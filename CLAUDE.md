# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Commands

**Development Setup:**
```bash
uv python install 3.14    # Install Python 3.14
uv sync                   # Install dependencies
```

**Common Tasks:**
```bash
uv run ruff format        # Format code
uv run ruff check         # Lint check
uv run ty check          # Type checking
uv run pytest            # Run tests
uv run pytest -k test_name -v  # Run single test
```

**Run the Server:**
```bash
# Terminal 1: Start proxy
uv run uvicorn server:app --host 0.0.0.0 --port 8082

# Terminal 2: Use Claude Code with proxy
ANTHROPIC_BASE_URL="http://localhost:8082" ANTHROPIC_AUTH_TOKEN="test" claude
```

**Pre-commit Checklist:**
All 5 checks must pass before committing:
```bash
uv run ruff format && uv run ruff check && uv run ty check && uv run pytest
```

---

## Coding Environment

- **Python Version**: 3.14 (see `.python-version`)
- **Package Manager**: `uv` (required; install via `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Always use**: `uv run` instead of global `python` command
- **Type Checking**: `ty` (no `# type: ignore` or `# ty: ignore` comments allowed—fix the underlying type issue)
- **Code Formatter**: Ruff (target Python 3.14, line length 88)

---

## High-Level Architecture

Free Claude Code is a **transparent proxy middleware** between Claude Code CLI/VSCode and multiple LLM providers:

```
Claude Code (CLI/VSCode)
        ↓ (Anthropic API format)
    [Proxy Server] ← Port 8082
        ↓ (OpenAI-compatible format)
    LLM Providers (NIM / OpenRouter / LM Studio / llama.cpp)
```

### Core Modules

**`api/`** — FastAPI routes & request handling
- `app.py`: Application factory, lifespan management (messaging platform init)
- `routes.py`: POST `/v1/messages` endpoint
- `detection.py`: Detect trivial requests (network probes, title generation, etc.)
- `optimization_handlers.py`: Intercept & respond locally to reduce API quota usage
- `dependencies.py`: Dependency injection for providers & request validation
- `models/`: Pydantic request/response models

**`providers/`** — LLM provider abstraction
- `base.py`: `BaseProvider` ABC & `ProviderConfig` (all providers extend this)
- `openai_compat.py`: `OpenAICompatibleProvider` (NIM, OpenRouter, LM Studio, llamacpp all use this)
- `common/`: Shared utilities
  - `message_converter.py`: Anthropic ↔ OpenAI message format conversion
  - `think_parser.py`: Parse `<think>` tags and `reasoning_content` into Claude thinking blocks
  - `heuristic_tool_parser.py`: Parse models' text tool calls into structured tool use
  - `sse_builder.py`: Build Anthropic SSE format responses
  - `error_mapping.py`: Map provider errors to Anthropic error format
- `nvidia_nim/`, `open_router/`, `lmstudio/`, `llamacpp/`: Provider-specific clients

**`config/`** — Configuration & settings
- `settings.py`: Pydantic `Settings` class (reads `.env`); defines all config variables
- `nim.py`: NVIDIA NIM-specific config (thinking model support via `chat_template_kwargs`)
- `logging_config.py`: Loguru logger setup

**`messaging/`** — Discord/Telegram bot integration (autonomous coding)
- `handler.py`: Main event loop, session management, tool interception
- `session.py`: Per-user session state (working directory, task history)
- `commands.py`: `/stop`, `/clear`, `/stats` commands
- `platforms/factory.py`: Create platform instance (Discord or Telegram)
- `platforms/discord.py`, `platforms/telegram.py`: Platform-specific logic
- `transcription.py`: Voice note support (local Whisper or NVIDIA NIM)
- `transcript.py`: LLM output rendering (thinking, tool calls, results)
- `trees/`: Message threading (reply = fork conversation)
- `rendering/`: Markdown to platform-specific formatting

**`cli/`** — CLI session & process management
- `entrypoints.py`: `free-claude-code` (serve) and `fcc-init` (init config)
- `process_registry.py`: Track subprocess cleanup

**`tests/`** — Pytest suite (mirrors source structure)
- Shared `conftest.py` with fixtures
- Tests auto-run with `-n auto` (parallel execution via pytest-xdist)

---

## Provider Routing & Model Mapping

**How it works:**
1. Claude Code sends request with `model="claude-3-5-sonnet-20241022"` (or Opus/Haiku)
2. Proxy resolves model name → `MODEL_SONNET` env var → actual LLM (e.g., `open_router/deepseek/deepseek-r1-0528:free`)
3. Provider prefix (`open_router/`, `nvidia_nim/`, etc.) determines which provider class to instantiate
4. Request is converted from Anthropic format → OpenAI format → sent to provider
5. Response is converted back → Anthropic SSE format → sent to Claude Code

**Model name resolution** (in `api/dependencies.py`):
```
claude-3-opus-20250729     → MODEL_OPUS env var (fallback: MODEL)
claude-3-5-sonnet-...      → MODEL_SONNET env var (fallback: MODEL)
claude-3-haiku-20250307    → MODEL_HAIKU env var (fallback: MODEL)
(unrecognized)             → MODEL env var (global fallback)
```

**Provider prefix detection** (in `providers/openai_compat.py`):
- `nvidia_nim/path/to/model` → NIM provider
- `open_router/path/to/model` → OpenRouter provider
- `lmstudio/model-name` → LM Studio (localhost:1234/v1)
- `llamacpp/model-name` → llama.cpp (localhost:8080/v1)
- Invalid prefix → error

---

## Request Optimization (Quota Savings)

5 categories of requests are intercepted & responded to **locally** without hitting the provider API:

1. **Network Probe** (`ENABLE_NETWORK_PROBE_MOCK=true`): Models test connectivity
2. **Title Generation** (`ENABLE_TITLE_GENERATION_SKIP=true`): Claude Code generates conversation titles
3. **Prefix Detection** (`FAST_PREFIX_DETECTION=true`): Models check completion support
4. **Suggestion Mode** (`ENABLE_SUGGESTION_MODE_SKIP=true`): Models generate code suggestions
5. **Filepath Extraction** (`ENABLE_FILEPATH_EXTRACTION_MOCK=true`): Extract output file paths

See `api/detection.py` & `api/optimization_handlers.py` for implementation.

---

## Rate Limiting & Concurrency

**Proactive Rate Limiting** (rolling window):
- `PROVIDER_RATE_LIMIT` requests per `PROVIDER_RATE_WINDOW` seconds (default: 40 req/60s for NIM)
- Exponential backoff on 429 errors

**Concurrency Cap**:
- `PROVIDER_MAX_CONCURRENCY` (default: 5) limits max simultaneous open provider streams

**Messaging Rate Limiting** (separate, for Discord/Telegram):
- `MESSAGING_RATE_LIMIT` & `MESSAGING_RATE_WINDOW` (default: 1 msg/1s)

---

## Thinking Token Support

Free Claude Code automatically converts:
- **Claude 3.5 thinking format**: `<think>...</think>` tags
- **Provider reasoning format**: `reasoning_content` field (OpenAI o1, Deepseek, etc.)

**Into native Claude thinking blocks** that Claude Code CLI/VSCode renders natively.

**NIM-specific**: Set `NIM_ENABLE_THINKING=true` to send `chat_template_kwargs` & `reasoning_budget` on thinking models (Kimi, Nemotron). Leave `false` for non-thinking models (Mistral).

---

## Tool Call Parsing

Models output tool calls in different formats. Free Claude Code provides:

1. **Heuristic text parser** (in `providers/common/heuristic_tool_parser.py`):
   - Detects tool calls in model text output (e.g., `<tool_name>{"arg": "value"}</tool_name>`)
   - Parses into structured tool use for Claude Code

2. **Subagent interception** (in `messaging/handler.py`):
   - Forces `run_in_background=False` on Task tool calls
   - Prevents runaway autonomous agents

---

## Architecture Principles

- **Shared utilities** in `providers/common/` — no inter-provider imports
- **DRY**: Extract common base classes; prefer composition over copy-paste
- **Encapsulation**: Use accessor methods for internal state (e.g., `set_current_task()`)
- **Provider-specific config** stays in provider constructors, not base `ProviderConfig`
- **Dead code removal**: Delete unused code; use settings instead of hardcoded values
- **Performance**: List accumulation for strings (not `+=` in loops), cache env vars at init
- **Platform-agnostic naming**: Use `PLATFORM_EDIT` not `TELEGRAM_EDIT` in shared code
- **Backward compatibility**: Re-export from old module locations when moving modules

---

## Configuration

### Core Provider
| Variable             | Default                                    | Notes |
|----------------------|--------------------------------------------|-------|
| `MODEL`              | `nvidia_nim/stepfun-ai/step-3.5-flash`    | Fallback |
| `MODEL_OPUS`         | `nvidia_nim/z-ai/glm4.7`                  | —     |
| `MODEL_SONNET`       | `open_router/arcee-ai/trinity-large-preview:free` | — |
| `MODEL_HAIKU`        | `open_router/stepfun/step-3.5-flash:free` | —     |
| `NVIDIA_NIM_API_KEY` | (required)                                 | —     |
| `NIM_ENABLE_THINKING`| `false`                                    | Set `true` for thinking models |
| `OPENROUTER_API_KEY` | (required)                                 | —     |
| `LM_STUDIO_BASE_URL` | `http://localhost:1234/v1`                 | —     |
| `LLAMACPP_BASE_URL`  | `http://localhost:8080/v1`                 | —     |

### Rate Limiting
| Variable                   | Default | Notes |
|----------------------------|---------|-------|
| `PROVIDER_RATE_LIMIT`      | `40`    | Requests per window |
| `PROVIDER_RATE_WINDOW`     | `60`    | Window in seconds |
| `PROVIDER_MAX_CONCURRENCY` | `5`     | Max concurrent streams |

### Messaging (Discord/Telegram)
| Variable                   | Default           | Notes |
|----------------------------|-------------------|-------|
| `MESSAGING_PLATFORM`       | `discord`         | `discord` or `telegram` |
| `DISCORD_BOT_TOKEN`        | `""`              | —     |
| `ALLOWED_DISCORD_CHANNELS` | `""`              | Comma-separated IDs |
| `TELEGRAM_BOT_TOKEN`       | `""`              | —     |
| `ALLOWED_TELEGRAM_USER_ID` | `""`              | —     |
| `CLAUDE_WORKSPACE`         | `./agent_workspace` | Agent working directory |

See `.env.example` for all variables.

---

## Testing

**Run all tests:**
```bash
uv run pytest -v
```

**Run single test file:**
```bash
uv run pytest tests/api/test_routes.py -v
```

**Run with coverage:**
```bash
uv run pytest --cov=. --cov-report=html
```

**Parallel execution** is enabled by default (`pytest-xdist`).

---

## CI/CD Checks (tests.yml)

All 5 checks must pass on push/PR:
1. No `# type: ignore` or `# ty: ignore` comments allowed
2. `uv run ruff format --check` (formatting)
3. `uv run ruff check` (linting)
4. `uv run ty check` (type checking)
5. `uv run pytest -v` (tests)

---

## Extending the Codebase

### Add a New Provider

Extend `OpenAICompatibleProvider` or `BaseProvider`:

```python
from providers.openai_compat import OpenAICompatibleProvider
from providers.base import ProviderConfig

class MyProvider(OpenAICompatibleProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="MYPROVIDER",
            base_url="https://api.example.com/v1",
            api_key=config.api_key,
        )
```

### Add a Messaging Platform

Extend `MessagingPlatform` in `messaging/platforms/`:

```python
from messaging.platforms.base import MessagingPlatform

class MyPlatform(MessagingPlatform):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def send_message(self, ...) -> None: ...
    async def on_message(self, ...) -> None: ...
```

---

## Important Files & Locations

- **Entry point**: `server.py`
- **Settings**: `config/settings.py`
- **Model mapping**: `api/dependencies.py` (resolve `model` name → provider model)
- **Provider selection**: `providers/openai_compat.py` (parse prefix → choose provider)
- **Request conversion**: `providers/common/message_converter.py`
- **Thinking tokens**: `providers/common/think_parser.py`
- **Tool parsing**: `providers/common/heuristic_tool_parser.py`
- **Messaging handler**: `messaging/handler.py`
- **CLI**: `cli/entrypoints.py`

---

## Development Workflow

1. **Read relevant files** — don't guess
2. **Map the logic** — understand flow before changing
3. **Fix root cause** — not the symptom
4. **Test incrementally** — run tests after each change
5. **Run all checks** — ensure they pass before committing
6. **Specific changes** — do exactly what's asked, nothing more

---

## Useful Links

- **README**: Setup, quick start, configuration, troubleshooting
- **`.env.example`**: All environment variables with defaults
- **`nvidia_nim_models.json`**: List of available NIM models
- **GitHub Issues**: Report bugs or suggest features
