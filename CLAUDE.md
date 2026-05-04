# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Keep this file in sync with [AGENTS.md](AGENTS.md).

## Coding Environment

- Install `uv` via `curl -LsSf https://astral.sh/uv/install.sh | sh` if absent; always `uv self update` first.
- Install Python 3.14 via `uv python install 3.14`.
- Always use `uv run` instead of the global `python` command.
- Python 3.14 supports `except TypeError, ValueError:` (no parentheses) — the ruff formatter is configured for `py314`.
- Read `.env.example` for all environment variables; `.env` is gitignored.

## Commands

```bash
uv run ruff format        # format
uv run ruff check         # lint
uv run ty check           # type check
uv run pytest             # unit + contract tests
```

Run in that order before pushing. CI enforces all four checks via `tests.yml`.

Run a single test file:
```bash
uv run pytest tests/path/to/test_file.py
```

Start the proxy:
```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8082
```

Live smoke tests (require real credentials or local services):
```bash
FCC_LIVE_SMOKE=1 uv run pytest smoke/ -n 0
```

## Architecture

`free-claude-code` is an Anthropic-compatible proxy (FastAPI on port 8082) that translates Claude Code's Anthropic Messages API calls to any of six provider backends.

### Request Flow

```
Claude Code CLI / IDE
        │  Anthropic Messages API
        ▼
server.py  (ASGI entry, exposes server:app)
        │
api/app.py  create_app()  ← preferred ASGI factory
        │
api/routes.py  →  api/services.py  →  providers/registry.py
        │                                      │
api/model_router.py                   per-provider transport
(resolves MODEL_OPUS/SONNET/HAIKU/MODEL)       │
        │                              upstream provider
api/optimization_handlers.py          (NIM / OpenRouter / DeepSeek /
(answers trivial Claude Code probes    LM Studio / llama.cpp / Ollama)
 locally to save latency and quota)
```

### Package Boundaries

Dependency direction (imports must flow this way):

```
config  ──►  api
config  ──►  providers
config  ──►  messaging
core    ──►  api
core    ──►  providers
core    ──►  messaging
providers ─► api
api ──────►  cli
api ──────►  messaging
```

Enforced by `tests/contracts/test_import_boundaries.py`. Key rules:
- `api/` may only import `providers.base`, `providers.exceptions`, and `providers.registry` — never per-adapter modules.
- `core/` has no imports from `api`, `messaging`, `cli`, `providers`, `config`, or `smoke`.
- `messaging/` does not import `api`, `cli`, or `smoke`.
- Shared Anthropic protocol helpers belong in `core/anthropic/`, not under any provider package.

### Key Modules

| Path | Responsibility |
|---|---|
| `server.py` | Module-level `app` instance (convenience entry point) |
| `api/app.py` | `create_app()` ASGI factory, lifespan, middleware |
| `api/runtime.py` | App composition, optional messaging startup, session/cleanup |
| `api/model_router.py` | Resolves Claude model tier → configured provider/model |
| `api/optimization_handlers.py` | Local short-circuit for trivial Claude Code probes |
| `api/services.py` | Request orchestration, streaming, error normalization |
| `providers/registry.py` | Provider factory, app-scoped `ProviderRegistry` |
| `providers/base.py` | `BaseProvider`, `ProviderConfig` |
| `providers/nvidia_nim/` | OpenAI chat-completions → Anthropic SSE translation |
| `providers/open_router/` | Anthropic Messages transport for OpenRouter |
| `providers/defaults.py` | Single constant per provider's default base URL |
| `core/anthropic/` | Protocol helpers, stream primitives, thinking, tool helpers, `stream_contracts.py` |
| `config/settings.py` | `Settings` (env-backed, loaded once at startup) |
| `config/provider_catalog.py` | Provider metadata and model listings |
| `messaging/` | Discord/Telegram adapters, sessions, voice transcription |
| `cli/` | Claude subprocess management, package entrypoints |
| `smoke/` | Opt-in live product smoke scenarios (`FCC_LIVE_SMOKE=1`) |

### Provider Transports

- **NVIDIA NIM**: extends `OpenAIChatTransport` — translates OpenAI chat-completions streaming into Anthropic SSE.
- **OpenRouter, DeepSeek, LM Studio, llama.cpp, Ollama**: extend `AnthropicMessagesTransport` — native Anthropic Messages style.
- To add a new OpenAI-compatible provider: extend `OpenAIChatTransport`; register in `config/provider_catalog.py` and `providers/registry.py`.
- To add a new Anthropic-compatible provider: extend `AnthropicMessagesTransport`; same registration steps.

### Production HTTP Handlers

Use `resolve_provider(request.app)` (not `get_provider()` / `get_provider_for_type()`) so the app-scoped `ProviderRegistry` is used. The process-cached helpers are for scripts and unit tests only.

## Code Standards

- No `# type: ignore` or `# ty: ignore` — fix the underlying type issue.
- No direct `_attribute` assignment to internal state from outside a class; use accessor methods.
- Provider-specific config fields belong in provider constructors, not in the base `ProviderConfig`.
- Use `settings.provider_type` (not string literals like `"nvidia_nim"`) for provider identification.
- String accumulation: use list + `"".join()`, not `+=` in loops.
- When moving a module: update all imports and remove compatibility shims in the same change.

## Summary Standards (for commit messages and PR descriptions)

Include: **[Files Changed]**, **[Logic Altered]**, **[Verification Method]**, **[Residual Risks]** (or "none").
