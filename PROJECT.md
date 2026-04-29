# PROJECT.md — Free Claude Code: Comprehensive Codebase Documentation

> **Purpose**: This document describes every module, class, data flow, and architectural
> contract in the `free-claude-code` project so that any developer or agent can
> understand the system, make changes safely, and avoid regressions.

---

## 1. Project Overview

**Free Claude Code** is an **Anthropic-compatible HTTP proxy** that lets
[Claude Code](https://docs.anthropic.com/en/docs/claude-code) (the official CLI,
VS Code extension, and JetBrains ACP plugin) talk to **non-Anthropic LLM backends**
— NVIDIA NIM, OpenRouter, DeepSeek, LM Studio, llama.cpp, and Ollama — while
keeping the exact same request/response contract that Claude Code expects.

### Core value proposition
| Concern | How the proxy solves it |
|---|---|
| **API compatibility** | Exposes `/v1/messages`, `/v1/messages/count_tokens`, `/v1/models` with Anthropic-shaped JSON and SSE. |
| **Provider routing** | Routes Opus / Sonnet / Haiku model tiers to different backends via `MODEL_*` env vars. |
| **Streaming conversion** | Translates OpenAI `chat.completions` chunks → Anthropic SSE lifecycle (message_start → content_block_start → deltas → stop). |
| **Native passthrough** | For providers with native Anthropic endpoints (OpenRouter, DeepSeek, LM Studio, llama.cpp, Ollama), streams SSE events directly with minimal transformation. |
| **Local optimizations** | Short-circuits common Claude Code probes (quota check, title generation, prefix detection, suggestions, filepath extraction) with instant local responses. |
| **Messaging bots** | Optional Discord / Telegram wrapper that spawns Claude Code CLI subprocesses and streams progress back to chat. |
| **Voice notes** | Optional voice-note transcription via local Whisper or NVIDIA NIM Riva, feeding transcripts into the messaging handler. |

### Tech stack
- **Python 3.14** (managed via `astral-sh/uv`)
- **FastAPI** + **Uvicorn** (ASGI)
- **Pydantic v2** + **pydantic-settings** (models, validation, env config)
- **httpx** (async HTTP for native Anthropic transports)
- **openai** Python SDK (async client for OpenAI-compat providers like NIM)
- **tiktoken** (token estimation)
- **loguru** (structured JSON logging)
- **python-telegram-bot** / **discord.py** (optional messaging)

---

## 2. Repository Layout

```
free-claude-code/
├── server.py                  # ASGI entry point (creates FastAPI app)
├── pyproject.toml             # Dependencies, scripts, tool config
├── .env.example               # Canonical env var template
├── AGENTS.md / CLAUDE.md      # Agent coding guidelines (identical)
├── PLAN.md                    # Architectural plan & dependency rules
│
├── api/                       # FastAPI application layer
│   ├── app.py                 # Application factory, lifespan, error handlers
│   ├── routes.py              # HTTP route definitions
│   ├── services.py            # ClaudeProxyService (orchestration)
│   ├── runtime.py             # AppRuntime (startup/shutdown lifecycle)
│   ├── dependencies.py        # Dependency injection (provider, auth)
│   ├── model_router.py        # Claude model → provider model resolution
│   ├── optimization_handlers.py  # Local fast-path handlers
│   ├── detection.py           # Request type detection utilities
│   ├── command_utils.py       # Shell command prefix/filepath extraction
│   ├── web_server_tools.py    # Compat re-exports for web_tools/
│   ├── models/
│   │   ├── anthropic.py       # Pydantic request models (MessagesRequest, etc.)
│   │   └── responses.py       # Pydantic response models (MessagesResponse, etc.)
│   └── web_tools/             # Local web_search / web_fetch server tool handler
│       ├── request.py         # Detect forced server tool requests
│       ├── streaming.py       # SSE streaming for web tool results
│       ├── egress.py          # URL allowlist / egress policy
│       ├── outbound.py        # Actual HTTP fetch / search execution
│       ├── parsers.py         # Query / URL extraction from messages
│       └── constants.py       # Web tool constants
│
├── config/                    # Application configuration
│   ├── settings.py            # Settings (pydantic-settings, env loading)
│   ├── constants.py           # Shared numeric constants
│   ├── nim.py                 # NimSettings (NVIDIA NIM sampling params)
│   ├── provider_catalog.py    # ProviderDescriptor registry (IDs, transport types, credentials)
│   ├── provider_ids.py        # Re-exports SUPPORTED_PROVIDER_IDS
│   └── logging_config.py      # Loguru JSON logging, secret redaction
│
├── core/                      # Shared protocol logic (no provider imports)
│   ├── rate_limit.py          # StrictSlidingWindowLimiter primitive
│   └── anthropic/             # Anthropic protocol helpers
│       ├── conversion.py      # AnthropicToOpenAIConverter (message/tool/system)
│       ├── sse.py             # SSEBuilder, ContentBlockManager, format_sse_event
│       ├── thinking.py        # ThinkTagParser (<think> streaming parser)
│       ├── tools.py           # HeuristicToolParser (text → tool_use)
│       ├── tokens.py          # get_token_count (tiktoken estimation)
│       ├── content.py         # extract_text_from_content, get_block_type/attr
│       ├── errors.py          # User-facing error message formatting
│       ├── native_messages_request.py  # Native Anthropic body builders
│       ├── native_sse_block_policy.py  # Thinking block filter for native SSE
│       ├── emitted_sse_tracker.py      # Mid-stream error recovery tracker
│       ├── provider_stream_error.py    # Error SSE event generator
│       ├── server_tool_sse.py          # Server tool SSE type constants
│       ├── stream_contracts.py         # Streaming contract test helpers
│       └── utils.py                    # set_if_not_none utility
│
├── providers/                 # LLM provider adapters
│   ├── base.py                # BaseProvider ABC, ProviderConfig model
│   ├── registry.py            # ProviderRegistry (factory, cache, cleanup)
│   ├── openai_compat.py       # OpenAIChatTransport (NIM base class)
│   ├── anthropic_messages.py  # AnthropicMessagesTransport (native base class)
│   ├── error_mapping.py       # map_error() (OpenAI/httpx → ProviderError)
│   ├── exceptions.py          # ProviderError hierarchy
│   ├── rate_limit.py          # GlobalRateLimiter (proactive + reactive + concurrency)
│   ├── defaults.py            # Re-exports default base URLs
│   ├── nvidia_nim/            # NvidiaNimProvider (OpenAI Chat transport)
│   │   ├── client.py          # Provider class with retry logic
│   │   └── request.py         # NIM-specific request body builder
│   ├── open_router/           # OpenRouterProvider (native Anthropic transport)
│   │   ├── client.py          # Provider class with SSE filtering
│   │   └── request.py         # OpenRouter-specific request body builder
│   ├── deepseek/              # DeepSeekProvider (native Anthropic transport)
│   │   ├── client.py          # Provider class
│   │   └── request.py         # DeepSeek-specific request body builder
│   ├── lmstudio/              # LMStudioProvider (native Anthropic transport)
│   │   └── client.py
│   ├── llamacpp/              # LlamaCppProvider (native Anthropic transport)
│   │   └── client.py
│   └── ollama/                # OllamaProvider (native Anthropic, /v1/messages)
│       └── client.py
│
├── messaging/                 # Optional Discord / Telegram bot layer
│   ├── handler.py             # ClaudeMessageHandler (core orchestration)
│   ├── platforms/
│   │   ├── base.py            # MessagingPlatform ABC, CLISession Protocol
│   │   ├── telegram.py        # TelegramPlatform adapter
│   │   └── discord.py         # DiscordPlatform adapter
│   ├── trees/                 # Tree-based message queue
│   │   ├── data.py            # MessageTree, MessageNode, MessageState
│   │   └── queue_manager.py   # TreeQueueManager (per-tree FIFO)
│   ├── rendering/             # Platform-specific markdown rendering
│   ├── session.py             # SessionStore (persistence)
│   ├── models.py              # IncomingMessage model
│   ├── commands.py            # Slash commands (/stop, /clear, /stats)
│   ├── command_dispatcher.py  # Command routing
│   ├── event_parser.py        # CLI JSON event → parsed events
│   ├── transcript.py          # TranscriptBuffer
│   ├── voice.py               # VoiceTranscriptionService
│   └── ...                    # UI updates, diagnostics, etc.
│
├── cli/                       # Claude Code CLI subprocess management
│   ├── entrypoints.py         # `free-claude-code` and `fcc-init` console scripts
│   ├── manager.py             # CLISessionManager (pool of CLISession instances)
│   ├── session.py             # CLISession (single subprocess lifecycle)
│   └── process_registry.py    # Global PID registry for cleanup
│
├── smoke/                     # Live integration / smoke tests
│   ├── prereq/                # Pre-requisite checks (API connectivity, auth)
│   ├── product/               # Product-level smoke tests
│   ├── lib/                   # Smoke test utilities (child process, config, HTTP)
│   └── capabilities.py        # CapabilityContract registry
│
└── tests/                     # Unit / integration tests (pytest)
    ├── api/                   # API layer tests
    ├── cli/                   # CLI session tests
    ├── config/                # Settings / logging tests
    ├── contracts/             # Architecture contract enforcement
    ├── core/                  # Core module tests
    ├── messaging/             # Messaging layer tests
    ├── providers/             # Provider adapter tests
    ├── conftest.py            # Shared fixtures
    └── stream_contract.py     # Stream lifecycle test helpers
```

---

## 3. Application Startup Flow

```
server.py
  └─ create_app()                              [api/app.py]
       ├─ configure_logging("server.log")       [config/logging_config.py]
       ├─ FastAPI(lifespan=lifespan)
       │    └─ lifespan():
       │         ├─ AppRuntime.for_app(app, settings)  [api/runtime.py]
       │         ├─ runtime.startup():
       │         │    ├─ ProviderRegistry()             [providers/registry.py]
       │         │    ├─ app.state.provider_registry = registry
       │         │    ├─ warn_if_process_auth_token()
       │         │    ├─ _start_messaging_if_configured():
       │         │    │    ├─ CLISessionManager(...)     [cli/manager.py]
       │         │    │    ├─ create_platform(...)       [messaging/platforms/factory.py]
       │         │    │    ├─ ClaudeMessageHandler(...)  [messaging/handler.py]
       │         │    │    ├─ SessionStore.restore_trees()
       │         │    │    └─ platform.start()
       │         │    └─ _publish_state()
       │         │
       │         └─ runtime.shutdown():
       │              ├─ platform.stop()
       │              ├─ cli_manager.stop_all()
       │              └─ provider_registry.cleanup_all()
       │
       ├─ app.include_router(router)            [api/routes.py]
       └─ exception handlers for ProviderError, ValidationError, Exception
```

### Key invariants at startup
1. `ProviderRegistry` is created empty; providers are lazily instantiated on first request.
2. `app.state.provider_registry` **must** be set before any request is served.
3. Messaging is only started when `MESSAGING_PLATFORM` ≠ `"none"`.
4. The log file `server.log` is **truncated** on each fresh start.

---

## 4. Request Processing Pipeline

### 4.1 POST /v1/messages (main path)

```
Client (Claude Code CLI / VS Code / JetBrains)
  │
  ▼
routes.create_message(request_data, service, _auth)    [api/routes.py:79]
  │
  ├─ require_api_key()                                  [api/dependencies.py]
  │    └─ checks x-api-key / Authorization / anthropic-auth-token
  │
  ├─ get_proxy_service() → ClaudeProxyService           [api/dependencies.py]
  │    ├─ get_settings()
  │    ├─ get_provider(request)  →  resolves provider via ProviderRegistry
  │    └─ ModelRouter(settings)
  │
  └─ service.create_message(request_data)               [api/services.py:101]
       │
       ├─ (1) try_optimizations(request_data, settings)  [api/optimization_handlers.py]
       │    ├─ try_prefix_detection()   — command prefix extraction
       │    ├─ try_quota_mock()         — fake quota response
       │    ├─ try_title_skip()         — skip title generation
       │    ├─ try_suggestion_skip()    — skip suggestion mode
       │    └─ try_filepath_mock()      — local filepath extraction
       │    → Returns MessagesResponse directly if matched (no provider call)
       │
       ├─ (2) is_web_server_tool_request() check         [api/web_tools/request.py]
       │    → If forced web_search/web_fetch, stream via local handler
       │
       ├─ (3) model_router.resolve(claude_model_name)    [api/model_router.py]
       │    → ResolvedModel(provider_id, provider_model, thinking_enabled)
       │
       ├─ (4) Strip Anthropic server tool definitions from tools list
       │       for OpenAI Chat providers (they can't handle them)
       │
       ├─ (5) get_token_count(request)                   [core/anthropic/tokens.py]
       │    → input_tokens estimate via tiktoken
       │
       └─ (6) StreamingResponse(provider.stream_response(...))
              │
              ▼
         Provider.stream_response()  (see §5)
```

### 4.2 POST /v1/messages/count_tokens

```
routes.count_tokens(request_data, service, _auth)
  └─ service.count_tokens(request_data)
       ├─ model_router.resolve_token_count(request_data)
       └─ get_token_count(request_data) → TokenCountResponse
```

### 4.3 GET /v1/models

Returns a static list of `SUPPORTED_CLAUDE_MODELS` in Anthropic `ModelsListResponse` format.
These are the Claude model IDs that Claude Code sends — the proxy accepts them and
re-routes them via the model router.

### 4.4 Other routes

| Route | Purpose |
|---|---|
| `GET /` | Root redirect / health info |
| `GET /health` | Health check (JSON) |
| `POST /stop` | Stop all CLI sessions (messaging mode) |
| `HEAD/OPTIONS /*` | Probe endpoints return 204 with Allow headers |

---

## 5. Provider Architecture

### 5.1 Two transport families

```
BaseProvider (ABC)                              [providers/base.py]
  ├─ OpenAIChatTransport                        [providers/openai_compat.py]
  │    └─ NvidiaNimProvider                     [providers/nvidia_nim/client.py]
  │
  └─ AnthropicMessagesTransport                 [providers/anthropic_messages.py]
       ├─ OpenRouterProvider                    [providers/open_router/client.py]
       ├─ DeepSeekProvider                      [providers/deepseek/client.py]
       ├─ LMStudioProvider                      [providers/lmstudio/client.py]
       ├─ LlamaCppProvider                      [providers/llamacpp/client.py]
       └─ OllamaProvider                        [providers/ollama/client.py]
```

### 5.2 OpenAIChatTransport (NIM)

- Uses the **OpenAI Python SDK** (`AsyncOpenAI`) for `/chat/completions`.
- **Converts** Anthropic → OpenAI format via `AnthropicToOpenAIConverter`:
  - Messages: role mapping, tool_result → tool role, thinking → `<think>` tags or `reasoning_content`
  - Tools: Anthropic tool schema → OpenAI function schema
  - System prompt: Anthropic system block → `{"role": "system"}` message
- **Converts** OpenAI response chunks → Anthropic SSE via `SSEBuilder`:
  - `reasoning_content` → thinking blocks
  - `delta.content` → text blocks (with `ThinkTagParser` for `<think>` tags)
  - `delta.tool_calls` → tool_use blocks
  - `finish_reason` → Anthropic `stop_reason`
- **Heuristic tool parsing**: Some models emit tool calls as text (`● <function=...>`);
  `HeuristicToolParser` detects and converts these.
- **Task subagent control**: `run_in_background` is forced to `false` for Task tool calls.
- **NIM-specific retry**: Retries once without `reasoning_budget`, `chat_template`, or
  `reasoning_content` if NIM rejects them with 400.

### 5.3 AnthropicMessagesTransport (OpenRouter, DeepSeek, local providers)

- Uses **httpx** directly to stream from native Anthropic-compatible `/messages` endpoints.
- **Minimal transformation**: Request body is serialized from Pydantic models; SSE events
  are streamed through with optional filtering.
- **Thinking block policy** (`NativeSseBlockPolicyState`): Filters or passes thinking/
  redacted_thinking blocks based on `thinking_enabled`.
- **Two chunk modes**:
  - `"line"` (default): Events are grouped and transformed, then re-split into lines.
  - `"event"` (OpenRouter): Events are grouped and transformed as whole SSE events.
- **Error recovery**: `EmittedNativeSseTracker` tracks open blocks; on mid-stream error,
  closes unclosed blocks and emits an Anthropic-compatible error tail.
- **Provider-specific overrides**:
  - OpenRouter: Custom headers (`anthropic-version`), terminal event filtering, reasoning policy.
  - DeepSeek: Custom auth header (`x-api-key`), dedicated request body builder.
  - Ollama: Hits `/v1/messages` instead of `/messages`.

### 5.4 ProviderRegistry

```python
class ProviderRegistry:                         [providers/registry.py]
    """Cache and clean up provider instances by provider id."""
```

- **Lazy instantiation**: Providers are created on first request via `create_provider()`.
- **Caching**: Provider instances are stored in `_providers` dict by provider ID.
- **Config-driven**: `PROVIDER_CATALOG` + `PROVIDER_DESCRIPTORS` define metadata;
  `PROVIDER_FACTORIES` map IDs to constructor functions.
- **Credential resolution**: `build_provider_config()` reads API keys from Settings,
  env vars, or static credentials defined in `ProviderDescriptor`.
- **Cleanup**: `cleanup_all()` calls `provider.cleanup()` on all cached instances.

### 5.5 Rate Limiting

```
core/rate_limit.py           → StrictSlidingWindowLimiter (primitive)
providers/rate_limit.py      → GlobalRateLimiter (proactive + reactive + concurrency)
```

- **Proactive**: Strict sliding window — at most N requests in any W-second window.
- **Reactive**: On 429, blocks all requests for a backoff period.
- **Concurrency**: Semaphore-based cap on simultaneous open streams.
- **Scoped instances**: Each provider gets its own limiter via `get_scoped_instance(scope)`.
- **Retry**: `execute_with_retry()` retries on 429 with exponential backoff + jitter.

### 5.6 Error Mapping

```python
map_error(e) → ProviderError subclass          [providers/error_mapping.py]
```

Maps `openai.*Error` and `httpx.HTTPStatusError` to the custom exception hierarchy:
- `AuthenticationError` (401/403)
- `RateLimitError` (429)
- `InvalidRequestError` (400)
- `OverloadedError` (502/503/504 or "overloaded" in message)
- `APIError` (other 5xx)

All `ProviderError` subclasses have `.to_anthropic_format()` for client-compatible responses.

---

## 6. Model Routing

```python
class ModelRouter:                               [api/model_router.py]
    def resolve(claude_model_name) → ResolvedModel
```

The model router maps incoming Claude model names to provider-specific models:

1. Client sends `model: "claude-sonnet-4-20250514"`.
2. `Settings.resolve_model()` checks `MODEL_SONNET` → e.g., `"nvidia_nim/qwen/qwen3-235b-a22b"`.
3. `Settings.parse_provider_type()` → `"nvidia_nim"`.
4. `Settings.parse_model_name()` → `"qwen/qwen3-235b-a22b"`.
5. `Settings.resolve_thinking()` checks `THINKING_SONNET` → `True/False`.
6. Result: `ResolvedModel(original="claude-sonnet-4-...", provider_id="nvidia_nim", model="qwen/...", thinking_enabled=True)`.

### Model tier mapping

| Claude tier | Settings key | Thinking key |
|---|---|---|
| Opus (`claude-*-opus*`) | `MODEL_OPUS` | `THINKING_OPUS` |
| Sonnet (`claude-*-sonnet*`) | `MODEL_SONNET` | `THINKING_SONNET` |
| Haiku (`claude-*-haiku*`) | `MODEL_HAIKU` | `THINKING_HAIKU` |
| Default (fallback) | `MODEL` | `THINKING` |

The provider prefix syntax is `provider_id/model_name` (e.g., `nvidia_nim/meta/llama-3.3-70b-instruct`).

---

## 7. Configuration System

### 7.1 Settings class

```python
class Settings(BaseSettings):                    [config/settings.py]
```

Uses `pydantic-settings` with env file loading. Precedence (highest → lowest):
1. Process environment variables
2. `FCC_ENV_FILE` (explicit override path)
3. `~/.config/free-claude-code/.env` (user config)
4. `./.env` (repo-local)

### 7.2 Key configuration groups

| Group | Env vars | Description |
|---|---|---|
| **Provider keys** | `NVIDIA_NIM_API_KEY`, `OPENROUTER_API_KEY`, `DEEPSEEK_API_KEY` | API credentials |
| **Local providers** | `LM_STUDIO_BASE_URL`, `LLAMACPP_BASE_URL`, `OLLAMA_BASE_URL` | Base URLs |
| **Model routing** | `MODEL`, `MODEL_OPUS`, `MODEL_SONNET`, `MODEL_HAIKU` | Provider-prefixed model refs |
| **Thinking** | `THINKING`, `THINKING_OPUS`, `THINKING_SONNET`, `THINKING_HAIKU` | Enable/disable per tier |
| **Auth** | `ANTHROPIC_AUTH_TOKEN` | Proxy auth token (optional) |
| **HTTP** | `HTTP_CONNECT_TIMEOUT`, `HTTP_READ_TIMEOUT`, `HTTP_WRITE_TIMEOUT` | Client timeouts |
| **Rate limits** | `RATE_LIMIT`, `RATE_WINDOW`, `MAX_CONCURRENCY` | Per-provider rate limiting |
| **Messaging** | `MESSAGING_PLATFORM`, `TELEGRAM_BOT_TOKEN`, `DISCORD_BOT_TOKEN` | Bot config |
| **Voice** | `WHISPER_DEVICE`, `WHISPER_MODEL` | Transcription backend |
| **Workspace** | `CLAUDE_WORKSPACE`, `CLAUDE_ALLOWED_DIRS`, `PLANS_DIRECTORY` | Agent workspace |
| **Logging** | `VERBOSE_HTTP_LOGGING`, `LOG_RAW_SSE_EVENTS`, etc. | Debug switches |

### 7.3 NimSettings

```python
class NimSettings(BaseModel):                    [config/nim.py]
```

Fixed NVIDIA NIM sampling parameters (temperature, top_p, top_k, max_tokens, penalties,
chat_template, etc.). Not loaded from env — configured programmatically.

### 7.4 Provider Catalog

```python
PROVIDER_CATALOG: dict[str, ProviderDescriptor]  [config/provider_catalog.py]
```

Metadata for each provider: ID, transport type (`openai_chat` or `anthropic_messages`),
credential env var, default base URL, capabilities tuple, and Settings attribute names
for credentials/proxy/base URL.

---

## 8. Core Anthropic Protocol Helpers

All shared between API, providers, and messaging. Located in `core/anthropic/`.

### 8.1 Message Conversion (`conversion.py`)

`AnthropicToOpenAIConverter` handles the complex translation of Anthropic message
format to OpenAI chat format:

- **Text blocks** → concatenated string content
- **Thinking blocks** → `<think>...</think>` tags (THINK_TAGS mode) or `reasoning_content` field (REASONING_CONTENT mode) or dropped (DISABLED mode)
- **Redacted thinking** → dropped (opaque provider data)
- **Tool use blocks** → `tool_calls` array with function schema
- **Tool result blocks** → `role: "tool"` messages
- **Image blocks** → raises `OpenAIConversionError` (not supported)
- **Server tool blocks** → raises `OpenAIConversionError` (use native transport)
- **Post-tool text** → deferred to after tool results (OpenAI constraint)
- **System prompt** → `role: "system"` message prepended

### 8.2 SSE Builder (`sse.py`)

`SSEBuilder` constructs the full Anthropic SSE streaming lifecycle:

```
message_start → [content_block_start → deltas → content_block_stop]* → message_delta → message_stop
```

- Manages content block indices (thinking, text, tool_use)
- Ensures blocks are properly opened/closed
- Tracks accumulated text/reasoning for token estimation
- `ContentBlockManager` handles tool call state (buffering, Task normalization)

### 8.3 ThinkTagParser (`thinking.py`)

Streaming parser for `<think>...</think>` tags in text content. Handles:
- Partial tags at chunk boundaries
- Orphan close tags
- Nested content
- Yields `ContentChunk(type=THINKING|TEXT, content=...)`

### 8.4 HeuristicToolParser (`tools.py`)

Detects tool calls emitted as text by some models:
- `● <function=name>` + `<parameter=key>value</parameter>` pattern
- JSON-style `WebFetch/WebSearch {"url": ...}` pattern
- Strips `<|control_tokens|>`

### 8.5 Token Counting (`tokens.py`)

`get_token_count()` estimates input tokens using tiktoken (`cl100k_base`):
- Counts message text, system prompt, tool schemas, images (fixed estimate), thinking blocks
- Used for both `/count_tokens` and input_tokens in SSE events

### 8.6 Native Messages Request (`native_messages_request.py`)

Two builders for native Anthropic transports:
- `build_base_native_anthropic_request_body()` — generic (LM Studio, llama.cpp, Ollama, DeepSeek)
- `build_openrouter_native_request_body()` — OpenRouter-specific (reasoning policy, system normalization, extra_body merge)

Both handle:
- Thinking policy sanitization (filter unsigned thinking blocks)
- Default max_tokens
- `extra_body` merge/validation

### 8.7 Native SSE Block Policy (`native_sse_block_policy.py`)

Filters native SSE events for thinking block policy:
- When thinking disabled: drops `content_block_start` with type=thinking, corresponding deltas and stops
- When thinking enabled: passes through signed thinking, drops unsigned
- Handles OpenRouter terminal `[DONE]` events

---

## 9. Local Optimizations

These bypass provider calls entirely for common Claude Code probes:

| Handler | Detects | Returns |
|---|---|---|
| `try_prefix_detection` | `max_tokens=1`, message asks for command prefix | `MessagesResponse` with extracted prefix |
| `try_quota_mock` | `max_tokens=1`, message contains "quota" | Fake quota OK response |
| `try_title_skip` | System prompt asks for conversation title | Pre-canned title response |
| `try_suggestion_skip` | System prompt in suggestion mode | Empty suggestion response |
| `try_filepath_mock` | `max_tokens=1`, message asks for filepaths | Extracted filepaths response |

Detection is done in `api/detection.py` by inspecting `MessagesRequest` fields
(max_tokens, message count, text content, system prompt patterns).

---

## 10. Web Server Tools

When `ENABLE_WEB_SERVER_TOOLS=true`, the proxy handles forced `web_search`/`web_fetch`
tool_choice locally:

1. `is_web_server_tool_request()` detects `tool_choice.type == "tool"` with `name in {web_search, web_fetch}`.
2. `stream_web_server_tool_response()` emits Anthropic SSE with:
   - `server_tool_use` content block
   - `web_search_tool_result` or `web_fetch_tool_result` block
   - Text summary block
3. Actual search/fetch is done via `outbound.py` (HTTP requests to search engines / URLs).
4. `WebFetchEgressPolicy` enforces URL allowlists.
5. For OpenAI Chat providers, listed server tools are stripped from the tools list
   (they can't process them); an error is returned if forced without web tools enabled.

---

## 11. Messaging Layer

### 11.1 Architecture

```
MessagingPlatform (ABC)                         [messaging/platforms/base.py]
  ├─ TelegramPlatform                           [messaging/platforms/telegram.py]
  └─ DiscordPlatform                            [messaging/platforms/discord.py]

ClaudeMessageHandler                            [messaging/handler.py]
  ├─ Uses: MessagingPlatform (send/edit/delete)
  ├─ Uses: SessionManagerInterface (CLISessionManager)
  ├─ Uses: SessionStore (persistence)
  └─ Uses: TreeQueueManager (message ordering)

CLISessionManager                               [cli/manager.py]
  └─ CLISession                                 [cli/session.py]
       └─ Spawns `claude` CLI subprocess
```

### 11.2 Message Flow

1. User sends message on Discord/Telegram.
2. Platform adapter creates `IncomingMessage`, calls `handler.handle_message()`.
3. Handler checks for commands (`/stop`, `/clear`, `/stats`).
4. Creates or extends a `MessageTree` (reply chains form trees).
5. Sends status message, enqueues node for processing.
6. `_process_node()`:
   - Gets or creates a `CLISession` (subprocess).
   - Calls `cli_session.start_task(prompt)`.
   - Iterates over CLI JSON events, parsing and updating transcript.
   - Edits the status message with live progress via `ThrottledTranscriptEditor`.
7. On completion, marks node as COMPLETED, frees session slot.

### 11.3 Tree Queue

```python
class MessageTree:                               [messaging/trees/data.py]
    """N-ary tree with per-tree FIFO queue."""

class TreeQueueManager:                          [messaging/trees/queue_manager.py]
    """Manages trees + per-tree processing queues."""
```

- New messages create tree roots.
- Replies become children of the replied-to node.
- Each tree has its own queue; nodes process one-at-a-time per tree.
- Cross-tree processing is parallel.
- States: PENDING → IN_PROGRESS → COMPLETED | ERROR.
- Error propagation: parent error cascades to pending children.

### 11.4 CLI Session

```python
class CLISession:                                [cli/session.py]
```

- Spawns `claude -p <prompt> --output-format stream-json --dangerously-skip-permissions --verbose`.
- Sets `ANTHROPIC_API_URL` to the proxy's own URL (self-referencing).
- Reads stdout line-by-line, parsing JSON events.
- Extracts session ID from events for resume/fork.
- Drains stderr concurrently (bounded capture).
- Supports `--resume <session_id>` and `--fork-session`.

### 11.5 Session Persistence

`SessionStore` persists message trees and node-to-tree mappings as JSON.
On restart, `restore_trees()` reloads trees and rebuilds the node index,
marking any IN_PROGRESS nodes as lost.

---

## 12. Voice Notes

When `WHISPER_DEVICE` is configured, the messaging platforms detect audio attachments:

- **Local Whisper** (`WHISPER_DEVICE=cpu|cuda`): Uses `transformers` + `torch` for local transcription.
- **NVIDIA NIM** (`WHISPER_DEVICE=nvidia_nim`): Uses NVIDIA Riva ASR API.
- Transcribed text is fed into `handle_message()` as the prompt.
- `PendingVoiceRegistry` tracks in-flight transcriptions to avoid duplicates.

---

## 13. Authentication

`require_api_key()` in `api/dependencies.py` checks (in order):
1. `x-api-key` header
2. `Authorization: Bearer <token>` header
3. `anthropic-auth-token` header

All compared against `ANTHROPIC_AUTH_TOKEN` setting. If not configured, all requests pass.

---

## 14. Logging

```python
configure_logging("server.log")                  [config/logging_config.py]
```

- **loguru** with JSON lines to `server.log`.
- Stdlib logging intercepted via `InterceptHandler`.
- Context vars (`request_id`, `node_id`, `chat_id`) promoted to top-level JSON keys.
- Sensitive substrings redacted (Telegram bot tokens, Authorization headers).
- Third-party loggers (httpx, httpcore, telegram) capped at WARNING unless verbose.

---

## 15. Testing Architecture

### 15.1 CI Pipeline

```yaml
# .github/workflows/tests.yml
1. Fail on `# type: ignore` / `# ty: ignore`
2. uv run ruff format --check
3. uv run ruff check
4. uv run ty check
5. uv run pytest -v --tb=short
```

### 15.2 Test categories

| Directory | Scope |
|---|---|
| `tests/api/` | Routes, services, auth, model router, optimizations, web tools |
| `tests/cli/` | CLI session, entrypoints, manager edge cases |
| `tests/config/` | Settings validation, logging config |
| `tests/contracts/` | Architecture contract enforcement, capability map, feature manifest |
| `tests/core/` | Core module unit tests |
| `tests/messaging/` | Handler, tree queue, platforms, voice, session store |
| `tests/providers/` | Provider adapters, streaming, conversion, errors |

### 15.3 Smoke tests

`smoke/` contains live integration tests that require actual provider credentials:
- `smoke/prereq/` — API connectivity and auth verification
- `smoke/product/` — End-to-end product features
- `smoke/capabilities.py` — `CapabilityContract` registry linking capabilities to test files

### 15.4 Architecture contracts

`tests/contracts/test_architecture_contracts.py` enforces:
- `PLAN.md` exists with required sections
- `api.__all__` matches expected exports
- No duplicate env template files
- No SSE shim in smoke/lib

---

## 16. Key Contracts and Invariants

### 16.1 Module dependency direction

```
config → (no app imports)
core   → config only
providers → core, config
api    → providers, core, config
messaging → api (via HTTP), core, config, cli
cli    → config only
smoke  → (external HTTP only)
```

**Rule**: `api/` and `messaging/` must NOT import provider internals. Use `BaseProvider` interface.

### 16.2 Provider contract

Every provider must:
1. Extend `BaseProvider` (or `OpenAIChatTransport` / `AnthropicMessagesTransport`).
2. Implement `stream_response()` → `AsyncIterator[str]` yielding Anthropic SSE format.
3. Implement `cleanup()` for resource release.
4. Be registered in `PROVIDER_CATALOG` and `PROVIDER_FACTORIES`.

### 16.3 SSE lifecycle contract

Every streaming response must produce:
```
event: message_start
event: content_block_start (index=0, type=thinking|text|tool_use)
event: content_block_delta  (0..N deltas)
event: content_block_stop
... (more blocks)
event: message_delta  (stop_reason, usage)
event: message_stop
```

On error mid-stream: close all open blocks, emit error text or top-level error event,
then `message_delta` + `message_stop`.

### 16.4 Request optimization contract

Optimizations must:
1. Return `MessagesResponse` (not SSE) or `None`.
2. Never call provider APIs.
3. Be idempotent and fast.
4. Fall through to provider when not matched.

### 16.5 Configuration contract

- No `# type: ignore` or `# ty: ignore` in any `.py` file.
- All CI checks must pass: format, lint, type check, tests.
- Dead env vars (e.g., `NIM_ENABLE_THINKING`) cause startup validation errors with migration guidance.

---

## 17. Adding a New Provider

1. Create `providers/<name>/client.py` extending `OpenAIChatTransport` or `AnthropicMessagesTransport`.
2. Implement `_build_request_body()` and any provider-specific overrides.
3. Add `ProviderDescriptor` to `config/provider_catalog.py`.
4. Add factory function to `PROVIDER_FACTORIES` in `providers/registry.py`.
5. Add credential / base URL fields to `config/settings.py` if needed.
6. Add `.env.example` entries.
7. Add tests in `tests/providers/`.
8. Update `smoke/capabilities.py` with capability contracts.

---

## 18. Common Pitfalls and Regression Risks

| Area | Risk | Mitigation |
|---|---|---|
| **SSE lifecycle** | Missing `content_block_stop` or `message_stop` crashes Claude Code | Stream contract tests in `tests/contracts/` |
| **Thinking policy** | Unsigned thinking blocks sent upstream cause errors | `sanitize_native_messages_thinking_policy()` filters them |
| **Tool call ordering** | Post-tool text in wrong position breaks OpenAI history | `_PendingAfterTools` defers text until tool results replay |
| **Task subagent** | `run_in_background=true` spawns uncontrolled subagents | Forced to `false` in `_normalize_task_run_in_background()` |
| **Rate limiting** | Shared singleton vs scoped instances | Each provider gets `get_scoped_instance(scope)` |
| **Server tools** | Listed web_search/web_fetch sent to OpenAI provider | Stripped from tools list; error if forced without handler |
| **Model routing** | Missing `MODEL` env var with no per-tier override | `Settings.resolve_model()` falls back gracefully |
| **Auth header order** | Claude Code sends different headers per client | `require_api_key()` checks all three header formats |
| **Native SSE passthrough** | Extra events from OpenRouter (e.g., `[DONE]`) | `NativeSseBlockPolicyState` + `is_terminal_openrouter_done_event()` filter them |
| **Mid-stream errors** | Unclosed blocks in partially streamed response | `EmittedNativeSseTracker` closes unclosed blocks before error tail |

---

## 19. Console Scripts

| Script | Entry point | Purpose |
|---|---|---|
| `free-claude-code` | `cli.entrypoints:serve` | Start the proxy server |
| `fcc-init` | `cli.entrypoints:init` | Scaffold `~/.config/free-claude-code/.env` |

---

## 20. Environment Variable Reference

See `.env.example` for the complete annotated list. Key groups:

- **Provider selection**: `MODEL=provider_id/model_name`
- **Per-tier overrides**: `MODEL_OPUS`, `MODEL_SONNET`, `MODEL_HAIKU`
- **Thinking control**: `THINKING=true`, `THINKING_SONNET=false`
- **Proxy/timeout**: `NVIDIA_NIM_PROXY`, `HTTP_READ_TIMEOUT=120`
- **Rate limits**: `RATE_LIMIT=40`, `RATE_WINDOW=60`, `MAX_CONCURRENCY=5`
- **Auth**: `ANTHROPIC_AUTH_TOKEN=your-secret`
- **Server binding**: `HOST=0.0.0.0`, `PORT=8082`
- **Web tools**: `ENABLE_WEB_SERVER_TOOLS=true`
- **Messaging**: `MESSAGING_PLATFORM=discord|telegram|none`
- **Voice**: `WHISPER_DEVICE=cpu|cuda|nvidia_nim`

---

*This document is auto-maintained. When adding new modules, update the relevant sections
to keep this documentation in sync with the codebase.*
