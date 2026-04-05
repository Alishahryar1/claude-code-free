# AGENTIC DIRECTIVE

> This file is identical to CLAUDE.md. Keep them in sync.

## CODING ENVIRONMENT

- Install astral uv using "curl -LsSf https://astral.sh/uv/install.sh | sh" if not already installed and if already installed then update it to the latest version
- Install Python 3.14 using `uv python install 3.14` if not already installed
- Always use `uv run` to run files instead of the global `python` command.
- Current uv ruff formatter is set to py314 which has supports multiple exception types without paranthesis (except TypeError, ValueError:)
- Read `.env.example` for environment variables.
- All CI checks must pass; failing checks block merge.
- Add tests for new changes (including edge cases), then run `uv run pytest`.
- Run checks in this order: `uv run ruff format`, `uv run ruff check`, `uv run ty check`, `uv run pytest`.
- Do not add `# type: ignore` or `# ty: ignore`; fix the underlying type issue.
- All 5 checks are enforced in `tests.yml` on push/merge.

## IDENTITY & CONTEXT

- You are an expert Software Architect and Systems Engineer.
- Goal: Zero-defect, root-cause-oriented engineering for bugs; test-driven engineering for new features. Think carefully; no need to rush.
- Code: Write the simplest code possible. Keep the codebase minimal and modular.

## ARCHITECTURE PRINCIPLES (see PLAN.md)

- **Shared utilities**: Extract common logic into shared packages (e.g. `providers/common/`). Do not have one provider import from another provider's utils.
- **DRY**: Extract shared base classes to eliminate duplication. Prefer composition over copy-paste.
- **Encapsulation**: Use accessor methods for internal state (e.g. `set_current_task()`), not direct `_attribute` assignment from outside.
- **Provider-specific config**: Keep provider-specific fields (e.g. `nim_settings`) in provider constructors, not in the base `ProviderConfig`.
- **Dead code**: Remove unused code, legacy systems, and hardcoded values. Use settings/config instead of literals (e.g. `settings.provider_type` not `"nvidia_nim"`).
- **Performance**: Use list accumulation for strings (not `+=` in loops), cache env vars at init, prefer iterative over recursive when stack depth matters.
- **Platform-agnostic naming**: Use generic names (e.g. `PLATFORM_EDIT`) not platform-specific ones (e.g. `TELEGRAM_EDIT`) in shared code.
- **No type ignores**: Do not add `# type: ignore` or `# ty: ignore`. Fix the underlying type issue.
- **Backward compatibility**: When moving modules, add re-exports from old locations so existing imports keep working.

## COGNITIVE WORKFLOW

1. **ANALYZE**: Read relevant files. Do not guess.
2. **PLAN**: Map out the logic. Identify root cause or required changes. Order changes by dependency.
3. **EXECUTE**: Fix the cause, not the symptom. Execute incrementally with clear commits.
4. **VERIFY**: Run ci checks. Confirm the fix via logs or output.
5. **SPECIFICITY**: Do exactly as much as asked; nothing more, nothing less.
6. **PROPAGATION**: Changes impact multiple files; propagate updates correctly.

## SUMMARY STANDARDS

- Summaries must be technical and granular.
- Include: [Files Changed], [Logic Altered], [Verification Method], [Residual Risks] (if no residual risks then say none).

## TOOLS

- Prefer built-in tools (grep, read_file, etc.) over manual workflows. Check tool availability before use.

---

## AVAILABLE PROVIDERS (`providers/` directory)

The project supports **31+ LLM providers**. Each provider implements the `BaseProvider` interface. See [`config/env.example`](config/env.example) for complete configuration.

### Provider Structure

```
providers/
├── base.py                  # BaseProvider abstract class
├── exceptions.py            # Provider-specific exceptions
├── openai_compat.py         # OpenAI-compatible base implementation
├── rate_limit.py            # Rate limiting logic
├── common/                  # Shared utilities (error_mapping, sse_builder, etc.)
│
├── [PAID / FREE CLOUD PROVIDERS]
├── nvidia_nim/              # NVIDIA NIM (40 req/min free) - RECOMMENDED
├── open_router/             # OpenRouter (hundreds of models)
├── groq/                    # Groq (fast inference)
├── together/                # Together AI (open-source models)
├── deepinfra/               # DeepInfra (diverse models)
├── anthropic/               # Anthropic (Claude models)
├── openai/                  # OpenAI (GPT models)
├── fireworks/               # Fireworks (optimized inference)
├── replicate/               # Replicate (model hosting)
├── huggingface/             # HuggingFace (free inference API)
├── cohere/                  # Cohere (Command models)
├── mistral/                 # Mistral (European models)
├── perplexity/              # Perplexity (search-enhanced)
├── cerebras/                # Cerebras (fast inference)
├── sambanova/               # SambaNova (enterprise AI)
├── textsynth/               # TextSynth (budget-friendly)
├── novita/                  # Novita (creative AI, multimodal)
├── ai21/                    # AI21 (Jurassic models)
├── anyscale/                # Anyscale (Ray-based serving)
├── predibase/               # Predibase (fine-tuning)
├── runpod/                  # RunPod (GPU cloud)
├── google/                  # Google (Gemini models)
├── xai/                     # xAI (Grok models)
├── kilo_gateway/            # Kilo Gateway (API gateway)
├── opencode_zen/            # OpenCode Zen (open-source platform)
│
├── [LOCAL PROVIDERS]
├── ollama/                  # Ollama (local models)
├── lmstudio/                # LM Studio (local GUI, no key)
├── llamacpp/                # llama.cpp (lightweight local)
├── vllm/                    # VLLM (local high-throughput)
│
└── custom/                  # Custom OpenAI-compatible API
```

### Provider Configuration Examples

**Recommended providers for daily use:**

NVIDIA NIM offers 40 requests per minute on the free tier and is recommended as a daily driver. You can configure it like this:

```dotenv
NVIDIA_NIM_API_KEY="nvapi-your-key-here"
MODEL_OPUS="nvidia_nim/z-ai/glm4.7"
MODEL_SONNET="nvidia_nim/moonshotai/kimi-k2-thinking"
MODEL_HAIKU="nvidia_nim/stepfun-ai/step-3.5-flash"
NIM_ENABLE_THINKING=true
```

OpenRouter provides hundreds of open-source models with free options:

```dotenv
OPENROUTER_API_KEY="sk-or-your-key-here"
MODEL_OPUS="open_router/deepseek/deepseek-r1-0528:free"
MODEL_SONNET="open_router/arcee-ai/trinity-large-preview:free"
MODEL_HAIKU="open_router/stepfun/step-3.5-flash:free"
```

Groq is known for fast inference with Llama models:

```dotenv
GROQ_API_KEY="gsk_your-key-here"
MODEL="groq/llama3-70b-8192"
```

**Fast inference providers:**

For low-latency applications, consider Fireworks or Together AI:

```dotenv
FIREWORKS_API_KEY="your-key-here"
MODEL="fireworks/accounts/fireworks/models/llama-v3-70b-instruct"
```

```dotenv
TOGETHER_API_KEY="your-key-here"
MODEL="together/meta-llama/Llama-3-70b-chat-hf"
```

**Specialized use cases:**

Perplexity offers search-enhanced responses, while Cohere provides specialized models optimized for specific tasks:

```dotenv
PERPLEXITY_API_KEY="pplx-your-key-here"
MODEL="perplexity/pplx-7b-online"
```

```dotenv
COHERE_API_KEY="your-key-here"
MODEL="cohere/command-r-v1:0"
```

**Enterprise providers:**

For enterprise deployments, SambaNova, Cerebras, and Anyscale offer advanced capabilities:

```dotenv
SAMBANOVA_API_KEY="your-key-here"
MODEL="sambanova/Meta-Llama-3-70B-Instruct"
```

```dotenv
CEREBRAS_API_KEY="your-key-here"
MODEL="cerebras/llama-3.1-70b"
```

```dotenv
ANYSCALE_API_KEY="your-key-here"
MODEL="anyscale/Meta-Llama-3.1-70B-Instruct"
```

**Running models locally without API keys:**

Ollama is the easiest way to get started with local models:

```dotenv
OLLAMA_BASE_URL="http://localhost:11434/v1"
MODEL="ollama/llama3.1:70b"
```

LM Studio provides a graphical interface for managing local models:

```dotenv
LM_STUDIO_BASE_URL="http://localhost:1234/v1"
MODEL="lmstudio/unsloth/MiniMax-M2.5-GGUF"
```

llama.cpp offers lightweight local inference with minimal resource requirements:

```dotenv
LLAMACPP_BASE_URL="http://localhost:8080/v1"
MODEL="llamacpp/local-model"
```

VLLM provides high-throughput serving for local deployments:

```dotenv
VLLM_BASE_URL="http://localhost:8000/v1"
MODEL="vllm/your-large-model"
```

**Additional options:**

For custom OpenAI-compatible APIs:

```dotenv
CUSTOM_API_KEY="your-key"
CUSTOM_BASE_URL="https://your-api.com/v1"
MODEL="custom/your-model-name"
```

You can also use other providers directly:

```dotenv
ANTHROPIC_API_KEY="sk-ant-your-key"
MODEL="anthropic/claude-opus-4-1"
```

```dotenv
OPENAI_API_KEY="sk-your-key"
MODEL="openai/gpt-4o"
```

```dotenv
GOOGLE_API_KEY="your-key"
MODEL="google/gemini-1.5-pro"
```

### Adding a New Provider

All providers follow this structure:

```
providers/new_provider/
├── __init__.py          # Exports Client and Request classes
├── client.py            # Implements BaseProvider interface
└── request.py           # Request/Response data models
```

**Key Files to Understand:**
- [`providers/base.py`](providers/base.py) - `BaseProvider` abstract class
- [`providers/openai_compat.py`](providers/openai_compat.py) - OpenAI-compatible base (most providers extend this)
- [`providers/common/`](providers/common/) - Shared utilities (error mapping, SSE streaming, message conversion)
