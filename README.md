<div align="center">

# 🤖 Free Claude Code

### Use Claude Code CLI & VSCode for free. No Anthropic API key required.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Python 3.14](https://img.shields.io/badge/python-3.14-3776ab.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json&style=for-the-badge)](https://github.com/astral-sh/uv)
[![Tested with Pytest](https://img.shields.io/badge/testing-Pytest-00c0ff.svg?style=for-the-badge)](https://github.com/Alishahryar1/free-claude-code/actions/workflows/tests.yml)
[![Type checking: Ty](https://img.shields.io/badge/type%20checking-ty-ffcc00.svg?style=for-the-badge)](https://pypi.org/project/ty/)
[![Code style: Ruff](https://img.shields.io/badge/code%20formatting-ruff-f5a623.svg?style=for-the-badge)](https://github.com/astral-sh/ruff)
[![Logging: Loguru](https://img.shields.io/badge/logging-loguru-4ecdc4.svg?style=for-the-badge)](https://github.com/Delgan/loguru)

A lightweight proxy that routes Claude Code's Anthropic API calls to **NVIDIA NIM** (40 req/min free), **OpenRouter** (hundreds of models), **LM Studio** (fully local), or **llama.cpp** (local with Anthropic endpoints).

[Quick Start](#quick-start) · [Providers](#providers) · [Discord Bot](#discord-bot) · [Configuration](#configuration) · [Development](#development) · [Contributing](#contributing)

---

</div>

<div align="center">
  <img src="pic.png" alt="Free Claude Code in action" width="700">
  <p><em>Claude Code running via NVIDIA NIM, completely free</em></p>
</div>

## Features

| Feature                    | Description                                                                                     |
| -------------------------- | ----------------------------------------------------------------------------------------------- |
| **Zero Cost**              | 40 req/min free on NVIDIA NIM. Free models on OpenRouter. Fully local with LM Studio            |
| **Drop-in Replacement**    | Set 2 env vars. No modifications to Claude Code CLI or VSCode extension needed                  |
| **31+ Providers**          | NVIDIA NIM, OpenRouter, Groq, Together, DeepInfra, HuggingFace, Ollama, and 25+ more           |
| **Per-Model Mapping**      | Route Opus / Sonnet / Haiku to different models and providers. Mix providers freely             |
| **Thinking Token Support** | Parses `<think>` tags and `reasoning_content` into native Claude thinking blocks                |
| **Heuristic Tool Parser**  | Models outputting tool calls as text are auto-parsed into structured tool use                   |
| **Request Optimization**   | 5 categories of trivial API calls intercepted locally, saving quota and latency                 |
| **Smart Rate Limiting**    | Proactive rolling-window throttle + reactive 429 exponential backoff + optional concurrency cap |
| **Discord / Telegram Bot** | Remote autonomous coding with tree-based threading, session persistence, and live progress      |
| **Subagent Control**       | Task tool interception forces `run_in_background=False`. No runaway subagents                   |
| **Extensible**             | Clean `BaseProvider` and `MessagingPlatform` ABCs. Add new providers or platforms easily        |

## Quick Start

### Prerequisites

1. Get an API key (or use LM Studio / llama.cpp locally):
   - **NVIDIA NIM**: [build.nvidia.com/settings/api-keys](https://build.nvidia.com/settings/api-keys)
   - **OpenRouter**: [openrouter.ai/keys](https://openrouter.ai/keys)
   - **LM Studio**: No API key needed. Run locally with [LM Studio](https://lmstudio.ai)
   - **llama.cpp**: No API key needed. Run `llama-server` locally.
2. Install [Claude Code](https://github.com/anthropics/claude-code)

### Install `uv`
```bash
# Install uv (required to run the project)
pip install uv
```
If uv is already installed, run uv self update to get the latest version.

### Clone & Configure

```bash
git clone https://github.com/Alishahryar1/free-claude-code.git
cd free-claude-code
cp .env.example .env
```

Choose your provider and edit `.env`:

<details>
<summary><b>NVIDIA NIM</b> (40 req/min free, recommended)</summary>

```dotenv
NVIDIA_NIM_API_KEY="nvapi-your-key-here"

MODEL_OPUS="nvidia_nim/z-ai/glm4.7"
MODEL_SONNET="nvidia_nim/moonshotai/kimi-k2-thinking"
MODEL_HAIKU="nvidia_nim/stepfun-ai/step-3.5-flash"
MODEL="nvidia_nim/z-ai/glm4.7"                     # fallback

# Enable for thinking models (kimi, nemotron). Leave false for others (e.g. Mistral).
NIM_ENABLE_THINKING=true
```

</details>

<details>
<summary><b>OpenRouter</b> (hundreds of models)</summary>

```dotenv
OPENROUTER_API_KEY="sk-or-your-key-here"

MODEL_OPUS="open_router/deepseek/deepseek-r1-0528:free"
MODEL_SONNET="open_router/openai/gpt-oss-120b:free"
MODEL_HAIKU="open_router/stepfun/step-3.5-flash:free"
MODEL="open_router/stepfun/step-3.5-flash:free"     # fallback
```

</details>

<details>
<summary><b>LM Studio</b> (fully local, no API key)</summary>

```dotenv
MODEL_OPUS="lmstudio/unsloth/MiniMax-M2.5-GGUF"
MODEL_SONNET="lmstudio/unsloth/Qwen3.5-35B-A3B-GGUF"
MODEL_HAIKU="lmstudio/unsloth/GLM-4.7-Flash-GGUF"
MODEL="lmstudio/unsloth/GLM-4.7-Flash-GGUF"         # fallback
```

</details>

<details>
<summary><b>llama.cpp</b> (fully local, no API key)</summary>

```dotenv
LLAMACPP_BASE_URL="http://localhost:8080/v1"

MODEL_OPUS="llamacpp/local-model"
MODEL_SONNET="llamacpp/local-model"
MODEL_HAIKU="llamacpp/local-model"
MODEL="llamacpp/local-model"
```

</details>

<details>
<summary><b>Mix providers</b></summary>

Each `MODEL_*` variable can use a different provider. `MODEL` is the fallback for unrecognized Claude models.

```dotenv
NVIDIA_NIM_API_KEY="nvapi-your-key-here"
OPENROUTER_API_KEY="sk-or-your-key-here"

MODEL_OPUS="nvidia_nim/moonshotai/kimi-k2.5"
MODEL_SONNET="open_router/deepseek/deepseek-r1-0528:free"
MODEL_HAIKU="lmstudio/unsloth/GLM-4.7-Flash-GGUF"
MODEL="nvidia_nim/z-ai/glm4.7"                      # fallback
```

</details>

<details>
<summary><b>Groq</b> (fast inference)</summary>

```dotenv
GROQ_API_KEY="gsk_your-key-here"

MODEL_OPUS="groq/llama3-70b-8192"
MODEL_SONNET="groq/llama3-8b-8192"
MODEL_HAIKU="groq/mixtral-8x7b-32768"
MODEL="groq/llama3-8b-8192"                          # fallback
```

</details>

<details>
<summary><b>Together AI</b> (open-source models)</summary>

```dotenv
TOGETHER_API_KEY="your-key-here"

MODEL_OPUS="together/meta-llama/Llama-3-70b-chat-hf"
MODEL_SONNET="together/mistralai/Mixtral-8x7B-Instruct-v0.1"
MODEL_HAIKU="together/microsoft/WizardLM-2-8x22B"
MODEL="together/meta-llama/Llama-3-8b-chat-hf"       # fallback
```

</details>

<details>
<summary><b>DeepInfra</b> (diverse models)</summary>

```dotenv
DEEPINFRA_API_KEY="your-key-here"

MODEL_OPUS="deepinfra/meta-llama/Meta-Llama-3-70B-Instruct"
MODEL_SONNET="deepinfra/mistralai/Mistral-7B-Instruct-v0.1"
MODEL_HAIKU="deepinfra/Qwen/Qwen2-72B-Instruct"
MODEL="deepinfra/meta-llama/Meta-Llama-3-8B-Instruct" # fallback
```

</details>

<details>
<summary><b>Custom provider</b> (any OpenAI-compatible API)</summary>

```dotenv
CUSTOM_API_KEY="your-api-key"
CUSTOM_BASE_URL="https://your-custom-endpoint.com/v1"

MODEL_OPUS="custom/your-opus-model"
MODEL_SONNET="custom/your-sonnet-model"
MODEL_HAIKU="custom/your-haiku-model"
MODEL="custom/your-fallback-model"                   # fallback
```

</details>

<details>
<summary><b>HuggingFace (Free) - Inference API</b></summary>

HuggingFace requires the model ID to be part of the base URL:

```dotenv
HUGGINGFACE_API_KEY="your-huggingface-token"
# Set HUGGINGFACE_BASE_URL with your model ID
HUGGINGFACE_BASE_URL="https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-hf/v1"

MODEL="huggingface/meta-llama/Llama-2-7b-hf"
```

Available models (examples - use any HF model with chat API support):
- `meta-llama/Llama-2-7b-hf`
- `mistralai/Mistral-7B-Instruct-v0.1`
- `google/flan-t5-large`

</details>

<details>
<summary><b>Ollama (Free, Local)</b></summary>

```dotenv
OLLAMA_BASE_URL="http://localhost:11434/v1"

MODEL_OPUS="ollama/llama3.1:70b"
MODEL_SONNET="ollama/llama3.1:8b"
MODEL_HAIKU="ollama/llama3.1:8b"
MODEL="ollama/llama3.1:8b"                            # fallback
```

</details>

<details>
<summary><b>VLLM (Free, Local)</b></summary>

```dotenv
VLLM_BASE_URL="http://localhost:8000/v1"

MODEL_OPUS="vllm/your-large-model"
MODEL_SONNET="vllm/your-medium-model"
MODEL_HAIKU="vllm/your-small-model"
MODEL="vllm/your-fallback-model"                      # fallback
```

</details>

<details>
<summary><b>Optional Authentication</b> (restrict access to your proxy)</summary>

Set `ANTHROPIC_AUTH_TOKEN` in `.env` to require clients to authenticate:

```dotenv
ANTHROPIC_AUTH_TOKEN="your-secret-token-here"
```

**How it works:**
- If `ANTHROPIC_AUTH_TOKEN` is empty (default), no authentication is required (backward compatible)
- If set, clients must provide the same token via the `ANTHROPIC_AUTH_TOKEN` header
- The `claude-pick` script automatically reads the token from `.env` if configured

**Example usage:**
```bash
# With authentication
ANTHROPIC_AUTH_TOKEN="your-secret-token-here" \
ANTHROPIC_BASE_URL="http://localhost:8082" claude

# claude-pick automatically uses the configured token
claude-pick
```

Use this feature if:
- Running the proxy on a public network
- Sharing the server with others but restricting access
- Wanting an additional layer of security

</details>

### Run It

**Terminal 1:** Start the proxy server:

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8082
```

**Terminal 2:** Run Claude Code:

#### Powershell
```powershell
$env:ANTHROPIC_AUTH_TOKEN="freecc"; $env:ANTHROPIC_BASE_URL="http://localhost:8082"; claude
```
#### Bash
```bash
ANTHROPIC_AUTH_TOKEN="freecc" ANTHROPIC_BASE_URL="http://localhost:8082" claude
```

That's it! Claude Code now uses your configured provider for free.

<details>
<summary><b>VSCode Extension Setup</b></summary>

1. Start the proxy server (same as above).
2. Open Settings (`Ctrl + ,`) and search for `claude-code.environmentVariables`.
3. Click **Edit in settings.json** and add:

```json
"claudeCode.environmentVariables": [
  { "name": "ANTHROPIC_BASE_URL", "value": "http://localhost:8082" },
  { "name": "ANTHROPIC_AUTH_TOKEN", "value": "freecc" }
]
```

4. Reload extensions.
5. **If you see the login screen**: Click **Anthropic Console**, then authorize. The extension will start working. You may be redirected to buy credits in the browser; ignore it — the extension already works.

To switch back to Anthropic models, comment out the added block and reload extensions.

</details>

<details>
<summary><b>Multi-Model Support (Model Picker)</b></summary>

`claude-pick` is an interactive model selector that lets you choose any model from your active provider each time you launch Claude, without editing `MODEL` in `.env`.

https://github.com/user-attachments/assets/9a33c316-90f8-4418-9650-97e7d33ad645

**1. Install [fzf](https://github.com/junegunn/fzf)**:

```bash
brew install fzf        # macOS/Linux
```

**2. Add the alias to `~/.zshrc` or `~/.bashrc`:**

```bash
alias claude-pick="/absolute/path/to/free-claude-code/claude-pick"
```

Then reload your shell (`source ~/.zshrc` or `source ~/.bashrc`) and run `claude-pick`.

**Or use a fixed model alias** (no picker needed):

```bash
alias claude-kimi='ANTHROPIC_BASE_URL="http://localhost:8082" ANTHROPIC_AUTH_TOKEN="freecc:moonshotai/kimi-k2.5" claude'
```

</details>

### Install as a Package (no clone needed)

```bash
uv tool install git+https://github.com/Alishahryar1/free-claude-code.git
fcc-init        # creates ~/.config/free-claude-code/.env from the built-in template
```

Edit `~/.config/free-claude-code/.env` with your API keys and model names, then:

```bash
free-claude-code    # starts the server
```

> To update: `uv tool upgrade free-claude-code`

---

## How It Works

```
┌─────────────────┐        ┌──────────────────────┐        ┌──────────────────┐
│  Claude Code    │───────>│  Free Claude Code    │───────>│  LLM Provider    │
│  CLI / VSCode   │<───────│  Proxy (:8082)       │<───────│  NIM / OR / LMS  │
└─────────────────┘        └──────────────────────┘        └──────────────────┘
   Anthropic API                                             OpenAI-compatible
   format (SSE)                                             format (SSE)
```

- **Transparent proxy**: Claude Code sends standard Anthropic API requests; the proxy forwards them to your configured provider
- **Per-model routing**: Opus / Sonnet / Haiku requests resolve to their model-specific backend, with `MODEL` as fallback
- **Request optimization**: 5 categories of trivial requests (quota probes, title generation, prefix detection, suggestions, filepath extraction) are intercepted and responded to locally without using API quota
- **Format translation**: Requests are translated from Anthropic format to the provider's OpenAI-compatible format and streamed back
- **Thinking tokens**: `<think>` tags and `reasoning_content` fields are converted into native Claude thinking blocks

---

## Providers

| Provider       | Cost         | Rate Limit | Best For                             |
| -------------- | ------------ | ---------- | ------------------------------------ |
| **NVIDIA NIM** | Free         | 40 req/min | Daily driver, generous free tier     |
| **OpenRouter** | Free / Paid  | Varies     | Model variety, fallback options      |
| **Groq**       | Free / Paid  | Varies     | Fast inference, Llama models         |
| **Together AI**| Free / Paid  | Varies     | Open-source models, fine-tuning      |
| **DeepInfra**  | Free / Paid  | Varies     | Diverse model selection              |
| **HuggingFace**| Free         | Varies     | Free inference API, research models  |
| **Replicate**  | Free / Paid  | Varies     | Model hosting, custom deployments    |
| **Fireworks**  | Free / Paid  | Varies     | Optimized inference, enterprise      |
| **Ollama**     | Free (local) | Unlimited  | Local models, easy setup             |
| **LM Studio**  | Free (local) | Unlimited  | Privacy, offline use, no rate limits |
| **llama.cpp**  | Free (local) | Unlimited  | Lightweight local inference engine   |
| **Anyscale**   | Free / Paid  | Varies     | Ray-based serving, scaling           |
| **Cohere**     | Free / Paid  | Varies     | Command models, RAG optimization     |
| **AI21**       | Free / Paid  | Varies     | Jurassic models, creative writing    |
| **Mistral**    | Free / Paid  | Varies     | European models, privacy-focused     |
| **Perplexity** | Free / Paid  | Varies     | Search-enhanced responses            |
| **Cerebras**   | Free / Paid  | Varies     | Fast inference, wafer-scale chips    |
| **SambaNova**  | Free / Paid  | Varies     | Enterprise AI, custom models         |
| **TextSynth**  | Free / Paid  | Varies     | Budget-friendly inference            |
| **Novita**     | Free / Paid  | Varies     | Creative AI, multimodal             |
| **Predibase**  | Free / Paid  | Varies     | Fine-tuning, model management        |
| **RunPod**     | Free / Paid  | Varies     | GPU cloud, serverless inference     |
| **VLLM**       | Free (local) | Unlimited  | High-throughput local serving        |
| **Google**     | Free / Paid  | Varies     | PaLM, Gemini models                  |
| **xAI**        | Free / Paid  | Varies     | Grok models, truth-seeking           |
| **Kilo Gateway** | Free / Paid | Varies     | API gateway aggregator               |
| **OpenCode Zen** | Free / Paid | Varies     | Open-source model platform           |
| **Anthropic**  | Paid         | Varies     | Claude models, safety-focused        |
| **OpenAI**     | Paid         | Varies     | GPT models, industry standard        |
| **Custom**     | Varies       | Varies     | User-defined OpenAI-compatible APIs  |

Models use a prefix format: `provider_prefix/model/name`. An invalid prefix causes an error.

| Provider   | `MODEL` prefix    | API Key Variable     | Default Base URL              |
| ---------- | ----------------- | -------------------- | ----------------------------- |
| NVIDIA NIM | `nvidia_nim/...`  | `NVIDIA_NIM_API_KEY` | `integrate.api.nvidia.com/v1` |
| OpenRouter | `open_router/...` | `OPENROUTER_API_KEY` | `openrouter.ai/api/v1`        |
| Groq       | `groq/...`        | `GROQ_API_KEY`       | `api.groq.com/openai/v1`      |
| Together AI| `together/...`    | `TOGETHER_API_KEY`   | `api.together.xyz/v1`         |
| DeepInfra  | `deepinfra/...`   | `DEEPINFRA_API_KEY`  | `api.deepinfra.com/v1/openai` |
| HuggingFace| `huggingface/...` | `HUGGINGFACE_API_KEY`| `api-inference.huggingface.co/models` |
| Replicate  | `replicate/...`   | `REPLICATE_API_KEY`  | `api.replicate.com/v1`        |
| Fireworks  | `fireworks/...`   | `FIREWORKS_API_KEY`  | `api.fireworks.ai/v1`         |
| Anyscale   | `anyscale/...`    | `ANYSCALE_API_KEY`   | `api.anyscale.com/v1`         |
| Cohere     | `cohere/...`      | `COHERE_API_KEY`     | `api.cohere.ai/v1`            |
| AI21       | `ai21/...`        | `AI21_API_KEY`       | `api.ai21.com/v1`             |
| Mistral    | `mistral/...`     | `MISTRAL_API_KEY`    | `api.mistral.ai/v1`           |
| Perplexity | `perplexity/...`  | `PERPLEXITY_API_KEY` | `api.perplexity.ai/v1`        |
| Cerebras   | `cerebras/...`    | `CEREBRAS_API_KEY`   | `api.cerebras.ai/v1`          |
| SambaNova  | `sambanova/...`   | `SAMBANOVA_API_KEY`  | `api.sambanova.ai/v1`         |
| TextSynth  | `textsynth/...`   | `TEXTSYNTH_API_KEY`  | `api.textsynth.com/v1`        |
| Novita     | `novita/...`      | `NOVITA_API_KEY`     | `api.novita.ai/v1`            |
| Predibase  | `predibase/...`   | `PREDIBASE_API_KEY`  | `api.predibase.com/v1`        |
| RunPod     | `runpod/...`      | `RUNPOD_API_KEY`     | `api.runpod.ai/v1`            |
| Google     | `google/...`      | `GOOGLE_API_KEY`     | `generativelanguage.googleapis.com/v1beta/openai/` |
| xAI        | `xai/...`         | `XAI_API_KEY`        | `api.x.ai/v1`                 |
| Kilo Gateway | `kilo_gateway/...` | `KILO_GATEWAY_API_KEY` | `api.kilogateway.com/v1`      |
| OpenCode Zen | `opencode_zen/...` | `OPENCODE_ZEN_API_KEY` | `api.opencodezen.com/v1`      |
| Anthropic  | `anthropic/...`   | `ANTHROPIC_API_KEY`  | `api.anthropic.com/v1`        |
| OpenAI     | `openai/...`      | `OPENAI_API_KEY`     | `api.openai.com/v1`           |
| Ollama     | `ollama/...`      | (none)               | `localhost:11434/v1`          |
| LM Studio  | `lmstudio/...`    | (none)               | `localhost:1234/v1`           |
| llama.cpp  | `llamacpp/...`    | (none)               | `localhost:8080/v1`           |
| VLLM       | `vllm/...`        | (none)               | User-defined                  |
| Custom     | `custom/...`      | `CUSTOM_API_KEY`     | User-defined                  |

<details>
<summary><b>NVIDIA NIM models</b></summary>

Popular models (full list in [`nvidia_nim_models.json`](nvidia_nim_models.json)):

- `nvidia_nim/minimaxai/minimax-m2.5`
- `nvidia_nim/qwen/qwen3.5-397b-a17b`
- `nvidia_nim/z-ai/glm5`
- `nvidia_nim/moonshotai/kimi-k2.5`
- `nvidia_nim/stepfun-ai/step-3.5-flash`

Browse: [build.nvidia.com](https://build.nvidia.com/explore/discover) · Update list: `curl "https://integrate.api.nvidia.com/v1/models" > nvidia_nim_models.json`

</details>

<details>
<summary><b>OpenRouter models</b></summary>

Popular free models:

- `open_router/arcee-ai/trinity-large-preview:free`
- `open_router/stepfun/step-3.5-flash:free`
- `open_router/deepseek/deepseek-r1-0528:free`
- `open_router/openai/gpt-oss-120b:free`

Browse: [openrouter.ai/models](https://openrouter.ai/models) · [Free models](https://openrouter.ai/collections/free-models)

</details>

<details>
<summary><b>LM Studio models</b></summary>

Run models locally with [LM Studio](https://lmstudio.ai). Load a model in the Chat or Developer tab, then set `MODEL` to its identifier.

Examples with native tool-use support:

- `LiquidAI/LFM2-24B-A2B-GGUF`
- `unsloth/MiniMax-M2.5-GGUF`
- `unsloth/GLM-4.7-Flash-GGUF`
- `unsloth/Qwen3.5-35B-A3B-GGUF`

Browse: [model.lmstudio.ai](https://model.lmstudio.ai)

</details>

<details>
<summary><b>llama.cpp models</b></summary>

Run models locally using `llama-server`. Ensure you have a tool-capable GGUF. Set `MODEL` to whatever arbitrary name you'd like (e.g. `llamacpp/my-model`), as `llama-server` ignores the model name when run via `/v1/messages`.

See the Unsloth docs for detailed instructions and capable models:
[https://unsloth.ai/docs/models/qwen3.5#qwen3.5-small-0.8b-2b-4b-9b](https://unsloth.ai/docs/models/qwen3.5#qwen3.5-small-0.8b-2b-4b-9b)

</details>

<details>
<summary><b>Groq models</b></summary>

Popular models available on Groq:

- `groq/llama3-8b-8192`
- `groq/llama3-70b-8192`
- `groq/mixtral-8x7b-32768`
- `groq/gemma-7b-it`

Browse: [console.groq.com/docs/models](https://console.groq.com/docs/models)

</details>

<details>
<summary><b>Together AI models</b></summary>

Popular models on Together AI:

- `together/meta-llama/Llama-3-70b-chat-hf`
- `together/mistralai/Mixtral-8x7B-Instruct-v0.1`
- `together/microsoft/WizardLM-2-8x22B`

Browse: [api.together.xyz/models](https://api.together.xyz/models)

</details>

<details>
<summary><b>DeepInfra models</b></summary>

Popular models on DeepInfra:

- `deepinfra/meta-llama/Meta-Llama-3-70B-Instruct`
- `deepinfra/mistralai/Mistral-7B-Instruct-v0.1`
- `deepinfra/Qwen/Qwen2-72B-Instruct`

Browse: [deepinfra.com/models](https://deepinfra.com/models)

</details>

<details>
<summary><b>Custom provider</b></summary>

Configure any OpenAI-compatible API endpoint:

```dotenv
CUSTOM_API_KEY="your-api-key"
CUSTOM_BASE_URL="https://your-api-endpoint.com/v1"
MODEL="custom/your-model-name"
```

The custom provider supports any OpenAI-compatible API that follows the standard chat completions format.

</details>

---

## Discord Bot

Control Claude Code remotely from Discord (or Telegram). Send tasks, watch live progress, and manage multiple concurrent sessions.

**Capabilities:**

- Tree-based message threading: reply to a message to fork the conversation
- Session persistence across server restarts
- Live streaming of thinking tokens, tool calls, and results
- Unlimited concurrent Claude CLI sessions (concurrency controlled by `PROVIDER_MAX_CONCURRENCY`)
- Voice notes: send voice messages; they are transcribed and processed as regular prompts
- Commands: `/stop` (cancel a task; reply to a message to stop only that task), `/clear` (reset all sessions, or reply to clear a branch), `/stats`

### Setup

1. **Create a Discord Bot**: Go to [Discord Developer Portal](https://discord.com/developers/applications), create an application, add a bot, and copy the token. Enable **Message Content Intent** under Bot settings.

2. **Edit `.env`:**

```dotenv
MESSAGING_PLATFORM="discord"
DISCORD_BOT_TOKEN="your_discord_bot_token"
ALLOWED_DISCORD_CHANNELS="123456789,987654321"
```

> Enable Developer Mode in Discord (Settings → Advanced), then right-click a channel and "Copy ID". Comma-separate multiple channels. If empty, no channels are allowed.

3. **Configure the workspace** (where Claude will operate):

```dotenv
CLAUDE_WORKSPACE="./agent_workspace"
ALLOWED_DIR="C:/Users/yourname/projects"
```

4. **Start the server:**

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8082
```

5. **Invite the bot** via OAuth2 URL Generator (scopes: `bot`, permissions: Read Messages, Send Messages, Manage Messages, Read Message History).

### Telegram

Set `MESSAGING_PLATFORM=telegram` and configure:

```dotenv
TELEGRAM_BOT_TOKEN="123456789:ABCdefGHIjklMNOpqrSTUvwxYZ"
ALLOWED_TELEGRAM_USER_ID="your_telegram_user_id"
```

Get a token from [@BotFather](https://t.me/BotFather); find your user ID via [@userinfobot](https://t.me/userinfobot).

### Voice Notes

Send voice messages on Discord or Telegram; they are transcribed and processed as regular prompts.

| Backend                     | Description                                                                                                   | API Key              |
| --------------------------- | ------------------------------------------------------------------------------------------------------------- | -------------------- |
| **Local Whisper** (default) | [Hugging Face Whisper](https://huggingface.co/openai/whisper-large-v3-turbo) — free, offline, CUDA compatible | not required         |
| **NVIDIA NIM**              | Whisper/Parakeet models via gRPC                                                                              | `NVIDIA_NIM_API_KEY` |

**Install the voice extras:**

```bash
# If you cloned the repo:
uv sync --extra voice_local          # Local Whisper
uv sync --extra voice                # NVIDIA NIM
uv sync --extra voice --extra voice_local  # Both

# If you installed as a package (no clone):
uv tool install "free-claude-code[voice_local] @ git+https://github.com/Alishahryar1/free-claude-code.git"
uv tool install "free-claude-code[voice] @ git+https://github.com/Alishahryar1/free-claude-code.git"
uv tool install "free-claude-code[voice,voice_local] @ git+https://github.com/Alishahryar1/free-claude-code.git"
```

Configure via `WHISPER_DEVICE` (`cpu` | `cuda` | `nvidia_nim`) and `WHISPER_MODEL`. See the [Configuration](#configuration) table for all voice variables and supported model values.

---

## Configuration

### Core

| Variable             | Description                                                           | Default                                           |
| -------------------- | --------------------------------------------------------------------- | ------------------------------------------------- |
| `MODEL`              | Fallback model (`provider/model/name` format; invalid prefix → error) | `nvidia_nim/stepfun-ai/step-3.5-flash`            |
| `MODEL_OPUS`         | Model for Claude Opus requests (falls back to `MODEL`)                | `nvidia_nim/z-ai/glm4.7`                          |
| `MODEL_SONNET`       | Model for Claude Sonnet requests (falls back to `MODEL`)              | `open_router/arcee-ai/trinity-large-preview:free` |
| `MODEL_HAIKU`        | Model for Claude Haiku requests (falls back to `MODEL`)               | `open_router/stepfun/step-3.5-flash:free`         |
| `NVIDIA_NIM_API_KEY`    | NVIDIA API key                                                        | required for NIM                                  |
| `NIM_ENABLE_THINKING`   | Send `chat_template_kwargs` + `reasoning_budget` on NIM requests. Enable for thinking models (kimi, nemotron); leave `false` for others (e.g. Mistral) | `false` |
| `OPENROUTER_API_KEY` | OpenRouter API key                                                    | required for OpenRouter                           |
| `GROQ_API_KEY`         | Groq API key                                                          | required for Groq                                 |
| `TOGETHER_API_KEY`     | Together AI API key                                                   | required for Together AI                          |
| `DEEPINFRA_API_KEY`    | DeepInfra API key                                                     | required for DeepInfra                            |
| `CUSTOM_API_KEY`       | Custom provider API key                                               | required for Custom provider                      |
| `CUSTOM_BASE_URL`      | Custom provider base URL                                              | required for Custom provider                      |
| `HUGGINGFACE_API_KEY`  | HuggingFace API token                                                 | required for HuggingFace                          |
| `REPLICATE_API_KEY`    | Replicate API key                                                     | required for Replicate                            |
| `FIREWORKS_API_KEY`    | Fireworks API key                                                     | required for Fireworks                            |
| `ANYSCALE_API_KEY`     | Anyscale API key                                                      | required for Anyscale                             |
| `COHERE_API_KEY`       | Cohere API key                                                        | required for Cohere                               |
| `AI21_API_KEY`         | AI21 API key                                                          | required for AI21                                 |
| `MISTRAL_API_KEY`      | Mistral API key                                                       | required for Mistral                              |
| `PERPLEXITY_API_KEY`   | Perplexity API key                                                    | required for Perplexity                           |
| `CEREBRAS_API_KEY`     | Cerebras API key                                                      | required for Cerebras                             |
| `SAMBANOVA_API_KEY`    | SambaNova API key                                                     | required for SambaNova                            |
| `TEXTSYNTH_API_KEY`    | TextSynth API key                                                     | required for TextSynth                            |
| `NOVITA_API_KEY`       | Novita API key                                                        | required for Novita                               |
| `PREDIBASE_API_KEY`    | Predibase API key                                                     | required for Predibase                            |
| `RUNPOD_API_KEY`       | RunPod API key                                                        | required for RunPod                               |
| `GOOGLE_API_KEY`       | Google AI API key                                                     | required for Google                               |
| `XAI_API_KEY`          | xAI API key                                                           | required for xAI                                  |
| `ANTHROPIC_API_KEY`    | Anthropic API key                                                     | required for Anthropic                            |
| `OPENAI_API_KEY`       | OpenAI API key                                                        | required for OpenAI                               |
| `OLLAMA_BASE_URL`      | Ollama server URL                                                     | `http://localhost:11434/v1`                       |
| `VLLM_BASE_URL`        | VLLM server URL                                                       | required for VLLM                                 |
| `LM_STUDIO_BASE_URL`   | LM Studio server URL                                                  | `http://localhost:1234/v1`                        |
| `LLAMACPP_BASE_URL`    | llama.cpp server URL                                                  | `http://localhost:8080/v1`                        |

### Rate Limiting & Timeouts

| Variable                   | Description                               | Default |
| -------------------------- | ----------------------------------------- | ------- |
| `PROVIDER_RATE_LIMIT`      | LLM API requests per window               | `40`    |
| `PROVIDER_RATE_WINDOW`     | Rate limit window (seconds)               | `60`    |
| `PROVIDER_MAX_CONCURRENCY` | Max simultaneous open provider streams    | `5`     |
| `HTTP_READ_TIMEOUT`        | Read timeout for provider requests (s)    | `120`   |
| `HTTP_WRITE_TIMEOUT`       | Write timeout for provider requests (s)   | `10`    |
| `HTTP_CONNECT_TIMEOUT`     | Connect timeout for provider requests (s) | `2`     |

### Messaging & Voice

| Variable                   | Description                                                                                                                                                        | Default             |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------- |
| `MESSAGING_PLATFORM`       | `discord` or `telegram`                                                                                                                                            | `discord`           |
| `DISCORD_BOT_TOKEN`        | Discord bot token                                                                                                                                                  | `""`                |
| `ALLOWED_DISCORD_CHANNELS` | Comma-separated channel IDs (empty = none allowed)                                                                                                                 | `""`                |
| `TELEGRAM_BOT_TOKEN`       | Telegram bot token                                                                                                                                                 | `""`                |
| `ALLOWED_TELEGRAM_USER_ID` | Allowed Telegram user ID                                                                                                                                           | `""`                |
| `CLAUDE_WORKSPACE`         | Directory where the agent operates                                                                                                                                 | `./agent_workspace` |
| `ALLOWED_DIR`              | Allowed directories for the agent                                                                                                                                  | `""`                |
| `MESSAGING_RATE_LIMIT`     | Messaging messages per window                                                                                                                                      | `1`                 |
| `MESSAGING_RATE_WINDOW`    | Messaging window (seconds)                                                                                                                                         | `1`                 |
| `VOICE_NOTE_ENABLED`       | Enable voice note handling                                                                                                                                         | `true`              |
| `WHISPER_DEVICE`           | `cpu` \| `cuda` \| `nvidia_nim`                                                                                                                                    | `cpu`               |
| `WHISPER_MODEL`            | Whisper model (local: `tiny`/`base`/`small`/`medium`/`large-v2`/`large-v3`/`large-v3-turbo`; NIM: `openai/whisper-large-v3`, `nvidia/parakeet-ctc-1.1b-asr`, etc.) | `base`              |
| `HF_TOKEN`                 | Hugging Face token for faster downloads (local Whisper, optional)                                                                                                  | —                   |

<details>
<summary><b>Advanced: Request optimization flags</b></summary>

These are enabled by default and intercept trivial Claude Code requests locally to save API quota.

| Variable                          | Description                    | Default |
| --------------------------------- | ------------------------------ | ------- |
| `FAST_PREFIX_DETECTION`           | Enable fast prefix detection   | `true`  |
| `ENABLE_NETWORK_PROBE_MOCK`       | Mock network probe requests    | `true`  |
| `ENABLE_TITLE_GENERATION_SKIP`    | Skip title generation requests | `true`  |
| `ENABLE_SUGGESTION_MODE_SKIP`     | Skip suggestion mode requests  | `true`  |
| `ENABLE_FILEPATH_EXTRACTION_MOCK` | Mock filepath extraction       | `true`  |

</details>

See [`.env.example`](.env.example) for all supported parameters.

---

## Development

### Project Structure

```
free-claude-code/
├── server.py              # Entry point
├── api/                   # FastAPI routes, request detection, optimization handlers
├── providers/             # BaseProvider, OpenAICompatibleProvider, NIM, OpenRouter, LM Studio, llamacpp
│   └── common/            # Shared utils (SSE builder, message converter, parsers, error mapping)
├── messaging/             # MessagingPlatform ABC + Discord/Telegram bots, session management
├── config/                # Settings, NIM config, logging
├── cli/                   # CLI session and process management
└── tests/                 # Pytest test suite
```

### Commands

```bash
uv run ruff format     # Format code
uv run ruff check      # Lint
uv run ty check        # Type checking
uv run pytest          # Run tests
```

### Extending

**Adding an OpenAI-compatible provider** (Groq, Together AI, etc.) — extend `OpenAICompatibleProvider`:

```python
from providers.openai_compat import OpenAICompatibleProvider
from providers.base import ProviderConfig

class MyProvider(OpenAICompatibleProvider):
    def __init__(self, config: ProviderConfig):
        super().__init__(config, provider_name="MYPROVIDER",
                         base_url="https://api.example.com/v1", api_key=config.api_key)
```

**Adding a fully custom provider** — extend `BaseProvider` directly and implement `stream_response()`.

**Adding a messaging platform** — extend `MessagingPlatform` in `messaging/` and implement `start()`, `stop()`, `send_message()`, `edit_message()`, and `on_message()`.

---

## Contributing

- Report bugs or suggest features via [Issues](https://github.com/Alishahryar1/free-claude-code/issues)
- Add new LLM providers (Groq, Together AI, etc.)
- Add new messaging platforms (Slack, etc.)
- Improve test coverage
- Not accepting Docker integration PRs for now

```bash
git checkout -b my-feature
uv run ruff format && uv run ruff check && uv run ty check && uv run pytest
# Open a pull request
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

Built with [FastAPI](https://fastapi.tiangolo.com/), [OpenAI Python SDK](https://github.com/openai/openai-python), [discord.py](https://github.com/Rapptz/discord.py), and [python-telegram-bot](https://python-telegram-bot.org/).
