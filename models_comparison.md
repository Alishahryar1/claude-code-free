# Model Comparison Table

Models discussed in recent thread with performance characteristics for agentic coding.

| Model | Provider | Use Case | Anthropic Analogy | Parameters | Est. Speed | Thinking/Reasoning | Tool Use | Notes |
|-------|----------|----------|-------------------|------------|------------|-------------------|----------|-------|
| **z-ai/glm5** | NVIDIA NIM | Deep reasoning, complex analysis | Opus/Sonnet | ~120B | 15-25 tok/s | ✓ (via NIM_ENABLE_THINKING) | ✓ | Current default. Strong but slow. Bilingual (ZH/EN). |
| **stepfun-ai/step-3.5-flash** | NVIDIA NIM | Quick iterations, fast responses | Haiku | ~7-10B | 40-60 tok/s | ✗ | ✓ | Current Haiku. Optimized for speed over depth. |
| **moonshotai/kimi-k2.5** | NVIDIA NIM | Long-context reasoning | Opus | ~130B | 10-20 tok/s | ✓ (implicit) | ✓ | Strong reasoning, good for large codebases. Slower. |
| **moonshotai/kimi-k2-thinking** | NVIDIA NIM | Chain-of-thought reasoning | Opus (with thinking) | ~130B | 8-15 tok/s | ✓✓ (explicit CoT) | ✓ | Explicit reasoning steps. Very slow but thorough. |
| **minimaxai/minimax-m2.5** | NVIDIA NIM | General purpose | Sonnet | ~100B | 12-22 tok/s | ✗ | ✓ | Older generation. Not recommended. |
| **meta/llama-3.3-70b-instruct** | NVIDIA NIM | Balanced quality/speed | Sonnet | 70B | 30-45 tok/s | ✗ | ✓ | Recommended alternative. Cleaner code output. Fast. |
| **qwen/qwen2.5-coder-32b-instruct** | NVIDIA NIM | Code generation | Sonnet | 32B | 35-50 tok/s | ✗ | ✓ | Code-specialized. Fast. Verbose output style. |
| **qwen/qwen3-next-80b-a3b-thinking** | NVIDIA NIM | Reasoning-heavy tasks | Opus | 80B (active 3B) | 12-20 tok/s | ✓ (explicit) | ✓ | MoE with thinking. Qwen's reasoning variant. |
| **qwen/qwen3.5-397b-a17b** | NVIDIA NIM | Heavy computation | Opus | 397B (active 17B) | 5-12 tok/s | ✗ | ✓ | MoE architecture. Very large but slow. Overkill? |
| **zai-org/GLM-5.1-FP8** | Modal | Free tier, unrestricted | Sonnet | ~120B (FP8) | 20-30 tok/s | ✓ (implicit) | ✓ | Free, but 1 concurrent request limit. FP8 quantization. |
| **deepseek-ai/deepseek-v3.2** | NVIDIA NIM | Code-focused reasoning | Sonnet | ~200B | 12-25 tok/s | ✓ (strong reasoning) | ✓ | Strong coding model. Worth trying if Qwen doesn't click. |

**Legend:**
- ✓ = Supported
- ✗ = Not supported / not a primary feature
- ✓✓ = Explicit/explicit chain-of-thought

## Speed Tiers

**Fast (30+ tok/s):**
- stepfun-ai/step-3.5-flash (Haiku candidate)
- meta/llama-3.3-70b-instruct (recommended Sonnet)
- qwen/qwen2.5-coder-32b-instruct

**Medium (15-30 tok/s):**
- z-ai/glm5
- zai-org/GLM-5.1-FP8 (Modal)
- deepseek-ai/deepseek-v3.2

**Slow (under 20 tok/s):**
- moonshotai/kimi-k2.5
- moonshotai/kimi-k2-thinking
- qwen/qwen3-next-80b-a3b-thinking
- qwen/qwen3.5-397b-a17b

## Rate Limits

| Provider | Rate Limit | Token Limit | Notes |
|----------|------------|-------------|-------|
| NVIDIA NIM | 40 req/min | Unlimited | Per-minute throughput, predictable capacity |
| Modal GLV5 | 1 concurrent | Unlimited | Single request at a time, queues block everything |

## Agentic Coding Suitability

**All models support tool use** - this is a baseline requirement for agentic coding. The table below focuses on other factors.

**Recommended for your workflow:**

```dotenv
# Deep reasoning (when you need it)
MODEL_OPUS="nvidia_nim/z-ai/glm5"

# Balanced work (faster, cleaner code)
MODEL_SONNET="nvidia_nim/meta/llama-3.3-70b-instruct"

# Quick iterations
MODEL_HAIKU="nvidia_nim/stepfun-ai/step-3.5-flash"
```

**Alternative if GLM-5 is too slow:**

```dotenv
# Fast, tested in agentic workflows
MODEL_OPUS="nvidia_nim/meta/llama-3.3-70b-instruct"

# Even faster, code-specialized
MODEL_SONNET="nvidia_nim/qwen/qwen2.5-coder-32b-instruct"

# Keep Step Flash
MODEL_HAIKU="nvidia_nim/stepfun-ai/step-3.5-flash"
```

**For reasoning-heavy tasks:**

```dotenv
# Explicit chain-of-thought (slowest, most thorough)
MODEL_OPUS="nvidia_nim/moonshotai/kimi-k2-thinking"

# Strong reasoning without explicit CoT
MODEL_SONNET="nvidia_nim/deepseek-ai/deepseek-v3.2"

# Fast iteration
MODEL_HAIKU="nvidia_nim/stepfun-ai/step-3.5-flash"
```

## Notes on Thinking Models

Models with thinking capabilities:

| Model | Thinking Type | When to Use |
|-------|---------------|-------------|
| GLM-5 (NIM) | Toggle via `NIM_ENABLE_THINKING=true` | Controlled via .env, adds 30-60% latency |
| Kimi-k2-thinking | Always-on explicit CoT | Complex debugging, architectural decisions |
| Kimi-k2.5 | Implicit reasoning | Large codebase analysis, multi-step tasks |
| Qwen3-next-thinking | Always-on explicit CoT | Mathematical reasoning, complex logic |
| DeepSeek-v3.2 | Strong implicit reasoning | Code architecture, refactoring decisions |

**Performance impact:**
- **Explicit thinking** (Kimi-k2-thinking, Qwen3-thinking): Slowest, but most thorough. Good for "think carefully" tasks.
- **Implicit reasoning** (Kimi-k2.5, DeepSeek): Strong reasoning without slowdown. Good balance.
- **Toggleable** (GLM-5): Best of both worlds. Test with `NIM_ENABLE_THINKING=false` first.

## Tool Use Quality

**All models support tool use, but quality varies:**

| Tier | Models | Tool Use Quality |
|------|--------|------------------|
| **Excellent** | Llama-3.3-70b, DeepSeek-v3.2, GLM-5 | Clean structured output, minimal parsing errors |
| **Good** | Kimi-k2.5, Qwen2.5-coder, Step-3.5-flash | Reliable, occasional format quirks |
| **Verbose** | Qwen family (all variants) | Works but outputs extra commentary, harder to parse |

**Note:** Your proxy has heuristic tool parsing that catches most edge cases. Even "verbose" models will work, but may feel less "clean".

---

**Speed estimates based on:** NVIDIA NIM infrastructure, typical batch sizes, sequence lengths 2k-8k tokens. Actual performance varies with request complexity and server load.

**Thinking/reasoning assessment:** Based on model architecture documentation and reported capabilities. "Implicit" = strong reasoning without explicit CoT steps. "Explicit" = visible chain-of-thought in outputs.
