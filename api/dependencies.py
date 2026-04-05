"""Dependency injection for FastAPI."""

from fastapi import Depends, HTTPException, Request
from loguru import logger

from config.settings import Settings
from config.settings import get_settings as _get_settings
from providers.ai21 import AI21_BASE_URL, Ai21Provider
from providers.anthropic import ANTHROPIC_BASE_URL, AnthropicProvider
from providers.anyscale import ANYSCALE_BASE_URL, AnyscaleProvider
from providers.base import BaseProvider, ProviderConfig
from providers.cerebras import CEREBRAS_BASE_URL, CerebrasProvider
from providers.cohere import COHERE_BASE_URL, CohereProvider
from providers.common import get_user_facing_error_message
from providers.custom import CustomProvider
from providers.deepinfra import DEEPINFRA_BASE_URL, DeepInfraProvider
from providers.exceptions import AuthenticationError
from providers.fireworks import FIREWORKS_BASE_URL, FireworksProvider
from providers.google import GOOGLE_BASE_URL, GoogleProvider
from providers.groq import GROQ_BASE_URL, GroqProvider
from providers.huggingface import HUGGINGFACE_BASE_URL, HuggingFaceProvider
from providers.llamacpp import LlamaCppProvider
from providers.lmstudio import LMStudioProvider
from providers.mistral import MISTRAL_BASE_URL, MistralProvider
from providers.novita import NOVITA_BASE_URL, NovitaProvider
from providers.nvidia_nim import NVIDIA_NIM_BASE_URL, NvidiaNimProvider
from providers.ollama import OLLAMA_BASE_URL, OllamaProvider
from providers.open_router import OPENROUTER_BASE_URL, OpenRouterProvider
from providers.openai import OPENAI_BASE_URL, OpenAiProvider
from providers.perplexity import PERPLEXITY_BASE_URL, PerplexityProvider
from providers.predibase import PREDIBASE_BASE_URL, PredibaseProvider
from providers.replicate import REPLICATE_BASE_URL, ReplicateProvider
from providers.kilo_gateway import KILO_GATEWAY_BASE_URL, KiloGatewayProvider
from providers.opencode_zen import OPENCODE_ZEN_BASE_URL, OpenCodeZenProvider
from providers.runpod import RUNPOD_BASE_URL, RunPodProvider
from providers.sambanova import SAMBANOVA_BASE_URL, SambaNovaProvider
from providers.textsynth import TEXTSYNTH_BASE_URL, TextSynthProvider
from providers.together import TOGETHER_BASE_URL, TogetherProvider
from providers.vllm import VLLM_BASE_URL, VllmProvider
from providers.xai import XAI_BASE_URL, XaiProvider

# Provider registry: keyed by provider type string, lazily populated
_providers: dict[str, BaseProvider] = {}


def get_settings() -> Settings:
    """Get application settings via dependency injection."""
    return _get_settings()


def _create_provider_for_type(provider_type: str, settings: Settings) -> BaseProvider:
    """Construct and return a new provider instance for the given provider type."""
    if provider_type == "nvidia_nim":
        if not settings.nvidia_nim_api_key or not settings.nvidia_nim_api_key.strip():
            raise AuthenticationError(
                "NVIDIA_NIM_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://build.nvidia.com/settings/api-keys"
            )
        config = ProviderConfig(
            api_key=settings.nvidia_nim_api_key,
            base_url=NVIDIA_NIM_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return NvidiaNimProvider(config, nim_settings=settings.nim)
    if provider_type == "open_router":
        if not settings.open_router_api_key or not settings.open_router_api_key.strip():
            raise AuthenticationError(
                "OPENROUTER_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://openrouter.ai/keys"
            )
        config = ProviderConfig(
            api_key=settings.open_router_api_key,
            base_url=OPENROUTER_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return OpenRouterProvider(config)
    if provider_type == "lmstudio":
        config = ProviderConfig(
            api_key="lm-studio",
            base_url=settings.lm_studio_base_url,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return LMStudioProvider(config)
    if provider_type == "llamacpp":
        config = ProviderConfig(
            api_key="llamacpp",
            base_url=settings.llamacpp_base_url,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return LlamaCppProvider(config)
    if provider_type == "groq":
        if not settings.groq_api_key or not settings.groq_api_key.strip():
            raise AuthenticationError(
                "GROQ_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://console.groq.com/keys"
            )
        config = ProviderConfig(
            api_key=settings.groq_api_key,
            base_url=GROQ_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return GroqProvider(config)
    if provider_type == "together":
        if not settings.together_api_key or not settings.together_api_key.strip():
            raise AuthenticationError(
                "TOGETHER_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://api.together.xyz/settings/api-keys"
            )
        config = ProviderConfig(
            api_key=settings.together_api_key,
            base_url=TOGETHER_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return TogetherProvider(config)
    if provider_type == "deepinfra":
        if not settings.deepinfra_api_key or not settings.deepinfra_api_key.strip():
            raise AuthenticationError(
                "DEEPINFRA_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://deepinfra.com/dash/api_keys"
            )
        config = ProviderConfig(
            api_key=settings.deepinfra_api_key,
            base_url=DEEPINFRA_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return DeepInfraProvider(config)
    if provider_type == "custom":
        if not settings.custom_api_key or not settings.custom_api_key.strip():
            raise AuthenticationError(
                "CUSTOM_API_KEY is not set. Add it to your .env file."
            )
        if not settings.custom_base_url or not settings.custom_base_url.strip():
            raise AuthenticationError(
                "CUSTOM_BASE_URL is not set. Add it to your .env file."
            )
        config = ProviderConfig(
            api_key=settings.custom_api_key,
            base_url=settings.custom_base_url,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return CustomProvider(config)
    if provider_type == "huggingface":
        config = ProviderConfig(
            api_key=settings.huggingface_api_key,
            base_url=settings.huggingface_base_url,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return HuggingFaceProvider(config)
    if provider_type == "replicate":
        if not settings.replicate_api_key or not settings.replicate_api_key.strip():
            raise AuthenticationError(
                "REPLICATE_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://replicate.com/account/api-tokens"
            )
        config = ProviderConfig(
            api_key=settings.replicate_api_key,
            base_url=REPLICATE_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return ReplicateProvider(config)
    if provider_type == "fireworks":
        if not settings.fireworks_api_key or not settings.fireworks_api_key.strip():
            raise AuthenticationError(
                "FIREWORKS_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://fireworks.ai/account/api-keys"
            )
        config = ProviderConfig(
            api_key=settings.fireworks_api_key,
            base_url=FIREWORKS_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return FireworksProvider(config)
    if provider_type == "anyscale":
        if not settings.anyscale_api_key or not settings.anyscale_api_key.strip():
            raise AuthenticationError(
                "ANYSCALE_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://app.anyscale.com/credentials"
            )
        config = ProviderConfig(
            api_key=settings.anyscale_api_key,
            base_url=ANYSCALE_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return AnyscaleProvider(config)
    if provider_type == "novita":
        if not settings.novita_api_key or not settings.novita_api_key.strip():
            raise AuthenticationError(
                "NOVITA_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://novita.ai/settings"
            )
        config = ProviderConfig(
            api_key=settings.novita_api_key,
            base_url=NOVITA_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return NovitaProvider(config)
    if provider_type == "cohere":
        if not settings.cohere_api_key or not settings.cohere_api_key.strip():
            raise AuthenticationError(
                "COHERE_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://dashboard.cohere.ai/api-keys"
            )
        config = ProviderConfig(
            api_key=settings.cohere_api_key,
            base_url=COHERE_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return CohereProvider(config)
    if provider_type == "ai21":
        if not settings.ai21_api_key or not settings.ai21_api_key.strip():
            raise AuthenticationError(
                "AI21_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://studio.ai21.com/account/api-key"
            )
        config = ProviderConfig(
            api_key=settings.ai21_api_key,
            base_url=AI21_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return Ai21Provider(config)
    if provider_type == "perplexity":
        if not settings.perplexity_api_key or not settings.perplexity_api_key.strip():
            raise AuthenticationError(
                "PERPLEXITY_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://www.perplexity.ai/settings/api"
            )
        config = ProviderConfig(
            api_key=settings.perplexity_api_key,
            base_url=PERPLEXITY_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return PerplexityProvider(config)
    if provider_type == "sambanova":
        if not settings.sambanova_api_key or not settings.sambanova_api_key.strip():
            raise AuthenticationError(
                "SAMBANOVA_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://sambanova.ai/console"
            )
        config = ProviderConfig(
            api_key=settings.sambanova_api_key,
            base_url=SAMBANOVA_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return SambaNovaProvider(config)
    if provider_type == "cerebras":
        if not settings.cerebras_api_key or not settings.cerebras_api_key.strip():
            raise AuthenticationError(
                "CEREBRAS_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://cloud.cerebras.ai/"
            )
        config = ProviderConfig(
            api_key=settings.cerebras_api_key,
            base_url=CEREBRAS_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return CerebrasProvider(config)
    if provider_type == "mistral":
        if not settings.mistral_api_key or not settings.mistral_api_key.strip():
            raise AuthenticationError(
                "MISTRAL_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://console.mistral.ai/"
            )
        config = ProviderConfig(
            api_key=settings.mistral_api_key,
            base_url=MISTRAL_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return MistralProvider(config)
    if provider_type == "google":
        if not settings.google_api_key or not settings.google_api_key.strip():
            raise AuthenticationError(
                "GOOGLE_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://makersuite.google.com/app/apikey"
            )
        config = ProviderConfig(
            api_key=settings.google_api_key,
            base_url=GOOGLE_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return GoogleProvider(config)
    if provider_type == "xai":
        if not settings.xai_api_key or not settings.xai_api_key.strip():
            raise AuthenticationError(
                "XAI_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://console.x.ai/"
            )
        config = ProviderConfig(
            api_key=settings.xai_api_key,
            base_url=XAI_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return XaiProvider(config)
    if provider_type == "anthropic":
        if not settings.anthropic_api_key or not settings.anthropic_api_key.strip():
            raise AuthenticationError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://console.anthropic.com/"
            )
        config = ProviderConfig(
            api_key=settings.anthropic_api_key,
            base_url=ANTHROPIC_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return AnthropicProvider(config)
    if provider_type == "openai":
        if not settings.openai_api_key or not settings.openai_api_key.strip():
            raise AuthenticationError(
                "OPENAI_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://platform.openai.com/api-keys"
            )
        config = ProviderConfig(
            api_key=settings.openai_api_key,
            base_url=OPENAI_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return OpenAiProvider(config)
    if provider_type == "ollama":
        config = ProviderConfig(
            api_key="ollama",
            base_url=settings.ollama_base_url,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return OllamaProvider(config)
    if provider_type == "vllm":
        if not settings.vllm_base_url or not settings.vllm_base_url.strip():
            raise AuthenticationError(
                "VLLM_BASE_URL is not set. Add it to your .env file."
            )
        config = ProviderConfig(
            api_key="vllm",
            base_url=settings.vllm_base_url,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return VllmProvider(config)
    if provider_type == "textsynth":
        if not settings.textsynth_api_key or not settings.textsynth_api_key.strip():
            raise AuthenticationError(
                "TEXTSYNTH_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://textsynth.com/settings"
            )
        config = ProviderConfig(
            api_key=settings.textsynth_api_key,
            base_url=TEXTSYNTH_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return TextSynthProvider(config)
    if provider_type == "predibase":
        if not settings.predibase_api_key or not settings.predibase_api_key.strip():
            raise AuthenticationError(
                "PREDIBASE_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://app.predibase.com/settings/api-keys"
            )
        config = ProviderConfig(
            api_key=settings.predibase_api_key,
            base_url=PREDIBASE_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return PredibaseProvider(config)
    if provider_type == "runpod":
        if not settings.runpod_api_key or not settings.runpod_api_key.strip():
            raise AuthenticationError(
                "RUNPOD_API_KEY is not set. Add it to your .env file. "
                "Get a key at https://www.runpod.io/console/user/settings"
            )
        config = ProviderConfig(
            api_key=settings.runpod_api_key,
            base_url=RUNPOD_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return RunPodProvider(config)
    if provider_type == "kilo_gateway":
        if (
            not settings.kilo_gateway_api_key
            or not settings.kilo_gateway_api_key.strip()
        ):
            raise AuthenticationError(
                "KILO_GATEWAY_API_KEY is not set. Add it to your .env file."
            )
        config = ProviderConfig(
            api_key=settings.kilo_gateway_api_key,
            base_url=KILO_GATEWAY_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return KiloGatewayProvider(config)
    if provider_type == "opencode_zen":
        if (
            not settings.opencode_zen_api_key
            or not settings.opencode_zen_api_key.strip()
        ):
            raise AuthenticationError(
                "OPENCODE_ZEN_API_KEY is not set. Add it to your .env file."
            )
        config = ProviderConfig(
            api_key=settings.opencode_zen_api_key,
            base_url=OPENCODE_ZEN_BASE_URL,
            rate_limit=settings.provider_rate_limit,
            rate_window=settings.provider_rate_window,
            max_concurrency=settings.provider_max_concurrency,
            http_read_timeout=settings.http_read_timeout,
            http_write_timeout=settings.http_write_timeout,
            http_connect_timeout=settings.http_connect_timeout,
        )
        return OpenCodeZenProvider(config)
    logger.error(
        "Unknown provider_type: '{}'. Supported providers include: nvidia_nim, open_router, lmstudio, llamacpp, groq, together, deepinfra, custom, huggingface, replicate, fireworks, anyscale, novita, cohere, ai21, perplexity, sambanova, cerebras, mistral, google, xai, anthropic, openai, ollama, vllm, textsynth, predibase, runpod, kilo_gateway, opencode_zen",
        provider_type,
    )
    raise ValueError(
        f"Unknown provider_type: '{provider_type}'. "
        f"Supported providers include: nvidia_nim, open_router, lmstudio, llamacpp, groq, together, deepinfra, custom, huggingface, replicate, fireworks, anyscale, novita, cohere, ai21, perplexity, sambanova, cerebras, mistral, google, xai, anthropic, openai, ollama, vllm, textsynth, predibase, runpod, kilo_gateway, opencode_zen"
    )


def get_provider_for_type(provider_type: str) -> BaseProvider:
    """Get or create a provider for the given provider type.

    Providers are cached in the registry and reused across requests.
    """
    if provider_type not in _providers:
        try:
            _providers[provider_type] = _create_provider_for_type(
                provider_type, get_settings()
            )
        except AuthenticationError as e:
            raise HTTPException(
                status_code=503, detail=get_user_facing_error_message(e)
            ) from e
        logger.info("Provider initialized: {}", provider_type)
    return _providers[provider_type]


def require_api_key(
    request: Request, settings: Settings = Depends(get_settings)
) -> None:
    """Require a server API key (Anthropic-style).

    Checks `x-api-key` header or `Authorization: Bearer ...` against
    `Settings.anthropic_auth_token`. If `ANTHROPIC_AUTH_TOKEN` is empty, this is a no-op.
    """
    anthropic_auth_token = settings.anthropic_auth_token
    if not anthropic_auth_token:
        # No API key configured -> allow
        return

    header = (
        request.headers.get("x-api-key")
        or request.headers.get("authorization")
        or request.headers.get("anthropic-auth-token")
    )
    if not header:
        raise HTTPException(status_code=401, detail="Missing API key")

    # Support both raw key in X-API-Key and Bearer token in Authorization
    token = header
    if header.lower().startswith("bearer "):
        token = header.split(" ", 1)[1]

    # Strip anything after the first colon to handle tokens with appended model names
    if token and ":" in token:
        token = token.split(":", 1)[0]

    if token != anthropic_auth_token:
        raise HTTPException(status_code=401, detail="Invalid API key")


def get_provider() -> BaseProvider:
    """Get or create the default provider (based on MODEL env var).

    Backward-compatible convenience for health/root endpoints and tests.
    """
    return get_provider_for_type(get_settings().provider_type)


async def cleanup_provider():
    """Cleanup all provider resources."""
    global _providers
    for provider in _providers.values():
        await provider.cleanup()
    _providers = {}
    logger.debug("Provider cleanup completed")
