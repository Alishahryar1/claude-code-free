"""Provider descriptors, factory, and runtime registry."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Callable, MutableMapping

import httpx
from loguru import logger

from config.provider_catalog import (
    PROVIDER_CATALOG,
    SUPPORTED_PROVIDER_IDS,
    ProviderDescriptor,
)
from config.settings import ConfiguredChatModelRef, Settings
from providers.base import BaseProvider, ProviderConfig
from providers.exceptions import (
    AuthenticationError,
    ModelListResponseError,
    ProviderError,
    ServiceUnavailableError,
    UnknownProviderTypeError,
)

ProviderFactory = Callable[[ProviderConfig, Settings], BaseProvider]

# Backwards-compatible name for the catalog (single source: ``config.provider_catalog``).
PROVIDER_DESCRIPTORS: dict[str, ProviderDescriptor] = PROVIDER_CATALOG


def _create_nvidia_nim(config: ProviderConfig, settings: Settings) -> BaseProvider:
    from providers.nvidia_nim import NvidiaNimProvider

    return NvidiaNimProvider(config, nim_settings=settings.nim)


def _create_open_router(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.open_router import OpenRouterProvider

    return OpenRouterProvider(config)


def _create_deepseek(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.deepseek import DeepSeekProvider

    return DeepSeekProvider(config)


def _create_lmstudio(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.lmstudio import LMStudioProvider

    return LMStudioProvider(config)


def _create_llamacpp(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.llamacpp import LlamaCppProvider

    return LlamaCppProvider(config)


def _create_ollama(config: ProviderConfig, _settings: Settings) -> BaseProvider:
    from providers.ollama import OllamaProvider

    return OllamaProvider(config)


PROVIDER_FACTORIES: dict[str, ProviderFactory] = {
    "nvidia_nim": _create_nvidia_nim,
    "open_router": _create_open_router,
    "deepseek": _create_deepseek,
    "lmstudio": _create_lmstudio,
    "llamacpp": _create_llamacpp,
    "ollama": _create_ollama,
}

if set(PROVIDER_DESCRIPTORS) != set(SUPPORTED_PROVIDER_IDS) or set(
    PROVIDER_FACTORIES
) != set(SUPPORTED_PROVIDER_IDS):
    raise AssertionError(
        "PROVIDER_DESCRIPTORS, PROVIDER_FACTORIES, and SUPPORTED_PROVIDER_IDS are out of sync: "
        f"descriptors={set(PROVIDER_DESCRIPTORS)!r} factories={set(PROVIDER_FACTORIES)!r} "
        f"ids={set(SUPPORTED_PROVIDER_IDS)!r}"
    )


def _string_attr(settings: Settings, attr_name: str | None, default: str = "") -> str:
    if attr_name is None:
        return default
    value = getattr(settings, attr_name, default)
    return value if isinstance(value, str) else default


def _credential_for(descriptor: ProviderDescriptor, settings: Settings) -> str:
    if descriptor.static_credential is not None:
        return descriptor.static_credential
    if descriptor.credential_attr:
        return _string_attr(settings, descriptor.credential_attr)
    return ""


def _require_credential(descriptor: ProviderDescriptor, credential: str) -> None:
    if descriptor.credential_env is None:
        return
    if credential and credential.strip():
        return
    message = f"{descriptor.credential_env} is not set. Add it to your .env file."
    if descriptor.credential_url:
        message = f"{message} Get a key at {descriptor.credential_url}"
    raise AuthenticationError(message)


def build_provider_config(
    descriptor: ProviderDescriptor, settings: Settings
) -> ProviderConfig:
    credential = _credential_for(descriptor, settings)
    _require_credential(descriptor, credential)
    base_url = _string_attr(
        settings, descriptor.base_url_attr, descriptor.default_base_url or ""
    )
    proxy = _string_attr(settings, descriptor.proxy_attr)
    return ProviderConfig(
        api_key=credential,
        base_url=base_url or descriptor.default_base_url,
        rate_limit=settings.provider_rate_limit,
        rate_window=settings.provider_rate_window,
        max_concurrency=settings.provider_max_concurrency,
        http_read_timeout=settings.http_read_timeout,
        http_write_timeout=settings.http_write_timeout,
        http_connect_timeout=settings.http_connect_timeout,
        enable_thinking=settings.enable_model_thinking,
        proxy=proxy,
        log_raw_sse_events=settings.log_raw_sse_events,
        log_api_error_tracebacks=settings.log_api_error_tracebacks,
    )


def create_provider(provider_id: str, settings: Settings) -> BaseProvider:
    descriptor = PROVIDER_DESCRIPTORS.get(provider_id)
    if descriptor is None:
        supported = "', '".join(PROVIDER_DESCRIPTORS)
        raise UnknownProviderTypeError(
            f"Unknown provider_type: '{provider_id}'. Supported: '{supported}'"
        )

    config = build_provider_config(descriptor, settings)
    factory = PROVIDER_FACTORIES.get(provider_id)
    if factory is None:
        raise AssertionError(f"Unhandled provider descriptor: {provider_id}")
    return factory(config, settings)


def _format_provider_query_failures(
    refs: list[ConfiguredChatModelRef],
    exc: BaseException,
    settings: Settings,
) -> list[str]:
    reason = _provider_query_failure_reason(exc, settings)
    return [_format_model_validation_failure(ref, reason) for ref in refs]


def _format_missing_model_failure(ref: ConfiguredChatModelRef) -> str:
    return _format_model_validation_failure(ref, "missing model")


def _format_model_validation_failure(ref: ConfiguredChatModelRef, problem: str) -> str:
    return (
        f"sources={','.join(ref.sources)} provider={ref.provider_id} "
        f"model={ref.model_id} problem={problem}"
    )


def _provider_query_failure_reason(
    exc: BaseException,
    settings: Settings,
) -> str:
    if isinstance(exc, ModelListResponseError):
        return f"malformed model-list response: {exc.message}"
    if isinstance(exc, httpx.HTTPStatusError):
        return f"query failure: HTTP {exc.response.status_code}"
    if isinstance(exc, AuthenticationError):
        return f"query failure: {exc.message}"
    if isinstance(exc, ProviderError) and settings.log_api_error_tracebacks:
        return f"query failure: {exc.message}"
    return f"query failure: {type(exc).__name__}"


class ProviderRegistry:
    """Cache and clean up provider instances by provider id."""

    def __init__(self, providers: MutableMapping[str, BaseProvider] | None = None):
        self._providers = providers if providers is not None else {}

    def is_cached(self, provider_id: str) -> bool:
        """Return whether a provider for this id is already in the cache."""
        return provider_id in self._providers

    def get(self, provider_id: str, settings: Settings) -> BaseProvider:
        if provider_id not in self._providers:
            self._providers[provider_id] = create_provider(provider_id, settings)
        return self._providers[provider_id]

    async def validate_configured_models(self, settings: Settings) -> None:
        """Fail fast unless every configured chat model exists upstream."""
        refs = settings.configured_chat_model_refs()
        refs_by_provider: dict[str, list[ConfiguredChatModelRef]] = defaultdict(list)
        failures: list[str] = []
        for ref in refs:
            if ref.provider_id == "nvidia_nim" and "/" not in ref.model_id:
                failures.append(
                    _format_model_validation_failure(
                        ref,
                        f"NIM model names require a vendor prefix "
                        f"(e.g. 'google/{ref.model_id}'). "
                        f"Browse models at https://build.nvidia.com/explore/discover",
                    )
                )
            else:
                refs_by_provider[ref.provider_id].append(ref)

        tasks: dict[str, asyncio.Task[frozenset[str]]] = {}
        for provider_id, provider_refs in refs_by_provider.items():
            try:
                provider = self.get(provider_id, settings)
            except Exception as exc:
                failures.extend(
                    _format_provider_query_failures(provider_refs, exc, settings)
                )
                continue
            tasks[provider_id] = asyncio.create_task(provider.list_model_ids())

        if tasks:
            results = await asyncio.gather(*tasks.values(), return_exceptions=True)
            for (provider_id, _task), result in zip(
                tasks.items(), results, strict=True
            ):
                provider_refs = refs_by_provider[provider_id]
                if isinstance(result, BaseException):
                    if isinstance(result, asyncio.CancelledError):
                        raise result
                    failures.extend(
                        _format_provider_query_failures(provider_refs, result, settings)
                    )
                    continue
                failures.extend(
                    _format_missing_model_failure(ref)
                    for ref in provider_refs
                    if ref.model_id not in result
                )

        if failures:
            message = "Configured model validation failed:\n" + "\n".join(
                f"- {failure}" for failure in failures
            )
            raise ServiceUnavailableError(message)

        logger.info(
            "Configured provider models validated: models={} providers={}",
            len(refs),
            len(refs_by_provider),
        )

    async def cleanup(self) -> None:
        """Call ``cleanup`` on every cached provider, then clear the cache.

        Attempts all providers even if one fails. A single failure is re-raised
        as-is; multiple failures are wrapped in :exc:`ExceptionGroup`.
        """
        items = list(self._providers.items())
        errors: list[Exception] = []
        try:
            for _pid, provider in items:
                try:
                    await provider.cleanup()
                except Exception as e:
                    errors.append(e)
        finally:
            self._providers.clear()
        if len(errors) == 1:
            raise errors[0]
        if len(errors) > 1:
            msg = "One or more provider cleanups failed"
            raise ExceptionGroup(msg, errors)
