"""Model routing for Claude-compatible requests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from loguru import logger

from config.provider_ids import SUPPORTED_PROVIDER_IDS
from config.settings import Settings

from .gateway_model_ids import decode_gateway_model_id
from . import auto_router
from .models.anthropic import MessagesRequest, TokenCountRequest


@dataclass(frozen=True, slots=True)
class ResolvedModel:
    original_model: str
    provider_id: str
    provider_model: str
    provider_model_ref: str
    thinking_enabled: bool
    auto_routed: bool = False
    routing_reasoning: str = ""


@dataclass(frozen=True, slots=True)
class RoutedMessagesRequest:
    request: MessagesRequest
    resolved: ResolvedModel


@dataclass(frozen=True, slots=True)
class RoutedTokenCountRequest:
    request: TokenCountRequest
    resolved: ResolvedModel


class ModelRouter:
    """Resolve incoming Claude model names to configured provider/model pairs."""

    def __init__(self, settings: Settings):
        self._settings = settings
        self._auto_router = auto_router.get_global_router()

    def resolve(self, claude_model_name: str) -> ResolvedModel:
        (
            direct_provider_id,
            direct_provider_model,
            force_thinking_enabled,
        ) = self._direct_provider_model(claude_model_name)
        if direct_provider_id is not None and direct_provider_model is not None:
            thinking_enabled = (
                force_thinking_enabled
                if force_thinking_enabled is not None
                else self._settings.resolve_thinking(direct_provider_model)
            )
            logger.debug(
                "MODEL DIRECT: '{}' -> provider='{}' model='{}' thinking={}",
                claude_model_name,
                direct_provider_id,
                direct_provider_model,
                thinking_enabled,
            )
            return ResolvedModel(
                original_model=claude_model_name,
                provider_id=direct_provider_id,
                provider_model=direct_provider_model,
                provider_model_ref=claude_model_name,
                thinking_enabled=thinking_enabled,
            )

        provider_model_ref = self._settings.resolve_model(claude_model_name)
        thinking_enabled = self._settings.resolve_thinking(claude_model_name)
        provider_id = Settings.parse_provider_type(provider_model_ref)
        provider_model = Settings.parse_model_name(provider_model_ref)
        if provider_model != claude_model_name:
            logger.debug(
                "MODEL MAPPING: '{}' -> '{}'", claude_model_name, provider_model
            )
        return ResolvedModel(
            original_model=claude_model_name,
            provider_id=provider_id,
            provider_model=provider_model,
            provider_model_ref=provider_model_ref,
            thinking_enabled=thinking_enabled,
        )

    def _direct_provider_model(
        self, model_name: str
    ) -> tuple[str | None, str | None, bool | None]:
        decoded = decode_gateway_model_id(model_name)
        if decoded is not None:
            if decoded.provider_id not in SUPPORTED_PROVIDER_IDS:
                return None, None, None
            return (
                decoded.provider_id,
                decoded.provider_model,
                decoded.force_thinking_enabled,
            )

        provider_id, separator, provider_model = model_name.partition("/")
        if not separator:
            return None, None, None
        if provider_id not in SUPPORTED_PROVIDER_IDS:
            return None, None, None
        if not provider_model:
            return None, None, None
        return provider_id, provider_model, None

    async def resolve_with_auto_routing(
        self,
        claude_model_name: str,
        messages: Sequence[Mapping[str, Any]] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> ResolvedModel:
        """Resolve model with optional auto-routing enabled."""
        if not self._settings.enable_auto_routing:
            return self.resolve(claude_model_name)

        if messages is None:
            return self.resolve(claude_model_name)

        try:
            from config.provider_catalog import PROVIDER_CATALOG
            from providers.defaults import NVIDIA_NIM_DEFAULT_BASE

            provider_descriptor = PROVIDER_CATALOG.get(
                self._settings.auto_routing_provider
            )
            if provider_descriptor is None:
                logger.warning(
                    "AUTO_ROUTING: provider '{}' not found, using static model",
                    self._settings.auto_routing_provider,
                )
                return self.resolve(claude_model_name)

            base_url = (
                getattr(self._settings, provider_descriptor.base_url_attr, None)
                if provider_descriptor.base_url_attr
                else None
            ) or provider_descriptor.default_base_url or NVIDIA_NIM_DEFAULT_BASE

            api_key = (
                getattr(
                    self._settings,
                    provider_descriptor.credential_attr,
                    None,
                )
                if provider_descriptor.credential_attr
                else None
            ) or ""

            if not api_key:
                logger.warning(
                    "AUTO_ROUTING: no API key for provider '{}', using static model",
                    self._settings.auto_routing_provider,
                )
                return self.resolve(claude_model_name)

            await self._auto_router.ensure_catalog_loaded(api_key, base_url)

            decision = await self._auto_router.route(
                messages=messages,
                tools=tools,
                settings=self._settings,
                fallback_model=self._settings.auto_routing_fallback_model,
            )

            if decision.confidence < self._settings.auto_routing_min_confidence:
                logger.debug(
                    "AUTO_ROUTER: confidence {:.2f} below threshold {:.2f}, using static model",
                    decision.confidence,
                    self._settings.auto_routing_min_confidence,
                )
                return self.resolve(claude_model_name)

            thinking_enabled = self._settings.resolve_thinking(claude_model_name)

            logger.info(
                "AUTO_ROUTER: selected model={} for query_type={} (confidence={:.2f})",
                decision.model_id,
                decision.query_type,
                decision.confidence,
            )

            return ResolvedModel(
                original_model=claude_model_name,
                provider_id=decision.provider_id,
                provider_model=decision.model_id,
                provider_model_ref=f"{decision.provider_id}/{decision.model_id}",
                thinking_enabled=thinking_enabled,
                auto_routed=True,
                routing_reasoning=decision.reasoning,
            )
        except Exception as e:
            logger.warning(
                "AUTO_ROUTER: error during routing, using static model: {}",
                str(e),
            )
            return self.resolve(claude_model_name)

    def resolve_messages_request(
        self, request: MessagesRequest
    ) -> RoutedMessagesRequest:
        """Return an internal routed request context."""
        resolved = self.resolve(request.model)
        routed = request.model_copy(deep=True)
        routed.model = resolved.provider_model
        return RoutedMessagesRequest(request=routed, resolved=resolved)

    async def resolve_messages_request_with_auto_routing(
        self, request: MessagesRequest
    ) -> RoutedMessagesRequest:
        """Return an internal routed request context with auto-routing."""
        messages_dict = [
            {"role": m.role, "content": m.content, "reasoning_content": m.reasoning_content}
            for m in request.messages
        ]
        tools_dict = (
            [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in request.tools
            ]
            if request.tools
            else None
        )

        resolved = await self.resolve_with_auto_routing(
            request.model, messages_dict, tools_dict
        )
        routed = request.model_copy(deep=True)
        routed.model = resolved.provider_model
        return RoutedMessagesRequest(request=routed, resolved=resolved)

    def resolve_token_count_request(
        self, request: TokenCountRequest
    ) -> RoutedTokenCountRequest:
        """Return an internal token-count request context."""
        resolved = self.resolve(request.model)
        routed = request.model_copy(
            update={"model": resolved.provider_model}, deep=True
        )
        return RoutedTokenCountRequest(request=routed, resolved=resolved)
