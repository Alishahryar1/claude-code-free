"""Auto router for dynamically selecting the best model for each query."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from loguru import logger

from core.model_catalog import ModelCatalog, ModelInfo, get_global_catalog
from config.settings import Settings
from core.query_classifier import (
    ClassificationResult,
    QueryClassifier,
    QueryType,
    get_global_classifier,
)


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Result of the auto-routing decision."""

    model_id: str
    provider_id: str
    query_type: QueryType
    confidence: float
    reasoning: str
    used_fallback: bool = False


class AutoRouter:
    """Automatically route queries to the best available model."""

    def __init__(
        self,
        catalog: ModelCatalog | None = None,
        classifier: QueryClassifier | None = None,
    ) -> None:
        self._catalog = catalog or get_global_catalog()
        self._classifier = classifier or get_global_classifier()

    async def route(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None = None,
        settings: Settings | None = None,
        fallback_model: str | None = None,
    ) -> RoutingDecision:
        """Route a query to the best available model."""
        if settings is None:
            from config.settings import get_settings

            settings = get_settings()

        classification = self._classifier.classify(messages, tools)

        best_model = self._catalog.find_best_model_for_query(
            classification.query_type, fallback_model
        )

        if best_model is None:
            logger.warning(
                "AUTO_ROUTER: no model found for query_type={}, using fallback",
                classification.query_type,
            )
            if fallback_model:
                return RoutingDecision(
                    model_id=fallback_model,
                    provider_id=settings.parse_provider_type(fallback_model),
                    query_type=classification.query_type,
                    confidence=classification.confidence,
                    reasoning=f"No model found, using fallback: {fallback_model}",
                    used_fallback=True,
                )
            return RoutingDecision(
                model_id=settings.model,
                provider_id=settings.provider_type,
                query_type=classification.query_type,
                confidence=classification.confidence,
                reasoning=f"No model found, using default: {settings.model}",
                used_fallback=True,
            )

        logger.info(
            "AUTO_ROUTER: routed to model={} for query_type={} (confidence={:.2f})",
            best_model.id,
            classification.query_type,
            classification.confidence,
        )

        return RoutingDecision(
            model_id=best_model.id,
            provider_id=best_model.provider_id,
            query_type=classification.query_type,
            confidence=classification.confidence,
            reasoning=classification.reasoning,
            used_fallback=False,
        )

    async def ensure_catalog_loaded(
        self, api_key: str, base_url: str = "https://integrate.api.nvidia.com/v1"
    ) -> None:
        """Ensure the model catalog is loaded with available models."""
        if not self._catalog.is_cache_valid() or not self._catalog.get_all_models():
            await self._catalog.fetch_nvidia_nim_models(api_key, base_url)


_global_router: AutoRouter | None = None


def get_global_router() -> AutoRouter:
    """Get or create the global auto router instance."""
    global _global_router
    if _global_router is None:
        _global_router = AutoRouter()
    return _global_router
