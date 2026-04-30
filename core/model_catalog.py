"""Model catalog for fetching and caching available models from providers."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx
from loguru import logger


from core.query_classifier import QueryType


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Metadata about a model available from a provider."""

    id: str
    name: str
    provider_id: str
    capabilities: frozenset[QueryType]
    context_length: int | None = None
    is_free: bool = True
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelCatalog:
    """Catalog of available models with caching and TTL support."""

    def __init__(self, cache_ttl: float = 300.0):
        self._models: dict[str, ModelInfo] = {}
        self._models_by_capability: dict[QueryType, list[ModelInfo]] = {qt: [] for qt in QueryType}
        self._last_fetch: float = 0.0
        self._cache_ttl: float = cache_ttl
        self._is_fetching: bool = False
        self._fetch_lock_instance: asyncio.Lock | None = None

    @property
    def _fetch_lock(self) -> asyncio.Lock:
        if self._fetch_lock_instance is None:
            self._fetch_lock_instance = asyncio.Lock()
        return self._fetch_lock_instance

    def is_cache_valid(self) -> bool:
        """Check if the cached model list is still valid."""
        return time.time() - self._last_fetch < self._cache_ttl

    def get_model(self, model_id: str) -> ModelInfo | None:
        """Get a model by ID."""
        return self._models.get(model_id)

    def get_models_for_capability(self, capability: QueryType) -> list[ModelInfo]:
        """Get all models that support a given capability."""
        return self._models_by_capability.get(capability, []).copy()

    def get_all_models(self) -> list[ModelInfo]:
        """Get all cached models."""
        return list(self._models.values())

    async def fetch_nvidia_nim_models(
        self, api_key: str, base_url: str = "https://integrate.api.nvidia.com/v1"
    ) -> list[ModelInfo]:
        """Fetch available models from NVIDIA NIM API."""
        if self.is_cache_valid() and self._models:
            logger.debug("MODEL_CATALOG: using cached models")
            return self.get_all_models()

        async with self._fetch_lock:
            if self.is_cache_valid() and self._models:
                return self.get_all_models()

            if self._is_fetching:
                logger.debug("MODEL_CATALOG: fetch already in progress, waiting")
                while self._is_fetching:
                    await asyncio.sleep(0.1)
                return self.get_all_models()

            self._is_fetching = True
            try:
                logger.info("MODEL_CATALOG: fetching models from NVIDIA NIM")
                models = await self._fetch_from_nvidia_nim(api_key, base_url)
                self._update_cache(models)
                return models
            finally:
                self._is_fetching = False

    async def _fetch_from_nvidia_nim(
        self, api_key: str, base_url: str
    ) -> list[ModelInfo]:
        """Internal method to fetch models from NVIDIA NIM."""
        url = f"{base_url}/models"
        headers = {"Authorization": f"Bearer {api_key}"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            data = response.json()

        models: list[ModelInfo] = []
        for model_data in data.get("data", []):
            model_id = model_data.get("id", "")
            if not model_id:
                continue

            capabilities = self._infer_capabilities_from_model_id(model_id)
            context_length = model_data.get("context_length")
            description = model_data.get("description", "")

            model_info = ModelInfo(
                id=model_id,
                name=model_id,
                provider_id="nvidia_nim",
                capabilities=capabilities,
                context_length=context_length,
                is_free=True,
                description=description,
                metadata=model_data,
            )
            models.append(model_info)

        logger.info("MODEL_CATALOG: fetched {} models from NVIDIA NIM", len(models))
        return models

    def _infer_capabilities_from_model_id(self, model_id: str) -> frozenset[QueryType]:
        """Infer model capabilities from the model ID."""
        model_id_lower = model_id.lower()
        capabilities: set[QueryType] = {QueryType.GENERAL}

        if any(
            keyword in model_id_lower
            for keyword in ["code", "coder", "instruct", "qwen", "deepseek"]
        ):
            capabilities.add(QueryType.CODE)

        if any(
            keyword in model_id_lower
            for keyword in ["chat", "conversational", "dialogue"]
        ):
            capabilities.add(QueryType.CHAT)

        if any(
            keyword in model_id_lower
            for keyword in ["summar", "summary", "compact"]
        ):
            capabilities.add(QueryType.SUMMARIZATION)

        if any(
            keyword in model_id_lower
            for keyword in ["vision", "image", "multimodal", "vl", "llava"]
        ):
            capabilities.add(QueryType.VISION)

        if any(
            keyword in model_id_lower
            for keyword in ["reason", "think", "r1", "distill"]
        ):
            capabilities.add(QueryType.REASONING)

        if any(
            keyword in model_id_lower
            for keyword in ["math", "arithmetic", "calculation"]
        ):
            capabilities.add(QueryType.MATH)

        if any(
            keyword in model_id_lower
            for keyword in ["writer", "writing", "creative", "story"]
        ):
            capabilities.add(QueryType.WRITING)

        return frozenset(capabilities)

    def _update_cache(self, models: list[ModelInfo]) -> None:
        """Update the internal cache with new model data."""
        self._models.clear()
        for qt in QueryType:
            self._models_by_capability[qt].clear()

        for model in models:
            self._models[model.id] = model
            for capability in model.capabilities:
                self._models_by_capability[capability].append(model)

        self._last_fetch = time.time()
        logger.info(
            "MODEL_CATALOG: updated cache with {} models", len(self._models)
        )

    def find_best_model_for_query(
        self, query_type: QueryType, fallback_model: str | None = None
    ) -> ModelInfo | None:
        """Find the best model for a given query type."""
        candidates = self.get_models_for_capability(query_type)

        if not candidates:
            logger.warning(
                "MODEL_CATALOG: no models found for query_type={}, using fallback",
                query_type,
            )
            if fallback_model:
                return self.get_model(fallback_model)
            return None

        candidates.sort(
            key=lambda m: (
                m.context_length or 0,
                len(m.capabilities),
            ),
            reverse=True,
        )

        best = candidates[0]
        logger.debug(
            "MODEL_CATALOG: selected model={} for query_type={}",
            best.id,
            query_type,
        )
        return best

    def clear_cache(self) -> None:
        """Clear the model cache."""
        self._models.clear()
        for qt in QueryType:
            self._models_by_capability[qt].clear()
        self._last_fetch = 0.0
        logger.info("MODEL_CATALOG: cache cleared")


_global_catalog: ModelCatalog | None = None


def get_global_catalog() -> ModelCatalog:
    """Get or create the global model catalog instance."""
    global _global_catalog
    if _global_catalog is None:
        _global_catalog = ModelCatalog()
    return _global_catalog
