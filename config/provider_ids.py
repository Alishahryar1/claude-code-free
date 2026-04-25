"""Canonical list of model provider type prefixes (provider_id values).

`providers.registry.PROVIDER_DESCRIPTORS` is the full metadata source; this
module holds the id set for config validation and must stay in sync
(registries assert in `providers.registry`).
"""

from __future__ import annotations

# Order matches docs / historical error text; must match PROVIDER_DESCRIPTORS keys.
SUPPORTED_PROVIDER_IDS: tuple[str, ...] = (
    "nvidia_nim",
    "open_router",
    "deepseek",
    "lmstudio",
    "llamacpp",
    "codex_cli",
)
