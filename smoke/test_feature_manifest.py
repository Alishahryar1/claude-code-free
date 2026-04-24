from __future__ import annotations

import pytest

from messaging.platforms.factory import create_messaging_platform
from providers.base import BaseProvider
from providers.deepseek import DeepSeekProvider
from providers.llamacpp import LlamaCppProvider
from providers.lmstudio import LMStudioProvider
from providers.nvidia_nim import NvidiaNimProvider
from providers.open_router import OpenRouterProvider
from smoke.features import FEATURE_SMOKES, README_FEATURES, smoke_ids

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("contract")]


def test_every_advertised_feature_has_a_smoke_entry() -> None:
    missing = sorted(set(README_FEATURES) - smoke_ids())
    extra = sorted(smoke_ids() - set(README_FEATURES))
    assert not missing, f"README features missing smoke entries: {missing}"
    assert not extra, f"smoke entries not tied to README features: {extra}"


def test_smoke_manifest_has_unique_ids_and_checks() -> None:
    ids = [feature.feature_id for feature in FEATURE_SMOKES]
    assert len(ids) == len(set(ids))
    for feature in FEATURE_SMOKES:
        assert feature.checks, feature
        assert feature.mode in {
            "live",
            "contract",
            "live_or_interactive",
            "live_or_skip",
        }


def test_provider_and_platform_registries_include_advertised_builtins() -> None:
    provider_classes = {
        "nvidia_nim": NvidiaNimProvider,
        "open_router": OpenRouterProvider,
        "deepseek": DeepSeekProvider,
        "lmstudio": LMStudioProvider,
        "llamacpp": LlamaCppProvider,
    }
    for provider_class in provider_classes.values():
        assert issubclass(provider_class, BaseProvider)

    assert create_messaging_platform("not-a-platform") is None
