from __future__ import annotations

import re
from pathlib import Path

from messaging.platforms.factory import create_messaging_platform
from providers.base import BaseProvider
from providers.deepseek import DeepSeekProvider
from providers.llamacpp import LlamaCppProvider
from providers.lmstudio import LMStudioProvider
from providers.nvidia_nim import NvidiaNimProvider
from providers.open_router import OpenRouterProvider
from smoke.features import FEATURE_INVENTORY, README_FEATURES, feature_ids

VALID_COVERAGE = {"pytest", "live_smoke", "both"}
VALID_SOURCE = {"readme", "public_surface"}


def test_every_readme_feature_has_inventory_entry() -> None:
    missing = sorted(set(README_FEATURES) - feature_ids(source="readme"))
    extra_readme = sorted(feature_ids(source="readme") - set(README_FEATURES))
    assert not missing, f"README features missing inventory entries: {missing}"
    assert not extra_readme, (
        f"README inventory entries not in README_FEATURES: {extra_readme}"
    )


def test_feature_inventory_is_unique_and_decision_complete() -> None:
    ids = [feature.feature_id for feature in FEATURE_INVENTORY]
    assert len(ids) == len(set(ids))
    assert "claude_pick" not in ids

    for feature in FEATURE_INVENTORY:
        assert feature.source in VALID_SOURCE, feature
        assert feature.coverage in VALID_COVERAGE, feature
        assert feature.title.strip(), feature
        assert feature.skip_policy.strip(), feature
        if feature.has_pytest_coverage:
            assert feature.pytest_tests, feature
        if feature.has_smoke_coverage:
            assert feature.smoke_tests, feature
            assert feature.smoke_targets, feature


def test_feature_inventory_test_owners_exist() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pytest_names = _collect_test_names(repo_root / "tests")
    smoke_names = _collect_test_names(repo_root / "smoke")

    for feature in FEATURE_INVENTORY:
        for owner in feature.pytest_tests:
            _assert_owner_exists(owner, repo_root, pytest_names)
        for owner in feature.smoke_tests:
            assert owner in smoke_names or owner in pytest_names, (feature, owner)


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


def _collect_test_names(root: Path) -> set[str]:
    names: set[str] = set()
    for path in root.rglob("test_*.py"):
        text = path.read_text(encoding="utf-8")
        names.update(re.findall(r"^\s*(?:async\s+)?def (test_[^(]+)", text, re.M))
    return names


def _assert_owner_exists(owner: str, repo_root: Path, test_names: set[str]) -> None:
    if owner.startswith("test_"):
        assert owner in test_names, owner
        return

    path_part, _, node_name = owner.partition("::")
    path = repo_root / path_part
    assert path.exists(), owner
    if node_name:
        assert node_name in test_names, owner
