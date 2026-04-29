from __future__ import annotations

import re
import tomllib
from pathlib import Path


def test_architecture_plan_exists() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    plan = repo_root / "PLAN.md"

    assert plan.exists()
    text = plan.read_text(encoding="utf-8")
    assert "Intended Dependency Direction" in text
    assert "Smoke Coverage Policy" in text
    assert "providers.nvidia_nim.voice" in text
    assert "no dedicated smoke SSE shim" in text


def test_smoke_lib_has_no_sse_shim_module() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    assert not (repo_root / "smoke" / "lib" / "sse.py").exists()


def test_api_package_exports_match_plan() -> None:
    import api

    assert set(api.__all__) == {
        "MessagesRequest",
        "MessagesResponse",
        "TokenCountRequest",
        "TokenCountResponse",
        "create_app",
    }


def test_root_env_example_is_the_single_template_source() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    root_example = repo_root / ".env.example"
    duplicate_example = repo_root / "config" / "env.example"

    assert root_example.is_file()
    assert not duplicate_example.exists()


def test_root_env_example_is_packaged_for_fcc_init() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text("utf-8"))

    force_include = pyproject["tool"]["hatch"]["build"]["targets"]["wheel"][
        "force-include"
    ]

    assert force_include[".env.example"] == "cli/env.example"


def test_pyproject_first_party_packages_match_packaged_roots() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r"known-first-party = \[(?P<items>[^\]]+)\]", pyproject)

    assert match is not None
    configured = {
        item.strip().strip('"')
        for item in match.group("items").split(",")
        if item.strip()
    }
    expected = {"api", "cli", "config", "core", "messaging", "providers", "smoke"}
    assert configured == expected
