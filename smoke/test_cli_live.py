from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from smoke.lib.config import SmokeConfig
from smoke.lib.server import start_server

pytestmark = [pytest.mark.live, pytest.mark.smoke_target("cli")]


def test_fcc_init_scaffolds_user_config(
    smoke_config: SmokeConfig, tmp_path: Path
) -> None:
    env = os.environ.copy()
    env["HOME"] = str(tmp_path)
    env["USERPROFILE"] = str(tmp_path)
    result = subprocess.run(
        ["uv", "run", "fcc-init"],
        cwd=smoke_config.root,
        env=env,
        capture_output=True,
        text=True,
        timeout=smoke_config.timeout_s,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert (tmp_path / ".config" / "free-claude-code" / ".env").is_file()


def test_free_claude_code_entrypoint_starts_server(smoke_config: SmokeConfig) -> None:
    with start_server(
        smoke_config,
        command=["uv", "run", "free-claude-code"],
        env_overrides={"MESSAGING_PLATFORM": "none"},
        name="entrypoint",
    ) as server:
        assert server.process.poll() is None


def test_claude_cli_prompt_when_available(
    smoke_config: SmokeConfig, tmp_path: Path
) -> None:
    claude_bin = shutil.which(smoke_config.claude_bin)
    if not claude_bin:
        pytest.skip(f"Claude CLI not found: {smoke_config.claude_bin}")
    models = smoke_config.provider_models()
    if not models:
        pytest.skip("no configured provider model available for Claude CLI smoke")

    with start_server(
        smoke_config,
        env_overrides={"MODEL": models[0].full_model, "MESSAGING_PLATFORM": "none"},
        name="claude-cli",
    ) as server:
        env = os.environ.copy()
        env["ANTHROPIC_BASE_URL"] = server.base_url
        if smoke_config.settings.anthropic_auth_token:
            env["ANTHROPIC_AUTH_TOKEN"] = smoke_config.settings.anthropic_auth_token
        result = subprocess.run(
            [claude_bin, "-p", "Reply with exactly FCC_SMOKE_PONG"],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=smoke_config.timeout_s,
            check=False,
        )
    assert result.returncode == 0, result.stderr or result.stdout
    assert "FCC_SMOKE_PONG" in result.stdout
