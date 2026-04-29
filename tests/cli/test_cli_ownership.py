from __future__ import annotations

from pathlib import Path

from cli.session import CLISession


def test_cli_session_owns_typed_runner_config(tmp_path: Path) -> None:
    session = CLISession(
        workspace_path=str(tmp_path),
        api_url="http://127.0.0.1:8082/v1",
        allowed_dirs=[str(tmp_path)],
        plans_directory=".plans",
        claude_bin="claude-test",
    )

    assert session.config.workspace_path == str(tmp_path)
    assert session.config.api_url == "http://127.0.0.1:8082/v1"
    assert session.config.allowed_dirs == [str(tmp_path)]
    assert session.config.plans_directory == ".plans"
    assert session.config.claude_bin == "claude-test"


def test_claude_pick_uses_project_python_runner() -> None:
    script = Path(__file__).resolve().parents[2] / "claude-pick"
    text = script.read_text(encoding="utf-8")

    assert "uv run python" in text
    assert "python3 -c" not in text
