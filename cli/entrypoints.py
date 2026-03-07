"""CLI entry points for the installed package."""

from __future__ import annotations


def serve() -> None:
    """Start the FastAPI server (registered as `free-claude-code` script)."""
    import uvicorn

    from cli.process_registry import kill_all_best_effort
    from config.settings import get_settings

    settings = get_settings()
    try:
        uvicorn.run(
            "api.app:app",
            host=settings.host,
            port=settings.port,
            log_level="debug",
            timeout_graceful_shutdown=5,
        )
    finally:
        kill_all_best_effort()


def init() -> None:
    """Scaffold config at ~/.config/free-claude-code/.env (registered as `fcc-init`)."""
    import importlib.resources
    from pathlib import Path

    config_dir = Path.home() / ".config" / "free-claude-code"
    env_file = config_dir / ".env"

    if env_file.exists():
        print(f"Config already exists at {env_file}")
        print("Delete it first if you want to reset to defaults.")
        return

    config_dir.mkdir(parents=True, exist_ok=True)
    template = (
        importlib.resources.files("config").joinpath("env.example").read_text("utf-8")
    )
    env_file.write_text(template, encoding="utf-8")
    print(f"Config created at {env_file}")
    print(
        "Edit it to set your API keys and model preferences, then run: free-claude-code"
    )
