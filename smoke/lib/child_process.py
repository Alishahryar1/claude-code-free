"""Child-process commands for smoke (avoid nested ``uv run`` on Windows).

Nested ``uv run`` can try to refresh console scripts while they are locked
(``free-claude-code.exe`` in use), causing flaky smoke. The smoke runner is
already executed under the project environment (``uv run pytest``), so children
should use the same interpreter.
"""

from __future__ import annotations

import sys


def python_exe() -> str:
    return sys.executable


def cmd_python_c(script: str) -> list[str]:
    return [python_exe(), "-c", script]


def cmd_uvicorn_server_app(
    host: str, port: int, *, graceful_shutdown_s: int = 5
) -> list[str]:
    return [
        python_exe(),
        "-m",
        "uvicorn",
        "server:app",
        "--host",
        host,
        "--port",
        str(port),
        "--timeout-graceful-shutdown",
        str(graceful_shutdown_s),
    ]


def cmd_fcc_init() -> list[str]:
    return [python_exe(), "-c", "from cli.entrypoints import init; init()"]


def cmd_free_claude_code_serve() -> list[str]:
    return [python_exe(), "-c", "from cli.entrypoints import serve; serve()"]
