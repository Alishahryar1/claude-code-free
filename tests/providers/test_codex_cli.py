from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest

from providers.base import ProviderConfig
from providers.codex_cli import CodexCliProvider


class EmptyStdout:
    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        raise StopAsyncIteration


class NeverStdout:
    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        await asyncio.sleep(999)
        raise StopAsyncIteration


class FakeStderr:
    def __init__(self, data: bytes = b""):
        self._data = data

    async def read(self) -> bytes:
        return self._data


class FakeStdin:
    def __init__(self) -> None:
        self.data = b""
        self.closed = False

    def write(self, data: bytes) -> None:
        self.data += data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class FakeProcess:
    def __init__(self, *, stdout: Any = None, stderr: bytes = b"", returncode: int = 0):
        self.stdin = FakeStdin()
        self.stdout = stdout if stdout is not None else EmptyStdout()
        self.stderr = FakeStderr(stderr)
        self.returncode: int | None = None
        self._final_returncode = returncode
        self.killed = False

    async def wait(self) -> int:
        self.returncode = self._final_returncode
        return self._final_returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def provider(**kwargs: Any) -> CodexCliProvider:
    return CodexCliProvider(ProviderConfig(api_key="codex-cli"), **kwargs)


def request(model: str = "default") -> SimpleNamespace:
    return SimpleNamespace(
        model=model,
        system="You are concise.",
        messages=[SimpleNamespace(role="user", content="Say hello")],
        tools=None,
        thinking=None,
    )


def test_command_construction_uses_codex_cli_bin() -> None:
    p = provider(codex_bin="/custom/codex", workspace="/tmp/work", codex_model="")

    command = p.build_command("hello", request_model="default")

    assert command[:8] == [
        "/custom/codex",
        "exec",
        "--json",
        "--color",
        "never",
        "--skip-git-repo-check",
        "-C",
        "/tmp/work",
    ]
    assert command[-1] == "-"
    assert "hello" not in command


@pytest.mark.asyncio
async def test_subprocess_invoked_without_shell_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def fake_create_subprocess_exec(*args: str, **kwargs: Any) -> FakeProcess:
        calls.append({"args": args, "kwargs": kwargs})
        return FakeProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    p = provider(codex_bin="codex-test", workspace="/tmp")

    assert [text async for text in p._run_codex(p.build_command("hello"), "/tmp")] == []

    assert calls
    assert calls[0]["args"][0] == "codex-test"
    assert "shell" not in calls[0]["kwargs"]
    assert calls[0]["kwargs"]["stdin"] == asyncio.subprocess.PIPE


@pytest.mark.asyncio
async def test_prompt_is_written_to_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeProcess()
    p = provider(workspace="/tmp")

    async def fake_start(command: list[str], cwd: str) -> FakeProcess:
        return fake

    monkeypatch.setattr(p, "_start_process", fake_start)

    assert [
        text
        async for text in p._run_codex(
            p.build_command("large prompt"), "/tmp", "large prompt"
        )
    ] == []

    assert fake.stdin.data == b"large prompt"
    assert fake.stdin.closed is True


@pytest.mark.asyncio
async def test_timeout_kills_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeProcess(stdout=NeverStdout(), stderr=b"still running", returncode=0)
    p = provider(workspace="/tmp", timeout=0.01)

    async def fake_start(command: list[str], cwd: str) -> FakeProcess:
        return fake

    monkeypatch.setattr(p, "_start_process", fake_start)

    with pytest.raises(RuntimeError, match="timed out"):
        _ = [text async for text in p._run_codex(p.build_command("hello"), "/tmp")]

    assert fake.killed is True


@pytest.mark.asyncio
async def test_stderr_becomes_visible_error(monkeypatch: pytest.MonkeyPatch) -> None:
    p = provider(workspace="/tmp")

    async def fake_start(command: list[str], cwd: str) -> FakeProcess:
        return FakeProcess(stderr=b"codex exploded", returncode=2)

    monkeypatch.setattr(p, "_start_process", fake_start)

    with pytest.raises(RuntimeError, match="codex exploded"):
        _ = [text async for text in p._run_codex(p.build_command("hello"), "/tmp")]


@pytest.mark.asyncio
async def test_stderr_with_empty_success_output_becomes_visible_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = provider(workspace="/tmp")

    async def fake_start(command: list[str], cwd: str) -> FakeProcess:
        return FakeProcess(stderr=b"codex wrote only stderr", returncode=0)

    monkeypatch.setattr(p, "_start_process", fake_start)

    with pytest.raises(RuntimeError, match="codex wrote only stderr"):
        _ = [text async for text in p._run_codex(p.build_command("hello"), "/tmp")]


@pytest.mark.asyncio
async def test_stdout_converts_to_valid_anthropic_sse_text_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    p = provider(workspace="/tmp")

    async def fake_run(command: list[str], cwd: str, prompt: str = ""):
        yield "hello from codex"

    monkeypatch.setattr(p, "_run_codex", fake_run)

    events = [
        event async for event in p.stream_response(request(), request_id="req_test")
    ]

    assert events[0].startswith("event: message_start")
    assert events[-1].startswith("event: message_stop")
    text_delta_events = [event for event in events if "content_block_delta" in event]
    assert text_delta_events
    payload = text_delta_events[0].split("data: ", 1)[1]
    data = json.loads(payload)
    assert data["delta"] == {"type": "text_delta", "text": "hello from codex"}
