"""Local Codex CLI provider.

This adapter intentionally uses only the installed `codex` binary. It does not
read Codex auth files and does not use OpenAI API keys or SDK clients.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from pathlib import Path
from typing import Any

from loguru import logger

from core.anthropic import SSEBuilder, append_request_id
from providers.base import BaseProvider, ProviderConfig

from .request import build_prompt
from .response import parse_jsonl_text


class CodexCliProvider(BaseProvider):
    """Provider that shells out to `codex exec` for local/dev text responses."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        codex_bin: str = "codex",
        workspace: str = "",
        timeout: float = 300.0,
        codex_model: str = "",
    ):
        super().__init__(config)
        self.codex_bin = codex_bin or "codex"
        self.workspace = workspace
        self.timeout = timeout
        self.codex_model = codex_model

    async def cleanup(self) -> None:
        """No persistent resources are held by the CLI adapter."""

    def _cwd(self) -> str:
        return self.workspace or str(Path.cwd())

    def build_command(self, _prompt: str, request_model: str = "") -> list[str]:
        """Build the Codex CLI command using only supported `codex exec` flags."""
        cwd = self._cwd()
        command = [
            self.codex_bin,
            "exec",
            "--json",
            "--color",
            "never",
            "--skip-git-repo-check",
            "-C",
            cwd,
        ]
        model = self.codex_model.strip() or (
            request_model if request_model and request_model != "default" else ""
        )
        if model:
            command.extend(["-m", model])
        command.append("-")
        return command

    async def _start_process(
        self, command: Sequence[str], cwd: str
    ) -> asyncio.subprocess.Process:
        return await asyncio.create_subprocess_exec(
            *command,
            cwd=cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _terminate(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        process.kill()
        await process.wait()

    async def _run_codex(
        self, command: Sequence[str], cwd: str, prompt: str = ""
    ) -> AsyncIterator[str]:
        process = await self._start_process(command, cwd)
        stderr_task = asyncio.create_task(
            process.stderr.read() if process.stderr else _empty_bytes()
        )
        emitted_text = False
        try:
            async with asyncio.timeout(self.timeout):
                await _write_stdin_prompt(process, prompt)
                if process.stdout is not None:
                    async for raw_line in process.stdout:
                        line = raw_line.decode(errors="replace").strip()
                        if text := parse_jsonl_text(line):
                            emitted_text = True
                            yield text
                return_code = await process.wait()
        except TimeoutError as e:
            await self._terminate(process)
            stderr = await _task_bytes(stderr_task)
            detail = stderr.decode(errors="replace").strip()
            msg = f"Codex CLI timed out after {self.timeout:g}s"
            if detail:
                msg = f"{msg}: {detail}"
            raise RuntimeError(msg) from e
        except asyncio.CancelledError:
            await self._terminate(process)
            raise

        stderr = (await _task_bytes(stderr_task)).decode(errors="replace").strip()
        if return_code != 0:
            msg = f"Codex CLI exited with code {return_code}"
            if stderr:
                msg = f"{msg}: {stderr}"
            raise RuntimeError(msg)
        if stderr and not emitted_text:
            raise RuntimeError(f"Codex CLI produced no text output: {stderr}")
        if stderr:
            logger.debug("CODEX_CLI_STDERR: {}", stderr)

    async def stream_response(
        self,
        request: Any,
        input_tokens: int = 0,
        *,
        request_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Stream Codex CLI stdout as Anthropic-compatible text SSE."""
        message_id = f"msg_{uuid.uuid4()}"
        sse = SSEBuilder(message_id, request.model, input_tokens)
        prompt = build_prompt(request)
        command = self.build_command(prompt, request_model=request.model)
        req_tag = f" request_id={request_id}" if request_id else ""
        logger.info(
            "CODEX_CLI_STREAM:{} command={} cwd={}", req_tag, command[:8], self._cwd()
        )

        yield sse.message_start()
        emitted_text = False
        error_occurred = False
        try:
            for event in sse.ensure_text_block():
                yield event
            async for text in self._run_codex(command, self._cwd(), prompt):
                emitted_text = True
                yield sse.emit_text_delta(text)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            error_occurred = True
            logger.error("CODEX_CLI_ERROR:{} {}", req_tag, e)
            for event in sse.close_content_blocks():
                yield event
            error_message = append_request_id(str(e), request_id)
            for event in sse.emit_error(error_message):
                yield event

        if not emitted_text and not error_occurred:
            yield sse.emit_text_delta(" ")
        for event in sse.close_all_blocks():
            yield event
        yield sse.message_delta("end_turn", sse.estimate_output_tokens())
        yield sse.message_stop()


async def _empty_bytes() -> bytes:
    return b""


async def _write_stdin_prompt(process: asyncio.subprocess.Process, prompt: str) -> None:
    if process.stdin is None:
        return
    try:
        if prompt:
            process.stdin.write(prompt.encode())
            await process.stdin.drain()
    except BrokenPipeError, ConnectionResetError:
        pass
    finally:
        process.stdin.close()
        wait_closed = getattr(process.stdin, "wait_closed", None)
        if wait_closed is not None:
            with suppress(BrokenPipeError, ConnectionResetError):
                await wait_closed()


async def _task_bytes(task: asyncio.Task[bytes]) -> bytes:
    try:
        return await task
    except Exception:
        return b""
