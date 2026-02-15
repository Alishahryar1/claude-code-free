import json
import uuid
import pytest
from unittest.mock import MagicMock, AsyncMock
from providers.nvidia_nim import NvidiaNimProvider
from providers.base import ProviderConfig


@pytest.mark.asyncio
async def test_task_tool_interception():
    # Setup provider
    config = ProviderConfig(api_key="test")
    provider = NvidiaNimProvider(config)

    # Mock request and sse builder
    request = MagicMock()
    request.model = "test-model"

    sse = MagicMock()
    sse.blocks = MagicMock()
    sse.blocks.tool_indices = {}
    sse.blocks.tool_names = {}
    sse.blocks.tool_started = {}
    sse.blocks.task_arg_buffer = {}
    sse.blocks.task_args_emitted = {}
    sse.blocks.tool_ids = {}

    # Tool call data (Task tool)
    tc = {
        "index": 0,
        "id": "tool_123",
        "function": {
            "name": "Task",
            "arguments": json.dumps(
                {
                    "description": "test task",
                    "prompt": "do something",
                    "run_in_background": True,
                }
            ),
        },
    }

    # Remove pre-filled tool name - _process_tool_call handles it
    # sse.blocks.tool_names[0] = "Task"

    # Call the method
    events = []
    # _process_tool_call is a synchronous generator in nvidia_nim.py
    for event in provider._process_tool_call(tc, sse):
        events.append(event)

    # Find the start_tool_block call or check the modified state
    calls = sse.emit_tool_delta.call_args_list
    assert len(calls) > 0
    args_passed = json.loads(calls[0][0][1])
    assert args_passed["run_in_background"] is False
    print("Verification successful: run_in_background was forced to False")


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_task_tool_interception())
