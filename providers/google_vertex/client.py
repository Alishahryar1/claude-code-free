"""Google Vertex AI Express provider implementation."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from config.provider_catalog import GOOGLE_VERTEX_DEFAULT_BASE
from providers.anthropic_messages import AnthropicMessagesTransport
from providers.base import ProviderConfig

from .request import build_request_body


def _parse_vertex_to_sse(text: str) -> list[str]:
    """Convert Vertex AI JSON response to SSE events."""
    events = []
    
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [f"data: {json.dumps({'text': text})}\n\n"]
    
    if not isinstance(data, list) or not data:
        return []
    
    result = data[0]
    
    event_id = 0
    
    events.append(f"event: message_start\ndata: {json.dumps({'type': 'message_start'})}\n\n")
    
    candidates = result.get("candidates", [])
    if candidates:
        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        role = content.get("role", "model")
        
        events.append(f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0})}\n\n")
        
        for i, part in enumerate(parts):
            if "text" in part:
                text_content = part["text"]
                events.append(
                    f"data: {json.dumps({'type': 'content_block_delta', 'index': i, 'delta': {'type': 'text_delta', 'text': text_content}})}\n\n"
                )
        
        events.append(f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n")
    
    usage = result.get("usageMetadata", {})
    if usage:
        events.append(f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'usage': usage, 'delta': {'stop_reason': result.get('finishReason', 'STOP')}})}\n\n")
    
    events.append(f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n")
    
    return events


class VertexExpressProvider(AnthropicMessagesTransport):
    """Google Vertex AI Express using ``https://aiplatform.googleapis.com/v1/``."""

    stream_chunk_mode = "event"

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="GOOGLE_VERTEX",
            default_base_url=GOOGLE_VERTEX_DEFAULT_BASE,
        )

    def _request_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "Accept": "text/event-stream"}

    def _build_request_body(
        self, request: Any, thinking_enabled: bool | None = None
    ) -> dict:
        """Build a Vertex AI request body."""
        thinking_enabled = self._is_thinking_enabled(request, thinking_enabled)
        return build_request_body(
            request,
            thinking_enabled=thinking_enabled,
        )

    def _get_model_path(self, model: str) -> str:
        """Convert model name to Vertex AI path format."""
        return f"/publishers/google/models/{model}:streamGenerateContent"

    async def _send_stream_request(self, body: dict) -> httpx.Response:
        """Create a streaming messages response with API key in query param."""
        model = body.get("model", "gemini-2.5-flash")
        path = self._get_model_path(model)

        body_copy = dict(body)

        request = self._client.build_request(
            "POST",
            path,
            json=body_copy,
            headers=self._request_headers(),
        )

        url = str(request.url)
        if "?" in url:
            url = f"{url}&key={self._api_key}"
        else:
            url = f"{url}?key={self._api_key}"

        new_url = httpx.URL(url)
        request = request.copy_with_url(new_url)

        return await self._client.send(request, stream=True)

    async def _iter_sse_events(self, response: httpx.Response) -> AsyncIterator[str]:
        """Parse Vertex AI JSON response and yield as SSE."""
        text = await response.aread()
        
        events = _parse_vertex_to_sse(text)
        
        for event in events:
            yield event