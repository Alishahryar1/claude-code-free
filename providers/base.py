"""Base provider interface - extend this to implement your own provider."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional
from pydantic import BaseModel


class ProviderConfig(BaseModel):
    """Configuration for a provider."""
    api_key: str
    base_url: Optional[str] = None
    rate_limit: Optional[int] = None
    rate_window: int = 60


class BaseProvider(ABC):
    """Base class for all providers. Extend this to add your own."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @abstractmethod
    async def complete(self, request: Any) -> dict:
        """Make a non-streaming completion request. Returns raw JSON response."""
        pass

    @abstractmethod
    async def stream_response(self, request: Any, input_tokens: int = 0) -> AsyncIterator[str]:
        """Stream response in Anthropic SSE format."""
        pass

    @abstractmethod
    def convert_response(self, response_json: dict, original_request: Any) -> Any:
        """Convert provider response to Anthropic format."""
        pass
