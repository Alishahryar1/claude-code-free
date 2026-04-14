import pytest
from unittest.mock import MagicMock

from providers.modal import ModalProvider
from providers.modal.client import MODAL_GLV5_BASE_URL
from providers.base import ProviderConfig


class TestModalProvider:
    """Test Modal provider initialization and configuration."""

    def test_modal_provider_uses_default_base_url(self):
        """Modal provider should use hardcoded Modal GLV5 endpoint."""
        config = ProviderConfig(
            api_key="test-key",
            base_url=None,
            rate_limit=10,
            rate_window=60,
            max_concurrency=5,
            http_read_timeout=300.0,
            http_write_timeout=10.0,
            http_connect_timeout=2.0,
        )
        provider = ModalProvider(config)

        assert provider._base_url == MODAL_GLV5_BASE_URL
        assert provider._base_url == "https://api.us-west-2.modal.direct/v1"

    def test_modal_provider_allows_custom_base_url(self):
        """Modal provider should allow custom base URL override."""
        custom_url = "https://custom.modal.endpoint/v1"
        config = ProviderConfig(
            api_key="test-key",
            base_url=custom_url,
            rate_limit=10,
            rate_window=60,
            max_concurrency=5,
            http_read_timeout=300.0,
            http_write_timeout=10.0,
            http_connect_timeout=2.0,
        )
        provider = ModalProvider(config)

        assert provider._base_url == custom_url

    def test_modal_provider_name(self):
        """Modal provider should have correct provider name."""
        config = ProviderConfig(
            api_key="test-key",
            base_url=None,
            rate_limit=10,
            rate_window=60,
            max_concurrency=5,
            http_read_timeout=300.0,
            http_write_timeout=10.0,
            http_connect_timeout=2.0,
        )
        provider = ModalProvider(config)

        assert provider._provider_name == "MODAL"

    def test_modal_provider_api_key(self):
        """Modal provider should store API key."""
        config = ProviderConfig(
            api_key="test-modal-key",
            base_url=None,
            rate_limit=10,
            rate_window=60,
            max_concurrency=5,
            http_read_timeout=300.0,
            http_write_timeout=10.0,
            http_connect_timeout=2.0,
        )
        provider = ModalProvider(config)

        assert provider._api_key == "test-modal-key"

    def test_modal_build_request_body(self):
        """Modal provider should build request body correctly."""
        config = ProviderConfig(
            api_key="test-key",
            base_url=None,
            rate_limit=10,
            rate_window=60,
            max_concurrency=5,
            http_read_timeout=300.0,
            http_write_timeout=10.0,
            http_connect_timeout=2.0,
        )
        provider = ModalProvider(config)

        # Mock request with standard Anthropic-style attributes
        mock_request = MagicMock()
        mock_request.model = "claude-sonnet-4-5"
        mock_request.messages = [MagicMock()]
        mock_request.max_tokens = 100

        body = provider._build_request_body(mock_request)

        assert "messages" in body
        assert "max_tokens" in body
        assert body["max_tokens"] == 100
