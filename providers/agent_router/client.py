"""Agent Router provider implementation."""

from providers.base import ProviderConfig
from providers.openai_compat import OpenAICompatibleProvider

AGENT_ROUTER_BASE_URL = "http://localhost:8000/v1"

class AgentRouterProvider(OpenAICompatibleProvider):
    """Agent Router provider using official OpenAI client."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="AGENT_ROUTER",
            base_url=config.base_url or AGENT_ROUTER_BASE_URL,
            api_key=config.api_key,
        )
