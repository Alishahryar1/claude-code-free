"""Centralized configuration using Pydantic Settings."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .nim import NimSettings


def _env_files() -> tuple[Path, ...]:
    """Return env file paths in priority order (later overrides earlier)."""
    files: list[Path] = [
        Path.home() / ".config" / "free-claude-code" / ".env",
        Path(".env"),
    ]
    if explicit := os.environ.get("FCC_ENV_FILE"):
        files.append(Path(explicit))
    return tuple(files)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ==================== OpenRouter Config ====================
    open_router_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")

    # ==================== Messaging Platform Selection ====================
    # Valid: "telegram" | "discord"
    messaging_platform: str = Field(
        default="discord", validation_alias="MESSAGING_PLATFORM"
    )

    # ==================== NVIDIA NIM Config ====================
    nvidia_nim_api_key: str = ""

    # ==================== LM Studio Config ====================
    lm_studio_base_url: str = Field(
        default="http://localhost:1234/v1",
        validation_alias="LM_STUDIO_BASE_URL",
    )

    # ==================== Llama.cpp Config ====================
    llamacpp_base_url: str = Field(
        default="http://localhost:8080/v1",
        validation_alias="LLAMACPP_BASE_URL",
    )

    # ==================== Groq Config ====================
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")

    # ==================== Together AI Config ====================
    together_api_key: str = Field(default="", validation_alias="TOGETHER_API_KEY")

    # ==================== DeepInfra Config ====================
    deepinfra_api_key: str = Field(default="", validation_alias="DEEPINFRA_API_KEY")

    # ==================== Custom Provider Config ====================
    custom_api_key: str = Field(default="", validation_alias="CUSTOM_API_KEY")
    custom_base_url: str = Field(default="", validation_alias="CUSTOM_BASE_URL")

    # ==================== HuggingFace Config ====================
    huggingface_api_key: str = Field(default="", validation_alias="HUGGINGFACE_API_KEY")
    huggingface_base_url: str = Field(
        default="", validation_alias="HUGGINGFACE_BASE_URL"
    )

    # ==================== Replicate Config ====================
    replicate_api_key: str = Field(default="", validation_alias="REPLICATE_API_KEY")

    # ==================== Fireworks Config ====================
    fireworks_api_key: str = Field(default="", validation_alias="FIREWORKS_API_KEY")

    # ==================== Anyscale Config ====================
    anyscale_api_key: str = Field(default="", validation_alias="ANYSCALE_API_KEY")

    # ==================== Novita Config ====================
    novita_api_key: str = Field(default="", validation_alias="NOVITA_API_KEY")

    # ==================== Cohere Config ====================
    cohere_api_key: str = Field(default="", validation_alias="COHERE_API_KEY")

    # ==================== AI21 Config ====================
    ai21_api_key: str = Field(default="", validation_alias="AI21_API_KEY")

    # ==================== Perplexity Config ====================
    perplexity_api_key: str = Field(default="", validation_alias="PERPLEXITY_API_KEY")

    # ==================== SambaNova Config ====================
    sambanova_api_key: str = Field(default="", validation_alias="SAMBANOVA_API_KEY")

    # ==================== Cerebras Config ====================
    cerebras_api_key: str = Field(default="", validation_alias="CEREBRAS_API_KEY")

    # ==================== Mistral Config ====================
    mistral_api_key: str = Field(default="", validation_alias="MISTRAL_API_KEY")

    # ==================== Google Config ====================
    google_api_key: str = Field(default="", validation_alias="GOOGLE_API_KEY")

    # ==================== xAI Config ====================
    xai_api_key: str = Field(default="", validation_alias="XAI_API_KEY")

    # ==================== Kilo Gateway Config ====================
    kilo_gateway_api_key: str = Field(
        default="", validation_alias="KILO_GATEWAY_API_KEY"
    )

    # ==================== OpenCode Zen Config ====================
    opencode_zen_api_key: str = Field(
        default="", validation_alias="OPENCODE_ZEN_API_KEY"
    )

    # ==================== Anthropic Config ====================
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")

    # ==================== OpenAI Config ====================
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")

    # ==================== Ollama Config ====================
    ollama_base_url: str = Field(
        default="http://localhost:11434/v1",
        validation_alias="OLLAMA_BASE_URL",
    )

    # ==================== VLLM Config ====================
    vllm_base_url: str = Field(default="", validation_alias="VLLM_BASE_URL")

    # ==================== TextSynth Config ====================
    textsynth_api_key: str = Field(default="", validation_alias="TEXTSYNTH_API_KEY")

    # ==================== Predibase Config ====================
    predibase_api_key: str = Field(default="", validation_alias="PREDIBASE_API_KEY")

    # ==================== RunPod Config ====================
    runpod_api_key: str = Field(default="", validation_alias="RUNPOD_API_KEY")

    # ==================== Model ====================
    # All Claude model requests are mapped to this single model (fallback)
    # Format: provider_type/model/name
    model: str = "nvidia_nim/meta/llama3-70b-instruct"

    # Per-model overrides (optional, falls back to MODEL)
    # Each can use a different provider
    model_opus: str | None = Field(default=None, validation_alias="MODEL_OPUS")
    model_sonnet: str | None = Field(default=None, validation_alias="MODEL_SONNET")
    model_haiku: str | None = Field(default=None, validation_alias="MODEL_HAIKU")

    # ==================== Provider Rate Limiting ====================
    provider_rate_limit: int = Field(default=40, validation_alias="PROVIDER_RATE_LIMIT")
    provider_rate_window: int = Field(
        default=60, validation_alias="PROVIDER_RATE_WINDOW"
    )
    provider_max_concurrency: int = Field(
        default=5, validation_alias="PROVIDER_MAX_CONCURRENCY"
    )

    # ==================== HTTP Client Timeouts ====================
    http_read_timeout: float = Field(
        default=300.0, validation_alias="HTTP_READ_TIMEOUT"
    )
    http_write_timeout: float = Field(
        default=10.0, validation_alias="HTTP_WRITE_TIMEOUT"
    )
    http_connect_timeout: float = Field(
        default=2.0, validation_alias="HTTP_CONNECT_TIMEOUT"
    )

    # ==================== Fast Prefix Detection ====================
    fast_prefix_detection: bool = True

    # ==================== Optimizations ====================
    enable_network_probe_mock: bool = True
    enable_title_generation_skip: bool = True
    enable_suggestion_mode_skip: bool = True
    enable_filepath_extraction_mock: bool = True

    # ==================== NIM Settings ====================
    nim: NimSettings = Field(default_factory=NimSettings)
    nim_enable_thinking: bool = Field(
        default=False, validation_alias="NIM_ENABLE_THINKING"
    )

    # ==================== Voice Note Transcription ====================
    voice_note_enabled: bool = Field(
        default=True, validation_alias="VOICE_NOTE_ENABLED"
    )
    # Device: "cpu" | "cuda" | "nvidia_nim"
    # - "cpu"/"cuda": local Whisper (requires voice_local extra: uv sync --extra voice_local)
    # - "nvidia_nim": NVIDIA NIM Whisper API (requires voice extra: uv sync --extra voice)
    whisper_device: str = Field(default="cpu", validation_alias="WHISPER_DEVICE")
    # Whisper model ID or short name (for local Whisper) or NVIDIA NIM model (for nvidia_nim)
    # Local Whisper: "tiny", "base", "small", "medium", "large-v2", "large-v3", "large-v3-turbo"
    # NVIDIA NIM: "nvidia/parakeet-ctc-1.1b-asr", "openai/whisper-large-v3", etc.
    whisper_model: str = Field(default="base", validation_alias="WHISPER_MODEL")
    # Hugging Face token for faster model downloads (optional, for local Whisper)
    hf_token: str = Field(default="", validation_alias="HF_TOKEN")

    # ==================== Bot Wrapper Config ====================
    telegram_bot_token: str | None = None
    allowed_telegram_user_id: str | None = None
    discord_bot_token: str | None = Field(
        default=None, validation_alias="DISCORD_BOT_TOKEN"
    )
    allowed_discord_channels: str | None = Field(
        default=None, validation_alias="ALLOWED_DISCORD_CHANNELS"
    )
    claude_workspace: str = "./agent_workspace"
    allowed_dir: str = ""

    # ==================== Server ====================
    host: str = "0.0.0.0"
    port: int = 8082
    log_file: str = "server.log"
    # Optional server API key to protect endpoints (Anthropic-style)
    # Set via env `ANTHROPIC_AUTH_TOKEN`. When empty, no auth is required.
    anthropic_auth_token: str = Field(
        default="", validation_alias="ANTHROPIC_AUTH_TOKEN"
    )

    # Handle empty strings for optional string fields
    @field_validator(
        "telegram_bot_token",
        "allowed_telegram_user_id",
        "discord_bot_token",
        "allowed_discord_channels",
        mode="before",
    )
    @classmethod
    def parse_optional_str(cls, v):
        if v == "":
            return None
        return v

    @field_validator("whisper_device")
    @classmethod
    def validate_whisper_device(cls, v: str) -> str:
        if v not in ("cpu", "cuda", "nvidia_nim"):
            raise ValueError(
                f"whisper_device must be 'cpu', 'cuda', or 'nvidia_nim', got {v!r}"
            )
        return v

    @field_validator("model", "model_opus", "model_sonnet", "model_haiku")
    @classmethod
    def validate_model_format(cls, v: str | None) -> str | None:
        if v is None:
            return None
        valid_providers = ("nvidia_nim", "open_router", "lmstudio", "llamacpp")
        if "/" not in v:
            raise ValueError(
                f"Model must be prefixed with provider type. "
                f"Valid providers: {', '.join(valid_providers)}. "
                f"Format: provider_type/model/name"
            )
        provider = v.split("/", 1)[0]
        if provider not in valid_providers:
            raise ValueError(
                f"Invalid provider: '{provider}'. "
                f"Supported: 'nvidia_nim', 'open_router', 'lmstudio', 'llamacpp'"
            )
        return v

    @model_validator(mode="after")
    def _inject_nim_thinking(self) -> Settings:
        self.nim = self.nim.model_copy(
            update={"enable_thinking": self.nim_enable_thinking}
        )
        return self

    @model_validator(mode="after")
    def check_nvidia_nim_api_key(self) -> Settings:
        if (
            self.voice_note_enabled
            and self.whisper_device == "nvidia_nim"
            and not self.nvidia_nim_api_key.strip()
        ):
            raise ValueError(
                "NVIDIA_NIM_API_KEY is required when WHISPER_DEVICE is 'nvidia_nim'. "
                "Set it in your .env file."
            )
        return self

    @property
    def provider_type(self) -> str:
        """Extract provider type from the default model string."""
        return self.model.split("/", 1)[0]

    @property
    def model_name(self) -> str:
        """Extract the actual model name from the default model string."""
        return self.model.split("/", 1)[1]

    def resolve_model(self, claude_model_name: str) -> str:
        """Resolve a Claude model name to the configured provider/model string.

        Classifies the incoming Claude model (opus/sonnet/haiku) and
        returns the model-specific override if configured, otherwise the fallback MODEL.
        """
        name_lower = claude_model_name.lower()
        if "opus" in name_lower and self.model_opus is not None:
            return self.model_opus
        if "haiku" in name_lower and self.model_haiku is not None:
            return self.model_haiku
        if "sonnet" in name_lower and self.model_sonnet is not None:
            return self.model_sonnet
        return self.model

    @staticmethod
    def parse_provider_type(model_string: str) -> str:
        """Extract provider type from any 'provider/model' string."""
        return model_string.split("/", 1)[0]

    @staticmethod
    def parse_model_name(model_string: str) -> str:
        """Extract model name from any 'provider/model' string."""
        return model_string.split("/", 1)[1]

    model_config = SettingsConfigDict(
        env_file=_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
