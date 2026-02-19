"""Pydantic models for Anthropic-compatible requests."""

from enum import StrEnum
from typing import Any, Literal, Self

from loguru import logger
from pydantic import BaseModel, field_validator, model_validator

from config.settings import get_settings
from providers.model_utils import normalize_model_name

# =============================================================================
# Content Block Types
# =============================================================================


class Role(StrEnum):
    user = "user"
    assistant = "assistant"
    system = "system"


class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_none_text(cls, data: Any) -> Any:
        """Model sometimes returns text: null — treat as empty string."""
        if isinstance(data, dict) and data.get("text") is None:
            return {**data, "text": ""}
        return data


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: dict[str, Any]


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: str | list[dict[str, Any]] | dict[str, Any] | list[Any] | Any


class ContentBlockThinking(BaseModel):
    type: Literal["thinking"]
    thinking: str
    signature: str | None = None  # Claude sometimes includes this field


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


# =============================================================================
# Message Types
# =============================================================================


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: (
        str
        | list[
            ContentBlockText
            | ContentBlockImage
            | ContentBlockToolUse
            | ContentBlockToolResult
            | ContentBlockThinking
        ]
    )
    reasoning_content: str | None = None


class Tool(BaseModel):
    name: str
    description: str | None = None
    input_schema: dict[str, Any]


class ThinkingConfig(BaseModel):
    """Supports both {enabled: true} and {type: "enabled"} formats."""

    enabled: bool = True
    type: str | None = None  # Claude Code sends {"type": "enabled"}

    @model_validator(mode="before")
    @classmethod
    def normalise(cls, data: Any) -> Any:
        """Accept {type: 'enabled'/'disabled'} as well as {enabled: bool}."""
        if isinstance(data, dict) and "type" in data and "enabled" not in data:
            return {**data, "enabled": data["type"] == "enabled"}
        return data


# =============================================================================
# Request Models
# =============================================================================


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int | None = None
    messages: list[Message]
    system: str | list[SystemContent] | None = None
    stop_sequences: list[str] | None = None
    stream: bool | None = True
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    metadata: dict[str, Any] | None = None
    tools: list[Tool] | None = None
    tool_choice: dict[str, Any] | None = None
    thinking: ThinkingConfig | None = None
    extra_body: dict[str, Any] | None = None
    original_model: str | None = None

    @model_validator(mode="after")
    def map_model(self) -> Self:
        """Map any Claude model name to the configured model."""
        settings = get_settings()
        if self.original_model is None:
            self.original_model = self.model

        # Use centralized model normalization
        normalized = normalize_model_name(self.model, settings.model)
        if normalized != self.model:
            self.model = normalized

        if self.model != self.original_model:
            logger.debug(f"MODEL MAPPING: '{self.original_model}' -> '{self.model}'")

        return self


class TokenCountRequest(BaseModel):
    model: str
    messages: list[Message]
    system: str | list[SystemContent] | None = None
    tools: list[Tool] | None = None
    thinking: ThinkingConfig | None = None
    tool_choice: dict[str, Any] | None = None

    @field_validator("model")
    @classmethod
    def validate_model_field(cls, v: str, info: Any) -> str:
        """Map any Claude model name to the configured model."""
        settings = get_settings()
        return normalize_model_name(v, settings.model)


# Force Pydantic to fully resolve all forward references.
# force=True is required for Python 3.14 + Pydantic 2.12 compat —
# without it the FastAPI TypeAdapter wrapper stays as a mock validator.
MessagesRequest.model_rebuild(force=True)
TokenCountRequest.model_rebuild(force=True)
