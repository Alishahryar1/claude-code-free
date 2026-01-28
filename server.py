"""
Claude Code Proxy - NVIDIA NIM Implementation

This server acts as a robust proxy between Anthropic API requests and NVIDIA NIM,
enabling Claude Code CLI to utilize NIM models with full support for:
- Streaming with SSE (Server-Sent Events)
- Thinking/Reasoning blocks and Reasoning-Split mode
- Native and heuristic tool use parsing
- Automatic model mapping (Haiku/Sonnet/Opus to NIM equivalents)
- Fast prefix detection for CLI policy specifications
"""

import os
import json
import logging
import uuid
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, field_validator, model_validator
from providers.nvidia_nim import NvidiaNimProvider, ProviderConfig
from providers.exceptions import ProviderError
import uvicorn
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import StreamingResponse, JSONResponse
import tiktoken

# Initialize tokenizer
try:
    ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    logger.warning("Could not load cl100k_base encoding, falling back to p50k_base")
    ENCODER = tiktoken.get_encoding("p50k_base")

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("server.log", encoding="utf-8", mode="w")],
)
logger = logging.getLogger(__name__)

logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("uvicorn.error").setLevel(logging.WARNING)

# =============================================================================
# Models
# =============================================================================

BIG_MODEL = os.getenv("BIG_MODEL", "moonshotai/kimi-k2-instruct")
SMALL_MODEL = os.getenv("SMALL_MODEL", "moonshotai/kimi-k2-instruct")


class ContentBlockText(BaseModel):
    type: Literal["text"]
    text: str


class ContentBlockImage(BaseModel):
    type: Literal["image"]
    source: Dict[str, Any]


class ContentBlockToolUse(BaseModel):
    type: Literal["tool_use"]
    id: str
    name: str
    input: Dict[str, Any]


class ContentBlockToolResult(BaseModel):
    type: Literal["tool_result"]
    tool_use_id: str
    content: Union[str, List[Dict[str, Any]], Dict[str, Any], List[Any], Any]


class ContentBlockThinking(BaseModel):
    type: Literal["thinking"]
    thinking: str


class SystemContent(BaseModel):
    type: Literal["text"]
    text: str


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: Union[
        str,
        List[
            Union[
                ContentBlockText,
                ContentBlockImage,
                ContentBlockToolUse,
                ContentBlockToolResult,
                ContentBlockThinking,
            ]
        ],
    ]
    reasoning_content: Optional[str] = None


class Tool(BaseModel):
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any]


class ThinkingConfig(BaseModel):
    enabled: bool = True


class MessagesRequest(BaseModel):
    model: str
    max_tokens: int
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    stop_sequences: Optional[List[str]] = None
    stream: Optional[bool] = False
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    tools: Optional[List[Tool]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    thinking: Optional[ThinkingConfig] = None
    extra_body: Optional[Dict[str, Any]] = None
    original_model: Optional[str] = None

    @model_validator(mode="after")
    def map_model(self) -> "MessagesRequest":
        if self.original_model is None:
            self.original_model = self.model

        clean_v = self.model
        for prefix in ["anthropic/", "openai/", "gemini/"]:
            if clean_v.startswith(prefix):
                clean_v = clean_v[len(prefix) :]
                break

        if "haiku" in clean_v.lower():
            self.model = SMALL_MODEL
        elif "sonnet" in clean_v.lower() or "opus" in clean_v.lower():
            self.model = BIG_MODEL

        if self.model != self.original_model:
            logger.debug(f"MODEL MAPPING: '{self.original_model}' -> '{self.model}'")

        return self


class TokenCountRequest(BaseModel):
    model: str
    messages: List[Message]
    system: Optional[Union[str, List[SystemContent]]] = None
    tools: Optional[List[Tool]] = None
    thinking: Optional[ThinkingConfig] = None
    tool_choice: Optional[Dict[str, Any]] = None

    @field_validator("model")
    @classmethod
    def validate_model_field(cls, v, info):
        clean_v = v
        for prefix in ["anthropic/", "openai/", "gemini/"]:
            if clean_v.startswith(prefix):
                clean_v = clean_v[len(prefix) :]
                break

        if "haiku" in clean_v.lower():
            return SMALL_MODEL
        elif "sonnet" in clean_v.lower() or "opus" in clean_v.lower():
            return BIG_MODEL
        return v


class TokenCountResponse(BaseModel):
    input_tokens: int


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class MessagesResponse(BaseModel):
    id: str
    model: str
    role: Literal["assistant"] = "assistant"
    content: List[
        Union[
            ContentBlockText, ContentBlockToolUse, ContentBlockThinking, Dict[str, Any]
        ]
    ]
    type: Literal["message"] = "message"
    stop_reason: Optional[
        Literal["end_turn", "max_tokens", "stop_sequence", "tool_use"]
    ] = None
    stop_sequence: Optional[str] = None
    usage: Usage


# =============================================================================
# Provider
# =============================================================================

provider_config = ProviderConfig(
    api_key=os.getenv("NVIDIA_NIM_API_KEY", ""),
    base_url=os.getenv("NVIDIA_NIM_BASE_URL", "https://integrate.api.nvidia.com/v1"),
    rate_limit=int(os.getenv("NVIDIA_NIM_RATE_LIMIT", "40")),
    rate_window=int(os.getenv("NVIDIA_NIM_RATE_WINDOW", "60")),
)

# Global provider instance for DI
_provider: Optional[NvidiaNimProvider] = None


def get_provider() -> NvidiaNimProvider:
    global _provider
    if _provider is None:
        _provider = NvidiaNimProvider(provider_config)
    return _provider


# =============================================================================
# FastAPI App
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting up...")
    yield
    logger.info("Server shutting down...")
    global _provider
    if _provider and hasattr(_provider, "_client"):
        await _provider._client.aclose()


FAST_PREFIX_DETECTION = os.getenv("FAST_PREFIX_DETECTION", "true").lower() == "true"


app = FastAPI(title="Claude Code Proxy", version="2.0.0", lifespan=lifespan)


@app.exception_handler(ProviderError)
async def provider_error_handler(request: Request, exc: ProviderError):
    """Handle provider-specific errors and return Anthropic format."""
    logger.error(f"Provider Error: {exc.error_type} - {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_anthropic_format(),
    )


@app.exception_handler(Exception)
async def general_error_handler(request: Request, exc: Exception):
    """Handle general errors and return Anthropic format."""
    logger.error(f"General Error: {str(exc)}")
    import traceback

    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "type": "error",
            "error": {
                "type": "api_error",
                "message": "An unexpected error occurred.",
            },
        },
    )


def extract_command_prefix(command: str) -> str:
    import shlex

    if "`" in command or "$(" in command:
        return "command_injection_detected"

    try:
        parts = shlex.split(command)
        if not parts:
            return "none"

        env_prefix = []
        cmd_start = 0
        for i, part in enumerate(parts):
            if "=" in part and not part.startswith("-"):
                env_prefix.append(part)
                cmd_start = i + 1
            else:
                break

        if cmd_start >= len(parts):
            return "none"

        cmd_parts = parts[cmd_start:]
        if not cmd_parts:
            return "none"

        first_word = cmd_parts[0]
        two_word_commands = {
            "git",
            "npm",
            "docker",
            "kubectl",
            "cargo",
            "go",
            "pip",
            "yarn",
        }

        if first_word in two_word_commands and len(cmd_parts) > 1:
            second_word = cmd_parts[1]
            if not second_word.startswith("-"):
                return f"{first_word} {second_word}"
            return first_word
        return first_word if not env_prefix else " ".join(env_prefix) + " " + first_word

    except ValueError:
        return command.split()[0] if command.split() else "none"


def is_prefix_detection_request(request_data: MessagesRequest) -> tuple[bool, str]:
    if len(request_data.messages) != 1 or request_data.messages[0].role != "user":
        return False, ""

    msg = request_data.messages[0]
    content = ""
    if isinstance(msg.content, str):
        content = msg.content
    elif isinstance(msg.content, list):
        for block in msg.content:
            if hasattr(block, "text"):
                content += block.text

    if "<policy_spec>" in content and "Command:" in content:
        try:
            cmd_start = content.rfind("Command:") + len("Command:")
            return True, content[cmd_start:].strip()
        except Exception:
            pass

    return False, ""


def get_token_count(messages, system=None, tools=None) -> int:
    total_tokens = 0

    if system:
        if isinstance(system, str):
            total_tokens += len(ENCODER.encode(system))
        elif isinstance(system, list):
            for block in system:
                if hasattr(block, "text"):
                    total_tokens += len(ENCODER.encode(block.text))

    for msg in messages:
        if isinstance(msg.content, str):
            total_tokens += len(ENCODER.encode(msg.content))
        elif isinstance(msg.content, list):
            for block in msg.content:
                # Handle dictionary or Pydantic model
                b_type = getattr(block, "type", None)

                if b_type == "text":
                    total_tokens += len(ENCODER.encode(getattr(block, "text", "")))
                elif b_type == "thinking":
                    # Thinking tokens are part of context if they are in history
                    total_tokens += len(ENCODER.encode(getattr(block, "thinking", "")))
                elif b_type == "tool_use":
                    name = getattr(block, "name", "")
                    inp = getattr(block, "input", {})
                    # Add tokens for definitions
                    total_tokens += len(ENCODER.encode(name))
                    total_tokens += len(ENCODER.encode(json.dumps(inp)))
                    total_tokens += 10  # Control tokens approximate
                elif b_type == "tool_result":
                    content = getattr(block, "content", "")
                    if isinstance(content, str):
                        total_tokens += len(ENCODER.encode(content))
                    else:
                        total_tokens += len(ENCODER.encode(json.dumps(content)))
                    total_tokens += 5  # Control tokens approximate

    if tools:
        for tool in tools:
            # Approximate tool definition tokens
            tool_str = (
                tool.name + (tool.description or "") + json.dumps(tool.input_schema)
            )
            total_tokens += len(ENCODER.encode(tool_str))

    # Add some overhead for message formatting (approx 3 tokens per message)
    total_tokens += len(messages) * 3
    if tools:
        total_tokens += len(tools) * 5  # Extra overhead for tool definitions

    return max(1, total_tokens)


def log_request_details(request_data: MessagesRequest):
    """Log detailed request content for debugging."""

    def sanitize(text: str, max_len: int = 200) -> str:
        """Escape newlines and truncate for single-line logging."""
        text = text.replace("\n", "\\n").replace("\r", "\\r")
        return text[:max_len] + "..." if len(text) > max_len else text

    for i, msg in enumerate(request_data.messages):
        role = msg.role
        if isinstance(msg.content, str):
            logger.debug(f"  [{i}] {role}: {sanitize(msg.content)}")
        elif isinstance(msg.content, list):
            text_acc = []
            for block in msg.content:
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    text_acc.append(getattr(block, "text", ""))
                else:
                    if text_acc:
                        logger.debug(
                            f"  [{i}] {role}/text: {sanitize(''.join(text_acc))}"
                        )
                        text_acc = []
                    if block_type == "tool_use":
                        name = getattr(block, "name", "unknown")
                        inp = getattr(block, "input", {})
                        logger.debug(
                            f"  [{i}] {role}/tool_use: {name}({sanitize(json.dumps(inp), 500)})"
                        )
                    elif block_type == "tool_result":
                        content = getattr(block, "content", "")
                        tool_use_id = getattr(block, "tool_use_id", "unknown")
                        logger.debug(
                            f"  [{i}] {role}/tool_result[{tool_use_id}]: {sanitize(str(content))}"
                        )
                    elif block_type == "thinking":
                        thinking = getattr(block, "thinking", "")
                        logger.debug(f"  [{i}] {role}/thinking: {sanitize(thinking)}")
            if text_acc:
                logger.debug(f"  [{i}] {role}/text: {sanitize(''.join(text_acc))}")


@app.post("/v1/messages")
async def create_message(
    request_data: MessagesRequest,
    raw_request: Request,
    provider: NvidiaNimProvider = Depends(get_provider),
):
    try:
        if FAST_PREFIX_DETECTION:
            is_prefix_req, command = is_prefix_detection_request(request_data)
            if is_prefix_req:
                import uuid

                return MessagesResponse(
                    id=f"msg_{uuid.uuid4()}",
                    model=request_data.model,
                    content=[{"type": "text", "text": extract_command_prefix(command)}],
                    stop_reason="end_turn",
                    usage=Usage(input_tokens=100, output_tokens=5),
                )

        logger.info(
            f"Request: model={request_data.model}, messages={len(request_data.messages)}, stream={request_data.stream}"
        )
        log_request_details(request_data)

        if request_data.stream:
            input_tokens = get_token_count(
                request_data.messages, request_data.system, request_data.tools
            )
            return StreamingResponse(
                provider.stream_response(request_data, input_tokens=input_tokens),
                media_type="text/event-stream",
                headers={
                    "X-Accel-Buffering": "no",
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )
        else:
            response_json = await provider.complete(request_data)
            return provider.convert_response(response_json, request_data)

    except ProviderError:
        # Re-raise ProviderError to be handled by the specialized exception handler
        raise
    except Exception as e:
        import traceback

        logger.error(f"Error: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=getattr(e, "status_code", 500), detail=str(e))


@app.post("/v1/messages/count_tokens")
async def count_tokens(request_data: TokenCountRequest):
    try:
        return TokenCountResponse(
            input_tokens=get_token_count(
                request_data.messages, request_data.system, request_data.tools
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    return {
        "status": "ok",
        "provider": "nvidia_nim",
        "big_model": BIG_MODEL,
        "small_model": SMALL_MODEL,
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8082, log_level="debug")
