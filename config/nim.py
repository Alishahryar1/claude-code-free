"""NVIDIA NIM settings (fixed values, no env config)."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

class NimSettings(BaseModel):
    """Fixed NVIDIA NIM settings (not configurable via env)."""

    temperature: float = Field(1.0, ge=0.0, le=2.0, description="Sampling temperature, must be >=0 and <=2.")  # Added le=2.0 for safety
    top_p: float = Field(1.0, ge=0.0, le=1.0, description="Nucleus sampling probability. [0,1]")
    top_k: int = -1
    max_tokens: int = Field(81920, ge=1, description="Maximum number of tokens in output.")
    presence_penalty: float = Field(0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(0.0, ge=-2.0, le=2.0)

    min_p: float = Field(0.0, ge=0.0, le=1.0, description="Minimum probability threshold [0,1].")
    repetition_penalty: float = Field(1.0, ge=0.0, description="Penalty for repeated tokens. Must be >=0.")

    seed: int | None = None
    stop: str | None = None

    parallel_tool_calls: bool = True
    ignore_eos: bool = False

    min_tokens: int = Field(0, ge=0, description="Minimum tokens in the response.")
    chat_template: str | None = None
    request_id: str | None = None

    model_config = ConfigDict(extra="forbid")

    # Validators
    @field_validator("top_k", mode="before")
    @classmethod
    def validate_top_k(cls, v):
        if v is None or v == "":
            return -1
        int_v = int(v)
        if int_v < -1:
            raise ValueError("top_k must be -1 or >= 0")
        return int_v

    @field_validator("temperature", "top_p", "min_p", "presence_penalty",
                     "frequency_penalty", "repetition_penalty", pre=True)
    @classmethod
    def validate_float_fields(cls, v, field):
        # Allow string conversion if possible, otherwise fail
        if v is None or v == "":
            return field.default
        try:
            val = float(v)
        except Exception:
            raise ValueError(f"{field.name} must be a float. Got {type(v).__name__}.")
        return val

    @field_validator("max_tokens", "min_tokens", pre=True)
    @classmethod
    def validate_int_fields(cls, v, field):
        if v is None or v == "":
            return field.default
        try:
            val = int(v)
        except Exception:
            raise ValueError(f"{field.name} must be an int. Got {type(v).__name__}.")
        return val

    @field_validator("seed", mode="before")
    @classmethod
    def parse_optional_int(cls, v):
        if v == "" or v is None:
            return None
        try:
            return int(v)
        except Exception:
            raise ValueError("seed must be an int or empty/None.")

    @field_validator("stop", "chat_template", "request_id", mode="before")
    @classmethod
    def parse_optional_str(cls, v):
        if v == "":
            return None
        if v is not None and not isinstance(v, str):
            return str(v)
        return v