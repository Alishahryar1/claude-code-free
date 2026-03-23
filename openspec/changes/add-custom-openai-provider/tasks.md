## 1. Configuration

- [x] 1.1 Add `custom_openai_base_url` field to `config/settings.py` with validation alias `CUSTOM_OPENAI_BASE_URL`
- [x] 1.2 Add `custom_openai_api_key` field to `config/settings.py` with validation alias `CUSTOM_OPENAI_API_KEY`
- [x] 1.3 Update `validate_model_format()` validator to accept `custom_openai` as valid provider
- [x] 1.4 Update error messages in validator to include `custom_openai` in supported providers list

## 2. Provider Implementation

- [x] 2.1 Create `providers/custom_openai/` directory
- [x] 2.2 Create `providers/custom_openai/__init__.py` that exports `CustomOpenAIProvider`
- [x] 2.3 Create `providers/custom_openai/request.py` with `build_request_body()` function using `build_base_request_body()`
- [x] 2.4 Create `providers/custom_openai/client.py` with `CustomOpenAIProvider` class extending `OpenAICompatibleProvider`
- [x] 2.5 Implement `__init__()` to call parent with `provider_name="CUSTOM_OPENAI"` and user-configured base URL

## 3. Provider Registration

- [x] 3.1 Add import for `CustomOpenAIProvider` in `api/dependencies.py`
- [x] 3.2 Add `custom_openai` case to `_create_provider_for_type()` function
- [x] 3.3 Add validation to raise `AuthenticationError` if `custom_openai_api_key` is missing or empty
- [x] 3.4 Add validation to raise error if `custom_openai_base_url` is missing or empty
- [x] 3.5 Create `ProviderConfig` with appropriate settings and return `CustomOpenAIProvider` instance
- [x] 3.6 Update error message in `ValueError` for unknown provider_type to include `custom_openai`

## 4. Documentation

- [x] 4.1 Add custom OpenAI provider section to `.env.example` with detailed examples
- [x] 4.2 Document `CUSTOM_OPENAI_BASE_URL` with example value and description
- [x] 4.3 Document `CUSTOM_OPENAI_API_KEY` with placeholder and description
- [x] 4.4 Add example model mappings for Opus, Sonnet, and Haiku (similar to other providers in README)
- [x] 4.5 Include at least 3 example configurations: Azure OpenAI, Groq, Together AI, or generic OpenAI-compatible API

## 5. Tests

- [x] 5.1 Create `tests/providers/test_custom_openai.py`
- [x] 5.2 Add test for provider initialization with valid configuration
- [x] 5.3 Add test for `AuthenticationError` when API key is missing
- [x] 5.4 Add test for error when base URL is missing
- [x] 5.5 Add test for `build_request_body()` with standard request
- [x] 5.6 Add test for request building with tools
- [x] 5.7 Add mock test for streaming response handling
- [x] 5.8 Run `uv run pytest tests/providers/test_custom_openai.py` to verify all tests pass

## 6. Validation

- [x] 6.1 Run `uv run ruff format` to check code formatting
- [x] 6.2 Run `uv run ruff check` to verify no linting errors
- [x] 6.3 Run `uv run ty check` to verify type checking passes
- [x] 6.4 Run full test suite with `uv run pytest` to ensure no regressions
- [x] 6.5 Verify server starts successfully with `uv run uvicorn server:app --host 0.0.0.0 --port 8082`
