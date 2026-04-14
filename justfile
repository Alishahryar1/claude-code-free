# liberated-claude-code automation
# Usage: just --list

default: (list)

# List available recipes
list:
    @just --list

# ── Proxy Management ────────────────────────────────────────────────────────────

# Start proxy server
proxy-start:
    uv run uvicorn server:app --host 0.0.0.0 --port 8082

# Stop proxy server
proxy-stop:
    -lsof -ti tcp:8082 | xargs kill -9 2>/dev/null || true

# Restart proxy server
proxy-restart: proxy-stop proxy-start

# Check proxy health
proxy-health:
    @curl -sf http://localhost:8082/health && echo "" || echo "✗ Proxy not running"

# ── Provider Switching ──────────────────────────────────────────────────────────

# Switch to NVIDIA NIM configuration
nvidia:
    @sed -i.bak 's/^MODEL=.*/MODEL=nvidia_nim\/meta\/llama3-70b-instruct/' .env
    @sed -i.bak 's/^MODEL_OPUS=.*/MODEL_OPUS=nvidia_nim\/moonshotai\/kimi-k2-thinking/' .env
    @sed -i.bak 's/^MODEL_SONNET=.*/MODEL_SONNET=nvidia_nim\/meta\/llama3-70b-instruct/' .env
    @sed -i.bak 's/^MODEL_HAIKU=.*/MODEL_HAIKU=nvidia_nim\/stepfun-ai\/step-3.5-flash/' .env
    @echo "✓ Switched to NVIDIA NIM"
    @echo "  Run 'just proxy-restart' to apply changes"

# Switch to Modal GLV5 configuration
modal:
    @sed -i.bak 's/^MODEL=.*/MODEL=modal\/zai-org\/GLM-5.1-FP8/' .env
    @sed -i.bak 's/^MODEL_OPUS=.*/MODEL_OPUS=modal\/zai-org\/GLM-5.1-FP8/' .env
    @sed -i.bak 's/^MODEL_SONNET=.*/MODEL_SONNET=modal\/zai-org\/GLM-5.1-FP8/' .env
    @sed -i.bak 's/^MODEL_HAIKU=.*/MODEL_HAIKU=modal\/zai-org\/GLM-5.1-FP8/' .env
    @echo "✓ Switched to Modal GLV5"
    @echo "  Run 'just proxy-restart' to apply changes"

# Show current provider configuration
provider-status:
    @echo "Current .env provider settings:"
    @grep "^MODEL" .env || echo "No MODEL vars found"

# ── Claude Code Launch ──────────────────────────────────────────────────────────

# Launch Claude Code with proxy
claude:
    ANTHROPIC_BASE_URL=http://localhost:8082 ANTHROPIC_API_KEY=freecc claude

# ── Testing ──────────────────────────────────────────────────────────────────────

# Run all tests
test:
    uv run pytest -v

# Run tests with coverage
test-cov:
    uv run pytest --cov=. --cov-report=term-missing

# Run specific provider tests
test-modal:
    uv run pytest tests/providers/test_modal.py -v

# ── Development ──────────────────────────────────────────────────────────────────

# Format code with ruff
fmt:
    uv run ruff format .

# Lint code
lint:
    uv run ruff check .

# Type check with ty
typecheck:
    uv run ty check

# ── VS Code Integration ──────────────────────────────────────────────────────────

# Configure VS Code for proxy usage
vscode-proxy:
    #!/usr/bin/env python3
    import json
    import os
    path = os.path.expanduser('~/Library/Application Support/Code/User/settings.json')
    s = json.load(open(path))
    s['claudeCode.environmentVariables'] = [
        {'name': 'ANTHROPIC_BASE_URL', 'value': 'http://localhost:8082'},
        {'name': 'ANTHROPIC_API_KEY', 'value': 'freecc'}
    ]
    json.dump(s, open(path, 'w'), indent=4)
    print('✓ VS Code configured for proxy')
    @echo "  Reload VS Code Claude extension to take effect"

# Restore VS Code to direct Anthropic connection
vscode-anthropic:
    #!/usr/bin/env python3
    import json
    import os
    path = os.path.expanduser('~/Library/Application Support/Code/User/settings.json')
    s = json.load(open(path))
    s.pop('claudeCode.environmentVariables', None)
    json.dump(s, open(path, 'w'), indent=4)
    print('✓ VS Code restored to direct Anthropic')
    @echo "  Reload VS Code Claude extension to take effect"
