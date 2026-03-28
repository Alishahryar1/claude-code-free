#!/usr/bin/env zsh
# claude-switch — toggle Claude Code between NVIDIA NIM (free) and Anthropic (subscription)
#
# Usage:
#   claude-switch nim        — start proxy, log out of Anthropic, verify NIM is live
#   claude-switch anthropic  — stop proxy, log in to Anthropic, verify billing session

set -euo pipefail

PROXY_DIR="$HOME/Projects/liberated-claude-code"
PROXY_PORT=8082
PROXY_URL="http://localhost:$PROXY_PORT"
PROXY_PIDFILE="/tmp/liberated-claude-code.pid"
PROXY_LOG="/tmp/proxy.log"
VSCODE_SETTINGS="$HOME/Library/Application Support/Code/User/settings.json"

# ── Helpers ──────────────────────────────────────────────────────────────────

_proxy_running() {
    curl -sf "$PROXY_URL/health" &>/dev/null
}

_start_proxy() {
    if _proxy_running; then
        echo "  ✓ Proxy already running on port $PROXY_PORT"
        return
    fi
    echo "  Starting proxy..."
    cd "$PROXY_DIR"
    uv run uvicorn server:app --host 0.0.0.0 --port $PROXY_PORT >> "$PROXY_LOG" 2>&1 &
    echo $! > "$PROXY_PIDFILE"
    # Wait up to 10s for proxy to become healthy
    local i=0
    while ! _proxy_running; do
        sleep 0.5
        i=$((i + 1))
        if (( i >= 20 )); then
            echo "  ✗ Proxy failed to start. Check $PROXY_LOG"
            exit 1
        fi
    done
    echo "  ✓ Proxy started (pid $(cat $PROXY_PIDFILE))"
}

_stop_proxy() {
    if [[ -f "$PROXY_PIDFILE" ]]; then
        local pid=$(cat "$PROXY_PIDFILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            echo "  ✓ Proxy stopped (pid $pid)"
        fi
        rm -f "$PROXY_PIDFILE"
    elif _proxy_running; then
        # Kill by port if pidfile missing
        local pid=$(lsof -ti tcp:$PROXY_PORT 2>/dev/null | head -1)
        [[ -n "$pid" ]] && kill "$pid" && echo "  ✓ Proxy stopped (pid $pid)"
    else
        echo "  ✓ Proxy not running"
    fi
}

_vscode_set_nim() {
    python3 -c "
import json, sys
path = sys.argv[1]
with open(path) as f:
    s = json.load(f)
s['claudeCode.environmentVariables'] = [
    {'name': 'ANTHROPIC_BASE_URL', 'value': 'http://localhost:$PROXY_PORT'},
    {'name': 'ANTHROPIC_API_KEY',  'value': 'freecc'},
]
with open(path, 'w') as f:
    json.dump(s, f, indent=4)
print('  ✓ VS Code settings updated — reload the Claude extension to take effect')
" "$VSCODE_SETTINGS"
}

_vscode_set_anthropic() {
    python3 -c "
import json, sys
path = sys.argv[1]
with open(path) as f:
    s = json.load(f)
s.pop('claudeCode.environmentVariables', None)
with open(path, 'w') as f:
    json.dump(s, f, indent=4)
print('  ✓ VS Code settings restored — reload the Claude extension to take effect')
" "$VSCODE_SETTINGS"
}

_verify_nim() {
    echo ""
    echo "  Verifying NIM routing..."
    local response
    response=$(curl -sf "$PROXY_URL/health") || { echo "  ✗ Proxy not responding"; exit 1; }
    echo "  ✓ Proxy health: $response"

    # Hit the proxy with a minimal request and confirm it returns something
    local model_check
    model_check=$(curl -sf -X POST "$PROXY_URL/v1/messages?beta=true" \
        -H "Content-Type: application/json" \
        -H "x-api-key: freecc" \
        -H "anthropic-version: 2023-06-01" \
        -d '{
            "model": "claude-sonnet-4-5",
            "max_tokens": 50,
            "messages": [{"role": "user", "content": "Reply with only: NVIDIA NIM confirmed"}]
        }' 2>/dev/null) || { echo "  ✗ Test request to proxy failed"; exit 1; }

    echo "  ✓ Test request succeeded"
    echo "  ✓ Auth: API key mode (no Anthropic OAuth)"
    echo ""
    echo "  ⚡ Active provider: NVIDIA NIM"
    echo "  Models: sonnet→kimi-k2-thinking, opus→glm4.7, haiku→step-3.5-flash"
}

_verify_anthropic() {
    echo ""
    echo "  Verifying Anthropic routing..."
    local auth_status
    auth_status=$(claude auth status 2>/dev/null)
    if echo "$auth_status" | grep -q '"loggedIn": true'; then
        local email=$(echo "$auth_status" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('email','unknown'))" 2>/dev/null)
        local plan=$(echo "$auth_status" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('subscriptionType','unknown'))" 2>/dev/null)
        echo "  ✓ Auth: OAuth active"
        echo "  ✓ Account: $email ($plan)"
        echo ""
        echo "  ⚡ Active provider: Anthropic"
    else
        echo "  ✗ Not logged in to Anthropic"
        echo "  Run: claude auth login --claudeai"
        exit 1
    fi
}

# ── Commands ─────────────────────────────────────────────────────────────────

cmd_nim() {
    echo ""
    echo "━━━ Switching to NVIDIA NIM ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    _start_proxy
    echo "  Logging out of Anthropic OAuth..."
    claude auth logout 2>/dev/null && echo "  ✓ Logged out" || echo "  ✓ Already logged out"
    _vscode_set_nim
    _verify_nim
    echo ""
    echo "  Launch CLI: claude-nim   (or: ANTHROPIC_BASE_URL=$PROXY_URL ANTHROPIC_API_KEY=freecc claude)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

cmd_anthropic() {
    echo ""
    echo "━━━ Switching to Anthropic ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    _stop_proxy
    _vscode_set_anthropic
    echo "  Logging in to Anthropic (browser will open)..."
    claude auth login --claudeai
    _verify_anthropic
    echo ""
    echo "  Launch CLI: claude"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

# ── Entry point ───────────────────────────────────────────────────────────────

case "${1:-}" in
    nim)       cmd_nim ;;
    anthropic) cmd_anthropic ;;
    status)
        echo ""
        if _proxy_running; then
            echo "  ⚡ Provider: NVIDIA NIM (proxy running on port $PROXY_PORT)"
        else
            echo "  ⚡ Provider: Anthropic (proxy not running)"
        fi
        claude auth status 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    if d.get('loggedIn'):
        print('  Auth: OAuth active (' + d.get('email','') + ' / ' + d.get('subscriptionType','') + ')')
    else:
        print('  Auth: No OAuth session (API key mode)')
except: pass
" 2>/dev/null
        echo ""
        ;;
    *)
        echo "Usage: claude-switch [nim|anthropic|status]"
        exit 1
        ;;
esac
