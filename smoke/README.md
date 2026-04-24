# Local Live Smoke Tests

These tests are for maintainers running against their own `.env`. They are not
part of CI and are not collected by plain `uv run pytest`.

## Safe Default Run

```powershell
$env:FCC_LIVE_SMOKE = "1"
uv run pytest smoke -n 0 -m live -s --tb=short
```

`-n 0` is recommended because the normal project pytest config enables xdist.
The smoke suite can run with workers, but one process gives clearer logs when a
real provider or bot fails.

## Targeted Runs

```powershell
$env:FCC_LIVE_SMOKE = "1"
$env:FCC_SMOKE_TARGETS = "api,providers,thinking,tools"
uv run pytest smoke -n 0 -m live -s --tb=short
```

Use `FCC_SMOKE_TARGETS=all` to include Telegram, Discord, and voice checks.
The default target set intentionally excludes those side-effectful integrations.

## Environment

- `FCC_ENV_FILE`: optional explicit dotenv path. The app still uses its normal
  env-file precedence.
- `FCC_SMOKE_PROVIDER_MATRIX`: comma-separated provider prefixes to test.
- `FCC_SMOKE_TIMEOUT_S`: per-request/subprocess timeout, default `45`.
- `FCC_SMOKE_CLAUDE_BIN`: Claude CLI executable name, default `claude`.
- `FCC_SMOKE_TELEGRAM_CHAT_ID`: Telegram chat/user ID for send/edit/delete.
- `FCC_SMOKE_DISCORD_CHANNEL_ID`: Discord channel ID for send/edit/delete.
- `FCC_SMOKE_INTERACTIVE=1`: enables manual inbound messaging checks.
- `FCC_SMOKE_RUN_VOICE=1`: allows the voice transcription backend to load/run.

## Results

Smoke artifacts are written to `.smoke-results/` and ignored by git. Reports and
logs redact env values whose names contain `KEY`, `TOKEN`, `SECRET`, `WEBHOOK`,
or `AUTH`.

## How To Read Failures

- `missing_env`: configure the required key, token, channel, or local base URL.
- `upstream_unavailable`: the provider/local model/bot API is not reachable.
- `product_failure`: the app returned the wrong shape or crashed.
- `harness_bug`: the smoke test itself made an invalid assumption.

The first real run is expected to find product failures. Fix those separately
from harness problems so the suite becomes a reliable regression signal.
