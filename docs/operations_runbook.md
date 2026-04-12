# Operations Runbook

This runbook covers the day-to-day operational concerns for running the
market prediction agent against live / public data sources. It assumes the
project has already been installed (`uv sync`) and all tests pass.

## 1. Environment and credentials

Copy `.env.example` to `.env` and fill in only the keys you actually use.
None of the primary public sources require a paid key, so minimal live
operation is possible without any credentials.

| Variable | Used by | Required? |
|---|---|---|
| `POLYGON_API_KEY` | Polygon OHLCV primary | Optional (fallbacks cover it) |
| `ALPHAVANTAGE_API_KEY` | Alpha Vantage OHLCV fallback | Optional |
| `FRED_API_KEY` | FRED API adapter | Optional (CSV mirror works without a key) |
| `COINGECKO_API_KEY` | Crypto adapter | Optional |
| `ANTHROPIC_API_KEY` | LLM helpers only | Optional |
| `AUDIT_SLACK_WEBHOOK_URL` | Scheduled audit Slack notifications | Optional |
| `AUDIT_EMAIL_SMTP_HOST` / `AUDIT_EMAIL_TO` | Scheduled audit email notifications | Optional |
| `AUDIT_NOTIFY_MIN_SEVERITY` | Notification threshold (`warning` default) | Optional |
| `CONFIRM_LIVE_TRADING` | Must stay `no` | **Required** |
| `EMERGENCY_STOP` | Kill switch | Required (`false` by default) |

All `trading.enable_live_trading` paths are explicitly disabled in
`config/default.yaml`. Live broker routing is out of scope.

## 2. Public data source chain

The operational default is the `full_light` public-audit profile, which
uses the following public data chain. All primary sources are keyless.

| Family | Primary | Secondary | Fallback |
|---|---|---|---|
| OHLCV | `polygon` (if key) | `alphavantage` (if key) | `yahoo_chart`, `stooq`, `fred_market_proxy` |
| News | `yahoo_finance_rss` | `google_news_rss` | `offline_news_proxy` |
| Fundamentals | `sec_companyfacts` | -- | `offline_fundamental_proxy` |
| Macro | FRED CSV mirror | -- | local cache |

Snapshot fallback lives under `storage/public_data/snapshots/<source>/`.
When the network is unavailable, the adapter will transparently fall back
to the most recent snapshot. You can seed snapshots in advance with
`scripts/seed_public_snapshots.py`.

## 3. Rate limits and retries

| Source | Rate limit (approx.) | Retry policy |
|---|---|---|
| Polygon | 5 req/min (free) | Retry 3x, exponential backoff |
| Alpha Vantage | 5 req/min (free) | Retry 3x |
| Yahoo chart / RSS | Unofficial — self throttle | Retry 2x |
| Google News RSS | Unofficial — self throttle | Retry 2x |
| SEC CompanyFacts | 10 req/sec | Retry 3x |
| FRED | 120 req/min | Retry 3x |

Retry counts and caches are visible in the `transport.requests[*]` section
of every audit / smoke output. A non-zero `retry_count` is expected on
transient failures; a non-empty `last_error` with `origin != "network"`
indicates the adapter fell back to cache or snapshot.

## 4. Smoke tests (run these before any scheduled audit)

```bash
# News chain (yahoo_finance_rss + google_news_rss)
uv run python scripts/run_news_live_smoke.py --tickers SPY,QQQ --lookback-days 5

# Fundamentals chain (sec_companyfacts + yahoo_chart for aux price)
uv run python scripts/run_fundamentals_live_smoke.py --tickers AAPL,MSFT --lookback-days 252

# MCP stdio server smoke
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"initialize"}' \
  '{"jsonrpc":"2.0","id":2,"method":"tools/list"}' \
  '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"market_data_fetcher","arguments":{"ticker":"TICK001","start_date":"2026-04-01","end_date":"2026-04-10","source":"dummy"}}}' \
  | uv run python scripts/mcp_server.py
```

Smoke tests pass if:

- `fallback_used` is false OR the fallback origin is a known cache /
  snapshot (not an exception).
- `record_count` / `headline_count` is non-zero.
- `stale_rate` is below the per-family threshold configured in
  `config/default.yaml`.

## 5. Scheduled audit jobs

Routine monitor audits use the `full_light` profile. Measured runtimes on
Windows 11, Python 3.13, dummy data, single machine (no concurrency):

| Profile | Runs | Dummy runtime (measured) | Live runtime (measured) |
|---|---|---|---|
| `fast` | 3 | **357s (6m)** | not measured |
| `standard` | ~5 | not measured | not measured |
| `full_light` | **12** | **1531s (25m 31s)** | **1366s (22m 46s)** |
| `full` | 12+ | not measured | not measured |

Surprisingly, the live run was faster than dummy on this machine because
the real-data ticker set (4 ETFs) is smaller than the default synthetic
set. Always measure in the target environment before scheduling.
Measurements above are from 2026-04-12 using anchor-date auto-generation
(`full_light` generates 12 weekly as-of dates).

```bash
# Manual run
uv run python scripts/run_public_audit_suite.py --profile full_light

# Deep-dive (ad-hoc)
uv run python scripts/run_public_audit_suite.py --profile full
```

### Automated scheduling (Windows Task Scheduler)

A weekly task is registered as `MarketPredictionAgent-WeeklyAudit`.

```powershell
# Register (requires Admin):
.\scripts\register_scheduled_task.ps1

# Change schedule:
.\scripts\register_scheduled_task.ps1 -DayOfWeek Sunday -Time "08:00"

# Remove:
.\scripts\register_scheduled_task.ps1 -Unregister

# Test immediately:
Start-ScheduledTask -TaskName 'MarketPredictionAgent-WeeklyAudit'
```

The wrapper script `scripts/scheduled_audit.ps1` handles logging and
status tracking:

- **Logs**: `storage/logs/scheduled_audits/<timestamp>.log`
- **Status**: `storage/logs/scheduled_audits/latest_run.json`

Check the latest run status:

```powershell
Get-Content .\storage\logs\scheduled_audits\latest_run.json
```

Suite output lands under
`storage/outputs/monitor_audit_suites/public_real_market/<date>/`. Each
run is schema-validated at write time.

## 6. Failure handling

1. **Snapshot fallback triggered** — expected under outages. Check
   `transport.by_source.<name>.origins` for `"snapshot"` and confirm the
   `fetched_at` is within the per-family stale threshold.
2. **Primary source fully failed, secondary succeeded** — the audit is
   still valid; inspect `feature_sources.*.used_source` in the audit
   report to verify which source was actually used.
3. **All sources failed** — the adapter raises, and the audit run fails
   fast. Rerun after addressing the outage; there is no silent
   degradation.
4. **Rate limit hit on primary** — retry with exponential backoff kicks
   in automatically. If persistent, lower the request concurrency or fall
   back to a different primary via `config/default.yaml`.
5. **Retraining monitor flagged `should_retrain = true`** — follow the
   retraining ledger in
   `storage/outputs/retraining_event_ledger.parquet` and decide whether
   the trigger is a legitimate drift or a transient regime shift.

## 7. Notifications

`scripts/scheduled_audit.ps1` invokes `scripts/notify_audit_status.py`
after every scheduled run. The notification hook reads
`storage/logs/scheduled_audits/latest_run.json`, locates the latest suite
artifact under `storage/outputs/monitor_audit_suites/public_real_market/`,
and evaluates these implemented alert conditions:

- Audit process failure (`success=false` or non-zero `exit_code`)
- Effective retraining rate above `AUDIT_NOTIFY_RETRAINING_RATE_THRESHOLD`
- Base/watch-only retraining, drift, or regime signals when
  `AUDIT_NOTIFY_INCLUDE_WATCH_ONLY=true`
- News fallback or news staleness above configured thresholds
- Cluster-adjusted PBO mean/max above configured thresholds

<!-- Legacy pre-implementation notes are superseded by the checks above:

- `suite_payload.run_count` — ensure all expected runs completed.
- `suite_payload.distribution_summary.retraining_trigger_rate` — alert
  if above a configured threshold.
- `suite_payload.distribution_summary.regime_shift_rate` — alert if
  elevated.
- Any non-zero `failed_sources` in the transport metadata.

-->

By default, the hook sends only `warning` or `critical` findings. A clean
run prints `No notification needed` into the audit log and exits with
code 0.

Configure Slack by setting:

```powershell
$env:AUDIT_SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/..."
```

Configure email by setting:

```powershell
$env:AUDIT_EMAIL_SMTP_HOST = "smtp.example.com"
$env:AUDIT_EMAIL_SMTP_PORT = "587"
$env:AUDIT_EMAIL_USE_TLS = "true"
$env:AUDIT_EMAIL_SMTP_USER = "user@example.com"
$env:AUDIT_EMAIL_SMTP_PASSWORD = "<secret>"
$env:AUDIT_EMAIL_FROM = "market-agent@example.com"
$env:AUDIT_EMAIL_TO = "operator@example.com"
```

Dry-run the current status without sending:

```powershell
uv run python .\scripts\notify_audit_status.py --dry-run
```

Useful thresholds:

| Variable | Default | Meaning |
|---|---:|---|
| `AUDIT_NOTIFY_MIN_SEVERITY` | `warning` | Minimum severity to send (`ok`, `info`, `warning`, `critical`) |
| `AUDIT_NOTIFY_ON_OK` | `false` | Send even when no findings exist |
| `AUDIT_NOTIFY_RETRAINING_RATE_THRESHOLD` | `0.0` | Critical if effective retraining rate is above this |
| `AUDIT_NOTIFY_BASE_RETRAINING_RATE_THRESHOLD` | `0.0` | Warning if base retraining trigger rate is above this |
| `AUDIT_NOTIFY_NEWS_FALLBACK_RATE_THRESHOLD` | `0.0` | Warning if news fallback rate is above this |
| `AUDIT_NOTIFY_NEWS_STALENESS_THRESHOLD` | `0.0` | Warning if mean news staleness is above this |
| `AUDIT_NOTIFY_CLUSTER_PBO_WARNING_MEAN` | `0.6` | Warning if mean cluster-adjusted PBO is above this |
| `AUDIT_NOTIFY_CLUSTER_PBO_CRITICAL_MAX` | `0.85` | Critical if max cluster-adjusted PBO is at or above this |

## 8. Backups

Parquet outputs under `storage/` are append-only. The most important
artifacts to back up are:

- `storage/outputs/retraining_event_ledger.parquet` — audit ledger
- `storage/outputs/audit_reports/` — audit history
- `storage/public_data/snapshots/` — frozen public-data fallbacks
- `storage/paper_trading/` — paper trading ledger

Everything else can be regenerated from the pipeline.

## 9. Scope notes

This runbook covers only paper-trading and audit operations. Anything
related to routing live orders is explicitly out of scope and requires a
separate compliance and broker-integration review.
