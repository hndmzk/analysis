# Changelog 2026-04-12

Summary of all changes made during the 2026-04-10 to 2026-04-12 sessions,
for use as context handoff.

## 1. Pydantic immutable migration fix (CLI scripts)

**Problem**: Pydantic migration (all config classes → `BaseModel(frozen=True)`)
was applied to `src/` and `tests/`, but 7 CLI scripts in `scripts/` still
used direct attribute assignment (`settings.data.X = ...`) and the old
`settings.model` accessor (renamed to `settings.model_settings`).

**Root cause**: Tests don't invoke CLI scripts directly, so pytest/ruff/mypy
did not catch the breakage.

**Files modified** (all use `update_settings()` helper now):
- `scripts/generate_dummy_data.py` — 1 assignment
- `scripts/run_backtest.py` — 3 assignments + `.model` → `.model_settings`
- `scripts/validate_pbo_ranges.py` — 10 assignments, fully restructured
- `scripts/run_public_audit_suite.py` — `.model.cpcv` → `.model_settings.cpcv`
- `scripts/run_fundamentals_live_smoke.py` — 1 assignment
- `scripts/run_news_live_smoke.py` — 1 assignment
- `scripts/sweep_learned_weighting.py` — 15 assignments, `deepcopy` removed

## 2. validate_pbo_ranges.py enhancement (Codex)

Codex rewrote `scripts/validate_pbo_ranges.py` to:
- Output both `candidate_level_pbo` and `cluster_adjusted_pbo`
- Keep `pbo` / `pbo_label` as backward-compatible aliases
- Remove hardcoded `predictable_momentum PBO <= 0.8` assertion
- Add `_require_pbo()` ([0,1] range check) and `_require_label()` (label
  membership check)
- Add CLI arguments: `--dummy-days`, `--dummy-ticker-count`,
  `--initial-train-days`, `--eval-days`, `--step-days`, `--cpcv-group-count`,
  `--cpcv-test-groups`, `--cpcv-max-splits`

**Claude follow-up fix**: 2 mypy errors (line 100-101, `object` not indexable)
resolved by extracting `cost_adjusted_metrics` and `aggregate_metrics` into
`cast()` variables.

## 3. Live smoke tests executed

| Script | Source | Result |
|---|---|---|
| `run_news_live_smoke.py` | yahoo_finance_rss + google_news_rss | 248 headlines, stale_rate=0, fallback unused |
| `run_fundamentals_live_smoke.py` | sec_companyfacts | AAPL + MSFT, record_count=4, stale_rate=0 |
| offline_news_proxy fallback | direct invocation | rows=10, functional |

## 4. MCP stdio server smoke test

Verified `initialize`, `tools/list`, and `tools/call` (market_data_fetcher
with dummy source) via stdin pipe to `scripts/mcp_server.py`. All 3
JSON-RPC methods returned valid responses.

## 5. Public audit suite execution baselines

| Profile | Source mode | Runs | Duration | Result |
|---|---|---|---|---|
| fast | dummy | 3 | 357s | success |
| full_light | dummy | 12 | 1531s | success |
| full_light | live | 12 | 1366s | success |
| fast | live (wrapper) | 3 | 314s | success |
| full_light | live (Task Scheduler) | 12 | 1255s | success |

Key live metrics (full_light):
- information_ratio: mean=2.99, std=0.68
- pbo (candidate): mean=0.42 (moderate); cluster_adjusted: mean=0.0
- retraining_rate: 0.0; base_retraining_rate: 0.167 (2/12 suppressed)
- news_fallback_rate: 0.0; selection_stability: mean=0.94

## 6. Scheduler integration

### New files
- `scripts/scheduled_audit.ps1` — wrapper for Task Scheduler invocation
- `scripts/register_scheduled_task.ps1` — register/update/remove task

### Registered task
- Name: `MarketPredictionAgent-WeeklyAudit`
- Schedule: Every Saturday at 06:00
- Next run: 2026-04-18

### scheduled_audit.ps1 fixes (Codex follow-up)

Initial version used `$ErrorActionPreference="Stop"` + `2>&1` which
swallowed Python tracebacks. First Task Scheduler run failed with
`Last Result=1` because:

1. stderr was treated as a PowerShell error, losing traceback detail
2. `Write-Error` inside the non-zero exit path re-triggered `catch`,
   overwriting `latest_run.json` error info with `WriteErrorException`
3. Separately, `run_public_audit_suite.py` hit `numpy._ArrayMemoryError`
   during live runs due to accumulated pipeline objects

**Codex fixes**:
- `scheduled_audit.ps1`: Replaced `2>&1` capture with `Start-Process`
  + separate stdout/stderr temp files, merged into log as
  `=== STDOUT ===` / `=== STDERR ===` sections. Non-zero exits log
  tail-40 as error context. No more `Write-Error`.
- `run_public_audit_suite.py`: Added `import gc`. Pipeline is now
  instantiated per-run (not per-suite). All large references are
  `del`-ed after each run, followed by `gc.collect()`. Added explicit
  `cast()` for `aggregate_metrics`, `drift_monitor`,
  `drift_supplementary_analysis` to satisfy mypy.

After fix: Task Scheduler `full_light/live` → `Last Result=0`,
`duration_sec=1254.6`, `run_count=12`.

## 7. Operations runbook

Created `docs/operations_runbook.md` covering:
- Environment and credentials
- Public data source chain and fallback behavior
- Rate limits and retries
- Smoke test commands
- Scheduled audit jobs (with measured runtimes)
- Failure handling procedures
- Notification hooks (spec only, not implemented)
- Backup guidance

Updated with measured live runtimes and scheduler section.

## 8. pyproject.toml

- Added `extend-exclude = ["*.ps1"]` to `[tool.ruff]` to prevent ruff
  from parsing PowerShell files as Python.

## 9. Quality gates (as of end of session)

| Check | Result |
|---|---|
| ruff (src/tests/scripts) | All checks passed |
| mypy (65 source files) | Success: no issues found |
| Unit tests | 156 passed, 5 skipped |
| Integration tests | 5 passed |
| Task Scheduler full_light/live | 12/12 runs, exit_code=0 |

## 10. Remaining tasks (Priority 2+)

1. CI torch environment (ml-extra) — transformer/lstm coverage 8-10% → 70%+
2. pipeline.py unit tests — coverage 39% → 70%+
3. Notification layer — Slack/email for retraining triggers
4. `full` profile baseline — not yet measured

## 11. Spec mapping status

- 41 items, all `implemented`
- 0 items `partially implemented` or `not implemented`
