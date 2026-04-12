from __future__ import annotations

import argparse
import json

from market_prediction_agent.config import load_settings, resolve_storage_path
from market_prediction_agent.schemas.validator import validate_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate persisted JSON outputs against schemas.")
    parser.add_argument("--config", default=None, help="Optional config path.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate every persisted artifact. Default validates only the latest JSON per schema folder.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    root = resolve_storage_path(settings) / "outputs"
    for schema_name, folder in [
        ("forecast_output", "forecasts"),
        ("evidence_bundle", "evidence"),
        ("risk_review", "risk_reviews"),
        ("report_payload", "reports"),
        ("backtest_result", "backtests"),
        ("paper_trading_batch", "paper_trading"),
        ("weekly_review", "weekly_reviews"),
        ("retraining_event", "retraining_events"),
        ("monitor_audit", "monitor_audits"),
        ("monitor_audit_suite", "monitor_audit_suites"),
        ("audit_report", "audit_reports"),
    ]:
        if not (root / folder).exists():
            continue
        paths = sorted((root / folder).rglob("*.json"), key=lambda path: path.stat().st_mtime)
        if not args.all and paths:
            paths = [paths[-1]]
        for path in paths:
            validate_payload(schema_name, json.loads(path.read_text(encoding="utf-8")))
            print(f"validated: {path}")


if __name__ == "__main__":
    main()
