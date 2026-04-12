from __future__ import annotations

import argparse
import os

from market_prediction_agent.notifications.audit import (
    SEVERITY_ORDER,
    build_report_from_files,
    send_configured_notifications,
    should_send_notification,
    thresholds_from_env,
)
from market_prediction_agent.utils.paths import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Notify on scheduled audit status and suite risk signals.")
    parser.add_argument(
        "--status-file",
        default="storage/logs/scheduled_audits/latest_run.json",
        help="Path to scheduled_audit.ps1 latest_run.json.",
    )
    parser.add_argument(
        "--suite-root",
        default="storage/outputs/monitor_audit_suites/public_real_market",
        help="Root directory containing monitor audit suite JSON artifacts.",
    )
    parser.add_argument("--suite-path", default=None, help="Optional explicit suite JSON path.")
    parser.add_argument(
        "--min-severity",
        choices=sorted(SEVERITY_ORDER),
        default=os.getenv("AUDIT_NOTIFY_MIN_SEVERITY", "warning"),
        help="Minimum severity required before sending a notification.",
    )
    parser.add_argument(
        "--notify-on-ok",
        action="store_true",
        default=os.getenv("AUDIT_NOTIFY_ON_OK", "").lower() in {"1", "true", "yes", "on"},
        help="Send a notification even when no alert findings are present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=os.getenv("AUDIT_NOTIFY_DRY_RUN", "").lower() in {"1", "true", "yes", "on"},
        help="Print the rendered notification instead of sending it.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    status_file = resolve_repo_path(args.status_file)
    suite_root = resolve_repo_path(args.suite_root)
    suite_path = resolve_repo_path(args.suite_path) if args.suite_path else None
    report = build_report_from_files(
        status_file=status_file,
        suite_root=suite_root,
        suite_path=suite_path,
        thresholds=thresholds_from_env(),
    )

    if args.dry_run:
        print(report.text)
        return 0

    if not should_send_notification(report, min_severity=args.min_severity, notify_on_ok=args.notify_on_ok):
        print(f"No notification needed. severity={report.severity}")
        return 0

    sent_count = send_configured_notifications(report)
    if sent_count == 0:
        print("No notification destination configured; skipping send.")
        print(report.text)
    else:
        print(f"Sent {sent_count} notification(s). severity={report.severity}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
