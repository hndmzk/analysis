from __future__ import annotations

import argparse
import json

from market_prediction_agent.config import load_settings
from market_prediction_agent.reporting.audit_reports import (
    build_audit_report,
    persist_audit_report,
    render_audit_report_markdown,
    resolve_artifact_bundle,
)
from market_prediction_agent.schemas.validator import validate_payload


LATEST_FLAGS = {
    "latest_monitor_audit": "monitor_audit",
    "latest_suite": "monitor_audit_suite",
    "latest_backtest": "backtest_result",
    "latest_weekly_review": "weekly_review",
    "latest_paper_trading_batch": "paper_trading_batch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render human-readable audit reports from persisted JSON artifacts.")
    parser.add_argument("--config", default=None, help="Optional config path.")
    parser.add_argument("--input", default=None, help="Explicit input artifact path.")
    parser.add_argument("--latest-monitor-audit", action="store_true", help="Use the latest monitor_audit artifact.")
    parser.add_argument("--latest-suite", action="store_true", help="Use the latest monitor_audit_suite artifact.")
    parser.add_argument("--latest-backtest", action="store_true", help="Use the latest backtest_result artifact.")
    parser.add_argument("--latest-weekly-review", action="store_true", help="Use the latest weekly_review artifact.")
    parser.add_argument(
        "--latest-paper-trading-batch",
        action="store_true",
        help="Use the latest paper_trading_batch artifact.",
    )
    parser.add_argument("--monitor-audit", default=None, help="Optional related monitor_audit path.")
    parser.add_argument("--suite", default=None, help="Optional related monitor_audit_suite path.")
    parser.add_argument("--backtest", default=None, help="Optional related backtest_result path.")
    parser.add_argument("--weekly-review", default=None, help="Optional related weekly_review path.")
    parser.add_argument("--paper-trading-batch", default=None, help="Optional related paper_trading_batch path.")
    parser.add_argument("--format", choices=["json", "markdown", "both"], default="both", help="Output format.")
    parser.add_argument("--output", default=None, help="Shared output path when --format is json or markdown.")
    parser.add_argument("--json-output", default=None, help="Explicit JSON output path.")
    parser.add_argument("--markdown-output", default=None, help="Explicit Markdown output path.")
    return parser.parse_args()


def _resolve_primary_latest_flag(args: argparse.Namespace) -> str | None:
    selected: list[str] = []
    for arg_name, artifact_type in LATEST_FLAGS.items():
        if getattr(args, arg_name):
            selected.append(artifact_type)
    if len(selected) > 1:
        raise ValueError("Specify only one latest-* flag at a time.")
    return selected[0] if selected else None


def _related_paths(args: argparse.Namespace) -> dict[str, str]:
    related: dict[str, str] = {}
    if args.monitor_audit:
        related["monitor_audit"] = args.monitor_audit
    if args.suite:
        related["monitor_audit_suite"] = args.suite
    if args.backtest:
        related["backtest_result"] = args.backtest
    if args.weekly_review:
        related["weekly_review"] = args.weekly_review
    if args.paper_trading_batch:
        related["paper_trading_batch"] = args.paper_trading_batch
    return related


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    latest_artifact_type = _resolve_primary_latest_flag(args)
    bundle = resolve_artifact_bundle(
        settings,
        primary_path=args.input,
        latest_artifact_type=latest_artifact_type,
        explicit_paths=_related_paths(args),
    )
    report = build_audit_report(bundle)
    validate_payload("audit_report", report)
    markdown = render_audit_report_markdown(report)

    json_output = args.json_output
    markdown_output = args.markdown_output
    if args.output:
        if args.format == "json":
            json_output = args.output
        elif args.format == "markdown":
            markdown_output = args.output
        else:
            raise ValueError("--output is only valid when --format is json or markdown. Use --json-output/--markdown-output for both.")
    json_path, markdown_path = persist_audit_report(
        settings,
        report,
        markdown,
        json_output=json_output if args.format in {"json", "both"} else None,
        markdown_output=markdown_output if args.format in {"markdown", "both"} else None,
        write_json=args.format in {"json", "both"},
        write_markdown=args.format in {"markdown", "both"},
    )
    result = {
        "report_type": report["report_type"],
        "primary_artifact_type": bundle.primary_type,
        "json_output": str(json_path) if args.format in {"json", "both"} else None,
        "markdown_output": str(markdown_path) if args.format in {"markdown", "both"} else None,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
