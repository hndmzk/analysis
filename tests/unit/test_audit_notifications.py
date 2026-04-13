from __future__ import annotations

from pathlib import Path
import json
import shutil
from uuid import uuid4

from market_prediction_agent.notifications.audit import (
    NotificationThresholds,
    build_report_from_files,
    evaluate_audit_status,
    find_latest_suite_path,
    should_send_notification,
    thresholds_from_env,
)


def _suite_payload(**summary_overrides: object) -> dict[str, object]:
    summary: dict[str, object] = {
        "retraining_rate": 0.0,
        "base_retraining_rate": 0.0,
        "news_fallback_rate": 0.0,
        "fundamental_fallback_rate": 0.0,
        "cluster_adjusted_pbo": {"mean": 0.0, "max": 0.0},
        "news_feature_staleness": {"mean": 0.0},
        "drift_dominated_analysis": {"raw_rate": 0.0},
        "regime_dominated_analysis": {"raw_rate": 0.0},
    }
    summary.update(summary_overrides)
    return {
        "suite_id": "11111111-1111-4111-8111-111111111111",
        "generated_at": "2026-04-12T00:00:00+00:00",
        "profile_name": "full_light",
        "run_count": 12,
        "distribution_summary": summary,
    }


def test_evaluate_audit_status_flags_failed_run() -> None:
    report = evaluate_audit_status(
        status={"success": False, "exit_code": 1, "profile": "full_light", "source_mode": "live"},
        suite_payload=None,
    )

    assert report.severity == "critical"
    assert report.findings[0].code == "audit_failed"
    assert should_send_notification(report, min_severity="warning", notify_on_ok=False)


def test_evaluate_audit_status_flags_retraining_and_watch_signals() -> None:
    report = evaluate_audit_status(
        status={"success": True, "exit_code": 0, "profile": "full_light", "source_mode": "live"},
        suite_payload=_suite_payload(
            retraining_rate=0.25,
            base_retraining_rate=0.5,
            drift_dominated_analysis={"raw_rate": 0.25},
            regime_dominated_analysis={"raw_rate": 0.25},
        ),
        thresholds=NotificationThresholds(),
    )

    assert report.severity == "critical"
    assert {finding.code for finding in report.findings} >= {
        "effective_retraining",
        "base_retraining",
        "watch_drift",
        "watch_regime",
    }


def test_evaluate_audit_status_stays_ok_for_clean_suite() -> None:
    report = evaluate_audit_status(
        status={"success": True, "exit_code": 0, "profile": "full_light", "source_mode": "live"},
        suite_payload=_suite_payload(),
        thresholds=NotificationThresholds(),
    )

    assert report.severity == "ok"
    assert report.findings == []
    assert not should_send_notification(report, min_severity="warning", notify_on_ok=False)


def test_evaluate_audit_status_tolerates_suppressed_watch_baseline() -> None:
    report = evaluate_audit_status(
        status={"success": True, "exit_code": 0, "profile": "full_light", "source_mode": "live"},
        suite_payload=_suite_payload(
            base_retraining_rate=1 / 6,
            fundamental_fallback_rate=1.0,
            drift_dominated_analysis={"raw_rate": 1 / 6},
            regime_dominated_analysis={"raw_rate": 1 / 6},
        ),
        thresholds=NotificationThresholds(),
    )

    assert report.severity == "ok"
    assert report.findings == []


def test_evaluate_audit_status_can_flag_fundamental_fallback_rate() -> None:
    report = evaluate_audit_status(
        status={"success": True, "exit_code": 0, "profile": "full_light", "source_mode": "live"},
        suite_payload=_suite_payload(fundamental_fallback_rate=1.0),
        thresholds=NotificationThresholds(fundamental_fallback_rate=0.5),
    )

    assert report.severity == "warning"
    assert {finding.code for finding in report.findings} == {"fundamental_fallback"}


def test_thresholds_from_env_accepts_watch_only_overrides() -> None:
    thresholds = thresholds_from_env(
        {
            "AUDIT_NOTIFY_BASE_RETRAINING_RATE_THRESHOLD": "0.30",
            "AUDIT_NOTIFY_FUNDAMENTAL_FALLBACK_RATE_THRESHOLD": "0.70",
            "AUDIT_NOTIFY_WATCH_DRIFT_RATE_THRESHOLD": "0.40",
            "AUDIT_NOTIFY_WATCH_REGIME_RATE_THRESHOLD": "0.50",
        }
    )

    assert thresholds.base_retraining_rate == 0.30
    assert thresholds.fundamental_fallback_rate == 0.70
    assert thresholds.watch_drift_rate == 0.40
    assert thresholds.watch_regime_rate == 0.50


def test_build_report_from_files_uses_latest_suite_artifact() -> None:
    root = Path(".test-artifacts") / uuid4().hex
    try:
        status_dir = root / "logs"
        suite_dir = root / "suites" / "2026-04-12"
        status_dir.mkdir(parents=True)
        suite_dir.mkdir(parents=True)
        status_file = status_dir / "latest_run.json"
        suite_file = suite_dir / "suite.json"
        status_file.write_text(json.dumps({"success": True, "exit_code": 0}), encoding="utf-8")
        suite_file.write_text(json.dumps(_suite_payload(news_fallback_rate=1.0)), encoding="utf-8")

        assert find_latest_suite_path(root / "suites") == suite_file
        report = build_report_from_files(status_file=status_file, suite_root=root / "suites")

        assert report.severity == "warning"
        assert report.suite_path == suite_file
        assert {finding.code for finding in report.findings} == {"news_fallback"}
    finally:
        shutil.rmtree(root, ignore_errors=True)
