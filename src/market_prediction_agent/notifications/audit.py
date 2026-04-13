from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from email.message import EmailMessage
import json
import os
from pathlib import Path
import smtplib
import ssl
from typing import Any
import urllib.request

from market_prediction_agent.utils.paths import resolve_repo_path


SEVERITY_ORDER = {"ok": 0, "info": 1, "warning": 2, "critical": 3}


@dataclass(frozen=True)
class NotificationFinding:
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class NotificationThresholds:
    effective_retraining_rate: float = 0.0
    base_retraining_rate: float = 0.20
    news_fallback_rate: float = 0.0
    fundamental_fallback_rate: float = 1.0
    news_staleness_mean: float = 0.0
    cluster_pbo_warning_mean: float = 0.6
    cluster_pbo_critical_max: float = 0.85
    include_watch_only: bool = True
    watch_drift_rate: float = 0.20
    watch_regime_rate: float = 0.20


@dataclass(frozen=True)
class AuditNotificationReport:
    severity: str
    title: str
    text: str
    findings: list[NotificationFinding]
    status: dict[str, Any]
    suite_path: Path | None


def _as_bool(value: object, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if not normalized:
        return default
    return normalized in {"1", "true", "yes", "y", "on"}


def _as_float(value: object, default: float = 0.0) -> float:
    if not isinstance(value, int | float | str):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: object, default: int = 0) -> int:
    if not isinstance(value, int | float | str):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_dict(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _format_rate(value: float) -> str:
    return f"{value:.1%}"


def _format_float(value: float) -> str:
    return f"{value:.3f}"


def _distribution_value(summary: Mapping[str, Any], key: str, stat: str) -> float:
    return _as_float(_as_dict(summary.get(key)).get(stat))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def find_latest_suite_path(suite_root: Path) -> Path | None:
    if not suite_root.exists():
        return None
    candidates = [path for path in suite_root.rglob("*.json") if path.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def thresholds_from_env(env: Mapping[str, str] | None = None) -> NotificationThresholds:
    source = env or os.environ
    return NotificationThresholds(
        effective_retraining_rate=_as_float(source.get("AUDIT_NOTIFY_RETRAINING_RATE_THRESHOLD"), 0.0),
        base_retraining_rate=_as_float(source.get("AUDIT_NOTIFY_BASE_RETRAINING_RATE_THRESHOLD"), 0.20),
        news_fallback_rate=_as_float(source.get("AUDIT_NOTIFY_NEWS_FALLBACK_RATE_THRESHOLD"), 0.0),
        fundamental_fallback_rate=_as_float(source.get("AUDIT_NOTIFY_FUNDAMENTAL_FALLBACK_RATE_THRESHOLD"), 1.0),
        news_staleness_mean=_as_float(source.get("AUDIT_NOTIFY_NEWS_STALENESS_THRESHOLD"), 0.0),
        cluster_pbo_warning_mean=_as_float(source.get("AUDIT_NOTIFY_CLUSTER_PBO_WARNING_MEAN"), 0.6),
        cluster_pbo_critical_max=_as_float(source.get("AUDIT_NOTIFY_CLUSTER_PBO_CRITICAL_MAX"), 0.85),
        include_watch_only=_as_bool(source.get("AUDIT_NOTIFY_INCLUDE_WATCH_ONLY"), True),
        watch_drift_rate=_as_float(source.get("AUDIT_NOTIFY_WATCH_DRIFT_RATE_THRESHOLD"), 0.20),
        watch_regime_rate=_as_float(source.get("AUDIT_NOTIFY_WATCH_REGIME_RATE_THRESHOLD"), 0.20),
    )


def _append_rate_finding(
    findings: list[NotificationFinding],
    *,
    severity: str,
    code: str,
    label: str,
    value: float,
    threshold: float,
) -> None:
    if value > threshold:
        findings.append(
            NotificationFinding(
                severity=severity,
                code=code,
                message=f"{label} is {_format_rate(value)} (threshold {_format_rate(threshold)}).",
            )
        )


def _max_severity(findings: Sequence[NotificationFinding]) -> str:
    if not findings:
        return "ok"
    return max(findings, key=lambda finding: SEVERITY_ORDER[finding.severity]).severity


def evaluate_audit_status(
    *,
    status: dict[str, Any],
    suite_payload: dict[str, Any] | None = None,
    suite_path: Path | None = None,
    thresholds: NotificationThresholds | None = None,
) -> AuditNotificationReport:
    resolved_thresholds = thresholds or NotificationThresholds()
    findings: list[NotificationFinding] = []
    exit_code = _as_int(status.get("exit_code"))
    success = bool(status.get("success", exit_code == 0))

    if not success or exit_code != 0:
        error_detail = str(status.get("error") or "").strip()
        suffix = f" Error: {error_detail[:500]}" if error_detail else ""
        findings.append(
            NotificationFinding(
                severity="critical",
                code="audit_failed",
                message=f"Scheduled audit failed with exit_code={exit_code}.{suffix}",
            )
        )

    if suite_payload is None:
        if success:
            findings.append(
                NotificationFinding(
                    severity="warning",
                    code="suite_missing",
                    message="Audit status succeeded, but no suite artifact was found for metric inspection.",
                )
            )
    else:
        run_count = _as_int(suite_payload.get("run_count"))
        if run_count <= 0:
            findings.append(
                NotificationFinding(
                    severity="critical",
                    code="empty_suite",
                    message=f"Suite artifact has invalid run_count={run_count}.",
                )
            )

        summary = _as_dict(suite_payload.get("distribution_summary"))
        retraining_rate = _as_float(summary.get("retraining_rate"))
        base_retraining_rate = _as_float(summary.get("base_retraining_rate"))
        news_fallback_rate = _as_float(summary.get("news_fallback_rate"))
        fundamental_fallback_rate = _as_float(summary.get("fundamental_fallback_rate"))
        news_staleness_mean = _distribution_value(summary, "news_feature_staleness", "mean")
        cluster_pbo_mean = _distribution_value(summary, "cluster_adjusted_pbo", "mean")
        cluster_pbo_max = _distribution_value(summary, "cluster_adjusted_pbo", "max")

        _append_rate_finding(
            findings,
            severity="critical",
            code="effective_retraining",
            label="Effective retraining rate",
            value=retraining_rate,
            threshold=resolved_thresholds.effective_retraining_rate,
        )
        _append_rate_finding(
            findings,
            severity="warning",
            code="base_retraining",
            label="Base retraining trigger rate",
            value=base_retraining_rate,
            threshold=resolved_thresholds.base_retraining_rate,
        )
        _append_rate_finding(
            findings,
            severity="warning",
            code="news_fallback",
            label="News fallback rate",
            value=news_fallback_rate,
            threshold=resolved_thresholds.news_fallback_rate,
        )
        _append_rate_finding(
            findings,
            severity="warning",
            code="fundamental_fallback",
            label="Fundamental fallback rate",
            value=fundamental_fallback_rate,
            threshold=resolved_thresholds.fundamental_fallback_rate,
        )
        _append_rate_finding(
            findings,
            severity="warning",
            code="news_staleness",
            label="News feature staleness",
            value=news_staleness_mean,
            threshold=resolved_thresholds.news_staleness_mean,
        )

        if cluster_pbo_max >= resolved_thresholds.cluster_pbo_critical_max:
            findings.append(
                NotificationFinding(
                    severity="critical",
                    code="cluster_pbo_critical",
                    message=(
                        "Cluster-adjusted PBO max is "
                        f"{_format_float(cluster_pbo_max)} "
                        f"(critical threshold {_format_float(resolved_thresholds.cluster_pbo_critical_max)})."
                    ),
                )
            )
        elif cluster_pbo_mean > resolved_thresholds.cluster_pbo_warning_mean:
            findings.append(
                NotificationFinding(
                    severity="warning",
                    code="cluster_pbo_warning",
                    message=(
                        "Cluster-adjusted PBO mean is "
                        f"{_format_float(cluster_pbo_mean)} "
                        f"(warning threshold {_format_float(resolved_thresholds.cluster_pbo_warning_mean)})."
                    ),
                )
            )

        if resolved_thresholds.include_watch_only:
            drift_raw_rate = _as_float(_as_dict(summary.get("drift_dominated_analysis")).get("raw_rate"))
            regime_raw_rate = _as_float(_as_dict(summary.get("regime_dominated_analysis")).get("raw_rate"))
            _append_rate_finding(
                findings,
                severity="warning",
                code="watch_drift",
                label="Watch-only drift-dominated rate",
                value=drift_raw_rate,
                threshold=resolved_thresholds.watch_drift_rate,
            )
            _append_rate_finding(
                findings,
                severity="warning",
                code="watch_regime",
                label="Watch-only regime-dominated rate",
                value=regime_raw_rate,
                threshold=resolved_thresholds.watch_regime_rate,
            )

    severity = _max_severity(findings)
    title = f"{severity.upper()}: MarketPredictionAgent scheduled audit"
    text = render_notification_text(
        title=title,
        severity=severity,
        status=status,
        suite_payload=suite_payload,
        suite_path=suite_path,
        findings=findings,
    )
    return AuditNotificationReport(
        severity=severity,
        title=title,
        text=text,
        findings=findings,
        status=status,
        suite_path=suite_path,
    )


def render_notification_text(
    *,
    title: str,
    severity: str,
    status: dict[str, Any],
    suite_payload: dict[str, Any] | None,
    suite_path: Path | None,
    findings: Sequence[NotificationFinding],
) -> str:
    lines = [
        title,
        f"severity={severity}",
        (
            "status="
            f"profile={status.get('profile', 'unknown')} "
            f"source_mode={status.get('source_mode', 'unknown')} "
            f"success={status.get('success')} "
            f"exit_code={status.get('exit_code')} "
            f"duration_sec={status.get('duration_sec', 'unknown')} "
            f"timestamp={status.get('timestamp', 'unknown')}"
        ),
    ]
    log_file = status.get("log_file")
    if log_file:
        lines.append(f"log_file={log_file}")
    if suite_payload is not None:
        summary = _as_dict(suite_payload.get("distribution_summary"))
        lines.append(
            "suite="
            f"suite_id={suite_payload.get('suite_id', 'unknown')} "
            f"profile={suite_payload.get('profile_name', 'unknown')} "
            f"run_count={suite_payload.get('run_count', 'unknown')} "
            f"generated_at={suite_payload.get('generated_at', 'unknown')}"
        )
        lines.append(
            "metrics="
            f"retraining_rate={_format_rate(_as_float(summary.get('retraining_rate')))} "
            f"base_retraining_rate={_format_rate(_as_float(summary.get('base_retraining_rate')))} "
            f"news_fallback_rate={_format_rate(_as_float(summary.get('news_fallback_rate')))} "
            f"fundamental_fallback_rate={_format_rate(_as_float(summary.get('fundamental_fallback_rate')))} "
            f"cluster_adjusted_pbo_mean={_format_float(_distribution_value(summary, 'cluster_adjusted_pbo', 'mean'))} "
            f"cluster_adjusted_pbo_max={_format_float(_distribution_value(summary, 'cluster_adjusted_pbo', 'max'))}"
        )
    if suite_path is not None:
        lines.append(f"suite_path={suite_path}")

    if findings:
        lines.append("findings:")
        lines.extend(f"- [{finding.severity}] {finding.code}: {finding.message}" for finding in findings)
    else:
        lines.append("findings: none")

    return "\n".join(lines)


def should_send_notification(report: AuditNotificationReport, *, min_severity: str, notify_on_ok: bool) -> bool:
    if report.severity == "ok":
        return notify_on_ok or SEVERITY_ORDER[min_severity] <= SEVERITY_ORDER["ok"]
    return SEVERITY_ORDER[report.severity] >= SEVERITY_ORDER[min_severity]


def send_slack_notification(text: str, webhook_url: str) -> None:
    payload = json.dumps({"text": text}).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=20):
        return


def _email_recipients(raw: str) -> list[str]:
    return [item.strip() for item in raw.replace(";", ",").split(",") if item.strip()]


def send_email_notification(*, subject: str, text: str, env: Mapping[str, str] | None = None) -> bool:
    source = env or os.environ
    host = source.get("AUDIT_EMAIL_SMTP_HOST")
    recipients_raw = source.get("AUDIT_EMAIL_TO")
    if not host or not recipients_raw:
        return False

    port = _as_int(source.get("AUDIT_EMAIL_SMTP_PORT"), 587)
    username = source.get("AUDIT_EMAIL_SMTP_USER")
    password = source.get("AUDIT_EMAIL_SMTP_PASSWORD") or ""
    sender = source.get("AUDIT_EMAIL_FROM") or username or "market-prediction-agent@localhost"
    recipients = _email_recipients(recipients_raw)
    if not recipients:
        return False

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = ", ".join(recipients)
    message.set_content(text)

    with smtplib.SMTP(host, port, timeout=20) as smtp:
        if _as_bool(source.get("AUDIT_EMAIL_USE_TLS"), True):
            smtp.starttls(context=ssl.create_default_context())
        if username:
            smtp.login(username, password)
        smtp.send_message(message)
    return True


def send_configured_notifications(report: AuditNotificationReport, env: Mapping[str, str] | None = None) -> int:
    source = env or os.environ
    sent_count = 0
    slack_webhook_url = source.get("AUDIT_SLACK_WEBHOOK_URL")
    if slack_webhook_url:
        send_slack_notification(report.text, slack_webhook_url)
        sent_count += 1
    if send_email_notification(subject=report.title, text=report.text, env=source):
        sent_count += 1
    return sent_count


def build_report_from_files(
    *,
    status_file: Path,
    suite_root: Path,
    suite_path: Path | None = None,
    thresholds: NotificationThresholds | None = None,
) -> AuditNotificationReport:
    status = load_json(status_file)
    resolved_suite_path = suite_path
    status_suite_path = status.get("suite_path")
    if resolved_suite_path is None and isinstance(status_suite_path, str) and status_suite_path:
        resolved_suite_path = resolve_repo_path(status_suite_path)
    if resolved_suite_path is None and bool(status.get("success", False)):
        resolved_suite_path = find_latest_suite_path(suite_root)

    suite_payload = load_json(resolved_suite_path) if resolved_suite_path is not None else None
    return evaluate_audit_status(
        status=status,
        suite_payload=suite_payload,
        suite_path=resolved_suite_path,
        thresholds=thresholds,
    )
