from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, cast
from uuid import uuid4

from market_prediction_agent.config import Settings, resolve_storage_path
from market_prediction_agent.utils.paths import resolve_repo_path


SECTION_ORDER = [
    "Executive Summary",
    "Run Context",
    "Model / Portfolio Performance",
    "Execution Diagnostics",
    "Drift / Calibration / Regime Review",
    "Retraining Decision",
    "Watch-Only Findings",
    "Data Sources and Transport",
    "Top Risks",
    "Recommended Next Actions",
    "Artifact References",
]

ARTIFACT_FOLDERS = {
    "monitor_audit": "monitor_audits",
    "monitor_audit_suite": "monitor_audit_suites",
    "weekly_review": "weekly_reviews",
    "backtest_result": "backtests",
    "paper_trading_batch": "paper_trading",
}


@dataclass(slots=True)
class ArtifactBundle:
    primary_type: str
    primary_payload: dict[str, Any]
    primary_path: Path | None
    monitor_audit: dict[str, Any] | None = None
    monitor_audit_path: Path | None = None
    monitor_audit_suite: dict[str, Any] | None = None
    monitor_audit_suite_path: Path | None = None
    weekly_review: dict[str, Any] | None = None
    weekly_review_path: Path | None = None
    backtest_result: dict[str, Any] | None = None
    backtest_result_path: Path | None = None
    paper_trading_batch: dict[str, Any] | None = None
    paper_trading_batch_path: Path | None = None
    suite_run_audits: dict[str, dict[str, Any]] = field(default_factory=dict)
    suite_run_audit_paths: dict[str, Path] = field(default_factory=dict)
    suite_run_backtests: dict[str, dict[str, Any]] = field(default_factory=dict)
    suite_run_backtest_paths: dict[str, Path] = field(default_factory=dict)


def load_json_payload(path: str | Path) -> dict[str, Any]:
    resolved = resolve_repo_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def detect_artifact_type(payload: dict[str, Any]) -> str:
    if "suite_id" in payload and "distribution_summary" in payload:
        return "monitor_audit_suite"
    if "audit_id" in payload and "backtest" in payload and "data_sources" in payload:
        return "monitor_audit"
    if "review_id" in payload and "week_id" in payload and "execution_diagnostics" in payload:
        return "weekly_review"
    if "batch_id" in payload and "trades" in payload and "execution_diagnostics" in payload:
        return "paper_trading_batch"
    if "backtest_id" in payload and "aggregate_metrics" in payload and "cost_adjusted_metrics" in payload:
        return "backtest_result"
    raise ValueError("Unsupported artifact payload for audit reporting.")


def latest_artifact_path(settings: Settings, artifact_type: str) -> Path | None:
    folder = ARTIFACT_FOLDERS[artifact_type]
    root = resolve_storage_path(settings) / "outputs" / folder
    if not root.exists():
        return None
    paths = sorted(root.rglob("*.json"), key=lambda path: path.stat().st_mtime)
    return paths[-1] if paths else None


def load_latest_artifact(settings: Settings, artifact_type: str) -> tuple[dict[str, Any], Path] | None:
    path = latest_artifact_path(settings, artifact_type)
    if path is None:
        return None
    return load_json_payload(path), path


def _load_all_artifacts(settings: Settings, artifact_type: str) -> list[tuple[Path, dict[str, Any]]]:
    folder = ARTIFACT_FOLDERS[artifact_type]
    root = resolve_storage_path(settings) / "outputs" / folder
    if not root.exists():
        return []
    records: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.rglob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        records.append((path, load_json_payload(path)))
    return records


def _find_matching_artifact(
    settings: Settings,
    artifact_type: str,
    predicate: Callable[[dict[str, Any]], bool],
) -> tuple[dict[str, Any], Path] | None:
    for path, payload in _load_all_artifacts(settings, artifact_type):
        if predicate(payload):
            return payload, path
    return None


def _set_bundle_artifact(bundle: ArtifactBundle, artifact_type: str, payload: dict[str, Any], path: Path | None) -> None:
    if artifact_type == "monitor_audit":
        bundle.monitor_audit = payload
        bundle.monitor_audit_path = path
    elif artifact_type == "monitor_audit_suite":
        bundle.monitor_audit_suite = payload
        bundle.monitor_audit_suite_path = path
    elif artifact_type == "weekly_review":
        bundle.weekly_review = payload
        bundle.weekly_review_path = path
    elif artifact_type == "backtest_result":
        bundle.backtest_result = payload
        bundle.backtest_result_path = path
    elif artifact_type == "paper_trading_batch":
        bundle.paper_trading_batch = payload
        bundle.paper_trading_batch_path = path


def resolve_artifact_bundle(
    settings: Settings,
    *,
    primary_path: str | Path | None = None,
    latest_artifact_type: str | None = None,
    explicit_paths: dict[str, str | Path] | None = None,
) -> ArtifactBundle:
    if primary_path is None and latest_artifact_type is None:
        raise ValueError("Either primary_path or latest_artifact_type must be provided.")
    if primary_path is not None:
        resolved_primary_path = resolve_repo_path(primary_path)
        primary_payload = load_json_payload(resolved_primary_path)
    else:
        latest = load_latest_artifact(settings, latest_artifact_type or "monitor_audit")
        if latest is None:
            raise FileNotFoundError(f"No persisted artifact found for {latest_artifact_type}.")
        primary_payload, resolved_primary_path = latest
    primary_type = detect_artifact_type(primary_payload)
    bundle = ArtifactBundle(primary_type=primary_type, primary_payload=primary_payload, primary_path=resolved_primary_path)
    _set_bundle_artifact(bundle, primary_type, primary_payload, resolved_primary_path)

    for artifact_type, artifact_path in (explicit_paths or {}).items():
        payload = load_json_payload(artifact_path)
        _set_bundle_artifact(bundle, artifact_type, payload, resolve_repo_path(artifact_path))

    if bundle.monitor_audit is None and primary_type == "backtest_result":
        backtest_id = str(primary_payload.get("backtest_id", ""))
        matched = _find_matching_artifact(
            settings,
            "monitor_audit",
            lambda payload: str(payload.get("backtest", {}).get("backtest_id", "")) == backtest_id,
        )
        if matched is not None:
            _set_bundle_artifact(bundle, "monitor_audit", matched[0], matched[1])
    if bundle.backtest_result is None and bundle.monitor_audit is not None:
        backtest_id = str(bundle.monitor_audit.get("backtest", {}).get("backtest_id", ""))
        if backtest_id:
            matched = _find_matching_artifact(
                settings,
                "backtest_result",
                lambda payload: str(payload.get("backtest_id", "")) == backtest_id,
            )
            if matched is not None:
                _set_bundle_artifact(bundle, "backtest_result", matched[0], matched[1])
    if bundle.paper_trading_batch is None and bundle.monitor_audit is not None:
        batch_id = str(bundle.monitor_audit.get("paper_trading_summary", {}).get("batch_id", ""))
        if batch_id:
            matched = _find_matching_artifact(
                settings,
                "paper_trading_batch",
                lambda payload: str(payload.get("batch_id", "")) == batch_id,
            )
            if matched is not None:
                _set_bundle_artifact(bundle, "paper_trading_batch", matched[0], matched[1])
    if bundle.weekly_review is None and bundle.paper_trading_batch is not None:
        week_id = str(bundle.paper_trading_batch.get("week_id", ""))
        if week_id:
            matched = _find_matching_artifact(
                settings,
                "weekly_review",
                lambda payload: str(payload.get("week_id", "")) == week_id,
            )
            if matched is not None:
                _set_bundle_artifact(bundle, "weekly_review", matched[0], matched[1])
    if bundle.weekly_review is None:
        latest_weekly = load_latest_artifact(settings, "weekly_review")
        if latest_weekly is not None:
            _set_bundle_artifact(bundle, "weekly_review", latest_weekly[0], latest_weekly[1])
    if bundle.paper_trading_batch is None:
        latest_paper = load_latest_artifact(settings, "paper_trading_batch")
        if latest_paper is not None:
            _set_bundle_artifact(bundle, "paper_trading_batch", latest_paper[0], latest_paper[1])
    if bundle.backtest_result is None:
        latest_backtest = load_latest_artifact(settings, "backtest_result")
        if latest_backtest is not None:
            _set_bundle_artifact(bundle, "backtest_result", latest_backtest[0], latest_backtest[1])
    if bundle.monitor_audit_suite is not None:
        audit_index = {str(payload.get("audit_id", "")): (payload, path) for path, payload in _load_all_artifacts(settings, "monitor_audit")}
        backtest_index = {
            str(payload.get("backtest_id", "")): (payload, path)
            for path, payload in _load_all_artifacts(settings, "backtest_result")
        }
        for run in bundle.monitor_audit_suite.get("runs", []):
            audit_id = str(run.get("audit_id", ""))
            backtest_id = str(run.get("backtest_id", ""))
            if audit_id in audit_index:
                payload, path = audit_index[audit_id]
                bundle.suite_run_audits[audit_id] = payload
                bundle.suite_run_audit_paths[audit_id] = path
            if backtest_id in backtest_index:
                payload, path = backtest_index[backtest_id]
                bundle.suite_run_backtests[backtest_id] = payload
                bundle.suite_run_backtest_paths[backtest_id] = path
    return bundle


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_number(value: Any, decimals: int = 4) -> str:
    numeric = _safe_float(value)
    if numeric is None:
        return "n/a"
    return f"{numeric:.{decimals}f}"


def _format_distribution(summary: dict[str, Any] | None) -> str:
    if not summary:
        return "n/a"
    count = _safe_int(summary.get("count"))
    mean_value = _safe_float(summary.get("mean"))
    median_value = _safe_float(summary.get("median"))
    min_value = _safe_float(summary.get("min"))
    max_value = _safe_float(summary.get("max"))
    if mean_value is None:
        return "n/a"
    parts = [f"mean={mean_value:.4f}", f"median={median_value:.4f}" if median_value is not None else None]
    if min_value is not None and max_value is not None:
        parts.append(f"range={min_value:.4f} to {max_value:.4f}")
    if count is not None:
        parts.append(f"count={count}")
    return ", ".join(part for part in parts if part)


def _top_items(counts: dict[str, Any] | None, limit: int = 3) -> list[str]:
    if not counts:
        return []
    normalized: list[tuple[str, int]] = []
    for key, value in counts.items():
        normalized.append((str(key), int(value)))
    normalized.sort(key=lambda item: (-item[1], item[0]))
    return [f"{key}={value}" for key, value in normalized[:limit]]


def _format_count_map(counts: dict[str, Any] | None) -> str:
    items = _top_items(counts, limit=10)
    return ", ".join(items) if items else "none"


def _news_weighting_summary(utility_comparison: dict[str, Any] | None) -> dict[str, Any]:
    if not utility_comparison:
        return {}
    def sort_key(item: dict[str, Any]) -> float:
        raw = item.get("mean_weight", 0.0)
        if isinstance(raw, dict):
            return float(raw.get("mean", 0.0) or 0.0)
        return float(raw or 0.0)

    weighting_modes = {
        str(item.get("mode", "")): dict(item)
        for item in _coerce_dict_list(utility_comparison.get("weighting_mode_comparison"))
        if str(item.get("mode", ""))
    }
    learned = dict(utility_comparison.get("learned_weighting", {}))
    source_weights = sorted(
        _coerce_dict_list(learned.get("source_weights")),
        key=sort_key,
        reverse=True,
    )
    session_weights = sorted(
        _coerce_dict_list(learned.get("session_weights")),
        key=sort_key,
        reverse=True,
    )
    return {
        "selected_mode": utility_comparison.get("selected_weighting_mode"),
        "selected_variant": utility_comparison.get("selected_weighting_variant"),
        "unweighted": weighting_modes.get("unweighted", {}),
        "fixed": weighting_modes.get("fixed", {}),
        "learned": weighting_modes.get("learned", {}),
        "multi_source_improvement": dict(utility_comparison.get("multi_source_weighting_improvement", {})),
        "learned_target": learned.get("target"),
        "learned_fallback_rate": learned.get("fallback_rate"),
        "learned_fit_day_count": learned.get("fit_day_count"),
        "top_source_weights": source_weights[:3],
        "top_session_weights": session_weights[:3],
    }


def _coerce_str_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    normalized: list[str] = []
    for item in values:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            status = str(item.get("status", "")).strip()
            detail = str(item.get("detail", "")).strip()
            parts = [part for part in [name, status] if part]
            prefix = " ".join(parts)
            if prefix and detail:
                normalized.append(f"{prefix}: {detail}")
            elif detail:
                normalized.append(detail)
            elif prefix:
                normalized.append(prefix)
            continue
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _parse_prefixed_note(notes: list[str], prefix: str) -> str | None:
    for note in notes:
        if note.startswith(prefix):
            return note[len(prefix) :]
    return None


def _join_tickers(values: list[str] | str | None) -> str:
    if values is None:
        return "n/a"
    if isinstance(values, str):
        return values
    if not values:
        return "n/a"
    return ",".join(values)


def _summarize_watch_only(
    base_names: list[str],
    effective_names: list[str],
    suppressed_names: list[str],
    observations: list[str],
) -> list[str]:
    findings: list[str] = []
    raw_only = [name for name in base_names if name not in effective_names]
    if raw_only:
        findings.append(f"Raw-only triggers: {', '.join(raw_only)}")
    if suppressed_names:
        findings.append(f"Suppressed triggers: {', '.join(suppressed_names)}")
    findings.extend(observations)
    return findings


def _suite_watch_only_counts(runs: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run in runs:
        base_names = _coerce_str_list(run.get("base_trigger_names"))
        effective_names = _coerce_str_list(run.get("trigger_names"))
        suppressed_names = _coerce_str_list(run.get("suppressed_trigger_names"))
        raw_only = [name for name in base_names if name not in effective_names]
        for name in raw_only + suppressed_names:
            counts[name] = counts.get(name, 0) + 1
    return counts


def _suite_distribution(values: list[float]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def _coerce_dict_list(values: Any) -> list[dict[str, Any]]:
    if not isinstance(values, list):
        return []
    return [dict(item) for item in values if isinstance(item, dict)]


def _top_feature_catalog_entries(
    feature_catalog: list[dict[str, Any]],
    *,
    key: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ranked = sorted(feature_catalog, key=lambda item: float(item.get(key, 0.0) or 0.0), reverse=True)
    return ranked[:limit]


def _aggregate_suite_feature_family_contribution(
    backtests: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for backtest in backtests:
        for item in _coerce_dict_list(backtest.get("feature_family_importance_summary")):
            family = str(item.get("feature_family", "unknown"))
            entry = grouped.setdefault(
                family,
                {
                    "feature_family": family,
                    "mean_abs_shap": [],
                    "mean_missing_rate": [],
                    "mean_stale_rate": [],
                    "data_sources": set(),
                    "run_count": 0,
                },
            )
            cast(list[float], entry["mean_abs_shap"]).append(float(item.get("mean_abs_shap", 0.0) or 0.0))
            cast(list[float], entry["mean_missing_rate"]).append(float(item.get("mean_missing_rate", 0.0) or 0.0))
            cast(list[float], entry["mean_stale_rate"]).append(float(item.get("mean_stale_rate", 0.0) or 0.0))
            cast(set[str], entry["data_sources"]).update(str(source) for source in item.get("data_sources", []))
            entry["run_count"] = int(entry["run_count"]) + 1
    summary: list[dict[str, Any]] = []
    for family, item in grouped.items():
        summary.append(
            {
                "feature_family": family,
                "mean_abs_shap": mean(cast(list[float], item["mean_abs_shap"])),
                "mean_missing_rate": mean(cast(list[float], item["mean_missing_rate"])),
                "mean_stale_rate": mean(cast(list[float], item["mean_stale_rate"])),
                "data_sources": sorted(cast(set[str], item["data_sources"])),
                "run_count": int(item["run_count"]),
            }
        )
    summary.sort(key=lambda item: float(item.get("mean_abs_shap", 0.0) or 0.0), reverse=True)
    return summary


def _aggregate_suite_feature_catalog(backtests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for backtest in backtests:
        for item in _coerce_dict_list(backtest.get("feature_catalog")):
            feature = str(item.get("feature", "unknown"))
            entry = grouped.setdefault(
                feature,
                {
                    "feature": feature,
                    "feature_family": str(item.get("feature_family", "unknown")),
                    "data_source": str(item.get("data_source", "unknown")),
                    "domain": str(item.get("domain", "unknown")),
                    "missing_rate": [],
                    "stale_rate": [],
                    "run_count": 0,
                },
            )
            cast(list[float], entry["missing_rate"]).append(float(item.get("missing_rate", 0.0) or 0.0))
            cast(list[float], entry["stale_rate"]).append(float(item.get("stale_rate", 0.0) or 0.0))
            entry["run_count"] = int(entry["run_count"]) + 1
    summary: list[dict[str, Any]] = []
    for feature, item in grouped.items():
        summary.append(
            {
                "feature": feature,
                "feature_family": item["feature_family"],
                "data_source": item["data_source"],
                "domain": item["domain"],
                "missing_rate": mean(cast(list[float], item["missing_rate"])),
                "stale_rate": mean(cast(list[float], item["stale_rate"])),
                "run_count": int(item["run_count"]),
            }
        )
    summary.sort(key=lambda item: (float(item.get("missing_rate", 0.0) or 0.0), float(item.get("stale_rate", 0.0) or 0.0)), reverse=True)
    return summary


def _execution_source(bundle: ArtifactBundle) -> tuple[dict[str, Any], str]:
    if bundle.weekly_review is not None:
        return dict(bundle.weekly_review.get("execution_diagnostics", {})), "weekly_review"
    if bundle.paper_trading_batch is not None:
        return dict(bundle.paper_trading_batch.get("execution_diagnostics", {})), "paper_trading_batch"
    if bundle.monitor_audit is not None:
        summary = dict(bundle.monitor_audit.get("paper_trading_summary", {}))
        return dict(summary.get("weekly_execution_diagnostics", {})) or dict(summary.get("execution_diagnostics", {})), "monitor_audit"
    return {}, "unavailable"


def _collect_top_risks(
    effective_triggers: list[str],
    watch_only_findings: list[str],
    execution_metrics: dict[str, Any],
    data_sources: dict[str, Any],
    candidate_pbo: float | None,
    adjusted_pbo: float | None,
) -> list[str]:
    risks: list[str] = []
    if effective_triggers:
        risks.append(f"Effective governance trigger(s): {', '.join(effective_triggers)}.")
    if watch_only_findings:
        risks.append(f"Watch-only findings remain active: {watch_only_findings[0]}")
    missed_trade_rate = _safe_float(execution_metrics.get("missed_trade_rate"))
    if missed_trade_rate is not None and missed_trade_rate > 0.0:
        risks.append(f"Execution missed_trade_rate={missed_trade_rate:.2%}.")
    partial_fill_rate = _safe_float(execution_metrics.get("partial_fill_rate"))
    if partial_fill_rate is not None and partial_fill_rate > 0.0:
        risks.append(f"Execution partial_fill_rate={partial_fill_rate:.2%}.")
    if candidate_pbo is not None and adjusted_pbo is not None and candidate_pbo > adjusted_pbo:
        risks.append(
            f"Candidate-level PBO={candidate_pbo:.4f} remains above cluster-adjusted PBO={adjusted_pbo:.4f}; treat raw PBO as explanation-only."
        )
    if data_sources.get("fallback_used"):
        risks.append(f"Fallback data source engaged: {data_sources.get('fallback_reason') or 'reason not provided'}.")
    if data_sources.get("proxy_ohlcv_used"):
        risks.append("Proxy OHLCV data is in use; treat volume/range diagnostics as proxy-sensitive.")
    return risks[:5]


def _collect_next_actions(
    profile_name: str | None,
    effective_triggers: list[str],
    watch_only_findings: list[str],
    execution_metrics: dict[str, Any],
) -> list[str]:
    actions: list[str] = []
    if profile_name == "full_light":
        actions.append("Keep using full_light as the routine monitoring profile.")
        actions.append("Use full only for replay or deep-dive investigations.")
    elif profile_name == "full":
        actions.append("Treat this report as a replay/deep-dive audit, not the routine live monitor.")
    if effective_triggers:
        actions.append(f"Review effective triggers before the next production-style research run: {', '.join(effective_triggers)}.")
    if watch_only_findings:
        actions.append("Keep raw watch-only findings on the monitoring queue without changing policy defaults.")
    fill_rate = _safe_float(execution_metrics.get("fill_rate"))
    if fill_rate is not None and fill_rate < 1.0:
        actions.append("Inspect execution realism diagnostics before interpreting realized PnL or exposure.")
    if not actions:
        actions.append("No action is required beyond routine monitoring and report archival.")
    return actions[:5]


def _artifact_reference_entries(bundle: ArtifactBundle) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for artifact_type, path in [
        (bundle.primary_type, bundle.primary_path),
        ("monitor_audit", bundle.monitor_audit_path),
        ("monitor_audit_suite", bundle.monitor_audit_suite_path),
        ("backtest_result", bundle.backtest_result_path),
        ("paper_trading_batch", bundle.paper_trading_batch_path),
        ("weekly_review", bundle.weekly_review_path),
    ]:
        if path is None:
            continue
        entries.append({"label": artifact_type, "path": str(path)})
    for audit_id, path in bundle.suite_run_audit_paths.items():
        entries.append({"label": f"suite_run_monitor_audit:{audit_id}", "path": str(path)})
    for backtest_id, path in bundle.suite_run_backtest_paths.items():
        entries.append({"label": f"suite_run_backtest:{backtest_id}", "path": str(path)})
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        key = (entry["label"], entry["path"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entry)
    return deduped


def _single_run_context(bundle: ArtifactBundle) -> dict[str, Any]:
    audit = bundle.monitor_audit
    backtest = bundle.backtest_result
    paper_batch = bundle.paper_trading_batch
    notes = _coerce_str_list(audit.get("notes") if audit is not None else [])
    profile_name = _parse_prefixed_note(notes, "suite_profile=")
    profile_role = _parse_prefixed_note(notes, "suite_profile_role=")
    as_of_date = _parse_prefixed_note(notes, "suite_as_of_date=")
    window = dict(audit.get("window", {})) if audit is not None else {}
    if as_of_date is None:
        as_of_date = str(window.get("end_date") or "") or None
    if as_of_date is None:
        as_of_date = str((paper_batch or {}).get("forecast_date", "")) or None
    if as_of_date is None:
        timestamp = str((audit or backtest or paper_batch or {}).get("generated_at") or (backtest or {}).get("completed_at") or "")
        as_of_date = timestamp[:10] if timestamp else None
    model_name = str((backtest or {}).get("config", {}).get("model_name", "unknown"))
    return {
        "source_artifact_type": bundle.primary_type,
        "profile_name": profile_name,
        "profile_role": profile_role,
        "report_scope": "single_run",
        "dataset_type": str((audit or {}).get("dataset_type", "unknown")),
        "analysis_mode": "single_run_report",
        "as_of_date": as_of_date,
        "window": window,
        "ticker_set": list((audit or {}).get("universe", [])),
        "primary_model": model_name,
        "profile_guidance": (
            "full_light is the routine live monitor; full is reserved for replay/deep-dive."
            if profile_name in {"full_light", "full"}
            else "Single-run report rendered from persisted artifacts."
        ),
    }


def _suite_run_context(bundle: ArtifactBundle) -> dict[str, Any]:
    suite = bundle.monitor_audit_suite or {}
    model_name = "unknown"
    if bundle.suite_run_backtests:
        first_backtest = next(iter(bundle.suite_run_backtests.values()))
        model_name = str(first_backtest.get("config", {}).get("model_name", "unknown"))
    elif bundle.backtest_result is not None:
        model_name = str(bundle.backtest_result.get("config", {}).get("model_name", "unknown"))
    as_of_dates = [str(item) for item in suite.get("as_of_dates", [])]
    return {
        "source_artifact_type": "monitor_audit_suite",
        "profile_name": str(suite.get("profile_name", "suite")),
        "profile_role": str(suite.get("profile_role", "aggregate_review")),
        "report_scope": "suite",
        "dataset_type": str(suite.get("dataset_type", "unknown")),
        "analysis_mode": str(suite.get("analysis_mode", "suite")),
        "as_of_date_range": {
            "start": as_of_dates[0] if as_of_dates else None,
            "end": as_of_dates[-1] if as_of_dates else None,
            "count": len(as_of_dates),
        },
        "ticker_sets": [list(items) for items in suite.get("ticker_sets", [])],
        "run_count": int(suite.get("run_count", 0)),
        "cpcv_max_splits": int(suite.get("cpcv_max_splits", 0) or 0),
        "primary_model": model_name,
        "profile_guidance": "full_light is the routine live monitor; full is reserved for replay/deep-dive.",
    }


def _build_single_report(bundle: ArtifactBundle) -> dict[str, Any]:
    audit = bundle.monitor_audit or {}
    backtest = bundle.backtest_result or {}
    weekly_review = bundle.weekly_review or {}
    paper_batch = bundle.paper_trading_batch or {}
    retraining_monitor = dict(audit.get("retraining_monitor", backtest.get("retraining_monitor", {})))
    regime_monitor = dict(audit.get("regime_monitor", backtest.get("regime_monitor", {})))
    drift_monitor = dict(audit.get("drift_monitor", backtest.get("drift_monitor", {})))
    data_sources = dict(audit.get("data_sources", {}))
    execution_metrics, execution_source = _execution_source(bundle)
    cost_metrics = dict(backtest.get("cost_adjusted_metrics", {}))
    aggregate_metrics = dict(backtest.get("aggregate_metrics", {}))
    feature_lineage = dict(audit.get("feature_lineage", {}))
    feature_catalog = _coerce_dict_list(feature_lineage.get("feature_catalog", backtest.get("feature_catalog")))
    feature_family_contribution = _coerce_dict_list(
        feature_lineage.get("feature_family_importance_summary", backtest.get("feature_family_importance_summary"))
    )
    feature_importance = _coerce_dict_list(
        feature_lineage.get("feature_importance_summary", backtest.get("feature_importance_summary"))
    )
    notes = _coerce_str_list(audit.get("notes"))

    run_context = _single_run_context(bundle)
    base_trigger_names = _coerce_str_list(retraining_monitor.get("base_trigger_names"))
    effective_trigger_names = _coerce_str_list(retraining_monitor.get("effective_trigger_names"))
    suppressed_trigger_names = _coerce_str_list(retraining_monitor.get("suppressed_trigger_names"))
    observations = _coerce_str_list(retraining_monitor.get("observations"))
    watch_only_findings = _summarize_watch_only(
        base_trigger_names,
        effective_trigger_names,
        suppressed_trigger_names,
        observations,
    )
    candidate_pbo = _safe_float(retraining_monitor.get("candidate_level_pbo", audit.get("backtest", {}).get("pbo")))
    adjusted_pbo = _safe_float(retraining_monitor.get("pbo", audit.get("backtest", {}).get("cluster_adjusted_pbo")))
    top_risks = _collect_top_risks(
        effective_trigger_names,
        watch_only_findings,
        execution_metrics,
        data_sources,
        candidate_pbo,
        adjusted_pbo,
    )
    next_actions = _collect_next_actions(
        str(run_context.get("profile_name")) if run_context.get("profile_name") else None,
        effective_trigger_names,
        watch_only_findings,
        execution_metrics,
    )
    transition_profile = str(retraining_monitor.get("regime_signal", {}).get("transition_profile", "not_transition"))
    feature_sources = dict(data_sources.get("feature_sources", {}))
    news_feature_source = dict(cast(dict[str, Any], feature_sources.get("news", {})))
    if news_feature_source:
        news_feature_source["weighting_summary"] = _news_weighting_summary(
            dict(news_feature_source.get("utility_comparison", {}))
        )
        feature_sources["news"] = news_feature_source
    approval_status = str(
        paper_batch.get("approval")
        or audit.get("paper_trading_summary", {}).get("approval")
        or weekly_review.get("approval_breakdown", {})
    )
    executive_summary = {
        "run_type": run_context.get("profile_name") or "single_run",
        "as_of_date": run_context.get("as_of_date"),
        "ticker_set": _join_tickers(run_context.get("ticker_set")),
        "primary_model": run_context.get("primary_model"),
        "approval_or_review_status": approval_status,
        "should_retrain": bool(retraining_monitor.get("should_retrain", False)),
        "effective_trigger_summary": effective_trigger_names or ["none"],
        "watch_only_summary": watch_only_findings or ["none"],
    }
    return {
        "report_id": str(uuid4()),
        "report_type": "single_monitor_audit_report",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "section_order": SECTION_ORDER,
        "source_artifacts": {
            "primary_artifact_type": bundle.primary_type,
            "primary_artifact_path": str(bundle.primary_path) if bundle.primary_path else None,
            "related_artifact_count": len(_artifact_reference_entries(bundle)) - 1,
        },
        "executive_summary": executive_summary,
        "run_context": run_context,
        "model_portfolio_performance": {
            "hit_rate_mean": _safe_float(backtest.get("aggregate_metrics", {}).get("hit_rate_mean", audit.get("backtest", {}).get("hit_rate_mean"))),
            "ece_mean": _safe_float(backtest.get("aggregate_metrics", {}).get("ece_mean", audit.get("backtest", {}).get("ece_mean"))),
            "information_ratio": _safe_float(
                backtest.get("cost_adjusted_metrics", {}).get("information_ratio", audit.get("backtest", {}).get("information_ratio"))
            ),
            "selection_stability": _safe_float(cost_metrics.get("selection_stability")),
            "candidate_level_pbo": candidate_pbo,
            "candidate_level_pbo_label": retraining_monitor.get("candidate_level_pbo_label")
            or audit.get("backtest", {}).get("pbo_summary", {}).get("label"),
            "cluster_adjusted_pbo": adjusted_pbo,
            "cluster_adjusted_pbo_label": retraining_monitor.get("pbo_label")
            or audit.get("backtest", {}).get("cluster_adjusted_pbo_summary", {}).get("label"),
            "avg_daily_turnover": _safe_float(cost_metrics.get("avg_daily_turnover")),
            "cost_drag_annual_return": _safe_float(cost_metrics.get("cost_drag_annual_return")),
            "portfolio_rule_analysis": dict(backtest.get("portfolio_rule_analysis", audit.get("backtest", {}).get("portfolio_rule_analysis", {}))),
            "feature_family_contribution": feature_family_contribution,
            "top_feature_contribution": feature_importance[:10],
        },
        "execution_diagnostics": {
            "source_scope": execution_source,
            "fill_rate": _safe_float(execution_metrics.get("fill_rate")),
            "partial_fill_rate": _safe_float(execution_metrics.get("partial_fill_rate")),
            "missed_trade_rate": _safe_float(execution_metrics.get("missed_trade_rate")),
            "realized_vs_intended_exposure": _safe_float(execution_metrics.get("realized_vs_intended_exposure")),
            "execution_cost_drag": _safe_float(execution_metrics.get("execution_cost_drag")),
            "execution_cost_drag_bps": _safe_float(execution_metrics.get("execution_cost_drag_bps")),
            "gap_slippage_bps": _safe_float(execution_metrics.get("gap_slippage_bps")),
            "raw_metrics": execution_metrics,
        },
        "drift_calibration_regime_review": {
            "dominant_cause": str(drift_monitor.get("supplementary_analysis", {}).get("primary_cause", "unknown")),
            "regime_classification": str(regime_monitor.get("current_regime", "unknown")),
            "transition_profile": transition_profile,
            "stable_transition": bool(retraining_monitor.get("regime_signal", {}).get("stable_transition", False)),
            "watch_transition": transition_profile == "watch_transition",
            "unstable_transition": bool(retraining_monitor.get("regime_signal", {}).get("unstable_transition", False)),
            "drift": {
                "max_psi": _safe_float(drift_monitor.get("max_psi")),
                "primary_cause": str(drift_monitor.get("supplementary_analysis", {}).get("primary_cause", "unknown")),
                "trigger_families": _coerce_str_list(retraining_monitor.get("drift_signal", {}).get("trigger_families")),
                "proxy_sensitive_profile": retraining_monitor.get("drift_signal", {}).get("proxy_sensitive_profile"),
            },
            "calibration": {
                "ece_mean": _safe_float(aggregate_metrics.get("ece_mean", audit.get("backtest", {}).get("ece_mean"))),
                "ece_breach_ratio": _safe_float(retraining_monitor.get("calibration_signal", {}).get("ece_breach_ratio")),
                "gap_breach_ratio": _safe_float(retraining_monitor.get("calibration_signal", {}).get("calibration_gap_breach_ratio")),
            },
            "regime": {
                "current_regime": str(regime_monitor.get("current_regime", "unknown")),
                "dominant_recent_regime": str(regime_monitor.get("dominant_recent_regime", "unknown")),
                "state_probability": _safe_float(regime_monitor.get("state_probability")),
                "transition_rate": _safe_float(regime_monitor.get("transition_rate")),
                "transition_profile": transition_profile,
            },
            "notes": notes,
        },
        "retraining_decision": {
            "base_retraining_rate": 1.0 if retraining_monitor.get("base_should_retrain", False) else 0.0,
            "effective_retraining_rate": 1.0 if retraining_monitor.get("should_retrain", False) else 0.0,
            "base_should_retrain": bool(retraining_monitor.get("base_should_retrain", False)),
            "should_retrain": bool(retraining_monitor.get("should_retrain", False)),
            "base_trigger_names": base_trigger_names,
            "effective_trigger_names": effective_trigger_names,
            "suppressed_trigger_names": suppressed_trigger_names,
            "watch_only_observations": observations,
            "policy_decision": str(retraining_monitor.get("policy_decision", "watch_only")),
            "policy_notes": _coerce_str_list(retraining_monitor.get("policy_notes")),
            "dominant_cause_family": (
                effective_trigger_names[0] if effective_trigger_names else base_trigger_names[0] if base_trigger_names else "none"
            ),
            "raw_signal_families": {
                "pbo": ["candidate_level_pbo" if candidate_pbo is not None else "none"],
                "calibration": [
                    name for name in base_trigger_names if "calibration" in name or name in {"ece", "calibration_ece", "calibration_gap"}
                ],
                "drift": [name for name in base_trigger_names if "drift" in name],
                "regime": [name for name in base_trigger_names if "regime" in name],
            },
            "effective_trigger_families": {
                "pbo": [name for name in effective_trigger_names if "pbo" in name],
                "calibration": [
                    name for name in effective_trigger_names if "calibration" in name or name in {"ece", "calibration_ece", "calibration_gap"}
                ],
                "drift": [name for name in effective_trigger_names if "drift" in name],
                "regime": [name for name in effective_trigger_names if "regime" in name],
            },
        },
        "watch_only_findings": {
            "findings": watch_only_findings,
            "raw_signal_names": [name for name in base_trigger_names if name not in effective_trigger_names],
            "suppressed_trigger_names": suppressed_trigger_names,
            "policy_notes": _coerce_str_list(retraining_monitor.get("policy_notes")),
        },
        "data_sources_transport": {
            "ohlcv_source": data_sources.get("ohlcv_source"),
            "macro_source": data_sources.get("macro_source"),
            "feature_sources": feature_sources,
            "proxy_ohlcv_used": data_sources.get("proxy_ohlcv_used"),
            "source_mode": data_sources.get("source_mode"),
            "dummy_mode": data_sources.get("dummy_mode"),
            "fallback_used": data_sources.get("fallback_used"),
            "fallback_reason": data_sources.get("fallback_reason"),
            "ohlcv_transport": dict(data_sources.get("ohlcv_transport", {})),
            "macro_transport": dict(data_sources.get("macro_transport", {})),
            "feature_lineage": {
                "feature_catalog": feature_catalog,
                "highest_missingness_features": _top_feature_catalog_entries(feature_catalog, key="missing_rate"),
                "highest_staleness_features": _top_feature_catalog_entries(feature_catalog, key="stale_rate"),
            },
            "stale_flags": {
                "present": False,
                "detail": "No explicit stale artifact summary is persisted in monitor_audit.",
            },
        },
        "top_risks": top_risks,
        "recommended_next_actions": next_actions,
        "artifact_references": _artifact_reference_entries(bundle),
    }


def _build_suite_report(bundle: ArtifactBundle) -> dict[str, Any]:
    suite = bundle.monitor_audit_suite or {}
    distribution_summary = dict(suite.get("distribution_summary", {}))
    runs = list(suite.get("runs", []))
    suite_backtests = list(bundle.suite_run_backtests.values())
    run_context = _suite_run_context(bundle)
    hit_rate_distribution = _suite_distribution(
        [float(run["hit_rate_mean"]) for run in runs if _safe_float(run.get("hit_rate_mean")) is not None]
    )
    suite_ece_distribution = _suite_distribution(
        [
            float(backtest.get("aggregate_metrics", {}).get("ece_mean"))
            for backtest in bundle.suite_run_backtests.values()
            if _safe_float(backtest.get("aggregate_metrics", {}).get("ece_mean")) is not None
        ]
    )
    suite_turnover_distribution = _suite_distribution(
        [
            float(backtest.get("cost_adjusted_metrics", {}).get("avg_daily_turnover"))
            for backtest in bundle.suite_run_backtests.values()
            if _safe_float(backtest.get("cost_adjusted_metrics", {}).get("avg_daily_turnover")) is not None
        ]
    )
    suite_cost_drag_distribution = _suite_distribution(
        [
            float(backtest.get("cost_adjusted_metrics", {}).get("cost_drag_annual_return"))
            for backtest in suite_backtests
            if _safe_float(backtest.get("cost_adjusted_metrics", {}).get("cost_drag_annual_return")) is not None
        ]
    )
    suite_feature_family_contribution = _aggregate_suite_feature_family_contribution(suite_backtests)
    suite_feature_catalog = _aggregate_suite_feature_catalog(suite_backtests)
    execution_metrics, execution_source = _execution_source(bundle)
    base_trigger_counts = dict(distribution_summary.get("base_trigger_counts", {}))
    effective_trigger_counts = dict(distribution_summary.get("effective_trigger_counts", {}))
    watch_only_counts = _suite_watch_only_counts(runs)
    candidate_pbo_summary = dict(distribution_summary.get("pbo", {}))
    adjusted_pbo_summary = dict(distribution_summary.get("cluster_adjusted_pbo", {}))
    effective_triggers = _top_items(effective_trigger_counts)
    watch_only_findings = _top_items(watch_only_counts)
    proxy_flag_values = [
        bool(audit.get("data_sources", {}).get("proxy_ohlcv_used", False))
        for audit in bundle.suite_run_audits.values()
        if "data_sources" in audit
    ]
    cache_usage_counts = {
        "ohlcv_cache_used": sum(
            1 for audit in bundle.suite_run_audits.values() if audit.get("data_sources", {}).get("ohlcv_transport", {}).get("cache_used")
        ),
        "ohlcv_snapshot_used": sum(
            1 for audit in bundle.suite_run_audits.values() if audit.get("data_sources", {}).get("ohlcv_transport", {}).get("snapshot_used")
        ),
        "macro_cache_used": sum(
            1 for audit in bundle.suite_run_audits.values() if audit.get("data_sources", {}).get("macro_transport", {}).get("cache_used")
        ),
        "macro_snapshot_used": sum(
            1 for audit in bundle.suite_run_audits.values() if audit.get("data_sources", {}).get("macro_transport", {}).get("snapshot_used")
        ),
    }
    suite_news_utility_comparison = dict(
        cast(
            dict[str, Any],
            distribution_summary.get(
                "news_utility_comparison",
                distribution_summary.get("news_coverage_analysis", {}),
            )
            or {},
        )
    )
    suite_news_weighting_summary = _news_weighting_summary(suite_news_utility_comparison)
    top_risks = _collect_top_risks(
        [item.split("=")[0] for item in effective_triggers],
        watch_only_findings,
        execution_metrics,
        {
            "proxy_ohlcv_used": any(proxy_flag_values),
            "fallback_used": False,
            "fallback_reason": None,
        },
        _safe_float(candidate_pbo_summary.get("mean")),
        _safe_float(adjusted_pbo_summary.get("mean")),
    )
    next_actions = _collect_next_actions(
        str(run_context.get("profile_name")),
        [item.split("=")[0] for item in effective_triggers],
        watch_only_findings,
        execution_metrics,
    )
    suite_retraining_rate = _safe_float(distribution_summary.get("retraining_rate"))
    executive_summary = {
        "run_type": run_context.get("profile_name"),
        "as_of_date": {
            "start": run_context.get("as_of_date_range", {}).get("start"),
            "end": run_context.get("as_of_date_range", {}).get("end"),
        },
        "ticker_set": " | ".join(_join_tickers(items) for items in run_context.get("ticker_sets", [])),
        "primary_model": run_context.get("primary_model"),
        "approval_or_review_status": f"{suite.get('profile_role', 'aggregate_review')} aggregate",
        "should_retrain": bool(suite_retraining_rate is not None and suite_retraining_rate > 0.0),
        "effective_trigger_summary": effective_triggers or ["none"],
        "watch_only_summary": watch_only_findings or ["none"],
    }
    return {
        "report_id": str(uuid4()),
        "report_type": "monitor_audit_suite_report",
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "section_order": SECTION_ORDER,
        "source_artifacts": {
            "primary_artifact_type": "monitor_audit_suite",
            "primary_artifact_path": str(bundle.primary_path) if bundle.primary_path else None,
            "related_artifact_count": len(_artifact_reference_entries(bundle)) - 1,
        },
        "executive_summary": executive_summary,
        "run_context": run_context,
        "model_portfolio_performance": {
            "hit_rate_mean": hit_rate_distribution,
            "ece_mean": suite_ece_distribution,
            "information_ratio": dict(distribution_summary.get("information_ratio", {})),
            "selection_stability": dict(distribution_summary.get("selection_stability", {})),
            "candidate_level_pbo": candidate_pbo_summary,
            "cluster_adjusted_pbo": adjusted_pbo_summary,
            "avg_daily_turnover": suite_turnover_distribution,
            "cost_drag_annual_return": suite_cost_drag_distribution,
            "candidate_pbo_competition_dominated_rate": _safe_float(distribution_summary.get("candidate_pbo_competition_dominated_rate")),
            "cluster_adjusted_pbo_competition_dominated_rate": _safe_float(distribution_summary.get("pbo_competition_dominated_rate")),
            "feature_family_contribution": suite_feature_family_contribution,
        },
        "execution_diagnostics": {
            "source_scope": execution_source,
            "fill_rate": _safe_float(execution_metrics.get("fill_rate")),
            "partial_fill_rate": _safe_float(execution_metrics.get("partial_fill_rate")),
            "missed_trade_rate": _safe_float(execution_metrics.get("missed_trade_rate")),
            "realized_vs_intended_exposure": _safe_float(execution_metrics.get("realized_vs_intended_exposure")),
            "execution_cost_drag": _safe_float(execution_metrics.get("execution_cost_drag")),
            "execution_cost_drag_bps": _safe_float(execution_metrics.get("execution_cost_drag_bps")),
            "gap_slippage_bps": _safe_float(execution_metrics.get("gap_slippage_bps")),
            "raw_metrics": execution_metrics,
        },
        "drift_calibration_regime_review": {
            "dominant_cause": next(iter(effective_trigger_counts)) if effective_trigger_counts else next(iter(base_trigger_counts)) if base_trigger_counts else "none",
            "regime_classification": dict(distribution_summary.get("regime_dominated_analysis", {}).get("current_regime_counts", {})),
            "transition_profile": dict(distribution_summary.get("regime_dominated_analysis", {}).get("transition_profile_counts", {})),
            "stable_transition": {
                "profile": "stable_transition",
                "count": int(distribution_summary.get("regime_dominated_analysis", {}).get("transition_profile_counts", {}).get("stable_transition", 0)),
            },
            "watch_transition": {
                "profile": "watch_transition",
                "count": int(distribution_summary.get("regime_dominated_analysis", {}).get("transition_profile_counts", {}).get("watch_transition", 0)),
            },
            "unstable_transition": {
                "profile": "unstable_transition",
                "count": int(distribution_summary.get("regime_dominated_analysis", {}).get("transition_profile_counts", {}).get("unstable_transition", 0)),
            },
            "drift_dominated_analysis": dict(distribution_summary.get("drift_dominated_analysis", {})),
            "calibration_dominated_analysis": dict(distribution_summary.get("calibration_dominated_analysis", {})),
            "regime_dominated_analysis": dict(distribution_summary.get("regime_dominated_analysis", {})),
        },
        "retraining_decision": {
            "base_retraining_rate": _safe_float(distribution_summary.get("base_retraining_rate")),
            "effective_retraining_rate": _safe_float(distribution_summary.get("retraining_rate")),
            "base_trigger_counts": base_trigger_counts,
            "effective_trigger_counts": effective_trigger_counts,
            "watch_only_counts": watch_only_counts,
            "policy_decision_counts": dict(distribution_summary.get("policy_decision_counts", {})),
            "dominant_cause_family": next(iter(effective_trigger_counts)) if effective_trigger_counts else next(iter(base_trigger_counts)) if base_trigger_counts else "none",
            "raw_signal_families": {
                "pbo": dict(distribution_summary.get("candidate_pbo_dominant_axis_counts", {})),
                "calibration": dict(distribution_summary.get("calibration_dominated_analysis", {}).get("trigger_metric_counts", {})),
                "drift": dict(distribution_summary.get("drift_dominated_analysis", {}).get("trigger_family_counts", {})),
                "regime": dict(distribution_summary.get("regime_dominated_analysis", {}).get("base_trigger_counts", {})),
            },
            "effective_trigger_families": {
                "pbo": {},
                "calibration": {},
                "drift": dict(distribution_summary.get("drift_dominated_analysis", {}).get("effective_trigger_family_counts", {})),
                "regime": dict(distribution_summary.get("regime_dominated_analysis", {}).get("effective_trigger_counts", {})),
            },
        },
        "watch_only_findings": {
            "findings": watch_only_findings,
            "raw_signal_names": sorted(watch_only_counts.keys()),
            "suppressed_trigger_names": sorted(watch_only_counts.keys()),
            "policy_notes": ["Raw watch-only regime_shift remains monitored without changing policy defaults."],
        },
        "data_sources_transport": {
            "ohlcv_source": dict(distribution_summary.get("drift_dominated_analysis", {}).get("ohlcv_source_counts", {})),
            "macro_source": {key: sum(1 for run in runs if str(run.get("macro_source", "")) == key) for key in {str(run.get("macro_source", "")) for run in runs}},
            "feature_sources": {
                "news": {
                    "coverage": dict(distribution_summary.get("news_feature_coverage", {})),
                    "missingness": dict(distribution_summary.get("news_feature_missing_rate", {})),
                    "staleness": dict(distribution_summary.get("news_feature_staleness", {})),
                    "used_source_counts": dict(distribution_summary.get("news_used_source_counts", {})),
                    "transport_origin_counts": dict(distribution_summary.get("news_transport_origin_counts", {})),
                    "fallback_rate": _safe_float(distribution_summary.get("news_fallback_rate")),
                    "coverage_analysis": dict(distribution_summary.get("news_coverage_analysis", {})),
                    "utility_comparison": suite_news_utility_comparison,
                    "weighting_summary": suite_news_weighting_summary,
                    "by_ticker_set": [
                        {
                            "ticker_set": item.get("ticker_set"),
                            "coverage": item.get("news_feature_coverage", {}),
                            "missingness": item.get("news_feature_missing_rate", {}),
                            "staleness": item.get("news_feature_staleness", {}),
                            "used_source_counts": item.get("news_used_source_counts", {}),
                            "transport_origin_counts": item.get("news_transport_origin_counts", {}),
                            "fallback_rate": item.get("news_fallback_rate"),
                            "coverage_analysis": item.get("news_coverage_analysis", {}),
                            "utility_comparison": item.get(
                                "news_utility_comparison",
                                item.get("news_coverage_analysis", {}),
                            ),
                        }
                        for item in cast(list[dict[str, Any]], bundle.primary_payload.get("by_ticker_set", []))
                    ],
                    "by_used_source": cast(list[dict[str, Any]], bundle.primary_payload.get("by_news_used_source", [])),
                    "by_transport_origin": cast(list[dict[str, Any]], bundle.primary_payload.get("by_news_transport_origin", [])),
                },
                "families": [
                    {
                        "feature_family": item.get("feature_family"),
                        "data_sources": item.get("data_sources", []),
                    }
                    for item in suite_feature_family_contribution
                ]
            },
            "proxy_ohlcv_used": {
                "true_count": sum(1 for value in proxy_flag_values if value),
                "false_count": sum(1 for value in proxy_flag_values if not value),
            },
            "source_mode": "live",
            "dummy_mode": None,
            "fallback_used": False,
            "fallback_reason": None,
            "ohlcv_transport": {
                "origin_counts": dict(distribution_summary.get("drift_dominated_analysis", {}).get("ohlcv_transport_origin_counts", {})),
                **cache_usage_counts,
            },
            "macro_transport": {
                "origin_counts": dict(distribution_summary.get("drift_dominated_analysis", {}).get("macro_transport_origin_counts", {})),
                **cache_usage_counts,
            },
            "feature_lineage": {
                "feature_catalog": suite_feature_catalog,
                "highest_missingness_features": _top_feature_catalog_entries(suite_feature_catalog, key="missing_rate"),
                "highest_staleness_features": _top_feature_catalog_entries(suite_feature_catalog, key="stale_rate"),
            },
            "stale_flags": {
                "present": False,
                "detail": "No explicit stale summary is aggregated in suite artifacts.",
            },
        },
        "top_risks": top_risks,
        "recommended_next_actions": next_actions,
        "artifact_references": _artifact_reference_entries(bundle),
    }


def build_audit_report(bundle: ArtifactBundle) -> dict[str, Any]:
    if bundle.monitor_audit_suite is not None and bundle.primary_type == "monitor_audit_suite":
        return _build_suite_report(bundle)
    return _build_single_report(bundle)


def _markdown_metric_lines(report: dict[str, Any]) -> list[str]:
    performance = dict(report.get("model_portfolio_performance", {}))
    return [
        f"- hit_rate_mean: {_format_distribution(performance.get('hit_rate_mean')) if isinstance(performance.get('hit_rate_mean'), dict) else _format_number(performance.get('hit_rate_mean'))}",
        f"- ece_mean: {_format_distribution(performance.get('ece_mean')) if isinstance(performance.get('ece_mean'), dict) else _format_number(performance.get('ece_mean'))}",
        f"- information_ratio: {_format_distribution(performance.get('information_ratio')) if isinstance(performance.get('information_ratio'), dict) else _format_number(performance.get('information_ratio'))}",
        f"- selection_stability: {_format_distribution(performance.get('selection_stability')) if isinstance(performance.get('selection_stability'), dict) else _format_number(performance.get('selection_stability'))}",
        f"- candidate-level PBO: {_format_distribution(performance.get('candidate_level_pbo')) if isinstance(performance.get('candidate_level_pbo'), dict) else _format_number(performance.get('candidate_level_pbo'))}",
        f"- cluster-adjusted PBO: {_format_distribution(performance.get('cluster_adjusted_pbo')) if isinstance(performance.get('cluster_adjusted_pbo'), dict) else _format_number(performance.get('cluster_adjusted_pbo'))}",
        f"- turnover: {_format_distribution(performance.get('avg_daily_turnover')) if isinstance(performance.get('avg_daily_turnover'), dict) else _format_number(performance.get('avg_daily_turnover'))}",
        f"- cost_drag: {_format_distribution(performance.get('cost_drag_annual_return')) if isinstance(performance.get('cost_drag_annual_return'), dict) else _format_number(performance.get('cost_drag_annual_return'))}",
    ]


def render_audit_report_markdown(report: dict[str, Any]) -> str:
    executive_summary = dict(report.get("executive_summary", {}))
    run_context = dict(report.get("run_context", {}))
    performance = dict(report.get("model_portfolio_performance", {}))
    execution = dict(report.get("execution_diagnostics", {}))
    governance = dict(report.get("drift_calibration_regime_review", {}))
    retraining = dict(report.get("retraining_decision", {}))
    watch_only = dict(report.get("watch_only_findings", {}))
    data_sources = dict(report.get("data_sources_transport", {}))
    news_feature_sources = dict(cast(dict[str, Any], data_sources.get("feature_sources", {})).get("news", {}))
    references = list(report.get("artifact_references", []))
    lines = ["# Audit Report", ""]
    lines.extend(
        [
            "## Executive Summary",
            f"- Run type: {executive_summary.get('run_type', 'n/a')}",
            f"- As-of date: {executive_summary.get('as_of_date', 'n/a')}",
            f"- Ticker set: {executive_summary.get('ticker_set', 'n/a')}",
            f"- Primary model: {executive_summary.get('primary_model', 'n/a')}",
            f"- Approval / review status: {executive_summary.get('approval_or_review_status', 'n/a')}",
            f"- should_retrain: {executive_summary.get('should_retrain', 'n/a')}",
            f"- Effective trigger summary: {', '.join(_coerce_str_list(executive_summary.get('effective_trigger_summary'))) or 'none'}",
            f"- Watch-only summary: {', '.join(_coerce_str_list(executive_summary.get('watch_only_summary'))) or 'none'}",
            "",
            "## Run Context",
            f"- Source artifact type: {run_context.get('source_artifact_type', 'n/a')}",
            f"- Dataset type: {run_context.get('dataset_type', 'n/a')}",
            f"- Analysis mode: {run_context.get('analysis_mode', 'n/a')}",
            f"- Profile name: {run_context.get('profile_name', 'n/a')}",
            f"- Profile role: {run_context.get('profile_role', 'n/a')}",
            f"- Profile guidance: {run_context.get('profile_guidance', 'n/a')}",
        ]
    )
    if "window" in run_context:
        lines.append(f"- Window: {json.dumps(run_context.get('window', {}), ensure_ascii=False)}")
    if "run_count" in run_context:
        lines.append(f"- Run count: {run_context.get('run_count', 'n/a')}")
    if "ticker_sets" in run_context:
        lines.append(f"- Ticker sets: {' | '.join(_join_tickers(item) for item in run_context.get('ticker_sets', []))}")
    lines.extend(["", "## Model / Portfolio Performance"])
    lines.extend(_markdown_metric_lines(report))
    if performance.get("candidate_pbo_competition_dominated_rate") is not None:
        lines.append(f"- candidate-level PBO competition-dominated rate: {_format_number(performance.get('candidate_pbo_competition_dominated_rate'))}")
    if performance.get("cluster_adjusted_pbo_competition_dominated_rate") is not None:
        lines.append(f"- cluster-adjusted PBO competition-dominated rate: {_format_number(performance.get('cluster_adjusted_pbo_competition_dominated_rate'))}")
    if performance.get("portfolio_rule_analysis"):
        lines.append(f"- portfolio_rule_analysis: {json.dumps(performance.get('portfolio_rule_analysis'), ensure_ascii=False)}")
    if performance.get("feature_family_contribution") is not None:
        lines.append(
            f"- feature_family_contribution: {json.dumps(performance.get('feature_family_contribution'), ensure_ascii=False)}"
        )
    if performance.get("top_feature_contribution") is not None:
        lines.append(
            f"- top_feature_contribution: {json.dumps(performance.get('top_feature_contribution'), ensure_ascii=False)}"
        )
    lines.extend(
        [
            "",
            "## Execution Diagnostics",
            f"- Source scope: {execution.get('source_scope', 'n/a')}",
            f"- fill_rate: {_format_number(execution.get('fill_rate'))}",
            f"- partial_fill_rate: {_format_number(execution.get('partial_fill_rate'))}",
            f"- missed_trade_rate: {_format_number(execution.get('missed_trade_rate'))}",
            f"- realized_vs_intended_exposure: {_format_number(execution.get('realized_vs_intended_exposure'))}",
            f"- execution_cost_drag: {_format_number(execution.get('execution_cost_drag'))}",
            f"- gap_slippage_bps: {_format_number(execution.get('gap_slippage_bps'))}",
            "",
            "## Drift / Calibration / Regime Review",
            f"- Dominant cause: {governance.get('dominant_cause', 'n/a')}",
            f"- Regime classification: {json.dumps(governance.get('regime_classification'), ensure_ascii=False)}",
            f"- Transition profile: {json.dumps(governance.get('transition_profile'), ensure_ascii=False)}",
            f"- stable_transition: {json.dumps(governance.get('stable_transition'), ensure_ascii=False)}",
            f"- watch_transition: {json.dumps(governance.get('watch_transition'), ensure_ascii=False)}",
            f"- unstable_transition: {json.dumps(governance.get('unstable_transition'), ensure_ascii=False)}",
        ]
    )
    for key in ["drift_dominated_analysis", "calibration_dominated_analysis", "regime_dominated_analysis", "drift", "calibration", "regime"]:
        if governance.get(key) is not None:
            lines.append(f"- {key}: {json.dumps(governance.get(key), ensure_ascii=False)}")
    lines.extend(
        [
            "",
            "## Retraining Decision",
            f"- base_retraining_rate: {_format_number(retraining.get('base_retraining_rate'))}",
            f"- effective_retraining_rate: {_format_number(retraining.get('effective_retraining_rate'))}",
            f"- Raw signal triggers: {_format_count_map(retraining.get('base_trigger_counts')) if isinstance(retraining.get('base_trigger_counts'), dict) else ', '.join(_coerce_str_list(retraining.get('base_trigger_names'))) or 'none'}",
            f"- Effective triggers: {_format_count_map(retraining.get('effective_trigger_counts')) if isinstance(retraining.get('effective_trigger_counts'), dict) else ', '.join(_coerce_str_list(retraining.get('effective_trigger_names'))) or 'none'}",
            f"- Dominant cause: {retraining.get('dominant_cause_family', 'none')}",
            f"- Policy decision: {retraining.get('policy_decision', 'n/a')}",
            f"- Policy notes: {', '.join(_coerce_str_list(retraining.get('policy_notes'))) or 'none'}",
            "",
            "## Watch-Only Findings",
            f"- Findings: {', '.join(_coerce_str_list(watch_only.get('findings'))) or 'none'}",
            f"- Raw watch-only names: {', '.join(_coerce_str_list(watch_only.get('raw_signal_names'))) or 'none'}",
            f"- Suppressed triggers: {', '.join(_coerce_str_list(watch_only.get('suppressed_trigger_names'))) or 'none'}",
            "",
            "## Data Sources and Transport",
            f"- ohlcv_source: {json.dumps(data_sources.get('ohlcv_source'), ensure_ascii=False)}",
            f"- macro_source: {json.dumps(data_sources.get('macro_source'), ensure_ascii=False)}",
            f"- feature_sources: {json.dumps(data_sources.get('feature_sources'), ensure_ascii=False)}",
            f"- news_utility_comparison: {json.dumps(news_feature_sources.get('utility_comparison', news_feature_sources.get('coverage_analysis', {})), ensure_ascii=False)}",
            f"- news_weighting_summary: {json.dumps(news_feature_sources.get('weighting_summary', {}), ensure_ascii=False)}",
            f"- proxy_ohlcv_used: {json.dumps(data_sources.get('proxy_ohlcv_used'), ensure_ascii=False)}",
            f"- ohlcv_transport: {json.dumps(data_sources.get('ohlcv_transport'), ensure_ascii=False)}",
            f"- macro_transport: {json.dumps(data_sources.get('macro_transport'), ensure_ascii=False)}",
            f"- feature_lineage: {json.dumps(data_sources.get('feature_lineage'), ensure_ascii=False)}",
            f"- fallback: used={data_sources.get('fallback_used', 'n/a')} reason={data_sources.get('fallback_reason', 'n/a')}",
            f"- stale_flags: {json.dumps(data_sources.get('stale_flags'), ensure_ascii=False)}",
            "",
            "## Top Risks",
        ]
    )
    for risk in report.get("top_risks", []):
        lines.append(f"- {risk}")
    if not report.get("top_risks"):
        lines.append("- none")
    lines.extend(["", "## Recommended Next Actions"])
    for action in report.get("recommended_next_actions", []):
        lines.append(f"- {action}")
    if not report.get("recommended_next_actions"):
        lines.append("- none")
    lines.extend(["", "## Artifact References"])
    for entry in references:
        lines.append(f"- {entry.get('label')}: {entry.get('path')}")
    if not references:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def default_report_output_paths(
    settings: Settings,
    report: dict[str, Any],
    *,
    json_output: str | Path | None = None,
    markdown_output: str | Path | None = None,
) -> tuple[Path, Path]:
    generated_date = str(report.get("generated_at", ""))[:10] or datetime.now(tz=timezone.utc).date().isoformat()
    report_type = str(report.get("report_type", "audit_report"))
    base_dir = resolve_storage_path(settings) / "outputs" / "audit_reports" / report_type / generated_date
    base_dir.mkdir(parents=True, exist_ok=True)
    report_id = str(report.get("report_id", uuid4()))
    resolved_json_output = resolve_repo_path(json_output) if json_output is not None else base_dir / f"{report_id}.json"
    resolved_markdown_output = resolve_repo_path(markdown_output) if markdown_output is not None else base_dir / f"{report_id}.md"
    resolved_json_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_markdown_output.parent.mkdir(parents=True, exist_ok=True)
    return resolved_json_output, resolved_markdown_output


def persist_audit_report(
    settings: Settings,
    report: dict[str, Any],
    markdown: str,
    *,
    json_output: str | Path | None = None,
    markdown_output: str | Path | None = None,
    write_json: bool = True,
    write_markdown: bool = True,
) -> tuple[Path | None, Path | None]:
    json_path, markdown_path = default_report_output_paths(
        settings,
        report,
        json_output=json_output,
        markdown_output=markdown_output,
    )
    written_json_path: Path | None = None
    written_markdown_path: Path | None = None
    if write_json:
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        written_json_path = json_path
    if write_markdown:
        markdown_path.write_text(markdown, encoding="utf-8")
        written_markdown_path = markdown_path
    return written_json_path, written_markdown_path
