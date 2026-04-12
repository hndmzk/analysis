from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json
from statistics import median
from typing import Any, cast
from uuid import uuid4

import pandas as pd

from market_prediction_agent.config import LearnedWeightingConfig, Settings, resolve_storage_path
from market_prediction_agent.evaluation.news_analysis import (
    build_news_feature_utility_comparison,
    summarize_news_feature_utility_comparison,
)
from market_prediction_agent.evaluation.retraining import build_retraining_history_entry, build_retraining_monitor


@dataclass(slots=True)
class PublicAuditProfile:
    name: str
    as_of_dates: list[str]
    ticker_sets: list[list[str]]
    history_days: int
    cpcv_max_splits: int
    role: str


def profile_role_for_name(profile_name: str | None) -> str | None:
    if profile_name is None:
        return None
    normalized = profile_name.strip().lower()
    roles = {
        "fast": "development_smoke",
        "standard": "research_comparison",
        "full_light": "routine_monitoring",
        "full": "replay_or_deep_dive",
    }
    return roles.get(normalized)


def parse_ticker_sets(spec: str | None, default_tickers: list[str]) -> list[list[str]]:
    if not spec:
        return [list(default_tickers)]
    ticker_sets: list[list[str]] = []
    for group in spec.split("|"):
        tickers = [item.strip().upper() for item in group.split(",") if item.strip()]
        if tickers:
            ticker_sets.append(tickers)
    return ticker_sets or [list(default_tickers)]


def parse_as_of_dates(spec: str | None) -> list[str]:
    if not spec:
        raise ValueError("At least one as-of date is required.")
    values = sorted({pd.Timestamp(item.strip()).date().isoformat() for item in spec.split(",") if item.strip()})
    if not values:
        raise ValueError("At least one as-of date is required.")
    return values


def _distribution(values: list[float]) -> dict[str, float | int]:
    clean = [float(value) for value in values]
    if not clean:
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    series = pd.Series(clean, dtype=float)
    return {
        "count": int(series.shape[0]),
        "mean": float(series.mean()),
        "median": float(median(clean)),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def _as_int(value: object) -> int:
    return int(cast(int | str | float, value))


def _as_float(value: object) -> float:
    return float(cast(int | str | float, value))


def _trigger_counts(runs: list[dict[str, object]], field: str) -> dict[str, int]:
    trigger_names = sorted({str(name) for run in runs for name in cast(list[object], run.get(field, []))})
    return {
        name: int(sum(name in cast(list[object], run.get(field, [])) for run in runs))
        for name in trigger_names
    }


def _policy_decision_counts(runs: list[dict[str, object]]) -> dict[str, int]:
    counts = pd.Series([str(run.get("policy_decision", "unknown")) for run in runs], dtype=str).value_counts()
    return {str(index): int(value) for index, value in counts.items()}


def _value_counts(runs: list[dict[str, object]], field: str) -> dict[str, int]:
    values = [str(run.get(field, "")) for run in runs if str(run.get(field, ""))]
    if not values:
        return {}
    counts = pd.Series(values, dtype=str).value_counts()
    return {str(index): int(value) for index, value in counts.items()}


def _value_counts_when_true(
    runs: list[dict[str, object]],
    *,
    field: str,
    condition_field: str,
) -> dict[str, int]:
    values = [
        str(run.get(field, ""))
        for run in runs
        if bool(run.get(condition_field, False)) and str(run.get(field, ""))
    ]
    if not values:
        return {}
    counts = pd.Series(values, dtype=str).value_counts()
    return {str(index): int(value) for index, value in counts.items()}


def _subset_distribution(
    runs: list[dict[str, object]],
    *,
    field: str,
    match_field: str,
    match_value: str,
) -> dict[str, float | int]:
    values = [
        float(cast(float, run[field]))
        for run in runs
        if str(run.get(match_field, "")) == match_value and field in run
    ]
    return _distribution(values)


def _group_distribution_by_value(
    runs: list[dict[str, object]],
    *,
    field: str,
    group_field: str,
) -> list[dict[str, object]]:
    groups = sorted({str(run.get(group_field, "")) for run in runs if str(run.get(group_field, ""))})
    return [
        {
            group_field: group,
            "run_count": int(sum(str(run.get(group_field, "")) == group for run in runs)),
            field: _subset_distribution(runs, field=field, match_field=group_field, match_value=group),
        }
        for group in groups
    ]


def build_news_feature_coverage_analysis(
    *,
    ohlcv: pd.DataFrame,
    news: pd.DataFrame,
    weighting_mode: str = "fixed",
    learned_weighting: LearnedWeightingConfig | None = None,
    lookback_windows: tuple[int, ...] = (1, 3, 5, 10),
    decay_halflives: tuple[int, ...] = (3, 5),
    min_effective_weight: float = 0.05,
) -> dict[str, object]:
    return build_news_feature_utility_comparison(
        ohlcv=ohlcv,
        news=news,
        weighting_mode=weighting_mode,
        learned_weighting=learned_weighting,
        lookback_windows=lookback_windows,
        decay_halflives=decay_halflives,
        min_effective_weight=min_effective_weight,
    )


def _news_coverage_summary(runs: list[dict[str, object]]) -> dict[str, object]:
    return summarize_news_feature_utility_comparison(runs)


def _normalize_tickers(tickers: list[str]) -> list[str]:
    return [str(ticker).upper() for ticker in tickers]


def _calibration_combo_label(*, ece: bool, gap: bool) -> str:
    if ece and gap:
        return "both"
    if ece:
        return "ece_only"
    if gap:
        return "gap_only"
    return "none"


def _calibration_trigger_combo(run: dict[str, object]) -> str:
    trigger_names = {str(name) for name in cast(list[object], run.get("trigger_names", []))}
    return _calibration_combo_label(
        ece="calibration_ece" in trigger_names,
        gap="calibration_gap" in trigger_names,
    )


def _calibration_dominated_runs(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    dominated: list[dict[str, object]] = []
    for run in runs:
        if not bool(run.get("should_retrain", False)):
            continue
        trigger_names = {str(name) for name in cast(list[object], run.get("trigger_names", []))}
        calibration_triggers = trigger_names.intersection({"calibration_ece", "calibration_gap"})
        if calibration_triggers and calibration_triggers == trigger_names:
            dominated.append(run)
    return dominated


def _drift_dominated_runs(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    dominated: list[dict[str, object]] = []
    for run in runs:
        raw_score = float(cast(float, run.get("drift_pre_suppression_weighted_score", 0.0) or 0.0))
        threshold = float(cast(float, run.get("drift_weighted_threshold", 0.0) or 0.0))
        if raw_score > 0.0 and raw_score >= threshold:
            dominated.append(run)
    return dominated


def _regime_dominated_runs(runs: list[dict[str, object]]) -> list[dict[str, object]]:
    regime_trigger_names = {"regime_shift", "transition_regime", "high_vol_regime"}
    dominated: list[dict[str, object]] = []
    for run in runs:
        trigger_names = {str(name) for name in cast(list[object], run.get("base_trigger_names", []))}
        if trigger_names.intersection(regime_trigger_names):
            dominated.append(run)
    return dominated


def _list_item_counts(runs: list[dict[str, object]], field: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for run in runs:
        for item in cast(list[object], run.get(field, [])):
            label = str(item)
            if label:
                counts[label] += 1
    return dict(counts)


def _effective_relief_count(
    runs: list[dict[str, object]],
    *,
    relief_field: str,
) -> int:
    count = 0
    for run in runs:
        trigger_names = {str(name) for name in cast(list[object], run.get("trigger_names", []))}
        non_drift = trigger_names.difference({"feature_drift"})
        if bool(run.get(relief_field, False)) and not non_drift:
            count += 1
    return count


def _default_secondary_ticker_set(default_tickers: list[str]) -> list[str] | None:
    primary = _normalize_tickers(default_tickers)
    if len(primary) < 4:
        return None
    if "DIA" in primary:
        secondary = [ticker for ticker in primary if ticker != "DIA"]
    else:
        secondary = primary[:-1]
    if len(secondary) < 3 or secondary == primary:
        return None
    return secondary


def _generated_as_of_dates(*, anchor_date: str | None, periods: int) -> list[str]:
    anchor = pd.Timestamp(anchor_date or pd.Timestamp.now(tz="UTC").date().isoformat())
    candidate = anchor.normalize()
    while candidate.weekday() != 4:
        candidate -= pd.Timedelta(days=1)
    return [
        (candidate - pd.Timedelta(days=7 * offset)).date().isoformat()
        for offset in range(periods - 1, -1, -1)
    ]


def resolve_public_audit_profile(
    *,
    profile_name: str,
    default_tickers: list[str],
    anchor_date: str | None = None,
    as_of_dates_override: str | None = None,
    ticker_sets_override: str | None = None,
    history_days_override: int | None = None,
    cpcv_max_splits_override: int | None = None,
) -> PublicAuditProfile:
    normalized_profile = profile_name.strip().lower()
    profile_defaults: dict[str, dict[str, object]] = {
        "fast": {
            "periods": 3,
            "history_days": 1100,
            "cpcv_max_splits": 1,
            "include_secondary": False,
            "role": "development_smoke",
        },
        "standard": {
            "periods": 6,
            "history_days": 1100,
            "cpcv_max_splits": 2,
            "include_secondary": True,
            "role": "research_comparison",
        },
        "full_light": {
            "periods": 12,
            "history_days": 1100,
            "cpcv_max_splits": 1,
            "include_secondary": False,
            "role": "routine_monitoring",
        },
        "full": {
            "periods": 12,
            "history_days": 1100,
            "cpcv_max_splits": 2,
            "include_secondary": True,
            "role": "replay_or_deep_dive",
        },
    }
    if normalized_profile not in profile_defaults:
        raise ValueError(f"Unsupported audit profile: {profile_name}")
    defaults = profile_defaults[normalized_profile]
    primary_tickers = _normalize_tickers(default_tickers)
    ticker_sets = parse_ticker_sets(ticker_sets_override, primary_tickers) if ticker_sets_override else [primary_tickers]
    if not ticker_sets_override and bool(defaults["include_secondary"]):
        secondary = _default_secondary_ticker_set(primary_tickers)
        if secondary is not None:
            ticker_sets.append(secondary)
    as_of_dates = (
        parse_as_of_dates(as_of_dates_override)
        if as_of_dates_override
        else _generated_as_of_dates(anchor_date=anchor_date, periods=_as_int(defaults["periods"]))
    )
    return PublicAuditProfile(
        name=normalized_profile,
        as_of_dates=as_of_dates,
        ticker_sets=ticker_sets,
        history_days=_as_int(history_days_override or defaults["history_days"]),
        cpcv_max_splits=_as_int(cpcv_max_splits_override or defaults["cpcv_max_splits"]),
        role=str(defaults["role"]),
    )


def build_public_audit_suite(
    *,
    runs: list[dict[str, object]],
    as_of_dates: list[str],
    ticker_sets: list[list[str]],
    cpcv_max_splits: int,
    profile_name: str | None = None,
    profile_role: str | None = None,
    analysis_mode: str = "live_suite",
    source_suite_id: str | None = None,
    source_suite_path: str | None = None,
) -> dict[str, object]:
    calibration_dominated_runs = _calibration_dominated_runs(runs)
    drift_dominated_runs = _drift_dominated_runs(runs)
    regime_dominated_runs = _regime_dominated_runs(runs)
    effective_drift_runs = [run for run in runs if "feature_drift" in cast(list[object], run.get("trigger_names", []))]
    effective_regime_runs = [
        run
        for run in runs
        if {
            str(name) for name in cast(list[object], run.get("trigger_names", []))
        }.intersection({"regime_shift", "transition_regime", "high_vol_regime"})
    ]
    base_retraining_rate = float(
        sum(bool(run.get("base_should_retrain", False)) for run in runs) / max(len(runs), 1)
    )
    retraining_rate = float(
        sum(bool(run.get("should_retrain", False)) for run in runs) / max(len(runs), 1)
    )
    distribution_summary: dict[str, object] = {
        "information_ratio": _distribution([float(cast(float, run["information_ratio"])) for run in runs]),
        "pbo": _distribution([float(cast(float, run["pbo"])) for run in runs]),
        "cluster_adjusted_pbo": _distribution(
            [float(cast(float, run.get("cluster_adjusted_pbo", run["pbo"]))) for run in runs]
        ),
        "selection_stability": _distribution([float(cast(float, run["selection_stability"])) for run in runs]),
        "news_feature_coverage": _distribution(
            [float(cast(float, run.get("news_feature_coverage", 0.0))) for run in runs]
        ),
        "news_feature_missing_rate": _distribution(
            [float(cast(float, run.get("news_feature_missing_rate", 1.0))) for run in runs]
        ),
        "news_feature_staleness": _distribution(
            [float(cast(float, run.get("news_feature_stale_rate", 0.0))) for run in runs]
        ),
        "news_used_source_counts": _value_counts(runs, "news_used_source"),
        "news_transport_origin_counts": _value_counts(runs, "news_transport_origin"),
        "news_fallback_rate": float(sum(bool(run.get("news_fallback_used", False)) for run in runs) / max(len(runs), 1)),
        "news_coverage_analysis": _news_coverage_summary(runs),
        "news_utility_comparison": _news_coverage_summary(runs),
        "base_retraining_rate": base_retraining_rate,
        "retraining_rate": retraining_rate,
        "base_trigger_counts": _trigger_counts(runs, "base_trigger_names"),
        "effective_trigger_counts": _trigger_counts(runs, "trigger_names"),
        "policy_decision_counts": _policy_decision_counts(runs),
        "pbo_competition_dominated_rate": float(
            sum(bool(run.get("pbo_competition_dominated", False)) for run in runs) / max(len(runs), 1)
        ),
        "candidate_pbo_competition_dominated_rate": float(
            sum(bool(run.get("candidate_pbo_competition_dominated", False)) for run in runs) / max(len(runs), 1)
        ),
        "pbo_dominant_axis_counts": _value_counts_when_true(
            runs,
            field="pbo_dominant_axis",
            condition_field="pbo_competition_dominated",
        ),
        "candidate_pbo_dominant_axis_counts": _value_counts_when_true(
            runs,
            field="candidate_pbo_dominant_axis",
            condition_field="candidate_pbo_competition_dominated",
        ),
        "drift_dominated_analysis": {
            "raw_run_count": len(drift_dominated_runs),
            "raw_rate": float(len(drift_dominated_runs) / max(len(runs), 1)),
            "effective_run_count": len(effective_drift_runs),
            "effective_rate": float(len(effective_drift_runs) / max(len(runs), 1)),
            "effective_drift_only_run_count": int(
                sum(
                    set(cast(list[object], run.get("trigger_names", []))).difference({"feature_drift"}) == set()
                    for run in effective_drift_runs
                )
            ),
            "effective_non_drift_co_trigger_counts": _list_item_counts(
                [
                    {
                        "labels": [
                            str(name)
                            for name in cast(list[object], run.get("trigger_names", []))
                            if str(name) != "feature_drift"
                        ]
                    }
                    for run in effective_drift_runs
                ],
                "labels",
            ),
            "trigger_family_counts": _list_item_counts(drift_dominated_runs, "drift_pre_suppression_trigger_families"),
            "effective_trigger_family_counts": _list_item_counts(effective_drift_runs, "drift_trigger_families"),
            "current_regime_counts": _value_counts(drift_dominated_runs, "current_regime"),
            "dominant_recent_regime_counts": _value_counts(drift_dominated_runs, "dominant_recent_regime"),
            "proxy_sensitive_profile_counts": _value_counts(drift_dominated_runs, "drift_proxy_sensitive_profile"),
            "ohlcv_source_counts": _value_counts(drift_dominated_runs, "ohlcv_source"),
            "ohlcv_transport_origin_counts": _value_counts(drift_dominated_runs, "ohlcv_transport_origin"),
            "macro_transport_origin_counts": _value_counts(drift_dominated_runs, "macro_transport_origin"),
            "history_matches": _distribution(
                [float(cast(float, run.get("drift_history_matches", 0))) for run in drift_dominated_runs]
            ),
            "persistence_span_business_days": _distribution(
                [float(cast(float, run.get("drift_span_business_days", 0))) for run in drift_dominated_runs]
            ),
            "family_persistence_relief_count": int(
                sum(bool(run.get("drift_family_persistence_would_suppress", False)) for run in drift_dominated_runs)
            ),
            "severity_threshold_low_vol_relief_count": int(
                sum(bool(run.get("drift_low_vol_threshold_would_suppress", False)) for run in drift_dominated_runs)
            ),
            "regime_suppression_relief_count": int(
                sum(bool(run.get("drift_stable_transition_suppression_would_suppress", False)) for run in drift_dominated_runs)
            ),
            "family_persistence_effective_relief_count": _effective_relief_count(
                effective_drift_runs,
                relief_field="drift_family_persistence_would_suppress",
            ),
            "severity_threshold_low_vol_effective_relief_count": _effective_relief_count(
                effective_drift_runs,
                relief_field="drift_low_vol_threshold_would_suppress",
            ),
            "regime_suppression_effective_relief_count": _effective_relief_count(
                effective_drift_runs,
                relief_field="drift_stable_transition_suppression_would_suppress",
            ),
            "threshold_delta_to_suppress": _distribution(
                [float(cast(float, run.get("drift_threshold_delta_to_suppress", 0.0))) for run in drift_dominated_runs]
            ),
            "stable_transition_suppressed_family_counts": _list_item_counts(
                drift_dominated_runs,
                "drift_stable_transition_suppressed_families",
            ),
        },
        "regime_dominated_analysis": {
            "raw_run_count": len(regime_dominated_runs),
            "raw_rate": float(len(regime_dominated_runs) / max(len(runs), 1)),
            "effective_run_count": len(effective_regime_runs),
            "effective_rate": float(len(effective_regime_runs) / max(len(runs), 1)),
            "base_trigger_counts": _list_item_counts(
                [
                    {
                        "labels": [
                            str(name)
                            for name in cast(list[object], run.get("base_trigger_names", []))
                            if str(name) in {"regime_shift", "transition_regime", "high_vol_regime"}
                        ]
                    }
                    for run in regime_dominated_runs
                ],
                "labels",
            ),
            "effective_trigger_counts": _list_item_counts(
                [
                    {
                        "labels": [
                            str(name)
                            for name in cast(list[object], run.get("trigger_names", []))
                            if str(name) in {"regime_shift", "transition_regime", "high_vol_regime"}
                        ]
                    }
                    for run in effective_regime_runs
                ],
                "labels",
            ),
            "current_regime_counts": _value_counts(regime_dominated_runs, "current_regime"),
            "dominant_recent_regime_counts": _value_counts(regime_dominated_runs, "dominant_recent_regime"),
            "transition_profile_counts": _value_counts(regime_dominated_runs, "transition_profile"),
            "state_probability_bucket_counts": _value_counts(regime_dominated_runs, "state_probability_bucket"),
            "transition_rate_bucket_counts": _value_counts(regime_dominated_runs, "transition_rate_bucket"),
            "shift_flag_counts": _value_counts(
                [
                    {"label": "shift" if bool(run.get("regime_shift_flag", False)) else "no_shift"}
                    for run in regime_dominated_runs
                ],
                "label",
            ),
            "transition_history_matches": _distribution(
                [float(cast(float, run.get("transition_history_matches", 0))) for run in regime_dominated_runs]
            ),
            "transition_observation_count": _distribution(
                [float(cast(float, run.get("transition_observation_count", 0))) for run in regime_dominated_runs]
            ),
            "transition_persistence_span_business_days": _distribution(
                [float(cast(float, run.get("transition_span_business_days", 0))) for run in regime_dominated_runs]
            ),
            "co_trigger_family_counts": _list_item_counts(
                [
                    {
                        "labels": [
                            str(family)
                            for family in cast(list[object], run.get("drift_trigger_families", []))
                        ]
                    }
                    for run in regime_dominated_runs
                    if "feature_drift" in cast(list[object], run.get("base_trigger_names", []))
                ],
                "labels",
            ),
            "effective_co_trigger_family_counts": _list_item_counts(
                [
                    {
                        "labels": [
                            str(family)
                            for family in cast(list[object], run.get("drift_trigger_families", []))
                        ]
                    }
                    for run in effective_regime_runs
                    if "feature_drift" in cast(list[object], run.get("trigger_names", []))
                ],
                "labels",
            ),
            "ohlcv_source_counts": _value_counts(regime_dominated_runs, "ohlcv_source"),
            "ohlcv_transport_origin_counts": _value_counts(regime_dominated_runs, "ohlcv_transport_origin"),
            "macro_transport_origin_counts": _value_counts(regime_dominated_runs, "macro_transport_origin"),
        },
        "calibration_dominated_analysis": {
            "run_count": len(calibration_dominated_runs),
            "rate": float(len(calibration_dominated_runs) / max(len(runs), 1)),
            "trigger_metric_counts": _value_counts(
                [
                    {"label": _calibration_trigger_combo(run)}
                    for run in calibration_dominated_runs
                ],
                "label",
            ),
            "fold_persistence_counts": _value_counts(
                [
                    {
                        "label": _calibration_combo_label(
                            ece=bool(run.get("ece_fold_persistent", False)),
                            gap=bool(run.get("calibration_gap_fold_persistent", False)),
                        )
                    }
                    for run in calibration_dominated_runs
                ],
                "label",
            ),
            "run_persistence_counts": _value_counts(
                [
                    {
                        "label": _calibration_combo_label(
                            ece=bool(run.get("ece_run_persistent", False)),
                            gap=bool(run.get("calibration_gap_run_persistent", False)),
                        )
                    }
                    for run in calibration_dominated_runs
                ],
                "label",
            ),
            "current_regime_counts": _value_counts(calibration_dominated_runs, "current_regime"),
            "dominant_recent_regime_counts": _value_counts(calibration_dominated_runs, "dominant_recent_regime"),
            "ticker_set_counts": _value_counts(calibration_dominated_runs, "ticker_set"),
            "ohlcv_source_counts": _value_counts(calibration_dominated_runs, "ohlcv_source"),
            "ohlcv_transport_origin_counts": _value_counts(calibration_dominated_runs, "ohlcv_transport_origin"),
            "macro_transport_origin_counts": _value_counts(calibration_dominated_runs, "macro_transport_origin"),
            "ece_history_matches": _distribution(
                [float(cast(float, run.get("ece_history_matches", 0))) for run in calibration_dominated_runs]
            ),
            "calibration_gap_history_matches": _distribution(
                [
                    float(cast(float, run.get("calibration_gap_history_matches", 0)))
                    for run in calibration_dominated_runs
                ]
            ),
            "ece_span_business_days": _distribution(
                [float(cast(float, run.get("ece_span_business_days", 0))) for run in calibration_dominated_runs]
            ),
            "calibration_gap_span_business_days": _distribution(
                [
                    float(cast(float, run.get("calibration_gap_span_business_days", 0)))
                    for run in calibration_dominated_runs
                ]
            ),
        },
    }
    by_ticker_set: list[dict[str, object]] = []
    for ticker_set in ticker_sets:
        label = ",".join(ticker_set)
        subset = [run for run in runs if str(run.get("ticker_set")) == label]
        by_ticker_set.append(
            {
                "ticker_set": label,
                "run_count": len(subset),
                "information_ratio": _distribution([float(cast(float, run["information_ratio"])) for run in subset]),
                "pbo": _distribution([float(cast(float, run["pbo"])) for run in subset]),
                "cluster_adjusted_pbo": _distribution(
                    [float(cast(float, run.get("cluster_adjusted_pbo", run["pbo"]))) for run in subset]
                ),
                "selection_stability": _distribution(
                    [float(cast(float, run["selection_stability"])) for run in subset]
                ),
                "base_retraining_rate": float(
                    sum(bool(run.get("base_should_retrain", False)) for run in subset) / max(len(subset), 1)
                ),
                "retraining_rate": float(
                    sum(bool(run.get("should_retrain", False)) for run in subset) / max(len(subset), 1)
                ),
                "news_feature_coverage": _distribution(
                    [float(cast(float, run.get("news_feature_coverage", 0.0))) for run in subset]
                ),
                "news_feature_missing_rate": _distribution(
                    [float(cast(float, run.get("news_feature_missing_rate", 1.0))) for run in subset]
                ),
                "news_feature_staleness": _distribution(
                    [float(cast(float, run.get("news_feature_stale_rate", 0.0))) for run in subset]
                ),
                "news_used_source_counts": _value_counts(subset, "news_used_source"),
                "news_transport_origin_counts": _value_counts(subset, "news_transport_origin"),
                "news_fallback_rate": float(
                    sum(bool(run.get("news_fallback_used", False)) for run in subset) / max(len(subset), 1)
                ),
                "news_coverage_analysis": _news_coverage_summary(subset),
                "news_utility_comparison": _news_coverage_summary(subset),
            }
        )
    by_as_of_date: list[dict[str, object]] = []
    for as_of_date in as_of_dates:
        subset = [run for run in runs if str(run.get("as_of_date")) == as_of_date]
        by_as_of_date.append(
            {
                "as_of_date": as_of_date,
                "run_count": len(subset),
                "information_ratio": _distribution([float(cast(float, run["information_ratio"])) for run in subset]),
                "pbo": _distribution([float(cast(float, run["pbo"])) for run in subset]),
                "cluster_adjusted_pbo": _distribution(
                    [float(cast(float, run.get("cluster_adjusted_pbo", run["pbo"]))) for run in subset]
                ),
                "selection_stability": _distribution(
                    [float(cast(float, run["selection_stability"])) for run in subset]
                ),
                "base_retraining_rate": float(
                    sum(bool(run.get("base_should_retrain", False)) for run in subset) / max(len(subset), 1)
                ),
                "retraining_rate": float(
                    sum(bool(run.get("should_retrain", False)) for run in subset) / max(len(subset), 1)
                ),
                "news_feature_coverage": _distribution(
                    [float(cast(float, run.get("news_feature_coverage", 0.0))) for run in subset]
                ),
                "news_feature_missing_rate": _distribution(
                    [float(cast(float, run.get("news_feature_missing_rate", 1.0))) for run in subset]
                ),
                "news_feature_staleness": _distribution(
                    [float(cast(float, run.get("news_feature_stale_rate", 0.0))) for run in subset]
                ),
                "news_used_source_counts": _value_counts(subset, "news_used_source"),
                "news_transport_origin_counts": _value_counts(subset, "news_transport_origin"),
                "news_fallback_rate": float(
                    sum(bool(run.get("news_fallback_used", False)) for run in subset) / max(len(subset), 1)
                ),
                "news_coverage_analysis": _news_coverage_summary(subset),
                "news_utility_comparison": _news_coverage_summary(subset),
            }
        )
    by_news_transport_origin = _group_distribution_by_value(
        runs,
        field="news_feature_coverage",
        group_field="news_transport_origin",
    )
    for entry in by_news_transport_origin:
        origin = str(entry.get("news_transport_origin", ""))
        entry["news_feature_missing_rate"] = _subset_distribution(
            runs,
            field="news_feature_missing_rate",
            match_field="news_transport_origin",
            match_value=origin,
        )
        entry["news_feature_staleness"] = _subset_distribution(
            runs,
            field="news_feature_stale_rate",
            match_field="news_transport_origin",
            match_value=origin,
        )
        entry["news_used_source_counts"] = _value_counts(
            [run for run in runs if str(run.get("news_transport_origin", "")) == origin],
            "news_used_source",
        )
        entry["news_coverage_analysis"] = _news_coverage_summary(
            [run for run in runs if str(run.get("news_transport_origin", "")) == origin]
        )
        entry["news_utility_comparison"] = entry["news_coverage_analysis"]
    by_news_used_source = _group_distribution_by_value(
        runs,
        field="news_feature_coverage",
        group_field="news_used_source",
    )
    for entry in by_news_used_source:
        source_name = str(entry.get("news_used_source", ""))
        subset = [run for run in runs if str(run.get("news_used_source", "")) == source_name]
        entry["news_feature_missing_rate"] = _subset_distribution(
            runs,
            field="news_feature_missing_rate",
            match_field="news_used_source",
            match_value=source_name,
        )
        entry["news_feature_staleness"] = _subset_distribution(
            runs,
            field="news_feature_stale_rate",
            match_field="news_used_source",
            match_value=source_name,
        )
        entry["news_transport_origin_counts"] = _value_counts(subset, "news_transport_origin")
        entry["news_fallback_rate"] = float(
            sum(bool(run.get("news_fallback_used", False)) for run in subset) / max(len(subset), 1)
        )
        entry["news_coverage_analysis"] = _news_coverage_summary(subset)
        entry["news_utility_comparison"] = entry["news_coverage_analysis"]
    payload: dict[str, object] = {
        "suite_id": str(uuid4()),
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "dataset_type": "public_real_market",
        "profile_name": profile_name,
        "profile_role": profile_role,
        "analysis_mode": analysis_mode,
        "source_suite_id": source_suite_id,
        "source_suite_path": source_suite_path,
        "as_of_dates": as_of_dates,
        "ticker_sets": ticker_sets,
        "run_count": len(runs),
        "cpcv_max_splits": cpcv_max_splits,
        "runs": runs,
        "distribution_summary": distribution_summary,
        "by_ticker_set": by_ticker_set,
        "by_as_of_date": by_as_of_date,
        "by_news_transport_origin": by_news_transport_origin,
        "by_news_used_source": by_news_used_source,
    }
    return payload


def load_public_audit_suite(path: str | Path) -> dict[str, object]:
    target = Path(path)
    return cast(dict[str, object], json.loads(target.read_text(encoding="utf-8")))


def replay_public_audit_suite(
    *,
    settings: Settings,
    source_payload: dict[str, object],
    source_suite_path: str | None = None,
    profile_name: str | None = None,
) -> dict[str, object]:
    storage_root = resolve_storage_path(settings)
    source_runs = cast(list[dict[str, object]], source_payload.get("runs", []))
    sorted_runs = sorted(source_runs, key=lambda item: (str(item.get("as_of_date", "")), str(item.get("ticker_set", ""))))
    ticker_sets = [
        [str(ticker) for ticker in cast(list[object], ticker_group)]
        for ticker_group in cast(list[list[object]], source_payload.get("ticker_sets", []))
    ]
    history_by_ticker_set: dict[str, list[dict[str, object]]] = {
        ",".join(ticker_set): [] for ticker_set in ticker_sets
    }
    replay_runs: list[dict[str, object]] = []
    for source_run in sorted_runs:
        backtest_id = str(source_run["backtest_id"])
        backtest_path = storage_root / "outputs" / "backtests" / f"{backtest_id}.json"
        backtest_payload = cast(dict[str, object], json.loads(backtest_path.read_text(encoding="utf-8")))
        ticker_label = str(source_run["ticker_set"])
        tickers = [item.strip() for item in ticker_label.split(",") if item.strip()]
        cpcv_result = cast(dict[str, object], backtest_payload.get("cpcv", {}))
        retraining_monitor = build_retraining_monitor(
            aggregate_metrics=cast(dict[str, float], backtest_payload["aggregate_metrics"]),
            drift_summary=cast(dict[str, object], backtest_payload.get("drift_monitor", {})),
            regime_summary=cast(dict[str, object], backtest_payload.get("regime_monitor", {})),
            pbo=cast(float | None, cpcv_result.get("cluster_adjusted_pbo", backtest_payload.get("pbo"))),
            pbo_summary=cast(
                dict[str, object],
                cpcv_result.get("cluster_adjusted_pbo_summary", cast(dict[str, object], cpcv_result.get("pbo_summary", {}))),
            ),
            pbo_diagnostics=cast(
                dict[str, object] | None,
                cpcv_result.get("cluster_adjusted_pbo_diagnostics", cpcv_result.get("pbo_diagnostics")),
            ),
            candidate_level_pbo=cast(float | None, backtest_payload.get("pbo")),
            candidate_level_pbo_summary=cast(dict[str, object] | None, cpcv_result.get("pbo_summary")),
            candidate_level_pbo_diagnostics=cast(dict[str, object] | None, cpcv_result.get("pbo_diagnostics")),
            settings=settings,
            policy_context={
                "as_of_date": str(source_run["as_of_date"]),
                "history": list(history_by_ticker_set.setdefault(ticker_label, [])),
            },
        )
        replay_run = dict(source_run)
        replay_run["base_should_retrain"] = bool(retraining_monitor.get("base_should_retrain", False))
        replay_run["should_retrain"] = bool(retraining_monitor.get("should_retrain", False))
        replay_run["policy_decision"] = str(retraining_monitor.get("policy_decision", "watch_only"))
        replay_run["trigger_names"] = [str(name) for name in cast(list[object], retraining_monitor.get("effective_trigger_names", []))]
        replay_run["base_trigger_names"] = [str(name) for name in cast(list[object], retraining_monitor.get("base_trigger_names", []))]
        replay_run["suppressed_trigger_names"] = [
            str(name) for name in cast(list[object], retraining_monitor.get("suppressed_trigger_names", []))
        ]
        replay_run["policy_notes"] = [str(note) for note in cast(list[object], retraining_monitor.get("policy_notes", []))]
        replay_runs.append(replay_run)
        history_by_ticker_set[ticker_label].append(
            build_retraining_history_entry(
                as_of_date=str(source_run["as_of_date"]),
                retraining_monitor=retraining_monitor,
                regime_summary=cast(dict[str, object], backtest_payload.get("regime_monitor", {})),
                tickers=tickers,
                source_mode="live",
                pbo=cast(float | None, backtest_payload.get("pbo")),
                pbo_summary=cast(
                    dict[str, object],
                    cast(dict[str, object], cast(dict[str, object], backtest_payload.get("cpcv", {})).get("pbo_summary", {})),
                ),
            )
        )
    payload = build_public_audit_suite(
        runs=replay_runs,
        as_of_dates=[str(value) for value in cast(list[object], source_payload.get("as_of_dates", []))],
        ticker_sets=ticker_sets,
        cpcv_max_splits=_as_int(source_payload.get("cpcv_max_splits", settings.model_settings.cpcv.max_splits)),
        profile_name=profile_name or cast(str | None, source_payload.get("profile_name")),
        profile_role=cast(str | None, source_payload.get("profile_role"))
        or profile_role_for_name(profile_name or cast(str | None, source_payload.get("profile_name"))),
        analysis_mode="retraining_policy_replay",
        source_suite_id=cast(str | None, source_payload.get("suite_id")),
        source_suite_path=source_suite_path,
    )
    source_distribution = cast(dict[str, object], source_payload.get("distribution_summary", {}))
    replay_distribution = cast(dict[str, object], payload["distribution_summary"])
    payload["comparison_to_source"] = {
        "source_base_retraining_rate": _as_float(source_distribution.get("base_retraining_rate", 0.0) or 0.0),
        "source_retraining_rate": _as_float(source_distribution.get("retraining_rate", 0.0) or 0.0),
        "replayed_base_retraining_rate": _as_float(replay_distribution.get("base_retraining_rate", 0.0) or 0.0),
        "replayed_retraining_rate": _as_float(replay_distribution.get("retraining_rate", 0.0) or 0.0),
        "base_retraining_rate_delta": _as_float(replay_distribution.get("base_retraining_rate", 0.0) or 0.0)
        - _as_float(source_distribution.get("base_retraining_rate", 0.0) or 0.0),
        "retraining_rate_delta": _as_float(replay_distribution.get("retraining_rate", 0.0) or 0.0)
        - _as_float(source_distribution.get("retraining_rate", 0.0) or 0.0),
    }
    return payload


def persist_public_audit_suite(settings: Settings, payload: dict[str, Any]) -> Path:
    storage_root = resolve_storage_path(settings)
    dataset_type = str(payload.get("dataset_type", "public_real_market"))
    generated_at = str(payload.get("generated_at", ""))[:10] or "unknown"
    target = (
        storage_root
        / "outputs"
        / "monitor_audit_suites"
        / dataset_type
        / generated_at
        / f"{payload['suite_id']}.json"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target

