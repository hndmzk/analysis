from __future__ import annotations

from collections import Counter
import json
from typing import cast

import numpy as np
import pandas as pd

from market_prediction_agent.config import Settings
from market_prediction_agent.evaluation.drift import feature_family


def _trigger(name: str, status: str, detail: str) -> dict[str, str]:
    return {"name": name, "status": status, "detail": detail}


def _regime_bucket(regime: str) -> str:
    if regime in {"high_vol", "transition"}:
        return "stress"
    if regime in {"low_vol", "unknown"}:
        return regime
    return "unknown"


def _threshold_regime_key(regime: str) -> str:
    return regime if regime in {"low_vol", "transition", "high_vol"} else "unknown"


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return float(cast(float, value))
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return int(cast(int, value))
    except (TypeError, ValueError):
        return default


def _normalized_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        return [str(item) for item in cast(list[object], parsed)]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in cast(list[object], list(value))]
    return [str(value)]


def _ticker_set_label(tickers: list[str] | None) -> str:
    if not tickers:
        return ""
    return ",".join(sorted({str(ticker).upper() for ticker in tickers}))


def _business_day_distance(previous_date: str, current_date: str) -> int:
    previous = pd.Timestamp(previous_date).date()
    current = pd.Timestamp(current_date).date()
    if current <= previous:
        return 0
    return int(np.busday_count(previous.isoformat(), current.isoformat()))


def _current_regime(regime_summary: dict[str, object]) -> str:
    return str(regime_summary.get("current_regime", "unknown"))


def _suppressed_families(settings: Settings, current_regime: str) -> set[str]:
    regime_key = _threshold_regime_key(current_regime)
    return {str(item) for item in settings.model_settings.retraining.family_retrain_suppression.get(regime_key, [])}


def _transition_history_metrics(
    *,
    history: list[dict[str, object]],
    as_of_date: str,
    persistence_business_days: int,
) -> tuple[int, int]:
    relevant_dates: list[str] = []
    for entry in history:
        entry_date = str(entry.get("as_of_date", ""))
        if not entry_date:
            continue
        distance = _business_day_distance(entry_date, as_of_date)
        if distance > persistence_business_days:
            continue
        if str(entry.get("current_regime", "")) == "transition":
            relevant_dates.append(entry_date)
    if not relevant_dates:
        return 0, 0
    earliest_date = min(relevant_dates)
    span_business_days = _business_day_distance(earliest_date, as_of_date)
    return len(relevant_dates), span_business_days


def _transition_state_probability_bucket(
    value: float,
    settings: Settings,
) -> str:
    if value < settings.model_settings.retraining.transition_regime_immediate_state_probability_threshold:
        return "very_low"
    if value < settings.model_settings.retraining.transition_regime_unstable_state_probability_threshold:
        return "low"
    if value < settings.model_settings.retraining.transition_regime_state_probability_threshold:
        return "medium"
    return "stable"


def _transition_rate_bucket(
    value: float,
    settings: Settings,
) -> str:
    if value <= 0.0:
        return "zero"
    if value <= settings.model_settings.retraining.transition_regime_transition_rate_threshold:
        return "low"
    if value <= settings.model_settings.retraining.transition_regime_immediate_transition_rate_threshold:
        return "medium"
    return "high"


def _transition_profile(
    *,
    regime_summary: dict[str, object],
    settings: Settings,
    policy_context: dict[str, object],
) -> dict[str, object]:
    current_regime = _current_regime(regime_summary)
    state_probability = _safe_float(regime_summary.get("state_probability"), 1.0)
    transition_rate = _safe_float(regime_summary.get("transition_rate"), 0.0)
    if current_regime != "transition":
        return {
            "is_transition": False,
            "classification": "not_transition",
            "history_matches": 0,
            "observation_count": 0,
            "span_business_days": 0,
            "persistent": False,
            "stable_transition": False,
            "unstable_transition": False,
            "immediate_transition": False,
            "state_probability_bucket": _transition_state_probability_bucket(state_probability, settings),
            "transition_rate_bucket": _transition_rate_bucket(transition_rate, settings),
        }
    history = cast(list[dict[str, object]], policy_context.get("history", []))
    as_of_date = str(policy_context.get("as_of_date", pd.Timestamp.now(tz="UTC").date().isoformat()))
    history_matches, span_business_days = _transition_history_metrics(
        history=history,
        as_of_date=as_of_date,
        persistence_business_days=int(settings.model_settings.retraining.transition_regime_persistence_business_days),
    )
    observation_count = history_matches + 1
    persistent = (
        observation_count >= settings.model_settings.retraining.transition_regime_min_persistent_runs
        and span_business_days >= settings.model_settings.retraining.transition_regime_min_persistent_span_business_days
    )
    stable_transition = (
        transition_rate <= settings.model_settings.retraining.transition_regime_transition_rate_threshold
        and (
            state_probability >= settings.model_settings.retraining.transition_regime_state_probability_threshold
            or persistent
        )
    )
    unstable_transition = (
        state_probability < settings.model_settings.retraining.transition_regime_unstable_state_probability_threshold
        or transition_rate > settings.model_settings.retraining.transition_regime_unstable_transition_rate_threshold
    )
    immediate_transition = (
        state_probability < settings.model_settings.retraining.transition_regime_immediate_state_probability_threshold
        or transition_rate > settings.model_settings.retraining.transition_regime_immediate_transition_rate_threshold
    )
    classification = "watch_transition"
    if stable_transition:
        classification = "stable_transition"
    elif unstable_transition:
        classification = "unstable_transition"
    return {
        "is_transition": True,
        "classification": classification,
        "history_matches": history_matches,
        "observation_count": observation_count,
        "span_business_days": span_business_days,
        "persistent": persistent,
        "stable_transition": stable_transition,
        "unstable_transition": unstable_transition,
        "immediate_transition": immediate_transition,
        "state_probability_bucket": _transition_state_probability_bucket(state_probability, settings),
        "transition_rate_bucket": _transition_rate_bucket(transition_rate, settings),
    }


def _stable_transition_suppressed_families(
    settings: Settings,
    transition_profile: dict[str, object],
) -> set[str]:
    if bool(transition_profile.get("stable_transition", False)):
        return {
            str(item)
            for item in settings.model_settings.retraining.stable_transition_family_retrain_suppression
        }
    return set()


def _drift_family_weights(settings: Settings) -> dict[str, float]:
    weights = settings.risk.drift.family_trigger_weights
    return {
        "price_momentum": float(weights.price_momentum),
        "volatility": float(weights.volatility),
        "volume": float(weights.volume),
        "macro": float(weights.macro),
        "calendar": float(weights.calendar),
    }


def _drift_signal(
    drift_summary: dict[str, object],
    regime_summary: dict[str, object],
    settings: Settings,
    policy_context: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, str]], list[dict[str, str]], list[str]]:
    supplementary = cast(dict[str, object], drift_summary.get("supplementary_analysis", {}))
    diagnostics = cast(list[dict[str, object]], supplementary.get("feature_diagnostics", []))
    weights = _drift_family_weights(settings)
    current_regime = _current_regime(regime_summary)
    history = cast(list[dict[str, object]], policy_context.get("history", []))
    as_of_date = str(policy_context.get("as_of_date", pd.Timestamp.now(tz="UTC").date().isoformat()))
    regime_key = _threshold_regime_key(current_regime)
    regime_multiplier = float(settings.model_settings.retraining.drift_threshold_multipliers.get(regime_key, 1.0))
    immediate_multiplier = float(
        settings.model_settings.retraining.drift_immediate_threshold_multipliers.get(regime_key, 1.0)
    )
    transition_profile = _transition_profile(
        regime_summary=regime_summary,
        settings=settings,
        policy_context={"history": history, "as_of_date": as_of_date},
    )
    stable_transition_suppressed = _stable_transition_suppressed_families(settings, transition_profile)
    suppressed_families = _suppressed_families(settings, current_regime).union(stable_transition_suppressed)
    family_scores: dict[str, float] = {}
    family_feature_counts: Counter[str] = Counter()
    raw_family_scores: dict[str, float] = {}
    raw_family_feature_counts: Counter[str] = Counter()
    suppressed_feature_count = 0
    trigger_features: list[str] = []
    trigger_families: set[str] = set()
    raw_trigger_features: list[str] = []
    raw_trigger_families: set[str] = set()
    proxy_sensitive_trigger_feature_count = 0
    non_proxy_trigger_feature_count = 0
    raw_proxy_sensitive_trigger_feature_count = 0
    raw_non_proxy_trigger_feature_count = 0
    observations: list[dict[str, str]] = []
    cause_keys: list[str] = []
    raw_cause_keys: list[str] = []

    for entry in diagnostics:
        status = str(entry.get("status", "PASS"))
        action = str(entry.get("retrain_action", "ignore"))
        if status not in {"WARNING", "FAIL"} or action != "trigger":
            continue
        family = str(entry.get("family") or feature_family(str(entry.get("feature", ""))))
        feature = str(entry.get("feature", "unknown"))
        proxy_sensitive = bool(entry.get("proxy_sensitive", False))
        raw_trigger_features.append(feature)
        raw_trigger_families.add(family)
        raw_family_feature_counts[family] += 1
        raw_family_scores[family] = float(weights.get(family, 0.0))
        raw_cause_keys.append(f"feature_drift:{family}:{_regime_bucket(current_regime)}")
        if proxy_sensitive:
            raw_proxy_sensitive_trigger_feature_count += 1
        else:
            raw_non_proxy_trigger_feature_count += 1
        if family in suppressed_families:
            suppressed_feature_count += 1
            continue
        if proxy_sensitive:
            proxy_sensitive_trigger_feature_count += 1
        else:
            non_proxy_trigger_feature_count += 1
        trigger_features.append(feature)
        trigger_families.add(family)
        family_feature_counts[family] += 1
        family_scores[family] = float(weights.get(family, 0.0))
        cause_keys.append(f"feature_drift:{family}:{_regime_bucket(current_regime)}")

    weighted_score = float(sum(family_scores.values()))
    pre_suppression_weighted_score = float(sum(raw_family_scores.values()))
    weighted_threshold = float(settings.model_settings.retraining.drift_family_weight_threshold * regime_multiplier)
    low_vol_threshold = float(
        settings.model_settings.retraining.drift_family_weight_threshold
        * settings.model_settings.retraining.drift_threshold_multipliers.get("low_vol", 1.0)
    )
    immediate_threshold = float(settings.model_settings.retraining.drift_immediate_weight_threshold * immediate_multiplier)
    family_persistence_min_feature_count = 2
    family_persistence_counterfactual_score = float(
        sum(
            float(weights.get(family, 0.0))
            for family, count in raw_family_feature_counts.items()
            if count >= family_persistence_min_feature_count
        )
    )
    family_persistence_would_suppress = family_persistence_counterfactual_score < weighted_threshold
    stable_transition_counterfactual_score = float(
        sum(
            float(weights.get(family, 0.0))
            for family, count in raw_family_feature_counts.items()
            if family not in stable_transition_suppressed
        )
    )
    stable_transition_suppression_would_suppress = (
        bool(stable_transition_suppressed) and stable_transition_counterfactual_score < weighted_threshold
    )
    threshold_delta_to_suppress = float(max(pre_suppression_weighted_score - weighted_threshold, 0.0))
    low_vol_threshold_would_suppress = pre_suppression_weighted_score < low_vol_threshold
    persistent = (
        len(trigger_features) >= settings.model_settings.retraining.drift_min_persistent_features
        and len(trigger_families) >= settings.model_settings.retraining.drift_min_persistent_families
    )
    triggered = persistent and weighted_score >= weighted_threshold
    immediate = weighted_score >= immediate_threshold and len(trigger_families) >= 1
    history_matches, span_business_days = _history_cause_matches(
        history=history,
        as_of_date=as_of_date,
        cause_keys=sorted(set(raw_cause_keys)),
        persistence_business_days=int(settings.model_settings.retraining.cause_cooloff_business_days),
    )

    if supplementary.get("primary_cause") == "proxy_artifact_likely" and not trigger_features:
        observations.append(
            _trigger(
                "feature_drift_proxy_watch",
                "INFO",
                "Only proxy-sensitive drift features were flagged; retraining remains in watch mode.",
            )
        )
    elif trigger_features and not persistent:
        observations.append(
            _trigger(
                "feature_drift_persistence_watch",
                "INFO",
                "Drift exceeded thresholds but did not meet persistent feature/family counts.",
            )
        )
    if suppressed_feature_count > 0:
        observations.append(
            _trigger(
                "feature_drift_family_suppression_watch",
                "INFO",
                f"Suppressed families for regime {current_regime}: {sorted(suppressed_families)}.",
            )
        )
    if stable_transition_suppressed:
        observations.append(
            _trigger(
                "feature_drift_stable_transition_suppression_watch",
                "INFO",
                (
                    "Stable transition regime applied targeted family suppression to reduce single-family noise: "
                    f"{sorted(stable_transition_suppressed)}."
                ),
            )
        )

    triggers: list[dict[str, str]] = []
    if triggered or immediate:
        triggers.append(
            _trigger(
                "feature_drift",
                "WARNING" if not immediate else "FAIL",
                (
                    f"Weighted drift score {weighted_score:.2f} across families {sorted(trigger_families)} "
                    f"(threshold={weighted_threshold:.2f}, immediate={immediate_threshold:.2f})."
                ),
            )
        )

    signal = {
        "weighted_score": weighted_score,
        "pre_suppression_weighted_score": pre_suppression_weighted_score,
        "weighted_threshold": weighted_threshold,
        "low_vol_threshold": low_vol_threshold,
        "immediate_threshold": immediate_threshold,
        "trigger_feature_count": len(trigger_features),
        "trigger_family_count": len(trigger_families),
        "pre_suppression_trigger_feature_count": len(raw_trigger_features),
        "pre_suppression_trigger_family_count": len(raw_trigger_families),
        "persistent": persistent,
        "triggered": bool(triggered or immediate),
        "family_scores": family_scores,
        "family_feature_counts": dict(family_feature_counts),
        "trigger_families": sorted(trigger_families),
        "pre_suppression_family_scores": raw_family_scores,
        "pre_suppression_family_feature_counts": dict(raw_family_feature_counts),
        "pre_suppression_trigger_families": sorted(raw_trigger_families),
        "suppressed_families": sorted(suppressed_families),
        "stable_transition_suppressed_families": sorted(stable_transition_suppressed),
        "suppressed_feature_count": suppressed_feature_count,
        "proxy_sensitive_trigger_feature_count": proxy_sensitive_trigger_feature_count,
        "non_proxy_trigger_feature_count": non_proxy_trigger_feature_count,
        "pre_suppression_proxy_sensitive_trigger_feature_count": raw_proxy_sensitive_trigger_feature_count,
        "pre_suppression_non_proxy_trigger_feature_count": raw_non_proxy_trigger_feature_count,
        "proxy_sensitive_profile": (
            "mixed"
            if raw_proxy_sensitive_trigger_feature_count and raw_non_proxy_trigger_feature_count
            else "proxy_only"
            if raw_proxy_sensitive_trigger_feature_count
            else "non_proxy_only"
            if raw_non_proxy_trigger_feature_count
            else "none"
        ),
        "history_matches": history_matches,
        "span_business_days": span_business_days,
        "family_persistence_min_feature_count": family_persistence_min_feature_count,
        "family_persistence_counterfactual_score": family_persistence_counterfactual_score,
        "family_persistence_would_suppress": family_persistence_would_suppress,
        "stable_transition_counterfactual_score": stable_transition_counterfactual_score,
        "stable_transition_suppression_would_suppress": stable_transition_suppression_would_suppress,
        "threshold_delta_to_suppress": threshold_delta_to_suppress,
        "low_vol_threshold_would_suppress": low_vol_threshold_would_suppress,
        "transition_profile": str(transition_profile.get("classification", "not_transition")),
        "transition_history_matches": _safe_int(transition_profile.get("history_matches"), 0),
        "transition_observation_count": _safe_int(transition_profile.get("observation_count"), 0),
        "transition_span_business_days": _safe_int(transition_profile.get("span_business_days"), 0),
        "transition_persistent": bool(transition_profile.get("persistent", False)),
        "transition_state_probability_bucket": str(
            transition_profile.get("state_probability_bucket", "stable")
        ),
        "transition_rate_bucket": str(transition_profile.get("transition_rate_bucket", "zero")),
    }
    return signal, triggers, observations, sorted(set(cause_keys))


def _history_cause_matches(
    *,
    history: list[dict[str, object]],
    as_of_date: str,
    cause_keys: list[str],
    persistence_business_days: int,
) -> tuple[int, int]:
    if not cause_keys:
        return 0, 0
    relevant_dates: list[str] = []
    for entry in history:
        entry_date = str(entry.get("as_of_date", ""))
        if not entry_date:
            continue
        distance = _business_day_distance(entry_date, as_of_date)
        if distance > persistence_business_days:
            continue
        entry_causes = set(_normalized_list(entry.get("base_cause_keys")))
        if not entry_causes:
            entry_triggers = set(_normalized_list(entry.get("base_trigger_names") or entry.get("trigger_names")))
            regime_bucket = str(entry.get("regime_bucket", "unknown"))
            if "calibration_ece" in entry_triggers:
                entry_causes.add(f"calibration_ece:{regime_bucket}")
            if "calibration_gap" in entry_triggers:
                entry_causes.add(f"calibration_gap:{regime_bucket}")
        if entry_causes.intersection(cause_keys):
            relevant_dates.append(entry_date)
    if not relevant_dates:
        return 0, 0
    earliest_date = min(relevant_dates)
    span_business_days = _business_day_distance(earliest_date, as_of_date)
    return len(relevant_dates), span_business_days


def _calibration_signal(
    aggregate_metrics: dict[str, float],
    regime_summary: dict[str, object],
    settings: Settings,
    policy_context: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, str]], list[dict[str, str]], list[str]]:
    current_regime = _current_regime(regime_summary)
    regime_key = _threshold_regime_key(current_regime)
    regime_bucket = _regime_bucket(current_regime)
    history = cast(list[dict[str, object]], policy_context.get("history", []))
    as_of_date = str(policy_context.get("as_of_date", pd.Timestamp.now(tz="UTC").date().isoformat()))
    min_ratio = float(
        settings.model_settings.retraining.calibration_min_fold_breach_ratio
        * settings.model_settings.retraining.calibration_breach_ratio_multipliers.get(regime_key, 1.0)
    )
    min_folds = int(settings.model_settings.retraining.calibration_min_persistent_folds)
    min_runs = int(settings.model_settings.retraining.calibration_min_persistent_runs)
    min_span_days = int(settings.model_settings.retraining.calibration_min_persistent_span_business_days)
    persistence_business_days = int(settings.model_settings.retraining.calibration_persistence_business_days)
    immediate_multiplier = float(settings.model_settings.retraining.calibration_immediate_multiplier)
    ece_warning = float(settings.model_settings.calibration.ece_warning)
    gap_warning = float(settings.model_settings.calibration.calibration_gap_warning)
    ece_mean = float(aggregate_metrics.get("ece_mean", 0.0) or 0.0)
    gap_mean = float(aggregate_metrics.get("calibration_gap_mean", 0.0) or 0.0)
    ece_breach_count = int(aggregate_metrics.get("ece_warning_breach_count", 0) or 0)
    ece_breach_ratio = float(aggregate_metrics.get("ece_warning_breach_ratio", 0.0) or 0.0)
    gap_breach_count = int(aggregate_metrics.get("calibration_gap_warning_breach_count", 0) or 0)
    gap_breach_ratio = float(aggregate_metrics.get("calibration_gap_warning_breach_ratio", 0.0) or 0.0)

    ece_fold_persistent = ece_breach_count >= min_folds and ece_breach_ratio >= min_ratio
    gap_fold_persistent = gap_breach_count >= min_folds and gap_breach_ratio >= min_ratio
    ece_cause_keys = [f"calibration_ece:{regime_bucket}"]
    gap_cause_keys = [f"calibration_gap:{regime_bucket}"]
    ece_history_matches, ece_span_business_days = _history_cause_matches(
        history=history,
        as_of_date=as_of_date,
        cause_keys=ece_cause_keys,
        persistence_business_days=persistence_business_days,
    )
    gap_history_matches, gap_span_business_days = _history_cause_matches(
        history=history,
        as_of_date=as_of_date,
        cause_keys=gap_cause_keys,
        persistence_business_days=persistence_business_days,
    )
    ece_observation_count = ece_history_matches + (1 if ece_fold_persistent else 0)
    gap_observation_count = gap_history_matches + (1 if gap_fold_persistent else 0)
    ece_run_persistent = (
        ece_fold_persistent
        and ece_observation_count >= min_runs
        and ece_span_business_days >= min_span_days
    )
    gap_run_persistent = (
        gap_fold_persistent
        and gap_observation_count >= min_runs
        and gap_span_business_days >= min_span_days
    )
    ece_immediate = ece_mean >= ece_warning * immediate_multiplier and ece_fold_persistent
    gap_immediate = gap_mean >= gap_warning * immediate_multiplier and gap_fold_persistent
    ece_triggered = ece_run_persistent or ece_immediate
    gap_triggered = gap_run_persistent or gap_immediate

    triggers: list[dict[str, str]] = []
    observations: list[dict[str, str]] = []
    cause_keys: list[str] = []
    if ece_triggered:
        triggers.append(
            _trigger(
                "calibration_ece",
                "WARNING",
                (
                    f"ECE mean={ece_mean:.3f}, breach_ratio={ece_breach_ratio:.2f}, breach_count={ece_breach_count}, "
                    f"run_observations={ece_observation_count}, span_business_days={ece_span_business_days}."
                ),
            )
        )
        cause_keys.extend(ece_cause_keys)
    elif ece_fold_persistent:
        observations.append(
            _trigger(
                "calibration_ece_run_persistence_watch",
                "INFO",
                (
                    "ECE met fold-persistence requirements but did not persist across enough runs/business-day span."
                ),
            )
        )
    elif ece_mean >= ece_warning:
        observations.append(
            _trigger(
                "calibration_ece_persistence_watch",
                "INFO",
                "ECE exceeded the warning threshold but did not persist across enough folds.",
            )
        )
    if gap_triggered:
        triggers.append(
            _trigger(
                "calibration_gap",
                "WARNING",
                (
                    f"Calibration gap mean={gap_mean:.3f}, breach_ratio={gap_breach_ratio:.2f}, "
                    f"breach_count={gap_breach_count}, run_observations={gap_observation_count}, "
                    f"span_business_days={gap_span_business_days}."
                ),
            )
        )
        cause_keys.extend(gap_cause_keys)
    elif gap_fold_persistent:
        observations.append(
            _trigger(
                "calibration_gap_run_persistence_watch",
                "INFO",
                (
                    "Calibration gap met fold-persistence requirements but did not persist across enough runs/business-day span."
                ),
            )
        )
    elif gap_mean >= gap_warning:
        observations.append(
            _trigger(
                "calibration_gap_persistence_watch",
                "INFO",
                "Calibration gap exceeded the warning threshold but did not persist across enough folds.",
            )
        )

    signal: dict[str, object] = {
        "ece_mean": ece_mean,
        "calibration_gap_mean": gap_mean,
        "ece_breach_count": ece_breach_count,
        "ece_breach_ratio": ece_breach_ratio,
        "calibration_gap_breach_count": gap_breach_count,
        "calibration_gap_breach_ratio": gap_breach_ratio,
        "minimum_persistent_folds": min_folds,
        "minimum_persistent_runs": min_runs,
        "minimum_persistent_span_business_days": min_span_days,
        "persistence_business_days": persistence_business_days,
        "minimum_breach_ratio": min_ratio,
        "immediate_multiplier": immediate_multiplier,
        "ece_warning_threshold": ece_warning,
        "calibration_gap_warning_threshold": gap_warning,
        "ece_fold_persistent": ece_fold_persistent,
        "calibration_gap_fold_persistent": gap_fold_persistent,
        "ece_run_persistent": ece_run_persistent,
        "calibration_gap_run_persistent": gap_run_persistent,
        "ece_history_matches": ece_history_matches,
        "calibration_gap_history_matches": gap_history_matches,
        "ece_observation_count": ece_observation_count,
        "calibration_gap_observation_count": gap_observation_count,
        "ece_span_business_days": ece_span_business_days,
        "calibration_gap_span_business_days": gap_span_business_days,
        "ece_persistent": ece_run_persistent,
        "calibration_gap_persistent": gap_run_persistent,
        "ece_triggered": ece_triggered,
        "calibration_gap_triggered": gap_triggered,
    }
    return signal, triggers, observations, cause_keys


def _regime_signal(
    regime_summary: dict[str, object],
    settings: Settings,
    policy_context: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, str]], list[dict[str, str]], list[str]]:
    current_regime = _current_regime(regime_summary)
    dominant_recent_regime = str(regime_summary.get("dominant_recent_regime", current_regime))
    shift_flag = bool(regime_summary.get("regime_shift_flag", False))
    state_probability = _safe_float(regime_summary.get("state_probability"), 1.0)
    transition_rate = _safe_float(regime_summary.get("transition_rate"), 0.0)
    transition_profile = _transition_profile(
        regime_summary=regime_summary,
        settings=settings,
        policy_context=policy_context,
    )
    triggers: list[dict[str, str]] = []
    observations: list[dict[str, str]] = []
    cause_keys: list[str] = []

    if shift_flag and settings.model_settings.retraining.regime_shift_requires_retrain:
        triggers.append(
            _trigger(
                "regime_shift",
                "WARNING",
                (
                    f"Recent regime changed from {dominant_recent_regime} to {current_regime} "
                    f"(state_probability={state_probability:.2f}, transition_rate={transition_rate:.2f})."
                ),
            )
        )
        cause_keys.append(f"regime_shift:{dominant_recent_regime}->{current_regime}")
    if current_regime == "transition" and settings.model_settings.retraining.transition_regime_requires_retrain:
        if bool(transition_profile.get("immediate_transition", False)) or (
            bool(transition_profile.get("unstable_transition", False))
            and bool(transition_profile.get("persistent", False))
        ):
            triggers.append(
                _trigger(
                    "transition_regime",
                    "WARNING",
                    (
                        "Transition regime remains unstable after persistence checks, "
                        "so retraining is recommended."
                    ),
                )
            )
            cause_keys.append("transition_regime:stress")
        elif bool(transition_profile.get("stable_transition", False)):
            observations.append(
                _trigger(
                    "transition_regime_watch",
                    "INFO",
                    (
                        "Transition regime is stable enough for watch-only mode after persistence checks "
                        f"(state_probability={state_probability:.2f}, transition_rate={transition_rate:.2f})."
                    ),
                )
            )
        else:
            observations.append(
                _trigger(
                    "transition_regime_watch",
                    "INFO",
                    (
                        "Transition regime is monitored in watch-only mode because it is not yet unstable enough "
                        f"after persistence checks (state_probability={state_probability:.2f}, transition_rate={transition_rate:.2f})."
                    ),
                )
            )
    if current_regime == "high_vol" and settings.model_settings.retraining.high_vol_requires_retrain:
        triggers.append(
            _trigger("high_vol_regime", "WARNING", "High-volatility regime is configured to require retraining.")
        )
        cause_keys.append("high_vol_regime:stress")

    if not triggers and current_regime in {"transition", "high_vol"}:
        observations.append(
            _trigger(
                "regime_watch",
                "INFO",
                f"Regime {current_regime} is being monitored without an automatic retraining trigger.",
            )
        )
    signal = {
        "shift_flag": shift_flag,
        "state_probability": state_probability,
        "transition_rate": transition_rate,
        "transition_profile": str(transition_profile.get("classification", "not_transition")),
        "transition_history_matches": _safe_int(transition_profile.get("history_matches"), 0),
        "transition_observation_count": _safe_int(transition_profile.get("observation_count"), 0),
        "transition_span_business_days": _safe_int(transition_profile.get("span_business_days"), 0),
        "transition_persistent": bool(transition_profile.get("persistent", False)),
        "stable_transition": bool(transition_profile.get("stable_transition", False)),
        "unstable_transition": bool(transition_profile.get("unstable_transition", False)),
        "immediate_transition": bool(transition_profile.get("immediate_transition", False)),
        "state_probability_bucket": str(transition_profile.get("state_probability_bucket", "stable")),
        "transition_rate_bucket": str(transition_profile.get("transition_rate_bucket", "zero")),
        "triggered": bool(triggers),
    }
    return signal, triggers, observations, cause_keys


def _pbo_signal(
    pbo: float | None,
    pbo_summary: dict[str, object],
    pbo_diagnostics: dict[str, object] | None,
    settings: Settings,
    policy_context: dict[str, object],
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str], dict[str, object]]:
    value = _safe_float(pbo, 0.0) if pbo is not None else None
    label = str(pbo_summary.get("label", "not_available"))
    if value is None:
        return [], [], [], {"value": None, "persistent": False, "history_matches": 0}

    history = cast(list[dict[str, object]], policy_context.get("history", []))
    as_of_date = str(policy_context.get("as_of_date", pd.Timestamp.now(tz="UTC").date().isoformat()))
    warning_threshold = float(settings.model_settings.retraining.pbo_warning)
    immediate_threshold = float(settings.model_settings.retraining.pbo_immediate_threshold)
    persistence_days = int(settings.model_settings.retraining.pbo_persistence_business_days)
    min_observations = int(settings.model_settings.retraining.pbo_min_persistent_observations)
    min_span_days = int(settings.model_settings.retraining.pbo_min_persistent_span_business_days)
    cause_key = f"pbo:{label}"
    diagnostics = pbo_diagnostics or {}
    competition = cast(dict[str, object], diagnostics.get("near_candidate_competition", {}))
    competition_dominated = bool(competition.get("competition_dominated", False))
    competition_reason = str(competition.get("competition_reason", "")).strip()

    recent_matches = 0
    oldest_match_date: str | None = None
    for entry in history:
        entry_date = str(entry.get("as_of_date", ""))
        if not entry_date:
            continue
        distance = _business_day_distance(entry_date, as_of_date)
        if distance > persistence_days:
            continue
        previous_pbo = _safe_float(entry.get("pbo"), 0.0)
        previous_causes = _normalized_list(entry.get("base_cause_keys"))
        if previous_pbo >= warning_threshold or cause_key in previous_causes:
            recent_matches += 1
            if oldest_match_date is None or entry_date < oldest_match_date:
                oldest_match_date = entry_date

    observation_count = recent_matches + 1 if value >= warning_threshold else recent_matches
    span_business_days = (
        _business_day_distance(oldest_match_date, as_of_date) if oldest_match_date is not None else 0
    )
    persistent = (
        value >= warning_threshold
        and observation_count >= min_observations
        and span_business_days >= min_span_days
    )
    immediate = value >= immediate_threshold and not competition_dominated
    should_trigger = persistent or immediate
    triggers: list[dict[str, str]] = []
    observations: list[dict[str, str]] = []
    cause_keys: list[str] = []
    if should_trigger:
        triggers.append(
            _trigger(
                "pbo",
                "WARNING" if not immediate else "FAIL",
                (
                    f"PBO={value:.3f} ({label}), history_matches={recent_matches}, "
                    f"observation_count={observation_count}, min_observations={min_observations}, "
                    f"span_business_days={span_business_days}, min_span_days={min_span_days}."
                ),
            )
        )
        cause_keys.append(cause_key)
    elif value >= immediate_threshold and competition_dominated:
        detail = (
            f"PBO={value:.3f} ({label}) was severe but treated as watch-only because close candidate competition "
            f"dominated the CPCV winners."
        )
        if competition_reason:
            detail += f" {competition_reason}"
        observations.append(_trigger("pbo_competition_watch", "INFO", detail))
    elif value >= warning_threshold:
        observations.append(
            _trigger(
                "pbo_persistence_watch",
                "INFO",
                (
                    f"PBO={value:.3f} ({label}) exceeded the warning threshold, "
                    f"but persistence/cool-off conditions were not met yet "
                    f"(observation_count={observation_count}, span_business_days={span_business_days})."
                ),
            )
        )
    signal: dict[str, object] = {
        "value": value,
        "label": label,
        "warning_threshold": warning_threshold,
        "immediate_threshold": immediate_threshold,
        "history_matches": recent_matches,
        "observation_count": observation_count,
        "minimum_persistent_observations": min_observations,
        "minimum_persistent_span_business_days": min_span_days,
        "persistence_business_days": persistence_days,
        "span_business_days": span_business_days,
        "competition_dominated": competition_dominated,
        "competition_reason": competition_reason,
        "persistent": persistent,
        "triggered": bool(should_trigger),
    }
    return triggers, observations, cause_keys, signal


def _latest_relevant_event(
    history: list[dict[str, object]],
    as_of_date: str,
    cause_keys: list[str],
    current_regime: str,
    settings: Settings,
) -> dict[str, object] | None:
    if not cause_keys:
        return None
    best_match: dict[str, object] | None = None
    best_distance: int | None = None
    for entry in history:
        if not bool(entry.get("should_retrain", False)):
            continue
        entry_causes = set(_normalized_list(entry.get("effective_cause_keys")))
        if not entry_causes:
            entry_triggers = set(_normalized_list(entry.get("effective_trigger_names")) or _normalized_list(entry.get("trigger_names")))
            entry_families = _normalized_list(entry.get("drift_trigger_families"))
            if "feature_drift" in entry_triggers:
                for family in entry_families:
                    entry_causes.add(f"feature_drift:{family}:{str(entry.get('regime_bucket', _regime_bucket(current_regime)))}")
            if "pbo" in entry_triggers:
                entry_causes.add(f"pbo:{str(entry.get('pbo_label', 'warning'))}")
            if "regime_shift" in entry_triggers:
                entry_causes.add(
                    f"regime_shift:{str(entry.get('dominant_recent_regime', 'unknown'))}->{str(entry.get('current_regime', current_regime))}"
                )
            if "transition_regime" in entry_triggers:
                entry_causes.add(f"transition_regime:{str(entry.get('regime_bucket', _regime_bucket(current_regime)))}")
            if "calibration_ece" in entry_triggers:
                entry_causes.add(f"calibration_ece:{str(entry.get('regime_bucket', _regime_bucket(current_regime)))}")
            if "calibration_gap" in entry_triggers:
                entry_causes.add(f"calibration_gap:{str(entry.get('regime_bucket', _regime_bucket(current_regime)))}")
        if not entry_causes.intersection(cause_keys):
            continue
        entry_date = str(entry.get("as_of_date", ""))
        if not entry_date:
            continue
        entry_regime_bucket = str(entry.get("regime_bucket") or _regime_bucket(str(entry.get("current_regime", current_regime))))
        if settings.model_settings.retraining.cooloff_same_regime_only and entry_regime_bucket != _regime_bucket(current_regime):
            continue
        distance = _business_day_distance(entry_date, as_of_date)
        if distance > int(settings.model_settings.retraining.cause_cooloff_business_days):
            continue
        if best_distance is None or distance < best_distance:
            best_match = entry
            best_distance = distance
    return best_match


def build_retraining_history_entry(
    *,
    as_of_date: str,
    retraining_monitor: dict[str, object],
    regime_summary: dict[str, object],
    tickers: list[str] | None = None,
    source_mode: str | None = None,
    dummy_mode: str | None = None,
    pbo: float | None = None,
    pbo_summary: dict[str, object] | None = None,
    created_at: str | None = None,
) -> dict[str, object]:
    current_regime = _current_regime(regime_summary)
    dominant_recent_regime = str(regime_summary.get("dominant_recent_regime", current_regime))
    drift_signal = cast(dict[str, object], retraining_monitor.get("drift_signal", {}))
    drift_trigger_families = _normalized_list(
        retraining_monitor.get("drift_trigger_families", drift_signal.get("trigger_families", []))
    )
    base_trigger_names = _normalized_list(retraining_monitor.get("base_trigger_names"))
    effective_trigger_names = _normalized_list(retraining_monitor.get("effective_trigger_names"))
    suppressed_trigger_names = _normalized_list(retraining_monitor.get("suppressed_trigger_names"))
    if not effective_trigger_names and retraining_monitor.get("triggers"):
        effective_trigger_names = [str(item["name"]) for item in cast(list[dict[str, object]], retraining_monitor["triggers"])]
    base_cause_keys = _normalized_list(retraining_monitor.get("base_cause_keys"))
    if not base_cause_keys:
        for family in drift_trigger_families:
            if "feature_drift" in base_trigger_names:
                base_cause_keys.append(f"feature_drift:{family}:{_regime_bucket(current_regime)}")
    effective_cause_keys = _normalized_list(retraining_monitor.get("effective_cause_keys"))
    if not effective_cause_keys:
        effective_cause_keys = list(base_cause_keys)
    if pbo is None:
        pbo = _safe_float(retraining_monitor.get("pbo"), 0.0)
    pbo_label = None
    if pbo_summary is not None:
        pbo_label = pbo_summary.get("label")
    if pbo_label is None:
        pbo_label = retraining_monitor.get("pbo_label")
    regime_bucket = _regime_bucket(current_regime)
    return {
        "created_at": created_at or pd.Timestamp.now(tz="UTC").isoformat(),
        "as_of_date": as_of_date,
        "ticker_set": _ticker_set_label(tickers),
        "source_mode": source_mode,
        "dummy_mode": dummy_mode,
        "current_regime": current_regime,
        "dominant_recent_regime": dominant_recent_regime,
        "regime_bucket": regime_bucket,
        "base_should_retrain": bool(retraining_monitor.get("base_should_retrain", False)),
        "should_retrain": bool(retraining_monitor.get("should_retrain", False)),
        "trigger_names": list(effective_trigger_names),
        "base_trigger_names": list(base_trigger_names),
        "effective_trigger_names": list(effective_trigger_names),
        "suppressed_trigger_names": list(suppressed_trigger_names),
        "drift_trigger_families": list(drift_trigger_families),
        "family_regime_keys": [f"{family}:{regime_bucket}" for family in drift_trigger_families],
        "base_cause_keys": list(base_cause_keys),
        "effective_cause_keys": list(effective_cause_keys),
        "pbo": pbo,
        "pbo_label": pbo_label,
        "policy_decision": str(retraining_monitor.get("policy_decision", "watch_only")),
    }


def build_retraining_monitor(
    *,
    aggregate_metrics: dict[str, float],
    drift_summary: dict[str, object],
    regime_summary: dict[str, object],
    pbo: float | None,
    pbo_summary: dict[str, object],
    pbo_diagnostics: dict[str, object] | None = None,
    candidate_level_pbo: float | None = None,
    candidate_level_pbo_summary: dict[str, object] | None = None,
    candidate_level_pbo_diagnostics: dict[str, object] | None = None,
    settings: Settings,
    policy_context: dict[str, object] | None = None,
) -> dict[str, object]:
    context = policy_context or {}
    history = cast(list[dict[str, object]], context.get("history", []))
    as_of_date = str(context.get("as_of_date", pd.Timestamp.now(tz="UTC").date().isoformat()))
    current_regime = _current_regime(regime_summary)
    dominant_recent_regime = str(regime_summary.get("dominant_recent_regime", current_regime))
    state_probability = _safe_float(regime_summary.get("state_probability"), 1.0)
    transition_rate = _safe_float(regime_summary.get("transition_rate"), 0.0)

    drift_signal, drift_triggers, drift_observations, drift_cause_keys = _drift_signal(
        drift_summary=drift_summary,
        regime_summary=regime_summary,
        settings=settings,
        policy_context={"history": history, "as_of_date": as_of_date},
    )
    calibration_signal, calibration_triggers, calibration_observations, calibration_cause_keys = _calibration_signal(
        aggregate_metrics=aggregate_metrics,
        regime_summary=regime_summary,
        settings=settings,
        policy_context={"history": history, "as_of_date": as_of_date},
    )
    regime_signal, regime_triggers, regime_observations, regime_cause_keys = _regime_signal(
        regime_summary=regime_summary,
        settings=settings,
        policy_context={"history": history, "as_of_date": as_of_date},
    )
    pbo_triggers, pbo_observations, pbo_cause_keys, pbo_signal = _pbo_signal(
        pbo=pbo,
        pbo_summary=pbo_summary,
        pbo_diagnostics=pbo_diagnostics,
        settings=settings,
        policy_context={"history": history, "as_of_date": as_of_date},
    )

    base_triggers = drift_triggers + calibration_triggers + regime_triggers + pbo_triggers
    observations = drift_observations + calibration_observations + regime_observations + pbo_observations
    base_trigger_names = [str(item["name"]) for item in base_triggers]
    base_cause_keys = drift_cause_keys + calibration_cause_keys + regime_cause_keys + pbo_cause_keys
    effective_triggers = list(base_triggers)
    effective_trigger_names = list(base_trigger_names)
    suppressed_trigger_names: list[str] = []
    effective_cause_keys = list(base_cause_keys)
    policy_notes: list[str] = []
    policy_decision = "trigger" if effective_triggers else "watch_only"

    if settings.model_settings.retraining.regime_shift_requires_confirmation:
        non_regime_triggers = [name for name in effective_trigger_names if name != "regime_shift"]
        standalone_regime_shift = effective_trigger_names == ["regime_shift"]
        if standalone_regime_shift and (
            state_probability >= settings.model_settings.retraining.regime_shift_standalone_state_probability
            and transition_rate <= settings.model_settings.retraining.regime_shift_standalone_transition_rate
        ):
            effective_triggers = []
            effective_trigger_names = []
            effective_cause_keys = []
            suppressed_trigger_names = ["regime_shift"]
            policy_decision = "watch_regime_shift_only"
            policy_notes.append(
                "Regime shift alone does not trigger retraining unless the state probability weakens or transition rate rises."
            )
            observations.append(
                _trigger(
                    "regime_shift_confirmation_watch",
                    "INFO",
                    (
                        f"Standalone regime shift {dominant_recent_regime}->{current_regime} is monitored because "
                        f"state_probability={state_probability:.2f} and transition_rate={transition_rate:.2f} are not severe."
                    ),
                )
            )
        elif "regime_shift" in effective_trigger_names and not non_regime_triggers:
            policy_notes.append("Regime shift remains the only active trigger after family suppression and persistence checks.")

    if (
        settings.model_settings.retraining.regime_shift_requires_drift_confirmation
        and "regime_shift" in effective_trigger_names
        and "feature_drift" not in effective_trigger_names
    ):
        effective_triggers = [item for item in effective_triggers if str(item["name"]) != "regime_shift"]
        effective_trigger_names = [name for name in effective_trigger_names if name != "regime_shift"]
        effective_cause_keys = [key for key in effective_cause_keys if not key.startswith("regime_shift:")]
        suppressed_trigger_names = sorted(set(suppressed_trigger_names + ["regime_shift"]))
        policy_notes.append("Regime shift is kept as watch-only unless feature drift co-occurs in the same run.")
        observations.append(
            _trigger(
                "regime_shift_drift_confirmation_watch",
                "INFO",
                "Regime shift did not retain an effective retraining trigger because feature drift did not co-occur.",
            )
        )

    latest_match = _latest_relevant_event(
        history=history,
        as_of_date=as_of_date,
        cause_keys=effective_cause_keys,
        current_regime=current_regime,
        settings=settings,
    )
    cooloff_active = False
    cooloff_reference_date: str | None = None
    cooloff_remaining_business_days = 0
    if latest_match is not None and effective_triggers:
        latest_date = str(latest_match.get("as_of_date", ""))
        distance = _business_day_distance(latest_date, as_of_date)
        remaining = max(settings.model_settings.retraining.cause_cooloff_business_days - distance, 0)
        if remaining > 0:
            cooloff_active = True
            cooloff_reference_date = latest_date
            cooloff_remaining_business_days = remaining
            suppressed_trigger_names = sorted(set(suppressed_trigger_names + effective_trigger_names))
            effective_triggers = []
            effective_trigger_names = []
            effective_cause_keys = []
            policy_decision = "suppressed_by_cooloff"
            policy_notes.append(
                "Recent retraining event with the same cause/regime bucket is inside the cool-off window; suppressing repeat retrain."
            )

    base_should_retrain = bool(base_triggers)
    should_retrain = bool(effective_triggers)
    if not base_should_retrain:
        policy_decision = "watch_only"
    elif should_retrain and policy_decision != "trigger":
        policy_decision = "trigger"

    return {
        "base_should_retrain": base_should_retrain,
        "should_retrain": should_retrain,
        "trigger_count": len(effective_triggers),
        "triggers": effective_triggers,
        "observations": observations,
        "drift_signal": drift_signal,
        "calibration_signal": calibration_signal,
        "regime_signal": regime_signal,
        "pbo_signal": pbo_signal,
        "policy_decision": policy_decision,
        "policy_notes": policy_notes,
        "cooloff_active": cooloff_active,
        "cooloff_reference_date": cooloff_reference_date,
        "cooloff_remaining_business_days": cooloff_remaining_business_days,
        "base_trigger_names": base_trigger_names,
        "effective_trigger_names": effective_trigger_names,
        "suppressed_trigger_names": suppressed_trigger_names,
        "drift_trigger_families": cast(list[str], drift_signal.get("trigger_families", [])),
        "base_cause_keys": base_cause_keys,
        "effective_cause_keys": effective_cause_keys,
        "history_lookback_count": len(history),
        "pbo": pbo,
        "pbo_label": pbo_summary.get("label"),
        "candidate_level_pbo": candidate_level_pbo,
        "candidate_level_pbo_label": (candidate_level_pbo_summary or {}).get("label"),
        "candidate_level_pbo_competition_dominated": bool(
            cast(dict[str, object], (candidate_level_pbo_diagnostics or {}).get("near_candidate_competition", {})).get(
                "competition_dominated",
                False,
            )
        ),
    }

