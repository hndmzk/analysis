from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import numpy as np
import pandas as pd


MACRO_FEATURES = {
    "fed_funds_rate",
    "yield_curve_slope",
    "vix",
    "vix_change_5d",
}
NEWS_FEATURES = {
    "news_sentiment_1d",
    "news_sentiment_5d",
    "news_sentiment_decay_5d",
    "news_relevance_5d",
    "news_novelty_5d",
    "news_source_diversity_5d",
    "news_volume_zscore_20d",
}
FUNDAMENTAL_FEATURES = {
    "fundamental_revenue_growth",
    "fundamental_earnings_yield",
    "fundamental_leverage",
    "fundamental_profitability",
}
SECTOR_FEATURES = {
    "sector_relative_momentum_20d",
    "sector_strength_20d",
    "sector_vol_spread_20d",
}
VOLATILITY_FEATURES = {
    "realized_vol_5d",
    "realized_vol_10d",
    "realized_vol_20d",
    "garman_klass_vol",
    "bb_width",
    "atr_14",
    "atr_ratio",
}
VOLUME_FEATURES = {
    "volume_ratio_5d",
    "volume_ratio_20d",
    "obv_slope_10d",
}
CALENDAR_FEATURES = {
    "day_of_week",
    "month",
    "is_month_end",
}
DEFAULT_PROXY_SENSITIVE_FEATURES = {
    "garman_klass_vol",
    "atr_14",
    "atr_ratio",
    "bb_width",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "obv_slope_10d",
}
DRIFT_FAMILIES = ("price_momentum", "volatility", "volume", "macro", "calendar")


def feature_family(feature: str) -> str:
    if feature in MACRO_FEATURES:
        return "macro"
    if feature in NEWS_FEATURES:
        return "news"
    if feature in FUNDAMENTAL_FEATURES:
        return "fundamental"
    if feature in SECTOR_FEATURES:
        return "sector"
    if feature in VOLATILITY_FEATURES:
        return "volatility"
    if feature in VOLUME_FEATURES:
        return "volume"
    if feature in CALENDAR_FEATURES:
        return "calendar"
    return "price_momentum"


def _clean_series(series: pd.Series) -> pd.Series:
    return series.replace([np.inf, -np.inf], np.nan).dropna().astype(float)


def _psi_bins(reference: pd.Series, bucket_count: int) -> np.ndarray:
    quantiles = np.linspace(0.0, 1.0, bucket_count + 1)
    edges = np.quantile(reference.to_numpy(dtype=float), quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        minimum = float(reference.min())
        maximum = float(reference.max())
        if minimum == maximum:
            maximum = minimum + 1.0
        edges = np.linspace(minimum, maximum, bucket_count + 1)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def population_stability_index(reference: pd.Series, current: pd.Series, bucket_count: int = 10) -> float:
    reference_clean = _clean_series(reference)
    current_clean = _clean_series(current)
    if reference_clean.empty or current_clean.empty:
        return 0.0
    edges = _psi_bins(reference_clean, bucket_count=bucket_count)
    reference_hist = np.histogram(reference_clean.to_numpy(dtype=float), bins=edges)[0].astype(float)
    current_hist = np.histogram(current_clean.to_numpy(dtype=float), bins=edges)[0].astype(float)
    reference_ratio = np.clip(reference_hist / reference_hist.sum(), 1e-6, 1.0)
    current_ratio = np.clip(current_hist / current_hist.sum(), 1e-6, 1.0)
    psi = np.sum((current_ratio - reference_ratio) * np.log(current_ratio / reference_ratio))
    return float(psi)


def _signed_log1p(series: pd.Series) -> pd.Series:
    return np.sign(series) * np.log1p(np.abs(series))


def _stability_transform(feature: str, reference: pd.Series, current: pd.Series) -> tuple[pd.Series, pd.Series, str]:
    reference_clean = _clean_series(reference)
    current_clean = _clean_series(current)
    family = feature_family(feature)
    transform_name = "identity"
    if family in {"macro", "volatility", "volume"}:
        reference_clean = _signed_log1p(reference_clean)
        current_clean = _signed_log1p(current_clean)
        transform_name = "signed_log1p"
    lower = float(reference_clean.quantile(0.01))
    upper = float(reference_clean.quantile(0.99))
    if lower >= upper:
        return reference_clean, current_clean, transform_name
    return reference_clean.clip(lower, upper), current_clean.clip(lower, upper), transform_name


def _mean_shift_sigma(reference: pd.Series, current: pd.Series) -> float:
    if reference.empty or current.empty:
        return 0.0
    reference_std = float(reference.std(ddof=0))
    if reference_std <= 1e-9:
        return 0.0 if float(reference.mean()) == float(current.mean()) else float("inf")
    return float(abs(float(current.mean()) - float(reference.mean())) / reference_std)


def _regime_active(regime_summary: dict[str, object] | None) -> bool:
    summary = regime_summary or {}
    current_regime = str(summary.get("current_regime", "unknown"))
    regime_shift_flag = bool(summary.get("regime_shift_flag", False))
    state_probability = cast(float, summary.get("state_probability", 1.0) or 1.0)
    return regime_shift_flag or current_regime in {"transition", "high_vol"} or state_probability < 0.6


def _entry_cause(
    *,
    status: str,
    proxy_sensitive: bool,
    proxy_ohlcv_used: bool,
    family: str,
    regime_active: bool,
) -> str:
    if status == "PASS":
        return "stable"
    if proxy_ohlcv_used and proxy_sensitive and regime_active:
        return "mixed_proxy_and_regime"
    if proxy_ohlcv_used and proxy_sensitive:
        return "proxy_sensitive"
    if family in {"macro", "volatility"} and regime_active:
        return "regime_sensitive"
    return "distribution_shift"


def _entry_retrain_action(*, status: str, cause: str) -> str:
    if status == "PASS":
        return "ignore"
    if cause in {"proxy_sensitive", "mixed_proxy_and_regime"}:
        return "watch"
    return "trigger"


def _normalize_family_thresholds(
    *,
    psi_warning: float,
    psi_critical: float,
    family_thresholds: Mapping[str, Mapping[str, float]] | None,
) -> dict[str, dict[str, float]]:
    thresholds: dict[str, dict[str, float]] = {
        family: {"warning": float(psi_warning), "critical": float(psi_critical)}
        for family in DRIFT_FAMILIES
    }
    if family_thresholds is None:
        return thresholds
    for family, values in family_thresholds.items():
        if family not in thresholds:
            continue
        warning = float(values.get("warning", thresholds[family]["warning"]))
        critical = float(values.get("critical", thresholds[family]["critical"]))
        thresholds[family] = {"warning": warning, "critical": critical}
    return thresholds


def _group_summary(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = sorted({str(entry["feature_family"]) for entry in entries})
    summary: list[dict[str, object]] = []
    for group in groups:
        group_entries = [entry for entry in entries if entry["feature_family"] == group]
        group_psi = [cast(float, entry["psi"]) for entry in group_entries]
        summary.append(
            {
                "group": group,
                "feature_count": len(group_entries),
                "max_psi": max(group_psi, default=0.0),
                "mean_psi": float(np.mean(group_psi)) if group_psi else 0.0,
                "flagged_features": [
                    str(entry["feature"]) for entry in group_entries if str(entry["status"]) != "PASS"
                ],
            }
        )
    summary.sort(key=lambda item: cast(float, item["max_psi"]), reverse=True)
    return summary


def _feature_diagnostics(entries: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    diagnostics: list[dict[str, object]] = []
    for entry in entries:
        diagnostics.append(
            {
                "feature": entry["feature"],
                "raw_psi": entry["raw_psi"],
                "adjusted_psi": entry["psi"],
                "family": entry["feature_family"],
                "proxy_sensitive": entry["proxy_sensitive"],
                "data_source": entry.get("data_source", "unknown"),
                "missing_rate": entry.get("missing_rate", 0.0),
                "stale_rate": entry.get("stale_rate", 0.0),
                "primary_cause": entry["primary_cause"],
                "retrain_action": entry["retrain_action"],
                "status": entry["status"],
            }
        )
    return diagnostics


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _supplementary_analysis(
    entries: list[dict[str, object]],
    *,
    proxy_ohlcv_used: bool,
    regime_summary: dict[str, object] | None,
) -> dict[str, object]:
    flagged_entries = [entry for entry in entries if str(entry["status"]) != "PASS"]
    proxy_entries = [entry for entry in flagged_entries if bool(entry["proxy_sensitive"])]
    non_proxy_entries = [entry for entry in flagged_entries if not bool(entry["proxy_sensitive"])]
    macro_vol_entries = [
        entry for entry in flagged_entries if str(entry["feature_family"]) in {"macro", "volatility"}
    ]
    regime_summary = regime_summary or {}
    current_regime = str(regime_summary.get("current_regime", "unknown"))
    regime_shift_flag = bool(regime_summary.get("regime_shift_flag", False))
    state_probability = cast(float, regime_summary.get("state_probability", 1.0) or 1.0)
    regime_active = _regime_active(regime_summary)
    reduction_values = [
        max(0.0, cast(float, entry["raw_psi"]) - cast(float, entry["psi"])) / max(cast(float, entry["raw_psi"]), 1e-9)
        for entry in flagged_entries
        if cast(float, entry["raw_psi"]) > 0
    ]
    reduction_mean = float(np.mean(reduction_values)) if reduction_values else 0.0

    if not flagged_entries:
        primary_cause = "stable"
    elif proxy_ohlcv_used and proxy_entries and not non_proxy_entries and not regime_active:
        primary_cause = "proxy_artifact_likely"
    elif regime_active and macro_vol_entries and len(non_proxy_entries) >= len(proxy_entries):
        primary_cause = "regime_shift_likely"
    elif proxy_ohlcv_used and proxy_entries and reduction_mean >= 0.35 and len(proxy_entries) >= len(non_proxy_entries):
        primary_cause = "proxy_artifact_likely"
    elif regime_active and macro_vol_entries:
        primary_cause = "mixed_proxy_and_regime"
    else:
        primary_cause = "mixed_or_unknown"

    notes: list[str] = []
    if primary_cause == "proxy_artifact_likely":
        notes.append("Flagged drift is concentrated in proxy-sensitive features and is more likely a transport/proxy artifact.")
    if primary_cause in {"regime_shift_likely", "mixed_proxy_and_regime"}:
        notes.append(
            f"Regime context is active: current_regime={current_regime}, regime_shift_flag={regime_shift_flag}, "
            f"state_probability={state_probability:.2f}."
        )
    if reduction_values:
        notes.append(f"Mean raw-to-adjusted PSI reduction={reduction_mean:.2%}.")
    return {
        "primary_cause": primary_cause,
        "proxy_ohlcv_used": proxy_ohlcv_used,
        "regime_active": regime_active,
        "current_regime": current_regime,
        "regime_shift_flag": regime_shift_flag,
        "state_probability": state_probability,
        "proxy_sensitive_flagged_features": [str(entry["feature"]) for entry in proxy_entries],
        "non_proxy_flagged_features": [str(entry["feature"]) for entry in non_proxy_entries],
        "macro_volatility_flagged_features": [str(entry["feature"]) for entry in macro_vol_entries],
        "raw_to_adjusted_reduction_mean": reduction_mean,
        "feature_diagnostics": _feature_diagnostics(entries),
        "notes": notes,
    }


def compute_feature_drift(
    reference_frame: pd.DataFrame,
    current_frame: pd.DataFrame,
    feature_columns: list[str],
    psi_warning: float,
    psi_critical: float,
    bucket_count: int = 10,
    *,
    proxy_ohlcv_used: bool = False,
    regime_summary: dict[str, object] | None = None,
    family_thresholds: Mapping[str, Mapping[str, float]] | None = None,
    proxy_sensitive_features: Sequence[str] | None = None,
    feature_catalog: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, object]:
    entries: list[dict[str, object]] = []
    psi_values: list[float] = []
    raw_psi_values: list[float] = []
    resolved_family_thresholds = _normalize_family_thresholds(
        psi_warning=psi_warning,
        psi_critical=psi_critical,
        family_thresholds=family_thresholds,
    )
    proxy_sensitive_lookup = set(proxy_sensitive_features or DEFAULT_PROXY_SENSITIVE_FEATURES)
    feature_catalog_lookup = {
        str(item.get("feature")): dict(item)
        for item in (feature_catalog or [])
        if item.get("feature") is not None
    }
    regime_active = _regime_active(regime_summary)
    for feature in feature_columns:
        raw_psi = population_stability_index(reference_frame[feature], current_frame[feature], bucket_count=bucket_count)
        reference_stable, current_stable, transform_name = _stability_transform(
            feature,
            reference_frame[feature],
            current_frame[feature],
        )
        psi = population_stability_index(reference_stable, current_stable, bucket_count=bucket_count)
        family = feature_family(feature)
        thresholds = resolved_family_thresholds.get(
            family,
            {"warning": float(psi_warning), "critical": float(psi_critical)},
        )
        warning_threshold = float(thresholds["warning"])
        critical_threshold = float(thresholds["critical"])
        status = "PASS"
        if psi >= critical_threshold:
            status = "FAIL"
        elif psi >= warning_threshold:
            status = "WARNING"
        proxy_sensitive = feature in proxy_sensitive_lookup
        cause = _entry_cause(
            status=status,
            proxy_sensitive=proxy_sensitive,
            proxy_ohlcv_used=proxy_ohlcv_used,
            family=family,
            regime_active=regime_active,
        )
        catalog_entry = feature_catalog_lookup.get(feature, {})
        entries.append(
            {
                "feature": feature,
                "feature_family": family,
                "data_source": str(catalog_entry.get("data_source", "unknown")),
                "missing_rate": _as_float(catalog_entry.get("missing_rate", 0.0)),
                "stale_rate": _as_float(catalog_entry.get("stale_rate", 0.0)),
                "warning_threshold": warning_threshold,
                "critical_threshold": critical_threshold,
                "proxy_sensitive": proxy_sensitive,
                "raw_psi": float(raw_psi),
                "psi": float(psi),
                "stability_transform": transform_name,
                "stability_reduction": float(max(0.0, raw_psi - psi)),
                "mean_shift_sigma": _mean_shift_sigma(reference_stable, current_stable),
                "status": status,
                "analysis_tag": cause,
                "primary_cause": cause,
                "retrain_action": _entry_retrain_action(status=status, cause=cause),
            }
        )
        psi_values.append(float(psi))
        raw_psi_values.append(float(raw_psi))
    warning_features = [str(entry["feature"]) for entry in entries if entry["status"] == "WARNING"]
    critical_features = [str(entry["feature"]) for entry in entries if entry["status"] == "FAIL"]
    max_psi = max(psi_values, default=0.0)
    mean_psi = float(np.mean(psi_values)) if psi_values else 0.0
    max_raw_psi = max(raw_psi_values, default=0.0)
    mean_raw_psi = float(np.mean(raw_psi_values)) if raw_psi_values else 0.0
    ordered_entries = [
        entry
        for _, entry in sorted(zip(psi_values, entries, strict=False), key=lambda item: item[0], reverse=True)
    ]
    supplementary_analysis = _supplementary_analysis(
        ordered_entries,
        proxy_ohlcv_used=proxy_ohlcv_used,
        regime_summary=regime_summary,
    )
    stable_feature_ratio = float(
        sum(str(entry["status"]) == "PASS" for entry in ordered_entries) / max(len(ordered_entries), 1)
    )
    return {
        "bucket_count": bucket_count,
        "warning_threshold": psi_warning,
        "critical_threshold": psi_critical,
        "family_thresholds": resolved_family_thresholds,
        "max_psi": max_psi,
        "mean_psi": mean_psi,
        "max_raw_psi": max_raw_psi,
        "mean_raw_psi": mean_raw_psi,
        "stable_feature_ratio": stable_feature_ratio,
        "warning_features": warning_features,
        "critical_features": critical_features,
        "feature_groups": _group_summary(ordered_entries),
        "supplementary_analysis": supplementary_analysis,
        "features": ordered_entries,
    }
