from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd

from market_prediction_agent.config import LearnedWeightingConfig
from market_prediction_agent.evaluation.learned_weighting import build_walk_forward_learned_weighting


LOOKBACK_WINDOWS = (1, 3, 5, 10)
DECAY_HALFLIVES = (3, 5)
FEATURE_VARIANTS = ("sentiment_1d", "headline_count_1d", "novelty_5d", "source_diversity_5d")
WEIGHTED_VARIANTS = (
    "unweighted_1d",
    "source_weighted_1d",
    "session_weighted_1d",
    "source_session_weighted_1d",
    "learned_weighted_1d",
)
WEIGHTING_MODES = ("unweighted", "fixed", "learned")
SESSION_BUCKET_ORDER = ("pre_market", "regular", "post_market", "weekend_shifted", "mixed", "unknown", "none")
SOURCE_DIVERSITY_BUCKETS = ("single_source", "multi_source")


def _distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    series = pd.Series(values, dtype=float)
    return {
        "count": int(series.shape[0]),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "max": float(series.max()),
    }


def _correlation_or_zero(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.concat([left, right], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    if float(aligned.iloc[:, 0].std(ddof=0) or 0.0) == 0.0 or float(aligned.iloc[:, 1].std(ddof=0) or 0.0) == 0.0:
        return 0.0
    correlation = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return 0.0 if pd.isna(correlation) else float(correlation)


def _as_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    return int(cast(int | str | float, value))


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(cast(int | str | float, value))


def _value_counts(values: list[object]) -> dict[str, int]:
    counts = pd.Series([str(value) for value in values if value not in {None, ""}], dtype=str).value_counts()
    return {str(index): int(value) for index, value in counts.items()}


def _best_entry(entries: list[dict[str, object]], *, key_field: str = "abs_ic") -> dict[str, object] | None:
    if not entries:
        return None
    return max(
        entries,
        key=lambda item: (
            float(cast(float, item.get(key_field, 0.0) or 0.0)),
            float(cast(float, item.get("overlap_rate", 0.0) or 0.0)),
            float(cast(float, item.get("coverage", 0.0) or 0.0)),
        ),
    )


def _combine_source_mix(values: pd.Series) -> str:
    components: set[str] = set()
    for value in values.astype(str):
        for part in value.split("|"):
            normalized = part.strip()
            if normalized and normalized != "none":
                components.add(normalized)
    if not components:
        return "none"
    return "|".join(sorted(components))


def _combine_session_bucket(values: pd.Series) -> str:
    normalized = [str(value).strip() for value in values.astype(str) if str(value).strip() and str(value).strip() != "none"]
    unique = sorted(set(normalized))
    if not unique:
        return "none"
    if len(unique) == 1:
        return unique[0]
    return "mixed"


def _weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0).clip(lower=0.0)
    total_weight = float(numeric_weights.sum())
    if total_weight <= 0.0:
        return float(numeric.mean() if not numeric.empty else 0.0)
    return float((numeric * numeric_weights).sum() / total_weight)


def _filled_string_series(
    frame: pd.DataFrame,
    column: str,
    default: str,
) -> pd.Series:
    if column in frame.columns:
        return frame[column].fillna(default).astype(str)
    return pd.Series(default, index=frame.index, dtype=str)


def _ensure_utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _signal_metrics(
    *,
    signal: pd.Series,
    active_mask: pd.Series,
    returns: pd.Series,
    source_diversity: pd.Series,
    source_count: pd.Series,
) -> dict[str, object]:
    mask = active_mask.fillna(False).astype(bool)
    signal_days = int(mask.sum())
    if signal_days == 0:
        return {
            "coverage": 0.0,
            "ic": 0.0,
            "abs_ic": 0.0,
            "overlap_rate": 0.0,
            "source_diversity_mean": 0.0,
            "source_count_mean": 0.0,
            "signal_days": 0,
        }
    overlap_rate = float((mask & returns.notna()).sum() / signal_days)
    return {
        "coverage": float(mask.mean()),
        "ic": _correlation_or_zero(signal.where(mask), returns),
        "abs_ic": abs(_correlation_or_zero(signal.where(mask), returns)),
        "overlap_rate": overlap_rate,
        "source_diversity_mean": float(source_diversity.where(mask).dropna().mean() or 0.0),
        "source_count_mean": float(source_count.where(mask).dropna().mean() or 0.0),
        "signal_days": signal_days,
    }


def _empty_analysis() -> dict[str, object]:
    return {
        "baseline_1d": {"window_days": 1, "coverage": 0.0, "ic": 0.0, "abs_ic": 0.0, "overlap_rate": 0.0},
        "return_alignment": "signal_day_close_to_close",
        "baseline_missing_rate": 1.0,
        "baseline_stale_rate": 0.0,
        "lookback_windows": [],
        "decay_halflives": [],
        "feature_variants": [],
        "weighted_variants": [],
        "weighting_mode_comparison": [],
        "session_buckets": [],
        "source_mix_buckets": [],
        "source_session_buckets": [],
        "source_diversity_buckets": [],
        "multi_source_weighting_comparison": [],
        "multi_source_weighting_improvement": {
            "learned_abs_ic_delta_vs_unweighted": 0.0,
            "learned_abs_ic_delta_vs_fixed": 0.0,
            "learned_overlap_rate_delta_vs_unweighted": 0.0,
            "learned_overlap_rate_delta_vs_fixed": 0.0,
            "improves_abs_ic_vs_unweighted": False,
            "improves_abs_ic_vs_fixed": False,
            "improves_overlap_rate_vs_unweighted": False,
            "improves_overlap_rate_vs_fixed": False,
            "improves_joint_vs_unweighted": False,
            "improves_joint_vs_fixed": False,
        },
        "selected_weighting_mode": "fixed",
        "selected_weighting_variant": "source_session_weighted_1d",
        "learned_weighting": {
            "mode": "learned",
            "target": "abs_ic",
            "lookback_days": 252,
            "regularization_lambda": 30.0,
            "min_samples": 20,
            "min_weight": 0.05,
            "fallback_mode": "fixed",
            "fit_day_count": 0,
            "fallback_day_count": 0,
            "fallback_rate": 1.0,
            "global_score_mean": 0.0,
            "training_sample_count_mean": 0.0,
            "eligible_combo_count_mean": 0.0,
            "applied_group_count": 0,
            "source_session_weights": [],
            "source_weights": [],
            "session_weights": [],
            "fallback_reasons": {},
            "comparison_ready": False,
        },
        "best_aggregation_window": None,
        "best_aggregation_coverage": 0.0,
        "best_aggregation_abs_ic": 0.0,
        "best_aggregation_overlap_rate": 0.0,
        "best_decay_halflife": None,
        "best_decay_coverage": 0.0,
        "best_decay_abs_ic": 0.0,
        "best_decay_overlap_rate": 0.0,
        "best_feature_variant": None,
        "best_feature_variant_coverage": 0.0,
        "best_feature_variant_abs_ic": 0.0,
        "best_feature_variant_overlap_rate": 0.0,
        "best_weighted_variant": None,
        "best_weighted_variant_coverage": 0.0,
        "best_weighted_variant_abs_ic": 0.0,
        "best_weighted_variant_overlap_rate": 0.0,
        "best_session_bucket": None,
        "best_session_coverage": 0.0,
        "best_session_abs_ic": 0.0,
        "best_session_overlap_rate": 0.0,
        "best_source_mix": None,
        "best_source_mix_coverage": 0.0,
        "best_source_mix_abs_ic": 0.0,
        "best_source_mix_overlap_rate": 0.0,
        "source_advantage_analysis": {
            "dominant_source_mix": None,
            "google_single_source_present": False,
            "google_single_source_abs_ic": 0.0,
            "google_single_source_overlap_rate": 0.0,
            "best_multi_source_mix": None,
            "best_multi_source_abs_ic": 0.0,
            "best_multi_source_overlap_rate": 0.0,
            "google_abs_ic_advantage_vs_best_multi_source": 0.0,
            "google_overlap_advantage_vs_best_multi_source": 0.0,
            "google_coverage_advantage_vs_best_multi_source": 0.0,
            "dominant_reason": "mixed_or_no_clear_advantage",
        },
        "mixed_session_conditions": {
            "mixed_session_present": False,
            "mixed_session_is_best": False,
            "mixed_session_abs_ic": 0.0,
            "mixed_session_overlap_rate": 0.0,
            "best_non_mixed_session": None,
            "best_non_mixed_abs_ic": 0.0,
            "best_non_mixed_overlap_rate": 0.0,
            "mixed_session_abs_ic_advantage": 0.0,
            "mixed_session_overlap_advantage": 0.0,
            "mixed_session_best_source_mix": None,
        },
        "aggregation_improves_coverage": False,
        "aggregation_improves_utility": False,
        "decay_improves_coverage": False,
        "decay_improves_utility": False,
    }


def _ensure_numeric_columns(frame: pd.DataFrame, defaults: dict[str, pd.Series | float]) -> pd.DataFrame:
    updated = frame.copy()
    for column, default in defaults.items():
        if isinstance(default, pd.Series):
            updated[column] = pd.to_numeric(updated.get(column, default), errors="coerce").fillna(default)
        else:
            updated[column] = pd.to_numeric(updated.get(column, default), errors="coerce").fillna(default)
    return updated


def _prepare_news_panel(
    *,
    coverage_panel: pd.DataFrame,
    news: pd.DataFrame,
) -> pd.DataFrame:
    if news.empty:
        news_panel = coverage_panel[["ticker", "date", "signal_day_return"]].copy()
        defaults = {
            "sentiment_score": 0.0,
            "sentiment_score_unweighted": 0.0,
            "sentiment_score_source_weighted": 0.0,
            "sentiment_score_session_weighted": 0.0,
            "sentiment_score_source_session_weighted": 0.0,
            "relevance_score": 0.0,
            "relevance_score_unweighted": 0.0,
            "relevance_score_source_weighted": 0.0,
            "relevance_score_session_weighted": 0.0,
            "relevance_score_source_session_weighted": 0.0,
            "headline_count": 0.0,
            "mapping_confidence": 0.0,
            "novelty_score": 0.0,
            "source_diversity": 0.0,
            "source_count": 0.0,
            "session_bucket": "none",
            "source_mix": "none",
            "stale_data_flag": False,
        }
        for column, default in defaults.items():
            news_panel[column] = default
        return news_panel

    normalized_news = news.copy()
    normalized_news["signal_date"] = pd.to_datetime(normalized_news["available_at"], utc=True).dt.normalize()
    normalized_news = _ensure_numeric_columns(
        normalized_news,
        {
            "sentiment_score": 0.0,
            "sentiment_score_unweighted": normalized_news.get("sentiment_score", 0.0),
            "sentiment_score_source_weighted": normalized_news.get("sentiment_score", 0.0),
            "sentiment_score_session_weighted": normalized_news.get("sentiment_score", 0.0),
            "sentiment_score_source_session_weighted": normalized_news.get("sentiment_score", 0.0),
            "relevance_score": 0.0,
            "relevance_score_unweighted": normalized_news.get("relevance_score", 0.0),
            "relevance_score_source_weighted": normalized_news.get("relevance_score", 0.0),
            "relevance_score_session_weighted": normalized_news.get("relevance_score", 0.0),
            "relevance_score_source_session_weighted": normalized_news.get("relevance_score", 0.0),
            "headline_count": 0.0,
            "mapping_confidence": 0.0,
            "novelty_score": 0.0,
            "source_diversity": 0.0,
            "source_count": 0.0,
        },
    )
    normalized_news["session_bucket"] = _filled_string_series(normalized_news, "session_bucket", "unknown")
    normalized_news["source_mix"] = _filled_string_series(normalized_news, "source_mix", "none")
    normalized_news["stale_data_flag"] = normalized_news.get("stale_data_flag", False).fillna(False).astype(bool)
    aggregation_rows: list[dict[str, object]] = []
    for (ticker, signal_date), group in normalized_news.groupby(["ticker", "signal_date"], as_index=False):
        aggregation_weight = group["headline_count"].clip(lower=0.0)
        if float(aggregation_weight.sum()) <= 0.0:
            aggregation_weight = pd.Series(np.ones(len(group), dtype=float), index=group.index)
        aggregation_rows.append(
            {
                "ticker": ticker,
                "date": signal_date,
                "published_at": pd.to_datetime(group["published_at"], utc=True).max()
                if "published_at" in group.columns
                else _ensure_utc_timestamp(signal_date),
                "available_at": pd.to_datetime(group["available_at"], utc=True).max(),
                "sentiment_score": _weighted_mean(group["sentiment_score"], aggregation_weight),
                "sentiment_score_unweighted": _weighted_mean(group["sentiment_score_unweighted"], aggregation_weight),
                "sentiment_score_source_weighted": _weighted_mean(group["sentiment_score_source_weighted"], aggregation_weight),
                "sentiment_score_session_weighted": _weighted_mean(group["sentiment_score_session_weighted"], aggregation_weight),
                "sentiment_score_source_session_weighted": _weighted_mean(
                    group["sentiment_score_source_session_weighted"],
                    aggregation_weight,
                ),
                "relevance_score": _weighted_mean(group["relevance_score"], aggregation_weight),
                "relevance_score_unweighted": _weighted_mean(group["relevance_score_unweighted"], aggregation_weight),
                "relevance_score_source_weighted": _weighted_mean(group["relevance_score_source_weighted"], aggregation_weight),
                "relevance_score_session_weighted": _weighted_mean(group["relevance_score_session_weighted"], aggregation_weight),
                "relevance_score_source_session_weighted": _weighted_mean(
                    group["relevance_score_source_session_weighted"],
                    aggregation_weight,
                ),
                "headline_count": float(pd.to_numeric(group["headline_count"], errors="coerce").sum()),
                "mapping_confidence": float(pd.to_numeric(group["mapping_confidence"], errors="coerce").mean()),
                "novelty_score": float(pd.to_numeric(group["novelty_score"], errors="coerce").mean()),
                "source_diversity": float(pd.to_numeric(group["source_diversity"], errors="coerce").mean()),
                "source_count": float(pd.to_numeric(group["source_count"], errors="coerce").max()),
                "session_bucket": _combine_session_bucket(group["session_bucket"]),
                "source_mix": _combine_source_mix(group["source_mix"]),
                "stale_data_flag": bool(group["stale_data_flag"].any()),
            }
        )
    daily_news = pd.DataFrame(aggregation_rows)
    news_panel = coverage_panel.merge(daily_news, on=["ticker", "date"], how="left")
    news_panel = _ensure_numeric_columns(
        news_panel,
        {
            "sentiment_score": 0.0,
            "sentiment_score_unweighted": news_panel.get("sentiment_score", 0.0),
            "sentiment_score_source_weighted": news_panel.get("sentiment_score", 0.0),
            "sentiment_score_session_weighted": news_panel.get("sentiment_score", 0.0),
            "sentiment_score_source_session_weighted": news_panel.get("sentiment_score", 0.0),
            "relevance_score": 0.0,
            "relevance_score_unweighted": news_panel.get("relevance_score", 0.0),
            "relevance_score_source_weighted": news_panel.get("relevance_score", 0.0),
            "relevance_score_session_weighted": news_panel.get("relevance_score", 0.0),
            "relevance_score_source_session_weighted": news_panel.get("relevance_score", 0.0),
            "headline_count": 0.0,
            "mapping_confidence": 0.0,
            "novelty_score": 0.0,
            "source_diversity": 0.0,
            "source_count": 0.0,
        },
    )
    news_panel["stale_data_flag"] = news_panel["stale_data_flag"].fillna(False).astype(bool)
    news_panel["session_bucket"] = news_panel["session_bucket"].fillna("none").astype(str)
    news_panel["source_mix"] = news_panel["source_mix"].fillna("none").astype(str)
    return news_panel


def _bucket_metrics(
    *,
    panel: pd.DataFrame,
    signal: pd.Series,
    active_mask: pd.Series,
    bucket_mask: pd.Series,
) -> dict[str, object]:
    return _signal_metrics(
        signal=signal,
        active_mask=active_mask & bucket_mask,
        returns=panel["signal_day_return"],
        source_diversity=panel["source_diversity"],
        source_count=panel["source_count"],
    )


def _default_learned_weighting_config() -> LearnedWeightingConfig:
    return LearnedWeightingConfig(
        regularization_lambda=30.0,
        min_samples=20,
        lookback_days=252,
        target="abs_ic",
        min_weight=0.05,
        fallback_mode="fixed",
    )


def _resolved_weighting_mode(mode: str) -> str:
    normalized = str(mode or "fixed").strip().lower()
    return normalized if normalized in WEIGHTING_MODES else "fixed"


def _effective_news_weight(
    *,
    panel: pd.DataFrame,
    relevance: pd.Series,
) -> pd.Series:
    clipped_relevance = pd.to_numeric(relevance, errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)
    base_weight = (
        panel["headline_count"].fillna(0.0).clip(lower=0.0)
        * clipped_relevance
        * panel["mapping_confidence"].fillna(0.0).clip(lower=0.0, upper=1.0)
    )
    return pd.Series(
        np.where(base_weight > 0.0, base_weight, panel["headline_count"].fillna(0.0).clip(lower=0.0)),
        index=panel.index,
        dtype=float,
    )


def _weighting_mode_metrics(
    *,
    panel: pd.DataFrame,
    signal: pd.Series,
    active_mask: pd.Series,
    mode: str,
    variant: str,
) -> dict[str, object]:
    return {
        "mode": mode,
        "variant": variant,
        **_signal_metrics(
            signal=signal,
            active_mask=active_mask,
            returns=panel["signal_day_return"],
            source_diversity=panel["source_diversity"],
            source_count=panel["source_count"],
        ),
    }


def build_news_feature_utility_comparison(
    *,
    ohlcv: pd.DataFrame,
    news: pd.DataFrame,
    weighting_mode: str = "fixed",
    learned_weighting: LearnedWeightingConfig | None = None,
    lookback_windows: tuple[int, ...] = LOOKBACK_WINDOWS,
    decay_halflives: tuple[int, ...] = DECAY_HALFLIVES,
    min_effective_weight: float = 0.05,
    coverage_extension_business_days: int = 3,
) -> dict[str, object]:
    if ohlcv.empty:
        return _empty_analysis()
    panel = ohlcv.loc[:, ["ticker", "timestamp_utc", "close"]].copy()
    panel["date"] = pd.to_datetime(panel["timestamp_utc"], utc=True).dt.normalize()
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["signal_day_return"] = panel.groupby("ticker")["close"].pct_change()
    latest_panel_date = cast(pd.Timestamp, panel["date"].max())
    future_dates = [
        (latest_panel_date + pd.tseries.offsets.BDay(offset)).normalize()
        for offset in range(1, max(int(coverage_extension_business_days), 0) + 1)
    ]
    if future_dates:
        latest_tickers = panel.groupby("ticker", as_index=False).tail(1).loc[:, ["ticker"]]
        future_rows = pd.concat(
            [
                latest_tickers.assign(
                    timestamp_utc=future_date,
                    close=np.nan,
                    date=future_date,
                    signal_day_return=np.nan,
                )
                for future_date in future_dates
            ],
            ignore_index=True,
        )
        coverage_panel = (
            pd.concat([panel, future_rows], ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
        )
    else:
        coverage_panel = panel.copy()
    news_panel = _prepare_news_panel(coverage_panel=coverage_panel, news=news)
    resolved_weighting_mode = _resolved_weighting_mode(weighting_mode)
    resolved_learned_weighting = learned_weighting or _default_learned_weighting_config()
    fallback_mode = _resolved_weighting_mode(resolved_learned_weighting.fallback_mode)
    fallback_signal = (
        news_panel["sentiment_score_unweighted"].fillna(news_panel["sentiment_score"]).fillna(0.0)
        if fallback_mode == "unweighted"
        else news_panel["sentiment_score_source_session_weighted"].fillna(news_panel["sentiment_score"]).fillna(0.0)
    )
    fallback_relevance = (
        news_panel["relevance_score_unweighted"].fillna(news_panel["relevance_score"]).clip(lower=0.0, upper=1.0)
        if fallback_mode == "unweighted"
        else news_panel["relevance_score_source_session_weighted"].fillna(news_panel["relevance_score"]).clip(
            lower=0.0,
            upper=1.0,
        )
    )
    learned_result = build_walk_forward_learned_weighting(
        news_panel=news_panel,
        news=news,
        config=resolved_learned_weighting,
        fallback_sentiment=fallback_signal,
        fallback_relevance=fallback_relevance,
    )
    news_panel["sentiment_score_learned"] = learned_result.sentiment
    news_panel["relevance_score_learned"] = learned_result.relevance

    weighting_signal_map = {
        "unweighted_1d": news_panel["sentiment_score_unweighted"].fillna(news_panel["sentiment_score"]).fillna(0.0),
        "source_weighted_1d": news_panel["sentiment_score_source_weighted"].fillna(news_panel["sentiment_score"]).fillna(0.0),
        "session_weighted_1d": news_panel["sentiment_score_session_weighted"].fillna(news_panel["sentiment_score"]).fillna(0.0),
        "source_session_weighted_1d": news_panel["sentiment_score_source_session_weighted"]
        .fillna(news_panel["sentiment_score"])
        .fillna(0.0),
        "learned_weighted_1d": news_panel["sentiment_score_learned"].fillna(fallback_signal).fillna(0.0),
    }
    weighting_relevance_map = {
        "unweighted_1d": news_panel["relevance_score_unweighted"].fillna(news_panel["relevance_score"]).clip(lower=0.0, upper=1.0),
        "source_weighted_1d": news_panel["relevance_score_source_weighted"].fillna(news_panel["relevance_score"]).clip(lower=0.0, upper=1.0),
        "session_weighted_1d": news_panel["relevance_score_session_weighted"].fillna(news_panel["relevance_score"]).clip(lower=0.0, upper=1.0),
        "source_session_weighted_1d": news_panel["relevance_score_source_session_weighted"]
        .fillna(news_panel["relevance_score"])
        .clip(lower=0.0, upper=1.0),
        "learned_weighted_1d": news_panel["relevance_score_learned"].fillna(fallback_relevance).clip(lower=0.0, upper=1.0),
    }
    weighting_weight_map = {
        variant: _effective_news_weight(panel=news_panel, relevance=relevance)
        for variant, relevance in weighting_relevance_map.items()
    }
    weighting_active_map = {
        variant: weight > 0.0
        for variant, weight in weighting_weight_map.items()
    }
    selected_variant = {
        "unweighted": "unweighted_1d",
        "fixed": "source_session_weighted_1d",
        "learned": "learned_weighted_1d",
    }[resolved_weighting_mode]
    news_panel["news_weight"] = weighting_weight_map[selected_variant]
    production_signal = weighting_signal_map[selected_variant]
    active_mask = weighting_active_map[selected_variant]
    news_panel["weighted_sentiment"] = production_signal * news_panel["news_weight"]
    news_panel["weighted_novelty"] = news_panel["novelty_score"].fillna(0.0) * news_panel["news_weight"]
    news_panel["weighted_source_diversity"] = news_panel["source_diversity"].fillna(0.0) * news_panel["news_weight"]
    grouped = news_panel.groupby("ticker", group_keys=False)

    lookback_results: list[dict[str, object]] = []
    for window in lookback_windows:
        numerator = grouped["weighted_sentiment"].transform(
            lambda series, win=window: series.fillna(0.0).rolling(win, min_periods=1).sum()
        )
        denominator = grouped["news_weight"].transform(
            lambda series, win=window: series.fillna(0.0).rolling(win, min_periods=1).sum()
        )
        signal = (numerator / denominator.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        lookback_results.append(
            {
                "window_days": window,
                **_signal_metrics(
                    signal=signal,
                    active_mask=denominator > 0.0,
                    returns=news_panel["signal_day_return"],
                    source_diversity=news_panel["source_diversity"],
                    source_count=news_panel["source_count"],
                ),
            }
        )

    decay_results: list[dict[str, object]] = []
    for halflife in decay_halflives:
        numerator = grouped["weighted_sentiment"].transform(
            lambda series, hl=halflife: series.fillna(0.0).ewm(halflife=hl, adjust=False).mean()
        )
        denominator = grouped["news_weight"].transform(
            lambda series, hl=halflife: series.fillna(0.0).ewm(halflife=hl, adjust=False).mean()
        )
        signal = (numerator / denominator.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        decay_results.append(
            {
                "halflife_days": halflife,
                **_signal_metrics(
                    signal=signal,
                    active_mask=denominator >= min_effective_weight,
                    returns=news_panel["signal_day_return"],
                    source_diversity=news_panel["source_diversity"],
                    source_count=news_panel["source_count"],
                ),
            }
        )

    rolling_weight_5d = grouped["news_weight"].transform(lambda series: series.fillna(0.0).rolling(5, min_periods=1).sum())
    novelty_5d_signal = grouped["weighted_novelty"].transform(
        lambda series: series.fillna(0.0).rolling(5, min_periods=1).sum()
    ) / rolling_weight_5d.replace(0.0, np.nan)
    diversity_5d_signal = grouped["weighted_source_diversity"].transform(
        lambda series: series.fillna(0.0).rolling(5, min_periods=1).sum()
    ) / rolling_weight_5d.replace(0.0, np.nan)

    feature_variants = [
        {
            "variant": "sentiment_1d",
            **_signal_metrics(
                signal=production_signal,
                active_mask=active_mask,
                returns=news_panel["signal_day_return"],
                source_diversity=news_panel["source_diversity"],
                source_count=news_panel["source_count"],
            ),
        },
        {
            "variant": "headline_count_1d",
            **_signal_metrics(
                signal=np.log1p(news_panel["headline_count"]),
                active_mask=news_panel["headline_count"] > 0.0,
                returns=news_panel["signal_day_return"],
                source_diversity=news_panel["source_diversity"],
                source_count=news_panel["source_count"],
            ),
        },
        {
            "variant": "novelty_5d",
            **_signal_metrics(
                signal=novelty_5d_signal,
                active_mask=rolling_weight_5d > 0.0,
                returns=news_panel["signal_day_return"],
                source_diversity=news_panel["source_diversity"],
                source_count=news_panel["source_count"],
            ),
        },
        {
            "variant": "source_diversity_5d",
            **_signal_metrics(
                signal=diversity_5d_signal,
                active_mask=rolling_weight_5d > 0.0,
                returns=news_panel["signal_day_return"],
                source_diversity=news_panel["source_diversity"],
                source_count=news_panel["source_count"],
            ),
        },
    ]

    weighted_variants = [
        {
            "variant": "unweighted_1d",
            **_signal_metrics(
                signal=weighting_signal_map["unweighted_1d"],
                active_mask=weighting_active_map["unweighted_1d"],
                returns=news_panel["signal_day_return"],
                source_diversity=news_panel["source_diversity"],
                source_count=news_panel["source_count"],
            ),
        },
        {
            "variant": "source_weighted_1d",
            **_signal_metrics(
                signal=weighting_signal_map["source_weighted_1d"],
                active_mask=weighting_active_map["source_weighted_1d"],
                returns=news_panel["signal_day_return"],
                source_diversity=news_panel["source_diversity"],
                source_count=news_panel["source_count"],
            ),
        },
        {
            "variant": "session_weighted_1d",
            **_signal_metrics(
                signal=weighting_signal_map["session_weighted_1d"],
                active_mask=weighting_active_map["session_weighted_1d"],
                returns=news_panel["signal_day_return"],
                source_diversity=news_panel["source_diversity"],
                source_count=news_panel["source_count"],
            ),
        },
        {
            "variant": "source_session_weighted_1d",
            **_signal_metrics(
                signal=weighting_signal_map["source_session_weighted_1d"],
                active_mask=weighting_active_map["source_session_weighted_1d"],
                returns=news_panel["signal_day_return"],
                source_diversity=news_panel["source_diversity"],
                source_count=news_panel["source_count"],
            ),
        },
        {
            "variant": "learned_weighted_1d",
            **_signal_metrics(
                signal=weighting_signal_map["learned_weighted_1d"],
                active_mask=weighting_active_map["learned_weighted_1d"],
                returns=news_panel["signal_day_return"],
                source_diversity=news_panel["source_diversity"],
                source_count=news_panel["source_count"],
            ),
        },
    ]
    weighting_mode_comparison = [
        _weighting_mode_metrics(
            panel=news_panel,
            signal=weighting_signal_map["unweighted_1d"],
            active_mask=weighting_active_map["unweighted_1d"],
            mode="unweighted",
            variant="unweighted_1d",
        ),
        _weighting_mode_metrics(
            panel=news_panel,
            signal=weighting_signal_map["source_session_weighted_1d"],
            active_mask=weighting_active_map["source_session_weighted_1d"],
            mode="fixed",
            variant="source_session_weighted_1d",
        ),
        _weighting_mode_metrics(
            panel=news_panel,
            signal=weighting_signal_map["learned_weighted_1d"],
            active_mask=weighting_active_map["learned_weighted_1d"],
            mode="learned",
            variant="learned_weighted_1d",
        ),
    ]

    present_buckets = {bucket for bucket in news_panel["session_bucket"].unique().tolist() if bucket not in {"", "none"}}
    session_results: list[dict[str, object]] = []
    for bucket in [bucket for bucket in SESSION_BUCKET_ORDER if bucket in present_buckets] or list(SESSION_BUCKET_ORDER):
        session_results.append(
            {
                "session_bucket": bucket,
                **_bucket_metrics(
                    panel=news_panel,
                    signal=production_signal,
                    active_mask=active_mask,
                    bucket_mask=news_panel["session_bucket"].eq(bucket),
                ),
            }
        )

    source_mix_names = sorted(
        {
            source_mix
            for source_mix in news_panel["source_mix"].astype(str).unique().tolist()
            if source_mix not in {"", "none"}
        }
    )
    source_mix_results: list[dict[str, object]] = []
    source_session_results: list[dict[str, object]] = []
    for source_mix in source_mix_names:
        source_mix_mask = news_panel["source_mix"].eq(source_mix)
        source_mix_results.append(
            {
                "source_mix": source_mix,
                **_bucket_metrics(
                    panel=news_panel,
                    signal=production_signal,
                    active_mask=active_mask,
                    bucket_mask=source_mix_mask,
                ),
            }
        )
        for session_bucket in sorted(
            {
                str(value)
                for value in news_panel.loc[source_mix_mask, "session_bucket"].astype(str).tolist()
                if str(value)
            }
        ):
            bucket_mask = source_mix_mask & news_panel["session_bucket"].eq(session_bucket)
            source_session_results.append(
                {
                    "source_mix": source_mix,
                    "session_bucket": session_bucket,
                    "source_session_bucket": f"{source_mix}::{session_bucket}",
                    **_bucket_metrics(
                        panel=news_panel,
                        signal=production_signal,
                        active_mask=active_mask,
                        bucket_mask=bucket_mask,
                    ),
                }
            )

    diversity_results: list[dict[str, object]] = []
    for bucket in SOURCE_DIVERSITY_BUCKETS:
        bucket_mask = active_mask & (
            news_panel["source_count"] <= 1.0 if bucket == "single_source" else news_panel["source_count"] >= 2.0
        )
        diversity_results.append(
            {
                "diversity_bucket": bucket,
                **_signal_metrics(
                    signal=production_signal,
                    active_mask=bucket_mask,
                    returns=news_panel["signal_day_return"],
                    source_diversity=news_panel["source_diversity"],
                    source_count=news_panel["source_count"],
                ),
            }
        )

    multi_source_weighting_comparison = [
        _weighting_mode_metrics(
            panel=news_panel,
            signal=weighting_signal_map["unweighted_1d"],
            active_mask=weighting_active_map["unweighted_1d"] & (news_panel["source_count"] >= 2.0),
            mode="unweighted",
            variant="unweighted_1d",
        ),
        _weighting_mode_metrics(
            panel=news_panel,
            signal=weighting_signal_map["source_session_weighted_1d"],
            active_mask=weighting_active_map["source_session_weighted_1d"] & (news_panel["source_count"] >= 2.0),
            mode="fixed",
            variant="source_session_weighted_1d",
        ),
        _weighting_mode_metrics(
            panel=news_panel,
            signal=weighting_signal_map["learned_weighted_1d"],
            active_mask=weighting_active_map["learned_weighted_1d"] & (news_panel["source_count"] >= 2.0),
            mode="learned",
            variant="learned_weighted_1d",
        ),
    ]

    baseline = next((item for item in lookback_results if _as_int(item.get("window_days")) == 1), None)
    if baseline is None:
        return _empty_analysis()
    best_aggregation = max(lookback_results, key=lambda item: (_as_float(item.get("coverage")), _as_float(item.get("abs_ic"))))
    best_aggregation_utility = _best_entry(lookback_results) or {}
    best_decay = max(decay_results, key=lambda item: (_as_float(item.get("coverage")), _as_float(item.get("abs_ic")))) if decay_results else None
    best_decay_utility = _best_entry(decay_results) if decay_results else None
    best_variant = _best_entry(feature_variants) or {}
    best_weighted_variant = _best_entry(weighted_variants) or {}
    best_session = _best_entry(session_results)
    best_source_mix = _best_entry(source_mix_results)

    google_single_source = next(
        (item for item in source_mix_results if str(item.get("source_mix")) == "google_news_rss"),
        None,
    )
    multi_source_candidates = [item for item in source_mix_results if "|" in str(item.get("source_mix", ""))]
    best_multi_source = _best_entry(multi_source_candidates) if multi_source_candidates else None
    google_abs_ic_advantage = _as_float(google_single_source.get("abs_ic") if google_single_source else 0.0) - _as_float(
        best_multi_source.get("abs_ic") if best_multi_source else 0.0
    )
    google_overlap_advantage = _as_float(
        google_single_source.get("overlap_rate") if google_single_source else 0.0
    ) - _as_float(best_multi_source.get("overlap_rate") if best_multi_source else 0.0)
    google_coverage_advantage = _as_float(
        google_single_source.get("coverage") if google_single_source else 0.0
    ) - _as_float(best_multi_source.get("coverage") if best_multi_source else 0.0)
    if google_abs_ic_advantage > 0.0 and google_overlap_advantage > 0.05:
        dominant_reason = "overlap_advantage"
    elif google_abs_ic_advantage > 0.0 and google_coverage_advantage > 0.002:
        dominant_reason = "coverage_advantage"
    elif google_abs_ic_advantage > 0.0 and best_multi_source is not None:
        dominant_reason = "multi_source_dilution"
    else:
        dominant_reason = "mixed_or_no_clear_advantage"

    mixed_session = next((item for item in session_results if str(item.get("session_bucket")) == "mixed"), None)
    non_mixed_sessions = [item for item in session_results if str(item.get("session_bucket")) not in {"mixed", "none"}]
    best_non_mixed_session = _best_entry(non_mixed_sessions) if non_mixed_sessions else None
    best_mixed_source_session = _best_entry(
        [item for item in source_session_results if str(item.get("session_bucket")) == "mixed"]
    )
    multi_source_mode_by_name = {str(item.get("mode")): item for item in multi_source_weighting_comparison}
    learned_multi_source = multi_source_mode_by_name.get("learned", {})
    fixed_multi_source = multi_source_mode_by_name.get("fixed", {})
    unweighted_multi_source = multi_source_mode_by_name.get("unweighted", {})

    baseline_coverage = _as_float(baseline.get("coverage"))
    baseline_abs_ic = _as_float(baseline.get("abs_ic"))
    return {
        "baseline_1d": baseline,
        "return_alignment": "signal_day_close_to_close",
        "baseline_missing_rate": float(1.0 - baseline_coverage),
        "baseline_stale_rate": float(news_panel["stale_data_flag"].mean()),
        "lookback_windows": lookback_results,
        "decay_halflives": decay_results,
        "feature_variants": feature_variants,
        "weighted_variants": weighted_variants,
        "weighting_mode_comparison": weighting_mode_comparison,
        "session_buckets": session_results,
        "source_mix_buckets": source_mix_results,
        "source_session_buckets": source_session_results,
        "source_diversity_buckets": diversity_results,
        "multi_source_weighting_comparison": multi_source_weighting_comparison,
        "multi_source_weighting_improvement": {
            "learned_abs_ic_delta_vs_unweighted": _as_float(learned_multi_source.get("abs_ic"))
            - _as_float(unweighted_multi_source.get("abs_ic")),
            "learned_abs_ic_delta_vs_fixed": _as_float(learned_multi_source.get("abs_ic"))
            - _as_float(fixed_multi_source.get("abs_ic")),
            "learned_overlap_rate_delta_vs_unweighted": _as_float(learned_multi_source.get("overlap_rate"))
            - _as_float(unweighted_multi_source.get("overlap_rate")),
            "learned_overlap_rate_delta_vs_fixed": _as_float(learned_multi_source.get("overlap_rate"))
            - _as_float(fixed_multi_source.get("overlap_rate")),
            "improves_abs_ic_vs_unweighted": _as_float(learned_multi_source.get("abs_ic"))
            > _as_float(unweighted_multi_source.get("abs_ic")),
            "improves_abs_ic_vs_fixed": _as_float(learned_multi_source.get("abs_ic"))
            > _as_float(fixed_multi_source.get("abs_ic")),
            "improves_overlap_rate_vs_unweighted": _as_float(learned_multi_source.get("overlap_rate"))
            > _as_float(unweighted_multi_source.get("overlap_rate")),
            "improves_overlap_rate_vs_fixed": _as_float(learned_multi_source.get("overlap_rate"))
            > _as_float(fixed_multi_source.get("overlap_rate")),
            "improves_joint_vs_unweighted": _as_float(learned_multi_source.get("abs_ic"))
            > _as_float(unweighted_multi_source.get("abs_ic"))
            and _as_float(learned_multi_source.get("overlap_rate")) > _as_float(unweighted_multi_source.get("overlap_rate")),
            "improves_joint_vs_fixed": _as_float(learned_multi_source.get("abs_ic"))
            > _as_float(fixed_multi_source.get("abs_ic"))
            and _as_float(fixed_multi_source.get("overlap_rate")) < _as_float(learned_multi_source.get("overlap_rate")),
        },
        "selected_weighting_mode": resolved_weighting_mode,
        "selected_weighting_variant": selected_variant,
        "learned_weighting": learned_result.summary,
        "best_aggregation_window": best_aggregation.get("window_days"),
        "best_aggregation_coverage": _as_float(best_aggregation.get("coverage")),
        "best_aggregation_abs_ic": _as_float(best_aggregation.get("abs_ic")),
        "best_aggregation_overlap_rate": _as_float(best_aggregation.get("overlap_rate")),
        "best_decay_halflife": None if best_decay is None else best_decay.get("halflife_days"),
        "best_decay_coverage": 0.0 if best_decay is None else _as_float(best_decay.get("coverage")),
        "best_decay_abs_ic": 0.0 if best_decay is None else _as_float(best_decay.get("abs_ic")),
        "best_decay_overlap_rate": 0.0 if best_decay is None else _as_float(best_decay.get("overlap_rate")),
        "best_feature_variant": best_variant.get("variant"),
        "best_feature_variant_coverage": _as_float(best_variant.get("coverage")),
        "best_feature_variant_abs_ic": _as_float(best_variant.get("abs_ic")),
        "best_feature_variant_overlap_rate": _as_float(best_variant.get("overlap_rate")),
        "best_weighted_variant": best_weighted_variant.get("variant"),
        "best_weighted_variant_coverage": _as_float(best_weighted_variant.get("coverage")),
        "best_weighted_variant_abs_ic": _as_float(best_weighted_variant.get("abs_ic")),
        "best_weighted_variant_overlap_rate": _as_float(best_weighted_variant.get("overlap_rate")),
        "best_session_bucket": None if best_session is None else best_session.get("session_bucket"),
        "best_session_coverage": 0.0 if best_session is None else _as_float(best_session.get("coverage")),
        "best_session_abs_ic": 0.0 if best_session is None else _as_float(best_session.get("abs_ic")),
        "best_session_overlap_rate": 0.0 if best_session is None else _as_float(best_session.get("overlap_rate")),
        "best_source_mix": None if best_source_mix is None else best_source_mix.get("source_mix"),
        "best_source_mix_coverage": 0.0 if best_source_mix is None else _as_float(best_source_mix.get("coverage")),
        "best_source_mix_abs_ic": 0.0 if best_source_mix is None else _as_float(best_source_mix.get("abs_ic")),
        "best_source_mix_overlap_rate": 0.0 if best_source_mix is None else _as_float(best_source_mix.get("overlap_rate")),
        "source_advantage_analysis": {
            "dominant_source_mix": None if best_source_mix is None else best_source_mix.get("source_mix"),
            "google_single_source_present": google_single_source is not None,
            "google_single_source_abs_ic": _as_float(google_single_source.get("abs_ic") if google_single_source else 0.0),
            "google_single_source_overlap_rate": _as_float(
                google_single_source.get("overlap_rate") if google_single_source else 0.0
            ),
            "best_multi_source_mix": None if best_multi_source is None else best_multi_source.get("source_mix"),
            "best_multi_source_abs_ic": _as_float(best_multi_source.get("abs_ic") if best_multi_source else 0.0),
            "best_multi_source_overlap_rate": _as_float(
                best_multi_source.get("overlap_rate") if best_multi_source else 0.0
            ),
            "google_abs_ic_advantage_vs_best_multi_source": google_abs_ic_advantage,
            "google_overlap_advantage_vs_best_multi_source": google_overlap_advantage,
            "google_coverage_advantage_vs_best_multi_source": google_coverage_advantage,
            "dominant_reason": dominant_reason,
        },
        "mixed_session_conditions": {
            "mixed_session_present": mixed_session is not None,
            "mixed_session_is_best": bool(best_session is not None and best_session.get("session_bucket") == "mixed"),
            "mixed_session_abs_ic": _as_float(mixed_session.get("abs_ic") if mixed_session else 0.0),
            "mixed_session_overlap_rate": _as_float(mixed_session.get("overlap_rate") if mixed_session else 0.0),
            "best_non_mixed_session": None if best_non_mixed_session is None else best_non_mixed_session.get("session_bucket"),
            "best_non_mixed_abs_ic": _as_float(
                best_non_mixed_session.get("abs_ic") if best_non_mixed_session else 0.0
            ),
            "best_non_mixed_overlap_rate": _as_float(
                best_non_mixed_session.get("overlap_rate") if best_non_mixed_session else 0.0
            ),
            "mixed_session_abs_ic_advantage": _as_float(mixed_session.get("abs_ic") if mixed_session else 0.0)
            - _as_float(best_non_mixed_session.get("abs_ic") if best_non_mixed_session else 0.0),
            "mixed_session_overlap_advantage": _as_float(
                mixed_session.get("overlap_rate") if mixed_session else 0.0
            ) - _as_float(best_non_mixed_session.get("overlap_rate") if best_non_mixed_session else 0.0),
            "mixed_session_best_source_mix": None
            if best_mixed_source_session is None
            else best_mixed_source_session.get("source_mix"),
        },
        "aggregation_improves_coverage": _as_float(best_aggregation.get("coverage")) > baseline_coverage,
        "aggregation_improves_utility": _as_float(best_aggregation_utility.get("abs_ic")) > baseline_abs_ic,
        "decay_improves_coverage": False if best_decay is None else _as_float(best_decay.get("coverage")) > baseline_coverage,
        "decay_improves_utility": False
        if best_decay_utility is None
        else _as_float(best_decay_utility.get("abs_ic")) > baseline_abs_ic,
    }


def _fallback_analysis_from_flattened_run(run: dict[str, object]) -> dict[str, object]:
    return {
        "baseline_1d": {
            "window_days": 1,
            "coverage": float(cast(float, run.get("news_coverage_lookback_1d", 0.0))),
            "ic": 0.0,
            "abs_ic": float(cast(float, run.get("news_abs_ic_lookback_1d", 0.0))),
            "overlap_rate": 0.0,
            "source_diversity_mean": 0.0,
            "source_count_mean": 0.0,
            "signal_days": 0,
        },
        "return_alignment": "signal_day_close_to_close",
        "baseline_missing_rate": float(cast(float, run.get("news_feature_missing_rate", 1.0))),
        "baseline_stale_rate": float(cast(float, run.get("news_feature_stale_rate", 0.0))),
        "lookback_windows": [
            {
                "window_days": window,
                "coverage": float(cast(float, run.get(f"news_coverage_lookback_{window}d", 0.0))),
                "ic": 0.0,
                "abs_ic": float(cast(float, run.get(f"news_abs_ic_lookback_{window}d", 0.0))),
                "overlap_rate": 0.0,
                "source_diversity_mean": 0.0,
                "source_count_mean": 0.0,
                "signal_days": 0,
            }
            for window in LOOKBACK_WINDOWS
        ],
        "decay_halflives": [
            {
                "halflife_days": halflife,
                "coverage": float(cast(float, run.get(f"news_decay_coverage_halflife_{halflife}d", 0.0))),
                "ic": 0.0,
                "abs_ic": float(cast(float, run.get(f"news_decay_abs_ic_halflife_{halflife}d", 0.0))),
                "overlap_rate": 0.0,
                "source_diversity_mean": 0.0,
                "source_count_mean": 0.0,
                "signal_days": 0,
            }
            for halflife in DECAY_HALFLIVES
        ],
        "feature_variants": [],
        "weighted_variants": [],
        "weighting_mode_comparison": [],
        "session_buckets": [],
        "source_mix_buckets": [],
        "source_session_buckets": [],
        "source_diversity_buckets": [],
        "multi_source_weighting_comparison": [],
        "multi_source_weighting_improvement": {
            "learned_abs_ic_delta_vs_unweighted": 0.0,
            "learned_abs_ic_delta_vs_fixed": 0.0,
            "learned_overlap_rate_delta_vs_unweighted": 0.0,
            "learned_overlap_rate_delta_vs_fixed": 0.0,
            "improves_abs_ic_vs_unweighted": False,
            "improves_abs_ic_vs_fixed": False,
            "improves_overlap_rate_vs_unweighted": False,
            "improves_overlap_rate_vs_fixed": False,
            "improves_joint_vs_unweighted": False,
            "improves_joint_vs_fixed": False,
        },
        "selected_weighting_mode": "fixed",
        "selected_weighting_variant": "source_session_weighted_1d",
        "learned_weighting": _empty_analysis()["learned_weighting"],
        "best_aggregation_window": run.get("news_best_aggregation_window"),
        "best_aggregation_coverage": float(cast(float, run.get("news_best_aggregation_coverage", 0.0))),
        "best_aggregation_abs_ic": float(cast(float, run.get("news_best_aggregation_abs_ic", 0.0))),
        "best_aggregation_overlap_rate": 0.0,
        "best_decay_halflife": run.get("news_best_decay_halflife"),
        "best_decay_coverage": float(cast(float, run.get("news_best_decay_coverage", 0.0))),
        "best_decay_abs_ic": float(cast(float, run.get("news_best_decay_abs_ic", 0.0))),
        "best_decay_overlap_rate": 0.0,
        "best_feature_variant": None,
        "best_feature_variant_coverage": 0.0,
        "best_feature_variant_abs_ic": 0.0,
        "best_feature_variant_overlap_rate": 0.0,
        "best_weighted_variant": None,
        "best_weighted_variant_coverage": 0.0,
        "best_weighted_variant_abs_ic": 0.0,
        "best_weighted_variant_overlap_rate": 0.0,
        "best_session_bucket": None,
        "best_session_coverage": 0.0,
        "best_session_abs_ic": 0.0,
        "best_session_overlap_rate": 0.0,
        "best_source_mix": None,
        "best_source_mix_coverage": 0.0,
        "best_source_mix_abs_ic": 0.0,
        "best_source_mix_overlap_rate": 0.0,
        "source_advantage_analysis": {
            "dominant_source_mix": None,
            "google_single_source_present": False,
            "google_single_source_abs_ic": 0.0,
            "google_single_source_overlap_rate": 0.0,
            "best_multi_source_mix": None,
            "best_multi_source_abs_ic": 0.0,
            "best_multi_source_overlap_rate": 0.0,
            "google_abs_ic_advantage_vs_best_multi_source": 0.0,
            "google_overlap_advantage_vs_best_multi_source": 0.0,
            "google_coverage_advantage_vs_best_multi_source": 0.0,
            "dominant_reason": "mixed_or_no_clear_advantage",
        },
        "mixed_session_conditions": {
            "mixed_session_present": False,
            "mixed_session_is_best": False,
            "mixed_session_abs_ic": 0.0,
            "mixed_session_overlap_rate": 0.0,
            "best_non_mixed_session": None,
            "best_non_mixed_abs_ic": 0.0,
            "best_non_mixed_overlap_rate": 0.0,
            "mixed_session_abs_ic_advantage": 0.0,
            "mixed_session_overlap_advantage": 0.0,
            "mixed_session_best_source_mix": None,
        },
        "aggregation_improves_coverage": bool(run.get("news_aggregation_improves_coverage", False)),
        "aggregation_improves_utility": bool(run.get("news_aggregation_improves_utility", False)),
        "decay_improves_coverage": bool(run.get("news_decay_improves_coverage", False)),
        "decay_improves_utility": bool(run.get("news_decay_improves_utility", False)),
    }


def _aggregate_bucket_summaries(
    analyses: list[dict[str, object]],
    *,
    list_key: str,
    id_key: str,
    default_ids: list[object] | None = None,
) -> list[dict[str, object]]:
    discovered_ids = {
        entry.get(id_key)
        for analysis in analyses
        for entry in cast(list[dict[str, object]], analysis.get(list_key, []))
        if entry.get(id_key) not in {None, ""}
    }
    ordered_ids: list[object] = []
    if default_ids:
        for item_id in default_ids:
            if item_id not in ordered_ids:
                ordered_ids.append(item_id)
        extras = sorted(
            [item_id for item_id in discovered_ids if item_id not in ordered_ids],
            key=lambda value: str(value),
        )
        ordered_ids.extend(extras)
    else:
        ordered_ids = sorted(discovered_ids, key=lambda value: str(value))
    results: list[dict[str, object]] = []
    for item_id in ordered_ids:
        entries = [
            next(
                (
                    cast(dict[str, object], entry)
                    for entry in cast(list[dict[str, object]], analysis.get(list_key, []))
                    if entry.get(id_key) == item_id
                ),
                {},
            )
            for analysis in analyses
        ]
        results.append(
            {
                id_key: item_id,
                "coverage": _distribution([_as_float(entry.get("coverage")) for entry in entries]),
                "abs_ic": _distribution([_as_float(entry.get("abs_ic")) for entry in entries]),
                "overlap_rate": _distribution([_as_float(entry.get("overlap_rate")) for entry in entries]),
                "source_diversity_mean": _distribution([_as_float(entry.get("source_diversity_mean")) for entry in entries]),
                "source_count_mean": _distribution([_as_float(entry.get("source_count_mean")) for entry in entries]),
            }
        )
    return results


def _aggregate_nested_weight_summaries(
    analyses: list[dict[str, object]],
    *,
    section_key: str,
    list_key: str,
    id_key: str,
) -> list[dict[str, object]]:
    discovered_ids = {
        entry.get(id_key)
        for analysis in analyses
        for entry in cast(
            list[dict[str, object]],
            cast(dict[str, object], analysis.get(section_key, {})).get(list_key, []),
        )
        if entry.get(id_key) not in {None, ""}
    }
    ordered_ids = sorted(discovered_ids, key=lambda value: str(value))
    results: list[dict[str, object]] = []
    for item_id in ordered_ids:
        entries = [
            next(
                (
                    cast(dict[str, object], entry)
                    for entry in cast(
                        list[dict[str, object]],
                        cast(dict[str, object], analysis.get(section_key, {})).get(list_key, []),
                    )
                    if entry.get(id_key) == item_id
                ),
                {},
            )
            for analysis in analyses
        ]
        results.append(
            {
                id_key: item_id,
                "mean_weight": _distribution([_as_float(entry.get("mean_weight")) for entry in entries]),
                "min_weight": _distribution([_as_float(entry.get("min_weight")) for entry in entries]),
                "max_weight": _distribution([_as_float(entry.get("max_weight")) for entry in entries]),
                "group_count": _distribution([_as_float(entry.get("group_count")) for entry in entries]),
            }
        )
    return results


def summarize_news_feature_utility_comparison(runs: list[dict[str, object]]) -> dict[str, object]:
    analyses = [
        cast(dict[str, object], run.get("news_coverage_analysis"))
        if isinstance(run.get("news_coverage_analysis"), dict)
        else _fallback_analysis_from_flattened_run(run)
        for run in runs
    ]
    if not analyses:
        return {
            "return_alignment": "signal_day_close_to_close",
            "baseline_missing_rate": _distribution([]),
            "baseline_stale_rate": _distribution([]),
            "lookback_windows": [],
            "decay_halflives": [],
            "feature_variants": [],
            "weighted_variants": [],
            "weighting_mode_comparison": [],
            "session_buckets": [],
            "source_mix_buckets": [],
            "source_session_buckets": [],
            "source_diversity_buckets": [],
            "multi_source_weighting_comparison": [],
            "multi_source_weighting_improvement": {},
            "selected_weighting_mode_counts": {},
            "selected_weighting_variant_counts": {},
            "learned_weighting": {
                "fit_day_count": _distribution([]),
                "fallback_day_count": _distribution([]),
                "fallback_rate": _distribution([]),
                "global_score_mean": _distribution([]),
                "training_sample_count_mean": _distribution([]),
                "eligible_combo_count_mean": _distribution([]),
                "source_session_weights": [],
                "source_weights": [],
                "session_weights": [],
                "fallback_reasons": {},
                "comparison_ready_rate": 0.0,
            },
            "best_aggregation_window_counts": {},
            "best_decay_halflife_counts": {},
            "best_feature_variant_counts": {},
            "best_weighted_variant_counts": {},
            "best_session_bucket_counts": {},
            "best_source_mix_counts": {},
            "best_aggregation_coverage": _distribution([]),
            "best_aggregation_abs_ic": _distribution([]),
            "best_aggregation_overlap_rate": _distribution([]),
            "best_decay_coverage": _distribution([]),
            "best_decay_abs_ic": _distribution([]),
            "best_decay_overlap_rate": _distribution([]),
            "best_feature_variant_coverage": _distribution([]),
            "best_feature_variant_abs_ic": _distribution([]),
            "best_feature_variant_overlap_rate": _distribution([]),
            "best_weighted_variant_coverage": _distribution([]),
            "best_weighted_variant_abs_ic": _distribution([]),
            "best_weighted_variant_overlap_rate": _distribution([]),
            "best_session_coverage": _distribution([]),
            "best_session_abs_ic": _distribution([]),
            "best_session_overlap_rate": _distribution([]),
            "best_source_mix_coverage": _distribution([]),
            "best_source_mix_abs_ic": _distribution([]),
            "best_source_mix_overlap_rate": _distribution([]),
            "source_advantage_analysis": {},
            "mixed_session_conditions": {},
            "aggregation_improves_coverage_rate": 0.0,
            "aggregation_improves_utility_rate": 0.0,
            "decay_improves_coverage_rate": 0.0,
            "decay_improves_utility_rate": 0.0,
        }
    lookback_windows = _aggregate_bucket_summaries(
        analyses,
        list_key="lookback_windows",
        id_key="window_days",
        default_ids=list(LOOKBACK_WINDOWS),
    )
    decay_halflives = _aggregate_bucket_summaries(
        analyses,
        list_key="decay_halflives",
        id_key="halflife_days",
        default_ids=list(DECAY_HALFLIVES),
    )
    feature_variants = _aggregate_bucket_summaries(
        analyses,
        list_key="feature_variants",
        id_key="variant",
        default_ids=list(FEATURE_VARIANTS),
    )
    weighted_variants = _aggregate_bucket_summaries(
        analyses,
        list_key="weighted_variants",
        id_key="variant",
        default_ids=list(WEIGHTED_VARIANTS),
    )
    weighting_mode_comparison = _aggregate_bucket_summaries(
        analyses,
        list_key="weighting_mode_comparison",
        id_key="mode",
        default_ids=list(WEIGHTING_MODES),
    )
    session_buckets = _aggregate_bucket_summaries(
        analyses,
        list_key="session_buckets",
        id_key="session_bucket",
        default_ids=list(SESSION_BUCKET_ORDER),
    )
    source_mix_buckets = _aggregate_bucket_summaries(analyses, list_key="source_mix_buckets", id_key="source_mix")
    source_session_buckets = _aggregate_bucket_summaries(
        analyses,
        list_key="source_session_buckets",
        id_key="source_session_bucket",
    )
    source_diversity_buckets = _aggregate_bucket_summaries(
        analyses,
        list_key="source_diversity_buckets",
        id_key="diversity_bucket",
        default_ids=list(SOURCE_DIVERSITY_BUCKETS),
    )
    multi_source_weighting_comparison = _aggregate_bucket_summaries(
        analyses,
        list_key="multi_source_weighting_comparison",
        id_key="mode",
        default_ids=list(WEIGHTING_MODES),
    )
    learned_source_session_weights = _aggregate_nested_weight_summaries(
        analyses,
        section_key="learned_weighting",
        list_key="source_session_weights",
        id_key="combo_key",
    )
    learned_source_weights = _aggregate_nested_weight_summaries(
        analyses,
        section_key="learned_weighting",
        list_key="source_weights",
        id_key="source_name",
    )
    learned_session_weights = _aggregate_nested_weight_summaries(
        analyses,
        section_key="learned_weighting",
        list_key="session_weights",
        id_key="session_bucket",
    )

    source_advantage_entries = [
        cast(dict[str, object], analysis.get("source_advantage_analysis", {}))
        for analysis in analyses
    ]
    mixed_session_entries = [
        cast(dict[str, object], analysis.get("mixed_session_conditions", {}))
        for analysis in analyses
    ]
    learned_entries = [
        cast(dict[str, object], analysis.get("learned_weighting", {}))
        for analysis in analyses
    ]
    multi_source_weighting_entries = [
        cast(dict[str, object], analysis.get("multi_source_weighting_improvement", {}))
        for analysis in analyses
    ]
    return {
        "return_alignment": "signal_day_close_to_close",
        "baseline_missing_rate": _distribution([_as_float(analysis.get("baseline_missing_rate", 1.0)) for analysis in analyses]),
        "baseline_stale_rate": _distribution([_as_float(analysis.get("baseline_stale_rate", 0.0)) for analysis in analyses]),
        "lookback_windows": lookback_windows,
        "decay_halflives": decay_halflives,
        "feature_variants": feature_variants,
        "weighted_variants": weighted_variants,
        "weighting_mode_comparison": weighting_mode_comparison,
        "session_buckets": session_buckets,
        "source_mix_buckets": source_mix_buckets,
        "source_session_buckets": source_session_buckets,
        "source_diversity_buckets": source_diversity_buckets,
        "multi_source_weighting_comparison": multi_source_weighting_comparison,
        "multi_source_weighting_improvement": {
            "learned_abs_ic_delta_vs_unweighted": _distribution(
                [_as_float(entry.get("learned_abs_ic_delta_vs_unweighted", 0.0)) for entry in multi_source_weighting_entries]
            ),
            "learned_abs_ic_delta_vs_fixed": _distribution(
                [_as_float(entry.get("learned_abs_ic_delta_vs_fixed", 0.0)) for entry in multi_source_weighting_entries]
            ),
            "learned_overlap_rate_delta_vs_unweighted": _distribution(
                [_as_float(entry.get("learned_overlap_rate_delta_vs_unweighted", 0.0)) for entry in multi_source_weighting_entries]
            ),
            "learned_overlap_rate_delta_vs_fixed": _distribution(
                [_as_float(entry.get("learned_overlap_rate_delta_vs_fixed", 0.0)) for entry in multi_source_weighting_entries]
            ),
            "improves_joint_vs_unweighted_rate": float(
                sum(bool(entry.get("improves_joint_vs_unweighted", False)) for entry in multi_source_weighting_entries)
                / max(len(multi_source_weighting_entries), 1)
            ),
            "improves_joint_vs_fixed_rate": float(
                sum(bool(entry.get("improves_joint_vs_fixed", False)) for entry in multi_source_weighting_entries)
                / max(len(multi_source_weighting_entries), 1)
            ),
        },
        "selected_weighting_mode_counts": _value_counts([analysis.get("selected_weighting_mode") for analysis in analyses]),
        "selected_weighting_variant_counts": _value_counts([analysis.get("selected_weighting_variant") for analysis in analyses]),
        "learned_weighting": {
            "fit_day_count": _distribution([_as_float(entry.get("fit_day_count", 0.0)) for entry in learned_entries]),
            "fallback_day_count": _distribution([_as_float(entry.get("fallback_day_count", 0.0)) for entry in learned_entries]),
            "fallback_rate": _distribution([_as_float(entry.get("fallback_rate", 1.0)) for entry in learned_entries]),
            "global_score_mean": _distribution([_as_float(entry.get("global_score_mean", 0.0)) for entry in learned_entries]),
            "training_sample_count_mean": _distribution(
                [_as_float(entry.get("training_sample_count_mean", 0.0)) for entry in learned_entries]
            ),
            "eligible_combo_count_mean": _distribution(
                [_as_float(entry.get("eligible_combo_count_mean", 0.0)) for entry in learned_entries]
            ),
            "source_session_weights": learned_source_session_weights,
            "source_weights": learned_source_weights,
            "session_weights": learned_session_weights,
            "fallback_reasons": {
                reason: int(
                    sum(
                        _as_int(cast(dict[str, object], entry.get("fallback_reasons", {})).get(reason, 0))
                        for entry in learned_entries
                    )
                )
                for reason in sorted(
                    {
                        reason
                        for entry in learned_entries
                        for reason in cast(dict[str, object], entry.get("fallback_reasons", {})).keys()
                    }
                )
            },
            "comparison_ready_rate": float(
                sum(bool(entry.get("comparison_ready", False)) for entry in learned_entries) / max(len(learned_entries), 1)
            ),
        },
        "best_aggregation_window_counts": _value_counts([analysis.get("best_aggregation_window") for analysis in analyses]),
        "best_decay_halflife_counts": _value_counts([analysis.get("best_decay_halflife") for analysis in analyses]),
        "best_feature_variant_counts": _value_counts([analysis.get("best_feature_variant") for analysis in analyses]),
        "best_weighted_variant_counts": _value_counts([analysis.get("best_weighted_variant") for analysis in analyses]),
        "best_session_bucket_counts": _value_counts([analysis.get("best_session_bucket") for analysis in analyses]),
        "best_source_mix_counts": _value_counts([analysis.get("best_source_mix") for analysis in analyses]),
        "best_aggregation_coverage": _distribution([_as_float(analysis.get("best_aggregation_coverage", 0.0)) for analysis in analyses]),
        "best_aggregation_abs_ic": _distribution([_as_float(analysis.get("best_aggregation_abs_ic", 0.0)) for analysis in analyses]),
        "best_aggregation_overlap_rate": _distribution([_as_float(analysis.get("best_aggregation_overlap_rate", 0.0)) for analysis in analyses]),
        "best_decay_coverage": _distribution([_as_float(analysis.get("best_decay_coverage", 0.0)) for analysis in analyses]),
        "best_decay_abs_ic": _distribution([_as_float(analysis.get("best_decay_abs_ic", 0.0)) for analysis in analyses]),
        "best_decay_overlap_rate": _distribution([_as_float(analysis.get("best_decay_overlap_rate", 0.0)) for analysis in analyses]),
        "best_feature_variant_coverage": _distribution([_as_float(analysis.get("best_feature_variant_coverage", 0.0)) for analysis in analyses]),
        "best_feature_variant_abs_ic": _distribution([_as_float(analysis.get("best_feature_variant_abs_ic", 0.0)) for analysis in analyses]),
        "best_feature_variant_overlap_rate": _distribution([_as_float(analysis.get("best_feature_variant_overlap_rate", 0.0)) for analysis in analyses]),
        "best_weighted_variant_coverage": _distribution([_as_float(analysis.get("best_weighted_variant_coverage", 0.0)) for analysis in analyses]),
        "best_weighted_variant_abs_ic": _distribution([_as_float(analysis.get("best_weighted_variant_abs_ic", 0.0)) for analysis in analyses]),
        "best_weighted_variant_overlap_rate": _distribution([_as_float(analysis.get("best_weighted_variant_overlap_rate", 0.0)) for analysis in analyses]),
        "best_session_coverage": _distribution([_as_float(analysis.get("best_session_coverage", 0.0)) for analysis in analyses]),
        "best_session_abs_ic": _distribution([_as_float(analysis.get("best_session_abs_ic", 0.0)) for analysis in analyses]),
        "best_session_overlap_rate": _distribution([_as_float(analysis.get("best_session_overlap_rate", 0.0)) for analysis in analyses]),
        "best_source_mix_coverage": _distribution([_as_float(analysis.get("best_source_mix_coverage", 0.0)) for analysis in analyses]),
        "best_source_mix_abs_ic": _distribution([_as_float(analysis.get("best_source_mix_abs_ic", 0.0)) for analysis in analyses]),
        "best_source_mix_overlap_rate": _distribution([_as_float(analysis.get("best_source_mix_overlap_rate", 0.0)) for analysis in analyses]),
        "source_advantage_analysis": {
            "dominant_source_mix_counts": _value_counts([entry.get("dominant_source_mix") for entry in source_advantage_entries]),
            "best_multi_source_mix_counts": _value_counts([entry.get("best_multi_source_mix") for entry in source_advantage_entries]),
            "dominant_reason_counts": _value_counts([entry.get("dominant_reason") for entry in source_advantage_entries]),
            "google_single_source_present_rate": float(
                sum(bool(entry.get("google_single_source_present", False)) for entry in source_advantage_entries)
                / max(len(source_advantage_entries), 1)
            ),
            "google_single_source_abs_ic": _distribution([_as_float(entry.get("google_single_source_abs_ic", 0.0)) for entry in source_advantage_entries]),
            "google_single_source_overlap_rate": _distribution([_as_float(entry.get("google_single_source_overlap_rate", 0.0)) for entry in source_advantage_entries]),
            "best_multi_source_abs_ic": _distribution([_as_float(entry.get("best_multi_source_abs_ic", 0.0)) for entry in source_advantage_entries]),
            "best_multi_source_overlap_rate": _distribution([_as_float(entry.get("best_multi_source_overlap_rate", 0.0)) for entry in source_advantage_entries]),
            "google_abs_ic_advantage_vs_best_multi_source": _distribution(
                [_as_float(entry.get("google_abs_ic_advantage_vs_best_multi_source", 0.0)) for entry in source_advantage_entries]
            ),
            "google_overlap_advantage_vs_best_multi_source": _distribution(
                [_as_float(entry.get("google_overlap_advantage_vs_best_multi_source", 0.0)) for entry in source_advantage_entries]
            ),
            "google_coverage_advantage_vs_best_multi_source": _distribution(
                [_as_float(entry.get("google_coverage_advantage_vs_best_multi_source", 0.0)) for entry in source_advantage_entries]
            ),
        },
        "mixed_session_conditions": {
            "mixed_session_present_rate": float(
                sum(bool(entry.get("mixed_session_present", False)) for entry in mixed_session_entries)
                / max(len(mixed_session_entries), 1)
            ),
            "mixed_session_is_best_rate": float(
                sum(bool(entry.get("mixed_session_is_best", False)) for entry in mixed_session_entries)
                / max(len(mixed_session_entries), 1)
            ),
            "best_non_mixed_session_counts": _value_counts([entry.get("best_non_mixed_session") for entry in mixed_session_entries]),
            "mixed_session_best_source_mix_counts": _value_counts([entry.get("mixed_session_best_source_mix") for entry in mixed_session_entries]),
            "mixed_session_abs_ic": _distribution([_as_float(entry.get("mixed_session_abs_ic", 0.0)) for entry in mixed_session_entries]),
            "mixed_session_overlap_rate": _distribution([_as_float(entry.get("mixed_session_overlap_rate", 0.0)) for entry in mixed_session_entries]),
            "best_non_mixed_abs_ic": _distribution([_as_float(entry.get("best_non_mixed_abs_ic", 0.0)) for entry in mixed_session_entries]),
            "best_non_mixed_overlap_rate": _distribution([_as_float(entry.get("best_non_mixed_overlap_rate", 0.0)) for entry in mixed_session_entries]),
            "mixed_session_abs_ic_advantage": _distribution([_as_float(entry.get("mixed_session_abs_ic_advantage", 0.0)) for entry in mixed_session_entries]),
            "mixed_session_overlap_advantage": _distribution([_as_float(entry.get("mixed_session_overlap_advantage", 0.0)) for entry in mixed_session_entries]),
        },
        "aggregation_improves_coverage_rate": float(
            sum(bool(analysis.get("aggregation_improves_coverage", False)) for analysis in analyses) / max(len(analyses), 1)
        ),
        "aggregation_improves_utility_rate": float(
            sum(bool(analysis.get("aggregation_improves_utility", False)) for analysis in analyses) / max(len(analyses), 1)
        ),
        "decay_improves_coverage_rate": float(
            sum(bool(analysis.get("decay_improves_coverage", False)) for analysis in analyses) / max(len(analyses), 1)
        ),
        "decay_improves_utility_rate": float(
            sum(bool(analysis.get("decay_improves_utility", False)) for analysis in analyses) / max(len(analyses), 1)
        ),
    }
