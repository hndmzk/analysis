from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json

import numpy as np
import pandas as pd

from market_prediction_agent.config import LearnedWeightingConfig


BREAKDOWN_COLUMN = "source_session_breakdown"
COMBO_SEPARATOR = "::"


@dataclass(slots=True)
class LearnedWeightingResult:
    sentiment: pd.Series
    relevance: pd.Series
    summary: dict[str, object]


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float | np.integer | np.floating):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _as_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        if np.isnan(value) or np.isinf(value):
            return default
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                return int(float(value))
            except ValueError:
                return default
    return default


def _correlation_or_zero(left: pd.Series, right: pd.Series) -> float:
    aligned = pd.concat([left, right], axis=1).dropna()
    if len(aligned) < 2:
        return 0.0
    if float(aligned.iloc[:, 0].std(ddof=0) or 0.0) == 0.0:
        return 0.0
    if float(aligned.iloc[:, 1].std(ddof=0) or 0.0) == 0.0:
        return 0.0
    correlation = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return 0.0 if pd.isna(correlation) else float(correlation)


def _target_score(*, abs_ic: float, overlap_rate: float, target: str) -> float:
    if target == "overlap_rate":
        return max(overlap_rate, 0.0)
    if target == "combined":
        return max(abs_ic, 0.0) * max(overlap_rate, 0.0)
    return max(abs_ic, 0.0)


def _normalize_with_min_weight(raw_weights: dict[str, float], min_weight: float) -> dict[str, float]:
    positive = {key: max(float(value), 0.0) for key, value in raw_weights.items()}
    if not positive:
        return {}
    total = float(sum(positive.values()))
    if total <= 0.0:
        return {}
    normalized = {key: value / total for key, value in positive.items()}
    floor = max(float(min_weight), 0.0)
    if floor <= 0.0:
        return normalized
    if floor * len(normalized) >= 1.0:
        uniform = 1.0 / max(len(normalized), 1)
        return {key: uniform for key in normalized}
    fixed: dict[str, float] = {}
    remaining = set(normalized)
    while remaining:
        remaining_total = float(sum(normalized[key] for key in remaining))
        remaining_mass = 1.0 - float(sum(fixed.values()))
        if remaining_total <= 0.0 or remaining_mass <= 0.0:
            uniform = 1.0 / max(len(normalized), 1)
            return {key: uniform for key in normalized}
        projected = {key: normalized[key] / remaining_total * remaining_mass for key in remaining}
        low = [key for key, value in projected.items() if value < floor]
        if not low:
            fixed.update(projected)
            break
        for key in low:
            fixed[key] = floor
            remaining.remove(key)
        if float(sum(fixed.values())) >= 1.0:
            uniform = 1.0 / max(len(normalized), 1)
            return {key: uniform for key in normalized}
    total_fixed = float(sum(fixed.values()))
    if total_fixed <= 0.0:
        return normalized
    return {key: value / total_fixed for key, value in fixed.items()}


def _split_sources(*, source_mix: object, source: object) -> list[str]:
    candidates = [part.strip() for part in str(source_mix or "").split("|") if part.strip() and part.strip() != "none"]
    if candidates:
        return sorted(dict.fromkeys(candidates))
    fallback = str(source or "").strip()
    if fallback and fallback != "none":
        return [fallback]
    return ["unknown"]


def _parse_breakdown(value: object) -> dict[str, dict[str, object]]:
    if isinstance(value, dict):
        return {
            str(key): dict(item)
            for key, item in value.items()
            if isinstance(key, str) and isinstance(item, dict)
        }
    if not isinstance(value, str):
        return {}
    text = value.strip()
    if not text or text == "{}":
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        str(key): dict(item)
        for key, item in payload.items()
        if isinstance(key, str) and isinstance(item, dict)
    }


def _fallback_breakdown_for_row(row: pd.Series) -> dict[str, dict[str, object]]:
    sources = _split_sources(source_mix=row.get("source_mix"), source=row.get("source"))
    session_bucket = str(row.get("session_bucket", "unknown") or "unknown")
    headline_count = max(_as_float(row.get("headline_count", 0.0), 0.0), 0.0)
    relevance = float(np.clip(_as_float(row.get("relevance_score", 0.0), 0.0), 0.0, 1.0))
    sentiment = _as_float(row.get("sentiment_score", 0.0), 0.0)
    base_weight_sum = headline_count * relevance
    if base_weight_sum <= 0.0 and headline_count > 0.0:
        base_weight_sum = headline_count
    if base_weight_sum <= 0.0:
        return {}
    per_source_weight = base_weight_sum / max(len(sources), 1)
    per_source_headline_count = max(headline_count / max(len(sources), 1), 0.0)
    breakdown: dict[str, dict[str, object]] = {}
    for source_name in sources:
        combo_key = f"{source_name}{COMBO_SEPARATOR}{session_bucket}"
        breakdown[combo_key] = {
            "source_name": source_name,
            "session_bucket": session_bucket,
            "base_weight_sum": per_source_weight,
            "sentiment_weighted_sum": sentiment * per_source_weight,
            "relevance_weighted_sum": relevance * per_source_weight,
            "headline_count": int(round(per_source_headline_count)),
            "article_count": int(max(round(per_source_headline_count), 1)),
        }
    return breakdown


def expand_source_session_breakdown(news: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "ticker",
        "date",
        "source_name",
        "session_bucket",
        "combo_key",
        "base_weight_sum",
        "sentiment_weighted_sum",
        "relevance_weighted_sum",
        "headline_count",
        "article_count",
    ]
    if news.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    signal_dates = pd.to_datetime(news["available_at"], utc=True).dt.normalize()
    for (_, row), signal_date in zip(news.iterrows(), signal_dates, strict=True):
        breakdown = _parse_breakdown(row.get(BREAKDOWN_COLUMN))
        if not breakdown:
            breakdown = _fallback_breakdown_for_row(row)
        ticker = str(row.get("ticker", ""))
        if not ticker:
            continue
        for combo_key, stats in breakdown.items():
            source_name = str(stats.get("source_name") or combo_key.split(COMBO_SEPARATOR, maxsplit=1)[0])
            if COMBO_SEPARATOR in combo_key:
                _, inferred_session = combo_key.split(COMBO_SEPARATOR, maxsplit=1)
            else:
                inferred_session = str(row.get("session_bucket", "unknown") or "unknown")
            session_bucket = str(stats.get("session_bucket") or inferred_session)
            rows.append(
                {
                    "ticker": ticker,
                    "date": signal_date,
                    "source_name": source_name,
                    "session_bucket": session_bucket,
                    "combo_key": combo_key,
                    "base_weight_sum": max(_as_float(stats.get("base_weight_sum"), 0.0), 0.0),
                    "sentiment_weighted_sum": _as_float(stats.get("sentiment_weighted_sum"), 0.0),
                    "relevance_weighted_sum": _as_float(stats.get("relevance_weighted_sum"), 0.0),
                    "headline_count": int(max(_as_float(stats.get("headline_count"), 0.0), 0.0)),
                    "article_count": int(max(_as_float(stats.get("article_count"), 0.0), 0.0)),
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    expanded = pd.DataFrame(rows)
    aggregated = (
        expanded.groupby(["ticker", "date", "source_name", "session_bucket", "combo_key"], as_index=False)
        .agg(
            base_weight_sum=("base_weight_sum", "sum"),
            sentiment_weighted_sum=("sentiment_weighted_sum", "sum"),
            relevance_weighted_sum=("relevance_weighted_sum", "sum"),
            headline_count=("headline_count", "sum"),
            article_count=("article_count", "sum"),
        )
        .reset_index(drop=True)
    )
    return aggregated[columns]


def _fit_combo_scores(
    history: pd.DataFrame,
    *,
    config: LearnedWeightingConfig,
) -> dict[str, object]:
    required_total_samples = max(int(config.min_samples), 1)
    if history.empty:
        return {"valid": False, "reason": "insufficient_total_samples", "total_samples": 0}
    history = history.copy()
    history["combo_signal"] = (
        history["sentiment_weighted_sum"] / history["base_weight_sum"].replace(0.0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    history["active"] = history["base_weight_sum"] > 0.0
    total_samples = int((history["active"] & history["signal_day_return"].notna()).sum())
    if total_samples < required_total_samples:
        return {"valid": False, "reason": "insufficient_total_samples", "total_samples": total_samples}
    combo_stats: list[dict[str, object]] = []
    for combo_key, group in history.groupby("combo_key", as_index=False):
        active = group["active"].fillna(False).astype(bool)
        sample_count = int((active & group["signal_day_return"].notna()).sum())
        if sample_count <= 0:
            continue
        abs_ic = abs(_correlation_or_zero(group["combo_signal"].where(active), group["signal_day_return"]))
        signal_days = int(active.sum())
        overlap_rate = float(sample_count / signal_days) if signal_days > 0 else 0.0
        target_score = _target_score(abs_ic=abs_ic, overlap_rate=overlap_rate, target=config.target)
        combo_stats.append(
            {
                "combo_key": str(combo_key),
                "source_name": str(group["source_name"].iloc[0]),
                "session_bucket": str(group["session_bucket"].iloc[0]),
                "sample_count": sample_count,
                "signal_days": signal_days,
                "abs_ic": abs_ic,
                "overlap_rate": overlap_rate,
                "target_score": target_score,
            }
        )
    if not combo_stats:
        return {"valid": False, "reason": "no_combo_samples", "total_samples": total_samples}
    eligible_combo_count = int(
        sum(_as_int(item.get("sample_count")) >= required_total_samples for item in combo_stats)
    )
    if eligible_combo_count <= 0:
        return {
            "valid": False,
            "reason": "insufficient_combo_samples",
            "total_samples": total_samples,
            "combo_stats": combo_stats,
        }
    weighted_target_sum = float(
        sum(_as_float(item.get("target_score")) * _as_int(item.get("sample_count")) for item in combo_stats)
    )
    sample_weight_sum = float(sum(_as_int(item.get("sample_count")) for item in combo_stats))
    global_score = weighted_target_sum / sample_weight_sum if sample_weight_sum > 0.0 else 0.0
    if global_score <= 0.0 and max(_as_float(item.get("target_score")) for item in combo_stats) <= 0.0:
        return {
            "valid": False,
            "reason": "zero_target_score",
            "total_samples": total_samples,
            "combo_stats": combo_stats,
        }
    shrink_lambda = max(float(config.regularization_lambda), 0.0)
    shrunk_scores: dict[str, float] = {}
    enriched_stats: list[dict[str, object]] = []
    for item in combo_stats:
        sample_count = _as_int(item.get("sample_count"))
        sample_score = _as_float(item.get("target_score"))
        shrunk_score = (
            (sample_count / (sample_count + shrink_lambda)) * sample_score
            + (shrink_lambda / (sample_count + shrink_lambda)) * global_score
            if sample_count + shrink_lambda > 0.0
            else sample_score
        )
        shrunk_scores[str(item["combo_key"])] = max(shrunk_score, 0.0)
        enriched = dict(item)
        enriched["shrunk_score"] = max(shrunk_score, 0.0)
        enriched_stats.append(enriched)
    if max(shrunk_scores.values(), default=0.0) <= 0.0:
        return {
            "valid": False,
            "reason": "zero_shrunk_score",
            "total_samples": total_samples,
            "combo_stats": enriched_stats,
        }
    normalized_weights = _normalize_with_min_weight(shrunk_scores, float(config.min_weight))
    return {
        "valid": True,
        "reason": "learned",
        "total_samples": total_samples,
        "eligible_combo_count": eligible_combo_count,
        "global_score": float(global_score),
        "shrunk_scores": shrunk_scores,
        "normalized_weights": normalized_weights,
        "combo_stats": enriched_stats,
    }


def fit_learned_source_session_weights(
    *,
    history: pd.DataFrame,
    config: LearnedWeightingConfig,
) -> dict[str, object]:
    return _fit_combo_scores(history, config=config)


def _fallback_reason_counts(counter: Counter[str]) -> dict[str, int]:
    return {reason: int(count) for reason, count in sorted(counter.items())}


def build_walk_forward_learned_weighting(
    *,
    news_panel: pd.DataFrame,
    news: pd.DataFrame,
    config: LearnedWeightingConfig,
    fallback_sentiment: pd.Series,
    fallback_relevance: pd.Series,
) -> LearnedWeightingResult:
    learned_sentiment = pd.to_numeric(fallback_sentiment, errors="coerce").fillna(0.0).copy()
    learned_relevance = pd.to_numeric(fallback_relevance, errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0).copy()
    expanded = expand_source_session_breakdown(news)
    summary: dict[str, object] = {
        "mode": "learned",
        "target": str(config.target),
        "lookback_days": int(config.lookback_days),
        "regularization_lambda": float(config.regularization_lambda),
        "min_samples": int(config.min_samples),
        "min_weight": float(config.min_weight),
        "fallback_mode": str(config.fallback_mode),
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
    }
    if news_panel.empty or expanded.empty:
        return LearnedWeightingResult(sentiment=learned_sentiment, relevance=learned_relevance, summary=summary)

    returns = news_panel.loc[:, ["ticker", "date", "signal_day_return"]].copy()
    history_frame = expanded.merge(returns, on=["ticker", "date"], how="left")
    combo_weight_stats: dict[str, dict[str, object]] = {}
    source_weight_stats: dict[str, dict[str, float | int | str]] = {}
    session_weight_stats: dict[str, dict[str, float | int | str]] = {}
    fallback_reasons: Counter[str] = Counter()
    fit_day_count = 0
    fallback_day_count = 0
    global_score_sum = 0.0
    training_sample_count_sum = 0.0
    eligible_combo_count_sum = 0.0
    applied_group_count = 0
    eval_dates = sorted(pd.to_datetime(expanded["date"], utc=True).unique().tolist())
    result_rows: list[pd.DataFrame] = []
    for eval_date_value in eval_dates:
        eval_date = pd.Timestamp(eval_date_value)
        history_start = (eval_date - pd.tseries.offsets.BDay(max(int(config.lookback_days), 1))).normalize()
        fit = _fit_combo_scores(
            history_frame.loc[(history_frame["date"] < eval_date) & (history_frame["date"] >= history_start)],
            config=config,
        )
        if not bool(fit.get("valid", False)):
            fallback_day_count += 1
            fallback_reasons[str(fit.get("reason", "fallback"))] += 1
            continue
        fit_day_count += 1
        global_score_sum += _as_float(fit.get("global_score"), 0.0)
        training_sample_count_sum += _as_float(fit.get("total_samples"), 0.0)
        eligible_combo_count_sum += _as_float(fit.get("eligible_combo_count"), 0.0)
        combo_stats_by_key: dict[str, dict[str, object]] = {}
        combo_stats_raw = fit.get("combo_stats", [])
        if isinstance(combo_stats_raw, list):
            for item in combo_stats_raw:
                if isinstance(item, dict) and "combo_key" in item:
                    combo_stats_by_key[str(item["combo_key"])] = dict(item)
        current = expanded.loc[expanded["date"] == eval_date].copy()
        if current.empty:
            continue
        for (_, _), group in current.groupby(["ticker", "date"], as_index=False):
            raw_scores = {
                str(combo_key): _as_float(
                    dict(combo_stats_by_key.get(str(combo_key), {})).get("shrunk_score"),
                    _as_float(fit.get("global_score"), 0.0),
                )
                for combo_key in group["combo_key"].astype(str).tolist()
            }
            applied_weights = _normalize_with_min_weight(raw_scores, float(config.min_weight))
            if not applied_weights:
                continue
            weighted_base = pd.Series(
                [applied_weights.get(str(combo_key), 0.0) for combo_key in group["combo_key"].astype(str)],
                index=group.index,
                dtype=float,
            ) * pd.to_numeric(group["base_weight_sum"], errors="coerce").fillna(0.0).clip(lower=0.0)
            combo_weight_series = pd.Series(
                [applied_weights.get(str(combo_key), 0.0) for combo_key in group["combo_key"].astype(str)],
                index=group.index,
                dtype=float,
            )
            denominator = float(weighted_base.sum())
            if denominator <= 0.0:
                continue
            weighted_sentiment_sum = float(
                (pd.to_numeric(group["sentiment_weighted_sum"], errors="coerce").fillna(0.0) * combo_weight_series).sum()
            )
            weighted_relevance_sum = float(
                (pd.to_numeric(group["relevance_weighted_sum"], errors="coerce").fillna(0.0) * combo_weight_series).sum()
            )
            applied_group_count += 1
            result_rows.append(
                pd.DataFrame(
                    {
                        "ticker": [str(group["ticker"].iloc[0])],
                        "date": [eval_date],
                        "sentiment_score_learned": [weighted_sentiment_sum / denominator],
                        "relevance_score_learned": [weighted_relevance_sum / denominator],
                    }
                )
            )
            source_daily_weights: dict[str, float] = {}
            session_daily_weights: dict[str, float] = {}
            for combo_key, weight in applied_weights.items():
                combo_stats = combo_stats_by_key.get(combo_key, {})
                source_name = str(combo_stats.get("source_name", combo_key.split(COMBO_SEPARATOR, maxsplit=1)[0]))
                session_bucket = str(
                    combo_stats.get(
                        "session_bucket",
                        combo_key.split(COMBO_SEPARATOR, maxsplit=1)[1] if COMBO_SEPARATOR in combo_key else "unknown",
                    )
                )
                combo_entry = combo_weight_stats.setdefault(
                    combo_key,
                    {
                        "combo_key": combo_key,
                        "source_name": source_name,
                        "session_bucket": session_bucket,
                        "weight_sum": 0.0,
                        "min_weight": float("inf"),
                        "max_weight": 0.0,
                        "group_count": 0,
                        "sample_count_sum": 0.0,
                        "sample_score_sum": 0.0,
                        "shrunk_score_sum": 0.0,
                    },
                )
                combo_entry["weight_sum"] = _as_float(combo_entry.get("weight_sum"), 0.0) + float(weight)
                combo_entry["min_weight"] = min(_as_float(combo_entry.get("min_weight"), float("inf")), float(weight))
                combo_entry["max_weight"] = max(_as_float(combo_entry.get("max_weight"), 0.0), float(weight))
                combo_entry["group_count"] = _as_int(combo_entry.get("group_count"), 0) + 1
                combo_entry["sample_count_sum"] = _as_float(combo_entry.get("sample_count_sum"), 0.0) + _as_float(
                    combo_stats.get("sample_count"),
                    0.0,
                )
                combo_entry["sample_score_sum"] = _as_float(combo_entry.get("sample_score_sum"), 0.0) + _as_float(
                    combo_stats.get("target_score"),
                    0.0,
                )
                combo_entry["shrunk_score_sum"] = _as_float(combo_entry.get("shrunk_score_sum"), 0.0) + _as_float(
                    combo_stats.get("shrunk_score"),
                    0.0,
                )
                source_daily_weights[source_name] = source_daily_weights.get(source_name, 0.0) + float(weight)
                session_daily_weights[session_bucket] = session_daily_weights.get(session_bucket, 0.0) + float(weight)
            for source_name, weight in source_daily_weights.items():
                source_entry = source_weight_stats.setdefault(
                    source_name,
                    {
                        "source_name": source_name,
                        "weight_sum": 0.0,
                        "min_weight": float("inf"),
                        "max_weight": 0.0,
                        "group_count": 0,
                    },
                )
                source_entry["weight_sum"] = _as_float(source_entry.get("weight_sum"), 0.0) + float(weight)
                source_entry["min_weight"] = min(_as_float(source_entry.get("min_weight"), float("inf")), float(weight))
                source_entry["max_weight"] = max(_as_float(source_entry.get("max_weight"), 0.0), float(weight))
                source_entry["group_count"] = _as_int(source_entry.get("group_count"), 0) + 1
            for session_bucket, weight in session_daily_weights.items():
                session_entry = session_weight_stats.setdefault(
                    session_bucket,
                    {
                        "session_bucket": session_bucket,
                        "weight_sum": 0.0,
                        "min_weight": float("inf"),
                        "max_weight": 0.0,
                        "group_count": 0,
                    },
                )
                session_entry["weight_sum"] = _as_float(session_entry.get("weight_sum"), 0.0) + float(weight)
                session_entry["min_weight"] = min(_as_float(session_entry.get("min_weight"), float("inf")), float(weight))
                session_entry["max_weight"] = max(_as_float(session_entry.get("max_weight"), 0.0), float(weight))
                session_entry["group_count"] = _as_int(session_entry.get("group_count"), 0) + 1

    if result_rows:
        learned_frame = pd.concat(result_rows, ignore_index=True)
        merged = news_panel.loc[:, ["ticker", "date"]].merge(learned_frame, on=["ticker", "date"], how="left")
        learned_sentiment = pd.to_numeric(
            merged["sentiment_score_learned"].combine_first(learned_sentiment),
            errors="coerce",
        ).fillna(0.0)
        learned_relevance = pd.to_numeric(
            merged["relevance_score_learned"].combine_first(learned_relevance),
            errors="coerce",
        ).fillna(0.0).clip(lower=0.0, upper=1.0)

    source_session_weights = [
        {
            "combo_key": str(entry["combo_key"]),
            "source_name": str(entry["source_name"]),
            "session_bucket": str(entry["session_bucket"]),
            "mean_weight": _as_float(entry.get("weight_sum"), 0.0) / max(_as_int(entry.get("group_count"), 0), 1),
            "min_weight": 0.0
            if _as_float(entry.get("min_weight"), 0.0) == float("inf")
            else _as_float(entry.get("min_weight"), 0.0),
            "max_weight": _as_float(entry.get("max_weight"), 0.0),
            "group_count": _as_int(entry.get("group_count"), 0),
            "sample_count_mean": _as_float(entry.get("sample_count_sum"), 0.0)
            / max(_as_int(entry.get("group_count"), 0), 1),
            "sample_score_mean": _as_float(entry.get("sample_score_sum"), 0.0)
            / max(_as_int(entry.get("group_count"), 0), 1),
            "shrunk_score_mean": _as_float(entry.get("shrunk_score_sum"), 0.0)
            / max(_as_int(entry.get("group_count"), 0), 1),
        }
        for entry in combo_weight_stats.values()
    ]
    source_session_weights.sort(key=lambda item: (-_as_float(item.get("mean_weight"), 0.0), str(item.get("combo_key"))))
    source_weights = [
        {
            "source_name": str(entry["source_name"]),
            "mean_weight": _as_float(entry.get("weight_sum"), 0.0) / max(_as_int(entry.get("group_count"), 0), 1),
            "min_weight": 0.0
            if _as_float(entry.get("min_weight"), 0.0) == float("inf")
            else _as_float(entry.get("min_weight"), 0.0),
            "max_weight": _as_float(entry.get("max_weight"), 0.0),
            "group_count": _as_int(entry.get("group_count"), 0),
        }
        for entry in source_weight_stats.values()
    ]
    source_weights.sort(key=lambda item: (-_as_float(item.get("mean_weight"), 0.0), str(item.get("source_name"))))
    session_weights = [
        {
            "session_bucket": str(entry["session_bucket"]),
            "mean_weight": _as_float(entry.get("weight_sum"), 0.0) / max(_as_int(entry.get("group_count"), 0), 1),
            "min_weight": 0.0
            if _as_float(entry.get("min_weight"), 0.0) == float("inf")
            else _as_float(entry.get("min_weight"), 0.0),
            "max_weight": _as_float(entry.get("max_weight"), 0.0),
            "group_count": _as_int(entry.get("group_count"), 0),
        }
        for entry in session_weight_stats.values()
    ]
    session_weights.sort(key=lambda item: (-_as_float(item.get("mean_weight"), 0.0), str(item.get("session_bucket"))))
    total_eval_days = fit_day_count + fallback_day_count
    summary.update(
        {
            "fit_day_count": fit_day_count,
            "fallback_day_count": fallback_day_count,
            "fallback_rate": float(fallback_day_count / max(total_eval_days, 1)),
            "global_score_mean": float(global_score_sum / max(fit_day_count, 1)),
            "training_sample_count_mean": float(training_sample_count_sum / max(fit_day_count, 1)),
            "eligible_combo_count_mean": float(eligible_combo_count_sum / max(fit_day_count, 1)),
            "applied_group_count": applied_group_count,
            "source_session_weights": source_session_weights,
            "source_weights": source_weights,
            "session_weights": session_weights,
            "fallback_reasons": _fallback_reason_counts(fallback_reasons),
            "comparison_ready": bool(fit_day_count > 0 and applied_group_count > 0),
        }
    )
    return LearnedWeightingResult(sentiment=learned_sentiment, relevance=learned_relevance, summary=summary)
