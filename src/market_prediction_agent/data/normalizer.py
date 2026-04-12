from __future__ import annotations

import pandas as pd

from market_prediction_agent.utils.time_utils import normalize_to_utc, to_utc_timestamp


OHLCV_COLUMNS = [
    "ticker",
    "timestamp_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
    "fetched_at",
    "data_age_hours",
    "stale_data_flag",
]
MACRO_COLUMNS = ["series_id", "date", "value", "available_at", "source"]
NEWS_COLUMNS = [
    "ticker",
    "published_at",
    "available_at",
    "session_bucket",
    "sentiment_score",
    "sentiment_score_unweighted",
    "sentiment_score_source_weighted",
    "sentiment_score_session_weighted",
    "sentiment_score_source_session_weighted",
    "relevance_score",
    "relevance_score_unweighted",
    "relevance_score_source_weighted",
    "relevance_score_session_weighted",
    "relevance_score_source_session_weighted",
    "headline_count",
    "mapping_confidence",
    "novelty_score",
    "source_diversity",
    "source_count",
    "source_mix",
    "source",
    "source_session_breakdown",
    "fetched_at",
    "data_age_hours",
    "stale_data_flag",
]
FUNDAMENTAL_COLUMNS = [
    "ticker",
    "report_date",
    "available_at",
    "revenue_growth",
    "earnings_yield",
    "debt_to_equity",
    "profitability",
    "source",
    "fetched_at",
    "data_age_hours",
    "stale_data_flag",
]
SECTOR_MAP_COLUMNS = [
    "ticker",
    "sector",
    "source",
    "fetched_at",
    "data_age_hours",
    "stale_data_flag",
]


def normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "timestamp_utc", "open", "high", "low", "close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
    normalized = frame.copy()
    normalized["timestamp_utc"] = normalize_to_utc(normalized["timestamp_utc"])
    normalized["fetched_at"] = normalize_to_utc(normalized.get("fetched_at", normalized["timestamp_utc"]))
    for column in ["open", "high", "low", "close", "volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["source"] = normalized.get("source", "unknown")
    if "data_age_hours" in normalized.columns:
        normalized["data_age_hours"] = pd.to_numeric(normalized["data_age_hours"], errors="coerce").fillna(0.0)
    else:
        normalized["data_age_hours"] = 0.0
    normalized["stale_data_flag"] = normalized.get(
        "stale_data_flag", pd.Series(False, index=normalized.index, dtype=bool)
    ).astype(bool)
    normalized = normalized.dropna(subset=["ticker", "timestamp_utc", "open", "high", "low", "close", "volume"])
    normalized = normalized.sort_values(["ticker", "timestamp_utc"]).drop_duplicates(["ticker", "timestamp_utc"])
    return normalized[OHLCV_COLUMNS].reset_index(drop=True)


def normalize_macro(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"series_id", "date", "value", "available_at"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing macro columns: {sorted(missing)}")
    normalized = frame.copy()
    normalized["date"] = normalize_to_utc(normalized["date"])
    normalized["available_at"] = normalize_to_utc(normalized["available_at"])
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    normalized["source"] = normalized.get("source", "unknown")
    normalized = normalized.dropna(subset=["series_id", "date", "value", "available_at"])
    normalized = normalized.sort_values(["series_id", "date"]).drop_duplicates(["series_id", "date"])
    return normalized[MACRO_COLUMNS].reset_index(drop=True)


def normalize_news(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "published_at", "available_at", "sentiment_score", "relevance_score", "headline_count"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing news columns: {sorted(missing)}")
    normalized = frame.copy()
    normalized["published_at"] = normalize_to_utc(normalized["published_at"])
    normalized["available_at"] = normalize_to_utc(normalized["available_at"])
    normalized["fetched_at"] = normalize_to_utc(normalized.get("fetched_at", normalized["available_at"]))
    normalized["sentiment_score"] = pd.to_numeric(normalized["sentiment_score"], errors="coerce")
    normalized["relevance_score"] = pd.to_numeric(normalized["relevance_score"], errors="coerce")
    normalized["headline_count"] = pd.to_numeric(normalized["headline_count"], errors="coerce")
    weighted_column_defaults = {
        "sentiment_score_unweighted": normalized["sentiment_score"],
        "sentiment_score_source_weighted": normalized["sentiment_score"],
        "sentiment_score_session_weighted": normalized["sentiment_score"],
        "sentiment_score_source_session_weighted": normalized["sentiment_score"],
        "relevance_score_unweighted": normalized["relevance_score"],
        "relevance_score_source_weighted": normalized["relevance_score"],
        "relevance_score_session_weighted": normalized["relevance_score"],
        "relevance_score_source_session_weighted": normalized["relevance_score"],
    }
    for column, default_series in weighted_column_defaults.items():
        normalized[column] = pd.to_numeric(normalized.get(column, default_series), errors="coerce").fillna(default_series)
    default_zero = pd.Series(0.0, index=normalized.index, dtype=float)
    normalized["mapping_confidence"] = pd.to_numeric(
        normalized.get("mapping_confidence", default_zero), errors="coerce"
    ).fillna(0.0)
    normalized["novelty_score"] = pd.to_numeric(normalized.get("novelty_score", default_zero), errors="coerce").fillna(
        0.0
    )
    normalized["source_diversity"] = pd.to_numeric(
        normalized.get("source_diversity", default_zero), errors="coerce"
    ).fillna(0.0)
    normalized["source_count"] = pd.to_numeric(normalized.get("source_count", default_zero), errors="coerce").fillna(
        0.0
    )
    if "source_mix" in normalized.columns:
        normalized["source_mix"] = normalized["source_mix"].fillna("none").astype(str)
    else:
        normalized["source_mix"] = "none"
    if "source_session_breakdown" in normalized.columns:
        normalized["source_session_breakdown"] = normalized["source_session_breakdown"].fillna("{}").astype(str)
    else:
        normalized["source_session_breakdown"] = "{}"
    if "session_bucket" in normalized.columns:
        normalized["session_bucket"] = normalized["session_bucket"].fillna("none").astype(str)
    else:
        normalized["session_bucket"] = "none"
    normalized["source"] = normalized.get("source", "unknown")
    if "data_age_hours" in normalized.columns:
        normalized["data_age_hours"] = pd.to_numeric(normalized["data_age_hours"], errors="coerce").fillna(0.0)
    else:
        normalized["data_age_hours"] = 0.0
    normalized["stale_data_flag"] = normalized.get(
        "stale_data_flag", pd.Series(False, index=normalized.index, dtype=bool)
    ).astype(bool)
    normalized = normalized.dropna(
        subset=["ticker", "published_at", "available_at", "sentiment_score", "relevance_score", "headline_count"]
    )
    normalized = normalized.sort_values(["ticker", "available_at"]).drop_duplicates(["ticker", "available_at"])
    return normalized[NEWS_COLUMNS].reset_index(drop=True)


def normalize_fundamentals(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "ticker",
        "report_date",
        "available_at",
        "revenue_growth",
        "earnings_yield",
        "debt_to_equity",
        "profitability",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing fundamentals columns: {sorted(missing)}")
    normalized = frame.copy()
    normalized["report_date"] = normalize_to_utc(normalized["report_date"])
    normalized["available_at"] = normalize_to_utc(normalized["available_at"])
    normalized["fetched_at"] = normalize_to_utc(normalized.get("fetched_at", normalized["available_at"]))
    for column in ["revenue_growth", "earnings_yield", "debt_to_equity", "profitability"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["source"] = normalized.get("source", "unknown")
    if "data_age_hours" in normalized.columns:
        normalized["data_age_hours"] = pd.to_numeric(normalized["data_age_hours"], errors="coerce").fillna(0.0)
    else:
        normalized["data_age_hours"] = 0.0
    normalized["stale_data_flag"] = normalized.get(
        "stale_data_flag", pd.Series(False, index=normalized.index, dtype=bool)
    ).astype(bool)
    normalized = normalized.dropna(
        subset=[
            "ticker",
            "report_date",
            "available_at",
            "revenue_growth",
            "earnings_yield",
            "debt_to_equity",
            "profitability",
        ]
    )
    normalized = normalized.sort_values(["ticker", "available_at"]).drop_duplicates(["ticker", "available_at"])
    return normalized[FUNDAMENTAL_COLUMNS].reset_index(drop=True)


def normalize_sector_map(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"ticker", "sector"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing sector-map columns: {sorted(missing)}")
    normalized = frame.copy()
    if "fetched_at" in normalized.columns:
        normalized["fetched_at"] = normalize_to_utc(normalized["fetched_at"])
    else:
        normalized["fetched_at"] = pd.Timestamp.now(tz="UTC")
    normalized["source"] = normalized.get("source", "unknown")
    if "data_age_hours" in normalized.columns:
        normalized["data_age_hours"] = pd.to_numeric(normalized["data_age_hours"], errors="coerce").fillna(0.0)
    else:
        normalized["data_age_hours"] = 0.0
    normalized["stale_data_flag"] = normalized.get(
        "stale_data_flag", pd.Series(False, index=normalized.index, dtype=bool)
    ).astype(bool)
    normalized = normalized.dropna(subset=["ticker", "sector"])
    normalized = normalized.sort_values(["ticker"]).drop_duplicates(["ticker"])
    return normalized[SECTOR_MAP_COLUMNS].reset_index(drop=True)


def apply_stale_flag(
    frame: pd.DataFrame,
    as_of_time: str | pd.Timestamp,
    threshold_hours: int,
) -> pd.DataFrame:
    if "fetched_at" not in frame.columns:
        updated = frame.copy()
        updated["data_age_hours"] = 0.0
        updated["stale_data_flag"] = False
        return updated
    updated = frame.copy()
    updated["fetched_at"] = normalize_to_utc(updated["fetched_at"])
    as_of_timestamp = to_utc_timestamp(as_of_time)
    age = (as_of_timestamp - updated["fetched_at"]).dt.total_seconds() / 3600.0
    updated["data_age_hours"] = age.clip(lower=0.0)
    updated["stale_data_flag"] = updated["data_age_hours"] > float(threshold_hours)
    return updated
