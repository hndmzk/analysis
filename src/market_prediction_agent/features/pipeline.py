from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from market_prediction_agent.features.labels import make_direction_label
from market_prediction_agent.utils.time_utils import point_in_time_filter


PRICE_FEATURE_COLUMNS = [
    "log_return_1d",
    "log_return_5d",
    "log_return_10d",
    "log_return_20d",
    "realized_vol_5d",
    "realized_vol_10d",
    "realized_vol_20d",
    "garman_klass_vol",
    "rsi_14",
    "macd",
    "macd_signal",
    "roc_5d",
    "roc_10d",
    "roc_20d",
    "volume_ratio_5d",
    "volume_ratio_20d",
    "obv_slope_10d",
    "price_vs_sma_5d",
    "price_vs_sma_20d",
    "price_vs_sma_50d",
    "price_vs_sma_200d",
    "bb_position",
    "bb_width",
    "atr_14",
    "atr_ratio",
]
MACRO_FEATURE_COLUMNS = [
    "fed_funds_rate",
    "yield_curve_slope",
    "vix",
    "vix_change_5d",
]
NEWS_FEATURE_COLUMNS = [
    "news_sentiment_1d",
    "news_sentiment_5d",
    "news_sentiment_decay_5d",
    "news_relevance_5d",
    "news_novelty_5d",
    "news_source_diversity_5d",
    "news_volume_zscore_20d",
]
NEWS_DIAGNOSTIC_RENAME_MAP: dict[str, str] = {
    "sentiment_score_unweighted": "news_sentiment_score_unweighted",
    "sentiment_score_source_weighted": "news_sentiment_score_source_weighted",
    "sentiment_score_session_weighted": "news_sentiment_score_session_weighted",
    "sentiment_score_source_session_weighted": "news_sentiment_score_source_session_weighted",
    "relevance_score_unweighted": "news_relevance_score_unweighted",
    "relevance_score_source_weighted": "news_relevance_score_source_weighted",
    "relevance_score_session_weighted": "news_relevance_score_session_weighted",
    "relevance_score_source_session_weighted": "news_relevance_score_source_session_weighted",
}
NEWS_DIAGNOSTIC_COLUMNS = list(NEWS_DIAGNOSTIC_RENAME_MAP.values())
FUNDAMENTAL_FEATURE_COLUMNS = [
    "fundamental_revenue_growth",
    "fundamental_earnings_yield",
    "fundamental_leverage",
    "fundamental_profitability",
]
SECTOR_FEATURE_COLUMNS = [
    "sector_relative_momentum_20d",
    "sector_strength_20d",
    "sector_vol_spread_20d",
]
CALENDAR_FEATURE_COLUMNS = [
    "day_of_week",
    "month",
    "is_month_end",
]

FEATURE_COLUMNS = (
    PRICE_FEATURE_COLUMNS
    + MACRO_FEATURE_COLUMNS
    + NEWS_FEATURE_COLUMNS
    + FUNDAMENTAL_FEATURE_COLUMNS
    + SECTOR_FEATURE_COLUMNS
    + CALENDAR_FEATURE_COLUMNS
)

PRICE_MOMENTUM_FEATURES = {
    "log_return_1d",
    "log_return_5d",
    "log_return_10d",
    "log_return_20d",
    "roc_5d",
    "roc_10d",
    "roc_20d",
    "rsi_14",
    "macd",
    "macd_signal",
    "price_vs_sma_5d",
    "price_vs_sma_20d",
    "price_vs_sma_50d",
    "price_vs_sma_200d",
    "bb_position",
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

FEATURE_FAMILY_MAP = {
    **{feature: "price_momentum" for feature in PRICE_MOMENTUM_FEATURES},
    **{feature: "volatility" for feature in VOLATILITY_FEATURES},
    **{feature: "volume" for feature in VOLUME_FEATURES},
    **{feature: "macro" for feature in MACRO_FEATURE_COLUMNS},
    **{feature: "news" for feature in NEWS_FEATURE_COLUMNS},
    **{feature: "fundamental" for feature in FUNDAMENTAL_FEATURE_COLUMNS},
    **{feature: "sector" for feature in SECTOR_FEATURE_COLUMNS},
    **{feature: "calendar" for feature in CALENDAR_FEATURE_COLUMNS},
}
FEATURE_DOMAIN_MAP = {
    **{feature: "ohlcv" for feature in PRICE_FEATURE_COLUMNS},
    **{feature: "macro" for feature in MACRO_FEATURE_COLUMNS},
    **{feature: "news" for feature in NEWS_FEATURE_COLUMNS},
    **{feature: "fundamental" for feature in FUNDAMENTAL_FEATURE_COLUMNS},
    **{feature: "sector" for feature in SECTOR_FEATURE_COLUMNS},
    **{feature: "calendar" for feature in CALENDAR_FEATURE_COLUMNS},
}
FAMILY_STALE_FLAG_COLUMN = {
    "price_momentum": "stale_data_flag",
    "volatility": "stale_data_flag",
    "volume": "stale_data_flag",
    "news": "news_stale_flag",
    "fundamental": "fundamental_stale_flag",
    "sector": "sector_stale_flag",
}


@dataclass(slots=True)
class FeatureEngineeringResult:
    feature_frame: pd.DataFrame
    feature_catalog: list[dict[str, object]]


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


def _feature_sources_metadata(source_metadata: dict[str, object]) -> dict[str, dict[str, object]]:
    feature_sources = source_metadata.get("feature_sources", {})
    if not isinstance(feature_sources, dict):
        return {}
    resolved: dict[str, dict[str, object]] = {}
    for key, value in feature_sources.items():
        if isinstance(key, str) and isinstance(value, dict):
            resolved[key] = cast(dict[str, object], value)
    return resolved


def _compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def _compute_macd(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    ema_fast = series.ewm(span=12, adjust=False).mean()
    ema_slow = series.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


def _compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume).cumsum()


def _forward_rolling_std(series: pd.Series, window: int) -> pd.Series:
    shifted = series.shift(-1)
    return shifted.iloc[::-1].rolling(window, min_periods=window).std().iloc[::-1]


def _merge_macro_features(frame: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    macro_filtered = point_in_time_filter(macro, frame["date"].max(), available_at_col="available_at")
    merged = pd.DataFrame({"date": sorted(frame["date"].unique())})
    merged["date"] = pd.to_datetime(merged["date"], utc=True).astype("datetime64[ns, UTC]")
    mapping = {
        "FEDFUNDS": "fed_funds_rate",
        "T10Y2Y": "yield_curve_slope",
        "VIXCLS": "vix",
    }
    for series_id, output_name in mapping.items():
        subset = macro_filtered.loc[macro_filtered["series_id"] == series_id, ["available_at", "value"]].copy()
        subset = subset.rename(columns={"available_at": "date", "value": output_name})
        subset["date"] = pd.to_datetime(subset["date"], utc=True).astype("datetime64[ns, UTC]")
        subset = subset.sort_values("date")
        merged = pd.merge_asof(merged.sort_values("date"), subset, on="date", direction="backward")
    merged["vix_change_5d"] = merged["vix"].pct_change(5).fillna(0.0)
    return frame.merge(merged, on="date", how="left")


def _merge_news_features(frame: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    if news.empty:
        updated = frame.copy()
        for column in NEWS_FEATURE_COLUMNS + NEWS_DIAGNOSTIC_COLUMNS:
            updated[column] = np.nan
        updated["news_stale_flag"] = False
        return updated
    news_features = news.rename(
        columns={
            "stale_data_flag": "news_stale_flag",
            "sentiment_score": "news_sentiment_score_raw",
            "relevance_score": "news_relevance_score_raw",
            "headline_count": "news_headline_count_raw",
            "mapping_confidence": "news_mapping_confidence_raw",
            "novelty_score": "news_novelty_score_raw",
            "source_diversity": "news_source_diversity_raw",
            **NEWS_DIAGNOSTIC_RENAME_MAP,
        }
    )[
        [
            "ticker",
            "available_at",
            "news_sentiment_score_raw",
            "news_relevance_score_raw",
            *NEWS_DIAGNOSTIC_COLUMNS,
            "news_headline_count_raw",
            "news_mapping_confidence_raw",
            "news_novelty_score_raw",
            "news_source_diversity_raw",
            "news_stale_flag",
        ]
    ].sort_values(["ticker", "available_at"])
    news_features["available_at"] = pd.to_datetime(news_features["available_at"], utc=True).astype("datetime64[ns, UTC]")
    merged_groups: list[pd.DataFrame] = []
    for ticker, group in frame.groupby("ticker", group_keys=False):
        right = news_features.loc[news_features["ticker"] == ticker].drop(columns=["ticker"])
        merged_group = pd.merge_asof(
            group.sort_values("date"),
            right.sort_values("available_at"),
            left_on="date",
            right_on="available_at",
            direction="backward",
        )
        merged_groups.append(merged_group)
    merged = pd.concat(merged_groups, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    for column in NEWS_DIAGNOSTIC_COLUMNS:
        merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
    grouped = merged.groupby("ticker", group_keys=False)
    merged["news_effective_weight_raw"] = (
        merged["news_headline_count_raw"].fillna(0.0).clip(lower=0.0)
        * merged["news_relevance_score_raw"].fillna(0.0).clip(lower=0.0, upper=1.0)
        * merged["news_mapping_confidence_raw"].fillna(0.0).clip(lower=0.0, upper=1.0)
    )
    merged["news_effective_weight_raw"] = np.where(
        merged["news_effective_weight_raw"] > 0.0,
        merged["news_effective_weight_raw"],
        merged["news_headline_count_raw"].fillna(0.0).clip(lower=0.0),
    )
    merged["news_weighted_sentiment_raw"] = merged["news_sentiment_score_raw"].fillna(0.0) * merged["news_effective_weight_raw"]
    merged["news_weighted_relevance_raw"] = merged["news_relevance_score_raw"].fillna(0.0) * merged["news_effective_weight_raw"]
    merged["news_weighted_novelty_raw"] = merged["news_novelty_score_raw"].fillna(0.0) * merged["news_effective_weight_raw"]
    merged["news_weighted_source_diversity_raw"] = (
        merged["news_source_diversity_raw"].fillna(0.0) * merged["news_effective_weight_raw"]
    )
    merged["news_sentiment_1d"] = merged["news_sentiment_score_raw"].fillna(0.0)
    sentiment_5d_numerator = grouped["news_weighted_sentiment_raw"].transform(
        lambda series: series.fillna(0.0).rolling(5, min_periods=1).sum()
    )
    sentiment_5d_denominator = grouped["news_effective_weight_raw"].transform(
        lambda series: series.fillna(0.0).rolling(5, min_periods=1).sum()
    )
    merged["news_sentiment_5d"] = sentiment_5d_numerator / sentiment_5d_denominator.replace(0.0, np.nan)
    sentiment_decay_numerator = grouped["news_weighted_sentiment_raw"].transform(
        lambda series: series.fillna(0.0).ewm(halflife=5, adjust=False).mean()
    )
    sentiment_decay_denominator = grouped["news_effective_weight_raw"].transform(
        lambda series: series.fillna(0.0).ewm(halflife=5, adjust=False).mean()
    )
    merged["news_sentiment_decay_5d"] = sentiment_decay_numerator / sentiment_decay_denominator.replace(0.0, np.nan)
    relevance_5d_numerator = grouped["news_weighted_relevance_raw"].transform(
        lambda series: series.fillna(0.0).rolling(5, min_periods=1).sum()
    )
    merged["news_relevance_5d"] = relevance_5d_numerator / sentiment_5d_denominator.replace(0.0, np.nan)
    novelty_5d_numerator = grouped["news_weighted_novelty_raw"].transform(
        lambda series: series.fillna(0.0).rolling(5, min_periods=1).sum()
    )
    merged["news_novelty_5d"] = novelty_5d_numerator / sentiment_5d_denominator.replace(0.0, np.nan)
    source_diversity_5d_numerator = grouped["news_weighted_source_diversity_raw"].transform(
        lambda series: series.fillna(0.0).rolling(5, min_periods=1).sum()
    )
    merged["news_source_diversity_5d"] = (
        source_diversity_5d_numerator / sentiment_5d_denominator.replace(0.0, np.nan)
    )
    rolling_headline_mean = grouped["news_headline_count_raw"].transform(
        lambda series: series.rolling(20, min_periods=5).mean()
    )
    rolling_headline_std = grouped["news_headline_count_raw"].transform(
        lambda series: series.rolling(20, min_periods=5).std()
    )
    merged["news_volume_zscore_20d"] = (
        (merged["news_headline_count_raw"] - rolling_headline_mean)
        / rolling_headline_std.replace(0.0, np.nan)
    )
    for column in NEWS_FEATURE_COLUMNS:
        merged[column] = merged[column].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    merged["news_stale_flag"] = merged["news_stale_flag"].fillna(False).astype(bool)
    return merged


def _merge_fundamental_features(frame: pd.DataFrame, fundamentals: pd.DataFrame) -> pd.DataFrame:
    if fundamentals.empty:
        updated = frame.copy()
        for column in FUNDAMENTAL_FEATURE_COLUMNS:
            updated[column] = np.nan
        updated["fundamental_stale_flag"] = False
        return updated
    fundamentals_features = fundamentals.rename(
        columns={
            "stale_data_flag": "fundamental_stale_flag",
            "revenue_growth": "fundamental_revenue_growth",
            "earnings_yield": "fundamental_earnings_yield",
            "debt_to_equity": "fundamental_leverage",
            "profitability": "fundamental_profitability",
        }
    )[
        [
            "ticker",
            "available_at",
            "fundamental_revenue_growth",
            "fundamental_earnings_yield",
            "fundamental_leverage",
            "fundamental_profitability",
            "fundamental_stale_flag",
        ]
    ].sort_values(["ticker", "available_at"])
    fundamentals_features["available_at"] = pd.to_datetime(
        fundamentals_features["available_at"], utc=True
    ).astype("datetime64[ns, UTC]")
    merged_groups: list[pd.DataFrame] = []
    for ticker, group in frame.groupby("ticker", group_keys=False):
        right = fundamentals_features.loc[fundamentals_features["ticker"] == ticker].drop(columns=["ticker"])
        merged_group = pd.merge_asof(
            group.sort_values("date"),
            right.sort_values("available_at"),
            left_on="date",
            right_on="available_at",
            direction="backward",
        )
        merged_groups.append(merged_group)
    merged = pd.concat(merged_groups, ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)
    merged["fundamental_stale_flag"] = merged["fundamental_stale_flag"].fillna(False).astype(bool)
    return merged


def _merge_sector_features(frame: pd.DataFrame, sector_map: pd.DataFrame) -> pd.DataFrame:
    if sector_map.empty:
        updated = frame.copy()
        for column in SECTOR_FEATURE_COLUMNS:
            updated[column] = np.nan
        updated["sector_stale_flag"] = updated["stale_data_flag"].fillna(False).astype(bool)
        return updated
    mapping = sector_map.rename(columns={"stale_data_flag": "sector_map_stale_flag"})[
        ["ticker", "sector", "sector_map_stale_flag"]
    ]
    merged = frame.merge(mapping, on="ticker", how="left")
    merged["sector"] = merged["sector"].fillna("unknown")
    sector_daily = (
        merged.groupby(["date", "sector"], as_index=False)
        .agg(
            sector_roc_20d=("roc_20d", "mean"),
            sector_realized_vol_20d=("realized_vol_20d", "mean"),
        )
    )
    market_daily = (
        merged.groupby("date", as_index=False)
        .agg(
            market_roc_20d=("roc_20d", "mean"),
            market_realized_vol_20d=("realized_vol_20d", "mean"),
        )
    )
    merged = merged.merge(sector_daily, on=["date", "sector"], how="left")
    merged = merged.merge(market_daily, on="date", how="left")
    merged["sector_relative_momentum_20d"] = merged["roc_20d"] - merged["sector_roc_20d"]
    merged["sector_strength_20d"] = merged["sector_roc_20d"] - merged["market_roc_20d"]
    merged["sector_vol_spread_20d"] = merged["sector_realized_vol_20d"] - merged["market_realized_vol_20d"]
    merged["sector_stale_flag"] = (
        merged["sector_map_stale_flag"].fillna(False).astype(bool)
        | merged["stale_data_flag"].fillna(False).astype(bool)
    )
    return merged


def _feature_source(feature: str, source_metadata: dict[str, object]) -> str:
    family = FEATURE_FAMILY_MAP.get(feature, "price_momentum")
    feature_sources = _feature_sources_metadata(source_metadata)
    if family in {"price_momentum", "volatility", "volume"}:
        return str(source_metadata.get("used_source", "unknown"))
    if family == "macro":
        return str(source_metadata.get("macro_source", "unknown"))
    if family == "news":
        return str(feature_sources.get("news", {}).get("used_source", "unknown"))
    if family == "fundamental":
        return str(feature_sources.get("fundamental", {}).get("used_source", "unknown"))
    if family == "sector":
        return str(feature_sources.get("sector", {}).get("used_source", "unknown"))
    return "derived_calendar"


def build_feature_catalog(
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
    source_metadata: dict[str, object],
) -> list[dict[str, object]]:
    catalog: list[dict[str, object]] = []
    feature_sources = _feature_sources_metadata(source_metadata)
    for feature in feature_columns:
        family = FEATURE_FAMILY_MAP.get(feature, "price_momentum")
        stale_flag_column = FAMILY_STALE_FLAG_COLUMN.get(family)
        stale_rate = 0.0
        if stale_flag_column is not None and stale_flag_column in feature_frame.columns:
            stale_rate = float(feature_frame[stale_flag_column].fillna(False).astype(bool).mean())
        elif family in {"news", "fundamental", "sector"}:
            domain_key = "fundamental" if family == "fundamental" else family
            stale_rate = _as_float(feature_sources.get(domain_key, {}).get("stale_rate", 0.0))
        catalog.append(
            {
                "feature": feature,
                "feature_family": family,
                "data_source": _feature_source(feature, source_metadata),
                "domain": FEATURE_DOMAIN_MAP.get(feature, family),
                "missing_rate": float(feature_frame[feature].isna().mean()) if feature in feature_frame.columns else 1.0,
                "stale_rate": stale_rate,
            }
        )
    return catalog


def build_feature_frame(
    ohlcv: pd.DataFrame,
    macro: pd.DataFrame,
    news: pd.DataFrame,
    fundamentals: pd.DataFrame,
    sector_map: pd.DataFrame,
    horizon_days: int,
    direction_threshold: float,
    source_metadata: dict[str, object] | None = None,
) -> FeatureEngineeringResult:
    frame = ohlcv.copy()
    frame["date"] = pd.to_datetime(frame["timestamp_utc"], utc=True).dt.normalize().astype("datetime64[ns, UTC]")
    frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
    grouped = frame.groupby("ticker", group_keys=False)

    frame["log_return_1d"] = grouped["close"].pct_change().pipe(lambda series: np.log1p(series))
    for lag in [5, 10, 20]:
        frame[f"log_return_{lag}d"] = grouped["close"].pct_change(lag).pipe(lambda series: np.log1p(series))
        frame[f"roc_{lag}d"] = grouped["close"].pct_change(lag)
    for window in [5, 10, 20]:
        frame[f"realized_vol_{window}d"] = grouped["log_return_1d"].transform(
            lambda series, win=window: series.rolling(win, min_periods=win).std() * np.sqrt(252)
        )

    log_hl = np.log(frame["high"] / frame["low"]).replace([np.inf, -np.inf], np.nan)
    log_co = np.log(frame["close"] / frame["open"]).replace([np.inf, -np.inf], np.nan)
    frame["garman_klass_vol"] = np.sqrt(np.maximum(0.0, 0.5 * log_hl.pow(2) - (2 * np.log(2) - 1) * log_co.pow(2)))
    frame["rsi_14"] = grouped["close"].transform(_compute_rsi)
    frame["macd"] = grouped["close"].transform(lambda series: _compute_macd(series)[0])
    frame["macd_signal"] = grouped["close"].transform(lambda series: _compute_macd(series)[1])

    frame["volume_ratio_5d"] = frame["volume"] / grouped["volume"].transform(
        lambda series: series.rolling(5, min_periods=5).mean()
    )
    frame["volume_ratio_20d"] = frame["volume"] / grouped["volume"].transform(
        lambda series: series.rolling(20, min_periods=20).mean()
    )
    frame["obv"] = grouped.apply(lambda group: _compute_obv(group["close"], group["volume"])).reset_index(level=0, drop=True)
    frame["obv_slope_10d"] = grouped["obv"].transform(lambda series: (series - series.shift(10)) / 10.0)

    for window in [5, 20, 50, 200]:
        sma = grouped["close"].transform(lambda series, win=window: series.rolling(win, min_periods=win).mean())
        frame[f"price_vs_sma_{window}d"] = frame["close"] / sma - 1.0

    rolling_mean = grouped["close"].transform(lambda series: series.rolling(20, min_periods=20).mean())
    rolling_std = grouped["close"].transform(lambda series: series.rolling(20, min_periods=20).std())
    upper_band = rolling_mean + (2 * rolling_std)
    lower_band = rolling_mean - (2 * rolling_std)
    band_range = (upper_band - lower_band).replace(0, np.nan)
    frame["bb_position"] = ((frame["close"] - lower_band) / band_range).clip(lower=0, upper=1)
    frame["bb_width"] = band_range / rolling_mean.replace(0, np.nan)

    previous_close = grouped["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr_14"] = true_range.groupby(frame["ticker"]).transform(
        lambda series: series.rolling(14, min_periods=14).mean()
    )
    frame["atr_ratio"] = frame["atr_14"] / frame["close"]

    frame = _merge_macro_features(frame, macro)
    frame = _merge_news_features(frame, news)
    frame = _merge_fundamental_features(frame, fundamentals)
    frame = _merge_sector_features(frame, sector_map)

    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["month"] = frame["date"].dt.month
    frame["is_month_end"] = frame["date"].dt.is_month_end.astype(int)

    grouped = frame.groupby("ticker", group_keys=False)
    frame["future_simple_return"] = grouped["close"].shift(-horizon_days) / frame["close"] - 1.0
    frame["target_return"] = np.log1p(frame["future_simple_return"])
    frame["direction_label"] = make_direction_label(frame["future_simple_return"], threshold=direction_threshold)
    frame["future_volatility_20d"] = grouped["log_return_1d"].transform(
        lambda series: _forward_rolling_std(series, 20) * np.sqrt(252)
    )
    frame["stale_data_flag"] = frame["stale_data_flag"].fillna(False).astype(bool)
    if "news_stale_flag" not in frame.columns:
        frame["news_stale_flag"] = False
    if "fundamental_stale_flag" not in frame.columns:
        frame["fundamental_stale_flag"] = False
    if "sector_stale_flag" not in frame.columns:
        frame["sector_stale_flag"] = frame["stale_data_flag"]
    frame["news_stale_flag"] = frame["news_stale_flag"].fillna(False).astype(bool)
    frame["fundamental_stale_flag"] = frame["fundamental_stale_flag"].fillna(False).astype(bool)
    frame["sector_stale_flag"] = frame["sector_stale_flag"].fillna(False).astype(bool)

    feature_catalog = build_feature_catalog(frame, FEATURE_COLUMNS, source_metadata or {})

    for column in NEWS_FEATURE_COLUMNS + FUNDAMENTAL_FEATURE_COLUMNS + SECTOR_FEATURE_COLUMNS:
        frame[column] = frame[column].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return FeatureEngineeringResult(feature_frame=frame, feature_catalog=feature_catalog)


def build_training_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    required = FEATURE_COLUMNS + ["ticker", "date", "direction_label", "target_return", "future_simple_return", "future_volatility_20d"]
    cleaned = feature_frame[required + ["stale_data_flag"]].copy()
    cleaned = cleaned.dropna(subset=["direction_label"] + FEATURE_COLUMNS)
    cleaned["direction_label"] = cleaned["direction_label"].astype(int)
    return cleaned.reset_index(drop=True)
