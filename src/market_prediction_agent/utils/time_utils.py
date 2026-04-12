from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd


UTC = "UTC"


def to_utc_timestamp(value: str | pd.Timestamp | date) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(UTC)
    return timestamp.tz_convert(UTC)


def normalize_to_utc(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, utc=True)
    return pd.Series(timestamps, index=series.index)


def point_in_time_filter(
    frame: pd.DataFrame,
    as_of_date: str | pd.Timestamp | date,
    available_at_col: str = "available_at",
) -> pd.DataFrame:
    as_of_ts = to_utc_timestamp(as_of_date)
    if available_at_col not in frame.columns:
        return frame.copy()
    filtered = frame.copy()
    filtered[available_at_col] = normalize_to_utc(filtered[available_at_col])
    return filtered.loc[filtered[available_at_col] <= as_of_ts].copy()


def is_business_day(value: str | pd.Timestamp | date) -> bool:
    timestamp = pd.Timestamp(value)
    return timestamp.dayofweek < 5


def business_dates_between(start: str | date, end: str | date) -> pd.DatetimeIndex:
    return pd.bdate_range(pd.Timestamp(start), pd.Timestamp(end), tz=UTC)


def ensure_sorted_unique(values: Iterable[pd.Timestamp]) -> list[pd.Timestamp]:
    return sorted({to_utc_timestamp(value) for value in values})

