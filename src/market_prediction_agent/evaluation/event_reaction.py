from __future__ import annotations

import math
from typing import Any

import pandas as pd

from market_prediction_agent.utils.time_utils import to_utc_timestamp


def _empty_event_metrics(event_date: str | pd.Timestamp, window: int) -> dict[str, Any]:
    return {
        "event_date": to_utc_timestamp(event_date).date().isoformat(),
        "window": int(window),
        "pre_event_ar": 0.0,
        "post_event_ar": 0.0,
        "cumulative_ar": 0.0,
        "t_statistic": 0.0,
        "observation_count": 0,
        "pre_event_days": 0,
        "post_event_days": 0,
    }


def _coerce_return_series(returns: pd.Series | pd.DataFrame) -> pd.Series:
    if isinstance(returns, pd.Series):
        series = pd.to_numeric(returns, errors="coerce")
        index = pd.to_datetime(series.index, utc=True).normalize()
        return (
            pd.Series(series.to_numpy(dtype=float), index=index, dtype=float)
            .dropna()
            .groupby(level=0)
            .mean()
            .sort_index()
        )

    frame = returns.copy()
    if frame.empty:
        return pd.Series(dtype=float)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.normalize()
    elif "timestamp_utc" in frame.columns:
        frame["date"] = pd.to_datetime(frame["timestamp_utc"], utc=True).dt.normalize()
    else:
        raise ValueError("Returns frame must contain either 'date' or 'timestamp_utc'.")

    value_column = None
    for candidate in ("return", "simple_return", "sector_return", "daily_return"):
        if candidate in frame.columns:
            value_column = candidate
            break
    if value_column is None:
        raise ValueError("Returns frame must contain a return column.")

    series = pd.Series(
        pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=float),
        index=frame["date"],
        dtype=float,
    )
    return series.dropna().groupby(level=0).mean().sort_index()


def compute_event_abnormal_return(
    ticker_returns: pd.Series | pd.DataFrame,
    sector_returns: pd.Series | pd.DataFrame,
    event_date: str | pd.Timestamp,
    window: int = 3,
) -> dict[str, Any]:
    if window < 1:
        raise ValueError("window must be at least 1 business day.")

    ticker_series = _coerce_return_series(ticker_returns)
    sector_series = _coerce_return_series(sector_returns)
    if ticker_series.empty or sector_series.empty:
        return _empty_event_metrics(event_date, window)

    event_timestamp = to_utc_timestamp(event_date).normalize()
    aligned = pd.DataFrame(
        {
            "ticker_return": ticker_series,
            "sector_return": sector_series,
        }
    ).dropna()
    if aligned.empty:
        return _empty_event_metrics(event_date, window)

    abnormal = aligned["ticker_return"] - aligned["sector_return"]
    business_window = pd.offsets.BDay(window)
    pre_event = abnormal.loc[
        (abnormal.index >= event_timestamp - business_window) & (abnormal.index < event_timestamp)
    ]
    post_event = abnormal.loc[
        (abnormal.index > event_timestamp) & (abnormal.index <= event_timestamp + business_window)
    ]
    event_window = pd.concat([pre_event, post_event]).sort_index()
    if event_window.empty:
        return _empty_event_metrics(event_date, window)

    std = float(event_window.std(ddof=1)) if len(event_window) > 1 else 0.0
    if len(event_window) < 2 or std == 0.0:
        t_statistic = 0.0
    else:
        t_statistic = float(event_window.mean() / (std / math.sqrt(len(event_window))))

    return {
        "event_date": event_timestamp.date().isoformat(),
        "window": int(window),
        "pre_event_ar": float(pre_event.sum()),
        "post_event_ar": float(post_event.sum()),
        "cumulative_ar": float(event_window.sum()),
        "t_statistic": t_statistic,
        "observation_count": int(len(event_window)),
        "pre_event_days": int(len(pre_event)),
        "post_event_days": int(len(post_event)),
    }


def detect_earnings_events(fundamentals: pd.DataFrame) -> pd.DataFrame:
    required = {"report_date", "available_at"}
    missing = required.difference(fundamentals.columns)
    if missing:
        raise ValueError(f"Missing fundamentals columns: {sorted(missing)}")

    if fundamentals.empty:
        empty = fundamentals.copy()
        empty["event_date"] = pd.Series(dtype="datetime64[ns, UTC]")
        empty["available_date"] = pd.Series(dtype="datetime64[ns, UTC]")
        empty["effective_event_date"] = pd.Series(dtype="datetime64[ns, UTC]")
        empty["availability_lag_days"] = pd.Series(dtype="int64")
        return empty

    events = fundamentals.copy()
    events["report_date"] = pd.to_datetime(events["report_date"], utc=True, errors="coerce")
    events["available_at"] = pd.to_datetime(events["available_at"], utc=True, errors="coerce")
    events = events.dropna(subset=["report_date", "available_at"]).copy()
    if events.empty:
        return detect_earnings_events(events.reindex(columns=fundamentals.columns))

    events["event_date"] = events["report_date"].dt.normalize()
    events["available_date"] = events["available_at"].dt.normalize()
    events["effective_event_date"] = pd.concat(
        [events["event_date"], events["available_date"]],
        axis=1,
    ).max(axis=1)
    events["availability_lag_days"] = (
        events["effective_event_date"] - events["event_date"]
    ).dt.days.astype(int)
    sort_columns = [column for column in ("ticker", "event_date", "available_at") if column in events.columns]
    dedupe_columns = [column for column in ("ticker", "event_date", "available_at") if column in events.columns]
    if sort_columns:
        events = events.sort_values(sort_columns)
    if dedupe_columns:
        events = events.drop_duplicates(dedupe_columns)
    return events.reset_index(drop=True)


def _prepare_ticker_returns(ohlcv: pd.DataFrame) -> pd.DataFrame:
    if ohlcv.empty:
        return pd.DataFrame(columns=["ticker", "date", "return"])
    required = {"ticker", "timestamp_utc", "close"}
    missing = required.difference(ohlcv.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")

    frame = ohlcv.copy()
    frame["date"] = pd.to_datetime(frame["timestamp_utc"], utc=True).dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["ticker", "date", "close"]).sort_values(["ticker", "date"])
    frame = frame.drop_duplicates(["ticker", "date"], keep="last")
    frame["return"] = frame.groupby("ticker")["close"].pct_change()
    return frame[["ticker", "date", "return"]].dropna().reset_index(drop=True)


def _prepare_sector_returns(sector_returns: pd.DataFrame) -> pd.DataFrame:
    if sector_returns.empty:
        return pd.DataFrame(columns=["sector", "date", "return"])

    frame = sector_returns.copy()
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.normalize()
    elif "timestamp_utc" in frame.columns:
        frame["date"] = pd.to_datetime(frame["timestamp_utc"], utc=True).dt.normalize()
    else:
        raise ValueError("sector_returns must contain either 'date' or 'timestamp_utc'.")

    if "return" not in frame.columns:
        if "sector_return" in frame.columns:
            frame = frame.rename(columns={"sector_return": "return"})
        elif "simple_return" in frame.columns:
            frame = frame.rename(columns={"simple_return": "return"})
        else:
            raise ValueError("sector_returns must contain 'return', 'sector_return', or 'simple_return'.")

    if "sector" not in frame.columns:
        frame["sector"] = "__all__"

    frame["return"] = pd.to_numeric(frame["return"], errors="coerce")
    frame = frame.dropna(subset=["sector", "date", "return"]).sort_values(["sector", "date"])
    return frame[["sector", "date", "return"]].reset_index(drop=True)


def build_event_reaction_summary(
    events: pd.DataFrame,
    ohlcv: pd.DataFrame,
    sector_returns: pd.DataFrame,
    *,
    window: int = 3,
) -> dict[str, Any]:
    if window < 1:
        raise ValueError("window must be at least 1 business day.")

    if events.empty:
        return {
            "window": int(window),
            "event_count": 0,
            "analyzed_event_count": 0,
            "average_pre_event_ar": 0.0,
            "average_post_event_ar": 0.0,
            "average_cumulative_ar": 0.0,
            "average_car": 0.0,
            "mean_t_statistic": 0.0,
            "significant_event_ratio": 0.0,
            "sector_summary": [],
            "events": [],
        }

    event_frame = detect_earnings_events(events) if "effective_event_date" not in events.columns else events.copy()
    if "effective_event_date" not in event_frame.columns:
        if "event_date" in event_frame.columns:
            event_frame["event_date"] = pd.to_datetime(event_frame["event_date"], utc=True).dt.normalize()
            event_frame["effective_event_date"] = event_frame["event_date"]
        else:
            raise ValueError("events must include report_date or effective_event_date.")
    else:
        event_frame["effective_event_date"] = pd.to_datetime(event_frame["effective_event_date"], utc=True).dt.normalize()

    ticker_return_frame = _prepare_ticker_returns(ohlcv)
    sector_return_frame = _prepare_sector_returns(sector_returns)
    sector_keys = sector_return_frame["sector"].drop_duplicates().tolist()

    if "sector" not in event_frame.columns:
        default_sector = "__all__" if "__all__" in sector_keys else (sector_keys[0] if len(sector_keys) == 1 else "unknown")
        event_frame["sector"] = default_sector

    event_results: list[dict[str, Any]] = []
    for _, row in event_frame.iterrows():
        ticker = str(row.get("ticker", "")).upper()
        ticker_series = ticker_return_frame.loc[ticker_return_frame["ticker"] == ticker, ["date", "return"]]
        sector = str(row.get("sector", "unknown"))
        if sector in sector_keys:
            sector_series = sector_return_frame.loc[sector_return_frame["sector"] == sector, ["date", "return"]]
        elif "__all__" in sector_keys:
            sector = "__all__"
            sector_series = sector_return_frame.loc[sector_return_frame["sector"] == sector, ["date", "return"]]
        else:
            sector_series = pd.DataFrame(columns=["date", "return"])
        metrics = compute_event_abnormal_return(
            ticker_series,
            sector_series,
            row["effective_event_date"],
            window=window,
        )
        event_results.append(
            {
                "ticker": ticker,
                "sector": sector,
                "event_date": pd.to_datetime(row.get("event_date", row["effective_event_date"]), utc=True).date().isoformat(),
                "effective_event_date": pd.to_datetime(row["effective_event_date"], utc=True).date().isoformat(),
                "available_at": (
                    pd.to_datetime(row["available_at"], utc=True).isoformat()
                    if row.get("available_at") is not None and not pd.isna(row.get("available_at"))
                    else None
                ),
                "availability_lag_days": int(row.get("availability_lag_days", 0) or 0),
                **metrics,
            }
        )

    result_frame = pd.DataFrame(event_results)
    analyzed = result_frame.loc[result_frame["observation_count"] > 0].copy()
    if analyzed.empty:
        sector_summary: list[dict[str, Any]] = []
    else:
        sector_summary = [
            {
                "sector": str(sector),
                "event_count": int(len(group)),
                "average_cumulative_ar": float(group["cumulative_ar"].mean()),
                "average_post_event_ar": float(group["post_event_ar"].mean()),
                "significant_event_ratio": float((group["t_statistic"].abs() >= 1.96).mean()),
            }
            for sector, group in analyzed.groupby("sector")
        ]
        sector_summary.sort(key=lambda item: str(item["sector"]))

    return {
        "window": int(window),
        "event_count": int(len(result_frame)),
        "analyzed_event_count": int(len(analyzed)),
        "average_pre_event_ar": float(analyzed["pre_event_ar"].mean()) if not analyzed.empty else 0.0,
        "average_post_event_ar": float(analyzed["post_event_ar"].mean()) if not analyzed.empty else 0.0,
        "average_cumulative_ar": float(analyzed["cumulative_ar"].mean()) if not analyzed.empty else 0.0,
        "average_car": float(analyzed["cumulative_ar"].mean()) if not analyzed.empty else 0.0,
        "mean_t_statistic": float(analyzed["t_statistic"].mean()) if not analyzed.empty else 0.0,
        "significant_event_ratio": float((analyzed["t_statistic"].abs() >= 1.96).mean()) if not analyzed.empty else 0.0,
        "sector_summary": sector_summary,
        "events": event_results,
    }
