from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from uuid import uuid4

import pandas as pd

from market_prediction_agent.agents.data_agent import DataAgent
from market_prediction_agent.config import load_settings, update_settings
from market_prediction_agent.data.adapters import NewsRequest
from market_prediction_agent.data.normalizer import apply_stale_flag, normalize_news
from market_prediction_agent.storage.parquet_store import ParquetStore
from market_prediction_agent.utils.paths import resolve_repo_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a live smoke check for the configured news source.")
    parser.add_argument("--config", default=None, help="Optional config path.")
    parser.add_argument("--tickers", default="SPY,QQQ,GLD", help="Comma-separated ticker list.")
    parser.add_argument("--lookback-days", type=int, default=7, help="Business-day lookback window.")
    parser.add_argument("--as-of-date", default=None, help="Optional as-of date in YYYY-MM-DD format.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    settings = update_settings(settings, data={"source_mode": "live"})
    tickers = [item.strip().upper() for item in args.tickers.split(",") if item.strip()]
    as_of_timestamp = pd.Timestamp(args.as_of_date or pd.Timestamp.now(tz="UTC").date().isoformat(), tz="UTC")
    start_date = pd.bdate_range(end=as_of_timestamp.normalize(), periods=max(args.lookback_days, 2), tz="UTC")[0].date().isoformat()
    end_date = as_of_timestamp.normalize().date().isoformat()
    temp_dir = resolve_repo_path(Path(".test-artifacts") / "news-live-smoke" / uuid4().hex)
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        agent = DataAgent(settings, ParquetStore(Path(temp_dir)))
        frame, metadata = agent._fetch_news_with_fallback(
            NewsRequest(tickers=tickers, start_date=start_date, end_date=end_date),
            as_of_timestamp=as_of_timestamp,
        )
        normalized = apply_stale_flag(
            normalize_news(frame),
            as_of_time=as_of_timestamp,
            threshold_hours=settings.data.stale_threshold_hours.news,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    result = {
        "tickers": tickers,
        "start_date": start_date,
        "end_date": end_date,
        "requested_source": metadata.get("requested_source"),
        "requested_sources": metadata.get("requested_sources", []),
        "used_source": metadata.get("used_source"),
        "used_sources": metadata.get("used_sources", []),
        "fallback_used": metadata.get("fallback_used"),
        "fallback_reason": metadata.get("fallback_reason"),
        "transport": metadata.get("transport", {}),
        "record_count": int(len(normalized)),
        "headline_count": int(normalized["headline_count"].fillna(0).sum()) if not normalized.empty else 0,
        "missing_rate": float(normalized["sentiment_score"].isna().mean()) if not normalized.empty else 1.0,
        "stale_rate": float(normalized["stale_data_flag"].mean()) if not normalized.empty else 0.0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
