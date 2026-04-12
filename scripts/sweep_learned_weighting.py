from __future__ import annotations

import argparse
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, cast
import urllib.parse
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

import market_prediction_agent.data.adapters as data_adapters_module
from market_prediction_agent.agents.data_agent import DataAgent
from market_prediction_agent.config import (
    LearnedWeightingConfig,
    Settings,
    load_settings,
    resolve_storage_path,
    update_settings,
)
from market_prediction_agent.data.adapters import (
    DummyMacroAdapter,
    DummyOHLCVAdapter,
    FredMarketProxyOHLCVAdapter,
    FundamentalsRequest,
    GoogleNewsRssAdapter,
    MultiSourceNewsAdapter,
    NewsRequest,
    OHLCVRequest,
    OfflineFundamentalProxyAdapter,
    PublicDataTransportConfig,
    US_MARKET_TZ,
    YahooChartOHLCVAdapter,
    YahooFinanceNewsAdapter,
    _headline_key,
    _stable_integer,
)
from market_prediction_agent.data.normalizer import apply_stale_flag, normalize_news, normalize_ohlcv
from market_prediction_agent.evaluation.learned_weighting import build_walk_forward_learned_weighting
from market_prediction_agent.evaluation.news_analysis import _effective_news_weight, _prepare_news_panel
from market_prediction_agent.storage.parquet_store import ParquetStore
from market_prediction_agent.utils.logging import configure_logging
from market_prediction_agent.utils.paths import resolve_repo_path
from market_prediction_agent.utils.time_utils import business_dates_between


OUTPUT_SUBDIR_BY_MODE = {
    "dummy": Path("outputs") / "sweep_learned_weighting",
    "public": Path("outputs") / "sweep_learned_weighting_public",
}
STAGE1_LAMBDAS = [5.0, 15.0, 30.0, 60.0, 120.0]
STAGE1_MIN_SAMPLES = [5, 10, 20, 40]
STAGE1_LOOKBACK_DAYS = [63, 126, 252, 504]
STAGE2_TARGETS = ["abs_ic", "overlap_rate", "combined"]
DEFAULT_REDUCED_TICKER_COUNT = 20
DEFAULT_REDUCED_DAY_COUNT = 400
DEFAULT_THREADSAFE_TICKER_COUNT = 8
DEFAULT_THREADSAFE_DAY_COUNT = 260
DEFAULT_MAX_WORKERS = 4
DEFAULT_TOP_K = 5
_SNAPSHOT_ONLY_ERROR = "network disabled for snapshot-only public sweep"
_GOOGLE_TICKER_RE = re.compile(r'"([A-Za-z0-9.\-]+)"')

_WORKER_CONTEXT: dict[str, Any] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep learned weighting hyperparameters on dummy or snapshot-backed public news data."
    )
    parser.add_argument("--config", default=None, help="Optional config path.")
    parser.add_argument("--mode", choices=["dummy", "public"], default="dummy", help="Dataset mode.")
    parser.add_argument("--ticker-count", type=int, default=None, help="Optional ticker-count override.")
    parser.add_argument("--days", type=int, default=None, help="Optional business-day history override.")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help="Worker process count.")
    return parser.parse_args()


def _as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float | np.integer | np.floating):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return default
        return numeric
    if isinstance(value, str):
        try:
            numeric = float(value)
        except ValueError:
            return default
        if math.isnan(numeric) or math.isinf(numeric):
            return default
        return numeric
    return default


def _as_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return default
        return int(numeric)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def _unit_interval(key: str) -> float:
    return (_stable_integer(key) % 1_000_000) / 1_000_000.0


def _local_midnight(signal_day: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(signal_day.date()).tz_localize(US_MARKET_TZ)


def _signal_day_return_lookup(frame: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], float]:
    normalized = frame.loc[:, ["ticker", "timestamp_utc", "close"]].copy()
    normalized["date"] = pd.to_datetime(normalized["timestamp_utc"], utc=True).dt.normalize()
    normalized = normalized.sort_values(["ticker", "date"]).reset_index(drop=True)
    normalized["signal_day_return"] = normalized.groupby("ticker")["close"].pct_change()
    lookup: dict[tuple[str, pd.Timestamp], float] = {}
    for _, row in normalized.dropna(subset=["signal_day_return"]).iterrows():
        lookup[(str(row["ticker"]), cast(pd.Timestamp, row["date"]).normalize())] = float(row["signal_day_return"])
    return lookup


def _published_at_for_signal_day(signal_day: pd.Timestamp, session_bucket: str, article_index: int) -> pd.Timestamp:
    local_signal_day = _local_midnight(signal_day)
    if session_bucket == "pre_market":
        hour = 7 if article_index % 2 == 0 else 8
        minute = 20 if article_index % 2 == 0 else 5
        published_local = local_signal_day + pd.Timedelta(hours=hour, minutes=minute)
    else:
        previous_signal_day = cast(pd.Timestamp, signal_day - pd.tseries.offsets.BDay(1))
        local_previous_day = _local_midnight(previous_signal_day)
        if session_bucket == "regular":
            hour = 11 if article_index % 2 == 0 else 14
            minute = 0 if article_index % 2 == 0 else 15
        else:
            hour = 17 if article_index % 2 == 0 else 18
            minute = 10 if article_index % 2 == 0 else 0
        published_local = local_previous_day + pd.Timedelta(hours=hour, minutes=minute)
    return published_local.tz_convert("UTC")


def _activation_state(ticker: str, signal_day: pd.Timestamp) -> str:
    draw = _unit_interval(f"active:{ticker}:{signal_day.date().isoformat()}")
    if draw < 0.40:
        return "both"
    if draw < 0.63:
        return "yahoo_finance_rss"
    if draw < 0.86:
        return "google_news_rss"
    return "none"


def _session_bucket_for_source(source_name: str, ticker: str, signal_day: pd.Timestamp) -> str:
    draw = _unit_interval(f"session:{source_name}:{ticker}:{signal_day.date().isoformat()}")
    if source_name == "yahoo_finance_rss":
        if draw < 0.60:
            return "pre_market"
        if draw < 0.82:
            return "post_market"
        return "regular"
    if draw < 0.18:
        return "pre_market"
    if draw < 0.52:
        return "regular"
    return "post_market"


def _source_session_skill(source_name: str, session_bucket: str) -> float:
    skill_map = {
        ("yahoo_finance_rss", "pre_market"): 0.08,
        ("yahoo_finance_rss", "regular"): 0.03,
        ("yahoo_finance_rss", "post_market"): 0.05,
        ("google_news_rss", "pre_market"): 0.02,
        ("google_news_rss", "regular"): 0.01,
        ("google_news_rss", "post_market"): 0.02,
    }
    return skill_map[(source_name, session_bucket)]


def _article_count(source_name: str, ticker: str, signal_day: pd.Timestamp, target_return: float) -> int:
    base_draw = _unit_interval(f"articles:{source_name}:{ticker}:{signal_day.date().isoformat()}")
    threshold = 0.20 + min(abs(target_return) / 0.03, 0.35)
    return 2 if base_draw < threshold else 1


def _article_sentiment(
    *,
    source_name: str,
    session_bucket: str,
    ticker: str,
    signal_day: pd.Timestamp,
    target_return: float,
    article_index: int,
) -> float:
    skill = _source_session_skill(source_name, session_bucket)
    scaled_return = float(np.clip(target_return / 0.015, -2.5, 2.5))
    source_bias = 0.00
    session_bias = {"pre_market": 0.01, "regular": -0.01, "post_market": 0.00}[session_bucket]
    noise_draw = _unit_interval(
        f"noise:{source_name}:{ticker}:{signal_day.date().isoformat()}:{article_index}"
    )
    noise = (noise_draw - 0.5) * (2.10 if source_name == "google_news_rss" else 1.80)
    sentiment = skill * scaled_return + source_bias + session_bias + noise
    return float(np.clip(sentiment, -1.0, 1.0))


def _article_relevance(
    *,
    source_name: str,
    session_bucket: str,
    ticker: str,
    signal_day: pd.Timestamp,
    sentiment: float,
    article_index: int,
) -> float:
    base = 0.42 + 0.20 * abs(sentiment)
    source_bonus = 0.05 if source_name == "yahoo_finance_rss" else 0.0
    session_bonus = {"pre_market": 0.05, "regular": -0.02, "post_market": 0.02}[session_bucket]
    noise = (_unit_interval(f"relevance:{source_name}:{ticker}:{signal_day.date().isoformat()}:{article_index}") - 0.5) * 0.12
    return float(np.clip(base + source_bonus + session_bonus + noise, 0.05, 1.0))


class _SyntheticRawNewsAdapter:
    def __init__(
        self,
        *,
        name: str,
        signal_day_return_lookup: dict[tuple[str, pd.Timestamp], float],
    ) -> None:
        self.name = name
        self.signal_day_return_lookup = signal_day_return_lookup
        self.last_fetch_metadata: dict[str, object] = {}

    def fetch_raw_items(self, request: NewsRequest) -> tuple[pd.DataFrame, dict[str, object]]:
        signal_days = business_dates_between(request.start_date, request.end_date)
        rows: list[dict[str, object]] = []
        for ticker in request.tickers:
            for signal_day_value in signal_days:
                signal_day = cast(pd.Timestamp, pd.Timestamp(signal_day_value).tz_convert("UTC").normalize())
                active_state = _activation_state(ticker, signal_day)
                if active_state not in {"both", self.name}:
                    continue
                session_bucket = _session_bucket_for_source(self.name, ticker, signal_day)
                target_return = self.signal_day_return_lookup.get((ticker, signal_day), 0.0)
                article_count = _article_count(self.name, ticker, signal_day, target_return)
                for article_index in range(article_count):
                    published_at = _published_at_for_signal_day(signal_day, session_bucket, article_index)
                    sentiment = _article_sentiment(
                        source_name=self.name,
                        session_bucket=session_bucket,
                        ticker=ticker,
                        signal_day=signal_day,
                        target_return=target_return,
                        article_index=article_index,
                    )
                    relevance = _article_relevance(
                        source_name=self.name,
                        session_bucket=session_bucket,
                        ticker=ticker,
                        signal_day=signal_day,
                        sentiment=sentiment,
                        article_index=article_index,
                    )
                    body = (
                        f"{ticker} synthetic {self.name} {session_bucket} "
                        f"signal_day={signal_day.date().isoformat()} article={article_index}"
                    )
                    rows.append(
                        {
                            "feed_ticker": ticker,
                            "published_at": published_at,
                            "body": body,
                            "headline_key": _headline_key(body),
                            "source_key": self.name,
                            "source_label": self.name,
                            "source_name": self.name,
                            "sentiment_score": sentiment,
                            "base_relevance_score": relevance,
                        }
                    )
        raw_items = pd.DataFrame(rows)
        metadata = {
            "origins": ["synthetic"],
            "cache_used": False,
            "snapshot_used": False,
            "stale_cache_used": False,
            "requests": [
                {
                    "origin": "synthetic",
                    "cache_used": False,
                    "snapshot_used": False,
                    "stale_cache_used": False,
                    "source_name": self.name,
                    "record_count": int(len(raw_items)),
                }
            ],
        }
        self.last_fetch_metadata = metadata
        return raw_items, metadata


class _SweepDataAgent(DataAgent):
    def __init__(self, settings: Settings, store: ParquetStore) -> None:
        super().__init__(settings, store)
        self._signal_day_return_lookup: dict[tuple[str, pd.Timestamp], float] = {}

    def _fetch_ohlcv_with_fallback(
        self,
        request: OHLCVRequest,
        as_of_timestamp: pd.Timestamp,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        frame = DummyOHLCVAdapter(seed=self.settings.app.seed, mode=self.settings.data.dummy_mode).fetch(request)
        self._signal_day_return_lookup = _signal_day_return_lookup(frame)
        return frame, {
            "requested_source": "dummy",
            "used_source": "dummy",
            "dummy_mode": self.settings.data.dummy_mode,
            "fallback_used": False,
            "fallback_reason": None,
            "attempt_errors": [],
            "as_of_time": as_of_timestamp.isoformat(),
        }

    def _build_macro_adapter(self) -> DummyMacroAdapter:
        return DummyMacroAdapter(seed=self.settings.app.seed)

    def _fetch_fundamentals_with_fallback(
        self,
        request: FundamentalsRequest,
        *,
        as_of_timestamp: pd.Timestamp,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        frame = OfflineFundamentalProxyAdapter(seed=self.settings.app.seed).fetch(request)
        return frame, {
            "requested_source": "offline_fundamental_proxy",
            "used_source": "offline_fundamental_proxy",
            "fallback_used": False,
            "fallback_reason": None,
            "attempt_errors": [],
            "as_of_time": as_of_timestamp.isoformat(),
            "transport": {},
        }

    def _build_news_adapter(self, source_name: str | None = None) -> _SyntheticRawNewsAdapter:
        resolved_source = source_name or self.settings.data.news_source
        if resolved_source not in {"yahoo_finance_rss", "google_news_rss"}:
            raise ValueError(f"Unsupported sweep news source: {resolved_source}")
        return _SyntheticRawNewsAdapter(
            name=resolved_source,
            signal_day_return_lookup=self._signal_day_return_lookup,
        )


@dataclass(slots=True)
class PublicSweepDatasetProfile:
    tickers: list[str]
    start_date: str
    end_date: str
    notes: list[str]
    snapshot_summary: dict[str, object]


def _public_transport(settings: Settings) -> PublicDataTransportConfig:
    public_data = settings.data.public_data
    return PublicDataTransportConfig(
        cache_dir=resolve_repo_path(public_data.cache_path),
        snapshot_dir=resolve_repo_path(public_data.snapshot_path),
        cache_ttl_hours=public_data.cache_ttl_hours,
        retry_count=public_data.retry_count,
        retry_backoff_seconds=public_data.retry_backoff_seconds,
    )


def _snapshot_only_transport(settings: Settings) -> PublicDataTransportConfig:
    transport = _public_transport(settings)
    return PublicDataTransportConfig(
        cache_dir=transport.snapshot_dir / "__snapshot_only_disabled_cache__",
        snapshot_dir=transport.snapshot_dir,
        cache_ttl_hours=transport.cache_ttl_hours,
        retry_count=transport.retry_count,
        retry_backoff_seconds=transport.retry_backoff_seconds,
    )


@contextmanager
def _disable_public_network() -> Iterator[None]:
    original = data_adapters_module._read_text_response

    def _blocked(url: str, headers: dict[str, str] | None = None) -> str:
        del headers
        raise RuntimeError(f"{_SNAPSHOT_ONLY_ERROR}: {url}")

    data_adapters_module._read_text_response = _blocked
    try:
        yield
    finally:
        data_adapters_module._read_text_response = original


def _load_snapshot_payloads(snapshot_dir: Path, namespace: str) -> list[dict[str, object]]:
    namespace_root = snapshot_dir / namespace
    if not namespace_root.exists():
        return []
    payloads: list[dict[str, object]] = []
    for path in sorted(namespace_root.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            payloads.append(payload)
    return payloads


def _snapshot_ticker(namespace: str, url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    if namespace == "yahoo_chart":
        return parsed.path.rsplit("/", maxsplit=1)[-1].upper()
    if namespace == "yahoo_finance_rss":
        return str(query.get("s", [""])[0]).upper()
    if namespace == "google_news_rss":
        query_text = str(query.get("q", [""])[0])
        match = _GOOGLE_TICKER_RE.search(query_text)
        return match.group(1).upper() if match else ""
    return ""


def _snapshot_ohlcv_bounds(url: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    period1 = query.get("period1")
    period2 = query.get("period2")
    if not period1 or not period2:
        return None
    start = pd.Timestamp(int(period1[0]), unit="s", tz="UTC").normalize()
    end = (pd.Timestamp(int(period2[0]), unit="s", tz="UTC") - pd.Timedelta(days=1)).normalize()
    return start, end


def _rss_article_bounds(feed_text: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    try:
        root = ET.fromstring(feed_text)
    except ET.ParseError:
        return None
    dates: list[pd.Timestamp] = []
    for item in root.findall(".//item"):
        published_text = item.findtext("pubDate") or item.findtext("published")
        published_at = pd.to_datetime(published_text, utc=True, errors="coerce")
        if pd.notna(published_at):
            dates.append(cast(pd.Timestamp, pd.Timestamp(published_at).normalize()))
    if not dates:
        return None
    return min(dates), max(dates)


def _summarize_snapshot_windows(
    *,
    payloads: list[dict[str, object]],
    namespace: str,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    by_ticker: dict[str, dict[str, object]] = {}
    for payload in payloads:
        url = str(payload.get("url", ""))
        ticker = _snapshot_ticker(namespace, url)
        if not ticker:
            continue
        bounds = (
            _snapshot_ohlcv_bounds(url)
            if namespace == "yahoo_chart"
            else _rss_article_bounds(str(payload.get("content", "")))
        )
        if bounds is None:
            continue
        start, end = bounds
        stats = by_ticker.setdefault(
            ticker,
            {
                "ticker": ticker,
                "file_count": 0,
                "min_start": start,
                "max_end": end,
            },
        )
        stats["file_count"] = _as_int(stats.get("file_count")) + 1
        stats["min_start"] = min(cast(pd.Timestamp, stats["min_start"]), start)
        stats["max_end"] = max(cast(pd.Timestamp, stats["max_end"]), end)
    summary_rows = [
        {
            "ticker": ticker,
            "file_count": _as_int(stats.get("file_count")),
            "min_start": cast(pd.Timestamp, stats["min_start"]).date().isoformat(),
            "max_end": cast(pd.Timestamp, stats["max_end"]).date().isoformat(),
        }
        for ticker, stats in sorted(by_ticker.items())
    ]
    return by_ticker, summary_rows


def _discover_public_dataset_profile(settings: Settings, args: argparse.Namespace) -> PublicSweepDatasetProfile:
    snapshot_dir = resolve_repo_path(settings.data.public_data.snapshot_path)
    yahoo_chart_payloads = _load_snapshot_payloads(snapshot_dir, "yahoo_chart")
    yahoo_rss_payloads = _load_snapshot_payloads(snapshot_dir, "yahoo_finance_rss")
    google_rss_payloads = _load_snapshot_payloads(snapshot_dir, "google_news_rss")
    ohlcv_by_ticker, ohlcv_summary = _summarize_snapshot_windows(payloads=yahoo_chart_payloads, namespace="yahoo_chart")
    yahoo_news_by_ticker, yahoo_news_summary = _summarize_snapshot_windows(
        payloads=yahoo_rss_payloads,
        namespace="yahoo_finance_rss",
    )
    google_news_by_ticker, google_news_summary = _summarize_snapshot_windows(
        payloads=google_rss_payloads,
        namespace="google_news_rss",
    )
    common_tickers = sorted(set(ohlcv_by_ticker) & set(yahoo_news_by_ticker) & set(google_news_by_ticker))
    if not common_tickers:
        raise RuntimeError(
            "No common tickers were found across yahoo_chart, yahoo_finance_rss, and google_news_rss snapshots."
        )
    max_tickers = DEFAULT_REDUCED_TICKER_COUNT if args.ticker_count is None else max(1, int(args.ticker_count))
    tickers = common_tickers[:max_tickers]
    if not tickers:
        raise RuntimeError("Snapshot discovery resolved an empty public ticker set.")
    ohlcv_common_start = max(cast(pd.Timestamp, ohlcv_by_ticker[ticker]["min_start"]) for ticker in tickers)
    ohlcv_common_end = min(cast(pd.Timestamp, ohlcv_by_ticker[ticker]["max_end"]) for ticker in tickers)
    yahoo_news_end = min(cast(pd.Timestamp, yahoo_news_by_ticker[ticker]["max_end"]) for ticker in tickers)
    google_news_start = min(cast(pd.Timestamp, google_news_by_ticker[ticker]["min_start"]) for ticker in tickers)
    google_news_end = min(cast(pd.Timestamp, google_news_by_ticker[ticker]["max_end"]) for ticker in tickers)
    start = max(ohlcv_common_start, google_news_start)
    end = min(ohlcv_common_end, yahoo_news_end, google_news_end)
    if args.days is not None:
        requested_start = cast(pd.Timestamp, (end - pd.tseries.offsets.BDay(max(int(args.days) - 1, 0))).normalize())
        start = max(start, requested_start)
    if start > end:
        raise RuntimeError(
            "Public snapshot discovery produced an empty shared date window. "
            f"start={start.date().isoformat()} end={end.date().isoformat()}"
        )
    notes = [
        "Public mode disables network reads and uses cache/snapshot payloads only.",
        f"Resolved common snapshot-backed tickers: {', '.join(tickers)}.",
        (
            "Google News article coverage starts earlier than Yahoo Finance RSS for this snapshot set, "
            "so learned weights train on a longer single-source lead-in than the multi-source overlap window."
        ),
    ]
    snapshot_summary = {
        "snapshot_dir": str(snapshot_dir),
        "file_counts": {
            "yahoo_chart": len(yahoo_chart_payloads),
            "yahoo_finance_rss": len(yahoo_rss_payloads),
            "google_news_rss": len(google_rss_payloads),
        },
        "available_tickers": {
            "yahoo_chart": sorted(ohlcv_by_ticker),
            "yahoo_finance_rss": sorted(yahoo_news_by_ticker),
            "google_news_rss": sorted(google_news_by_ticker),
            "common": common_tickers,
            "selected": tickers,
        },
        "window": {
            "ohlcv_common_start": ohlcv_common_start.date().isoformat(),
            "ohlcv_common_end": ohlcv_common_end.date().isoformat(),
            "google_news_start": google_news_start.date().isoformat(),
            "google_news_end": google_news_end.date().isoformat(),
            "yahoo_finance_news_end": yahoo_news_end.date().isoformat(),
            "selected_start": start.date().isoformat(),
            "selected_end": end.date().isoformat(),
        },
        "per_ticker": {
            "yahoo_chart": ohlcv_summary,
            "yahoo_finance_rss": yahoo_news_summary,
            "google_news_rss": google_news_summary,
        },
    }
    return PublicSweepDatasetProfile(
        tickers=tickers,
        start_date=start.date().isoformat(),
        end_date=end.date().isoformat(),
        notes=notes,
        snapshot_summary=snapshot_summary,
    )


def _build_public_ohlcv_adapter(
    source_name: str,
    *,
    transport: PublicDataTransportConfig,
) -> YahooChartOHLCVAdapter | FredMarketProxyOHLCVAdapter:
    if source_name == "yahoo_chart":
        return YahooChartOHLCVAdapter(transport=transport)
    if source_name == "fred_market_proxy":
        return FredMarketProxyOHLCVAdapter(transport=transport)
    raise ValueError(f"Unsupported public OHLCV source for sweep: {source_name}")


def _empty_raw_news_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
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
    )


def _fetch_public_ohlcv(
    *,
    settings: Settings,
    transport: PublicDataTransportConfig,
    tickers: list[str],
    start_date: str,
    end_date: str,
    as_of_time: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, object]]:
    requested_sources = [settings.data.primary_source]
    if settings.data.fallback_source and settings.data.fallback_source not in requested_sources:
        requested_sources.append(settings.data.fallback_source)
    frames: list[pd.DataFrame] = []
    request_records: list[dict[str, object]] = []
    ticker_records: list[dict[str, object]] = []
    skipped_tickers: list[dict[str, object]] = []
    for ticker in tickers:
        attempt_errors: list[dict[str, str]] = []
        first_error: str | None = None
        for index, source_name in enumerate(requested_sources):
            try:
                adapter = _build_public_ohlcv_adapter(source_name, transport=transport)
                frame = adapter.fetch(OHLCVRequest(tickers=[ticker], start_date=start_date, end_date=end_date))
                if frame.empty:
                    raise RuntimeError(f"{source_name} returned no OHLCV rows for {ticker}.")
                transport_metadata = cast(dict[str, object], getattr(adapter, "last_fetch_metadata", {}))
                request_records.extend(cast(list[dict[str, object]], transport_metadata.get("requests", [])) or [])
                ticker_records.append(
                    {
                        "ticker": ticker,
                        "used_source": source_name,
                        "fallback_used": index > 0,
                        "fallback_reason": first_error,
                        "attempt_errors": attempt_errors,
                    }
                )
                frames.append(frame)
                break
            except (RuntimeError, ValueError, OSError) as exc:
                attempt_errors.append({"source": source_name, "error": str(exc)})
                if first_error is None:
                    first_error = str(exc)
        else:
            skipped_tickers.append({"ticker": ticker, "attempt_errors": attempt_errors})
    if not frames:
        raise RuntimeError("Public sweep could not load OHLCV data from snapshots for any ticker.")
    return pd.concat(frames, ignore_index=True), {
        "requested_source": settings.data.primary_source,
        "requested_sources": requested_sources,
        "used_tickers": [str(item["ticker"]) for item in ticker_records],
        "skipped_tickers": skipped_tickers,
        "ticker_records": ticker_records,
        "transport": cast(dict[str, object], data_adapters_module._summarize_public_fetches(request_records)),
        "as_of_time": as_of_time.isoformat(),
    }


def _fetch_public_news(
    *,
    settings: Settings,
    transport: PublicDataTransportConfig,
    tickers: list[str],
    start_date: str,
    end_date: str,
    as_of_time: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, object]]:
    requested_sources = [settings.data.news_source]
    for source_name in settings.data.news_secondary_sources:
        if source_name and source_name not in requested_sources:
            requested_sources.append(source_name)
    frames: list[pd.DataFrame] = []
    request_records: list[dict[str, object]] = []
    ticker_records: list[dict[str, object]] = []
    skipped_tickers: list[dict[str, object]] = []
    for ticker in tickers:
        try:
            adapter = MultiSourceNewsAdapter(
                [
                    YahooFinanceNewsAdapter(
                        transport=transport,
                        source_weights=settings.data.news_source_weights,
                        session_weights=settings.data.news_session_weights,
                    ),
                    GoogleNewsRssAdapter(
                        transport=transport,
                        source_weights=settings.data.news_source_weights,
                        session_weights=settings.data.news_session_weights,
                    ),
                ],
                source_weights=settings.data.news_source_weights,
                session_weights=settings.data.news_session_weights,
            )
            frame = adapter.fetch(NewsRequest(tickers=[ticker], start_date=start_date, end_date=end_date))
            transport_metadata = cast(dict[str, object], getattr(adapter, "last_fetch_metadata", {}))
            request_records.extend(cast(list[dict[str, object]], transport_metadata.get("requests", [])) or [])
            ticker_records.append(
                {
                    "ticker": ticker,
                    "used_sources": [str(item) for item in cast(list[object], transport_metadata.get("used_sources", []))],
                    "failed_sources": cast(list[dict[str, str]], transport_metadata.get("failed_sources", [])),
                }
            )
            frames.append(frame)
        except (RuntimeError, ValueError, OSError) as exc:
            skipped_tickers.append({"ticker": ticker, "error": str(exc)})
    raw_news = pd.concat(frames, ignore_index=True) if frames else _empty_raw_news_frame()
    return raw_news, {
        "requested_sources": requested_sources,
        "used_tickers": [str(item["ticker"]) for item in ticker_records],
        "skipped_tickers": skipped_tickers,
        "ticker_records": ticker_records,
        "transport": cast(dict[str, object], data_adapters_module._summarize_public_fetches(request_records)),
        "as_of_time": as_of_time.isoformat(),
    }


def _generate_public_sweep_data(
    *,
    settings: Settings,
    args: argparse.Namespace,
    as_of_time: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, PublicSweepDatasetProfile, dict[str, object]]:
    profile = _discover_public_dataset_profile(settings, args)
    public_settings = update_settings(
        settings,
        data={
            "source_mode": "public",
            "primary_source": "yahoo_chart",
            "fallback_source": "fred_market_proxy",
            "news_source": "yahoo_finance_rss",
            "news_secondary_sources": ["google_news_rss"],
            "news_fallback_source": "",
        },
    )
    transport = _snapshot_only_transport(public_settings)
    with _disable_public_network():
        raw_ohlcv, ohlcv_metadata = _fetch_public_ohlcv(
            settings=public_settings,
            transport=transport,
            tickers=profile.tickers,
            start_date=profile.start_date,
            end_date=profile.end_date,
            as_of_time=as_of_time,
        )
        raw_news, news_metadata = _fetch_public_news(
            settings=public_settings,
            transport=transport,
            tickers=profile.tickers,
            start_date=profile.start_date,
            end_date=profile.end_date,
            as_of_time=as_of_time,
        )
    processed_ohlcv = apply_stale_flag(
        normalize_ohlcv(raw_ohlcv),
        as_of_time=as_of_time,
        threshold_hours=public_settings.data.stale_threshold_hours.ohlcv,
    )
    processed_news = apply_stale_flag(
        normalize_news(raw_news),
        as_of_time=as_of_time,
        threshold_hours=public_settings.data.stale_threshold_hours.news,
    )
    metadata = {
        "ohlcv": ohlcv_metadata,
        "news": news_metadata,
        "snapshot_summary": profile.snapshot_summary,
    }
    return processed_ohlcv, processed_news, profile, metadata


def _build_coverage_panel(ohlcv: pd.DataFrame, *, coverage_extension_business_days: int = 3) -> pd.DataFrame:
    panel = ohlcv.loc[:, ["ticker", "timestamp_utc", "close"]].copy()
    panel["date"] = pd.to_datetime(panel["timestamp_utc"], utc=True).dt.normalize()
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel["signal_day_return"] = panel.groupby("ticker")["close"].pct_change()
    latest_panel_date = cast(pd.Timestamp, panel["date"].max())
    future_dates = [
        cast(pd.Timestamp, (latest_panel_date + pd.tseries.offsets.BDay(offset)).normalize())
        for offset in range(1, max(int(coverage_extension_business_days), 0) + 1)
    ]
    if not future_dates:
        return panel
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
    return pd.concat([panel, future_rows], ignore_index=True).sort_values(["ticker", "date"]).reset_index(drop=True)


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


def _signal_summary(
    *,
    panel: pd.DataFrame,
    signal: pd.Series,
    relevance: pd.Series,
    multi_source_only: bool,
) -> dict[str, float | int]:
    weight = _effective_news_weight(panel=panel, relevance=relevance)
    active_mask = weight > 0.0
    if multi_source_only:
        active_mask = active_mask & (panel["source_count"] >= 2.0)
    signal_days = int(active_mask.sum())
    overlap_rate = float((active_mask & panel["signal_day_return"].notna()).sum() / signal_days) if signal_days else 0.0
    return {
        "coverage": float(active_mask.mean()),
        "signal_days": signal_days,
        "abs_ic": abs(_correlation_or_zero(signal.where(active_mask), panel["signal_day_return"])),
        "overlap_rate": overlap_rate,
    }


def _fixed_signal(panel: pd.DataFrame) -> pd.Series:
    return panel["sentiment_score_source_session_weighted"].fillna(panel["sentiment_score"]).fillna(0.0)


def _fixed_relevance(panel: pd.DataFrame) -> pd.Series:
    return panel["relevance_score_source_session_weighted"].fillna(panel["relevance_score"]).clip(lower=0.0, upper=1.0)


def _unweighted_signal(panel: pd.DataFrame) -> pd.Series:
    return panel["sentiment_score_unweighted"].fillna(panel["sentiment_score"]).fillna(0.0)


def _unweighted_relevance(panel: pd.DataFrame) -> pd.Series:
    return panel["relevance_score_unweighted"].fillna(panel["relevance_score"]).clip(lower=0.0, upper=1.0)


def _build_eval_context(ohlcv: pd.DataFrame, news: pd.DataFrame) -> dict[str, Any]:
    coverage_panel = _build_coverage_panel(ohlcv)
    news_panel = _prepare_news_panel(coverage_panel=coverage_panel, news=news)
    fixed_signal = _fixed_signal(news_panel)
    fixed_relevance = _fixed_relevance(news_panel)
    unweighted_signal = _unweighted_signal(news_panel)
    unweighted_relevance = _unweighted_relevance(news_panel)
    return {
        "news": news,
        "news_panel": news_panel,
        "fixed_signal": fixed_signal,
        "fixed_relevance": fixed_relevance,
        "unweighted_signal": unweighted_signal,
        "unweighted_relevance": unweighted_relevance,
        "fixed_overall": _signal_summary(
            panel=news_panel,
            signal=fixed_signal,
            relevance=fixed_relevance,
            multi_source_only=False,
        ),
        "fixed_multi_source": _signal_summary(
            panel=news_panel,
            signal=fixed_signal,
            relevance=fixed_relevance,
            multi_source_only=True,
        ),
        "unweighted_overall": _signal_summary(
            panel=news_panel,
            signal=unweighted_signal,
            relevance=unweighted_relevance,
            multi_source_only=False,
        ),
        "unweighted_multi_source": _signal_summary(
            panel=news_panel,
            signal=unweighted_signal,
            relevance=unweighted_relevance,
            multi_source_only=True,
        ),
    }


def _worker_init(ohlcv_path: str, news_path: str) -> None:
    ohlcv = pd.read_parquet(ohlcv_path)
    news = pd.read_parquet(news_path)
    _WORKER_CONTEXT.clear()
    _WORKER_CONTEXT.update(_build_eval_context(ohlcv=ohlcv, news=news))


def _evaluate_config(config_payload: dict[str, object]) -> dict[str, object]:
    panel = cast(pd.DataFrame, _WORKER_CONTEXT["news_panel"])
    news = cast(pd.DataFrame, _WORKER_CONTEXT["news"])
    fixed_signal = cast(pd.Series, _WORKER_CONTEXT["fixed_signal"])
    fixed_relevance = cast(pd.Series, _WORKER_CONTEXT["fixed_relevance"])
    config = LearnedWeightingConfig(
        regularization_lambda=_as_float(config_payload["regularization_lambda"]),
        min_samples=_as_int(config_payload["min_samples"]),
        lookback_days=_as_int(config_payload["lookback_days"]),
        target=str(config_payload["target"]),
        min_weight=_as_float(config_payload["min_weight"]),
        fallback_mode=str(config_payload["fallback_mode"]),
    )
    learned_result = build_walk_forward_learned_weighting(
        news_panel=panel,
        news=news,
        config=config,
        fallback_sentiment=fixed_signal,
        fallback_relevance=fixed_relevance,
    )
    learned_overall = _signal_summary(
        panel=panel,
        signal=learned_result.sentiment,
        relevance=learned_result.relevance,
        multi_source_only=False,
    )
    learned_multi_source = _signal_summary(
        panel=panel,
        signal=learned_result.sentiment,
        relevance=learned_result.relevance,
        multi_source_only=True,
    )
    fixed_multi_source = cast(dict[str, float | int], _WORKER_CONTEXT["fixed_multi_source"])
    unweighted_multi_source = cast(dict[str, float | int], _WORKER_CONTEXT["unweighted_multi_source"])
    summary = cast(dict[str, object], learned_result.summary)
    fit_day_count = _as_int(summary.get("fit_day_count"))
    fallback_day_count = _as_int(summary.get("fallback_day_count"))
    total_eval_days = fit_day_count + fallback_day_count
    return {
        "regularization_lambda": config.regularization_lambda,
        "min_samples": config.min_samples,
        "lookback_days": config.lookback_days,
        "target": config.target,
        "abs_ic": _as_float(learned_multi_source.get("abs_ic")),
        "overlap_rate": _as_float(learned_multi_source.get("overlap_rate")),
        "coverage": _as_float(learned_multi_source.get("coverage")),
        "signal_days": _as_int(learned_multi_source.get("signal_days")),
        "fallback_rate": _as_float(summary.get("fallback_rate")),
        "fit_day_count": fit_day_count,
        "fallback_day_count": fallback_day_count,
        "total_eval_days": total_eval_days,
        "fit_rate": float(fit_day_count / max(total_eval_days, 1)),
        "abs_ic_vs_fixed": _as_float(learned_multi_source.get("abs_ic")) - _as_float(fixed_multi_source.get("abs_ic")),
        "abs_ic_vs_unweighted": _as_float(learned_multi_source.get("abs_ic"))
        - _as_float(unweighted_multi_source.get("abs_ic")),
        "overlap_rate_vs_fixed": _as_float(learned_multi_source.get("overlap_rate"))
        - _as_float(fixed_multi_source.get("overlap_rate")),
        "overlap_rate_vs_unweighted": _as_float(learned_multi_source.get("overlap_rate"))
        - _as_float(unweighted_multi_source.get("overlap_rate")),
        "overall_abs_ic": _as_float(learned_overall.get("abs_ic")),
        "overall_overlap_rate": _as_float(learned_overall.get("overlap_rate")),
        "overall_coverage": _as_float(learned_overall.get("coverage")),
    }


def _ranking_key(result: dict[str, object]) -> tuple[float, float, float, float, int]:
    return (
        _as_float(result.get("abs_ic")),
        _as_float(result.get("overlap_rate")),
        _as_float(result.get("abs_ic_vs_fixed")),
        -_as_float(result.get("fallback_rate")),
        _as_int(result.get("fit_day_count")),
    )


def _distribution(results: list[dict[str, object]], field: str) -> dict[str, float]:
    values = [_as_float(result.get(field)) for result in results]
    if not values:
        return {"best": 0.0, "worst": 0.0, "median": 0.0}
    return {
        "best": float(max(values)),
        "worst": float(min(values)),
        "median": float(np.median(np.asarray(values, dtype=float))),
    }


def _stage_summary(results: list[dict[str, object]], *, top_k: int) -> dict[str, object]:
    return {
        "run_count": len(results),
        "top_results": results[:top_k],
        "best": results[0] if results else None,
        "worst": results[-1] if results else None,
        "distributions": {
            "abs_ic": _distribution(results, "abs_ic"),
            "overlap_rate": _distribution(results, "overlap_rate"),
            "fallback_rate": _distribution(results, "fallback_rate"),
            "abs_ic_vs_fixed": _distribution(results, "abs_ic_vs_fixed"),
            "abs_ic_vs_unweighted": _distribution(results, "abs_ic_vs_unweighted"),
        },
    }


def _run_stage(
    *,
    stage_name: str,
    configs: list[dict[str, object]],
    ohlcv_path: Path,
    news_path: Path,
    max_workers: int,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    worker_count = max(1, min(max_workers, len(configs)))
    print(f"{stage_name}: running {len(configs)} configurations with up to {worker_count} worker(s)", flush=True)

    def _collect(executor: ProcessPoolExecutor | ThreadPoolExecutor) -> None:
        future_map = {executor.submit(_evaluate_config, config): config for config in configs}
        completed = 0
        total = len(configs)
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            completed += 1
            if completed == total or completed % max(worker_count, 5) == 0:
                print(f"{stage_name}: completed {completed}/{total}", flush=True)

    try:
        with ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_worker_init,
            initargs=(str(ohlcv_path), str(news_path)),
        ) as executor:
            _collect(executor)
    except PermissionError:
        print(f"{stage_name}: process workers unavailable, falling back to thread pool", flush=True)
        _worker_init(str(ohlcv_path), str(news_path))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            _collect(executor)

    results.sort(key=_ranking_key, reverse=True)
    for rank, result in enumerate(results, start=1):
        result["rank"] = rank
    return results


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _process_pool_available() -> bool:
    try:
        with ProcessPoolExecutor(max_workers=1) as executor:
            executor.shutdown(wait=True, cancel_futures=True)
        return True
    except PermissionError:
        return False


def _dataset_profile(
    settings: Settings,
    args: argparse.Namespace,
    *,
    process_pool_available: bool,
) -> tuple[int, int, list[str]]:
    notes: list[str] = []
    if args.ticker_count is not None or args.days is not None:
        return (
            args.ticker_count or settings.data.dummy_ticker_count,
            args.days or settings.data.dummy_days,
            notes,
        )
    ticker_count = settings.data.dummy_ticker_count
    day_count = settings.data.dummy_days
    if process_pool_available and (ticker_count > DEFAULT_REDUCED_TICKER_COUNT or day_count > DEFAULT_REDUCED_DAY_COUNT):
        notes.append(
            "Reduced dummy profile to 20 tickers x 400 business days because the configured profile "
            "would exceed the offline sweep budget."
        )
        return DEFAULT_REDUCED_TICKER_COUNT, DEFAULT_REDUCED_DAY_COUNT, notes
    if (not process_pool_available) and (
        ticker_count > DEFAULT_THREADSAFE_TICKER_COUNT or day_count > DEFAULT_THREADSAFE_DAY_COUNT
    ):
        notes.append(
            "Process workers are unavailable in this environment, so the dummy profile was reduced to "
            "8 tickers x 260 business days to keep the offline sweep runnable."
        )
        return DEFAULT_THREADSAFE_TICKER_COUNT, DEFAULT_THREADSAFE_DAY_COUNT, notes
    return ticker_count, day_count, notes


def _build_stage1_configs(min_weight: float, fallback_mode: str) -> list[dict[str, object]]:
    return [
        {
            "regularization_lambda": regularization_lambda,
            "min_samples": min_samples,
            "lookback_days": lookback_days,
            "target": "abs_ic",
            "min_weight": min_weight,
            "fallback_mode": fallback_mode,
        }
        for regularization_lambda in STAGE1_LAMBDAS
        for min_samples in STAGE1_MIN_SAMPLES
        for lookback_days in STAGE1_LOOKBACK_DAYS
    ]


def _build_stage2_configs(
    stage1_results: list[dict[str, object]],
    *,
    min_weight: float,
    fallback_mode: str,
    top_k: int,
) -> list[dict[str, object]]:
    return [
        {
            "regularization_lambda": _as_float(result.get("regularization_lambda")),
            "min_samples": _as_int(result.get("min_samples")),
            "lookback_days": _as_int(result.get("lookback_days")),
            "target": target,
            "min_weight": min_weight,
            "fallback_mode": fallback_mode,
        }
        for result in stage1_results[:top_k]
        for target in STAGE2_TARGETS
    ]


def _recommended_preset(best_result: dict[str, object], *, min_weight: float, fallback_mode: str) -> dict[str, object]:
    return {
        "regularization_lambda": _as_float(best_result.get("regularization_lambda")),
        "min_samples": _as_int(best_result.get("min_samples")),
        "lookback_days": _as_int(best_result.get("lookback_days")),
        "target": str(best_result.get("target", "abs_ic")),
        "min_weight": min_weight,
        "fallback_mode": fallback_mode,
        "evaluation": {
            "abs_ic": _as_float(best_result.get("abs_ic")),
            "overlap_rate": _as_float(best_result.get("overlap_rate")),
            "fallback_rate": _as_float(best_result.get("fallback_rate")),
            "fit_day_count": _as_int(best_result.get("fit_day_count")),
            "abs_ic_vs_fixed": _as_float(best_result.get("abs_ic_vs_fixed")),
            "abs_ic_vs_unweighted": _as_float(best_result.get("abs_ic_vs_unweighted")),
        },
    }


def _generate_sweep_data(
    *,
    settings: Settings,
    ticker_count: int,
    day_count: int,
    as_of_time: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sweep_settings = update_settings(
        settings,
        data={
            "source_mode": "live",
            "dummy_mode": settings.data.dummy_mode,
            "storage_path": settings.data.storage_path,
            "news_source": "yahoo_finance_rss",
            "news_secondary_sources": ["google_news_rss"],
            "news_fallback_source": "offline_news_proxy",
            "dummy_ticker_count": ticker_count,
            "dummy_days": day_count,
        },
    )
    store = ParquetStore(resolve_storage_path(sweep_settings))
    agent = _SweepDataAgent(sweep_settings, store)
    end_date = as_of_time.normalize()
    start_date = cast(pd.Timestamp, end_date - pd.tseries.offsets.BDay(day_count - 1))
    tickers = [f"TICK{index:03d}" for index in range(ticker_count)]
    artifacts = agent.generate_or_fetch(
        tickers=tickers,
        start_date=start_date.date().isoformat(),
        end_date=end_date.date().isoformat(),
        as_of_time=as_of_time,
    )
    return artifacts.processed_ohlcv, artifacts.processed_news


def _transport_brief(transport: dict[str, object]) -> dict[str, object]:
    return {
        "origins": [str(item) for item in cast(list[object], transport.get("origins", []))],
        "cache_used": bool(transport.get("cache_used", False)),
        "snapshot_used": bool(transport.get("snapshot_used", False)),
        "stale_cache_used": bool(transport.get("stale_cache_used", False)),
    }


def _dataset_quality_summary(
    *,
    processed_ohlcv: pd.DataFrame,
    processed_news: pd.DataFrame,
    news_panel: pd.DataFrame,
) -> dict[str, object]:
    active_news_mask = processed_news["headline_count"] > 0 if not processed_news.empty else pd.Series(dtype=bool)
    active_panel_mask = news_panel["headline_count"] > 0 if not news_panel.empty else pd.Series(dtype=bool)
    return {
        "ticker_count": int(processed_ohlcv["ticker"].nunique()) if not processed_ohlcv.empty else 0,
        "tickers": sorted(processed_ohlcv["ticker"].astype(str).unique().tolist()) if not processed_ohlcv.empty else [],
        "business_day_count": (
            int(pd.to_datetime(processed_ohlcv["timestamp_utc"], utc=True).dt.normalize().nunique())
            if not processed_ohlcv.empty
            else 0
        ),
        "ohlcv_rows": int(len(processed_ohlcv)),
        "news_daily_rows": int(len(processed_news)),
        "news_article_count": int(pd.to_numeric(processed_news["headline_count"], errors="coerce").fillna(0).sum())
        if not processed_news.empty
        else 0,
        "news_active_days": int(active_panel_mask.sum()),
        "news_missing_rate": float(1.0 - active_panel_mask.mean()) if len(news_panel) else 1.0,
        "news_stale_rate": float(processed_news["stale_data_flag"].mean()) if not processed_news.empty else 0.0,
        "return_ready_rows": int(news_panel["signal_day_return"].notna().sum()) if not news_panel.empty else 0,
        "multi_source_rows": int((news_panel["source_count"] >= 2.0).sum()) if not news_panel.empty else 0,
        "multi_source_rate": float((news_panel["source_count"] >= 2.0).mean()) if len(news_panel) else 0.0,
        "source_mix_counts": (
            processed_news.loc[active_news_mask, "source_mix"].astype(str).value_counts().astype(int).to_dict()
            if not processed_news.empty
            else {}
        ),
        "session_bucket_counts": (
            processed_news.loc[active_news_mask, "session_bucket"].astype(str).value_counts().astype(int).to_dict()
            if not processed_news.empty
            else {}
        ),
        "source_count_distribution": (
            news_panel.loc[active_panel_mask, "source_count"].astype(int).value_counts().astype(int).to_dict()
            if not news_panel.empty
            else {}
        ),
    }


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    configure_logging(settings.app.log_level)
    process_pool_available = _process_pool_available()
    min_weight = float(settings.data.learned_weighting.min_weight)
    fallback_mode = str(settings.data.learned_weighting.fallback_mode)
    max_workers = max(1, int(args.max_workers))
    storage_path = resolve_storage_path(settings)
    output_dir = storage_path / OUTPUT_SUBDIR_BY_MODE[args.mode]
    output_dir.mkdir(parents=True, exist_ok=True)
    as_of_time = pd.Timestamp("2026-04-08T00:00:00Z")
    dataset_profile_summary: dict[str, object]
    notes: list[str]
    metadata: dict[str, object] = {}
    if args.mode == "public":
        processed_ohlcv, processed_news, public_profile, metadata = _generate_public_sweep_data(
            settings=settings,
            args=args,
            as_of_time=as_of_time,
        )
        notes = list(public_profile.notes)
        dataset_profile_summary = {
            "mode": "public",
            "source_mode": "public",
            "tickers": public_profile.tickers,
            "ticker_count": len(public_profile.tickers),
            "start_date": public_profile.start_date,
            "end_date": public_profile.end_date,
            "as_of_date": as_of_time.date().isoformat(),
            "snapshot_summary": public_profile.snapshot_summary,
        }
        print(
            "Generating snapshot-backed public sweep data for "
            f"{len(public_profile.tickers)} tickers from {public_profile.start_date} to {public_profile.end_date}",
            flush=True,
        )
    else:
        ticker_count, day_count, notes = _dataset_profile(
            settings,
            args,
            process_pool_available=process_pool_available,
        )
        dataset_profile_summary = {
            "mode": "dummy",
            "source_mode": "dummy",
            "dummy_mode": settings.data.dummy_mode,
            "ticker_count": ticker_count,
            "business_days": day_count,
            "as_of_date": as_of_time.date().isoformat(),
        }
        print(
            f"Generating synthetic multi-source sweep data for {ticker_count} tickers x {day_count} business days "
            f"at {as_of_time.date().isoformat()}",
            flush=True,
        )
        processed_ohlcv, processed_news = _generate_sweep_data(
            settings=settings,
            ticker_count=ticker_count,
            day_count=day_count,
            as_of_time=as_of_time,
        )
    if notes:
        for note in notes:
            print(f"note: {note}", flush=True)
    ohlcv_path = output_dir / "processed_ohlcv.parquet"
    news_path = output_dir / "processed_news.parquet"
    processed_ohlcv.to_parquet(ohlcv_path, index=False)
    processed_news.to_parquet(news_path, index=False)
    eval_context = _build_eval_context(ohlcv=processed_ohlcv, news=processed_news)
    news_panel = cast(pd.DataFrame, eval_context["news_panel"])
    multi_source_mask = news_panel["source_count"] >= 2.0
    if args.mode == "dummy" and int(multi_source_mask.sum()) <= 0:
        raise RuntimeError("Synthetic sweep dataset did not produce any multi-source rows.")
    _WORKER_CONTEXT.clear()
    _WORKER_CONTEXT.update(eval_context)
    current_preset_eval = _evaluate_config(
        {
            "regularization_lambda": float(settings.data.learned_weighting.regularization_lambda),
            "min_samples": int(settings.data.learned_weighting.min_samples),
            "lookback_days": int(settings.data.learned_weighting.lookback_days),
            "target": str(settings.data.learned_weighting.target),
            "min_weight": min_weight,
            "fallback_mode": fallback_mode,
        }
    )
    stage1_results = _run_stage(
        stage_name="Stage 1",
        configs=_build_stage1_configs(min_weight=min_weight, fallback_mode=fallback_mode),
        ohlcv_path=ohlcv_path,
        news_path=news_path,
        max_workers=max_workers,
    )
    stage2_results = _run_stage(
        stage_name="Stage 2",
        configs=_build_stage2_configs(
            stage1_results,
            min_weight=min_weight,
            fallback_mode=fallback_mode,
            top_k=DEFAULT_TOP_K,
        ),
        ohlcv_path=ohlcv_path,
        news_path=news_path,
        max_workers=max_workers,
    )
    best_result = stage2_results[0]
    recommended_preset = _recommended_preset(best_result, min_weight=min_weight, fallback_mode=fallback_mode)
    dataset_profile_summary.update(_dataset_quality_summary(
        processed_ohlcv=processed_ohlcv,
        processed_news=processed_news,
        news_panel=news_panel,
    ))
    summary = {
        "dataset_profile": dataset_profile_summary,
        "notes": notes,
        "stage1": _stage_summary(stage1_results, top_k=DEFAULT_TOP_K),
        "stage2": _stage_summary(stage2_results, top_k=3),
        "baselines": {
            "overall": {
                "unweighted": cast(dict[str, float | int], eval_context["unweighted_overall"]),
                "fixed": cast(dict[str, float | int], eval_context["fixed_overall"]),
            },
            "multi_source": {
                "unweighted": cast(dict[str, float | int], eval_context["unweighted_multi_source"]),
                "fixed": cast(dict[str, float | int], eval_context["fixed_multi_source"]),
                "recommended_learned": {
                    "abs_ic": _as_float(best_result.get("abs_ic")),
                    "overlap_rate": _as_float(best_result.get("overlap_rate")),
                    "coverage": _as_float(best_result.get("coverage")),
                },
            },
        },
        "current_config_preset": {
            "regularization_lambda": float(settings.data.learned_weighting.regularization_lambda),
            "min_samples": int(settings.data.learned_weighting.min_samples),
            "lookback_days": int(settings.data.learned_weighting.lookback_days),
            "target": str(settings.data.learned_weighting.target),
            "min_weight": min_weight,
            "fallback_mode": fallback_mode,
            "evaluation": current_preset_eval,
        },
        "current_config_matches_recommended": {
            "regularization_lambda": float(settings.data.learned_weighting.regularization_lambda)
            == _as_float(best_result.get("regularization_lambda")),
            "min_samples": int(settings.data.learned_weighting.min_samples) == _as_int(best_result.get("min_samples")),
            "lookback_days": int(settings.data.learned_weighting.lookback_days)
            == _as_int(best_result.get("lookback_days")),
            "target": str(settings.data.learned_weighting.target) == str(best_result.get("target")),
        },
        "data_sources": {
            "ohlcv_transport": _transport_brief(cast(dict[str, object], metadata.get("ohlcv", {}).get("transport", {}))),
            "news_transport": _transport_brief(cast(dict[str, object], metadata.get("news", {}).get("transport", {}))),
        }
        if args.mode == "public"
        else {},
        "recommended_preset": recommended_preset,
    }
    _write_json(output_dir / "stage1_results.json", stage1_results)
    _write_json(output_dir / "stage2_results.json", stage2_results)
    _write_json(output_dir / "recommended_preset.json", recommended_preset)
    _write_json(output_dir / "sweep_summary.json", summary)
    print(
        json.dumps({"output_dir": str(output_dir), "recommended_preset": recommended_preset}, ensure_ascii=False, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
