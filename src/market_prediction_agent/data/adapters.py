from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
import gzip
import html
import hashlib
from io import StringIO
import json
from pathlib import Path
import re
import time
from typing import cast
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from market_prediction_agent.utils.paths import resolve_repo_path
from market_prediction_agent.utils.time_utils import business_dates_between, to_utc_timestamp


@dataclass(slots=True)
class OHLCVRequest:
    tickers: list[str]
    start_date: str
    end_date: str


@dataclass(slots=True)
class MacroRequest:
    series_ids: list[str]
    start_date: str
    end_date: str


@dataclass(slots=True)
class NewsRequest:
    tickers: list[str]
    start_date: str
    end_date: str


@dataclass(slots=True)
class FundamentalsRequest:
    tickers: list[str]
    start_date: str
    end_date: str


@dataclass(slots=True)
class SectorRequest:
    tickers: list[str]


@dataclass(slots=True)
class PublicDataTransportConfig:
    cache_dir: Path
    snapshot_dir: Path
    cache_ttl_hours: int
    retry_count: int
    retry_backoff_seconds: float


class MissingCredentialsError(RuntimeError):
    """Raised when a live adapter is requested without credentials."""


class OHLCVAdapter(ABC):
    name: str

    @abstractmethod
    def fetch(self, request: OHLCVRequest) -> pd.DataFrame:
        raise NotImplementedError


class MacroAdapter(ABC):
    name: str

    @abstractmethod
    def fetch(self, request: MacroRequest) -> pd.DataFrame:
        raise NotImplementedError


class NewsAdapter(ABC):
    name: str

    @abstractmethod
    def fetch(self, request: NewsRequest) -> pd.DataFrame:
        raise NotImplementedError


class FundamentalsAdapter(ABC):
    name: str

    @abstractmethod
    def fetch(self, request: FundamentalsRequest) -> pd.DataFrame:
        raise NotImplementedError


class SectorAdapter(ABC):
    name: str

    @abstractmethod
    def fetch(self, request: SectorRequest) -> pd.DataFrame:
        raise NotImplementedError


def _default_public_data_transport_config() -> PublicDataTransportConfig:
    return PublicDataTransportConfig(
        cache_dir=resolve_repo_path("storage/public_data/cache"),
        snapshot_dir=resolve_repo_path("storage/public_data/snapshots"),
        cache_ttl_hours=24,
        retry_count=2,
        retry_backoff_seconds=1.0,
    )


def _cache_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _read_text_response(url: str, headers: dict[str, str] | None = None) -> str:
    request_headers = {
        "User-Agent": "market-prediction-agent/0.1 (+https://example.invalid/offline-safe)",
    }
    if headers:
        request_headers.update(headers)
    request = urllib.request.Request(
        url,
        headers=request_headers,
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read()
        content_encoding = str(response.headers.get("Content-Encoding", "")).lower()
        if "gzip" in content_encoding:
            payload = gzip.decompress(payload)
        return payload.decode("utf-8")


def _payload_path(root: Path, namespace: str, url: str) -> Path:
    return root / namespace / f"{_cache_key(url)}.json"


def _read_cached_payload(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _write_cached_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _range_signature(namespace: str, url: str) -> tuple[str, pd.Timestamp, pd.Timestamp] | None:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    if namespace == "yahoo_chart":
        period1 = query.get("period1")
        period2 = query.get("period2")
        if not period1 or not period2:
            return None
        resource = parsed.path.rsplit("/", maxsplit=1)[-1]
        start = pd.Timestamp(int(period1[0]), unit="s", tz="UTC").normalize()
        end = (pd.Timestamp(int(period2[0]), unit="s", tz="UTC") - pd.Timedelta(days=1)).normalize()
        return resource, start, end
    if namespace in {"fred_csv", "fred_market_proxy"}:
        series_id = query.get("id")
        start = query.get("cosd")
        end = query.get("coed")
        if not series_id or not start or not end:
            return None
        return (
            series_id[0],
            pd.Timestamp(start[0], tz="UTC").normalize(),
            pd.Timestamp(end[0], tz="UTC").normalize(),
        )
    if namespace == "yahoo_finance_rss":
        ticker = query.get("s")
        start = query.get("start")
        end = query.get("end")
        if not ticker or not start or not end:
            return None
        return (
            ticker[0],
            pd.Timestamp(start[0], tz="UTC").normalize(),
            pd.Timestamp(end[0], tz="UTC").normalize(),
        )
    if namespace == "coingecko_market_chart_range":
        from_timestamp = query.get("from")
        to_timestamp = query.get("to")
        if not from_timestamp or not to_timestamp:
            return None
        path_parts = [part for part in parsed.path.split("/") if part]
        try:
            coin_index = path_parts.index("coins")
            coin_id = path_parts[coin_index + 1]
        except (ValueError, IndexError):
            return None
        start = pd.Timestamp(int(from_timestamp[0]), unit="s", tz="UTC").normalize()
        end = (pd.Timestamp(int(to_timestamp[0]), unit="s", tz="UTC") - pd.Timedelta(days=1)).normalize()
        return coin_id, start, end
    return None


def _find_compatible_payload(
    *,
    root: Path,
    namespace: str,
    url: str,
) -> tuple[dict[str, object], Path] | None:
    requested_signature = _range_signature(namespace, url)
    if requested_signature is None:
        return None
    requested_resource, requested_start, requested_end = requested_signature
    namespace_root = root / namespace
    if not namespace_root.exists():
        return None
    best_match: tuple[tuple[int, float], dict[str, object], Path] | None = None
    for path in namespace_root.rglob("*.json"):
        payload = _read_cached_payload(path)
        if payload is None:
            continue
        candidate_url = payload.get("url")
        if not isinstance(candidate_url, str):
            continue
        candidate_signature = _range_signature(namespace, candidate_url)
        if candidate_signature is None:
            continue
        candidate_resource, candidate_start, candidate_end = candidate_signature
        if candidate_resource != requested_resource:
            continue
        if candidate_start > requested_start or candidate_end < requested_end:
            continue
        coverage_days = int((candidate_end - candidate_start).days)
        fetched_at = payload.get("fetched_at")
        freshness = 0.0
        if isinstance(fetched_at, str):
            freshness = pd.Timestamp(fetched_at).timestamp()
        score = (coverage_days, -freshness)
        if best_match is None or score < best_match[0]:
            best_match = (score, payload, path)
    if best_match is None:
        return None
    return best_match[1], best_match[2]


def _fetch_public_text(
    *,
    url: str,
    namespace: str,
    transport: PublicDataTransportConfig,
    headers: dict[str, str] | None = None,
) -> tuple[str, dict[str, object]]:
    now = to_utc_timestamp(pd.Timestamp.now(tz="UTC"))
    cache_path = _payload_path(transport.cache_dir, namespace, url)
    snapshot_path = _payload_path(transport.snapshot_dir, namespace, url)
    cached_payload = _read_cached_payload(cache_path)
    if cached_payload is not None:
        fetched_at = to_utc_timestamp(cached_payload["fetched_at"])
        age_hours = (now - fetched_at).total_seconds() / 3600.0
        if age_hours <= transport.cache_ttl_hours:
            return cast(str, cached_payload["content"]), {
                "origin": "cache",
                "cache_used": True,
                "snapshot_used": False,
                "stale_cache_used": False,
                "retry_count": 0,
                "cache_path": str(cache_path.as_posix()),
                "snapshot_path": str(snapshot_path.as_posix()),
                "last_error": None,
            }
    compatible_cache = _find_compatible_payload(root=transport.cache_dir, namespace=namespace, url=url)
    if compatible_cache is not None:
        compatible_payload, compatible_path = compatible_cache
        fetched_at = to_utc_timestamp(compatible_payload["fetched_at"])
        age_hours = (now - fetched_at).total_seconds() / 3600.0
        if age_hours <= transport.cache_ttl_hours:
            return cast(str, compatible_payload["content"]), {
                "origin": "compatible_cache",
                "cache_used": True,
                "snapshot_used": False,
                "stale_cache_used": False,
                "retry_count": 0,
                "cache_path": str(compatible_path.as_posix()),
                "snapshot_path": str(snapshot_path.as_posix()),
                "last_error": None,
            }

    last_error: str | None = None
    total_attempts = max(1, transport.retry_count + 1)
    for attempt_index in range(total_attempts):
        try:
            content = _read_text_response(url, headers=headers) if headers else _read_text_response(url)
            payload = {
                "url": url,
                "fetched_at": now.isoformat(),
                "content": content,
            }
            _write_cached_payload(cache_path, payload)
            _write_cached_payload(snapshot_path, payload)
            return content, {
                "origin": "network",
                "cache_used": False,
                "snapshot_used": False,
                "stale_cache_used": False,
                "retry_count": attempt_index,
                "cache_path": str(cache_path.as_posix()),
                "snapshot_path": str(snapshot_path.as_posix()),
                "last_error": None,
            }
        except (OSError, RuntimeError, ValueError) as exc:
            last_error = str(exc)
            if attempt_index < total_attempts - 1 and transport.retry_backoff_seconds > 0:
                time.sleep(transport.retry_backoff_seconds * (2**attempt_index))

    if cached_payload is not None:
        return cast(str, cached_payload["content"]), {
            "origin": "stale_cache",
            "cache_used": True,
            "snapshot_used": False,
            "stale_cache_used": True,
            "retry_count": total_attempts - 1,
            "cache_path": str(cache_path.as_posix()),
            "snapshot_path": str(snapshot_path.as_posix()),
            "last_error": last_error,
        }
    if compatible_cache is not None:
        compatible_payload, compatible_path = compatible_cache
        return cast(str, compatible_payload["content"]), {
            "origin": "compatible_stale_cache",
            "cache_used": True,
            "snapshot_used": False,
            "stale_cache_used": True,
            "retry_count": total_attempts - 1,
            "cache_path": str(compatible_path.as_posix()),
            "snapshot_path": str(snapshot_path.as_posix()),
            "last_error": last_error,
        }
    snapshot_payload = _read_cached_payload(snapshot_path)
    if snapshot_payload is not None:
        return cast(str, snapshot_payload["content"]), {
            "origin": "snapshot",
            "cache_used": False,
            "snapshot_used": True,
            "stale_cache_used": False,
            "retry_count": total_attempts - 1,
            "cache_path": str(cache_path.as_posix()),
            "snapshot_path": str(snapshot_path.as_posix()),
            "last_error": last_error,
        }
    compatible_snapshot = _find_compatible_payload(root=transport.snapshot_dir, namespace=namespace, url=url)
    if compatible_snapshot is not None:
        compatible_payload, compatible_path = compatible_snapshot
        return cast(str, compatible_payload["content"]), {
            "origin": "compatible_snapshot",
            "cache_used": False,
            "snapshot_used": True,
            "stale_cache_used": False,
            "retry_count": total_attempts - 1,
            "cache_path": str(cache_path.as_posix()),
            "snapshot_path": str(compatible_path.as_posix()),
            "last_error": last_error,
        }
    raise RuntimeError(f"Public data fetch failed with no cache or snapshot fallback. last_error={last_error}")


def _summarize_public_fetches(requests: list[dict[str, object]]) -> dict[str, object]:
    return {
        "network_used": any(item["origin"] == "network" for item in requests),
        "cache_used": any(bool(item["cache_used"]) for item in requests),
        "snapshot_used": any(bool(item["snapshot_used"]) for item in requests),
        "stale_cache_used": any(bool(item["stale_cache_used"]) for item in requests),
        "origins": sorted({cast(str, item["origin"]) for item in requests}),
        "requests": requests,
    }


class DummyOHLCVAdapter(OHLCVAdapter):
    name = "dummy"

    def __init__(self, seed: int = 42, mode: str = "null_random_walk") -> None:
        self.seed = seed
        self.mode = mode

    def fetch(self, request: OHLCVRequest) -> pd.DataFrame:
        if self.mode == "predictable_momentum":
            return self._fetch_predictable_momentum(request)
        if self.mode == "null_random_walk":
            return self._fetch_null_random_walk(request)
        raise ValueError(f"Unsupported dummy mode: {self.mode}")

    def _fetch_null_random_walk(self, request: OHLCVRequest) -> pd.DataFrame:
        dates = business_dates_between(request.start_date, request.end_date)
        rng = np.random.default_rng(self.seed)
        rows: list[dict[str, object]] = []
        for ticker in request.tickers:
            returns = rng.normal(0.0, 0.012, len(dates))
            close = 100 * np.exp(np.cumsum(returns))
            open_px = close * (1 + rng.normal(0, 0.002, len(dates)))
            high = np.maximum(open_px, close) * (1 + rng.uniform(0.0005, 0.01, len(dates)))
            low = np.minimum(open_px, close) * (1 - rng.uniform(0.0005, 0.01, len(dates)))
            volume = np.maximum(1_000_000, 2_000_000 * (1 + np.abs(rng.normal(0, 0.15, len(dates)))))
            for date_value, open_value, high_value, low_value, close_value, volume_value in zip(
                dates, open_px, high, low, close, volume, strict=True
            ):
                rows.append(
                    {
                        "ticker": ticker,
                        "timestamp_utc": date_value,
                        "open": float(open_value),
                        "high": float(high_value),
                        "low": float(low_value),
                        "close": float(close_value),
                        "volume": float(volume_value),
                        "source": self.name,
                        "fetched_at": to_utc_timestamp(pd.Timestamp.now(tz="UTC")),
                        "stale_data_flag": False,
                    }
                )
        return pd.DataFrame(rows)

    def _fetch_predictable_momentum(self, request: OHLCVRequest) -> pd.DataFrame:
        dates = business_dates_between(request.start_date, request.end_date)
        rng = np.random.default_rng(self.seed)
        market = rng.normal(0.0002, 0.01, len(dates))
        regime = np.where(np.arange(len(dates)) % 200 < 120, 0.8, 1.4)
        rows: list[dict[str, object]] = []
        for index, ticker in enumerate(request.tickers):
            idiosyncratic = rng.normal(0, 0.012 + index * 0.00001, len(dates))
            momentum = np.zeros(len(dates))
            for day in range(1, len(dates)):
                momentum[day] = 0.18 * momentum[day - 1] + idiosyncratic[day]
            returns = market * (0.6 + (index % 10) * 0.05) + momentum * regime
            close = 100 * np.exp(np.cumsum(returns))
            open_px = close * (1 + rng.normal(0, 0.002, len(dates)))
            high = np.maximum(open_px, close) * (1 + rng.uniform(0.0005, 0.01, len(dates)))
            low = np.minimum(open_px, close) * (1 - rng.uniform(0.0005, 0.01, len(dates)))
            volume = np.maximum(1_000_000, 2_000_000 * (1 + np.abs(returns) * 40 + rng.normal(0, 0.1, len(dates))))
            for date_value, open_value, high_value, low_value, close_value, volume_value in zip(
                dates, open_px, high, low, close, volume, strict=True
            ):
                rows.append(
                    {
                        "ticker": ticker,
                        "timestamp_utc": date_value,
                        "open": float(open_value),
                        "high": float(high_value),
                        "low": float(low_value),
                        "close": float(close_value),
                        "volume": float(volume_value),
                        "source": self.name,
                        "fetched_at": to_utc_timestamp(pd.Timestamp.now(tz="UTC")),
                        "stale_data_flag": False,
                    }
                )
        return pd.DataFrame(rows)


class CoinGeckoAdapter(OHLCVAdapter):
    name = "coingecko"
    TICKER_TO_COIN_ID = {
        "BTC-USD": "bitcoin",
        "ETH-USD": "ethereum",
    }

    def __init__(
        self,
        *,
        transport: PublicDataTransportConfig | None = None,
        source_mode: str = "live",
        seed: int = 42,
        dummy_mode: str = "null_random_walk",
    ) -> None:
        self.transport = transport or _default_public_data_transport_config()
        self.source_mode = source_mode
        self.seed = seed
        self.dummy_mode = dummy_mode
        self.last_fetch_metadata: dict[str, object] = {}

    def fetch(self, request: OHLCVRequest) -> pd.DataFrame:
        if self.source_mode == "dummy":
            self.last_fetch_metadata = {
                "network_used": False,
                "cache_used": False,
                "snapshot_used": False,
                "stale_cache_used": False,
                "origins": ["dummy"],
                "requests": [
                    {
                        "origin": "dummy",
                        "cache_used": False,
                        "snapshot_used": False,
                        "stale_cache_used": False,
                        "source_name": self.name,
                    }
                ],
            }
            return DummyOHLCVAdapter(seed=self.seed, mode=self.dummy_mode).fetch(request)

        start = pd.Timestamp(request.start_date, tz="UTC")
        end = pd.Timestamp(request.end_date, tz="UTC")
        fetched_at = to_utc_timestamp(pd.Timestamp.now(tz="UTC"))
        frames: list[pd.DataFrame] = []
        fetch_requests: list[dict[str, object]] = []
        for ticker in request.tickers:
            coin_id = self.TICKER_TO_COIN_ID.get(ticker.upper())
            if coin_id is None:
                raise ValueError(f"CoinGecko adapter does not support ticker {ticker}.")
            frame, metadata = self._fetch_ticker_history(
                ticker=ticker,
                coin_id=coin_id,
                start=start,
                end=end,
                fetched_at=fetched_at,
            )
            frames.append(frame)
            fetch_requests.append({**metadata, "ticker": ticker, "coin_id": coin_id})
        if not frames:
            raise RuntimeError("CoinGecko returned no crypto history.")
        self.last_fetch_metadata = _summarize_public_fetches(fetch_requests)
        return pd.concat(frames, ignore_index=True)

    def _fetch_ticker_history(
        self,
        *,
        ticker: str,
        coin_id: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        fetched_at: pd.Timestamp,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        query = urllib.parse.urlencode(
            {
                "vs_currency": "usd",
                "from": int(start.timestamp()),
                "to": int((end + pd.Timedelta(days=1)).timestamp()),
            }
        )
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart/range?{query}"
        payload_text, metadata = _fetch_public_text(
            url=url,
            namespace="coingecko_market_chart_range",
            transport=self.transport,
        )
        payload = cast(dict[str, object], json.loads(payload_text))
        prices = cast(list[list[float | int]], payload.get("prices", []))
        total_volumes = cast(list[list[float | int]], payload.get("total_volumes", []))
        if not prices:
            raise RuntimeError(f"CoinGecko returned no prices for ticker {ticker}.")
        price_frame = pd.DataFrame(prices, columns=["timestamp_ms", "price"])
        volume_frame = (
            pd.DataFrame(total_volumes, columns=["timestamp_ms", "volume"])
            if total_volumes
            else pd.DataFrame(columns=["timestamp_ms", "volume"])
        )
        frame = price_frame.merge(volume_frame, on="timestamp_ms", how="left")
        frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True, errors="coerce")
        frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
        frame["volume"] = pd.to_numeric(frame["volume"], errors="coerce")
        frame = frame.dropna(subset=["timestamp_utc", "price"]).sort_values("timestamp_utc")
        frame = frame.loc[
            (frame["timestamp_utc"] >= start)
            & (frame["timestamp_utc"] < (end + pd.Timedelta(days=1)))
        ].copy()
        if frame.empty:
            raise RuntimeError(f"CoinGecko returned no rows for ticker {ticker} in the requested window.")
        frame["trade_date"] = frame["timestamp_utc"].dt.normalize()
        rows: list[dict[str, object]] = []
        for trade_date, group in frame.groupby("trade_date", sort=True):
            volumes = pd.to_numeric(group["volume"], errors="coerce").ffill().dropna()
            volume_value = float(volumes.iloc[-1]) if not volumes.empty else 0.0
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp_utc": cast(pd.Timestamp, trade_date),
                    "open": float(group["price"].iloc[0]),
                    "high": float(group["price"].max()),
                    "low": float(group["price"].min()),
                    "close": float(group["price"].iloc[-1]),
                    "volume": volume_value,
                    "source": self.name,
                    "fetched_at": fetched_at,
                    "stale_data_flag": False,
                }
            )
        if not rows:
            raise RuntimeError(f"CoinGecko returned no daily bars for ticker {ticker}.")
        return pd.DataFrame(rows), metadata


class DummyMacroAdapter(MacroAdapter):
    name = "dummy"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def fetch(self, request: MacroRequest) -> pd.DataFrame:
        dates = business_dates_between(request.start_date, request.end_date)
        rng = np.random.default_rng(self.seed + 7)
        rows: list[dict[str, object]] = []
        base_series = {
            "FEDFUNDS": 4.5,
            "T10Y2Y": 0.5,
            "VIXCLS": 18.0,
            "CPIAUCSL": 310.0,
            "UNRATE": 4.1,
        }
        for series_id in request.series_ids:
            value = base_series.get(series_id, 1.0)
            for index, date_value in enumerate(dates):
                drift = 0.001 * index
                noise = rng.normal(0, 0.05 if series_id == "VIXCLS" else 0.01)
                level = value + drift + noise
                if series_id == "VIXCLS":
                    level = max(10.0, 18 + 6 * np.sin(index / 45) + rng.normal(0, 1.5))
                rows.append(
                    {
                        "series_id": series_id,
                        "date": date_value,
                        "value": float(level),
                        "available_at": date_value + timedelta(days=1),
                        "source": self.name,
                    }
                )
        return pd.DataFrame(rows)


SECTOR_BUCKETS = [
    "technology",
    "financials",
    "healthcare",
    "industrials",
    "consumer",
    "energy",
    "materials",
    "utilities",
]

KNOWN_SECTOR_OVERRIDES = {
    "SPY": "broad_market",
    "QQQ": "technology_growth",
    "DIA": "industrial_value",
    "GLD": "precious_metals",
}


def _stable_integer(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def _sector_for_ticker(ticker: str) -> str:
    normalized = ticker.upper()
    if normalized in KNOWN_SECTOR_OVERRIDES:
        return KNOWN_SECTOR_OVERRIDES[normalized]
    return SECTOR_BUCKETS[_stable_integer(f"sector:{normalized}") % len(SECTOR_BUCKETS)]


NEWS_POSITIVE_TERMS = {
    "beat",
    "beats",
    "bullish",
    "buyback",
    "expands",
    "gain",
    "gains",
    "growth",
    "improves",
    "improved",
    "outperform",
    "profit",
    "profits",
    "raises",
    "rally",
    "record",
    "strong",
    "surge",
    "surges",
    "upgrade",
    "upgrades",
}
NEWS_NEGATIVE_TERMS = {
    "bearish",
    "cut",
    "cuts",
    "decline",
    "declines",
    "downgrade",
    "downgrades",
    "drop",
    "drops",
    "fall",
    "falls",
    "lawsuit",
    "loss",
    "losses",
    "miss",
    "misses",
    "risk",
    "slump",
    "underperform",
    "warning",
    "warns",
    "weak",
}
NEWS_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z'/-]*")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
NEWS_ALIAS_NORMALIZE_PATTERN = re.compile(r"[^A-Za-z0-9]+")
SEC_HEADERS = {
    "User-Agent": "market-prediction-agent/0.1 research-support contact: openai@example.invalid",
    "Accept-Encoding": "gzip, deflate",
}
COMPANY_SUFFIX_TOKENS = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "holdings",
    "holding",
    "group",
    "plc",
    "ltd",
    "limited",
    "trust",
    "fund",
    "funds",
    "etf",
    "shares",
    "class",
    "common",
}
STATIC_NEWS_ALIASES = {
    "AAPL": ["apple", "apple inc"],
    "MSFT": ["microsoft", "microsoft corp", "microsoft corporation"],
    "NVDA": ["nvidia", "nvidia corp", "nvidia corporation"],
    "AMZN": ["amazon", "amazon com", "amazon.com", "amazon com inc"],
    "GOOG": ["google", "alphabet", "alphabet inc"],
    "GOOGL": ["google", "alphabet", "alphabet inc"],
    "META": ["meta", "meta platforms", "facebook", "facebook parent"],
    "TSLA": ["tesla", "tesla inc"],
    "SPY": ["spdr s&p 500", "spdr sp 500", "s&p 500 etf", "sp500 etf"],
    "QQQ": ["invesco qqq", "nasdaq 100 etf", "nasdaq-100 etf", "qqq etf"],
    "DIA": ["spdr dow", "dow etf", "dow jones industrial average etf"],
    "GLD": ["spdr gold", "gold etf", "spdr gold shares"],
}
US_MARKET_TZ = ZoneInfo("America/New_York")
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16


def _strip_html(text: str) -> str:
    return html.unescape(HTML_TAG_PATTERN.sub(" ", text or "")).strip()


def _xml_child_text(element: ET.Element, local_name: str) -> str:
    for child in element:
        tag_name = child.tag.rsplit("}", maxsplit=1)[-1]
        if tag_name == local_name:
            return (child.text or "").strip()
    return ""


def _headline_sentiment_score(text: str) -> float:
    tokens = [token.lower() for token in NEWS_TOKEN_PATTERN.findall(text)]
    positive_hits = sum(token in NEWS_POSITIVE_TERMS for token in tokens)
    negative_hits = sum(token in NEWS_NEGATIVE_TERMS for token in tokens)
    total_hits = positive_hits + negative_hits
    if total_hits == 0:
        return 0.0
    return float(np.clip((positive_hits - negative_hits) / total_hits, -1.0, 1.0))


def _headline_relevance_score(text: str, ticker: str) -> float:
    normalized_text = text.upper()
    ticker_hits = normalized_text.count(ticker.upper())
    token_count = max(len(NEWS_TOKEN_PATTERN.findall(text)), 1)
    density = min(ticker_hits / token_count, 0.25)
    return float(np.clip(0.55 + 0.20 * min(ticker_hits, 2) + 0.40 * density, 0.0, 1.0))


def _normalize_alias_text(text: str) -> str:
    return NEWS_ALIAS_NORMALIZE_PATTERN.sub(" ", text.lower()).strip()


def _headline_key(text: str) -> str:
    return " ".join(_normalize_alias_text(text).split())


def _canonical_company_aliases(ticker: str) -> set[str]:
    aliases = {ticker.lower()}
    aliases.update(alias.lower() for alias in STATIC_NEWS_ALIASES.get(ticker.upper(), []))
    return {alias for alias in aliases if alias}


def _company_token_aliases(alias: str) -> set[str]:
    tokens = [token for token in _normalize_alias_text(alias).split() if token and token not in COMPANY_SUFFIX_TOKENS]
    aliases: set[str] = set()
    if len(tokens) >= 2:
        aliases.add(" ".join(tokens))
    aliases.update(token for token in tokens if len(token) >= 4)
    return aliases


def _news_alias_lookup(tickers: list[str]) -> dict[str, set[str]]:
    lookup: dict[str, set[str]] = {}
    for ticker in tickers:
        normalized_ticker = ticker.upper()
        aliases = _canonical_company_aliases(normalized_ticker)
        expanded: set[str] = set()
        for alias in aliases:
            expanded.add(alias)
            expanded.update(_company_token_aliases(alias))
        lookup[normalized_ticker] = {alias for alias in expanded if alias}
    return lookup


def _headline_mapping_score(
    *,
    text: str,
    ticker: str,
    alias_lookup: dict[str, set[str]],
) -> tuple[str | None, float]:
    normalized_text = f" {_normalize_alias_text(text)} "
    symbol_pattern = re.compile(rf"\b{re.escape(ticker.lower())}\b")
    symbol_hits = len(symbol_pattern.findall(normalized_text))
    if symbol_hits > 0:
        confidence = float(np.clip(0.80 + 0.10 * min(symbol_hits, 2), 0.0, 1.0))
        return "symbol", confidence
    alias_hits = sum(f" {alias} " in normalized_text for alias in alias_lookup.get(ticker.upper(), set()))
    if alias_hits > 0:
        confidence = float(np.clip(0.62 + 0.12 * min(alias_hits, 2), 0.0, 0.95))
        return "alias", confidence
    return None, 0.0


def _source_domain_from_link(link: str) -> str:
    if not link:
        return "finance.yahoo.com"
    parsed = urllib.parse.urlparse(link)
    return parsed.netloc or "finance.yahoo.com"


def _source_label_from_item(item: ET.Element, link: str, default: str) -> str:
    source_text = _xml_child_text(item, "source")
    if source_text:
        return source_text
    return _source_domain_from_link(link) or default


def _local_business_day(value: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(value.date()).tz_localize(US_MARKET_TZ)


def _next_local_business_day(value: pd.Timestamp) -> pd.Timestamp:
    next_business = pd.Timestamp(value.date()) + pd.tseries.offsets.BDay(1)
    return pd.Timestamp(next_business.date()).tz_localize(US_MARKET_TZ)


def _session_aligned_available_at(published_at: pd.Timestamp) -> tuple[pd.Timestamp, str]:
    local_timestamp = to_utc_timestamp(published_at).tz_convert(US_MARKET_TZ)
    local_business_day = _local_business_day(local_timestamp)
    market_open = local_business_day + pd.Timedelta(hours=MARKET_OPEN_HOUR, minutes=MARKET_OPEN_MINUTE)
    market_close = local_business_day + pd.Timedelta(hours=MARKET_CLOSE_HOUR)
    if local_timestamp.weekday() >= 5:
        signal_day = _next_local_business_day(local_timestamp)
        session_bucket = "weekend_shifted"
    elif local_timestamp < market_open:
        signal_day = local_business_day
        session_bucket = "pre_market"
    elif local_timestamp < market_close:
        signal_day = _next_local_business_day(local_timestamp)
        session_bucket = "regular"
    else:
        signal_day = _next_local_business_day(local_timestamp)
        session_bucket = "post_market"
    return signal_day.tz_convert("UTC").normalize(), session_bucket


def _session_bucket_label(values: pd.Series) -> str:
    non_empty = [str(value) for value in values if str(value)]
    if not non_empty:
        return "none"
    unique = sorted(set(non_empty))
    if len(unique) == 1:
        return unique[0]
    return "mixed"


def _session_bucket_counts(values: pd.Series) -> dict[str, int]:
    counts = pd.Series([str(value) for value in values if str(value)], dtype=str).value_counts()
    return {str(index): int(value) for index, value in counts.items()}


def _source_mix_label(values: pd.Series) -> str:
    normalized = sorted({str(value) for value in values if str(value)})
    if not normalized:
        return "none"
    return "|".join(normalized)


def _weight_lookup(weights: dict[str, float], key: str, default: float = 1.0) -> float:
    value = weights.get(key, default)
    return float(value if value > 0.0 else default)


def _weighted_average(
    frame: pd.DataFrame,
    *,
    value_column: str,
    weight_column: str,
) -> float:
    values = pd.to_numeric(frame[value_column], errors="coerce")
    weights = pd.to_numeric(frame[weight_column], errors="coerce").fillna(0.0)
    if values.notna().sum() == 0:
        return 0.0
    effective = weights.where(values.notna(), 0.0)
    total_weight = float(effective.sum())
    if total_weight <= 0.0:
        return float(values.fillna(0.0).mean())
    return float((values.fillna(0.0) * effective).sum() / total_weight)


def _query_aliases_for_ticker(ticker: str) -> list[str]:
    aliases = list(_canonical_company_aliases(ticker))
    phrase_aliases = [alias for alias in aliases if " " in alias]
    preferred = sorted(phrase_aliases or aliases, key=lambda item: (-len(item), item))
    deduped: list[str] = []
    for alias in [ticker.lower(), *preferred]:
        normalized = alias.strip()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped[:3]


def _google_news_query_variants(ticker: str, lookback_days: int) -> list[tuple[str, str]]:
    aliases = _query_aliases_for_ticker(ticker)
    variants: list[tuple[str, str]] = []
    quoted_terms = " OR ".join(f"\"{alias}\"" for alias in aliases)
    if quoted_terms:
        variants.append(("aliases_or", f"{quoted_terms} when:{lookback_days}d"))
    if aliases:
        variants.append(("primary_alias", f"\"{aliases[0]}\" stock when:{lookback_days}d"))
    variants.append(("ticker_stock", f"\"{ticker}\" stock when:{lookback_days}d"))
    deduped: list[tuple[str, str]] = []
    seen_queries: set[str] = set()
    for label, query in variants:
        if query not in seen_queries:
            deduped.append((label, query))
            seen_queries.add(query)
    return deduped


def _empty_daily_news_frame(
    *,
    ticker: str,
    dates: pd.DatetimeIndex,
    fetched_at: pd.Timestamp,
    source: str,
) -> pd.DataFrame:
    published_at = dates + pd.Timedelta(hours=16)
    available_at = dates + pd.Timedelta(days=1)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "news_date": dates,
            "published_at": published_at,
            "available_at": available_at,
            "sentiment_score": np.zeros(len(dates), dtype=float),
            "sentiment_score_unweighted": np.zeros(len(dates), dtype=float),
            "sentiment_score_source_weighted": np.zeros(len(dates), dtype=float),
            "sentiment_score_session_weighted": np.zeros(len(dates), dtype=float),
            "sentiment_score_source_session_weighted": np.zeros(len(dates), dtype=float),
            "relevance_score": np.zeros(len(dates), dtype=float),
            "relevance_score_unweighted": np.zeros(len(dates), dtype=float),
            "relevance_score_source_weighted": np.zeros(len(dates), dtype=float),
            "relevance_score_session_weighted": np.zeros(len(dates), dtype=float),
            "relevance_score_source_session_weighted": np.zeros(len(dates), dtype=float),
            "headline_count": np.zeros(len(dates), dtype=int),
            "mapping_confidence": np.zeros(len(dates), dtype=float),
            "novelty_score": np.zeros(len(dates), dtype=float),
            "source_diversity": np.zeros(len(dates), dtype=float),
            "source_count": np.zeros(len(dates), dtype=float),
            "source_mix": ["none"] * len(dates),
            "session_bucket": ["none"] * len(dates),
            "source_session_breakdown": ["{}"] * len(dates),
            "source": source,
            "fetched_at": fetched_at,
            "stale_data_flag": False,
        }
    )


def _empty_fundamentals_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ticker",
            "report_date",
            "available_at",
            "revenue_growth",
            "earnings_yield",
            "debt_to_equity",
            "profitability",
            "source",
            "fetched_at",
            "stale_data_flag",
        ]
    )


def _select_sec_metric_records(
    facts: dict[str, object],
    *,
    concept_names: list[str],
    unit_prefixes: tuple[str, ...],
    min_duration_days: int | None = None,
    max_duration_days: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for priority, concept_name in enumerate(concept_names):
        concept = cast(dict[str, object], facts.get(concept_name, {}))
        units = cast(dict[str, object], concept.get("units", {}))
        for unit_name, values in units.items():
            if unit_prefixes and not any(str(unit_name).startswith(prefix) for prefix in unit_prefixes):
                continue
            for item in cast(list[object], values):
                if not isinstance(item, dict):
                    continue
                report_date = pd.to_datetime(item.get("end"), utc=True, errors="coerce")
                filed_at = pd.to_datetime(item.get("filed"), utc=True, errors="coerce")
                start = pd.to_datetime(item.get("start"), utc=True, errors="coerce")
                numeric_value = pd.to_numeric(item.get("val"), errors="coerce")
                if pd.isna(report_date) or pd.isna(filed_at) or pd.isna(numeric_value):
                    continue
                duration_days: int | None = None
                if not pd.isna(start):
                    duration_days = int((cast(pd.Timestamp, report_date) - cast(pd.Timestamp, start)).days)
                if min_duration_days is not None and (duration_days is None or duration_days < min_duration_days):
                    continue
                if max_duration_days is not None and (duration_days is None or duration_days > max_duration_days):
                    continue
                rows.append(
                    {
                        "report_date": cast(pd.Timestamp, report_date).normalize(),
                        "filed_at": cast(pd.Timestamp, filed_at),
                        "value": float(numeric_value),
                        "concept_name": concept_name,
                        "concept_priority": priority,
                        "duration_days": duration_days,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=["report_date", "filed_at", "value", "concept_name", "concept_priority"])
    frame = pd.DataFrame(rows).sort_values(
        ["report_date", "concept_priority", "filed_at"],
        ascending=[True, True, False],
    )
    return frame.drop_duplicates(["report_date"], keep="first").reset_index(drop=True)


def _nearest_close_on_or_before(frame: pd.DataFrame, timestamp: pd.Timestamp) -> float | None:
    if frame.empty:
        return None
    eligible = frame.loc[frame["timestamp_utc"] <= timestamp, "close"]
    if eligible.empty:
        return None
    return float(eligible.iloc[-1])


class OfflineNewsProxyAdapter(NewsAdapter):
    name = "offline_news_proxy"

    def __init__(self, seed: int = 42, mode: str = "live_proxy") -> None:
        self.seed = seed
        self.mode = mode

    def fetch(self, request: NewsRequest) -> pd.DataFrame:
        dates = business_dates_between(request.start_date, request.end_date)
        rows: list[dict[str, object]] = []
        fetched_at = to_utc_timestamp(pd.Timestamp.now(tz="UTC"))
        mode = self.mode
        for ticker_index, ticker in enumerate(request.tickers):
            rng = np.random.default_rng(self.seed + _stable_integer(f"news:{ticker}") % 10_000)
            latent = 0.0
            for index, date_value in enumerate(dates):
                market_wave = np.sin((index + ticker_index) / 18.0)
                shock_scale = 0.28 if mode == "null_random_walk" else 0.16
                persistence = 0.10 if mode == "null_random_walk" else 0.35
                signal_strength = 0.03 if mode == "null_random_walk" else 0.18 if mode == "predictable_momentum" else 0.09
                latent = persistence * latent + signal_strength * market_wave + rng.normal(0.0, shock_scale)
                sentiment = float(np.clip(latent, -1.0, 1.0))
                relevance = float(np.clip(0.40 + 0.20 * abs(sentiment) + rng.normal(0.0, 0.05), 0.0, 1.0))
                headline_count = int(max(rng.poisson(1.5 + 3.5 * relevance), 0))
                published_at = pd.Timestamp(date_value).tz_convert("UTC") + timedelta(hours=16)
                available_at = pd.Timestamp(date_value).tz_convert("UTC") + timedelta(days=1)
                rows.append(
                    {
                        "ticker": ticker,
                        "published_at": published_at,
                        "available_at": available_at,
                        "sentiment_score": sentiment,
                        "relevance_score": relevance,
                        "headline_count": headline_count,
                        "mapping_confidence": 1.0 if headline_count > 0 else 0.0,
                        "novelty_score": float(np.clip(0.75 - 0.05 * max(headline_count - 1, 0), 0.1, 1.0)),
                        "source_diversity": float(min(max(headline_count, 0), 3)),
                        "source_count": 1.0 if headline_count > 0 else 0.0,
                        "source_mix": self.name if headline_count > 0 else "none",
                        "session_bucket": "post_market" if headline_count > 0 else "none",
                        "source": self.name,
                        "fetched_at": fetched_at,
                        "stale_data_flag": False,
                    }
                )
        return pd.DataFrame(rows)


def _matched_tickers_for_headline(
    *,
    body: str,
    feed_ticker: str,
    alias_lookup: dict[str, set[str]],
) -> list[tuple[str, str, float]]:
    matches: list[tuple[str, str, float]] = []
    for ticker in alias_lookup:
        method, confidence = _headline_mapping_score(text=body, ticker=ticker, alias_lookup=alias_lookup)
        if method is not None:
            matches.append((ticker, method, confidence))
    if matches:
        return matches
    return [(feed_ticker.upper(), "feed_scope", 0.55)]


def _compute_novelty(mapped: pd.DataFrame) -> pd.Series:
    novelty_scores: list[float] = []
    history_by_ticker: dict[str, dict[str, int]] = {}
    for _, row in mapped.iterrows():
        ticker = str(row["ticker"])
        headline_key = str(row["headline_key"])
        ticker_history = history_by_ticker.setdefault(ticker, {})
        prior_count = ticker_history.get(headline_key, 0)
        novelty_scores.append(float(1.0 / (1.0 + prior_count)))
        ticker_history[headline_key] = prior_count + 1
    return pd.Series(novelty_scores, index=mapped.index, dtype=float)


def _aggregate_news_items_to_daily_frame(
    *,
    tickers: list[str],
    raw_items: pd.DataFrame,
    alias_lookup: dict[str, set[str]],
    business_dates: pd.DatetimeIndex,
    fetched_at: pd.Timestamp,
    output_source: str,
    source_weights: dict[str, float] | None = None,
    session_weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    resolved_source_weights = source_weights or {}
    resolved_session_weights = session_weights or {}
    if raw_items.empty:
        return pd.concat(
            [
                _empty_daily_news_frame(
                    ticker=ticker,
                    dates=business_dates,
                    fetched_at=fetched_at,
                    source=output_source,
                ).drop(columns=["news_date"])
                for ticker in tickers
            ],
            ignore_index=True,
        )
    deduplicated = (
        raw_items.sort_values(["published_at", "base_relevance_score"], ascending=[True, False])
        .drop_duplicates(["headline_key", "published_at", "source_key"], keep="first")
        .reset_index(drop=True)
    )
    mapped_rows: list[dict[str, object]] = []
    for _, row in deduplicated.iterrows():
        body = str(row["body"])
        matched = _matched_tickers_for_headline(
            body=body,
            feed_ticker=str(row["feed_ticker"]),
            alias_lookup=alias_lookup,
        )
        aligned_available_at, session_bucket = _session_aligned_available_at(cast(pd.Timestamp, row["published_at"]))
        for matched_ticker, mapping_method, mapping_confidence in matched:
            mapped_rows.append(
                {
                    "ticker": matched_ticker,
                    "published_at": row["published_at"],
                    "available_at": aligned_available_at,
                    "session_bucket": session_bucket,
                    "sentiment_score": float(row["sentiment_score"]),
                    "relevance_score": float(
                        np.clip(
                            0.50 * float(row["base_relevance_score"])
                            + 0.50 * float(mapping_confidence),
                            0.0,
                            1.0,
                        )
                    ),
                    "headline_count": 1,
                    "mapping_confidence": float(mapping_confidence),
                    "mapping_method": mapping_method,
                    "source_key": str(row["source_key"]),
                    "source_label": str(row["source_label"]),
                    "source_name": str(row["source_name"]),
                    "headline_key": str(row["headline_key"]),
                    "source": output_source,
                    "fetched_at": fetched_at,
                    "stale_data_flag": False,
                }
            )
    if not mapped_rows:
        return pd.concat(
            [
                _empty_daily_news_frame(
                    ticker=ticker,
                    dates=business_dates,
                    fetched_at=fetched_at,
                    source=output_source,
                ).drop(columns=["news_date"])
                for ticker in tickers
            ],
            ignore_index=True,
        )
    mapped = pd.DataFrame(mapped_rows).sort_values(["ticker", "published_at", "headline_key"]).reset_index(drop=True)
    mapped["novelty_score"] = _compute_novelty(mapped)
    mapped["source_weight"] = mapped["source_name"].map(
        lambda value: _weight_lookup(resolved_source_weights, str(value), 1.0)
    )
    mapped["session_weight"] = mapped["session_bucket"].map(
        lambda value: _weight_lookup(resolved_session_weights, str(value), 1.0)
    )
    mapped["source_session_weight"] = mapped["source_weight"] * mapped["session_weight"]
    mapped["signal_weight_unweighted"] = mapped["relevance_score"].clip(lower=0.0, upper=1.0)
    mapped["signal_weight_source"] = mapped["signal_weight_unweighted"] * mapped["source_weight"]
    mapped["signal_weight_session"] = mapped["signal_weight_unweighted"] * mapped["session_weight"]
    mapped["signal_weight_source_session"] = mapped["signal_weight_unweighted"] * mapped["source_session_weight"]
    mapped["news_date"] = pd.to_datetime(mapped["available_at"], utc=True).dt.normalize()
    daily_rows: list[dict[str, object]] = []
    for (ticker, news_date), group in mapped.groupby(["ticker", "news_date"], as_index=False):
        source_session_breakdown: dict[str, dict[str, float | int | str]] = {}
        for (source_name, session_bucket), combo_group in group.groupby(["source_name", "session_bucket"], as_index=False):
            combo_weights = pd.to_numeric(combo_group["signal_weight_unweighted"], errors="coerce").fillna(0.0).clip(lower=0.0)
            sentiment_values = pd.to_numeric(combo_group["sentiment_score"], errors="coerce").fillna(0.0)
            relevance_values = pd.to_numeric(combo_group["relevance_score"], errors="coerce").fillna(0.0)
            combo_key = f"{source_name}::{session_bucket}"
            source_session_breakdown[combo_key] = {
                "source_name": str(source_name),
                "session_bucket": str(session_bucket),
                "base_weight_sum": float(combo_weights.sum()),
                "sentiment_weighted_sum": float((sentiment_values * combo_weights).sum()),
                "relevance_weighted_sum": float((relevance_values * combo_weights).sum()),
                "headline_count": int(pd.to_numeric(combo_group["headline_count"], errors="coerce").fillna(0).sum()),
                "article_count": int(len(combo_group)),
            }
        daily_rows.append(
            {
                "ticker": ticker,
                "news_date": news_date,
                "published_at": group["published_at"].max(),
                "available_at": group["available_at"].max(),
                "sentiment_score_unweighted": _weighted_average(
                    group,
                    value_column="sentiment_score",
                    weight_column="signal_weight_unweighted",
                ),
                "sentiment_score_source_weighted": _weighted_average(
                    group,
                    value_column="sentiment_score",
                    weight_column="signal_weight_source",
                ),
                "sentiment_score_session_weighted": _weighted_average(
                    group,
                    value_column="sentiment_score",
                    weight_column="signal_weight_session",
                ),
                "sentiment_score_source_session_weighted": _weighted_average(
                    group,
                    value_column="sentiment_score",
                    weight_column="signal_weight_source_session",
                ),
                "relevance_score_unweighted": _weighted_average(
                    group,
                    value_column="relevance_score",
                    weight_column="signal_weight_unweighted",
                ),
                "relevance_score_source_weighted": _weighted_average(
                    group,
                    value_column="relevance_score",
                    weight_column="signal_weight_source",
                ),
                "relevance_score_session_weighted": _weighted_average(
                    group,
                    value_column="relevance_score",
                    weight_column="signal_weight_session",
                ),
                "relevance_score_source_session_weighted": _weighted_average(
                    group,
                    value_column="relevance_score",
                    weight_column="signal_weight_source_session",
                ),
                "sentiment_score": _weighted_average(
                    group,
                    value_column="sentiment_score",
                    weight_column="signal_weight_source_session",
                ),
                "relevance_score": _weighted_average(
                    group,
                    value_column="relevance_score",
                    weight_column="signal_weight_source_session",
                ),
                "headline_count": int(group["headline_count"].sum()),
                "mapping_confidence": float(pd.to_numeric(group["mapping_confidence"], errors="coerce").mean()),
                "novelty_score": float(pd.to_numeric(group["novelty_score"], errors="coerce").mean()),
                "source_diversity": int(group["source_key"].nunique()),
                "source_count": int(group["source_name"].nunique()),
                "source_mix": _source_mix_label(group["source_name"]),
                "session_bucket": _session_bucket_label(group["session_bucket"]),
                "source_session_breakdown": json.dumps(
                    source_session_breakdown,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    daily = pd.DataFrame(daily_rows).sort_values(["ticker", "news_date"])
    frames: list[pd.DataFrame] = []
    for ticker in tickers:
        daily_ticker = daily.loc[daily["ticker"] == ticker].drop(columns=["ticker"])
        frame_dates = business_dates
        if not daily_ticker.empty:
            frame_dates = pd.DatetimeIndex(
                sorted(set(business_dates.tolist()).union(pd.to_datetime(daily_ticker["news_date"], utc=True).tolist()))
            )
        frame = _empty_daily_news_frame(
            ticker=ticker,
            dates=frame_dates,
            fetched_at=fetched_at,
            source=output_source,
        )
        merged = frame.merge(daily_ticker, on="news_date", how="left", suffixes=("", "_daily"))
        for column in [
            "published_at",
            "available_at",
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
            "session_bucket",
            "source_session_breakdown",
        ]:
            daily_column = f"{column}_daily"
            if daily_column in merged.columns:
                merged[column] = merged[daily_column].combine_first(merged[column])
        merged["headline_count"] = merged["headline_count"].fillna(0).astype(int)
        for column in [
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
            "mapping_confidence",
            "novelty_score",
            "source_diversity",
            "source_count",
        ]:
            merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(0.0)
        merged["source_mix"] = merged["source_mix"].fillna("none").astype(str)
        merged["session_bucket"] = merged["session_bucket"].fillna("none").astype(str)
        merged["source_session_breakdown"] = merged["source_session_breakdown"].fillna("{}").astype(str)
        frames.append(
            merged.drop(
                columns=[
                    "news_date",
                    "published_at_daily",
                    "available_at_daily",
                    "sentiment_score_daily",
                    "sentiment_score_unweighted_daily",
                    "sentiment_score_source_weighted_daily",
                    "sentiment_score_session_weighted_daily",
                    "sentiment_score_source_session_weighted_daily",
                    "relevance_score_daily",
                    "relevance_score_unweighted_daily",
                    "relevance_score_source_weighted_daily",
                    "relevance_score_session_weighted_daily",
                    "relevance_score_source_session_weighted_daily",
                    "headline_count_daily",
                    "mapping_confidence_daily",
                    "novelty_score_daily",
                    "source_diversity_daily",
                    "source_count_daily",
                    "source_mix_daily",
                    "session_bucket_daily",
                    "source_session_breakdown_daily",
                ],
                errors="ignore",
            )
        )
    return pd.concat(frames, ignore_index=True)


class YahooFinanceNewsAdapter(NewsAdapter):
    name = "yahoo_finance_rss"

    def __init__(
        self,
        transport: PublicDataTransportConfig | None = None,
        *,
        source_weights: dict[str, float] | None = None,
        session_weights: dict[str, float] | None = None,
    ) -> None:
        self.transport = transport or _default_public_data_transport_config()
        self.source_weights = source_weights or {}
        self.session_weights = session_weights or {}
        self.last_fetch_metadata: dict[str, object] = {}

    def fetch(self, request: NewsRequest) -> pd.DataFrame:
        business_dates = pd.DatetimeIndex(business_dates_between(request.start_date, request.end_date))
        fetched_at = to_utc_timestamp(pd.Timestamp.now(tz="UTC"))
        alias_lookup = _news_alias_lookup(request.tickers)
        raw_items, metadata = self.fetch_raw_items(request)
        self.last_fetch_metadata = metadata
        return _aggregate_news_items_to_daily_frame(
            tickers=request.tickers,
            raw_items=raw_items,
            alias_lookup=alias_lookup,
            business_dates=business_dates,
            fetched_at=fetched_at,
            output_source=self.name,
            source_weights=self.source_weights,
            session_weights=self.session_weights,
        )

    def fetch_raw_items(self, request: NewsRequest) -> tuple[pd.DataFrame, dict[str, object]]:
        start = pd.Timestamp(request.start_date, tz="UTC")
        end = pd.Timestamp(request.end_date, tz="UTC")
        fetch_requests: list[dict[str, object]] = []
        item_frames: list[pd.DataFrame] = []
        for ticker in request.tickers:
            items, metadata = self._fetch_ticker_feed_items(
                ticker=ticker,
                start=start,
                end=end,
            )
            item_frames.append(items)
            fetch_requests.append(metadata)
        raw_items = pd.concat(item_frames, ignore_index=True) if item_frames else pd.DataFrame()
        metadata = _summarize_public_fetches(fetch_requests)
        metadata["used_sources"] = [self.name]
        return raw_items, metadata

    def _fetch_ticker_feed_items(
        self,
        *,
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        query = urllib.parse.urlencode(
            {
                "s": ticker,
                "region": "US",
                "lang": "en-US",
                "start": start.date().isoformat(),
                "end": end.date().isoformat(),
            }
        )
        url = f"https://finance.yahoo.com/rss/headline?{query}"
        payload_text, metadata = _fetch_public_text(url=url, namespace="yahoo_finance_rss", transport=self.transport)
        items = self._parse_feed_items(payload_text, ticker=ticker, start=start, end=end)
        return items, {**metadata, "ticker": ticker, "source_name": self.name}

    def _parse_feed_items(
        self,
        feed_text: str,
        *,
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        try:
            root = ET.fromstring(feed_text)
        except ET.ParseError as exc:
            raise RuntimeError(f"Yahoo Finance RSS returned invalid XML for ticker {ticker}: {exc}") from exc
        rows: list[dict[str, object]] = []
        upper_bound = end + pd.Timedelta(days=1)
        for item in root.findall(".//item"):
            title = _strip_html(_xml_child_text(item, "title"))
            description = _strip_html(_xml_child_text(item, "description"))
            link = _xml_child_text(item, "link")
            published_text = _xml_child_text(item, "pubDate") or _xml_child_text(item, "published")
            published_at = pd.to_datetime(published_text, utc=True, errors="coerce")
            if pd.isna(published_at):
                continue
            published_at = cast(pd.Timestamp, published_at)
            if published_at < start or published_at >= upper_bound:
                continue
            body = f"{title} {description}".strip()
            source_label = _source_label_from_item(item, link, "finance.yahoo.com")
            rows.append(
                {
                    "feed_ticker": ticker,
                    "published_at": published_at,
                    "body": body,
                    "headline_key": _headline_key(body),
                    "source_key": source_label.lower(),
                    "source_label": source_label,
                    "source_name": self.name,
                    "sentiment_score": _headline_sentiment_score(body),
                    "base_relevance_score": _headline_relevance_score(body, ticker),
                }
            )
        return pd.DataFrame(rows)


class GoogleNewsRssAdapter(NewsAdapter):
    name = "google_news_rss"

    def __init__(
        self,
        transport: PublicDataTransportConfig | None = None,
        *,
        source_weights: dict[str, float] | None = None,
        session_weights: dict[str, float] | None = None,
    ) -> None:
        self.transport = transport or _default_public_data_transport_config()
        self.source_weights = source_weights or {}
        self.session_weights = session_weights or {}
        self.last_fetch_metadata: dict[str, object] = {}

    def fetch(self, request: NewsRequest) -> pd.DataFrame:
        business_dates = pd.DatetimeIndex(business_dates_between(request.start_date, request.end_date))
        fetched_at = to_utc_timestamp(pd.Timestamp.now(tz="UTC"))
        alias_lookup = _news_alias_lookup(request.tickers)
        raw_items, metadata = self.fetch_raw_items(request)
        self.last_fetch_metadata = metadata
        return _aggregate_news_items_to_daily_frame(
            tickers=request.tickers,
            raw_items=raw_items,
            alias_lookup=alias_lookup,
            business_dates=business_dates,
            fetched_at=fetched_at,
            output_source=self.name,
            source_weights=self.source_weights,
            session_weights=self.session_weights,
        )

    def fetch_raw_items(self, request: NewsRequest) -> tuple[pd.DataFrame, dict[str, object]]:
        start = pd.Timestamp(request.start_date, tz="UTC")
        end = pd.Timestamp(request.end_date, tz="UTC")
        fetch_requests: list[dict[str, object]] = []
        item_frames: list[pd.DataFrame] = []
        for ticker in request.tickers:
            items, metadata = self._fetch_ticker_feed_items(ticker=ticker, start=start, end=end)
            item_frames.append(items)
            request_records = cast(list[dict[str, object]], metadata.get("requests", [])) or []
            if request_records:
                fetch_requests.extend(request_records)
            else:
                fetch_requests.append(metadata)
        raw_items = pd.concat(item_frames, ignore_index=True) if item_frames else pd.DataFrame()
        metadata = _summarize_public_fetches(fetch_requests)
        metadata["used_sources"] = [self.name]
        return raw_items, metadata

    def _fetch_ticker_feed_items(
        self,
        *,
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        lookback_days = int(np.clip((end.normalize() - start.normalize()).days + 1, 7, 30))
        query_variants = _google_news_query_variants(ticker, lookback_days)
        fetch_requests: list[dict[str, object]] = []
        item_frames: list[pd.DataFrame] = []
        errors: list[dict[str, str]] = []
        successful_variants: list[str] = []
        for label, query_text in query_variants:
            query = urllib.parse.urlencode(
                {
                    "q": query_text,
                    "hl": "en-US",
                    "gl": "US",
                    "ceid": "US:en",
                }
            )
            url = f"https://news.google.com/rss/search?{query}"
            try:
                payload_text, metadata = _fetch_public_text(
                    url=url,
                    namespace="google_news_rss",
                    transport=self.transport,
                )
                items = self._parse_feed_items(payload_text, ticker=ticker, start=start, end=end)
                if not items.empty:
                    item_frames.append(items)
                fetch_requests.append({**metadata, "ticker": ticker, "source_name": self.name, "query_variant": label})
                successful_variants.append(label)
            except (RuntimeError, ValueError, OSError) as exc:
                errors.append({"query_variant": label, "reason": str(exc)})
        if not fetch_requests:
            raise RuntimeError(
                "Google News query variants failed for ticker "
                f"{ticker}: " + "; ".join(f"{item['query_variant']}={item['reason']}" for item in errors)
            )
        items = pd.concat(item_frames, ignore_index=True) if item_frames else pd.DataFrame()
        if not items.empty:
            items = (
                items.sort_values(["published_at", "headline_key", "source_key"])
                .drop_duplicates(["headline_key", "published_at", "source_key"], keep="first")
                .reset_index(drop=True)
            )
        metadata = _summarize_public_fetches(fetch_requests)
        metadata["query_variants"] = [label for label, _ in query_variants]
        metadata["successful_query_variants"] = successful_variants
        metadata["failed_query_variants"] = errors
        return items, {**metadata, "ticker": ticker, "source_name": self.name}

    def _parse_feed_items(
        self,
        feed_text: str,
        *,
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        try:
            root = ET.fromstring(feed_text)
        except ET.ParseError as exc:
            raise RuntimeError(f"Google News RSS returned invalid XML for ticker {ticker}: {exc}") from exc
        rows: list[dict[str, object]] = []
        upper_bound = end + pd.Timedelta(days=1)
        for item in root.findall(".//item"):
            title = _strip_html(_xml_child_text(item, "title"))
            description = _strip_html(_xml_child_text(item, "description"))
            link = _xml_child_text(item, "link")
            published_text = _xml_child_text(item, "pubDate") or _xml_child_text(item, "published")
            published_at = pd.to_datetime(published_text, utc=True, errors="coerce")
            if pd.isna(published_at):
                continue
            published_at = cast(pd.Timestamp, published_at)
            if published_at < start or published_at >= upper_bound:
                continue
            body = f"{title} {description}".strip()
            source_label = _source_label_from_item(item, link, "news.google.com")
            rows.append(
                {
                    "feed_ticker": ticker,
                    "published_at": published_at,
                    "body": body,
                    "headline_key": _headline_key(body),
                    "source_key": source_label.lower(),
                    "source_label": source_label,
                    "source_name": self.name,
                    "sentiment_score": _headline_sentiment_score(body),
                    "base_relevance_score": _headline_relevance_score(body, ticker),
                }
            )
        return pd.DataFrame(rows)


class MultiSourceNewsAdapter(NewsAdapter):
    name = "multi_source_news"

    def __init__(
        self,
        adapters: list[NewsAdapter],
        *,
        source_weights: dict[str, float] | None = None,
        session_weights: dict[str, float] | None = None,
    ) -> None:
        if not adapters:
            raise ValueError("MultiSourceNewsAdapter requires at least one underlying adapter.")
        self.adapters = adapters
        self.source_weights = source_weights or {}
        self.session_weights = session_weights or {}
        self.last_fetch_metadata: dict[str, object] = {}

    def fetch(self, request: NewsRequest) -> pd.DataFrame:
        business_dates = pd.DatetimeIndex(business_dates_between(request.start_date, request.end_date))
        fetched_at = to_utc_timestamp(pd.Timestamp.now(tz="UTC"))
        alias_lookup = _news_alias_lookup(request.tickers)
        fetch_requests: list[dict[str, object]] = []
        raw_frames: list[pd.DataFrame] = []
        errors: list[dict[str, str]] = []
        used_sources: list[str] = []
        by_source: dict[str, dict[str, object]] = {}
        for adapter in self.adapters:
            source_name = cast(str, getattr(adapter, "name", adapter.__class__.__name__.lower()))
            try:
                fetch_raw_items = getattr(adapter, "fetch_raw_items")
                raw_items, metadata = cast(
                    tuple[pd.DataFrame, dict[str, object]],
                    fetch_raw_items(request),
                )
                if not raw_items.empty:
                    raw_frames.append(raw_items)
                request_records = cast(list[dict[str, object]], metadata.get("requests", [])) or []
                if request_records:
                    fetch_requests.extend(request_records)
                else:
                    fetch_requests.append({**metadata, "source_name": source_name})
                used_sources.append(source_name)
                by_source[source_name] = metadata
            except (AttributeError, RuntimeError, ValueError, OSError) as exc:
                errors.append({"source": source_name, "reason": str(exc)})
        if not used_sources:
            raise RuntimeError(
                "All configured live news sources failed. "
                + "; ".join(f"{item['source']}={item['reason']}" for item in errors)
            )
        raw_items = pd.concat(raw_frames, ignore_index=True) if raw_frames else pd.DataFrame()
        metadata = _summarize_public_fetches(fetch_requests)
        metadata["used_sources"] = used_sources
        metadata["failed_sources"] = errors
        metadata["by_source"] = by_source
        self.last_fetch_metadata = metadata
        return _aggregate_news_items_to_daily_frame(
            tickers=request.tickers,
            raw_items=raw_items,
            alias_lookup=alias_lookup,
            business_dates=business_dates,
            fetched_at=fetched_at,
            output_source=_source_mix_label(pd.Series(used_sources, dtype=str)),
            source_weights=self.source_weights,
            session_weights=self.session_weights,
        )


class SecCompanyFactsAdapter(FundamentalsAdapter):
    name = "sec_companyfacts"
    TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
    REVENUE_CONCEPTS = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
    ]
    NET_INCOME_CONCEPTS = ["NetIncomeLoss"]
    EPS_CONCEPTS = [
        "EarningsPerShareDiluted",
        "EarningsPerShareBasicAndDiluted",
        "EarningsPerShareBasic",
    ]
    DEBT_CONCEPTS = [
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "Liabilities",
    ]
    EQUITY_CONCEPTS = [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "StockholdersEquityAttributableToParent",
    ]

    def __init__(self, transport: PublicDataTransportConfig | None = None) -> None:
        self.transport = transport or _default_public_data_transport_config()
        self.last_fetch_metadata: dict[str, object] = {}

    def fetch(self, request: FundamentalsRequest) -> pd.DataFrame:
        ticker_map, ticker_map_metadata = self._fetch_ticker_map()
        fetched_at = to_utc_timestamp(pd.Timestamp.now(tz="UTC"))
        start = pd.Timestamp(request.start_date, tz="UTC")
        end = pd.Timestamp(request.end_date, tz="UTC")
        fetch_requests: list[dict[str, object]] = [ticker_map_metadata]
        frames: list[pd.DataFrame] = []
        price_adapter = YahooChartOHLCVAdapter(transport=self.transport)
        for ticker in request.tickers:
            cik = ticker_map.get(ticker.upper())
            if cik is None:
                continue
            try:
                companyfacts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
                payload_text, metadata = _fetch_public_text(
                    url=companyfacts_url,
                    namespace="sec_companyfacts",
                    transport=self.transport,
                    headers=SEC_HEADERS,
                )
                fetch_requests.append({**metadata, "ticker": ticker, "cik": cik})
                companyfacts = cast(dict[str, object], json.loads(payload_text))
                price_frame = price_adapter.fetch(
                    OHLCVRequest(tickers=[ticker], start_date=request.start_date, end_date=request.end_date)
                )
                price_requests = cast(list[dict[str, object]], price_adapter.last_fetch_metadata.get("requests", []))
                fetch_requests.extend([{**item, "ticker": ticker, "namespace": "yahoo_chart"} for item in price_requests])
                frame = self._build_ticker_frame(
                    ticker=ticker,
                    companyfacts=companyfacts,
                    start=start,
                    end=end,
                    price_frame=price_frame,
                    fetched_at=fetched_at,
                )
                if not frame.empty:
                    frames.append(frame)
            except (RuntimeError, ValueError, OSError, KeyError, json.JSONDecodeError):
                continue
        if not frames:
            raise RuntimeError("SEC companyfacts returned no usable fundamentals rows.")
        self.last_fetch_metadata = _summarize_public_fetches(fetch_requests)
        return pd.concat(frames, ignore_index=True)

    def _fetch_ticker_map(self) -> tuple[dict[str, str], dict[str, object]]:
        payload_text, metadata = _fetch_public_text(
            url=self.TICKER_MAP_URL,
            namespace="sec_ticker_map",
            transport=self.transport,
            headers=SEC_HEADERS,
        )
        payload = cast(dict[str, object], json.loads(payload_text))
        mapping: dict[str, str] = {}
        for item in payload.values():
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("ticker", "")).upper()
            cik_number = item.get("cik_str")
            if not ticker or cik_number in {None, ""}:
                continue
            mapping[ticker] = f"{int(cast(int | str, cik_number)):010d}"
        return mapping, metadata

    def _build_ticker_frame(
        self,
        *,
        ticker: str,
        companyfacts: dict[str, object],
        start: pd.Timestamp,
        end: pd.Timestamp,
        price_frame: pd.DataFrame,
        fetched_at: pd.Timestamp,
    ) -> pd.DataFrame:
        facts_root = cast(dict[str, object], companyfacts.get("facts", {}))
        us_gaap = cast(dict[str, object], facts_root.get("us-gaap", {}))
        revenue = _select_sec_metric_records(
            us_gaap,
            concept_names=self.REVENUE_CONCEPTS,
            unit_prefixes=("USD",),
            min_duration_days=70,
            max_duration_days=110,
        )
        if revenue.empty:
            revenue = _select_sec_metric_records(
                us_gaap,
                concept_names=self.REVENUE_CONCEPTS,
                unit_prefixes=("USD",),
                min_duration_days=300,
                max_duration_days=380,
            )
        if revenue.empty:
            return _empty_fundamentals_frame()
        net_income = _select_sec_metric_records(
            us_gaap,
            concept_names=self.NET_INCOME_CONCEPTS,
            unit_prefixes=("USD",),
            min_duration_days=70,
            max_duration_days=110,
        )
        if net_income.empty:
            net_income = _select_sec_metric_records(
                us_gaap,
                concept_names=self.NET_INCOME_CONCEPTS,
                unit_prefixes=("USD",),
                min_duration_days=300,
                max_duration_days=380,
            )
        eps = _select_sec_metric_records(
            us_gaap,
            concept_names=self.EPS_CONCEPTS,
            unit_prefixes=("USD",),
            min_duration_days=70,
            max_duration_days=110,
        )
        if eps.empty:
            eps = _select_sec_metric_records(
                us_gaap,
                concept_names=self.EPS_CONCEPTS,
                unit_prefixes=("USD",),
                min_duration_days=300,
                max_duration_days=380,
            )
        debt = _select_sec_metric_records(
            us_gaap,
            concept_names=self.DEBT_CONCEPTS,
            unit_prefixes=("USD",),
        )
        equity = _select_sec_metric_records(
            us_gaap,
            concept_names=self.EQUITY_CONCEPTS,
            unit_prefixes=("USD",),
        )
        frame = revenue.rename(columns={"value": "revenue", "filed_at": "revenue_filed_at"}).drop(
            columns=["concept_name", "concept_priority", "duration_days"]
        )
        for metric_frame, value_name, filed_name in [
            (net_income, "net_income", "net_income_filed_at"),
            (eps, "eps", "eps_filed_at"),
            (debt, "debt", "debt_filed_at"),
            (equity, "equity", "equity_filed_at"),
        ]:
            if metric_frame.empty:
                continue
            subset = metric_frame.rename(columns={"value": value_name, "filed_at": filed_name}).drop(
                columns=["concept_name", "concept_priority", "duration_days"]
            )
            frame = frame.merge(subset, on="report_date", how="left")
        if frame.empty:
            return _empty_fundamentals_frame()
        frame = frame.sort_values("report_date").reset_index(drop=True)
        filed_columns = [column for column in frame.columns if column.endswith("_filed_at")]
        frame["available_at"] = frame[filed_columns].bfill(axis=1).iloc[:, 0]
        for column in filed_columns:
            frame["available_at"] = frame[["available_at", column]].max(axis=1)
        frame = frame.dropna(subset=["available_at"])
        frame = frame.loc[(frame["report_date"] >= start) & (frame["report_date"] <= end)].copy()
        if frame.empty:
            return _empty_fundamentals_frame()
        frame["revenue_growth"] = frame["revenue"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        frame["profitability"] = (frame["net_income"] / frame["revenue"]).replace([np.inf, -np.inf], np.nan)
        frame["debt_to_equity"] = (frame["debt"] / frame["equity"]).replace([np.inf, -np.inf], np.nan)
        frame["annualized_eps"] = np.where(
            frame["eps"].notna(),
            np.where(frame["report_date"].diff().dt.days.fillna(91) <= 130, frame["eps"] * 4.0, frame["eps"]),
            np.nan,
        )
        frame["close_at_available"] = frame["available_at"].apply(
            lambda timestamp: _nearest_close_on_or_before(price_frame, cast(pd.Timestamp, timestamp))
        )
        frame["earnings_yield"] = (frame["annualized_eps"] / frame["close_at_available"]).replace([np.inf, -np.inf], np.nan)
        for column in ["earnings_yield", "debt_to_equity", "profitability"]:
            frame[column] = frame[column].ffill()
        frame = frame.dropna(subset=["revenue_growth", "earnings_yield", "debt_to_equity", "profitability"])
        if frame.empty:
            return _empty_fundamentals_frame()
        return pd.DataFrame(
            {
                "ticker": ticker,
                "report_date": frame["report_date"],
                "available_at": frame["available_at"],
                "revenue_growth": frame["revenue_growth"].astype(float),
                "earnings_yield": frame["earnings_yield"].astype(float),
                "debt_to_equity": frame["debt_to_equity"].astype(float),
                "profitability": frame["profitability"].astype(float),
                "source": self.name,
                "fetched_at": fetched_at,
                "stale_data_flag": False,
            }
        ).reset_index(drop=True)


class OfflineFundamentalProxyAdapter(FundamentalsAdapter):
    name = "offline_fundamental_proxy"

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def fetch(self, request: FundamentalsRequest) -> pd.DataFrame:
        all_dates = business_dates_between(request.start_date, request.end_date)
        report_dates = all_dates[::63]
        if len(report_dates) == 0:
            report_dates = all_dates[:1]
        rows: list[dict[str, object]] = []
        fetched_at = to_utc_timestamp(pd.Timestamp.now(tz="UTC"))
        for ticker in request.tickers:
            rng = np.random.default_rng(self.seed + _stable_integer(f"fundamental:{ticker}") % 10_000)
            quality = ((_stable_integer(f"quality:{ticker}") % 200) / 100.0) - 1.0
            for index, report_date in enumerate(report_dates):
                cycle = np.sin(index / 2.0 + quality)
                revenue_growth = 0.03 + 0.04 * quality + 0.015 * cycle + rng.normal(0.0, 0.005)
                earnings_yield = 0.045 + 0.02 * quality + rng.normal(0.0, 0.004)
                debt_to_equity = max(0.05, 1.10 - 0.45 * quality + rng.normal(0.0, 0.08))
                profitability = 0.08 + 0.03 * quality + 0.01 * cycle + rng.normal(0.0, 0.004)
                rows.append(
                    {
                        "ticker": ticker,
                        "report_date": pd.Timestamp(report_date).tz_convert("UTC"),
                        "available_at": pd.Timestamp(report_date).tz_convert("UTC") + timedelta(days=3),
                        "revenue_growth": float(revenue_growth),
                        "earnings_yield": float(earnings_yield),
                        "debt_to_equity": float(debt_to_equity),
                        "profitability": float(profitability),
                        "source": self.name,
                        "fetched_at": fetched_at,
                        "stale_data_flag": False,
                    }
                )
        return pd.DataFrame(rows)


class StaticSectorMapAdapter(SectorAdapter):
    name = "static_sector_map"

    def fetch(self, request: SectorRequest) -> pd.DataFrame:
        fetched_at = to_utc_timestamp(pd.Timestamp.now(tz="UTC"))
        rows = [
            {
                "ticker": ticker,
                "sector": _sector_for_ticker(ticker),
                "source": self.name,
                "fetched_at": fetched_at,
                "stale_data_flag": False,
            }
            for ticker in request.tickers
        ]
        return pd.DataFrame(rows)


class PolygonOHLCVAdapter(OHLCVAdapter):
    name = "polygon"

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise MissingCredentialsError("POLYGON_API_KEY is required for Polygon adapter.")
        self.api_key = api_key

    def fetch(self, request: OHLCVRequest) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for ticker in request.tickers:
            url = (
                f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
                f"{request.start_date}/{request.end_date}"
                f"?adjusted=true&sort=asc&apiKey={urllib.parse.quote(self.api_key)}"
            )
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            for item in payload.get("results", []):
                rows.append(
                    {
                        "ticker": ticker,
                        "timestamp_utc": pd.Timestamp(item["t"], unit="ms", tz="UTC"),
                        "open": float(item["o"]),
                        "high": float(item["h"]),
                        "low": float(item["l"]),
                        "close": float(item["c"]),
                        "volume": float(item["v"]),
                        "source": self.name,
                        "fetched_at": to_utc_timestamp(pd.Timestamp.now(tz="UTC")),
                        "stale_data_flag": False,
                    }
                )
        return pd.DataFrame(rows)


class AlphaVantageOHLCVAdapter(OHLCVAdapter):
    name = "alphavantage"

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise MissingCredentialsError("ALPHAVANTAGE_API_KEY is required for Alpha Vantage adapter.")
        self.api_key = api_key

    def fetch(self, request: OHLCVRequest) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for ticker in request.tickers:
            query = urllib.parse.urlencode(
                {
                    "function": "TIME_SERIES_DAILY_ADJUSTED",
                    "symbol": ticker,
                    "outputsize": "full",
                    "apikey": self.api_key,
                }
            )
            url = f"https://www.alphavantage.co/query?{query}"
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            daily = payload.get("Time Series (Daily)", {})
            for day_text, values in daily.items():
                day = pd.Timestamp(day_text, tz="UTC")
                if day < pd.Timestamp(request.start_date, tz="UTC") or day > pd.Timestamp(request.end_date, tz="UTC"):
                    continue
                rows.append(
                    {
                        "ticker": ticker,
                        "timestamp_utc": day,
                        "open": float(values["1. open"]),
                        "high": float(values["2. high"]),
                        "low": float(values["3. low"]),
                        "close": float(values["4. close"]),
                        "volume": float(values["6. volume"]),
                        "source": self.name,
                        "fetched_at": to_utc_timestamp(pd.Timestamp.now(tz="UTC")),
                        "stale_data_flag": False,
                    }
                )
        return pd.DataFrame(rows)


def _stooq_symbol(ticker: str) -> str:
    normalized = ticker.strip().upper()
    if normalized.endswith(".T"):
        return f"{normalized[:-2].lower()}.jp"
    if normalized.endswith(".JP") or normalized.endswith(".US"):
        return normalized.lower()
    return f"{normalized.lower()}.us"


class StooqOHLCVAdapter(OHLCVAdapter):
    name = "stooq"

    def __init__(self, max_pages: int = 60, transport: PublicDataTransportConfig | None = None) -> None:
        self.max_pages = max_pages
        self.transport = transport or _default_public_data_transport_config()
        self.last_fetch_metadata: dict[str, object] = {}

    def fetch(self, request: OHLCVRequest) -> pd.DataFrame:
        start = pd.Timestamp(request.start_date, tz="UTC")
        end = pd.Timestamp(request.end_date, tz="UTC")
        fetched_at = to_utc_timestamp(pd.Timestamp.now(tz="UTC"))
        ticker_frames: list[pd.DataFrame] = []
        fetch_requests: list[dict[str, object]] = []
        for ticker in request.tickers:
            history, metadata = self._fetch_ticker_history(ticker=ticker, start=start, end=end, fetched_at=fetched_at)
            ticker_frames.append(history)
            fetch_requests.extend(cast(list[dict[str, object]], metadata["requests"]))
        if not ticker_frames:
            raise RuntimeError("Stooq returned no ticker history.")
        self.last_fetch_metadata = _summarize_public_fetches(fetch_requests)
        return pd.concat(ticker_frames, ignore_index=True)

    def _fetch_ticker_history(
        self,
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        fetched_at: pd.Timestamp,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        collected: list[pd.DataFrame] = []
        previous_oldest: pd.Timestamp | None = None
        page_requests: list[dict[str, object]] = []
        for page in range(1, self.max_pages + 1):
            raw_page, page_metadata = self._fetch_page(ticker=ticker, page=page, fetched_at=fetched_at)
            page_requests.append(page_metadata)
            if raw_page.empty:
                break
            page_frame = raw_page.loc[(raw_page["timestamp_utc"] >= start) & (raw_page["timestamp_utc"] <= end)].copy()
            if not page_frame.empty:
                collected.append(page_frame)
            oldest_seen = page_frame["timestamp_utc"].min() if not page_frame.empty else None
            if oldest_seen is None:
                raw_oldest = raw_page["timestamp_utc"].min()
                if raw_oldest <= start:
                    break
                oldest_seen = raw_oldest
            if oldest_seen <= start or previous_oldest == oldest_seen:
                break
            previous_oldest = oldest_seen
        if not collected:
            raise RuntimeError(f"Stooq returned no rows for ticker {ticker} in the requested window.")
        result = pd.concat(collected, ignore_index=True).drop_duplicates(subset=["ticker", "timestamp_utc"])
        return result.sort_values("timestamp_utc").reset_index(drop=True), {"requests": page_requests}

    def _fetch_page(
        self,
        ticker: str,
        page: int,
        fetched_at: pd.Timestamp,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        symbol = _stooq_symbol(ticker)
        query = {"s": symbol, "i": "d"}
        if page > 1:
            query["l"] = str(page)
        url = f"https://stooq.com/q/d/?{urllib.parse.urlencode(query)}"
        html, metadata = _fetch_public_text(url=url, namespace="stooq", transport=self.transport)
        tables = pd.read_html(StringIO(html))
        history_table: pd.DataFrame | None = None
        for table in tables:
            normalized_columns = {str(column).strip() for column in table.columns}
            if {"Date", "Open", "High", "Low", "Close", "Volume"}.issubset(normalized_columns):
                history_table = table.copy()
                break
        if history_table is None or history_table.empty:
            return pd.DataFrame(), metadata
        history_table.columns = [str(column).strip() for column in history_table.columns]
        frame = history_table.loc[:, ["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        frame["timestamp_utc"] = pd.to_datetime(frame["Date"], format="%d %b %Y", utc=True, errors="coerce")
        for column in ["Open", "High", "Low", "Close", "Volume"]:
            frame[column] = pd.to_numeric(frame[column].astype(str).str.replace(",", "", regex=False), errors="coerce")
        frame = frame.dropna(subset=["timestamp_utc", "Open", "High", "Low", "Close", "Volume"])
        if frame.empty:
            return pd.DataFrame(), metadata
        return pd.DataFrame(
            {
                "ticker": ticker,
                "timestamp_utc": frame["timestamp_utc"],
                "open": frame["Open"].astype(float),
                "high": frame["High"].astype(float),
                "low": frame["Low"].astype(float),
                "close": frame["Close"].astype(float),
                "volume": frame["Volume"].astype(float),
                "source": self.name,
                "fetched_at": fetched_at,
                "stale_data_flag": False,
            }
        ), metadata


class YahooChartOHLCVAdapter(OHLCVAdapter):
    name = "yahoo_chart"

    def __init__(self, transport: PublicDataTransportConfig | None = None) -> None:
        self.transport = transport or _default_public_data_transport_config()
        self.last_fetch_metadata: dict[str, object] = {}

    def fetch(self, request: OHLCVRequest) -> pd.DataFrame:
        start = pd.Timestamp(request.start_date, tz="UTC")
        end = pd.Timestamp(request.end_date, tz="UTC")
        fetched_at = to_utc_timestamp(pd.Timestamp.now(tz="UTC"))
        frames: list[pd.DataFrame] = []
        fetch_requests: list[dict[str, object]] = []
        for ticker in request.tickers:
            frame, metadata = self._fetch_ticker_history(
                ticker=ticker,
                start=start,
                end=end,
                fetched_at=fetched_at,
            )
            frames.append(frame)
            fetch_requests.append(metadata)
        if not frames:
            raise RuntimeError("Yahoo chart returned no ticker history.")
        self.last_fetch_metadata = _summarize_public_fetches(fetch_requests)
        return pd.concat(frames, ignore_index=True)

    def _fetch_ticker_history(
        self,
        *,
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        fetched_at: pd.Timestamp,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        period_start = int(start.timestamp())
        period_end = int((end + pd.Timedelta(days=1)).timestamp())
        query = urllib.parse.urlencode(
            {
                "interval": "1d",
                "period1": period_start,
                "period2": period_end,
                "includeAdjustedClose": "true",
                "events": "div,splits",
            }
        )
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(ticker)}?{query}"
        payload_text, metadata = _fetch_public_text(url=url, namespace="yahoo_chart", transport=self.transport)
        payload = cast(dict[str, object], json.loads(payload_text))
        chart = cast(dict[str, object], payload.get("chart", {}))
        if chart.get("error"):
            raise RuntimeError(f"Yahoo chart returned error for ticker {ticker}: {chart['error']}")
        results = cast(list[dict[str, object]], chart.get("result", []))
        if not results:
            raise RuntimeError(f"Yahoo chart returned no result for ticker {ticker}.")
        result = results[0]
        timestamps = cast(list[int], result.get("timestamp", []))
        indicators = cast(dict[str, object], result.get("indicators", {}))
        quote_series = cast(list[dict[str, object]], indicators.get("quote", [{}]))
        quote_items = quote_series[0]
        if not timestamps:
            raise RuntimeError(f"Yahoo chart returned no rows for ticker {ticker} in the requested window.")
        frame = pd.DataFrame(
            {
                "timestamp_utc": pd.to_datetime(timestamps, unit="s", utc=True),
                "open": pd.to_numeric(quote_items.get("open"), errors="coerce"),
                "high": pd.to_numeric(quote_items.get("high"), errors="coerce"),
                "low": pd.to_numeric(quote_items.get("low"), errors="coerce"),
                "close": pd.to_numeric(quote_items.get("close"), errors="coerce"),
                "volume": pd.to_numeric(quote_items.get("volume"), errors="coerce"),
            }
        )
        frame = frame.dropna(subset=["timestamp_utc", "open", "high", "low", "close", "volume"])
        frame = frame.loc[
            (frame["timestamp_utc"] >= start)
            & (frame["timestamp_utc"] < (end + pd.Timedelta(days=1)))
        ].copy()
        if frame.empty:
            raise RuntimeError(f"Yahoo chart returned no rows for ticker {ticker} in the requested window.")
        return (
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "timestamp_utc": frame["timestamp_utc"],
                    "open": frame["open"].astype(float),
                    "high": frame["high"].astype(float),
                    "low": frame["low"].astype(float),
                    "close": frame["close"].astype(float),
                    "volume": frame["volume"].astype(float),
                    "source": self.name,
                    "fetched_at": fetched_at,
                    "stale_data_flag": False,
                }
            ),
            {**metadata, "ticker": ticker},
        )


class FredMarketProxyOHLCVAdapter(OHLCVAdapter):
    name = "fred_market_proxy"
    SERIES_MAP = {
        "SPY": "SP500",
        "QQQ": "NASDAQCOM",
        "DIA": "DJIA",
        "GLD": "GOLDAMGBD228NLBM",
    }

    def __init__(self, transport: PublicDataTransportConfig | None = None) -> None:
        self.transport = transport or _default_public_data_transport_config()
        self.last_fetch_metadata: dict[str, object] = {}

    def fetch(self, request: OHLCVRequest) -> pd.DataFrame:
        fetched_at = to_utc_timestamp(pd.Timestamp.now(tz="UTC"))
        business_dates = pd.DatetimeIndex(business_dates_between(request.start_date, request.end_date))
        frames: list[pd.DataFrame] = []
        fetch_requests: list[dict[str, object]] = []
        for ticker in request.tickers:
            series_id = self.SERIES_MAP.get(ticker.upper())
            if series_id is None:
                raise ValueError(f"Ticker {ticker} is not mapped for FRED market proxy adapter.")
            query = urllib.parse.urlencode(
                {
                    "id": series_id,
                    "cosd": request.start_date,
                    "coed": request.end_date,
                }
            )
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?{query}"
            payload_text, metadata = _fetch_public_text(url=url, namespace="fred_market_proxy", transport=self.transport)
            fetch_requests.append({**metadata, "ticker": ticker, "series_id": series_id})
            payload = pd.read_csv(StringIO(payload_text))
            if payload.empty:
                raise RuntimeError(f"FRED market proxy returned no data for {ticker}/{series_id}.")
            date_column = payload.columns[0]
            value_column = payload.columns[-1]
            payload = payload.rename(columns={date_column: "date", value_column: "value"})
            payload["date"] = pd.to_datetime(payload["date"], utc=True, errors="coerce")
            payload["value"] = pd.to_numeric(payload["value"], errors="coerce")
            payload = payload.dropna(subset=["date", "value"]).sort_values("date")
            series = payload.set_index("date")["value"].reindex(business_dates, method="ffill").dropna()
            if series.empty:
                raise RuntimeError(f"FRED market proxy returned no rows for ticker {ticker} in the requested window.")
            previous_close = series.shift(1).fillna(series)
            abs_move = (series / previous_close - 1.0).abs().fillna(0.001).clip(lower=0.0005, upper=0.03)
            open_px = previous_close.astype(float)
            close_px = series.astype(float)
            high_px = np.maximum(open_px, close_px) * (1.0 + abs_move * 0.5)
            low_px = np.minimum(open_px, close_px) * (1.0 - abs_move * 0.5)
            frame = pd.DataFrame(
                {
                    "ticker": ticker,
                    "timestamp_utc": series.index,
                    "open": open_px.to_numpy(dtype=float),
                    "high": high_px.to_numpy(dtype=float),
                    "low": low_px.to_numpy(dtype=float),
                    "close": close_px.to_numpy(dtype=float),
                    "volume": np.full(len(series), 1_000_000_000.0),
                    "source": self.name,
                    "fetched_at": fetched_at,
                    "stale_data_flag": False,
                }
            )
            frames.append(frame.reset_index(drop=True))
        self.last_fetch_metadata = _summarize_public_fetches(fetch_requests)
        return pd.concat(frames, ignore_index=True)


class FredMacroAdapter(MacroAdapter):
    name = "fred"

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise MissingCredentialsError("FRED_API_KEY is required for FRED adapter.")
        self.api_key = api_key

    def fetch(self, request: MacroRequest) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for series_id in request.series_ids:
            query = urllib.parse.urlencode(
                {
                    "series_id": series_id,
                    "api_key": self.api_key,
                    "file_type": "json",
                    "observation_start": request.start_date,
                    "observation_end": request.end_date,
                }
            )
            url = f"https://api.stlouisfed.org/fred/series/observations?{query}"
            with urllib.request.urlopen(url, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            for item in payload.get("observations", []):
                value = item.get("value")
                if value in {".", ""}:
                    continue
                observation_date = pd.Timestamp(item["date"], tz="UTC")
                rows.append(
                    {
                        "series_id": series_id,
                        "date": observation_date,
                        "value": float(value),
                        "available_at": observation_date + timedelta(days=1),
                        "source": self.name,
                    }
                )
        return pd.DataFrame(rows)


class FredCsvMacroAdapter(MacroAdapter):
    name = "fred_csv"

    def __init__(self, transport: PublicDataTransportConfig | None = None) -> None:
        self.transport = transport or _default_public_data_transport_config()
        self.last_fetch_metadata: dict[str, object] = {}

    def fetch(self, request: MacroRequest) -> pd.DataFrame:
        start = pd.Timestamp(request.start_date, tz="UTC")
        end = pd.Timestamp(request.end_date, tz="UTC")
        rows: list[dict[str, object]] = []
        fetch_requests: list[dict[str, object]] = []
        for series_id in request.series_ids:
            query = urllib.parse.urlencode(
                {
                    "id": series_id,
                    "cosd": request.start_date,
                    "coed": request.end_date,
                }
            )
            url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?{query}"
            payload_text, metadata = _fetch_public_text(url=url, namespace="fred_csv", transport=self.transport)
            fetch_requests.append({**metadata, "series_id": series_id})
            payload = pd.read_csv(StringIO(payload_text))
            if payload.empty:
                continue
            date_column = payload.columns[0]
            value_column = payload.columns[-1]
            payload = payload.rename(columns={date_column: "date", value_column: "value"})
            payload["date"] = pd.to_datetime(payload["date"], utc=True, errors="coerce")
            payload["value"] = pd.to_numeric(payload["value"], errors="coerce")
            payload = payload.dropna(subset=["date", "value"])
            payload = payload.loc[(payload["date"] >= start) & (payload["date"] <= end)].copy()
            for _, row in payload.iterrows():
                observation_date = pd.Timestamp(row["date"]).tz_convert("UTC")
                rows.append(
                    {
                        "series_id": series_id,
                        "date": observation_date,
                        "value": float(row["value"]),
                        "available_at": observation_date + timedelta(days=1),
                        "source": self.name,
                    }
                )
        if not rows:
            raise RuntimeError("FRED CSV adapter returned no macro observations.")
        self.last_fetch_metadata = _summarize_public_fetches(fetch_requests)
        return pd.DataFrame(rows)
