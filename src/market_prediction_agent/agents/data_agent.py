from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import cast

import pandas as pd

from market_prediction_agent.config import Settings
from market_prediction_agent.data.adapters import (
    AlphaVantageOHLCVAdapter,
    CoinGeckoAdapter,
    DummyMacroAdapter,
    DummyOHLCVAdapter,
    FundamentalsAdapter,
    FundamentalsRequest,
    FredMarketProxyOHLCVAdapter,
    FredCsvMacroAdapter,
    FredMacroAdapter,
    GoogleNewsRssAdapter,
    MacroAdapter,
    MacroRequest,
    MissingCredentialsError,
    MultiSourceNewsAdapter,
    NewsAdapter,
    NewsRequest,
    OfflineFundamentalProxyAdapter,
    OfflineNewsProxyAdapter,
    OHLCVAdapter,
    OHLCVRequest,
    PolygonOHLCVAdapter,
    PublicDataTransportConfig,
    SectorAdapter,
    SectorRequest,
    SecCompanyFactsAdapter,
    StaticSectorMapAdapter,
    StooqOHLCVAdapter,
    YahooFinanceNewsAdapter,
    YahooChartOHLCVAdapter,
)
from market_prediction_agent.data.normalizer import (
    apply_stale_flag,
    normalize_fundamentals,
    normalize_macro,
    normalize_news,
    normalize_ohlcv,
    normalize_sector_map,
)
from market_prediction_agent.data.universe import resolve_active_constituents, resolve_default_tickers
from market_prediction_agent.storage.parquet_store import ParquetStore
from market_prediction_agent.utils.paths import resolve_repo_path
from market_prediction_agent.utils.time_utils import to_utc_timestamp


logger = logging.getLogger(__name__)


def _used_source_label(requested_sources: list[str], used_sources: list[str]) -> str:
    if not used_sources:
        return requested_sources[0] if requested_sources else ""
    ordered = [source for source in requested_sources if source in used_sources]
    remaining = [source for source in used_sources if source not in ordered]
    resolved = ordered + remaining
    if len(resolved) == 1:
        return resolved[0]
    return "+".join(resolved)


@dataclass(slots=True)
class DataArtifacts:
    raw_ohlcv: pd.DataFrame
    raw_macro: pd.DataFrame
    raw_news: pd.DataFrame
    raw_fundamentals: pd.DataFrame
    raw_sector_map: pd.DataFrame
    processed_ohlcv: pd.DataFrame
    processed_macro: pd.DataFrame
    processed_news: pd.DataFrame
    processed_fundamentals: pd.DataFrame
    processed_sector_map: pd.DataFrame
    ohlcv_metadata: dict[str, object]


class DataAgent:
    def __init__(self, settings: Settings, store: ParquetStore) -> None:
        self.settings = settings
        self.store = store

    def default_tickers(self) -> list[str]:
        return resolve_default_tickers(self.settings)

    def active_universe_constituents(
        self,
        *,
        as_of_time: str | pd.Timestamp,
    ) -> list[str] | None:
        return resolve_active_constituents(self.settings, as_of_date=as_of_time)

    def generate_or_fetch(
        self,
        tickers: list[str],
        start_date: str,
        end_date: str,
        as_of_time: str | pd.Timestamp | None = None,
    ) -> DataArtifacts:
        as_of_timestamp = to_utc_timestamp(as_of_time or pd.Timestamp.now(tz="UTC"))
        request = OHLCVRequest(tickers=tickers, start_date=start_date, end_date=end_date)
        macro_adapter: MacroAdapter
        if self.settings.data.source_mode == "dummy":
            raw_ohlcv = self._fetch_dummy_ohlcv(request)
            ohlcv_metadata = {
                "requested_source": "dummy",
                "used_source": "dummy",
                "used_sources": ["dummy"],
                "dummy_mode": self.settings.data.dummy_mode,
                "fallback_used": False,
                "fallback_reason": None,
                "attempt_errors": [],
                "as_of_time": as_of_timestamp.isoformat(),
            }
            macro_adapter = DummyMacroAdapter(seed=self.settings.app.seed)
        else:
            raw_ohlcv, ohlcv_metadata = self._fetch_ohlcv_with_fallback(request, as_of_timestamp=as_of_timestamp)
            macro_adapter = self._build_macro_adapter()
        if self.settings.data.source_mode == "dummy":
            news_adapter = self._build_news_adapter("offline_news_proxy")
            raw_news = news_adapter.fetch(NewsRequest(tickers=tickers, start_date=start_date, end_date=end_date))
            news_metadata = {
                "requested_source": self.settings.data.news_source,
                "used_source": cast(str, getattr(news_adapter, "name", "offline_news_proxy")),
                "fallback_used": False,
                "fallback_reason": None,
                "attempt_errors": [],
                "as_of_time": as_of_timestamp.isoformat(),
                "transport": cast(dict[str, object], getattr(news_adapter, "last_fetch_metadata", {})),
            }
        else:
            raw_news, news_metadata = self._fetch_news_with_fallback(
                NewsRequest(tickers=tickers, start_date=start_date, end_date=end_date),
                as_of_timestamp=as_of_timestamp,
            )
        if self.settings.data.source_mode == "dummy":
            fundamentals_adapter = self._build_fundamentals_adapter("offline_fundamental_proxy")
            raw_fundamentals = fundamentals_adapter.fetch(
                FundamentalsRequest(tickers=tickers, start_date=start_date, end_date=end_date)
            )
            fundamentals_metadata = {
                "requested_source": self.settings.data.fundamentals_source,
                "used_source": cast(
                    str,
                    getattr(fundamentals_adapter, "name", "offline_fundamental_proxy"),
                ),
                "fallback_used": False,
                "fallback_reason": None,
                "attempt_errors": [],
                "as_of_time": as_of_timestamp.isoformat(),
                "transport": cast(dict[str, object], getattr(fundamentals_adapter, "last_fetch_metadata", {})),
            }
        else:
            raw_fundamentals, fundamentals_metadata = self._fetch_fundamentals_with_fallback(
                FundamentalsRequest(tickers=tickers, start_date=start_date, end_date=end_date),
                as_of_timestamp=as_of_timestamp,
            )
        sector_adapter = self._build_sector_adapter()
        raw_macro = macro_adapter.fetch(MacroRequest(series_ids=self.settings.data.macro_series, start_date=start_date, end_date=end_date))
        raw_sector_map = sector_adapter.fetch(SectorRequest(tickers=tickers))
        macro_source_name = cast(str, getattr(macro_adapter, "name", macro_adapter.__class__.__name__.lower()))
        ohlcv_metadata["macro_source"] = macro_source_name
        if self.settings.data.source_mode != "dummy":
            ohlcv_metadata["macro_public_data_transport"] = cast(
                dict[str, object],
                getattr(macro_adapter, "last_fetch_metadata", {}),
            )
        else:
            ohlcv_metadata["macro_public_data_transport"] = {}
        processed_news = apply_stale_flag(
            normalize_news(raw_news),
            as_of_time=as_of_timestamp,
            threshold_hours=self.settings.data.stale_threshold_hours.news,
        )
        processed_fundamentals = apply_stale_flag(
            normalize_fundamentals(raw_fundamentals),
            as_of_time=as_of_timestamp,
            threshold_hours=self.settings.data.stale_threshold_hours.fundamentals,
        )
        processed_sector_map = apply_stale_flag(
            normalize_sector_map(raw_sector_map),
            as_of_time=as_of_timestamp,
            threshold_hours=self.settings.data.stale_threshold_hours.sector,
        )
        processed_ohlcv = normalize_ohlcv(raw_ohlcv)
        processed_ohlcv = apply_stale_flag(
            processed_ohlcv,
            as_of_time=as_of_timestamp,
            threshold_hours=self.settings.data.stale_threshold_hours.ohlcv,
        )
        processed_macro = normalize_macro(raw_macro)
        ohlcv_metadata["feature_sources"] = {
            "news": {
                "requested_source": cast(str, news_metadata.get("requested_source", self.settings.data.news_source)),
                "used_source": cast(str, news_metadata.get("used_source", self.settings.data.news_source)),
                "requested_sources": cast(
                    list[str],
                    news_metadata.get("requested_sources", [self.settings.data.news_source]),
                ),
                "used_sources": cast(list[str], news_metadata.get("used_sources", [])),
                "record_count": int(len(processed_news)),
                "stale_rate": float(processed_news["stale_data_flag"].mean()) if not processed_news.empty else 0.0,
                "session_bucket_counts": (
                    processed_news["session_bucket"].fillna("none").astype(str).value_counts().astype(int).to_dict()
                    if not processed_news.empty and "session_bucket" in processed_news.columns
                    else {}
                ),
                "source_diversity_mean": (
                    float(processed_news["source_diversity"].mean())
                    if not processed_news.empty and "source_diversity" in processed_news.columns
                    else 0.0
                ),
                "fallback_used": bool(news_metadata.get("fallback_used", False)),
                "fallback_reason": news_metadata.get("fallback_reason"),
                "transport": cast(dict[str, object], news_metadata.get("transport", {})),
            },
            "fundamental": {
                "requested_source": cast(
                    str,
                    fundamentals_metadata.get("requested_source", self.settings.data.fundamentals_source),
                ),
                "used_source": cast(
                    str,
                    fundamentals_metadata.get("used_source", self.settings.data.fundamentals_source),
                ),
                "record_count": int(len(processed_fundamentals)),
                "stale_rate": (
                    float(processed_fundamentals["stale_data_flag"].mean()) if not processed_fundamentals.empty else 0.0
                ),
                "fallback_used": bool(fundamentals_metadata.get("fallback_used", False)),
                "fallback_reason": fundamentals_metadata.get("fallback_reason"),
                "transport": cast(dict[str, object], fundamentals_metadata.get("transport", {})),
            },
            "sector": {
                "used_source": cast(str, getattr(sector_adapter, "name", self.settings.data.sector_source)),
                "record_count": int(len(processed_sector_map)),
                "stale_rate": (
                    float(processed_sector_map["stale_data_flag"].mean()) if not processed_sector_map.empty else 0.0
                ),
                "transport": {},
            },
        }
        self.store.write_frame(Path("raw") / "ohlcv" / "ohlcv.parquet", raw_ohlcv)
        self.store.write_frame(Path("raw") / "macro" / "macro.parquet", raw_macro)
        self.store.write_frame(Path("raw") / "news" / "news.parquet", raw_news)
        self.store.write_frame(Path("raw") / "fundamentals" / "fundamentals.parquet", raw_fundamentals)
        self.store.write_frame(Path("raw") / "sector" / "sector_map.parquet", raw_sector_map)
        self.store.write_frame(Path("processed") / "ohlcv" / "ohlcv.parquet", processed_ohlcv)
        self.store.write_frame(Path("processed") / "macro" / "macro.parquet", processed_macro)
        self.store.write_frame(Path("processed") / "news" / "news.parquet", processed_news)
        self.store.write_frame(Path("processed") / "fundamentals" / "fundamentals.parquet", processed_fundamentals)
        self.store.write_frame(Path("processed") / "sector" / "sector_map.parquet", processed_sector_map)
        return DataArtifacts(
            raw_ohlcv=raw_ohlcv,
            raw_macro=raw_macro,
            raw_news=raw_news,
            raw_fundamentals=raw_fundamentals,
            raw_sector_map=raw_sector_map,
            processed_ohlcv=processed_ohlcv,
            processed_macro=processed_macro,
            processed_news=processed_news,
            processed_fundamentals=processed_fundamentals,
            processed_sector_map=processed_sector_map,
            ohlcv_metadata=ohlcv_metadata,
        )

    def fallback_adapter_name(self) -> str:
        if self.settings.data.fallback_source == "alphavantage":
            return AlphaVantageOHLCVAdapter.name
        return self.settings.data.fallback_source

    def _public_data_transport(self) -> PublicDataTransportConfig:
        public_data = self.settings.data.public_data
        return PublicDataTransportConfig(
            cache_dir=resolve_repo_path(public_data.cache_path),
            snapshot_dir=resolve_repo_path(public_data.snapshot_path),
            cache_ttl_hours=public_data.cache_ttl_hours,
            retry_count=public_data.retry_count,
            retry_backoff_seconds=public_data.retry_backoff_seconds,
        )

    def _known_crypto_tickers(self) -> set[str]:
        configured = {ticker.upper() for ticker in self.settings.data.crypto_tickers}
        return configured.union(CoinGeckoAdapter.TICKER_TO_COIN_ID)

    def _split_ohlcv_tickers(self, tickers: list[str]) -> tuple[list[str], list[str]]:
        known_crypto = self._known_crypto_tickers()
        crypto_tickers = [ticker for ticker in tickers if ticker.upper() in known_crypto]
        equity_tickers = [ticker for ticker in tickers if ticker.upper() not in known_crypto]
        if crypto_tickers and not self.settings.data.crypto_enabled:
            raise ValueError("Crypto OHLCV requested but data.crypto_enabled is false.")
        return equity_tickers, crypto_tickers

    def _split_equity_market_tickers(self, tickers: list[str]) -> tuple[list[str], list[str]]:
        jp_equity_tickers = [ticker for ticker in tickers if ticker.upper().endswith(".T")]
        primary_equity_tickers = [ticker for ticker in tickers if ticker.upper() not in {item.upper() for item in jp_equity_tickers}]
        return primary_equity_tickers, jp_equity_tickers

    def _fetch_dummy_ohlcv(self, request: OHLCVRequest) -> pd.DataFrame:
        equity_tickers, crypto_tickers = self._split_ohlcv_tickers(request.tickers)
        frames: list[pd.DataFrame] = []
        if equity_tickers:
            frames.append(
                DummyOHLCVAdapter(seed=self.settings.app.seed, mode=self.settings.data.dummy_mode).fetch(
                    OHLCVRequest(
                        tickers=equity_tickers,
                        start_date=request.start_date,
                        end_date=request.end_date,
                    )
                )
            )
        if crypto_tickers:
            frames.append(
                CoinGeckoAdapter(
                    transport=self._public_data_transport(),
                    source_mode="dummy",
                    seed=self.settings.app.seed,
                    dummy_mode=self.settings.data.dummy_mode,
                ).fetch(
                    OHLCVRequest(
                        tickers=crypto_tickers,
                        start_date=request.start_date,
                        end_date=request.end_date,
                    )
                )
            )
        if not frames:
            raise RuntimeError("No OHLCV tickers were provided.")
        return pd.concat(frames, ignore_index=True)

    def _build_macro_adapter(self) -> MacroAdapter:
        if self.settings.data.macro_source == "fred":
            return FredMacroAdapter(self.settings.api_keys.fred)
        if self.settings.data.macro_source == "fred_csv":
            return FredCsvMacroAdapter(transport=self._public_data_transport())
        if self.settings.data.macro_source == "auto" and self.settings.api_keys.fred:
            return FredMacroAdapter(self.settings.api_keys.fred)
        return FredCsvMacroAdapter(transport=self._public_data_transport())

    def _build_news_adapter(self, source_name: str | None = None) -> NewsAdapter:
        resolved_source = source_name or self.settings.data.news_source
        if resolved_source == "offline_news_proxy":
            mode = self.settings.data.dummy_mode if self.settings.data.source_mode == "dummy" else "live_proxy"
            return OfflineNewsProxyAdapter(seed=self.settings.app.seed, mode=mode)
        if resolved_source == "yahoo_finance_rss":
            return YahooFinanceNewsAdapter(
                transport=self._public_data_transport(),
                source_weights=self.settings.data.news_source_weights,
                session_weights=self.settings.data.news_session_weights,
            )
        if resolved_source == "google_news_rss":
            return GoogleNewsRssAdapter(
                transport=self._public_data_transport(),
                source_weights=self.settings.data.news_source_weights,
                session_weights=self.settings.data.news_session_weights,
            )
        raise ValueError(f"Unsupported news source: {resolved_source}")

    def _build_fundamentals_adapter(self, source_name: str | None = None) -> FundamentalsAdapter:
        resolved_source = source_name or self.settings.data.fundamentals_source
        if resolved_source == "offline_fundamental_proxy":
            return OfflineFundamentalProxyAdapter(seed=self.settings.app.seed)
        if resolved_source == "sec_companyfacts":
            return SecCompanyFactsAdapter(transport=self._public_data_transport())
        raise ValueError(f"Unsupported fundamentals source: {resolved_source}")

    def _build_sector_adapter(self) -> SectorAdapter:
        if self.settings.data.sector_source == "static_sector_map":
            return StaticSectorMapAdapter()
        raise ValueError(f"Unsupported sector source: {self.settings.data.sector_source}")

    def _build_ohlcv_adapter(self, source_name: str) -> OHLCVAdapter:
        if source_name == "polygon":
            return PolygonOHLCVAdapter(self.settings.api_keys.polygon)
        if source_name == "alphavantage":
            return AlphaVantageOHLCVAdapter(self.settings.api_keys.alphavantage)
        if source_name == "coingecko":
            return CoinGeckoAdapter(
                transport=self._public_data_transport(),
                source_mode=self.settings.data.source_mode,
                seed=self.settings.app.seed,
                dummy_mode=self.settings.data.dummy_mode,
            )
        if source_name == "stooq":
            return StooqOHLCVAdapter(transport=self._public_data_transport())
        if source_name == "yahoo_chart":
            return YahooChartOHLCVAdapter(transport=self._public_data_transport())
        if source_name == "fred_market_proxy":
            return FredMarketProxyOHLCVAdapter(transport=self._public_data_transport())
        raise ValueError(f"Unsupported OHLCV source: {source_name}")

    def _fetch_jp_equity_ohlcv(
        self,
        request: OHLCVRequest,
        *,
        as_of_timestamp: pd.Timestamp,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        source_name = self.settings.data.jp_equity.source
        try:
            adapter = self._build_ohlcv_adapter(source_name)
            frame = adapter.fetch(request)
        except (MissingCredentialsError, RuntimeError, ValueError, OSError) as exc:
            logger.warning("JP equity OHLCV fetch failed using source '%s': %s", source_name, exc)
            raise RuntimeError(f"JP equity OHLCV fetch failed using source '{source_name}': {exc}") from exc
        return (
            frame,
            {
                "requested_source": source_name,
                "used_source": source_name,
                "used_sources": [source_name],
                "dummy_mode": None,
                "fallback_used": False,
                "fallback_reason": None,
                "attempt_errors": [],
                "as_of_time": as_of_timestamp.isoformat(),
                "public_data_transport": cast(dict[str, object], getattr(adapter, "last_fetch_metadata", {})),
            },
        )

    def _fetch_equity_ohlcv_with_fallback(
        self,
        request: OHLCVRequest,
        as_of_timestamp: pd.Timestamp,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        sources = [self.settings.data.primary_source]
        if self.settings.data.fallback_source and self.settings.data.fallback_source not in sources:
            sources.append(self.settings.data.fallback_source)
        attempt_errors: list[dict[str, str]] = []
        first_error: str | None = None
        for index, source_name in enumerate(sources):
            try:
                adapter = self._build_ohlcv_adapter(source_name)
                frame = adapter.fetch(request)
                if index > 0:
                    logger.warning(
                        "Primary OHLCV source failed; fallback source '%s' succeeded. reason=%s",
                        source_name,
                        first_error,
                    )
                metadata = {
                    "requested_source": self.settings.data.primary_source,
                    "used_source": source_name,
                    "used_sources": [source_name],
                    "dummy_mode": None,
                    "fallback_used": index > 0,
                    "fallback_reason": first_error,
                    "attempt_errors": attempt_errors,
                    "as_of_time": as_of_timestamp.isoformat(),
                    "public_data_transport": cast(dict[str, object], getattr(adapter, "last_fetch_metadata", {})),
                }
                return frame, metadata
            except (MissingCredentialsError, RuntimeError, ValueError, OSError) as exc:
                logger.warning("OHLCV fetch failed using source '%s': %s", source_name, exc)
                attempt_errors.append({"source": source_name, "error": str(exc)})
                if first_error is None:
                    first_error = str(exc)
        error_summary = "; ".join(f"{item['source']}: {item['error']}" for item in attempt_errors)
        raise RuntimeError(f"OHLCV fetch failed for all configured sources. {error_summary}")

    def _fetch_news_with_fallback(
        self,
        request: NewsRequest,
        *,
        as_of_timestamp: pd.Timestamp,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        primary_sources = [self.settings.data.news_source]
        for source_name in self.settings.data.news_secondary_sources:
            if source_name and source_name not in primary_sources:
                primary_sources.append(source_name)
        attempt_errors: list[dict[str, str]] = []
        first_error: str | None = None
        try:
            live_adapters = [self._build_news_adapter(source_name) for source_name in primary_sources]
            adapter: NewsAdapter
            if len(live_adapters) == 1:
                adapter = live_adapters[0]
            else:
                adapter = MultiSourceNewsAdapter(
                    live_adapters,
                    source_weights=self.settings.data.news_source_weights,
                    session_weights=self.settings.data.news_session_weights,
                )
            frame = adapter.fetch(request)
            transport_metadata = cast(dict[str, object], getattr(adapter, "last_fetch_metadata", {}))
            used_sources = [str(item) for item in cast(list[object], transport_metadata.get("used_sources", []))]
            return frame, {
                "requested_source": self.settings.data.news_source,
                "requested_sources": primary_sources,
                "used_source": _used_source_label(primary_sources, used_sources),
                "used_sources": used_sources,
                "fallback_used": False,
                "fallback_reason": None,
                "attempt_errors": cast(list[dict[str, str]], transport_metadata.get("failed_sources", [])),
                "as_of_time": as_of_timestamp.isoformat(),
                "transport": transport_metadata,
            }
        except (MissingCredentialsError, RuntimeError, ValueError, OSError) as exc:
            logger.warning("Primary live news ingestion failed: %s", exc)
            attempt_errors.append({"source": "+".join(primary_sources), "error": str(exc)})
            first_error = str(exc)
        fallback_sources = [self.settings.data.news_fallback_source]
        for source_name in fallback_sources:
            try:
                adapter = self._build_news_adapter(source_name)
                frame = adapter.fetch(request)
                metadata = {
                    "requested_source": self.settings.data.news_source,
                    "requested_sources": primary_sources,
                    "used_source": source_name,
                    "used_sources": [source_name],
                    "fallback_used": True,
                    "fallback_reason": first_error,
                    "attempt_errors": attempt_errors,
                    "as_of_time": as_of_timestamp.isoformat(),
                    "transport": cast(dict[str, object], getattr(adapter, "last_fetch_metadata", {})),
                }
                logger.warning(
                    "Primary live news ingestion failed; fallback news source '%s' succeeded. reason=%s",
                    source_name,
                    first_error,
                )
                return frame, metadata
            except (MissingCredentialsError, RuntimeError, ValueError, OSError) as exc:
                logger.warning("News fetch failed using source '%s': %s", source_name, exc)
                attempt_errors.append({"source": source_name, "error": str(exc)})
        error_summary = "; ".join(f"{item['source']}: {item['error']}" for item in attempt_errors)
        raise RuntimeError(f"News fetch failed for all configured sources. {error_summary}")

    def _fetch_ohlcv_with_fallback(
        self,
        request: OHLCVRequest,
        as_of_timestamp: pd.Timestamp,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        equity_tickers, crypto_tickers = self._split_ohlcv_tickers(request.tickers)
        primary_equity_tickers, jp_equity_tickers = self._split_equity_market_tickers(equity_tickers)
        frames: list[pd.DataFrame] = []
        used_sources: list[str] = []
        attempt_errors: list[dict[str, str]] = []
        fallback_used = False
        fallback_reason: str | None = None
        public_data_transport: dict[str, object] = {}
        if primary_equity_tickers:
            equity_frame, equity_metadata = self._fetch_equity_ohlcv_with_fallback(
                OHLCVRequest(
                    tickers=primary_equity_tickers,
                    start_date=request.start_date,
                    end_date=request.end_date,
                ),
                as_of_timestamp=as_of_timestamp,
            )
            frames.append(equity_frame)
            used_sources.extend(cast(list[str], equity_metadata.get("used_sources", [equity_metadata["used_source"]])))
            attempt_errors.extend(cast(list[dict[str, str]], equity_metadata.get("attempt_errors", [])))
            fallback_used = bool(equity_metadata.get("fallback_used", False))
            fallback_reason = cast(str | None, equity_metadata.get("fallback_reason"))
            public_data_transport["equity"] = cast(dict[str, object], equity_metadata.get("public_data_transport", {}))
        if jp_equity_tickers:
            jp_equity_frame, jp_equity_metadata = self._fetch_jp_equity_ohlcv(
                OHLCVRequest(
                    tickers=jp_equity_tickers,
                    start_date=request.start_date,
                    end_date=request.end_date,
                ),
                as_of_timestamp=as_of_timestamp,
            )
            frames.append(jp_equity_frame)
            used_sources.extend(
                cast(list[str], jp_equity_metadata.get("used_sources", [jp_equity_metadata["used_source"]]))
            )
            public_data_transport["jp_equity"] = cast(
                dict[str, object],
                jp_equity_metadata.get("public_data_transport", {}),
            )
        if crypto_tickers:
            adapter = self._build_ohlcv_adapter(self.settings.data.crypto_source)
            try:
                crypto_frame = adapter.fetch(
                    OHLCVRequest(
                        tickers=crypto_tickers,
                        start_date=request.start_date,
                        end_date=request.end_date,
                    )
                )
            except (MissingCredentialsError, RuntimeError, ValueError, OSError) as exc:
                logger.warning("Crypto OHLCV fetch failed using source '%s': %s", self.settings.data.crypto_source, exc)
                attempt_errors.append({"source": self.settings.data.crypto_source, "error": str(exc)})
                error_summary = "; ".join(f"{item['source']}: {item['error']}" for item in attempt_errors)
                raise RuntimeError(f"OHLCV fetch failed for configured sources. {error_summary}") from exc
            frames.append(crypto_frame)
            used_sources.append(self.settings.data.crypto_source)
            public_data_transport["crypto"] = cast(dict[str, object], getattr(adapter, "last_fetch_metadata", {}))
        if not frames:
            raise RuntimeError("No OHLCV tickers were provided.")
        used_sources = list(dict.fromkeys(used_sources))
        requested_sources = [self.settings.data.primary_source]
        if jp_equity_tickers and self.settings.data.jp_equity.source not in requested_sources:
            requested_sources.append(self.settings.data.jp_equity.source)
        if crypto_tickers and self.settings.data.crypto_source not in requested_sources:
            requested_sources.append(self.settings.data.crypto_source)
        if len(public_data_transport) == 1:
            combined_transport = next(iter(public_data_transport.values()))
        else:
            combined_transport = public_data_transport
        return (
            pd.concat(frames, ignore_index=True),
            {
                "requested_source": self.settings.data.primary_source,
                "used_source": _used_source_label(requested_sources, used_sources),
                "used_sources": used_sources,
                "dummy_mode": None,
                "fallback_used": fallback_used,
                "fallback_reason": fallback_reason,
                "attempt_errors": attempt_errors,
                "as_of_time": as_of_timestamp.isoformat(),
                "public_data_transport": combined_transport,
            },
        )

    def _fetch_fundamentals_with_fallback(
        self,
        request: FundamentalsRequest,
        *,
        as_of_timestamp: pd.Timestamp,
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        sources = [self.settings.data.fundamentals_source]
        if (
            self.settings.data.fundamentals_fallback_source
            and self.settings.data.fundamentals_fallback_source not in sources
        ):
            sources.append(self.settings.data.fundamentals_fallback_source)
        attempt_errors: list[dict[str, str]] = []
        first_error: str | None = None
        for index, source_name in enumerate(sources):
            try:
                adapter = self._build_fundamentals_adapter(source_name)
                frame = adapter.fetch(request)
                if frame.empty:
                    raise RuntimeError(f"{source_name} returned no fundamentals rows.")
                if index > 0:
                    logger.warning(
                        "Primary fundamentals source failed; fallback source '%s' succeeded. reason=%s",
                        source_name,
                        first_error,
                    )
                metadata = {
                    "requested_source": self.settings.data.fundamentals_source,
                    "used_source": source_name,
                    "fallback_used": index > 0,
                    "fallback_reason": first_error,
                    "attempt_errors": attempt_errors,
                    "as_of_time": as_of_timestamp.isoformat(),
                    "transport": cast(dict[str, object], getattr(adapter, "last_fetch_metadata", {})),
                }
                return frame, metadata
            except (MissingCredentialsError, RuntimeError, ValueError, OSError) as exc:
                logger.warning("Fundamentals fetch failed using source '%s': %s", source_name, exc)
                attempt_errors.append({"source": source_name, "error": str(exc)})
                if first_error is None:
                    first_error = str(exc)
        error_summary = "; ".join(f"{item['source']}: {item['error']}" for item in attempt_errors)
        raise RuntimeError(f"Fundamentals fetch failed for all configured sources. {error_summary}")
