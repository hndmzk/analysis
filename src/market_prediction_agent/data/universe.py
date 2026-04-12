from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import pandas as pd

from market_prediction_agent.config import Settings
from market_prediction_agent.utils.paths import resolve_repo_path
from market_prediction_agent.utils.time_utils import to_utc_timestamp


DEFAULT_UNIVERSE_HISTORY_PATH = "config/universe_history.json"


def _normalize_tickers(tickers: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for ticker in tickers:
        label = str(ticker).strip().upper()
        if label and label not in seen:
            seen.add(label)
            ordered.append(label)
    return ordered


def _load_json(path: str | Path) -> object:
    return json.loads(resolve_repo_path(path).read_text(encoding="utf-8"))


@dataclass(slots=True)
class UniverseChange:
    date: pd.Timestamp
    added: list[str]
    removed: list[str]


@dataclass(slots=True)
class PointInTimeUniverse:
    base_date: pd.Timestamp
    base_constituents: list[str]
    changes: list[UniverseChange] = field(default_factory=list)

    def get_constituents(self, as_of_date: str | pd.Timestamp) -> list[str]:
        as_of_timestamp = to_utc_timestamp(as_of_date).normalize()
        constituents = list(_normalize_tickers(self.base_constituents))
        active = set(constituents)
        for change in sorted(self.changes, key=lambda item: item.date):
            if change.date > as_of_timestamp:
                break
            for ticker in change.removed:
                if ticker in active:
                    active.remove(ticker)
                    constituents = [value for value in constituents if value != ticker]
            for ticker in change.added:
                if ticker not in active:
                    active.add(ticker)
                    constituents.append(ticker)
        return constituents

    def add_change(
        self,
        date: str | pd.Timestamp,
        *,
        added: list[str],
        removed: list[str],
    ) -> None:
        self.changes.append(
            UniverseChange(
                date=to_utc_timestamp(date).normalize(),
                added=_normalize_tickers(added),
                removed=_normalize_tickers(removed),
            )
        )
        self.changes.sort(key=lambda item: item.date)

    def all_tickers(self) -> list[str]:
        tickers = list(_normalize_tickers(self.base_constituents))
        seen = set(tickers)
        for change in sorted(self.changes, key=lambda item: item.date):
            for ticker in [*change.removed, *change.added]:
                if ticker not in seen:
                    seen.add(ticker)
                    tickers.append(ticker)
        return tickers

    @classmethod
    def from_static(cls, tickers: list[str]) -> PointInTimeUniverse:
        return cls(
            base_date=to_utc_timestamp("1970-01-01").normalize(),
            base_constituents=_normalize_tickers(tickers),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> PointInTimeUniverse:
        payload = _load_json(path)
        if not isinstance(payload, dict):
            raise ValueError("Universe history payload must be a JSON object.")
        base_constituents = payload.get("base_constituents")
        if not isinstance(base_constituents, list):
            raise ValueError("Universe history payload must include a base_constituents array.")
        universe = cls(
            base_date=to_utc_timestamp(str(payload["base_date"])).normalize(),
            base_constituents=_normalize_tickers([str(item) for item in base_constituents]),
        )
        changes = payload.get("changes", [])
        if not isinstance(changes, list):
            raise ValueError("Universe history changes must be a JSON array.")
        for change in changes:
            if not isinstance(change, dict):
                raise ValueError("Universe history changes must be JSON objects.")
            universe.add_change(
                str(change["date"]),
                added=[str(item) for item in change.get("added", [])],
                removed=[str(item) for item in change.get("removed", [])],
            )
        return universe


def load_ticker_list(path: str | Path) -> list[str]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError("Ticker list payload must be a JSON array.")
    return _normalize_tickers([str(item) for item in payload])


def load_point_in_time_universe(path: str | Path = DEFAULT_UNIVERSE_HISTORY_PATH) -> PointInTimeUniverse:
    return PointInTimeUniverse.from_json(path)


def resolve_default_tickers(settings: Settings) -> list[str]:
    if settings.data.source_mode != "dummy":
        if settings.data.universe == "sp500_pit":
            tickers = load_point_in_time_universe().all_tickers()
        elif settings.data.default_tickers:
            tickers = list(settings.data.default_tickers)
        else:
            tickers = [f"TICK{i:03d}" for i in range(settings.data.dummy_ticker_count)]
    else:
        tickers = [f"TICK{i:03d}" for i in range(settings.data.dummy_ticker_count)]

    if settings.data.jp_equity.enabled:
        tickers.extend(load_ticker_list(settings.data.jp_equity.tickers_file))
    if settings.data.crypto_enabled:
        tickers.extend(settings.data.crypto_tickers)
    return _normalize_tickers(tickers)


def resolve_active_constituents(
    settings: Settings,
    *,
    as_of_date: str | pd.Timestamp,
) -> list[str] | None:
    if settings.data.universe != "sp500_pit":
        return None
    tickers = load_point_in_time_universe().get_constituents(as_of_date)
    if settings.data.jp_equity.enabled:
        tickers.extend(load_ticker_list(settings.data.jp_equity.tickers_file))
    return _normalize_tickers(tickers)
