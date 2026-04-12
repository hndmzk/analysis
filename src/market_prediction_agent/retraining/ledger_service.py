from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd

from market_prediction_agent.config import Settings, resolve_storage_path
from market_prediction_agent.storage.parquet_store import ParquetStore


EVENT_LEDGER_RELATIVE_PATH = Path("outputs") / "retraining_events" / "retraining_event_ledger.parquet"
EVENT_LEDGER_COLUMNS = [
    "created_at",
    "as_of_date",
    "ticker_set",
    "source_mode",
    "dummy_mode",
    "current_regime",
    "dominant_recent_regime",
    "regime_bucket",
    "base_should_retrain",
    "should_retrain",
    "trigger_names",
    "base_trigger_names",
    "effective_trigger_names",
    "suppressed_trigger_names",
    "drift_trigger_families",
    "family_regime_keys",
    "base_cause_keys",
    "effective_cause_keys",
    "pbo",
    "pbo_label",
    "policy_decision",
]
LIST_COLUMNS = [
    "trigger_names",
    "base_trigger_names",
    "effective_trigger_names",
    "suppressed_trigger_names",
    "drift_trigger_families",
    "family_regime_keys",
    "base_cause_keys",
    "effective_cause_keys",
]


def _normalized_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        return [str(item) for item in parsed]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _ticker_set_label(tickers: list[str]) -> str:
    return ",".join(sorted({str(ticker).upper() for ticker in tickers}))


def _normalize_ticker_set_label(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        tickers = [item.strip().upper() for item in value.split(",") if item.strip()]
        return _ticker_set_label(tickers) if tickers else ""
    if isinstance(value, (list, tuple, set)):
        return _ticker_set_label([str(item) for item in value])
    return str(value)


@dataclass(slots=True)
class RetrainingEventLedgerService:
    settings: Settings
    store: ParquetStore | None = None

    def __post_init__(self) -> None:
        if self.store is None:
            self.store = ParquetStore(resolve_storage_path(self.settings))

    @property
    def path(self) -> Path:
        assert self.store is not None
        return self.store.resolve(EVENT_LEDGER_RELATIVE_PATH)

    def empty_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=EVENT_LEDGER_COLUMNS)

    def normalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return self.empty_frame()
        normalized = frame.copy()
        for column in EVENT_LEDGER_COLUMNS:
            if column not in normalized.columns:
                normalized[column] = None
        for column in LIST_COLUMNS:
            normalized[column] = normalized[column].apply(
                lambda value: json.dumps(_normalized_list(value), ensure_ascii=False)
            )
        normalized["ticker_set"] = normalized["ticker_set"].apply(_normalize_ticker_set_label)
        normalized["source_mode"] = normalized["source_mode"].fillna("")
        normalized["base_should_retrain"] = normalized["base_should_retrain"].fillna(False).astype(bool)
        normalized["should_retrain"] = normalized["should_retrain"].fillna(False).astype(bool)
        normalized["created_at"] = pd.to_datetime(normalized["created_at"], utc=True, errors="coerce").fillna(
            pd.Timestamp.now(tz="UTC")
        )
        return normalized.loc[:, EVENT_LEDGER_COLUMNS].copy()

    def load_frame(self) -> pd.DataFrame:
        assert self.store is not None
        if not self.path.exists():
            return self.empty_frame()
        frame = self.store.read_frame(EVENT_LEDGER_RELATIVE_PATH)
        if frame.empty:
            return self.empty_frame()
        return self.normalize_frame(frame)

    def append_entry(self, entry: dict[str, object]) -> Path:
        assert self.store is not None
        frame = self.load_frame()
        frame = pd.concat([frame, pd.DataFrame([entry])], ignore_index=True)
        frame = self.normalize_frame(frame)
        return self.store.write_frame(EVENT_LEDGER_RELATIVE_PATH, frame)

    def load_policy_history(
        self,
        *,
        tickers: list[str],
        as_of_date: str,
        limit: int = 128,
        source_mode: str = "live",
    ) -> list[dict[str, object]]:
        frame = self.load_frame()
        if frame.empty:
            return []
        ticker_label = _ticker_set_label(tickers)
        filtered = frame.loc[
            (frame["ticker_set"] == ticker_label)
            & (frame["source_mode"].fillna("") == source_mode)
            & (frame["as_of_date"].fillna("") < as_of_date),
            :,
        ].sort_values("as_of_date")
        if filtered.empty:
            return []
        rows = filtered.tail(limit).to_dict(orient="records")
        history: list[dict[str, object]] = []
        for row in rows:
            decoded = dict(row)
            for column in LIST_COLUMNS:
                decoded[column] = _normalized_list(decoded.get(column))
            history.append(decoded)
        return history
