from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any, cast
from uuid import uuid4

import pandas as pd

from market_prediction_agent.config import Settings
from market_prediction_agent.data.universe import resolve_default_tickers
from market_prediction_agent.storage.parquet_store import ParquetStore
from market_prediction_agent.utils.paths import resolve_repo_path
from market_prediction_agent.utils.time_utils import to_utc_timestamp


MCPHandler = Callable[[dict[str, Any], Settings], dict[str, Any]]


@dataclass(slots=True)
class MCPTool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: MCPHandler
    read_only: bool = True


def clone_settings(settings: Settings) -> Settings:
    return settings.model_copy(deep=True)


def dataframe_response(
    frame: pd.DataFrame,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "columns": list(frame.columns),
        "row_count": int(len(frame)),
        "data": cast(list[dict[str, Any]], json.loads(frame.to_json(orient="records", date_format="iso"))),
    }
    if metadata is not None:
        payload["metadata"] = metadata
    return payload


@contextmanager
def temporary_store() -> Iterator[ParquetStore]:
    temp_root = resolve_repo_path(".test-artifacts/mcp-temp")
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_dir = temp_root / f"market-prediction-agent-mcp-{uuid4().hex}"
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    try:
        yield ParquetStore(Path(temp_dir))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def default_runner_tickers(settings: Settings) -> list[str]:
    return resolve_default_tickers(settings)


def history_window_bounds(
    as_of_date: str | pd.Timestamp,
    business_days: int,
) -> tuple[str, str, pd.Timestamp]:
    as_of_timestamp = to_utc_timestamp(as_of_date).normalize()
    start_date = (
        as_of_timestamp - pd.tseries.offsets.BDay(max(business_days - 1, 0))
    ).date().isoformat()
    end_date = as_of_timestamp.date().isoformat()
    return start_date, end_date, as_of_timestamp
