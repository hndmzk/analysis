from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from market_prediction_agent.agents.data_agent import DataAgent
from market_prediction_agent.config import Settings
from market_prediction_agent.data.adapters import FundamentalsRequest
from market_prediction_agent.data.normalizer import normalize_fundamentals

from .base import MCPTool, clone_settings, dataframe_response, temporary_store


SEC_FILING_READER_INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ticker": {"type": "string", "minLength": 1},
        "filing_type": {"type": "string", "enum": ["10-K", "10-Q"]},
        "limit": {"type": "integer", "minimum": 1, "default": 5},
    },
    "required": ["ticker", "filing_type"],
    "additionalProperties": False,
}


def _infer_filing_types(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        updated = frame.copy()
        updated["filing_type"] = pd.Series(dtype="object")
        return updated

    updated = frame.sort_values(["ticker", "report_date"]).reset_index(drop=True).copy()
    report_dates = pd.to_datetime(updated["report_date"], utc=True)
    prev_gap = report_dates.groupby(updated["ticker"]).diff().dt.days.abs()
    next_gap = report_dates.groupby(updated["ticker"]).shift(-1).sub(report_dates).dt.days.abs()
    cadence_days = prev_gap.fillna(next_gap).fillna(91.0)
    updated["filing_type"] = np.where(cadence_days >= 250.0, "10-K", "10-Q")
    return updated


def handle_sec_filing_reader(params: dict[str, Any], settings: Settings) -> dict[str, Any]:
    ticker = str(params["ticker"]).upper()
    filing_type = str(params["filing_type"])
    limit = int(params.get("limit", 5))
    end_date = pd.Timestamp.now(tz="UTC").normalize()
    start_date = (end_date - pd.Timedelta(days=3650)).date().isoformat()

    tool_settings = clone_settings(settings)
    with temporary_store() as store:
        agent = DataAgent(tool_settings, store)
        if tool_settings.data.source_mode == "dummy":
            adapter = agent._build_fundamentals_adapter("offline_fundamental_proxy")
            used_source = "offline_fundamental_proxy"
        else:
            adapter = agent._build_fundamentals_adapter("sec_companyfacts")
            used_source = "sec_companyfacts"
        frame = normalize_fundamentals(
            adapter.fetch(
                FundamentalsRequest(
                    tickers=[ticker],
                    start_date=start_date,
                    end_date=end_date.date().isoformat(),
                )
            )
        )

    enriched = _infer_filing_types(frame)
    filtered = (
        enriched.loc[enriched["filing_type"] == filing_type]
        .sort_values("report_date", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )
    return dataframe_response(
        filtered,
        metadata={
            "ticker": ticker,
            "filing_type": filing_type,
            "limit": limit,
            "used_source": used_source,
            "transport": cast(dict[str, Any], getattr(adapter, "last_fetch_metadata", {})),
        },
    )


SEC_FILING_READER = MCPTool(
    name="sec_filing_reader",
    description="Fetch normalized SEC CompanyFacts fundamentals and filter them to recent 10-K or 10-Q style filings.",
    input_schema=SEC_FILING_READER_INPUT_SCHEMA,
    handler=handle_sec_filing_reader,
)
