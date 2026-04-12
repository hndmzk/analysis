from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4

import pandas as pd

from market_prediction_agent.agents.data_agent import DataAgent
from market_prediction_agent.config import load_settings, update_settings
from market_prediction_agent.data.adapters import FundamentalsRequest, OfflineFundamentalProxyAdapter
from market_prediction_agent.storage.parquet_store import ParquetStore
from market_prediction_agent.utils.paths import resolve_repo_path


class _FailingFundamentalsAdapter:
    name = "sec_companyfacts"

    def fetch(self, request: FundamentalsRequest) -> pd.DataFrame:
        raise RuntimeError("companyfacts unavailable")


def _workspace_temp_dir() -> Path:
    path = resolve_repo_path(Path(".test-artifacts") / "test-data-agent-fundamentals" / uuid4().hex)
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_data_agent_fundamentals_fall_back_to_proxy_with_metadata() -> None:
    settings = update_settings(
        load_settings("config/default.yaml"),
        data={
            "source_mode": "live",
            "fundamentals_source": "sec_companyfacts",
            "fundamentals_fallback_source": "offline_fundamental_proxy",
        },
    )
    temp_dir = _workspace_temp_dir()
    try:
        agent = DataAgent(settings, ParquetStore(temp_dir))

        def fake_build_adapter(source_name: str | None = None):
            if source_name == "sec_companyfacts":
                return _FailingFundamentalsAdapter()
            if source_name == "offline_fundamental_proxy":
                return OfflineFundamentalProxyAdapter(seed=42)
            raise AssertionError(source_name)

        agent._build_fundamentals_adapter = fake_build_adapter  # type: ignore[assignment]
        frame, metadata = agent._fetch_fundamentals_with_fallback(
            FundamentalsRequest(tickers=["AAPL", "MSFT"], start_date="2024-01-01", end_date="2025-12-31"),
            as_of_timestamp=pd.Timestamp("2026-04-04T00:00:00Z"),
        )
        assert not frame.empty
        assert metadata["requested_source"] == "sec_companyfacts"
        assert metadata["used_source"] == "offline_fundamental_proxy"
        assert metadata["fallback_used"] is True
        assert "companyfacts unavailable" in str(metadata["fallback_reason"])
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
