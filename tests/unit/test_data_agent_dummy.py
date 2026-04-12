from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4

from market_prediction_agent.agents.data_agent import DataAgent
from market_prediction_agent.config import load_settings, update_settings
from market_prediction_agent.data.normalizer import FUNDAMENTAL_COLUMNS, MACRO_COLUMNS, NEWS_COLUMNS, OHLCV_COLUMNS, SECTOR_MAP_COLUMNS
from market_prediction_agent.storage.parquet_store import ParquetStore
from market_prediction_agent.utils.paths import resolve_repo_path


def _workspace_temp_dir() -> Path:
    path = resolve_repo_path(Path(".test-artifacts") / "test-data-agent-dummy" / uuid4().hex)
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _generate_dummy_artifacts():
    temp_dir = _workspace_temp_dir()
    settings = update_settings(
        load_settings("config/default.yaml"),
        data={
            "source_mode": "dummy",
            "crypto_enabled": False,
        },
    )
    agent = DataAgent(settings, ParquetStore(temp_dir))
    artifacts = agent.generate_or_fetch(
        tickers=["AAA", "BBB", "CCC"],
        start_date="2024-01-01",
        end_date="2024-03-29",
        as_of_time="2024-04-01T00:00:00Z",
    )
    return settings, artifacts, temp_dir


def test_data_agent_dummy_mode_generates_ohlcv_data() -> None:
    settings, artifacts, temp_dir = _generate_dummy_artifacts()
    del settings
    try:
        assert not artifacts.raw_ohlcv.empty
        assert artifacts.raw_ohlcv["source"].unique().tolist() == ["dummy"]
        assert artifacts.processed_ohlcv.columns.tolist() == OHLCV_COLUMNS
        assert artifacts.ohlcv_metadata["used_source"] == "dummy"
        assert artifacts.ohlcv_metadata["used_sources"] == ["dummy"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_data_agent_dummy_mode_generates_macro_data() -> None:
    settings, artifacts, temp_dir = _generate_dummy_artifacts()
    try:
        assert artifacts.processed_macro.columns.tolist() == MACRO_COLUMNS
        assert sorted(artifacts.processed_macro["series_id"].unique().tolist()) == sorted(settings.data.macro_series)
        assert artifacts.ohlcv_metadata["macro_source"] == "dummy"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_data_agent_static_sector_map_is_normalized() -> None:
    settings, artifacts, temp_dir = _generate_dummy_artifacts()
    del settings
    try:
        assert artifacts.processed_sector_map.columns.tolist() == SECTOR_MAP_COLUMNS
        assert artifacts.processed_sector_map["ticker"].tolist() == ["AAA", "BBB", "CCC"]
        assert artifacts.processed_sector_map["source"].unique().tolist() == ["static_sector_map"]
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_data_agent_generate_or_fetch_records_normalized_feature_sources() -> None:
    settings, artifacts, temp_dir = _generate_dummy_artifacts()
    del settings
    try:
        feature_sources = artifacts.ohlcv_metadata["feature_sources"]
        assert artifacts.processed_news.columns.tolist() == NEWS_COLUMNS
        assert artifacts.processed_fundamentals.columns.tolist() == FUNDAMENTAL_COLUMNS
        assert feature_sources["news"]["used_source"] == "offline_news_proxy"
        assert feature_sources["fundamental"]["used_source"] == "offline_fundamental_proxy"
        assert feature_sources["sector"]["used_source"] == "static_sector_map"
        assert feature_sources["news"]["record_count"] == len(artifacts.processed_news)
        assert feature_sources["fundamental"]["record_count"] == len(artifacts.processed_fundamentals)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
