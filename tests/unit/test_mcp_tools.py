from __future__ import annotations

from jsonschema import Draft202012Validator

from market_prediction_agent.config import load_settings, update_settings
from market_prediction_agent.mcp import MCP_TOOLS
from market_prediction_agent.schemas.validator import validate_payload


FAST_FEATURE_COLUMNS = [
    "log_return_1d",
    "log_return_5d",
    "log_return_20d",
    "realized_vol_5d",
    "garman_klass_vol",
    "bb_width",
    "atr_ratio",
    "volume_ratio_20d",
    "price_vs_sma_20d",
    "price_vs_sma_50d",
    "fed_funds_rate",
    "yield_curve_slope",
    "vix",
    "vix_change_5d",
    "news_sentiment_5d",
    "news_relevance_5d",
    "fundamental_revenue_growth",
    "fundamental_profitability",
    "sector_relative_momentum_20d",
    "sector_strength_20d",
    "day_of_week",
    "month",
    "is_month_end",
]


def _mcp_settings(monkeypatch) -> object:
    monkeypatch.setattr("market_prediction_agent.features.pipeline.FEATURE_COLUMNS", FAST_FEATURE_COLUMNS)
    monkeypatch.setattr("market_prediction_agent.agents.forecast_agent.FEATURE_COLUMNS", FAST_FEATURE_COLUMNS)
    return update_settings(
        load_settings("config/default.yaml"),
        data={
            "source_mode": "dummy",
            "dummy_mode": "predictable_momentum",
            "crypto_enabled": False,
            "dummy_ticker_count": 4,
            "dummy_days": 180,
        },
        model_settings={
            "walk_forward": {
                "initial_train_days": 80,
                "eval_days": 20,
                "step_days": 20,
                "embargo_days": 0,
            },
            "cpcv": {
                "group_count": 4,
                "test_groups": 1,
                "max_splits": 2,
            },
            "lightgbm": {
                "n_estimators": 20,
                "num_leaves": 7,
                "min_child_samples": 5,
                "max_shap_samples": 25,
            },
            "calibration": {
                "min_days": 10,
                "fraction": 0.1,
            },
            "comparison_models": [],
        },
    )


def test_mcp_tools_registry_contains_expected_read_only_tools() -> None:
    assert set(MCP_TOOLS) == {
        "market_data_fetcher",
        "sec_filing_reader",
        "macro_data_fetcher",
        "forecast_runner",
        "backtest_runner",
    }
    assert all(tool.read_only for tool in MCP_TOOLS.values())


def test_mcp_tool_input_schemas_are_valid_json_schema() -> None:
    for tool in MCP_TOOLS.values():
        Draft202012Validator.check_schema(tool.input_schema)


def test_market_data_fetcher_handler_returns_dummy_ohlcv(monkeypatch) -> None:
    settings = _mcp_settings(monkeypatch)
    payload = MCP_TOOLS["market_data_fetcher"].handler(
        {
            "ticker": "AAA",
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
        },
        settings,
    )
    assert isinstance(payload, dict)
    assert payload["row_count"] > 0
    assert {"ticker", "timestamp_utc", "close", "source"}.issubset(payload["columns"])
    assert payload["metadata"]["used_source"] == "dummy"


def test_sec_filing_reader_handler_returns_dummy_fundamentals(monkeypatch) -> None:
    settings = _mcp_settings(monkeypatch)
    payload = MCP_TOOLS["sec_filing_reader"].handler(
        {
            "ticker": "AAA",
            "filing_type": "10-Q",
            "limit": 2,
        },
        settings,
    )
    assert isinstance(payload, dict)
    assert payload["row_count"] > 0
    assert "filing_type" in payload["columns"]
    assert {item["filing_type"] for item in payload["data"]} == {"10-Q"}


def test_macro_data_fetcher_handler_returns_dummy_macro_series(monkeypatch) -> None:
    settings = _mcp_settings(monkeypatch)
    payload = MCP_TOOLS["macro_data_fetcher"].handler(
        {
            "series_id": "FEDFUNDS",
            "start_date": "2024-01-01",
            "end_date": "2024-03-31",
        },
        settings,
    )
    assert isinstance(payload, dict)
    assert payload["row_count"] > 0
    assert payload["metadata"]["used_source"] == "dummy"
    assert {item["series_id"] for item in payload["data"]} == {"FEDFUNDS"}


def test_forecast_runner_handler_returns_schema_compliant_payload(monkeypatch) -> None:
    settings = _mcp_settings(monkeypatch)
    payload = MCP_TOOLS["forecast_runner"].handler(
        {
            "tickers": ["AAA", "BBB"],
            "as_of_date": "2024-09-30",
        },
        settings,
    )
    validate_payload("forecast_output", payload)
    assert len(payload["predictions"]) == 2


def test_backtest_runner_handler_returns_schema_compliant_payload(monkeypatch) -> None:
    settings = _mcp_settings(monkeypatch)
    payload = MCP_TOOLS["backtest_runner"].handler(
        {
            "model_name": settings.model_settings.primary,
            "start_date": "2024-01-01",
            "end_date": "2024-09-30",
        },
        settings,
    )
    validate_payload("backtest_result", payload)
    assert payload["config"]["model_name"] == settings.model_settings.primary
    assert payload["folds"]

