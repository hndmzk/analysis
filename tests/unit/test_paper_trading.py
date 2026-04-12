from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4

import pandas as pd

from market_prediction_agent.config import load_settings, update_settings
from market_prediction_agent.paper_trading.service import LEDGER_PATH, PaperTradingService
from market_prediction_agent.retraining.ledger_service import EVENT_LEDGER_RELATIVE_PATH
from market_prediction_agent.storage.parquet_store import ParquetStore


def _base_forecast_output(generated_at: str, forecast_id: str) -> dict[str, object]:
    return {
        "forecast_id": forecast_id,
        "generated_at": generated_at,
        "model_version": "v0.1.0",
        "horizon": "1d",
        "regime": "low_vol",
        "predictions": [
            {
                "ticker": "AAA",
                "direction": "UP",
                "probabilities": {"UP": 0.6, "DOWN": 0.2, "FLAT": 0.2},
                "expected_return": 0.01,
                "predicted_volatility": 0.2,
                "confidence": "high",
                "top_features": [],
                "stale_data_flag": False,
            }
        ],
    }


def _risk_review(should_retrain: bool) -> dict[str, object]:
    return {
        "approval": "MANUAL_REVIEW_REQUIRED",
        "retraining_monitor": {
            "should_retrain": should_retrain,
            "trigger_count": 1 if should_retrain else 0,
            "triggers": [{"name": "pbo", "status": "WARNING", "detail": "x"}] if should_retrain else [],
        },
    }


def _backtest_result() -> dict[str, object]:
    return {
        "aggregate_metrics": {"ece_mean": 0.01},
        "cpcv": {"pbo_summary": {"label": "low_overfit_risk"}},
        "pbo": 0.1,
    }


def _market_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _make_service() -> tuple[PaperTradingService, ParquetStore]:
    artifact_root = Path(".test-artifacts") / uuid4().hex
    shutil.rmtree(artifact_root, ignore_errors=True)
    settings = update_settings(
        load_settings("config/default.yaml"),
        data={"storage_path": str(artifact_root / "storage")},
    )
    store = ParquetStore(Path(settings.data.storage_path))
    return PaperTradingService(settings, store), store


def test_paper_trading_service_queues_executes_and_settles_trade() -> None:
    service, store = _make_service()

    day1 = service.update(
        forecast_output=_base_forecast_output("2026-01-05T00:00:00Z", str(uuid4())),
        backtest_result=_backtest_result(),
        risk_review=_risk_review(False),
        source_metadata={"used_source": "dummy", "dummy_mode": "null_random_walk"},
        feature_frame=_market_frame(
            [
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-01-05", tz="UTC"),
                    "open": 99.0,
                    "close": 100.0,
                    "volume": 10_000_000.0,
                }
            ]
        ),
    )
    queued_trade = day1.batch_log["trades"][0]
    assert queued_trade["status"] == "QUEUED"
    assert queued_trade["signal_time"] is not None
    assert queued_trade["execution_time"] is not None
    assert queued_trade["intended_order_quantity"] > 0.0
    assert queued_trade["filled_quantity"] == 0.0

    day2 = service.update(
        forecast_output=_base_forecast_output("2026-01-06T00:00:00Z", str(uuid4())),
        backtest_result=_backtest_result(),
        risk_review=_risk_review(False),
        source_metadata={"used_source": "dummy", "dummy_mode": "null_random_walk"},
        feature_frame=_market_frame(
            [
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-01-05", tz="UTC"),
                    "open": 99.0,
                    "close": 100.0,
                    "volume": 10_000_000.0,
                },
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-01-06", tz="UTC"),
                    "open": 101.0,
                    "close": 103.0,
                    "volume": 10_000_000.0,
                },
            ]
        ),
    )
    ledger_after_execution = store.read_frame(LEDGER_PATH)
    executed = ledger_after_execution.loc[ledger_after_execution["forecast_date"] == pd.Timestamp("2026-01-05", tz="UTC")]
    assert not executed.empty
    executed_row = executed.iloc[0]
    assert executed_row["status"] == "PENDING"
    assert float(executed_row["execution_price"]) == 101.0
    assert float(executed_row["fill_rate"]) == 1.0
    assert bool(executed_row["partial_fill"]) is False
    assert bool(executed_row["missed_trade"]) is False
    assert float(executed_row["gap_slippage_bps"]) > 0.0
    assert day2.batch_log["execution_diagnostics"]["fill_rate"] == 1.0
    assert day2.weekly_review["execution_diagnostics"]["realized_vs_intended_exposure"] > 0.0

    day3 = service.update(
        forecast_output=_base_forecast_output("2026-01-07T00:00:00Z", str(uuid4())),
        backtest_result=_backtest_result(),
        risk_review=_risk_review(True),
        source_metadata={"used_source": "dummy", "dummy_mode": "null_random_walk"},
        feature_frame=_market_frame(
            [
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-01-05", tz="UTC"),
                    "open": 99.0,
                    "close": 100.0,
                    "volume": 10_000_000.0,
                },
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-01-06", tz="UTC"),
                    "open": 101.0,
                    "close": 103.0,
                    "volume": 10_000_000.0,
                },
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-01-07", tz="UTC"),
                    "open": 102.0,
                    "close": 104.0,
                    "volume": 10_000_000.0,
                },
            ]
        ),
    )
    persisted = store.read_frame(LEDGER_PATH)
    settled = persisted.loc[
        (persisted["forecast_date"] == pd.Timestamp("2026-01-05", tz="UTC")) & (persisted["status"] == "SETTLED")
    ]
    assert not settled.empty
    settled_row = settled.iloc[0]
    assert bool(settled_row["hit"]) is True
    assert round(float(settled_row["gross_trade_return"]), 4) == round((103.0 / 101.0) - 1.0, 4)
    assert day3.weekly_review["execution_diagnostics"]["fill_rate"] > 0.0
    assert day3.retraining_event is not None
    event_ledger = store.read_frame(EVENT_LEDGER_RELATIVE_PATH)
    assert not event_ledger.empty
    assert bool(event_ledger.iloc[-1]["should_retrain"]) is True


def test_paper_trading_records_partial_fill_and_gap_slippage() -> None:
    artifact_root = Path(".test-artifacts") / uuid4().hex
    shutil.rmtree(artifact_root, ignore_errors=True)
    settings = update_settings(
        load_settings("config/default.yaml"),
        data={"storage_path": str(artifact_root / "storage")},
        trading={"max_participation_rate": 0.005},
    )
    store = ParquetStore(Path(settings.data.storage_path))
    service = PaperTradingService(settings, store)

    service.update(
        forecast_output=_base_forecast_output("2026-01-05T00:00:00Z", str(uuid4())),
        backtest_result=_backtest_result(),
        risk_review=_risk_review(False),
        source_metadata={"used_source": "dummy", "dummy_mode": "predictable_momentum"},
        feature_frame=_market_frame(
            [
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-01-05", tz="UTC"),
                    "open": 100.0,
                    "close": 100.0,
                    "volume": 50_000.0,
                }
            ]
        ),
    )

    day2 = service.update(
        forecast_output=_base_forecast_output("2026-01-06T00:00:00Z", str(uuid4())),
        backtest_result=_backtest_result(),
        risk_review=_risk_review(False),
        source_metadata={"used_source": "dummy", "dummy_mode": "predictable_momentum"},
        feature_frame=_market_frame(
            [
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-01-05", tz="UTC"),
                    "open": 100.0,
                    "close": 100.0,
                    "volume": 50_000.0,
                },
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-01-06", tz="UTC"),
                    "open": 102.0,
                    "close": 101.0,
                    "volume": 50_000.0,
                },
            ]
        ),
    )
    persisted = store.read_frame(LEDGER_PATH)
    partial = persisted.loc[persisted["forecast_date"] == pd.Timestamp("2026-01-05", tz="UTC")].iloc[0]
    assert partial["status"] == "PENDING"
    assert bool(partial["partial_fill"]) is True
    assert 0.0 < float(partial["fill_rate"]) < 1.0
    assert float(partial["filled_quantity"]) < float(partial["intended_order_quantity"])
    assert float(partial["unfilled_quantity"]) > 0.0
    assert float(partial["gap_slippage_bps"]) > 0.0
    assert float(partial["execution_cost_drag"]) > 0.0
    assert day2.batch_log["execution_diagnostics"]["partial_fill_rate"] == 1.0
    assert day2.batch_log["execution_diagnostics"]["realized_vs_intended_exposure"] < 1.0


def test_paper_trading_skips_trade_when_execution_liquidity_is_insufficient() -> None:
    service, store = _make_service()

    service.update(
        forecast_output=_base_forecast_output("2026-01-05T00:00:00Z", str(uuid4())),
        backtest_result=_backtest_result(),
        risk_review=_risk_review(False),
        source_metadata={"used_source": "yahoo_chart", "dummy_mode": None},
        feature_frame=_market_frame(
            [
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-01-05", tz="UTC"),
                    "open": 10.0,
                    "close": 10.0,
                    "volume": 10_000.0,
                }
            ]
        ),
    )

    day2 = service.update(
        forecast_output=_base_forecast_output("2026-01-06T00:00:00Z", str(uuid4())),
        backtest_result=_backtest_result(),
        risk_review=_risk_review(False),
        source_metadata={"used_source": "yahoo_chart", "dummy_mode": None},
        feature_frame=_market_frame(
            [
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-01-05", tz="UTC"),
                    "open": 10.0,
                    "close": 10.0,
                    "volume": 10_000.0,
                },
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-01-06", tz="UTC"),
                    "open": 10.0,
                    "close": 10.0,
                    "volume": 5_000.0,
                },
            ]
        ),
    )
    persisted = store.read_frame(LEDGER_PATH)
    missed = persisted.loc[persisted["forecast_date"] == pd.Timestamp("2026-01-05", tz="UTC")].iloc[0]
    assert missed["status"] == "SKIPPED"
    assert missed["execution_reason"] == "insufficient_liquidity"
    assert bool(missed["missed_trade"]) is True
    assert float(missed["filled_quantity"]) == 0.0
    assert float(missed["unfilled_quantity"]) == float(missed["intended_order_quantity"])
    assert day2.batch_log["execution_diagnostics"]["missed_trade_rate"] == 1.0
    assert day2.weekly_review["execution_diagnostics"]["missed_trade_rate"] >= 1.0 / 1.0


def test_paper_trading_reinitializes_corrupt_ledger() -> None:
    service, store = _make_service()
    corrupt_path = store.resolve(LEDGER_PATH)
    corrupt_path.parent.mkdir(parents=True, exist_ok=True)
    corrupt_path.write_bytes(b"not-a-valid-parquet")

    day1 = service.update(
        forecast_output=_base_forecast_output("2026-01-05T00:00:00Z", str(uuid4())),
        backtest_result=_backtest_result(),
        risk_review=_risk_review(False),
        source_metadata={"used_source": "dummy", "dummy_mode": "null_random_walk"},
        feature_frame=_market_frame(
            [
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-01-05", tz="UTC"),
                    "open": 99.0,
                    "close": 100.0,
                    "volume": 10_000_000.0,
                }
            ]
        ),
    )

    persisted = store.read_frame(LEDGER_PATH)
    assert len(persisted) == 1
    assert day1.batch_log["metrics"]["new_trades"] == 1


def test_paper_trading_can_use_detailed_execution_model_when_enabled() -> None:
    artifact_root = Path(".test-artifacts") / uuid4().hex
    shutil.rmtree(artifact_root, ignore_errors=True)
    settings = update_settings(
        load_settings("config/default.yaml"),
        data={"storage_path": str(artifact_root / "storage")},
        execution={"enabled": True},
    )
    store = ParquetStore(Path(settings.data.storage_path))
    service = PaperTradingService(settings, store)

    service.update(
        forecast_output=_base_forecast_output("2026-01-05T00:00:00Z", str(uuid4())),
        backtest_result=_backtest_result(),
        risk_review=_risk_review(False),
        source_metadata={"used_source": "dummy", "dummy_mode": "null_random_walk"},
        feature_frame=_market_frame(
            [
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-01-05", tz="UTC"),
                    "open": 99.0,
                    "close": 100.0,
                    "volume": 10_000_000.0,
                }
            ]
        ),
    )
    service.update(
        forecast_output=_base_forecast_output("2026-01-06T00:00:00Z", str(uuid4())),
        backtest_result=_backtest_result(),
        risk_review=_risk_review(False),
        source_metadata={"used_source": "dummy", "dummy_mode": "null_random_walk"},
        feature_frame=_market_frame(
            [
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-01-05", tz="UTC"),
                    "open": 99.0,
                    "close": 100.0,
                    "volume": 10_000_000.0,
                },
                {
                    "ticker": "AAA",
                    "date": pd.Timestamp("2026-01-06", tz="UTC"),
                    "open": 101.0,
                    "close": 103.0,
                    "volume": 10_000_000.0,
                },
            ]
        ),
    )

    persisted = store.read_frame(LEDGER_PATH)
    executed = persisted.loc[persisted["forecast_date"] == pd.Timestamp("2026-01-05", tz="UTC")].iloc[0]
    assert executed["status"] == "PENDING"
    assert float(executed["execution_price"]) > 101.0
    assert float(executed["round_trip_slippage_bps"]) > 10.0
