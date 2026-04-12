from __future__ import annotations

import pandas as pd
import pytest

from market_prediction_agent.backtest.walk_forward import (
    build_walk_forward_windows,
    run_model_comparisons,
    run_walk_forward_backtest,
)
from market_prediction_agent.config import load_settings, update_settings


def _make_training_frame(*, ticker_count: int, day_count: int) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=day_count, tz="UTC")
    rows: list[dict[str, object]] = []
    tickers = [f"TICK{i:02d}" for i in range(ticker_count)]
    for day_index, date in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            base = float(day_index + ticker_index + 1)
            direction_label = (day_index + ticker_index) % 3
            centered_direction = direction_label - 1
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "stale_data_flag": False,
                    "feature_a": base,
                    "direction_label": direction_label,
                    "target_return": 0.001 * centered_direction + 0.0001 * base,
                    "future_simple_return": 0.002 * centered_direction + 0.0002 * base,
                    "future_volatility_20d": 0.10 + 0.002 * base,
                    "volume_ratio_20d": 1.0 + 0.01 * ticker_index,
                }
            )
    return pd.DataFrame(rows)


class _FakeWalkForwardModel:
    calibration_summary = {
        "enabled": False,
        "method": "none",
        "samples": 0,
        "calibration_start": None,
        "calibration_end": None,
    }

    def fit(self, frame: pd.DataFrame, feature_columns: list[str]) -> None:
        del frame, feature_columns

    def predict(self, frame: pd.DataFrame, include_explanations: bool = True) -> pd.DataFrame:
        del include_explanations
        predictions = frame[["ticker", "date", "stale_data_flag"]].copy()
        base = frame["feature_a"].to_numpy(dtype=float)
        predictions["prob_down"] = 0.2
        predictions["prob_flat"] = 0.3
        predictions["prob_up"] = 0.5
        predictions["signal"] = base
        predictions["direction"] = "UP"
        predictions["expected_return"] = 0.001 * base
        predictions["predicted_volatility"] = 0.09 + 0.004 * base
        predictions["confidence"] = "medium"
        predictions["top_features"] = [[] for _ in range(len(frame))]
        return predictions

    def feature_importance_top(self, frame: pd.DataFrame, limit: int = 10) -> list[dict[str, float | str]]:
        del frame, limit
        return []


def test_walk_forward_windows_include_embargo_gap() -> None:
    dates = list(pd.bdate_range("2020-01-01", periods=200, tz="UTC"))
    windows = build_walk_forward_windows(
        dates=dates,
        initial_train_days=100,
        eval_days=20,
        step_days=20,
        embargo_days=2,
    )
    assert windows
    first = windows[0]
    assert first.train_end == dates[99]
    assert first.eval_start == dates[102]
    assert first.fold_id == 1


def test_run_walk_forward_backtest_includes_regression_metrics(monkeypatch) -> None:
    dates = pd.bdate_range("2024-01-01", periods=24, tz="UTC")
    rows: list[dict[str, object]] = []
    tickers = ["AAA", "BBB", "CCC"]
    for day_index, date in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            base = float(day_index + ticker_index + 1)
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "stale_data_flag": False,
                    "feature_a": base,
                    "direction_label": ticker_index,
                    "target_return": 0.0015 * base,
                    "future_simple_return": 0.002 * base,
                    "future_volatility_20d": 0.10 + 0.005 * base,
                    "volume_ratio_20d": 1.0,
                }
            )
    training_frame = pd.DataFrame(rows)

    class FakeModel:
        calibration_summary = {
            "enabled": False,
            "method": "none",
            "samples": 0,
            "calibration_start": None,
            "calibration_end": None,
        }

        def fit(self, frame: pd.DataFrame, feature_columns: list[str]) -> None:
            del frame, feature_columns

        def predict(self, frame: pd.DataFrame, include_explanations: bool = True) -> pd.DataFrame:
            del include_explanations
            predictions = frame[["ticker", "date", "stale_data_flag"]].copy()
            base = frame["feature_a"].to_numpy(dtype=float)
            predictions["prob_down"] = 0.2
            predictions["prob_flat"] = 0.3
            predictions["prob_up"] = 0.5
            predictions["signal"] = base
            predictions["direction"] = "UP"
            predictions["expected_return"] = 0.001 * base
            predictions["predicted_volatility"] = 0.09 + 0.004 * base
            predictions["confidence"] = "medium"
            predictions["top_features"] = [[] for _ in range(len(frame))]
            return predictions

        def feature_importance_top(self, frame: pd.DataFrame, limit: int = 10) -> list[dict[str, float | str]]:
            del frame, limit
            return []

    monkeypatch.setattr(
        "market_prediction_agent.backtest.walk_forward.build_model",
        lambda *, settings, model_name, version: FakeModel(),
    )
    settings = update_settings(
        load_settings("config/default.yaml"),
        model_settings={
            "walk_forward": {
                "initial_train_days": 10,
                "eval_days": 4,
                "step_days": 4,
                "embargo_days": 0,
            }
        },
    )

    backtest_result, predictions = run_walk_forward_backtest(
        training_frame=training_frame,
        feature_columns=["feature_a"],
        settings=settings,
        include_feature_importance=False,
        include_cpcv=False,
    )

    fold_metrics = backtest_result["folds"][0]["metrics"]
    for metric_name in [
        "return_ic",
        "return_rank_ic",
        "return_mae",
        "return_rmse",
        "vol_mae",
        "vol_mape",
        "vol_rmse",
    ]:
        assert metric_name in fold_metrics
        assert f"{metric_name}_mean" in backtest_result["aggregate_metrics"]
    assert "target_return" in predictions.columns
    assert "future_volatility_20d" in predictions.columns


def test_run_walk_forward_backtest_populates_metrics_for_multiple_folds(monkeypatch) -> None:
    training_frame = _make_training_frame(ticker_count=10, day_count=200)
    monkeypatch.setattr(
        "market_prediction_agent.backtest.walk_forward.build_model",
        lambda *, settings, model_name, version: _FakeWalkForwardModel(),
    )
    settings = update_settings(
        load_settings("config/default.yaml"),
        model_settings={
            "walk_forward": {
                "initial_train_days": 100,
                "eval_days": 20,
                "step_days": 20,
                "embargo_days": 0,
            }
        },
    )

    backtest_result, predictions = run_walk_forward_backtest(
        training_frame=training_frame,
        feature_columns=["feature_a"],
        settings=settings,
        include_feature_importance=False,
        include_cpcv=False,
    )

    assert len(backtest_result["folds"]) == 5
    assert predictions["fold_id"].nunique() == 5
    for fold in backtest_result["folds"]:
        metrics = fold["metrics"]
        for metric_name in [
            "hit_rate",
            "log_loss",
            "ic",
            "rank_ic",
            "return_ic",
            "return_rank_ic",
            "return_mae",
            "vol_mae",
            "vol_rmse",
        ]:
            assert metric_name in metrics
    for aggregate_name in [
        "hit_rate_mean",
        "log_loss_mean",
        "ic_mean",
        "rank_ic_mean",
        "return_ic_mean",
        "vol_mae_mean",
    ]:
        assert aggregate_name in backtest_result["aggregate_metrics"]


def test_run_walk_forward_backtest_filters_sp500_pit_constituents_per_fold(monkeypatch) -> None:
    training_frame = _make_training_frame(ticker_count=2, day_count=40)
    monkeypatch.setattr(
        "market_prediction_agent.backtest.walk_forward.build_model",
        lambda *, settings, model_name, version: _FakeWalkForwardModel(),
    )

    def fake_resolve_active_constituents(settings, *, as_of_date):
        del settings
        if pd.Timestamp(as_of_date) < pd.Timestamp("2024-02-15", tz="UTC"):
            return ["TICK00"]
        return ["TICK01"]

    monkeypatch.setattr(
        "market_prediction_agent.backtest.walk_forward.resolve_active_constituents",
        fake_resolve_active_constituents,
    )

    settings = update_settings(
        load_settings("config/default.yaml"),
        data={"universe": "sp500_pit"},
        model_settings={
            "walk_forward": {
                "initial_train_days": 20,
                "eval_days": 10,
                "step_days": 10,
                "embargo_days": 0,
            }
        },
    )

    _, predictions = run_walk_forward_backtest(
        training_frame=training_frame,
        feature_columns=["feature_a"],
        settings=settings,
        include_feature_importance=False,
        include_cpcv=False,
    )

    fold_one = predictions.loc[predictions["fold_id"] == 1, "ticker"].unique().tolist()
    fold_two = predictions.loc[predictions["fold_id"] == 2, "ticker"].unique().tolist()
    assert fold_one == ["TICK00"]
    assert fold_two == ["TICK01"]


def test_build_walk_forward_windows_returns_no_folds_for_short_history() -> None:
    dates = list(pd.bdate_range("2024-01-01", periods=30, tz="UTC"))
    windows = build_walk_forward_windows(
        dates=dates,
        initial_train_days=25,
        eval_days=10,
        step_days=5,
        embargo_days=0,
    )
    assert windows == []


def test_build_walk_forward_windows_returns_no_folds_when_embargo_is_too_large() -> None:
    dates = list(pd.bdate_range("2024-01-01", periods=40, tz="UTC"))
    windows = build_walk_forward_windows(
        dates=dates,
        initial_train_days=20,
        eval_days=10,
        step_days=10,
        embargo_days=15,
    )
    assert windows == []


def test_run_model_comparisons_returns_completed_and_error_results(monkeypatch) -> None:
    settings = update_settings(
        load_settings("config/default.yaml"),
        model_settings={"comparison_models": ["alt_model", "broken_model"]},
    )
    primary_backtest_result = {
        "config": {"model_name": "lightgbm_multiclass_calibrated"},
        "aggregate_metrics": {"hit_rate_mean": 0.50, "log_loss_mean": 1.00, "ece_mean": 0.08},
        "cost_adjusted_metrics": {"information_ratio": 0.10},
    }

    def fake_run_walk_forward_backtest(
        training_frame: pd.DataFrame,
        feature_columns: list[str],
        settings,
        *,
        model_name: str | None = None,
        include_feature_importance: bool = True,
        include_cpcv: bool = True,
        feature_catalog: list[dict[str, object]] | None = None,
    ) -> tuple[dict[str, object], pd.DataFrame]:
        del training_frame, feature_columns, settings, include_feature_importance, include_cpcv, feature_catalog
        if model_name == "broken_model":
            raise RuntimeError("comparison failed")
        return (
            {
                "aggregate_metrics": {"hit_rate_mean": 0.55, "log_loss_mean": 0.95, "ece_mean": 0.04},
                "cost_adjusted_metrics": {"information_ratio": 0.18},
                "diebold_mariano": {"vs_baseline": "class_prior", "dm_statistic": 0.0, "p_value": 1.0},
            },
            pd.DataFrame(),
        )

    monkeypatch.setattr(
        "market_prediction_agent.backtest.walk_forward.run_walk_forward_backtest",
        fake_run_walk_forward_backtest,
    )

    comparisons = run_model_comparisons(
        training_frame=_make_training_frame(ticker_count=3, day_count=30),
        feature_columns=["feature_a"],
        settings=settings,
        primary_backtest_result=primary_backtest_result,
    )

    completed = next(item for item in comparisons if item["model_name"] == "alt_model")
    assert completed["status"] == "completed"
    assert completed["comparison_to_primary"]["hit_rate_mean_delta"] == pytest.approx(0.05)
    assert completed["comparison_to_primary"]["information_ratio_delta"] == pytest.approx(0.08)
    errored = next(item for item in comparisons if item["model_name"] == "broken_model")
    assert errored["status"] == "error"
    assert "comparison failed" in errored["error"]

