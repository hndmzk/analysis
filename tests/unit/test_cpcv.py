from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_prediction_agent.backtest.cpcv import (
    _aggregate_pbo_diagnostics,
    build_cpcv_splits,
    run_cpcv_backtest,
)
from market_prediction_agent.config import load_settings, update_settings


def test_cpcv_builds_limited_purged_splits() -> None:
    dates = list(pd.bdate_range("2020-01-01", periods=120, tz="UTC"))
    splits = build_cpcv_splits(
        dates=dates,
        group_count=6,
        test_groups=2,
        embargo_days=2,
        max_splits=4,
    )
    assert splits
    assert len(splits) <= 4
    first = splits[0]
    assert first.train_dates
    assert first.test_dates
    assert set(first.train_dates).isdisjoint(set(first.test_dates))


def test_pbo_diagnostics_decompose_candidate_axes_and_competition() -> None:
    settings = load_settings("config/default.yaml")
    diagnostics = _aggregate_pbo_diagnostics(
        [
            {
                "overfit_flag": True,
                "best_strategy_name": "classified_directional",
                "best_threshold_key": "0.30",
                "best_bucket_key": "top=0.50|bottom=0.50",
                "best_holding_days_key": "5",
                "best_vs_runner_up_in_sample_margin": 0.04,
                "best_vs_runner_up_oos_margin": -0.02,
                "best_candidate_test_information_ratio": 0.10,
                "same_family_runner_up": True,
                "same_threshold_runner_up": False,
                "same_bucket_runner_up": True,
                "same_holding_days_runner_up": True,
            },
            {
                "overfit_flag": True,
                "best_strategy_name": "classified_directional",
                "best_threshold_key": "0.30",
                "best_bucket_key": "top=0.50|bottom=0.50",
                "best_holding_days_key": "5",
                "best_vs_runner_up_in_sample_margin": 0.05,
                "best_vs_runner_up_oos_margin": -0.01,
                "best_candidate_test_information_ratio": 0.08,
                "same_family_runner_up": True,
                "same_threshold_runner_up": False,
                "same_bucket_runner_up": False,
                "same_holding_days_runner_up": True,
            },
            {
                "overfit_flag": False,
                "best_strategy_name": "classified_directional",
                "best_threshold_key": "0.35",
                "best_bucket_key": "top=0.25|bottom=0.25",
                "best_holding_days_key": "3",
                "best_vs_runner_up_in_sample_margin": 0.30,
                "best_vs_runner_up_oos_margin": 0.04,
                "best_candidate_test_information_ratio": 0.15,
                "same_family_runner_up": True,
                "same_threshold_runner_up": True,
                "same_bucket_runner_up": True,
                "same_holding_days_runner_up": False,
            },
        ],
        settings,
    )
    assert diagnostics["threshold_contribution"][0]["probability_threshold"] == "0.30"
    assert diagnostics["bucket_contribution"][0]["bucket_pair"] == "top=0.50|bottom=0.50"
    assert diagnostics["holding_days_contribution"][0]["holding_days"] == "5"
    assert diagnostics["near_candidate_competition"]["competition_dominated"] is True
    assert diagnostics["near_candidate_competition"]["close_split_ratio"] > 0.5


def _make_training_frame(*, ticker_count: int = 3, day_count: int = 60) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=day_count, tz="UTC")
    rows: list[dict[str, object]] = []
    tickers = [f"TICK{i:02d}" for i in range(ticker_count)]
    for day_index, date in enumerate(dates):
        for ticker_index, ticker in enumerate(tickers):
            base = float(day_index + ticker_index + 1)
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "stale_data_flag": False,
                    "feature_a": base,
                    "direction_label": (day_index + ticker_index) % 3,
                    "target_return": 0.0005 * base,
                    "future_simple_return": 0.001 * base,
                    "future_volatility_20d": 0.10 + 0.001 * base,
                    "volume_ratio_20d": 1.0,
                }
            )
    return pd.DataFrame(rows)


def _fake_cost_metrics(information_ratio: float) -> dict[str, float]:
    return {
        "information_ratio": information_ratio,
        "gross_information_ratio": information_ratio + 0.05,
        "annual_return": information_ratio / 10.0,
        "gross_annual_return": information_ratio / 10.0 + 0.02,
        "cost_drag_annual_return": 0.01,
        "active_days_ratio": 0.30,
        "signal_days_ratio": 0.30,
        "avg_daily_turnover": 0.05,
        "total_cost_bps": 10.0,
        "selection_stability": 0.40,
        "two_sided_signal_days_ratio": 0.10,
        "one_sided_signal_days_ratio": 0.20,
        "rebalance_thinned_days_ratio": 0.05,
        "avg_target_turnover": 0.08,
    }


def test_cpcv_builds_expected_combination_count() -> None:
    dates = list(pd.bdate_range("2020-01-01", periods=120, tz="UTC"))
    splits = build_cpcv_splits(
        dates=dates,
        group_count=6,
        test_groups=2,
        embargo_days=0,
        max_splits=15,
    )
    assert len(splits) == 15
    assert [split.split_id for split in splits] == list(range(1, 16))
    assert {len(split.test_dates) for split in splits} == {40}


def test_run_cpcv_backtest_returns_expected_pbo_for_known_candidate_ranks(monkeypatch) -> None:
    settings = update_settings(
        load_settings("config/default.yaml"),
        model_settings={
            "walk_forward": {"embargo_days": 0},
            "cpcv": {
                "group_count": 3,
                "test_groups": 1,
                "max_splits": 2,
                "strategy_names": ["classified_directional"],
                "portfolio_thresholds": [0.30, 0.35, 0.40],
                "top_bucket_fractions": [0.25],
                "bottom_bucket_fractions": [0.25],
                "holding_days": [5],
                "threshold_cluster_tolerance": 0.0,
                "bucket_cluster_tolerance": 0.0,
                "holding_days_cluster_tolerance": 0,
            },
        },
    )

    class FakeModel:
        def __init__(self, settings, version) -> None:
            del settings, version

        def fit(self, frame: pd.DataFrame, feature_columns: list[str]) -> None:
            del frame, feature_columns

    call_index = {"value": 0}
    expected_thresholds = [0.30, 0.30, 0.35, 0.35, 0.40, 0.40] * 2
    information_ratios = [
        0.90,
        0.10,
        0.80,
        0.60,
        0.70,
        0.90,
        0.40,
        0.30,
        0.85,
        0.95,
        0.70,
        0.50,
    ]

    def fake_prepare_predictions(model: FakeModel, frame: pd.DataFrame, baseline_prior: pd.Series) -> pd.DataFrame:
        del model, baseline_prior
        return pd.DataFrame(
            {
                "ticker": frame["ticker"].to_numpy(),
                "date": frame["date"].to_numpy(),
                "stale_data_flag": frame["stale_data_flag"].to_numpy(dtype=bool),
                "prob_down": np.full(len(frame), 0.2),
                "prob_flat": np.full(len(frame), 0.3),
                "prob_up": np.full(len(frame), 0.5),
                "direction_label": frame["direction_label"].to_numpy(dtype=int),
                "future_simple_return": frame["future_simple_return"].to_numpy(dtype=float),
                "volume_ratio_20d": frame["volume_ratio_20d"].to_numpy(dtype=float),
            }
        )

    def fake_compute_cost_adjusted_metrics(
        predictions: pd.DataFrame,
        *,
        probability_threshold: float,
        top_bucket_fraction: float,
        bottom_bucket_fraction: float,
        strategy_name: str,
        holding_days: int,
        **controls,
    ) -> dict[str, float]:
        del predictions, top_bucket_fraction, bottom_bucket_fraction, strategy_name, holding_days, controls
        threshold_index = call_index["value"]
        assert probability_threshold == pytest.approx(expected_thresholds[threshold_index])
        metrics = _fake_cost_metrics(information_ratios[threshold_index])
        call_index["value"] += 1
        return metrics

    monkeypatch.setattr("market_prediction_agent.backtest.cpcv.LightGBMCalibratedModel", FakeModel)
    monkeypatch.setattr("market_prediction_agent.backtest.cpcv._prepare_predictions", fake_prepare_predictions)
    monkeypatch.setattr(
        "market_prediction_agent.backtest.cpcv.compute_fold_metrics",
        lambda predictions: ({"hit_rate": 0.4, "log_loss": 1.0, "ece": 0.05}, np.array([0.0])),
    )
    monkeypatch.setattr(
        "market_prediction_agent.backtest.cpcv.compute_cost_adjusted_metrics",
        fake_compute_cost_adjusted_metrics,
    )

    result, pbo = run_cpcv_backtest(_make_training_frame(), ["feature_a"], settings)

    assert pbo == pytest.approx(0.5)
    assert result["pbo_summary"]["label"] == "high_overfit_risk"
    assert len(result["splits"]) == 2
    assert [split["best_threshold_key"] for split in result["splits"]] == ["0.30", "0.35"]

