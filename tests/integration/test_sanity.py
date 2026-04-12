from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4

from market_prediction_agent.config import load_settings, update_settings
from market_prediction_agent.pipeline import MarketPredictionPipeline


def test_null_random_walk_sanity_check(monkeypatch) -> None:
    monkeypatch.setenv("CONFIG_PATH", str(Path("config") / "default.yaml"))
    artifact_root = Path(".test-artifacts") / uuid4().hex
    shutil.rmtree(artifact_root, ignore_errors=True)
    settings = update_settings(
        load_settings(),
        data={
            "storage_path": str(artifact_root / "storage"),
            "dummy_mode": "null_random_walk",
            "dummy_ticker_count": 20,
            "dummy_days": 900,
        },
        model_settings={
            "walk_forward": {
                "initial_train_days": 500,
                "eval_days": 60,
                "step_days": 60,
            },
            "cpcv": {
                "group_count": 4,
                "test_groups": 1,
                "max_splits": 2,
            },
            "portfolio_rule": {
                "min_edge": 0.0,
                "bucket_hysteresis": 0.0,
                "hysteresis_edge_buffer": 0.0,
                "reentry_cooldown_days": 0,
                "max_turnover_per_day": 0.0,
                "participation_volume_floor": 1.0,
                "participation_volume_ceiling": 1.0,
            },
        },
    )
    result = MarketPredictionPipeline(settings).run()
    hit_rate = result.backtest_result["aggregate_metrics"]["hit_rate_mean"]
    information_ratio = result.backtest_result["cost_adjusted_metrics"]["information_ratio"]
    pbo = result.backtest_result["pbo"]
    pbo_label = result.backtest_result["cpcv"]["pbo_summary"]["label"]
    assert 0.28 <= hit_rate <= 0.38
    assert abs(information_ratio) < 1.0
    assert 0.0 <= pbo <= 1.0
    assert pbo_label in {
        "low_overfit_risk",
        "moderate_overfit_risk",
        "high_overfit_risk",
        "severe_overfit_risk",
    }
    assert result.risk_review["approval"] != "AUTO_APPROVED"
    assert any("Sanity dummy runs" in note for note in result.risk_review["approval_notes"])


def test_predictable_momentum_is_not_forced_to_blocked(monkeypatch) -> None:
    monkeypatch.setenv("CONFIG_PATH", str(Path("config") / "default.yaml"))
    artifact_root = Path(".test-artifacts") / uuid4().hex
    shutil.rmtree(artifact_root, ignore_errors=True)
    settings = update_settings(
        load_settings(),
        data={
            "storage_path": str(artifact_root / "storage"),
            "dummy_mode": "predictable_momentum",
            "dummy_ticker_count": 20,
            "dummy_days": 900,
        },
        model_settings={
            "walk_forward": {
                "initial_train_days": 500,
                "eval_days": 60,
                "step_days": 60,
            },
            "cpcv": {
                "group_count": 4,
                "test_groups": 1,
                "max_splits": 2,
            },
        },
    )
    result = MarketPredictionPipeline(settings).run()
    pbo = result.backtest_result["pbo"]
    pbo_label = result.backtest_result["cpcv"]["pbo_summary"]["label"]
    assert 0.0 <= pbo <= 1.0
    assert pbo_label in {
        "low_overfit_risk",
        "moderate_overfit_risk",
        "high_overfit_risk",
        "severe_overfit_risk",
    }
    assert result.risk_review["approval"] in {"AUTO_APPROVED", "MANUAL_REVIEW_REQUIRED"}
    assert any("development/demo only" in note for note in result.risk_review["approval_notes"])

