from __future__ import annotations

from pathlib import Path
import shutil
from uuid import uuid4

from market_prediction_agent.config import load_settings, update_settings
from market_prediction_agent.retraining.ledger_service import RetrainingEventLedgerService


def test_retraining_ledger_service_appends_and_filters_history() -> None:
    artifact_root = Path(".test-artifacts") / uuid4().hex
    shutil.rmtree(artifact_root, ignore_errors=True)
    settings = update_settings(
        load_settings("config/default.yaml"),
        data={"storage_path": str(artifact_root / "storage")},
    )
    service = RetrainingEventLedgerService(settings)

    service.append_entry(
        {
            "created_at": "2026-03-20T00:00:00+00:00",
            "as_of_date": "2026-03-20",
            "ticker_set": "SPY,QQQ,GLD",
            "source_mode": "live",
            "dummy_mode": None,
            "current_regime": "low_vol",
            "dominant_recent_regime": "low_vol",
            "regime_bucket": "low_vol",
            "base_should_retrain": True,
            "should_retrain": True,
            "trigger_names": ["feature_drift"],
            "base_trigger_names": ["feature_drift"],
            "effective_trigger_names": ["feature_drift"],
            "suppressed_trigger_names": [],
            "drift_trigger_families": ["macro", "volatility"],
            "family_regime_keys": ["macro:low_vol", "volatility:low_vol"],
            "base_cause_keys": ["feature_drift:macro:low_vol", "feature_drift:volatility:low_vol"],
            "effective_cause_keys": ["feature_drift:macro:low_vol", "feature_drift:volatility:low_vol"],
            "pbo": 0.1,
            "pbo_label": "low_overfit_risk",
            "policy_decision": "trigger",
        }
    )
    service.append_entry(
        {
            "created_at": "2026-03-27T00:00:00+00:00",
            "as_of_date": "2026-03-27",
            "ticker_set": "SPY,QQQ,GLD",
            "source_mode": "dummy",
            "dummy_mode": "null_random_walk",
            "current_regime": "low_vol",
            "dominant_recent_regime": "low_vol",
            "regime_bucket": "low_vol",
            "base_should_retrain": False,
            "should_retrain": False,
            "trigger_names": [],
            "base_trigger_names": [],
            "effective_trigger_names": [],
            "suppressed_trigger_names": [],
            "drift_trigger_families": [],
            "family_regime_keys": [],
            "base_cause_keys": [],
            "effective_cause_keys": [],
            "pbo": 0.0,
            "pbo_label": "low_overfit_risk",
            "policy_decision": "watch_only",
        }
    )

    history = service.load_policy_history(tickers=["QQQ", "SPY", "GLD"], as_of_date="2026-04-03")

    assert len(history) == 1
    assert history[0]["source_mode"] == "live"
    assert history[0]["as_of_date"] == "2026-03-20"
    assert history[0]["effective_cause_keys"] == [
        "feature_drift:macro:low_vol",
        "feature_drift:volatility:low_vol",
    ]
