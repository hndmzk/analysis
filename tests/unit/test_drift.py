from __future__ import annotations

import numpy as np
import pandas as pd

from market_prediction_agent.evaluation.drift import compute_feature_drift, population_stability_index


def test_population_stability_index_is_small_for_similar_distributions() -> None:
    reference = pd.Series(np.linspace(-1.0, 1.0, 200))
    current = pd.Series(np.linspace(-1.05, 0.95, 200))
    psi = population_stability_index(reference, current)
    assert psi < 0.1


def test_compute_feature_drift_flags_large_distribution_shift() -> None:
    reference_frame = pd.DataFrame(
        {
            "feature_a": np.linspace(-1.0, 1.0, 200),
            "feature_b": np.linspace(-0.5, 0.5, 200),
        }
    )
    current_frame = pd.DataFrame(
        {
            "feature_a": np.linspace(2.0, 4.0, 50),
            "feature_b": np.linspace(-0.45, 0.55, 50),
        }
    )
    summary = compute_feature_drift(
        reference_frame=reference_frame,
        current_frame=current_frame,
        feature_columns=["feature_a", "feature_b"],
        psi_warning=0.2,
        psi_critical=0.25,
    )
    assert summary["max_psi"] >= 0.25
    assert "feature_a" in summary["critical_features"]
    assert summary["max_raw_psi"] >= summary["max_psi"]


def test_compute_feature_drift_identifies_proxy_artifact_pattern() -> None:
    reference_frame = pd.DataFrame(
        {
            "garman_klass_vol": np.linspace(0.01, 0.03, 200),
            "volume_ratio_5d": np.linspace(0.9, 1.1, 200),
        }
    )
    current_frame = pd.DataFrame(
        {
            "garman_klass_vol": np.linspace(0.20, 0.40, 50),
            "volume_ratio_5d": np.linspace(2.0, 3.0, 50),
        }
    )
    summary = compute_feature_drift(
        reference_frame=reference_frame,
        current_frame=current_frame,
        feature_columns=["garman_klass_vol", "volume_ratio_5d"],
        psi_warning=0.2,
        psi_critical=0.25,
        proxy_ohlcv_used=True,
        regime_summary={"current_regime": "low_vol", "regime_shift_flag": False, "state_probability": 0.9},
    )
    assert summary["supplementary_analysis"]["primary_cause"] == "proxy_artifact_likely"
    assert "garman_klass_vol" in summary["supplementary_analysis"]["proxy_sensitive_flagged_features"]
    diagnostics = summary["supplementary_analysis"]["feature_diagnostics"]
    assert diagnostics[0]["feature"] == "garman_klass_vol"
    assert diagnostics[0]["proxy_sensitive"] is True
    assert diagnostics[0]["retrain_action"] == "watch"
    assert diagnostics[0]["primary_cause"] == "proxy_sensitive"


def test_compute_feature_drift_identifies_regime_shift_pattern() -> None:
    reference_frame = pd.DataFrame(
        {
            "vix": np.linspace(12.0, 18.0, 200),
            "realized_vol_20d": np.linspace(0.08, 0.16, 200),
        }
    )
    current_frame = pd.DataFrame(
        {
            "vix": np.linspace(28.0, 40.0, 50),
            "realized_vol_20d": np.linspace(0.28, 0.45, 50),
        }
    )
    summary = compute_feature_drift(
        reference_frame=reference_frame,
        current_frame=current_frame,
        feature_columns=["vix", "realized_vol_20d"],
        psi_warning=0.2,
        psi_critical=0.25,
        proxy_ohlcv_used=False,
        regime_summary={"current_regime": "transition", "regime_shift_flag": True, "state_probability": 0.52},
    )
    assert summary["supplementary_analysis"]["primary_cause"] == "regime_shift_likely"
    assert "vix" in summary["supplementary_analysis"]["macro_volatility_flagged_features"]


def test_compute_feature_drift_uses_family_threshold_tuning() -> None:
    reference_frame = pd.DataFrame(
        {
            "vix": np.linspace(12.0, 18.0, 200),
            "log_return_1d": np.linspace(-0.02, 0.02, 200),
        }
    )
    current_frame = pd.DataFrame(
        {
            "vix": np.linspace(21.0, 25.0, 50),
            "log_return_1d": np.linspace(0.10, 0.14, 50),
        }
    )
    summary = compute_feature_drift(
        reference_frame=reference_frame,
        current_frame=current_frame,
        feature_columns=["vix", "log_return_1d"],
        psi_warning=0.2,
        psi_critical=0.25,
        family_thresholds={
            "macro": {"warning": 20.0, "critical": 25.0},
            "price_momentum": {"warning": 0.2, "critical": 0.25},
        },
    )
    feature_map = {entry["feature"]: entry for entry in summary["features"]}
    assert feature_map["vix"]["status"] == "PASS"
    assert feature_map["vix"]["warning_threshold"] == 20.0
    assert feature_map["log_return_1d"]["warning_threshold"] == 0.2
