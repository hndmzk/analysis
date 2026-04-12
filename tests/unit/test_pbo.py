from __future__ import annotations

from market_prediction_agent.evaluation.pbo import interpret_pbo


def test_pbo_interpretation_thresholds_are_fixed() -> None:
    assert interpret_pbo(0.10)["label"] == "low_overfit_risk"
    assert interpret_pbo(0.30)["label"] == "moderate_overfit_risk"
    assert interpret_pbo(0.60)["label"] == "high_overfit_risk"
    assert interpret_pbo(0.85)["label"] == "severe_overfit_risk"
