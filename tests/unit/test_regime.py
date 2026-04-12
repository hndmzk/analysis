from __future__ import annotations

import numpy as np
import pandas as pd

from market_prediction_agent.config import load_settings
from market_prediction_agent.evaluation.regime import detect_regime


def test_hmm_regime_detection_returns_summary() -> None:
    settings = load_settings("config/default.yaml")
    dates = pd.bdate_range("2024-01-01", periods=180, tz="UTC")
    low_vol = np.random.default_rng(1).normal(0.0005, 0.005, size=90)
    high_vol = np.random.default_rng(2).normal(-0.001, 0.03, size=90)
    returns = np.concatenate([low_vol, high_vol])
    rows = []
    for ticker in ["AAA", "BBB", "CCC"]:
        for index, date in enumerate(dates):
            rows.append(
                {
                    "ticker": ticker,
                    "date": date,
                    "log_return_1d": returns[index],
                    "realized_vol_20d": abs(returns[index]) * 10 + (0.1 if index >= 90 else 0.02),
                    "vix": 30.0 if index >= 90 else 15.0,
                    "vix_change_5d": 0.1 if index >= 90 else -0.02,
                }
            )
    feature_frame = pd.DataFrame(rows)
    macro = pd.DataFrame(
        [{"series_id": "VIXCLS", "available_at": dates[-1], "value": 30.0}],
    )
    regime = detect_regime(feature_frame=feature_frame, macro=macro, settings=settings)
    assert regime["current_regime"] in {"low_vol", "high_vol", "transition"}
    assert regime["method"] in {"gaussian_hmm", "vix_threshold_fallback"}
    assert "regime_shift_flag" in regime
