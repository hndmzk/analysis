from __future__ import annotations

from market_prediction_agent.data.adapters import DummyOHLCVAdapter, OHLCVRequest


def test_dummy_modes_have_different_predictability_profiles() -> None:
    request = OHLCVRequest(tickers=["AAA"], start_date="2020-01-01", end_date="2024-12-31")
    null_frame = DummyOHLCVAdapter(seed=42, mode="null_random_walk").fetch(request)
    predictable_frame = DummyOHLCVAdapter(seed=42, mode="predictable_momentum").fetch(request)
    null_autocorr = null_frame["close"].pct_change().dropna().autocorr()
    predictable_autocorr = predictable_frame["close"].pct_change().dropna().autocorr()
    assert abs(null_autocorr) < 0.12
    assert predictable_autocorr > null_autocorr + 0.03

