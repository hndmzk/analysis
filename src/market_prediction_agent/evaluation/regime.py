from __future__ import annotations

from collections import Counter

from hmmlearn.hmm import GaussianHMM
import numpy as np
import pandas as pd

from market_prediction_agent.config import Settings


def _fallback_regime(macro: pd.DataFrame, settings: Settings, reason: str) -> dict[str, object]:
    vix_rows = macro.loc[macro["series_id"] == "VIXCLS"].sort_values("available_at")
    current_regime = "unknown"
    current_vix = None
    if not vix_rows.empty:
        current_vix = float(vix_rows["value"].iloc[-1])
        current_regime = "high_vol" if current_vix > settings.risk.vix_stress_threshold else "low_vol"
    return {
        "method": "vix_threshold_fallback",
        "current_regime": current_regime,
        "previous_regime": current_regime,
        "dominant_recent_regime": current_regime,
        "regime_shift_flag": False,
        "state_probability": 1.0,
        "transition_rate": 0.0,
        "reason": reason,
        "state_statistics": [],
        "current_vix": current_vix,
    }


def _build_regime_observation_frame(feature_frame: pd.DataFrame) -> pd.DataFrame:
    observations = (
        feature_frame.groupby("date", as_index=False)
        .agg(
            market_return_1d=("log_return_1d", "mean"),
            market_volatility_20d=("realized_vol_20d", "median"),
            market_vix=("vix", "median"),
            market_vix_change_5d=("vix_change_5d", "median"),
        )
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )
    return observations


def _state_mapping(state_stats: list[dict[str, float | int]]) -> dict[int, str]:
    ordered = sorted(
        state_stats,
        key=lambda item: (float(item["mean_volatility"]), float(item["mean_vix"])),
    )
    if len(ordered) == 1:
        return {int(ordered[0]["state"]): "low_vol"}
    if len(ordered) == 2:
        return {
            int(ordered[0]["state"]): "low_vol",
            int(ordered[1]["state"]): "high_vol",
        }
    mapping: dict[int, str] = {}
    mapping[int(ordered[0]["state"])] = "low_vol"
    mapping[int(ordered[-1]["state"])] = "high_vol"
    for item in ordered[1:-1]:
        mapping[int(item["state"])] = "transition"
    return mapping


def detect_regime(feature_frame: pd.DataFrame, macro: pd.DataFrame, settings: Settings) -> dict[str, object]:
    observations = _build_regime_observation_frame(feature_frame)
    if len(observations) < settings.model_settings.hmm.min_history_days:
        return _fallback_regime(macro, settings, reason="Insufficient history for HMM regime detection.")

    columns = ["market_return_1d", "market_volatility_20d", "market_vix", "market_vix_change_5d"]
    matrix = observations[columns].to_numpy(dtype=float)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    scale = matrix.std(axis=0, keepdims=True)
    scale = np.where(scale == 0, 1.0, scale)
    normalized = centered / scale
    try:
        model = GaussianHMM(
            n_components=settings.model_settings.hmm.n_states,
            covariance_type="full",
            n_iter=200,
            random_state=settings.app.seed,
        )
        model.fit(normalized)
        hidden_states = model.predict(normalized)
        probabilities = model.predict_proba(normalized)
    except Exception:
        return _fallback_regime(macro, settings, reason="HMM fit failed; fallback to VIX threshold.")

    state_stats: list[dict[str, float | int]] = []
    for state in range(settings.model_settings.hmm.n_states):
        mask = hidden_states == state
        if not mask.any():
            continue
        state_stats.append(
            {
                "state": state,
                "mean_return": float(observations.loc[mask, "market_return_1d"].mean()),
                "mean_volatility": float(observations.loc[mask, "market_volatility_20d"].mean()),
                "mean_vix": float(observations.loc[mask, "market_vix"].mean()),
                "sample_count": int(mask.sum()),
            }
        )
    if not state_stats:
        return _fallback_regime(macro, settings, reason="HMM produced no usable states.")

    mapping = _state_mapping(state_stats)
    current_state = int(hidden_states[-1])
    lookback = min(settings.model_settings.hmm.regime_shift_lookback_days, max(len(hidden_states) - 1, 1))
    recent_history = hidden_states[-(lookback + 1) : -1]
    if len(recent_history) == 0:
        recent_history = hidden_states[:-1]
    previous_state = int(recent_history[-1]) if len(recent_history) else current_state
    dominant_recent_state = Counter(recent_history.tolist()).most_common(1)[0][0] if len(recent_history) else current_state
    recent_window = hidden_states[-(lookback + 1) :]
    transition_rate = 0.0
    if len(recent_window) > 1:
        transition_rate = float(np.mean(recent_window[1:] != recent_window[:-1]))
    current_regime = mapping.get(current_state, "unknown")
    previous_regime = mapping.get(previous_state, current_regime)
    dominant_recent_regime = mapping.get(int(dominant_recent_state), current_regime)
    return {
        "method": "gaussian_hmm",
        "current_regime": current_regime,
        "previous_regime": previous_regime,
        "dominant_recent_regime": dominant_recent_regime,
        "regime_shift_flag": current_regime != dominant_recent_regime,
        "state_probability": float(probabilities[-1, current_state]),
        "transition_rate": transition_rate,
        "reason": "Gaussian HMM fitted on daily market return, volatility, and VIX observations.",
        "state_mapping": {str(state): regime for state, regime in mapping.items()},
        "state_statistics": state_stats,
        "current_vix": float(observations["market_vix"].iloc[-1]),
    }

