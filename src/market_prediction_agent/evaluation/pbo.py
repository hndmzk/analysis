from __future__ import annotations


PBO_DEFINITION = (
    "Probability of Backtest Overfitting is estimated as the fraction of CPCV splits "
    "where the best in-sample portfolio-rule candidate lands below the median "
    "out-of-sample rank, equivalently logit(omega) < 0."
)


def interpret_pbo(pbo: float | None) -> dict[str, object]:
    thresholds = {
        "low": 0.20,
        "moderate": 0.50,
        "high": 0.80,
    }
    if pbo is None:
        return {
            "definition": PBO_DEFINITION,
            "thresholds": thresholds,
            "label": "not_available",
            "detail": "CPCV splits were insufficient to estimate PBO.",
        }
    if pbo < thresholds["low"]:
        label = "low_overfit_risk"
    elif pbo < thresholds["moderate"]:
        label = "moderate_overfit_risk"
    elif pbo < thresholds["high"]:
        label = "high_overfit_risk"
    else:
        label = "severe_overfit_risk"
    return {
        "definition": PBO_DEFINITION,
        "thresholds": thresholds,
        "label": label,
        "detail": (
            f"PBO={pbo:.3f}; <{thresholds['low']:.2f} is low, "
            f"{thresholds['low']:.2f}-{thresholds['moderate']:.2f} is moderate, "
            f"{thresholds['moderate']:.2f}-{thresholds['high']:.2f} is high, "
            f">={thresholds['high']:.2f} is severe."
        ),
    }
