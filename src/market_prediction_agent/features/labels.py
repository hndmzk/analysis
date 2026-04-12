from __future__ import annotations

import numpy as np
import pandas as pd


CLASS_TO_LABEL = {0: "DOWN", 1: "FLAT", 2: "UP"}
LABEL_TO_CLASS = {value: key for key, value in CLASS_TO_LABEL.items()}


def make_direction_label(returns: pd.Series, threshold: float = 0.005) -> pd.Series:
    bins = [-np.inf, -threshold, threshold, np.inf]
    labeled = pd.cut(returns, bins=bins, labels=[0, 1, 2])
    return pd.Series(labeled, index=returns.index, dtype="float")

