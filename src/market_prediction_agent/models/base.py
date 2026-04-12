from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class ForecastModel(ABC):
    @abstractmethod
    def fit(self, frame: pd.DataFrame, feature_columns: list[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, frame: pd.DataFrame, include_explanations: bool = True) -> pd.DataFrame:
        raise NotImplementedError
