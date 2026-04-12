from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from market_prediction_agent.features.labels import CLASS_TO_LABEL
from market_prediction_agent.models.base import ForecastModel


@dataclass(slots=True)
class BaselineGaussianRidgeModel(ForecastModel):
    version: str
    ridge_alpha: float = 1.0
    trained_at: str | None = None
    feature_columns: list[str] = field(default_factory=list)
    fill_values: pd.Series | None = None
    scale_mean: np.ndarray | None = None
    scale_std: np.ndarray | None = None
    priors: np.ndarray | None = None
    class_mean: np.ndarray | None = None
    class_var: np.ndarray | None = None
    return_coef: np.ndarray | None = None
    vol_coef: np.ndarray | None = None
    training_samples: int = 0

    def fit(self, frame: pd.DataFrame, feature_columns: list[str]) -> None:
        self.feature_columns = feature_columns
        self.fill_values = frame[feature_columns].median()
        x_raw = frame[feature_columns].fillna(self.fill_values).to_numpy(dtype=float)
        self.scale_mean = x_raw.mean(axis=0)
        self.scale_std = np.where(x_raw.std(axis=0) == 0, 1.0, x_raw.std(axis=0))
        x = (x_raw - self.scale_mean) / self.scale_std
        y_class = frame["direction_label"].to_numpy(dtype=int)
        self.training_samples = len(frame)

        priors: list[float] = []
        means: list[np.ndarray] = []
        variances: list[np.ndarray] = []
        for klass in [0, 1, 2]:
            class_mask = y_class == klass
            if not class_mask.any():
                priors.append(1e-9)
                means.append(np.zeros(x.shape[1]))
                variances.append(np.ones(x.shape[1]))
                continue
            class_values = x[class_mask]
            priors.append(float(class_values.shape[0] / x.shape[0]))
            means.append(class_values.mean(axis=0))
            variances.append(np.where(class_values.var(axis=0) < 1e-6, 1e-6, class_values.var(axis=0)))
        self.priors = np.array(priors)
        self.class_mean = np.vstack(means)
        self.class_var = np.vstack(variances)
        self.return_coef = self._fit_ridge(x, frame["target_return"].fillna(0.0).to_numpy(dtype=float))
        vol_target = frame["future_volatility_20d"].fillna(frame["future_volatility_20d"].median())
        self.vol_coef = self._fit_ridge(x, vol_target.to_numpy(dtype=float))
        self.trained_at = datetime.now(timezone.utc).isoformat()

    def _fit_ridge(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        design = np.hstack([np.ones((x.shape[0], 1)), x])
        identity = np.eye(design.shape[1])
        identity[0, 0] = 0.0
        return np.linalg.pinv(design.T @ design + self.ridge_alpha * identity) @ design.T @ y

    def _transform(self, frame: pd.DataFrame) -> np.ndarray:
        if self.fill_values is None or self.scale_mean is None or self.scale_std is None:
            raise RuntimeError("Model must be fit before predict.")
        x_raw = frame[self.feature_columns].fillna(self.fill_values).to_numpy(dtype=float)
        return (x_raw - self.scale_mean) / self.scale_std

    def predict(self, frame: pd.DataFrame, include_explanations: bool = True) -> pd.DataFrame:
        if any(item is None for item in [self.priors, self.class_mean, self.class_var, self.return_coef, self.vol_coef]):
            raise RuntimeError("Model must be fit before predict.")
        assert self.priors is not None
        assert self.class_mean is not None
        assert self.class_var is not None
        assert self.return_coef is not None
        assert self.vol_coef is not None
        x = self._transform(frame)
        log_probabilities = []
        for class_index in range(3):
            mean = self.class_mean[class_index]
            variance = self.class_var[class_index]
            log_density = -0.5 * np.sum(np.log(2 * np.pi * variance) + ((x - mean) ** 2) / variance, axis=1)
            log_probabilities.append(np.log(self.priors[class_index]) + log_density)
        log_prob_matrix = np.vstack(log_probabilities).T
        max_log = np.max(log_prob_matrix, axis=1, keepdims=True)
        probability_matrix = np.exp(log_prob_matrix - max_log)
        probability_matrix = probability_matrix / probability_matrix.sum(axis=1, keepdims=True)

        design = np.hstack([np.ones((x.shape[0], 1)), x])
        expected_return = design @ self.return_coef
        predicted_volatility = np.maximum(0.0, design @ self.vol_coef)
        predicted_class = probability_matrix.argmax(axis=1)
        max_prob = probability_matrix.max(axis=1)
        confidence = np.where(max_prob > 0.5, "high", np.where(max_prob > 0.4, "medium", "low"))

        predictions = frame[["ticker", "date", "stale_data_flag"]].copy()
        predictions["prob_down"] = probability_matrix[:, 0]
        predictions["prob_flat"] = probability_matrix[:, 1]
        predictions["prob_up"] = probability_matrix[:, 2]
        predictions["signal"] = predictions["prob_up"] - predictions["prob_down"]
        predictions["direction"] = [CLASS_TO_LABEL[int(index)] for index in predicted_class]
        predictions["expected_return"] = expected_return
        predictions["predicted_volatility"] = predicted_volatility
        predictions["confidence"] = confidence
        predictions["top_features"] = self._top_features(x)
        return predictions

    def _top_features(self, x: np.ndarray) -> list[list[dict[str, float | str]]]:
        if self.return_coef is None:
            raise RuntimeError("Model must be fit before predict.")
        assert self.return_coef is not None
        coefficients = np.abs(self.return_coef[1:])
        results: list[list[dict[str, float | str]]] = []
        for row in x:
            contributions = np.abs(row) * coefficients
            top_indices = np.argsort(contributions)[::-1][:5]
            results.append(
                [
                    {
                        "name": self.feature_columns[index],
                        "shap_value": float(contributions[index]),
                    }
                    for index in top_indices
                ]
            )
        return results


BaselineModel = BaselineGaussianRidgeModel
