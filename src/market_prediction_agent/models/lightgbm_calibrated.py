from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from market_prediction_agent.config import Settings
from market_prediction_agent.features.labels import CLASS_TO_LABEL
from market_prediction_agent.models.base import ForecastModel


def _safe_softmax(matrix: np.ndarray) -> np.ndarray:
    shifted = matrix - matrix.max(axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


@dataclass(slots=True)
class LightGBMCalibratedModel(ForecastModel):
    settings: Settings
    version: str
    trained_at: str | None = None
    feature_columns: list[str] = field(default_factory=list)
    training_samples: int = 0
    fill_values: pd.Series | None = None
    classifier: Any | None = None
    return_regressor: Any | None = None
    volatility_regressor: Any | None = None
    calibrator: LogisticRegression | None = None
    global_feature_importance_map: dict[str, float] = field(default_factory=dict)
    calibration_summary: dict[str, Any] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame, feature_columns: list[str]) -> None:
        ordered = frame.sort_values(["date", "ticker"]).reset_index(drop=True)
        fit_frame, calibration_frame = self._split_fit_and_calibration(ordered)
        self.feature_columns = feature_columns
        self.training_samples = int(len(ordered))
        self.fill_values = fit_frame[feature_columns].median()
        x_fit = self._prepare_features(fit_frame)
        y_fit = fit_frame["direction_label"].to_numpy(dtype=int)
        lightgbm_params: dict[str, Any] = {
            "n_estimators": self.settings.model_settings.lightgbm.n_estimators,
            "learning_rate": self.settings.model_settings.lightgbm.learning_rate,
            "num_leaves": self.settings.model_settings.lightgbm.num_leaves,
            "min_child_samples": self.settings.model_settings.lightgbm.min_child_samples,
            "feature_fraction": self.settings.model_settings.lightgbm.feature_fraction,
            "bagging_fraction": self.settings.model_settings.lightgbm.bagging_fraction,
            "bagging_freq": self.settings.model_settings.lightgbm.bagging_freq,
            "reg_lambda": self.settings.model_settings.lightgbm.reg_lambda,
            "random_state": self.settings.app.seed,
            "n_jobs": 1,
            "verbosity": -1,
        }
        self.classifier = lgb.LGBMClassifier(objective="multiclass", num_class=3, **lightgbm_params)
        self.classifier.fit(x_fit, y_fit)

        self.return_regressor = lgb.LGBMRegressor(objective="regression", **lightgbm_params)
        self.return_regressor.fit(x_fit, fit_frame["target_return"].fillna(0.0).to_numpy(dtype=float))

        volatility_target = fit_frame["future_volatility_20d"].fillna(fit_frame["future_volatility_20d"].median())
        self.volatility_regressor = lgb.LGBMRegressor(objective="regression", **lightgbm_params)
        self.volatility_regressor.fit(x_fit, volatility_target.fillna(0.0).to_numpy(dtype=float))

        self.calibrator = None
        if self._can_fit_calibrator(calibration_frame):
            x_calibration = self._prepare_features(calibration_frame)
            raw_probabilities = self._predict_raw_probabilities(x_calibration)
            transformed_probabilities = self._transform_probabilities(raw_probabilities)
            self.calibrator = LogisticRegression(max_iter=500, random_state=self.settings.app.seed)
            self.calibrator.fit(transformed_probabilities, calibration_frame["direction_label"].to_numpy(dtype=int))
            self.calibration_summary = {
                "enabled": True,
                "method": self.settings.model_settings.calibration.method,
                "samples": int(len(calibration_frame)),
                "calibration_start": calibration_frame["date"].min().date().isoformat(),
                "calibration_end": calibration_frame["date"].max().date().isoformat(),
            }
        else:
            self.calibration_summary = {
                "enabled": False,
                "method": self.settings.model_settings.calibration.method,
                "samples": 0,
                "calibration_start": None,
                "calibration_end": None,
            }

        shap_reference = calibration_frame if not calibration_frame.empty else fit_frame
        self.global_feature_importance_map = self._compute_shap_importance(shap_reference)
        self.trained_at = datetime.now(timezone.utc).isoformat()

    def predict(self, frame: pd.DataFrame, include_explanations: bool = True) -> pd.DataFrame:
        if self.classifier is None or self.return_regressor is None or self.volatility_regressor is None:
            raise RuntimeError("Model must be fit before predict.")
        x = self._prepare_features(frame)
        probability_matrix = self._predict_probabilities(x)
        expected_return = self.return_regressor.predict(x)
        predicted_volatility = np.maximum(0.0, self.volatility_regressor.predict(x))
        predicted_class = probability_matrix.argmax(axis=1)
        max_prob = probability_matrix.max(axis=1)
        confidence = np.where(max_prob > 0.6, "high", np.where(max_prob > 0.45, "medium", "low"))

        predictions = frame[["ticker", "date", "stale_data_flag"]].copy()
        predictions["prob_down"] = probability_matrix[:, 0]
        predictions["prob_flat"] = probability_matrix[:, 1]
        predictions["prob_up"] = probability_matrix[:, 2]
        predictions["signal"] = predictions["prob_up"] - predictions["prob_down"]
        predictions["direction"] = [CLASS_TO_LABEL[int(index)] for index in predicted_class]
        predictions["expected_return"] = expected_return
        predictions["predicted_volatility"] = predicted_volatility
        predictions["confidence"] = confidence
        if include_explanations:
            predictions["top_features"] = self.top_features(frame, probability_matrix)
        else:
            predictions["top_features"] = [[] for _ in range(len(predictions))]
        return predictions

    def feature_importance_top(
        self,
        frame: pd.DataFrame | None = None,
        limit: int = 10,
    ) -> list[dict[str, float | str]]:
        importance_map = self.global_feature_importance_map if frame is None else self._compute_shap_importance(frame)
        ordered = sorted(importance_map.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [
            {
                "feature": feature,
                "mean_abs_shap": float(value),
            }
            for feature, value in ordered
        ]

    def hyperparameters(self) -> dict[str, float | int | str]:
        return {
            "model_name": self.settings.model_settings.primary,
            "calibration_method": self.settings.model_settings.calibration.method,
            "n_estimators": self.settings.model_settings.lightgbm.n_estimators,
            "learning_rate": self.settings.model_settings.lightgbm.learning_rate,
            "num_leaves": self.settings.model_settings.lightgbm.num_leaves,
            "min_child_samples": self.settings.model_settings.lightgbm.min_child_samples,
            "feature_fraction": self.settings.model_settings.lightgbm.feature_fraction,
            "bagging_fraction": self.settings.model_settings.lightgbm.bagging_fraction,
            "bagging_freq": self.settings.model_settings.lightgbm.bagging_freq,
            "reg_lambda": self.settings.model_settings.lightgbm.reg_lambda,
            "max_shap_samples": self.settings.model_settings.lightgbm.max_shap_samples,
        }

    def top_features(
        self,
        frame: pd.DataFrame,
        probability_matrix: np.ndarray | None = None,
        limit: int = 5,
    ) -> list[list[dict[str, float | str]]]:
        if probability_matrix is None:
            probability_matrix = self._predict_probabilities(self._prepare_features(frame))
        predicted_class = probability_matrix.argmax(axis=1)
        shap_values = self._predicted_class_shap_values(frame, predicted_class)
        results: list[list[dict[str, float | str]]] = []
        for row in shap_values:
            top_indices = np.argsort(np.abs(row))[::-1][:limit]
            results.append(
                [
                    {
                        "name": self.feature_columns[index],
                        "shap_value": float(row[index]),
                    }
                    for index in top_indices
                ]
            )
        return results

    def _split_fit_and_calibration(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        unique_dates = sorted(frame["date"].drop_duplicates().tolist())
        calibration_days = max(
            self.settings.model_settings.calibration.min_days,
            int(round(len(unique_dates) * self.settings.model_settings.calibration.fraction)),
        )
        if len(unique_dates) <= calibration_days + 5:
            return frame, frame.iloc[0:0].copy()
        calibration_start = unique_dates[-calibration_days]
        fit_frame = frame.loc[frame["date"] < calibration_start].copy()
        calibration_frame = frame.loc[frame["date"] >= calibration_start].copy()
        if fit_frame.empty or calibration_frame.empty:
            return frame, frame.iloc[0:0].copy()
        return fit_frame, calibration_frame

    def _can_fit_calibrator(self, frame: pd.DataFrame) -> bool:
        if frame.empty:
            return False
        return set(frame["direction_label"].astype(int).tolist()) == {0, 1, 2}

    def _prepare_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.fill_values is None:
            raise RuntimeError("Model must be fit before predict.")
        return frame[self.feature_columns].fillna(self.fill_values).astype(float)

    def _predict_raw_probabilities(self, x: pd.DataFrame) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("Model must be fit before predict.")
        probabilities = np.asarray(self.classifier.predict_proba(x), dtype=float)
        if probabilities.ndim != 2 or probabilities.shape[1] != 3:
            raise RuntimeError("LightGBM classifier returned an unexpected probability shape.")
        return probabilities

    def _transform_probabilities(self, probability_matrix: np.ndarray) -> np.ndarray:
        clipped = np.clip(probability_matrix, 1e-6, 1 - 1e-6)
        return np.log(clipped) - np.log(clipped).mean(axis=1, keepdims=True)

    def _predict_probabilities(self, x: pd.DataFrame) -> np.ndarray:
        raw_probabilities = self._predict_raw_probabilities(x)
        if self.calibrator is None:
            return raw_probabilities
        transformed = self._transform_probabilities(raw_probabilities)
        calibrated_logits = self.calibrator.decision_function(transformed)
        calibrated_matrix = np.asarray(calibrated_logits, dtype=float)
        if calibrated_matrix.ndim == 1:
            calibrated_matrix = np.column_stack([-calibrated_matrix, np.zeros_like(calibrated_matrix), calibrated_matrix])
        return _safe_softmax(calibrated_matrix)

    def _sample_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        limit = self.settings.model_settings.lightgbm.max_shap_samples
        if len(frame) <= limit:
            return frame
        sampled = frame.sample(n=limit, random_state=self.settings.app.seed)
        return sampled.sort_values(["date", "ticker"]).reset_index(drop=True)

    def _raw_shap_values(self, frame: pd.DataFrame) -> np.ndarray:
        if self.classifier is None:
            raise RuntimeError("Model must be fit before predict.")
        x = self._prepare_features(frame)
        raw_values = self.classifier.booster_.predict(x, pred_contrib=True)
        feature_count = len(self.feature_columns)
        class_count = 3
        if isinstance(raw_values, list):
            matrices = [np.asarray(item, dtype=float)[:, :feature_count] for item in raw_values]
            return np.stack(matrices, axis=1)
        values = np.asarray(raw_values, dtype=float)
        if values.ndim == 2 and values.shape[1] == class_count * (feature_count + 1):
            reshaped = values.reshape(len(values), class_count, feature_count + 1)
            return reshaped[:, :, :feature_count]
        if values.ndim == 3 and values.shape[1] == class_count:
            return values[:, :, :feature_count]
        if values.ndim == 3 and values.shape[2] == class_count:
            return np.transpose(values[:, :feature_count, :], (0, 2, 1))
        raise RuntimeError("LightGBM SHAP values returned an unexpected shape.")

    def _predicted_class_shap_values(self, frame: pd.DataFrame, predicted_class: np.ndarray) -> np.ndarray:
        shap_values = self._raw_shap_values(frame)
        return shap_values[np.arange(len(frame)), predicted_class, :]

    def _compute_shap_importance(self, frame: pd.DataFrame) -> dict[str, float]:
        if frame.empty:
            return {}
        sampled = self._sample_frame(frame)
        probabilities = self._predict_probabilities(self._prepare_features(sampled))
        predicted_class = probabilities.argmax(axis=1)
        shap_values = self._predicted_class_shap_values(sampled, predicted_class)
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        total = float(np.sum(mean_abs))
        if total <= 0:
            uniform = 1.0 / max(len(self.feature_columns), 1)
            return {feature: uniform for feature in self.feature_columns}
        return {
            feature: float(mean_abs[index] / total)
            for index, feature in enumerate(self.feature_columns)
        }

