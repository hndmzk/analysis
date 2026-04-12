from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from market_prediction_agent.config import Settings
from market_prediction_agent.features.labels import CLASS_TO_LABEL
from market_prediction_agent.models.base import ForecastModel

try:
    import torch
    import torch.nn as torch_nn
    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.utils.data import TensorDataset as TorchTensorDataset
except ImportError as exc:  # pragma: no cover - exercised via runtime error path
    torch = None
    LSTM_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no branch
    LSTM_IMPORT_ERROR = None


def _date_key(value: object) -> str:
    return pd.Timestamp(value).date().isoformat()


@dataclass(slots=True)
class _SequenceBatch:
    features: np.ndarray
    metadata: pd.DataFrame
    direction_labels: np.ndarray | None = None
    target_returns: np.ndarray | None = None
    target_volatility: np.ndarray | None = None


if torch is None:
    LSTMDirectionModel = None
else:
    class _LSTMForecastNetwork(torch_nn.Module):
        def __init__(self, *, n_features: int, hidden_size: int, num_layers: int, dropout: float) -> None:
            super().__init__()
            effective_dropout = dropout if num_layers > 1 else 0.0
            self.lstm = torch_nn.LSTM(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=effective_dropout,
                batch_first=True,
            )
            self.direction_head = torch_nn.Linear(hidden_size, 3)
            self.return_head = torch_nn.Linear(hidden_size, 1)
            self.volatility_head = torch_nn.Linear(hidden_size, 1)

        def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            _, (hidden_state, _) = self.lstm(features)
            representation = hidden_state[-1]
            logits = self.direction_head(representation)
            expected_return = self.return_head(representation).squeeze(-1)
            predicted_volatility = self.volatility_head(representation).squeeze(-1)
            return logits, expected_return, predicted_volatility


    @dataclass(slots=True)
    class _LSTMDirectionModel(ForecastModel):
        settings: Settings
        version: str
        trained_at: str | None = None
        feature_columns: list[str] = field(default_factory=list)
        training_samples: int = 0
        fill_values: pd.Series | None = None
        feature_mean: pd.Series | None = None
        feature_std: pd.Series | None = None
        history_frame: pd.DataFrame | None = None
        network: _LSTMForecastNetwork | None = None
        calibration_summary: dict[str, Any] = field(
            default_factory=lambda: {
                "enabled": False,
                "method": "none",
                "samples": 0,
                "calibration_start": None,
                "calibration_end": None,
            }
        )
        global_feature_importance_map: dict[str, float] = field(default_factory=dict)

        def fit(self, frame: pd.DataFrame, feature_columns: list[str]) -> None:
            torch.manual_seed(self.settings.app.seed)
            ordered = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
            self.feature_columns = list(feature_columns)
            self.training_samples = int(len(ordered))
            self.fill_values = ordered[self.feature_columns].median()
            filled_features = ordered[self.feature_columns].fillna(self.fill_values).astype(float)
            self.feature_mean = filled_features.mean()
            self.feature_std = filled_features.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
            self.history_frame = ordered[["ticker", "date", "stale_data_flag", *self.feature_columns]].copy()

            sequence_batch = self._build_sequences(ordered, include_targets=True)
            if sequence_batch.direction_labels is None:
                raise RuntimeError("Direction labels are required for training.")
            if sequence_batch.target_returns is None or sequence_batch.target_volatility is None:
                raise RuntimeError("Regression targets are required for training.")
            if len(sequence_batch.features) < 2:
                raise ValueError("Not enough sequence windows to fit the LSTM model.")

            train_batch, validation_batch = self._split_train_validation(sequence_batch)
            self.network = _LSTMForecastNetwork(
                n_features=len(self.feature_columns),
                hidden_size=self.settings.model_settings.lstm.hidden_size,
                num_layers=self.settings.model_settings.lstm.num_layers,
                dropout=self.settings.model_settings.lstm.dropout,
            )
            optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=self.settings.model_settings.lstm.learning_rate,
            )
            direction_loss = torch_nn.CrossEntropyLoss()
            regression_loss = torch_nn.MSELoss()
            train_loader = self._build_data_loader(train_batch, shuffle=True)
            validation_loader = self._build_data_loader(validation_batch, shuffle=False)

            best_state: dict[str, Any] | None = None
            best_validation_loss = float("inf")
            epochs_without_improvement = 0
            for _ in range(self.settings.model_settings.lstm.max_epochs):
                self.network.train()
                for features, labels, target_returns, target_volatility in train_loader:
                    optimizer.zero_grad()
                    logits, predicted_returns, predicted_volatility = self.network(features)
                    loss = (
                        direction_loss(logits, labels)
                        + regression_loss(predicted_returns, target_returns)
                        + regression_loss(predicted_volatility, target_volatility)
                    )
                    loss.backward()
                    optimizer.step()

                validation_loss = self._evaluate_loss(
                    validation_loader=validation_loader,
                    direction_loss=direction_loss,
                    regression_loss=regression_loss,
                )
                if validation_loss < best_validation_loss - 1e-6:
                    best_validation_loss = validation_loss
                    best_state = copy.deepcopy(self.network.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.settings.model_settings.lstm.patience:
                        break

            if best_state is not None:
                self.network.load_state_dict(best_state)
            self.trained_at = datetime.now(timezone.utc).isoformat()
            self.global_feature_importance_map = {}

        def predict(self, frame: pd.DataFrame, include_explanations: bool = True) -> pd.DataFrame:
            if self.network is None:
                raise RuntimeError("Model must be fit before predict.")
            if frame.empty:
                return pd.DataFrame(
                    columns=[
                        "ticker",
                        "date",
                        "stale_data_flag",
                        "prob_down",
                        "prob_flat",
                        "prob_up",
                        "signal",
                        "direction",
                        "expected_return",
                        "predicted_volatility",
                        "confidence",
                        "top_features",
                    ]
                )
            combined = self._combine_history_with_frame(frame)
            endpoint_keys = [(str(row["ticker"]), _date_key(row["date"])) for _, row in frame.iterrows()]
            sequence_batch = self._build_sequences(combined, include_targets=False, endpoint_keys=endpoint_keys)
            self.network.eval()
            with torch.no_grad():
                features = torch.tensor(sequence_batch.features, dtype=torch.float32)
                logits, expected_return, predicted_volatility = self.network(features)
                probability_matrix = torch.softmax(logits, dim=1).cpu().numpy()
                expected_return_array = expected_return.cpu().numpy()
                predicted_volatility_array = np.maximum(0.0, predicted_volatility.cpu().numpy())

            predicted_class = probability_matrix.argmax(axis=1)
            max_prob = probability_matrix.max(axis=1)
            confidence = np.where(max_prob > 0.6, "high", np.where(max_prob > 0.45, "medium", "low"))

            predictions = sequence_batch.metadata.copy()
            predictions["prob_down"] = probability_matrix[:, 0]
            predictions["prob_flat"] = probability_matrix[:, 1]
            predictions["prob_up"] = probability_matrix[:, 2]
            predictions["signal"] = predictions["prob_up"] - predictions["prob_down"]
            predictions["direction"] = [CLASS_TO_LABEL[int(index)] for index in predicted_class]
            predictions["expected_return"] = expected_return_array
            predictions["predicted_volatility"] = predicted_volatility_array
            predictions["confidence"] = confidence
            predictions["top_features"] = [[] for _ in range(len(predictions))]
            if not include_explanations:
                predictions["top_features"] = [[] for _ in range(len(predictions))]
            return predictions

        def feature_importance_top(
            self,
            frame: pd.DataFrame | None = None,
            limit: int = 10,
        ) -> list[dict[str, float | str]]:
            del frame, limit
            return []

        def hyperparameters(self) -> dict[str, float | int | str]:
            return {
                "model_name": "lstm_direction",
                "hidden_size": self.settings.model_settings.lstm.hidden_size,
                "num_layers": self.settings.model_settings.lstm.num_layers,
                "dropout": self.settings.model_settings.lstm.dropout,
                "sequence_length": self.settings.model_settings.lstm.sequence_length,
                "max_epochs": self.settings.model_settings.lstm.max_epochs,
                "batch_size": self.settings.model_settings.lstm.batch_size,
                "learning_rate": self.settings.model_settings.lstm.learning_rate,
                "patience": self.settings.model_settings.lstm.patience,
            }

        def top_features(self, frame: pd.DataFrame, limit: int = 5) -> list[list[dict[str, float | str]]]:
            del limit
            return [[] for _ in range(len(frame))]

        def _combine_history_with_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
            if self.history_frame is None:
                raise RuntimeError("Model must be fit before predict.")
            expected_columns = ["ticker", "date", "stale_data_flag", *self.feature_columns]
            missing_columns = [column for column in expected_columns if column not in frame.columns]
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"Predict frame is missing required columns: {missing}")
            combined = pd.concat(
                [
                    self.history_frame,
                    frame[expected_columns],
                ],
                ignore_index=True,
            )
            combined = combined.drop_duplicates(subset=["ticker", "date"], keep="last")
            return combined.sort_values(["ticker", "date"]).reset_index(drop=True)

        def _build_sequences(
            self,
            frame: pd.DataFrame,
            *,
            include_targets: bool,
            endpoint_keys: list[tuple[str, str]] | None = None,
        ) -> _SequenceBatch:
            if self.fill_values is None or self.feature_mean is None or self.feature_std is None:
                raise RuntimeError("Model must be fit before building sequences.")

            required_columns = ["ticker", "date", "stale_data_flag", *self.feature_columns]
            if include_targets:
                required_columns.extend(["direction_label", "target_return", "future_volatility_20d"])
            missing_columns = [column for column in required_columns if column not in frame.columns]
            if missing_columns:
                missing = ", ".join(sorted(missing_columns))
                raise ValueError(f"Sequence frame is missing required columns: {missing}")

            prepared = frame[required_columns].copy()
            prepared["date"] = pd.to_datetime(prepared["date"])
            filled = prepared[self.feature_columns].fillna(self.fill_values).astype(float)
            prepared.loc[:, self.feature_columns] = ((filled - self.feature_mean) / self.feature_std).astype(np.float32)
            if include_targets:
                prepared["target_return"] = prepared["target_return"].fillna(0.0).astype(float)
                prepared["future_volatility_20d"] = (
                    prepared["future_volatility_20d"]
                    .fillna(prepared["future_volatility_20d"].median())
                    .fillna(0.0)
                    .astype(float)
                )
            prepared = prepared.sort_values(["ticker", "date"]).reset_index(drop=True)

            requested_keys = set(endpoint_keys or [])
            endpoint_order = {
                key: index
                for index, key in enumerate(endpoint_keys or [])
            }
            sequences: list[np.ndarray] = []
            metadata_rows: list[dict[str, Any]] = []
            direction_labels: list[int] = []
            target_returns: list[float] = []
            target_volatility: list[float] = []
            sequence_length = self.settings.model_settings.lstm.sequence_length

            for ticker, group in prepared.groupby("ticker", sort=False):
                ticker_group = group.reset_index(drop=True)
                if len(ticker_group) < sequence_length:
                    continue
                for end_index in range(sequence_length - 1, len(ticker_group)):
                    end_row = ticker_group.iloc[end_index]
                    key = (str(ticker), _date_key(end_row["date"]))
                    if requested_keys and key not in requested_keys:
                        continue
                    window = ticker_group.iloc[end_index - sequence_length + 1 : end_index + 1]
                    sequences.append(window[self.feature_columns].to_numpy(dtype=np.float32))
                    metadata_rows.append(
                        {
                            "ticker": str(ticker),
                            "date": pd.Timestamp(end_row["date"]),
                            "stale_data_flag": bool(end_row["stale_data_flag"]),
                            "_order": endpoint_order.get(key, len(metadata_rows)),
                        }
                    )
                    if include_targets:
                        direction_labels.append(int(end_row["direction_label"]))
                        target_returns.append(float(end_row["target_return"]))
                        target_volatility.append(float(end_row["future_volatility_20d"]))

            if not sequences:
                raise ValueError("Not enough history to build sequence windows for the requested rows.")

            metadata = pd.DataFrame(metadata_rows)
            order = metadata.sort_values("_order").index.to_numpy(dtype=int)
            metadata = metadata.iloc[order].drop(columns="_order").reset_index(drop=True)
            if endpoint_keys is not None and len(metadata) != len(endpoint_keys):
                raise ValueError("Not enough history to build sequence windows for every requested prediction row.")

            features = np.asarray(sequences, dtype=np.float32)[order]
            batch = _SequenceBatch(features=features, metadata=metadata)
            if include_targets:
                batch.direction_labels = np.asarray(direction_labels, dtype=np.int64)[order]
                batch.target_returns = np.asarray(target_returns, dtype=np.float32)[order]
                batch.target_volatility = np.asarray(target_volatility, dtype=np.float32)[order]
            return batch

        def _split_train_validation(self, batch: _SequenceBatch) -> tuple[_SequenceBatch, _SequenceBatch]:
            train_size = max(1, int(len(batch.features) * 0.8))
            train_size = min(train_size, len(batch.features) - 1)
            train_batch = _SequenceBatch(
                features=batch.features[:train_size],
                metadata=batch.metadata.iloc[:train_size].reset_index(drop=True),
                direction_labels=batch.direction_labels[:train_size] if batch.direction_labels is not None else None,
                target_returns=batch.target_returns[:train_size] if batch.target_returns is not None else None,
                target_volatility=batch.target_volatility[:train_size] if batch.target_volatility is not None else None,
            )
            validation_batch = _SequenceBatch(
                features=batch.features[train_size:],
                metadata=batch.metadata.iloc[train_size:].reset_index(drop=True),
                direction_labels=batch.direction_labels[train_size:] if batch.direction_labels is not None else None,
                target_returns=batch.target_returns[train_size:] if batch.target_returns is not None else None,
                target_volatility=batch.target_volatility[train_size:] if batch.target_volatility is not None else None,
            )
            return train_batch, validation_batch

        def _build_data_loader(self, batch: _SequenceBatch, *, shuffle: bool) -> TorchDataLoader:
            if batch.direction_labels is None or batch.target_returns is None or batch.target_volatility is None:
                raise RuntimeError("Training batches require classification and regression targets.")
            dataset = TorchTensorDataset(
                torch.tensor(batch.features, dtype=torch.float32),
                torch.tensor(batch.direction_labels, dtype=torch.long),
                torch.tensor(batch.target_returns, dtype=torch.float32),
                torch.tensor(batch.target_volatility, dtype=torch.float32),
            )
            return TorchDataLoader(
                dataset,
                batch_size=min(self.settings.model_settings.lstm.batch_size, len(dataset)),
                shuffle=shuffle,
            )

        def _evaluate_loss(
            self,
            *,
            validation_loader: TorchDataLoader,
            direction_loss: torch_nn.CrossEntropyLoss,
            regression_loss: torch_nn.MSELoss,
        ) -> float:
            if self.network is None:
                raise RuntimeError("Model must be fit before validation.")
            self.network.eval()
            losses: list[float] = []
            with torch.no_grad():
                for features, labels, target_returns, target_volatility in validation_loader:
                    logits, predicted_returns, predicted_volatility = self.network(features)
                    loss = (
                        direction_loss(logits, labels)
                        + regression_loss(predicted_returns, target_returns)
                        + regression_loss(predicted_volatility, target_volatility)
                    )
                    losses.append(float(loss.item()))
            if not losses:
                raise ValueError("Validation split is empty; cannot evaluate early stopping.")
            return float(np.mean(losses))


    LSTMDirectionModel = _LSTMDirectionModel


__all__ = ["LSTMDirectionModel", "LSTM_IMPORT_ERROR", "torch"]

