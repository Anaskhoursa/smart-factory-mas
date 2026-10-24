"""
Loads the two trained checkpoints (RUL LSTM, anomaly LSTM-autoencoder)
and exposes a PredictiveEngine that consumes the live factory
simulation one SensorReading at a time, per machine, and returns
RUL / anomaly predictions once each machine has enough history.

This module intentionally re-declares the model architectures
(RULLSTM, LSTMAutoencoder) rather than importing them from the
training scripts, since those scripts are meant to be run standalone
(python train_rul_lstm.py) and importing them would re-trigger their
argparse/__main__ setup. Keep the class definitions here in sync with
train_rul_lstm.py / train_anomaly_lstm.py if you change the
architecture.
"""

from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

# --------------------------------------------------
# Constants -- must match the training scripts exactly
# --------------------------------------------------

RAW_SENSORS = ["vibration", "temperature", "throughput", "error_rate"]
ROLLING_FEATURES = (
    [f"{c}_roll_mean" for c in RAW_SENSORS]
    + [f"{c}_roll_std" for c in RAW_SENSORS]
    + [f"{c}_trend" for c in RAW_SENSORS]
)
RUL_FEATURES = RAW_SENSORS + ROLLING_FEATURES  # 12 columns, scaled
MACHINE_IDS = ["M1", "M2", "M3", "M4"]  # fixed order, must match training

ROLLING_WINDOW = 10   # must match --rolling-window used in generate_dataset.py
RUL_WINDOW_SIZE = 30  # must match WINDOW_SIZE in train_rul_lstm.py
ANOMALY_WINDOW_SIZE = 30  # must match WINDOW_SIZE in train_anomaly_lstm.py

# How much raw history each machine needs before the rolling features
# (which look back ROLLING_WINDOW steps) are valid for every row of a
# full RUL_WINDOW_SIZE window.
RUL_HISTORY_NEEDED = RUL_WINDOW_SIZE + ROLLING_WINDOW


# --------------------------------------------------
# Model architectures (must match the training scripts)
# --------------------------------------------------

class RULLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        sequence_output, _ = self.lstm(x)
        final_output = sequence_output[:, -1, :]
        return self.regression_head(final_output).squeeze(-1)


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_size=32, num_layers=1, window_size=ANOMALY_WINDOW_SIZE):
        super().__init__()
        self.window_size = window_size
        self.encoder = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        bottleneck = hidden[-1]
        repeated = bottleneck.unsqueeze(1).repeat(1, self.window_size, 1)
        decoded_seq, _ = self.decoder(repeated)
        return self.output_layer(decoded_seq)


# --------------------------------------------------
# Live rolling-feature computation
# --------------------------------------------------

def compute_rolling_features(buffer_df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """Same formulas as generate_dataset.py's add_rolling_features,
    applied to a short live buffer instead of a full offline episode.
    Only the LAST RUL_WINDOW_SIZE rows of the result should be used as
    the model input -- the first `window` rows are a warmup region
    where the rolling stats are based on less history than at
    training time (see PredictiveEngine docstring)."""
    df = buffer_df.copy()
    for col in RAW_SENSORS:
        series = df[col]
        df[f"{col}_roll_mean"] = series.rolling(window=window, min_periods=1).mean()
        df[f"{col}_roll_std"] = series.rolling(window=window, min_periods=1).std().fillna(0.0)
        shifted = series.shift(window)
        shifted = shifted.fillna(series.iloc[0])
        df[f"{col}_trend"] = (series - shifted) / window
    return df


# --------------------------------------------------
# PredictiveEngine
# --------------------------------------------------

class PredictiveEngine:
    """Wraps both trained models and maintains a rolling per-machine
    buffer of raw sensor readings, so main.py can call `update` once
    per machine per simulation step and `predict` to get the current
    RUL estimate and anomaly flag.

    Both models need a few steps of warm-up history before they
    return a prediction (RUL needs RUL_HISTORY_NEEDED steps, anomaly
    needs ANOMALY_WINDOW_SIZE steps) -- `predict` returns None for
    whichever isn't ready yet, rather than guessing.
    """

    def __init__(
        self,
        rul_checkpoint_path: Path | str | None,
        anomaly_checkpoint_path: Path | str | None,
        device: str | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.rul_model = None
        self.rul_scaler_mean = None
        self.rul_scaler_scale = None
        self.rul_cap = None
        if rul_checkpoint_path is not None:
            self._load_rul_model(Path(rul_checkpoint_path))

        self.anomaly_model = None
        self.anomaly_scaler_mean = None
        self.anomaly_scaler_scale = None
        self.anomaly_threshold = None
        if anomaly_checkpoint_path is not None:
            self._load_anomaly_model(Path(anomaly_checkpoint_path))

        # Per-machine raw sensor history, long enough to serve both models.
        buffer_len = max(RUL_HISTORY_NEEDED, ANOMALY_WINDOW_SIZE)
        self._buffers: dict[str, deque] = {}
        self._buffer_len = buffer_len

    def _load_rul_model(self, path: Path):
        if not path.exists():
            print(f"[PredictiveEngine] WARNING: RUL checkpoint not found at {path}, "
                  f"RUL predictions will be disabled.")
            return
        checkpoint = torch.load(path, map_location=self.device)
        n_features = len(checkpoint["features"])  # RUL_FEATURES + one-hot
        model = RULLSTM(
            input_size=n_features,
            hidden_size=checkpoint.get("hidden_size", 64),
            num_layers=checkpoint.get("num_layers", 2),
            dropout=checkpoint.get("dropout", 0.2),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        self.rul_model = model
        self.rul_scaler_mean = np.array(checkpoint["scaler_mean"], dtype=np.float32)
        self.rul_scaler_scale = np.array(checkpoint["scaler_scale"], dtype=np.float32)
        self.rul_cap = checkpoint.get("rul_cap", 400)
        print(f"[PredictiveEngine] Loaded RUL model from {path} (best_epoch={checkpoint.get('best_epoch')})")

    def _load_anomaly_model(self, path: Path):
        if not path.exists():
            print(f"[PredictiveEngine] WARNING: anomaly checkpoint not found at {path}, "
                  f"anomaly predictions will be disabled.")
            return
        checkpoint = torch.load(path, map_location=self.device)
        model = LSTMAutoencoder(
            n_features=len(checkpoint["features"]),
            hidden_size=checkpoint.get("hidden_size", 32),
            num_layers=checkpoint.get("num_layers", 1),
            window_size=checkpoint.get("window_size", ANOMALY_WINDOW_SIZE),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        self.anomaly_model = model
        self.anomaly_scaler_mean = np.array(checkpoint["scaler_mean"], dtype=np.float32)
        self.anomaly_scaler_scale = np.array(checkpoint["scaler_scale"], dtype=np.float32)
        self.anomaly_threshold = checkpoint["threshold"]
        print(f"[PredictiveEngine] Loaded anomaly model from {path} "
              f"(threshold={self.anomaly_threshold:.5f}, best_epoch={checkpoint.get('best_epoch')})")

    def update(self, machine_id: str, reading: dict) -> None:
        """Append one step's sensor reading for a machine. `reading`
        is a plain mapping with the four raw sensor keys (vibration,
        temperature, throughput, error_rate) -- deliberately NOT a
        SensorReading/MachineTelemetry object, so this module has no
        import dependency on the simulator package and can be reused
        by anything that can produce a dict (the MQTT agent, a
        notebook, a different simulator, etc.)."""
        if machine_id not in self._buffers:
            self._buffers[machine_id] = deque(maxlen=self._buffer_len)
        self._buffers[machine_id].append(
            {
                "vibration": float(reading["vibration"]),
                "temperature": float(reading["temperature"]),
                "throughput": float(reading["throughput"]),
                "error_rate": float(reading["error_rate"]),
            }
        )

    def _scale(self, values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        return (values - mean) / scale

    def _predict_rul(self, machine_id: str, buffer_rows: list[dict]) -> float | None:
        if self.rul_model is None or len(buffer_rows) < RUL_HISTORY_NEEDED:
            return None

        buffer_df = pd.DataFrame(buffer_rows)
        featured = compute_rolling_features(buffer_df, window=ROLLING_WINDOW)
        window_df = featured.iloc[-RUL_WINDOW_SIZE:]  # drop the rolling-warmup rows

        raw_features = window_df[RUL_FEATURES].to_numpy(dtype=np.float32)
        scaled = self._scale(raw_features, self.rul_scaler_mean, self.rul_scaler_scale)

        one_hot = np.zeros(len(MACHINE_IDS), dtype=np.float32)
        if machine_id in MACHINE_IDS:
            one_hot[MACHINE_IDS.index(machine_id)] = 1.0
        one_hot_tiled = np.tile(one_hot, (RUL_WINDOW_SIZE, 1))

        model_input = np.concatenate([scaled, one_hot_tiled], axis=1)
        tensor = torch.from_numpy(model_input).unsqueeze(0).to(self.device)  # (1, window, features)

        with torch.no_grad():
            prediction = self.rul_model(tensor).item()

        return float(np.clip(prediction, 0.0, self.rul_cap))

    def _predict_anomaly(self, buffer_rows: list[dict]) -> tuple[float, bool] | tuple[None, None]:
        if self.anomaly_model is None or len(buffer_rows) < ANOMALY_WINDOW_SIZE:
            return None, None

        window_rows = buffer_rows[-ANOMALY_WINDOW_SIZE:]
        raw_features = pd.DataFrame(window_rows)[RAW_SENSORS].to_numpy(dtype=np.float32)
        scaled = self._scale(raw_features, self.anomaly_scaler_mean, self.anomaly_scaler_scale)

        tensor = torch.from_numpy(scaled.astype(np.float32)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            reconstruction = self.anomaly_model(tensor)
            error = torch.mean((reconstruction - tensor) ** 2).item()

        return float(error), bool(error > self.anomaly_threshold)

    def predict(self, machine_id: str) -> dict:
        """Returns a dict with whatever predictions are currently
        available for this machine. Values are None until that
        model's warmup history has been accumulated."""
        buffer_rows = list(self._buffers.get(machine_id, []))

        rul = self._predict_rul(machine_id, buffer_rows)
        anomaly_error, is_anomaly = self._predict_anomaly(buffer_rows)

        return {
            "predicted_rul": rul,
            "anomaly_score": anomaly_error,
            "is_anomaly": is_anomaly,
        }