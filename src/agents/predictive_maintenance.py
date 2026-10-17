"""MQTT-driven predictive-maintenance agent for the factory simulator.

Run from the repository root:
    python -m agents.predictive_maintenance

The service consumes factory/machines/+/telemetry, maintains independent
per-machine windows, runs the existing RUL LSTM and anomaly LSTM-autoencoder,
and publishes JSON reports to factory/maintenance/predictions.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import paho.mqtt.client as mqtt
import torch
from pydantic import BaseModel, Field, ValidationError
from torch import nn

LOG = logging.getLogger("predictive-agent")
ROOT = Path(__file__).resolve().parents[2]
RUL_CHECKPOINT = Path(os.getenv(
    "RUL_CHECKPOINT", ROOT / "predictive/artifacts/lstm_factory/best_lstm.pt"
))
ANOMALY_CHECKPOINT = Path(os.getenv(
    "ANOMALY_CHECKPOINT", ROOT / "predictive/artifacts/anomaly_lstm/best_autoencoder.pt"
))
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
TELEMETRY_TOPIC = "factory/machines/+/telemetry"
PREDICTION_TOPIC = "factory/maintenance/predictions"

SENSORS = ["vibration", "temperature", "throughput", "error_rate"]
MACHINE_IDS = ["M1", "M2", "M3", "M4"]
ROLLING_WINDOW = int(os.getenv("ROLLING_WINDOW", "10"))
WINDOW_SIZE = 30


class RULLSTM(nn.Module):
    """Architecture kept in sync with predictive/train_lstm.py."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.regression_head(output[:, -1, :]).squeeze(-1)


class LSTMAutoencoder(nn.Module):
    """Architecture kept in sync with predictive/train_anomaly_lstm.py."""

    def __init__(self, n_features: int, hidden_size: int, num_layers: int, window_size: int):
        super().__init__()
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.encoder = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.encoder(x)
        bottleneck = hidden[-1]
        repeated = bottleneck.unsqueeze(1).repeat(1, self.window_size, 1)
        decoded, _ = self.decoder(repeated)
        return self.output_layer(decoded)


class SensorPayload(BaseModel):
    machine_id: str
    timestamp: datetime | None = None
    vibration: float = Field(ge=0)
    temperature: float = Field(ge=0)
    throughput: float = Field(ge=0)
    error_rate: float = Field(ge=0, le=1)


class PredictionReport(BaseModel):
    schema_version: str = "1.0"
    agent: str = "predictive_maintenance"
    machine_id: str
    timestamp: datetime
    sequence: int
    rul_steps: float
    anomaly_score: float
    anomaly_threshold: float
    is_anomaly: bool
    severity: str
    recommendation: str
    evidence: list[str]
    models: dict[str, str]


class ModelBundle:
    def __init__(self, rul_path: Path, anomaly_path: Path, device: torch.device):
        for path in (rul_path, anomaly_path):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Model artifact not found: {path}. Train the models first."
                )
        self.device = device
        # Checkpoints are project-generated artifacts; use weights_only to avoid
        # unpickling arbitrary Python objects.
        rul = torch.load(rul_path, map_location=device, weights_only=True)
        anomaly = torch.load(anomaly_path, map_location=device, weights_only=True)

        self.rul_features = list(rul["features"])
        self.rul_window = int(rul["window_size"])
        self.rul_cap = float(rul["rul_cap"])
        self.rul_mean = np.asarray(rul["scaler_mean"], dtype=np.float32)
        self.rul_scale = np.asarray(rul["scaler_scale"], dtype=np.float32)
        self.rul_model = RULLSTM(
            input_size=len(self.rul_features),
            hidden_size=int(rul["hidden_size"]),
            num_layers=int(rul["num_layers"]),
            dropout=float(rul["dropout"]),
        ).to(device)
        self.rul_model.load_state_dict(rul["model_state_dict"])
        self.rul_model.eval()

        self.anomaly_features = list(anomaly["features"])
        self.anomaly_window = int(anomaly["window_size"])
        self.anomaly_mean = np.asarray(anomaly["scaler_mean"], dtype=np.float32)
        self.anomaly_scale = np.asarray(anomaly["scaler_scale"], dtype=np.float32)
        self.anomaly_threshold = float(anomaly["threshold"])
        self.anomaly_model = LSTMAutoencoder(
            n_features=len(self.anomaly_features),
            hidden_size=int(anomaly["hidden_size"]),
            num_layers=int(anomaly["num_layers"]),
            window_size=self.anomaly_window,
        ).to(device)
        self.anomaly_model.load_state_dict(anomaly["model_state_dict"])
        self.anomaly_model.eval()

    def predict(self, rows: list[dict[str, float]], machine_id: str) -> tuple[float, float]:
        if len(rows) < max(self.rul_window, self.anomaly_window):
            raise ValueError("Not enough telemetry to form both model windows")
        raw = np.asarray([[row[s] for s in SENSORS] for row in rows], dtype=np.float32)

        # Match generate_dataset.py: rolling features use window=10 by default,
        # pandas sample std (ddof=1), and (current - value_window_steps_ago)/window.
        engineered = []
        for idx, row in enumerate(rows):
            start = max(0, idx - ROLLING_WINDOW + 1)
            segment = rows[start:idx + 1]
            features = {s: float(row[s]) for s in SENSORS}
            for sensor in SENSORS:
                vals = np.asarray([item[sensor] for item in segment], dtype=np.float32)
                features[f"{sensor}_roll_mean"] = float(vals.mean())
                features[f"{sensor}_roll_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                baseline_idx = max(0, idx - ROLLING_WINDOW)
                baseline = rows[baseline_idx][sensor]
                features[f"{sensor}_trend"] = float((row[sensor] - baseline) / ROLLING_WINDOW)
            engineered.append(features)

        # RUL model expects sensor features followed by the fixed M1..M4 one-hot flags.
        rul_values = []
        for row in engineered[-self.rul_window:]:
            values = []
            for feature in self.rul_features:
                if feature in row:
                    values.append(row[feature])
                elif feature.startswith("is_"):
                    values.append(float(feature == f"is_{machine_id}"))
                else:
                    raise ValueError(f"Unsupported RUL feature in checkpoint: {feature}")
            rul_values.append(values)
        rul_array = np.asarray(rul_values, dtype=np.float32)
        if len(rul_array) < self.rul_window:
            pad = np.repeat(rul_array[:1], self.rul_window - len(rul_array), axis=0)
            rul_array = np.concatenate([pad, rul_array], axis=0)
        # The training scaler was fitted only on the 12 continuous sensor-derived\n        # features; machine one-hot flags are concatenated afterwards and stay 0/1.\n        continuous_indices = [i for i, name in enumerate(self.rul_features) if not name.startswith("is_")]\n        rul_array[:, continuous_indices] = (\n            rul_array[:, continuous_indices] - self.rul_mean\n        ) / np.maximum(self.rul_scale, 1e-8)

        # Anomaly model has its own raw-sensor scaler and its own threshold.
        anomaly_raw = raw[-self.anomaly_window:]
        anomaly_scaled = (anomaly_raw - self.anomaly_mean) / np.maximum(self.anomaly_scale, 1e-8)

        with torch.inference_mode():
            x_rul = torch.from_numpy(rul_array[None, ...]).to(self.device)
            rul = float(self.rul_model(x_rul).item())
            x_anomaly = torch.from_numpy(anomaly_scaled[None, ...]).to(self.device)
            reconstructed = self.anomaly_model(x_anomaly)
            score = float(torch.mean((reconstructed - x_anomaly) ** 2, dim=(1, 2)).item())
        return float(np.clip(rul, 0.0, self.rul_cap)), score


class PredictiveMaintenanceAgent:
    def __init__(self, models: ModelBundle):
        self.models = models
        self.history: dict[str, deque[dict[str, float]]] = defaultdict(
            lambda: deque(maxlen=max(WINDOW_SIZE, ROLLING_WINDOW) + ROLLING_WINDOW)
        )
        self.sequence: dict[str, int] = defaultdict(int)
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="predictive-maintenance-agent",
            protocol=mqtt.MQTTv5,
        )
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        if reason_code.is_failure:
            LOG.error("MQTT connection failed: %s", reason_code)
            return
        LOG.info("Connected to MQTT broker %s:%s", MQTT_HOST, MQTT_PORT)
        client.subscribe(TELEMETRY_TOPIC, qos=1)

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        LOG.warning("Disconnected from MQTT: %s", reason_code)

    def _on_message(self, client, userdata, message):
        try:
            payload = json.loads(message.payload.decode("utf-8"))
            # Supports either the simulator's nested MachineTelemetry JSON or a
            # flat SensorReading JSON. Never use health/wear/status as model input.
            sensor_data = payload.get("sensors", payload)
            machine_id = payload.get("machine_id") or sensor_data.get("machine_id")
            if not machine_id:
                machine_id = message.topic.split("/")[-2]
            if machine_id not in MACHINE_IDS:
                raise ValueError(f"Unknown machine_id {machine_id!r}; expected {MACHINE_IDS}")
            reading = SensorPayload(
                machine_id=machine_id,
                timestamp=payload.get("timestamp") or sensor_data.get("timestamp"),
                **{key: sensor_data[key] for key in SENSORS},
            )
            row = {key: float(getattr(reading, key)) for key in SENSORS}
            self.history[machine_id].append(row)
            self.sequence[machine_id] += 1

            needed = max(self.models.rul_window, self.models.anomaly_window)
            if len(self.history[machine_id]) < needed:
                LOG.info("%s warming up model window: %d/%d", machine_id, len(self.history[machine_id]), needed)
                return

            rows = list(self.history[machine_id])
            rul_steps, anomaly_score = self.models.predict(rows, machine_id)
            is_anomaly = anomaly_score > self.models.anomaly_threshold
            severity, recommendation, evidence = self._assess(rul_steps, is_anomaly, anomaly_score)
            report = PredictionReport(
                machine_id=machine_id,
                timestamp=reading.timestamp or datetime.now(timezone.utc),
                sequence=self.sequence[machine_id],
                rul_steps=round(rul_steps, 3),
                anomaly_score=round(anomaly_score, 8),
                anomaly_threshold=self.models.anomaly_threshold,
                is_anomaly=is_anomaly,
                severity=severity,
                recommendation=recommendation,
                evidence=evidence,
                models={"rul": RUL_CHECKPOINT.name, "anomaly": ANOMALY_CHECKPOINT.name},
            )
            client.publish(PREDICTION_TOPIC, report.model_dump_json(), qos=1, retain=False)
            LOG.warning("Prediction published: %s", report.model_dump_json())
        except (ValueError, KeyError, TypeError, ValidationError, json.JSONDecodeError) as exc:
            LOG.exception("Rejected telemetry message on %s: %s", message.topic, exc)
        except Exception:
            LOG.exception("Prediction failed for message on %s", message.topic)

    def _assess(self, rul_steps: float, is_anomaly: bool, anomaly_score: float):
        evidence = [f"Predicted RUL: {rul_steps:.1f} simulation steps"]
        if is_anomaly:
            evidence.append(
                f"Anomaly reconstruction error {anomaly_score:.6g} exceeds calibrated threshold "
                f"{self.models.anomaly_threshold:.6g}"
            )
        if rul_steps <= 10:
            return "critical", "Request urgent maintenance review; supervisor approval required for shutdown.", evidence
        if is_anomaly and rul_steps <= 50:
            return "critical", "Request urgent maintenance assessment and notify scheduling agent.", evidence
        if is_anomaly or rul_steps <= 50:
            return "high", "Request maintenance scheduling assessment; do not stop the machine automatically.", evidence
        if rul_steps <= 100:
            return "warning", "Monitor closely and ask scheduling agent to evaluate a maintenance window.", evidence
        return "normal", "Continue monitoring; no maintenance action requested.", evidence

    def run(self):
        self.client.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
        self.client.loop_forever()


def main():
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOG.info("Loading predictive models on %s", device)
    models = ModelBundle(RUL_CHECKPOINT, ANOMALY_CHECKPOINT, device)
    agent = PredictiveMaintenanceAgent(models)
    agent.run()


if __name__ == "__main__":
    main()
