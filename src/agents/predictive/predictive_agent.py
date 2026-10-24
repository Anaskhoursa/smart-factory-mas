"""Predictive Maintenance Agent.

Receives per-machine telemetry over MQTT, runs the RUL LSTM and the
anomaly LSTM-autoencoder, and publishes the resulting RUL and anomaly
values back on the factory/v1/health/* topics every step (once each
model has warmed up).

LLM assessments are called at most once per (machine, band) and only
after the band has persisted for STREAK_THRESHOLD consecutive steps.
The band, not the raw value, is what the memory keys on — see
health_state.py and assessment_memory.py. An assessment is attached
only to the message published in the step it was generated; the memory
exists solely to suppress re-calls.

Threading model (unchanged from the previous version, and essential):
paho's on_message callback runs on the client's network thread, which
is also responsible for reading the next packet — including any PUBACK
a blocking publish() would be waiting for. on_message therefore does
the minimum possible work: it drops the raw message onto a queue.Queue
and returns. All model inference, LLM calls, and publishes happen on
the main thread's run loop.
"""
from __future__ import annotations

import argparse
import json
import logging
import logging.handlers
import os
import queue
import time
from datetime import datetime, timezone
from pathlib import Path

from messaging.client import MQTTClient
from messaging.subscriptions import subscriptions_for, QOS_EVENTS
from messaging import topics

from predictive_models import PredictiveEngine

from assessment_memory import AssessmentMemory
from health_state import HealthState
from llm_assessor import PredictiveLLMAssessor

logger = logging.getLogger("predictive_agent")

TREND_WINDOW = 10
STREAK_THRESHOLD = 3


# --------------------------------------------------
# Logging
# --------------------------------------------------

def setup_logging(
    log_file: str | Path,
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    console: bool = True,
) -> None:
    logger.setLevel(level.upper())
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info(
        "Logging to %s (rotating at %d MB, keeping %d backups)",
        log_path.resolve(), max_bytes // (1024 * 1024), backup_count,
    )


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def extract_machine_id(topic: str) -> str | None:
    parts = topic.split("/")
    if "machines" in parts:
        idx = parts.index("machines")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def sensor_trend_summary(buffer_rows: list[dict], window: int = TREND_WINDOW) -> str:
    if len(buffer_rows) < 2:
        return "insufficient history"
    recent = buffer_rows[-window:]
    lines = []
    for key in ("vibration", "temperature", "throughput", "error_rate"):
        start = recent[0][key]
        end = recent[-1][key]
        delta = end - start
        threshold = 0.01 * (abs(start) + 1e-6)
        direction = "rising" if delta > threshold else (
            "falling" if delta < -threshold else "stable"
        )
        lines.append(f"{key}: {direction} ({start:.3f} -> {end:.3f})")
    return "; ".join(lines)


# --------------------------------------------------
# Agent
# --------------------------------------------------

class PredictiveAgent:
    def __init__(
        self,
        rul_model_path: str | None,
        anomaly_model_path: str | None,
        mqtt_host: str = "localhost",
        mqtt_port: int = 1883,
        groq_api_key: str | None = None,
        enable_llm: bool = True,
    ):
        self.engine = PredictiveEngine(
            rul_checkpoint_path=rul_model_path,
            anomaly_checkpoint_path=anomaly_model_path,
        )

        self.assessor = (
            PredictiveLLMAssessor(api_key=groq_api_key) if enable_llm else None
        )
        if self.assessor is not None and not self.assessor.enabled:
            logger.info("Predictive LLM assessor constructed but disabled "
                        "(missing key or package); continuing rules-free.")

        self.memory = AssessmentMemory(streak_threshold=STREAK_THRESHOLD)
        self._step_counter: dict[str, int] = {}
        self._messages_received = 0

        if self.engine.rul_model is None and self.engine.anomaly_model is None:
            logger.warning("*** NO MODELS LOADED *** -- this agent will receive "
                           "telemetry but will never publish or log a prediction.")

        self._queue: "queue.Queue[tuple[str, bytes, int]]" = queue.Queue()
        self.mqtt = MQTTClient(
            client_id="predictive-agent",
            host=mqtt_host,
            port=mqtt_port,
            on_message=self._enqueue_message,
        )

    # --- lifecycle ---

    def start(self):
        self.mqtt.connect()
        for topic, qos in subscriptions_for("predictive"):
            self.mqtt.subscribe(topic, qos=qos)
            logger.info("Subscribed to %s (qos=%s)", topic, qos)

    def stop(self):
        self.mqtt.disconnect()

    # --- network thread: enqueue only ---

    def _enqueue_message(self, topic: str, payload: bytes, qos: int):
        self._queue.put((topic, payload, qos))

    # --- main thread: all real work ---

    def run_forever(self):
        logger.info("Running. Ctrl+C to stop.")
        try:
            while True:
                topic, payload, qos = self._queue.get()
                try:
                    self._handle_message(topic, payload, qos)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Error handling message on %s: %s", topic, exc)
        except KeyboardInterrupt:
            logger.info("Shutting down.")
            self.stop()

    def _handle_message(self, topic: str, payload: bytes, qos: int):
        machine_id = extract_machine_id(topic)
        if machine_id is None:
            return

        try:
            data = json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            logger.warning("Could not parse payload on %s: %s", topic, exc)
            return

        sensors = data.get("sensors")
        if sensors is None:
            return

        self._messages_received += 1
        self._step_counter[machine_id] = self._step_counter.get(machine_id, 0) + 1
        step = self._step_counter[machine_id]

        if step == 1 or step % 50 == 0:
            logger.info("[%s] telemetry received (step %d)", machine_id, step)

        self.engine.update(machine_id, sensors)
        prediction = self.engine.predict(machine_id)
        state = HealthState.from_prediction(prediction)

        self.memory.update_streaks(
            machine_id,
            rul_streak_continues=state.rul_streak_continues(),
            anomaly_streak_continues=state.anomaly_streak_continues(),
        )

        self._publish_health(machine_id, prediction, state, step)

    def _publish_health(self, machine_id: str, prediction: dict, state: HealthState, step: int):
        now = datetime.now(timezone.utc).isoformat()

        rul_assessment = self._maybe_assess_rul(machine_id, state)
        anomaly_assessment = self._maybe_assess_anomaly(machine_id, state)

        if state.rul is not None:
            rul_message = {
                "machine_id": machine_id,
                "timestamp": now,
                "predicted_rul": state.rul,
                "rul_cap": self.engine.rul_cap,
                "rul_band": state.rul_band,
                "assessment": rul_assessment,
            }
            self.mqtt.publish(topics.V1_HEALTH_RUL, json.dumps(rul_message), qos=QOS_EVENTS)

        if state.anomaly_score is not None:
            anomaly_message = {
                "machine_id": machine_id,
                "timestamp": now,
                "anomaly_score": state.anomaly_score,
                "is_anomaly": state.is_anomaly,
                "anomaly_flag": state.anomaly_flag,
                "threshold": self.engine.anomaly_threshold,
                "assessment": anomaly_assessment,
            }
            self.mqtt.publish(topics.V1_HEALTH_ANOMALY, json.dumps(anomaly_message), qos=QOS_EVENTS)

        rul_str = f"{state.rul:.1f}" if state.rul is not None else "warming up"
        score_str = f"{state.anomaly_score:.5f}" if state.anomaly_score is not None else "warming up"
        flag = " [ANOMALY]" if state.is_anomaly else ""
        assessed = []
        if rul_assessment is not None:
            assessed.append("rul")
        if anomaly_assessment is not None:
            assessed.append("anomaly")
        assessed_str = f" assessed={','.join(assessed)}" if assessed else ""
        logger.info(
            "[%s] step=%d RUL=%s band=%s anomaly_score=%s flag=%s%s%s",
            machine_id, step, rul_str, state.rul_band, score_str, state.anomaly_flag, flag, assessed_str,
        )
    def _maybe_assess_rul(self, machine_id: str, state: HealthState) -> dict | None:
        if self.assessor is None or not self.assessor.enabled:
            return None
        if not self.memory.should_assess_rul(machine_id, state.rul_band):
            return None

        # Count the attempt before calling the model, so a failure still
        # consumes one of the bounded retries. Without this, a failing LLM
        # would be re-called on every step until the band changes.
        self.memory.record_rul_attempt(machine_id, state.rul_band)

        trend = sensor_trend_summary(list(self.engine._buffers.get(machine_id, [])))
        assessment = self.assessor.assess_rul(
            machine_id=machine_id,
            rul=state.rul,
            rul_band=state.rul_band,
            rul_cap=self.engine.rul_cap,
            trend_summary=trend,
        )
        if assessment is None:
            logger.warning(
                "[%s] RUL assessment failed for band=%s (attempt %d/%d)",
                machine_id, state.rul_band,
                self.memory.snapshot(machine_id)["rul_attempts"].get(state.rul_band, 0),
                3,
            )
            return None
        self.memory.record_rul_assessment(machine_id, state.rul_band, assessment)
        logger.info("[%s] RUL assessment generated for band=%s", machine_id, state.rul_band)
        return assessment

    def _maybe_assess_anomaly(self, machine_id: str, state: HealthState) -> dict | None:
        if self.assessor is None or not self.assessor.enabled:
            return None
        if not self.memory.should_assess_anomaly(machine_id, state.anomaly_flag):
            return None

        self.memory.record_anomaly_attempt(machine_id, state.anomaly_flag)

        trend = sensor_trend_summary(list(self.engine._buffers.get(machine_id, [])))
        assessment = self.assessor.assess_anomaly(
            machine_id=machine_id,
            anomaly_score=state.anomaly_score,
            threshold=self.engine.anomaly_threshold,
            is_anomaly=state.is_anomaly,
            anomaly_flag=state.anomaly_flag,
            trend_summary=trend,
        )
        if assessment is None:
            logger.warning(
                "[%s] anomaly assessment failed for flag=%s (attempt %d/%d)",
                machine_id, state.anomaly_flag,
                self.memory.snapshot(machine_id)["anomaly_attempts"].get(state.anomaly_flag, 0),
                3,
            )
            return None
        self.memory.record_anomaly_assessment(machine_id, state.anomaly_flag, assessment)
        logger.info("[%s] anomaly assessment generated for flag=%s", machine_id, state.anomaly_flag)
        return assessment


# --------------------------------------------------
# CLI
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Predictive Maintenance Agent")
    parser.add_argument("--mqtt-host", type=str, default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument(
        "--rul-model", type=str,
        default=str(Path(__file__).resolve().parent / "artifacts" / "lstm_factory" / "best_lstm.pt"),
    )
    parser.add_argument(
        "--anomaly-model", type=str,
        default=str(Path(__file__).resolve().parent / "artifacts" / "anomaly_lstm" / "best_autoencoder.pt"),
    )
    parser.add_argument("--no-llm", action="store_true", help="Disable Groq LLM assessments.")
    parser.add_argument("--groq-api-key", type=str, default=None, help="Defaults to $GROQ_API_KEY.")
    parser.add_argument(
        "--log-file", type=str,
        default=str(Path(__file__).resolve().parent / "logs" / "predictive_agent.log"),
    )
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--max-log-mb", type=int, default=10)
    parser.add_argument("--log-backup-count", type=int, default=5)
    parser.add_argument("--no-console-log", action="store_true")
    args = parser.parse_args()

    setup_logging(
        log_file=args.log_file,
        level=args.log_level,
        max_bytes=args.max_log_mb * 1024 * 1024,
        backup_count=args.log_backup_count,
        console=not args.no_console_log,
    )

    agent = PredictiveAgent(
        rul_model_path=args.rul_model,
        anomaly_model_path=args.anomaly_model,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        groq_api_key=args.groq_api_key,
        enable_llm=not args.no_llm,
    )
    agent.start()
    agent.run_forever()


if __name__ == "__main__":
    main()