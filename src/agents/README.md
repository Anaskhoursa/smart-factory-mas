# Predictive maintenance agent

The first agent consumes live simulator telemetry, runs the existing RUL LSTM and
anomaly LSTM-autoencoder, and publishes a structured maintenance assessment. It
does not stop machines or issue maintenance commands; downstream scheduling and
supervisor agents own those decisions.

## Install

From the repository root:

```bash
pip install -e ".[predictive]"
docker compose up -d mqtt
```

The agent expects the trained checkpoints at:

- `predictive/artifacts/lstm_factory/best_lstm.pt`
- `predictive/artifacts/anomaly_lstm/best_autoencoder.pt`

Override them with `RUL_CHECKPOINT` and `ANOMALY_CHECKPOINT` if your artifacts
are stored elsewhere. The live RUL feature engineering must use the same
`ROLLING_WINDOW` used to generate the training CSV; the current dataset
generator defaults to 10. Override with `ROLLING_WINDOW` if you trained with
another value.

## Run end-to-end

Terminal 1 — start the broker from the repository root:

```bash
docker compose up -d mqtt
```

Terminal 2 — start the MQTT-enabled simulator. The simulator modules currently
use local imports, so run this script from its package directory:

```bash
cd src/factory_sim
python mqtt_main.py --steps 120 --inject-at 30 --machine M2 --failure bearing_wear
```

Terminal 3 — run the predictive agent from the repository root:

```bash
python -m agents.predictive_maintenance
```

The simulator publishes each machine's `MachineTelemetry` JSON to
`factory/machines/{machine_id}/telemetry`. The agent subscribes to
`factory/machines/+/telemetry` and publishes reports to
`factory/maintenance/predictions` (QoS 1).

Watch reports from another terminal:

```bash
docker exec -it smart-factory-mqtt mosquitto_sub -h localhost -t factory/maintenance/predictions -v
```

If Mosquitto tools are installed on the host, use
`mosquitto_sub -h localhost -t factory/maintenance/predictions -v` instead.

## Report contract

Reports include `schema_version`, `machine_id`, `timestamp`, per-machine
`sequence`, `rul_steps`, `anomaly_score`, the model's saved anomaly threshold,
`is_anomaly`, `severity`, a recommendation, evidence, and checkpoint names.
RUL is in simulator steps, not hours. Thresholds for operational severity are
initial policy defaults and should be calibrated against the evaluation results
before being used for real equipment.

The model uses only the four sensor readings. Simulator-only fields such as
`health`, `wear`, and `active_failure` are deliberately not passed to either
model.

## Integration notes

- Keep each machine's history separate; do not mix windows from different
  machines.
- The RUL model scaler applies to the 12 sensor-derived features only. The four
  machine identity flags remain unscaled, matching training.
- The anomaly score is mean squared reconstruction error. It is not a probability;
  `is_anomaly` compares it with the threshold saved during training.
- The first report is emitted only after enough telemetry arrives to fill a
  30-step window.
- Missing/invalid telemetry is logged and rejected. No physical action is
  executed by this agent.
