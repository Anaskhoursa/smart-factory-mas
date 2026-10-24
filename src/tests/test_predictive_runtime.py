import numpy as np
import pytest

from src.agents.predictive_runtime import (
    PredictiveModels,
    
    RAW_SENSORS,
)


def test_loads_both_checkpoints():
    models = PredictiveModels()

    assert models.rul_model is not None
    assert models.anomaly_model is not None
    assert models.anomaly_threshold >= 0


def test_synthetic_window_runs_both_models():
    rng = np.random.default_rng(42)
    history = MachineWindow()

    # Plausible synthetic sensor values for shape/contract testing.
    # This is not a model-accuracy test.
    for _ in range(40):
        history.add({
            "vibration": float(rng.normal(1.5, 0.15)),
            "temperature": float(rng.normal(50.0, 2.0)),
            "throughput": float(rng.normal(75.0, 5.0)),
            "error_rate": float(abs(rng.normal(0.02, 0.005))),
        })

    raw = history.raw_window()
    rul_features = history.rul_feature_window()

    assert raw.shape == (30, 4)
    assert rul_features.shape == (30, 12)
    assert np.isfinite(raw).all()
    assert np.isfinite(rul_features).all()

    models = PredictiveModels()
    result = models.predict(
        machine_id="M1",
        raw_window=raw,
        rul_feature_window=rul_features,
    )

    assert result["machine_id"] == "M1"

    anomaly = result["anomaly"]
    rul = result["rul"]
    metadata = result["metadata"]

    assert np.isfinite(anomaly["score"])
    assert np.isfinite(anomaly["threshold"])
    assert anomaly["is_anomaly"] == (
        anomaly["score"] > anomaly["threshold"]
    )

    assert np.isfinite(rul["estimate"])
    assert 0.0 <= rul["estimate"] <= 400.0
    assert rul["unit"] == "training_steps"

    assert metadata["anomaly_input_shape"] == [1, 30, 4]
    assert metadata["anomaly_output_shape"] == [1, 30, 4]
    assert metadata["rul_input_shape"] == [1, 30, 16]
    assert metadata["rul_output_shape"] == [1]


def test_rejects_invalid_machine_id():
    models = PredictiveModels()

    with pytest.raises(ValueError):
        models.predict(
            machine_id="UNKNOWN",
            raw_window=np.zeros((30, 4), dtype=np.float32),
            rul_feature_window=np.zeros((30, 12), dtype=np.float32),
        )
