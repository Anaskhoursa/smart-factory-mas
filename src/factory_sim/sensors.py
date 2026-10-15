
import random
from datetime import datetime, timezone

from config import MachineConfig
from degradation import DegradationState
from models import SensorReading


def generate_reading(
    machine_id: str,
    config: MachineConfig,
    state: DegradationState,
    load: float,
    rng: random.Random,
    noise: float = 0.05,
    temperature: float | None = None,
    failed: bool = False,
    timestamp: datetime | None = None,
) -> SensorReading:
    wear = state.wear
    misalignment = state.misalignment
    overheating = state.overheating
    shock = state.shock_damage

    # Underlying physical signals (simplified).
    vibration_mean = (
        config.base_vibration
        * (1 + 3.0 * wear + 2.0 * misalignment + 2.5 * shock)
    )

    temperature_mean = (
        config.base_temperature
        + 20.0 * load
        + 50.0 * overheating
        + 8.0 * wear
    )

    throughput_mean = (
        config.nominal_throughput
        * load
        * max(0.0, 1.0 - 0.35 * wear - 0.2 * shock)
    )

    error_mean = min(
        1.0,
        0.005
        + 0.08 * wear
        + 0.12 * misalignment
        + 0.15 * overheating
        + 0.2 * shock,
    )

    if failed:
        throughput_mean = 0.0
        error_mean = max(error_mean, 0.5)

    def noisy(mean: float, scale: float) -> float:
        return max(0.0, rng.gauss(mean, max(scale, 1e-9)))

    measured_temp = (
        temperature if temperature is not None else temperature_mean
    )

    return SensorReading(
        machine_id=machine_id,
        timestamp=timestamp or datetime.now(timezone.utc),
        vibration=round(
            noisy(vibration_mean, noise * max(vibration_mean, 0.1)), 4
        ),
        temperature=round(
            noisy(measured_temp, noise * max(measured_temp, 1.0)), 2
        ),
        throughput=round(
            noisy(throughput_mean, noise * max(throughput_mean, 1.0)), 2
        ),
        error_rate=round(
            min(1.0, noisy(error_mean, noise * 0.1)), 5
        ),
    )