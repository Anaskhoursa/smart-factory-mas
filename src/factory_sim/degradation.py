
from dataclasses import dataclass

from models import FailureMode


@dataclass
class DegradationState:
    wear: float = 0.0
    misalignment: float = 0.0
    overheating: float = 0.0
    shock_damage: float = 0.0
    active_failure: FailureMode | None = None
    fault_severity: float = 0.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def calculate_health(state: DegradationState) -> float:
    """Return normalized health in the range [0, 1]."""
    damage = (
        0.55 * state.wear
        + 0.15 * state.misalignment
        + 0.15 * state.overheating
        + 0.35 * state.shock_damage
    )
    return clamp(1.0 - damage, 0.0, 1.0)


def advance_degradation(
    state: DegradationState,
    wear_rate_per_hour: float,
    dt_hours: float,
    load: float,
) -> None:
    """Advance gradual mechanical wear by one simulation step."""
    load_factor = 0.25 + 0.75 * load
    state.wear = clamp(
        state.wear + wear_rate_per_hour * dt_hours * load_factor,
        0.0,
        1.0,
    )

    # Faults can progress while the machine is running.
    if state.active_failure == FailureMode.BEARING_WEAR:
        state.wear = clamp(
            state.wear + 0.015 * dt_hours * 60
            * state.fault_severity,
            0.0,
            1.0,
        )

    elif state.active_failure == FailureMode.MISALIGNMENT:
        state.misalignment = clamp(
            state.misalignment
            + 0.003 * dt_hours * 60 * state.fault_severity,
            0.0,
            1.0,
        )

    elif state.active_failure == FailureMode.OVERHEATING:
        state.overheating = clamp(
            state.overheating
            + 0.002 * dt_hours * 60 * state.fault_severity,
            0.0,
            1.0,
        )

    elif state.active_failure == FailureMode.RANDOM_SHOCK:
        state.shock_damage = clamp(
            state.shock_damage + 0.02 * state.fault_severity,
            0.0,
            1.0,
        )