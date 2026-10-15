
import random
from datetime import datetime, timezone

from config import MachineConfig, SimulationConfig
from degradation import (
    DegradationState,
    advance_degradation,
    calculate_health,
    clamp,
)
from models import (
    FailureMode,
    MachineStatus,
    MachineTelemetry,
)
from sensors import generate_reading


class Machine:
    def __init__(
        self,
        config: MachineConfig,
        sim_config: SimulationConfig,
        seed_offset: int = 0,
    ):
        self.config = config
        self.sim_config = sim_config
        self.rng = random.Random(sim_config.seed + seed_offset)

        self.state = DegradationState()
        self.status = MachineStatus.RUNNING
        self.load = sim_config.initial_load

        self.temperature = config.base_temperature
        self.total_operating_hours = 0.0
        self.total_produced = 0.0
        self.failure_count = 0

    @property
    def health(self) -> float:
        return calculate_health(self.state)

    def inject_failure(
        self,
        mode: FailureMode,
        severity: float = 0.5,
    ) -> None:
        """Inject a fault, where severity is between 0 and 1."""
        if self.status == MachineStatus.FAILED:
            raise RuntimeError("Cannot inject a fault into a failed machine.")

        if not 0 < severity <= 1:
            raise ValueError("severity must be in (0, 1].")

        self.state.active_failure = mode
        self.state.fault_severity = severity

        if mode == FailureMode.BEARING_WEAR:
            self.state.wear = clamp(
                self.state.wear + 0.4 * severity, 0.0, 1.0
            )

        elif mode == FailureMode.OVERHEATING:
            self.state.overheating = max(
                self.state.overheating, 0.6 * severity
            )

        elif mode == FailureMode.MISALIGNMENT:
            self.state.misalignment = max(
                self.state.misalignment, 0.5 * severity
            )

        elif mode == FailureMode.RANDOM_SHOCK:
            self.state.shock_damage = clamp(
                self.state.shock_damage + 0.7 * severity,
                0.0,
                1.0,
            )

    def repair(self) -> None:
        """Perform a simplified full repair and return to service."""
        self.state = DegradationState()
        self.temperature = self.config.base_temperature
        self.status = MachineStatus.RUNNING

    def step(self, dt_minutes: float, timestamp: datetime | None = None) -> MachineTelemetry:
        if dt_minutes <= 0:
            raise ValueError("dt_minutes must be positive.")

        dt_hours = dt_minutes / 60.0

        if self.status != MachineStatus.FAILED:
            advance_degradation(
                state=self.state,
                wear_rate_per_hour=self.config.wear_rate_per_hour,
                dt_hours=dt_hours,
                load=self.load,
            )

            # A simple first-order thermal model.
            target_temperature = (
                self.config.base_temperature
                + 20.0 * self.load
                + 50.0 * self.state.overheating
                + 8.0 * self.state.wear
            )

            # Thermal time constant: approximately 30 minutes.
            alpha = 1.0 - pow(2.718281828, -dt_minutes / 30.0)
            self.temperature += alpha * (
                target_temperature - self.temperature
            )

            self.total_operating_hours += dt_hours

            # Failure conditions.
            failed = (
                self.state.wear >= self.config.max_wear
                or self.temperature >= self.config.max_temperature
                or self.state.shock_damage >= 0.9
            )

            # Optional spontaneous failure, modeled as a hazard rate.
            if not failed:
                p_failure = 1.0 - (
                    1.0
                    - self.sim_config.spontaneous_failure_rate
                ) ** dt_hours
                failed = self.rng.random() < p_failure

            if failed:
                self.status = MachineStatus.FAILED
                self.failure_count += 1

            elif (
                self.health < 0.75
                or self.state.active_failure is not None
            ):
                self.status = MachineStatus.DEGRADED

            else:
                self.status = MachineStatus.RUNNING

        if self.status in (
            MachineStatus.RUNNING,
            MachineStatus.DEGRADED,
        ):
            produced = (
                self.config.nominal_throughput
                * self.load
                * dt_hours
                * max(0.0, 1.0 - 0.35 * self.state.wear)
            )
            self.total_produced += produced
        else:
            produced = 0.0

        reading = generate_reading(
            machine_id=self.config.machine_id,
            config=self.config,
            state=self.state,
            load=self.load,
            rng=self.rng,
            noise=self.sim_config.sensor_noise,
            temperature=self.temperature,
            failed=self.status == MachineStatus.FAILED,
            timestamp=timestamp,
        )

        return MachineTelemetry(
            machine_id=self.config.machine_id,
            timestamp=datetime.now(timezone.utc),
            status=self.status,
            health=round(self.health, 4),
            wear=round(self.state.wear, 4),
            load=self.load,
            active_failure=self.state.active_failure,
            sensors=reading,
        )