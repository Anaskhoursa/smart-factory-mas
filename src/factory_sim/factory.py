
import uuid
from datetime import datetime, timedelta, timezone

from config import (
    DEFAULT_MACHINES,
    MachineConfig,
    SimulationConfig,
)
from machine import Machine
from models import (
    FactoryEvent,
    FactorySnapshot,
    FailureMode,
    MachineStatus,
)


class Factory:
    def __init__(
        self,
        machines: list[MachineConfig] | None = None,
        config: SimulationConfig | None = None,
    ):
        self.config = config or SimulationConfig()
        machine_configs = machines or DEFAULT_MACHINES

        self.machines = {
            cfg.machine_id: Machine(
                config=cfg,
                sim_config=self.config,
                seed_offset=index,
            )
            for index, cfg in enumerate(machine_configs)
        }

        self.clock = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.total_production = 0.0
        self.events: list[FactoryEvent] = []
        self._previous_status = {
            machine_id: machine.status
            for machine_id, machine in self.machines.items()
        }

    def inject_failure(
        self,
        machine_id: str,
        mode: FailureMode,
        severity: float = 0.5,
    ) -> FactoryEvent:
        machine = self.machines[machine_id]
        machine.inject_failure(mode, severity)

        event = FactoryEvent(
            event_id=str(uuid.uuid4()),
            timestamp=self.clock,
            machine_id=machine_id,
            event_type="failure_injected",
            severity="warning",
            details={
                "mode": mode.value,
                "severity": severity,
            },
        )
        self.events.append(event)
        return event

    def _collect_events(self) -> list[FactoryEvent]:
        new_events = []

        for machine_id, machine in self.machines.items():
            previous = self._previous_status[machine_id]
            current = machine.status

            if previous != MachineStatus.FAILED and (
                current == MachineStatus.FAILED
            ):
                event = FactoryEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=self.clock,
                    machine_id=machine_id,
                    event_type="machine_failure",
                    severity="critical",
                    details={
                        "health": round(machine.health, 4),
                        "wear": round(machine.state.wear, 4),
                    },
                )
                new_events.append(event)

            if previous != MachineStatus.DEGRADED and (
                current == MachineStatus.DEGRADED
            ):
                event = FactoryEvent(
                    event_id=str(uuid.uuid4()),
                    timestamp=self.clock,
                    machine_id=machine_id,
                    event_type="machine_degraded",
                    severity="warning",
                    details={
                        "health": round(machine.health, 4),
                        "wear": round(machine.state.wear, 4),
                    },
                )
                new_events.append(event)

            self._previous_status[machine_id] = current

        self.events.extend(new_events)
        return new_events

    def step(self) -> FactorySnapshot:
        dt = self.config.step_minutes
        self.clock += timedelta(minutes=dt)

        telemetry = [
            machine.step(dt, timestamp=self.clock)
            for machine in self.machines.values()
        ]

        # Sequential production line: the slowest operating
        # machine determines the line's output.
        if all(
            machine.status in (
                MachineStatus.RUNNING,
                MachineStatus.DEGRADED,
            )
            for machine in self.machines.values()
        ):
            output_rates = [
                machine.config.nominal_throughput
                * machine.load
                * max(0.0, 1.0 - 0.35 * machine.state.wear)
                for machine in self.machines.values()
            ]
            output = min(output_rates) * dt / 60.0
        else:
            output = 0.0

        self.total_production += output

        return FactorySnapshot(
            timestamp=self.clock,
            total_production=round(self.total_production, 4),
            machines=telemetry,
            events=self._collect_events(),
        )

    def run(self, steps: int) -> list[FactorySnapshot]:
        if steps < 0:
            raise ValueError("steps must be non-negative.")
        return [self.step() for _ in range(steps)]