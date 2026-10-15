
from dataclasses import dataclass


@dataclass(frozen=True)
class MachineConfig:
    machine_id: str
    name: str
    nominal_throughput: float  # units per hour
    base_temperature: float    # Celsius
    base_vibration: float      # mm/s
    wear_rate_per_hour: float
    max_temperature: float = 110.0
    max_wear: float = 1.0


@dataclass(frozen=True)
class SimulationConfig:
    step_minutes: float = 1.0
    seed: int = 42
    sensor_noise: float = 0.05
    spontaneous_failure_rate: float = 0.00001
    initial_load: float = 0.8


DEFAULT_MACHINES = [
    MachineConfig(
        machine_id="M1",
        name="Raw Material Cutter",
        nominal_throughput=120,
        base_temperature=42,
        base_vibration=1.5,
        wear_rate_per_hour=0.0008,
    ),
    MachineConfig(
        machine_id="M2",
        name="CNC Machining Center",
        nominal_throughput=100,
        base_temperature=55,
        base_vibration=2.0,
        wear_rate_per_hour=0.0012,
    ),
    MachineConfig(
        machine_id="M3",
        name="Assembly Station",
        nominal_throughput=90,
        base_temperature=38,
        base_vibration=1.0,
        wear_rate_per_hour=0.0006,
    ),
    MachineConfig(
        machine_id="M4",
        name="Quality Inspection",
        nominal_throughput=110,
        base_temperature=35,
        base_vibration=0.8,
        wear_rate_per_hour=0.0005,
    ),
]