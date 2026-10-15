
from datetime import datetime
from enum import Enum

class StrEnum(str, Enum):
    pass

class YourEnum(StrEnum):
    MY_VALUE = "my_value"


from pydantic import BaseModel, Field


class MachineStatus(StrEnum):
    RUNNING = "running"
    DEGRADED = "degraded"
    FAILED = "failed"
    MAINTENANCE = "maintenance"


class FailureMode(StrEnum):
    BEARING_WEAR = "bearing_wear"
    OVERHEATING = "overheating"
    MISALIGNMENT = "misalignment"
    RANDOM_SHOCK = "random_shock"


class SensorReading(BaseModel):
    machine_id: str
    timestamp: datetime
    vibration: float
    temperature: float
    throughput: float
    error_rate: float


class MachineTelemetry(BaseModel):
    machine_id: str
    timestamp: datetime
    status: MachineStatus
    health: float = Field(ge=0, le=1)
    wear: float = Field(ge=0, le=1)
    load: float = Field(ge=0, le=1)
    active_failure: FailureMode | None = None
    sensors: SensorReading


class FactoryEvent(BaseModel):
    event_id: str
    timestamp: datetime
    machine_id: str
    event_type: str
    severity: str
    details: dict[str, str | float | int | bool]


class FactorySnapshot(BaseModel):
    timestamp: datetime
    total_production: float
    machines: list[MachineTelemetry]
    events: list[FactoryEvent]