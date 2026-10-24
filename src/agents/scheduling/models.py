"""Pydantic schemas for the scheduling agent.

Two groups:

  Input models    -- mirror the five JSON files (factory_calendar,
                     production_orders, technicians, spare_parts,
                     maintenance_tasks). Validated on load.
  Internal models -- types produced and consumed by the five modules
                     (Priority, Candidate, FeasibleCandidate, Proposal).

Time convention (per the data README): every duration and deadline is
in *simulation minutes*. Calendar windows and technician shifts are
minute-of-day (0-1439). No wall-clock time appears anywhere in this
package's logic; the only wall-clock use is the LLM cooldown in
workflow.py, which is about "don't spam the API", not about scheduling.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ------------------------------------------------------------------
# Input models
# ------------------------------------------------------------------

class OperatingWindow(BaseModel):
    name: str
    start_minute_of_day: int = Field(ge=0, lt=1440)
    end_minute_of_day: int = Field(gt=0, le=1440)


class MaintenanceWindowDef(BaseModel):
    window_id: str
    name: str
    start_minute_of_day: int = Field(ge=0, lt=1440)
    end_minute_of_day: int = Field(gt=0, le=1440)
    allowed_tasks: list[str]


class CalendarRules(BaseModel):
    windows_repeat_daily: bool
    minutes_per_day: int = 1440
    maintenance_during_production_requires_approval: bool
    emergency_maintenance_may_override_window: bool
    emergency_override_requires_supervisor_approval: bool


class FactoryCalendar(BaseModel):
    schema_version: int
    time_basis: str
    description: str = ""
    operating_windows: list[OperatingWindow]
    maintenance_windows: list[MaintenanceWindowDef]
    rules: CalendarRules


class ProductionOrder(BaseModel):
    order_id: str
    product: str
    quantity: int = Field(gt=0)
    completed_quantity: int = Field(ge=0)
    due_at_sim_minute: int = Field(ge=0)
    priority: Literal["low", "normal", "high"]
    required_machines: list[str]
    status: Literal["open", "in_progress", "done", "cancelled"]


class ProductionOrders(BaseModel):
    schema_version: int
    description: str = ""
    orders: list[ProductionOrder]


class Technician(BaseModel):
    technician_id: str
    name: str
    skills: list[str]
    shift_start_minute_of_day: int = Field(ge=0, lt=1440)
    shift_end_minute_of_day: int = Field(gt=0, le=1440)
    available: bool
    max_concurrent_jobs: int = Field(ge=1)


class Technicians(BaseModel):
    schema_version: int
    time_basis: str
    description: str = ""
    technicians: list[Technician]


class SparePart(BaseModel):
    part_id: str
    name: str
    compatible_machines: list[str]
    compatible_failures: list[str]
    quantity_on_hand: int = Field(ge=0)
    reserved_quantity: int = Field(ge=0)
    reorder_level: int = Field(ge=0)
    lead_time_minutes: int = Field(ge=0)

    @property
    def available_quantity(self) -> int:
        return max(0, self.quantity_on_hand - self.reserved_quantity)


class SpareParts(BaseModel):
    schema_version: int
    description: str = ""
    parts: list[SparePart]


class RequiredPart(BaseModel):
    part_id: str
    quantity: int = Field(ge=1)


class MaintenanceTaskDef(BaseModel):
    task_id: str
    failure_mode: str
    task_name: str
    required_skills: list[str]
    required_parts: list[RequiredPart]
    estimated_duration_minutes: int = Field(gt=0)
    duration_uncertainty_minutes: int = Field(ge=0)
    task_category: Literal[
        "inspection", "minor_repair", "planned_repair", "preventive_maintenance"
    ]
    requires_machine_stopped: bool


class MaintenanceTasks(BaseModel):
    schema_version: int
    description: str = ""
    tasks: list[MaintenanceTaskDef]


# ------------------------------------------------------------------
# Internal models
# ------------------------------------------------------------------

PriorityLevel = Literal["normal", "medium", "high", "critical"]


class PriorityAssessment(BaseModel):
    priority: PriorityLevel
    reasons: list[str]
    rul_band: str
    anomaly_flag: str
    stale_signals: list[str]
    # Urgency horizon in simulation minutes. The optimizer uses this to
    # constrain when a maintenance window must begin.
    urgency_horizon_minutes: int = Field(ge=0)


class Candidate(BaseModel):
    """A maintenance window the optimizer thinks is plausible in time."""
    window_id: str
    window_name: str
    start_sim_minute: int = Field(ge=0)
    end_sim_minute: int = Field(gt=0)
    task_id: str
    task_name: str
    estimated_duration_minutes: int = Field(gt=0)
    duration_uncertainty_minutes: int = Field(ge=0)
    requires_machine_stopped: bool
    within_window: bool
    rationale: str


class FeasibleCandidate(BaseModel):
    """A candidate that survived the resources module."""
    candidate: Candidate
    assigned_technicians: list[str]
    consumed_parts: dict[str, int]


class InfeasibilityReason(BaseModel):
    window_id: str
    reason: str


class Proposal(BaseModel):
    proposal_id: str
    agent_id: str = "maintenance-scheduling-agent"
    machine_id: str
    generated_at: str
    status: Literal["proposed"] = "proposed"
    priority: PriorityLevel
    reasons: list[str]
    health_evidence: dict
    predictive_assessments: dict
    feasible_options: list[FeasibleCandidate]
    infeasible_reasons: list[InfeasibilityReason]
    llm_assessment: dict | None
    production_impact: dict
    stale_signals: list[str]
    requires_supervisor_review: bool
    machine_control_authorized: Literal[False] = False
    feasibility: Literal["feasible", "infeasible", "no_candidates"]
    audit: dict