"""Module 2 -- candidate maintenance windows.

Given a PriorityAssessment and the factory calendar, produce a short
list of Candidate windows: when could this machine be serviced?

Time model: simulation minutes from start. Calendar windows repeat
daily, defined as minute-of-day ranges. A candidate's absolute start
is an absolute simulation minute; its minute-of-day is the start mod
1440.

Emergency override rule (from factory_calendar.json):
  - If priority is "critical" AND emergency_maintenance_may_override_window
    is True, the task may be scheduled inside an operating window,
    provided emergency_override_requires_supervisor_approval is True
    (the proposal will be flagged for supervisor review automatically
    because critical priority already does that).
  - Otherwise, the task must fall entirely inside a maintenance window
    whose `allowed_tasks` includes the task's category.

Anchoring: candidates are generated at:
  - The earliest sim minute after `now_sim_minute` that satisfies the
    window constraint AND the urgency horizon.
  - Every subsequent window boundary up to `now + 2 days`, so the
    reasoner has real alternatives to compare, not one forced option.

The optimizer never picks the winning window. It only enumerates
plausible ones. Selection is left to the supervisor, informed by the
resource check and the LLM narrative.
"""
from __future__ import annotations

from models import (
    Candidate,
    FactoryCalendar,
    MaintenanceTaskDef,
    PriorityAssessment,
)
from models import ProductionOrders


MAX_CANDIDATES = 4


def _next_window_boundary(
    start_sim: int,
    window_start: int,
    window_end: int,
    minutes_per_day: int,
) -> int:
    """Smallest sim minute >= start_sim that falls at a day boundary
    where the given window opens."""
    day = start_sim // minutes_per_day
    for k in range(0, 3):  # search today and the next two days
        base = (day + k) * minutes_per_day
        candidate_start = base + window_start
        if candidate_start >= start_sim:
            return candidate_start
    return (day + 3) * minutes_per_day + window_start


def _candidate_fits(
    start_sim: int,
    duration_minutes: int,
    window_start: int,
    window_end: int,
    minutes_per_day: int,
) -> bool:
    """Does [start_sim, start_sim+duration] fit inside the window on
    the day of start_sim? Assumes the window does not straddle midnight."""
    minute_of_day = start_sim % minutes_per_day
    if minute_of_day < window_start:
        return False
    if minute_of_day + duration_minutes > window_end:
        return False
    return True


def generate_candidates(
    machine_id: str,
    priority: PriorityAssessment,
    now_sim_minute: int,
    task: MaintenanceTaskDef | None,
    calendar: FactoryCalendar,
    orders: ProductionOrders,
) -> list[Candidate]:
    if task is None:
        return []

    minutes_per_day = calendar.rules.minutes_per_day
    duration = task.estimated_duration_minutes
    urgent_minute = now_sim_minute + priority.urgency_horizon_minutes

    emergency_allowed = (
        priority.priority == "critical"
        and calendar.rules.emergency_maintenance_may_override_window
    )

    candidates: list[Candidate] = []

    if emergency_allowed:
        # Emergency: schedule as soon as possible, even inside production.
        candidates.append(Candidate(
            window_id="EMERGENCY-NOW",
            window_name="Emergency override (production window)",
            start_sim_minute=now_sim_minute,
            end_sim_minute=now_sim_minute + duration,
            task_id=task.task_id,
            task_name=task.task_name,
            estimated_duration_minutes=duration,
            duration_uncertainty_minutes=task.duration_uncertainty_minutes,
            requires_machine_stopped=task.requires_machine_stopped,
            within_window=False,
            rationale=(
                "Critical priority with emergency override enabled; "
                "supervisor approval is mandatory before execution."
            ),
        ))

    # Then enumerate maintenance-window candidates.
    for window in calendar.maintenance_windows:
        if task.task_category not in window.allowed_tasks:
            continue

        first_start = _next_window_boundary(
            urgent_minute, window.start_minute_of_day, window.end_minute_of_day,
            minutes_per_day,
        )
        # Also try the window that *contains* now if it fits, so we
        # don't skip a window that is already open right now.
        today_start = (
            now_sim_minute // minutes_per_day * minutes_per_day
            + window.start_minute_of_day
        )
        starts_to_try = sorted({first_start, today_start})
        for start in starts_to_try:
            if start < now_sim_minute:
                continue
            if not _candidate_fits(
                start, duration, window.start_minute_of_day,
                window.end_minute_of_day, minutes_per_day,
            ):
                continue
            if len(candidates) >= MAX_CANDIDATES:
                break
            candidates.append(Candidate(
                window_id=window.window_id,
                window_name=window.name,
                start_sim_minute=start,
                end_sim_minute=start + duration,
                task_id=task.task_id,
                task_name=task.task_name,
                estimated_duration_minutes=duration,
                duration_uncertainty_minutes=task.duration_uncertainty_minutes,
                requires_machine_stopped=task.requires_machine_stopped,
                within_window=True,
                rationale=(
                    f"Fits inside {window.name} and begins within the "
                    f"{priority.urgency_horizon_minutes}-minute urgency horizon."
                ),
            ))

    # De-duplicate by (window_id, start_sim) and sort by start.
    seen: set[tuple[str, int]] = set()
    unique: list[Candidate] = []
    for c in sorted(candidates, key=lambda x: x.start_sim_minute):
        key = (c.window_id, c.start_sim_minute)
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique[:MAX_CANDIDATES]


def production_impact(
    candidate: Candidate,
    orders: ProductionOrders,
    now_sim_minute: int,
) -> dict:
    """Cheap summary of which open orders overlap the candidate window."""
    affected = []
    for order in orders.orders:
        if order.status != "open":
            continue
        if order.due_at_sim_minute < now_sim_minute:
            continue
        if (
            order.due_at_sim_minute >= candidate.start_sim_minute
            and order.due_at_sim_minute <= candidate.end_sim_minute
        ):
            affected.append(order.order_id)
    return {
        "orders_due_during_downtime": affected,
        "downtime_minutes": candidate.estimated_duration_minutes,
    }