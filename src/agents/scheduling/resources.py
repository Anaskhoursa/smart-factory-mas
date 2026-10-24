"""Module 3 -- feasibility filter.

A Candidate is feasible only if:
  1. Every required part has enough available stock
     (quantity_on_hand - reserved_quantity >= required quantity).
  2. Every required skill is held by at least one technician whose
     shift covers the full candidate window.
  3. At least one such technician is available (max_concurrent_jobs
     not exhausted -- here, always true, since we assume no other
     concurrent assignments; the guard exists so the model can be
     extended later without changing callers).

The module does not reserve stock or assign technicians. It only
*reports* feasibility. Reservation is a side effect that belongs to a
higher layer (supervisor approval), not to the evaluator.
"""
from __future__ import annotations

from models import (
    Candidate,
    FeasibleCandidate,
    InfeasibilityReason,
    SpareParts,
    Technicians,
)


def _available_stock(parts: SpareParts) -> dict[str, int]:
    return {p.part_id: p.available_quantity for p in parts.parts}


def _shift_covers(
    technician,
    start_sim: int,
    end_sim: int,
    minutes_per_day: int,
) -> bool:
    """Does the technician's daily shift cover the candidate entirely?
    Assumes the candidate is within one day (optimizer guarantees it)."""
    start_mod = start_sim % minutes_per_day
    end_mod = end_sim % minutes_per_day
    if end_sim - start_sim >= minutes_per_day:
        return False  # shift < full day; can't cover a >=day task
    return (
        start_mod >= technician.shift_start_minute_of_day
        and end_mod <= technician.shift_end_minute_of_day
    )


def filter_feasible(
    candidates: list[Candidate],
    task_required_parts: dict[str, int],
    task_required_skills: list[str],
    parts: SpareParts,
    technicians: Technicians,
    minutes_per_day: int,
) -> tuple[list[FeasibleCandidate], list[InfeasibilityReason]]:
    available = _available_stock(parts)
    feasible: list[FeasibleCandidate] = []
    rejected: list[InfeasibilityReason] = []

    for cand in candidates:
        # Parts check
        missing = []
        for part_id, needed in task_required_parts.items():
            if available.get(part_id, 0) < needed:
                missing.append(f"{part_id} (need {needed}, have {available.get(part_id, 0)})")
        if missing:
            rejected.append(InfeasibilityReason(
                window_id=cand.window_id,
                reason="Insufficient stock: " + ", ".join(missing),
            ))
            continue

        # Technician check
        skilled = [
            t for t in technicians.technicians
            if t.available and all(skill in t.skills for skill in task_required_skills)
        ]
        if not skilled:
            rejected.append(InfeasibilityReason(
                window_id=cand.window_id,
                reason=(
                    "No available technician has all required skills: "
                    + ", ".join(task_required_skills)
                ),
            ))
            continue

        covering = [
            t for t in skilled
            if _shift_covers(t, cand.start_sim_minute, cand.end_sim_minute, minutes_per_day)
        ]
        if not covering:
            rejected.append(InfeasibilityReason(
                window_id=cand.window_id,
                reason=(
                    "No skilled technician's shift covers "
                    f"{cand.start_sim_minute}..{cand.end_sim_minute} sim minutes."
                ),
            ))
            continue

        # Pick the first covering technician (deterministic; assignment
        # is not this module's job, only feasibility reporting).
        chosen = covering[0]
        feasible.append(FeasibleCandidate(
            candidate=cand,
            assigned_technicians=[chosen.technician_id],
            consumed_parts=dict(task_required_parts),
        ))

    return feasible, rejected