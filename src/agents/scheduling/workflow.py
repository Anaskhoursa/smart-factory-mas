"""Module 5 -- proposal construction, dedup, audit log.

Owns:
  - the per-machine proposal history used for dedup
  - the append-only audit JSONL log
  - the exact JSON shape published on V1_MAINTENANCE_PROPOSALS

Dedup signature includes `assessment_id` (a fresh UUID generated on
each successful LLM call). This is the fix for the previous bug where
a fresh LLM narrative could be computed and then dropped because the
rest of the signature was unchanged. Any fresh assessment always
reaches the bus.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from models import (
    FeasibleCandidate,
    InfeasibilityReason,
    PriorityAssessment,
    Proposal,
)

logger = logging.getLogger("scheduling_agent.workflow")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_feasible(feasible: list[FeasibleCandidate]) -> str:
    payload = [
        {
            "window_id": f.candidate.window_id,
            "task": f.candidate.task_id,
            "tech": f.assigned_technicians,
            "parts": f.consumed_parts,
        }
        for f in feasible
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

class AuditLog:
    """One JSONL line per LLM call and per published proposal."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event_type: str, payload: dict) -> None:
        line = json.dumps(
            {"ts": _utc_now_iso(), "event": event_type, **payload},
            ensure_ascii=False, allow_nan=False,
        )
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


class Workflow:
    def __init__(self, audit_log_path: str | Path, proposal_refresh_seconds: float = 60.0):
        self._last_publish_signature: dict[str, tuple] = {}
        self._last_publish_at: dict[str, float] = {}
        self._last_priority: dict[str, str] = {}
        self._last_feasible_hash: dict[str, str] = {}
        self._last_assessment_id: dict[str, str] = {}
        self.proposal_refresh_seconds = proposal_refresh_seconds
        self.audit = AuditLog(audit_log_path)

    def should_narrate(self, machine_id: str, priority: PriorityAssessment, feasible: list[FeasibleCandidate], requested: bool) -> bool:
        if requested:
            return True
        if self._last_priority.get(machine_id) != priority.priority:
            return True
        if self._last_feasible_hash.get(machine_id) != _hash_feasible(feasible):
            return True
        return False

    def build(
        self,
        machine_id: str,
        priority: PriorityAssessment,
        feasible: list[FeasibleCandidate],
        infeasible: list[InfeasibilityReason],
        llm_assessment: dict | None,
        health_evidence: dict,
        predictive_assessments: dict,
        production_impact: dict,
    ) -> Proposal:
        assessment_id = str(uuid.uuid4()) if llm_assessment is not None else ""
        feasibility = (
            "no_candidates" if not feasible and not infeasible
            else "feasible" if feasible
            else "infeasible"
        )
        proposal = Proposal(
            proposal_id=str(uuid.uuid4()),
            machine_id=machine_id,
            generated_at=_utc_now_iso(),
            priority=priority.priority,
            reasons=priority.reasons,
            health_evidence=health_evidence,
            predictive_assessments=predictive_assessments,
            feasible_options=feasible,
            infeasible_reasons=infeasible,
            llm_assessment=llm_assessment,
            production_impact=production_impact,
            stale_signals=priority.stale_signals,
            requires_supervisor_review=priority.priority in {"high", "critical"},
            feasibility=feasibility,
            audit={
                "assessment_id": assessment_id,
                "priority_changed_at_iso": _utc_now_iso()
                if self._last_priority.get(machine_id) != priority.priority else None,
                "feasible_hash": _hash_feasible(feasible),
            },
        )
        return proposal

    def should_publish(self, proposal: Proposal) -> bool:
        machine_id = proposal.machine_id
        signature = (
            proposal.priority,
            proposal.feasibility,
            proposal.audit["feasible_hash"],
            proposal.audit["assessment_id"],
            tuple(proposal.stale_signals),
        )
        now = time.monotonic()
        if (
            signature == self._last_publish_signature.get(machine_id)
            and now - self._last_publish_at.get(machine_id, 0.0) < self.proposal_refresh_seconds
        ):
            return False
        self._last_publish_signature[machine_id] = signature
        self._last_publish_at[machine_id] = now
        self._last_priority[machine_id] = proposal.priority
        self._last_feasible_hash[machine_id] = proposal.audit["feasible_hash"]
        self._last_assessment_id[machine_id] = proposal.audit["assessment_id"]
        return True

    def log_llm_call(self, machine_id: str, context: dict, assessment: dict | None) -> None:
        self.audit.append("llm_call", {
            "agent": "scheduling",
            "machine_id": machine_id,
            "context": context,
            "assessment": assessment,
        })

    def log_publish(self, proposal: Proposal) -> None:
        self.audit.append("proposal_published", {
            "machine_id": proposal.machine_id,
            "proposal_id": proposal.proposal_id,
            "priority": proposal.priority,
            "feasibility": proposal.feasibility,
            "assessment_id": proposal.audit["assessment_id"],
        })