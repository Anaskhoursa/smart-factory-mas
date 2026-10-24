"""Per-machine LLM-assessment dedup memory.

The predictive agent calls its LLM once per (machine, band) — not once
per step, not once per raw-value change. This class owns that memory.

Two independent axes:
  - RUL band memory: which RUL bands we've already narrated for this
    machine. Bands are {critical, near, medium, far, unknown}, but only
    {critical, near} are ever eligible for a call (see health_state).
  - Anomaly flag memory: which anomaly flags we've already narrated for
    this machine. Flags are {flagged, clear, unknown}, but only
    {flagged} is ever eligible for a call.

Memory is in-process and cleared on restart. That's correct for the
simulator. If the agent is later deployed across restarts, persist
this dict to disk — see the plan doc.

The streak counters live here too because they are per-machine and are
updated on exactly the same cadence as the memory.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from health_state import (
    ANOMALY_FLAGGED,
    RUL_BAND_CRITICAL,
    RUL_BAND_NEAR,
)

ASSESSABLE_RUL_BANDS = frozenset({RUL_BAND_NEAR, RUL_BAND_CRITICAL})
ASSESSABLE_ANOMALY_FLAGS = frozenset({ANOMALY_FLAGGED})

RUL_LLM_MAX_ATTEMPTS = 3
ANOMALY_LLM_MAX_ATTEMPTS = 3

@dataclass
class MachineMemory:
    rul_streak: int = 0
    anomaly_streak: int = 0
    assessed_rul_bands: set[str] = field(default_factory=set)
    assessed_anomaly_flags: set[str] = field(default_factory=set)
    last_rul_assessment: dict | None = None
    last_anomaly_assessment: dict | None = None
    rul_attempts: dict[str, int] = field(default_factory=dict)
    anomaly_attempts: dict[str, int] = field(default_factory=dict)


class AssessmentMemory:
    """Owns per-machine streaks and band memories."""

    def __init__(self, streak_threshold: int = 3):
        if streak_threshold < 1:
            raise ValueError("streak_threshold must be >= 1")
        self.streak_threshold = streak_threshold
        self._machines: dict[str, MachineMemory] = {}

    def _get(self, machine_id: str) -> MachineMemory:
        return self._machines.setdefault(machine_id, MachineMemory())

    def update_streaks(self, machine_id: str, rul_streak_continues: bool, anomaly_streak_continues: bool) -> None:
        memory = self._get(machine_id)
        memory.rul_streak = memory.rul_streak + 1 if rul_streak_continues else 0
        memory.anomaly_streak = (
            memory.anomaly_streak + 1 if anomaly_streak_continues else 0
        )

    def should_assess_rul(self, machine_id: str, rul_band: str) -> bool:
        if rul_band not in ASSESSABLE_RUL_BANDS:
            return False
        memory = self._get(machine_id)
        if memory.rul_streak < self.streak_threshold:
            return False
        if rul_band in memory.assessed_rul_bands:
            return False
        if memory.rul_attempts.get(rul_band, 0) >= RUL_LLM_MAX_ATTEMPTS:
            return False
        return True

    def should_assess_anomaly(self, machine_id: str, anomaly_flag: str) -> bool:
        if anomaly_flag not in ASSESSABLE_ANOMALY_FLAGS:
            return False
        memory = self._get(machine_id)
        if memory.anomaly_streak < self.streak_threshold:
            return False
        if anomaly_flag in memory.assessed_anomaly_flags:
            return False
        if memory.anomaly_attempts.get(anomaly_flag, 0) >= ANOMALY_LLM_MAX_ATTEMPTS:
            return False
        return True

    def record_rul_assessment(self, machine_id: str, rul_band: str, assessment: dict) -> None:
        memory = self._get(machine_id)
        memory.assessed_rul_bands.add(rul_band)
        memory.last_rul_assessment = assessment

    def record_anomaly_assessment(self, machine_id: str, anomaly_flag: str, assessment: dict) -> None:
        memory = self._get(machine_id)
        memory.assessed_anomaly_flags.add(anomaly_flag)
        memory.last_anomaly_assessment = assessment

    def snapshot(self, machine_id: str) -> dict:
        m = self._machines.get(machine_id)
        if m is None:
            return {
                "rul_streak": 0, "anomaly_streak": 0,
                "assessed_rul_bands": [], "assessed_anomaly_flags": [],
                "rul_attempts": {}, "anomaly_attempts": {},
            }
        return {
            "rul_streak": m.rul_streak,
            "anomaly_streak": m.anomaly_streak,
            "assessed_rul_bands": sorted(m.assessed_rul_bands),
            "assessed_anomaly_flags": sorted(m.assessed_anomaly_flags),
            "rul_attempts": dict(m.rul_attempts),
            "anomaly_attempts": dict(m.anomaly_attempts),
        }
        
    def record_rul_attempt(self, machine_id: str, rul_band: str) -> None:
        memory = self._get(machine_id)
        memory.rul_attempts[rul_band] = memory.rul_attempts.get(rul_band, 0) + 1

    def record_anomaly_attempt(self, machine_id: str, anomaly_flag: str) -> None:
        memory = self._get(machine_id)
        memory.anomaly_attempts[anomaly_flag] = (
            memory.anomaly_attempts.get(anomaly_flag, 0) + 1
        )