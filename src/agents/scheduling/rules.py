"""Module 1 -- deterministic priority and reasons.

Pure functions. No MQTT, no LLM, no I/O. Given a machine state dict
(as maintained by scheduling_agent.py from the health topics) and the
current simulation minute, return a PriorityAssessment.

Rules (matching the predictive agent's bands so the two agree):

  RUL band critical (<30)  -> critical
  RUL band near (30..60)   -> high
  RUL band medium (60..120)-> medium
  RUL band far (>=120)     -> normal
  RUL unknown              -> normal

  Anomaly flagged escalates one level (normal->medium->high->critical).
  Stale signal (a health feed the scheduler has stopped hearing from)
  escalates normal->medium, never further.

  Urgency horizon (simulation minutes) is derived from priority:
    critical -> 30 min   (must happen within ~half an hour of sim time)
    high     -> 120 min
    medium   -> 480 min
    normal   -> 1440 min (next daily cycle)
"""
from __future__ import annotations

from models import PriorityAssessment, PriorityLevel

RUL_CRITICAL = 30.0
RUL_NEAR = 60.0
RUL_MEDIUM = 120.0

URGENCY_BY_PRIORITY: dict[str, int] = {
    "critical": 30,
    "high": 120,
    "medium": 480,
    "normal": 1440,
}

_ORDER = ["normal", "medium", "high", "critical"]


def _rul_band(rul: float | None) -> str:
    if rul is None:
        return "unknown"
    if rul < RUL_CRITICAL:
        return "critical"
    if rul < RUL_NEAR:
        return "near"
    if rul < RUL_MEDIUM:
        return "medium"
    return "far"


def _escalate(level: str) -> str:
    return _ORDER[min(_ORDER.index(level) + 1, len(_ORDER) - 1)]


def prioritize(
    machine_id: str,
    rul: float | None,
    is_anomaly: bool | None,
    stale_signals: list[str],
) -> PriorityAssessment:
    rul_band = _rul_band(rul)
    anomaly_flag = (
        "unknown" if is_anomaly is None else ("flagged" if is_anomaly else "clear")
    )

    reasons: list[str] = []

    if rul_band == "unknown" and anomaly_flag == "unknown":
        # Nothing to act on yet; still return a valid assessment so the
        # rest of the pipeline runs uniformly.
        reasons.append("Waiting for health predictions.")
        return PriorityAssessment(
            priority="normal",
            reasons=reasons,
            rul_band=rul_band,
            anomaly_flag=anomaly_flag,
            stale_signals=stale_signals,
            urgency_horizon_minutes=URGENCY_BY_PRIORITY["normal"],
        )

    priority: PriorityLevel = "normal"

    if rul_band == "critical":
        priority = "critical"
        reasons.append(
            f"Predicted remaining useful life is very low ({rul:.1f} simulation minutes)."
        )
    elif rul_band == "near":
        priority = "high"
        reasons.append(
            f"Predicted remaining useful life is low ({rul:.1f} simulation minutes)."
        )
    elif rul_band == "medium":
        priority = "medium"
        reasons.append(
            f"Predicted remaining useful life warrants preventive planning ({rul:.1f} simulation minutes)."
        )

    if anomaly_flag == "flagged":
        reasons.append("The anomaly detector flagged unusual sensor behaviour.")
        priority = _escalate(priority)

    if stale_signals:
        reasons.extend(stale_signals)
        if priority == "normal":
            priority = "medium"

    if not reasons:
        reasons.append("No warning is indicated by the currently available health signals.")

    return PriorityAssessment(
        priority=priority,
        reasons=reasons,
        rul_band=rul_band,
        anomaly_flag=anomaly_flag,
        stale_signals=stale_signals,
        urgency_horizon_minutes=URGENCY_BY_PRIORITY[priority],
    )