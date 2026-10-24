"""Pure derivation of coarse health bands from raw model outputs.

The predictive agent's LLM trigger and assessment-memory dedup both
operate on *bands*, never raw values. This module owns the band
definitions so the predictive agent and the scheduling agent agree on
what "near failure" means.

Bands are deliberately coarse. The goal is that a slowly drifting RUL
does not re-trigger an LLM call every step, while a genuine regime
change (far -> medium, clear -> flagged) does. See the plan doc for
the reasoning.

Boundaries (matching scheduling_agent.py's existing thresholds):
  RUL:        critical  <  30
              near      30..60
              medium    60..120
              far       >= 120
  Anomaly:    flagged   is_anomaly == True
              clear     is_anomaly == False
              unknown   is_anomaly is None (model not warm yet)
"""
from __future__ import annotations

from dataclasses import dataclass

RUL_CRITICAL = 30.0
RUL_NEAR = 60.0
RUL_MEDIUM = 120.0

RUL_BAND_CRITICAL = "critical"
RUL_BAND_NEAR = "near"
RUL_BAND_MEDIUM = "medium"
RUL_BAND_FAR = "far"
RUL_BAND_UNKNOWN = "unknown"

ANOMALY_CLEAR = "clear"
ANOMALY_FLAGGED = "flagged"
ANOMALY_UNKNOWN = "unknown"

# Bands that trigger the RUL assessment streak. "near" and "critical"
# are the RUL regimes where a human narrative is materially useful.
RUL_STREAK_BANDS = frozenset({RUL_BAND_NEAR, RUL_BAND_CRITICAL})


def derive_rul_band(rul: float | None) -> str:
    """Map a raw RUL value to its coarse band. None -> unknown."""
    if rul is None:
        return RUL_BAND_UNKNOWN
    if rul < RUL_CRITICAL:
        return RUL_BAND_CRITICAL
    if rul < RUL_NEAR:
        return RUL_BAND_NEAR
    if rul < RUL_MEDIUM:
        return RUL_BAND_MEDIUM
    return RUL_BAND_FAR


def derive_anomaly_flag(is_anomaly: bool | None) -> str:
    """Map a raw is_anomaly bool to a string flag."""
    if is_anomaly is None:
        return ANOMALY_UNKNOWN
    return ANOMALY_FLAGGED if is_anomaly else ANOMALY_CLEAR


@dataclass(frozen=True)
class HealthState:
    """Coarse health snapshot for one machine, at one step."""
    rul: float | None
    rul_band: str
    anomaly_score: float | None
    is_anomaly: bool | None
    anomaly_flag: str

    @classmethod
    def from_prediction(cls, prediction: dict) -> "HealthState":
        rul = prediction.get("predicted_rul")
        is_anomaly = prediction.get("is_anomaly")
        return cls(
            rul=rul,
            rul_band=derive_rul_band(rul),
            anomaly_score=prediction.get("anomaly_score"),
            is_anomaly=is_anomaly,
            anomaly_flag=derive_anomaly_flag(is_anomaly),
        )

    def rul_streak_continues(self) -> bool:
        return self.rul_band in RUL_STREAK_BANDS

    def anomaly_streak_continues(self) -> bool:
        return self.anomaly_flag == ANOMALY_FLAGGED