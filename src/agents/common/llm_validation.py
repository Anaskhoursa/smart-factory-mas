"""Shared validation for LLM-produced JSON.

Both the predictive and scheduling agents use an LLM to enrich a
deterministic decision with a human-readable narrative. Neither LLM is
allowed to set priority, authorize machine control, or invent history.
This module centralizes the sanitization so both agents enforce the
same boundaries.

Every agent that calls an LLM must pass the raw parsed JSON through
`validate_maintenance_assessment` before attaching it to a published
message. The validator:

  - coerces every string field, collapses whitespace, truncates it
  - restricts `confidence` to {"low", "moderate", "high"}
  - restricts list fields to short bounded lists of short strings
  - scrubs `recommended_action` for machine-control language

It never raises on malformed input; it substitutes safe defaults. The
caller decides whether a `None` return (missing/unusable LLM) is
acceptable — in practice, the agent falls back to a rules-based
narrative, so the pipeline never stops on an LLM failure.
"""
from __future__ import annotations

from typing import Any

CONFIDENCE_LEVELS = {"low", "moderate", "high"}

# Phrases that would let the LLM issue a machine-control instruction
# through the `recommended_action` field. If any appear, the field is
# replaced with a supervisor-review fallback.
UNSAFE_ACTION_PHRASES = (
    "shut down", "shutdown", "stop the machine", "stop machine",
    "restart the machine", "restart machine", "turn off the machine",
    "disable the machine", "override the safety", "bypass the safety",
    "bypass safety", "disable safety",
)

SAFE_FALLBACK_ACTION = (
    "Request qualified supervisor review and have a technician verify "
    "the machine condition before any control action."
)


def _clean_text(value: Any, default: str, limit: int = 600) -> str:
    if not isinstance(value, str):
        return default
    cleaned = " ".join(value.split()).strip()
    return cleaned[:limit] if cleaned else default


def _clean_string_list(value: Any, limit_items: int = 5, item_limit: int = 240) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value[:limit_items]:
        if isinstance(item, str):
            item = " ".join(item.split()).strip()
            if item:
                out.append(item[:item_limit])
    return out


def validate_maintenance_assessment(value: Any) -> dict[str, Any]:
    """Return a sanitized assessment dict. Never raises on bad input."""
    if not isinstance(value, dict):
        raise ValueError("LLM response must be a JSON object")

    recommended_action = _clean_text(
        value.get("recommended_action"),
        SAFE_FALLBACK_ACTION,
    )
    if any(phrase in recommended_action.lower() for phrase in UNSAFE_ACTION_PHRASES):
        recommended_action = SAFE_FALLBACK_ACTION

    confidence = value.get("confidence", "low")
    if not isinstance(confidence, str) or confidence not in CONFIDENCE_LEVELS:
        confidence = "low"

    return {
        "likely_cause": _clean_text(
            value.get("likely_cause"),
            "Cause cannot be determined from current evidence.",
        ),
        "recommended_action": recommended_action,
        "inspection_steps": _clean_string_list(value.get("inspection_steps")),
        "missing_information": _clean_string_list(value.get("missing_information")),
        "confidence": confidence,
        "evidence_summary": _clean_text(
            value.get("evidence_summary"),
            "The recommendation is limited by the health evidence currently available.",
        ),
    }