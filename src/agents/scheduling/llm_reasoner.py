"""Optional Groq-backed reasoning for maintenance proposals.

The LLM enriches a deterministic assessment. It does not set priority,
schedule downtime, or issue machine-control commands.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

logger = logging.getLogger("scheduling_agent.llm")

DEFAULT_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """
You are the maintenance-reasoning component of an industrial multi-agent system.
You receive only machine-health model outputs and a small amount of operational
context. You do not know the true internal machine state or confirmed failure
cause. Treat model outputs as evidence, not certainty.

Return one JSON object with exactly these keys:
{
  "likely_cause": "short hypothesis, explicitly uncertain where appropriate",
  "recommended_action": "one practical inspection/review recommendation",
  "inspection_steps": ["step 1", "step 2"],
  "missing_information": ["information that would help confirm the hypothesis"],
  "confidence": "low|moderate|high",
  "evidence_summary": "one or two sentences grounded only in supplied evidence"
}

Rules:
- Do not change or propose a numeric priority; deterministic software owns priority.
- Do not invent machine history, component names, sensor readings, or root causes.
- Do not claim that a failure is confirmed from an anomaly score or RUL estimate.
- Do not issue commands to stop, restart, or alter a machine. A qualified supervisor
  must decide any shutdown or consequential action.
- RUL is expressed in simulation steps, not hours or minutes. Never convert it to time.
- If evidence is sparse or signals conflict, say so and use low confidence.
- Keep the response concise and return JSON only.
""".strip()


class GroqMaintenanceReasoner:
    """Thin optional wrapper around the existing Groq provider."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = None
        try:
            from groq import Groq
        except ImportError:
            logger.warning("groq package is not installed; using rules-only proposals.")
            return

        key = ''
        if not key:
            logger.warning("GROQ_API_KEY is not set; using rules-only proposals.")
            return

        self.client = Groq(api_key=key)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def assess(self, context: dict[str, Any]) -> dict[str, Any] | None:
        """Return validated, bounded LLM fields or None on any failure."""
        if self.client is None:
            return None

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(context, ensure_ascii=False, allow_nan=False),
                    },
                ],
                temperature=0.2,
                max_tokens=500,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty LLM response")
            parsed = json.loads(content)
            return self._validate(parsed)
        except Exception as exc:  # LLM failure must not kill the agent
            logger.warning("LLM maintenance reasoning failed; using rules-only proposal: %s", exc)
            return None

    @staticmethod
    def _validate(value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("LLM response must be a JSON object")

        def short_text(key: str, default: str, limit: int = 600) -> str:
            raw = value.get(key, default)
            if not isinstance(raw, str):
                return default
            cleaned = " ".join(raw.split()).strip()
            return cleaned[:limit] if cleaned else default

        confidence = value.get("confidence", "low")
        if not isinstance(confidence, str) or confidence not in {"low", "moderate", "high"}:
            confidence = "low"

        def string_list(key: str, limit_items: int = 5) -> list[str]:
            raw = value.get(key, [])
            if not isinstance(raw, list):
                return []
            output = []
            for item in raw[:limit_items]:
                if isinstance(item, str):
                    item = " ".join(item.split()).strip()
                    if item:
                        output.append(item[:240])
            return output

        recommended_action = short_text(
            "recommended_action",
            "Have a qualified technician review the available machine-health evidence.",
        )
        # Defense in depth: never pass an LLM-generated direct machine-control
        # instruction through as the proposed action.
        unsafe_phrases = (
            "shut down", "shutdown", "stop the machine", "stop machine",
            "restart the machine", "restart machine", "turn off the machine",
            "disable the machine", "override the safety", "bypass the safety",
        )
        if any(phrase in recommended_action.lower() for phrase in unsafe_phrases):
            recommended_action = (
                "Request qualified supervisor review and have a technician verify "
                "the machine condition before any control action."
            )

        return {
            "likely_cause": short_text("likely_cause", "Cause cannot be determined from current evidence."),
            "recommended_action": recommended_action,
            "inspection_steps": string_list("inspection_steps"),
            "missing_information": string_list("missing_information"),
            "confidence": confidence,
            "evidence_summary": short_text(
                "evidence_summary",
                "The recommendation is limited by the health evidence currently available.",
            ),
        }
