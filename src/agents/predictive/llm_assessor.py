"""Predictive-agent LLM assessor.

Two distinct prompts, one per trigger axis:

  - RUL assessment: given the predicted remaining useful life (in
    simulation steps), its band, and the recent sensor trend, explain
    likely cause, urgency, and a recommended action.
  - Anomaly assessment: given the anomaly reconstruction score, its
    threshold, the flag, and the recent sensor trend, explain the
    same.

Both prompts explicitly say the LLM does not have access to internal
machine state (health, wear, injected fault) — that boundary is
enforced here as well as in the caller.

This module never touches MQTT, the model, or the memory. It returns
a dict (or None) per call. The caller stores the result in memory and
attaches it to the published message.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from agents.common.llm_validation import validate_maintenance_assessment

logger = logging.getLogger("predictive_agent.llm")

DEFAULT_MODEL = "openai/gpt-oss-120b"

_COMMON_RULES = (
    "You are the Predictive Maintenance Agent in an industrial factory "
    "multi-agent system. You receive model outputs and a short sensor "
    "trend summary for one machine. You do NOT have access to the "
    "machine's internal wear state, health, or confirmed failure cause "
    "— only what a real sensor deployment would see. Treat model "
    "outputs as evidence, not certainty. "
    "Respond with STRICT JSON only, no other text. "
)

_RUL_SYSTEM_PROMPT = _COMMON_RULES + (
    "The input is a predicted Remaining Useful Life (RUL), measured in "
    "simulation steps, and its band (critical/near/medium/far). "
    "Return exactly this shape: "
    '{"likely_cause": "<short hypothesis>", '
    '"recommended_action": "<one practical inspection/review recommendation>", '
    '"inspection_steps": ["step 1", "step 2"], '
    '"missing_information": ["..."], '
    '"confidence": "low|moderate|high", '
    '"evidence_summary": "<one or two sentences grounded only in the supplied evidence>"} '
    "Do not propose a numeric priority; deterministic software owns priority. "
    "Do not claim a failure is confirmed. Do not convert simulation steps to clock time. "
    "Do not issue commands to stop, restart, or alter the machine."
)

_ANOMALY_SYSTEM_PROMPT = _COMMON_RULES + (
    "The input is an anomaly reconstruction score compared against a "
    "threshold, the resulting flag, and a sensor trend summary. "
    "Return exactly this shape: "
    '{"likely_cause": "<short hypothesis>", '
    '"recommended_action": "<one practical inspection/review recommendation>", '
    '"inspection_steps": ["step 1", "step 2"], '
    '"missing_information": ["..."], '
    '"confidence": "low|moderate|high", '
    '"evidence_summary": "<one or two sentences grounded only in the supplied evidence>"} '
    "An anomaly flag indicates unusual sensor behaviour, not a confirmed "
    "root cause. Do not propose a numeric priority. Do not issue commands "
    "to stop, restart, or alter the machine."
)


class PredictiveLLMAssessor:
    """Thin wrapper around Groq, optional by construction."""

    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = None
        try:
            from groq import Groq
        except ImportError:
            logger.warning("groq package is not installed; LLM assessments disabled.")
            return
        key = ""
        if not key:
            logger.warning("GROQ_API_KEY is not set; LLM assessments disabled.")
            return
        self.client = Groq(api_key=key)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def assess_rul(
        self,
        machine_id: str,
        rul: float | None,
        rul_band: str,
        rul_cap: float | None,
        trend_summary: str,
    ) -> dict | None:
        user_prompt = (
            f"Machine: {machine_id}\n"
            f"Predicted RUL (simulation steps): "
            f"{rul if rul is not None else 'unavailable'}\n"
            f"RUL cap: {rul_cap if rul_cap is not None else 'unknown'}\n"
            f"RUL band: {rul_band}\n"
            f"Recent sensor trend: {trend_summary}\n"
        )
        return self._call(_RUL_SYSTEM_PROMPT, user_prompt, machine_id, "rul")

    def assess_anomaly(
        self,
        machine_id: str,
        anomaly_score: float | None,
        threshold: float | None,
        is_anomaly: bool | None,
        anomaly_flag: str,
        trend_summary: str,
    ) -> dict | None:
        user_prompt = (
            f"Machine: {machine_id}\n"
            f"Anomaly reconstruction score: "
            f"{anomaly_score if anomaly_score is not None else 'unavailable'}\n"
            f"Threshold: {threshold if threshold is not None else 'unknown'}\n"
            f"Anomaly flagged: {is_anomaly}\n"
            f"Anomaly band: {anomaly_flag}\n"
            f"Recent sensor trend: {trend_summary}\n"
        )
        return self._call(_ANOMALY_SYSTEM_PROMPT, user_prompt, machine_id, "anomaly")

    def _call(self, system_prompt: str, user_prompt: str, machine_id: str, kind: str) -> dict | None:
        if self.client is None:
            return None
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                max_tokens=1000,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty LLM response")
            parsed = json.loads(content)
            return validate_maintenance_assessment(parsed)
        except json.JSONDecodeError:
            logger.warning("LLM %s response for %s was not valid JSON; skipping.", kind, machine_id)
            return None
        except Exception as exc:  # noqa: BLE001 — never let LLM failure kill the agent
            logger.warning("LLM %s call failed for %s: %s", kind, machine_id, exc)
            return None