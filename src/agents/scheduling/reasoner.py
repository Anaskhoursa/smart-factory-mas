"""Module 4 -- LLM narration.

One call per proposal. Re-called only when the priority changes or the
feasible-candidate set changes materially. Never on a timer.

The prompt receives:
  - health evidence (RUL, anomaly score, flags, thresholds)
  - the predictive agent's latest RUL and anomaly assessments
    (so the scheduler builds on them, not from scratch)
  - deterministic reasons (why this priority)
  - stale warnings
  - the feasible candidates (window, downtime, required resources)
  - an explicit "no feasible candidates" note when applicable

It does NOT receive:
  - raw sensor telemetry (the predictive agent owns that)
  - the scheduling algorithm itself (it can't compute schedules)
  - the ability to set priority

The validator from agents/common/llm_validation.py sanitizes the
output. Failures return None and the agent falls back to the
deterministic action.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any

from agents.common.llm_validation import validate_maintenance_assessment

logger = logging.getLogger("scheduling_agent.llm")

DEFAULT_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = """\
You are the maintenance-reasoning component of an industrial multi-agent system.
You receive machine-health model outputs, the predictive agent's prior assessments,
deterministic scheduling reasons, and a set of feasible maintenance windows chosen
by a separate deterministic optimizer. You do not know the true internal machine
state or confirmed failure cause. Treat model outputs as evidence, not certainty.

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
- Do not compute or propose alternative schedules. The feasible windows are given.
- Do not invent machine history, component names, sensor readings, or root causes.
- Do not claim a failure is confirmed from an anomaly score or RUL estimate.
- Do not issue commands to stop, restart, or alter a machine.
- RUL is expressed in simulation minutes, not wall-clock time.
- If the feasible set is empty, say so and explain what information or resource
  would be needed to make a schedule possible. Do not invent a window.
- If evidence is sparse or signals conflict, say so and use low confidence.
- Keep the response concise and return JSON only.
""".strip()


class SchedulingReasoner:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = None
        try:
            from groq import Groq
        except ImportError:
            logger.warning("groq package is not installed; LLM narration disabled.")
            return
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            logger.warning("GROQ_API_KEY is not set; LLM narration disabled.")
            return
        self.client = Groq(api_key=key)

    @property
    def enabled(self) -> bool:
        return self.client is not None

    def narrate(self, context: dict[str, Any]) -> dict | None:
        if self.client is None:
            return None
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(context, ensure_ascii=False, allow_nan=False)},
                ],
                temperature=0.2,
                max_tokens=600,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty LLM response")
            parsed = json.loads(content)
            return validate_maintenance_assessment(parsed)
        except json.JSONDecodeError:
            logger.warning("Scheduling LLM response was not valid JSON; skipping.")
            return None
        except Exception as exc:  # noqa: BLE001
            logger.warning("Scheduling LLM call failed: %s", exc)
            return None