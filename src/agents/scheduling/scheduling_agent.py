"""LLM-enhanced Maintenance Scheduling Agent.

Wires the five modules (rules, optimizer, resources, reasoner, workflow)
around an MQTT client. Consumes V1_HEALTH_RUL, V1_HEALTH_ANOMALY, and
V1_PRODUCTION_CONSTRAINTS. Publishes V1_MAINTENANCE_PROPOSALS.

From repository root (Windows PowerShell shown; adjust for shell):
    $env:PYTHONPATH = "src"
    python -m agents.scheduling.scheduling_agent --data-dir <path to JSON files>

The five modules are pure; this class only manages per-machine state,
the queue hand-off from the network thread, and MQTT I/O. Everything
the modules compute is testable without MQTT.
"""
from __future__ import annotations

import argparse
import json
import logging
import queue
import time
from pathlib import Path
from typing import Any

from messaging.client import MQTTClient
from messaging.subscriptions import subscriptions_for, QOS_EVENTS
from messaging import topics

from data_loader import SchedulingData, load_scheduling_data
from models import PriorityAssessment
from optimizer import generate_candidates, production_impact
from reasoner import SchedulingReasoner
from resources import filter_feasible
from rules import prioritize
from workflow import Workflow

logger = logging.getLogger("scheduling_agent")

STALE_AFTER_SECONDS = 120.0
PROPOSAL_REFRESH_SECONDS = 60.0


def _float_or_none(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result or result == float("inf") or result == float("-inf"):
        return None
    return result


class MaintenanceSchedulingAgent:
    def __init__(
        self,
        data: SchedulingData,
        mqtt_host: str = "localhost",
        mqtt_port: int = 1883,
        enable_llm: bool = True,
        groq_api_key: str | None = None,
        groq_model: str = "openai/gpt-oss-120b",
        stale_after_seconds: float = STALE_AFTER_SECONDS,
        proposal_refresh_seconds: float = PROPOSAL_REFRESH_SECONDS,
        audit_log_path: str | Path = "scheduling_audit.jsonl",
        now_sim_minute_provider=None,
    ):
        self.data = data
        self.stale_after_seconds = max(0.0, stale_after_seconds)

        self.reasoner = (
            SchedulingReasoner(api_key=groq_api_key, model=groq_model)
            if enable_llm else None
        )
        self.workflow = Workflow(
            audit_log_path=audit_log_path,
            proposal_refresh_seconds=proposal_refresh_seconds,
        )

        # In this build the agent does not advance simulation time itself;
        # it uses a monotonic wall-clock estimate of "sim minutes elapsed
        # since process start". A production deployment would receive the
        # simulator's clock over MQTT. For now, per the README, the sim
        # is 1 sim minute per telemetry step. We approximate: sim minute
        # = monotonic seconds since the agent's first message. If you
        # later publish a clock topic, wire it here.
        self._sim_minute: int = 0

        self._queue: queue.Queue[tuple[str, bytes, int]] = queue.Queue()
        self._machines: dict[str, dict[str, Any]] = {}

        self.mqtt = MQTTClient(
            client_id="maintenance-scheduling-agent",
            host=mqtt_host,
            port=mqtt_port,
            on_message=self._enqueue_message,
        )

    # --- lifecycle ---

    def start(self) -> None:
        self.mqtt.connect()
        for topic_name, qos in subscriptions_for("scheduling"):
            self.mqtt.subscribe(topic_name, qos=qos)
            logger.info("Subscribed to %s (qos=%s)", topic_name, qos)
        logger.info(
            "Maintenance Scheduling Agent started (LLM=%s).",
            bool(self.reasoner and self.reasoner.enabled),
        )

    def stop(self) -> None:
        self.mqtt.disconnect()

    # --- network thread ---

    def _enqueue_message(self, topic_name: str, payload: bytes, qos: int) -> None:
        self._queue.put((topic_name, payload, qos))

    # --- main loop ---

    def run_forever(self) -> None:
        logger.info("Running. Ctrl+C to stop.")
        try:
            while True:
                try:
                    topic_name, payload, _ = self._queue.get(timeout=1.0)
                    self._handle_message(topic_name, payload)
                except queue.Empty:
                    pass
                except Exception:
                    logger.exception("Could not process incoming message")
                self._refresh_stale_states()
        except KeyboardInterrupt:
            logger.info("Shutting down.")
        finally:
            self.stop()

    # --- state ---

    def _now_sim_minute(self) -> int:
        return self._sim_minute

    def _new_state(self) -> dict[str, Any]:
        return {
            "rul": None,
            "rul_cap": None,
            "anomaly_score": None,
            "is_anomaly": None,
            "anomaly_threshold": None,
            "rul_received_at": None,
            "anomaly_received_at": None,
            "production_constraints": None,
            "constraints_received_at": None,
            "last_received_at": None,
            "predictive_assessments": {"rul": None, "anomaly": None},
            "reassessment_requested": False,
        }

    def _stale_signals(self, state: dict, now: float) -> list[str]:
        if self.stale_after_seconds <= 0:
            return []
        stale = []
        for key, label in (
            ("rul_received_at", "RUL prediction is stale"),
            ("anomaly_received_at", "anomaly prediction is stale"),
        ):
            received = state[key]
            if received is not None and now - received > self.stale_after_seconds:
                stale.append(label)
        return stale

    # --- message handling ---

    def _handle_message(self, topic_name: str, payload: bytes) -> None:
        try:
            message = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            logger.warning("Ignoring invalid JSON on %s", topic_name)
            return
        if not isinstance(message, dict):
            return
        if topic_name not in {
            topics.V1_HEALTH_RUL,
            topics.V1_HEALTH_ANOMALY,
            topics.V1_PRODUCTION_CONSTRAINTS,
        }:
            return
        if topic_name == topics.V1_SIM_CLOCK:
            sim_minute = message.get("sim_minute")
            if isinstance(sim_minute, int) and sim_minute >= 0:
                self._sim_minute = sim_minute
            else:
                logger.warning("Ignoring invalid sim clock message: %r", message)
            return
        machine_id = message.get("machine_id")
        if not isinstance(machine_id, str) or not machine_id.strip():
            return
        machine_id = machine_id.strip()
        state = self._machines.setdefault(machine_id, self._new_state())
        now_mono = time.monotonic()

        if topic_name == topics.V1_HEALTH_RUL:
            rul = _float_or_none(message.get("predicted_rul"))
            if rul is None:
                return
            state["rul"] = rul
            state["rul_cap"] = _float_or_none(message.get("rul_cap"))
            state["rul_received_at"] = now_mono
            assessment = message.get("assessment")
            if isinstance(assessment, dict):
                state["predictive_assessments"]["rul"] = assessment

        elif topic_name == topics.V1_HEALTH_ANOMALY:
            score = _float_or_none(message.get("anomaly_score"))
            if score is None:
                return
            is_anomaly = message.get("is_anomaly")
            if not isinstance(is_anomaly, bool):
                return
            state["anomaly_score"] = score
            state["is_anomaly"] = is_anomaly
            state["anomaly_threshold"] = _float_or_none(message.get("threshold"))
            state["anomaly_received_at"] = now_mono
            assessment = message.get("assessment")
            if isinstance(assessment, dict):
                state["predictive_assessments"]["anomaly"] = assessment

        else:  # production constraints
            state["production_constraints"] = message
            state["constraints_received_at"] = now_mono

        state["last_received_at"] = now_mono
        self._evaluate_and_publish(machine_id, state)

    # --- main pipeline per machine ---

    def _evaluate_and_publish(self, machine_id: str, state: dict) -> None:
        now_mono = time.monotonic()
        now_sim = self._now_sim_minute()
        stale = self._stale_signals(state, now_mono)

        priority = prioritize(
            machine_id=machine_id,
            rul=state["rul"],
            is_anomaly=state["is_anomaly"],
            stale_signals=stale,
        )

        # Which failure mode does the predictive evidence point at?
        suspected = self._suspected_failure_mode(state)
        task = self.data.task_for_failure(suspected)

        candidates = generate_candidates(
            machine_id=machine_id,
            priority=priority,
            now_sim_minute=now_sim,
            task=task,
            calendar=self.data.calendar,
            orders=self.data.orders,
        )

        task_parts = (
            {req.part_id: req.quantity for req in task.required_parts}
            if task is not None else {}
        )
        task_skills = task.required_skills if task is not None else []

        feasible, infeasible = filter_feasible(
            candidates=candidates,
            task_required_parts=task_parts,
            task_required_skills=task_skills,
            parts=self.data.parts,
            technicians=self.data.technicians,
            minutes_per_day=self.data.calendar.rules.minutes_per_day,
        )

        requested = bool(state.get("reassessment_requested"))
        llm_assessment = None
        if self.reasoner and self.reasoner.enabled and self.workflow.should_narrate(
            machine_id, priority, feasible, requested,
        ):
            context = self._llm_context(
                machine_id, state, priority, feasible, infeasible,
            )
            llm_assessment = self.reasoner.narrate(context)
            self.workflow.log_llm_call(machine_id, context, llm_assessment)
            state["reassessment_requested"] = False

        health_evidence = {
            "predicted_rul_steps": state["rul"],
            "rul_cap": state["rul_cap"],
            "anomaly_score": state["anomaly_score"],
            "is_anomaly": state["is_anomaly"],
            "anomaly_threshold": state["anomaly_threshold"],
        }

        impact = (
            production_impact(feasible[0].candidate, self.data.orders, now_sim)
            if feasible else {"orders_due_during_downtime": [], "downtime_minutes": 0}
        )

        proposal = self.workflow.build(
            machine_id=machine_id,
            priority=priority,
            feasible=feasible,
            infeasible=infeasible,
            llm_assessment=llm_assessment,
            health_evidence=health_evidence,
            predictive_assessments=state["predictive_assessments"],
            production_impact=impact,
        )

        if not self.workflow.should_publish(proposal):
            return

        self.mqtt.publish(
            topics.V1_MAINTENANCE_PROPOSALS,
            proposal.model_dump_json(),
            qos=QOS_EVENTS,
        )
        self.workflow.log_publish(proposal)
        logger.info(
            "[%s] proposal published priority=%s feasibility=%s LLM=%s",
            machine_id, proposal.priority, proposal.feasibility,
            bool(proposal.llm_assessment),
        )

    def _suspected_failure_mode(self, state: dict) -> str | None:
        """Pick the failure mode to look up a task for.

        The predictive agent's assessments include a `likely_cause`
        string but not a structured failure_mode. We map a small set of
        keywords in the anomaly assessment's likely_cause. If nothing
        matches, return None -> the 'unknown' fallback task is used.
        """
        anomaly = state["predictive_assessments"].get("anomaly")
        if not anomaly:
            return None
        text = (anomaly.get("likely_cause") or "").lower()
        for mode, keywords in (
            ("bearing_wear", ("bearing", "wear", "grinding")),
            ("overheating", ("heat", "overheat", "cooling", "thermal")),
            ("misalignment", ("align", "misalign", "shaft")),
            ("random_shock", ("shock", "impact", "mount")),
        ):
            if any(k in text for k in keywords):
                return mode
        return None

    def _llm_context(
        self,
        machine_id: str,
        state: dict,
        priority: PriorityAssessment,
        feasible: list,
        infeasible: list,
    ) -> dict:
        return {
            "machine_id": machine_id,
            "deterministic_priority": priority.priority,
            "urgency_horizon_minutes": priority.urgency_horizon_minutes,
            "health_evidence": {
                "predicted_rul_steps": state["rul"],
                "rul_cap": state["rul_cap"],
                "anomaly_score": state["anomaly_score"],
                "anomaly_flagged": state["is_anomaly"],
                "anomaly_threshold": state["anomaly_threshold"],
            },
            "predictive_agent_assessments": state["predictive_assessments"],
            "deterministic_reasons": priority.reasons,
            "stale_data_warnings": priority.stale_signals,
            "feasible_options": [
                {
                    "window_id": f.candidate.window_id,
                    "window_name": f.candidate.window_name,
                    "start_sim_minute": f.candidate.start_sim_minute,
                    "end_sim_minute": f.candidate.end_sim_minute,
                    "task_name": f.candidate.task_name,
                    "downtime_minutes": f.candidate.estimated_duration_minutes,
                    "assigned_technicians": f.assigned_technicians,
                    "consumed_parts": f.consumed_parts,
                }
                for f in feasible
            ],
            "infeasible_reasons": [r.model_dump() for r in infeasible],
            "no_feasible_options": len(feasible) == 0,
            "production_constraints": state["production_constraints"],
            "important_limitations": [
                "RUL is in simulation minutes; no step-to-clock-time mapping is configured.",
                "An anomaly flag indicates unusual sensor behaviour, not a confirmed failure cause.",
                "Feasible options are generated by a deterministic optimizer; do not propose alternatives.",
            ],
        }

    def _refresh_stale_states(self) -> None:
        if self.stale_after_seconds <= 0:
            return
        now = time.monotonic()
        for machine_id, state in list(self._machines.items()):
            if state["last_received_at"] is None:
                continue
            # Avoid re-running the pipeline every loop tick when nothing
            # has changed -- only re-evaluate machines whose last signal
            # is near or past the staleness boundary.
            if now - state["last_received_at"] >= self.stale_after_seconds:
                self._evaluate_and_publish(machine_id, state)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Maintenance Scheduling Agent")
    parser.add_argument("--mqtt-host", default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument(
        "--data-dir", required=True,
        help="Directory containing factory_calendar.json, production_orders.json, "
             "technicians.json, spare_parts.json, maintenance_tasks.json.",
    )
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--groq-api-key", default=None)
    parser.add_argument("--groq-model", default="openai/gpt-oss-120b")
    parser.add_argument("--stale-after", type=float, default=STALE_AFTER_SECONDS)
    parser.add_argument("--proposal-refresh", type=float, default=PROPOSAL_REFRESH_SECONDS)
    parser.add_argument(
        "--audit-log", default="scheduling_audit.jsonl",
        help="Path to append-only JSONL audit log.",
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    data = load_scheduling_data(args.data_dir)

    agent = MaintenanceSchedulingAgent(
        data=data,
        mqtt_host=args.mqtt_host,
        mqtt_port=args.mqtt_port,
        enable_llm=not args.no_llm,
        groq_api_key=args.groq_api_key,
        groq_model=args.groq_model,
        stale_after_seconds=args.stale_after,
        proposal_refresh_seconds=args.proposal_refresh,
        audit_log_path=args.audit_log,
    )
    agent.start()
    agent.run_forever()


if __name__ == "__main__":
    main()