"""Subscription filters for each independent factory agent/service.

Values are (topic_filter, qos) pairs. MQTT wildcards are valid in filters only.
This module declares routing policy; it does not start clients or execute actions.
"""

from __future__ import annotations

from . import topics

# QoS 0 is sufficient for high-frequency telemetry. State changes, proposals,
# decisions, approvals, and command messages use QoS 1 for at-least-once delivery.
TELEMETRY_QOS = 0
EVENT_QOS = 1

SUBSCRIPTIONS: dict[str, tuple[tuple[str, int], ...]] = {
    "predictive": (
        # Legacy topic supports the current simulator during migration.
        ("factory/machines/+/telemetry", TELEMETRY_QOS),
        (topics.ALL_MACHINE_TELEMETRY, TELEMETRY_QOS),
    ),
    "scheduling": (
        (topics.V1_HEALTH_ANOMALY, EVENT_QOS),
        (topics.V1_HEALTH_RUL, EVENT_QOS),
        (topics.V1_PRODUCTION_CONSTRAINTS, EVENT_QOS),
        (topics.SCHEDULING_REQUESTS, EVENT_QOS),
    ),
    "production": (
        (topics.ALL_MACHINE_TELEMETRY, TELEMETRY_QOS),
        ("factory/machines/+/telemetry", TELEMETRY_QOS),
        (topics.ALL_MACHINE_EVENTS, EVENT_QOS),
        ("factory/machines/+/events", EVENT_QOS),
    ),
    "supervisor": (
        (topics.V1_HEALTH_ANOMALY, EVENT_QOS),
        (topics.V1_HEALTH_RUL, EVENT_QOS),
        (topics.V1_MAINTENANCE_PROPOSALS, EVENT_QOS),
        (topics.V1_PRODUCTION_CONSTRAINTS, EVENT_QOS),
        (topics.V1_PRODUCTION_STATUS, EVENT_QOS),
        (topics.V1_DECISION_RESULTS, EVENT_QOS),
        (topics.V1_APPROVAL_RESPONSES, EVENT_QOS),
        (topics.V1_COMMAND_RESULTS, EVENT_QOS),
        (topics.SUPERVISOR_ALERTS, EVENT_QOS),
        (topics.SUPERVISOR_APPROVALS, EVENT_QOS),
    ),
    # OR-Tools is a deterministic service, not a fifth agent.
    "optimizer": (
        (topics.V1_DECISION_REQUESTS, EVENT_QOS),
    ),
    # The command handler should validate authorization and approval state
    # before executing any request; it must never blindly actuate on receipt.
    "command_handler": (
        (topics.V1_COMMAND_REQUESTS, EVENT_QOS),
    ),
    "approval_ui": (
        (topics.V1_APPROVAL_REQUESTS, EVENT_QOS),
    ),
    "dashboard": (
        (topics.V1_HEALTH_ANOMALY, EVENT_QOS),
        (topics.V1_HEALTH_RUL, EVENT_QOS),
        (topics.V1_PRODUCTION_STATUS, EVENT_QOS),
        (topics.V1_COMMAND_RESULTS, EVENT_QOS),
        (topics.ALL_AGENT_STATUS, EVENT_QOS),
    ),
}


def subscriptions_for(component: str) -> list[tuple[str, int]]:
    """Return a copy of the subscription filters for a named component."""
    try:
        return list(SUBSCRIPTIONS[component])
    except KeyError as exc:
        available = ", ".join(sorted(SUBSCRIPTIONS))
        raise ValueError(
            f"Unknown MQTT component {component!r}. Available: {available}"
        ) from exc
