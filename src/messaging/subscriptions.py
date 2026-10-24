"""MQTT subscriptions for each factory agent and service."""

from . import topics

QOS_TELEMETRY = 0
QOS_EVENTS = 1

SUBSCRIPTIONS = {
    "predictive": [
        ("factory/machines/+/telemetry", QOS_TELEMETRY),
        (topics.ALL_MACHINE_TELEMETRY, QOS_TELEMETRY),
    ],
    "scheduling": [
        (topics.V1_HEALTH_ANOMALY, QOS_EVENTS),
        (topics.V1_HEALTH_RUL, QOS_EVENTS),
        (topics.V1_PRODUCTION_CONSTRAINTS, QOS_EVENTS),
        (topics.SCHEDULING_REQUESTS, QOS_EVENTS),
    ],
    "production": [
        ("factory/machines/+/telemetry", QOS_TELEMETRY),
        ("factory/machines/+/events", QOS_EVENTS),
        (topics.ALL_MACHINE_TELEMETRY, QOS_TELEMETRY),
        (topics.ALL_MACHINE_EVENTS, QOS_EVENTS),
    ],
    "supervisor": [
        (topics.V1_HEALTH_ANOMALY, QOS_EVENTS),
        (topics.V1_HEALTH_RUL, QOS_EVENTS),
        (topics.V1_MAINTENANCE_PROPOSALS, QOS_EVENTS),
        (topics.V1_PRODUCTION_CONSTRAINTS, QOS_EVENTS),
        (topics.V1_PRODUCTION_STATUS, QOS_EVENTS),
        (topics.V1_DECISION_RESULTS, QOS_EVENTS),
        (topics.V1_APPROVAL_RESPONSES, QOS_EVENTS),
        (topics.V1_COMMAND_RESULTS, QOS_EVENTS),
        (topics.SUPERVISOR_ALERTS, QOS_EVENTS),
        (topics.SUPERVISOR_APPROVALS, QOS_EVENTS),
    ],
    "optimizer": [
        (topics.V1_DECISION_REQUESTS, QOS_EVENTS),
    ],
    "command_handler": [
        (topics.V1_COMMAND_REQUESTS, QOS_EVENTS),
    ],
    "approval_ui": [
        (topics.V1_APPROVAL_REQUESTS, QOS_EVENTS),
    ],
    "dashboard": [
        (topics.V1_HEALTH_ANOMALY, QOS_EVENTS),
        (topics.V1_HEALTH_RUL, QOS_EVENTS),
        (topics.V1_PRODUCTION_STATUS, QOS_EVENTS),
        (topics.V1_COMMAND_RESULTS, QOS_EVENTS),
        (topics.ALL_AGENT_STATUS, QOS_EVENTS),
    ],
    "scheduling": [
    (topics.V1_SIM_CLOCK, QOS_EVENTS),          # ← new
    (topics.V1_HEALTH_ANOMALY, QOS_EVENTS),
    (topics.V1_HEALTH_RUL, QOS_EVENTS),
    (topics.V1_PRODUCTION_CONSTRAINTS, QOS_EVENTS),
    (topics.SCHEDULING_REQUESTS, QOS_EVENTS),
],
}


def subscriptions_for(component: str):
    """Return a copy of the subscriptions for a component."""
    try:
        return list(SUBSCRIPTIONS[component])
    except KeyError as exc:
        raise ValueError(
            f"Unknown component {component!r}. "
            f"Available: {', '.join(sorted(SUBSCRIPTIONS))}"
        ) from exc
