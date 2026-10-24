"""Canonical MQTT topic names for the smart-factory multi-agent system.

Legacy topics remain available while existing simulator code is migrated.
New agent-to-agent messages use the versioned factory/v1 namespace.
"""

# -------------------------------------------------------------------
# Legacy topics — retained for compatibility with existing code
# -------------------------------------------------------------------

FACTORY_TELEMETRY = "factory/machines/{machine_id}/telemetry"
FACTORY_EVENTS = "factory/machines/{machine_id}/events"

MAINTENANCE_PREDICTIONS = "factory/maintenance/predictions"
MAINTENANCE_COMMANDS = "factory/maintenance/commands"

SCHEDULING_REQUESTS = "factory/scheduling/requests"
SCHEDULING_DECISIONS = "factory/scheduling/decisions"

SUPERVISOR_ALERTS = "factory/supervisor/alerts"
SUPERVISOR_APPROVALS = "factory/supervisor/approvals"


# -------------------------------------------------------------------
# Versioned namespace for the multi-agent system
# -------------------------------------------------------------------

V1_PREFIX = "factory/v1"


# Machine telemetry and events
V1_MACHINE_TELEMETRY = f"{V1_PREFIX}/machines/{{machine_id}}/telemetry"
V1_MACHINE_EVENTS = f"{V1_PREFIX}/machines/{{machine_id}}/events"


# Predictive maintenance / machine health agent
V1_HEALTH_ANOMALY = f"{V1_PREFIX}/health/anomaly"
V1_HEALTH_RUL = f"{V1_PREFIX}/health/rul"


# Maintenance scheduling agent
V1_MAINTENANCE_PROPOSALS = f"{V1_PREFIX}/maintenance/proposals"


# Production agent
V1_PRODUCTION_CONSTRAINTS = f"{V1_PREFIX}/production/constraints"
V1_PRODUCTION_STATUS = f"{V1_PREFIX}/production/status"


# Supervisor / orchestrator and optimization service
V1_DECISION_REQUESTS = f"{V1_PREFIX}/decisions/requests"
V1_DECISION_RESULTS = f"{V1_PREFIX}/decisions/results"


# Human approval workflow
V1_APPROVAL_REQUESTS = f"{V1_PREFIX}/approvals/requests"
V1_APPROVAL_RESPONSES = f"{V1_PREFIX}/approvals/responses"


# Command handler
V1_COMMAND_REQUESTS = f"{V1_PREFIX}/commands/requests"
V1_COMMAND_RESULTS = f"{V1_PREFIX}/commands/results"


# Agent health and lifecycle status
V1_AGENT_STATUS = f"{V1_PREFIX}/system/agents/{{agent_id}}/status"


# -------------------------------------------------------------------
# MQTT subscription filters
#
# Wildcards (+ and #) are valid in subscription filters, not in
# topics used to publish individual messages.
# -------------------------------------------------------------------

ALL_MACHINE_TELEMETRY = f"{V1_PREFIX}/machines/+/telemetry"
ALL_MACHINE_EVENTS = f"{V1_PREFIX}/machines/+/events"
ALL_AGENT_STATUS = f"{V1_PREFIX}/system/agents/+/status"


# Simulation clock (published once per factory step by the simulator)
V1_SIM_CLOCK = f"{V1_PREFIX}/sim/clock"