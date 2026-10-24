import argparse
import json
import time

from factory import Factory
from models import FailureMode


def main():
    parser = argparse.ArgumentParser(
        description="Smart Factory Simulator"
    )
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inject-at", type=int, default=30)
    parser.add_argument(
        "--machine",
        type=str,
        default="M2",
    )
    parser.add_argument(
        "--failure",
        type=str,
        choices=[mode.value for mode in FailureMode],
        default="bearing_wear",
    )
    parser.add_argument("--severity", type=float, default=0.5)
    parser.add_argument(
        "--publish-mqtt",
        action="store_true",
        help="Publish each machine's telemetry/events to MQTT as the simulation runs.",
    )
    parser.add_argument("--mqtt-host", type=str, default="localhost")
    parser.add_argument("--mqtt-port", type=int, default=1883)
    parser.add_argument(
        "--step-delay",
        type=float,
        default=0.0,
        help="Seconds to sleep between steps (useful with --publish-mqtt, "
             "so a live agent can process messages at a readable pace "
             "instead of thousands of steps firing instantly).",
    )
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Suppress the stdout JSON snapshot output.",
    )
    args = parser.parse_args()

    if args.steps < 0:
        parser.error("--steps must be non-negative")
    if args.inject_at < 0:
        parser.error("--inject-at must be non-negative")
    if not 0 < args.severity <= 1:
        parser.error("--severity must be in (0, 1]")

    from config import SimulationConfig

    factory = Factory(
        config=SimulationConfig(seed=args.seed)
    )

    mqtt_client = None
    topics = None
    if args.publish_mqtt:
        # Imported lazily so the simulator still runs standalone
        # (python main.py, no --publish-mqtt) without the mqtt
        # package or a broker being required.
        from messaging.client import MQTTClient
        from messaging import topics as topics_module
        from messaging.subscriptions import QOS_TELEMETRY, QOS_EVENTS

        topics = topics_module
        mqtt_client = MQTTClient(
            client_id="factory-simulator",
            host=args.mqtt_host,
            port=args.mqtt_port,
        )
        mqtt_client.connect()
        print(f"[Simulator] Publishing telemetry to MQTT at {args.mqtt_host}:{args.mqtt_port}", flush=True)

    try:
        for i in range(args.steps):
            if i == args.inject_at:
                factory.inject_failure(
                    machine_id=args.machine,
                    mode=FailureMode(args.failure),
                    severity=args.severity,
                )

            snapshot = factory.step()

            # if not args.no_print:
            #     print(
            #         json.dumps(
            #             snapshot.model_dump(mode="json"),
            #             separators=(",", ":"),
            #         ), flush=True)
            from datetime import datetime, timezone

            if mqtt_client is not None:
                # Publish the simulation clock first, so the scheduler sees the
                # current sim minute before any telemetry from the same step.
                mqtt_client.publish(
                    topics.V1_SIM_CLOCK,
                    json.dumps({
                        "sim_minute": int(i * factory.config.step_minutes),
                        "step": i,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }),
                    qos=QOS_EVENTS,
                )

                for machine_telemetry in snapshot.machines:
                    topic = topics.V1_MACHINE_TELEMETRY.format(
                        machine_id=machine_telemetry.machine_id
                    )
                    mqtt_client.publish(
                        topic,
                        machine_telemetry.model_dump_json(),
                        qos=QOS_TELEMETRY,
                    )

                for event in snapshot.events:
                    topic = topics.V1_MACHINE_EVENTS.format(
                        machine_id=event.machine_id
                    )
                    mqtt_client.publish(
                        topic,
                        event.model_dump_json(),
                        qos=QOS_EVENTS,
                    )

            if args.step_delay > 0:
                time.sleep(args.step_delay)

    finally:
        if mqtt_client is not None:
            mqtt_client.disconnect()


if __name__ == "__main__":
    main()