"""Run the simulator and publish per-machine telemetry over MQTT.

Example:
    python -m factory_sim.mqtt_main --steps 120 --inject-at 30 --machine M2
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time

import paho.mqtt.client as mqtt

from config import SimulationConfig
from factory import Factory
from models import FailureMode

LOG = logging.getLogger("factory-mqtt-simulator")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart Factory Simulator with MQTT telemetry")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--inject-at", type=int, default=30)
    parser.add_argument("--machine", type=str, default="M2")
    parser.add_argument(
        "--failure", choices=[mode.value for mode in FailureMode],
        default="bearing_wear",
    )
    parser.add_argument("--severity", type=float, default=0.5)
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Real seconds between simulated steps")
    parser.add_argument("--host", default=os.getenv("MQTT_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MQTT_PORT", "1883")))
    args = parser.parse_args()

    if args.steps < 0 or args.inject_at < 0:
        parser.error("--steps and --inject-at must be non-negative")
    if not 0 < args.severity <= 1:
        parser.error("--severity must be in (0, 1]")
    if args.interval < 0:
        parser.error("--interval must be non-negative")

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id="factory-simulator",
        protocol=mqtt.MQTTv5,
    )
    client.connect(args.host, args.port, keepalive=60)
    client.loop_start()

    factory = Factory(config=SimulationConfig(seed=args.seed))
    try:
        # Give the network loop a moment to finish the initial connection.
        time.sleep(0.2)
        for step in range(args.steps):
            if step == args.inject_at:
                factory.inject_failure(
                    machine_id=args.machine,
                    mode=FailureMode(args.failure),
                    severity=args.severity,
                )
                LOG.warning("Injected %s on %s at step %d",
                            args.failure, args.machine, step)

            snapshot = factory.step()
            for machine in snapshot.machines:
                topic = f"factory/machines/{machine.machine_id}/telemetry"
                payload = machine.model_dump_json()
                info = client.publish(topic, payload, qos=1, retain=False)
                info.wait_for_publish()
            LOG.info("Published telemetry for %d machines at simulation step %d",
                     len(snapshot.machines), step)
            if args.interval:
                time.sleep(args.interval)
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
