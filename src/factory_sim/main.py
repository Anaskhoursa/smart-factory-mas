
import argparse
import json

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

    for i in range(args.steps):
        if i == args.inject_at:
            factory.inject_failure(
                machine_id=args.machine,
                mode=FailureMode(args.failure),
                severity=args.severity,
            )

        snapshot = factory.step()

        print(
            json.dumps(
                snapshot.model_dump(mode="json"),
                separators=(",", ":"),
            )
        )


if __name__ == "__main__":
    main()