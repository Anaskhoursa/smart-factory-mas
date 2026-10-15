"""
Generate a labeled Remaining-Useful-Life (RUL) dataset from the
factory simulator, across all machines.

How it works
------------
1. Each "episode" runs one machine, in isolation, forward in time using
   the same Machine/DegradationState/sensor model as the live factory,
   until it reaches MachineStatus.FAILED (or hits max_steps).
2. ~85% of episodes get an injected fault, chosen from --failure-modes
   (random mode within that set, random timing, random severity), to
   produce clear degrade-to-failure trajectories. ~15% of episodes are
   left "natural": only ordinary wear accumulation and the tiny
   spontaneous-failure hazard apply. These teach the model what
   healthy / slowly-aging behavior looks like.
3. Once an episode ends in failure, RUL is labeled backward from the
   failure step, then clipped at --rul-cap.
4. Engineered rolling-window features (mean, std, trend) are computed
   for each sensor BEFORE any downsampling, on the full, contiguous
   per-step data -- computing them after thinning the flat region
   would corrupt the windows with gaps.
5. Episodes that never fail within max_steps are "censored" and are
   dropped rather than mislabeled.
6. Rows stuck at the RUL cap (the long "healthy" stretch) are
   downsampled via --flat-sample-rate to fix the severe class
   imbalance that comes from episodes running thousands of steps but
   being capped much lower.
7. Runs are split into train/test by run_id (whole episodes go to one
   side or the other).

Why --failure-modes matters: RANDOM_SHOCK failures jump most of their
damage (shock_damage) the instant they're injected (see machine.py /
degradation.py), rather than announcing themselves through a gradual
sensor trend the way bearing wear or overheating do. That makes them
close to unpredictable from sensor history by design -- including a
lot of them in training/eval will silently cap how good any RUL model
can look, independent of model quality. Use --failure-modes to leave
random_shock out of injected faults entirely (cleaner signal), or keep
it but rely on the "injected_fault" column already written to the CSV
to isolate/exclude it later at evaluation time instead.

Output columns: run_id, machine_id, step, vibration, temperature,
throughput, error_rate, <sensor>_roll_mean / _roll_std / _trend for
each sensor, health, wear, status, injected_fault, RUL.
("health"/"wear" are for analysis/sanity-checking only -- drop them
before feeding a model, since a real deployment wouldn't have access
to the internal degradation state.)
"""

import argparse
import random
from dataclasses import dataclass

import pandas as pd

from config import DEFAULT_MACHINES, MachineConfig, SimulationConfig
from factory import Factory
from models import FailureMode, MachineStatus

SENSOR_COLUMNS = ["vibration", "temperature", "throughput", "error_rate"]


@dataclass
class EpisodeResult:
    df: pd.DataFrame
    censored: bool  # True if the machine never failed within max_steps


def run_episode(
    run_id: int,
    machine_config: MachineConfig,
    max_steps: int,
    natural_fraction: float,
    allowed_failure_modes: list[FailureMode],
) -> EpisodeResult:
    """Run a single machine, alone, until failure or max_steps."""
    rng = random.Random(run_id)
    factory = Factory(machines=[machine_config], config=SimulationConfig(seed=run_id))
    machine_id = machine_config.machine_id

    is_natural = rng.random() < natural_fraction
    inject_at = None
    failure_mode = None
    severity = None
    if not is_natural:
        inject_at = rng.randint(20, max(21, max_steps // 2))
        failure_mode = rng.choice(allowed_failure_modes)
        severity = rng.uniform(0.2, 0.9)

    rows = []
    failed = False
    for step in range(max_steps):
        if not is_natural and step == inject_at:
            factory.inject_failure(machine_id, failure_mode, severity)

        snapshot = factory.step()
        telemetry = snapshot.machines[0]  # only one machine in this factory

        rows.append(
            {
                "run_id": run_id,
                "machine_id": machine_id,
                "step": step,
                "vibration": telemetry.sensors.vibration,
                "temperature": telemetry.sensors.temperature,
                "throughput": telemetry.sensors.throughput,
                "error_rate": telemetry.sensors.error_rate,
                "health": telemetry.health,
                "wear": telemetry.wear,
                "status": telemetry.status.value,
                "injected_fault": failure_mode.value if failure_mode else "none",
            }
        )

        if telemetry.status == MachineStatus.FAILED:
            failed = True
            break

    df = pd.DataFrame(rows)
    return EpisodeResult(df=df, censored=not failed)


def label_rul(df: pd.DataFrame, rul_cap: int) -> pd.DataFrame:
    """Backward-label RUL within one episode and clip at rul_cap."""
    final_step = df["step"].iloc[-1]
    df = df.copy()
    df["RUL"] = (final_step - df["step"]).clip(upper=rul_cap)
    return df


def add_rolling_features(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add rolling mean / std / trend per sensor. Must be called on the
    full, contiguous per-step rows of ONE episode, before any
    downsampling -- computing these over a gapped/thinned series would
    mix non-adjacent timesteps into a single "rolling" statistic."""
    df = df.copy()
    for col in SENSOR_COLUMNS:
        series = df[col]
        df[f"{col}_roll_mean"] = (
            series.rolling(window=window, min_periods=1).mean()
        )
        df[f"{col}_roll_std"] = (
            series.rolling(window=window, min_periods=1).std().fillna(0.0)
        )
        # Simple trend: change over the last `window` steps, per step.
        # Near the start of an episode (fewer than `window` prior rows)
        # this falls back to change-since-start, so it's always defined.
        shifted = series.shift(window)
        shifted = shifted.fillna(series.iloc[0])
        df[f"{col}_trend"] = (series - shifted) / window
    return df


def downsample_flat_region(
    df: pd.DataFrame, rul_cap: int, flat_sample_rate: float, seed: int
) -> pd.DataFrame:
    """Thin out rows stuck at the RUL cap (the long, mostly-redundant
    'healthy' portion of a run), while keeping every row once RUL
    actually starts counting down."""
    flat = df[df["RUL"] == rul_cap]
    declining = df[df["RUL"] < rul_cap]
    if flat_sample_rate >= 1.0 or flat.empty:
        return df
    flat_sampled = flat.sample(frac=flat_sample_rate, random_state=seed)
    return pd.concat([flat_sampled, declining], ignore_index=True).sort_values(
        ["run_id", "step"]
    ).reset_index(drop=True)


def generate_episodes(
    machines: list[MachineConfig],
    runs_per_machine: int,
    max_steps: int,
    natural_fraction: float,
    rul_cap: int,
    flat_sample_rate: float,
    rolling_window: int,
    allowed_failure_modes: list[FailureMode],
    seed_offset: int,
) -> tuple[list[pd.DataFrame], int]:
    """Generate labeled episodes for every machine. Returns (episodes, n_censored)."""
    episodes = []
    n_censored = 0
    run_id = seed_offset

    for machine_config in machines:
        for _ in range(runs_per_machine):
            result = run_episode(
                run_id=run_id,
                machine_config=machine_config,
                max_steps=max_steps,
                natural_fraction=natural_fraction,
                allowed_failure_modes=allowed_failure_modes,
            )

            if result.censored:
                n_censored += 1
                run_id += 1
                continue  # true RUL unknown -> drop rather than mislabel

            labeled = label_rul(result.df, rul_cap=rul_cap)
            # Rolling features computed on the full, contiguous episode
            # BEFORE downsampling -- order matters, see docstring above.
            featured = add_rolling_features(labeled, window=rolling_window)
            thinned = downsample_flat_region(
                featured, rul_cap=rul_cap, flat_sample_rate=flat_sample_rate, seed=run_id
            )
            episodes.append(thinned)
            run_id += 1

    return episodes, n_censored


def split_train_test(
    episodes: list[pd.DataFrame], test_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split whole episodes (by run_id) into train/test, never splitting
    a single run's timeline across both sides."""
    rng = random.Random(seed)
    shuffled = episodes[:]
    rng.shuffle(shuffled)

    n_test = max(1, round(len(shuffled) * test_fraction))
    test_episodes = shuffled[:n_test]
    train_episodes = shuffled[n_test:]

    train_df = pd.concat(train_episodes, ignore_index=True) if train_episodes else pd.DataFrame()
    test_df = pd.concat(test_episodes, ignore_index=True) if test_episodes else pd.DataFrame()
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate a labeled RUL dataset from the factory simulator."
    )
    parser.add_argument(
        "--runs-per-machine",
        type=int,
        default=150,
        help="Number of episodes to simulate per machine config.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=6000,
        help="Max simulated steps per episode before giving up (censored).",
    )
    parser.add_argument(
        "--natural-fraction",
        type=float,
        default=1,
        help="Fraction of episodes with no injected fault (ordinary wear only).",
    )
    parser.add_argument(
        "--rul-cap",
        type=int,
        default=400,
        help="Clip RUL labels at this value (CMAPSS-style ceiling).",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.2,
        help="Fraction of episodes (whole runs) held out as the test set.",
    )
    parser.add_argument(
        "--flat-sample-rate",
        type=float,
        default=0.03,
        help=(
            "Fraction of rows at the RUL cap (the flat 'healthy' region) to keep. "
            "1.0 keeps everything (no downsampling). Rows where RUL < cap are "
            "always kept in full."
        ),
    )
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=10,
        help=(
            "Window size (in steps) for engineered rolling mean / std / trend "
            "features per sensor. Computed before flat-region downsampling."
        ),
    )
    parser.add_argument(
        "--failure-modes",
        type=str,
        nargs="+",
        default=["bearing_wear", "overheating", "misalignment", "random_shock"],
        choices=[mode.value for mode in FailureMode],
        help=(
            "Which failure modes can be injected. Drop random_shock from this "
            "list (e.g. --failure-modes bearing_wear overheating misalignment) "
            "to exclude it entirely, since its damage jumps instantly rather "
            "than announcing itself through a gradual sensor trend -- including "
            "it caps how good any RUL model can look regardless of features. "
            "It's still safe to isolate it later instead via the 'injected_fault' "
            "column, if you'd rather keep it in the data and just exclude it "
            "from training/eval splits."
        ),
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed / run_id offset.")
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="factory_rul",
        help="Output files are written as <prefix>_train.csv and <prefix>_test.csv.",
    )
    args = parser.parse_args()

    if not 0 <= args.natural_fraction <= 1:
        parser.error("--natural-fraction must be in [0, 1]")
    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be in (0, 1)")
    if args.rolling_window < 1:
        parser.error("--rolling-window must be >= 1")

    allowed_failure_modes = [FailureMode(m) for m in args.failure_modes]

    episodes, n_censored = generate_episodes(
        machines=DEFAULT_MACHINES,
        runs_per_machine=args.runs_per_machine,
        max_steps=args.max_steps,
        natural_fraction=args.natural_fraction,
        rul_cap=args.rul_cap,
        flat_sample_rate=args.flat_sample_rate,
        rolling_window=args.rolling_window,
        allowed_failure_modes=allowed_failure_modes,
        seed_offset=args.seed,
    )

    total_requested = args.runs_per_machine * len(DEFAULT_MACHINES)
    print(f"Requested episodes  : {total_requested}")
    print(f"Censored (dropped)  : {n_censored}")
    print(f"Usable episodes     : {len(episodes)}")
    print(f"Failure modes used  : {[m.value for m in allowed_failure_modes]}")

    if not episodes:
        raise SystemExit(
            "No usable (failed) episodes were generated -- try increasing "
            "--max-steps or lowering --natural-fraction."
        )

    train_df, test_df = split_train_test(
        episodes, test_fraction=args.test_fraction, seed=args.seed
    )

    train_path = f"{args.out_prefix}_train.csv"
    test_path = f"{args.out_prefix}_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nTrain episodes : {train_df['run_id'].nunique()}  rows: {len(train_df)}")
    print(f"Test episodes  : {test_df['run_id'].nunique()}  rows: {len(test_df)}")
    print(
        f"Rows at RUL cap ({args.rul_cap}), train: "
        f"{(train_df['RUL'] == args.rul_cap).mean():.1%}"
    )
    print("\nFault mix (train), by injected_fault:")
    print(train_df.drop_duplicates("run_id")["injected_fault"].value_counts())
    print("\nEpisode length summary (steps to failure), train set:")
    print(train_df.groupby("run_id")["step"].max().describe())
    print(f"\nWrote {train_path} and {test_path}")
    print(
        f"Columns: {list(train_df.columns)}"
    )


if __name__ == "__main__":
    main()