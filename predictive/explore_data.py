
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from dataset import add_train_rul, load_cmapss


OUTPUT_DIR = Path(__file__).resolve().parent / "reports" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    train, test, test_rul = load_cmapss()
    train = add_train_rul(train)

    sensor_cols = [f"sensor_{i}" for i in range(1, 22)]

    print("\n=== Dataset overview ===")
    print(f"Training rows: {len(train):,}")
    print(f"Training engines: {train['unit_number'].nunique()}")
    print(f"Sensor columns: {len(sensor_cols)}")
    print(f"RUL range: {train['RUL'].min()} to {train['RUL'].max()}")

    # 1. Sensor variability
    stats = train[sensor_cols].agg(["mean", "std", "min", "max"]).T
    stats["unique_values"] = train[sensor_cols].nunique()
    stats["variance"] = train[sensor_cols].var()

    stats = stats.sort_values("variance")
    print("\n=== Sensors with the lowest variance ===")
    print(stats.head(10).round(4))

    constant_sensors = stats.index[stats["unique_values"] <= 1].tolist()
    print("\nConstant sensors:", constant_sensors)

    # Save the complete summary.
    stats.to_csv(OUTPUT_DIR / "sensor_statistics.csv")

    # 2. Correlation with RUL.
    # This is a diagnostic, not proof of causation or usefulness.
    correlations = (
        train[sensor_cols]
        .corrwith(train["RUL"])
        .rename("correlation_with_RUL")
        .sort_values()
    )

    print("\n=== Sensor correlation with RUL ===")
    print(correlations.round(3))

    correlations.to_csv(OUTPUT_DIR / "sensor_rul_correlations.csv")

    plt.figure(figsize=(10, 6))
    correlations.plot(kind="bar")
    plt.axhline(0, linewidth=1)
    plt.title("Sensor correlation with training RUL")
    plt.xlabel("Sensor")
    plt.ylabel("Pearson correlation")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sensor_rul_correlations.png", dpi=150)
    plt.close()

    # 3. Plot several sensors for one engine.
    # We choose common candidate sensors for inspection.
    selected_sensors = [
        "sensor_2",
        "sensor_3",
        "sensor_4",
        "sensor_7",
        "sensor_11",
        "sensor_12",
        "sensor_15",
        "sensor_21",
    ]

    engine_id = 1
    engine = train[train["unit_number"] == engine_id].sort_values(
        "time_in_cycles"
    )

    fig, axes = plt.subplots(4, 2, figsize=(13, 12), sharex=True)
    for ax, sensor in zip(axes.flat, selected_sensors):
        ax.plot(engine["time_in_cycles"], engine[sensor])
        ax.set_title(sensor)
        ax.set_ylabel("Reading")
        ax.grid(True, alpha=0.3)

    axes[-1, 0].set_xlabel("Operating cycle")
    axes[-1, 1].set_xlabel("Operating cycle")
    fig.suptitle(f"Sensor trajectories for training engine {engine_id}")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "engine_1_sensor_trends.png", dpi=150)
    plt.close(fig)

    # 4. Compare early-life and late-life observations.
    # This is a simple exploratory comparison, not a model split.
    train = train.copy()
    train["life_stage"] = pd.cut(
        train["RUL"],
        bins=[-1, 30, 75, float("inf")],
        labels=["Late life (0-30)", "Mid life (31-75)", "Early life (>75)"],
    )

    stage_means = train.groupby(
        "life_stage", observed=False
    )[sensor_cols].mean().T

    print("\n=== Mean sensor readings by life stage ===")
    print(stage_means.round(3).to_string())

    stage_means.to_csv(OUTPUT_DIR / "sensor_means_by_life_stage.csv")

    stage_means.plot(kind="bar", figsize=(14, 6))
    plt.title("Average sensor readings by RUL stage")
    plt.xlabel("Sensor")
    plt.ylabel("Mean reading")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sensor_means_by_life_stage.png", dpi=150)
    plt.close()

    print(f"\nPlots and reports saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()