
from pathlib import Path

import numpy as np
import pandas as pd


# Change this path if your dataset is stored elsewhere.
DATA_DIR = (
    Path(__file__).resolve().parent
    / "data"
    / "cmapss"
)

DATASET_ID = "FD001"

# The raw files have no header row.
COLUMN_NAMES = [
    "unit_number",
    "time_in_cycles",
    "setting_1",
    "setting_2",
    "setting_3",
] + [f"sensor_{i}" for i in range(1, 22)]


def load_cmapss(dataset_id=DATASET_ID, data_dir=DATA_DIR):
    """Load the raw training, test, and test-RUL files."""
    data_dir = Path(data_dir)

    train_path = data_dir / f"train_{dataset_id}.txt"
    test_path = data_dir / f"test_{dataset_id}.txt"
    rul_path = data_dir / f"RUL_{dataset_id}.txt"

    # Fail early with a useful message if a file is missing.
    for path in (train_path, test_path, rul_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Cannot find {path}\n"
                "Check DATA_DIR and confirm the dataset was extracted."
            )

    # The files are whitespace-separated, not comma-separated.
    train = pd.read_csv(
        train_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
    )

    test = pd.read_csv(
        test_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
    )

    test_rul = pd.read_csv(
        rul_path,
        sep=r"\s+",
        header=None,
        names=["RUL"],
    )

    return train, test, test_rul


def add_train_rul(train):
    """Calculate RUL for every training row."""
    train = train.copy()

    last_cycle = train.groupby("unit_number")[
        "time_in_cycles"
    ].transform("max")

    train["RUL"] = last_cycle - train["time_in_cycles"]

    return train


def add_test_rul(test, test_rul):
    """
    Calculate RUL for each test row.

    RUL_FD001 contains the true RUL at the FINAL observed
    cycle of each test engine, not at every earlier cycle.
    """
    test = test.copy()

    # There should be one RUL value per test engine, in
    # ascending engine-ID order.
    test_rul = test_rul.copy()

    if test["unit_number"].nunique() != len(test_rul):
        raise ValueError(
            "Number of test engines does not match "
            "number of values in the RUL file."
        )

    final_cycles = (
        test.groupby("unit_number")["time_in_cycles"]
        .max()
        .sort_index()
    )

    # Associate each engine with its final observed cycle
    # and the supplied RUL at that cycle.
    engine_targets = pd.DataFrame({
        "unit_number": final_cycles.index,
        "final_observed_cycle": final_cycles.values,
        "RUL_at_final_observation": test_rul["RUL"].to_numpy(),
    })

    test = test.merge(engine_targets, on="unit_number")

    # Earlier observations have more remaining cycles than
    # the final observation of the same engine.
    test["RUL"] = (
        test["RUL_at_final_observation"]
        + test["final_observed_cycle"]
        - test["time_in_cycles"]
    )

    return test.drop(
        columns=[
            "final_observed_cycle",
            "RUL_at_final_observation",
        ]
    )


def inspect_data(train, test, test_rul):
    """Print basic checks and useful dataset summaries."""
    print("\n--- Dataset shapes ---")
    print(f"Train:    {train.shape}")
    print(f"Test:     {test.shape}")
    print(f"Test RUL: {test_rul.shape}")

    print("\n--- First training rows ---")
    print(train.head())

    print("\n--- Data types ---")
    print(train.dtypes)

    print("\n--- Missing values ---")
    print("Train:", int(train.isna().sum().sum()))
    print("Test: ", int(test.isna().sum().sum()))

    print("\n--- Engine counts ---")
    print("Train engines:", train["unit_number"].nunique())
    print("Test engines: ", test["unit_number"].nunique())

    train_lengths = train.groupby("unit_number")[
        "time_in_cycles"
    ].max()

    print("\n--- Training trajectory lengths ---")
    print(train_lengths.describe())

    print("\n--- Test RUL summary ---")
    print(test_rul["RUL"].describe())


if __name__ == "__main__":
    train, test, test_rul = load_cmapss()

    train = add_train_rul(train)
    test = add_test_rul(test, test_rul)

    inspect_data(train, test, test_rul)

    print("\n--- Training data with RUL ---")
    print(train[
        ["unit_number", "time_in_cycles", "RUL"]
    ].head(10))

    print("\n--- Test data with RUL ---")
    print(test[
        ["unit_number", "time_in_cycles", "RUL"]
    ].head(10))