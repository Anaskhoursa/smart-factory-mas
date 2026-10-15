import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

SEED = 42
WINDOW_SIZE = 30

# RUL is already capped inside the CSVs by generate_dataset.py's --rul-cap.
# This just needs to match (or be <=) whatever value was used there, so
# np.minimum below is a no-op safety net rather than a second real cap.
RUL_CAP = 400

BATCH_SIZE = 128
EPOCHS = 60
PATIENCE = 8
LEARNING_RATE = 1e-3

HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2
VALIDATION_SIZE = 0.2

# Raw sensors + the engineered rolling features generate_dataset.py now
# writes for each one (mean / std / trend over its --rolling-window).
# Deliberately NOT including health/wear/status: a real deployment
# wouldn't have access to the internal degradation state.
RAW_SENSORS = [
    "vibration",
    "temperature",
    "throughput",
    "error_rate",
]
ROLLING_FEATURES = (
    [f"{c}_roll_mean" for c in RAW_SENSORS]
    + [f"{c}_roll_std" for c in RAW_SENSORS]
    + [f"{c}_trend" for c in RAW_SENSORS]
)
FEATURES = RAW_SENSORS + ROLLING_FEATURES  # 12 columns, all scaled

MACHINE_IDS = ["M1", "M2", "M3", "M4"]  # fixed, known order
ALL_FEATURES = FEATURES + [f"is_{m}" for m in MACHINE_IDS]  # 16 columns total

GROUP_COL = "run_id"
TIME_COL = "step"
TARGET_COL = "RUL"
FAULT_COL = "injected_fault"

BASE_DIR = Path(__file__).resolve().parent
TRAIN_CSV = BASE_DIR / "factory_rul_train.csv"
TEST_CSV = BASE_DIR / "factory_rul_test.csv"
OUTPUT_DIR = BASE_DIR / "artifacts" / "lstm_factory"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# 2. Build sequence windows
# --------------------------------------------------
#
# IMPORTANT: generate_dataset.py downsamples rows stuck at the RUL cap
# (the long "healthy" portion of a run) to fix a severe class
# imbalance. That means consecutive rows for a given run_id are NOT
# guaranteed to be consecutive timesteps -- there can be gaps where
# flat rows were dropped. Every candidate window's `step` values are
# checked for strict +1 contiguity; windows spanning a gap are skipped.
# The rolling-feature columns themselves were computed upstream (in
# generate_dataset.py) on the full, pre-downsampling data, so they're
# correct even though the raw step sequence here has gaps.

def make_windows(
    frame,
    group_ids,
    scaler,
    window_size=WINDOW_SIZE,
):
    """
    Create sequences independently for each run_id, combining scaled
    continuous features (raw sensors + rolling stats) with unscaled
    machine one-hot flags. Also returns, per window, the run's
    injected_fault label (constant within a run), for later breakdown.
    """
    sequences = []
    targets = []
    run_labels = []
    fault_labels = []
    skipped_gaps = 0

    selected = frame[frame[GROUP_COL].isin(group_ids)]
    machine_flag_cols = [f"is_{m}" for m in MACHINE_IDS]

    for run_id, run in selected.groupby(GROUP_COL):
        run = run.sort_values(TIME_COL)

        scaled = scaler.transform(run[FEATURES].to_numpy(dtype=np.float32))
        machine_flags = run[machine_flag_cols].to_numpy(dtype=np.float32)
        values = np.concatenate([scaled, machine_flags], axis=1)

        steps = run[TIME_COL].to_numpy()
        rul_values = run[TARGET_COL].to_numpy(dtype=np.float32)
        fault_label = run[FAULT_COL].iloc[0] if FAULT_COL in run.columns else "unknown"

        n = len(run)

        if n < window_size:
            padding_size = window_size - n
            padding = np.repeat(values[:1], repeats=padding_size, axis=0)
            values_padded = np.concatenate([padding, values], axis=0)

            sequences.append(values_padded)
            targets.append(rul_values[-1])
            run_labels.append(run_id)
            fault_labels.append(fault_label)
            continue

        for end in range(window_size - 1, n):
            start = end - window_size + 1

            step_slice = steps[start:end + 1]
            if not np.all(np.diff(step_slice) == 1):
                skipped_gaps += 1
                continue

            sequences.append(values[start:end + 1])
            targets.append(rul_values[end])
            run_labels.append(run_id)
            fault_labels.append(fault_label)

    X = np.asarray(sequences, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    groups = np.asarray(run_labels)
    faults = np.asarray(fault_labels)

    return X, y, groups, faults, skipped_gaps


def make_final_windows(
    frame,
    scaler,
    window_size=WINDOW_SIZE,
):
    """Build one final observed sequence per run_id, using both
    scaled features and machine flags."""
    sequences = []
    targets = []
    run_ids = []
    machine_flag_cols = [f"is_{m}" for m in MACHINE_IDS]

    for run_id, run in frame.groupby(GROUP_COL):
        run = run.sort_values(TIME_COL)

        scaled = scaler.transform(run[FEATURES].to_numpy(dtype=np.float32))
        machine_flags = run[machine_flag_cols].to_numpy(dtype=np.float32)
        values_all = np.concatenate([scaled, machine_flags], axis=1)

        steps = run[TIME_COL].to_numpy()

        if len(run) < window_size:
            padding_size = window_size - len(run)
            padding = np.repeat(values_all[:1], repeats=padding_size, axis=0)
            values = np.concatenate([padding, values_all], axis=0)
        else:
            tail = values_all[-window_size:]
            tail_steps = steps[-window_size:]
            if np.all(np.diff(tail_steps) == 1):
                values = tail
            else:
                diffs = np.diff(steps)
                break_points = np.where(diffs != 1)[0]
                last_break = break_points[-1] + 1 if len(break_points) else 0
                contiguous = values_all[last_break:]
                if len(contiguous) >= window_size:
                    values = contiguous[-window_size:]
                else:
                    padding = np.repeat(
                        contiguous[:1],
                        repeats=window_size - len(contiguous),
                        axis=0,
                    )
                    values = np.concatenate([padding, contiguous], axis=0)

        sequences.append(values)
        targets.append(float(run[TARGET_COL].iloc[-1]))
        run_ids.append(run_id)

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(run_ids),
    )


# --------------------------------------------------
# 3. Define the LSTM
# --------------------------------------------------

class RULLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        sequence_output, _ = self.lstm(x)
        final_output = sequence_output[:, -1, :]
        prediction = self.regression_head(final_output)
        return prediction.squeeze(-1)


# --------------------------------------------------
# 4. Metrics and prediction helpers
# --------------------------------------------------

def calculate_metrics(actual, predicted):
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(set(actual.tolist())) > 1 else None,
        "n": int(len(actual)),
    }


def breakdown_by_fault(actual, predicted, fault_labels):
    """Per-failure-mode error breakdown, so e.g. random_shock windows
    (which have little to no leading sensor signal by design) don't
    silently drag down the headline number without being visible."""
    breakdown = {}
    for fault in sorted(set(fault_labels.tolist())):
        mask = fault_labels == fault
        if mask.sum() == 0:
            continue
        breakdown[fault] = calculate_metrics(actual[mask], predicted[mask])
    return breakdown


def breakdown_by_rul_range(actual, predicted, rul_cap):
    """Per-RUL-range error breakdown. Far-from-failure windows have a
    much harder (near-flat) signal to predict precisely than
    close-to-failure windows, so this splits the score accordingly."""
    bins = [
        (0, 50, "0-50 (imminent)"),
        (50, 150, "50-150 (approaching)"),
        (150, 300, "150-300 (early decline)"),
        (300, rul_cap + 1, "300+ (far from failure)"),
    ]
    breakdown = {}
    for low, high, label in bins:
        mask = (actual >= low) & (actual < high)
        if mask.sum() == 0:
            continue
        breakdown[label] = calculate_metrics(actual[mask], predicted[mask])
    return breakdown


def predict(model, X, device):
    model.eval()
    predictions = []

    loader = DataLoader(
        TensorDataset(torch.from_numpy(X)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            output = model(batch)
            predictions.extend(output.cpu().numpy())

    return np.asarray(predictions, dtype=np.float32)


# --------------------------------------------------
# 5. Main training pipeline
# --------------------------------------------------

def main():
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading train CSV: {TRAIN_CSV}")
    print(f"Loading test CSV:  {TEST_CSV}")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    required_cols = FEATURES + [TARGET_COL, GROUP_COL, TIME_COL, "machine_id"]
    missing_train = [c for c in required_cols if c not in train_df.columns]
    if missing_train:
        raise ValueError(
            f"Train CSV is missing expected columns: {missing_train}. "
            "Make sure it was generated with the v2 generate_dataset.py "
            "that writes the rolling-feature columns."
        )

    # Build machine one-hot flags once, up front.
    for m in MACHINE_IDS:
        train_df[f"is_{m}"] = (train_df["machine_id"] == m).astype(float)
        test_df[f"is_{m}"] = (test_df["machine_id"] == m).astype(float)

    # Split by run_id, not by individual rows -- otherwise near-identical
    # neighboring rows from the same run leak between train and val.
    run_ids = train_df[GROUP_COL].unique()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
    )

    train_idx, val_idx = next(splitter.split(run_ids, groups=run_ids))
    train_run_ids = run_ids[train_idx]
    val_run_ids = run_ids[val_idx]

    print(f"Training runs:   {len(train_run_ids)}")
    print(f"Validation runs: {len(val_run_ids)}")

    # Fit normalization (raw sensors + rolling features) on training runs only.
    training_rows = train_df[train_df[GROUP_COL].isin(train_run_ids)]
    scaler = StandardScaler()
    scaler.fit(training_rows[FEATURES])

    X_train, y_train_raw, _, _, skipped_train = make_windows(
        train_df, train_run_ids, scaler,
    )
    X_val, y_val_raw, val_groups, _, skipped_val = make_windows(
        train_df, val_run_ids, scaler,
    )

    print(f"Train windows: {len(y_train_raw)} (skipped {skipped_train} for step gaps)")
    print(f"Val windows:   {len(y_val_raw)} (skipped {skipped_val} for step gaps)")

    if len(y_train_raw) == 0 or len(y_val_raw) == 0:
        raise RuntimeError(
            "No usable windows were built -- check WINDOW_SIZE against "
            "your episode lengths, or regenerate the dataset with a "
            "higher --flat-sample-rate so fewer rows are sparse."
        )

    y_train = np.minimum(y_train_raw, RUL_CAP)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_train),
            torch.from_numpy(y_train.astype(np.float32)),
        ),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = RULLSTM(input_size=len(ALL_FEATURES)).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    loss_function = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None

    # --------------------------------------------------
    # Training loop with early stopping
    # --------------------------------------------------

    for epoch in range(1, EPOCHS + 1):
        model.train()
        training_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            predictions = model(X_batch)
            loss = loss_function(predictions, y_batch)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            training_losses.append(loss.item())

        val_predictions = predict(model, X_val, device)
        y_val_capped = np.minimum(y_val_raw, RUL_CAP)

        val_loss = float(np.mean((val_predictions - y_val_capped) ** 2))
        train_loss = float(np.mean(training_losses))

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train MSE: {train_loss:.2f} | "
            f"val MSE: {val_loss:.2f}", flush=True
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping: validation loss stopped improving.")
            break

    model.load_state_dict(best_state)
    model.to(device)

    print(f"\nBest epoch: {best_epoch}")

    # --------------------------------------------------
    # 6. Evaluate held-out validation runs
    # --------------------------------------------------

    val_predictions = predict(model, X_val, device)

    validation_metrics = {
        "capped_target_metrics": calculate_metrics(
            np.minimum(y_val_raw, RUL_CAP), val_predictions,
        ),
        "raw_target_metrics": calculate_metrics(
            y_val_raw, val_predictions,
        ),
    }

    print("\nValidation metrics:")
    print(json.dumps(validation_metrics, indent=2))

    # --------------------------------------------------
    # 7. Evaluate the held-out test set
    # --------------------------------------------------
    # Sliding-window view across whole trajectories (closer to live
    # deployment), a single final-window-per-run view (CMAPSS-style,
    # degenerate here since every run ends at RUL=0 -- kept only for
    # continuity), plus breakdowns by failure mode and RUL range.

    X_test_full, y_test_full_raw, _, test_faults, skipped_test = make_windows(
        test_df, test_df[GROUP_COL].unique(), scaler,
    )
    test_predictions_full = predict(model, X_test_full, device)

    X_test_final, y_test_final_raw, test_run_ids = make_final_windows(
        test_df, scaler,
    )
    test_predictions_final = predict(model, X_test_final, device)

    test_metrics = {
        "sliding_window_all_steps": {
            "capped_target_metrics": calculate_metrics(
                np.minimum(y_test_full_raw, RUL_CAP),
                np.minimum(test_predictions_full, RUL_CAP),
            ),
            "raw_target_metrics": calculate_metrics(
                y_test_full_raw, test_predictions_full,
            ),
            "n_windows": int(len(y_test_full_raw)),
            "skipped_for_step_gaps": int(skipped_test),
            "by_failure_mode": breakdown_by_fault(
                np.minimum(y_test_full_raw, RUL_CAP),
                np.minimum(test_predictions_full, RUL_CAP),
                test_faults,
            ),
            "by_rul_range": breakdown_by_rul_range(
                y_test_full_raw, test_predictions_full, RUL_CAP,
            ),
        },
        "final_window_per_run": {
            "note": (
                "Degenerate here (not useful): every run ends at RUL=0 "
                "by construction, so the target has zero variance and "
                "R2 is undefined. Kept only for continuity with the "
                "CMAPSS-style convention -- use sliding_window_all_steps."
            ),
            "capped_target_metrics": calculate_metrics(
                np.minimum(y_test_final_raw, RUL_CAP),
                np.minimum(test_predictions_final, RUL_CAP),
            ),
            "n_runs": int(len(y_test_final_raw)),
        },
    }

    print("\nTest metrics:")
    print(json.dumps(test_metrics, indent=2))

    # Save predictions from the final-window-per-run view (one row per run).
    prediction_df = pd.DataFrame({
        "run_id": test_run_ids,
        "actual_rul": y_test_final_raw,
        "predicted_rul": test_predictions_final,
        "predicted_rul_capped": np.minimum(test_predictions_final, RUL_CAP),
    })
    prediction_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "features": ALL_FEATURES,
            "window_size": WINDOW_SIZE,
            "rul_cap": RUL_CAP,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "best_epoch": best_epoch,
        },
        OUTPUT_DIR / "best_lstm.pt",
    )

    results = {
        "window_size": WINDOW_SIZE,
        "features": ALL_FEATURES,
        "rul_cap": RUL_CAP,
        "best_epoch": best_epoch,
        "validation": validation_metrics,
        "test": test_metrics,
    }

    with open(OUTPUT_DIR / "metrics.json", "w") as file:
        json.dump(results, file, indent=2)

    print(f"\nSaved artifacts to: {OUTPUT_DIR}")
    print(" - best_lstm.pt")
    print(" - metrics.json")
    print(" - test_predictions.csv")


if __name__ == "__main__":
    main()