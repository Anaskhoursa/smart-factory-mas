"""
Train an LSTM-autoencoder for anomaly detection on the factory
simulator's sensor data, plus an Isolation Forest baseline on the
engineered rolling features for comparison.

Data and labels
----------------
Reuses factory_rul_train.csv / factory_rul_test.csv (the same files
produced for the RUL model -- no new simulation needed).

- TRAINING is unsupervised and uses ONLY raw sensors (vibration,
  temperature, throughput, error_rate) from windows that are entirely
  "healthy" (health > NORMAL_HEALTH_THRESHOLD for every row in the
  window). The model never sees health / wear / injected_fault.
- EVALUATION uses health / injected_fault only as a post-hoc answer
  key: a window is labeled a true anomaly if its final row's health <
  ANOMALY_HEALTH_THRESHOLD (stricter than the training threshold, so
  there's a buffer zone and no ambiguous windows are used to grade
  the model).

Two models are trained and compared:
1. LSTM-autoencoder on raw sensor windows -- the main model, since
   anomalies here are temporal (a rising trend), not single bad
   readings, and a recurrent model can see the trend shape directly.
2. Isolation Forest on the engineered rolling features (mean/std/
   trend per sensor), as a cheap classical baseline for comparison.
"""

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
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

SEED = 42
WINDOW_SIZE = 30

NORMAL_HEALTH_THRESHOLD = 0.90   # windows used for autoencoder training
ANOMALY_HEALTH_THRESHOLD = 0.80  # windows used as the "true anomaly" eval label
# Rows with health strictly between ANOMALY_HEALTH_THRESHOLD and
# NORMAL_HEALTH_THRESHOLD are a deliberate buffer zone: never used for
# training, and not counted as a confident true-anomaly label either
# (they're excluded from eval precision/recall, see build_eval_set).

BATCH_SIZE = 128
EPOCHS = 60
PATIENCE = 8
LEARNING_RATE = 1e-3

HIDDEN_SIZE = 32   # smaller than the RUL LSTM: this is a bottleneck by design
NUM_LAYERS = 1
DROPOUT = 0.0
VALIDATION_SIZE = 0.2

THRESHOLD_PERCENTILE = 99  # anomaly cutoff = this percentile of normal-val reconstruction error

RAW_SENSORS = ["vibration", "temperature", "throughput", "error_rate"]
ROLLING_FEATURES = (
    [f"{c}_roll_mean" for c in RAW_SENSORS]
    + [f"{c}_roll_std" for c in RAW_SENSORS]
    + [f"{c}_trend" for c in RAW_SENSORS]
)

GROUP_COL = "run_id"
TIME_COL = "step"
HEALTH_COL = "health"
FAULT_COL = "injected_fault"

BASE_DIR = Path(__file__).resolve().parent
TRAIN_CSV = BASE_DIR / "factory_rul_train.csv"
TEST_CSV = BASE_DIR / "factory_rul_test.csv"
OUTPUT_DIR = BASE_DIR / "artifacts" / "anomaly_lstm"
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
# Same step-contiguity guard as the RUL script: generate_dataset.py
# downsamples the long "healthy" region, so consecutive rows for a
# run_id aren't guaranteed to be consecutive timesteps. Any window
# spanning a gap is skipped rather than silently stitched together.

def make_windows(frame, group_ids, feature_cols, window_size=WINDOW_SIZE):
    """Build every valid (gap-free) window per run_id. Returns the raw
    (unscaled) feature sequences plus, per window, the final row's
    health, injected_fault and rolling-feature snapshot (for the
    Isolation Forest baseline and for eval labeling)."""
    sequences = []
    run_labels = []
    final_health = []
    final_fault = []
    final_rolling = []
    skipped_gaps = 0

    selected = frame[frame[GROUP_COL].isin(group_ids)]

    for run_id, run in selected.groupby(GROUP_COL):
        run = run.sort_values(TIME_COL)

        values = run[feature_cols].to_numpy(dtype=np.float32)
        rolling_values = run[ROLLING_FEATURES].to_numpy(dtype=np.float32)
        steps = run[TIME_COL].to_numpy()
        health = run[HEALTH_COL].to_numpy(dtype=np.float32)
        fault = run[FAULT_COL].to_numpy() if FAULT_COL in run.columns else None

        n = len(run)
        if n < window_size:
            continue  # too short to form a real window; skip (unlike RUL, no padding needed here)

        for end in range(window_size - 1, n):
            start = end - window_size + 1
            step_slice = steps[start:end + 1]
            if not np.all(np.diff(step_slice) == 1):
                skipped_gaps += 1
                continue

            sequences.append(values[start:end + 1])
            run_labels.append(run_id)
            final_health.append(health[end])
            final_fault.append(fault[end] if fault is not None else "unknown")
            final_rolling.append(rolling_values[end])

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(run_labels),
        np.asarray(final_health, dtype=np.float32),
        np.asarray(final_fault),
        np.asarray(final_rolling, dtype=np.float32),
        skipped_gaps,
    )


def select_fully_normal_windows(sequences, run_labels, final_health, raw_df, group_ids, window_size):
    """Keep only windows where EVERY row (not just the last) has
    health > NORMAL_HEALTH_THRESHOLD -- training should never see a
    window that contains the start of a degradation trend."""
    # Recompute per-window health arrays by re-walking the same windows.
    mask = []
    selected = raw_df[raw_df[GROUP_COL].isin(group_ids)]
    per_run_health = {
        run_id: grp.sort_values(TIME_COL)[HEALTH_COL].to_numpy(dtype=np.float32)
        for run_id, grp in selected.groupby(GROUP_COL)
    }
    per_run_steps = {
        run_id: grp.sort_values(TIME_COL)[TIME_COL].to_numpy()
        for run_id, grp in selected.groupby(GROUP_COL)
    }

    keep = np.zeros(len(sequences), dtype=bool)
    idx = 0
    for run_id, grp in selected.groupby(GROUP_COL):
        grp = grp.sort_values(TIME_COL)
        health = grp[HEALTH_COL].to_numpy(dtype=np.float32)
        steps = grp[TIME_COL].to_numpy()
        n = len(grp)
        if n < window_size:
            continue
        for end in range(window_size - 1, n):
            start = end - window_size + 1
            step_slice = steps[start:end + 1]
            if not np.all(np.diff(step_slice) == 1):
                continue
            window_health = health[start:end + 1]
            keep[idx] = np.all(window_health > NORMAL_HEALTH_THRESHOLD)
            idx += 1

    assert idx == len(sequences), (
        "Window count mismatch between make_windows and the normal-window "
        "filter -- they must walk the data in exactly the same order."
    )
    return sequences[keep], run_labels[keep]


# --------------------------------------------------
# 3. Define the LSTM-autoencoder
# --------------------------------------------------

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, window_size=WINDOW_SIZE):
        super().__init__()
        self.window_size = window_size
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, n_features)

    def forward(self, x):
        # Encode the whole window down to its final hidden state (the bottleneck).
        _, (hidden, _) = self.encoder(x)
        bottleneck = hidden[-1]  # (batch, hidden_size)

        # Repeat the bottleneck across time as the decoder's input at every step.
        repeated = bottleneck.unsqueeze(1).repeat(1, self.window_size, 1)

        decoded_seq, _ = self.decoder(repeated)
        reconstruction = self.output_layer(decoded_seq)
        return reconstruction


def reconstruction_errors(model, X, device, batch_size=BATCH_SIZE):
    """Per-window mean squared reconstruction error."""
    model.eval()
    errors = []
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            recon = model(batch)
            err = torch.mean((recon - batch) ** 2, dim=(1, 2))
            errors.extend(err.cpu().numpy())
    return np.asarray(errors, dtype=np.float32)


# --------------------------------------------------
# 4. Evaluation helpers
# --------------------------------------------------

def build_eval_labels(final_health):
    """True-anomaly label per window, with a buffer zone excluded.
    Returns (labels, keep_mask): labels are 1 (anomaly) / 0 (normal)
    for kept windows only."""
    is_anomaly = final_health < ANOMALY_HEALTH_THRESHOLD
    is_normal = final_health > NORMAL_HEALTH_THRESHOLD
    keep = is_anomaly | is_normal  # drop the ambiguous buffer zone
    labels = is_anomaly[keep].astype(int)
    return labels, keep


def score_at_threshold(scores, labels, threshold):
    preds = (scores > threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_flagged": int(preds.sum()),
        "n_total": int(len(labels)),
    }


def breakdown_by_fault(scores, labels, threshold, fault_labels):
    breakdown = {}
    for fault in sorted(set(fault_labels.tolist())):
        mask = fault_labels == fault
        if mask.sum() == 0 or labels[mask].sum() == 0:
            continue  # skip groups with no true anomalies to score against
        breakdown[fault] = score_at_threshold(scores[mask], labels[mask], threshold)
    return breakdown


# --------------------------------------------------
# 5. Main pipeline
# --------------------------------------------------

def main():
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading train CSV: {TRAIN_CSV}")
    print(f"Loading test CSV:  {TEST_CSV}")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    required = RAW_SENSORS + ROLLING_FEATURES + [GROUP_COL, TIME_COL, HEALTH_COL]
    missing = [c for c in required if c not in train_df.columns]
    if missing:
        raise ValueError(f"Train CSV is missing expected columns: {missing}")

    # --- split runs: train / val, both drawn from the train CSV ---
    run_ids = train_df[GROUP_COL].unique()
    splitter = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_SIZE, random_state=SEED)
    train_idx, val_idx = next(splitter.split(run_ids, groups=run_ids))
    train_run_ids = run_ids[train_idx]
    val_run_ids = run_ids[val_idx]
    print(f"Training runs:   {len(train_run_ids)}")
    print(f"Validation runs: {len(val_run_ids)}")

    # --- build all windows (raw sensors) for train/val/test ---
    X_train_all, train_groups_all, train_health_all, _, _, skipped_train = make_windows(
        train_df, train_run_ids, RAW_SENSORS,
    )
    X_val_all, val_groups_all, val_health_all, val_fault_all, val_rolling_all, skipped_val = make_windows(
        train_df, val_run_ids, RAW_SENSORS,
    )
    X_test_all, test_groups_all, test_health_all, test_fault_all, test_rolling_all, skipped_test = make_windows(
        test_df, test_df[GROUP_COL].unique(), RAW_SENSORS,
    )

    print(
        f"Windows built -- train(all): {len(X_train_all)} (skipped {skipped_train}), "
        f"val(all): {len(X_val_all)} (skipped {skipped_val}), "
        f"test(all): {len(X_test_all)} (skipped {skipped_test})"
    )

    # --- keep only FULLY normal windows for training ---
    X_train_normal, _ = select_fully_normal_windows(
        X_train_all, train_groups_all, train_health_all, train_df, train_run_ids, WINDOW_SIZE,
    )
    X_val_normal, _ = select_fully_normal_windows(
        X_val_all, val_groups_all, val_health_all, train_df, val_run_ids, WINDOW_SIZE,
    )
    print(f"Fully-normal windows -- train: {len(X_train_normal)}, val: {len(X_val_normal)}")

    if len(X_train_normal) == 0 or len(X_val_normal) == 0:
        raise RuntimeError(
            "No fully-normal windows found -- lower NORMAL_HEALTH_THRESHOLD, "
            "increase --natural-fraction in generate_dataset.py, or raise "
            "--flat-sample-rate so more healthy rows remain."
        )

    # --- scale on normal training windows only ---
    scaler = StandardScaler()
    scaler.fit(X_train_normal.reshape(-1, len(RAW_SENSORS)))

    def scale_windows(X):
        shape = X.shape
        flat = scaler.transform(X.reshape(-1, shape[-1]))
        return flat.reshape(shape).astype(np.float32)

    X_train_normal_s = scale_windows(X_train_normal)
    X_val_normal_s = scale_windows(X_val_normal)
    X_val_all_s = scale_windows(X_val_all)
    X_test_all_s = scale_windows(X_test_all)

    # --------------------------------------------------
    # Train the LSTM-autoencoder (unsupervised, normal windows only)
    # --------------------------------------------------

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train_normal_s)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    model = LSTMAutoencoder(n_features=len(RAW_SENSORS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = None

    X_val_normal_tensor = torch.from_numpy(X_val_normal_s)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_recon = model(X_val_normal_tensor.to(device))
            val_loss = loss_fn(val_recon, X_val_normal_tensor.to(device)).item()

        train_loss = float(np.mean(losses))
        print(f"Epoch {epoch:02d}/{EPOCHS} | train MSE: {train_loss:.5f} | val MSE: {val_loss:.5f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping: validation reconstruction loss stopped improving.")
            break

    model.load_state_dict(best_state)
    model.to(device)
    print(f"\nBest epoch: {best_epoch}")

    # --------------------------------------------------
    # Pick the anomaly threshold from normal validation windows
    # --------------------------------------------------

    normal_val_errors = reconstruction_errors(model, X_val_normal_s, device)
    threshold = float(np.percentile(normal_val_errors, THRESHOLD_PERCENTILE))
    print(f"\nAnomaly threshold (p{THRESHOLD_PERCENTILE} of normal-val reconstruction error): {threshold:.5f}")

    # --------------------------------------------------
    # Evaluate on the held-out test set
    # --------------------------------------------------

    test_labels, test_keep = build_eval_labels(test_health_all)
    test_scores = reconstruction_errors(model, X_test_all_s[test_keep], device)
    test_fault_kept = test_fault_all[test_keep]

    lstm_metrics = score_at_threshold(test_scores, test_labels, threshold)
    try:
        lstm_metrics["roc_auc"] = float(roc_auc_score(test_labels, test_scores))
    except ValueError:
        lstm_metrics["roc_auc"] = None  # only one class present
    lstm_metrics["by_failure_mode"] = breakdown_by_fault(test_scores, test_labels, threshold, test_fault_kept)

    print("\nLSTM-autoencoder test metrics:")
    print(json.dumps(lstm_metrics, indent=2))

    # --------------------------------------------------
    # Isolation Forest baseline on engineered rolling features
    # --------------------------------------------------

    train_rolling_normal = train_df[train_df[GROUP_COL].isin(train_run_ids) & (train_df[HEALTH_COL] > NORMAL_HEALTH_THRESHOLD)][ROLLING_FEATURES].to_numpy(dtype=np.float32)

    iso_scaler = StandardScaler().fit(train_rolling_normal)
    iso_forest = IsolationForest(n_estimators=200, contamination="auto", random_state=SEED)
    iso_forest.fit(iso_scaler.transform(train_rolling_normal))

    test_rolling_kept = test_rolling_all[test_keep]
    iso_scores = -iso_forest.score_samples(iso_scaler.transform(test_rolling_kept))  # higher = more anomalous
    iso_threshold = float(
        np.percentile(
            -iso_forest.score_samples(
                iso_scaler.transform(
                    train_df[
                        train_df[GROUP_COL].isin(val_run_ids)
                        & (train_df[HEALTH_COL] > NORMAL_HEALTH_THRESHOLD)
                    ][ROLLING_FEATURES].to_numpy(dtype=np.float32)
                )
            ),
            THRESHOLD_PERCENTILE,
        )
    )

    iso_metrics = score_at_threshold(iso_scores, test_labels, iso_threshold)
    try:
        iso_metrics["roc_auc"] = float(roc_auc_score(test_labels, iso_scores))
    except ValueError:
        iso_metrics["roc_auc"] = None
    iso_metrics["by_failure_mode"] = breakdown_by_fault(iso_scores, test_labels, iso_threshold, test_fault_kept)

    print("\nIsolation Forest baseline test metrics:")
    print(json.dumps(iso_metrics, indent=2))

    # --------------------------------------------------
    # Save artifacts
    # --------------------------------------------------

    predictions_df = pd.DataFrame({
        "run_id": test_groups_all[test_keep],
        "true_label": test_labels,
        "health": test_health_all[test_keep],
        "injected_fault": test_fault_kept,
        "lstm_reconstruction_error": test_scores,
        "lstm_flagged": (test_scores > threshold).astype(int),
        "iso_forest_score": iso_scores,
        "iso_forest_flagged": (iso_scores > iso_threshold).astype(int),
    })
    predictions_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "features": RAW_SENSORS,
            "window_size": WINDOW_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "threshold": threshold,
            "threshold_percentile": THRESHOLD_PERCENTILE,
            "normal_health_threshold": NORMAL_HEALTH_THRESHOLD,
            "anomaly_health_threshold": ANOMALY_HEALTH_THRESHOLD,
            "best_epoch": best_epoch,
        },
        OUTPUT_DIR / "best_autoencoder.pt",
    )

    results = {
        "window_size": WINDOW_SIZE,
        "normal_health_threshold": NORMAL_HEALTH_THRESHOLD,
        "anomaly_health_threshold": ANOMALY_HEALTH_THRESHOLD,
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "lstm_autoencoder": {**lstm_metrics, "threshold": threshold, "best_epoch": best_epoch},
        "isolation_forest_baseline": {**iso_metrics, "threshold": iso_threshold},
    }
    with open(OUTPUT_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved artifacts to: {OUTPUT_DIR}")
    print(" - best_autoencoder.pt")
    print(" - metrics.json")
    print(" - test_predictions.csv")


if __name__ == "__main__":
    main()