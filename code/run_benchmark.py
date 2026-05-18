"""
MagBridge-Battery v1.0 — Classical-ML benchmark.

Runs the four classical benchmark tasks (T1 SOH regression, T2 second-life
binary classification, T3 anomaly detection 3-class, T4 anomaly subtype
4-class), using the by_cell_primary leakage-safe split. Reports mean ± std
across 5 cell-subsampling seeds.

Data directory:
  By default, looks for the dataset in ./data/ (relative to where the script
  is run). Override with the MAGBRIDGE_DATA environment variable or the
  --data-dir command-line argument.

  Expected layout:
    {data}/data/shard_*.parquet
    {data}/splits/by_cell_primary.json

To get the data, download the release bundle from Zenodo (see README) and
unzip it into the directory you point this script at.
"""
from __future__ import annotations
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression, RidgeClassifier
from sklearn.svm import SVR, LinearSVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--data-dir",
        default=os.environ.get("MAGBRIDGE_DATA", "./data"),
        help="Path to the unzipped Zenodo bundle (default: ./data, or $MAGBRIDGE_DATA)",
    )
    return p.parse_args()


_args = _parse_args()
BASE = Path(_args.data_dir).expanduser().resolve()
DATA = BASE / "data"
SPLITS = BASE / "splits"

if not DATA.exists() or not SPLITS.exists():
    raise SystemExit(
        f"Dataset not found at {BASE}.\n"
        f"Expected layout:\n"
        f"  {DATA}/shard_*.parquet\n"
        f"  {SPLITS}/by_cell_primary.json\n\n"
        f"Download the v1.0 bundle from Zenodo (see README) and unzip it,\n"
        f"or pass --data-dir <path> to this script."
    )

SEEDS = [0, 1, 2, 3, 4]
SIGNAL_CHANNELS = ["B_s1Y", "B_s1Z", "B_s2Y", "B_s2Z", "B_s1C5", "B_s2C6"]


# --- 1. Load data and split ---
print("Loading shards...")
df = pd.concat([pd.read_parquet(s) for s in sorted(DATA.glob("shard_*.parquet"))], ignore_index=True)
print(f"  Loaded {len(df)} rows")

with open(SPLITS / "by_cell_primary.json") as f:
    split = json.load(f)
train_ids = set(split["train_samples"])
val_ids = set(split["val_samples"])
test_ids = set(split["test_samples"])
print(f"  Split: train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")


# --- 2. Feature extraction: 57-feature static descriptor ---
# 9 features per channel × 6 channels = 54, plus 3 cross-channel correlations = 57
def per_channel_feats(x: np.ndarray) -> np.ndarray:
    """9 features for one length-100 signal."""
    early = x[: len(x) // 4].mean()
    late = x[-len(x) // 4 :].mean()
    return np.array(
        [
            x.mean(), x.std(), x.min(), x.max(), x.max() - x.min(),
            np.median(x), np.dot(x, x), (x[-1] - x[0]) / max(1, len(x)),
            late - early,
        ]
    )

def extract_features(row) -> np.ndarray:
    feats = []
    sigs = {c: np.asarray(row[c], dtype=np.float64) for c in SIGNAL_CHANNELS}
    for c in SIGNAL_CHANNELS:
        feats.append(per_channel_feats(sigs[c]))
    # 3 cross-channel correlations
    pairs = [("B_s1Y", "B_s1Z"), ("B_s2Y", "B_s2Z"), ("B_s1Y", "B_s2Y")]
    for a, b in pairs:
        c = np.corrcoef(sigs[a], sigs[b])[0, 1]
        feats.append(np.array([0.0 if np.isnan(c) else c]))
    return np.concatenate(feats)

print("Extracting features (57 per sample)...")
t0 = time.time()
X = np.stack([extract_features(row) for _, row in df.iterrows()])
print(f"  X shape: {X.shape}  ({time.time()-t0:.1f}s)")


# --- 3. Helper: get train/test indices for a row-level mask ---
def subset_indices(row_mask):
    """Given a boolean mask on df, return train/test indices respecting by_cell_primary."""
    sids = df.loc[row_mask, "sample_id"].values
    tr = np.array([i for i, s in zip(df.index[row_mask], sids) if s in train_ids])
    te = np.array([i for i, s in zip(df.index[row_mask], sids) if s in test_ids])
    return tr, te

def seed_subsample(tr_idx, seed, frac=0.8):
    """Bootstrap-style: each seed sees a different 80% of training samples,
    so the seed-to-seed variance reflects real sensitivity to training composition."""
    rng = np.random.default_rng(seed)
    n = int(len(tr_idx) * frac)
    return tr_idx[rng.choice(len(tr_idx), n, replace=False)]


# --- 4. T1: SOH regression on grounded clean samples ---
print("\n=== T1: SOH regression ===")
mask_t1 = (df["anomaly_subtype"] == "none") & (df["regime"] == "grounded") & (df["soh"].notna())
tr_idx, te_idx = subset_indices(mask_t1)
print(f"  train={len(tr_idx)}  test={len(te_idx)}")
y = df["soh"].values

t1_results = {}
for name, model_factory in [
    ("Ridge",  lambda: Pipeline([("sc", StandardScaler()), ("m", Ridge(alpha=1.0))])),
    ("SVR-RBF", lambda: Pipeline([("sc", StandardScaler()), ("m", SVR(kernel="rbf"))])),
    ("RF",     lambda: RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1)),
]:
    seed_scores = []
    for seed in SEEDS:
        # Each seed sees a different 80% of training samples
        sub = seed_subsample(tr_idx, seed)
        model = model_factory()
        if name == "RF":
            model.set_params(random_state=seed)
        model.fit(X[sub], y[sub])
        pred = model.predict(X[te_idx])
        seed_scores.append(r2_score(y[te_idx], pred))
    arr = np.array(seed_scores)
    t1_results[name] = (arr.mean(), arr.std())
    print(f"  {name:8s}: R² = {arr.mean():+.4f} ± {arr.std():.4f}")
best_t1 = max(t1_results.items(), key=lambda kv: kv[1][0])
print(f"  BEST: {best_t1[0]} (R² = {best_t1[1][0]:+.4f} ± {best_t1[1][1]:.4f})")


# --- 5. T2: Second-life classification (binary) on grounded clean samples ---
print("\n=== T2: Second-life classification ===")
mask_t2 = (df["anomaly_subtype"] == "none") & (df["regime"] == "grounded") & (df["second_life_class"].notna())
tr_idx, te_idx = subset_indices(mask_t2)
print(f"  train={len(tr_idx)}  test={len(te_idx)}")
y2 = (df["second_life_class"] == "reuse").astype(int).values

t2_results = {}
for name, model_factory in [
    ("LogReg",     lambda: Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000))])),
    ("RidgeCls",   lambda: Pipeline([("sc", StandardScaler()), ("m", RidgeClassifier())])),
    ("LinSVC",     lambda: Pipeline([("sc", StandardScaler()), ("m", LinearSVC(max_iter=5000, dual="auto"))])),
    ("RF",         lambda: RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)),
]:
    seed_scores = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        sub = seed_subsample(tr_idx, seed)
        model = model_factory()
        if name == "RF":
            model.set_params(random_state=seed)
        model.fit(X[sub], y2[sub])
        pred = model.predict(X[te_idx])
        seed_scores.append(balanced_accuracy_score(y2[te_idx], pred))
    arr = np.array(seed_scores)
    t2_results[name] = (arr.mean(), arr.std())
    print(f"  {name:8s}: bal_acc = {arr.mean():.4f} ± {arr.std():.4f}")
best_t2 = max(t2_results.items(), key=lambda kv: kv[1][0])
print(f"  BEST: {best_t2[0]} (bal_acc = {best_t2[1][0]:.4f} ± {best_t2[1][1]:.4f})")


# --- 6. T3: Anomaly detection (3-class: clean grounded / synthetic anomaly / Regime-B) ---
print("\n=== T3: Anomaly detection (3-class) ===")
# Construct label: 0 = clean grounded, 1 = synthetic anomaly, 2 = Regime-B
def t3_label(row):
    if row["anomaly_subtype"] == "none" and row["regime"] == "grounded":
        return 0
    if row["anomaly_subtype"] == "low_voltage_regime_B":
        return 2
    return 1  # one of the 4 synthetic anomaly subtypes

y3 = df.apply(t3_label, axis=1).values
mask_t3 = np.ones(len(df), dtype=bool)  # all samples participate
tr_idx, te_idx = subset_indices(mask_t3)
print(f"  train={len(tr_idx)}  test={len(te_idx)}")
print(f"  train class balance: {pd.Series(y3[tr_idx]).value_counts().to_dict()}")
print(f"  test class balance:  {pd.Series(y3[te_idx]).value_counts().to_dict()}")

t3_results = {}
for name, model_factory in [
    ("LogReg",   lambda: Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000))])),
    ("RidgeCls", lambda: Pipeline([("sc", StandardScaler()), ("m", RidgeClassifier())])),
    ("LinSVC",   lambda: Pipeline([("sc", StandardScaler()), ("m", LinearSVC(max_iter=5000, dual="auto"))])),
    ("RF",       lambda: RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)),
]:
    seed_scores = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        sub = seed_subsample(tr_idx, seed)
        model = model_factory()
        if name == "RF":
            model.set_params(random_state=seed)
        try:
            model.fit(X[sub], y3[sub])
            pred = model.predict(X[te_idx])
            seed_scores.append(balanced_accuracy_score(y3[te_idx], pred))
        except Exception as e:
            print(f"    [{name} seed {seed}] FAILED: {e}")
            seed_scores.append(np.nan)
    arr = np.array(seed_scores)
    t3_results[name] = (np.nanmean(arr), np.nanstd(arr))
    print(f"  {name:8s}: bal_acc = {np.nanmean(arr):.4f} ± {np.nanstd(arr):.4f}")
best_t3 = max(t3_results.items(), key=lambda kv: kv[1][0])
print(f"  BEST: {best_t3[0]} (bal_acc = {best_t3[1][0]:.4f} ± {best_t3[1][1]:.4f})")


# --- 7. T4: Anomaly subtype classification (4-class, on anomalies only) ---
print("\n=== T4: Anomaly subtype classification (4-class) ===")
subtype_map = {
    "sensor_dropout": 0, "calibration_drift": 1, "temporal_warp": 2, "periodic_interference": 3
}
mask_t4 = df["anomaly_subtype"].isin(subtype_map.keys())
tr_idx, te_idx = subset_indices(mask_t4)
print(f"  train={len(tr_idx)}  test={len(te_idx)}")
y4 = df["anomaly_subtype"].map(lambda s: subtype_map.get(s, -1)).values

t4_results = {}
for name, model_factory in [
    ("LogReg",   lambda: Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=2000))])),
    ("RidgeCls", lambda: Pipeline([("sc", StandardScaler()), ("m", RidgeClassifier())])),
    ("LinSVC",   lambda: Pipeline([("sc", StandardScaler()), ("m", LinearSVC(max_iter=5000, dual="auto"))])),
    ("RF",       lambda: RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)),
]:
    seed_scores = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        sub = seed_subsample(tr_idx, seed)
        model = model_factory()
        if name == "RF":
            model.set_params(random_state=seed)
        try:
            model.fit(X[sub], y4[sub])
            pred = model.predict(X[te_idx])
            seed_scores.append(balanced_accuracy_score(y4[te_idx], pred))
        except Exception as e:
            print(f"    [{name} seed {seed}] FAILED: {e}")
            seed_scores.append(np.nan)
    arr = np.array(seed_scores)
    t4_results[name] = (np.nanmean(arr), np.nanstd(arr))
    print(f"  {name:8s}: bal_acc = {np.nanmean(arr):.4f} ± {np.nanstd(arr):.4f}")
best_t4 = max(t4_results.items(), key=lambda kv: kv[1][0])
print(f"  BEST: {best_t4[0]} (bal_acc = {best_t4[1][0]:.4f} ± {best_t4[1][1]:.4f})")


# --- 8. Final Table III summary ---
print("\n" + "="*60)
print("TABLE III: MagBridge-Battery v1.0 — full-release benchmark")
print("="*60)
print(f"  T1 SOH regression                : R²       = {best_t1[1][0]:+.3f} ± {best_t1[1][1]:.3f}  ({best_t1[0]})")
print(f"  T2 Second-life classification    : bal_acc  =  {best_t2[1][0]:.3f} ± {best_t2[1][1]:.3f}  ({best_t2[0]})")
print(f"  T3 Anomaly detection (3-class)   : bal_acc  =  {best_t3[1][0]:.3f} ± {best_t3[1][1]:.3f}  ({best_t3[0]})")
print(f"  T4 Anomaly subtype (4-class)     : bal_acc  =  {best_t4[1][0]:.3f} ± {best_t4[1][1]:.3f}  ({best_t4[0]})")
print()
print("Per-model details for paper appendix:")
print(f"  T1: {t1_results}")
print(f"  T2: {t2_results}")
print(f"  T3: {t3_results}")
print(f"  T4: {t4_results}")
