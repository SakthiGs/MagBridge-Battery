"""
MagBridge-Battery v1.0 — Deep-learning benchmark (lean version).

Runs three off-the-shelf neural architectures (MLP, 1D-CNN, LSTM) on one
benchmark task at a time.

Usage:
  python run_dl_bench_lean.py {t1|t2|t3|t4} [--data-dir PATH]

  t1 = SOH regression
  t2 = Second-life classification
  t3 = Anomaly detection (3-class)
  t4 = Anomaly subtype classification (4-class)

By default, looks for the dataset in ./data/. Override with --data-dir or
the MAGBRIDGE_DATA environment variable.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, balanced_accuracy_score


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("task", choices=["t1", "t2", "t3", "t4"], help="Which benchmark task to run")
    p.add_argument(
        "--data-dir",
        default=os.environ.get("MAGBRIDGE_DATA", "./data"),
        help="Path to unzipped Zenodo bundle (default: ./data, or $MAGBRIDGE_DATA)",
    )
    return p.parse_args()


_args = _parse_args()
torch.set_num_threads(4)
DEVICE = torch.device("cpu")

BASE = Path(_args.data_dir).expanduser().resolve()
DATA = BASE / "data"
SPLITS = BASE / "splits"

if not DATA.exists() or not SPLITS.exists():
    raise SystemExit(
        f"Dataset not found at {BASE}.\n"
        f"Download the v1.0 bundle from Zenodo (see README) and unzip it,\n"
        f"or pass --data-dir <path>."
    )

SEEDS = [0, 1, 2]
SIGNAL_CHANNELS = ["B_s1Y", "B_s1Z", "B_s2Y", "B_s2Z", "B_s1C5", "B_s2C6"]
T_LEN = 100
C = 6
EPOCHS = 15
BATCH = 64

print("Loading data...")
df = pd.concat([pd.read_parquet(s) for s in sorted(DATA.glob("shard_*.parquet"))], ignore_index=True)
with open(SPLITS / "by_cell_primary.json") as f:
    split = json.load(f)
train_ids = set(split["train_samples"])
test_ids = set(split["test_samples"])

sig_arrays = [np.stack(df[c].values) for c in SIGNAL_CHANNELS]
X_raw = np.stack(sig_arrays, axis=-1).astype(np.float32)
print(f"  {len(df)} rows, X shape {X_raw.shape}")


def subset_indices(row_mask):
    sids = df.loc[row_mask, "sample_id"].values
    tr = np.array([i for i, s in zip(df.index[row_mask], sids) if s in train_ids])
    te = np.array([i for i, s in zip(df.index[row_mask], sids) if s in test_ids])
    return tr, te

def seed_subsample(tr_idx, seed, frac=0.8):
    rng = np.random.default_rng(seed)
    n = int(len(tr_idx) * frac)
    return tr_idx[rng.choice(len(tr_idx), n, replace=False)]


# -------------------- Models (smaller than v1) --------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, out_dim),
        )
    def forward(self, x): return self.net(x)

class CNN1D(nn.Module):
    def __init__(self, c_in, out_dim, base=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(c_in, base, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(base, base*2, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(base*2, base*2, kernel_size=3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(base*2, out_dim))
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        return self.head(x)

class LSTMNet(nn.Module):
    def __init__(self, c_in, out_dim, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(c_in, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(hidden, out_dim))
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def train_eval(model_factory, X_tr, y_tr, X_te, y_te, task_type, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    mu = X_tr.reshape(-1, X_tr.shape[-1]).mean(axis=0, keepdims=True)
    sd = X_tr.reshape(-1, X_tr.shape[-1]).std(axis=0, keepdims=True) + 1e-6
    X_tr_s = ((X_tr - mu) / sd).astype(np.float32)
    X_te_s = ((X_te - mu) / sd).astype(np.float32)

    model = model_factory().to(DEVICE)
    if task_type == "reg":
        crit = nn.MSELoss()
        y_tr_t = torch.tensor(y_tr.astype(np.float32)).view(-1, 1)
    else:
        crit = nn.CrossEntropyLoss()
        y_tr_t = torch.tensor(y_tr.astype(np.int64))

    X_tr_t = torch.tensor(X_tr_s)
    X_te_t = torch.tensor(X_te_s)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH, shuffle=True)

    best_loss = float("inf"); patience = 0
    for ep in range(EPOCHS):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item()
        if ep_loss < best_loss - 1e-4:
            best_loss = ep_loss; patience = 0
        else:
            patience += 1
            if patience >= 3:
                break

    model.eval()
    with torch.no_grad():
        pred = model(X_te_t)
    if task_type == "reg":
        return r2_score(y_te, pred.numpy().ravel())
    else:
        return balanced_accuracy_score(y_te, pred.argmax(dim=1).numpy())


# Task definitions
def get_task(name):
    if name == "t1":
        m = (df["anomaly_subtype"] == "none") & (df["regime"] == "grounded") & (df["soh"].notna())
        return m, df["soh"].values.astype(np.float32), "reg", 1, "T1 SOH regression (R²)"
    if name == "t2":
        m = (df["anomaly_subtype"] == "none") & (df["regime"] == "grounded") & (df["second_life_class"].notna())
        return m, (df["second_life_class"] == "reuse").astype(np.int64).values, "clf", 2, "T2 Second-life (bal_acc)"
    if name == "t3":
        def lbl(row):
            if row["anomaly_subtype"] == "none" and row["regime"] == "grounded": return 0
            if row["anomaly_subtype"] == "low_voltage_regime_B": return 2
            return 1
        m = np.ones(len(df), dtype=bool)
        return m, df.apply(lbl, axis=1).values.astype(np.int64), "clf", 3, "T3 Anomaly 3-class (bal_acc)"
    if name == "t4":
        subtype_map = {"sensor_dropout": 0, "calibration_drift": 1, "temporal_warp": 2, "periodic_interference": 3}
        m = df["anomaly_subtype"].isin(subtype_map.keys()).values
        y = df["anomaly_subtype"].map(lambda s: subtype_map.get(s, -1)).values.astype(np.int64)
        return m, y, "clf", 4, "T4 Anomaly subtype 4-class (bal_acc)"

task = _args.task
mask, y_all, task_type, out_dim, label = get_task(task)
tr_idx, te_idx = subset_indices(mask)
X_te = X_raw[te_idx]; y_te = y_all[te_idx]

print(f"\n=== {label} ===  train_pool={len(tr_idx)}  test={len(te_idx)}")

def make_mlp(): return MLP(T_LEN * C, out_dim)
def make_cnn(): return CNN1D(C, out_dim)
def make_lstm(): return LSTMNet(C, out_dim)

results = {}
for mname, factory in [("MLP", make_mlp), ("1D CNN", make_cnn), ("LSTM", make_lstm)]:
    scores = []
    for seed in SEEDS:
        sub = seed_subsample(tr_idx, seed)
        t0 = time.time()
        s = train_eval(factory, X_raw[sub], y_all[sub], X_te, y_te, task_type, seed=seed)
        scores.append(s)
        print(f"  {mname:7s} seed={seed}  score={s:+.4f}  ({time.time()-t0:.1f}s)")
    arr = np.array(scores)
    results[mname] = (arr.mean(), arr.std())
    print(f"  {mname:7s} MEAN: {arr.mean():+.4f} ± {arr.std():.4f}")

print(f"\nResults for {label}:")
for mname, (m, s) in results.items():
    print(f"  {mname:7s}: {m:+.4f} ± {s:.4f}")
