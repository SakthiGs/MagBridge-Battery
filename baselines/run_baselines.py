"""
MagBridge-Battery v1.0 — baseline evaluation.

Two baselines per task:
  A. statistical_features:  per-channel statistics + cross-channel correlations
                            (~57 hand-engineered features per sample)
  B. flattened_signal:      raw signals concatenated into one long vector
                            (600 features per sample = 100 timesteps x 6 channels)

Tasks:
  T1. SOH regression                  (clean grounded samples, by_cell split)
  T2. Second-life classification     (clean grounded samples, by_cell split)
  T3. Anomaly detection (binary)     (all samples, by_cell split)
  T4. Anomaly subtype classification (synthetic anomalies, by_cell split)

Both baselines use the same readout (Ridge / Logistic Regression) so any
performance gap reflects the FEATURE REPRESENTATION, not the classifier.

Results are reported with mean ± std across 5 random seeds (different
test/train sample ordering and classifier random_state).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


# =============================================================================
# Data loading
# =============================================================================
def load_dataset(bundle_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load full dataset from parquet shards + load by_cell_primary split.

    Returns:
        (df_all, by_cell_split_dict)
    """
    shards = sorted((bundle_dir / "data").glob("shard_*.parquet"))
    dfs = [pq.read_table(s).to_pandas() for s in shards]
    df_all = pd.concat(dfs, ignore_index=True)
    with open(bundle_dir / "splits" / "by_cell_primary.json") as f:
        split = json.load(f)
    return df_all, split


def assign_split_label(df: pd.DataFrame, split: dict) -> pd.Series:
    """Return a Series mapping each sample_id to 'train'/'val'/'test'/'unassigned'."""
    sid_to_split = {}
    for s in split.get("train_samples", []) or []:
        sid_to_split[s] = "train"
    for s in split.get("val_samples", []) or []:
        sid_to_split[s] = "val"
    for s in split.get("test_samples", []) or []:
        sid_to_split[s] = "test"
    return df["sample_id"].map(sid_to_split).fillna("unassigned")


# =============================================================================
# Feature extraction
# =============================================================================
SIGNAL_COLS = ["B_s1Y", "B_s1Z", "B_s2Y", "B_s2Z", "B_s1Mag", "B_s2Mag"]


def extract_statistical_features(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Extract per-channel statistics + cross-channel correlations.

    Per channel (6 channels x 9 stats):
        mean, std, min, max, range, median, energy, slope, early-late delta

    Plus 3 cross-channel correlations:
        corr(s1Y, s1Z), corr(s2Y, s2Z), corr(s1Mag, s2Mag)

    Total: 6 * 9 + 3 = 57 features.
    """
    feats = []
    names = []
    sigs = {col: np.array(df[col].tolist()) for col in SIGNAL_COLS}
    n_samples = len(df)
    feat_per_ch = 9
    n_feats = len(SIGNAL_COLS) * feat_per_ch + 3
    X = np.zeros((n_samples, n_feats), dtype=np.float32)
    col_idx = 0

    for ch in SIGNAL_COLS:
        sig = sigs[ch]   # (N, 100)
        X[:, col_idx + 0] = sig.mean(axis=1)
        X[:, col_idx + 1] = sig.std(axis=1)
        X[:, col_idx + 2] = sig.min(axis=1)
        X[:, col_idx + 3] = sig.max(axis=1)
        X[:, col_idx + 4] = X[:, col_idx + 3] - X[:, col_idx + 2]
        X[:, col_idx + 5] = np.median(sig, axis=1)
        X[:, col_idx + 6] = (sig ** 2).sum(axis=1)
        # Linear slope via simple regression
        t = np.arange(sig.shape[1])
        t_centered = t - t.mean()
        denom = (t_centered ** 2).sum()
        X[:, col_idx + 7] = (sig * t_centered).sum(axis=1) / denom
        # Early-late delta (last quartile - first quartile)
        q = sig.shape[1] // 4
        X[:, col_idx + 8] = sig[:, -q:].mean(axis=1) - sig[:, :q].mean(axis=1)
        for stat in ["mean", "std", "min", "max", "range", "median", "energy", "slope", "early_late_delta"]:
            names.append(f"{ch}_{stat}")
        col_idx += feat_per_ch

    # Cross-channel correlations
    for a, b in [("B_s1Y", "B_s1Z"), ("B_s2Y", "B_s2Z"), ("B_s1Mag", "B_s2Mag")]:
        sa, sb = sigs[a], sigs[b]
        sa_c = sa - sa.mean(axis=1, keepdims=True)
        sb_c = sb - sb.mean(axis=1, keepdims=True)
        num = (sa_c * sb_c).sum(axis=1)
        den = np.sqrt((sa_c ** 2).sum(axis=1) * (sb_c ** 2).sum(axis=1)) + 1e-12
        X[:, col_idx] = num / den
        names.append(f"corr_{a}_{b}")
        col_idx += 1

    return X, names


def extract_flattened_signal(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Concatenate all 6 channels into a flat vector per sample.

    Output: (N, 600). 100 timesteps × 6 channels.
    """
    parts = [np.array(df[ch].tolist(), dtype=np.float32) for ch in SIGNAL_COLS]
    X = np.concatenate(parts, axis=1)
    names = [f"{ch}_t{t}" for ch in SIGNAL_COLS for t in range(100)]
    return X, names


# =============================================================================
# Task evaluation
# =============================================================================
def evaluate_regression(
    X_train, y_train, X_test, y_test, n_seeds=5, bootstrap_frac=0.8,
) -> dict:
    """Train Ridge regression across n_seeds bootstrap subsamples.

    Each seed draws a 80% subsample (with replacement) of the training set,
    fits Ridge, and evaluates on the fixed test set. Gives us genuine
    variance estimates that closed-form Ridge alone would not.
    """
    r2s, maes, rmses = [], [], []
    n_train = len(X_train)
    boot_size = int(n_train * bootstrap_frac)
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_train, size=boot_size, replace=True)
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X_train[idx])
        Xe = scaler.transform(X_test)
        model = Ridge(alpha=1.0, random_state=seed)
        model.fit(Xt, y_train[idx])
        pred = model.predict(Xe)
        r2s.append(r2_score(y_test, pred))
        maes.append(mean_absolute_error(y_test, pred))
        rmses.append(np.sqrt(mean_squared_error(y_test, pred)))
    return {
        "r2_mean": float(np.mean(r2s)), "r2_std": float(np.std(r2s)),
        "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
        "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
        "n_seeds": n_seeds,
    }


def evaluate_classification(
    X_train, y_train, X_test, y_test, n_seeds=5, multiclass=False,
    bootstrap_frac=0.8,
) -> dict:
    """Train Logistic Regression across n_seeds bootstrap subsamples."""
    bal_accs, macro_f1s, roc_aucs = [], [], []
    n_train = len(X_train)
    boot_size = int(n_train * bootstrap_frac)
    for seed in range(n_seeds):
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_train, size=boot_size, replace=True)
        scaler = StandardScaler()
        Xt = scaler.fit_transform(X_train[idx])
        Xe = scaler.transform(X_test)
        model = LogisticRegression(max_iter=2000, random_state=seed)
        model.fit(Xt, y_train[idx])
        pred = model.predict(Xe)
        bal_accs.append(balanced_accuracy_score(y_test, pred))
        avg = "macro"
        macro_f1s.append(f1_score(y_test, pred, average=avg, zero_division=0))
        if not multiclass:
            try:
                pred_prob = model.predict_proba(Xe)[:, 1]
                roc_aucs.append(roc_auc_score(y_test, pred_prob))
            except Exception:
                roc_aucs.append(float("nan"))
    out = {
        "balanced_accuracy_mean": float(np.mean(bal_accs)),
        "balanced_accuracy_std": float(np.std(bal_accs)),
        "macro_f1_mean": float(np.mean(macro_f1s)),
        "macro_f1_std": float(np.std(macro_f1s)),
        "n_seeds": n_seeds,
    }
    if not multiclass and roc_aucs:
        out["roc_auc_mean"] = float(np.nanmean(roc_aucs))
        out["roc_auc_std"] = float(np.nanstd(roc_aucs))
    return out


def chance_baseline_classification(y_train, y_test) -> dict:
    """Compute chance-level performance for classification (majority class)."""
    from sklearn.dummy import DummyClassifier
    model = DummyClassifier(strategy="most_frequent")
    model.fit(np.zeros((len(y_train), 1)), y_train)
    pred = model.predict(np.zeros((len(y_test), 1)))
    return {
        "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
    }


# =============================================================================
# Main evaluation driver
# =============================================================================
def run_all_baselines(bundle_dir: Path) -> dict:
    """Run all 4 tasks × 2 feature types and return results dict."""
    df, split = load_dataset(bundle_dir)
    df["split"] = assign_split_label(df, split)
    print(f"Loaded {len(df)} samples; split labels:")
    print(df["split"].value_counts().to_dict())

    results = {}

    # -------- Feature extraction (do once per representation) --------
    print("\nExtracting statistical features...")
    t0 = time.time()
    X_stat, _ = extract_statistical_features(df)
    print(f"  Shape: {X_stat.shape}, time: {time.time()-t0:.1f}s")

    print("Extracting flattened signal features...")
    t0 = time.time()
    X_flat, _ = extract_flattened_signal(df)
    print(f"  Shape: {X_flat.shape}, time: {time.time()-t0:.1f}s")

    # Masks once
    is_clean = df["anomaly_origin"] == "none"
    is_train = df["split"] == "train"
    is_test = df["split"] == "test"

    # =========================================================================
    # T1: SOH regression (clean grounded samples only)
    # =========================================================================
    print("\n" + "=" * 70)
    print("T1: SOH regression (clean grounded only, by_cell split)")
    print("=" * 70)
    t1_mask = is_clean
    t1_train = t1_mask & is_train
    t1_test = t1_mask & is_test
    y = df["soh"].values
    print(f"  Train samples: {t1_train.sum()}, Test samples: {t1_test.sum()}")
    print(f"  Test SOH range: [{y[t1_test].min():.3f}, {y[t1_test].max():.3f}], "
          f"std={y[t1_test].std():.3f}")
    results["T1_soh_regression"] = {}
    for name, X in [("statistical", X_stat), ("flattened", X_flat)]:
        t0 = time.time()
        r = evaluate_regression(X[t1_train], y[t1_train], X[t1_test], y[t1_test])
        r["wall_time_s"] = time.time() - t0
        results["T1_soh_regression"][name] = r
        print(f"  [{name:11s}] R²={r['r2_mean']:.3f}±{r['r2_std']:.3f}, "
              f"MAE={r['mae_mean']:.4f}±{r['mae_std']:.4f}, "
              f"RMSE={r['rmse_mean']:.4f} ({r['wall_time_s']:.1f}s)")

    # =========================================================================
    # T2: Second-life classification (clean grounded only, binary)
    # =========================================================================
    print("\n" + "=" * 70)
    print("T2: Second-life classification (clean grounded only, by_cell split)")
    print("=" * 70)
    y_t2 = df["second_life_class"].map({"reuse": 0, "recondition": 1}).values
    print(f"  Train samples: {t1_train.sum()}, Test samples: {t1_test.sum()}")
    print(f"  Train class balance: {pd.Series(y_t2[t1_train]).value_counts().to_dict()}")
    print(f"  Test class balance: {pd.Series(y_t2[t1_test]).value_counts().to_dict()}")
    chance = chance_baseline_classification(y_t2[t1_train], y_t2[t1_test])
    print(f"  Chance baseline (majority class): bal_acc={chance['balanced_accuracy']:.3f}")
    results["T2_second_life"] = {"chance": chance}
    for name, X in [("statistical", X_stat), ("flattened", X_flat)]:
        t0 = time.time()
        r = evaluate_classification(X[t1_train], y_t2[t1_train], X[t1_test], y_t2[t1_test])
        r["wall_time_s"] = time.time() - t0
        results["T2_second_life"][name] = r
        print(f"  [{name:11s}] bal_acc={r['balanced_accuracy_mean']:.3f}±{r['balanced_accuracy_std']:.3f}, "
              f"macro_f1={r['macro_f1_mean']:.3f}, ROC-AUC={r.get('roc_auc_mean', float('nan')):.3f} "
              f"({r['wall_time_s']:.1f}s)")

    # =========================================================================
    # T3: Anomaly detection (all samples, binary)
    # =========================================================================
    print("\n" + "=" * 70)
    print("T3: Anomaly detection (all samples, by_cell split)")
    print("=" * 70)
    y_t3 = df["anomaly_flag"].astype(int).values
    print(f"  Train samples: {is_train.sum()}, Test samples: {is_test.sum()}")
    print(f"  Train class balance: {pd.Series(y_t3[is_train]).value_counts().to_dict()}")
    print(f"  Test class balance: {pd.Series(y_t3[is_test]).value_counts().to_dict()}")
    chance = chance_baseline_classification(y_t3[is_train], y_t3[is_test])
    print(f"  Chance baseline (majority class): bal_acc={chance['balanced_accuracy']:.3f}")
    results["T3_anomaly_detection"] = {"chance": chance}
    for name, X in [("statistical", X_stat), ("flattened", X_flat)]:
        t0 = time.time()
        r = evaluate_classification(X[is_train], y_t3[is_train], X[is_test], y_t3[is_test])
        r["wall_time_s"] = time.time() - t0
        results["T3_anomaly_detection"][name] = r
        print(f"  [{name:11s}] bal_acc={r['balanced_accuracy_mean']:.3f}±{r['balanced_accuracy_std']:.3f}, "
              f"macro_f1={r['macro_f1_mean']:.3f}, ROC-AUC={r.get('roc_auc_mean', float('nan')):.3f} "
              f"({r['wall_time_s']:.1f}s)")

    # =========================================================================
    # T4: Anomaly subtype classification (synthetic anomalies only)
    # =========================================================================
    print("\n" + "=" * 70)
    print("T4: Anomaly subtype classification (synthetic anomalies only, by_cell)")
    print("=" * 70)
    synth_mask = df["anomaly_origin"] == "synthetic_sensor_perturbation"
    t4_train = synth_mask & is_train
    t4_test = synth_mask & is_test
    subtype_codes = {"sensor_dropout": 0, "calibration_drift": 1,
                      "timebase_jitter": 2, "periodic_interference": 3}
    y_t4 = df["anomaly_subtype"].map(subtype_codes).fillna(-1).astype(int).values
    print(f"  Train samples: {t4_train.sum()}, Test samples: {t4_test.sum()}")
    if t4_test.sum() == 0:
        print("  No anomalies in test split — task skipped")
        results["T4_anomaly_subtype"] = {"skipped": True, "reason": "no test anomalies"}
    else:
        chance_acc = 1.0 / 4   # 4 classes, balanced chance
        print(f"  Chance baseline (4-class uniform): bal_acc={chance_acc:.3f}")
        results["T4_anomaly_subtype"] = {"chance_balanced_acc": chance_acc}
        for name, X in [("statistical", X_stat), ("flattened", X_flat)]:
            t0 = time.time()
            r = evaluate_classification(
                X[t4_train], y_t4[t4_train], X[t4_test], y_t4[t4_test],
                multiclass=True,
            )
            r["wall_time_s"] = time.time() - t0
            results["T4_anomaly_subtype"][name] = r
            print(f"  [{name:11s}] bal_acc={r['balanced_accuracy_mean']:.3f}±{r['balanced_accuracy_std']:.3f}, "
                  f"macro_f1={r['macro_f1_mean']:.3f} ({r['wall_time_s']:.1f}s)")

    return results


if __name__ == "__main__":
    import sys
    bundle_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/claude/v1_0_inspect")
    print(f"Bundle dir: {bundle_dir}")
    results = run_all_baselines(bundle_dir)

    print("\n" + "=" * 70)
    print("FINAL RESULTS (JSON)")
    print("=" * 70)
    print(json.dumps(results, indent=2))

    out_path = bundle_dir.parent / "baseline_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
    print(f"\nSaved to {out_path}")
