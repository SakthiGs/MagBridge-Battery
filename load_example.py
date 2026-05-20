"""
MagBridge-Battery v1.0 — Minimal usage example.

This script demonstrates how to load the MagBridge-Battery dataset
from its Zenodo release bundle (DOI: 10.5281/zenodo.20260147) and
perform the most common operations a user would do:

    1. Load all data shards from Parquet files.
    2. Apply the leakage-safe cell-disjoint primary split.
    3. Filter to the grounded regime for SOH regression.
    4. Plot one example magnetic signature.

Usage
-----
    # After downloading and unzipping the Zenodo bundle:
    python load_example.py --data-dir ./magbridge_battery_v1_0/

The expected directory layout after unzipping is:

    magbridge_battery_v1_0/
    ├── data/
    │   ├── shard_0.parquet
    │   ├── shard_1.parquet
    │   ├── ... (5 shards, 1,352 rows each)
    │   └── metadata.parquet
    ├── splits/
    │   ├── by_cell_primary.json
    │   └── by_record_optimistic_baseline.json
    ├── manifest.json
    └── checksums.sha256

Requirements
------------
    pandas, numpy, matplotlib, pyarrow

Citation
--------
If you use MagBridge-Battery, please cite both the paper and the dataset DOI.
See CITING.md in the release bundle for full citation guidance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CHANNEL_NAMES = ["B_s1Y", "B_s1Z", "B_s2Y", "B_s2Z", "B_s1C5", "B_s2C6"]


def load_dataset(data_dir: Path) -> pd.DataFrame:
    """Load all five MagBridge-Battery v1.0 shards into a single DataFrame.

    Each row is one magnetic signature: 6 signal channels of length 100,
    plus metadata fields (sample_id, cell_id, regime, soh, etc).

    Parameters
    ----------
    data_dir : Path
        Path to the unzipped Zenodo bundle root.

    Returns
    -------
    pd.DataFrame
        6,760 rows × ~25 columns. The six signal channels and the
        time_norm column hold length-100 numpy arrays.
    """
    shard_files = sorted((data_dir / "data").glob("shard_*.parquet"))
    if not shard_files:
        raise FileNotFoundError(
            f"No shard_*.parquet files found in {data_dir / 'data'}. "
            f"Did you unzip the Zenodo bundle?"
        )
    df = pd.concat([pd.read_parquet(f) for f in shard_files], ignore_index=True)
    print(f"  Loaded {len(df):,} rows from {len(shard_files)} shards.")
    return df


def load_split(data_dir: Path, split_name: str = "by_cell_primary") -> dict:
    """Load a benchmark split from JSON.

    The split JSON contains 'train_samples', 'val_samples', 'test_samples'
    lists of sample_id strings (plus cell lists and metadata).

    Parameters
    ----------
    data_dir : Path
        Path to the unzipped Zenodo bundle root.
    split_name : str
        Either 'by_cell_primary' (recommended, cell-disjoint, parent-child
        leakage-free) or 'by_record_optimistic_baseline' (contrast split with
        known leakage; not recommended for benchmark reporting).

    Returns
    -------
    dict
        Split dict with 'train_samples', 'val_samples', 'test_samples' keys, each
        a list of sample_id strings.
    """
    split_file = data_dir / "splits" / f"{split_name}.json"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file) as f:
        split = json.load(f)
    print(
        f"  Loaded split '{split_name}': "
        f"train={len(split['train_samples']):,}, "
        f"val={len(split['val_samples']):,}, "
        f"test={len(split['test_samples']):,}"
    )
    return split


def filter_to_grounded(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to grounded-regime clean samples only.

    SOH regression (T1) and second-life classification (T2) are defined
    on grounded-regime clean samples. Extrapolation-regime samples have soh=NaN by
    design and must be excluded from these tasks.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.

    Returns
    -------
    pd.DataFrame
        Filtered dataset (typically 5,600 grounded clean rows).
    """
    grounded = df[
        (df["regime"] == "grounded") & (~df["anomaly_flag"])
    ].copy()
    print(
        f"  Filtered to grounded clean: {len(grounded):,} rows "
        f"(SOH range: {grounded['soh'].min():.3f} – {grounded['soh'].max():.3f})"
    )
    return grounded


def plot_example_signal(
    df: pd.DataFrame, sample_id: str, output_path: Path
) -> None:
    """Plot one sample's six magnetic-signature channels.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (or any subset containing the requested sample_id).
    sample_id : str
        sample_id to plot.
    output_path : Path
        Output PNG path.
    """
    row = df[df["sample_id"] == sample_id]
    if len(row) == 0:
        raise ValueError(f"sample_id '{sample_id}' not found in dataset.")
    row = row.iloc[0]

    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    time = np.asarray(row["time_norm"])
    for ax, ch_name in zip(axes.flatten(), CHANNEL_NAMES):
        signal = np.asarray(row[ch_name])
        ax.plot(time, signal, linewidth=1.2)
        ax.set_title(ch_name, fontsize=10)
        ax.set_xlabel("time_norm (a.u.)", fontsize=8)
        ax.set_ylabel("field (a.u.)", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Sample {sample_id}  "
        f"|  voltage={row['voltage']:.2f}V  "
        f"|  SOC={row['soc']:.0f}%  "
        f"|  SOH={row['soh']:.3f}  "
        f"|  regime={row['regime']}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved example signal plot: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Minimal usage example for MagBridge-Battery v1.0."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Path to the unzipped Zenodo bundle (containing data/, splits/, "
        "manifest.json, etc).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("example_signal.png"),
        help="Output PNG path for the example signal plot.",
    )
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {args.data_dir}\n"
            f"Download from https://doi.org/10.5281/zenodo.20260147 and unzip."
        )

    print("=" * 60)
    print("MagBridge-Battery v1.0 — load_example.py")
    print("=" * 60)
    print()

    # 1. Load all shards.
    print("[1/4] Loading all data shards...")
    df = load_dataset(args.data_dir)
    print()

    # 2. Apply the leakage-safe primary split.
    print("[2/4] Loading by_cell_primary split (cell-disjoint, "
          "parent-child leakage-free)...")
    split = load_split(args.data_dir, split_name="by_cell_primary")
    train = df[df["sample_id"].isin(split["train_samples"])]
    val = df[df["sample_id"].isin(split["val_samples"])]
    test = df[df["sample_id"].isin(split["test_samples"])]
    assert len(train) + len(val) + len(test) == len(df), \
        "Split coverage mismatch."
    print()

    # 3. Filter to grounded regime for SOH regression (T1) workflow.
    print("[3/4] Filtering to grounded clean samples for T1 SOH regression...")
    grounded_train = filter_to_grounded(train)
    grounded_test = filter_to_grounded(test)
    print(
        f"  Final T1 train pool: {len(grounded_train):,} samples; "
        f"T1 test pool: {len(grounded_test):,} samples."
    )
    print()

    # 4. Plot one example to verify the data loads correctly.
    print("[4/4] Plotting one example grounded-regime signal...")
    example_id = grounded_test.iloc[0]["sample_id"]
    plot_example_signal(df, example_id, args.output)
    print()

    # Summary.
    print("=" * 60)
    print("Done.")
    n_grounded_clean = int(((df["regime"] == "grounded") & (~df["anomaly_flag"])).sum())
    n_anomaly = int(((df["regime"] == "grounded") & (df["anomaly_flag"])).sum())
    n_extrap = int((df["regime"] == "extrapolation").sum())
    print(
        f"  Total samples in dataset:           {len(df):,}\n"
        f"  Grounded clean (for T1/T2):         {n_grounded_clean:,}\n"
        f"  Synthetic anomalies (for T3/T4):    {n_anomaly:,}\n"
        f"  Extrapolation / Regime-B (OOD/T3):  {n_extrap:,}\n"
    )
    print("Next steps:")
    print("  - Train a baseline: python3 code/run_benchmark.py")
    print("  - Read the dataset card: docs/dataset_card.md")
    print("  - Cite the paper and dataset DOI: see CITING.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
