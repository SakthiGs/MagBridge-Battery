"""Integration test for the generator.

Runs a small-scale version of the v1.0 pipeline end-to-end:
  - 5 cells (so 50 clean samples)
  - 8 synthetic anomalies (2 per subtype)
  - 8 regime-B samples
  - Total: 66 samples

Verifies the pipeline produces a manifest, splits, and Parquet shards
that all pass validation. This is NOT a full-dataset test (which takes
~6 hours); it's a structural integrity check on the whole pipeline.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from magbridge.generate import generate_dataset
from magbridge.schema import DatasetManifest, SplitFile


# Project-root-relative path: tests/<this file>.parent.parent = project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "v1.0"


@pytest.fixture
def mini_config_and_data(tmp_path):
    """Create a downsized config + PulseBat subset for a fast integration test.

    Uses 5 cells * 10 SOC = 50 clean, 8 anomalies (2 per subtype), 8 regime-B.
    """
    # Subset PulseBat to first 5 cells
    pulsebat_full = pd.read_csv(DATA_DIR / "pulsebat_lfp.csv")
    first_5_cells = pulsebat_full["No"].drop_duplicates().head(5).tolist()
    pulsebat_mini = pulsebat_full[pulsebat_full["No"].isin(first_5_cells)].copy()
    assert len(pulsebat_mini) == 50

    # Stage minimal data dir
    mini_data = tmp_path / "data_mini"
    mini_data.mkdir()
    # Copy the bridge artifacts as-is (they're not downsized)
    for fname in ["anchor_stats.npz", "lda_fit.npz", "qrec_embeddings.npz",
                   "osf_sequences.npz"]:
        target = mini_data / fname
        target.write_bytes((DATA_DIR / fname).read_bytes())
    # Write the mini PulseBat
    pulsebat_mini.to_csv(mini_data / "pulsebat_lfp.csv", index=False)

    # Mini config — same structure as v1.0 but smaller counts.
    # Splits work on 5 cells -> 3 train / 1 val / 1 test (smallest valid split)
    mini_cfg = {
        "dataset": {"name": "MagBridge-Battery-MINI", "version": "0.0"},
        "bridge": {
            "version": "v1.3",
            "qrec_drift_strength": 800.0,
            "cone_half_angle_deg": 75.0,
            "decode_k": 8,
            "decode_kernel_sigma": 50.0,
            "amplitude_strength": 0.30,
            "spectral_strength": 0.05,
            "sensor_noise_fraction": 0.05,
            "soc_fluctuation_strength": 0.04,
            "cone_min_candidates": 8,
            "cone_disable_below_delta": 0.02,
        },
        "generation": {
            "n_clean_variants_per_record": 1,    # one variant per row for fast mini test
            "n_clean_grounded_samples": 50,
            "n_synthetic_anomaly_samples": 8,
            "n_regime_b_extrapolation_samples": 8,
            "total_expected_samples": 66,
            "anomaly_severity_min": 0.2,
            "anomaly_severity_max": 1.0,
            "rng_seed": 20260512,
        },
        "anomaly_subtypes": {
            "sensor_dropout": 2,
            "calibration_drift": 2,
            "temporal_warp": 2,
            "periodic_interference": 2,
        },
        "regime_b": {"n_samples": 8},
        "splits": {
            "primary": {
                "rng_seed": 42,
                "n_train_cells": 3,
                "n_val_cells": 1,
                "n_test_cells": 1,
            },
            "secondary": {"rng_seed": 42},
        },
    }
    import yaml
    cfg_path = tmp_path / "mini_config.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(mini_cfg, f)

    return cfg_path, mini_data, tmp_path / "output"


@pytest.mark.slow
def test_generator_end_to_end(mini_config_and_data):
    """Full-pipeline integration test: 66 samples through the entire pipeline.

    SLOW: ~3 min wall clock for 66 samples through the bridge.
    Run with `pytest -m slow` to opt in.
    """
    cfg_path, mini_data, output_dir = mini_config_and_data

    # Run the full pipeline
    generate_dataset(
        config_path=cfg_path,
        data_dir=mini_data,
        output_dir=output_dir,
        bridge_code_commit="MINI_INTEGRATION_TEST",
    )

    # ---- Verify outputs exist ----
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "splits" / "by_cell_primary.json").is_file()
    assert (output_dir / "splits" / "by_record_optimistic_baseline.json").is_file()
    # 5 shards expected (small dataset still gets 5 shards by design)
    shards = sorted((output_dir / "data").glob("shard_*.parquet"))
    assert len(shards) >= 1   # may be fewer than 5 if 66 samples / 5_shards rounds
    assert (output_dir / "data" / "metadata.parquet").is_file()

    # ---- Manifest is valid ----
    with open(output_dir / "manifest.json") as f:
        manifest_data = json.load(f)
    # Strip split_summaries which DatasetManifest doesn't model
    manifest_data.pop("split_summaries", None)
    manifest = DatasetManifest.model_validate(manifest_data)
    assert manifest.n_total_samples == 66
    assert manifest.n_clean_grounded_samples == 50
    assert manifest.n_synthetic_anomaly_samples == 8
    assert manifest.n_regime_b_extrapolation_samples == 8
    assert manifest.bridge_code_commit == "MINI_INTEGRATION_TEST"

    # ---- Splits are valid and roundtrip through pydantic ----
    with open(output_dir / "splits" / "by_cell_primary.json") as f:
        by_cell_data = json.load(f)
    by_cell = SplitFile.model_validate(by_cell_data)
    assert by_cell.split_type == "by_cell_primary"
    assert by_cell.n_train_cells == 3
    assert by_cell.n_val_cells == 1
    assert by_cell.n_test_cells == 1
    # Total samples in split should equal total catalog
    assert by_cell.n_train_samples + by_cell.n_val_samples + by_cell.n_test_samples == 66

    with open(output_dir / "splits" / "by_record_optimistic_baseline.json") as f:
        by_record_data = json.load(f)
    by_record = SplitFile.model_validate(by_record_data)
    assert by_record.split_type == "by_record_optimistic_baseline"
    assert by_record.warning is not None
    assert by_record.n_train_samples + by_record.n_val_samples + by_record.n_test_samples == 66

    # ---- Parquet shards have the right schema ----
    all_rows = []
    for shard in shards:
        df = pq.read_table(shard).to_pandas()
        all_rows.append(df)
    df_all = pd.concat(all_rows, ignore_index=True)
    assert len(df_all) == 66

    # Required signal columns present, each cell is a 100-element list
    for ch in ["B_s1Y", "B_s1Z", "B_s2Y", "B_s2Z", "B_s1C5", "B_s2C6", "time_norm"]:
        assert ch in df_all.columns, f"missing column {ch}"
        first_signal = df_all[ch].iloc[0]
        assert len(first_signal) == 100, f"{ch} should be length 100, got {len(first_signal)}"

    # Schema fields present
    for col in ["sample_id", "parent_sample_id", "cell_id", "anomaly_flag",
                 "anomaly_subtype", "anomaly_origin", "anomaly_severity",
                 "voltage", "soc", "soh", "regime"]:
        assert col in df_all.columns

    # ---- Category counts on the Parquet match expected ----
    clean_count = (df_all["anomaly_origin"] == "none").sum()
    synth_count = (df_all["anomaly_origin"] == "synthetic_sensor_perturbation").sum()
    regime_b_count = (df_all["anomaly_origin"] == "bridge_extrapolation").sum()
    assert clean_count == 50
    assert synth_count == 8
    assert regime_b_count == 8

    # ---- Anomaly subtype distribution matches config ----
    sub_counts = df_all[df_all["anomaly_origin"] == "synthetic_sensor_perturbation"]["anomaly_subtype"].value_counts().to_dict()
    assert sub_counts.get("sensor_dropout") == 2
    assert sub_counts.get("calibration_drift") == 2
    assert sub_counts.get("temporal_warp") == 2
    assert sub_counts.get("periodic_interference") == 2

    # ---- Pair invariant: every synthetic anomaly has a parent that's clean ----
    clean_ids = set(df_all[df_all["anomaly_origin"] == "none"]["sample_id"].tolist())
    for _, anom_row in df_all[df_all["anomaly_origin"] == "synthetic_sensor_perturbation"].iterrows():
        assert anom_row["parent_sample_id"] in clean_ids, (
            f"anomaly {anom_row['sample_id']} parent missing from clean set"
        )

    # ---- Metadata file is lighter and signal-free ----
    meta = pq.read_table(output_dir / "data" / "metadata.parquet").to_pandas()
    assert len(meta) == 66
    for forbidden_col in ["B_s1Y", "B_s1Z", "time_norm"]:
        assert forbidden_col not in meta.columns


@pytest.mark.slow
def test_generator_determinism(mini_config_and_data, tmp_path):
    """Running the generator twice with the same inputs must produce identical samples.

    SLOW: runs the bridge twice on 66 samples (~6 min total). Run with
    `pytest -m slow` to opt in.

    Compares the first 5 samples' B_s1Y signals between two runs."""
    cfg_path, mini_data, _ = mini_config_and_data

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    generate_dataset(cfg_path, mini_data, out1, bridge_code_commit="A")
    generate_dataset(cfg_path, mini_data, out2, bridge_code_commit="B")

    # Compare shard_0000 from both runs
    df1 = pq.read_table(out1 / "data" / "shard_0000.parquet").to_pandas()
    df2 = pq.read_table(out2 / "data" / "shard_0000.parquet").to_pandas()
    assert df1["sample_id"].tolist() == df2["sample_id"].tolist()
    # B_s1Y signals must be byte-identical
    for i in range(min(5, len(df1))):
        sig1 = np.array(df1["B_s1Y"].iloc[i])
        sig2 = np.array(df2["B_s1Y"].iloc[i])
        np.testing.assert_array_equal(sig1, sig2)


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call(["pytest", "-v", "--tb=short", __file__]))
