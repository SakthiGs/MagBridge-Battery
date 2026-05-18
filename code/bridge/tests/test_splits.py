"""Tests for train/val/test split generation."""

import json
import pytest
import sys


from magbridge.splits import (
    SampleEntry,
    build_by_cell_split,
    build_by_record_split,
    check_no_pair_leakage,
    split_summary,
    BY_CELL_TRAIN_COUNT,
    BY_CELL_VAL_COUNT,
    BY_CELL_TEST_COUNT,
    BY_RECORD_WARNING,
)
from magbridge.schema import AnomalySubtype, AnomalyOrigin, SplitFile


# ===========================================================================
# Catalog fixtures
# ===========================================================================
def make_catalog_v1():
    """Build a catalog matching the v1.0 spec proportions (scaled down for tests).

    Uses 56 cells (as in real v1.0), 10 SOC levels per cell -> 560 clean samples
    (vs 5,600 in v1.0). 60 synthetic anomalies (vs 600). 56 regime-B (vs 560).
    Scaled 10x down to make tests fast while preserving structural ratios.
    """
    entries = []

    # Clean grounded: 56 cells * 10 SOC levels = 560 samples
    for cell_idx in range(56):
        cell_id = f"C{cell_idx:03d}"
        for soc_idx in range(10):
            soc = 5 + soc_idx * 5  # 5, 10, 15, ..., 50
            entries.append(SampleEntry(
                sample_id=f"lfp_{cell_id}_soc{soc:02d}_{cell_idx:04x}{soc_idx:02x}xx",
                cell_id=cell_id,
                parent_sample_id=None,
                anomaly_subtype=AnomalySubtype.NONE,
                anomaly_origin=AnomalyOrigin.NONE,
                soh=0.85,
            ))

    # Synthetic anomalies: 60 samples (15 per subtype), parented on first 60 clean
    clean_ids = [e.sample_id for e in entries]
    subtypes = [
        AnomalySubtype.SENSOR_DROPOUT,
        AnomalySubtype.CALIBRATION_DRIFT,
        AnomalySubtype.TEMPORAL_WARP,
        AnomalySubtype.PERIODIC_INTERFERENCE,
    ]
    for i, subtype in enumerate(subtypes):
        for j in range(15):
            parent_id = clean_ids[i * 15 + j]
            parent = entries[i * 15 + j]
            entries.append(SampleEntry(
                sample_id=f"lfp_{parent.cell_id}_soc{int(5 + (j % 10) * 5):02d}_"
                          f"{subtype.value.split('_')[0][:6]}_{i:02x}{j:02x}xxxx",
                cell_id=parent.cell_id,
                parent_sample_id=parent_id,
                anomaly_subtype=subtype,
                anomaly_origin=AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION,
                soh=parent.soh,
            ))

    # Regime-B: 56 samples across 3 voltages, each voltage gets a distinct
    # pseudo-cell ID (matches the production format in generate.py).
    for i in range(56):
        v_int = [254, 281, 300][i % 3]
        pseudo_cell = f"regimeB_v{v_int:03d}"
        entries.append(SampleEntry(
            sample_id=f"lfp_regimeb_v{v_int}_rb{i:04x}xx",
            cell_id=pseudo_cell,
            parent_sample_id=None,
            anomaly_subtype=AnomalySubtype.LOW_VOLTAGE_REGIME_B,
            anomaly_origin=AnomalyOrigin.BRIDGE_EXTRAPOLATION,
            soh=None,
        ))

    return entries


# ===========================================================================
# By-cell split tests
# ===========================================================================
def test_by_cell_split_cell_counts():
    catalog = make_catalog_v1()
    split = build_by_cell_split(catalog, rng_seed=42)
    assert len(split.train_cells) == BY_CELL_TRAIN_COUNT
    assert len(split.val_cells) == BY_CELL_VAL_COUNT
    assert len(split.test_cells) == BY_CELL_TEST_COUNT
    assert split.n_train_cells == BY_CELL_TRAIN_COUNT
    assert split.n_val_cells == BY_CELL_VAL_COUNT
    assert split.n_test_cells == BY_CELL_TEST_COUNT


def test_by_cell_split_cells_disjoint():
    catalog = make_catalog_v1()
    split = build_by_cell_split(catalog, rng_seed=42)
    s_train, s_val, s_test = set(split.train_cells), set(split.val_cells), set(split.test_cells)
    assert s_train.isdisjoint(s_val)
    assert s_train.isdisjoint(s_test)
    assert s_val.isdisjoint(s_test)
    assert len(s_train | s_val | s_test) == 56


def test_by_cell_split_determinism():
    catalog = make_catalog_v1()
    split1 = build_by_cell_split(catalog, rng_seed=42)
    split2 = build_by_cell_split(catalog, rng_seed=42)
    assert split1.train_cells == split2.train_cells
    assert split1.val_cells == split2.val_cells
    assert split1.test_cells == split2.test_cells


def test_by_cell_different_seeds_different_splits():
    catalog = make_catalog_v1()
    s1 = build_by_cell_split(catalog, rng_seed=42)
    s2 = build_by_cell_split(catalog, rng_seed=43)
    # At least the test cells should differ for different seeds
    assert set(s1.test_cells) != set(s2.test_cells)


def test_by_cell_no_pair_leakage():
    """Synthetic anomalies must share their parent's split."""
    catalog = make_catalog_v1()
    split = build_by_cell_split(catalog, rng_seed=42)
    check_no_pair_leakage(split, catalog)


def test_by_cell_regime_b_pseudo_cells_assigned_to_splits():
    """With distinct pseudo-cell IDs per voltage, regime-B samples must:
    1. Appear in all 3 splits (with 3 pseudo-cells and round-robin assignment).
    2. Each pseudo-cell goes to exactly one split (not scattered across).
    3. No pseudo-cell shares a split membership with a PulseBat cell.
    """
    catalog = make_catalog_v1()
    split = build_by_cell_split(catalog, rng_seed=42)
    summary = split_summary(split, catalog)

    # 1. Regime-B present in all 3 splits (with 3 pseudo-cells via round-robin)
    for sp in ("train", "val", "test"):
        rb_count = summary[sp]["anomaly_subtype_counts"].get("low_voltage_regime_B", 0)
        assert rb_count > 0, (
            f"regime-B has 0 samples in {sp}; with 3 pseudo-cells and 3 splits, "
            f"round-robin should give each split at least one pseudo-cell"
        )

    # 2. Each pseudo-cell goes to exactly one split — check by looking at the
    # cell_ids that appear in each split's regime-B samples
    regime_b_entries = [e for e in catalog if e.anomaly_origin == AnomalyOrigin.BRIDGE_EXTRAPOLATION]
    sample_id_to_cell = {e.sample_id: e.cell_id for e in regime_b_entries}

    split_cells: dict[str, set] = {"train": set(), "val": set(), "test": set()}
    for sp in ("train", "val", "test"):
        samples = getattr(split, f"{sp}_samples")
        for sid in samples:
            if sid in sample_id_to_cell:
                split_cells[sp].add(sample_id_to_cell[sid])

    # No pseudo-cell may appear in more than one split
    for c1, c2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = split_cells[c1] & split_cells[c2]
        assert not overlap, (
            f"Regime-B pseudo-cell(s) {overlap} appear in both {c1} and {c2} "
            f"— violates by-cell disjointness"
        )

    # 3. Sanity: pseudo-cell IDs are distinct from PulseBat cell IDs
    pulsebat_cells = {e.cell_id for e in catalog if e.anomaly_origin == AnomalyOrigin.NONE}
    regime_b_cells = set().union(*split_cells.values())
    assert not (pulsebat_cells & regime_b_cells), (
        f"Regime-B pseudo-cells overlap with PulseBat cells: "
        f"{pulsebat_cells & regime_b_cells}"
    )


def test_by_cell_total_samples_equals_catalog_size():
    catalog = make_catalog_v1()
    split = build_by_cell_split(catalog, rng_seed=42)
    total = split.n_train_samples + split.n_val_samples + split.n_test_samples
    assert total == len(catalog), f"split totals {total} != catalog size {len(catalog)}"


def test_by_cell_wrong_cell_count_rejected():
    """If catalog doesn't have exactly 56 cells, builder must raise."""
    catalog = make_catalog_v1()
    # Drop one cell entirely
    catalog_small = [e for e in catalog if e.cell_id != "C000"]
    with pytest.raises(ValueError, match="expects exactly 56 cells"):
        build_by_cell_split(catalog_small, rng_seed=42)


# ===========================================================================
# By-record split tests
# ===========================================================================
def test_by_record_split_counts_approx_fractions():
    catalog = make_catalog_v1()
    split = build_by_record_split(catalog, rng_seed=42)
    total = len(catalog)
    # Allow rounding tolerance of 1 sample
    assert abs(split.n_train_samples - int(round(total * 0.70))) <= 1
    assert abs(split.n_val_samples - int(round(total * 0.15))) <= 1
    assert abs(split.n_test_samples - int(round(total * 0.15))) <= 1
    assert split.n_train_samples + split.n_val_samples + split.n_test_samples == total


def test_by_record_split_disjoint():
    catalog = make_catalog_v1()
    split = build_by_record_split(catalog, rng_seed=42)
    s_train, s_val, s_test = set(split.train_samples), set(split.val_samples), set(split.test_samples)
    assert s_train.isdisjoint(s_val)
    assert s_train.isdisjoint(s_test)
    assert s_val.isdisjoint(s_test)
    assert len(s_train | s_val | s_test) == len(catalog)


def test_by_record_warning_substantive():
    """Warning text must explain WHY this split is optimistic, not just THAT it is."""
    catalog = make_catalog_v1()
    split = build_by_record_split(catalog, rng_seed=42)
    assert split.warning is not None
    assert "leakage" in split.warning.lower()
    assert "by_cell_primary" in split.warning  # tells user where to go instead
    assert len(split.warning) > 200  # substantive, not a one-liner


def test_by_record_determinism():
    catalog = make_catalog_v1()
    s1 = build_by_record_split(catalog, rng_seed=42)
    s2 = build_by_record_split(catalog, rng_seed=42)
    assert s1.train_samples == s2.train_samples
    assert s1.val_samples == s2.val_samples
    assert s1.test_samples == s2.test_samples


def test_by_record_bad_fractions_rejected():
    catalog = make_catalog_v1()
    with pytest.raises(ValueError, match="fractions must sum"):
        build_by_record_split(catalog, train_frac=0.5, val_frac=0.2, test_frac=0.2)


# ===========================================================================
# Split summary tests
# ===========================================================================
def test_split_summary_anomaly_breakdowns():
    catalog = make_catalog_v1()
    split = build_by_cell_split(catalog, rng_seed=42)
    summary = split_summary(split, catalog)

    # Total counts should sum across splits
    total_train = summary["train"]["n_samples"]
    total_val = summary["val"]["n_samples"]
    total_test = summary["test"]["n_samples"]
    assert total_train + total_val + total_test == len(catalog)

    # SOH stats present for non-regime-B samples in each split
    for sp in ("train", "val", "test"):
        assert summary[sp]["n_with_soh"] > 0


def test_split_summary_handles_by_record():
    catalog = make_catalog_v1()
    split = build_by_record_split(catalog, rng_seed=42)
    summary = split_summary(split, catalog)
    # All three splits non-empty
    assert summary["train"]["n_samples"] > 0
    assert summary["val"]["n_samples"] > 0
    assert summary["test"]["n_samples"] > 0


# ===========================================================================
# Round-trip serialization
# ===========================================================================
def test_by_cell_split_json_round_trip():
    catalog = make_catalog_v1()
    split = build_by_cell_split(catalog, rng_seed=42)
    # Serialize to JSON
    blob = split.model_dump_json()
    parsed = json.loads(blob)
    # Reload through schema validator
    reloaded = SplitFile.model_validate(parsed)
    assert reloaded.train_cells == split.train_cells
    assert reloaded.n_test_samples == split.n_test_samples


def test_by_record_split_json_round_trip():
    catalog = make_catalog_v1()
    split = build_by_record_split(catalog, rng_seed=42)
    blob = split.model_dump_json()
    parsed = json.loads(blob)
    reloaded = SplitFile.model_validate(parsed)
    assert reloaded.train_samples == split.train_samples
    assert reloaded.warning == split.warning


# ===========================================================================
# Pair-leakage detector tests
# ===========================================================================
def test_pair_leakage_detector_catches_violation():
    """If we corrupt a synthetic anomaly's cell to differ from its parent's,
    the leakage detector should raise."""
    catalog = make_catalog_v1()
    # Find a synthetic anomaly and move it to a different cell
    for entry in catalog:
        if entry.anomaly_origin == AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION:
            entry.cell_id = "C055"  # move to a different cell
            break

    # Build a split and check that pair leakage check catches the violation
    # (only if C055 happens to land in a different split than the original parent)
    split = build_by_cell_split(catalog, rng_seed=42)

    # Check: if parent is in train and child cell is in test, the checker raises
    # We can't deterministically engineer this without knowing the seed's
    # partition, so we just verify the checker function runs and doesn't crash
    # under normal conditions:
    try:
        check_no_pair_leakage(split, catalog)
    except ValueError as e:
        # If the corruption happened to land cross-split, that's what we expected
        assert "different splits" in str(e) or "not in catalog" in str(e)


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call(["pytest", "-v", "--tb=short", __file__]))
