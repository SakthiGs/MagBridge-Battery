"""Tests for the dataset-level validator (D1-D9 rules)."""

import pytest
import sys


from magbridge.validator import (
    validate_dataset,
    DatasetValidationError,
    ValidationFailure,
)
from magbridge.schema import AnomalySubtype, AnomalyOrigin
from magbridge.splits import SampleEntry


# ===========================================================================
# Helper: build a small valid catalog
# ===========================================================================
def make_valid_catalog(n_cells: int = 4, soc_levels: int = 5):
    """Build a valid catalog: n_cells * soc_levels clean + 2 paired anomalies + 2 regime-B."""
    entries = []
    # Clean grounded
    for cell_idx in range(n_cells):
        cell_id = f"C{cell_idx:03d}"
        for soc_idx in range(soc_levels):
            soc = 5 + soc_idx * 5
            entries.append(SampleEntry(
                sample_id=f"lfp_{cell_id}_soc{soc:02d}_clean{cell_idx:02d}{soc_idx:02d}",
                cell_id=cell_id,
                parent_sample_id=None,
                anomaly_subtype=AnomalySubtype.NONE,
                anomaly_origin=AnomalyOrigin.NONE,
                soh=0.85,
            ))

    # 2 synthetic anomalies parented on first 2 clean
    parents = entries[:2]
    for i, parent in enumerate(parents):
        entries.append(SampleEntry(
            sample_id=f"lfp_{parent.cell_id}_soc05_dropout_anom{i:04d}",
            cell_id=parent.cell_id,
            parent_sample_id=parent.sample_id,
            anomaly_subtype=AnomalySubtype.SENSOR_DROPOUT,
            anomaly_origin=AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION,
            soh=0.85,
        ))

    # 2 regime-B samples
    for i in range(2):
        entries.append(SampleEntry(
            sample_id=f"lfp_regimeb_v281_rb{i:04d}",
            cell_id="regimeB_v281",
            parent_sample_id=None,
            anomaly_subtype=AnomalySubtype.LOW_VOLTAGE_REGIME_B,
            anomaly_origin=AnomalyOrigin.BRIDGE_EXTRAPOLATION,
            soh=None,
        ))
    return entries


# ===========================================================================
# Happy path
# ===========================================================================
def test_valid_catalog_passes():
    catalog = make_valid_catalog()
    failures = validate_dataset(catalog)
    assert failures == []


def test_valid_catalog_passes_with_all_expectations():
    catalog = make_valid_catalog(n_cells=4, soc_levels=5)
    failures = validate_dataset(
        catalog,
        expected_counts={
            "n_clean_grounded_samples": 20,
            "n_synthetic_anomaly_samples": 2,
            "n_regime_b_extrapolation_samples": 2,
            "total_expected_samples": 24,
        },
        expected_subtype_counts={"sensor_dropout": 2},
        expected_clean_per_cell=5,
    )
    assert failures == []


# ===========================================================================
# D1: unique IDs
# ===========================================================================
def test_d1_duplicate_ids_caught():
    catalog = make_valid_catalog()
    # Inject a duplicate
    catalog.append(SampleEntry(
        sample_id=catalog[0].sample_id,  # duplicate of first sample
        cell_id="C000",
        parent_sample_id=None,
        anomaly_subtype=AnomalySubtype.NONE,
        anomaly_origin=AnomalyOrigin.NONE,
        soh=0.85,
    ))
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_dataset(catalog)
    failures = excinfo.value.failures
    assert any(f.rule == "D1" for f in failures)


# ===========================================================================
# D2: parent must exist
# ===========================================================================
def test_d2_orphan_parent_caught():
    catalog = make_valid_catalog()
    catalog.append(SampleEntry(
        sample_id="lfp_C000_soc05_dropout_orphan",
        cell_id="C000",
        parent_sample_id="lfp_C000_soc05_DOES_NOT_EXIST",
        anomaly_subtype=AnomalySubtype.SENSOR_DROPOUT,
        anomaly_origin=AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION,
        soh=0.85,
    ))
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_dataset(catalog)
    failures = excinfo.value.failures
    assert any(f.rule == "D2" for f in failures)


# ===========================================================================
# D3: parent must be clean grounded (no anomaly chains)
# ===========================================================================
def test_d3_anomaly_chain_caught():
    catalog = make_valid_catalog()
    # Find an existing synthetic anomaly to use as a (bad) parent
    anomaly_parent = next(e for e in catalog if e.is_synthetic_anomaly())
    catalog.append(SampleEntry(
        sample_id="lfp_C000_soc05_dropout_child",
        cell_id="C000",
        parent_sample_id=anomaly_parent.sample_id,  # chain forbidden
        anomaly_subtype=AnomalySubtype.SENSOR_DROPOUT,
        anomaly_origin=AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION,
        soh=0.85,
    ))
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_dataset(catalog)
    failures = excinfo.value.failures
    assert any(f.rule == "D3" for f in failures)


def test_d3_regime_b_as_parent_caught():
    catalog = make_valid_catalog()
    regime_b_parent = next(e for e in catalog if e.is_regime_b())
    catalog.append(SampleEntry(
        sample_id="lfp_C000_soc05_dropout_rbchild",
        cell_id="C000",
        parent_sample_id=regime_b_parent.sample_id,  # also forbidden
        anomaly_subtype=AnomalySubtype.SENSOR_DROPOUT,
        anomaly_origin=AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION,
        soh=0.85,
    ))
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_dataset(catalog)
    failures = excinfo.value.failures
    assert any(f.rule == "D3" for f in failures)


# ===========================================================================
# D4: per-cell sample counts
# ===========================================================================
def test_d4_uneven_cell_counts_caught():
    catalog = make_valid_catalog(n_cells=4, soc_levels=5)
    # Drop one sample from cell C000 to make it have 4 instead of 5
    catalog = [e for e in catalog if not (e.cell_id == "C000" and e.is_clean_grounded() and "soc05" in e.sample_id)]
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_dataset(catalog, expected_clean_per_cell=5)
    failures = excinfo.value.failures
    assert any(f.rule == "D4" for f in failures)


def test_d4_skipped_when_unspecified():
    catalog = make_valid_catalog()
    # Even with uneven cells, no failure if expected_clean_per_cell is None
    failures = validate_dataset(catalog)  # no D4 expectation
    assert not any(f.rule == "D4" for f in failures)


# ===========================================================================
# D5: anomaly subtype counts match config
# ===========================================================================
def test_d5_subtype_count_mismatch_caught():
    catalog = make_valid_catalog()
    # Catalog has 2 sensor_dropout, but config expects 4
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_dataset(catalog, expected_subtype_counts={"sensor_dropout": 4})
    failures = excinfo.value.failures
    assert any(f.rule == "D5" for f in failures)


def test_d5_unexpected_subtype_present_caught():
    catalog = make_valid_catalog()
    # Add a calibration_drift sample, but config only declares sensor_dropout
    clean_parent = next(e for e in catalog if e.is_clean_grounded())
    catalog.append(SampleEntry(
        sample_id="lfp_C000_soc05_caldrift_unexpected",
        cell_id=clean_parent.cell_id,
        parent_sample_id=clean_parent.sample_id,
        anomaly_subtype=AnomalySubtype.CALIBRATION_DRIFT,
        anomaly_origin=AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION,
        soh=0.85,
    ))
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_dataset(catalog, expected_subtype_counts={"sensor_dropout": 2})
    failures = excinfo.value.failures
    assert any(f.rule == "D5" and "not declared" in f.message for f in failures)


# ===========================================================================
# D6: total counts
# ===========================================================================
def test_d6_total_mismatch_caught():
    catalog = make_valid_catalog()
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_dataset(
            catalog,
            expected_counts={
                "n_clean_grounded_samples": 999,  # wrong on purpose
                "n_synthetic_anomaly_samples": 2,
                "n_regime_b_extrapolation_samples": 2,
                "total_expected_samples": 24,
            },
        )
    failures = excinfo.value.failures
    d6_failures = [f for f in failures if f.rule == "D6"]
    assert len(d6_failures) >= 1
    assert "clean_grounded" in d6_failures[0].message


# ===========================================================================
# D8: regime-B no parent
# ===========================================================================
def test_d8_regime_b_with_parent_caught():
    catalog = make_valid_catalog()
    clean_parent = next(e for e in catalog if e.is_clean_grounded())
    catalog.append(SampleEntry(
        sample_id="lfp_regimeb_v281_bad_with_parent",
        cell_id="regimeB_v281",
        parent_sample_id=clean_parent.sample_id,  # forbidden
        anomaly_subtype=AnomalySubtype.LOW_VOLTAGE_REGIME_B,
        anomaly_origin=AnomalyOrigin.BRIDGE_EXTRAPOLATION,
        soh=None,
    ))
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_dataset(catalog)
    failures = excinfo.value.failures
    assert any(f.rule == "D8" for f in failures)


# ===========================================================================
# D9: clean grounded must have SOH
# ===========================================================================
def test_d9_clean_missing_soh_caught():
    catalog = make_valid_catalog()
    catalog.append(SampleEntry(
        sample_id="lfp_C099_soc30_no_soh",
        cell_id="C099",
        parent_sample_id=None,
        anomaly_subtype=AnomalySubtype.NONE,
        anomaly_origin=AnomalyOrigin.NONE,
        soh=None,  # missing — should be flagged
    ))
    with pytest.raises(DatasetValidationError) as excinfo:
        validate_dataset(catalog)
    failures = excinfo.value.failures
    assert any(f.rule == "D9" for f in failures)


# ===========================================================================
# Multiple failures collected together
# ===========================================================================
def test_multiple_failures_collected():
    """All failures should be reported together, not just the first."""
    catalog = make_valid_catalog()

    # Inject TWO different problems
    # 1. Orphan parent (D2)
    catalog.append(SampleEntry(
        sample_id="lfp_C000_soc05_dropout_orphan",
        cell_id="C000",
        parent_sample_id="lfp_C000_soc05_DOES_NOT_EXIST",
        anomaly_subtype=AnomalySubtype.SENSOR_DROPOUT,
        anomaly_origin=AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION,
        soh=0.85,
    ))
    # 2. Clean sample without SOH (D9)
    catalog.append(SampleEntry(
        sample_id="lfp_C099_soc30_no_soh",
        cell_id="C099",
        parent_sample_id=None,
        anomaly_subtype=AnomalySubtype.NONE,
        anomaly_origin=AnomalyOrigin.NONE,
        soh=None,
    ))

    with pytest.raises(DatasetValidationError) as excinfo:
        validate_dataset(catalog)
    failures = excinfo.value.failures
    rules_seen = {f.rule for f in failures}
    assert "D2" in rules_seen
    assert "D9" in rules_seen


# ===========================================================================
# DatasetValidationError formatting
# ===========================================================================
def test_validation_error_str_format():
    failure = ValidationFailure(
        rule="D1",
        severity="fatal",
        message="test message",
        sample_ids=["s1", "s2"],
    )
    error = DatasetValidationError([failure])
    err_str = str(error)
    assert "1 fatal" in err_str
    assert "D1" in err_str
    assert "test message" in err_str


def test_validation_error_truncates_long_sample_lists():
    failure = ValidationFailure(
        rule="D1",
        severity="fatal",
        message="many failures",
        sample_ids=[f"s{i}" for i in range(50)],
    )
    error = DatasetValidationError([failure])
    err_str = str(error)
    assert "+45 more" in err_str  # 50 - 5 head = 45 truncated


def test_validation_error_fatal_failures_property():
    failures = [
        ValidationFailure("D1", "fatal", "fatal one"),
        ValidationFailure("D5", "warning", "warning one"),
        ValidationFailure("D6", "fatal", "fatal two"),
    ]
    error = DatasetValidationError(failures)
    fatal = error.fatal_failures
    assert len(fatal) == 2
    assert all(f.severity == "fatal" for f in fatal)


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call(["pytest", "-v", "--tb=short", __file__]))
