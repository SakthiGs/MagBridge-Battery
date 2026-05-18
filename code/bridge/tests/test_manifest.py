"""Tests for dataset manifest building and provenance hashing."""

import json
import pytest
import sys
import tempfile
from pathlib import Path


from magbridge.manifest import (
    build_manifest,
    write_manifest_with_summaries,
    hash_file,
    hash_config,
    resolve_git_commit,
    render_citation,
    verify_catalog_consistent_with_config,
    HASH_LENGTH_PROVENANCE,
    HASH_LENGTH_CONFIG,
)
from magbridge.schema import (
    DatasetManifest,
    AnomalySubtype,
    AnomalyOrigin,
)
from magbridge.splits import SampleEntry, build_by_cell_split, build_by_record_split


# ===========================================================================
# Fixtures
# ===========================================================================
@pytest.fixture
def tmp_input_files(tmp_path):
    """Create dummy OSF and PulseBat files for hashing."""
    osf = tmp_path / "osf_sequences.npz"
    pulsebat = tmp_path / "pulsebat_lfp.csv"
    osf.write_bytes(b"FAKE_OSF_DATA_FOR_TESTING" * 100)
    pulsebat.write_text("U1,U2,SOH\n0.1,0.2,0.85\n")
    return osf, pulsebat


@pytest.fixture
def small_catalog():
    """Minimal catalog: 4 clean + 2 anomaly + 2 regime-B = 8 samples."""
    entries = []
    # Clean grounded
    for i in range(4):
        entries.append(SampleEntry(
            sample_id=f"lfp_C{i:03d}_soc30_clean{i:04d}",
            cell_id=f"C{i:03d}",
            parent_sample_id=None,
            anomaly_subtype=AnomalySubtype.NONE,
            anomaly_origin=AnomalyOrigin.NONE,
            soh=0.85,
        ))
    # Synthetic anomalies (parented on first 2 clean)
    for i in range(2):
        entries.append(SampleEntry(
            sample_id=f"lfp_C{i:03d}_soc30_dropout_anom{i:04d}",
            cell_id=f"C{i:03d}",
            parent_sample_id=f"lfp_C{i:03d}_soc30_clean{i:04d}",
            anomaly_subtype=AnomalySubtype.SENSOR_DROPOUT,
            anomaly_origin=AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION,
            soh=0.85,
        ))
    # Regime-B
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


@pytest.fixture
def bridge_config():
    return {
        "qrec_drift_strength": 800.0,
        "cone_half_angle_deg": 75.0,
        "decode_k": 8,
        "decode_kernel_sigma": 50.0,
        "amplitude_strength": 0.30,
        "spectral_strength": 0.05,
    }


# ===========================================================================
# File hashing
# ===========================================================================
def test_hash_file_deterministic(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello, world")
    h1 = hash_file(f)
    h2 = hash_file(f)
    assert h1 == h2
    assert len(h1) == HASH_LENGTH_PROVENANCE


def test_hash_file_differs_for_different_content(tmp_path):
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("contents A")
    b.write_text("contents B")
    assert hash_file(a) != hash_file(b)


def test_hash_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        hash_file(tmp_path / "does_not_exist.npz")


# ===========================================================================
# Config hashing
# ===========================================================================
def test_hash_config_deterministic():
    cfg = {"a": 1.0, "b": "x"}
    assert hash_config(cfg) == hash_config(cfg)
    assert len(hash_config(cfg)) == HASH_LENGTH_CONFIG


def test_hash_config_key_order_invariant():
    """Same dict in different orderings must hash identically."""
    cfg_a = {"a": 1.0, "b": 2.0}
    cfg_b = {"b": 2.0, "a": 1.0}
    assert hash_config(cfg_a) == hash_config(cfg_b)


def test_hash_config_detects_value_change():
    cfg1 = {"qrec_drift_strength": 800.0, "cone_half_angle_deg": 75.0}
    cfg2 = {"qrec_drift_strength": 800.0, "cone_half_angle_deg": 80.0}
    assert hash_config(cfg1) != hash_config(cfg2)


def test_hash_config_detects_added_param():
    cfg1 = {"a": 1.0}
    cfg2 = {"a": 1.0, "b": 2.0}
    assert hash_config(cfg1) != hash_config(cfg2)


# ===========================================================================
# Git commit resolution
# ===========================================================================
def test_resolve_git_commit_fallback_when_no_repo(tmp_path):
    """In a tmp dir with no git, resolver returns the fallback string."""
    result = resolve_git_commit(tmp_path)
    assert result == "NOT_IN_GIT"


# ===========================================================================
# Citation rendering
# ===========================================================================
def test_citation_has_placeholders():
    c = render_citation("1.0", "1.0", "2026-05-12T12:00:00Z")
    assert "ZENODO_DOI_HERE" in c
    assert "ZENODO_URL_HERE" in c
    assert "magbridge_battery_v1_0" in c
    assert "2026" in c


def test_citation_year_extracted_from_timestamp():
    c = render_citation("1.0", "1.0", "2027-08-15T00:00:00Z")
    assert "year   = {2027}" in c


# ===========================================================================
# Manifest builder
# ===========================================================================
def test_build_manifest_basic(small_catalog, bridge_config, tmp_input_files):
    osf, pulsebat = tmp_input_files
    m = build_manifest(
        catalog=small_catalog,
        bridge_config=bridge_config,
        bridge_version="v1.3",
        osf_data_path=osf,
        pulsebat_data_path=pulsebat,
    )
    assert m.n_total_samples == 8
    assert m.n_clean_grounded_samples == 4
    assert m.n_synthetic_anomaly_samples == 2
    assert m.n_regime_b_extrapolation_samples == 2
    assert m.bridge_version == "v1.3"
    assert m.dataset_version == "1.0"
    assert m.license_dataset == "CC-BY-4.0"
    assert m.bridge_config == bridge_config
    assert len(m.osf_data_hash) == HASH_LENGTH_PROVENANCE
    assert len(m.config_hash) == HASH_LENGTH_CONFIG


def test_build_manifest_with_splits(small_catalog, bridge_config, tmp_input_files):
    """When splits are provided, summaries are attached."""
    osf, pulsebat = tmp_input_files
    # Need 56 cells for by-cell split, so use a larger catalog instead
    catalog = []
    for cell_idx in range(56):
        catalog.append(SampleEntry(
            sample_id=f"lfp_C{cell_idx:03d}_soc30_clean{cell_idx:04d}",
            cell_id=f"C{cell_idx:03d}",
            parent_sample_id=None,
            anomaly_subtype=AnomalySubtype.NONE,
            anomaly_origin=AnomalyOrigin.NONE,
            soh=0.85,
        ))

    by_cell = build_by_cell_split(catalog, rng_seed=42)
    m = build_manifest(
        catalog=catalog,
        bridge_config=bridge_config,
        bridge_version="v1.3",
        osf_data_path=osf,
        pulsebat_data_path=pulsebat,
        by_cell_split=by_cell,
    )
    # Summaries attached as sidecar
    assert hasattr(m, "_split_summaries")
    assert "by_cell_primary" in m._split_summaries


def test_build_manifest_total_mismatch_raises(bridge_config, tmp_input_files, monkeypatch):
    """If category counts don't sum to total, builder must raise.

    SampleEntry uses __slots__, so we use monkeypatch on the class methods
    to simulate a sample that doesn't satisfy any is_* predicate.
    """
    osf, pulsebat = tmp_input_files
    bad_entry = SampleEntry(
        sample_id="lfp_C000_soc30_weird",
        cell_id="C000",
        parent_sample_id=None,
        anomaly_subtype=AnomalySubtype.NONE,
        anomaly_origin=AnomalyOrigin.NONE,
        soh=0.85,
    )
    # Patch class-level methods to return False for everything, simulating an
    # entry that doesn't match any category. The builder must catch this.
    monkeypatch.setattr(SampleEntry, "is_clean_grounded", lambda self: False)
    monkeypatch.setattr(SampleEntry, "is_synthetic_anomaly", lambda self: False)
    monkeypatch.setattr(SampleEntry, "is_regime_b", lambda self: False)

    with pytest.raises(ValueError, match="do not sum to total"):
        build_manifest(
            catalog=[bad_entry],
            bridge_config=bridge_config,
            bridge_version="v1.3",
            osf_data_path=osf,
            pulsebat_data_path=pulsebat,
        )


def test_build_manifest_commit_override(small_catalog, bridge_config, tmp_input_files):
    """Caller can override the bridge_code_commit at release time."""
    osf, pulsebat = tmp_input_files
    m = build_manifest(
        catalog=small_catalog,
        bridge_config=bridge_config,
        bridge_version="v1.3",
        osf_data_path=osf,
        pulsebat_data_path=pulsebat,
        bridge_code_commit_override="abcdef123456",
    )
    assert m.bridge_code_commit == "abcdef123456"


def test_manifest_timestamp_iso_format(small_catalog, bridge_config, tmp_input_files):
    """generated_at_utc must be ISO 8601 with Z (UTC)."""
    osf, pulsebat = tmp_input_files
    m = build_manifest(
        catalog=small_catalog,
        bridge_config=bridge_config,
        bridge_version="v1.3",
        osf_data_path=osf,
        pulsebat_data_path=pulsebat,
    )
    # Format: YYYY-MM-DDTHH:MM:SSZ
    import re
    pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
    assert re.match(pattern, m.generated_at_utc), \
        f"timestamp not ISO 8601 UTC: {m.generated_at_utc}"


# ===========================================================================
# Manifest writer (JSON round-trip)
# ===========================================================================
def test_write_manifest_round_trip(small_catalog, bridge_config, tmp_input_files, tmp_path):
    osf, pulsebat = tmp_input_files
    m = build_manifest(
        catalog=small_catalog,
        bridge_config=bridge_config,
        bridge_version="v1.3",
        osf_data_path=osf,
        pulsebat_data_path=pulsebat,
    )
    out = tmp_path / "manifest.json"
    write_manifest_with_summaries(m, out)
    assert out.is_file()
    # Reload and re-validate through pydantic
    with open(out) as f:
        data = json.load(f)
    # Some sidecar keys won't validate via DatasetManifest; strip them
    data.pop("split_summaries", None)
    m2 = DatasetManifest.model_validate(data)
    assert m2.n_total_samples == m.n_total_samples
    assert m2.bridge_config == m.bridge_config
    assert m2.config_hash == m.config_hash


def test_write_manifest_keys_sorted(small_catalog, bridge_config, tmp_input_files, tmp_path):
    """The JSON output should have stably sorted keys.

    This makes diffs between manifests semantically clean and ensures
    byte-identical output for identical inputs.
    """
    osf, pulsebat = tmp_input_files
    m = build_manifest(
        catalog=small_catalog,
        bridge_config=bridge_config,
        bridge_version="v1.3",
        osf_data_path=osf,
        pulsebat_data_path=pulsebat,
    )
    out = tmp_path / "manifest.json"
    write_manifest_with_summaries(m, out)
    text = out.read_text()
    # Top-level keys should be alphabetically sorted
    data = json.loads(text)
    keys = list(data.keys())
    assert keys == sorted(keys), f"top-level keys not sorted: {keys}"


def test_write_manifest_creates_parent_dirs(small_catalog, bridge_config, tmp_input_files, tmp_path):
    """Writer must create parent directories if they don't exist."""
    osf, pulsebat = tmp_input_files
    m = build_manifest(
        catalog=small_catalog,
        bridge_config=bridge_config,
        bridge_version="v1.3",
        osf_data_path=osf,
        pulsebat_data_path=pulsebat,
    )
    nested = tmp_path / "release" / "v1.0" / "manifest.json"
    write_manifest_with_summaries(m, nested)
    assert nested.is_file()


# ===========================================================================
# Catalog/config consistency check
# ===========================================================================
def test_verify_catalog_consistent_passes(small_catalog):
    """Catalog with the right counts passes the consistency check."""
    expected = {
        "n_clean_grounded_samples": 4,
        "n_synthetic_anomaly_samples": 2,
        "n_regime_b_extrapolation_samples": 2,
        "total_expected_samples": 8,
    }
    verify_catalog_consistent_with_config(small_catalog, expected)  # no raise


def test_verify_catalog_clean_mismatch_raises(small_catalog):
    expected = {
        "n_clean_grounded_samples": 999,   # wrong
        "n_synthetic_anomaly_samples": 2,
        "n_regime_b_extrapolation_samples": 2,
        "total_expected_samples": 8,
    }
    with pytest.raises(ValueError, match="clean grounded count mismatch"):
        verify_catalog_consistent_with_config(small_catalog, expected)


def test_verify_catalog_synth_mismatch_raises(small_catalog):
    expected = {
        "n_clean_grounded_samples": 4,
        "n_synthetic_anomaly_samples": 999,   # wrong
        "n_regime_b_extrapolation_samples": 2,
        "total_expected_samples": 8,
    }
    with pytest.raises(ValueError, match="synthetic anomaly count mismatch"):
        verify_catalog_consistent_with_config(small_catalog, expected)


def test_verify_catalog_regime_b_mismatch_raises(small_catalog):
    expected = {
        "n_clean_grounded_samples": 4,
        "n_synthetic_anomaly_samples": 2,
        "n_regime_b_extrapolation_samples": 999,
        "total_expected_samples": 8,
    }
    with pytest.raises(ValueError, match="regime-B count mismatch"):
        verify_catalog_consistent_with_config(small_catalog, expected)


def test_verify_catalog_total_mismatch_raises(small_catalog):
    expected = {
        "n_clean_grounded_samples": 4,
        "n_synthetic_anomaly_samples": 2,
        "n_regime_b_extrapolation_samples": 2,
        "total_expected_samples": 999,
    }
    with pytest.raises(ValueError, match="total count mismatch"):
        verify_catalog_consistent_with_config(small_catalog, expected)


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call(["pytest", "-v", "--tb=short", __file__]))
