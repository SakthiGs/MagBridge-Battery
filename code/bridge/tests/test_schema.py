"""Tests for schema integrity rules.

Each test maps to one of the cross-field invariants documented in
SampleRecord._check_anomaly_consistency.
"""

import pytest
import sys

from magbridge.schema import (
    SampleRecord, SplitFile, DatasetManifest,
    Regime, AnomalySubtype, AnomalyOrigin, SecondLifeClass, Chemistry,
    SIGNAL_N_TIMESTEPS,
)


# ---------------------------------------------------------------------------
# Sample factory helpers
# ---------------------------------------------------------------------------
def _make_signal():
    return [0.0] * SIGNAL_N_TIMESTEPS


def make_clean_grounded(**overrides):
    """Build a valid clean grounded sample. Override fields as needed."""
    base = dict(
        sample_id="lfp_C042_soc30_a8f3e2d1",
        parent_sample_id=None,
        cell_id="C042",
        generation_seed=12345,
        bridge_version="v1.3",
        bridge_config_hash="abc12345",
        voltage=3.10,
        soc=30.0,
        soh=0.85,
        chemistry=Chemistry.LFP,
        u_features=[0.0] * 21,
        regime=Regime.GROUNDED,
        nearest_anchor=3.10,
        anomaly_flag=False,
        anomaly_subtype=AnomalySubtype.NONE,
        anomaly_origin=AnomalyOrigin.NONE,
        anomaly_severity=0.0,
        second_life_class=SecondLifeClass.REUSE,
        B_s1Y=_make_signal(),
        B_s1Z=_make_signal(),
        B_s2Y=_make_signal(),
        B_s2Z=_make_signal(),
        B_s1C5=_make_signal(),
        B_s2C6=_make_signal(),
        time_norm=[i / 99 for i in range(100)],
    )
    base.update(overrides)
    return SampleRecord(**base)


def make_synthetic_anomaly(**overrides):
    """Build a valid synthetic-anomaly sample with parent."""
    base = dict(
        sample_id="lfp_C042_soc30_dropout_b9f4d3e2",
        parent_sample_id="lfp_C042_soc30_a8f3e2d1",
        cell_id="C042",
        generation_seed=67890,
        bridge_version="v1.3",
        bridge_config_hash="abc12345",
        voltage=3.10,
        soc=30.0,
        soh=0.85,
        chemistry=Chemistry.LFP,
        u_features=[0.0] * 21,
        regime=Regime.GROUNDED,
        nearest_anchor=3.10,
        anomaly_flag=True,
        anomaly_subtype=AnomalySubtype.SENSOR_DROPOUT,
        anomaly_origin=AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION,
        anomaly_severity=0.5,
        second_life_class=SecondLifeClass.REUSE,
        B_s1Y=_make_signal(),
        B_s1Z=_make_signal(),
        B_s2Y=_make_signal(),
        B_s2Z=_make_signal(),
        B_s1C5=_make_signal(),
        B_s2C6=_make_signal(),
        time_norm=[i / 99 for i in range(100)],
    )
    base.update(overrides)
    return SampleRecord(**base)


def make_regime_b(**overrides):
    """Build a valid regime-B sample (no parent, no SOH required)."""
    base = dict(
        sample_id="lfp_regime_b_2_81V_c0d1e2f3",
        parent_sample_id=None,
        cell_id="C042",
        generation_seed=999,
        bridge_version="v1.3",
        bridge_config_hash="abc12345",
        voltage=2.81,
        soc=30.0,
        soh=None,
        chemistry=Chemistry.LFP,
        u_features=None,
        regime=Regime.EXTRAPOLATION,
        nearest_anchor=2.81,
        anomaly_flag=True,
        anomaly_subtype=AnomalySubtype.LOW_VOLTAGE_REGIME_B,
        anomaly_origin=AnomalyOrigin.BRIDGE_EXTRAPOLATION,
        anomaly_severity=1.0,
        second_life_class=None,
        B_s1Y=_make_signal(),
        B_s1Z=_make_signal(),
        B_s2Y=_make_signal(),
        B_s2Z=_make_signal(),
        B_s1C5=_make_signal(),
        B_s2C6=_make_signal(),
        time_norm=[i / 99 for i in range(100)],
    )
    base.update(overrides)
    return SampleRecord(**base)


# ---------------------------------------------------------------------------
# Valid-construction tests
# ---------------------------------------------------------------------------
def test_clean_grounded_builds():
    s = make_clean_grounded()
    assert s.anomaly_flag is False
    assert s.parent_sample_id is None


def test_synthetic_anomaly_builds():
    s = make_synthetic_anomaly()
    assert s.anomaly_flag is True
    assert s.parent_sample_id == "lfp_C042_soc30_a8f3e2d1"


def test_regime_b_builds():
    s = make_regime_b()
    assert s.anomaly_subtype == AnomalySubtype.LOW_VOLTAGE_REGIME_B
    assert s.anomaly_origin == AnomalyOrigin.BRIDGE_EXTRAPOLATION
    assert s.soh is None
    assert s.parent_sample_id is None


# ---------------------------------------------------------------------------
# Cross-field invariant tests (the strict validators)
# ---------------------------------------------------------------------------
def test_R1_flag_true_requires_nonNone_subtype():
    with pytest.raises(ValueError, match="anomaly_flag=True but anomaly_subtype=NONE"):
        make_clean_grounded(anomaly_flag=True)


def test_R1_flag_false_requires_NONE_subtype():
    with pytest.raises(ValueError, match="anomaly_flag=False but anomaly_subtype"):
        make_clean_grounded(anomaly_subtype=AnomalySubtype.SENSOR_DROPOUT)


def test_R3_regime_b_subtype_requires_bridge_origin():
    with pytest.raises(ValueError, match="low_voltage_regime_B subtype must have bridge"):
        make_regime_b(anomaly_origin=AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION)


def test_R4_synthetic_subtype_requires_synthetic_origin():
    with pytest.raises(ValueError, match="must have synthetic_sensor_perturbation origin"):
        make_synthetic_anomaly(anomaly_origin=AnomalyOrigin.BRIDGE_EXTRAPOLATION)


def test_R5_parent_id_only_for_synthetic_origin():
    with pytest.raises(ValueError, match="parent_sample_id set but origin"):
        make_regime_b(parent_sample_id="lfp_C042_soc30_a8f3e2d1")


def test_R5_synthetic_origin_requires_parent_id():
    with pytest.raises(ValueError, match="must have parent_sample_id"):
        make_synthetic_anomaly(parent_sample_id=None)


def test_R6_regime_b_requires_extrapolation_regime():
    with pytest.raises(ValueError, match="low_voltage_regime_B must have regime=extrapolation"):
        make_regime_b(regime=Regime.GROUNDED)


def test_R7_clean_sample_has_zero_severity():
    with pytest.raises(ValueError, match="clean sample must have anomaly_severity=0.0"):
        make_clean_grounded(anomaly_severity=0.5)


def test_R8_synthetic_severity_in_range():
    with pytest.raises(ValueError, match="severity in"):
        make_synthetic_anomaly(anomaly_severity=0.1)


def test_signal_length_enforced():
    with pytest.raises(ValueError, match="length 100"):
        make_clean_grounded(B_s1Y=[0.0] * 99)


def test_time_norm_length_enforced():
    with pytest.raises(ValueError, match="time_norm must have length 100"):
        make_clean_grounded(time_norm=[0.0] * 99)


def test_u_features_length_enforced():
    with pytest.raises(ValueError, match="u_features must have length 21"):
        make_clean_grounded(u_features=[0.0] * 20)


# ---------------------------------------------------------------------------
# Split file tests
# ---------------------------------------------------------------------------
def test_split_by_cell_overlap_rejected():
    with pytest.raises(ValueError, match="overlap"):
        SplitFile(
            split_type="by_cell_primary",
            rng_seed=42,
            train_cells=["C001", "C002"],
            val_cells=["C002", "C003"],   # C002 overlap
            test_cells=["C004"],
            train_samples=["s1", "s2"],
            val_samples=["s3", "s4"],
            test_samples=["s5"],
            n_train_cells=2,
            n_val_cells=2,
            n_test_cells=1,
            n_train_samples=200,
            n_val_samples=200,
            n_test_samples=100,
        )


def test_split_by_record_requires_warning():
    with pytest.raises(ValueError, match="warning about within-cell leakage"):
        SplitFile(
            split_type="by_record_optimistic_baseline",
            rng_seed=42,
            train_samples=["s1"],
            val_samples=["s2"],
            test_samples=["s3"],
            n_train_samples=1,
            n_val_samples=1,
            n_test_samples=1,
            # warning intentionally missing
        )


# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------
def test_manifest_total_must_match_sum():
    with pytest.raises(ValueError, match="does not match the sum"):
        DatasetManifest(
            generated_at_utc="2026-05-12T00:00:00Z",
            n_total_samples=999,                  # wrong
            n_clean_grounded_samples=5600,
            n_synthetic_anomaly_samples=600,
            n_regime_b_extrapolation_samples=560,
            osf_data_hash="x",
            pulsebat_data_hash="y",
            bridge_code_commit="z",
            config_hash="w",
            bridge_version="v1.3",
            bridge_config={},
        )


def test_manifest_correct_total_accepted():
    m = DatasetManifest(
        generated_at_utc="2026-05-12T00:00:00Z",
        n_total_samples=6760,
        n_clean_grounded_samples=5600,
        n_synthetic_anomaly_samples=600,
        n_regime_b_extrapolation_samples=560,
        osf_data_hash="x",
        pulsebat_data_hash="y",
        bridge_code_commit="z",
        config_hash="w",
        bridge_version="v1.3",
        bridge_config={"qrec_drift_strength": 800.0},
    )
    assert m.n_total_samples == 6760


if __name__ == "__main__":
    import subprocess, sys
    sys.exit(subprocess.call(["pytest", "-v", "--tb=short", __file__]))
