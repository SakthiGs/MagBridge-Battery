"""Tests for sample ID generation, parsing, and collision detection."""

import pytest
import sys


from magbridge.sample_id import (
    make_sample_id,
    parse_sample_id,
    check_no_collisions,
    derive_per_sample_seed,
    SUBTYPE_ABBREVIATIONS,
    HASH_LENGTH,
)
from magbridge.schema import AnomalySubtype, AnomalyOrigin


# Convenient base kwargs for clean grounded samples
CLEAN_BASE = dict(
    cell_id="C042",
    voltage=3.10,
    soc=30.0,
    soh=0.85,
    generation_seed=12345,
    anomaly_subtype=AnomalySubtype.NONE,
    anomaly_origin=AnomalyOrigin.NONE,
    anomaly_severity=0.0,
    parent_sample_id=None,
    bridge_version="v1.3",
    bridge_config_hash="abc12345",
)


# ===========================================================================
# Format tests: each category produces the expected ID shape
# ===========================================================================
def test_clean_grounded_id_format():
    sid = make_sample_id(**CLEAN_BASE)
    # Format: lfp_C042_soc30_<8 hex chars>
    parts = sid.split("_")
    assert len(parts) == 4
    assert parts[0] == "lfp"
    assert parts[1] == "C042"
    assert parts[2] == "soc30"
    assert len(parts[3]) == HASH_LENGTH
    assert all(c in "0123456789abcdef" for c in parts[3])


def test_synthetic_anomaly_id_format():
    sid = make_sample_id(
        **{**CLEAN_BASE,
           "anomaly_subtype": AnomalySubtype.SENSOR_DROPOUT,
           "anomaly_origin": AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION,
           "anomaly_severity": 0.5,
           "parent_sample_id": "lfp_C042_soc30_a8f3e2d1"}
    )
    parts = sid.split("_")
    assert len(parts) == 5
    assert parts[0] == "lfp"
    assert parts[1] == "C042"
    assert parts[2] == "soc30"
    assert parts[3] == "dropout"
    assert len(parts[4]) == HASH_LENGTH


def test_regime_b_id_format():
    sid = make_sample_id(
        cell_id="C042",                  # passed but not in ID for regime-B
        voltage=2.81,
        soc=30.0,
        soh=None,
        generation_seed=999,
        anomaly_subtype=AnomalySubtype.LOW_VOLTAGE_REGIME_B,
        anomaly_origin=AnomalyOrigin.BRIDGE_EXTRAPOLATION,
        anomaly_severity=1.0,
        parent_sample_id=None,
        bridge_version="v1.3",
        bridge_config_hash="abc12345",
    )
    parts = sid.split("_")
    assert len(parts) == 4
    assert parts[0] == "lfp"
    assert parts[1] == "regimeb"
    assert parts[2] == "v281"          # 2.81V -> v281
    assert len(parts[3]) == HASH_LENGTH


def test_all_four_subtype_abbreviations_appear():
    """Each synthetic subtype should produce a recognizable prefix."""
    for subtype, expected_abbr in SUBTYPE_ABBREVIATIONS.items():
        sid = make_sample_id(
            **{**CLEAN_BASE,
               "anomaly_subtype": subtype,
               "anomaly_origin": AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION,
               "anomaly_severity": 0.5,
               "parent_sample_id": "lfp_C042_soc30_a8f3e2d1"}
        )
        assert f"_{expected_abbr}_" in sid, f"{subtype.value} should produce '{expected_abbr}' in ID"


# ===========================================================================
# Determinism tests
# ===========================================================================
def test_same_inputs_same_id():
    sid1 = make_sample_id(**CLEAN_BASE)
    sid2 = make_sample_id(**CLEAN_BASE)
    assert sid1 == sid2


def test_different_seed_different_id():
    sid1 = make_sample_id(**CLEAN_BASE)
    sid2 = make_sample_id(**{**CLEAN_BASE, "generation_seed": 99999})
    assert sid1 != sid2


def test_different_cell_different_id():
    sid1 = make_sample_id(**CLEAN_BASE)
    sid2 = make_sample_id(**{**CLEAN_BASE, "cell_id": "C099"})
    assert sid1 != sid2


def test_different_soh_different_id():
    sid1 = make_sample_id(**CLEAN_BASE)
    sid2 = make_sample_id(**{**CLEAN_BASE, "soh": 0.75})
    assert sid1 != sid2


def test_different_voltage_different_id():
    sid1 = make_sample_id(**CLEAN_BASE)
    sid2 = make_sample_id(**{**CLEAN_BASE, "voltage": 3.34})
    assert sid1 != sid2


def test_different_bridge_version_different_id():
    """Changing bridge_version MUST change the ID — different code, different sample."""
    sid1 = make_sample_id(**CLEAN_BASE)
    sid2 = make_sample_id(**{**CLEAN_BASE, "bridge_version": "v1.4"})
    assert sid1 != sid2


def test_different_config_hash_different_id():
    """Changing bridge_config_hash MUST change the ID."""
    sid1 = make_sample_id(**CLEAN_BASE)
    sid2 = make_sample_id(**{**CLEAN_BASE, "bridge_config_hash": "deadbeef"})
    assert sid1 != sid2


def test_float_precision_robust():
    """Tiny float differences below identity precision shouldn't change the ID.

    SOC is rounded to 4 decimals in the identity. So 30.0 and 30.00000001
    must produce the same ID, but 30.0 and 30.001 must not."""
    sid1 = make_sample_id(**CLEAN_BASE)  # soc=30.0
    sid2 = make_sample_id(**{**CLEAN_BASE, "soc": 30.00000001})
    sid3 = make_sample_id(**{**CLEAN_BASE, "soc": 30.001})
    assert sid1 == sid2, "subprecision float change should not affect ID"
    assert sid1 != sid3, "0.001 SOC change must affect ID"


# ===========================================================================
# Input validation
# ===========================================================================
def test_synthetic_without_parent_rejected():
    with pytest.raises(ValueError, match="parent_sample_id"):
        make_sample_id(
            **{**CLEAN_BASE,
               "anomaly_subtype": AnomalySubtype.SENSOR_DROPOUT,
               "anomaly_origin": AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION,
               "anomaly_severity": 0.5,
               "parent_sample_id": None}
        )


def test_clean_with_parent_rejected():
    with pytest.raises(ValueError, match="must not have parent_sample_id"):
        make_sample_id(
            **{**CLEAN_BASE,
               "parent_sample_id": "lfp_C042_soc30_a8f3e2d1"}
        )


def test_clean_with_nonNone_subtype_rejected():
    with pytest.raises(ValueError, match="must have anomaly_subtype=NONE"):
        make_sample_id(
            **{**CLEAN_BASE,
               "anomaly_subtype": AnomalySubtype.SENSOR_DROPOUT}
            # origin still NONE -> clean path, but subtype is non-NONE -> error
        )


def test_accepts_string_enum_values():
    """make_sample_id should accept string values OR enum instances for enums."""
    sid_enum = make_sample_id(**CLEAN_BASE)
    sid_str = make_sample_id(
        **{**CLEAN_BASE,
           "anomaly_subtype": "none",      # string
           "anomaly_origin": "none"}        # string
    )
    assert sid_enum == sid_str


# ===========================================================================
# Round-trip parsing
# ===========================================================================
def test_parse_clean():
    sid = make_sample_id(**CLEAN_BASE)
    parsed = parse_sample_id(sid)
    assert parsed["category"] == "clean"
    assert parsed["cell_id"] == "C042"
    assert parsed["soc"] == 30
    assert len(parsed["hash"]) == HASH_LENGTH


def test_parse_synthetic():
    sid = make_sample_id(
        **{**CLEAN_BASE,
           "anomaly_subtype": AnomalySubtype.CALIBRATION_DRIFT,
           "anomaly_origin": AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION,
           "anomaly_severity": 0.7,
           "parent_sample_id": "lfp_C042_soc30_a8f3e2d1"}
    )
    parsed = parse_sample_id(sid)
    assert parsed["category"] == "synthetic_anomaly"
    assert parsed["cell_id"] == "C042"
    assert parsed["soc"] == 30
    assert parsed["subtype"] == AnomalySubtype.CALIBRATION_DRIFT


def test_parse_regime_b():
    sid = make_sample_id(
        cell_id="C042", voltage=2.81, soc=30.0, soh=None, generation_seed=999,
        anomaly_subtype=AnomalySubtype.LOW_VOLTAGE_REGIME_B,
        anomaly_origin=AnomalyOrigin.BRIDGE_EXTRAPOLATION,
        anomaly_severity=1.0, parent_sample_id=None,
        bridge_version="v1.3", bridge_config_hash="abc12345",
    )
    parsed = parse_sample_id(sid)
    assert parsed["category"] == "regime_b"
    assert parsed["voltage_centivolts"] == 281
    assert len(parsed["hash"]) == HASH_LENGTH


def test_parse_malformed_rejected():
    with pytest.raises(ValueError):
        parse_sample_id("not_a_real_id")
    with pytest.raises(ValueError):
        parse_sample_id("lfp_C042_socXX_abc12345")          # SOC not numeric
    with pytest.raises(ValueError):
        parse_sample_id("lfp_C042_soc30_unknown_abc12345")  # unknown subtype abbr


# ===========================================================================
# Collision detection
# ===========================================================================
def test_collision_check_passes_unique():
    ids = ["lfp_A_soc10_1", "lfp_B_soc20_2", "lfp_regimeb_v281_3"]
    check_no_collisions(ids)  # no raise


def test_collision_check_catches_dup():
    ids = ["lfp_A_soc10_aaaaaaaa", "lfp_B_soc20_bbbbbbbb", "lfp_A_soc10_aaaaaaaa"]
    with pytest.raises(ValueError, match="collision"):
        check_no_collisions(ids)


# ===========================================================================
# Per-sample seed derivation
# ===========================================================================
def test_derived_seeds_deterministic():
    s1 = derive_per_sample_seed(20260512, "C042", "soc30", 0)
    s2 = derive_per_sample_seed(20260512, "C042", "soc30", 0)
    assert s1 == s2


def test_derived_seeds_distinct_for_different_paths():
    s1 = derive_per_sample_seed(20260512, "C042", "soc30", 0)
    s2 = derive_per_sample_seed(20260512, "C042", "soc30", 1)
    s3 = derive_per_sample_seed(20260512, "C043", "soc30", 0)
    assert s1 != s2
    assert s1 != s3


def test_derived_seed_in_int32_range():
    """Per-sample seeds must fit numpy's seed range (non-negative int32)."""
    for i in range(100):
        s = derive_per_sample_seed(20260512, "test", i)
        assert 0 <= s < 2**31


# ===========================================================================
# Real-world scale collision sanity test
# ===========================================================================
def test_full_dataset_no_collisions_simulation():
    """Generate IDs at full-dataset scale and check no collisions occur.

    Approximates the v1.0 release: 56 cells * 10 SOC levels * (clean + 4 subtypes)
    plus 560 regime-B samples. Real generation will have varying SOH/seed which
    makes this an under-estimate of distinctness, so passing here is necessary
    but not quite sufficient.
    """
    ids = []

    # Clean grounded: 56 cells x 10 SOC levels = 560 (this is per-(cell,SOC),
    # real dataset has 5600 because each (cell,SOC) appears 10 times with
    # different SOH/seeds). Use seed as discriminator here.
    socs = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    for cell_idx in range(56):
        cell_id = f"C{cell_idx:03d}"
        for soc in socs:
            for soh_pct in range(75, 100, 5):  # 5 SOH values
                ids.append(make_sample_id(
                    **{**CLEAN_BASE,
                       "cell_id": cell_id,
                       "soc": float(soc),
                       "soh": soh_pct / 100,
                       "generation_seed": cell_idx * 100 + soc}
                ))

    # Synthetic anomalies: 150 of each subtype, parented on first 150 clean IDs
    for i, subtype in enumerate([
        AnomalySubtype.SENSOR_DROPOUT,
        AnomalySubtype.CALIBRATION_DRIFT,
        AnomalySubtype.TEMPORAL_WARP,
        AnomalySubtype.PERIODIC_INTERFERENCE,
    ]):
        for j in range(150):
            parent = ids[j % len(ids)]
            parsed = parse_sample_id(parent)
            ids.append(make_sample_id(
                cell_id=parsed["cell_id"],
                voltage=3.10,
                soc=float(parsed["soc"]),
                soh=0.85,
                generation_seed=10000 + i * 200 + j,
                anomaly_subtype=subtype,
                anomaly_origin=AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION,
                anomaly_severity=0.5,
                parent_sample_id=parent,
                bridge_version="v1.3",
                bridge_config_hash="abc12345",
            ))

    # Regime-B: 560 samples across 3 voltages
    for i in range(560):
        v = [2.54, 2.81, 3.00][i % 3]
        ids.append(make_sample_id(
            cell_id="C042", voltage=v, soc=30.0, soh=None,
            generation_seed=50000 + i,
            anomaly_subtype=AnomalySubtype.LOW_VOLTAGE_REGIME_B,
            anomaly_origin=AnomalyOrigin.BRIDGE_EXTRAPOLATION,
            anomaly_severity=1.0,
            parent_sample_id=None,
            bridge_version="v1.3", bridge_config_hash="abc12345",
        ))

    # Should be at full-dataset scale with no collisions
    print(f"\nGenerated {len(ids)} IDs in collision-check simulation")
    check_no_collisions(ids)


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call(["pytest", "-v", "--tb=short", __file__]))
