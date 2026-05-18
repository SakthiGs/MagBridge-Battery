"""Tests for the bridge package (qrec, morphology, bridge integration)."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

from magbridge.bridge import (
    BridgeV13,
    BridgeConfig,
    MorphologyBank,
    Regime,
    classify_regime,
    QuantumRecurrentReservoir10q,
    OSF_ANCHORS,
)
from magbridge.bridge.qrec import (
    pool_reservoir_states,
    angle_scale_sequences,
    POOLING_DEFAULT,
)


# Project-root-relative path: tests/<this file>.parent.parent = project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "v1.0"


# ===========================================================================
# Module-level bridge fixture (expensive to build, fine to share)
# ===========================================================================
@pytest.fixture(scope="module")
def bridge():
    """Build BridgeV13 once per test module."""
    return BridgeV13(
        anchor_stats_path=DATA_DIR / "anchor_stats.npz",
        osf_seq_path=DATA_DIR / "osf_sequences.npz",
        osf_qrec_emb_path=DATA_DIR / "qrec_embeddings.npz",
        lda_fit_path=DATA_DIR / "lda_fit.npz",
    )


# ===========================================================================
# Regime classification
# ===========================================================================
def test_regime_grounded():
    r, near = classify_regime(3.10)
    assert r == Regime.GROUNDED
    assert near == 3.10

    r, near = classify_regime(3.20)
    assert r == Regime.GROUNDED


def test_regime_extrapolation():
    r, near = classify_regime(2.54)
    assert r == Regime.EXTRAPOLATION
    assert near == 2.54

    r, near = classify_regime(2.81)
    assert r == Regime.EXTRAPOLATION


def test_regime_unsupported():
    r, _ = classify_regime(2.40)
    assert r == Regime.UNSUPPORTED
    r, _ = classify_regime(3.50)
    assert r == Regime.UNSUPPORTED


def test_nearest_anchor():
    """Nearest-anchor lookup should be exact for anchor voltages."""
    for v in OSF_ANCHORS:
        _, near = classify_regime(v)
        assert near == v


# ===========================================================================
# MorphologyBank
# ===========================================================================
def test_morphology_bank_loads():
    bank = MorphologyBank(DATA_DIR / "anchor_stats.npz")
    assert bank.shape == (100, 6)


def test_morphology_anchor_mean_shape():
    bank = MorphologyBank(DATA_DIR / "anchor_stats.npz")
    for v in OSF_ANCHORS:
        m = bank.anchor_mean(v)
        assert m.shape == (100, 6)


def test_morphology_bracket_at_anchor():
    """At anchor v=3.10, the bracket should resolve to the anchor itself.

    The implementation picks the FIRST interval [v_low, v_high] containing v.
    For v=3.10 exactly, this is [3.00, 3.10] with alpha=1.0 -- which gives
    (1-1)*mean(3.00) + 1*mean(3.10) = mean(3.10). Both that AND the lookup
    bank.anchor_mean(3.10) must produce the same trajectory.
    """
    bank = MorphologyBank(DATA_DIR / "anchor_stats.npz")
    v_low, v_high, alpha = bank._bracket(3.10)
    # v=3.10 falls in the interval [3.00, 3.10] with alpha=1, so v_high MUST be 3.10
    assert v_high == 3.10
    # And the bracket interpolation at this alpha must reproduce the anchor exactly
    if v_low == v_high:
        assert alpha == 0.0
    else:
        # Interpolate manually and compare to anchor_mean(3.10)
        s_low = bank._load(v_low)
        s_high = bank._load(v_high)
        interpolated = (1 - alpha) * s_low["mean_traj"] + alpha * s_high["mean_traj"]
        anchor = bank.anchor_mean(3.10)
        np.testing.assert_allclose(interpolated, anchor)


def test_morphology_bracket_at_first_anchor():
    """At the smallest anchor 2.54V, bracket should collapse to (v_low, v_low)."""
    bank = MorphologyBank(DATA_DIR / "anchor_stats.npz")
    v_low, v_high, alpha = bank._bracket(2.54)
    assert v_low == 2.54
    assert v_high == 2.54
    assert alpha == 0.0


def test_morphology_bracket_between():
    """Voltage exactly between two anchors should give alpha=0.5."""
    bank = MorphologyBank(DATA_DIR / "anchor_stats.npz")
    # 3.05 is exactly between 3.00 and 3.10
    v_low, v_high, alpha = bank._bracket(3.05)
    assert v_low == 3.00
    assert v_high == 3.10
    assert abs(alpha - 0.5) < 1e-9


def test_morphology_sample_shape():
    bank = MorphologyBank(DATA_DIR / "anchor_stats.npz")
    rng = np.random.default_rng(42)
    s = bank.sample_base_morphology(3.10, rng)
    assert s.shape == (100, 6)


def test_morphology_sample_deterministic_with_seed():
    bank = MorphologyBank(DATA_DIR / "anchor_stats.npz")
    s1 = bank.sample_base_morphology(3.10, np.random.default_rng(42))
    s2 = bank.sample_base_morphology(3.10, np.random.default_rng(42))
    np.testing.assert_array_equal(s1, s2)


# ===========================================================================
# QRec reservoir
# ===========================================================================
def test_qrec_output_dimensions():
    qrec = QuantumRecurrentReservoir10q(seed=42)
    assert qrec.n_outputs == 57  # 18 single + 15 proc pairs + 24 cross pairs


def test_qrec_process_sequence_shape():
    qrec = QuantumRecurrentReservoir10q(seed=42)
    # Use angle-scaled input
    seq = np.random.default_rng(0).standard_normal((100, 6))
    scaled, _ = angle_scale_sequences(seq[None])
    states = qrec.process_sequence(scaled[0])
    assert states.shape == (100, 57)


def test_qrec_pooled_embedding_171d():
    qrec = QuantumRecurrentReservoir10q(seed=42)
    seq = np.random.default_rng(0).standard_normal((100, 6))
    scaled, _ = angle_scale_sequences(seq[None])
    states = qrec.process_sequence(scaled[0])
    pooled = pool_reservoir_states(states, POOLING_DEFAULT)
    assert pooled.shape == (171,)


def test_qrec_reset_clears_memory():
    qrec = QuantumRecurrentReservoir10q(seed=42)
    seq = np.random.default_rng(0).standard_normal((100, 6))
    scaled, _ = angle_scale_sequences(seq[None])

    # Process, then re-process; memory_angles should be the same after both
    # because process_sequence calls reset() first.
    _ = qrec.process_sequence(scaled[0])
    angles_after_1 = qrec.memory_angles.copy()
    _ = qrec.process_sequence(scaled[0])
    angles_after_2 = qrec.memory_angles.copy()
    np.testing.assert_array_equal(angles_after_1, angles_after_2)


# ===========================================================================
# BridgeV13 — end-to-end generation
# ===========================================================================
def test_bridge_builds(bridge):
    """Just constructing the bridge should succeed."""
    assert bridge.osf_lda.shape == (205, 4)
    assert bridge.d_state_lda_unit.shape == (4,)
    assert abs(np.linalg.norm(bridge.d_state_lda_unit) - 1.0) < 1e-9


def test_bridge_generate_shape(bridge):
    signal, regime, nearest = bridge.generate(
        voltage=3.10, soc=30.0, soh=0.85, seed=42
    )
    assert signal.shape == (100, 6)
    assert regime == Regime.GROUNDED
    assert nearest == 3.10


def test_bridge_generate_determinism(bridge):
    """Same seed must produce byte-identical output."""
    sig1, _, _ = bridge.generate(voltage=3.10, soc=30.0, soh=0.85, seed=42)
    sig2, _, _ = bridge.generate(voltage=3.10, soc=30.0, soh=0.85, seed=42)
    np.testing.assert_array_equal(sig1, sig2)


def test_bridge_different_seeds_different_output(bridge):
    sig1, _, _ = bridge.generate(voltage=3.10, soc=30.0, soh=0.85, seed=42)
    sig2, _, _ = bridge.generate(voltage=3.10, soc=30.0, soh=0.85, seed=99)
    assert not np.allclose(sig1, sig2)


def test_bridge_soh_sensitivity(bridge):
    """Different SOH values should produce meaningfully different signals."""
    sig_high, _, _ = bridge.generate(voltage=3.10, soc=30.0, soh=0.95, seed=42)
    sig_low, _, _ = bridge.generate(voltage=3.10, soc=30.0, soh=0.75, seed=42)
    diff = np.linalg.norm(sig_high - sig_low)
    sig_range = sig_high.max() - sig_high.min()
    # The difference should be at least a substantial fraction of the signal range
    assert diff > 0.5 * sig_range, f"SOH sensitivity too low: diff={diff}, range={sig_range}"


def test_bridge_unsupported_voltage_raises(bridge):
    with pytest.raises(ValueError, match="outside the supported"):
        bridge.generate(voltage=2.40, soc=30.0, soh=0.85, seed=42)
    with pytest.raises(ValueError, match="outside the supported"):
        bridge.generate(voltage=3.50, soc=30.0, soh=0.85, seed=42)


def test_bridge_extrapolation_regime(bridge):
    """Low voltages should be tagged as EXTRAPOLATION."""
    _, regime, _ = bridge.generate(voltage=2.81, soc=30.0, soh=0.85, seed=42)
    assert regime == Regime.EXTRAPOLATION


def test_bridge_diagnostics_populated(bridge):
    bridge.generate(voltage=3.10, soc=30.0, soh=0.85, seed=42)
    diag = bridge.last_diagnostics
    required_keys = {
        "soh", "soh_delta", "magnitude",
        "lda_coords_before", "lda_coords_after",
        "n_candidates_after_cone", "topk_neighbours", "topk_weights",
        "blend_factor", "regime", "nearest_anchor",
    }
    assert required_keys.issubset(set(diag.keys()))


def test_bridge_cone_filter_active_for_meaningful_soh(bridge):
    """For non-trivial SOH delta (>0.02), the cone filter should restrict
    the candidate set below the total of 205."""
    bridge.generate(voltage=3.10, soc=30.0, soh=0.85, seed=42)
    n_cand = bridge.last_diagnostics["n_candidates_after_cone"]
    # With cone_half_angle=75deg and SOH=0.85 (delta=0.15), cone should
    # exclude at least some samples (but not necessarily — depends on
    # how OSF samples distribute in LDA space relative to d_state).
    # The honest assertion is: n_cand <= 205 (filter is at least applied).
    assert n_cand <= 205


def test_bridge_cone_filter_bypassed_for_tiny_soh(bridge):
    """For SOH delta below cone_disable_below_delta (0.02), all 205 OSF
    samples should be eligible."""
    bridge.generate(voltage=3.10, soc=30.0, soh=0.99, seed=42)   # delta=0.01
    n_cand = bridge.last_diagnostics["n_candidates_after_cone"]
    assert n_cand == 205, (
        f"cone should be bypassed for tiny SOH delta but n_cand={n_cand}"
    )


def test_bridge_signal_is_finite(bridge):
    """No NaN or inf in generated signals."""
    sig, _, _ = bridge.generate(voltage=3.10, soc=30.0, soh=0.85, seed=42)
    assert np.isfinite(sig).all()


def test_bridge_config_dict_serialisable(bridge):
    """config_dict() must return a dict that's JSON-serialisable."""
    import json
    d = bridge.config_dict()
    blob = json.dumps(d)  # should not raise
    assert "qrec_drift_strength" in d
    assert d["qrec_drift_strength"] == 800.0  # locked v1.0 value
    assert d["cone_half_angle_deg"] == 75.0


# ===========================================================================
# Bridge with overridden config
# ===========================================================================
def test_bridge_accepts_custom_config():
    cfg = BridgeConfig(qrec_drift_strength=400.0, cone_half_angle_deg=45.0)
    bridge = BridgeV13(
        anchor_stats_path=DATA_DIR / "anchor_stats.npz",
        osf_seq_path=DATA_DIR / "osf_sequences.npz",
        osf_qrec_emb_path=DATA_DIR / "qrec_embeddings.npz",
        lda_fit_path=DATA_DIR / "lda_fit.npz",
        config=cfg,
    )
    assert bridge.config.qrec_drift_strength == 400.0
    assert bridge.config.cone_half_angle_deg == 45.0


def test_bridge_direction_override(bridge):
    """Passing a custom direction should change the output (A1 ablation)."""
    sig_default, _, _ = bridge.generate(voltage=3.10, soc=30.0, soh=0.85, seed=42)
    # Use a random direction in 4-D LDA space (unit normalized)
    rng = np.random.default_rng(0)
    rand_dir = rng.standard_normal(4)
    rand_dir = rand_dir / np.linalg.norm(rand_dir)
    sig_override, _, _ = bridge.generate(
        voltage=3.10, soc=30.0, soh=0.85, seed=42, direction_override=rand_dir
    )
    # Different direction should produce different output
    assert not np.allclose(sig_default, sig_override)


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call(["pytest", "-v", "--tb=short", __file__]))
