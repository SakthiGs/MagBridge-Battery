"""Tests for the four anomaly injection functions."""

import sys
import numpy as np
import pytest


from magbridge.anomalies import (
    inject_sensor_dropout,
    inject_calibration_drift,
    inject_temporal_warp,
    inject_periodic_interference,
    inject_anomaly,
    INJECTOR_BY_SUBTYPE,
    N_TIMESTEPS,
    N_CHANNELS,
)


# ===========================================================================
# Fixtures
# ===========================================================================
def make_clean_signal(seed: int = 0):
    """Generate a clean signal that mimics realistic OSF morphology shape."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, N_TIMESTEPS)
    # 6 channels with different amplitude scales (mimics s1Y/Z, s2Y/Z, s1C5, s2C6)
    base_amps = np.array([100.0, 50.0, 120.0, 60.0, 150.0, 180.0])
    sig = np.zeros((N_TIMESTEPS, N_CHANNELS))
    for c in range(N_CHANNELS):
        sig[:, c] = base_amps[c] * (np.sin(2 * np.pi * (1 + c) * t) +
                                     0.1 * rng.standard_normal(N_TIMESTEPS))
    time_norm = t
    return sig, time_norm


# ===========================================================================
# Shape and contract tests (applies to all injectors)
# ===========================================================================
@pytest.mark.parametrize("subtype", list(INJECTOR_BY_SUBTYPE.keys()))
def test_output_shape_preserved(subtype):
    sig, t = make_clean_signal()
    rng = np.random.default_rng(42)
    out_sig, out_t = inject_anomaly(subtype, sig, t, 0.5, rng)
    assert out_sig.shape == (N_TIMESTEPS, N_CHANNELS)
    assert out_t.shape == (N_TIMESTEPS,)


@pytest.mark.parametrize("subtype", list(INJECTOR_BY_SUBTYPE.keys()))
def test_does_not_mutate_input(subtype):
    """Injectors must not modify input arrays in place."""
    sig, t = make_clean_signal()
    sig_copy = sig.copy()
    t_copy = t.copy()
    rng = np.random.default_rng(42)
    inject_anomaly(subtype, sig, t, 0.5, rng)
    np.testing.assert_array_equal(sig, sig_copy)
    np.testing.assert_array_equal(t, t_copy)


@pytest.mark.parametrize("subtype", list(INJECTOR_BY_SUBTYPE.keys()))
def test_determinism(subtype):
    """Same inputs + same seed must produce byte-identical outputs."""
    sig, t = make_clean_signal()
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    out1, t1 = inject_anomaly(subtype, sig, t, 0.5, rng1)
    out2, t2 = inject_anomaly(subtype, sig, t, 0.5, rng2)
    np.testing.assert_array_equal(out1, out2)
    np.testing.assert_array_equal(t1, t2)


@pytest.mark.parametrize("subtype", list(INJECTOR_BY_SUBTYPE.keys()))
def test_different_seeds_produce_different_outputs(subtype):
    sig, t = make_clean_signal()
    out1, t1 = inject_anomaly(subtype, sig, t, 0.5, np.random.default_rng(42))
    out2, t2 = inject_anomaly(subtype, sig, t, 0.5, np.random.default_rng(43))
    # Either signal or time_norm should differ
    differs = not (np.allclose(out1, out2) and np.allclose(t1, t2))
    assert differs


def test_unknown_subtype_rejected():
    sig, t = make_clean_signal()
    with pytest.raises(ValueError, match="Unknown anomaly subtype"):
        inject_anomaly("nonexistent_anomaly", sig, t, 0.5, np.random.default_rng(0))


@pytest.mark.parametrize("subtype", list(INJECTOR_BY_SUBTYPE.keys()))
def test_severity_out_of_range_rejected(subtype):
    sig, t = make_clean_signal()
    with pytest.raises(ValueError, match="severity"):
        inject_anomaly(subtype, sig, t, 0.1, np.random.default_rng(0))    # < 0.2
    with pytest.raises(ValueError, match="severity"):
        inject_anomaly(subtype, sig, t, 1.5, np.random.default_rng(0))    # > 1.0


# ===========================================================================
# sensor_dropout
# ===========================================================================
def test_sensor_dropout_zeros_channels():
    """At least one channel should have a contiguous near-zero region."""
    sig, t = make_clean_signal()
    out, _ = inject_sensor_dropout(sig, t, severity=0.8, rng=np.random.default_rng(42))
    # At least one channel must have a window of much-lower magnitude than original
    found_dropout = False
    for ch in range(N_CHANNELS):
        orig_amp = np.std(sig[:, ch])
        # Find any window where the post-dropout signal is much smaller
        rolling = np.abs(out[:, ch])
        # Looking for a window of >= 5 contiguous samples with magnitude <
        # 10% of original std
        below_threshold = rolling < (0.10 * orig_amp)
        # contiguous-run detection
        if below_threshold.sum() >= 5:
            # Check at least one run is >= 5 long
            runs = []
            cur = 0
            for v in below_threshold:
                if v:
                    cur += 1
                else:
                    if cur > 0:
                        runs.append(cur)
                    cur = 0
            if cur > 0:
                runs.append(cur)
            if runs and max(runs) >= 5:
                found_dropout = True
                break
    assert found_dropout, "sensor_dropout did not produce a detectable zero region"


def test_sensor_dropout_severity_scales():
    """Higher severity should affect more channels or wider windows."""
    sig, t = make_clean_signal()

    def n_changed_channels(severity):
        out, _ = inject_sensor_dropout(sig, t, severity, np.random.default_rng(42))
        diffs = np.abs(out - sig).max(axis=0)  # max change per channel
        return int((diffs > 1.0).sum())  # >1 unit change in any channel counts

    n_low = n_changed_channels(0.2)
    n_high = n_changed_channels(1.0)
    assert n_high >= n_low, f"severity scaling broken: low={n_low}, high={n_high}"


def test_sensor_dropout_does_not_affect_time_norm():
    sig, t = make_clean_signal()
    _, out_t = inject_sensor_dropout(sig, t, 0.5, np.random.default_rng(42))
    np.testing.assert_array_equal(out_t, t)


# ===========================================================================
# calibration_drift
# ===========================================================================
def test_calibration_drift_affects_all_channels():
    """Calibration drift is system-wide; all 6 channels should be perturbed."""
    sig, t = make_clean_signal()
    out, _ = inject_calibration_drift(sig, t, 0.8, np.random.default_rng(42))
    # All channels should differ from input
    for ch in range(N_CHANNELS):
        assert not np.allclose(out[:, ch], sig[:, ch], rtol=0.001), \
            f"channel {ch} unaffected by calibration drift"


def test_calibration_drift_starts_unity():
    """At t=0, the gain should be 1.0 (no drift yet)."""
    sig, t = make_clean_signal()
    out, _ = inject_calibration_drift(sig, t, 1.0, np.random.default_rng(42))
    # First sample should be approximately equal to input
    np.testing.assert_allclose(out[0, :], sig[0, :], atol=1e-9)


def test_calibration_drift_end_magnitude_scales_with_severity():
    """Severity 1.0 should produce a larger end-of-sequence shift than severity 0.2."""
    sig, t = make_clean_signal()
    # Take a channel with no zero-crossing for a stable ratio
    ch = 4  # B_s1C5, high amplitude
    out_low, _ = inject_calibration_drift(sig, t, 0.2, np.random.default_rng(42))
    out_high, _ = inject_calibration_drift(sig, t, 1.0, np.random.default_rng(42))

    # Compare end-vs-start ratios. Severity 1.0 should produce a larger
    # magnitude shift in the gain at t=1.
    gain_low_end = out_low[-1, ch] / sig[-1, ch]
    gain_high_end = out_high[-1, ch] / sig[-1, ch]
    assert abs(gain_high_end - 1.0) > abs(gain_low_end - 1.0), \
        f"severity scaling: low gain shift={gain_low_end - 1}, high={gain_high_end - 1}"


def test_calibration_drift_does_not_affect_time_norm():
    sig, t = make_clean_signal()
    _, out_t = inject_calibration_drift(sig, t, 0.5, np.random.default_rng(42))
    np.testing.assert_array_equal(out_t, t)


# ===========================================================================
# temporal_warp
# ===========================================================================
def test_temporal_warp_signal_is_modified():
    """Resampling at jittered timepoints must produce a signal that differs
    from the clean parent. This is the critical correctness test: a no-op
    temporal_warp (signal == parent) is mislabeled data."""
    sig, t = make_clean_signal()
    out_sig, _ = inject_temporal_warp(sig, t, 0.5, np.random.default_rng(42))
    # Signal must be different from the original (the bug we just fixed:
    # the old implementation returned signal.copy() unchanged).
    assert not np.array_equal(out_sig, sig), (
        "temporal_warp produced identical signal to parent — anomaly is a no-op"
    )
    # But the perturbation should be small (we're jittering, not destroying).
    rel_change = np.linalg.norm(out_sig - sig) / np.linalg.norm(sig)
    assert rel_change < 0.5, (
        f"temporal_warp changed signal too much: relative change={rel_change}"
    )


def test_temporal_warp_time_norm_preserved():
    """The returned time_norm should equal the input time_norm (the new design
    returns the ORIGINAL uniform axis; the jitter lives in the signal values).
    """
    sig, t = make_clean_signal()
    _, out_t = inject_temporal_warp(sig, t, 1.0, np.random.default_rng(42))
    np.testing.assert_array_equal(out_t, t)


def test_temporal_warp_endpoints_preserved():
    """The signal at t=0 and t=1 should match the parent (jittered time pins
    endpoints, so np.interp at those points returns the original values)."""
    sig, t = make_clean_signal()
    out_sig, _ = inject_temporal_warp(sig, t, 1.0, np.random.default_rng(42))
    np.testing.assert_array_equal(out_sig[0], sig[0])
    np.testing.assert_array_equal(out_sig[-1], sig[-1])


def test_temporal_warp_severity_scales():
    """Higher severity → larger deviation of signal from parent."""
    sig, t = make_clean_signal()
    out_low, _ = inject_temporal_warp(sig, t, 0.2, np.random.default_rng(42))
    out_high, _ = inject_temporal_warp(sig, t, 1.0, np.random.default_rng(42))

    low_change = np.linalg.norm(out_low - sig)
    high_change = np.linalg.norm(out_high - sig)
    assert high_change > low_change, (
        f"higher severity should cause larger signal change: low={low_change}, high={high_change}"
    )


def test_temporal_warp_shape_preserved():
    """Output shape must match input (resampling doesn't change array shape)."""
    sig, t = make_clean_signal()
    out_sig, out_t = inject_temporal_warp(sig, t, 1.0, np.random.default_rng(42))
    assert out_sig.shape == sig.shape
    assert out_t.shape == t.shape


def test_temporal_warp_zero_severity_is_no_op():
    """At severity=0.0 (or near-zero), the jitter should produce near-identical
    signal — both arrays should match very closely (within float tolerance)."""
    sig, t = make_clean_signal()
    # Minimum severity allowed is 0.2 per _check_inputs; check that's still small
    out_sig, _ = inject_temporal_warp(sig, t, 0.2, np.random.default_rng(42))
    rel_change = np.linalg.norm(out_sig - sig) / np.linalg.norm(sig)
    # At lowest severity (0.2), perturbation is ±2% of nominal spacing
    # → very small signal change
    assert rel_change < 0.1, (
        f"low-severity jitter should change signal little: rel_change={rel_change}"
    )


# ===========================================================================
# periodic_interference
# ===========================================================================
def test_periodic_interference_adds_sinusoidal_energy():
    """The perturbed signal should have additive sinusoidal contamination."""
    sig, t = make_clean_signal()
    out, _ = inject_periodic_interference(sig, t, 1.0, np.random.default_rng(42))
    # The difference between out and sig should be the interference itself.
    # Its energy should be non-trivial and concentrated in some frequency band.
    diff = out - sig
    # Total energy difference should be > 0 for at least one channel
    energy_per_ch = (diff ** 2).sum(axis=0)
    assert energy_per_ch.max() > 0, "periodic_interference added no energy"


def test_periodic_interference_severity_scales():
    """Higher severity → more channels affected and larger amplitude."""
    sig, t = make_clean_signal()
    out_low, _ = inject_periodic_interference(sig, t, 0.2, np.random.default_rng(42))
    out_high, _ = inject_periodic_interference(sig, t, 1.0, np.random.default_rng(42))

    affected_low = ((out_low != sig).any(axis=0)).sum()
    affected_high = ((out_high != sig).any(axis=0)).sum()
    assert affected_high >= affected_low


def test_periodic_interference_does_not_affect_time_norm():
    sig, t = make_clean_signal()
    _, out_t = inject_periodic_interference(sig, t, 0.5, np.random.default_rng(42))
    np.testing.assert_array_equal(out_t, t)


# ===========================================================================
# Dispatch
# ===========================================================================
def test_inject_anomaly_dispatch_matches_direct_call():
    """inject_anomaly(subtype, ...) should match calling the function directly."""
    sig, t = make_clean_signal()
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    out1, t1 = inject_anomaly("sensor_dropout", sig, t, 0.5, rng1)
    out2, t2 = inject_sensor_dropout(sig, t, 0.5, rng2)
    np.testing.assert_array_equal(out1, out2)
    np.testing.assert_array_equal(t1, t2)


# ===========================================================================
# Severity boundary values
# ===========================================================================
@pytest.mark.parametrize("subtype", list(INJECTOR_BY_SUBTYPE.keys()))
def test_severity_boundary_min_accepted(subtype):
    """severity = 0.2 (exact lower bound) should be accepted."""
    sig, t = make_clean_signal()
    out, _ = inject_anomaly(subtype, sig, t, 0.2, np.random.default_rng(42))
    assert out.shape == (N_TIMESTEPS, N_CHANNELS)


@pytest.mark.parametrize("subtype", list(INJECTOR_BY_SUBTYPE.keys()))
def test_severity_boundary_max_accepted(subtype):
    """severity = 1.0 (exact upper bound) should be accepted."""
    sig, t = make_clean_signal()
    out, _ = inject_anomaly(subtype, sig, t, 1.0, np.random.default_rng(42))
    assert out.shape == (N_TIMESTEPS, N_CHANNELS)


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call(["pytest", "-v", "--tb=short", __file__]))
