"""
MagBridge-Battery — synthetic anomaly injection.

Four injection functions, each parameterized by `severity ∈ [0.2, 1.0]`:

    sensor_dropout:          one or more channels go near-zero for a contiguous
                             window (physical: solder/ADC/EMI failure)
    calibration_drift:       smooth multiplicative gain drift across all 6
                             channels (physical: amp gain or supply drift)
    temporal_warp:           signal resampled at perturbed timepoints (physical:
                             clock jitter, irregular sampling; perturbation is
                             in signal values, time_norm stays uniform)
    periodic_interference:   sinusoidal additive contamination on selected
                             channels (physical: mains hum, motor noise)

Each function takes (clean_signal, time_norm, severity, rng) and returns
(perturbed_signal, perturbed_time_norm). One anomaly per sample, never
stacked — the schema invariants enforce this.

Reproducibility: every call uses a numpy Generator that the caller seeds
deterministically. Same inputs + same seed → byte-identical output.
"""

from __future__ import annotations

from typing import Optional
import numpy as np


# Signal shape constants (must match schema.SIGNAL_N_TIMESTEPS / CHANNELS)
N_TIMESTEPS = 100
N_CHANNELS = 6


# =============================================================================
# Anomaly 1: sensor_dropout
# =============================================================================
def inject_sensor_dropout(
    signal: np.ndarray,
    time_norm: np.ndarray,
    severity: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """One or more channels zeroed for a contiguous window.

    Real dropouts leave residual ADC noise rather than exact zero; we model
    this by replacing the affected window with Gaussian noise at ~3% of the
    channel's pre-dropout amplitude.

    Severity mapping:
        severity 0.2 → 1 channel, window = 10% of T
        severity 1.0 → 3 channels, window = 50% of T

    Window start sampled uniformly within the sequence.
    """
    _check_inputs(signal, time_norm, severity)
    sig = signal.copy()

    # Resolve severity to (n_channels_affected, window_width)
    n_channels = int(np.round(1 + 2 * (severity - 0.2) / 0.8))   # 1..3
    n_channels = max(1, min(3, n_channels))
    window_frac = 0.10 + 0.40 * (severity - 0.2) / 0.8           # 0.10..0.50
    window_width = max(1, int(np.round(window_frac * N_TIMESTEPS)))

    # Select channels (without replacement) and start position
    channel_idxs = rng.choice(N_CHANNELS, size=n_channels, replace=False)
    max_start = N_TIMESTEPS - window_width
    start = int(rng.integers(0, max_start + 1))
    end = start + window_width

    # Zero out the window, but leave residual noise so the dropout isn't
    # a hard discontinuity (an exact-zero block is a giveaway).
    for ch in channel_idxs:
        pre_std = float(np.std(sig[:start, ch])) if start > 0 else float(np.std(sig[:, ch]))
        if pre_std == 0:
            pre_std = 1e-6
        residual_noise = rng.standard_normal(window_width) * (0.03 * pre_std)
        sig[start:end, ch] = residual_noise

    return sig, time_norm.copy()


# =============================================================================
# Anomaly 2: calibration_drift
# =============================================================================
def inject_calibration_drift(
    signal: np.ndarray,
    time_norm: np.ndarray,
    severity: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth multiplicative gain drift applied uniformly to all 6 channels.

    Models amplifier gain drift or supply voltage shift — system-wide,
    not channel-specific. The drift profile is a low-order polynomial
    (linear or weak quadratic) so the perturbation is visible but doesn't
    look like a sudden glitch.

    Severity mapping:
        severity 0.2 → ±5% end-to-end drift
        severity 1.0 → ±25% end-to-end drift

    Direction (positive or negative) randomized per sample.
    """
    _check_inputs(signal, time_norm, severity)

    # End-to-end magnitude scales with severity
    max_drift = 0.05 + 0.20 * (severity - 0.2) / 0.8     # 0.05..0.25
    direction = 1.0 if rng.random() < 0.5 else -1.0
    end_shift = direction * max_drift

    # Drift profile: linear by default, with weak quadratic curvature
    # to break perfect linearity (more physically realistic).
    t = np.linspace(0, 1, N_TIMESTEPS)
    # Curvature in [-0.3, +0.3] of the linear component
    curvature = float(rng.uniform(-0.3, 0.3)) * end_shift
    gain = 1.0 + end_shift * t + curvature * (t * t - t)
    # gain at t=0 is 1.0; gain at t=1 is 1 + end_shift (linear part dominates)

    sig = signal * gain[:, None]   # apply to every channel
    return sig, time_norm.copy()


# =============================================================================
# Anomaly 3: temporal_warp
# =============================================================================
def inject_temporal_warp(
    signal: np.ndarray,
    time_norm: np.ndarray,
    severity: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Non-uniform timebase: resample signal at jittered timestamps.

    Models a sensor with a perturbed sampling clock. Each timestep's
    nominal sampling time is shifted by a small random amount, then
    monotonicity is enforced by sorting. The signal is then resampled
    via linear interpolation at the perturbed timepoints — so the
    signal values DO change relative to the clean parent.

    Implementation:
      1. Build jittered_time = time_norm + small_perturbation
      2. Sort middle values to enforce monotonicity (endpoints fixed)
      3. For each channel, resample the signal:
           signal_jittered[t] = interp(jittered_time[t], time_norm, signal[:, ch])
         This treats jittered_time as the "true clock" at which the sensor
         fired, and reads the corresponding value from the original signal.

    The returned time_norm is the ORIGINAL uniform axis [0, 1, ..., 100]; the
    jitter manifests purely in the signal values. This is the right design for
    downstream anomaly detection: any model looking at the signal channels can
    detect the perturbation without needing to also look at time_norm.

    Severity mapping:
        severity 0.2 → each timestep perturbed by up to ±2% of nominal spacing
        severity 1.0 → up to ±15% of nominal spacing
    """
    _check_inputs(signal, time_norm, severity)

    max_perturbation = 0.02 + 0.13 * (severity - 0.2) / 0.8   # 0.02..0.15
    # Nominal spacing between timesteps is 1/(T-1) ≈ 0.01
    nominal_dt = 1.0 / (N_TIMESTEPS - 1)
    perturbation = rng.uniform(
        -max_perturbation * nominal_dt,
        +max_perturbation * nominal_dt,
        size=N_TIMESTEPS,
    )

    # Build the jittered time axis, with endpoints pinned and middle sorted.
    perturbed = time_norm + perturbation
    perturbed[0] = 0.0
    perturbed[-1] = 1.0
    middle = np.sort(perturbed[1:-1])
    jittered_time = np.concatenate([[0.0], middle, [1.0]])
    jittered_time = np.clip(jittered_time, 0.0, 1.0)

    # Resample each channel at the jittered timepoints.
    # The original signal is sampled on time_norm; we read its value at
    # jittered_time[t] for each output timestep t.
    new_signal = np.zeros_like(signal)
    for ch in range(signal.shape[1]):
        new_signal[:, ch] = np.interp(jittered_time, time_norm, signal[:, ch])

    # Return the original uniform time axis; the perturbation lives in the
    # signal values (which is where downstream anomaly detectors look).
    return new_signal, time_norm.copy()


# =============================================================================
# Anomaly 4: periodic_interference
# =============================================================================
def inject_periodic_interference(
    signal: np.ndarray,
    time_norm: np.ndarray,
    severity: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Additive sinusoidal contamination on selected channels.

    Models electrical interference (mains hum, motor noise, RF pickup).
    Amplitude scales with per-channel std so high-amplitude channels get
    proportionally larger interference. Frequency is sampled within a
    physically plausible band (2-10 cycles across the sequence).

    Severity mapping:
        severity 0.2 → amplitude 10% of channel std, 1 frequency, 1-2 channels
        severity 1.0 → amplitude 40% of channel std, possibly 2 frequencies, all 6 channels
    """
    _check_inputs(signal, time_norm, severity)
    sig = signal.copy()

    # Resolve severity
    amp_frac = 0.10 + 0.30 * (severity - 0.2) / 0.8       # 0.10..0.40
    n_channels_affected = int(np.round(1 + 5 * (severity - 0.2) / 0.8))  # 1..6
    n_channels_affected = max(1, min(N_CHANNELS, n_channels_affected))
    # Two frequencies only at very high severity
    n_freqs = 2 if severity > 0.7 else 1

    channel_idxs = rng.choice(N_CHANNELS, size=n_channels_affected, replace=False)
    # Use the perturbed time axis if provided (jitter+interference combo not allowed
    # per schema, but be safe). time_norm should already be roughly [0, 1].
    t = time_norm

    for ch in channel_idxs:
        ch_std = float(np.std(sig[:, ch]))
        if ch_std == 0:
            ch_std = 1e-6
        amp = amp_frac * ch_std

        interference = np.zeros(N_TIMESTEPS)
        for _ in range(n_freqs):
            # Frequency in cycles per unit time (2 to 10 cycles across [0,1])
            freq_cycles = float(rng.uniform(2.0, 10.0))
            phase = float(rng.uniform(0, 2 * np.pi))
            interference += amp * np.sin(2 * np.pi * freq_cycles * t + phase)

        sig[:, ch] = sig[:, ch] + interference

    return sig, time_norm.copy()


# =============================================================================
# Dispatch table
# =============================================================================
INJECTOR_BY_SUBTYPE = {
    "sensor_dropout":          inject_sensor_dropout,
    "calibration_drift":       inject_calibration_drift,
    "temporal_warp":           inject_temporal_warp,
    "periodic_interference":   inject_periodic_interference,
}


def inject_anomaly(
    subtype: str,
    signal: np.ndarray,
    time_norm: np.ndarray,
    severity: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to the appropriate injector for the given subtype.

    Args:
        subtype: one of {sensor_dropout, calibration_drift, temporal_warp,
                          periodic_interference}.
        signal: clean signal of shape (N_TIMESTEPS, N_CHANNELS).
        time_norm: clean time axis of shape (N_TIMESTEPS,).
        severity: value in [0.2, 1.0].
        rng: numpy Generator, seeded deterministically by the caller.

    Returns:
        (perturbed_signal, perturbed_time_norm).

    Raises:
        ValueError: for unknown subtypes or out-of-range severity.
    """
    if subtype not in INJECTOR_BY_SUBTYPE:
        raise ValueError(
            f"Unknown anomaly subtype: {subtype!r}. "
            f"Valid: {sorted(INJECTOR_BY_SUBTYPE.keys())}"
        )
    return INJECTOR_BY_SUBTYPE[subtype](signal, time_norm, severity, rng)


# =============================================================================
# Helpers
# =============================================================================
def _check_inputs(
    signal: np.ndarray,
    time_norm: np.ndarray,
    severity: float,
) -> None:
    if signal.shape != (N_TIMESTEPS, N_CHANNELS):
        raise ValueError(
            f"signal must have shape ({N_TIMESTEPS}, {N_CHANNELS}); got {signal.shape}"
        )
    if time_norm.shape != (N_TIMESTEPS,):
        raise ValueError(
            f"time_norm must have shape ({N_TIMESTEPS},); got {time_norm.shape}"
        )
    if not (0.2 <= severity <= 1.0):
        raise ValueError(f"severity must be in [0.2, 1.0]; got {severity}")
