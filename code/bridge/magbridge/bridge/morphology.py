"""
OSF magnetic morphology bank.

Holds per-voltage statistics extracted from the OSF archive:
  - mean_traj: (T=100, C=6) mean trajectory per voltage anchor
  - std_traj: (T, C) per-step std
  - corr_mean, corr_std: cross-channel correlation matrices
  - psd, envelope, amplitude_per_pos: spectral and amplitude descriptors

The MorphologyBank provides:
  - `anchor_mean(v)`: exact OSF anchor mean trajectory at v
  - `sample_base_morphology(v, rng)`: jittered mean + voltage interpolation
    (linear interpolation between OSF anchors for voltages between them)
  - `_bracket(v)`: finds the two surrounding OSF anchors for any v

Regime classification:
  - GROUNDED: voltage in [3.06, 3.34] V (within OSF anchor span)
  - EXTRAPOLATION: voltage in [2.54, 3.06) V (low-voltage regime-B)
  - UNSUPPORTED: voltage outside [2.54, 3.34] V (rejected)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union

import numpy as np


# =============================================================================
# OSF anchor constants
# =============================================================================
OSF_ANCHORS: tuple[float, ...] = (2.54, 2.81, 3.00, 3.10, 3.34)
GROUNDED_LOW: float = 3.06       # voltages >= this are GROUNDED
GROUNDED_HIGH: float = 3.34      # max supported voltage
EXTRAPOLATION_LOW: float = 2.54  # min supported voltage


# =============================================================================
# Regime classification
# =============================================================================
class Regime(str, Enum):
    GROUNDED = "grounded"
    EXTRAPOLATION = "extrapolation"
    UNSUPPORTED = "unsupported"


def classify_regime(voltage: float) -> tuple[Regime, float]:
    """Classify operating voltage and return (regime, nearest OSF anchor).

    UNSUPPORTED voltages (outside [2.54, 3.34]) should raise an error in the
    caller; the bridge does not generate samples for unsupported voltages.

    Args:
        voltage: operating voltage in V.

    Returns:
        (regime, nearest_anchor_voltage)
    """
    if not (EXTRAPOLATION_LOW <= voltage <= GROUNDED_HIGH):
        r = Regime.UNSUPPORTED
    elif GROUNDED_LOW <= voltage <= GROUNDED_HIGH:
        r = Regime.GROUNDED
    else:
        r = Regime.EXTRAPOLATION
    nearest = min(OSF_ANCHORS, key=lambda a: abs(a - voltage))
    return r, nearest


# =============================================================================
# Morphology bank
# =============================================================================
class MorphologyBank:
    """Loads OSF anchor statistics from anchor_stats.npz and provides
    voltage-interpolated base morphologies."""

    def __init__(self, anchor_stats_path: Union[str, Path]):
        self._npz = np.load(anchor_stats_path)
        self._voltages = sorted(OSF_ANCHORS)
        self._cache: dict = {}

    def _load(self, v: float) -> dict:
        """Load and cache statistics for OSF anchor voltage v."""
        if v in self._cache:
            return self._cache[v]
        tag = f"v{v:.2f}".replace(".", "_")
        s = {
            "voltage": v,
            "mean_traj": self._npz[f"{tag}__mean_traj"],
            "std_traj": self._npz[f"{tag}__std_traj"],
            "corr_mean": self._npz[f"{tag}__corr_mean"],
        }
        self._cache[v] = s
        return s

    @property
    def shape(self) -> tuple[int, int]:
        """(T, C) shape of stored morphology trajectories."""
        return self._load(self._voltages[0])["mean_traj"].shape

    def anchor_mean(self, v: float) -> np.ndarray:
        """Exact mean trajectory at OSF anchor voltage v."""
        return self._load(v)["mean_traj"].copy()

    def _bracket(self, v: float) -> tuple[float, float, float]:
        """Find the two surrounding OSF anchors and the interpolation weight.

        Returns:
            (v_low, v_high, alpha) where alpha=0 means use v_low alone,
            alpha=1 means use v_high alone, intermediate means linear
            interpolation. For v outside the OSF range, alpha=0 and
            v_low=v_high=nearest endpoint.
        """
        if v <= self._voltages[0]:
            return self._voltages[0], self._voltages[0], 0.0
        if v >= self._voltages[-1]:
            return self._voltages[-1], self._voltages[-1], 0.0
        for i in range(len(self._voltages) - 1):
            v_low = self._voltages[i]
            v_high = self._voltages[i + 1]
            if v_low <= v <= v_high:
                a = 0.0 if v_high == v_low else (v - v_low) / (v_high - v_low)
                return v_low, v_high, a
        raise RuntimeError(f"bracket failed for {v}")

    def sample_base_morphology(
        self,
        voltage: float,
        rng: np.random.Generator,
        within_pos_jitter: float = 1.0,
    ) -> np.ndarray:
        """Sample a base morphology for given voltage with within-position jitter.

        For voltages at OSF anchors: returns mean_traj + Gaussian noise scaled
        by std_traj.

        For voltages between anchors: linearly interpolates the mean across the
        bracketing anchors, mixes the variances, and adds noise.

        Args:
            voltage: target operating voltage.
            rng: numpy Generator for the noise.
            within_pos_jitter: scale factor on the std-driven jitter (1.0 = default).

        Returns:
            (T, C) base morphology trajectory.
        """
        v_low, v_high, alpha = self._bracket(voltage)
        s_low = self._load(v_low)
        s_high = self._load(v_high) if v_high != v_low else s_low

        if v_high == v_low:
            mean = s_low["mean_traj"].copy()
            std = s_low["std_traj"].copy()
        else:
            mean = (1 - alpha) * s_low["mean_traj"] + alpha * s_high["mean_traj"]
            var_mix = (1 - alpha) * s_low["std_traj"] ** 2 + alpha * s_high["std_traj"] ** 2
            std = np.sqrt(var_mix)

        return mean + rng.standard_normal(mean.shape) * std * within_pos_jitter
