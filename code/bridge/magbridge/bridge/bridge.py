"""
BridgeV13 — the v1.3 magnetic-signature bridge.

This is the validated bridge used to produce the MagBridge-Battery v1.0
dataset. It implements the architecture documented in:

    bridge(voltage, soc, soh, u_features, seed) ->
        morphology -> QRec embed -> LDA project -> perturb by SOH-delta
        in d_state direction -> cone-filtered top-k decode -> blend with
        base + amplitude + spectral + sensor + SOC-fluctuation noise.

Differences from v1.2:
    v1.3 adds a CONE FILTER to the top-k decoder. After the SOH-delta
    perturbation pushes lda_coords by `magnitude * direction`, the
    candidate set is restricted to OSF samples whose displacement from
    the unperturbed coords roughly aligns with the perturbation
    direction. This makes the decoder direction-sensitive in cases where
    cone_half_angle_deg < 90 and the perturbation is large enough to
    matter (>= cone_disable_below_delta in SOH-delta units).

    For small SOH deltas, the cone filter is bypassed (all OSF samples
    eligible), since direction is meaningless when magnitude is near zero.

    The cone is parameterized by:
        cone_half_angle_deg     opening angle in 4-D LDA space
        cone_min_candidates     fall back if filtering leaves too few
        cone_disable_below_delta SOH-delta threshold below which to skip cone

Reproducibility:
    Bridge generation is deterministic given (voltage, soc, soh, seed,
    u_features, all fitted artifacts). The seed drives the RNG used for
    morphology jitter, amplitude scaling, spectral noise, sensor noise,
    and SOC fluctuation noise.

    The fitted artifacts (anchor_stats.npz, lda_fit.npz, qrec_embeddings.npz,
    osf_sequences.npz) are version-locked v1.0 inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Union

import numpy as np

from magbridge.bridge.morphology import (
    MorphologyBank,
    Regime,
    classify_regime,
)
from magbridge.bridge.qrec import (
    QREC_DEFAULTS,
    POOLING_DEFAULT,
    QuantumRecurrentReservoir10q,
    angle_scale_sequences,
    pool_reservoir_states,
)


# =============================================================================
# Bridge config (mirrors generation_config.yaml -> bridge section)
# =============================================================================
@dataclass
class BridgeConfig:
    """v1.3 bridge configuration.

    Default values match the LOCKED v1.0 release config. Do not change
    these without invalidating the generated dataset (use a new release
    version instead).
    """
    # SOH degradation lever
    soh_baseline: float = 1.0
    qrec_drift_strength: float = 800.0    # in LDA-space units (v1.0 locked = 800)

    # Decoder
    decode_k: int = 8
    decode_kernel_sigma: float = 50.0     # softmin temperature in LDA-space units

    # v1.3 cone-decoder extension
    cone_half_angle_deg: float = 75.0      # max angle to perturbation direction
    cone_min_candidates: int = 8           # fall back if filter leaves fewer
    cone_disable_below_delta: float = 0.02 # disable for SOH-delta below this

    # Per-channel noise stack
    amplitude_strength: float = 0.30       # multiplicative amplitude jitter
    spectral_strength: float = 0.05        # broadband noise scale
    sensor_noise_fraction: float = 0.05    # fraction of channel std
    soc_fluctuation_strength: float = 0.04 # low-frequency SOC noise

    # SOC reference
    soc_rest: float = 50.0
    soc_range: float = 50.0


# =============================================================================
# BridgeV13
# =============================================================================
class BridgeV13:
    """Bridge v1.3 — degradation in 4-D LDA space with cone-filtered top-k decode.

    Usage:
        bridge = BridgeV13(
            anchor_stats_path="data/v1.0/anchor_stats.npz",
            osf_seq_path="data/v1.0/osf_sequences.npz",
            osf_qrec_emb_path="data/v1.0/qrec_embeddings.npz",
            lda_fit_path="data/v1.0/lda_fit.npz",
        )
        signal, regime, nearest_anchor = bridge.generate(
            voltage=3.10, soc=30.0, soh=0.85, seed=42, u_features=u_vec
        )
        # signal shape: (T=100, C=6)
    """

    def __init__(
        self,
        anchor_stats_path: Union[str, Path],
        osf_seq_path: Union[str, Path],
        osf_qrec_emb_path: Union[str, Path],
        lda_fit_path: Union[str, Path],
        config: Optional[BridgeConfig] = None,
        qrec_kwargs: Optional[dict] = None,
    ):
        self.config = config or BridgeConfig()

        # Load morphology bank
        self.morphology = MorphologyBank(anchor_stats_path)

        # Load LDA fit (171-D -> 4-D projection)
        lda = np.load(lda_fit_path)
        self.scaler_mean = lda["scaler_mean"]
        self.scaler_scale = lda["scaler_scale"]
        self.lda_components = lda["lda_components"][:, :4]   # (171, 4)
        self.lda_xbar = lda["lda_xbar"]                       # (171,)
        self.d_state_lda_unit = lda["d_state_lda_unit"]       # (4,) unit vector

        # Load real OSF sequences and pre-computed QRec embeddings
        self.osf_seqs = np.load(osf_seq_path)["X_seq"]                       # (205, 100, 6)
        self.osf_emb = np.load(osf_qrec_emb_path)["embeddings"]              # (205, 171)

        # Pre-project OSF embeddings into LDA space (cached for decoding)
        emb_std = (self.osf_emb - self.scaler_mean) / (self.scaler_scale + 1e-12)
        self.osf_lda = (emb_std - self.lda_xbar) @ self.lda_components       # (205, 4)

        # Fit the global QRec scaler on real OSF sequences (used for all embeddings)
        _, self._global_scaler = angle_scale_sequences(self.osf_seqs)

        # Build the QRec reservoir
        qrec_args = dict(QREC_DEFAULTS)
        if qrec_kwargs:
            qrec_args.update(qrec_kwargs)
        self.qrec = QuantumRecurrentReservoir10q(**qrec_args)

        # Last-generation diagnostics (for inspection / debugging)
        self.last_diagnostics: dict = {}

    # ---------------------------------------------------------------- internals
    def _embed_171d(self, sequence: np.ndarray) -> np.ndarray:
        """Embed a (T, C) sequence into the 171-D QRec pooled space."""
        X_scaled, _ = angle_scale_sequences(sequence[None], scaler=self._global_scaler)
        states = self.qrec.process_sequence(X_scaled[0])
        return pool_reservoir_states(states, pooling=POOLING_DEFAULT)

    def _project_to_lda(self, emb_171d: np.ndarray) -> np.ndarray:
        """Project a 171-D QRec embedding into the 4-D LDA subspace."""
        std = (emb_171d - self.scaler_mean) / (self.scaler_scale + 1e-12)
        return (std - self.lda_xbar) @ self.lda_components

    def _perturb_lda(
        self,
        lda_coords: np.ndarray,
        soh: float,
        direction: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Apply SOH-driven LDA-space perturbation.

        Magnitude is proportional to (soh_baseline - soh); direction is
        the d_state (or override). Negative sign means lower SOH pushes
        AGAINST the d_state direction (which we defined as "ageing"
        direction at fit time, so against = more aged).
        """
        delta = self.config.soh_baseline - float(soh)
        magnitude = -self.config.qrec_drift_strength * delta
        return lda_coords + magnitude * direction, magnitude

    def _cone_filter(
        self,
        perturbed_lda: np.ndarray,
        unperturbed_lda: np.ndarray,
        direction: np.ndarray,
        soh_delta: float,
    ) -> np.ndarray:
        """v1.3: restrict candidate set to OSF samples within a cone around
        the perturbation direction.

        Args:
            perturbed_lda: (4,) target LDA coordinates after SOH perturbation
            unperturbed_lda: (4,) starting LDA coordinates (for displacement)
            direction: (4,) unit perturbation direction
            soh_delta: |soh_baseline - soh|; small values bypass the filter

        Returns:
            Indices (into self.osf_lda) of candidate OSF samples after filtering.
            Falls back to all indices if too few pass the cone filter, OR
            if soh_delta < cone_disable_below_delta.
        """
        cfg = self.config
        n_total = len(self.osf_lda)

        # For tiny SOH deltas, direction is meaningless; skip the cone.
        if soh_delta < cfg.cone_disable_below_delta:
            return np.arange(n_total)

        # Displacement vectors from unperturbed -> each OSF sample
        disp = self.osf_lda - unperturbed_lda[None, :]                    # (205, 4)
        disp_norm = np.linalg.norm(disp, axis=1) + 1e-12

        # Project onto perturbation direction (unit vector)
        dir_unit = direction / (np.linalg.norm(direction) + 1e-12)
        cos_theta = (disp @ dir_unit) / disp_norm                          # (205,)

        # Cone threshold: keep candidates with cos_theta >= cos(half_angle)
        cos_threshold = np.cos(np.deg2rad(cfg.cone_half_angle_deg))
        mask = cos_theta >= cos_threshold
        n_pass = int(mask.sum())

        if n_pass < cfg.cone_min_candidates:
            # Not enough candidates aligned; fall back to all OSF samples.
            return np.arange(n_total)
        return np.where(mask)[0]

    def _decode_lda(
        self,
        perturbed_lda: np.ndarray,
        candidate_idx: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Softmin decoder over OSF samples restricted to candidate_idx.

        Returns:
            (decoded_signal[T,C], topk_idx_into_full_osf, topk_weights)
        """
        cfg = self.config
        # Distances from perturbed point to each candidate OSF sample
        cand_lda = self.osf_lda[candidate_idx]
        d = np.linalg.norm(cand_lda - perturbed_lda[None, :], axis=1)
        k = min(cfg.decode_k, len(d))
        topk_local = np.argsort(d)[:k]
        topk_idx = candidate_idx[topk_local]
        d_topk = d[topk_local]
        # Softmin weighting: closer samples get more weight
        w = np.exp(-d_topk / (cfg.decode_kernel_sigma + 1e-12))
        w = w / (w.sum() + 1e-12)
        decoded = np.einsum("k,ktc->tc", w, self.osf_seqs[topk_idx])
        return decoded, topk_idx, w

    # ----------------------------------------------------------------- generate
    def generate(
        self,
        voltage: float,
        soc: float,
        soh: float,
        seed: int = 42,
        direction_override: Optional[np.ndarray] = None,
        u_features: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, Regime, float]:
        """Generate one synthetic magnetic signature for (voltage, soc, soh).

        Args:
            voltage: operating voltage in V (must be in [2.54, 3.34]).
            soc: state of charge in %.
            soh: state of health in [0, 1].
            seed: RNG seed for the per-sample randomness.
            direction_override: optional 4-D LDA-space direction to use
                instead of d_state (used in ablation studies; A0 uses None).
            u_features: optional PulseBat U1-U21 features (currently unused
                in v1.3 generation; reserved for future cross-conditioning).

        Returns:
            (signal[T=100, C=6], regime, nearest_anchor_voltage)

        Raises:
            ValueError: for unsupported voltages.
        """
        regime, nearest = classify_regime(voltage)
        if regime == Regime.UNSUPPORTED:
            raise ValueError(
                f"voltage {voltage} is outside the supported OSF range [2.54, 3.34]"
            )
        rng = np.random.default_rng(seed)
        cfg = self.config

        # 1. Sample base morphology at the requested voltage
        base = self.morphology.sample_base_morphology(voltage, rng)

        # 2. Embed via QRec -> 171-D pooled vector
        base_emb = self._embed_171d(base)

        # 3. Project to 4-D LDA space
        lda_coords = self._project_to_lda(base_emb)

        # 4. SOH-driven perturbation (direction = d_state by default)
        direction = (direction_override if direction_override is not None
                     else self.d_state_lda_unit)
        perturbed, magnitude = self._perturb_lda(lda_coords, soh, direction)

        # 5. v1.3 cone filter on candidate set
        soh_delta = abs(cfg.soh_baseline - float(soh))
        candidate_idx = self._cone_filter(perturbed, lda_coords, direction, soh_delta)

        # 6. Decode: softmin-weighted blend over top-k filtered OSF samples
        decoded, topk_idx, topk_w = self._decode_lda(perturbed, candidate_idx)

        # 7. Blend base + decoded; blend factor scales with SOH delta
        delta_pos = max(0.0, cfg.soh_baseline - soh)  # positive part of SOH gap
        blend = float(min(1.0, delta_pos * 1.5))
        signal = (1.0 - blend) * base + blend * decoded

        # 8. Amplitude jitter scaled by SOH delta
        amp = 1.0 + cfg.amplitude_strength * delta_pos * rng.standard_normal(signal.shape[1]) * 0.5
        amp = np.clip(amp, 1.0 - cfg.amplitude_strength, 1.0 + cfg.amplitude_strength)
        signal = signal * amp[None, :]

        # 9. Spectral broadening (per-channel, scaled by SOH delta)
        per_ch_range = (signal.max(axis=0) - signal.min(axis=0)) + 1e-9
        sig_b = cfg.spectral_strength * delta_pos
        signal = signal + rng.standard_normal(signal.shape) * sig_b * per_ch_range[None, :]

        # 10. Per-channel sensor noise (always present)
        std_ref = signal.std(axis=0)
        sensor = rng.standard_normal(signal.shape) * (cfg.sensor_noise_fraction * std_ref[None, :])

        # 11. SOC-driven low-frequency fluctuation
        soc_d = abs(soc - cfg.soc_rest) / max(cfg.soc_range, 1e-9)
        soc_d = float(np.clip(soc_d, 0, 1))
        raw = rng.standard_normal(signal.shape)
        klen = max(5, signal.shape[0] // 10)
        kk = np.ones(klen) / klen
        lf = np.stack(
            [np.convolve(raw[:, c], kk, mode="same") for c in range(signal.shape[1])],
            axis=1,
        )
        lf -= lf.mean(axis=0, keepdims=True)
        lf /= (lf.std(axis=0, keepdims=True) + 1e-12)
        soc_n = lf * (cfg.soc_fluctuation_strength * soc_d) * per_ch_range[None, :]

        # 12. Combine and record diagnostics
        final_signal = signal + sensor + soc_n
        self.last_diagnostics = {
            "soh": soh,
            "soh_delta": soh_delta,
            "magnitude": magnitude,
            "lda_coords_before": lda_coords.tolist(),
            "lda_coords_after": perturbed.tolist(),
            "n_candidates_after_cone": len(candidate_idx),
            "topk_neighbours": topk_idx.tolist(),
            "topk_weights": topk_w.tolist(),
            "blend_factor": blend,
            "regime": regime.value,
            "nearest_anchor": nearest,
        }
        return final_signal, regime, nearest

    def config_dict(self) -> dict:
        """Return the bridge config as a plain dict (for hashing/manifest)."""
        return asdict(self.config)
