"""
MagBridge-Battery — deterministic sample ID generation.

Sample IDs combine a human-readable prefix with an 8-character hash of the
sample's identity fields. The hash ensures uniqueness; the prefix makes IDs
scannable for humans.

Format by sample category:
    Clean grounded:      lfp_<cell>_soc<NN>_<hash8>
    Synthetic anomaly:   lfp_<cell>_soc<NN>_<subtype>_<hash8>
    Regime-B:            lfp_regimeb_v<VVV>_<hash8>

Hash algorithm: SHA-256 of canonicalised JSON, truncated to 8 hex chars.
This gives ~4e9 possible hash values per prefix. With 6,760 samples and
typically <200 samples sharing any prefix, collision probability is
negligible. The validator additionally enforces zero collisions across
the full dataset.

Determinism contract:
    For two samples with identical (cell_id, soc, voltage, soh,
    generation_seed, anomaly_subtype, anomaly_severity, parent_sample_id,
    bridge_version, bridge_config_hash), make_sample_id() MUST return the
    same string. Any change to one of these inputs MUST change the hash.

The signal arrays are deliberately NOT in the hash input. Two samples
generated from the same identity inputs but with different signal arrays
indicate a bug somewhere — they'd get the same ID and the validator
would catch the inconsistency at the parquet layer.
"""

from __future__ import annotations

import hashlib
import json
from typing import Optional, Union

from magbridge.schema import (
    AnomalyOrigin,
    AnomalySubtype,
)


# =============================================================================
# Subtype abbreviations for IDs
# =============================================================================
SUBTYPE_ABBREVIATIONS: dict[AnomalySubtype, str] = {
    AnomalySubtype.SENSOR_DROPOUT: "dropout",
    AnomalySubtype.CALIBRATION_DRIFT: "caldrift",
    AnomalySubtype.TEMPORAL_WARP: "twarp",
    AnomalySubtype.PERIODIC_INTERFERENCE: "interf",
}

# Reverse map for parsing IDs back into subtypes (validator use)
ABBREVIATION_TO_SUBTYPE: dict[str, AnomalySubtype] = {
    v: k for k, v in SUBTYPE_ABBREVIATIONS.items()
}

# Hash length (hex chars)
HASH_LENGTH = 8


# =============================================================================
# Identity field assembly
# =============================================================================
def _identity_fields(
    cell_id: str,
    voltage: float,
    soc: float,
    soh: Optional[float],
    generation_seed: int,
    anomaly_subtype: AnomalySubtype,
    anomaly_origin: AnomalyOrigin,
    anomaly_severity: float,
    parent_sample_id: Optional[str],
    bridge_version: str,
    bridge_config_hash: str,
) -> dict:
    """Assemble the canonical identity dict used for hashing.

    Order is fixed; values are normalised to types that JSON-serialise
    deterministically (no floats with surprising rounding).
    """
    return {
        "cell_id": cell_id,
        # Voltage and SOC rounded to fixed precision to avoid float fragility.
        # 4 decimals is more than enough for any operating voltage.
        "voltage": round(float(voltage), 4),
        "soc": round(float(soc), 4),
        "soh": None if soh is None else round(float(soh), 6),
        "generation_seed": int(generation_seed),
        "anomaly_subtype": anomaly_subtype.value,
        "anomaly_origin": anomaly_origin.value,
        "anomaly_severity": round(float(anomaly_severity), 6),
        "parent_sample_id": parent_sample_id,
        "bridge_version": bridge_version,
        "bridge_config_hash": bridge_config_hash,
    }


def _hash_identity(identity: dict) -> str:
    """Hash an identity dict to an 8-char hex string."""
    # sort_keys=True ensures field ordering doesn't affect the hash.
    blob = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:HASH_LENGTH]


# =============================================================================
# Public API: make_sample_id
# =============================================================================
def make_sample_id(
    *,
    cell_id: str,
    voltage: float,
    soc: float,
    soh: Optional[float],
    generation_seed: int,
    anomaly_subtype: Union[AnomalySubtype, str],
    anomaly_origin: Union[AnomalyOrigin, str],
    anomaly_severity: float,
    parent_sample_id: Optional[str],
    bridge_version: str,
    bridge_config_hash: str,
) -> str:
    """Generate a deterministic, human-readable sample ID.

    Format dispatched by anomaly_origin:
        BRIDGE_EXTRAPOLATION       -> lfp_regimeb_v<VVV>_<hash8>
        SYNTHETIC_SENSOR_PERTURBATION -> lfp_<cell>_soc<NN>_<subtype>_<hash8>
        NONE (clean grounded)      -> lfp_<cell>_soc<NN>_<hash8>

    Raises:
        ValueError: if anomaly_origin is unknown or if inputs are inconsistent
                    (e.g. synthetic origin without parent_sample_id).
    """
    # Normalise enums (accept either Enum instances or their string values)
    if isinstance(anomaly_subtype, str):
        anomaly_subtype = AnomalySubtype(anomaly_subtype)
    if isinstance(anomaly_origin, str):
        anomaly_origin = AnomalyOrigin(anomaly_origin)

    identity = _identity_fields(
        cell_id=cell_id,
        voltage=voltage,
        soc=soc,
        soh=soh,
        generation_seed=generation_seed,
        anomaly_subtype=anomaly_subtype,
        anomaly_origin=anomaly_origin,
        anomaly_severity=anomaly_severity,
        parent_sample_id=parent_sample_id,
        bridge_version=bridge_version,
        bridge_config_hash=bridge_config_hash,
    )
    h = _hash_identity(identity)

    # Dispatch by origin
    if anomaly_origin == AnomalyOrigin.BRIDGE_EXTRAPOLATION:
        # Regime-B: voltage in the prefix, no cell_id (extrapolation samples
        # aren't tied to a specific PulseBat cell in a meaningful way).
        v_int = int(round(voltage * 100))  # 2.81 -> 281
        return f"lfp_regimeb_v{v_int:03d}_{h}"

    if anomaly_origin == AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION:
        if parent_sample_id is None:
            raise ValueError(
                "synthetic_sensor_perturbation requires parent_sample_id "
                "(synthetic anomalies must be derived from a clean parent)"
            )
        if anomaly_subtype not in SUBTYPE_ABBREVIATIONS:
            raise ValueError(
                f"synthetic anomaly subtype {anomaly_subtype.value} has no "
                f"abbreviation; check SUBTYPE_ABBREVIATIONS"
            )
        abbr = SUBTYPE_ABBREVIATIONS[anomaly_subtype]
        soc_int = int(round(soc))
        return f"lfp_{cell_id}_soc{soc_int:02d}_{abbr}_{h}"

    if anomaly_origin == AnomalyOrigin.NONE:
        if parent_sample_id is not None:
            raise ValueError(
                "clean sample (origin=NONE) must not have parent_sample_id"
            )
        if anomaly_subtype != AnomalySubtype.NONE:
            raise ValueError(
                f"clean sample must have anomaly_subtype=NONE, got {anomaly_subtype.value}"
            )
        soc_int = int(round(soc))
        return f"lfp_{cell_id}_soc{soc_int:02d}_{h}"

    raise ValueError(f"unknown anomaly_origin: {anomaly_origin}")


# =============================================================================
# Helpers for the generator and validator
# =============================================================================
def derive_per_sample_seed(master_seed: int, *path_components: Union[str, int]) -> int:
    """Derive a deterministic per-sample seed from the master seed.

    Used by the generator to make every sample's randomness reproducible
    from (master_seed, sample_path). Different samples get different seeds
    via the path components.

    Returns a 32-bit non-negative integer suitable for numpy default_rng.
    """
    h = hashlib.sha256(
        f"{master_seed}::{'/'.join(str(c) for c in path_components)}".encode("utf-8")
    ).hexdigest()
    # Use first 8 hex chars = 32 bits, masked to 31 bits (non-negative int32)
    return int(h[:8], 16) & 0x7FFFFFFF


def parse_sample_id(sample_id: str) -> dict:
    """Parse a sample ID back into its components.

    Useful for the validator and for users who want to know what a sample is
    without loading metadata. Returns a dict with at least:
        category: "clean" | "synthetic_anomaly" | "regime_b"
        hash: the 8-char hash
    Plus category-specific fields (cell_id, soc, subtype, voltage_centivolts).

    Raises ValueError if the ID does not match any expected format.
    """
    parts = sample_id.split("_")
    if len(parts) < 4 or parts[0] != "lfp":
        raise ValueError(f"sample_id must start with 'lfp_': {sample_id}")

    # Regime-B: lfp_regimeb_v<VVV>_<hash>
    if parts[1] == "regimeb":
        if len(parts) != 4:
            raise ValueError(f"regime-B ID must have 4 underscore-segments: {sample_id}")
        v_part = parts[2]
        if not (v_part.startswith("v") and v_part[1:].isdigit()):
            raise ValueError(f"regime-B voltage component malformed: {sample_id}")
        return {
            "category": "regime_b",
            "voltage_centivolts": int(v_part[1:]),
            "hash": parts[3],
        }

    # Grounded clean or synthetic anomaly: lfp_<cell>_soc<NN>_[<subtype>_]<hash>
    cell_id = parts[1]
    soc_part = parts[2]
    if not (soc_part.startswith("soc") and soc_part[3:].isdigit()):
        raise ValueError(f"SOC component malformed: {sample_id}")
    soc = int(soc_part[3:])

    if len(parts) == 4:
        # Clean: lfp_C042_soc30_a8f3e2d1
        return {
            "category": "clean",
            "cell_id": cell_id,
            "soc": soc,
            "hash": parts[3],
        }
    elif len(parts) == 5:
        # Synthetic anomaly: lfp_C042_soc30_dropout_b9f4d3e2
        abbr = parts[3]
        if abbr not in ABBREVIATION_TO_SUBTYPE:
            raise ValueError(
                f"unknown subtype abbreviation '{abbr}' in {sample_id}"
            )
        return {
            "category": "synthetic_anomaly",
            "cell_id": cell_id,
            "soc": soc,
            "subtype": ABBREVIATION_TO_SUBTYPE[abbr],
            "hash": parts[4],
        }
    else:
        raise ValueError(f"sample_id has unexpected structure: {sample_id}")


def check_no_collisions(sample_ids: list[str]) -> None:
    """Raise if any sample_id appears more than once. Validator helper."""
    seen = {}
    for sid in sample_ids:
        if sid in seen:
            raise ValueError(
                f"sample_id collision: '{sid}' appears at least twice. "
                f"This indicates either (a) a hash collision (vanishingly rare) "
                f"or (b) two samples with identical identity fields (a bug). "
                f"Re-examine the generator's sample identity inputs."
            )
        seen[sid] = True
