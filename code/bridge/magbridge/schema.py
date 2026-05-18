"""
MagBridge-Battery v1.0 schemas
==============================

Pydantic models defining the canonical structure of:
  - SampleRecord: one row in the dataset
  - SplitFile: train/val/test assignment file
  - DatasetManifest: top-level provenance and summary

These schemas are the single source of truth for:
  - what fields exist in each sample
  - which values are valid for categorical fields
  - what invariants must hold across fields

The validator module (validator.py) uses these schemas to enforce the
9 integrity rules specified in the engineering decisions doc.

DO NOT add fields casually. Every change is a schema-version bump.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


# =============================================================================
# Constants (must match generation_config.yaml)
# =============================================================================
SIGNAL_N_TIMESTEPS = 100
SIGNAL_N_CHANNELS = 6
# Channel naming:
#   B_s1Y, B_s1Z: sensor 1, Y and Z components of the magnetic field (in nT,
#     centred near zero; sign convention from OSF source data)
#   B_s2Y, B_s2Z: sensor 2, same convention
#   B_s1C5, B_s2C6: the 5th and 6th channels in the OSF source data. These
#     were originally labelled "Mag" in OSF but may not be a magnitude in
#     the strict sense (sqrt(Y^2 + Z^2)) — values can be negative. We name
#     them "C5" and "C6" (channel-5, channel-6) to avoid implying physical
#     interpretation we cannot verify from the source.
SIGNAL_CHANNEL_NAMES = ["B_s1Y", "B_s1Z", "B_s2Y", "B_s2Z", "B_s1C5", "B_s2C6"]
SCHEMA_VERSION = "1.0"


# =============================================================================
# Enums for categorical fields
# =============================================================================
class Regime(str, Enum):
    GROUNDED = "grounded"
    EXTRAPOLATION = "extrapolation"


class AnomalySubtype(str, Enum):
    NONE = "none"
    SENSOR_DROPOUT = "sensor_dropout"
    CALIBRATION_DRIFT = "calibration_drift"
    TEMPORAL_WARP = "temporal_warp"
    PERIODIC_INTERFERENCE = "periodic_interference"
    LOW_VOLTAGE_REGIME_B = "low_voltage_regime_B"


class AnomalyOrigin(str, Enum):
    NONE = "none"
    SYNTHETIC_SENSOR_PERTURBATION = "synthetic_sensor_perturbation"
    BRIDGE_EXTRAPOLATION = "bridge_extrapolation"


class SecondLifeClass(str, Enum):
    REUSE = "reuse"
    RECONDITION = "recondition"


class Chemistry(str, Enum):
    LFP = "LFP"


# Synthetic-anomaly subtypes (used by validator to check origin pairing).
SYNTHETIC_SENSOR_SUBTYPES = {
    AnomalySubtype.SENSOR_DROPOUT,
    AnomalySubtype.CALIBRATION_DRIFT,
    AnomalySubtype.TEMPORAL_WARP,
    AnomalySubtype.PERIODIC_INTERFERENCE,
}


# =============================================================================
# Sample record schema
# =============================================================================
class SampleRecord(BaseModel):
    """One row of the MagBridge-Battery dataset."""

    # -- Identity and provenance -------------------------------------------
    sample_id: str = Field(..., description="Deterministic canonical sample ID")
    parent_sample_id: Optional[str] = Field(
        default=None,
        description="For paired anomalies: ID of the clean parent. None for clean and regime-B."
    )
    cell_id: str = Field(..., description="PulseBat cell ID (e.g., 'C042')")
    generation_seed: int = Field(..., description="RNG seed used for this specific sample")
    bridge_version: str = Field(..., description="Bridge architecture version, e.g. 'v1.3'")
    bridge_config_hash: str = Field(..., description="Short hash (8 chars) of bridge config used")
    schema_version: str = Field(default=SCHEMA_VERSION, description="Schema version, fixed at 1.0")

    # -- Conditioning inputs (what the bridge saw) -------------------------
    voltage: float = Field(..., ge=2.54, le=3.34, description="Operating voltage (V)")
    soc: float = Field(..., ge=0.0, le=100.0, description="State of charge (%)")
    soh: Optional[float] = Field(default=None, ge=0.0, le=1.0,
                                  description="State of health (0-1). None for regime-B.")
    chemistry: Chemistry = Field(default=Chemistry.LFP)
    u_features: Optional[list[float]] = Field(
        default=None,
        description="PulseBat U1-U21 pulse-response features. None for non-grounded samples."
    )

    # -- Regime and anomaly labels -----------------------------------------
    regime: Regime
    nearest_anchor: float = Field(..., description="Nearest OSF voltage anchor (V)")
    anomaly_flag: bool = Field(..., description="True iff ANY anomaly applied")
    anomaly_subtype: AnomalySubtype
    anomaly_origin: AnomalyOrigin
    anomaly_severity: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="0 for clean samples; uniform[0.2, 1.0] for synthetic anomalies; "
                    "1.0 for regime-B (categorical extrapolation, not graded)."
    )

    # -- Task labels -------------------------------------------------------
    second_life_class: Optional[SecondLifeClass] = Field(
        default=None,
        description="Reuse/recondition; None for samples without SOH (regime-B)."
    )

    # -- Signal data -------------------------------------------------------
    # Six fixed signal columns, each length SIGNAL_N_TIMESTEPS
    B_s1Y: list[float]
    B_s1Z: list[float]
    B_s2Y: list[float]
    B_s2Z: list[float]
    B_s1C5: list[float]
    B_s2C6: list[float]
    time_norm: list[float] = Field(
        ...,
        description="Length-100 normalised time axis, always [0, 1/99, ..., 1] (uniform). For temporal_warp anomalies, the perturbation manifests in the signal channels — time_norm itself stays uniform."
    )

    # -- Validators --------------------------------------------------------
    @field_validator("u_features")
    @classmethod
    def _check_u_features_length(cls, v):
        if v is not None and len(v) != 21:
            raise ValueError(f"u_features must have length 21 (got {len(v)})")
        return v

    @field_validator("B_s1Y", "B_s1Z", "B_s2Y", "B_s2Z", "B_s1C5", "B_s2C6")
    @classmethod
    def _check_signal_length(cls, v, info):
        if len(v) != SIGNAL_N_TIMESTEPS:
            raise ValueError(
                f"signal channel {info.field_name} must have length {SIGNAL_N_TIMESTEPS} "
                f"(got {len(v)})"
            )
        return v

    @field_validator("time_norm")
    @classmethod
    def _check_time_norm_length(cls, v):
        if len(v) != SIGNAL_N_TIMESTEPS:
            raise ValueError(f"time_norm must have length {SIGNAL_N_TIMESTEPS} (got {len(v)})")
        return v

    @field_validator("sample_id", "cell_id", "bridge_version", "bridge_config_hash")
    @classmethod
    def _no_whitespace(cls, v, info):
        if any(c.isspace() for c in v):
            raise ValueError(f"{info.field_name} must not contain whitespace")
        if not v:
            raise ValueError(f"{info.field_name} must be non-empty")
        return v

    # -- Cross-field invariants (model-level) ------------------------------
    @model_validator(mode="after")
    def _check_anomaly_consistency(self):
        """Enforces the integrity rules for anomaly fields and parent linkage.

        Rules enforced here (validator.py enforces dataset-level rules separately):
            (R1) anomaly_flag == True  iff  anomaly_subtype != NONE
            (R2) anomaly_flag == True  iff  anomaly_origin  != NONE
            (R3) low_voltage_regime_B  iff  origin == bridge_extrapolation
            (R4) synthetic sensor subtype  iff  origin == synthetic_sensor_perturbation
            (R5) parent_sample_id != None  iff  anomaly_origin == synthetic_sensor_perturbation
            (R6) regime-B samples are extrapolation regime
            (R7) anomaly_severity == 0 for clean samples
            (R8) anomaly_severity in [0.2, 1.0] for synthetic anomalies
        """
        # R1 + R2
        if self.anomaly_flag and self.anomaly_subtype == AnomalySubtype.NONE:
            raise ValueError(
                f"anomaly_flag=True but anomaly_subtype=NONE (sample_id={self.sample_id})"
            )
        if (not self.anomaly_flag) and self.anomaly_subtype != AnomalySubtype.NONE:
            raise ValueError(
                f"anomaly_flag=False but anomaly_subtype={self.anomaly_subtype.value} "
                f"(sample_id={self.sample_id})"
            )
        if self.anomaly_flag and self.anomaly_origin == AnomalyOrigin.NONE:
            raise ValueError(
                f"anomaly_flag=True but anomaly_origin=NONE (sample_id={self.sample_id})"
            )
        if (not self.anomaly_flag) and self.anomaly_origin != AnomalyOrigin.NONE:
            raise ValueError(
                f"anomaly_flag=False but anomaly_origin={self.anomaly_origin.value} "
                f"(sample_id={self.sample_id})"
            )

        # R3: low_voltage_regime_B subtype requires bridge_extrapolation origin
        if self.anomaly_subtype == AnomalySubtype.LOW_VOLTAGE_REGIME_B:
            if self.anomaly_origin != AnomalyOrigin.BRIDGE_EXTRAPOLATION:
                raise ValueError(
                    f"low_voltage_regime_B subtype must have bridge_extrapolation origin "
                    f"(sample_id={self.sample_id}, got origin={self.anomaly_origin.value})"
                )

        # R4: synthetic sensor subtypes require synthetic_sensor_perturbation origin
        if self.anomaly_subtype in SYNTHETIC_SENSOR_SUBTYPES:
            if self.anomaly_origin != AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION:
                raise ValueError(
                    f"Synthetic sensor subtype {self.anomaly_subtype.value} must have "
                    f"synthetic_sensor_perturbation origin "
                    f"(sample_id={self.sample_id}, got origin={self.anomaly_origin.value})"
                )

        # R5: parent_sample_id pairing only for synthetic_sensor_perturbation
        if self.parent_sample_id is not None:
            if self.anomaly_origin != AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION:
                raise ValueError(
                    f"parent_sample_id set but origin is {self.anomaly_origin.value}; "
                    f"only synthetic_sensor_perturbation samples may have parents "
                    f"(sample_id={self.sample_id})"
                )
        if self.parent_sample_id is None:
            if self.anomaly_origin == AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION:
                raise ValueError(
                    f"synthetic_sensor_perturbation sample must have parent_sample_id "
                    f"(sample_id={self.sample_id})"
                )

        # R6: regime-B == extrapolation regime
        if self.anomaly_subtype == AnomalySubtype.LOW_VOLTAGE_REGIME_B:
            if self.regime != Regime.EXTRAPOLATION:
                raise ValueError(
                    f"low_voltage_regime_B must have regime=extrapolation "
                    f"(sample_id={self.sample_id}, got regime={self.regime.value})"
                )

        # R7 + R8: severity rules
        if (not self.anomaly_flag) and self.anomaly_severity != 0.0:
            raise ValueError(
                f"clean sample must have anomaly_severity=0.0 "
                f"(sample_id={self.sample_id}, got {self.anomaly_severity})"
            )
        if (self.anomaly_flag
                and self.anomaly_origin == AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION):
            if not (0.2 <= self.anomaly_severity <= 1.0):
                raise ValueError(
                    f"synthetic anomaly must have severity in [0.2, 1.0] "
                    f"(sample_id={self.sample_id}, got {self.anomaly_severity})"
                )

        return self


# =============================================================================
# Split file schema
# =============================================================================
class SplitFile(BaseModel):
    """A train/val/test split assignment."""

    version: str = Field(default="1.0")
    split_type: Literal["by_cell_primary", "by_record_optimistic_baseline"]
    rng_seed: int

    # by-cell splits: lists of cell IDs
    train_cells: Optional[list[str]] = None
    val_cells: Optional[list[str]] = None
    test_cells: Optional[list[str]] = None

    # by-record splits: lists of sample IDs
    train_samples: Optional[list[str]] = None
    val_samples: Optional[list[str]] = None
    test_samples: Optional[list[str]] = None

    n_train_cells: Optional[int] = None
    n_val_cells: Optional[int] = None
    n_test_cells: Optional[int] = None

    n_train_samples: int
    n_val_samples: int
    n_test_samples: int

    warning: Optional[str] = Field(
        default=None,
        description="For by_record split, explains the leakage caveat."
    )

    @model_validator(mode="after")
    def _check_split_consistency(self):
        if self.split_type == "by_cell_primary":
            # by-cell splits MUST populate both:
            #   - cell lists (the partition of physical cells)
            #   - sample lists (the resolved per-sample assignment, including
            #     regime-B samples which don't belong to any cell)
            if any(x is None for x in [self.train_cells, self.val_cells, self.test_cells]):
                raise ValueError("by_cell_primary split must populate train/val/test_cells lists")
            if any(x is None for x in [self.train_samples, self.val_samples, self.test_samples]):
                raise ValueError(
                    "by_cell_primary split must also populate train/val/test_samples lists "
                    "(explicit per-sample assignment, including regime-B samples not tied to cells)"
                )
            # Check cell partitions are disjoint
            train_c, val_c, test_c = set(self.train_cells), set(self.val_cells), set(self.test_cells)
            if train_c & val_c:
                raise ValueError(f"train and val cells overlap: {train_c & val_c}")
            if train_c & test_c:
                raise ValueError(f"train and test cells overlap: {train_c & test_c}")
            if val_c & test_c:
                raise ValueError(f"val and test cells overlap: {val_c & test_c}")
            # Check sample partitions are also disjoint
            train_s, val_s, test_s = set(self.train_samples), set(self.val_samples), set(self.test_samples)
            if train_s & val_s:
                raise ValueError(f"train and val samples overlap: {len(train_s & val_s)} samples")
            if train_s & test_s:
                raise ValueError(f"train and test samples overlap: {len(train_s & test_s)} samples")
            if val_s & test_s:
                raise ValueError(f"val and test samples overlap: {len(val_s & test_s)} samples")

        elif self.split_type == "by_record_optimistic_baseline":
            if any(x is None for x in [self.train_samples, self.val_samples, self.test_samples]):
                raise ValueError(
                    "by_record_optimistic_baseline split must populate train/val/test_samples"
                )
            if not self.warning or len(self.warning) < 50:
                raise ValueError(
                    "by_record split must carry a substantive warning about within-cell leakage"
                )

        return self


# =============================================================================
# Manifest schema
# =============================================================================
class DatasetManifest(BaseModel):
    """Top-level dataset manifest with provenance and summary."""

    dataset_name: str = Field(default="MagBridge-Battery")
    dataset_version: str = Field(default="1.0")
    schema_version: str = Field(default=SCHEMA_VERSION)
    license_dataset: str = Field(default="CC-BY-4.0")
    license_code: str = Field(default="Apache-2.0")
    generated_at_utc: str = Field(..., description="ISO 8601 UTC timestamp of generation")

    # Sample counts (must match the actual generated data)
    n_total_samples: int
    n_clean_grounded_samples: int
    n_synthetic_anomaly_samples: int
    n_regime_b_extrapolation_samples: int

    # Provenance
    osf_data_hash: str
    pulsebat_data_hash: str
    bridge_code_commit: str
    config_hash: str

    # Bridge config (locked, embedded for reproducibility)
    bridge_version: str
    bridge_config: dict

    # Signal channel documentation (explains C5/C6 naming)
    # Default value is set in build_manifest; included here for schema completeness.
    signal_channels: dict = Field(
        default_factory=dict,
        description=(
            "Per-channel description. Each key is a channel name (e.g. 'B_s1Y') "
            "mapping to a short description. C5/C6 channels are NOT strict "
            "magnitudes — values can be negative — they are the 5th and 6th "
            "channels from the OSF source data."
        ),
    )

    # Citation
    citation: Optional[str] = None

    @model_validator(mode="after")
    def _check_total_matches(self):
        expected = (self.n_clean_grounded_samples
                    + self.n_synthetic_anomaly_samples
                    + self.n_regime_b_extrapolation_samples)
        if self.n_total_samples != expected:
            raise ValueError(
                f"n_total_samples ({self.n_total_samples}) does not match the sum "
                f"of components ({expected})"
            )
        return self
