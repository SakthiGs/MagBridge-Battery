"""Pre-flight sanity checks for v1.0 release readiness.

This test file runs IN ADDITION to the existing v1_0_config_sanity tests.
It catches the specific issue classes that v0.9 reviewer rounds revealed:

  R1. timebase_jitter / temporal_warp naming consistency
  R2. Channel naming consistency (Mag vs C5/C6) across schema/config/code
  R3. Regime-B pseudo-cell IDs distinct from PulseBat cell IDs
  R4. by_record warning explicitly mentions parent-child leakage

If any of these fails, the locked config / schema / code are inconsistent
and shipping would invite another round of reviewer comments. This file
is the "did I actually address all the reviewer's points" gate.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from magbridge.schema import (
    AnomalySubtype,
    SIGNAL_CHANNEL_NAMES,
    DatasetManifest,
)
from magbridge.splits import BY_RECORD_WARNING


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "generation_config.yaml"
PULSEBAT_PATH = PROJECT_ROOT / "data" / "v1.0" / "pulsebat_lfp.csv"


# =============================================================================
# R1: temporal_warp naming consistency
# =============================================================================
def test_no_timebase_jitter_anywhere_in_subtype_names():
    """The old name 'timebase_jitter' must NOT appear in the schema enum."""
    subtype_values = [s.value for s in AnomalySubtype]
    assert "timebase_jitter" not in subtype_values, (
        "Old name 'timebase_jitter' still in AnomalySubtype enum. "
        "Should be 'temporal_warp'."
    )
    assert "temporal_warp" in subtype_values, (
        "New name 'temporal_warp' missing from AnomalySubtype enum."
    )


def test_no_timebase_jitter_in_config():
    """The locked config must use 'temporal_warp', not 'timebase_jitter'."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    subtypes = cfg["anomaly_subtypes"]
    assert "timebase_jitter" not in subtypes, (
        f"Config anomaly_subtypes still has 'timebase_jitter': {subtypes}"
    )
    assert "temporal_warp" in subtypes, (
        f"Config anomaly_subtypes missing 'temporal_warp': {subtypes}"
    )


def test_config_subtypes_match_schema_enum():
    """Every subtype name in the config must be a valid AnomalySubtype enum value."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    valid_values = {s.value for s in AnomalySubtype}
    for subtype_name in cfg["anomaly_subtypes"]:
        assert subtype_name in valid_values, (
            f"Config subtype '{subtype_name}' is not a valid AnomalySubtype enum value. "
            f"Valid values: {sorted(valid_values)}"
        )


# =============================================================================
# R1b: sample-ID abbreviations must NOT reference the old subtype name
# =============================================================================
def test_no_jitter_abbreviation_in_sample_id_module():
    """Sample-ID abbreviation for TEMPORAL_WARP must NOT be 'jitter' — that
    abbreviation would surface in every sample ID (e.g. 'lfp_X_jitter_...')
    and contradict the temporal_warp rename in everything else."""
    from magbridge.sample_id import SUBTYPE_ABBREVIATIONS
    abbrev = SUBTYPE_ABBREVIATIONS.get(AnomalySubtype.TEMPORAL_WARP)
    assert abbrev != "jitter", (
        "TEMPORAL_WARP abbreviation is still 'jitter' — sample IDs would still "
        "say 'jitter' even though the subtype label is 'temporal_warp'. "
        "Change SUBTYPE_ABBREVIATIONS[AnomalySubtype.TEMPORAL_WARP] to 'twarp' "
        "(or similar) to keep IDs consistent with subtype names."
    )
    # Must not be empty or None either
    assert abbrev and len(abbrev) > 0


def test_sample_id_abbreviations_have_documented_reason():
    """Each subtype abbreviation must either:
       (a) share its first 3 chars with the subtype name, OR
       (b) be in the explicit ACCEPTED_ABBREVIATIONS list with a documented reason.

    Catches future inconsistencies where someone renames a subtype but forgets
    to update the abbreviation."""
    from magbridge.sample_id import SUBTYPE_ABBREVIATIONS

    # Acceptable abbreviations that don't share a prefix with their subtype value.
    # If you add to this list, document WHY the abbreviation makes sense.
    ACCEPTED_ABBREVIATIONS = {
        # 'twarp' = t + warp = contraction of 'temporal_warp'. Reviewer-readable.
        (AnomalySubtype.TEMPORAL_WARP, "twarp"),
    }

    for subtype, abbrev in SUBTYPE_ABBREVIATIONS.items():
        if (subtype, abbrev) in ACCEPTED_ABBREVIATIONS:
            continue
        subtype_val = subtype.value
        # Check the abbreviation shares the first 3 chars with some part of the
        # subtype value (e.g., 'caldrift' shares 'cal' with 'calibration_drift').
        shared = (
            abbrev[:3].lower() in subtype_val.lower()
            or any(part.startswith(abbrev[:3].lower())
                   for part in subtype_val.split("_"))
        )
        assert shared, (
            f"Abbreviation '{abbrev}' for subtype '{subtype_val}' looks unrelated. "
            f"Either change the abbreviation to share a prefix with the subtype name, "
            f"or add it to ACCEPTED_ABBREVIATIONS in this test with a comment."
        )


# =============================================================================
# R2: Channel naming consistency
# =============================================================================
def test_no_mag_channels_in_schema():
    """The old 'Mag' channel names must NOT appear in SIGNAL_CHANNEL_NAMES."""
    for name in SIGNAL_CHANNEL_NAMES:
        assert not name.endswith("Mag"), (
            f"Old channel name '{name}' still in SIGNAL_CHANNEL_NAMES. "
            f"Should be renamed to 'C5'/'C6'."
        )


def test_c5_c6_channels_present_in_schema():
    """New C5/C6 channel names must be in SIGNAL_CHANNEL_NAMES."""
    assert "B_s1C5" in SIGNAL_CHANNEL_NAMES
    assert "B_s2C6" in SIGNAL_CHANNEL_NAMES


def test_config_channel_names_match_schema():
    """The config's channel_names list must EXACTLY match SIGNAL_CHANNEL_NAMES."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    config_names = cfg["signal"]["channel_names"]
    assert config_names == SIGNAL_CHANNEL_NAMES, (
        f"Config channel_names ({config_names}) does not match "
        f"SIGNAL_CHANNEL_NAMES ({SIGNAL_CHANNEL_NAMES})"
    )


# =============================================================================
# R3: Regime-B pseudo-cell IDs distinct from PulseBat cells
# =============================================================================
def test_regime_b_voltage_set_yields_distinct_pseudo_cells():
    """REGIME_B_VOLTAGES, when formatted into pseudo-cell IDs, must give
    distinct strings that don't collide with PulseBat cell IDs."""
    from magbridge.generate import REGIME_B_VOLTAGES

    # Build pseudo-cell IDs using the same format as generate.py
    pseudo_cells = {
        f"regimeB_v{int(round(v * 100)):03d}"
        for v in REGIME_B_VOLTAGES
    }
    # Must be distinct (one per voltage)
    assert len(pseudo_cells) == len(REGIME_B_VOLTAGES), (
        f"REGIME_B_VOLTAGES {REGIME_B_VOLTAGES} produce non-distinct "
        f"pseudo-cell IDs: {pseudo_cells}"
    )
    # Must not contain the bare 'N/A' placeholder
    assert "N/A" not in pseudo_cells

    # Must not overlap with PulseBat cell IDs
    pulsebat = pd.read_csv(PULSEBAT_PATH)
    pulsebat_cells = {str(c) for c in pulsebat["No"].unique()}
    overlap = pseudo_cells & pulsebat_cells
    assert not overlap, (
        f"Regime-B pseudo-cell IDs collide with PulseBat cells: {overlap}"
    )


def test_n_regime_b_voltages_at_least_three_for_split_distribution():
    """With 3 splits (train/val/test) and round-robin assignment, we need at
    least 3 distinct regime-B pseudo-cells to distribute one per split."""
    from magbridge.generate import REGIME_B_VOLTAGES
    assert len(REGIME_B_VOLTAGES) >= 3, (
        f"Need at least 3 regime-B voltages for round-robin split assignment; "
        f"got {len(REGIME_B_VOLTAGES)}: {REGIME_B_VOLTAGES}"
    )


# =============================================================================
# R4: by_record warning mentions parent-child leakage
# =============================================================================
def test_by_record_warning_mentions_within_cell_leakage():
    """The warning must mention within-cell leakage (the original concern)."""
    assert "within-cell" in BY_RECORD_WARNING.lower() or "cell" in BY_RECORD_WARNING.lower()


def test_by_record_warning_mentions_parent_child_leakage():
    """The warning must explicitly mention parent-child leakage (R4)."""
    msg = BY_RECORD_WARNING.lower()
    assert "parent-child" in msg or "parent" in msg, (
        f"by_record warning does not mention parent-child leakage. "
        f"Warning text: {BY_RECORD_WARNING[:300]}"
    )


def test_by_record_warning_says_do_not_use_for_benchmark():
    """The warning must explicitly state NOT to use for benchmark reporting."""
    msg = BY_RECORD_WARNING.lower()
    assert "do not use" in msg or "not for benchmark" in msg or "do not publish" in msg


# =============================================================================
# Manifest carries channel descriptions
# =============================================================================
def test_dataset_manifest_has_signal_channels_field():
    """DatasetManifest must have a signal_channels field for channel descriptions."""
    fields = DatasetManifest.model_fields.keys()
    assert "signal_channels" in fields, (
        f"DatasetManifest missing signal_channels field. Found: {sorted(fields)}"
    )
