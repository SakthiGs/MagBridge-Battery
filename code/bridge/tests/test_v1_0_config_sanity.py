"""Sanity test for the LOCKED v1.0 config.

Verifies WITHOUT running the bridge that:
  - PulseBat has the expected number of records
  - The configured variants-per-record produces the expected clean count
  - All three category counts sum to total_expected_samples

This catches the "560 vs 5,600" bug class before committing to a 6-hour
generation run. If this test fails, the locked config has internal
inconsistency and you'd waste compute time discovering it.
"""

from pathlib import Path

import pandas as pd
import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "generation_config.yaml"
PULSEBAT_PATH = PROJECT_ROOT / "data" / "v1.0" / "pulsebat_lfp.csv"


def test_v1_0_config_is_internally_consistent():
    """Locked v1.0 config: 560 records * 10 variants + 600 anom + 560 regime-B = 6,760."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    pulsebat = pd.read_csv(PULSEBAT_PATH)

    n_records = len(pulsebat)
    n_variants = cfg["generation"]["n_clean_variants_per_record"]
    n_clean_expected = n_records * n_variants
    n_anomaly = cfg["generation"]["n_synthetic_anomaly_samples"]
    n_regime_b = cfg["generation"]["n_regime_b_extrapolation_samples"]
    n_total = n_clean_expected + n_anomaly + n_regime_b

    # PulseBat has 56 cells x 10 SOC = 560 records (locked source)
    assert n_records == 560, (
        f"PulseBat must have exactly 560 records (56 cells x 10 SOC), got {n_records}"
    )

    # Config consistency: clean = records * variants
    assert n_clean_expected == cfg["generation"]["n_clean_grounded_samples"], (
        f"Computed clean count ({n_clean_expected}) does not match config "
        f"({cfg['generation']['n_clean_grounded_samples']}). Either change "
        f"n_clean_grounded_samples or n_clean_variants_per_record."
    )

    # Config consistency: total = clean + anomaly + regime-B
    assert n_total == cfg["generation"]["total_expected_samples"], (
        f"Computed total ({n_total}) does not match total_expected_samples "
        f"({cfg['generation']['total_expected_samples']})"
    )

    # Anomaly subtype counts must sum to n_synthetic_anomaly_samples
    subtype_sum = sum(cfg["anomaly_subtypes"].values())
    assert subtype_sum == n_anomaly, (
        f"Anomaly subtype counts sum to {subtype_sum}, but "
        f"n_synthetic_anomaly_samples is {n_anomaly}"
    )

    # Anomalies must be a subset of clean parents (no replacement)
    assert n_anomaly <= n_clean_expected, (
        f"Cannot generate {n_anomaly} paired anomalies from only {n_clean_expected} "
        f"clean parents without replacement"
    )

    # Regime-B count must equal config value
    assert n_regime_b == cfg["regime_b"]["n_samples"], (
        f"regime_b section count ({cfg['regime_b']['n_samples']}) does not match "
        f"generation section count ({n_regime_b})"
    )


def test_v1_0_runtime_estimate_reasonable():
    """Sanity check: estimated runtime should be in the 4-10 hour range."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    # Bridge is called for every clean + regime-B sample (anomalies derive from clean)
    n_bridge_calls = (
        cfg["generation"]["n_clean_grounded_samples"]
        + cfg["generation"]["n_regime_b_extrapolation_samples"]
    )
    # Conservative estimate: 3-5 seconds per call on Colab free CPU
    est_hours = n_bridge_calls * 3.5 / 3600
    assert 3.0 <= est_hours <= 12.0, (
        f"Estimated runtime {est_hours:.1f}h is outside the expected 3-12h band. "
        f"Either the spec has changed dramatically or runtime estimate needs updating."
    )
