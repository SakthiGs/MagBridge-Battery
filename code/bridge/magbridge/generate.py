"""
MagBridge-Battery — main dataset generator.

Orchestrates the full v1.0 generation pipeline:
    1. Load locked config + bridge artifacts + PulseBat CSV
    2. Phase A: generate 5,600 clean grounded samples (one per cell × SOC)
    3. Phase B: generate 600 paired synthetic anomalies (4 subtypes × 150)
    4. Phase C: generate 560 regime-B extrapolation samples
    5. Build catalog and splits
    6. Write Parquet shards + manifest + split files
    7. Run dataset-level validator

Usage:
    python -m magbridge.generate \\
        --config configs/generation_config.yaml \\
        --data-dir data/v1.0 \\
        --output-dir /tmp/magbridge-output

Output structure:
    output_dir/
        manifest.json
        splits/by_cell_primary.json
        splits/by_record_optimistic_baseline.json
        data/metadata.parquet
        data/shard_0000.parquet ... shard_0004.parquet

Determinism: fully reproducible given the same config + bridge artifacts
+ PulseBat CSV. The master seed (config.generation.rng_seed) drives all
per-sample randomness through `derive_per_sample_seed`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from magbridge.anomalies import inject_anomaly, N_TIMESTEPS, N_CHANNELS
from magbridge.bridge import BridgeV13, BridgeConfig, Regime, OSF_ANCHORS
from magbridge.manifest import (
    build_manifest,
    hash_config,
    write_manifest_with_summaries,
)
from magbridge.sample_id import derive_per_sample_seed, make_sample_id
from magbridge.schema import (
    SampleRecord,
    AnomalyOrigin,
    AnomalySubtype,
    Chemistry,
    Regime as SchemaRegime,
    SecondLifeClass,
    SIGNAL_CHANNEL_NAMES,
    SIGNAL_N_TIMESTEPS,
)
from magbridge.splits import (
    SampleEntry,
    build_by_cell_split,
    build_by_record_split,
)
from magbridge.validator import validate_dataset


# =============================================================================
# Sample-record builders for each phase
# =============================================================================

# U-feature columns in the PulseBat CSV
U_COLS = [f"U{i}" for i in range(1, 22)]

# Voltages used for regime-B extrapolation
REGIME_B_VOLTAGES = (2.54, 2.81, 3.00)


def _build_sample_record(
    *,
    signal: np.ndarray,
    time_norm: np.ndarray,
    sample_id: str,
    parent_sample_id: Optional[str],
    cell_id: str,
    generation_seed: int,
    bridge_version: str,
    bridge_config_hash: str,
    voltage: float,
    soc: float,
    soh: Optional[float],
    u_features: Optional[list[float]],
    regime: SchemaRegime,
    nearest_anchor: float,
    anomaly_flag: bool,
    anomaly_subtype: AnomalySubtype,
    anomaly_origin: AnomalyOrigin,
    anomaly_severity: float,
    second_life_class: Optional[SecondLifeClass],
) -> SampleRecord:
    """Build a SampleRecord with validation."""
    return SampleRecord(
        sample_id=sample_id,
        parent_sample_id=parent_sample_id,
        cell_id=cell_id,
        generation_seed=generation_seed,
        bridge_version=bridge_version,
        bridge_config_hash=bridge_config_hash,
        voltage=voltage,
        soc=soc,
        soh=soh,
        chemistry=Chemistry.LFP,
        u_features=u_features,
        regime=regime,
        nearest_anchor=nearest_anchor,
        anomaly_flag=anomaly_flag,
        anomaly_subtype=anomaly_subtype,
        anomaly_origin=anomaly_origin,
        anomaly_severity=anomaly_severity,
        second_life_class=second_life_class,
        B_s1Y=signal[:, 0].tolist(),
        B_s1Z=signal[:, 1].tolist(),
        B_s2Y=signal[:, 2].tolist(),
        B_s2Z=signal[:, 3].tolist(),
        B_s1C5=signal[:, 4].tolist(),
        B_s2C6=signal[:, 5].tolist(),
        time_norm=time_norm.tolist(),
    )


# =============================================================================
# Phase A: clean grounded generation
# =============================================================================
def generate_clean_grounded(
    pulsebat: pd.DataFrame,
    bridge: BridgeV13,
    bridge_config_hash: str,
    bridge_version: str,
    master_seed: int,
    n_variants_per_record: int = 10,
    progress_every: int = 100,
) -> list[SampleRecord]:
    """Generate N synthetic samples per (cell, SOC) row in pulsebat.

    Each PulseBat row defines one unique physical state (cell, SOC, SOH,
    U-features). For each row, we generate `n_variants_per_record` samples
    that share the same physical state but use distinct generation seeds.
    The resulting samples differ only in bridge randomness (morphology
    jitter, amplitude noise, sensor noise, SOC fluctuation) -- modelling
    the same cell measured multiple times with independent noise.

    Voltage is derived from each row's U-features (mean across U1..U21,
    clipped to grounded range), matching what the v1.2 pilot does.

    Returns the full list of SampleRecord objects.
    """
    target_n = len(pulsebat) * n_variants_per_record
    print(f"\n=== Phase A: clean grounded generation ===")
    print(f"  Target: {target_n} samples "
          f"({len(pulsebat)} rows x {n_variants_per_record} noise variants each)")
    records: list[SampleRecord] = []
    t_start = time.time()
    samples_done = 0

    for i, row in pulsebat.iterrows():
        cell_id = str(row["No"])
        soc = float(row["SOC"])
        soh = float(row["SOH"])
        u_vec = row[U_COLS].values.astype(float)
        # Voltage from mean of U-features, clipped to grounded regime
        v_op = float(np.clip(u_vec.mean(), 3.06, 3.34))
        # second_life_class from PulseBat CSV (shared across variants)
        slc = SecondLifeClass(row["second_life_class"])

        for variant in range(n_variants_per_record):
            # Deterministic per-sample seed -- variant index disambiguates
            seed = derive_per_sample_seed(
                master_seed, "clean", cell_id, int(soc), variant,
            )

            # Generate signal via bridge
            signal, regime_bridge, nearest = bridge.generate(
                voltage=v_op, soc=soc, soh=soh, seed=seed, u_features=u_vec,
            )
            # Time axis is uniform [0, 1] for clean samples
            time_norm = np.linspace(0.0, 1.0, N_TIMESTEPS)

            # Make canonical sample ID -- distinct per variant because the
            # seed enters the hash inputs
            sample_id = make_sample_id(
                cell_id=cell_id,
                voltage=v_op,
                soc=soc,
                soh=soh,
                generation_seed=seed,
                anomaly_subtype=AnomalySubtype.NONE,
                anomaly_origin=AnomalyOrigin.NONE,
                anomaly_severity=0.0,
                parent_sample_id=None,
                bridge_version=bridge_version,
                bridge_config_hash=bridge_config_hash,
            )

            # Map bridge regime to schema regime
            schema_regime = (SchemaRegime.GROUNDED if regime_bridge == Regime.GROUNDED
                              else SchemaRegime.EXTRAPOLATION)

            rec = _build_sample_record(
                signal=signal,
                time_norm=time_norm,
                sample_id=sample_id,
                parent_sample_id=None,
                cell_id=cell_id,
                generation_seed=seed,
                bridge_version=bridge_version,
                bridge_config_hash=bridge_config_hash,
                voltage=v_op,
                soc=soc,
                soh=soh,
                u_features=u_vec.tolist(),
                regime=schema_regime,
                nearest_anchor=nearest,
                anomaly_flag=False,
                anomaly_subtype=AnomalySubtype.NONE,
                anomaly_origin=AnomalyOrigin.NONE,
                anomaly_severity=0.0,
                second_life_class=slc,
            )
            records.append(rec)
            samples_done += 1

            if samples_done % progress_every == 0:
                elapsed = time.time() - t_start
                rate = samples_done / elapsed
                remaining = (target_n - samples_done) / rate
                print(f"  [{samples_done}/{target_n}] "
                      f"{samples_done*100/target_n:.1f}% — "
                      f"rate {rate:.2f}/s — ETA {remaining/60:.1f} min",
                      flush=True)

    print(f"  Done: {len(records)} clean samples in {time.time()-t_start:.1f}s")
    return records


# =============================================================================
# Phase B: synthetic anomaly generation (paired)
# =============================================================================
def generate_synthetic_anomalies(
    clean_records: list[SampleRecord],
    subtype_counts: dict[str, int],
    severity_range: tuple[float, float],
    master_seed: int,
    bridge_version: str,
    bridge_config_hash: str,
) -> list[SampleRecord]:
    """Generate paired anomalies by perturbing 600 randomly-chosen clean samples.

    Each parent is used for EXACTLY ONE anomaly subtype. Subtype assignment is
    interleaved: parent[i] gets subtype[i % 4].
    """
    print(f"\n=== Phase B: synthetic anomalies ===")
    total = sum(subtype_counts.values())
    print(f"  Target: {total} paired anomalies (subtype counts: {dict(subtype_counts)})")

    # Sample 600 parents uniformly from clean records (deterministic)
    rng = np.random.default_rng(master_seed)
    parent_indices = rng.choice(len(clean_records), size=total, replace=False)
    # The order of subtype assignment matters for reproducibility — we use the
    # subtype_counts as an ordered iterable.
    subtype_order = []
    for st_name, count in subtype_counts.items():
        subtype_order.extend([st_name] * count)
    # Shuffle the subtype assignment to avoid clustering
    rng.shuffle(subtype_order)
    assert len(parent_indices) == len(subtype_order)

    records: list[SampleRecord] = []
    t_start = time.time()
    sev_lo, sev_hi = severity_range

    for i, (parent_idx, subtype_name) in enumerate(zip(parent_indices, subtype_order)):
        parent = clean_records[int(parent_idx)]
        subtype_enum = AnomalySubtype(subtype_name)

        # Per-anomaly seed (deterministic from parent + subtype + i)
        anom_seed = derive_per_sample_seed(
            master_seed, "anomaly", parent.sample_id, subtype_name, i
        )
        anom_rng = np.random.default_rng(anom_seed)
        # Sample severity uniformly in [0.2, 1.0]
        severity = float(anom_rng.uniform(sev_lo, sev_hi))

        # Reconstruct parent's signal/time_norm
        parent_signal = np.column_stack([
            parent.B_s1Y, parent.B_s1Z, parent.B_s2Y, parent.B_s2Z,
            parent.B_s1C5, parent.B_s2C6,
        ])
        parent_time_norm = np.array(parent.time_norm)

        # Apply anomaly (uses its own deterministic sub-RNG)
        inj_rng = np.random.default_rng(anom_seed + 1)
        anom_signal, anom_time_norm = inject_anomaly(
            subtype_name, parent_signal, parent_time_norm, severity, inj_rng,
        )

        # Build the anomaly sample ID
        anom_sample_id = make_sample_id(
            cell_id=parent.cell_id,
            voltage=parent.voltage,
            soc=parent.soc,
            soh=parent.soh,
            generation_seed=anom_seed,
            anomaly_subtype=subtype_enum,
            anomaly_origin=AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION,
            anomaly_severity=severity,
            parent_sample_id=parent.sample_id,
            bridge_version=bridge_version,
            bridge_config_hash=bridge_config_hash,
        )

        rec = _build_sample_record(
            signal=anom_signal,
            time_norm=anom_time_norm,
            sample_id=anom_sample_id,
            parent_sample_id=parent.sample_id,
            cell_id=parent.cell_id,
            generation_seed=anom_seed,
            bridge_version=bridge_version,
            bridge_config_hash=bridge_config_hash,
            voltage=parent.voltage,
            soc=parent.soc,
            soh=parent.soh,
            u_features=parent.u_features,
            regime=parent.regime,
            nearest_anchor=parent.nearest_anchor,
            anomaly_flag=True,
            anomaly_subtype=subtype_enum,
            anomaly_origin=AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION,
            anomaly_severity=severity,
            second_life_class=parent.second_life_class,
        )
        records.append(rec)

    print(f"  Done: {len(records)} anomaly samples in {time.time()-t_start:.1f}s")
    subtype_actual = Counter(r.anomaly_subtype.value for r in records)
    print(f"  Subtype counts (actual): {dict(subtype_actual)}")
    return records


# =============================================================================
# Phase C: regime-B (low-voltage extrapolation) generation
# =============================================================================
def generate_regime_b(
    n_samples: int,
    pulsebat: pd.DataFrame,
    bridge: BridgeV13,
    bridge_config_hash: str,
    bridge_version: str,
    master_seed: int,
    progress_every: int = 100,
) -> list[SampleRecord]:
    """Generate regime-B (extrapolation) samples at low voltages.

    Each sample picks a voltage uniformly from REGIME_B_VOLTAGES, SOC and SOH
    from a random PulseBat row (so values are realistic but the SOH at these
    voltages isn't physically meaningful — schema sets soh=None for regime-B).
    """
    print(f"\n=== Phase C: regime-B (extrapolation) ===")
    print(f"  Target: {n_samples} samples")
    rng = np.random.default_rng(master_seed + 2)  # distinct sub-seed
    records: list[SampleRecord] = []
    t_start = time.time()

    for i in range(n_samples):
        # Pick voltage uniformly from low-V set
        v_op = float(rng.choice(REGIME_B_VOLTAGES))
        # Pick a random PulseBat row for SOC + driving conditions
        idx = int(rng.integers(0, len(pulsebat)))
        row = pulsebat.iloc[idx]
        soc = float(row["SOC"])
        # The bridge needs an SOH input for its perturbation; use SOH=1.0
        # (no degradation) for regime-B. The schema stores soh=None to
        # signal that SOH at these voltages isn't physically meaningful.
        soh_for_bridge = 1.0
        u_vec = row[U_COLS].values.astype(float)

        # Distinct pseudo-cell IDs per voltage (regimeB_v254, regimeB_v281, regimeB_v300).
        # This ensures the by_cell split builder treats them as distinct cells
        # — they go to one split each, not all three. Avoids the "cell_id=N/A
        # appears in train/val/test" labeling inconsistency. Format: integer
        # centivolts to keep the ID compact and disjoint from PulseBat cells
        # (which are integer strings like "1", "2", ..., "56").
        regime_b_cell_id = f"regimeB_v{int(round(v_op * 100)):03d}"

        seed = derive_per_sample_seed(master_seed, "regimeb", i, v_op)

        signal, regime_bridge, nearest = bridge.generate(
            voltage=v_op, soc=soc, soh=soh_for_bridge, seed=seed, u_features=u_vec,
        )
        time_norm = np.linspace(0.0, 1.0, N_TIMESTEPS)

        sample_id = make_sample_id(
            cell_id=regime_b_cell_id,
            voltage=v_op,
            soc=soc,
            soh=None,
            generation_seed=seed,
            anomaly_subtype=AnomalySubtype.LOW_VOLTAGE_REGIME_B,
            anomaly_origin=AnomalyOrigin.BRIDGE_EXTRAPOLATION,
            anomaly_severity=1.0,
            parent_sample_id=None,
            bridge_version=bridge_version,
            bridge_config_hash=bridge_config_hash,
        )

        rec = _build_sample_record(
            signal=signal,
            time_norm=time_norm,
            sample_id=sample_id,
            parent_sample_id=None,
            cell_id=regime_b_cell_id,
            generation_seed=seed,
            bridge_version=bridge_version,
            bridge_config_hash=bridge_config_hash,
            voltage=v_op,
            soc=soc,
            soh=None,
            u_features=None,
            regime=SchemaRegime.EXTRAPOLATION,
            nearest_anchor=nearest,
            anomaly_flag=True,
            anomaly_subtype=AnomalySubtype.LOW_VOLTAGE_REGIME_B,
            anomaly_origin=AnomalyOrigin.BRIDGE_EXTRAPOLATION,
            anomaly_severity=1.0,
            second_life_class=None,
        )
        records.append(rec)

        if (i + 1) % progress_every == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            remaining = (n_samples - i - 1) / rate
            print(f"  [{i+1}/{n_samples}] {(i+1)*100/n_samples:.1f}% — "
                  f"rate {rate:.2f}/s — ETA {remaining/60:.1f} min")

    print(f"  Done: {len(records)} regime-B samples in {time.time()-t_start:.1f}s")
    return records


# =============================================================================
# Parquet writers
# =============================================================================
def write_shards(
    records: list[SampleRecord],
    output_dir: Path,
    n_shards: int = 5,
) -> list[Path]:
    """Write samples into N Parquet shards (with signal data)."""
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Stable shard assignment by record order
    shard_size = (len(records) + n_shards - 1) // n_shards
    shard_paths = []

    for shard_idx in range(n_shards):
        lo = shard_idx * shard_size
        hi = min(lo + shard_size, len(records))
        if lo >= hi:
            break

        chunk = records[lo:hi]
        rows = [r.model_dump(mode="json") for r in chunk]
        df = pd.DataFrame(rows)

        path = data_dir / f"shard_{shard_idx:04d}.parquet"
        df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
        shard_paths.append(path)
        print(f"  Wrote {path.name}: {len(chunk)} samples")
    return shard_paths


def write_metadata_index(
    records: list[SampleRecord],
    output_dir: Path,
) -> Path:
    """Write a small Parquet file with metadata only (no signal columns).

    Lets users query/filter the dataset without loading signal data.
    """
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Strip signal/time_norm/u_features columns to keep this lightweight
    drop_cols = {"B_s1Y", "B_s1Z", "B_s2Y", "B_s2Z", "B_s1C5", "B_s2C6",
                 "time_norm", "u_features"}
    rows = []
    for r in records:
        d = r.model_dump(mode="json")
        for k in drop_cols:
            d.pop(k, None)
        rows.append(d)
    df = pd.DataFrame(rows)
    path = data_dir / "metadata.parquet"
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=False)
    print(f"  Wrote {path.name}: {len(df)} rows, {len(df.columns)} columns")
    return path


# =============================================================================
# Main entry point
# =============================================================================
def generate_dataset(
    config_path: Path,
    data_dir: Path,
    output_dir: Path,
    bridge_code_commit: Optional[str] = None,
) -> None:
    """Run the full v1.0 generation pipeline end to end."""
    t_total = time.time()

    # 1. Load locked config
    print(f"Loading config: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    bridge_cfg_dict = cfg["bridge"]
    bridge_version = bridge_cfg_dict["version"]
    expected_total = cfg["generation"]["total_expected_samples"]
    master_seed = cfg["generation"]["rng_seed"]

    print(f"  Dataset: {cfg['dataset']['name']} v{cfg['dataset']['version']}")
    print(f"  Bridge: {bridge_version}")
    print(f"  Expected total: {expected_total}")
    print(f"  Master seed: {master_seed}")

    # 2. Build bridge
    print(f"\nBuilding bridge from {data_dir}...")
    bridge_config = BridgeConfig(
        qrec_drift_strength=bridge_cfg_dict["qrec_drift_strength"],
        cone_half_angle_deg=bridge_cfg_dict["cone_half_angle_deg"],
        decode_k=bridge_cfg_dict["decode_k"],
        decode_kernel_sigma=bridge_cfg_dict["decode_kernel_sigma"],
        amplitude_strength=bridge_cfg_dict["amplitude_strength"],
        spectral_strength=bridge_cfg_dict["spectral_strength"],
        sensor_noise_fraction=bridge_cfg_dict["sensor_noise_fraction"],
        soc_fluctuation_strength=bridge_cfg_dict["soc_fluctuation_strength"],
        cone_min_candidates=bridge_cfg_dict["cone_min_candidates"],
        cone_disable_below_delta=bridge_cfg_dict["cone_disable_below_delta"],
    )
    bridge = BridgeV13(
        anchor_stats_path=data_dir / "anchor_stats.npz",
        osf_seq_path=data_dir / "osf_sequences.npz",
        osf_qrec_emb_path=data_dir / "qrec_embeddings.npz",
        lda_fit_path=data_dir / "lda_fit.npz",
        config=bridge_config,
    )
    bridge_cfg_hash = hash_config(asdict(bridge_config))
    print(f"  Bridge config hash: {bridge_cfg_hash}")

    # 3. Load PulseBat
    pulsebat_path = data_dir / "pulsebat_lfp.csv"
    pulsebat = pd.read_csv(pulsebat_path)
    print(f"  Loaded PulseBat: {len(pulsebat)} rows, {pulsebat['No'].nunique()} cells")

    # 4. Phase A: clean grounded
    n_variants = cfg["generation"].get("n_clean_variants_per_record", 10)
    clean_records = generate_clean_grounded(
        pulsebat, bridge, bridge_cfg_hash, bridge_version, master_seed,
        n_variants_per_record=n_variants,
    )

    # 5. Phase B: synthetic anomalies
    anomaly_records = generate_synthetic_anomalies(
        clean_records=clean_records,
        subtype_counts=cfg["anomaly_subtypes"],
        severity_range=(
            cfg["generation"]["anomaly_severity_min"],
            cfg["generation"]["anomaly_severity_max"],
        ),
        master_seed=master_seed,
        bridge_version=bridge_version,
        bridge_config_hash=bridge_cfg_hash,
    )

    # 6. Phase C: regime-B
    regime_b_records = generate_regime_b(
        n_samples=cfg["regime_b"]["n_samples"],
        pulsebat=pulsebat,
        bridge=bridge,
        bridge_config_hash=bridge_cfg_hash,
        bridge_version=bridge_version,
        master_seed=master_seed,
    )

    # 7. Assemble full catalog
    all_records = clean_records + anomaly_records + regime_b_records
    print(f"\n=== Catalog assembled: {len(all_records)} samples ===")
    print(f"  Clean grounded: {len(clean_records)}")
    print(f"  Synthetic anomalies: {len(anomaly_records)}")
    print(f"  Regime-B: {len(regime_b_records)}")
    assert len(all_records) == expected_total, (
        f"Got {len(all_records)} samples but config expects {expected_total}"
    )

    # 8. Build SampleEntry catalog for the validator and splitter
    catalog = [
        SampleEntry(
            sample_id=r.sample_id,
            cell_id=r.cell_id,
            parent_sample_id=r.parent_sample_id,
            anomaly_subtype=r.anomaly_subtype,
            anomaly_origin=r.anomaly_origin,
            soh=r.soh,
        )
        for r in all_records
    ]

    # 9. Run dataset-level validator
    # Each PulseBat cell has 10 SOC levels and each (cell, SOC) row produces
    # n_variants clean samples, so per-cell clean count = 10 * n_variants.
    n_soc_levels_per_cell = 10
    expected_clean_per_cell = n_soc_levels_per_cell * n_variants
    print(f"\n=== Running validator ===")
    failures = validate_dataset(
        catalog,
        expected_counts={
            "n_clean_grounded_samples": cfg["generation"]["n_clean_grounded_samples"],
            "n_synthetic_anomaly_samples": cfg["generation"]["n_synthetic_anomaly_samples"],
            "n_regime_b_extrapolation_samples": cfg["generation"]["n_regime_b_extrapolation_samples"],
            "total_expected_samples": cfg["generation"]["total_expected_samples"],
        },
        expected_subtype_counts=cfg["anomaly_subtypes"],
        expected_clean_per_cell=expected_clean_per_cell,
    )
    print(f"  Validation: {len(failures)} failures")
    assert not failures, f"Validation failed with {len(failures)} issues"

    # 10. Build splits
    print(f"\n=== Building splits ===")
    primary_cfg = cfg["splits"]["primary"]
    # Use explicit train/val/test counts from config when present; otherwise
    # fall back to the locked v1.0 defaults (39/8/9). This lets integration
    # tests use a downsized split (e.g., 3/1/1 for 5 cells).
    by_cell_kwargs = {"rng_seed": primary_cfg["rng_seed"]}
    if "n_train_cells" in primary_cfg:
        by_cell_kwargs["train_n"] = primary_cfg["n_train_cells"]
        by_cell_kwargs["val_n"] = primary_cfg["n_val_cells"]
        by_cell_kwargs["test_n"] = primary_cfg["n_test_cells"]
    by_cell = build_by_cell_split(catalog, **by_cell_kwargs)
    by_record = build_by_record_split(catalog, rng_seed=cfg["splits"]["secondary"]["rng_seed"])
    print(f"  by_cell:   {by_cell.n_train_samples}/{by_cell.n_val_samples}/{by_cell.n_test_samples}")
    print(f"  by_record: {by_record.n_train_samples}/{by_record.n_val_samples}/{by_record.n_test_samples}")

    # 11. Write outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Writing outputs to {output_dir} ===")

    # Parquet shards + metadata index
    write_shards(all_records, output_dir, n_shards=5)
    write_metadata_index(all_records, output_dir)

    # Split files
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    with open(splits_dir / "by_cell_primary.json", "w") as f:
        json.dump(by_cell.model_dump(mode="json"), f, indent=2, sort_keys=True)
    with open(splits_dir / "by_record_optimistic_baseline.json", "w") as f:
        json.dump(by_record.model_dump(mode="json"), f, indent=2, sort_keys=True)
    print(f"  Wrote splits/by_cell_primary.json")
    print(f"  Wrote splits/by_record_optimistic_baseline.json")

    # Manifest
    manifest = build_manifest(
        catalog=catalog,
        bridge_config=asdict(bridge_config),
        bridge_version=bridge_version,
        osf_data_path=data_dir / "osf_sequences.npz",
        pulsebat_data_path=data_dir / "pulsebat_lfp.csv",
        by_cell_split=by_cell,
        by_record_split=by_record,
        bridge_code_commit_override=bridge_code_commit,
    )
    write_manifest_with_summaries(manifest, output_dir / "manifest.json")
    print(f"  Wrote manifest.json")
    print(f"    config_hash:        {manifest.config_hash}")
    print(f"    osf_data_hash:      {manifest.osf_data_hash}")
    print(f"    pulsebat_data_hash: {manifest.pulsebat_data_hash}")

    # 12. Done
    print(f"\n=== Generation complete in {(time.time()-t_total)/60:.1f} min ===")


def main():
    p = argparse.ArgumentParser(description="MagBridge-Battery v1.0 dataset generator")
    p.add_argument("--config", type=Path, required=True,
                    help="Path to generation_config.yaml")
    p.add_argument("--data-dir", type=Path, required=True,
                    help="Directory containing bridge artifacts and PulseBat CSV")
    p.add_argument("--output-dir", type=Path, required=True,
                    help="Where to write Parquet shards, manifest, splits")
    p.add_argument("--bridge-code-commit", type=str, default=None,
                    help="Override bridge code commit SHA (else auto-detected via git)")
    args = p.parse_args()

    generate_dataset(
        config_path=args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        bridge_code_commit=args.bridge_code_commit,
    )


if __name__ == "__main__":
    main()
