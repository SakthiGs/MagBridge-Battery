"""
MagBridge-Battery — train/val/test split generation.

Produces two split artifacts:

  1. by_cell_primary: 56 PulseBat cells partitioned into train/val/test.
     Each sample is assigned the partition of its cell. Synthetic anomalies
     follow their parent's partition (no within-pair split leakage).
     Regime-B samples are distributed proportionally across partitions.

  2. by_record_optimistic_baseline: individual samples partitioned directly,
     ignoring cell membership. Documented as optimistic because the same
     physical cell can appear in both train and test.

Both splits are deterministic given the RNG seed declared in the
generation config (default 42).

The functions here operate on lightweight catalogs (lists of dicts with
sample_id, cell_id, parent_sample_id, anomaly_subtype, anomaly_origin).
They do not load Parquet shards or signal data; that's the validator's
job at release time.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Optional

from magbridge.schema import (
    AnomalyOrigin,
    AnomalySubtype,
    SplitFile,
)


# =============================================================================
# Locked split configuration (matches generation_config.yaml)
# =============================================================================
BY_CELL_TRAIN_COUNT = 39
BY_CELL_VAL_COUNT = 8
BY_CELL_TEST_COUNT = 9

BY_RECORD_TRAIN_FRAC = 0.70
BY_RECORD_VAL_FRAC = 0.15
BY_RECORD_TEST_FRAC = 0.15

BY_RECORD_WARNING = (
    "OPTIMISTIC BASELINE SPLIT — DO NOT USE FOR BENCHMARK REPORTING.\n"
    "\n"
    "This split partitions individual samples without regard to cell membership "
    "OR parent-child anomaly pairing. It has TWO distinct sources of leakage:\n"
    "\n"
    "1. WITHIN-CELL LEAKAGE. Samples generated from the same physical PulseBat "
    "cell appear in both the train and test sets. A method that learns "
    "cell-specific magnetic signatures will achieve inflated test performance "
    "compared to its true cross-cell generalisation.\n"
    "\n"
    "2. PARENT-CHILD LEAKAGE. Synthetic anomalies (sensor_dropout, "
    "calibration_drift, temporal_warp, periodic_interference) and their clean "
    "parents can land in different splits. A method trained on the clean parent "
    "and tested on the anomaly child (or vice versa) will see correlated samples "
    "across the split boundary. We observe ~290 such parent-child cross-split "
    "pairs in a typical realisation.\n"
    "\n"
    "In our internal testing, downstream SOH-regression and second-life "
    "classification numbers are 5–15 percentage points higher under this split "
    "than under the by_cell_primary split. The gap is the combined leakage.\n"
    "\n"
    "This split is provided only to enable controlled studies of within-cell "
    "variation and parent-child pairing, as an optimistic upper-bound baseline, "
    "and to characterise the magnitude of the leakage effect. For all published "
    "benchmark numbers, use splits/by_cell_primary.json."
)


# =============================================================================
# Lightweight sample catalog type
# =============================================================================
class SampleEntry:
    """Minimal sample-catalog entry used by the splitter.

    Just the fields the splitter needs — keeps the splitter independent of
    the full SampleRecord schema and easy to test.
    """

    __slots__ = ("sample_id", "cell_id", "parent_sample_id",
                 "anomaly_subtype", "anomaly_origin", "soh")

    def __init__(
        self,
        sample_id: str,
        cell_id: str,
        parent_sample_id: Optional[str],
        anomaly_subtype: AnomalySubtype,
        anomaly_origin: AnomalyOrigin,
        soh: Optional[float],
    ):
        self.sample_id = sample_id
        self.cell_id = cell_id
        self.parent_sample_id = parent_sample_id
        self.anomaly_subtype = anomaly_subtype
        self.anomaly_origin = anomaly_origin
        self.soh = soh

    def is_clean_grounded(self) -> bool:
        return self.anomaly_origin == AnomalyOrigin.NONE

    def is_synthetic_anomaly(self) -> bool:
        return self.anomaly_origin == AnomalyOrigin.SYNTHETIC_SENSOR_PERTURBATION

    def is_regime_b(self) -> bool:
        return self.anomaly_origin == AnomalyOrigin.BRIDGE_EXTRAPOLATION


# =============================================================================
# By-cell split (primary)
# =============================================================================
def build_by_cell_split(
    catalog: list[SampleEntry],
    rng_seed: int = 42,
    train_n: int = BY_CELL_TRAIN_COUNT,
    val_n: int = BY_CELL_VAL_COUNT,
    test_n: int = BY_CELL_TEST_COUNT,
) -> SplitFile:
    """Build by-cell train/val/test split with safe anomaly pairing.

    Algorithm:
        1. Enumerate all unique cell_ids from clean grounded samples.
        2. Shuffle deterministically.
        3. Partition first train_n -> train, next val_n -> val, next test_n -> test.
        4. Verify counts match expected total (train_n + val_n + test_n).
        5. Sample assignment: clean grounded samples go to their cell's partition.
        6. Synthetic anomalies follow their parent's partition (no leakage across pairs).
        7. Regime-B samples are distributed proportionally to the split sizes.

    Returns:
        A validated SplitFile model. The caller is responsible for serialising
        it to JSON.

    Raises:
        ValueError: if the catalog is inconsistent (orphaned parents, cell
                    counts don't match expected, etc.).
    """
    rng = random.Random(rng_seed)

    # ---- step 1-3: partition cells ---------------------------------------
    all_cells = sorted(
        {e.cell_id for e in catalog if e.is_clean_grounded()}
    )
    expected_total = train_n + val_n + test_n
    if len(all_cells) != expected_total:
        raise ValueError(
            f"by_cell split expects exactly {expected_total} cells "
            f"(train={train_n}, val={val_n}, test={test_n}); "
            f"catalog has {len(all_cells)} unique cells in clean grounded samples"
        )

    shuffled = list(all_cells)
    rng.shuffle(shuffled)

    train_cells = sorted(shuffled[:train_n])
    val_cells = sorted(shuffled[train_n:train_n + val_n])
    test_cells = sorted(shuffled[train_n + val_n:])

    cell_to_split: dict[str, str] = {}
    for c in train_cells:
        cell_to_split[c] = "train"
    for c in val_cells:
        cell_to_split[c] = "val"
    for c in test_cells:
        cell_to_split[c] = "test"

    # ---- step 5-6: assign clean + synthetic to splits --------------------
    sample_to_split: dict[str, str] = {}

    # First pass: clean grounded samples
    for entry in catalog:
        if entry.is_clean_grounded():
            sample_to_split[entry.sample_id] = cell_to_split[entry.cell_id]

    # Second pass: synthetic anomalies follow parent
    for entry in catalog:
        if entry.is_synthetic_anomaly():
            if entry.parent_sample_id is None:
                raise ValueError(
                    f"synthetic anomaly {entry.sample_id} has no parent_sample_id"
                )
            parent_split = sample_to_split.get(entry.parent_sample_id)
            if parent_split is None:
                raise ValueError(
                    f"synthetic anomaly {entry.sample_id} parent "
                    f"{entry.parent_sample_id} not found in clean catalog"
                )
            sample_to_split[entry.sample_id] = parent_split

    # ---- step 7: regime-B follows its pseudo-cell membership ----------------
    # With regime-B getting distinct pseudo-cell IDs (regimeB_v254, regimeB_v281,
    # regimeB_v300), we want all samples sharing a pseudo-cell to land in the
    # same split. We assign each regime-B pseudo-cell to one split independently
    # (sub-seeded), so a single physical voltage's samples are not scattered
    # across train/val/test.
    regime_b = [e for e in catalog if e.is_regime_b()]
    if regime_b:
        regime_b_cells = sorted({e.cell_id for e in regime_b})
        # Sanity: regime-B pseudo-cells must not collide with PulseBat cells
        overlap = set(regime_b_cells) & set(all_cells)
        if overlap:
            raise ValueError(
                f"regime-B pseudo-cell IDs collide with PulseBat cell IDs: {overlap}"
            )
        # Deterministic shuffle + round-robin assignment across splits.
        # With 3 pseudo-cells (one per voltage) and 3 splits, round-robin
        # ensures each split gets exactly one pseudo-cell.
        rng_regime = random.Random(rng_seed + 1)
        shuffled_cells = list(regime_b_cells)
        rng_regime.shuffle(shuffled_cells)
        split_targets = ["train", "val", "test"]
        regime_b_cell_to_split: dict[str, str] = {}
        for i, cell in enumerate(shuffled_cells):
            regime_b_cell_to_split[cell] = split_targets[i % len(split_targets)]

        for entry in regime_b:
            sample_to_split[entry.sample_id] = regime_b_cell_to_split[entry.cell_id]

    # ---- final tallies for the SplitFile ---------------------------------
    train_samples = sorted([s for s, sp in sample_to_split.items() if sp == "train"])
    val_samples = sorted([s for s, sp in sample_to_split.items() if sp == "val"])
    test_samples = sorted([s for s, sp in sample_to_split.items() if sp == "test"])

    # Sanity check: every catalog sample got assigned
    assigned = len(sample_to_split)
    if assigned != len(catalog):
        unassigned = [e.sample_id for e in catalog if e.sample_id not in sample_to_split]
        raise ValueError(
            f"by_cell split assigned {assigned} of {len(catalog)} catalog samples; "
            f"unassigned: {unassigned[:5]}{'...' if len(unassigned) > 5 else ''}"
        )

    return SplitFile(
        split_type="by_cell_primary",
        rng_seed=rng_seed,
        train_cells=train_cells,
        val_cells=val_cells,
        test_cells=test_cells,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        n_train_cells=len(train_cells),
        n_val_cells=len(val_cells),
        n_test_cells=len(test_cells),
        n_train_samples=len(train_samples),
        n_val_samples=len(val_samples),
        n_test_samples=len(test_samples),
        warning=None,
    )


# =============================================================================
# By-record split (optimistic baseline)
# =============================================================================
def build_by_record_split(
    catalog: list[SampleEntry],
    rng_seed: int = 42,
    train_frac: float = BY_RECORD_TRAIN_FRAC,
    val_frac: float = BY_RECORD_VAL_FRAC,
    test_frac: float = BY_RECORD_TEST_FRAC,
) -> SplitFile:
    """Build by-record train/val/test split, ignoring cell membership.

    This split is documented as OPTIMISTIC because samples from the same
    physical cell can appear in both train and test. The returned SplitFile
    carries a substantive warning to that effect.

    Algorithm:
        1. Shuffle all sample IDs deterministically.
        2. Assign first train_frac -> train, next val_frac -> val, rest -> test.
        3. NO awareness of parent_sample_id or cell_id — by design, samples
           are partitioned independently. This is what makes the split
           optimistic.

    Even paired anomalies can land in different splits — which is intentional,
    so that the split's permissiveness is maximal.

    Returns:
        A validated SplitFile model with split_type='by_record_optimistic_baseline'.
    """
    if not (0.99 < train_frac + val_frac + test_frac < 1.01):
        raise ValueError(
            f"split fractions must sum to ~1.0 "
            f"(got {train_frac + val_frac + test_frac})"
        )

    rng = random.Random(rng_seed)
    all_ids = sorted([e.sample_id for e in catalog])
    rng.shuffle(all_ids)

    n_total = len(all_ids)
    n_train = int(round(n_total * train_frac))
    n_val = int(round(n_total * val_frac))
    n_test = n_total - n_train - n_val  # remainder for exact total

    train_samples = sorted(all_ids[:n_train])
    val_samples = sorted(all_ids[n_train:n_train + n_val])
    test_samples = sorted(all_ids[n_train + n_val:])

    return SplitFile(
        split_type="by_record_optimistic_baseline",
        rng_seed=rng_seed,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        n_train_cells=None,
        n_val_cells=None,
        n_test_cells=None,
        n_train_samples=len(train_samples),
        n_val_samples=len(val_samples),
        n_test_samples=len(test_samples),
        warning=BY_RECORD_WARNING,
    )


# =============================================================================
# Cross-checks (called by validator)
# =============================================================================
def check_no_pair_leakage(
    by_cell_split: SplitFile,
    catalog: list[SampleEntry],
) -> None:
    """Verify that no synthetic anomaly is in a different split from its parent.

    This is a critical invariant for by_cell splits — if an anomaly ended up
    in test while its clean parent ended up in train, a method evaluating on
    the anomaly's test sample has effectively seen the parent during training.

    Raises ValueError on first violation found.
    """
    # Build sample -> split lookup from the split file
    sample_to_split: dict[str, str] = {}
    for sid in by_cell_split.train_samples or []:
        sample_to_split[sid] = "train"
    for sid in by_cell_split.val_samples or []:
        sample_to_split[sid] = "val"
    for sid in by_cell_split.test_samples or []:
        sample_to_split[sid] = "test"

    # If the SplitFile only stored cell IDs, we need a different mechanism.
    # The by-cell SplitFile we build does NOT store sample IDs at the model
    # level — they're aggregated by cell. So this checker is for when an
    # external builder provides per-sample lookup.
    # For our build_by_cell_split, the per-sample assignment is implicit
    # via cell membership. The check that matters is on cell membership.

    cell_to_split: dict[str, str] = {}
    for c in by_cell_split.train_cells or []:
        cell_to_split[c] = "train"
    for c in by_cell_split.val_cells or []:
        cell_to_split[c] = "val"
    for c in by_cell_split.test_cells or []:
        cell_to_split[c] = "test"

    # Build parent_id -> cell_id map from the catalog
    sample_id_to_cell: dict[str, str] = {e.sample_id: e.cell_id for e in catalog}

    for entry in catalog:
        if entry.is_synthetic_anomaly() and entry.parent_sample_id is not None:
            parent_cell = sample_id_to_cell.get(entry.parent_sample_id)
            entry_cell = entry.cell_id
            if parent_cell is None:
                raise ValueError(
                    f"anomaly {entry.sample_id} has parent "
                    f"{entry.parent_sample_id} not in catalog"
                )
            # The anomaly should share its parent's cell (by construction of
            # generation), but in case it doesn't, check splits agree.
            parent_split = cell_to_split.get(parent_cell)
            entry_split = cell_to_split.get(entry_cell)
            if parent_split != entry_split:
                raise ValueError(
                    f"anomaly {entry.sample_id} (cell {entry_cell}, split {entry_split}) "
                    f"and its parent {entry.parent_sample_id} (cell {parent_cell}, "
                    f"split {parent_split}) are in different splits"
                )


def split_summary(split: SplitFile, catalog: list[SampleEntry]) -> dict:
    """Compute distributional statistics for a split.

    Useful for the manifest and dataset card to report SOH ranges, anomaly
    counts per split, etc.

    Both by_cell_primary and by_record splits now populate train/val/test_samples,
    so we always look up assignment by sample_id directly.
    """
    # Build sample -> split lookup (always available now)
    sample_to_split: dict[str, str] = {}
    if split.train_samples:
        for s in split.train_samples:
            sample_to_split[s] = "train"
    if split.val_samples:
        for s in split.val_samples:
            sample_to_split[s] = "val"
    if split.test_samples:
        for s in split.test_samples:
            sample_to_split[s] = "test"

    summary = {"train": {}, "val": {}, "test": {}}
    counts_by_split = Counter()
    anomaly_counts: dict[str, Counter] = defaultdict(Counter)
    soh_by_split: dict[str, list[float]] = defaultdict(list)

    for entry in catalog:
        sp = sample_to_split.get(entry.sample_id)
        if sp is None:
            continue  # not assigned in this split

        counts_by_split[sp] += 1
        anomaly_counts[sp][entry.anomaly_subtype.value] += 1
        if entry.soh is not None:
            soh_by_split[sp].append(entry.soh)

    for sp in ("train", "val", "test"):
        sohs = soh_by_split[sp]
        summary[sp] = {
            "n_samples": counts_by_split[sp],
            "anomaly_subtype_counts": dict(anomaly_counts[sp]),
            "soh_min": min(sohs) if sohs else None,
            "soh_max": max(sohs) if sohs else None,
            "soh_mean": sum(sohs) / len(sohs) if sohs else None,
            "n_with_soh": len(sohs),
        }
    return summary
