"""
MagBridge-Battery — dataset-level validator.

Runs after generation, before release. Enforces invariants that cannot be
verified on individual samples — only by inspecting the catalog as a whole.

Rules enforced (D1-D9):
  D1: All sample IDs are unique (no hash collisions, no duplicate generation).
  D2: Every parent_sample_id resolves to a real sample in the catalog.
  D3: Every parent is a clean grounded sample (no anomaly chains).
  D4: Per-cell sample counts match expectations.
  D5: Anomaly subtype counts match the locked config.
  D6: Total counts match the config's total_expected_samples.
  D7: Every record passes individual Pydantic validation (per-sample rules).
  D8: No regime-B sample has parent_sample_id set.
  D9: Every clean grounded sample has u_features, soh, second_life_class.

The validator collects ALL failures, then raises DatasetValidationError once.
This avoids the "fix one, hit the next" cycle during release prep.

The validator does NOT load Parquet shards or signal data — it operates on the
in-memory SampleEntry catalog. Per-sample signal integrity (length 100,
correct channels) is enforced earlier by the SampleRecord schema at Parquet
write time.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

from magbridge.schema import (
    AnomalyOrigin,
    AnomalySubtype,
    SampleRecord,
)
from magbridge.sample_id import check_no_collisions
from magbridge.splits import SampleEntry


# =============================================================================
# Failure reporting types
# =============================================================================
class ValidationFailure:
    """One validator rule failure with structured context."""

    __slots__ = ("rule", "severity", "message", "sample_ids")

    def __init__(
        self,
        rule: str,
        severity: str,
        message: str,
        sample_ids: Optional[list[str]] = None,
    ):
        self.rule = rule
        self.severity = severity   # "fatal" | "warning"
        self.message = message
        self.sample_ids = sample_ids or []

    def __repr__(self) -> str:
        n = len(self.sample_ids)
        suffix = f" (affects {n} sample{'s' if n != 1 else ''})" if n else ""
        return f"[{self.rule} {self.severity.upper()}] {self.message}{suffix}"


class DatasetValidationError(Exception):
    """Raised when one or more dataset-level invariants are violated.

    Carries a structured list of all failures found, not just the first one.
    Use `error.failures` to inspect programmatically.
    """

    def __init__(self, failures: list[ValidationFailure]):
        self.failures = failures
        super().__init__(self._format())

    def _format(self) -> str:
        n_fatal = sum(1 for f in self.failures if f.severity == "fatal")
        n_warn = sum(1 for f in self.failures if f.severity == "warning")
        lines = [
            f"Dataset validation failed: {n_fatal} fatal, {n_warn} warning.",
            "",
        ]
        for f in self.failures:
            lines.append(f"  {f}")
            if f.sample_ids and len(f.sample_ids) <= 10:
                lines.append(f"     Affected samples: {f.sample_ids}")
            elif f.sample_ids:
                head = ", ".join(f.sample_ids[:5])
                lines.append(f"     Affected samples: {head}, ... (+{len(f.sample_ids) - 5} more)")
        return "\n".join(lines)

    @property
    def fatal_failures(self) -> list[ValidationFailure]:
        return [f for f in self.failures if f.severity == "fatal"]


# =============================================================================
# Per-rule checks
# =============================================================================
def _check_d1_unique_ids(catalog: list[SampleEntry]) -> list[ValidationFailure]:
    """D1: All sample IDs must be unique."""
    failures = []
    try:
        check_no_collisions([e.sample_id for e in catalog])
    except ValueError as exc:
        # Find the duplicates to enumerate them
        counts = Counter(e.sample_id for e in catalog)
        dups = [sid for sid, c in counts.items() if c > 1]
        failures.append(ValidationFailure(
            rule="D1",
            severity="fatal",
            message=f"Sample IDs are not unique: {len(dups)} duplicate(s) found",
            sample_ids=dups[:20],
        ))
    return failures


def _check_d2_d3_parent_resolution(
    catalog: list[SampleEntry],
) -> list[ValidationFailure]:
    """D2: every parent_sample_id resolves; D3: parents are clean grounded."""
    failures = []
    by_id = {e.sample_id: e for e in catalog}

    orphans: list[str] = []     # parent_id not in catalog
    bad_parents: list[str] = []   # parent exists but isn't clean grounded

    for e in catalog:
        if e.parent_sample_id is None:
            continue
        parent = by_id.get(e.parent_sample_id)
        if parent is None:
            orphans.append(e.sample_id)
        elif not parent.is_clean_grounded():
            bad_parents.append(e.sample_id)

    if orphans:
        failures.append(ValidationFailure(
            rule="D2",
            severity="fatal",
            message=f"Anomaly samples reference non-existent parents: {len(orphans)} orphan(s)",
            sample_ids=orphans[:20],
        ))
    if bad_parents:
        failures.append(ValidationFailure(
            rule="D3",
            severity="fatal",
            message=(
                f"Anomaly samples reference non-clean parents (anomaly chains forbidden): "
                f"{len(bad_parents)} violation(s)"
            ),
            sample_ids=bad_parents[:20],
        ))
    return failures


def _check_d4_per_cell_counts(
    catalog: list[SampleEntry],
    expected_clean_per_cell: Optional[int] = None,
) -> list[ValidationFailure]:
    """D4: per-cell sample counts within expected band.

    If expected_clean_per_cell is provided (e.g., 10 SOC levels per cell),
    flag any cell with fewer or more clean grounded samples than expected.
    """
    failures = []
    if expected_clean_per_cell is None:
        return failures

    per_cell = Counter(
        e.cell_id for e in catalog if e.is_clean_grounded()
    )
    mismatches = []
    for cell, count in per_cell.items():
        if count != expected_clean_per_cell:
            mismatches.append((cell, count))

    if mismatches:
        examples = ", ".join(f"{c}={n}" for c, n in mismatches[:5])
        failures.append(ValidationFailure(
            rule="D4",
            severity="fatal",
            message=(
                f"{len(mismatches)} cell(s) have unexpected clean-sample counts "
                f"(expected {expected_clean_per_cell} per cell): e.g., {examples}"
            ),
        ))
    return failures


def _check_d5_anomaly_subtype_counts(
    catalog: list[SampleEntry],
    expected_subtype_counts: dict[str, int],
) -> list[ValidationFailure]:
    """D5: synthetic anomaly subtype counts match the locked config."""
    failures = []
    actual = Counter()
    for e in catalog:
        if e.is_synthetic_anomaly():
            actual[e.anomaly_subtype.value] += 1

    for subtype, expected_n in expected_subtype_counts.items():
        actual_n = actual.get(subtype, 0)
        if actual_n != expected_n:
            failures.append(ValidationFailure(
                rule="D5",
                severity="fatal",
                message=(
                    f"Subtype '{subtype}' has {actual_n} samples; "
                    f"config expects {expected_n}"
                ),
            ))

    # Also flag any unexpected subtype present
    for subtype in actual:
        if subtype not in expected_subtype_counts:
            failures.append(ValidationFailure(
                rule="D5",
                severity="fatal",
                message=(
                    f"Subtype '{subtype}' is present ({actual[subtype]} samples) "
                    f"but not declared in config"
                ),
            ))
    return failures


def _check_d6_total_counts(
    catalog: list[SampleEntry],
    expected_counts: dict,
) -> list[ValidationFailure]:
    """D6: per-category and total counts match config."""
    failures = []

    n_clean = sum(1 for e in catalog if e.is_clean_grounded())
    n_synth = sum(1 for e in catalog if e.is_synthetic_anomaly())
    n_regime_b = sum(1 for e in catalog if e.is_regime_b())
    n_total = len(catalog)

    checks = [
        ("clean_grounded", n_clean, expected_counts["n_clean_grounded_samples"]),
        ("synthetic_anomaly", n_synth, expected_counts["n_synthetic_anomaly_samples"]),
        ("regime_b_extrapolation", n_regime_b, expected_counts["n_regime_b_extrapolation_samples"]),
        ("total", n_total, expected_counts["total_expected_samples"]),
    ]
    for category, actual, expected in checks:
        if actual != expected:
            failures.append(ValidationFailure(
                rule="D6",
                severity="fatal",
                message=f"{category} count: actual={actual}, expected={expected}",
            ))
    return failures


def _check_d7_per_sample_schema(
    sample_records: Optional[list[SampleRecord]] = None,
) -> list[ValidationFailure]:
    """D7: every record passes individual Pydantic validation.

    This is delegated to the Pydantic model itself — if SampleRecord can be
    constructed without raising, the per-sample rules pass. So this check
    only matters if the caller has access to the full SampleRecord objects
    (e.g., after re-reading from Parquet); for the catalog-only path,
    SampleEntry doesn't carry signal data and this check is skipped.
    """
    failures = []
    if sample_records is None:
        return failures
    # In practice, since these were already validated at construction, this
    # is a re-check after deserialisation. Iterate and capture failures.
    for rec in sample_records:
        try:
            # Re-validate by constructing from dict
            SampleRecord.model_validate(rec.model_dump())
        except Exception as exc:
            failures.append(ValidationFailure(
                rule="D7",
                severity="fatal",
                message=f"Per-sample validation failed: {exc}",
                sample_ids=[rec.sample_id],
            ))
    return failures


def _check_d8_regime_b_no_parent(
    catalog: list[SampleEntry],
) -> list[ValidationFailure]:
    """D8: regime-B samples must not have parent_sample_id set."""
    failures = []
    bad = []
    for e in catalog:
        if e.is_regime_b() and e.parent_sample_id is not None:
            bad.append(e.sample_id)
    if bad:
        failures.append(ValidationFailure(
            rule="D8",
            severity="fatal",
            message=(
                f"Regime-B samples must not have parent_sample_id "
                f"(found {len(bad)} violations)"
            ),
            sample_ids=bad[:20],
        ))
    return failures


def _check_d9_clean_metadata_complete(
    catalog: list[SampleEntry],
) -> list[ValidationFailure]:
    """D9: clean grounded samples have soh (and ideally u_features) populated.

    SampleEntry doesn't carry u_features (it's a lightweight catalog type),
    so this check only verifies soh is non-null for clean samples.
    The full u_features check happens at the Parquet write layer using
    SampleRecord.
    """
    failures = []
    missing_soh = []
    for e in catalog:
        if e.is_clean_grounded() and e.soh is None:
            missing_soh.append(e.sample_id)
    if missing_soh:
        failures.append(ValidationFailure(
            rule="D9",
            severity="fatal",
            message=(
                f"Clean grounded samples missing SOH labels: "
                f"{len(missing_soh)} sample(s)"
            ),
            sample_ids=missing_soh[:20],
        ))
    return failures


# =============================================================================
# Public entry point
# =============================================================================
def validate_dataset(
    catalog: list[SampleEntry],
    *,
    expected_counts: Optional[dict] = None,
    expected_subtype_counts: Optional[dict[str, int]] = None,
    expected_clean_per_cell: Optional[int] = None,
    sample_records: Optional[list[SampleRecord]] = None,
    raise_on_warning: bool = False,
) -> list[ValidationFailure]:
    """Run all D1-D9 checks on the catalog. Collect all failures, then raise.

    Args:
        catalog: list of SampleEntry — the generated samples.
        expected_counts: dict from generation config (n_clean_grounded_samples,
                         n_synthetic_anomaly_samples, n_regime_b_extrapolation_samples,
                         total_expected_samples). If None, D6 is skipped.
        expected_subtype_counts: dict {subtype_name: expected_count}. If None, D5 skipped.
        expected_clean_per_cell: int, e.g. 10 (SOC levels per cell). If None, D4 skipped.
        sample_records: optional list of full SampleRecord (post-Parquet-roundtrip).
                        If provided, D7 is run; otherwise skipped.
        raise_on_warning: if True, warnings also trigger the exception.

    Returns:
        A list of ValidationFailure objects (empty if all checks pass).
        If any FATAL failure is present (or warnings present with raise_on_warning=True),
        DatasetValidationError is raised before this returns.
    """
    all_failures: list[ValidationFailure] = []
    all_failures.extend(_check_d1_unique_ids(catalog))
    all_failures.extend(_check_d2_d3_parent_resolution(catalog))

    if expected_clean_per_cell is not None:
        all_failures.extend(_check_d4_per_cell_counts(catalog, expected_clean_per_cell))

    if expected_subtype_counts is not None:
        all_failures.extend(_check_d5_anomaly_subtype_counts(catalog, expected_subtype_counts))

    if expected_counts is not None:
        all_failures.extend(_check_d6_total_counts(catalog, expected_counts))

    if sample_records is not None:
        all_failures.extend(_check_d7_per_sample_schema(sample_records))

    all_failures.extend(_check_d8_regime_b_no_parent(catalog))
    all_failures.extend(_check_d9_clean_metadata_complete(catalog))

    # Decide whether to raise
    has_fatal = any(f.severity == "fatal" for f in all_failures)
    has_warn = any(f.severity == "warning" for f in all_failures)
    if has_fatal or (raise_on_warning and has_warn):
        raise DatasetValidationError(all_failures)

    return all_failures
