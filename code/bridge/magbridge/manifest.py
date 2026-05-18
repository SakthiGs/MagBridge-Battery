"""
MagBridge-Battery — dataset manifest builder.

The manifest is the dataset's birth certificate. It records:
  - identity:    name, version, license, schema version
  - provenance:  hashes of input data, bridge code commit, config hash
  - intent:      the full bridge config used (verbatim)
  - contents:    sample counts by category, split summaries
  - citation:    BibTeX template with placeholders for release-time DOI

A future user reading the manifest can verify they have the same dataset
we describe, and can reproduce generation given the same inputs and the
bridge code at the recorded commit.

Reproducibility contract:
  Two generations with the same inputs, bridge code, and config MUST
  produce manifests that differ only in `generated_at_utc`. All other
  fields must be byte-identical. This is what makes the dataset
  reproducible.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from magbridge.schema import (
    DatasetManifest,
    SplitFile,
)
from magbridge.splits import SampleEntry, split_summary


# =============================================================================
# Hash helpers
# =============================================================================
HASH_LENGTH_PROVENANCE = 16   # for file hashes (more bits = less collision risk)
HASH_LENGTH_CONFIG = 12       # for config hashes (still effectively unique)


def hash_file(path: Union[str, Path], length: int = HASH_LENGTH_PROVENANCE) -> str:
    """SHA256 hash of a file, truncated to `length` hex chars.

    Reads the file in chunks so it handles large inputs without loading into memory.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"hash_file: not a file: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def hash_config(config: dict, length: int = HASH_LENGTH_CONFIG) -> str:
    """Hash a config dict via canonicalised JSON.

    Same dict in different key orders -> same hash. Floats are stable.

    NOTE (v1.0): currently called only with the BRIDGE config dict, so the
    resulting `config_hash` field in the manifest captures bridge
    hyperparameters only. The full generation_config.yaml (sample counts,
    anomaly subtype counts, severity range, etc.) is NOT included in the
    hash. Two datasets generated with the same bridge config but different
    sample counts will have identical config_hash values.

    Workaround for users wanting full provenance: also check the
    generated_at_utc + osf_data_hash + pulsebat_data_hash + sample-count
    fields in the manifest. v1.0.1 will add a separate
    generation_config_hash field covering the full YAML.
    """
    blob = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:length]


def resolve_git_commit(repo_root: Optional[Union[str, Path]] = None) -> str:
    """Return the git commit SHA of the bridge code, or 'NOT_IN_GIT' if unavailable.

    Uses subprocess to call `git rev-parse HEAD`. Returns 12-char short SHA.
    Falls back gracefully — release in Colab or zip-extracted environments
    may not have git context. The release script can override this value.
    """
    cmd = ["git", "rev-parse", "--short=12", "HEAD"]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        pass
    return "NOT_IN_GIT"


# =============================================================================
# Citation template
# =============================================================================
CITATION_TEMPLATE = """\
@dataset{{magbridge_battery_v{version_underscore},
  title  = {{MagBridge-Battery v{version}}},
  author = {{Gunasekar, Sakthi Prabhu and Rangarajan, Prasanna Kumar}},
  year   = {{{year}}},
  doi    = {{ZENODO_DOI_HERE}},
  url    = {{ZENODO_URL_HERE}},
  license = {{CC-BY-4.0}},
  note   = {{Schema version {schema_version}; generated {generated_at_utc}}},
}}"""


def render_citation(version: str, schema_version: str, generated_at_utc: str) -> str:
    """Render the BibTeX citation template with placeholders for release DOI/URL."""
    version_underscore = version.replace(".", "_")
    year = generated_at_utc[:4]
    return CITATION_TEMPLATE.format(
        version=version,
        version_underscore=version_underscore,
        year=year,
        schema_version=schema_version,
        generated_at_utc=generated_at_utc,
    )


# =============================================================================
# Manifest builder
# =============================================================================
def build_manifest(
    *,
    catalog: list[SampleEntry],
    bridge_config: dict,
    bridge_version: str,
    osf_data_path: Union[str, Path],
    pulsebat_data_path: Union[str, Path],
    by_cell_split: Optional[SplitFile] = None,
    by_record_split: Optional[SplitFile] = None,
    repo_root: Optional[Union[str, Path]] = None,
    bridge_code_commit_override: Optional[str] = None,
) -> DatasetManifest:
    """Build a DatasetManifest from a generated catalog and bridge inputs.

    Args:
        catalog: list of SampleEntry (the generated samples).
        bridge_config: dict of bridge hyperparameters (drift strength, cone
                       angle, etc.). Embedded verbatim in the manifest.
        bridge_version: e.g. 'v1.3'.
        osf_data_path: path to the OSF sequences file used for generation.
        pulsebat_data_path: path to the PulseBat CSV used for generation.
        by_cell_split: optional SplitFile; if provided, split_summary is embedded.
        by_record_split: optional SplitFile; if provided, split_summary is embedded.
        repo_root: directory containing the bridge code git repo (for commit lookup).
        bridge_code_commit_override: explicit commit SHA (release-time override).

    Returns:
        A validated DatasetManifest model. Caller serialises to JSON.

    Raises:
        ValueError: if catalog sample counts don't agree with config, etc.
        FileNotFoundError: if input data paths are not readable.
    """
    # Provenance hashes
    osf_hash = hash_file(osf_data_path)
    pulsebat_hash = hash_file(pulsebat_data_path)
    config_h = hash_config(bridge_config)

    if bridge_code_commit_override:
        bridge_commit = bridge_code_commit_override
    else:
        bridge_commit = resolve_git_commit(repo_root)

    # Sample counts from catalog (ground truth — not from config)
    n_clean = sum(1 for e in catalog if e.is_clean_grounded())
    n_synth = sum(1 for e in catalog if e.is_synthetic_anomaly())
    n_regime_b = sum(1 for e in catalog if e.is_regime_b())
    n_total = len(catalog)

    if n_clean + n_synth + n_regime_b != n_total:
        raise ValueError(
            f"Catalog category counts ({n_clean}+{n_synth}+{n_regime_b}={n_clean+n_synth+n_regime_b}) "
            f"do not sum to total ({n_total}). Some samples have unexpected anomaly_origin."
        )

    # Timestamp
    now = datetime.now(timezone.utc).replace(microsecond=0)
    generated_at_utc = now.isoformat().replace("+00:00", "Z")

    # Per-split summaries (optional, embedded if splits provided)
    extra_summaries: dict = {}
    if by_cell_split is not None:
        extra_summaries["by_cell_primary"] = split_summary(by_cell_split, catalog)
    if by_record_split is not None:
        extra_summaries["by_record_optimistic_baseline"] = split_summary(by_record_split, catalog)

    # Citation
    citation = render_citation(
        version="1.0",
        schema_version="1.0",
        generated_at_utc=generated_at_utc,
    )

    # Per-channel descriptions (acknowledges that C5/C6 are not strict magnitudes).
    signal_channels = {
        "B_s1Y": "Sensor 1, Y component of magnetic field (nT, signed; from OSF source data).",
        "B_s1Z": "Sensor 1, Z component of magnetic field (nT, signed; from OSF source data).",
        "B_s2Y": "Sensor 2, Y component of magnetic field (nT, signed; from OSF source data).",
        "B_s2Z": "Sensor 2, Z component of magnetic field (nT, signed; from OSF source data).",
        "B_s1C5": (
            "5th channel from the OSF source (originally labelled 'Mag' in OSF). "
            "Values CAN BE NEGATIVE — this is not a strict magnitude. We name it 'C5' "
            "(channel-5) to avoid implying a sqrt(Y^2+Z^2) interpretation. "
            "Users wanting a non-negative magnitude can compute sqrt(B_s1Y^2 + B_s1Z^2) themselves."
        ),
        "B_s2C6": (
            "6th channel from the OSF source (originally labelled 'Mag' in OSF). "
            "Same caveats as B_s1C5 — values can be negative; not a strict magnitude."
        ),
    }

    manifest = DatasetManifest(
        dataset_name="MagBridge-Battery",
        dataset_version="1.0",
        schema_version="1.0",
        license_dataset="CC-BY-4.0",
        license_code="Apache-2.0",
        generated_at_utc=generated_at_utc,
        n_total_samples=n_total,
        n_clean_grounded_samples=n_clean,
        n_synthetic_anomaly_samples=n_synth,
        n_regime_b_extrapolation_samples=n_regime_b,
        osf_data_hash=osf_hash,
        pulsebat_data_hash=pulsebat_hash,
        bridge_code_commit=bridge_commit,
        config_hash=config_h,
        bridge_version=bridge_version,
        bridge_config=bridge_config,
        signal_channels=signal_channels,
        citation=citation,
    )

    # Attach extras as a non-schema field for downstream consumption.
    # The DatasetManifest model doesn't include split summaries by design
    # (they're a release artifact, not core provenance), so we attach via
    # a sidecar dict that the release script writes alongside the manifest.
    # See `write_manifest_with_summaries` below for the canonical writer.
    manifest._split_summaries = extra_summaries  # type: ignore[attr-defined]
    return manifest


# =============================================================================
# Manifest writer
# =============================================================================
def write_manifest_with_summaries(
    manifest: DatasetManifest,
    output_path: Union[str, Path],
    *,
    include_summaries: bool = True,
) -> Path:
    """Serialise a manifest to JSON, optionally with split summaries attached.

    The DatasetManifest model itself doesn't include split summaries (they're
    a release artifact). This writer combines the model fields with the
    sidecar summaries dict attached by build_manifest, producing a single
    self-contained manifest.json.

    Args:
        manifest: the DatasetManifest model.
        output_path: where to write the JSON.
        include_summaries: whether to embed per-split summaries.

    Returns:
        the Path written to.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dump model to dict
    data = manifest.model_dump(mode="json")

    # Attach split summaries if present and requested
    if include_summaries:
        summaries = getattr(manifest, "_split_summaries", None)
        if summaries:
            data["split_summaries"] = summaries

    # Write with stable formatting (sorted keys, 2-space indent)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")  # trailing newline for unix-friendly files

    return output_path


# =============================================================================
# Consistency checks (validator helpers)
# =============================================================================
def verify_catalog_consistent_with_config(
    catalog: list[SampleEntry],
    expected_counts: dict,
) -> None:
    """Verify that the catalog matches the count expectations from the config.

    Args:
        catalog: the SampleEntry list.
        expected_counts: dict with keys:
            'n_clean_grounded_samples'
            'n_synthetic_anomaly_samples'
            'n_regime_b_extrapolation_samples'
            'total_expected_samples'

    Raises:
        ValueError: on first mismatch found.
    """
    n_clean = sum(1 for e in catalog if e.is_clean_grounded())
    n_synth = sum(1 for e in catalog if e.is_synthetic_anomaly())
    n_regime_b = sum(1 for e in catalog if e.is_regime_b())
    n_total = len(catalog)

    expected_clean = expected_counts["n_clean_grounded_samples"]
    expected_synth = expected_counts["n_synthetic_anomaly_samples"]
    expected_regime_b = expected_counts["n_regime_b_extrapolation_samples"]
    expected_total = expected_counts["total_expected_samples"]

    if n_clean != expected_clean:
        raise ValueError(
            f"clean grounded count mismatch: catalog has {n_clean}, config expects {expected_clean}"
        )
    if n_synth != expected_synth:
        raise ValueError(
            f"synthetic anomaly count mismatch: catalog has {n_synth}, config expects {expected_synth}"
        )
    if n_regime_b != expected_regime_b:
        raise ValueError(
            f"regime-B count mismatch: catalog has {n_regime_b}, config expects {expected_regime_b}"
        )
    if n_total != expected_total:
        raise ValueError(
            f"total count mismatch: catalog has {n_total}, config expects {expected_total}"
        )
