"""
Pytest configuration for MagBridge-Battery v1.0 tests.

Behaviour:
- If the raw upstream files (osf_sequences.npz, osf_sample_metadata.csv,
  pulsebat_lfp.csv) are present in data/v1.0/, all 194 tests run.
- If those files are absent — the default state in the public GitHub repo,
  because MagBridge-Battery does not redistribute raw upstream data — the
  31 tests that need them are skipped with a clear reason, rather than
  erroring out.

This produces a clean "163 passed, 31 skipped, 2 deselected" result in
the public repo (where 2 slow tests are deselected via pytest markers),
without modifying any individual test file.

To obtain the missing files and run the full suite, see
code/bridge/README.md (Missing files section).
"""
from __future__ import annotations
from pathlib import Path

import pytest

_BRIDGE_ROOT = Path(__file__).parent.parent
_DATA_V10 = _BRIDGE_ROOT / "data" / "v1.0"

_UPSTREAM = {
    "osf_sequences":      _DATA_V10 / "osf_sequences.npz",
    "osf_sample_metadata": _DATA_V10 / "osf_sample_metadata.csv",
    "pulsebat_lfp":       _DATA_V10 / "pulsebat_lfp.csv",
}
_MISSING = {name: path for name, path in _UPSTREAM.items() if not path.exists()}

# Exact set of tests that require raw upstream files.
# Format: "{filename}::{test_name}" or "{filename}" for whole-file dependence.
#
# The BridgeV13 fixture in test_bridge.py loads osf_sequences at setup, so
# every test in that file is affected.
_TESTS_REQUIRING_UPSTREAM = {
    # Whole file needs osf_sequences (via the bridge fixture)
    "test_bridge.py": "osf_sequences",
    # Specific tests that load pulsebat/osf directly
    "test_release_readiness.py::test_regime_b_voltage_set_yields_distinct_pseudo_cells": "pulsebat_lfp",
    "test_v1_0_config_sanity.py::test_v1_0_config_is_internally_consistent": "pulsebat_lfp",
}


def pytest_collection_modifyitems(config, items):
    if not _MISSING:
        return  # all upstream files present; full test suite runs

    for item in items:
        # Construct a "filename::testname" key from the pytest nodeid
        # nodeid example: "tests/test_bridge.py::test_bridge_soh_sensitivity"
        nodeid = item.nodeid
        if "::" in nodeid:
            filename_part, test_part = nodeid.split("::", 1)
            filename = filename_part.split("/")[-1]
            full_key = f"{filename}::{test_part}"
        else:
            filename = nodeid.split("/")[-1]
            full_key = filename

        # Check if this exact test or its file is in the required-upstream set
        required = None
        for pattern, file_label in _TESTS_REQUIRING_UPSTREAM.items():
            if "::" in pattern:
                if full_key == pattern:
                    required = file_label
                    break
            else:
                if filename == pattern:
                    required = file_label
                    break

        if required and required in _MISSING:
            item.add_marker(
                pytest.mark.skip(
                    reason=(
                        f"Requires raw upstream file '{_UPSTREAM[required].name}' "
                        f"which is not redistributed (see code/bridge/README.md)."
                    )
                )
            )


def pytest_report_header(config):
    if _MISSING:
        names = ", ".join(sorted(_MISSING.keys()))
        return [
            f"MagBridge upstream files: MISSING ({names}). Affected tests will be skipped.",
            "                          See code/bridge/README.md (Missing files section).",
        ]
    return ["MagBridge upstream files: all present. Full test suite enabled."]
