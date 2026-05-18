# MagBridge-Battery bridge implementation (v1.3)

This directory contains the bridge code that produced **MagBridge-Battery v1.0** — the `BridgeV13` class, its supporting modules, the generation orchestrator, the validation/manifest/splits infrastructure, and 194 unit tests.

The committed code commit is `MAGBRIDGE_V1_0_PHASE2_FIXED_20260515`, matching the `bridge_code_commit` field in the released dataset's `manifest.json`.

---

## When to read this

You probably **don't** need the bridge code if you only want to use the dataset for benchmarking — the released v1.0 dataset on Zenodo is fully self-contained, and the parent repo's `run_benchmark.py` / `run_dl_bench_lean.py` scripts will reproduce the headline numbers from the paper without touching this directory.

You **do** need the bridge code if you want to:

- Re-generate the dataset from scratch (with different seeds, different sample counts, or different conditioning)
- Audit the bridge architecture in detail beyond the paper appendix
- Implement an extension (e.g. v1.1 with chemistry transfer, or a learned decoder replacement)
- Verify our reported sanity invariants and ablation results

---

## Layout

```
bridge/
├── magbridge/                       The Python package
│   ├── __init__.py
│   ├── schema.py                    Pydantic schemas for samples, metadata, manifest
│   ├── bridge/
│   │   ├── bridge.py                BridgeV13 — the core architecture
│   │   ├── morphology.py            Per-anchor morphology bank
│   │   └── qrec.py                  Quantum reservoir embedding (MagBridge-Embed)
│   ├── anomalies.py                 4 anomaly subtype injectors
│   ├── generate.py                  Orchestrator that produces the full v1.0 release
│   ├── manifest.py                  Manifest assembly + hash pinning
│   ├── sample_id.py                 Deterministic sample-ID generation
│   ├── splits.py                    Cell-disjoint + leaky split builders
│   └── validator.py                 Integrity audits (no NaN, no leakage, etc.)
├── tests/                           194 unit tests; pytest tests/
├── configs/
│   └── generation_config.yaml       The locked v1.0 generation configuration
├── notebooks/
│   └── magbridge_v1_0_generate.ipynb   Colab orchestrator (uploads project ZIP, runs generate.py)
├── data/v1.0/                       Derived statistical artifacts (see "Missing files" below)
│   ├── anchor_stats.npz             Per-anchor mean trajectories + std (OSF-derived aggregate)
│   ├── lda_fit.npz                  Fitted LDA model (5 anchor classes → 4-D subspace)
│   └── qrec_embeddings.npz          Precomputed MagBridge-Embed of OSF samples (171-D)
└── pytest.ini
```

---

## Missing files (must be obtained separately)

To run `generate.py` end-to-end, you need three additional files that are **not redistributed** here for license reasons:

| File | Source | Reason not shipped |
|---|---|---|
| `data/v1.0/osf_sequences.npz` | OSF magnetometry archive (DOI [10.17605/OSF.IO/CW8ZV](https://doi.org/10.17605/OSF.IO/CW8ZV)) | OSF project lists "No License declared"; raw signals are not redistributed by MagBridge-Battery |
| `data/v1.0/osf_sample_metadata.csv` | OSF magnetometry archive | Same as above |
| `data/v1.0/pulsebat_lfp.csv` | PulseBat dataset (DOI [10.5281/zenodo.13360631](https://doi.org/10.5281/zenodo.13360631)) | Raw PulseBat data not redistributed; users obtain directly from upstream under their license |

Each of those upstream sources is freely accessible — see the parent repo's `LICENSE` file and the published paper for upstream citation details.

The three files that **are** shipped (`anchor_stats.npz`, `lda_fit.npz`, `qrec_embeddings.npz`) are derived statistical artifacts: per-anchor means and variances, the LDA fit, and 171-D embeddings. These compress and transform the upstream data; they cannot be inverted back to raw signals.

---

## Quick start

### Run the test suite

```bash
cd code/bridge
pip install -r ../../requirements.txt
PYTHONPATH=. python3 -m pytest tests/ -q
```

In the public GitHub repo, expect the tests requiring raw upstream files to be skipped (see the "Missing files" section above). A clean run reports approximately **163 passed, 31 skipped, 2 deselected** in about 5 seconds. The 31 skipped tests are auto-skipped by `tests/conftest.py` with a clear reason ("raw upstream file X not redistributed"). If you obtain the three missing upstream files and place them in `data/v1.0/`, the full 194-test suite runs.

### Inspect the bridge architecture

```python
import sys; sys.path.insert(0, "code/bridge")
from magbridge.bridge.bridge import BridgeV13, BridgeConfig
from magbridge.schema import SIGNAL_CHANNEL_NAMES, AnomalySubtype

cfg = BridgeConfig()
print(f"Default drift strength: {cfg.qrec_drift_strength}")     # → 800.0
print(f"Signal channels: {SIGNAL_CHANNEL_NAMES}")
print(f"Anomaly subtypes: {[s.value for s in AnomalySubtype]}")
```

### Regenerate the v1.0 dataset

After obtaining the three missing upstream files and placing them in `data/v1.0/`:

```bash
cd code/bridge
PYTHONPATH=. python3 -m magbridge.generate \
    --config configs/generation_config.yaml \
    --data-dir data/v1.0 \
    --output-dir /tmp/magbridge_output \
    --bridge-code-commit MAGBRIDGE_V1_0_PHASE2_FIXED_20260515
```

Expected runtime: ~6 hours on a Colab T4 GPU. The generation produces 6,760 samples (5,600 grounded + 600 anomaly + 560 Regime-B), the two benchmark splits, the manifest, and SHA-256 checksums — bit-identical to the Zenodo release.

---

## How this maps to the paper

| Paper section | Code location |
|---|---|
| §III. Bridge architecture | `magbridge/bridge/bridge.py` (`BridgeV13` class) |
| §III. Morphology bank | `magbridge/bridge/morphology.py` |
| §III. MagBridge-Embed (171-D QRC embedding) | `magbridge/bridge/qrec.py` |
| §III. Regime classifier | inside `BridgeV13.classify_regime()` |
| §III. Synthetic anomalies (4 subtypes) | `magbridge/anomalies.py` |
| §IV. Sanity invariants | `magbridge/validator.py` |
| §IV. Benchmark splits | `magbridge/splits.py` |
| Appendix A. Bridge equations | Tracked in `bridge.py` docstrings and inline comments |
| Dataset manifest fields | `magbridge/manifest.py` |

The `bridge_config_hash` field is computed from the `BridgeConfig` hyperparameters used by the bridge. It does not hash the full `generation_config.yaml`; sample counts, anomaly counts, and split settings are documented separately in the config and manifest. Two datasets generated with the same bridge hyperparameters but different sample counts will share a `bridge_config_hash`; users wanting full provenance should also compare `generated_at_utc`, `osf_data_hash`, `pulsebat_data_hash`, and the sample-count fields.

---

## License

This bridge code is released under **Apache-2.0** (see top-level `LICENSE` file). The derived statistical artifacts in `data/v1.0/` are also Apache-2.0 because they are creative compressions of upstream data, not raw upstream content.
