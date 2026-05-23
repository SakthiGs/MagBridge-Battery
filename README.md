# MagBridge-Battery

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Data license: CC-BY-4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20260147-blue)](https://doi.org/10.5281/zenodo.20260147)
[![arXiv](https://img.shields.io/badge/arXiv-2605.20240-b31b1b.svg)](https://arxiv.org/abs/2605.20240)

This is the code repository for **MagBridge-Battery: A Synthetic Bridge Dataset for Li-ion Magnetometry and State-of-Health Diagnostics** by Sakthi Prabhu Gunasekar and Prasanna Kumar Rangarajan (2026).

> **The dataset itself (40 MB of Parquet shards plus metadata) is on Zenodo, not in this repo.** See [Dataset access](#dataset-access) below.

This repo contains:
- The paper source (LaTeX + bibliography)
- The bridge implementation that produced the dataset
- Reproducible benchmark scripts (classical + deep-learning baselines)
- Documentation (dataset card, citation guidance, license notices)

---

## Quick start

```bash
git clone https://github.com/SakthiGs/MagBridge-Battery.git
cd MagBridge-Battery

# Get the dataset from Zenodo (one-time, ~40MB):
mkdir -p data
wget https://zenodo.org/records/20260147/files/magbridge_battery_v1_0_release.zip
unzip magbridge_battery_v1_0_release.zip -d data/
# After this, you should have data/data/shard_*.parquet and data/splits/*.json

# Install dependencies
pip install -r requirements.txt

# Run the classical benchmark (Table III in the paper)
python3 code/run_benchmark.py --data-dir ./data

# Run the deep-learning benchmark (Table IV), one task at a time
python3 code/run_dl_bench_lean.py t1 --data-dir ./data   # SOH regression
python3 code/run_dl_bench_lean.py t2 --data-dir ./data   # Second-life classification
python3 code/run_dl_bench_lean.py t3 --data-dir ./data   # Anomaly detection (3-class)
python3 code/run_dl_bench_lean.py t4 --data-dir ./data   # Anomaly subtype (4-class)
```

You can also set `MAGBRIDGE_DATA=./data` as an environment variable to avoid passing `--data-dir` every time.

---

## Dataset access

The 6,760-sample MagBridge-Battery v1.0 dataset is archived on Zenodo:

- **DOI:** [10.5281/zenodo.20260147](https://doi.org/10.5281/zenodo.20260147)
- **License:** CC-BY-4.0
- **Size:** ~40 MB (5 Parquet shards + metadata + splits + manifest + docs)
- **Format:** Parquet (loadable via `pandas.read_parquet`)

The Zenodo bundle is self-contained — it includes a `load_example.py` if you only want the data and not this repo.

---

## What's in this repo

```
MagBridge-Battery/
├── paper/                              The paper (LaTeX source + compiled PDF)
│   ├── magbridge_battery.tex
│   ├── magbridge_battery.pdf
│   └── references.bib
├── code/                               Code
│   ├── run_benchmark.py                Classical-ML benchmark (Table III)
│   ├── run_dl_bench_lean.py            Deep-learning benchmark (Table IV)
│   └── bridge/                         The bridge implementation
│       ├── magbridge/                  Python package (3,731 LOC)
│       ├── tests/                      194 unit tests
│       ├── configs/                    Generation config
│       ├── data/v1.0/                  OSF-derived aggregate statistics
│       ├── notebooks/                  Colab orchestrator
│       └── README.md                   Package usage and missing-files note
├── docs/                               Documentation
│   ├── dataset_card.md                 Full dataset card
│   └── CITING.md                       Citation guidance
├── requirements.txt                    Python dependencies (root level)
├── CITATION.cff                        Machine-readable citation; GitHub renders the "Cite this repository" button from this
├── LICENSE                             Apache-2.0 (covers code)
├── LICENSE-DATA                        CC-BY-4.0 reference (data is on Zenodo)
├── NOTICE-PULSEBAT                     Upstream MIT notice from PulseBat
├── NOTICE-OSF                          Upstream attribution for OSF magnetometry archive
└── README.md                           This file
```

---

## Reproducing the paper's numbers

All numbers in the paper come from the scripts in `code/`. To reproduce:

**Table III (classical baselines):**
```bash
python3 code/run_benchmark.py
```
Expected output: T1 R² ≈ 0.675, T2 bal_acc ≈ 0.907, T3 bal_acc ≈ 0.789, T4 bal_acc ≈ 0.725 (within seed variance).
Time: ~5 minutes on a laptop CPU.

**Table IV (deep-learning baselines):**
```bash
for task in t1 t2 t3 t4; do
  python3 code/run_dl_bench_lean.py $task
done
```
Expected output: best DL per task as reported in Table IV.
Time: ~25 minutes on a laptop CPU; ~5 minutes on a Colab T4 GPU.

Both scripts use the same `by_cell_primary` cell-disjoint split as defined in the paper, with the same feature extraction pipeline. Seed variation is bounded by repeated cell-subsampling (each seed sees a different 80% of the training cells).

---

## How to cite

If you use MagBridge-Battery in your work, please cite **both** the paper and the dataset DOI. The paper describes the bridge construction, validation, and benchmark protocol; the dataset DOI uniquely identifies the v1.0 data artifact.

**Paper:**
```bibtex
@article{magbridge2026,
  author        = {Gunasekar, Sakthi Prabhu and Rangarajan, Prasanna Kumar},
  title         = {{MagBridge-Battery}: A Synthetic Bridge Dataset for
                   {Li}-ion Magnetometry and State-of-Health Diagnostics},
  journal       = {arXiv preprint},
  eprint        = {2605.20240},
  archivePrefix = {arXiv},
  year          = {2026}
}
```

**Dataset:**
```bibtex
@misc{magbridge_battery_v1_0,
  author    = {Gunasekar, Sakthi Prabhu and Rangarajan, Prasanna Kumar},
  title     = {{MagBridge-Battery v1.0}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20260147}
}
```

See `docs/CITING.md` for full guidance including upstream-source citations (OSF magnetometry archive, PulseBat dataset).

---

## Licenses

- **Code in this repo** (everything under `code/`, `paper/`, `docs/`): [Apache-2.0](LICENSE)
- **Dataset on Zenodo** (the Parquet shards and metadata): [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)
- **Upstream sources:**
  - OSF magnetometry archive (Mohammadi–Jerschow, [DOI 10.17605/OSF.IO/CW8ZV](https://doi.org/10.17605/OSF.IO/CW8ZV)) — used to derive aggregate statistics; no raw data redistributed.
  - PulseBat dataset (Tao et al., [DOI 10.5281/zenodo.13360631](https://doi.org/10.5281/zenodo.13360631)) — used for conditioning labels; declared CC-BY-4.0 on Zenodo and MIT on the [code repo](https://github.com/terencetaothucb/Pulse-Voltage-Response-Generation). See `NOTICE-PULSEBAT`.

MagBridge-Battery does not redistribute raw upstream data. See the LICENSE file in the Zenodo bundle for full details.

---

## Issues, questions, contributions

- **Bug in the data or paper?** Open an issue on GitHub or email the authors (see the paper or CITATION.cff).
- **Want to contribute a benchmark result or method?** Open an issue or PR. Method submissions that beat the Table III / Table IV baselines are particularly welcome.
- **Want to use the dataset commercially?** CC-BY-4.0 permits this; please cite the paper and dataset DOI per [How to cite](#how-to-cite) above.

---

## Related work

- The Mohammadi–Jerschow [OSF magnetometry archive](https://osf.io/cw8zv/) (the upstream magnetic morphology source).
- The [PulseBat dataset](https://github.com/terencetaothucb/Pulse-Voltage-Response-Generation) (Tao et al., 2024) (the upstream SOH/SOC label source).
- The [QuaLiProM consortium](https://www.ifam.fraunhofer.de/) (BMBF, 2024–2026) — building paired magnetic–electrochemical data privately; MagBridge-Battery is a public stand-in until that data is released.

---

*This repository accompanies the v1.0 release on Zenodo (DOI: 10.5281/zenodo.20260147) and the arXiv preprint (arXiv:2605.20240).*
