# Contributing to MagBridge-Battery

Thank you for your interest in MagBridge-Battery. We welcome bug reports, dataset issues, benchmark contributions, and method papers built on the dataset.

## Ways to contribute

### Reporting a dataset issue

If you find a bug or inconsistency in the v1.0 dataset (incorrect labels, broken sample IDs, schema violations, anything that fails an integrity check), please open a [GitHub issue](https://github.com/SakthiGs/MagBridge-Battery/issues) with:

- The specific `sample_id`(s) affected, if applicable
- The expected vs. observed behaviour
- The script or code that reproduces the issue

We track these for the v1.1 patch release.

### Reporting a benchmark result

If you've run a new method on the MagBridge-Battery v1.0 benchmarks (T1 SOH regression, T2 second-life classification, T3 anomaly detection, T4 anomaly subtype), we'd love to know — open an issue tagged `benchmark` with:

- The method name and citation
- The exact metric (e.g. R² for T1, balanced accuracy for T2/T3/T4)
- The split used (must be `by_cell_primary` for primary benchmark comparison)
- A link to the implementation if public

We may compile community benchmark results into future paper versions or a leaderboard.

### Contributing code

Pull requests are welcome for:

- Bug fixes in the bridge or benchmark scripts
- Additional benchmark scripts for new tasks
- Improved documentation or examples
- Reproducibility improvements (e.g. Docker containers, pinned dependency snapshots)

Before opening a PR:

1. Run the test suite: `cd code/bridge && PYTHONPATH=. python3 -m pytest tests/ -q`. Without raw upstream files, expect approximately 163 passed, 31 skipped, 2 deselected. With raw upstream files present, the full 194-test suite should pass.
2. Run the benchmark scripts: `python3 code/run_benchmark.py` and `python3 code/run_dl_bench_lean.py t1`. Numbers should match Table III / Table IV in the paper within seed variance.
3. Keep changes focused; one logical change per PR.

### Proposing v1.1 / future versions

For larger changes — extending to other chemistries (NMC, LMO), adding new anomaly subtypes, replacing the retrieve-and-blend decoder with a learned generative model, integrating QuaLiProM-style data when it becomes public — please open a discussion issue first. We'd like to coordinate to avoid duplicate effort.

## Licensing for contributions

By contributing code to this repository, you agree that your contributions are released under the Apache License 2.0 (matching the repository LICENSE).

By contributing benchmark results or methods, you are simply sharing information about your independent work; you retain all rights to your method and may license it however you choose.

## Code of conduct

Be respectful. Engage in good faith. Disagreements are normal in research; ad-hominem is not.

## Contact

For questions that don't fit a GitHub issue, please email the authors (contact info in `CITATION.cff` and on the paper).
