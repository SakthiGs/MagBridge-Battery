# How to cite MagBridge-Battery

If you use MagBridge-Battery in your research, in a publication, in a thesis,
in a benchmark, or in any other public-facing work, **please cite both the
paper and the dataset**.

We ask for two citations because each one tracks a different artifact and
gives credit through a different system:

1. **The paper** is the primary intellectual contribution (the bridge
   architecture, the validation methodology, the released benchmark). Cite it
   first.
2. **The dataset DOI** specifically identifies the v1.0 data artifact and is
   how Zenodo, DataCite, and similar trackers count dataset use.

If your reference list has only room for one entry, cite the paper.
For dataset-paper conventions (Scientific Data, Data in Brief, etc.) cite
both.

---

## Copy-paste citations

### Paper (primary citation)

**Plain text:**
> Gunasekar, S. P. and Rangarajan, P. K. "MagBridge-Battery: A Synthetic
> Bridge Dataset for Li-ion Magnetometry and State-of-Health Diagnostics."
> arXiv preprint arXiv:XXXX.XXXXX, 2026.

**BibTeX:**
```bibtex
@article{magbridge2026,
  author        = {Gunasekar, Sakthi Prabhu and Rangarajan, Prasanna Kumar},
  title         = {{MagBridge-Battery}: A Synthetic Bridge Dataset for
                   {Li}-ion Magnetometry and State-of-Health Diagnostics},
  journal       = {arXiv preprint},
  eprint        = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  year          = {2026},
  url           = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

> **Note:** The `arXiv:XXXX.XXXXX` placeholder will be replaced with the real
> arXiv identifier after submission. Once the paper is published in a
> peer-reviewed venue, the BibTeX above will be updated to point at the
> published version of record. The arXiv ID will remain a valid alternative.


### Dataset (secondary citation)

**Plain text:**
> Gunasekar, S. P. and Rangarajan, P. K. *MagBridge-Battery v1.0* [Data set].
> Zenodo, 2026. https://doi.org/10.5281/zenodo.20260147

**BibTeX:**
```bibtex
@misc{magbridge_battery_v1_0,
  author    = {Gunasekar, Sakthi Prabhu and Rangarajan, Prasanna Kumar},
  title     = {{MagBridge-Battery v1.0}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20260147},
  url       = {https://doi.org/10.5281/zenodo.20260147},
  note      = {Data set}
}
```

---

## Upstream attribution

In addition to citing MagBridge-Battery itself, please cite the upstream
sources whose data the bridge depends on. These are NOT substitutes for
citing MagBridge-Battery; they are additional citations expected by the
upstream authors and consistent with the LICENSE notice in this bundle.

**OSF battery magnetometry archive (Mohammadi, Jerschow, Hu et al.)**

```bibtex
@article{hu2020,
  author  = {Hu, Y. and Iwata, G. Z. and Mohammadi, M. and Silletta, E. V. and
             Wickenbrock, A. and Blanchard, J. W. and Budker, D. and
             Jerschow, A.},
  title   = {Sensitive magnetometry reveals inhomogeneities in charge storage
             and weak transient internal currents in {Li}-ion cells},
  journal = {Proceedings of the National Academy of Sciences},
  volume  = {117},
  number  = {20},
  pages   = {10667--10672},
  year    = {2020},
  doi     = {10.1073/pnas.1917172117}
}

@misc{osfdata,
  author       = {Mohammadi, M. and Jerschow, A.},
  title        = {Battery Magnetometry Data 2019-2020 (OSF)},
  year         = {2019--2020},
  howpublished = {Open Science Framework},
  doi          = {10.17605/OSF.IO/CW8ZV},
  url          = {https://osf.io/cw8zv/}
}
```

**PulseBat dataset (Tao et al.)**

```bibtex
@article{tao2024pulsebat,
  author  = {Tao, S. and Ma, R. and Zhao, Z. and others},
  title   = {Generative learning assisted state-of-health estimation for
             sustainable battery recycling with random retirement conditions},
  journal = {Nature Communications},
  volume  = {15},
  pages   = {10154},
  year    = {2024},
  doi     = {10.1038/s41467-024-54454-0}
}
```

---

## Why we ask for two citations to our own work

Citation indexers handle paper-DOIs and dataset-DOIs differently:

- Google Scholar primarily tracks paper-DOI and arXiv ID citations. Web of
  Science and Scopus typically only count the published version of record.
- Zenodo and DataCite track citations to the dataset DOI separately.
- Some downstream users will only cite the paper; others, by convention or
  habit, will also cite the dataset. Asking for both ensures both indexers
  see the credit.

The paper-first ordering reflects that the paper is the intellectual
contribution and the dataset is the artifact described by it. Cite the
paper as your main reference; include the dataset DOI when the data
specifically is what you are using.
