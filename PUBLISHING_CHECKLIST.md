# MagBridge-Battery — Publishing Checklist

This is the step-by-step playbook for publishing MagBridge-Battery v1.0 across
Zenodo, arXiv, and GitHub. Follow it in order. Each step produces an identifier
that the next step needs.

> **Status of this checklist for v1.0:**
> - ✅ **Step 1 (Zenodo)**: complete. The v1.0 dataset is published at DOI
>   `10.5281/zenodo.20260147` (https://doi.org/10.5281/zenodo.20260147).
> - 🔄 **Step 2 (arXiv)**: submission in moderation; arXiv ID to be backfilled
>   into Zenodo metadata and this repo once accepted.
> - 🔄 **Step 3 (GitHub public)**: pending arXiv acceptance.
>
> Below: the historical checklist used to reach Step 1. The placeholder
> `10.5281/zenodo.XXXXXXXX` shown in the examples is what users see *before*
> reserving a Zenodo DOI — once reserved, it becomes the real DOI like
> `10.5281/zenodo.20260147`.

> **Time budget:** roughly 60–90 minutes of active work spread over 2–3 calendar
> days (mostly waiting on arXiv moderation).

---

## Step 0 — Pre-flight verification

Before starting, verify locally that everything compiles and runs.

- [ ] **Local environment ready.** `pip install pandas pyarrow scikit-learn torch`
- [ ] **Paper compiles cleanly.** `pdflatex magbridge_battery.tex && bibtex magbridge_battery && pdflatex magbridge_battery.tex && pdflatex magbridge_battery.tex` produces a 9-page PDF with no errors.
- [ ] **Bundle is intact.** Unzip the release ZIP and run `sha256sum -c checksums.sha256` — every line should report `OK`.
- [ ] **Benchmark scripts run.** `python3 run_benchmark.py` and `python3 run_dl_bench_lean.py t1` complete without errors and produce numbers matching Table III / Table IV in the paper (within seed variance).
- [ ] **Endorsement for arXiv ready.** If you have not submitted to arXiv `cs.LG` before, line up an endorser now (advisor, colleague who has 3+ submissions in cs.LG). Endorsement takes 30 seconds for the endorser but blocks submission if not in place. You can check your endorsement status at <https://arxiv.org/auth/endorse>.

If any of these fail, stop and fix before proceeding.

---

## Step 1 — Zenodo (reserve DOI, then upload)

### 1a. Reserve the DOI without publishing

1. Log in to Zenodo (<https://zenodo.org>) — register if needed, ORCID-linked.
2. Click **"New Upload"**.
3. Fill in the metadata form:
   - **Title:** `MagBridge-Battery: A Synthetic Bridge Dataset for Li-ion Magnetometry and State-of-Health Diagnostics`
     *(Use this exact title. Same title across Zenodo + arXiv + paper is critical for Scholar indexing. The dataset version `1.0` belongs in the Zenodo "Version" field, not in the title.)*
   - **Authors:** add Sakthi Prabhu Gunasekar and Prasanna Kumar Rangarajan, each with their ORCID.
   - **Resource type:** `Dataset`
   - **License:** `Creative Commons Attribution 4.0 International` (CC-BY-4.0)
   - **Description:** copy from `dataset_card.md` Summary section, or paste from `README.md` opening section.
   - **Keywords:** `battery`, `lithium iron phosphate`, `LFP`, `magnetometry`, `state of health`, `SOH`, `synthetic dataset`, `benchmark`, `anomaly detection`
   - **Funding** (if applicable): add your institutional or grant info.
   - **Related identifiers:** leave empty for now; we add the arXiv ID after Step 3.
4. **Without uploading files yet**, click **"Save draft"**.
5. Click **"Reserve DOI"** (near the top of the page). Zenodo gives you a permanent DOI like `10.5281/zenodo.18847291`.
6. **Write the reserved DOI down somewhere safe.** You'll paste it into several files in the next step.

> **Why reserve before uploading:** the DOI is referenced inside the bundle's
> own files (manifest.json, README.md, CITATION.cff, CITING.md). Uploading
> first then editing would leave broken placeholders inside the archived bundle.

### 1b. Fix placeholders in the bundle, then re-package

Edit these files in the release bundle directory:

- [ ] **`manifest.json`** — replace every `10.5281/zenodo.XXXXXXXX` with the reserved DOI.
- [ ] **`README.md`** — same replacement in the Citation section.
- [ ] **`CITATION.cff`** — same replacement in both `identifiers.value` and `references[0].doi`.
- [ ] **`CITING.md`** — same replacement in both BibTeX blocks.
- [ ] **`LICENSE`** — replace `<to be filled at upload>` in the suggested attribution block with the reserved DOI.
- [ ] **`dataset_card.md`** — no DOI in current draft; no change needed.

> If you change your mind and want to update the paper's DOI placeholder too,
> do it now so the same DOI appears in both bundle and paper. The paper has it at:
> `\\texttt{10.5281/zenodo.XXXXXXXX}` in the "Code and data availability" section.

Then **regenerate the checksums** (they must match the new file contents):

```bash
cd <bundle root>
rm -f checksums.sha256
find . -type f ! -name checksums.sha256 -print0 | xargs -0 sha256sum | sort -k2 > checksums.sha256
sha256sum -c checksums.sha256   # confirm all OK
```

Then **re-zip the bundle**:

```bash
cd <parent of bundle>
zip -r magbridge_battery_v1_0_release.zip <bundle_dir> -x "*.DS_Store"
```

Verify the new ZIP:

```bash
unzip -l magbridge_battery_v1_0_release.zip | tail -5  # should show ~18 files
```

### 1c. Upload and publish on Zenodo

1. Return to your Zenodo draft.
2. Drag-and-drop the **final** ZIP (the one with real DOI inside) into the upload area.
3. Wait for upload to finish.
4. **Verify metadata is still correct** — title, authors, ORCIDs, license, description.
5. Click **"Publish"**. ⚠️ This is permanent. Once published, you cannot change files (only metadata).
6. Confirm the DOI on the published page matches what you reserved.
7. **Bookmark the Zenodo URL.** You'll cite it from GitHub and arXiv next.

✅ **Step 1 done.** You now have: permanent Zenodo DOI + live Zenodo URL.

---

## Step 2 — arXiv submission

> **arXiv first, then GitHub public.** This is the opposite of what you might
> expect, but it lets the GitHub README link to a live arXiv preprint on day one.

### 2a. Prepare the arXiv submission tarball

In the `paper/` directory of your local working area, you should have:

- `magbridge_battery.tex`
- `references.bib`
- (no figure files needed — the paper uses only TikZ figures, which compile from source)

Before tarballing:

- [ ] **Update the Zenodo DOI in the paper** if you didn't do it in Step 1b. Change `10.5281/zenodo.XXXXXXXX` to the real DOI.
- [ ] **Compile one more time** to confirm clean output: `pdflatex && bibtex && pdflatex && pdflatex`. Visual check the PDF.

Create the tarball:

```bash
cd paper/
tar czf magbridge_arxiv_submission.tar.gz magbridge_battery.tex references.bib
ls -lh magbridge_arxiv_submission.tar.gz
```

### 2b. Submit to arXiv

1. Log in to <https://arxiv.org/submit>.
2. Click **"Start New Submission"**.
3. **Category:** primary `cs.LG`; cross-list to `eess.SP` and `physics.app-ph`.
   - If you don't have endorsement for `cs.LG`, arXiv will tell you here. Resolve via Step 0 if needed.
4. **Upload** the tarball. arXiv compiles it on their servers.
5. **Wait for the preview PDF**. Open it and check:
   - All 9 pages render
   - Both TikZ figures (Fig. 1 composition, Fig. 2 architecture) appear correctly
   - All citations resolve (no `[?]`)
   - The Zenodo DOI in the "Code and data availability" section shows the real number, not `XXXXXXXX`
6. **Metadata page** — fill in:
   - **Title:** same as Zenodo and the paper (exact match)
   - **Authors:** comma-separated, with ORCIDs
   - **Abstract:** paste from the LaTeX `\begin{abstract}…\end{abstract}` block. Strip LaTeX commands manually if arXiv complains.
   - **Comments:** `9 pages, 2 figures, 4 tables. Dataset: <Zenodo DOI URL>`
   - **License:** `arXiv.org perpetual non-exclusive license` (default; fine for preprint)
   - **MSC / ACM class:** optional, skip
   - **Journal-ref:** leave blank; we'll fill this when/if the paper is accepted to a conference.
   - **DOI:** leave blank; arXiv-side DOI is the journal DOI, which we don't have yet.
   - **Related identifier:** if arXiv has this field, add the Zenodo DOI as `IsSupplementedBy`.
7. **Submit** for moderation.

### 2c. Wait for arXiv moderation

- Typical wait: 1–2 business days (longer for first-time submitters).
- arXiv will email you when accepted, with the arXiv ID (e.g. `arXiv:2606.12345`).
- If rejected: read the rejection email carefully. Common reasons: category mismatch, formatting issues, missing endorsement.

✅ **Step 2 done.** You now have: permanent arXiv ID + live arXiv preprint URL.

---

## Step 3 — Backfill, then GitHub public

### 3a. Update Zenodo metadata with the arXiv ID

Zenodo allows metadata edits on a published record without minting a new DOI version — perfect for adding the arXiv ID after-the-fact.

1. Go to your published Zenodo record.
2. Click **"Edit"** (top right).
3. Scroll to **"Related identifiers"**.
4. Add:
   - **Identifier:** `arXiv:XXXX.XXXXX` (your actual ID)
   - **Relation:** `is supplement to`
   - **Resource type:** `Publication / preprint`
5. Add a paragraph to the description: *"Associated paper: arXiv:XXXX.XXXXX"*
6. **"Save"** — the change is live within a minute. The DOI does NOT change.

### 3b. Prepare GitHub repo

Locally, you should have the prepared GitHub repo structure (we'll build this together in the next message). Before pushing public:

- [ ] **README.md** has been updated with the real Zenodo DOI **and** real arXiv ID.
- [ ] **CITATION.cff** has both real identifiers.
- [ ] **NOTICE-PULSEBAT** has the verified copyright line from the upstream repo.
- [ ] **paper/magbridge_battery.pdf** is the latest compiled version.
- [ ] No private files committed — `git log --stat` and verify no leaked secrets, API keys, internal paths.
- [ ] **LICENSE** is present (Apache-2.0 for code).
- [ ] **No .ipynb_checkpoints, __pycache__, .DS_Store** committed. Check `.gitignore` covers them.

### 3c. Flip GitHub repo to public

1. Go to `github.com/SakthiGs/MagBridge-Battery` → **Settings**.
2. Scroll to **"Danger Zone"** → **"Change repository visibility"** → **Make public**.
3. Confirm.
4. Visit the repo as if you were a stranger. Check:
   - Top-right shows **"Cite this repository"** button (from CITATION.cff)
   - README renders correctly with working links to Zenodo and arXiv
   - License badge shows Apache-2.0 on the right sidebar
   - The two figures (if any are committed as PNGs) render in the README

### 3d. (Optional) Create a GitHub Release

This is good hygiene for tagging which commit corresponds to the paper version:

1. **Releases** → **"Create a new release"**.
2. Tag: `v1.0`
3. Title: `MagBridge-Battery v1.0 (paper release)`
4. Description:
   ```
   This release corresponds to the v1.0 dataset (Zenodo DOI: 10.5281/zenodo.20260147)
   and the arXiv preprint (arXiv:XXXX.XXXXX).
   ```
5. **Publish release**.

Bonus: GitHub releases get a Zenodo DOI of their own if you've enabled the Zenodo–GitHub integration. Optional, but it gives the code a separate DOI for citation if anyone needs to cite the code specifically rather than the dataset.

✅ **Step 3 done.** All three artifacts (paper on arXiv, data on Zenodo, code on GitHub) are public and cross-reference each other.

---

## Step 4 — Post-publication housekeeping

Within the first week after publication:

- [ ] **Set up Google Scholar profile** if you don't have one. Add the paper manually if Scholar hasn't auto-discovered it within 2 weeks.
- [ ] **Update your ORCID** to include the new publication.
- [ ] **Update your CV / institutional profile** with the citation.
- [ ] **Tweet / LinkedIn post** announcing the dataset — short, includes the Zenodo link and the arXiv link, optionally a screenshot of Figure 2 (the architecture diagram).
- [ ] **Email Prof. Jerschow** (the OSF data author) with the link to the published Zenodo and arXiv — informational, not asking for endorsement. Good academic etiquette since you derived from his data.
- [ ] **Email the PulseBat team (Tao et al.)** with the same — informational, good etiquette.
- [ ] **Watch the Zenodo download counter** for the first month — gives you early signal on whether anyone is using it.

Within the first month:

- [ ] If anyone files a GitHub issue or emails about the data: respond, fix any v1.0 bugs found, and plan a v1.1 release if needed.
- [ ] If a conference deadline is approaching, prepare the conference submission version (potentially anonymized for double-blind venues).

---

## Cross-references at a glance

After Steps 1–3 complete, every artifact references every other:

| Artifact | References | Located in |
|---|---|---|
| arXiv paper | Zenodo DOI | "Code and data availability" section |
| Zenodo record | arXiv ID | "Related identifiers" + description |
| GitHub repo | Zenodo DOI + arXiv ID | README.md |
| Zenodo bundle README | arXiv ID + GitHub URL | README.md |
| Zenodo bundle CITING.md | arXiv ID + Zenodo DOI | both BibTeX blocks |
| Zenodo bundle CITATION.cff | arXiv ID + Zenodo DOI | preferred-citation + references |
| Zenodo bundle manifest.json | arXiv ID + Zenodo DOI | citations block |

The single source of truth for each is:
- **Zenodo DOI:** chosen by Zenodo at reservation time (Step 1a)
- **arXiv ID:** chosen by arXiv at acceptance time (Step 2c)
- **GitHub URL:** chosen by you (`github.com/SakthiGs/MagBridge-Battery`)

---

## Troubleshooting

**Zenodo upload fails / hangs.** ZIP might be too large for the browser upload. Use `zenodo-cli` (`pip install zenodo-client`) for resumable uploads.

**arXiv submission rejected with "endorsement required".** Resolve via your endorser before resubmitting. The submission itself can be saved as a draft until endorsement clears.

**arXiv PDF preview looks broken** (e.g. TikZ figures missing). arXiv occasionally has TikZ rendering issues. The fix is usually one of: include `\\pdfoutput=1` at the top of the .tex, or pre-compile locally and upload the PDF instead. The current paper uses standard TikZ and should be fine.

**Google Scholar doesn't pick up the paper.** Wait 4-6 weeks. If still missing, add manually via your Scholar profile.

**Citation count split across multiple Scholar entries** (e.g. arXiv version and conference version listed separately). After publication, update the arXiv "Journal-ref" field to point at the conference paper. Scholar typically merges within a few weeks of that change.
