# Hosted classical baseline

`01_baseline_kaggle.ipynb` is the execution-only entry point. Research design and interpretation remain in `notebooks/02_baseline.ipynb`.

The current sklearn TF-IDF/LR/LinearSVC pipeline uses CPU sparse matrices even when the Kaggle session has T4 GPUs. The estimator recipe is unchanged. A T4 runtime is supported as a host, but GPU acceleration has not been introduced. CPU is also sufficient. GPU support is estimator-specific: see the [scikit-learn FAQ](https://scikit-learn.org/stable/faq.html#will-you-add-gpu-support).

## Prepare and upload

Build local artifacts after changing any shared code or notebook:

```bash
uv run python scripts/build_kaggle_bundles.py
```

For a new bundle version, upload the new code ZIP, reimport the matching execution
notebook and restart the session before launching. Updating a local ZIP does not
update an already uploaded notebook or an active runtime's imported modules.

1. Create a **private** Kaggle Dataset and upload both `dist/kaggle/classical_baseline_code.zip` and `dist/kaggle/classical_baseline_split.zip`. The split contains private IDs/labels and must stay private. The code archive contains no competition CSVs, credentials, checkpoints or internal reports.
2. Create a **private** Kaggle Notebook and import `experiments/01_baseline_kaggle.ipynb` using the notebook file-upload workflow.
3. Attach the competition dataset containing `train.csv`, `test.csv` and `sample_submission.csv`, plus your private bundle dataset. Kaggle extracts ZIP uploads; keep both top-level `hm3_baseline_code/` and `hm3_baseline_split/` directories. See [Kaggle dataset documentation](https://www.kaggle.com/docs/datasets).
4. Enable Internet for installing the exact pinned Python packages. Select T4 x2 if desired, or CPU. No external text data or model weights are retrieved.
5. Run setup and audit cells first. Check the discovered paths, environment, disk/RAM information, four-candidate matrix and split counts. Setup does not fit models.
6. Run the training cell, or Run All. **Run All trains by default:** 12 CV fits on the same 30,002 frozen records, followed by the selected model fitted on all 217,881 development texts and evaluated once on 54,471 validation texts. Set `RUN_TRAINING = False` for audit only. No real hosted training is launched during local packaging verification.

`RUN_FINAL_SUBMISSION` defaults to false. Enable it after selection to refit on all training records and export the sample-aligned CSV. For inference-only Run All, set `RUN_TRAINING = False`, `RUN_FINAL_SUBMISSION = True` and `FINAL_STAGE = "inference"`. The training cell also skips fitting automatically in this mode. No Kaggle submission is automatic.

The final export uses `scripts/run_bounded_submission.py`. It constructs the same
word/character vocabulary and IDF from streamed full-training frequencies, then
transforms texts in sparse batches. This avoids the unpruned full-corpus matrices
following SIGKILL during the first all-data refit. Vectorizers, the final sparse
matrix and fitted model have separate verified checkpoints. The classifier, C,
feature caps and labeled training population are unchanged. Rebuild the code
bundle before using this updated export path; existing Kaggle source datasets
remain unchanged until a new bundle is uploaded.

## Hosted compatibility exception

The project targets local Python 3.14. Kaggle's interpreter may differ, so bootstrap supports Python 3.12–3.14 and creates an isolated environment using the hosted interpreter. It installs hash-pinned dependencies from `config/requirements_kaggle.txt` with uv instead of applying the Python 3.14 project lock. CUDA, PyTorch and kernel packages are untouched. A missing uv binary is fetched from an official release with a pinned SHA-256. Actual Python, sklearn, NumPy, SciPy, Polars and joblib versions are recorded per run.

`config/requirements_kaggle.in` pins the direct runtime packages. Regenerate the complete lock with:

```bash
uv pip compile config/requirements_kaggle.in --python-version 3.12 --python-platform x86_64-unknown-linux-gnu --only-binary :all: --generate-hashes --output-file config/requirements_kaggle.txt
```

`config/baseline_kaggle.toml` is one baseline-component configuration with root keys: seed, frozen CV target, word/character vocabulary caps, min_df, the LR/SVM C lists, minimum free disk reserve, and exact source/split hashes. All keys are required; there are no hidden defaults or environment overrides. The default scientific settings match the research notebook. Changing source/split or sampling settings requires a compatible frozen manifest, never an automatic new split on Kaggle.

## Outputs and recovery

All persistent results are under `/kaggle/working/hm3_baseline/`; isolated packages/cache are under `/kaggle/temp/hm3_baseline/` and are not saved as notebook outputs. Save the private notebook output version or download the output directory before session expiry. [Kaggle notebook documentation](https://www.kaggle.com/docs/notebooks) describes reattaching saved notebook outputs as inputs.

Each recipe directory contains environment and input identity, split audit, completed-fold scores/OOF predictions, comparison tables, selected-model holdout metrics/predictions, subgroup diagnostics, short error examples and confusion-matrix HTML. Optional final model/CSV provenance is kept alongside the export. There are no full competition-text copies in the code package or saved results.

To resume, attach the same private code/split dataset and competition input, plus exactly one prior saved output version for that recipe. The runner restores that directory into writable storage and verifies all completed artifacts before using them. Completed folds and holdout evaluation are skipped; a completed final model restarts inference without refitting. An active sklearn fit cannot resume mid-solver and repeats if interrupted. Runtime/library changes produce a distinct identity. Stop interactive training before launching another writer through Save Version.

Local tests and a compatible local Python environment can verify bootstrap, archive integrity, split restoration and synthetic model execution. They do not prove real Kaggle session duration, T4 availability, full-data peak RAM, convergence or completed competition results. No live Kaggle run is claimed until it is actually executed.
