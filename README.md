# UCU AI 2026 HW3: AI-generated text detection

English and German document classification into `human`, `ai` and `fusion`.

Use Python 3.14 and `uv`:

```bash
make setup
uv run jupyter lab notebooks/
make check
uv run pytest
```

Place the competition `train.csv` in `data/`. Raw CSVs, generated metadata, model outputs and internal research notes are excluded from Git. No external text data is used.

- `notebooks/01_eda.ipynb`: training-data integrity, distributions, tokenizer lengths and leakage investigation.
- `notebooks/02_baseline.ipynb`: frozen split audit, text-only sparse TF-IDF experiments, three-fold CV, selected-model holdout evaluation and optional submission export.
- `notebooks/03_xlm_roberta.ipynb`: shared encoder sampling audit and controlled
  XLM-R head-only versus head-plus-tail experiment review.

The baseline notebook defaults to analysis without model training. Enable `RUN_CV` for the four-candidate, three-fold comparison, then `RUN_HOLDOUT` for the CV-selected model fitted on all development records. A separate `RUN_SUBMISSION` stage requires the local test and sample CSVs, performs an all-training-data refit and exports predictions without submitting to Kaggle. Review RAM and runtime before enabling training.

Keep `data/metadata/validation_manifest.parquet` and its JSON provenance together in durable private storage. They freeze IDs, labels, normalized exact-text groups, holdout assignments, CV folds and the shared approximately 30k CV sample. Future experiments must restore and reuse these IDs. Existing incompatible records fail rather than regenerate. The manifest's deterministic reconstruction requires the recorded source data, seed, recipe and scikit-learn version; Git does not store the generated manifest.

Results are saved under a recipe-specific `runs/classical_baseline/` directory with source and output fingerprints. Completed CV and holdout measurements are restored on rerun. A final-model checkpoint can resume test inference after interruption without repeating its fit. Unknown related-document families remain a potential validation leakage source.

For Kaggle upload, use `experiments/01_baseline_kaggle.ipynb`. Build the private code and
frozen-split ZIPs with `uv run python scripts/build_kaggle_bundles.py`. The hosted notebook
runs training by default when Run All is selected; the research notebook keeps training
disabled.

XLM-R uses a separate [GPU entry point and upload guide](experiments/XLMR_README.md).
Build its private archives with `uv run python scripts/build_xlmr_bundles.py`.
The GPU notebook defaults to preflight; training requires `MODE = "train"`.
