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
- `notebooks/04_sentence_rewrite_experiments.ipynb`: training-set document families,
  sentence rewrite scores added to the TF-IDF baseline, a fusion decision offset and a
  TF-IDF/XLM-R probability blend, all compared on the frozen holdout.

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

## Sentence rewrite, offset and blend experiments

`config/rewrite_experiments.toml` holds every setting: input paths, the baseline
vocabulary caps, sentence alignment thresholds, the sentence classifier, the stage gate,
the offset grid and the fixed blend weight. All keys are required. The stages run on CPU
and must be executed in this order; each verifies and skips completed outputs:

```bash
uv run python scripts/run_rewrite_experiments.py --stage families
uv run python scripts/run_rewrite_experiments.py --stage reference
uv run python scripts/run_rewrite_experiments.py --stage sentence-gate
uv run python scripts/run_rewrite_experiments.py --stage sentence-holdout
uv run python scripts/run_rewrite_experiments.py --stage offset
uv run python scripts/run_rewrite_experiments.py --stage blend
uv run python scripts/run_rewrite_experiments.py --stage final
```

Sentence labels come only from `train.csv`: a fusion sentence is labelled rewritten when
its normalized text is absent from every linked human document of the same training
population. Holdout evaluation uses development families only and excludes families linked
to holdout documents from sentence-model training. The `final` stage needs the saved
all-training baseline (`final_pipeline.joblib`, `fitted_vectorizers.joblib` and
`training_features.npz`) from the `hw3-tf-idf-baseline-final-submission` Kaggle output and
the XLM-R test predictions; it writes validated CSVs only for decisions that met their
predeclared holdout criteria. Nothing is submitted to Kaggle automatically. Peak memory
reaches roughly 20–30 GiB in the `reference` and `final` stages.

### XLM-R sentence detector

`config/xlmr_sentence.toml` configures a GPU sentence classifier that replaces nothing and
adds seven XLM-R aggregates to the character-sentence model. Development family fold 0
(excluding families linked to the holdout) trains the encoder; documents from those families
are excluded from every later logistic-regression fit, with class weights restoring the
original class totals.

```bash
uv run python scripts/run_rewrite_experiments.py --stage transformer-prepare
uv run python scripts/run_rewrite_experiments.py --stage char-all-train
# Kaggle: experiments/03_xlmr_sentence_kaggle.ipynb with MODE = "train" on a T4 GPU
uv run python scripts/run_rewrite_experiments.py --stage transformer-holdout
uv run python scripts/run_rewrite_experiments.py --stage transformer-final
```

`transformer-prepare` writes the private Kaggle bundle to `dist/kaggle_sentence_transformer/`:
sentence keys and labels, sentence counts and the sentence splitter, without competition
text. The hosted notebook verifies input fingerprints and identical sentence splitting, fine-tunes
for one epoch and scores every eligible train and test sentence. Download its
`sentence_scores.parquet` and `sentence_scores.json` to the paths in the configuration before
running the evaluation stages.

### XLM-R long-text policies

`config/xlmr_long_text.toml` lists the downloaded output directories of the original
head-only run and the corrected head-only and head-plus-tail runs of
`experiments/02_xlm_roberta_kaggle.ipynb`, together with the predeclared adoption
criteria. Download each kernel's outputs without model weights, then run:

```bash
uv run python scripts/run_rewrite_experiments.py --stage xlmr-long-text
```

The stage verifies every holdout receipt and selected-model manifest, requires identical
recipes apart from the truncation policy and writes paired comparisons and length-group
macro-F1 that `notebooks/05_xlmr_long_text_policies.ipynb` displays.
