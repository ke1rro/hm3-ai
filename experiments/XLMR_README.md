# XLM-R on Kaggle

Use a **new private Kaggle notebook and a separate private code dataset**. Keep the
running classical baseline and its input dataset unchanged. The research comparison
lives in `notebooks/03_xlm_roberta.ipynb`; this directory contains the GPU entry point.

## Prepare and upload

1. Build the two archives with `uv run python scripts/build_xlmr_bundles.py`.
   Upload `dist/xlmr_kaggle/xlmr_code.zip` and `xlmr_split.zip` to a **private** Kaggle
   dataset. Kaggle must extract both archives. The split archive contains IDs,
   labels and assignments, with no competition text; it must remain private.
2. Import `experiments/02_xlm_roberta_kaggle.ipynb` as a new private notebook.
3. Attach both the new code/split dataset and the **original competition data**,
   including `train.csv`, `test.csv`, and `sample_submission.csv`. The bootstrap
   verifies the training CSV fingerprint before installation. Repackaged CSVs with
   changed bytes are rejected; use the original files.
4. Select a T4 GPU runtime and enable Internet for pinned open-weight model files
   and auxiliary packages only. If Kaggle exposes two T4s, the launcher deliberately
   exposes only GPU 0, keeping the effective batch size at 32 documents.
5. Run All with the default `MODE = "preflight"`. Review source fingerprints,
   GPU capacity, free disk, class mappings and role counts. No model weights are
   loaded and no training is started in this mode; tokenizer files may download.

## Start the first training run

Set these variables in the notebook configuration cell:

```python
EXPERIMENT = "xlmr_head_only_40k"
MODE = "train"
MODEL_SCOPE = "sampled"
EVALUATE_HOLDOUT_AFTER_TRAIN = True
```

Use **Save Version → Save & Run All** to preserve completed notebook outputs. This
trains one selected recipe, selects its checkpoint using development-only macro-F1,
then evaluates the complete fixed holdout once. It never starts head-plus-tail.
The shared recipe is in `config/xlmr.toml`: 512 total tokens, two initial epochs,
learning rate `2e-5`, weight decay `0.01`, train batch 4, accumulation 8, FP16,
gradient checkpointing and dynamic padding. Evaluation/checkpointing occurs every
250 optimizer steps; patience is two development evaluations without improvement.
Early stopping can end training before two epochs. Epoch-budget all-data refit
has no development evaluation or early stopping.

The frozen selection contains **40,001 training**, **2,001 development evaluation**
and **54,471 fixed holdout** documents. Rounding respects complete confirmed
duplicate groups. The maximum joint class/language/domain deviation from the
development partition is approximately 0.00134 percentage points for training and
0.03850 for development evaluation. No joint stratum is missing in development
evaluation. Unknown source-document families remain an unresolved leakage risk.

The runtime uses the host's Python 3.12–3.14 and **native PyTorch/CUDA**, requiring
PyTorch 2.6 or later. An isolated environment installs hash-pinned auxiliary
dependencies with `uv`; it does not install Torch or CUDA wheels. The original
Torch file, version and CUDA build are checked before and after setup. Local
development uses Python 3.14 and the project lock, rather than applying that lock
to Kaggle's different interpreter.

Allow one GPU with at least 12 GiB total VRAM; T4 has approximately 16 GiB. Actual
free VRAM is reported before model allocation. Reserve at least 12 GiB free disk
for downloads, best/latest Trainer checkpoints, selected weights and predictions.
VRAM fit and throughput require measurement on the actual runtime; there is no
promised training duration or model score.
The fixed seed controls sampling and model initialization; GPU kernels can still
introduce numerical nondeterminism across hardware and native runtime versions.

## Outputs and restoration

Outputs live under `/kaggle/working/hm3_xlmr/runs/<experiment>/<recipe_hash>/`:

- `run_metadata.json`, copied manifests and configuration identity;
- `history.json`, loss and development macro/per-class F1;
- `checkpoints/checkpoint-*`, optimizer, scheduler, RNG, FP16 scaler and weights;
- `selected_model/`, verified inference weights, tokenizer and selection state;
- holdout metrics, subgroup tables, confusion figures and ID-aligned predictions
  containing true labels, predicted labels, logits and class probabilities;
- an internal measured findings record after holdout evaluation;
- optional validated submission CSV and provenance.

`/kaggle/working` is temporary until outputs are saved. Save/download the complete
run directory before runtime expiry; a crashed saved run may not publish outputs.
There is no external storage synchronization configured. Never rely on a browser
disconnect as proof that the job or outputs survived.

To restore, attach **one saved output version** to the next notebook, keep the same
source bundle, native runtime, output root, experiment and scope, then rerun the
same mode. Only matching identities and checksummed complete checkpoints are
accepted. Training restores optimizer/scheduler/RNG/scaler and progress. A newer
incomplete generation is skipped only when an earlier verified checkpoint exists.
Do not move the run root during training restoration: Trainer best-checkpoint
paths refer to the original output location. Completed selected-model inference
does not require the intermediate Trainer checkpoint directories.

## Other explicit modes

- `MODE = "holdout"`: evaluate a restored sampled model without training. Completed
  verified predictions are reused.
- `EXPERIMENT = "xlmr_head_tail_40k"`, `MODE = "train"`: run the controlled second
  experiment separately, with identical IDs and optimizer settings. At length 512,
  two special tokens leave 510 content tokens, split into 255 head and 255 tail.
  Short documents keep all content. No chunk labels are created. Further model
  selection between policies should use their development scores; viewing repeated
  holdout results must not become a tuning loop.
- `MODE = "refit"`: explicitly initialize the selected pretrained recipe and train
  on **all 272,352 labeled rows**, for the fixed two-epoch budget. This creates a
  separate `_all_train` run and never evaluates holdout or uses it for stopping.
  This mode performs substantially more computation and is never automatic.
- `MODE = "inference"`, `MODEL_SCOPE = "sampled"` or `"all_train"`: restore the
  corresponding selected model, skip training, predict all 25,000 test IDs and
  export `id,label` in sample-submission order. No Kaggle submission is sent.

Test inference uses the original tokenizer revision and truncation policy. Saved
probability/logit columns follow **human, ai, fusion**, matching output IDs 0, 1, 2.
Subgroup metrics always average over these three classes, assigning zero for
undefined class scores; missing-class support is reported explicitly. Treat small
subgroups as descriptive measurements.

## Local verification

```bash
uv run pytest tests/test_encoder.py
uv run ty check
uv run python scripts/build_xlmr_bundles.py
```

Local tests use tiny synthetic models/tokenizers only. They verify contracts and
actual Trainer state restoration on CPU; they do not establish live T4 training
performance or produce competition scores.

## Completed head-only run and loss normalization

The completed `ke1rro/notebookf475a1574b` run used the original source bundle.
Its verified holdout macro-F1 is 0.813427; the selected checkpoint is step 500.
Transformers 5.24 inferred accumulation-normalized loss from XLM-R's forward
kwargs, although its classification head returns mean document cross-entropy.
A tiny CPU reproduction confirmed an eightfold scaling error at accumulation 8.
The current loader declares `accepts_loss_kwargs=False`.

With that correction, `ke1rro/hw3-xlmr-head-only-corrected` reached holdout macro-F1
0.811687 (early stop at step 1,000, selected step 500) and
`ke1rro/hw3-xlmr-head-tail-corrected` reached 0.903012 (two full epochs, selected step
2,500). The paired comparison, length groups and the early-stopping caveat are in
`notebooks/05_xlmr_long_text_policies.ipynb`, produced by the `xlmr-long-text` stage of
`scripts/run_rewrite_experiments.py` from the downloaded outputs listed in
`config/xlmr_long_text.toml`.

Keep the historical model and its metrics unchanged. Restore or run inference for
that model with its original code dataset and runtime; the matching local source
archive is `dist/xlmr_kaggle/xlmr_code_legacy_57dfda5d5e0e.zip`. Its inference does
not use training loss. Do not resume the old optimizer state with changed
normalization.
