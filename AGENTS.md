# AGENTS.md

## Project Context

This repository is for the UCU AI 2026 AI-Generated Text Detection (HW3) homework.
Classify English and German texts as `human`, `ai`, or `fusion` (human-written text
partially rewritten by an LLM). The competition data contains 272,352 labeled
training texts and 25,000 unlabeled test texts, with `id`, `text`, `lang`, and
`domain` columns. The evaluation metric is three-class macro-F1. The main
deliverable is a reproducible, notebook-first comparison of NLP baselines and
fine-tuned multilingual encoders, supported by controlled experiments and a
justified final prediction pipeline.

Reserve `notebooks/` for research notebooks and results intended for the report. Store
execution-only Colab and Kaggle notebooks in `experiments/`. Keep reusable
code in `src/` when it improves clarity, reduces duplication, or makes testing easier.
Do not introduce unnecessary infrastructure or hide all experiments behind library calls.

Follow the competition restrictions: train only on `train.csv`; pretrained model
weights are allowed, but no external data may be used to improve predictions.
Never use proprietary LLMs, including chat or API interfaces, to label, generate,
or extract features from competition data. AI coding assistance is permitted;
the student must write the final 1–2 page technical report independently.

## Development Workflow

1. Inspect relevant code and configuration before making changes. Follow existing project conventions.
2. Make the smallest coherent change that solves the task. Avoid unrelated refactoring and unnecessary abstractions.
3. Use `uv` exclusively for Python environments and dependency management. Run project commands through `uv run`.
4. Update `pyproject.toml` and `uv.lock` together when changing dependencies.
5. Run relevant tests and quality checks after modifications. Never claim a check passed unless it was executed.
6. Report changes, verification results, and remaining issues concisely.
7. Never stage, commit, push, or create branches unless explicitly requested.
8. Keep Python, PyTorch, CUDA, tokenizers, Transformers, and NLP-library versions compatible with the actual runtime. Do not carry over unrelated project dependency pins. Document any necessary Python-version or GPU-specific exceptions.

## Python Style

01. Target Python 3.14 when supported by the required NLP and GPU stack. Use a documented compatible Python version when a required dependency or hosted GPU runtime does not support 3.14. Write clean, idiomatic, readable Python.
02. Add parameter and return type annotations to every function and method. Prefer built-in types such as `list`, `dict`, `tuple`, and `str`, and modern unions such as `str | None`.
03. Do not use `from __future__ import annotations`. Avoid importing `typing` when built-in syntax is sufficient.
04. Prefer comprehensions and standard Python constructs when they improve readability.
05. Avoid code comments in reusable modules. Prefer expressive naming and straightforward implementations. In notebooks, use Markdown cells to explain decisions and brief code comments only where needed to clarify non-obvious steps.
06. Avoid leading underscores for public variables and functions. Use one leading underscore for private names, Python protocols, or external APIs where appropriate.
07. Use `pathlib` for filesystem operations.
08. Keep functions focused and avoid unnecessary classes, wrappers, and helper functions.
09. For imports do not use `import *`. You should import `polars` as `pl`, `polars.selectors` as `cs`, `plotly.express` as `px`, `plotly.graph_objects` as `go`, `plotly.subplots` as `sp`, `numpy` as `np`, `streamlit` as `st`. For all other libraries, use the default import name or use `from <module> import <class/function>` for specific classes or functions. Avoid aliasing unless necessary to avoid name conflicts or improve clarity.
10. Limit the length of lines to 100 characters, as required by Flake8 configuration. For long strings, use implicit concatenation with parentheses instead of backslashes. Avoid using `# noqa` to bypass style checks.
11. Read environment variables and `.env` configuration using `decouple` via `from decouple import config`. Do not use `os.environ` or `os.getenv` to read settings or secrets.
12. For all Python scripts and modules, use `logging` to log messages. Use `info` level for normal operation messages, where processing takes a long time. Avoid using `print()` for logging or debugging. For each standalone script, configure logging to output to stdout with a timestamp, log level, and logger name as follows:

```python
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
```

## Naming Conventions

1. Use snake_case for variables, functions, and methods. Use PascalCase for classes. Use UPPER_SNAKE_CASE for constants. Prefix internal/private attributes and methods with a single underscore (e.g., \_internal_state).
2. For boolean variables, use names that clearly indicate their meaning. Always use prefixes such as `is_`, `has_`, `can_`, or `should_`. For example, use `is_valid`, `has_permission`, `can_execute`, or `should_retry`.
3. Use strong verbs that describe the exact action being performed. Avoid ambiguous names like `process_data()`, `handle_event()`, or `do_work()`. Instead, use `parse_market_data()`, `route_incoming_event()`, or `calculate_volatility()`.
4. Specific Class Nouns: Use descriptive nouns that define the entity's exact role. Avoid vague suffix crutches like `Manager`, `Handler`, or `Processor`. Instead of `DataProcessor`, use `DataNormalizer` or `MetricsAggregator`.
5. Avoid arbitrary abbreviations, but accept domain-standard acronyms (e.g., `http`, `uuid`). While descriptive loop variables (e.g., for record in records) are strongly preferred, standard mathematical or index variables (`i`, `j`, `x`, `y`) are acceptable for simple iterations or coordinate math. Always use _ for throwaway variables in unpacks or loops. Do not use `df` acronyms for DataFrames. Use descriptive names like `user_data`, `transaction_records`, or `sensor_readings` instead.

## Configuration Conventions

1. Store human-edited project runtime configurations as TOML in `config/`. Keep framework-required formats, such as pretrained tokenizer/model configuration files, when applicable. Use JSON for metadata and generated result records. Keep tooling files in their required formats, such as `pyproject.toml`.
2. Use `snake_case` keys. A configuration for one component stores its options at the root. A configuration composing multiple components groups their options in named tables, such as `[dataset]`, `[training]`, and `[evaluation]` for NLP classification experiments.
3. Use `backend` to select an implementation when configurable, `model` for the classifier architecture, and `model_id` for a pretrained checkpoint identifier or path. Keep backend options in the same table, using the names accepted by the implementation. Avoid wrapper tables or unused fields added only to enforce a uniform shape.
4. Omit optional settings to use documented defaults. TOML has no `null` value. Use native TOML values for booleans, numbers, strings, and arrays.
5. Keep secrets in environment variables or `.env`, read through `decouple`. Configuration may name a secret variable, such as `api_key_env = "WANDB_API_KEY"`, but must not contain its value.
6. Document required tables, supported backends, and defaults in the component's README. Keep configuration filenames, loader behavior, CLI defaults, and examples consistent.
7. Keep reusable dataset, tokenization, training, evaluation, and inference logic in focused `src/` modules when needed. Keep model selection, experiment design, comparisons, and conclusions visible in notebooks. Do not add a framework abstraction unless it eliminates real duplication.

## Docstrings

1. Add docstrings to all public functions, methods, and classes.
2. Do not add docstrings to private functions, private methods, private classes, special methods, any code in Jupyter notebooks, or in the header of a file.
3. Use imperative mood for function and method docstrings. Use descriptive mood for class docstrings.
4. For parameters, attributes, and return values, provide 1-sentence description, unless the type is nested or complex, in which case provide a brief description of the structure and contents (e.g. key and value descriptions for a dictionary).
5. For large functions that have more than 60 lines of code, include a brief description of the algorithm between the summary and the `Args` section. Use blank lines to separate the summary, algorithm description, and `Args` section.
6. Use modified Google-style docstrings with a 1-to-3 sentence summary. Do not include types in the docstring, as they are already specified in the function signature.
7. Do not use `Raises` section in docstrings. For important exceptions, include them in the summary or algorithm description instead. But do not include every possible exception, only the ones that are relevant, very important and require the caller to handle them explicitly.

Example of docstrings:

```python
def function(param1: int, param2: str) -> bool:
    """{1-3 sentence summary of the function}

    {brief description of the algorithm if the function is large}

    Args:
        param1: {description of `param1`}.
        param2: {description of `param2`}.

    Returns:
        {description of the return value}.
    """

def generator(param1: int) -> Generator[int, None, None]:
    """{1-3 sentence summary of the generator function}

    {brief description of the algorithm if the function is large}

    Args:
        param1: {description of `param1`}.

    Yields:
        {description of the yielded values}.
    """

class Class:
    """{1-3 sentence summary of the class}

    Attributes:
        attr1: {description of `attr1`}.
        attr2: {description of `attr2`}.

    Args:
        param1: {description of `param1`}.
        param2: {description of `param2`}.
        param3: {description of `param3`}.
    """
    attr1: str
    attr2: int

    def __init__(self, param1: str, param2: int = 0, param3: list[str] | None = None) -> None:
        self.attr1 = param1
        self.attr2 = param2
        self.attr3 = [] if param3 is None else param3

    def method(self, param1: int) -> bool:
        """{1-3 sentence summary of the method}

        {brief description of the algorithm if the method is large}

        Args:
            param1: {description of `param1`}.

        Returns:
            {description of the return value}.
        """

    @property
    def readwrite_property(self) -> list[str]:
        """{description of `readwrite_property`}"""

    @readwrite_property.setter
    def readwrite_property(self, value: list[str]) -> None:
        ...
```

## Simplicity and Error Handling

1. Reason through requirements, assumptions, and likely failure modes before writing code. Prefer simple code over defensive programming. Add guards only for concrete requirements or observed failures, not hypothetical cases.
2. Assume internal inputs satisfy documented contracts. Do not repeatedly validate values already guaranteed by the calling code.
3. Avoid speculative `if` checks, redundant `raise` statements, and excessive `try/except` blocks.
4. Validate inputs at meaningful system boundaries, especially when reading external data, configuration, or API responses.
5. Catch exceptions only when recovery or useful additional context is possible. Never silently swallow errors.
6. Do not use catch-all and reraise patterns for cleaning. Instead, use context manager where possible and if not, then `try/finally` pattern.
7. Do not use `assert` for production input validation. Use normal pytest assertions in tests.
8. Do not introduce fallback behavior unless it is explicitly required.
9. Never throw errors for operations where the semantic intent can be gracefully satisfied, for example:
   - Enforce mathematical identities where they exist: `sum([])` should return `0`. Do not invent values for undefined statistics such as `mean([])`.
   - Make deletions idempotent: `delete(missing_record)` or `rm(missing_file)` must return success. The intent ("ensure it's gone") is already met.
   - Forgive boundaries: array slicing outside existing bounds should yield an empty array, not an `IndexOutOfBounds` exception.

## Tooling and Verification

| Purpose                      | Tool                                                                 |
| ---------------------------- | -------------------------------------------------------------------- |
| Dependencies and environment | uv                                                                   |
| Configuration and secrets    | decouple                                                             |
| Formatting                   | Black                                                                |
| Import sorting               | isort                                                                |
| Linting                      | flake8                                                               |
| Static type checking         | ty                                                                   |
| Tests                        | pytest                                                               |
| Git hooks                    | pre-commit                                                           |
| Interactive notebooks        | JupyterLab                                                           |
| Tabular data                 | polars                                                               |
| Plotting                     | plotly, plotly-express                                               |
| Classical NLP                | scikit-learn, scipy                                                  |
| Text vectorization           | scikit-learn TF-IDF word and character n-grams                       |
| Deep learning                | torch                                                                |
| NLP encoders and tokenizers  | transformers, tokenizers; datasets as needed                         |
| Classification evaluation    | scikit-learn metrics                                                 |
| Explainability               | SHAP, LIME, or token attributions, if used                            |
| Experiment tracking          | Notebook tables and saved artifacts; external trackers if configured |

Use the existing configuration instead of introducing alternative tooling.

After relevant Python changes, run:

```bash
uv run isort --check-only .
uv run black --check .
uv run flake8 src/ scripts/ tests/
uv run ty check
uv run pytest
```

Use `uv run pre-commit run --all-files` for complete repository validation.
Run directory-specific checks only for paths that exist; follow the repository configuration.

Report failures rather than weakening checks to make them pass.

## Data and Experiments

01. Prefer Polars for tabular data processing. Use native expressions, `with_columns`. Avoid Python loops for operations that can be expressed directly in Polars.
02. Use lazy evaluation only for multi-step pipelines that benefit from query optimization or streaming processing. Avoid lazy evaluation for single-step operations.
03. Preserve existing experiment seeds and splits unless explicitly changing the experiment. Use seed 2026 for new experiments. Keep experiments reproducible; document unavoidable GPU nondeterminism.
04. Keep training, validation, and testing boundaries clear. Prevent data leakage. Fit vectorizers, tokenizers trained from scratch, scalers, feature selection, and learned preprocessing only on the corresponding training fold. Pretrained tokenizers and weights are allowed.
05. Record configurations, seeds, metrics, and relevant outputs in the experiment notebook and appropriate project output directories. Store dataset metadata in `data/metadata/`. Keep a concise comparison table containing the model, changed parameter, validation protocol, and result.
06. Compare experiments against a consistent baseline and fixed validation split. Do not report unmeasured improvements or invent results.
07. Examine exact and near-duplicate texts and possible human/AI/fusion variants before making the local split. Use group-aware splitting when reliable document families can be established from training data alone; otherwise use a fixed stratified split and document residual leakage risk. Never tune preprocessing or hyperparameters on test texts or leaderboard feedback.
08. Track class balance by `lang` and `domain`, their intersections, word and tokenizer lengths, nulls, duplicates, and representative examples of each class during EDA. Inspect whether `fusion` differs locally from both other classes without assuming a fixed pattern.
09. Preserve text content and language-specific information during preprocessing. Test normalization decisions rather than silently deleting punctuation, capitalization, whitespace, or symbols that may be predictive of authorship. Distinguish word counts from tokenizer token counts.
10. Keep label conversions explicit: `human`, `ai`, and `fusion` are the only submission labels. Document model class order, probability-column order, and any integer mapping; never assume alphabetic order matches a model's output head.
11. Report consistent three-class macro-F1, per-class precision/recall/F1/support, and a confusion matrix. Include subgroup results for language, domain, and text length when meaningful. Distinguish local validation from Kaggle public and private scores.
12. Test tokenization, padding, truncation, attention masks, and chunk-to-document aggregation. Models with a maximum sequence length must have an explicit long-text policy. Do not assign the document's `fusion` label to every chunk as if chunk-level ground truth were available.
13. Validate submissions against `sample_submission.csv`: exactly 25,000 rows, precisely `id,label` columns, unique and complete test IDs in sample order, and only lowercase `human`, `ai`, or `fusion` values. Align predictions by `id` after sorting, batching, or parallel inference. Do not include an index.
14. Use descriptive experiment names in notebook prose, code, tables, plots, captions, logs, and reports. Name the model and changed setting, such as `tfidf_char_word_logreg`, `xlmr_head_only`, or `xlmr_head_tail`, and include the data selection when relevant, such as `xlmr_head_tail_50k`. Use readable labels such as “XLM-R with head and tail tokens (50k training texts)” in presentation. Do not use opaque numbered run codes as public names. Keep historical checkpoint IDs only at the storage boundary; do not rename existing run directories or change saved checkpoint configurations just to relabel results.
15. Store completed experiment metrics, training logs, predictions, and submission metadata in a durable location when configured. Restore local records only when needed for rerunning, then delete temporary copies unless the user asks to keep them. Version research notebook outputs, reusable code, and instructions for restoring the records. Never upload competition data or private credentials to a public repository.

## Notebooks

01. Use Jupyter for notebooks. Store report and research notebooks as `.ipynb` files in `notebooks/`; keep execution-only notebooks in `experiments/`. Use the project's Python 3.14 environment when compatible, or a documented GPU-compatible kernel locally or remotely. Launch a local JupyterLab session from the repository root with `uv run jupyter lab notebooks/` when appropriate.
02. Use dedicated Markdown cells for static explanatory text. Avoid generating narrative paragraphs through `print()` or other output wrappers.
03. If narrative text has to be generated programmatically (e.g., for reporting variables), use `IPython.display.Markdown` with `display()`. Use `IPython.display.Image` for images and `ipywidgets` for interactive controls.
04. Do not use `print()` for output in notebooks. Use cell return values or `IPython.display.display()` for multiple outputs.
05. For slow iterative computations, use `tqdm.notebook.tqdm(iterator)` to display progress bars. Avoid verbose console output.
06. Let notebook cells display their results directly when possible. Avoid verbose console output.
07. Use consistent mathematical notation across code, plots, and explanations.
08. Treat notebooks as the primary presentation and experiment interface. Show data decisions, model configuration, the relevant training/evaluation calls, plots, comparisons, and conclusions there. Move repeated or testable implementation details into `src/` instead of duplicating them across notebooks.
09. Inspect notebook code and metadata for hardware, data, dependency, side effect, and runtime requirements before execution. Prefer an existing configured remote kernel with the required environment when available. Use a local kernel only for verified lightweight CPU notebooks.
10. Do not automatically rerun notebooks requiring a GPU, model training or fine-tuning, or other expensive computation, locally or remotely. Such runs require explicit user authorization and suitable compute.
11. When execution is appropriate, restart the selected kernel and run all cells in order. If suitable compute is unavailable, requirements are unclear, or a run is not authorized, validate notebook schema and code syntax without executing cells. Preserve saved outputs when skipping execution and report that execution was not run, why it was skipped, and which outputs may be stale. Follow `notebooks/README.md` for naming and output conventions, if present.
12. Prefer numbered notebooks such as `00_data.ipynb`, `01_eda.ipynb`, classical NLP baselines, multilingual encoder experiments, and final inference/submission. Adapt names to the repository instead of enforcing a rigid notebook count. Keep numbered files in logical dependency and presentation order, with consecutive prefixes within each directory. Number `notebooks/` and `experiments/` independently. After adding, moving or deleting a numbered file, check the complete sequence and update affected filenames, links and references together. Do not leave gaps from removed notebooks or rename historical experiment IDs and checkpoint directories to match notebook numbering.
13. Keep notebook outputs concise and useful for review. Avoid huge logs, embedded texts or datasets, and unnecessary checkpoint outputs. Make major results understandable without executing expensive training.
14. Write the notebook as a student research narrative in first-person singular when a personal statement is useful: use **I**, never **We**. Prefer concise neutral phrasing where repeated first-person statements would be distracting. Follow the concise structure of `00_data.ipynb` and `01_eda.ipynb`: question, assumptions, visible configuration, execution, measured results, and limitations. Keep the experiment matrix and scientific choices visible even when implementation is imported from `src/`.
15. Keep AI working notes, agent handoffs, internal task prompts, permission history, and development-status reports out of notebooks. Do not mention AI artifacts or `.artifacts/` in notebook prose, code, metadata, or outputs. Put platform operation instructions here, and keep only concise reproducibility notes in research notebooks. Do not invent student conclusions or replace the student-authored final report.
16. Use `$...$` for inline Markdown math and `$$...$$` for display math, with display delimiters on their own lines. Do not use `\(...\)` or `\[...\]` in Markdown; this rule does not apply to `.tex` sources.

## Hosted Experiment Execution

Use existing hosted notebooks and shared modules as working examples only when they
belong to this NLP repository. Do not reuse task-specific training code, paths, credentials, experiment identities,
or checkpoint formats from unrelated homework. If there is no
established hosted pipeline, create the smallest reproducible Colab or Kaggle
notebook needed for the selected experiment; do not pretend legacy infrastructure
already exists. Keep the research comparison in `notebooks/` and execution-only
hosted entry points in `experiments/`.

Hosted runs must use the same fixed validation IDs, class mapping, and metric as
local comparisons. Record the actual environment and model identity. Kaggle allows
up to five submissions per day and up to two selected final submissions.

### Shared Python modules

- Keep shared CSV loading, schema checks, split manifests, and dataset fingerprints
  in focused `src/` modules when multiple notebooks need them. A split manifest
  must preserve exact row IDs and fixed labels for local validation.
- Keep text vectorization and reusable stylometric feature extraction separate
  from classifier fitting. Fit every learned transformation on training data only;
  transform validation and test without refitting.
- Keep reusable tokenizer, sampling, batching, model training, checkpoint loading,
  and inference functions in focused `src/` modules only when they reduce real
  duplication. Preserve tokenizer/checkpoint compatibility.
- Reuse existing storage and checkpoint utilities if this NLP repository already
  has them. Otherwise use a small, explicit implementation rather than importing
  unrelated project utilities or adding an experiment-management framework.
- Keep a small bootstrap cell for dependencies and making `src/` importable when
  opening a hosted notebook from a project ZIP. Show actual model configurations,
  data selection, training calls, validation results, and conclusions in notebooks.

### Input bundle and paths

1. Use only `train.csv` for supervised training, fitting text vectorizers, building
   learned features, and selecting parameters. `test.csv` is for final inference,
   and `sample_submission.csv` determines submission IDs and row order. Never
   obtain the original GhostWriter dataset or other external data to improve
   predictions. Pretrained model weights are permitted.
2. Keep raw input files unchanged and store derived metadata and reproducible split
   manifests under `data/metadata/` when creating them. For local work, use
   repository-relative paths or configurable input locations. For Kaggle, inspect
   the actual attached competition dataset path; for Colab, stage verified files
   to fast local runtime storage if needed.
3. Bundle only the code and configurations needed by the hosted experiment.
   Exclude `.env`, credentials, `.artifacts/`, caches, competition CSVs, downloaded
   pretrained model weights, and unnecessary large outputs. Verify imports and
   archive integrity before use; local computer paths do not exist on hosted GPUs.
4. Use descriptive local output directories such as
   `runs/<experiment_group>/<run_id>/`. If Google Drive persistence is configured,
   keep a matching durable per-run layout and make its root configurable. Do not
   assume Drive folders or token secrets from unrelated projects exist.
5. When changing shared modules, update and verify their hosted consumers together
   before the next launch. Do not overwrite a running runtime's source tree,
   checkpoint files, or active experiment for cleanup.
6. Create every destination's parent directory before opening a download,
   temporary file, or archive. A fresh Kaggle or Colab runtime may not have
   previously created temporary directories. Verify disk capacity for model
   weights, tokenized data, checkpoints, and outputs.

### Colab launch

1. Open the Colab entry notebook manually, select a suitable CPU or GPU runtime,
   and inspect its setup before execution. Use native Drive mounting only when
   Drive persistence is actually configured and the user completes authorization.
2. Retain the hosted runtime's compatible CUDA/PyTorch stack. Install missing
   auxiliary packages with `uv` when supported; do not apply a Python 3.14 lock
   to an incompatible hosted interpreter. Record Python, CUDA, GPU, Transformers,
   tokenizers, scikit-learn, and other relevant library versions.
3. Inspect the fixed split, selected training sample, effective max token length,
   pretrained checkpoint, class mapping, and experiment table before training.
   Keep synthetic storage/GPU probes out of the training path. Do not launch
   simultaneous runs that share the same checkpoint destination.
4. A notebook setup cell must not silently trigger fine-tuning. Make training an
   explicit stage. If using a Run All workflow, state whether it will train and
   whether user authorization is required.
5. Browser disconnection may leave the runtime running, but session expiry and GPU
   quota can terminate it. Inspect active execution and saved progress before
   restarting. Do not promise unattended completion or use anti-idle workarounds.
6. If a hosted execution method has not been verified in this repository, do not
   claim it works. Prefer supported native notebook execution over unverified
   browser/CLI automation.

### Kaggle launch with the same Google Drive storage

1. Import the selected experiment notebook into a private Kaggle notebook. Attach
   the competition data, select an available GPU, and enable Internet only if
   needed and permitted for downloading pretrained open-weight models or packages.
   Never use Internet access to retrieve external text data for prediction.
2. Confirm competition input files, compute quota, writeable paths, and available
   GPU memory. Kaggle sessions are temporary; `/kaggle/working` must not be treated
   as durable storage without saving outputs or publishing a notebook version.
3. When Drive synchronization is already configured, use the repository's existing
   authentication approach. If adding it, require a user-authorized OAuth client
   and store tokens only in Kaggle Secrets or private local configuration. Do not
   reuse unrelated project tokens or hardcode a nonexistent secret name.
4. Read secrets through the platform's supported secrets interface when necessary.
   Never place tokens in notebook cells, CSVs, logs, project ZIPs, or Git. Verify
   remote folder permissions and actual upload/download behavior before depending
   on remote checkpoint restoration.
5. Stage the exact selected training subset and validation IDs; verify input
   fingerprints and tokenizer files. Restore only an experiment with a compatible
   config, model, tokenizer, label mapping, and split. Check available disk and
   storage quota before training.
6. Execute setup, inspect effective configuration and restored progress, then start
   the explicit training cell. For a native saved Kaggle run, use the platform's
   Save Version workflow if suitable. Stop interactive training before starting
   another writer to the same run directory.
7. Session limits bound one invocation, not the total training budget. Leave time
   for checkpoint persistence, final validation, and complete test inference.
   On interruption, rerun with the same run identity and exact recipe; skip
   completed work and resume only from verified compatible checkpoints.

### Test inference and submission export

1. Prepare test inference and CSV export with every final candidate model. Prefer
   performing GPU inference in the hosted environment holding its checkpoint.
   Do not default to downloading large checkpoints to the local machine. Local
   heavy inference requires an explicit request.
2. Keep inference-only execution for completed checkpoints. It must restore the
   selected final or explicitly chosen best model and its tokenizer, skip training,
   and predict every test row. Run the same preprocessing and long-text policy
   used during validation.
3. Record checkpoint and input-file fingerprints, model and tokenizer identity,
   class mapping, `max_length`, chunking/aggregation settings, and selection rule
   alongside each CSV. If blending, preserve the exact model order, probability
   columns, weights, and validation protocol. Align results by original `id`.
4. Export a CSV with exactly `id,label` in `sample_submission.csv` row order,
   exactly 25,000 distinct test IDs, and labels from `human`, `ai`, or `fusion`.
   Do not include probability columns, an index, or missing predictions.
5. Write descriptive filenames under a submissions output directory. Save a fully
   written CSV atomically when possible, validate it against the sample file,
   and persist export metadata. Reuse completed exports only when input,
   checkpoint, config, and CSV hashes match. Do not submit automatically.
6. Select at most two final Kaggle submissions by the stated deadline. Keep the
   number of submissions within the daily limit and identify the selected files
   for the student-authored technical report.

### Required pipeline for every future experiment

01. Keep the operational pipeline consistent across classical NLP baselines,
    encoder backbones, and long-text experiments. Reuse data loading, split
    manifests, metric computation, artifact export, and verified checkpoints
    rather than creating a new framework for each model.
02. Use only authorized competition inputs and permitted pretrained model weights.
    Preserve all `id`, `text`, `lang`, `domain`, and `label` values in raw records.
    Never add the public GhostWriter corpus, scraped text, synthetic training
    examples, or proprietary LLM features to improve the score.
03. Package and verify the code needed by the hosted notebook. Keep secrets,
    datasets, checkpoints, caches, and outputs outside source ZIPs. Updating a
    ZIP does not change notebook cells already uploaded to Kaggle; identify
    precisely which bundle or notebook needs replacing.
04. Make setup work in an empty runtime. Create missing parent directories,
    verify dependency versions, hardware, credentials when used, write access,
    disk capacity, and actual CSV schemas. Verify fixed train/validation IDs,
    class counts, language/domain distributions, and test input availability
    when submission export is configured.
05. Show effective configuration, dataset subset, seed, train/validation split,
    model checkpoint, tokenizer length limits, output paths, and run identity
    before training. Make all environment overrides visible. Preserve existing
    scientific settings when fixing operational issues.
06. Start only explicitly requested training. Keep synthetic GPU/storage probes
    out of the normal training path. Log effective training controls so skipped
    training is distinguishable from actual training, and do not make a probe
    file a prerequisite for real work.
07. Persist complete training state using atomic local writes and verified
    durable copies when remote persistence is configured. Save periodically,
    after validation, and on completion. Resume only when the exact run recipe
    and checkpoint metadata match; stop on persistence failure.
08. Test actual setup, dataset loading, split restoration, tokenization,
    checkpoint loading, and submission export with focused CPU or mocked tests.
    Syntax/type checks alone do not verify hosted setup. Distinguish simulated,
    real API, and GPU verification in reporting; never run full training merely
    to test bootstrap or formatting.
09. Store per-run configuration, split fingerprints, environment information,
    training history, macro-F1, per-class metrics, confusion matrices, and logs
    alongside checkpoints. Save comparison tables and configured plots after
    completion; persist submission CSVs and export metadata for selected runs.
10. Confirm completion from a finished validation and verified final artifacts,
    not epoch count alone. Report training time separately from elapsed time.
    Distinguish measured results from supplied logs and independently inspected
    files. Never invent metrics or restart completed training just to inspect
    status.

### Automatic checkpoints and recovery

Training entry points must work in a fresh hosted runtime without diagnostic
switches or temporary probe files. Keep training disabled in research notebooks
unless explicitly authorized. If a hosted run trains multiple configurations,
use one run directory per experiment, check storage and session limits, skip
completed runs, and stop the queue when a run pauses. Preserve checkpoint
integrity checks and never assume recovery is available until verified.

1. Save complete training state as appropriate: model, optimizer, scheduler,
   mixed-precision scaler, exact config, completed epoch, optimizer step,
   training history, and RNG states. Record the seed, dataset and split hashes,
   class mapping, tokenizer identity, max token length, and sampling order.
   Preserve the sampler/batch cursor when supported by the training loop.
2. Write checkpoints locally using atomic replacement, then publish a finished
   durable snapshot if remote storage is configured. Checkpoint periodically
   and at epoch boundaries, after validation, and on completion. Define best
   checkpoint selection before examining results, usually by validation
   macro-F1; do not silently change the selection protocol between runs.
3. For remote snapshots, verify destination checksums, publish the manifest
   after its files, and update the current verified generation only after
   successful verification. Retain a preceding known-good generation when
   practical and reuse unchanged files rather than uploading the same bytes.
4. A persistence failure must stop training if successful recovery is a
   required part of the experiment. Never rely solely on temporary Colab or
   Kaggle paths for durable checkpoints. Save metrics, class-wise reports,
   logs, environment details, and timing with the selected checkpoint.
5. Before resuming, verify experiment ID, model backbone, classifier head,
   tokenizer vocabulary/checkpoint, label order, training-subset and split
   hashes, sequence length, chunk policy, optimizer, scheduler, and epoch
   budget. Restore the original trainable-parameter/frozen-layer policy before
   loading optimizer state. Never silently restart from pretrained weights.
6. Work since the last durable save may repeat. Report rollback or runtime
   changes and do not claim bit-for-bit GPU determinism or guaranteed session
   duration. Preserve validation coverage and training budget.
7. Verify checkpoint changes with focused CPU save/restore tests and notebook
   schema/syntax checks. Do not launch training, overwrite live states, or
   rerun GPU notebooks merely to check formatting or documentation.

## Plotting

1. Use Plotly Express for quick visualizations where possible. Only if Plotly Express does not support the required functionality, use Plotly's `graph_objects` API.
2. Use Plotly for charts; do not add another plotting library or custom style without a concrete need. Use confusion matrices, class/language/domain distributions, text-length histograms, and training curves as appropriate. Framework-native evaluation plots are acceptable when relevant.
3. Always label axes, legends, and titles. Use clear and descriptive labels. Avoid generic labels such as "x", "y", or "value". Axes, legends and titles should be capitalized and use sentence case. Avoid all caps, camel case, or snake case.

## Writing

1. Write concise, technical explanations without unnecessary repetition.
2. Avoid dash punctuation in prose.
3. Keep mathematical symbols and terminology consistent.
4. Prefer clear descriptions of assumptions, decisions, and results over generic commentary. Follow the assignment reporting rules: the student authors the final report; AI assistance is for code and review, not an AI-written final report.

## Git Conventions

Use Conventional Commits with these prefixes:

`feat:`, `fix:`, `docs:`, `exp:`, `refactor:`, `test:`, `chore:`.

Commit subjects should describe the change. Commit bodies should explain **why the change was made**, including important reasoning and tradeoffs, rather than repeating the diff.

For `exp:` commits, include the measured result, evaluation setup, and location of detailed experiment records. Never invent metrics.

Do not combine unrelated changes into one commit. Do not create commits unless explicitly instructed.

## Artifacts and Research Notes

01. Reserve `.artifacts/` exclusively for AI working notes, temporary investigation outputs, and agent-generated reports. It is not a project data or runtime output directory.
02. Store useful intermediate findings when they help explain decisions, reproduce experiments, or continue work in a later session.
03. Do not reference or depend on `.artifacts/` in project code, notebooks, or runtime configuration. Do not store datasets, prepared data, metadata caches, or image previews there. Store project data and derived metadata under `data/`, such as `data/metadata/` and `data/previews/`, and keep competition CSVs, generated metadata under `data/metadata/`, token caches, and text previews excluded from Git.
04. Organize artifacts by task or experiment. Use descriptive filenames and avoid unnecessary files.
05. Prefer updating an existing artifact over creating multiple overlapping reports.
06. Do not create documentation or artifacts for trivial changes.
07. Record relevant commands, configurations, seeds, metrics, and conclusions when reporting experimental results.
08. Clearly distinguish measured results, hypotheses, assumptions, and unverified observations.
09. Keep `.artifacts/` excluded from Git. Do not stage or commit its contents.
10. Do not store credentials, tokens, private information, or unnecessary copies of large datasets in artifacts.
11. Important validated findings intended for the project history should be promoted to tracked documentation when explicitly requested.
12. Summarize significant findings and reference relevant artifact paths in the final response.

## Code Changes and Cleanup

01. All substantial code changes must pass the relevant test suite, formatting, linting, and static type checks.
02. Preserve existing functionality unless the task explicitly requires changing it.
03. When replacing an implementation, remove obsolete functions, classes, imports, configuration, and references that are no longer used.
04. Do not leave commented-out implementations, redundant wrappers, or compatibility code that serves no purpose.
05. Do not write large numbers of tests. Keep the test suite small and focused on critical behavior and important regressions. Update affected tests when behavior or interfaces change. Exhaustive coverage of every function, branch, or component is not required. Do not add tests solely to increase coverage or mirror implementation details; reason about correctness before deciding what needs testing.
06. Prefer modifying existing code over introducing parallel implementations of the same behavior.
07. Keep changes minimal, readable, idiomatic, and consistent with project conventions.
08. Do not perform unrelated cleanup outside the task scope.
09. Before removing code, verify that it is no longer referenced by relevant callers, tests, or supported interfaces.
10. Do not suppress type errors, disable tests, or weaken quality checks to make changes pass.
11. If verification fails, investigate and fix issues introduced by the change. Report pre-existing failures separately.
12. Do not describe a task as complete when required checks are failing or have not been executed. State the remaining blockers explicitly.
13. For substantial changes, optionally record implementation decisions, removed legacy code, verification results, and unresolved issues in `.artifacts/`.
