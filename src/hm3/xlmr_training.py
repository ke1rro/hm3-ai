import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import EvalPrediction

from hm3.encoder_data import ID2LABEL, LABEL2ID, TokenizedDocuments, prediction_table
from hm3.validation import LABELS
from hm3.xlmr_checkpoints import publish_checkpoint, verify_checkpoint

logger = logging.getLogger(__name__)


def write_record(path: Path, record: dict[str, object]) -> None:
    """Atomically write a generated experiment or completion record.

    Args:
        path: JSON destination.
        record: JSON-compatible experiment data.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_classifier(
    model_id: str,
    revision: str | None,
    is_initial: bool,
    cache_dir: Path | None = None,
) -> PreTrainedModel:
    """Load XLM-R with explicit labels and declare its mean document-loss semantics.

    Prevent Trainer from inferring accumulation-normalized loss from forward kwargs.

    Args:
        model_id: Open-weight checkpoint identifier or verified local model directory.
        revision: Pinned Hub commit, or omitted for local checkpoints.
        is_initial: Initialize the three-class head for the first pretrained load.
        cache_dir: Optional writable cache for downloaded open-weight files.

    Returns:
        A document-level model with human=0, ai=1 and fusion=2.
    """
    options = {"num_labels": 3, "id2label": ID2LABEL, "label2id": LABEL2ID} if is_initial else {}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, revision=revision, cache_dir=cache_dir, local_files_only=not is_initial, **options
    )
    if (
        model.config.id2label != ID2LABEL
        or model.config.label2id != LABEL2ID
        or model.config.num_labels != 3
    ):
        raise ValueError("Classifier output mapping is incompatible with human/ai/fusion ordering")
    model.accepts_loss_kwargs = False
    return model


def inspect_gpu() -> dict[str, object]:
    """Require one visible CUDA GPU and report capacity before allocating model weights.

    Returns:
        GPU identity, CUDA/PyTorch versions and free/total device memory.
    """
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Select a CUDA GPU runtime and expose exactly one GPU for this experiment")
    properties = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info(0)
    if total < 12 * 1024**3:
        raise ValueError("This initial batch configuration requires at least 12 GiB GPU memory")
    return {
        "gpu": properties.name,
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "free_vram_gib": free / 1024**3,
        "total_vram_gib": total / 1024**3,
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "visible_gpu_count": 1,
    }


def development_metrics(prediction: EvalPrediction) -> dict[str, float]:
    """Calculate checkpoint-selection metrics from development-only predictions.

    Args:
        prediction: Trainer logits and known development evaluation labels.

    Returns:
        Fixed three-class macro-F1 and F1 for each explicit class.
    """
    logits = (
        prediction.predictions[0]
        if isinstance(prediction.predictions, tuple)
        else prediction.predictions
    )
    predicted = np.asarray(logits).argmax(axis=1)
    labels = np.asarray(prediction.label_ids)
    per_class = f1_score(labels, predicted, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "macro_f1": float(np.mean(per_class)),
        **{f"f1_{label}": float(per_class[index]) for index, label in enumerate(LABELS)},
    }


class CheckpointReceipts(TrainerCallback):
    """Publish complete Trainer state generations and concise training history.

    Attributes:
        identity: Exact run identity required for checkpoint compatibility.
        run_path: Experiment directory containing history and checkpoints.
    """

    def __init__(self, identity: dict[str, object], run_path: Path) -> None:
        self.identity = identity
        self.run_path = run_path

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> TrainerControl:
        """Publish checksums only after Trainer completes its checkpoint writes."""
        publish_checkpoint(
            self.run_path / "checkpoints" / f"checkpoint-{state.global_step}", self.identity, True
        )
        return control

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> TrainerControl:
        """Persist loss and development metrics after each logging event."""
        write_record(self.run_path / "history.json", {"events": state.log_history})
        return control


def find_resume_checkpoint(run_path: Path, identity: dict[str, object]) -> Path | None:
    """Select the latest published complete training checkpoint without trusting partial writes.

    Args:
        run_path: Experiment directory containing Trainer checkpoints.
        identity: Required run identity.

    Returns:
        Latest verified checkpoint or no checkpoint for a fresh run.
    """
    checkpoints = sorted(
        (run_path / "checkpoints").glob("checkpoint-*"),
        key=lambda path: int(path.name.split("-")[-1]),
        reverse=True,
    )
    for path in checkpoints:
        if (path / "checkpoint_manifest.json").exists():
            verify_checkpoint(path, identity, True)
            state = json.loads((path / "trainer_state.json").read_text())
            if state.get("best_model_checkpoint"):
                best = Path(state["best_model_checkpoint"])
                if not best.resolve().is_relative_to((run_path / "checkpoints").resolve()):
                    raise ValueError("Restore the original run root before resuming training")
                verify_checkpoint(best, identity, True)
            return path
    if checkpoints:
        raise ValueError(
            "Only incomplete checkpoints exist; restore a verified generation before training"
        )
    return None


def checkpoint_is_terminal(path: Path, is_refit: bool) -> bool:
    """Recognize exhausted step budgets or restored early-stopping patience.

    Args:
        path: Verified Trainer checkpoint.
        is_refit: Ignore early-stopping state for fixed-budget all-data refit.

    Returns:
        Whether selected weights can be finalized without additional optimizer steps.
    """
    state = json.loads((path / "trainer_state.json").read_text())
    if state["global_step"] >= state["max_steps"]:
        return True
    stopping = state.get("stateful_callbacks", {}).get("EarlyStoppingCallback")
    return bool(
        not is_refit
        and stopping
        and stopping["attributes"]["early_stopping_patience_counter"]
        >= stopping["args"]["early_stopping_patience"]
    )


def train_encoder(
    training: pl.DataFrame,
    evaluation: pl.DataFrame | None,
    tokenizer: PreTrainedTokenizerBase,
    config: dict,
    run_path: Path,
    identity: dict[str, object],
    is_refit: bool = False,
    cache_dir: Path | None = None,
) -> Path:
    """Train or resume one selected truncation experiment on explicitly supplied document IDs.

    Use development-only macro-F1 for early stopping and checkpoint selection. Full
    all-data refit uses the configured fixed epoch budget with no holdout evaluation.
    Publish inference weights atomically; subsequent invocations restore completion.

    Args:
        training: Tokenized training documents only.
        evaluation: Separate tokenized development documents, omitted for all-data refit.
        tokenizer: Matching pinned pretrained tokenizer.
        config: Visible experiment settings.
        run_path: Separate per-policy/per-mode directory.
        identity: Exact model/data/tokenizer/recipe/environment identity.
        is_refit: Run all-data fixed-budget training without early stopping.
        cache_dir: Writable pretrained download cache, separate from saved run outputs.

    Returns:
        Verified completed selected-model directory for inference.
    """
    inspect_gpu()
    selected_path = run_path / "selected_model"
    completion = run_path / "training_complete.json"
    if completion.exists():
        record = json.loads(completion.read_text())
        if record["identity"] != identity:
            raise ValueError("Completed training has an incompatible recipe")
        verify_checkpoint(selected_path, identity, False)
        return selected_path
    if selected_path.exists():
        verify_checkpoint(selected_path, identity, False)
        state_file = selected_path / "selection_state.json"
        selection = json.loads(state_file.read_text())
        write_record(completion, {"identity": identity, **selection})
        return selected_path
    run_path.mkdir(parents=True, exist_ok=True)
    set_seed(config["seed"])
    resume = find_resume_checkpoint(run_path, identity)
    has_terminal_checkpoint = resume is not None and checkpoint_is_terminal(resume, is_refit)
    restore_path = resume
    restored_state = (
        TrainerState.load_from_json(str(resume / "trainer_state.json")) if resume else None
    )
    if has_terminal_checkpoint and not is_refit and restored_state is not None:
        restore_path = Path(restored_state.best_model_checkpoint)
    model = load_classifier(
        str(restore_path) if restore_path else config["model_id"],
        None if resume else config["revision"],
        not bool(resume),
        cache_dir,
    )
    arguments = TrainingArguments(
        output_dir=str(run_path / "checkpoints"),
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        fp16=config["fp16"],
        bf16=False,
        optim="adamw_torch",
        gradient_checkpointing=config["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="no" if is_refit else "steps",
        eval_steps=config["eval_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        logging_steps=config["logging_steps"],
        save_total_limit=2,
        load_best_model_at_end=not is_refit,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        seed=config["seed"],
        data_seed=config["seed"],
        report_to="none",
        push_to_hub=False,
        dataloader_num_workers=0,
        restore_callback_states_from_checkpoint=True,
        ignore_data_skip=False,
        disable_tqdm=True,
    )
    callbacks: list[TrainerCallback] = [CheckpointReceipts(identity, run_path)]
    if not is_refit:
        if evaluation is None:
            raise ValueError("Early stopping requires a separate development evaluation dataset")
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=config["early_stopping_patience"])
        )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=TokenizedDocuments(training),
        eval_dataset=TokenizedDocuments(evaluation) if evaluation is not None else None,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        processing_class=tokenizer,
        compute_metrics=development_metrics,
        callbacks=callbacks,
    )
    logger.info("Training %s; resume checkpoint: %s", config["experiment"], resume)
    if has_terminal_checkpoint and restored_state is not None:
        trainer.state = restored_state
        train_metrics = {"restored_terminal_checkpoint": True}
        logger.info("Finalizing terminal checkpoint without additional optimizer steps")
    else:
        result = trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        train_metrics = result.metrics
    selection = {
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_development_macro_f1": trainer.state.best_metric,
        "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch,
        "train_metrics": train_metrics,
        "selection_rule": (
            "last fixed-budget model" if is_refit else "best development-only macro-F1"
        ),
    }
    with tempfile.TemporaryDirectory(prefix=".selected-", dir=run_path) as staging:
        staged = Path(staging) / "selected_model"
        trainer.save_model(str(staged))
        tokenizer.save_pretrained(staged)
        write_record(staged / "selection_state.json", selection)
        publish_checkpoint(staged, identity, False)
        staged.replace(selected_path)
    write_record(completion, {"identity": identity, **selection})
    return selected_path


def predict_encoder(
    model_path: Path,
    tokens: pl.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    config: dict,
    identity: dict[str, object],
    true_labels: list[str] | None = None,
) -> pl.DataFrame:
    """Restore verified selected weights and produce ID-aligned logits and probabilities.

    Args:
        model_path: Published selected-model directory.
        tokens: Token arrays in deterministic ID order.
        tokenizer: Matching tokenizer.
        config: Batch size and FP16 settings.
        identity: Required selected-model provenance.
        true_labels: Labels aligned with token IDs for labeled evaluation.

    Returns:
        One prediction row per document with explicit class-column order.
    """
    inspect_gpu()
    verify_checkpoint(model_path, identity, False)
    model = load_classifier(str(model_path), None, False)
    arguments = TrainingArguments(
        output_dir=str(model_path.parent / "inference"),
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        fp16=config["fp16"],
        report_to="none",
        disable_tqdm=True,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
    )
    predictions = trainer.predict(
        TokenizedDocuments(tokens.drop("labels") if "labels" in tokens.columns else tokens)
    )
    logits = (
        predictions.predictions[0]
        if isinstance(predictions.predictions, tuple)
        else predictions.predictions
    )
    return prediction_table(tokens["id"].to_list(), np.asarray(logits), true_labels)
