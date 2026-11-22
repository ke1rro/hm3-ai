import argparse
import hashlib
import json
import logging
import platform
import shutil
import sys
import tempfile
import tomllib
from importlib.metadata import version
from pathlib import Path
from typing import cast

import plotly.express as px
import polars as pl
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from hm3.encoder_data import tokenize_records, validate_encoder_manifest
from hm3.evaluation import classification_metrics, export_submission, subgroup_metrics
from hm3.kaggle_bootstrap import find_competition_inputs, verify_bundle
from hm3.validation import LABELS, fingerprint_file, load_training_data, validate_manifest
from hm3.xlmr_checkpoints import verify_checkpoint
from hm3.xlmr_training import inspect_gpu, predict_encoder, train_encoder, write_record

logger = logging.getLogger(__name__)


def _restore_run(inputs: Path, destination: Path, identity: dict[str, object]) -> None:
    if destination.exists():
        saved = json.loads((destination / "run_metadata.json").read_text())
        if saved["identity"] != identity:
            raise ValueError("Working run identity is incompatible")
        return
    matches = []
    for path in inputs.rglob("run_metadata.json"):
        prior = json.loads(path.read_text()).get("identity", {})
        if prior == identity:
            matches.append(path.parent)
        elif (
            prior.get("configuration") == identity["configuration"]
            and prior.get("scope") == identity["scope"]
        ):
            raise ValueError(
                "Attached XLM-R recipe has an incompatible runtime or source identity; "
                "restore its original runtime/bundle instead of silently restarting"
            )
    if len(matches) > 1:
        raise ValueError("Attach one prior saved-output version for this XLM-R recipe")
    if matches:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".restore-", dir=destination.parent) as staging:
            staged = Path(staging) / destination.name
            shutil.copytree(matches[0], staged)
            staged.replace(destination)
        logger.info("Restored prior XLM-R outputs: %s", destination)


def _prepare(args: argparse.Namespace) -> tuple:
    verify_bundle(args.code_root, "xlmr_code")
    verify_bundle(args.split_root, "xlmr_split")
    document = tomllib.loads((args.code_root / "config/xlmr.toml").read_text())
    if args.experiment not in document["experiments"]:
        raise ValueError("Choose one declared XLM-R experiment")
    config = {key: value for key, value in document.items() if key != "experiments"}
    config.update(document["experiments"][args.experiment])
    config["experiment"] = args.experiment
    inputs = find_competition_inputs(args.input_root, config["source_sha256"])
    frozen_path = args.split_root / "data/metadata/validation_manifest.parquet"
    sample_path = args.split_root / "data/metadata/xlmr_development_40k.parquet"
    if (
        fingerprint_file(frozen_path) != config["frozen_manifest_sha256"]
        or fingerprint_file(sample_path) != config["encoder_manifest_sha256"]
    ):
        raise ValueError("Uploaded frozen split or common encoder sample differs from the recipe")
    data = load_training_data(inputs["train"])
    frozen, selected = pl.read_parquet(frozen_path), pl.read_parquet(sample_path)
    validate_manifest(data, frozen)
    validate_encoder_manifest(frozen, selected)
    sample_record = json.loads(sample_path.with_suffix(".json").read_text())
    if (
        sample_record["parent_manifest_sha256"] != config["frozen_manifest_sha256"]
        or sample_record["train_target"] != config["train_target"]
        or sample_record["evaluation_target"] != config["evaluation_target"]
        or sample_record["seed"] != config["seed"]
    ):
        raise ValueError("Encoder sampling provenance is incompatible")
    gpu = inspect_gpu()
    if config["fp16"] and torch.cuda.get_device_capability(0)[0] < 7:
        raise ValueError("This FP16 recipe requires a supported CUDA GPU")
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    free_disk = shutil.disk_usage(args.output_root).free / 1024**3
    if free_disk < config["minimum_free_disk_gib"]:
        raise ValueError("Insufficient free disk for best/latest checkpoints and model downloads")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"], revision=config["revision"], cache_dir=args.cache_root, use_fast=True
    )
    if tokenizer is None or not tokenizer.is_fast:
        raise ValueError("This experiment requires the pinned fast XLM-R tokenizer")
    tokenizer_hash = hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()
    environment = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": gpu["gpu"],
        **{
            name: version(name)
            for name in [
                "transformers",
                "accelerate",
                "tokenizers",
                "numpy",
                "polars",
                "scikit-learn",
            ]
        },
    }
    scope = "all_train" if args.mode == "refit" else args.scope
    if args.mode in {"train", "holdout"} and scope != "sampled":
        raise ValueError(
            "Training/holdout modes use sampled development only; choose refit explicitly"
        )
    identity: dict[str, object] = {
        "configuration": config,
        "scope": scope,
        "environment": environment,
        "tokenizer_sha256": tokenizer_hash,
        "label_order": list(LABELS),
        "precision": "fp16" if config["fp16"] else "fp32",
        "source_modules": {
            name: fingerprint_file(args.code_root / f"src/hm3/{name}.py")
            for name in [
                "encoder_data",
                "xlmr_training",
                "xlmr_checkpoints",
                "xlmr_kaggle",
                "validation",
                "evaluation",
            ]
        },
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    name = config["experiment"] + ("_all_train" if scope == "all_train" else "")
    run = args.output_root / "runs" / name / digest
    _restore_run(args.input_root, run, identity)
    run.mkdir(parents=True, exist_ok=True)
    if not (run / "run_metadata.json").exists():
        write_record(
            run / "run_metadata.json",
            {"identity": identity, "gpu": gpu, "free_disk_gib": free_disk, "run_id": digest},
        )
    saved = run / "data/metadata"
    saved.mkdir(parents=True, exist_ok=True)
    for source in [
        frozen_path,
        frozen_path.with_suffix(".json"),
        sample_path,
        sample_path.with_suffix(".json"),
    ]:
        destination = saved / source.name
        if destination.exists() and fingerprint_file(destination) != fingerprint_file(source):
            raise ValueError("Saved run contains a different split manifest")
        if not destination.exists():
            shutil.copy2(source, destination)
    write_record(
        args.output_root / "current_run.json",
        {
            "run_directory": str(run.relative_to(args.output_root)),
            "experiment": config["experiment"],
            "scope": scope,
        },
    )
    logger.info(
        "Preflight complete: %s; scope=%s; GPU=%s; roles=%s",
        config["experiment"],
        scope,
        gpu,
        sample_record["role_counts"],
    )
    logger.info("Effective model/training settings: %s", config)
    return data, selected, tokenizer, config, identity, run, inputs, scope


def _tokens(
    records: pl.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    config: dict,
    run: Path,
    role: str,
) -> pl.DataFrame:
    records = records.sort("id")
    cache = run / "token_cache" / f"{role}.parquet"
    receipt = cache.with_suffix(".json")
    identity = {
        "ids_labels_sha256": hashlib.sha256(
            records.select(*[name for name in ["id", "label"] if name in records.columns])
            .write_csv()
            .encode()
        ).hexdigest(),
        "max_length": config["max_length"],
        "policy": config["truncation_policy"],
        "revision": config["revision"],
    }
    if cache.exists() and receipt.exists():
        record = json.loads(receipt.read_text())
        if record["identity"] != identity or record["sha256"] != fingerprint_file(cache):
            raise ValueError("Cached tokens are incompatible or corrupted")
        return pl.read_parquet(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tokens = tokenize_records(records, tokenizer, config["max_length"], config["truncation_policy"])
    temporary = cache.with_suffix(".parquet.tmp")
    tokens.write_parquet(temporary)
    temporary.replace(cache)
    write_record(receipt, {"identity": identity, "sha256": fingerprint_file(cache)})
    return tokens


def _role(data: pl.DataFrame, selected: pl.DataFrame, role: str) -> pl.DataFrame:
    return data.join(
        selected.filter(pl.col("role") == role).select("id"), on="id", how="inner", validate="1:1"
    ).sort("id")


def _holdout(
    data: pl.DataFrame,
    selected: pl.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    config: dict,
    identity: dict[str, object],
    run: Path,
) -> None:
    model = run / "selected_model"
    verify_checkpoint(model, identity, False)
    receipt = run / "holdout_record.json"
    model_hash = fingerprint_file(model / "checkpoint_manifest.json")
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved["identity"] != identity or saved["model_sha256"] != model_hash:
            raise ValueError("Completed holdout result belongs to another checkpoint")
        for name, expected in saved["files"].items():
            if fingerprint_file(run / name) != expected:
                raise ValueError("Saved holdout predictions or metrics changed")
        logger.info("Restored verified complete holdout evaluation")
        return
    records = _role(data, selected, "holdout")
    tokens = _tokens(records, tokenizer, config, run, "holdout")
    predictions = predict_encoder(
        model, tokens, tokenizer, config, identity, records["label"].to_list()
    )
    metadata = records.select(
        "id", "lang", "domain", pl.col("text").str.count_matches(r"\S+").alias("words")
    )
    predictions = predictions.join(
        metadata, on="id", how="left", maintain_order="left"
    ).with_columns(
        pl.when(pl.col("words") <= 128)
        .then(pl.lit("001-128"))
        .when(pl.col("words") <= 512)
        .then(pl.lit("129-512"))
        .when(pl.col("words") <= 1024)
        .then(pl.lit("513-1024"))
        .otherwise(pl.lit("1025+"))
        .alias("length_bucket")
    )
    predictions.write_parquet(run / "holdout_predictions.parquet")
    metrics = classification_metrics(
        predictions["label"].to_list(), predictions["prediction"].to_list()
    )
    write_record(run / "holdout_metrics.json", metrics)
    for columns in [["lang"], ["domain"], ["length_bucket"], ["lang", "domain"]]:
        subgroup_metrics(predictions, columns).write_json(
            run / f"subgroup_{'_'.join(columns)}.json"
        )
    for key in ["confusion_matrix", "row_normalized_confusion_matrix"]:
        px.imshow(
            metrics[key],
            x=list(LABELS),
            y=list(LABELS),
            text_auto=True,
            title=f"{config['experiment']}: {key.replace('_', ' ')}",
            labels={"x": "Predicted class", "y": "True class", "color": "Value"},
        ).write_html(run / f"{key}.html", include_plotlyjs="directory")
    write_record(
        run / "holdout_prediction_metadata.json",
        {
            "identity": identity,
            "model_sha256": model_hash,
            "rows": predictions.height,
            "label_order": list(LABELS),
            "token_policy": config["truncation_policy"],
            "selection": json.loads((model / "selection_state.json").read_text()),
            "columns": predictions.columns,
        },
    )
    matrix = cast(list[list[int]], metrics["confusion_matrix"])
    report = (
        f"# Internal measured XLM-R findings\n\nExperiment: {config['experiment']}. "
        f"Model: {config['model_id']} at {config['revision']}.\n\n"
        f"Fixed-holdout rows: {predictions.height}. Macro-F1: {metrics['macro_f1']:.6f}; "
        f"accuracy: {metrics['accuracy']:.6f}; weighted-F1: {metrics['weighted_f1']:.6f}.\n\n"
        f"Per-class measurements: {json.dumps(metrics['per_class'])}.\n\n"
        f"Fusion to human: {matrix[2][0]}; fusion to ai: {matrix[2][1]}; "
        f"human to fusion: {matrix[0][2]}; ai to fusion: {matrix[1][2]}.\n\n"
        "Checkpoint selection used development evaluation only. Unknown document-family "
        "leakage remains possible. Baseline and truncation-policy superiority have not "
        "been inferred from these numbers. This is an internal engineering record, "
        "not the student-authored final report.\n"
    )
    (run / "internal_findings.md").write_text(report, encoding="utf-8")
    write_record(
        receipt,
        {
            "identity": identity,
            "model_sha256": model_hash,
            "files": {
                name: fingerprint_file(run / name)
                for name in ["holdout_predictions.parquet", "holdout_metrics.json"]
            },
        },
    )
    logger.info("Complete fixed-holdout metrics: %s", metrics)


def _inference(
    data_inputs: dict[str, Path],
    tokenizer: PreTrainedTokenizerBase,
    config: dict,
    identity: dict[str, object],
    run: Path,
) -> None:
    model = run / "selected_model"
    verify_checkpoint(model, identity, False)
    output = run / "submissions" / f"{config['experiment']}_{identity['scope']}.csv"
    expected = {
        "identity": identity,
        "model_sha256": fingerprint_file(model / "checkpoint_manifest.json"),
        "test_sha256": fingerprint_file(data_inputs["test"]),
        "sample_sha256": fingerprint_file(data_inputs["sample"]),
    }
    if output.exists() and output.with_suffix(".json").exists():
        record = json.loads(output.with_suffix(".json").read_text())
        if any(record[key] != value for key, value in expected.items()) or record[
            "csv_sha256"
        ] != fingerprint_file(output):
            raise ValueError("Completed test export is incompatible")
        logger.info("Restored verified submission %s", output)
        return
    test = pl.read_csv(data_inputs["test"], schema_overrides={"id": pl.String}).sort("id")
    sample = pl.read_csv(data_inputs["sample"], schema_overrides={"id": pl.String})
    tokens = _tokens(test, tokenizer, config, run, "test")
    predictions = predict_encoder(model, tokens, tokenizer, config, identity)
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(output.parent / "test_predictions.parquet")
    export_submission(
        predictions.select("id", pl.col("prediction").alias("label")),
        test.select("id"),
        sample,
        output,
        {**expected, "label_order": list(LABELS)},
    )
    logger.info("Validated test export %s; no submission performed", output)


def main() -> None:
    """Execute one explicit XLM-R preflight, training, holdout, refit or inference mode."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    parser = argparse.ArgumentParser(description="XLM-R initial controlled experiment")
    for name in ["code-root", "split-root", "input-root", "output-root", "cache-root"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--experiment", default="xlmr_head_only_40k")
    parser.add_argument(
        "--mode",
        choices=["preflight", "train", "holdout", "refit", "inference"],
        default="preflight",
    )
    parser.add_argument("--scope", choices=["sampled", "all_train"], default="sampled")
    args = parser.parse_args()
    data, selected, tokenizer, config, identity, run, inputs, scope = _prepare(args)
    if args.mode in {"train", "refit"}:
        training = data.sort("id") if scope == "all_train" else _role(data, selected, "train")
        evaluation = None if scope == "all_train" else _role(data, selected, "development_eval")
        train_tokens = _tokens(training, tokenizer, config, run, "train")
        eval_tokens = (
            _tokens(evaluation, tokenizer, config, run, "development_eval")
            if evaluation is not None
            else None
        )
        train_encoder(
            train_tokens,
            eval_tokens,
            tokenizer,
            config,
            run,
            identity,
            scope == "all_train",
            args.cache_root,
        )
    elif args.mode == "holdout":
        _holdout(data, selected, tokenizer, config, identity, run)
    elif args.mode == "inference":
        _inference(inputs, tokenizer, config, identity, run)
    logger.info("XLM-R mode %s complete; artifacts: %s", args.mode, run)


if __name__ == "__main__":
    main()
