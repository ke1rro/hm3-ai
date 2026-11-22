import argparse
import hashlib
import json
import logging
import platform
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import plotly.express as px
import polars as pl

from hm3.baseline import (
    BaselineCandidate,
    fit_holdout,
    fit_or_restore_final_pipeline,
    run_cross_validation,
    summarize_cv,
)
from hm3.evaluation import export_submission, subgroup_metrics
from hm3.kaggle_bootstrap import find_competition_inputs, verify_bundle
from hm3.validation import LABELS, fingerprint_file, get_or_create_manifest, load_training_data

logger = logging.getLogger(__name__)


def _write_json(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(path)


def _candidates(config: dict) -> list[BaselineCandidate]:
    return [
        BaselineCandidate(
            f"{representation}_tfidf_logistic_regression_c{c:g}",
            representation,
            "logistic_regression",
            c,
        )
        for representation in ["word", "char", "word_char"]
        for c in config["logistic_regression_c"]
    ] + [
        BaselineCandidate(f"word_char_tfidf_linear_svm_c{c:g}", "word_char", "linear_svm", c)
        for c in config["linear_svm_c"]
    ]


def _hardware() -> dict[str, object]:
    gpu = shutil.which("nvidia-smi")
    gpu_description = "No nvidia-smi; CPU execution"
    if gpu:
        result = subprocess.run(
            [gpu, "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
        gpu_description = (
            result.stdout.strip() if result.returncode == 0 else "No usable GPU reported"
        )
    memory = Path("/proc/meminfo")
    return {
        "platform": platform.platform(),
        "gpu": gpu_description,
        "backend": "scikit-learn CPU sparse matrices",
        "memory": memory.read_text() if memory.exists() else "Unavailable on this platform",
    }


def _restore_saved_run(input_root: Path, destination: Path, identity: str) -> None:
    if destination.exists():
        return
    matches = [
        path.parent
        for path in input_root.rglob("run_metadata.json")
        if json.loads(path.read_text()).get("recipe_sha256") == identity
    ]
    if len(matches) > 1:
        raise ValueError("Attach only one saved-output version of this run for restoration")
    if matches:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".restore-", dir=destination.parent) as staging:
            staged = Path(staging) / destination.name
            shutil.copytree(matches[0], staged)
            staged.replace(destination)
        logger.info("Restored previous saved notebook outputs from %s", matches[0])


def _prepare(args: argparse.Namespace) -> tuple:
    verify_bundle(args.code_root, "code")
    verify_bundle(args.split_root, "split")
    config = tomllib.loads((args.code_root / "config/baseline_kaggle.toml").read_text())
    inputs = find_competition_inputs(args.input_root, config["source_sha256"])
    source_manifest = args.split_root / "data/metadata/validation_manifest.parquet"
    if fingerprint_file(source_manifest) != config["manifest_sha256"]:
        raise ValueError("Uploaded split differs from the frozen local manifest")
    args.output_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.output_root).free / 1024**3
    if free < config["minimum_free_disk_gib"]:
        raise ValueError(f"Only {free:.1f} GiB free; free space before launching the baseline")
    logger.info(
        "Verified competition paths: %s", {name: str(path) for name, path in inputs.items()}
    )
    data = load_training_data(inputs["train"])
    manifest, metadata = get_or_create_manifest(
        data, source_manifest, config["source_sha256"], config["seed"], config["cv_sample_target"]
    )
    saved_split = args.output_root / "data/metadata"
    saved_split.mkdir(parents=True, exist_ok=True)
    for source in [source_manifest, source_manifest.with_suffix(".json")]:
        target = saved_split / source.name
        if target.exists() and fingerprint_file(target) != fingerprint_file(source):
            raise ValueError("Working output contains a different frozen split")
        if not target.exists():
            shutil.copy2(source, target)
    features = {key: config[key] for key in ["word_max_features", "char_max_features", "min_df"]}
    candidates = _candidates(config)
    environment = {
        "python": platform.python_version(),
        **{name: version(name) for name in ["scikit-learn", "numpy", "scipy", "polars", "joblib"]},
    }
    recipe = {
        "seed": config["seed"],
        "source_sha256": config["source_sha256"],
        "manifest_sha256": config["manifest_sha256"],
        "features": features,
        "candidates": [asdict(candidate) for candidate in candidates],
        "environment": environment,
        "source_modules": {
            name: fingerprint_file(args.code_root / f"src/hm3/{name}.py")
            for name in ["baseline", "evaluation", "validation", "kaggle_run"]
        },
    }
    identity = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    run = args.output_root / "runs/classical_baseline" / identity
    _restore_saved_run(args.input_root, run, identity)
    run.mkdir(parents=True, exist_ok=True)
    record_path = run / "run_metadata.json"
    if record_path.exists() and json.loads(record_path.read_text())["recipe"] != recipe:
        raise ValueError("Existing run metadata is incompatible")
    if not record_path.exists():
        _write_json(
            record_path,
            {
                "recipe_sha256": identity,
                "recipe": recipe,
                "hardware": _hardware(),
                "free_disk_gib": free,
            },
        )
    _write_json(
        run / "split_audit.json", json.loads(source_manifest.with_suffix(".json").read_text())
    )
    split_audit = json.loads(source_manifest.with_suffix(".json").read_text())["audit"]
    _write_json(
        args.output_root / "current_run.json",
        {
            "recipe_sha256": identity,
            "run_directory": str(run.relative_to(args.output_root)),
        },
    )
    logger.info(
        "Run %s; source rows %s; partitions %s; CV sample folds %s",
        identity,
        data.height,
        split_audit["partition_counts"],
        split_audit["cv_sample_fold_counts"],
    )
    logger.info(
        "Candidate recipes: %s; feature settings: %s", [asdict(c) for c in candidates], features
    )
    return data, manifest, candidates, features, inputs, config, identity, run


def _verify_outputs(run: Path, record: dict) -> None:
    for name, expected in record["files"].items():
        if fingerprint_file(run / name) != expected:
            raise ValueError(f"Completed result checksum mismatch: {name}")


def _cv_holdout(
    data: pl.DataFrame,
    manifest: pl.DataFrame,
    candidates: list[BaselineCandidate],
    features: dict,
    identity: str,
    run: Path,
) -> None:
    scores, oof = run_cross_validation(
        data,
        manifest,
        candidates,
        **features,
        checkpoint_path=run / "cv_folds",
        run_identity=identity,
    )
    comparison = summarize_cv(scores, oof)
    scores.write_parquet(run / "cv_scores.parquet")
    oof.write_parquet(run / "cv_predictions.parquet")
    comparison.write_csv(run / "cv_comparison.csv")
    selected = next(c for c in candidates if c.name == comparison["candidate"][0])
    _write_json(
        run / "selection.json",
        {
            "recipe_sha256": identity,
            "candidate": asdict(selected),
            "rule": "highest mean fold macro-F1; lexical name tie-break",
        },
    )
    completion = run / "holdout_record.json"
    if completion.exists():
        record = json.loads(completion.read_text())
        if record["recipe_sha256"] != identity or record["candidate"] != asdict(selected):
            raise ValueError("Saved holdout evaluation is incompatible")
        _verify_outputs(run, record)
        predictions = pl.read_parquet(run / "holdout_predictions.parquet")
        metrics = json.loads((run / "holdout_metrics.json").read_text())
        logger.info("Restored completed holdout evaluation")
    else:
        logger.info("Fitting CV-selected %s on all development records", selected.name)
        _, predictions, metrics = fit_holdout(data, manifest, selected, **features)
        predictions.write_parquet(run / "holdout_predictions.parquet")
        _write_json(run / "holdout_metrics.json", metrics)
        _write_json(
            completion,
            {
                "recipe_sha256": identity,
                "candidate": asdict(selected),
                "files": {
                    name: fingerprint_file(run / name)
                    for name in ["holdout_predictions.parquet", "holdout_metrics.json"]
                },
            },
        )
    comparison = comparison.with_columns(
        pl.when(pl.col("candidate") == selected.name)
        .then(pl.lit(metrics["macro_f1"]))
        .otherwise(None)
        .alias("holdout_macro_f1")
    )
    comparison.write_csv(run / "model_comparison.csv")
    for columns in [["lang"], ["domain"], ["length_bucket"], ["lang", "domain"]]:
        subgroup_metrics(predictions, columns).write_json(
            run / f"subgroup_{'_'.join(columns)}.json"
        )
    errors = predictions.filter(pl.col("label") != pl.col("prediction"))
    errors.group_by(["label", "prediction"]).len().write_json(run / "error_counts.json")
    if errors.height:
        examples = (
            errors.group_by(["label", "prediction"], maintain_order=True)
            .map_groups(lambda group: group.sample(n=min(2, group.height), seed=2026))
            .join(data.select("id", pl.col("text").str.slice(0, 180).alias("excerpt")), on="id")
        )
        examples.write_parquet(run / "error_examples.parquet")
    charts = run / "charts"
    charts.mkdir(exist_ok=True)
    for key, title in [
        ("confusion_matrix", "Fixed-validation confusion counts"),
        ("row_normalized_confusion_matrix", "Fixed-validation confusion proportions"),
    ]:
        px.imshow(
            metrics[key],
            x=list(LABELS),
            y=list(LABELS),
            text_auto=True,
            title=title,
            labels={"x": "Predicted class", "y": "True class", "color": "Value"},
        ).write_html(charts / f"{key}.html", include_plotlyjs="directory")
    logger.info(
        "Completed holdout: macro-F1 %.6f; weighted-F1 %.6f; accuracy %.6f",
        metrics["macro_f1"],
        metrics["weighted_f1"],
        metrics["accuracy"],
    )


def _submission(
    data: pl.DataFrame,
    features: dict,
    inputs: dict[str, Path],
    config: dict,
    identity: str,
    run: Path,
    is_inference_only: bool,
) -> None:
    if not (run / "holdout_record.json").exists():
        raise ValueError("Complete the CV-selected holdout evaluation before final export")
    holdout_record = json.loads((run / "holdout_record.json").read_text())
    _verify_outputs(run, holdout_record)
    selection = json.loads((run / "selection.json").read_text())
    if (
        selection["recipe_sha256"] != identity
        or selection["candidate"] != holdout_record["candidate"]
    ):
        raise ValueError("Saved model selection is incompatible")
    selected = BaselineCandidate(**selection["candidate"])
    model_path = run / "submissions" / f"{selected.name}_all_train.joblib"
    output = model_path.with_suffix(".csv")
    input_identity = {
        "source_sha256": config["source_sha256"],
        "test_sha256": fingerprint_file(inputs["test"]),
        "sample_sha256": fingerprint_file(inputs["sample"]),
        "recipe_sha256": identity,
    }
    if output.exists() and output.with_suffix(".json").exists():
        record = json.loads(output.with_suffix(".json").read_text())
        if (
            any(record[key] != value for key, value in input_identity.items())
            or record["model_sha256"] != fingerprint_file(model_path)
            or record["csv_sha256"] != fingerprint_file(output)
        ):
            raise ValueError("Completed export is incompatible with its inputs or checkpoint")
        logger.info("Restored verified submission %s", output)
        return
    if is_inference_only and not model_path.exists():
        raise ValueError("Inference-only requires a completed final model checkpoint")
    model, metadata = fit_or_restore_final_pipeline(
        data, selected, model_path, identity, config["source_sha256"], **features
    )
    test = pl.read_csv(inputs["test"], schema_overrides={"id": pl.String})
    sample = pl.read_csv(inputs["sample"], schema_overrides={"id": pl.String})
    labels = []
    for start in range(0, test.height, 512):
        labels.extend(model.predict(test["text"][start : start + 512].to_list()).tolist())
        logger.info("Test inference %s/%s", min(start + 512, test.height), test.height)
    predictions = test.select("id").with_columns(pl.Series("label", labels))
    export_submission(
        predictions,
        test.select("id"),
        sample,
        output,
        {
            **input_identity,
            "model_sha256": metadata["model_sha256"],
            "candidate": asdict(selected),
            "seed": config["seed"],
            "model_class_order": model.classes_.tolist(),
            "training_rows": data.height,
        },
    )
    logger.info("Validated final submission: %s", output)


def main() -> None:
    """Run an explicit Kaggle audit, classical CV/holdout or final export stage."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    parser = argparse.ArgumentParser(description="Frozen-split classical baseline for Kaggle")
    for name in ["code-root", "split-root", "input-root", "output-root"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=["audit", "cv-holdout", "submission", "inference"], default="audit"
    )
    args = parser.parse_args()
    data, manifest, candidates, features, inputs, config, identity, run = _prepare(args)
    if args.stage == "cv-holdout":
        _cv_holdout(data, manifest, candidates, features, identity, run)
    elif args.stage in {"submission", "inference"}:
        _submission(data, features, inputs, config, identity, run, args.stage == "inference")
    logger.info("Stage %s completed; outputs: %s", args.stage, run)


if __name__ == "__main__":
    main()
