import argparse
import json
import logging
import sys
import tomllib
from dataclasses import asdict
from pathlib import Path

import joblib
import polars as pl


def main() -> None:
    """Restore the selected baseline and export predictions after a bounded final refit.

    Verify input and selection receipts, restore the completed model or explicitly
    fit it on all labeled rows, then validate and publish the sample-aligned CSV.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    logger = logging.getLogger("bounded_final_submission")
    parser = argparse.ArgumentParser()
    for name in ["code-root", "run-path", "input-root"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--inference-only", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.code_root / "src"))

    from hm3.baseline import BaselineCandidate
    from hm3.baseline_export import fit_bounded_final_pipeline
    from hm3.evaluation import export_submission
    from hm3.kaggle_bootstrap import find_competition_inputs, verify_bundle
    from hm3.validation import fingerprint_file, load_training_data

    verify_bundle(args.code_root, "code")
    config = tomllib.loads((args.code_root / "config/baseline_kaggle.toml").read_text())
    inputs = find_competition_inputs(args.input_root, config["source_sha256"])
    metadata = json.loads((args.run_path / "run_metadata.json").read_text())
    selection = json.loads((args.run_path / "selection.json").read_text())
    receipt = json.loads((args.run_path / "holdout_record.json").read_text())
    if (
        selection["recipe_sha256"] != metadata["recipe_sha256"]
        or receipt["recipe_sha256"] != selection["recipe_sha256"]
        or selection["candidate"] != receipt["candidate"]
    ):
        raise ValueError("Restore the verified CV-selected baseline configuration")
    for name, expected in receipt["files"].items():
        if fingerprint_file(args.run_path / name) != expected:
            raise ValueError("Changed original holdout receipt")
    candidate = BaselineCandidate(**selection["candidate"])
    features = metadata["recipe"]["features"]
    if features != {"word_max_features": 40000, "char_max_features": 60000, "min_df": 3}:
        raise ValueError("Unexpected scientific feature recipe")
    output_dir = args.run_path / "submissions"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{candidate.name}_all_train.csv"
    model_path = output_dir / "bounded_refit/final_pipeline.joblib"
    expected_export = {
        "source_sha256": config["source_sha256"],
        "test_sha256": fingerprint_file(inputs["test"]),
        "sample_sha256": fingerprint_file(inputs["sample"]),
        "recipe_sha256": metadata["recipe_sha256"],
        "candidate": asdict(candidate),
        "seed": config["seed"],
        "feature_options": features,
        "feature_implementation_sha256": fingerprint_file(
            args.code_root / "src/hm3/baseline_export.py"
        ),
        "export_implementation_sha256": fingerprint_file(Path(__file__)),
    }
    if output.exists() and output.with_suffix(".json").exists():
        record = json.loads(output.with_suffix(".json").read_text())
        if (
            any(record.get(key) != value for key, value in expected_export.items())
            or record["model_sha256"] != fingerprint_file(model_path)
            or record["csv_sha256"] != fingerprint_file(output)
        ):
            raise ValueError("Completed export has incompatible inputs, model or exporter")
        logger.info("Restored completed submission %s", output)
        return
    if model_path.exists():
        checkpoint = json.loads(model_path.with_suffix(".joblib.json").read_text())
        identity = checkpoint["identity"]
        expected_model = {
            "source_sha256": config["source_sha256"],
            "candidate": asdict(candidate),
            **features,
            "rows": 272352,
            "implementation_sha256": expected_export["feature_implementation_sha256"],
        }
        if any(identity.get(key) != value for key, value in expected_model.items()) or checkpoint[
            "sha256"
        ] != fingerprint_file(model_path):
            raise ValueError("Final classifier checkpoint is incompatible or corrupted")
        pipeline = joblib.load(model_path)
        training_rows = identity["rows"]
        logger.info("Restored verified all-training-data classifier; no fitting required")
    elif args.inference_only:
        raise ValueError("Inference-only requires the completed bounded final model")
    else:
        data = load_training_data(inputs["train"])
        texts, labels = data["text"].to_list(), data["label"].to_list()
        training_rows = data.height
        del data
        logger.info("Fitting the selected model on all %s labeled rows", training_rows)
        pipeline = fit_bounded_final_pipeline(
            texts,
            labels,
            candidate,
            output_dir / "bounded_refit",
            config["source_sha256"],
            **features,
        )
        del texts, labels
    test = pl.read_csv(inputs["test"], schema_overrides={"id": pl.String})
    sample = pl.read_csv(inputs["sample"], schema_overrides={"id": pl.String})
    labels = []
    for start in range(0, test.height, 512):
        labels.extend(pipeline.predict(test["text"][start : start + 512].to_list()).tolist())
        logger.info("Test inference %s/%s", min(start + 512, test.height), test.height)
    export_submission(
        test.select("id").with_columns(pl.Series("label", labels)),
        test.select("id"),
        sample,
        output,
        {
            **expected_export,
            "model_sha256": fingerprint_file(model_path),
            "model_class_order": pipeline.classes_.tolist(),
            "training_rows": training_rows,
            "vectorizer_fit_method": "streamed frequencies; chunked sparse transform",
        },
    )
    logger.info("Validated complete submission %s; no Kaggle submission sent", output)


if __name__ == "__main__":
    main()
