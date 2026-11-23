import argparse
import hashlib
import json
import logging
import sys
import tomllib
from dataclasses import asdict
from pathlib import Path
from resource import RUSAGE_SELF, getrusage
from time import perf_counter

import joblib
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from scipy import sparse
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hm3.baseline import BaselineCandidate, build_pipeline  # noqa: E402
from hm3.baseline_export import fit_streaming_vectorizer  # noqa: E402
from hm3.decision import (  # noqa: E402
    PROBABILITY_COLUMNS,
    blend_probabilities,
    predict_with_fusion_offset,
    select_fusion_offset,
)
from hm3.evaluation import classification_metrics, export_submission  # noqa: E402
from hm3.families import (  # noqa: E402
    assign_families,
    assign_family_folds,
    find_strong_links,
    label_rewritten_sentences,
    split_sentences,
)
from hm3.sentence_scores import (  # noqa: E402
    AGGREGATE_COLUMNS,
    aggregate_document_scores,
    cross_fit_sentence_scores,
    fit_sentence_model,
    score_sentences,
)
from hm3.validation import LABELS, SEED, fingerprint_file, load_training_data  # noqa: E402

logger = logging.getLogger("rewrite_experiments")
CANDIDATE = BaselineCandidate(
    "word_char_tfidf_logistic_regression_c1", "word_char", "logistic_regression", 1.0
)
AGGREGATE_SCALING = (
    "log1p(count_over_threshold); min-max fitted on training rows with clipping; block divided "
    "by sqrt(7) so its squared row norm is at most 1, keeping saga step sizes near the reference"
)
STAGES = ["families", "reference", "sentence-gate", "sentence-holdout", "offset", "blend", "final"]
TRANSFORMER_STAGES = [
    "transformer-prepare",
    "char-all-train",
    "transformer-holdout",
    "transformer-final",
]


def _peak_rss_gib() -> float:
    return getrusage(RUSAGE_SELF).ru_maxrss / (1024**3 if sys.platform == "darwin" else 1024**2)


def _write_json(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    frame.write_parquet(temporary)
    temporary.replace(path)


def _write_matrix(matrix: sparse.csr_matrix, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    sparse.save_npz(temporary, matrix, compressed=False)
    temporary.replace(path)


def _sentence_settings(config: dict) -> dict:
    model = config["sentence_model"]
    return {
        "ngram_range": tuple(model["ngram_range"]),
        "min_df": model["min_df"],
        "max_features": model["max_features"],
        "c": model["c"],
    }


def _link_settings(config: dict) -> dict:
    alignment = config["alignment"]
    return {
        "maximum_documents": alignment["maximum_sentence_documents"],
        "minimum_shared": alignment["minimum_shared_sentences"],
        "minimum_share": alignment["minimum_shared_share"],
    }


def _identity(config: dict, stage: str) -> str:
    modules = ["families", "sentence_scores", "decision", "baseline", "baseline_export"]
    record = {
        "stage": stage,
        "config": config,
        "modules": {name: fingerprint_file(ROOT / f"src/hm3/{name}.py") for name in modules},
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def _is_complete(directory: Path, identity: str) -> bool:
    record = directory / "record.json"
    if not record.exists():
        return False
    saved = json.loads(record.read_text())
    if saved["identity"] != identity:
        raise ValueError(f"{directory} holds a different recipe; move it before rerunning")
    for name, expected in saved["files"].items():
        if fingerprint_file(directory / name) != expected:
            raise ValueError(f"Completed output changed: {directory / name}")
    return True


def _complete(directory: Path, identity: str, files: list[str], summary: dict) -> None:
    _write_json(
        directory / "record.json",
        {
            "identity": identity,
            "files": {name: fingerprint_file(directory / name) for name in files},
            "peak_rss_gib": _peak_rss_gib(),
            **summary,
        },
    )


def _load_inputs(config: dict) -> tuple[pl.DataFrame, pl.DataFrame]:
    train_path = ROOT / config["inputs"]["train"]
    manifest_path = ROOT / config["inputs"]["manifest"]
    if fingerprint_file(train_path) != config["source_sha256"]:
        raise ValueError("train.csv differs from the configured source fingerprint")
    if fingerprint_file(manifest_path) != config["manifest_sha256"]:
        raise ValueError("Restore the frozen validation manifest before running experiments")
    data = load_training_data(train_path)
    manifest = pl.read_parquet(manifest_path).select("id", "label", "lang", "domain", "partition")
    return data, manifest


def _aggregate_values(frame: pl.DataFrame) -> np.ndarray:
    return (
        frame.select(AGGREGATE_COLUMNS)
        .with_columns(pl.col("count_over_threshold").log1p())
        .to_numpy()
    )


def _aggregate_block(scaler: MinMaxScaler, frame: pl.DataFrame) -> sparse.csr_matrix:
    scaled = scaler.transform(_aggregate_values(frame)) / np.sqrt(len(AGGREGATE_COLUMNS))
    return sparse.csr_matrix(scaled.astype(np.float32))


def _transform_batch(features: FeatureUnion, texts: list[str]) -> sparse.csr_matrix:
    return features.transform(texts).tocsr()


def _transform_in_batches(
    features: FeatureUnion, texts: list[str], n_jobs: int, batch_size: int = 4096
) -> sparse.csr_matrix:
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_transform_batch)(features, texts[start : start + batch_size])
        for start in range(0, len(texts), batch_size)
    )
    return sparse.vstack(parts, format="csr")


def _ordered_probabilities(classes: list[str], probabilities: np.ndarray) -> np.ndarray:
    return probabilities[:, [classes.index(label) for label in LABELS]]


def _probability_table(
    frame: pl.DataFrame, probabilities: np.ndarray, columns: list[str]
) -> pl.DataFrame:
    return frame.select(columns).with_columns(
        *[
            pl.Series(name, probabilities[:, index])
            for index, name in enumerate(PROBABILITY_COLUMNS)
        ],
        pl.Series("prediction", np.asarray(LABELS)[probabilities.argmax(axis=1)]),
    )


def _macro_f1(truth: np.ndarray, predicted: np.ndarray) -> float:
    return float(f1_score(truth, predicted, labels=list(LABELS), average="macro", zero_division=0))


def _cluster_bootstrap(
    truth: np.ndarray,
    reference: np.ndarray,
    candidate: np.ndarray,
    clusters: np.ndarray,
    repetitions: int = 1000,
) -> dict[str, float | int | str]:
    codes = {label: index for index, label in enumerate(LABELS)}
    true_codes = np.vectorize(codes.get)(truth)
    cells = [true_codes * 3 + np.vectorize(codes.get)(values) for values in (reference, candidate)]
    _, cluster_index = np.unique(clusters, return_inverse=True)
    generator = np.random.default_rng(SEED)
    differences = []
    for _ in range(repetitions):
        weights = np.bincount(
            generator.integers(0, cluster_index.max() + 1, cluster_index.max() + 1),
            minlength=cluster_index.max() + 1,
        )[cluster_index]
        scores = []
        for cell in cells:
            matrix = np.bincount(cell, weights=weights, minlength=9).reshape(3, 3)
            precision = np.diag(matrix) / np.maximum(matrix.sum(axis=0), 1e-12)
            recall = np.diag(matrix) / np.maximum(matrix.sum(axis=1), 1e-12)
            scores.append(np.mean(2 * precision * recall / np.maximum(precision + recall, 1e-12)))
        differences.append(scores[1] - scores[0])
    values = np.asarray(differences)
    return {
        "difference": _macro_f1(truth, candidate) - _macro_f1(truth, reference),
        "bootstrap_se": float(values.std(ddof=1)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "repetitions": repetitions,
        "resampling_unit": "all-training document family",
    }


def _evaluation(
    holdout: pl.DataFrame, predicted: np.ndarray, reference: np.ndarray | None = None
) -> dict:
    truth = holdout["label"].to_numpy()
    clean = holdout["is_family_clean"].to_numpy()
    result = {
        "holdout": classification_metrics(truth.tolist(), predicted.tolist()),
        "family_clean": classification_metrics(truth[clean].tolist(), predicted[clean].tolist()),
        "family_clean_rows": int(clean.sum()),
    }
    if reference is not None:
        result["paired_holdout"] = _cluster_bootstrap(
            truth, reference, predicted, holdout["train_family"].to_numpy()
        )
        result["paired_family_clean"] = {
            "difference": _macro_f1(truth[clean], predicted[clean])
            - _macro_f1(truth[clean], reference[clean])
        }
    return result


def _holdout_frame(config: dict, assignments: pl.DataFrame) -> pl.DataFrame:
    saved = pl.read_parquet(ROOT / config["inputs"]["baseline_run"] / "holdout_predictions.parquet")
    return (
        saved.select("id", "label", "lang", "domain", "words", "length_bucket")
        .join(
            assignments.select(
                "id", "is_family_clean", "train_family", "retained_share", "rewritten_sentences"
            ),
            on="id",
            how="left",
            validate="1:1",
        )
        .sort("id")
    )


def run_families(config: dict, directory: Path) -> None:
    """Measure sentence-overlap families, family folds, purge sets and rewrite extent."""
    data, manifest = _load_inputs(config)
    links_settings = _link_settings(config)
    sentences = split_sentences(
        data.select("id", "text"), config["alignment"]["minimum_sentence_words"]
    ).drop("sentence")
    del data
    development = manifest.filter(pl.col("partition") == "development")
    development_sentences = sentences.join(development.select("id"), on="id")
    development_links = find_strong_links(development_sentences, **links_settings)
    development_families = assign_families(development["id"], development_links)
    folds = assign_family_folds(
        development, development_families, config["alignment"]["family_folds"], config["seed"]
    )
    all_links = find_strong_links(sentences, **links_settings)
    train_families = assign_families(manifest["id"], all_links).rename({"family": "train_family"})
    partitions = manifest.select("id", "partition")
    crossing = (
        all_links.join(partitions, on="id")
        .join(partitions.rename({"id": "id_other", "partition": "partition_other"}), on="id_other")
        .filter(pl.col("partition") != pl.col("partition_other"))
    )
    linked_ids = pl.concat([crossing.select("id"), crossing.select(id="id_other")]).unique()
    linked = linked_ids.join(partitions, on="id")
    purged_families = folds.join(
        linked.filter(pl.col("partition") == "development").select("id"), on="id"
    )["family"].unique()
    labelled = label_rewritten_sentences(
        sentences.with_columns(pl.lit("").alias("sentence")), manifest, all_links
    )
    extent = (
        labelled.join(manifest.filter(pl.col("label") == "fusion").select("id"), on="id")
        .group_by("id")
        .agg(
            pl.col("is_rewritten").sum().alias("rewritten_sentences"),
            (1 - pl.col("is_rewritten").mean()).alias("retained_share"),
        )
    )
    assignments = (
        manifest.join(folds, on="id", how="left")
        .join(train_families, on="id", validate="1:1")
        .join(extent, on="id", how="left")
        .with_columns(
            pl.col("family").is_in(purged_families.implode()).fill_null(False).alias("is_purged"),
            (
                (pl.col("partition") == "validation")
                & pl.col("id").is_in(linked["id"].implode()).not_()
            ).alias("is_family_clean"),
        )
        .sort("id")
    )
    _write_parquet(assignments, directory / "assignments.parquet")
    _write_parquet(development_links, directory / "development_links.parquet")
    _write_parquet(all_links, directory / "all_links.parquet")
    summary = {
        "development_links": development_links.height,
        "all_links": all_links.height,
        "development_families": int(development_families["family"].n_unique()),
        "purged_development_documents": int(assignments["is_purged"].sum()),
        "family_clean_holdout_documents": int(assignments["is_family_clean"].sum()),
        "family_fold_counts": folds.group_by("family_fold").len().sort("family_fold").to_dicts(),
    }
    logger.info("Families: %s", summary)
    _write_json(directory / "summary.json", summary)


def run_reference(config: dict, directory: Path) -> None:
    """Refit the frozen baseline on development rows and save holdout probabilities."""
    data, manifest = _load_inputs(config)
    assigned = data.join(manifest.select("id", "partition"), on="id", how="inner", validate="1:1")
    training = assigned.filter(pl.col("partition") == "development")
    validation = assigned.filter(pl.col("partition") == "validation")
    del data, assigned
    pipeline = build_pipeline(CANDIDATE, **config["baseline"])
    features = pipeline.named_steps["features"]
    start = perf_counter()
    for _, vectorizer in features.transformer_list:
        fit_streaming_vectorizer(vectorizer, training["text"].to_list())
    fit_vocabulary_seconds = perf_counter() - start
    start = perf_counter()
    training_matrix = _transform_in_batches(features, training["text"].to_list(), config["n_jobs"])
    validation_matrix = _transform_in_batches(
        features, validation["text"].to_list(), config["n_jobs"]
    )
    transform_seconds = perf_counter() - start
    logger.info(
        "Reference features: train %s nnz %s; holdout %s; peak RSS %.2f GiB",
        training_matrix.shape,
        training_matrix.nnz,
        validation_matrix.shape,
        _peak_rss_gib(),
    )
    _write_matrix(training_matrix, directory / "development_features.npz")
    _write_matrix(validation_matrix, directory / "holdout_features.npz")
    _write_parquet(training.select("id", "label"), directory / "development_rows.parquet")
    _write_parquet(validation.select("id", "label"), directory / "holdout_rows.parquet")
    classifier = pipeline.named_steps["classifier"]
    start = perf_counter()
    classifier.fit(training_matrix, training["label"].to_list())
    classifier_seconds = perf_counter() - start
    probabilities = _ordered_probabilities(
        classifier.classes_.tolist(), classifier.predict_proba(validation_matrix)
    )
    table = _probability_table(validation, probabilities, ["id", "label"])
    saved = pl.read_parquet(
        ROOT / config["inputs"]["baseline_run"] / "holdout_predictions.parquet"
    ).select("id", pl.col("prediction").alias("saved_prediction"))
    agreement = table.join(saved, on="id", validate="1:1").select(
        (pl.col("prediction") == pl.col("saved_prediction")).mean()
    )
    joblib.dump(classifier, directory / "classifier.joblib")
    _write_parquet(table, directory / "holdout_probabilities.parquet")
    metrics = classification_metrics(table["label"].to_list(), table["prediction"].to_list())
    summary = {
        "macro_f1": metrics["macro_f1"],
        "agreement_with_saved_kaggle_predictions": agreement.item(),
        "classifier_iterations": int(np.max(classifier.n_iter_)),
        "fit_vocabulary_seconds": fit_vocabulary_seconds,
        "transform_seconds": transform_seconds,
        "classifier_seconds": classifier_seconds,
        "development_matrix_nnz": int(training_matrix.nnz),
    }
    logger.info("Reference refit: %s", summary)
    _write_json(directory / "metrics.json", {**metrics, **summary})


def run_sentence_gate(config: dict, directory: Path, families: Path) -> None:
    """Cross-fit the sentence detector inside development, score holdout and test the gate."""
    data, manifest = _load_inputs(config)
    assignments = pl.read_parquet(families / "assignments.parquet")
    development_links = pl.read_parquet(families / "development_links.parquet")
    settings = _sentence_settings(config)
    sentences = split_sentences(
        data.select("id", "text"), config["alignment"]["minimum_sentence_words"]
    )
    del data
    development = assignments.filter(pl.col("partition") == "development")
    holdout = assignments.filter(pl.col("partition") == "validation")
    development_sentences = sentences.join(development.select("id"), on="id")
    holdout_sentences = sentences.join(holdout.select("id"), on="id")
    del sentences
    training = label_rewritten_sentences(
        development_sentences, development.select("id", "label"), development_links
    ).join(development.filter(~pl.col("is_purged")).select("id"), on="id")
    logger.info(
        "Sentence training rows %s, rewritten share %.4f, documents %s",
        training.height,
        training["is_rewritten"].mean(),
        training["id"].n_unique(),
    )
    out_of_fold, fold_records = cross_fit_sentence_scores(
        development_sentences.select("id", "position", "sentence"),
        training,
        development.select("id", "family_fold"),
        settings,
        config["n_jobs"],
    )
    start = perf_counter()
    model = fit_sentence_model(
        settings, training["sentence"].to_list(), training["is_rewritten"].to_numpy()
    )
    full_fit_seconds = perf_counter() - start
    holdout_scores = holdout_sentences.select("id", "position").with_columns(
        pl.Series(
            "score",
            score_sentences(model, holdout_sentences["sentence"].to_list(), config["n_jobs"]),
        )
    )
    threshold = config["sentence_model"]["score_threshold"]
    development_aggregates = aggregate_document_scores(out_of_fold, development["id"], threshold)
    holdout_aggregates = aggregate_document_scores(holdout_scores, holdout["id"], threshold)
    labelled_scores = training.join(out_of_fold, on=["id", "position"], validate="1:1")
    pair = development_aggregates.join(development.select("id", "label"), on="id").filter(
        pl.col("label").is_in(["human", "fusion"])
    )
    gate = {
        "sentence_oof_auc": float(
            roc_auc_score(labelled_scores["is_rewritten"], labelled_scores["score"])
        ),
        "document_oof_auc_max_human_vs_fusion": float(
            roc_auc_score(pair["label"] == "fusion", pair["max_score"])
        ),
        "training_sentences": training.height,
        "training_documents": int(training["id"].n_unique()),
        "full_fit_seconds": full_fit_seconds,
        "folds": fold_records.to_dicts(),
    }
    gate["passed"] = (
        gate["sentence_oof_auc"] >= config["gate"]["minimum_sentence_auc"]
        and gate["document_oof_auc_max_human_vs_fusion"] >= config["gate"]["minimum_document_auc"]
    )
    logger.info("Sentence gate: %s", {k: v for k, v in gate.items() if k != "folds"})
    joblib.dump(model, directory / "sentence_model.joblib")
    _write_parquet(development_aggregates, directory / "development_aggregates.parquet")
    _write_parquet(holdout_aggregates, directory / "holdout_aggregates.parquet")
    _write_json(directory / "gate.json", gate)


def run_sentence_holdout(
    config: dict, directory: Path, families: Path, reference: Path, gate: Path
) -> None:
    """Fit word/character TF-IDF plus sentence aggregates on development and evaluate once."""
    gate_record = json.loads((gate / "gate.json").read_text())
    if not gate_record["passed"]:
        raise ValueError("Sentence gate failed; the holdout comparison is not authorized")
    assignments = pl.read_parquet(families / "assignments.parquet")
    development_rows = pl.read_parquet(reference / "development_rows.parquet")
    holdout_rows = pl.read_parquet(reference / "holdout_rows.parquet")
    development_aggregates = development_rows.join(
        pl.read_parquet(gate / "development_aggregates.parquet"),
        on="id",
        how="left",
        maintain_order="left",
    )
    holdout_aggregates = holdout_rows.join(
        pl.read_parquet(gate / "holdout_aggregates.parquet"),
        on="id",
        how="left",
        maintain_order="left",
    )
    scaler = MinMaxScaler(clip=True).fit(_aggregate_values(development_aggregates))
    matrices = []
    for name, frame in [
        ("development_features.npz", development_aggregates),
        ("holdout_features.npz", holdout_aggregates),
    ]:
        matrices.append(
            sparse.hstack(
                [sparse.load_npz(reference / name), _aggregate_block(scaler, frame)], format="csr"
            )
        )
        logger.info("Joint matrix %s ready; peak RSS %.2f GiB", matrices[-1].shape, _peak_rss_gib())
    classifier = build_pipeline(CANDIDATE, **config["baseline"]).named_steps["classifier"]
    start = perf_counter()
    classifier.fit(matrices[0], development_rows["label"].to_list())
    classifier_seconds = perf_counter() - start
    probabilities = _ordered_probabilities(
        classifier.classes_.tolist(), classifier.predict_proba(matrices[1])
    )
    del matrices
    table = _probability_table(holdout_rows, probabilities, ["id", "label"])
    reference_table = pl.read_parquet(reference / "holdout_probabilities.parquet")
    holdout = (
        _holdout_frame(config, assignments)
        .join(table.select("id", "prediction"), on="id", validate="1:1")
        .join(
            reference_table.select("id", pl.col("prediction").alias("reference")),
            on="id",
            validate="1:1",
        )
    )
    evaluation = _evaluation(
        holdout, holdout["prediction"].to_numpy(), holdout["reference"].to_numpy()
    )
    extent = (
        holdout.filter(pl.col("label") == "fusion")
        .with_columns(
            pl.when(pl.col("rewritten_sentences").is_null())
            .then(pl.lit("no linked human"))
            .otherwise(pl.col("rewritten_sentences").clip(upper_bound=4).cast(pl.String))
            .alias("rewritten_sentences_capped")
        )
        .group_by("rewritten_sentences_capped")
        .agg(
            pl.len().alias("fusion_documents"),
            (pl.col("reference") == "fusion").mean().alias("reference_recall"),
            (pl.col("prediction") == "fusion").mean().alias("candidate_recall"),
        )
        .sort("rewritten_sentences_capped")
    )
    evaluation_reference = _evaluation(holdout, holdout["reference"].to_numpy())
    human_f1 = {
        name: next(row["f1"] for row in metrics["holdout"]["per_class"] if row["label"] == "human")
        for name, metrics in [("reference", evaluation_reference), ("candidate", evaluation)]
    }
    paired = evaluation["paired_holdout"]
    success = {
        "minimum_gain": paired["difference"] >= 0.003,
        "positive_interval": paired["ci95_low"] > 0,
        "family_clean_not_worse": evaluation["paired_family_clean"]["difference"] >= 0,
        "human_f1_drop_within_limit": human_f1["candidate"] - human_f1["reference"] >= -0.003,
    }
    success["passed"] = all(success.values())
    _write_parquet(table, directory / "holdout_probabilities.parquet")
    _write_parquet(extent, directory / "fusion_recall_by_rewritten_sentences.parquet")
    _write_json(
        directory / "metrics.json",
        {
            "candidate": evaluation,
            "reference": evaluation_reference,
            "success": success,
            "classifier_seconds": classifier_seconds,
            "classifier_iterations": int(np.max(classifier.n_iter_)),
            "aggregate_columns": AGGREGATE_COLUMNS,
            "aggregate_scaling": AGGREGATE_SCALING,
            "aggregate_coefficients": {
                label: dict(
                    zip(
                        AGGREGATE_COLUMNS,
                        classifier.coef_[index, -len(AGGREGATE_COLUMNS) :].tolist(),
                    )
                )
                for index, label in enumerate(classifier.classes_.tolist())
            },
        },
    )
    logger.info(
        "Sentence holdout: candidate %.6f, reference %.6f, paired %s, success %s",
        evaluation["holdout"]["macro_f1"],
        evaluation_reference["holdout"]["macro_f1"],
        paired,
        success,
    )
    logger.info("Fusion recall by rewritten sentences:\n%s", extent)


def _fit_cv_fold(
    texts: list[str], labels: list[str], validation_texts: list[str], baseline: dict
) -> tuple[list[str], np.ndarray, list[str]]:
    pipeline = build_pipeline(CANDIDATE, **baseline).fit(texts, labels)
    return (
        pipeline.classes_.tolist(),
        pipeline.predict_proba(validation_texts),
        pipeline.predict(validation_texts).tolist(),
    )


def run_offset(config: dict, directory: Path, families: Path, reference: Path) -> None:
    """Tune one fusion log-probability offset on development OOF and evaluate it once."""
    data, manifest = _load_inputs(config)
    assignments = pl.read_parquet(families / "assignments.parquet")
    frozen = pl.read_parquet(ROOT / config["inputs"]["manifest"]).filter(pl.col("is_cv_sample"))
    sample = (
        data.join(frozen.select("id", "cv_fold"), on="id", validate="1:1")
        .join(assignments.select("id", "family"), on="id", validate="1:1")
        .sort("id")
    )
    del data
    grouped = assign_family_folds(
        sample.select("id", "label"),
        sample.select("id", "family"),
        3,
        config["seed"],
    )
    sample = sample.join(grouped.select("id", "family_fold"), on="id", validate="1:1")
    columns = {"frozen_cv_folds": "cv_fold", "family_grouped_folds": "family_fold"}
    tasks = [(scheme, fold) for scheme in columns for fold in range(3)]
    start = perf_counter()
    results = Parallel(n_jobs=min(3, config["n_jobs"]))(
        delayed(_fit_cv_fold)(
            sample.filter(pl.col(columns[scheme]) != fold)["text"].to_list(),
            sample.filter(pl.col(columns[scheme]) != fold)["label"].to_list(),
            sample.filter(pl.col(columns[scheme]) == fold)["text"].to_list(),
            config["baseline"],
        )
        for scheme, fold in tasks
    )
    fit_seconds = perf_counter() - start
    out_of_fold = []
    for (scheme, fold), (classes, probabilities, predicted) in zip(tasks, results, strict=True):
        rows = sample.filter(pl.col(columns[scheme]) == fold).select("id", "label")
        out_of_fold.append(
            _probability_table(
                rows, _ordered_probabilities(classes, probabilities), ["id", "label"]
            )
            .with_columns(pl.lit(scheme).alias("scheme"), pl.lit(fold).alias("fold"))
            .with_columns(pl.Series("pipeline_prediction", predicted))
        )
    out_of_fold = pl.concat(out_of_fold)
    saved = pl.read_parquet(ROOT / config["inputs"]["baseline_run"] / "cv_predictions.parquet")
    saved = saved.filter(pl.col("candidate") == CANDIDATE.name).select(
        "id", pl.col("prediction").alias("saved_prediction")
    )
    reproduction = (
        out_of_fold.filter(pl.col("scheme") == "frozen_cv_folds")
        .join(saved, on="id", validate="1:1")
        .select((pl.col("pipeline_prediction") == pl.col("saved_prediction")).mean())
        .item()
    )
    grid = np.round(
        np.arange(
            config["offset"]["grid_minimum"],
            config["offset"]["grid_maximum"] + config["offset"]["grid_step"] / 2,
            config["offset"]["grid_step"],
        ),
        6,
    )
    reference_table = pl.read_parquet(reference / "holdout_probabilities.parquet")
    holdout = _holdout_frame(config, assignments).join(
        reference_table.select("id", *PROBABILITY_COLUMNS, pl.col("prediction").alias("reference")),
        on="id",
        validate="1:1",
    )
    holdout_probabilities = holdout.select(PROBABILITY_COLUMNS).to_numpy()
    curves, schemes = [], {}
    for scheme in ["frozen_cv_folds", "family_grouped_folds"]:
        rows = out_of_fold.filter(pl.col("scheme") == scheme)
        offset, curve = select_fusion_offset(
            rows["label"].to_numpy(), rows.select(PROBABILITY_COLUMNS).to_numpy(), grid
        )
        curves.append(curve.with_columns(pl.lit(scheme).alias("scheme")))
        predicted = predict_with_fusion_offset(holdout_probabilities, offset)
        schemes[scheme] = {
            "selected_offset": offset,
            "oof_macro_f1_at_zero": float(curve.filter(pl.col("offset") == 0)["macro_f1"][0]),
            "oof_macro_f1_selected": float(curve.select(pl.col("macro_f1").max()).item()),
            **_evaluation(holdout, predicted, holdout["reference"].to_numpy()),
        }
    family_scheme = schemes["family_grouped_folds"]
    adoption = {
        "offset": family_scheme["selected_offset"],
        "holdout_not_worse": family_scheme["paired_holdout"]["difference"] >= 0,
        "family_clean_not_worse": family_scheme["paired_family_clean"]["difference"] >= 0,
        "nonzero_offset": family_scheme["selected_offset"] != 0,
    }
    adoption["adopted"] = all(
        adoption[key] for key in ["holdout_not_worse", "family_clean_not_worse", "nonzero_offset"]
    )
    _write_parquet(out_of_fold, directory / "oof_probabilities.parquet")
    _write_parquet(pl.concat(curves), directory / "oof_offset_curves.parquet")
    _write_json(
        directory / "metrics.json",
        {
            "schemes": schemes,
            "adoption": adoption,
            "frozen_fold_label_reproduction": reproduction,
            "fit_seconds": fit_seconds,
            "cv_sample_rows": sample.height,
        },
    )
    logger.info(
        "Offset: reproduction %.4f; %s; adoption %s",
        reproduction,
        {
            scheme: (
                values["selected_offset"],
                values["paired_holdout"]["difference"],
                values["paired_family_clean"]["difference"],
            )
            for scheme, values in schemes.items()
        },
        adoption,
    )


def run_blend(config: dict, directory: Path, families: Path, reference: Path) -> None:
    """Evaluate a predeclared equal-weight TF-IDF and XLM-R probability average once."""
    assignments = pl.read_parquet(families / "assignments.parquet")
    reference_table = pl.read_parquet(reference / "holdout_probabilities.parquet")
    xlmr = pl.read_parquet(ROOT / config["inputs"]["xlmr_holdout"])
    blended = blend_probabilities(reference_table, xlmr, config["blend"]["xlmr_weight"])
    holdout = (
        _holdout_frame(config, assignments)
        .join(blended.select("id", "prediction"), on="id", validate="1:1")
        .join(
            reference_table.select("id", pl.col("prediction").alias("reference")),
            on="id",
            validate="1:1",
        )
    )
    evaluation = _evaluation(
        holdout, holdout["prediction"].to_numpy(), holdout["reference"].to_numpy()
    )
    evaluation["adopted"] = (
        evaluation["paired_holdout"]["difference"] > 0
        and evaluation["paired_family_clean"]["difference"] >= 0
    )
    _write_parquet(blended, directory / "holdout_probabilities.parquet")
    _write_json(
        directory / "metrics.json", {"xlmr_weight": config["blend"]["xlmr_weight"], **evaluation}
    )
    logger.info(
        "Blend: macro-F1 %.6f; paired %s; adopted %s",
        evaluation["holdout"]["macro_f1"],
        evaluation["paired_holdout"],
        evaluation["adopted"],
    )


def _verified_final_artifact(path: Path, rows: int) -> Path:
    record = json.loads(path.with_suffix(path.suffix + ".json").read_text())
    if record["identity"]["rows"] != rows or record["sha256"] != fingerprint_file(path):
        raise ValueError(f"Final baseline artifact is incomplete or changed: {path}")
    return path


def _char_aggregates_all_train(
    config: dict, data: pl.DataFrame, manifest: pl.DataFrame, test: pl.DataFrame, families: Path
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, Pipeline]:
    settings = _sentence_settings(config)
    threshold = config["sentence_model"]["score_threshold"]
    all_links = pl.read_parquet(families / "all_links.parquet")
    family_table = pl.read_parquet(families / "assignments.parquet").select(
        "id", pl.col("train_family").alias("family")
    )
    folds = assign_family_folds(
        manifest.select("id", "label"),
        family_table,
        config["alignment"]["family_folds"],
        config["seed"],
    )
    sentences = split_sentences(
        data.select("id", "text"), config["alignment"]["minimum_sentence_words"]
    )
    training = label_rewritten_sentences(sentences, manifest.select("id", "label"), all_links)
    out_of_fold, fold_records = cross_fit_sentence_scores(
        sentences.select("id", "position", "sentence"),
        training,
        folds,
        settings,
        config["n_jobs"],
    )
    model = fit_sentence_model(
        settings, training["sentence"].to_list(), training["is_rewritten"].to_numpy()
    )
    test_sentences = split_sentences(
        test.select("id", "text"), config["alignment"]["minimum_sentence_words"]
    )
    test_scores = test_sentences.select("id", "position").with_columns(
        pl.Series(
            "score",
            score_sentences(model, test_sentences["sentence"].to_list(), config["n_jobs"]),
        )
    )
    return (
        aggregate_document_scores(out_of_fold, data["id"], threshold),
        aggregate_document_scores(test_scores, test["id"], threshold),
        fold_records,
        model,
    )


def run_final(config: dict, directory: Path, stages: dict[str, Path]) -> None:
    """Restore the all-training baseline, apply adopted decisions and export validated CSVs."""
    data, manifest = _load_inputs(config)
    test = pl.read_csv(ROOT / config["inputs"]["test"], schema_overrides={"id": pl.String})
    sample = pl.read_csv(ROOT / config["inputs"]["sample"], schema_overrides={"id": pl.String})
    final_root = ROOT / config["inputs"]["final_baseline"]
    pipeline = joblib.load(
        _verified_final_artifact(final_root / "final_pipeline.joblib", data.height)
    )
    classifier = pipeline.named_steps["classifier"]
    features = pipeline.named_steps["features"]
    start = perf_counter()
    test_matrix = _transform_in_batches(features, test["text"].to_list(), config["n_jobs"])
    baseline_probabilities = _ordered_probabilities(
        classifier.classes_.tolist(), classifier.predict_proba(test_matrix)
    )
    baseline_table = _probability_table(test, baseline_probabilities, ["id"])
    submitted = pl.read_csv(
        next((ROOT / config["inputs"]["final_baseline"]).parent.glob("*_all_train.csv")),
        schema_overrides={"id": pl.String},
    )
    reproduction = (
        baseline_table.join(submitted, on="id", validate="1:1")
        .select((pl.col("prediction") == pl.col("label")).mean())
        .item()
    )
    logger.info("Baseline test reproduction of submitted CSV: %.6f", reproduction)
    _write_parquet(baseline_table, directory / "baseline_test_probabilities.parquet")
    base_metadata = {
        "source_sha256": config["source_sha256"],
        "test_sha256": fingerprint_file(ROOT / config["inputs"]["test"]),
        "sample_sha256": fingerprint_file(ROOT / config["inputs"]["sample"]),
        "baseline_model_sha256": fingerprint_file(final_root / "final_pipeline.joblib"),
        "baseline_candidate": asdict(CANDIDATE),
        "class_order": list(LABELS),
        "seed": config["seed"],
        "training_rows": data.height,
    }
    exports = {"baseline_test_reproduction": reproduction}

    def export(name: str, labels: np.ndarray, metadata: dict) -> None:
        output = directory / "submissions" / f"{name}.csv"
        export_submission(
            test.select("id").with_columns(pl.Series("label", labels)),
            test.select("id"),
            sample,
            output,
            {**base_metadata, **metadata},
        )
        exports[name] = {
            "path": str(output.relative_to(ROOT)),
            "predicted_share": {label: float(np.mean(labels == label)) for label in LABELS},
        }
        logger.info("Exported validated submission %s", output)

    offset = json.loads((stages["offset"] / "metrics.json").read_text())["adoption"]
    if offset["adopted"]:
        export(
            "word_char_logreg_fusion_offset_all_train",
            predict_with_fusion_offset(baseline_probabilities, offset["offset"]),
            {"fusion_log_probability_offset": offset["offset"]},
        )
    blend = json.loads((stages["blend"] / "metrics.json").read_text())
    if blend["adopted"]:
        xlmr = pl.read_parquet(ROOT / config["inputs"]["xlmr_test"])
        blended = blend_probabilities(baseline_table, xlmr, config["blend"]["xlmr_weight"])
        export(
            "word_char_logreg_xlmr_blend_all_train",
            blended["prediction"].to_numpy(),
            {
                "xlmr_weight": config["blend"]["xlmr_weight"],
                "xlmr_test_sha256": fingerprint_file(ROOT / config["inputs"]["xlmr_test"]),
            },
        )
    holdout_path = stages["sentence-holdout"] / "metrics.json"
    if holdout_path.exists() and json.loads(holdout_path.read_text())["success"]["passed"]:
        settings = _sentence_settings(config)
        threshold = config["sentence_model"]["score_threshold"]
        training_aggregates, test_aggregates, fold_records, model = _char_aggregates_all_train(
            config, data, manifest, test, stages["families"]
        )
        scaler = MinMaxScaler(clip=True).fit(_aggregate_values(training_aggregates))
        training_matrix = sparse.hstack(
            [
                sparse.load_npz(
                    _verified_final_artifact(final_root / "training_features.npz", data.height)
                ),
                _aggregate_block(scaler, training_aggregates),
            ],
            format="csr",
        )
        logger.info(
            "Final joint matrix %s; peak RSS %.2f GiB", training_matrix.shape, _peak_rss_gib()
        )
        joint = build_pipeline(CANDIDATE, **config["baseline"]).named_steps["classifier"]
        joint.fit(training_matrix, data["label"].to_list())
        del training_matrix
        test_joint = sparse.hstack(
            [
                test_matrix,
                _aggregate_block(scaler, test_aggregates),
            ],
            format="csr",
        )
        probabilities = _ordered_probabilities(
            joint.classes_.tolist(), joint.predict_proba(test_joint)
        )
        joblib.dump({"classifier": joint, "scaler": scaler}, directory / "joint_model.joblib")
        joblib.dump(model, directory / "sentence_model.joblib")
        _write_parquet(
            _probability_table(test, probabilities, ["id"]),
            directory / "sentence_rewrite_test_probabilities.parquet",
        )
        export(
            "word_char_logreg_sentence_rewrite_all_train",
            np.asarray(LABELS)[probabilities.argmax(axis=1)],
            {
                "sentence_settings": {**settings, "ngram_range": list(settings["ngram_range"])},
                "sentence_folds": fold_records.to_dicts(),
                "aggregate_columns": AGGREGATE_COLUMNS,
                "aggregate_scaling": AGGREGATE_SCALING,
                "score_threshold": threshold,
                "joint_model_sha256": fingerprint_file(directory / "joint_model.joblib"),
            },
        )
    exports["seconds"] = perf_counter() - start
    _write_json(directory / "exports.json", exports)


def run_transformer_prepare(
    config: dict, transformer: dict, directory: Path, families: Path
) -> None:
    """Write the private Kaggle bundle of sentence keys and labels for the XLM-R detector."""
    data, manifest = _load_inputs(config)
    test = pl.read_csv(ROOT / config["inputs"]["test"], schema_overrides={"id": pl.String})
    assignments = pl.read_parquet(families / "assignments.parquet")
    development_links = pl.read_parquet(families / "development_links.parquet")
    minimum_words = config["alignment"]["minimum_sentence_words"]
    sentences = split_sentences(data.select("id", "text"), minimum_words)
    test_sentences = split_sentences(test.select("id", "text"), minimum_words)
    del data
    development = assignments.filter(pl.col("partition") == "development")
    labels = label_rewritten_sentences(
        sentences.join(development.select("id"), on="id"),
        development.select("id", "label"),
        development_links,
    ).join(development.filter(~pl.col("is_purged")).select("id", "family", "family_fold"), on="id")
    training = labels.filter(pl.col("family_fold") == transformer["training_fold"])
    monitor = (
        labels.filter(pl.col("family_fold") != transformer["training_fold"])
        .sort("id", "position")
        .sample(n=transformer["monitor_sentences"], seed=transformer["seed"])
    )
    excluded = development.filter(
        pl.col("family").is_in(training["family"].unique().implode())
    ).select("id")
    counts = pl.concat(
        [
            frame.group_by("id").agg(pl.len().alias("sentences"))
            for frame in [sentences, test_sentences]
        ]
    ).sort("id")
    bundle = ROOT / transformer["bundle_directory"]
    bundle.mkdir(parents=True, exist_ok=True)
    files = {
        "transformer_labels.parquet": training.select("id", "position", "is_rewritten"),
        "monitor_labels.parquet": monitor.select("id", "position", "is_rewritten"),
        "sentence_counts.parquet": counts,
    }
    for name, frame in files.items():
        _write_parquet(frame, bundle / name)
    for name in ["families.py", "validation.py"]:
        (bundle / name).write_bytes((ROOT / "src/hm3" / name).read_bytes())
    metadata = {
        "source_sha256": config["source_sha256"],
        "test_sha256": fingerprint_file(ROOT / config["inputs"]["test"]),
        "transformer": transformer,
        "minimum_sentence_words": minimum_words,
        "files": {name: fingerprint_file(bundle / name) for name in [*files, "families.py"]},
        "training_sentences": training.height,
        "training_rewritten_share": float(training.select(pl.col("is_rewritten").mean()).item()),
        "training_documents": int(training["id"].n_unique()),
        "monitor_sentences": monitor.height,
        "scored_sentences": int(counts["sentences"].sum()),
        "excluded_development_documents": excluded.height,
    }
    _write_json(bundle / "bundle_metadata.json", metadata)
    _write_json(
        bundle / "dataset-metadata.json",
        {
            "title": "HW3 sentence transformer bundle",
            "id": transformer["kaggle_dataset"],
            "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
        },
    )
    _write_parquet(
        training.select("id").unique().sort("id"), directory / "training_documents.parquet"
    )
    _write_parquet(excluded.sort("id"), directory / "excluded_development_documents.parquet")
    _write_json(directory / "summary.json", metadata)
    logger.info(
        "Transformer bundle %s: %s", bundle, {k: v for k, v in metadata.items() if k != "files"}
    )


def run_char_all_train(config: dict, directory: Path, families: Path) -> None:
    """Persist all-training out-of-fold character sentence aggregates and test aggregates."""
    data, manifest = _load_inputs(config)
    test = pl.read_csv(ROOT / config["inputs"]["test"], schema_overrides={"id": pl.String})
    training_aggregates, test_aggregates, fold_records, _ = _char_aggregates_all_train(
        config, data, manifest, test, families
    )
    _write_parquet(training_aggregates, directory / "training_aggregates.parquet")
    _write_parquet(test_aggregates, directory / "test_aggregates.parquet")
    _write_json(directory / "folds.json", {"folds": fold_records.to_dicts()})


def _transformer_scores(transformer: dict, prepare: Path) -> pl.DataFrame:
    metadata = json.loads((ROOT / transformer["score_metadata"]).read_text())
    bundle = json.loads((prepare / "summary.json").read_text())
    if (
        metadata["bundle_files"] != bundle["files"]
        or metadata["revision"] != transformer["revision"]
    ):
        raise ValueError(
            "Kaggle sentence scores do not match the prepared bundle or model revision"
        )
    scores = pl.read_parquet(ROOT / transformer["scores"])
    if scores.height != bundle["scored_sentences"]:
        raise ValueError("Kaggle sentence scores do not cover every eligible sentence")
    return scores


def _class_weights(labels: pl.Series, reference: pl.Series) -> np.ndarray:
    kept = labels.value_counts()
    full = reference.value_counts().rename({"count": "full"})
    factors = kept.join(full, on=labels.name).with_columns(
        (pl.col("full") / pl.col("count")).alias("weight")
    )
    return (
        pl.DataFrame({labels.name: labels})
        .join(
            factors.select(labels.name, "weight"), on=labels.name, how="left", maintain_order="left"
        )["weight"]
        .to_numpy()
    )


def run_transformer_holdout(
    config: dict, transformer: dict, directory: Path, stages: dict[str, Path]
) -> None:
    """Compare character-only and character plus XLM-R sentence aggregates on identical rows."""
    threshold = config["sentence_model"]["score_threshold"]
    scores = _transformer_scores(transformer, stages["transformer-prepare"])
    excluded = pl.read_parquet(
        stages["transformer-prepare"] / "excluded_development_documents.parquet"
    )
    reference = stages["reference"]
    development_rows = pl.read_parquet(reference / "development_rows.parquet")
    holdout_rows = pl.read_parquet(reference / "holdout_rows.parquet")
    partitions = [("development", development_rows), ("holdout", holdout_rows)]
    blocks = {
        "char": [
            rows.select("id").join(
                pl.read_parquet(stages["sentence-gate"] / f"{name}_aggregates.parquet"),
                on="id",
                how="left",
                maintain_order="left",
            )
            for name, rows in partitions
        ],
        "transformer": [
            aggregate_document_scores(
                scores.join(rows.select("id"), on="id"), rows["id"], threshold
            )
            for _, rows in partitions
        ],
    }
    keep = development_rows["id"].is_in(excluded["id"].implode()).not_().to_numpy()
    kept_labels = development_rows.filter(pl.Series(keep))["label"]
    weights = _class_weights(kept_labels, development_rows["label"])
    pair = (
        blocks["transformer"][0]
        .with_columns(development_rows["label"], pl.Series("is_kept", keep))
        .filter(pl.col("is_kept") & pl.col("label").is_in(["human", "fusion"]))
    )
    gate = {
        "monitor_sentence_auc": json.loads((ROOT / transformer["score_metadata"]).read_text())[
            "monitor_auc"
        ],
        "document_auc_max_human_vs_fusion": float(
            roc_auc_score(pair["label"] == "fusion", pair["max_score"])
        ),
        "kept_development_rows": int(keep.sum()),
        "excluded_development_rows": int((~keep).sum()),
    }
    logger.info("Transformer gate: %s", gate)
    scalers = {
        name: MinMaxScaler(clip=True).fit(_aggregate_values(frames[0].filter(pl.Series(keep))))
        for name, frames in blocks.items()
    }
    development_matrix = sparse.load_npz(reference / "development_features.npz")[keep]
    holdout_matrix = sparse.load_npz(reference / "holdout_features.npz")
    variants = {
        "control_char": ["char"],
        "candidate_char_transformer": ["char", "transformer"],
        "ablation_transformer": ["transformer"],
    }
    assignments = pl.read_parquet(stages["families"] / "assignments.parquet")
    holdout = _holdout_frame(config, assignments)
    predictions: dict[str, np.ndarray] = {}
    records: dict[str, dict] = {}
    for variant, names in variants.items():
        training_matrix = sparse.hstack(
            [
                development_matrix,
                *[
                    _aggregate_block(scalers[name], blocks[name][0].filter(pl.Series(keep)))
                    for name in names
                ],
            ],
            format="csr",
        )
        classifier = build_pipeline(CANDIDATE, **config["baseline"]).named_steps["classifier"]
        start = perf_counter()
        classifier.fit(training_matrix, kept_labels.to_list(), sample_weight=weights)
        seconds = perf_counter() - start
        del training_matrix
        test_matrix = sparse.hstack(
            [holdout_matrix, *[_aggregate_block(scalers[name], blocks[name][1]) for name in names]],
            format="csr",
        )
        probabilities = _ordered_probabilities(
            classifier.classes_.tolist(), classifier.predict_proba(test_matrix)
        )
        table = _probability_table(holdout_rows, probabilities, ["id", "label"])
        _write_parquet(table, directory / f"{variant}_holdout_probabilities.parquet")
        predictions[variant] = (
            holdout.select("id")
            .join(table, on="id", maintain_order="left")["prediction"]
            .to_numpy()
        )
        records[variant] = {
            "classifier_seconds": seconds,
            "classifier_iterations": int(np.max(classifier.n_iter_)),
        }
        logger.info(
            "Variant %s fitted in %.1fs; peak RSS %.2f GiB", variant, seconds, _peak_rss_gib()
        )
    previous = pl.read_parquet(stages["sentence-holdout"] / "holdout_probabilities.parquet")
    predictions["previous_char_all_rows"] = (
        holdout.select("id").join(previous, on="id", maintain_order="left")["prediction"].to_numpy()
    )
    evaluations = {
        "control_char": _evaluation(holdout, predictions["control_char"]),
        "candidate_char_transformer": _evaluation(
            holdout, predictions["candidate_char_transformer"], predictions["control_char"]
        ),
        "ablation_transformer": _evaluation(
            holdout, predictions["ablation_transformer"], predictions["control_char"]
        ),
        "candidate_versus_previous": _evaluation(
            holdout,
            predictions["candidate_char_transformer"],
            predictions["previous_char_all_rows"],
        ),
    }
    candidate = evaluations["candidate_char_transformer"]
    human_f1 = {
        name: next(
            row["f1"]
            for row in evaluations[name]["holdout"]["per_class"]
            if row["label"] == "human"
        )
        for name in ["control_char", "candidate_char_transformer"]
    }
    success = {
        "minimum_gain": candidate["paired_holdout"]["difference"] >= 0.003,
        "positive_interval": candidate["paired_holdout"]["ci95_low"] > 0,
        "family_clean_not_worse": candidate["paired_family_clean"]["difference"] >= 0,
        "human_f1_drop_within_limit": human_f1["candidate_char_transformer"]
        - human_f1["control_char"]
        >= -0.003,
    }
    success["passed"] = all(success.values())
    _write_json(
        directory / "metrics.json",
        {
            "gate": gate,
            "evaluations": evaluations,
            "fits": records,
            "success": success,
            "weights": "class weights restore full-development class totals on kept rows",
            "aggregate_scaling": AGGREGATE_SCALING,
        },
    )
    logger.info(
        "Transformer holdout: %s; success %s",
        {name: round(value["holdout"]["macro_f1"], 6) for name, value in evaluations.items()},
        success,
    )


def run_transformer_final(
    config: dict, transformer: dict, directory: Path, stages: dict[str, Path]
) -> None:
    """Fit character plus XLM-R aggregates on all eligible training rows and export a CSV."""
    holdout = json.loads((stages["transformer-holdout"] / "metrics.json").read_text())
    if not holdout["success"]["passed"]:
        raise ValueError("Transformer aggregates failed their holdout criteria; no export")
    threshold = config["sentence_model"]["score_threshold"]
    data, manifest = _load_inputs(config)
    test = pl.read_csv(ROOT / config["inputs"]["test"], schema_overrides={"id": pl.String})
    sample = pl.read_csv(ROOT / config["inputs"]["sample"], schema_overrides={"id": pl.String})
    scores = _transformer_scores(transformer, stages["transformer-prepare"])
    assignments = pl.read_parquet(stages["families"] / "assignments.parquet")
    training_documents = pl.read_parquet(
        stages["transformer-prepare"] / "training_documents.parquet"
    )
    excluded_families = assignments.join(training_documents, on="id")["train_family"].unique()
    excluded = assignments.filter(pl.col("train_family").is_in(excluded_families.implode()))
    keep = data["id"].is_in(excluded["id"].implode()).not_().to_numpy()
    kept_labels = data.filter(pl.Series(keep))["label"]
    weights = _class_weights(kept_labels, data["label"])
    char = [
        pl.read_parquet(stages["char-all-train"] / f"{name}_aggregates.parquet")
        for name in ["training", "test"]
    ]
    char = [
        ids.join(frame, on="id", how="left", maintain_order="left")
        for ids, frame in zip([data.select("id"), test.select("id")], char, strict=True)
    ]
    transformer_aggregates = [
        aggregate_document_scores(scores.join(frame.select("id"), on="id"), frame["id"], threshold)
        for frame in [data, test]
    ]
    blocks = {"char": char, "transformer": transformer_aggregates}
    scalers = {
        name: MinMaxScaler(clip=True).fit(_aggregate_values(frames[0].filter(pl.Series(keep))))
        for name, frames in blocks.items()
    }
    final_root = ROOT / config["inputs"]["final_baseline"]
    tfidf = sparse.load_npz(
        _verified_final_artifact(final_root / "training_features.npz", data.height)
    )[keep]
    training_matrix = sparse.hstack(
        [
            tfidf,
            *[
                _aggregate_block(scalers[name], blocks[name][0].filter(pl.Series(keep)))
                for name in blocks
            ],
        ],
        format="csr",
    )
    del tfidf
    logger.info(
        "Final transformer matrix %s; peak RSS %.2f GiB", training_matrix.shape, _peak_rss_gib()
    )
    classifier = build_pipeline(CANDIDATE, **config["baseline"]).named_steps["classifier"]
    classifier.fit(training_matrix, kept_labels.to_list(), sample_weight=weights)
    del training_matrix
    pipeline = joblib.load(
        _verified_final_artifact(final_root / "final_pipeline.joblib", data.height)
    )
    test_matrix = sparse.hstack(
        [
            _transform_in_batches(
                pipeline.named_steps["features"], test["text"].to_list(), config["n_jobs"]
            ),
            *[_aggregate_block(scalers[name], blocks[name][1]) for name in blocks],
        ],
        format="csr",
    )
    probabilities = _ordered_probabilities(
        classifier.classes_.tolist(), classifier.predict_proba(test_matrix)
    )
    joblib.dump({"classifier": classifier, "scalers": scalers}, directory / "joint_model.joblib")
    _write_parquet(
        _probability_table(test, probabilities, ["id"]), directory / "test_probabilities.parquet"
    )
    name = "word_char_logreg_xlmr_sentence_rewrite_all_train"
    output = directory / "submissions" / f"{name}.csv"
    labels = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    export_submission(
        test.select("id").with_columns(pl.Series("label", labels)),
        test.select("id"),
        sample,
        output,
        {
            "source_sha256": config["source_sha256"],
            "test_sha256": fingerprint_file(ROOT / config["inputs"]["test"]),
            "sample_sha256": fingerprint_file(ROOT / config["inputs"]["sample"]),
            "scores_sha256": fingerprint_file(ROOT / transformer["scores"]),
            "transformer": transformer,
            "joint_model_sha256": fingerprint_file(directory / "joint_model.joblib"),
            "training_rows": int(keep.sum()),
            "excluded_transformer_family_rows": int((~keep).sum()),
            "class_order": list(LABELS),
            "aggregate_scaling": AGGREGATE_SCALING,
        },
    )
    shares = {label: float(np.mean(labels == label)) for label in LABELS}
    _write_json(
        directory / "exports.json",
        {name: {"path": str(output.relative_to(ROOT)), "predicted_share": shares}},
    )
    logger.info("Exported validated submission %s; shares %s", output, shares)


def main() -> None:
    """Run one explicit stage of the sentence-rewrite, offset and blend experiments."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    parser = argparse.ArgumentParser(description="Sentence-rewrite and decision experiments")
    parser.add_argument("--config", type=Path, default=ROOT / "config/rewrite_experiments.toml")
    parser.add_argument(
        "--transformer-config", type=Path, default=ROOT / "config/xlmr_sentence.toml"
    )
    parser.add_argument("--stage", choices=STAGES + TRANSFORMER_STAGES, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    transformer = tomllib.loads(args.transformer_config.read_text())
    root = ROOT / config["output_root"]
    stages = {stage: root / stage.replace("-", "_") for stage in STAGES + TRANSFORMER_STAGES}
    directory = stages[args.stage]
    identity = _identity(
        config if args.stage in STAGES else {**config, "transformer": transformer}, args.stage
    )
    if _is_complete(directory, identity):
        logger.info("Stage %s already complete; verified outputs in %s", args.stage, directory)
        return
    directory.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    if args.stage == "families":
        run_families(config, directory)
    elif args.stage == "reference":
        run_reference(config, directory)
    elif args.stage == "sentence-gate":
        run_sentence_gate(config, directory, stages["families"])
    elif args.stage == "sentence-holdout":
        run_sentence_holdout(
            config, directory, stages["families"], stages["reference"], stages["sentence-gate"]
        )
    elif args.stage == "offset":
        run_offset(config, directory, stages["families"], stages["reference"])
    elif args.stage == "blend":
        run_blend(config, directory, stages["families"], stages["reference"])
    elif args.stage == "final":
        run_final(config, directory, stages)
    elif args.stage == "transformer-prepare":
        run_transformer_prepare(config, transformer, directory, stages["families"])
    elif args.stage == "char-all-train":
        run_char_all_train(config, directory, stages["families"])
    elif args.stage == "transformer-holdout":
        run_transformer_holdout(config, transformer, directory, stages)
    else:
        run_transformer_final(config, transformer, directory, stages)
    files = sorted(
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file()
        and path.suffix in {".json", ".parquet", ".csv"}
        and path.name != "record.json"
    )
    _complete(directory, identity, files, {"stage": args.stage, "seconds": perf_counter() - start})
    logger.info(
        "Stage %s completed in %.1fs; peak RSS %.2f GiB",
        args.stage,
        perf_counter() - start,
        _peak_rss_gib(),
    )


if __name__ == "__main__":
    main()
