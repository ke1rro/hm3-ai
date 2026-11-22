import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from hm3.evaluation import classification_metrics
from hm3.validation import LABELS, SEED, fingerprint_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BaselineCandidate:
    """An explicit text-representation and linear-classifier experiment.

    Attributes:
        name: Descriptive unique experiment name.
        representation: Word, char or word_char sparse TF-IDF representation.
        classifier: Logistic regression or linear SVM implementation.
        c: Inverse regularization strength.
    """

    name: str
    representation: str
    classifier: str
    c: float


def build_pipeline(
    candidate: BaselineCandidate,
    word_max_features: int = 40000,
    char_max_features: int = 60000,
    min_df: int = 3,
) -> Pipeline:
    """Construct an unfitted sparse text-only pipeline with fold-local preprocessing.

    Args:
        candidate: Representation, classifier and regularization setting.
        word_max_features: Maximum word unigram and bigram vocabulary.
        char_max_features: Maximum character three-to-five-gram vocabulary.
        min_df: Minimum training-fold document frequency.

    Returns:
        Pipeline with features and classifier stages; original case is preserved.
    """
    word = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=min_df,
        max_features=word_max_features,
        lowercase=False,
        norm="l2",
        dtype=np.float32,
    )
    char = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=min_df,
        max_features=char_max_features,
        lowercase=False,
        norm="l2",
        dtype=np.float32,
    )
    if candidate.representation == "word":
        features = word
    elif candidate.representation == "char":
        features = char
    elif candidate.representation == "word_char":
        features = FeatureUnion([("word", word), ("char", char)], n_jobs=1)
    else:
        raise ValueError(f"Unknown representation: {candidate.representation}")
    if candidate.classifier == "logistic_regression":
        classifier = LogisticRegression(
            C=candidate.c,
            solver="saga",
            max_iter=300,
            tol=1e-3,
            random_state=SEED,
            class_weight=None,
        )
    elif candidate.classifier == "linear_svm":
        classifier = LinearSVC(
            C=candidate.c,
            dual="auto",
            max_iter=5000,
            tol=1e-3,
            random_state=SEED,
            class_weight=None,
        )
    else:
        raise ValueError(f"Unknown classifier: {candidate.classifier}")
    return Pipeline([("features", features), ("classifier", classifier)])


def run_cross_validation(
    data: pl.DataFrame,
    manifest: pl.DataFrame,
    candidates: list[BaselineCandidate],
    word_max_features: int = 40000,
    char_max_features: int = 60000,
    min_df: int = 3,
    checkpoint_path: Path | None = None,
    run_identity: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compare candidates using identical frozen development-only CV rows and folds.

    Fit a fresh pipeline inside each fold. Retain out-of-fold predictions with IDs
    and report macro-F1, wall time and coefficient-memory estimates. Fold scores
    and pooled OOF scores are different summaries and must be reported separately.

    Args:
        data: Full labeled source data.
        manifest: Validated frozen assignments including the shared CV sample.
        candidates: Controlled experiment matrix with unique names.
        word_max_features: Word vocabulary bound.
        char_max_features: Character vocabulary bound.
        min_df: Minimum training-fold document frequency.
        checkpoint_path: Optional directory for completed-fold scores and predictions.
        run_identity: Required source, split, recipe and environment identity when checkpointing.

    Returns:
        Fold-level measurements and aligned OOF predictions for every candidate.
    """
    if checkpoint_path is not None and run_identity is None:
        raise ValueError("CV checkpointing requires an explicit run identity")
    if len({candidate.name for candidate in candidates}) != len(candidates):
        raise ValueError("Candidate names must be unique")
    selected = manifest.filter(
        (pl.col("partition") == "development") & pl.col("is_cv_sample")
    ).select("id", "cv_fold")
    if set(selected["cv_fold"]) != {0, 1, 2}:
        raise ValueError("CV requires the frozen sample to represent all three folds")
    sample = data.join(selected, on="id", how="inner", validate="1:1").sort("id")
    if sample.height != selected.height:
        raise ValueError("CV assignments contain missing source IDs")
    measurements, predictions = [], []
    for candidate in candidates:
        for fold in range(3):
            training = sample.filter(pl.col("cv_fold") != fold)
            validation = sample.filter(pl.col("cv_fold") == fold)
            checkpoint = (
                checkpoint_path / candidate.name / f"fold_{fold}.json"
                if checkpoint_path is not None
                else None
            )
            identity: dict[str, object] = {
                "run_identity": run_identity,
                "candidate": asdict(candidate),
                "fold": fold,
                "word_max_features": word_max_features,
                "char_max_features": char_max_features,
                "min_df": min_df,
            }
            if checkpoint is not None and checkpoint.exists():
                record = json.loads(checkpoint.read_text())
                prediction_path = checkpoint.with_suffix(".parquet")
                if record["identity"] != identity or record[
                    "predictions_sha256"
                ] != fingerprint_file(prediction_path):
                    raise ValueError("CV checkpoint is incompatible with the run recipe")
                restored = pl.read_parquet(prediction_path)
                if (
                    not restored.select("id", "label")
                    .sort("id")
                    .equals(validation.select("id", "label").sort("id"))
                ):
                    raise ValueError("CV checkpoint is incompatible with the validation rows")
                measurements.append(record["score"])
                predictions.append(restored)
                logger.info("Restored %s, CV fold %s", candidate.name, fold)
                continue
            pipeline = build_pipeline(candidate, word_max_features, char_max_features, min_df)
            logger.info("Fitting %s, CV fold %s", candidate.name, fold)
            start = perf_counter()
            pipeline.fit(training["text"].to_list(), training["label"].to_list())
            fit_seconds = perf_counter() - start
            start = perf_counter()
            predicted = pipeline.predict(validation["text"].to_list()).tolist()
            predict_seconds = perf_counter() - start
            classifier = pipeline.named_steps["classifier"]
            measurements.append(
                {
                    "candidate": candidate.name,
                    "representation": candidate.representation,
                    "classifier": candidate.classifier,
                    "c": candidate.c,
                    "fold": fold,
                    "train_rows": training.height,
                    "validation_rows": validation.height,
                    "macro_f1": f1_score(
                        validation["label"].to_list(),
                        predicted,
                        labels=list(LABELS),
                        average="macro",
                        zero_division=0,
                    ),
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    "features": int(classifier.coef_.shape[1]),
                    "coefficient_mib": classifier.coef_.nbytes / 1024**2,
                    "hit_iteration_limit": bool(np.max(classifier.n_iter_) >= classifier.max_iter),
                }
            )
            predictions.append(
                validation.select("id", "label").with_columns(
                    pl.Series("prediction", predicted),
                    pl.lit(candidate.name).alias("candidate"),
                    pl.lit(fold).alias("fold"),
                )
            )
            if checkpoint is not None:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                prediction_path = checkpoint.with_suffix(".parquet")
                temporary_predictions = prediction_path.with_suffix(".parquet.tmp")
                predictions[-1].write_parquet(temporary_predictions)
                temporary_predictions.replace(prediction_path)
                temporary_record = checkpoint.with_suffix(".json.tmp")
                temporary_record.write_text(
                    json.dumps(
                        {
                            "identity": identity,
                            "score": measurements[-1],
                            "predictions_sha256": fingerprint_file(prediction_path),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                temporary_record.replace(checkpoint)
    return pl.DataFrame(measurements), pl.concat(predictions)


def summarize_cv(scores: pl.DataFrame, predictions: pl.DataFrame) -> pl.DataFrame:
    """Rank mean fold macro-F1 and distinguish it from pooled OOF performance.

    Args:
        scores: Three measured fold rows per candidate.
        predictions: OOF predictions generated by run_cross_validation.

    Returns:
        Mean, sample standard deviation, pooled OOF macro-F1 and resource summaries.
    """
    pooled = pl.DataFrame(
        [
            {
                "candidate": group["candidate"][0],
                "pooled_oof_macro_f1": f1_score(
                    group["label"].to_list(),
                    group["prediction"].to_list(),
                    labels=list(LABELS),
                    average="macro",
                    zero_division=0,
                ),
            }
            for group in predictions.partition_by("candidate")
        ]
    )
    return (
        scores.group_by("candidate")
        .agg(
            pl.col("representation").first(),
            pl.col("classifier").first(),
            pl.col("c").first(),
            pl.col("macro_f1").mean().alias("cv_macro_f1_mean"),
            pl.col("macro_f1").std(ddof=1).alias("cv_macro_f1_std"),
            pl.col("fit_seconds").sum().alias("total_fit_seconds"),
            pl.col("coefficient_mib").max().alias("max_coefficient_mib"),
            pl.col("hit_iteration_limit").any(),
        )
        .join(pooled, on="candidate")
        .sort(["cv_macro_f1_mean", "candidate"], descending=[True, False])
    )


def fit_holdout(
    data: pl.DataFrame,
    manifest: pl.DataFrame,
    candidate: BaselineCandidate,
    word_max_features: int = 40000,
    char_max_features: int = 60000,
    min_df: int = 3,
) -> tuple[Pipeline, pl.DataFrame, dict[str, object]]:
    """Refit a CV-selected candidate on all development rows and evaluate the holdout.

    Args:
        data: Full labeled source data.
        manifest: Validated frozen holdout and CV assignments.
        candidate: Configuration selected exclusively using CV.
        word_max_features: Word vocabulary bound.
        char_max_features: Character vocabulary bound.
        min_df: Minimum development-set document frequency.

    Returns:
        Fitted pipeline, validation predictions and fixed-label evaluation metrics.
    """
    assigned = data.join(manifest.select("id", "partition"), on="id", how="inner", validate="1:1")
    if assigned.height != data.height:
        raise ValueError("Holdout assignments do not cover all source IDs")
    training = assigned.filter(pl.col("partition") == "development")
    validation = assigned.filter(pl.col("partition") == "validation")
    pipeline = build_pipeline(candidate, word_max_features, char_max_features, min_df)
    start = perf_counter()
    pipeline.fit(training["text"].to_list(), training["label"].to_list())
    predicted = pipeline.predict(validation["text"].to_list()).tolist()
    fit_predict_seconds = perf_counter() - start
    predictions = (
        validation.select("id", "label", "lang", "domain")
        .with_columns(
            pl.Series("prediction", predicted),
            validation["text"].str.count_matches(r"\S+").alias("words"),
        )
        .with_columns(
            pl.when(pl.col("words") <= 128)
            .then(pl.lit("001-128"))
            .when(pl.col("words") <= 512)
            .then(pl.lit("129-512"))
            .when(pl.col("words") <= 1024)
            .then(pl.lit("513-1024"))
            .otherwise(pl.lit("1025+"))
            .alias("length_bucket")
        )
    )
    metrics = classification_metrics(validation["label"].to_list(), predicted)
    metrics.update(
        {
            "candidate": candidate.name,
            "training_rows": training.height,
            "validation_rows": validation.height,
            "fit_predict_seconds": fit_predict_seconds,
            "model_class_order": pipeline.classes_.tolist(),
        }
    )
    return pipeline, predictions, metrics


def fit_or_restore_final_pipeline(
    data: pl.DataFrame,
    candidate: BaselineCandidate,
    path: Path,
    recipe_sha256: str,
    source_sha256: str,
    word_max_features: int = 40000,
    char_max_features: int = 60000,
    min_df: int = 3,
) -> tuple[Pipeline, dict[str, object]]:
    """Restore a verified local final checkpoint or fit and persist it once.

    Persist model provenance before inference starts, so interrupted prediction can
    restart without training again. An orphaned model or incompatible record fails
    clearly instead of silently replacing a completed fit.

    Args:
        data: All labeled training rows.
        candidate: CV-selected model recipe.
        path: Local joblib checkpoint with a neighboring joblib.json record.
        recipe_sha256: Complete experiment recipe identity.
        source_sha256: Labeled CSV identity.
        word_max_features: Word vocabulary bound.
        char_max_features: Character vocabulary bound.
        min_df: Minimum labeled-training document frequency.

    Returns:
        Fitted or restored pipeline and verified checkpoint provenance.
    """
    metadata_path = path.with_suffix(".joblib.json")
    expected: dict[str, object] = {
        "recipe_sha256": recipe_sha256,
        "source_sha256": source_sha256,
        "candidate": asdict(candidate),
        "seed": SEED,
        "training_rows": data.height,
        "feature_options": {
            "word_max_features": word_max_features,
            "char_max_features": char_max_features,
            "min_df": min_df,
        },
    }
    if path.exists() != metadata_path.exists():
        raise ValueError("Incomplete final checkpoint: restore its model or provenance record")
    if path.exists():
        metadata = json.loads(metadata_path.read_text())
        if any(metadata[key] != value for key, value in expected.items()) or metadata[
            "model_sha256"
        ] != fingerprint_file(path):
            raise ValueError("Final checkpoint is incompatible with the source or recipe")
        pipeline = joblib.load(path)
        if (
            not isinstance(pipeline, Pipeline)
            or pipeline.classes_.tolist() != metadata["model_class_order"]
        ):
            raise ValueError("Final checkpoint has an incompatible model or class order")
        return pipeline, metadata
    path.parent.mkdir(parents=True, exist_ok=True)
    pipeline = build_pipeline(candidate, word_max_features, char_max_features, min_df)
    pipeline.fit(data["text"].to_list(), data["label"].to_list())
    temporary = path.with_suffix(".joblib.tmp")
    joblib.dump(pipeline, temporary, compress=3)
    temporary.replace(path)
    metadata = {
        **expected,
        "model_sha256": fingerprint_file(path),
        "model_class_order": pipeline.classes_.tolist(),
    }
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary_metadata.replace(metadata_path)
    return pipeline, metadata
