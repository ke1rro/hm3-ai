import logging
from time import perf_counter

import numpy as np
import polars as pl
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from hm3.validation import SEED

logger = logging.getLogger(__name__)
AGGREGATE_COLUMNS = [
    "max_score",
    "top2_mean_score",
    "mean_score",
    "count_over_threshold",
    "share_over_threshold",
    "log_sentences",
    "has_no_sentences",
]


def build_sentence_model(
    ngram_range: tuple[int, int] = (2, 5),
    min_df: int = 3,
    max_features: int = 300000,
    c: float = 1.0,
) -> Pipeline:
    """Construct the unfitted character n-gram rewritten-sentence classifier.

    Args:
        ngram_range: Inclusive character n-gram lengths within word boundaries.
        min_df: Minimum training sentence frequency of an n-gram.
        max_features: Maximum vocabulary size, selected by corpus frequency.
        c: Inverse L2 regularization strength.

    Returns:
        Pipeline whose positive class is a rewritten sentence; case is preserved.
    """
    return Pipeline(
        [
            (
                "features",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=ngram_range,
                    min_df=min_df,
                    max_features=max_features,
                    sublinear_tf=True,
                    lowercase=False,
                    dtype=np.float32,
                ),
            ),
            ("classifier", LogisticRegression(C=c, max_iter=1000, random_state=SEED)),
        ]
    )


def fit_sentence_model(settings: dict, sentences: list[str], is_rewritten: np.ndarray) -> Pipeline:
    """Fit the sentence classifier and drop its pruned-term set to keep worker copies small.

    Args:
        settings: Keyword arguments for `build_sentence_model`.
        sentences: Training sentences.
        is_rewritten: Aligned boolean targets.

    Returns:
        Fitted pipeline; `stop_words_` is cleared because transform uses only the vocabulary.
    """
    model = build_sentence_model(**settings).fit(sentences, is_rewritten)
    model.named_steps["features"].stop_words_ = None
    return model


def _score_batch(model: Pipeline, sentences: list[str]) -> np.ndarray:
    return model.predict_proba(sentences)[:, 1].astype(np.float32)


def score_sentences(
    model: Pipeline, sentences: list[str], n_jobs: int = 6, batch_size: int = 50000
) -> np.ndarray:
    """Return rewritten-sentence probabilities, transforming batches in parallel processes.

    Args:
        model: Fitted sentence classifier.
        sentences: Sentences to score, in the order of the returned array.
        n_jobs: Maximum worker processes.
        batch_size: Sentences per worker task.

    Returns:
        Float32 probabilities aligned with `sentences`.
    """
    if not sentences:
        return np.zeros(0, dtype=np.float32)
    batches = Parallel(n_jobs=n_jobs)(
        delayed(_score_batch)(model, sentences[start : start + batch_size])
        for start in range(0, len(sentences), batch_size)
    )
    return np.concatenate(batches)


def cross_fit_sentence_scores(
    sentences: pl.DataFrame,
    training: pl.DataFrame,
    folds: pl.DataFrame,
    settings: dict,
    n_jobs: int = 6,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Score every sentence with a model that never trained on its document family.

    For each fold, fit the sentence classifier on labeled sentences from documents in the
    other folds, then score all eligible sentences of documents in the held-out fold.
    Labeled sentences receive out-of-fold scores used for the stage gate.

    Args:
        sentences: Eligible sentences (`id`, `position`, `sentence`) of all documents to score.
        training: Labeled sentences (`id`, `position`, `sentence`, `is_rewritten`) allowed for
            fitting; it may be a purged subset of the scored population.
        folds: Rows with `id` and `family_fold` covering every scored document.
        settings: Keyword arguments for `build_sentence_model`.
        n_jobs: Maximum scoring worker processes.

    Returns:
        Scored sentences (`id`, `position`, `score`) and per-fold fit records.
    """
    assigned = sentences.join(folds.select("id", "family_fold"), on="id", how="left")
    if assigned["family_fold"].null_count():
        raise ValueError("Every scored document requires a family fold")
    training = training.join(folds.select("id", "family_fold"), on="id")
    scored, records = [], []
    for fold in sorted(assigned["family_fold"].unique().to_list()):
        fit_rows = training.filter(pl.col("family_fold") != fold)
        start = perf_counter()
        model = fit_sentence_model(
            settings, fit_rows["sentence"].to_list(), fit_rows["is_rewritten"].to_numpy()
        )
        fit_seconds = perf_counter() - start
        target = assigned.filter(pl.col("family_fold") == fold)
        start = perf_counter()
        scores = score_sentences(model, target["sentence"].to_list(), n_jobs)
        records.append(
            {
                "fold": fold,
                "fit_sentences": fit_rows.height,
                "fit_rewritten_share": float(fit_rows.select(pl.col("is_rewritten").mean()).item()),
                "scored_sentences": target.height,
                "fit_seconds": fit_seconds,
                "score_seconds": perf_counter() - start,
            }
        )
        logger.info("Sentence fold %s: %s", fold, records[-1])
        scored.append(target.select("id", "position").with_columns(pl.Series("score", scores)))
    return pl.concat(scored).sort("id", "position"), pl.DataFrame(records)


def aggregate_document_scores(
    scored: pl.DataFrame, ids: pl.Series, threshold: float = 0.5
) -> pl.DataFrame:
    """Summarize sentence scores into fixed document features.

    Documents without eligible sentences receive zeros and `has_no_sentences = 1`.

    Args:
        scored: Rows with `id` and sentence `score`.
        ids: Documents that require features, in the desired output order.
        threshold: Score threshold for the count and share features.

    Returns:
        Rows with `id` followed by `AGGREGATE_COLUMNS` as float64 values.
    """
    summary = scored.group_by("id").agg(
        pl.col("score").max().alias("max_score"),
        pl.col("score").top_k(2).mean().alias("top2_mean_score"),
        pl.col("score").mean().alias("mean_score"),
        (pl.col("score") >= threshold).sum().alias("count_over_threshold"),
        (pl.col("score") >= threshold).mean().alias("share_over_threshold"),
        pl.len().log1p().alias("log_sentences"),
    )
    return (
        pl.DataFrame({"id": ids})
        .join(summary, on="id", how="left", maintain_order="left")
        .with_columns(pl.col("max_score").is_null().cast(pl.Float64).alias("has_no_sentences"))
        .with_columns(pl.col(AGGREGATE_COLUMNS).cast(pl.Float64).fill_null(0.0))
        .select("id", *AGGREGATE_COLUMNS)
    )
