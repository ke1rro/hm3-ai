from pathlib import Path

import polars as pl
import pytest
from scipy.sparse import issparse
from sklearn.pipeline import Pipeline


@pytest.mark.parametrize(
    ("representation", "classifier"),
    [
        ("word", "logistic_regression"),
        ("char", "logistic_regression"),
        ("word_char", "logistic_regression"),
        ("word_char", "linear_svm"),
    ],
)
def test_requested_pipeline_preserves_sparse_features(representation: str, classifier: str) -> None:
    """Catch dense concatenation, invalid classifiers or lost class labels."""
    from hm3.baseline import BaselineCandidate, build_pipeline

    texts = ["human natural personal", "ai synthetic generated", "fusion mixed rewritten"] * 3
    labels = ["human", "ai", "fusion"] * 3
    pipeline = build_pipeline(
        BaselineCandidate("candidate", representation, classifier, 1.0), min_df=1
    )
    pipeline.fit(texts, labels)
    assert issparse(pipeline.named_steps["features"].transform(texts))
    assert set(pipeline.predict(texts)) <= {"human", "ai", "fusion"}
    assert set(pipeline.classes_) == {"human", "ai", "fusion"}


def test_cv_vocabulary_never_learns_validation_or_holdout_words(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch fitting a vectorizer before the fold boundary or including holdout rows."""
    from hm3 import baseline

    data = pl.DataFrame(
        [
            {"id": f"{fold}-{label}", "text": f"common sentinel{fold} {label}", "label": label}
            for fold in range(3)
            for label in ["human", "ai", "fusion"]
        ]
        + [{"id": "holdout", "text": "holdoutonly", "label": "human"}]
    )
    manifest = data.select("id").with_columns(
        pl.Series("partition", ["development"] * 9 + ["validation"]),
        pl.Series("cv_fold", [0] * 3 + [1] * 3 + [2] * 3 + [-1]),
        pl.Series("is_cv_sample", [True] * 9 + [False]),
    )
    original_builder = baseline.build_pipeline
    pipelines = []

    def record_pipeline(
        candidate: baseline.BaselineCandidate,
        word_max_features: int = 40000,
        char_max_features: int = 60000,
        min_df: int = 3,
    ) -> Pipeline:
        pipeline = original_builder(candidate, word_max_features, char_max_features, min_df)
        pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(baseline, "build_pipeline", record_pipeline)
    scores, predictions = baseline.run_cross_validation(
        data,
        manifest,
        [baseline.BaselineCandidate("word", "word", "logistic_regression", 1.0)],
        min_df=1,
    )
    assert scores.height == 3
    assert predictions.height == 9
    assert "holdout" not in predictions["id"]
    for fold, pipeline in enumerate(pipelines):
        vocabulary = pipeline.named_steps["features"].vocabulary_
        assert "holdoutonly" not in vocabulary
        assert f"sentinel{fold}" not in vocabulary
        assert f"sentinel{(fold + 1) % 3}" in vocabulary


def test_metrics_keep_three_class_definition_for_missing_classes() -> None:
    """Catch subgroup macro-F1 silently changing to an observed-class average."""
    from hm3.evaluation import classification_metrics

    metrics = classification_metrics(["human", "human"], ["human", "human"])
    assert metrics["macro_f1"] == pytest.approx(1 / 3)
    assert metrics["accuracy"] == 1
    assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 0, 0], [0, 0, 0]]


def test_submission_aligns_by_id_and_validates_schema(tmp_path: Path) -> None:
    """Catch positional export, incomplete IDs or illegal labels."""
    from hm3.evaluation import export_submission

    sample = pl.DataFrame({"id": ["b", "a", "c"], "label": ["ai"] * 3})
    test_ids = pl.DataFrame({"id": ["c", "b", "a"]})
    predictions = pl.DataFrame({"id": ["a", "b", "c"], "label": ["fusion", "human", "ai"]})
    path = tmp_path / "submission.csv"
    export_submission(predictions, test_ids, sample, path, {"seed": 2026}, expected_rows=3)
    result = pl.read_csv(path)
    assert result.columns == ["id", "label"]
    assert result["label"].to_list() == ["human", "fusion", "ai"]
    for invalid in [
        predictions.head(2),
        pl.concat([predictions, predictions.head(1)]),
        predictions.with_columns(pl.lit("invalid").alias("label")),
    ]:
        with pytest.raises(ValueError):
            export_submission(invalid, test_ids, sample, path, {}, expected_rows=3)


def test_final_pipeline_restores_without_refitting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch repeating an expensive final fit after interrupted inference."""
    from hm3.baseline import BaselineCandidate, fit_or_restore_final_pipeline

    data = pl.DataFrame(
        {
            "text": ["human natural personal", "ai synthetic generated", "fusion mixed rewritten"]
            * 3,
            "label": ["human", "ai", "fusion"] * 3,
        }
    )
    candidate = BaselineCandidate("word", "word", "logistic_regression", 1.0)
    path = tmp_path / "model.joblib"
    first, record = fit_or_restore_final_pipeline(
        data, candidate, path, "recipe", "source", min_df=1
    )
    expected = first.predict(data["text"].to_list()).tolist()

    def reject_refit(self: Pipeline, *args: object, **kwargs: object) -> Pipeline:
        raise RuntimeError("A compatible checkpoint must not be refitted")

    monkeypatch.setattr(Pipeline, "fit", reject_refit)
    restored, restored_record = fit_or_restore_final_pipeline(
        data, candidate, path, "recipe", "source", min_df=1
    )
    assert restored.predict(data["text"].to_list()).tolist() == expected
    assert restored_record["model_sha256"] == record["model_sha256"]
    with pytest.raises(ValueError, match="incompatible"):
        fit_or_restore_final_pipeline(data, candidate, path, "different", "source", min_df=1)


def test_cv_reuses_completed_fold_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch retraining completed CV folds after a hosted session interruption."""
    from hm3 import baseline

    data = pl.DataFrame(
        [
            {"id": f"{fold}-{label}", "text": f"common sentinel{fold} {label}", "label": label}
            for fold in range(3)
            for label in ["human", "ai", "fusion"]
        ]
    )
    manifest = data.select("id").with_columns(
        pl.lit("development").alias("partition"),
        pl.Series("cv_fold", [0] * 3 + [1] * 3 + [2] * 3),
        pl.lit(True).alias("is_cv_sample"),
    )
    candidate = baseline.BaselineCandidate("word", "word", "logistic_regression", 1.0)
    scores, predictions = baseline.run_cross_validation(
        data, manifest, [candidate], min_df=1, checkpoint_path=tmp_path, run_identity="recipe"
    )

    def reject_refit(self: Pipeline, *args: object, **kwargs: object) -> Pipeline:
        raise RuntimeError("Completed folds must not be fitted again")

    monkeypatch.setattr(Pipeline, "fit", reject_refit)
    restored_scores, restored_predictions = baseline.run_cross_validation(
        data, manifest, [candidate], min_df=1, checkpoint_path=tmp_path, run_identity="recipe"
    )
    assert restored_scores.equals(scores)
    assert restored_predictions.equals(predictions)
    with pytest.raises(ValueError, match="incompatible"):
        baseline.run_cross_validation(
            data, manifest, [candidate], min_df=1, checkpoint_path=tmp_path, run_identity="changed"
        )
