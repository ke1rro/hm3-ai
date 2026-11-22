from pathlib import Path

import polars as pl
import pytest


@pytest.fixture
def training_data() -> pl.DataFrame:
    """Create small joint strata with a confirmed normalized duplicate pair."""
    rows = [
        {
            "id": f"{label}-{lang}-{domain}-{index}",
            "text": f"{label} {lang} {domain} unique{index}",
            "label": label,
            "lang": lang,
            "domain": domain,
        }
        for label in ["human", "ai", "fusion"]
        for lang in ["en-EN", "de-DE"]
        for domain in ["news", "books"]
        for index in range(15)
    ]
    rows.extend(
        [
            {**rows[0], "id": "duplicate-one", "text": "Ａ shared\n text"},
            {**rows[0], "id": "duplicate-two", "text": "A shared text"},
        ]
    )
    return pl.DataFrame(rows)


def test_split_is_deterministic_group_safe_and_has_complete_cv(training_data: pl.DataFrame) -> None:
    """Catch row-order dependent allocation, overlapping groups and missing folds."""
    from hm3.validation import build_manifest

    first, metadata = build_manifest(training_data, "source", cv_sample_size=60)
    second, _ = build_manifest(training_data.reverse(), "source", cv_sample_size=60)
    assert first.equals(second)
    assert first["id"].n_unique() == training_data.height
    assert set(first["partition"]) == {"development", "validation"}
    development = first.filter(pl.col("partition") == "development")
    assert set(development["cv_fold"]) == {0, 1, 2}
    assert first.filter(pl.col("partition") == "validation")["cv_fold"].unique().to_list() == [-1]
    assert first.group_by("group_id").agg(pl.col("partition").n_unique())["partition"].max() == 1
    assert development.group_by("group_id").agg(pl.col("cv_fold").n_unique())["cv_fold"].max() == 1
    pair = first.filter(pl.col("id").is_in(["duplicate-one", "duplicate-two"]))
    assert pair["group_id"].n_unique() == 1
    assert pair["is_cv_sample"].n_unique() == 1
    assert not first.filter(pl.col("partition") == "validation")["is_cv_sample"].any()
    assert metadata["source_sha256"] == "source"


def test_existing_manifest_is_reused_and_rejects_changed_recipe(
    training_data: pl.DataFrame, tmp_path: Path
) -> None:
    """Catch silent replacement of a frozen split or mismatched data identity."""
    from hm3.validation import get_or_create_manifest

    path = tmp_path / "manifest.parquet"
    first, _ = get_or_create_manifest(training_data, path, "source", cv_sample_size=60)
    before = path.read_bytes()
    second, _ = get_or_create_manifest(training_data, path, "source", cv_sample_size=60)
    assert first.equals(second)
    assert path.read_bytes() == before
    with pytest.raises(ValueError, match="incompatible"):
        get_or_create_manifest(training_data, path, "changed", cv_sample_size=60)
    with pytest.raises(ValueError, match="incompatible"):
        get_or_create_manifest(training_data, path, "source", cv_sample_size=90)
    changed = training_data.with_columns(
        pl.when(pl.col("id") == "duplicate-two")
        .then(pl.lit("different source text"))
        .otherwise(pl.col("text"))
        .alias("text")
    )
    with pytest.raises(ValueError):
        get_or_create_manifest(changed, path, "source", cv_sample_size=60)


def test_manifest_rejects_fold_tampering(training_data: pl.DataFrame) -> None:
    """Catch validation rows incorrectly assigned to a training CV fold."""
    from hm3.validation import build_manifest, validate_manifest

    manifest, _ = build_manifest(training_data, "source", cv_sample_size=60)
    corrupted = manifest.with_columns(pl.lit(0).alias("cv_fold"))
    with pytest.raises(ValueError):
        validate_manifest(training_data, corrupted)


def test_rare_joint_stratum_uses_documented_fallback(training_data: pl.DataFrame) -> None:
    """Catch silent loss of a rare joint stratum during stratification."""
    from hm3.validation import build_manifest

    rare = training_data.head(1).with_columns(
        pl.lit("rare").alias("domain"), pl.lit("rare").alias("id")
    )
    _, metadata = build_manifest(pl.concat([training_data, rare]), "source", cv_sample_size=60)
    assert metadata["holdout_stratification"] == ["label", "lang"]
