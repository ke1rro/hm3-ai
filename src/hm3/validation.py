import hashlib
import json
import logging
import unicodedata
from importlib.metadata import version
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold

LABELS = ("human", "ai", "fusion")
SEED = 2026
logger = logging.getLogger(__name__)


def fingerprint_file(path: Path) -> str:
    """Return the SHA-256 identity of a file.

    Args:
        path: File to hash.

    Returns:
        Hexadecimal content fingerprint.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_training_data(path: Path) -> pl.DataFrame:
    """Load the labeled CSV and validate its schema and identity columns.

    Args:
        path: Competition training CSV.

    Returns:
        Original texts and explicit string labels without text normalization.
    """
    data = pl.read_csv(path, schema_overrides={"id": pl.String})
    if set(data.columns) != {"id", "text", "label", "lang", "domain"}:
        raise ValueError("Training data must contain id, text, label, lang and domain")
    if any(data.null_count().row(0)) or data["id"].n_unique() != data.height:
        raise ValueError("Training data has null values or duplicate IDs")
    if set(data["label"]) != set(LABELS):
        raise ValueError("Training data must contain exactly human, ai and fusion labels")
    if data.filter(pl.col("text").str.strip_chars() == "").height:
        raise ValueError("Training data contains blank text")
    return data


def _normalized_hash(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _identify_groups(data: pl.DataFrame) -> pl.DataFrame:
    return (
        data.select("id", "label", "lang", "domain", "text")
        .with_columns(
            pl.col("text").map_elements(_normalized_hash, return_dtype=pl.String).alias("group_id")
        )
        .drop("text")
        .sort("id")
    )


def _choose_stratification(data: pl.DataFrame, folds: int) -> list[str]:
    for columns in [["label", "lang", "domain"], ["label", "lang"], ["label"]]:
        counts = data.group_by(columns).agg(pl.col("group_id").n_unique().alias("groups"))
        if counts.filter(pl.col("groups") < folds).is_empty():
            return columns
    raise ValueError(f"At least {folds} independent groups per class are required")


def _strata(data: pl.DataFrame, columns: list[str]) -> np.ndarray:
    return data.select(pl.concat_str(columns, separator="|"))["label"].to_numpy()


def _cv_sample(manifest: pl.DataFrame, target: int | None, seed: int) -> pl.Series:
    development = manifest.filter(pl.col("partition") == "development")
    if target is None or target >= development.height:
        return manifest["partition"] == "development"
    if target < 1:
        raise ValueError("CV sample target must be positive")
    group_table = (
        development.group_by("group_id")
        .agg(
            pl.col("cv_fold").first(),
            pl.col("label").sort().first(),
            pl.col("lang").sort().first(),
            pl.col("domain").sort().first(),
        )
        .sort(["cv_fold", "label", "lang", "domain", "group_id"])
    )
    generator = np.random.default_rng(seed)
    selected = []
    for bucket in group_table.partition_by(
        ["cv_fold", "label", "lang", "domain"], maintain_order=True
    ):
        count = max(1, round(bucket.height * target / development.height))
        selected.extend(
            generator.choice(bucket["group_id"].to_list(), count, replace=False).tolist()
        )
    return manifest["group_id"].is_in(selected)


def validate_manifest(data: pl.DataFrame, manifest: pl.DataFrame) -> None:
    """Reject incompatible IDs, labels, groups, partitions or CV assignments.

    Args:
        data: Current labeled source data.
        manifest: Frozen ID-level assignments to validate.
    """
    expected = _identify_groups(data)
    columns = expected.columns
    if (
        manifest.height != data.height
        or manifest["id"].n_unique() != data.height
        or not manifest.select(columns).sort("id").equals(expected)
    ):
        raise ValueError("Manifest is incompatible with the source rows, labels or text groups")
    if set(manifest["partition"]) != {"development", "validation"}:
        raise ValueError("Manifest must contain development and validation partitions")
    development = manifest.filter(pl.col("partition") == "development")
    validation = manifest.filter(pl.col("partition") == "validation")
    if set(development["cv_fold"]) != {0, 1, 2} or set(validation["cv_fold"]) != {-1}:
        raise ValueError("Manifest contains invalid CV assignments")
    if validation["is_cv_sample"].any() or manifest["is_cv_sample"].null_count():
        raise ValueError("CV sample must contain development rows only")
    group_counts = manifest.group_by("group_id").agg(
        pl.col("partition").n_unique().alias("partitions"),
        pl.col("cv_fold").n_unique().alias("folds"),
        pl.col("is_cv_sample").n_unique().alias("samples"),
    )
    if group_counts.select(pl.any_horizontal(pl.all().exclude("group_id") > 1).any()).item():
        raise ValueError("Known duplicate group crosses partition, fold or CV sample boundaries")
    if set(development.filter(pl.col("is_cv_sample"))["cv_fold"]) != {0, 1, 2}:
        raise ValueError("CV sample must represent all three folds")


def audit_manifest(manifest: pl.DataFrame) -> dict[str, object]:
    """Summarize support, subgroup balance and confirmed group separation.

    Args:
        manifest: Validated frozen assignments.

    Returns:
        JSON-compatible counts and maximum absolute proportion deviations in percentage points.
    """
    development = manifest.filter(pl.col("partition") == "development")
    audit: dict[str, object] = {
        "partition_counts": manifest.group_by("partition").len().sort("partition").to_dicts(),
        "fold_counts": development.group_by("cv_fold").len().sort("cv_fold").to_dicts(),
        "cv_sample_fold_counts": development.filter(pl.col("is_cv_sample"))
        .group_by("cv_fold")
        .len()
        .sort("cv_fold")
        .to_dicts(),
        "known_group_partition_overlap": manifest.group_by("group_id")
        .agg(pl.col("partition").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
        .height,
        "known_group_fold_overlap": development.group_by("group_id")
        .agg(pl.col("cv_fold").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
        .height,
        "duplicate_groups": manifest.group_by("group_id").len().filter(pl.col("len") > 1).height,
    }
    for name, population in [
        ("cv", development),
        ("cv_sample", development.filter(pl.col("is_cv_sample"))),
    ]:
        audit[f"{name}_joint_counts"] = (
            population.group_by(["cv_fold", "label", "lang", "domain"])
            .len()
            .sort(["cv_fold", "label", "lang", "domain"])
            .to_dicts()
        )
    deviations = []
    for name, columns in [
        ("class", ["label"]),
        ("language", ["lang"]),
        ("domain", ["domain"]),
        ("joint", ["label", "lang", "domain"]),
    ]:
        counts = manifest.group_by(["partition", *columns]).len().sort(["partition", *columns])
        audit[f"{name}_counts"] = counts.to_dicts()
        for population, assignment, scope in [
            (manifest, "partition", "holdout"),
            (development, "cv_fold", "cv"),
            (development.filter(pl.col("is_cv_sample")), "cv_fold", "cv_sample"),
        ]:
            reference = (
                population.group_by(columns)
                .len()
                .with_columns((pl.col("len") / population.height).alias("reference"))
                .select(*columns, "reference")
            )
            actual = (
                population.group_by([assignment, *columns])
                .len()
                .with_columns(
                    (pl.col("len") / pl.col("len").sum().over(assignment)).alias("proportion")
                )
            )
            complete = (
                population.select(assignment)
                .unique()
                .join(reference, how="cross")
                .join(actual, on=[assignment, *columns], how="left")
                .with_columns(pl.col("proportion").fill_null(0))
            )
            maximum = complete.select(
                ((pl.col("proportion") - pl.col("reference")).abs().max() * 100)
            ).item()
            deviations.append({"scope": scope, "grouping": name, "max_deviation_pp": maximum})
        reference = (
            development.group_by(columns)
            .len()
            .with_columns((pl.col("len") / development.height).alias("reference"))
            .select(*columns, "reference")
        )
        sampled = development.filter(pl.col("is_cv_sample"))
        actual = (
            sampled.group_by(columns)
            .len()
            .with_columns((pl.col("len") / sampled.height).alias("proportion"))
            .select(*columns, "proportion")
        )
        maximum = (
            reference.join(actual, on=columns, how="left")
            .select(((pl.col("proportion").fill_null(0) - pl.col("reference")).abs().max() * 100))
            .item()
        )
        deviations.append(
            {
                "scope": "cv_sample_vs_development",
                "grouping": name,
                "max_deviation_pp": maximum,
            }
        )
    audit["distribution_deviations"] = deviations
    return audit


def build_manifest(
    data: pl.DataFrame, source_sha256: str, seed: int = SEED, cv_sample_size: int | None = 30000
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Allocate a group-safe fifth as holdout and three development-only CV folds.

    Choose joint stratification when there are enough independent groups, otherwise
    coarsen to class/language and then class. Record the actual choice. A fixed
    group-safe CV sample is allocated within the resulting fold and joint strata.

    Args:
        data: Labeled training rows with original text and unique IDs.
        source_sha256: Fingerprint of the source CSV.
        seed: Fixed split and sampling seed.
        cv_sample_size: Approximate shared CV sample size, or all development rows when omitted.

    Returns:
        ID-level assignments and provenance with distribution and leakage audits.
    """
    prepared = _identify_groups(data)
    holdout_columns = _choose_stratification(prepared, 5)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_indices, validation_indices = next(
        splitter.split(
            np.zeros(prepared.height),
            _strata(prepared, holdout_columns),
            prepared["group_id"].to_numpy(),
        )
    )
    partitions = np.full(prepared.height, "development", dtype="<U11")
    partitions[validation_indices] = "validation"
    development = prepared[train_indices]
    cv_columns = _choose_stratification(development, 3)
    folds = np.full(prepared.height, -1)
    cv_splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
    for fold, (_, local_validation) in enumerate(
        cv_splitter.split(
            np.zeros(development.height),
            _strata(development, cv_columns),
            development["group_id"].to_numpy(),
        )
    ):
        folds[train_indices[local_validation]] = fold
    manifest = prepared.with_columns(
        pl.Series("partition", partitions), pl.Series("cv_fold", folds)
    )
    manifest = manifest.with_columns(
        _cv_sample(manifest, cv_sample_size, seed).alias("is_cv_sample")
    )
    validate_manifest(data, manifest)
    metadata: dict[str, object] = {
        "version": 1,
        "scikit_learn_version": version("scikit-learn"),
        "source_sha256": source_sha256,
        "seed": seed,
        "cv_sample_target": cv_sample_size,
        "rows": data.height,
        "holdout_method": "first of five shuffled StratifiedGroupKFold folds",
        "cv_method": "three shuffled StratifiedGroupKFold folds within development",
        "group_definition": "SHA-256 of NFKC plus collapsed whitespace; original text preserved",
        "holdout_stratification": holdout_columns,
        "cv_stratification": cv_columns,
        "cv_sample_method": (
            "seeded group sampling within CV fold and joint strata; approximate target"
        ),
        "leakage_limitations": "Unidentified related rewrites may cross boundaries",
        "audit": audit_manifest(manifest),
    }
    return manifest, metadata


def get_or_create_manifest(
    data: pl.DataFrame,
    path: Path,
    source_sha256: str,
    seed: int = SEED,
    cv_sample_size: int | None = 30000,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Restore compatible frozen assignments or create them once without overwriting.

    Args:
        data: Current labeled data.
        path: Parquet manifest destination with a neighboring JSON provenance file.
        source_sha256: Current source fingerprint.
        seed: Required split seed.
        cv_sample_size: Required approximate shared CV sample size.

    Returns:
        Validated manifest and its provenance.
    """
    metadata_path = path.with_suffix(".json")
    if path.exists() != metadata_path.exists():
        raise ValueError("Incomplete frozen manifest; restore the missing file before continuing")
    if path.exists():
        metadata = json.loads(metadata_path.read_text())
        if any(
            metadata[key] != value
            for key, value in {
                "version": 1,
                "source_sha256": source_sha256,
                "seed": seed,
                "cv_sample_target": cv_sample_size,
            }.items()
        ) or metadata["manifest_sha256"] != fingerprint_file(path):
            raise ValueError(
                "Existing manifest is incompatible with the requested source or recipe"
            )
        manifest = pl.read_parquet(path)
        validate_manifest(data, manifest)
        logger.info("Restored frozen assignments from %s", path)
        return manifest, metadata
    manifest, metadata = build_manifest(data, source_sha256, seed, cv_sample_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    manifest.write_parquet(temporary)
    temporary.replace(path)
    metadata["manifest_sha256"] = fingerprint_file(path)
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary_metadata.replace(metadata_path)
    logger.info("Saved frozen assignments to %s", path)
    return manifest, metadata
