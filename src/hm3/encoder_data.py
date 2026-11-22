import json
import logging
from pathlib import Path

import numpy as np
import polars as pl
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from hm3.validation import LABELS, fingerprint_file

LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))
logger = logging.getLogger(__name__)


def validate_encoder_manifest(frozen: pl.DataFrame, selected: pl.DataFrame) -> None:
    """Validate common transformer assignments against the original frozen split.

    Args:
        frozen: Original baseline ID assignments, never modified.
        selected: Encoder assignments with a role column.
    """
    columns = ["id", "label", "lang", "domain", "group_id", "partition"]
    if (
        selected.height != frozen.height
        or selected["id"].n_unique() != frozen.height
        or not selected.select(columns).sort("id").equals(frozen.select(columns).sort("id"))
    ):
        raise ValueError("Encoder manifest is incompatible with frozen IDs or labels")
    if not set(selected["role"]) <= {"train", "development_eval", "holdout", "unused"}:
        raise ValueError("Unknown encoder selection role")
    if selected.filter(
        (pl.col("partition") == "validation") != (pl.col("role") == "holdout")
    ).height:
        raise ValueError("Fixed holdout IDs differ from the baseline")
    if (
        selected.group_by("group_id")
        .agg(pl.col("role").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
        .height
    ):
        raise ValueError("Known duplicate group crosses encoder roles")
    for role in ["train", "development_eval"]:
        if set(selected.filter(pl.col("role") == role)["label"]) != set(LABELS):
            raise ValueError(f"Encoder {role} must represent all three classes")


def get_or_create_encoder_manifest(
    frozen: pl.DataFrame,
    path: Path,
    parent_manifest_sha256: str,
    train_target: int = 40000,
    evaluation_target: int = 2000,
    seed: int = 2026,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Freeze group-safe training and early-stopping IDs inside development only.

    Sample entire groups independently within joint class/language/domain buckets.
    Reserve evaluation first while leaving at least one group for training where
    possible. Group rounding makes targets approximate; scarce groups prioritize
    training and missing evaluation strata are explicitly audited.

    Args:
        frozen: Validated original holdout assignments.
        path: New encoder Parquet manifest with neighboring JSON provenance.
        parent_manifest_sha256: Original manifest identity.
        train_target: Approximate training size.
        evaluation_target: Approximate development evaluation size.
        seed: Fixed sampling seed shared across truncation experiments.

    Returns:
        Stable assignments and measured support/deviation metadata.
    """
    metadata_path = path.with_suffix(".json")
    recipe = {
        "version": 1,
        "parent_manifest_sha256": parent_manifest_sha256,
        "train_target": train_target,
        "evaluation_target": evaluation_target,
        "seed": seed,
    }
    if path.exists() != metadata_path.exists():
        raise ValueError("Incomplete encoder manifest; restore the missing file")
    if path.exists():
        metadata = json.loads(metadata_path.read_text())
        if any(metadata[key] != value for key, value in recipe.items()) or metadata[
            "encoder_manifest_sha256"
        ] != fingerprint_file(path):
            raise ValueError("Existing encoder manifest is incompatible with the requested recipe")
        selected = pl.read_parquet(path)
        validate_encoder_manifest(frozen, selected)
        return selected, metadata
    development = frozen.filter(pl.col("partition") == "development")
    if (
        min(train_target, evaluation_target) < 1
        or train_target + evaluation_target >= development.height
    ):
        raise ValueError("Positive training/evaluation targets must leave unused development rows")
    groups = (
        development.sort(["label", "lang", "domain", "id"])
        .group_by("group_id", maintain_order=True)
        .agg(pl.col("label").first(), pl.col("lang").first(), pl.col("domain").first())
        .sort(["label", "lang", "domain", "group_id"])
    )
    generator = np.random.default_rng(seed)
    train_groups, evaluation_groups = [], []
    scarce = []
    for bucket in groups.partition_by(["label", "lang", "domain"], maintain_order=True):
        order = generator.permutation(bucket["group_id"].to_numpy()).tolist()
        evaluation_count = min(
            max(1, round(bucket.height * evaluation_target / development.height)), bucket.height - 1
        )
        training_count = min(
            max(1, round(bucket.height * train_target / development.height)),
            bucket.height - evaluation_count,
        )
        evaluation_groups.extend(order[:evaluation_count])
        train_groups.extend(order[evaluation_count : evaluation_count + training_count])
        if evaluation_count == 0:
            scarce.append(bucket.select("label", "lang", "domain").row(0, named=True))
    selected = (
        frozen.select("id", "label", "lang", "domain", "group_id", "partition")
        .with_columns(
            pl.when(pl.col("partition") == "validation")
            .then(pl.lit("holdout"))
            .when(pl.col("group_id").is_in(evaluation_groups))
            .then(pl.lit("development_eval"))
            .when(pl.col("group_id").is_in(train_groups))
            .then(pl.lit("train"))
            .otherwise(pl.lit("unused"))
            .alias("role")
        )
        .sort("id")
    )
    validate_encoder_manifest(frozen, selected)
    reference = (
        development.group_by(["label", "lang", "domain"])
        .len()
        .with_columns((pl.col("len") / development.height).alias("reference"))
        .drop("len")
    )
    deviations = []
    for role in ["train", "development_eval"]:
        population = selected.filter(pl.col("role") == role)
        actual = (
            population.group_by(["label", "lang", "domain"])
            .len()
            .with_columns((pl.col("len") / population.height).alias("proportion"))
            .drop("len")
        )
        maximum = (
            reference.join(actual, on=["label", "lang", "domain"], how="left")
            .select(((pl.col("proportion").fill_null(0) - pl.col("reference")).abs().max() * 100))
            .item()
        )
        deviations.append({"role": role, "max_joint_deviation_pp": maximum})
    metadata: dict[str, object] = {
        **recipe,
        "role_counts": selected.group_by("role").len().sort("role").to_dicts(),
        "joint_counts": selected.group_by(["role", "label", "lang", "domain"])
        .len()
        .sort(["role", "label", "lang", "domain"])
        .to_dicts(),
        "distribution_deviations": deviations,
        "evaluation_strata_missing": scarce,
        "method": "seeded whole-group joint-stratum sampling; scarce buckets prioritize training",
        "known_group_overlap": 0,
        "unresolved_risk": "Related rewrites without confirmed group IDs may cross roles",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    selected.write_parquet(temporary)
    temporary.replace(path)
    metadata["encoder_manifest_sha256"] = fingerprint_file(path)
    temporary_metadata = metadata_path.with_suffix(".json.tmp")
    temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary_metadata.replace(metadata_path)
    return selected, metadata


def tokenize_document(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    policy: str,
) -> dict[str, list[int]]:
    """Select beginning or beginning/end tokens within a special-token-aware budget.

    Args:
        text: Original document text.
        tokenizer: Matching pretrained tokenizer.
        max_length: Total budget including required special tokens.
        policy: Head or head_tail selection.

    Returns:
        Unpadded input IDs and matching attention mask for one document.
    """
    budget = max_length - tokenizer.num_special_tokens_to_add(pair=False)
    if budget < 1 or policy not in {"head", "head_tail"}:
        raise ValueError("Invalid token budget or truncation policy")
    if (
        tokenizer.num_special_tokens_to_add(pair=False) != 2
        or tokenizer.bos_token_id is None
        or tokenizer.eos_token_id is None
    ):
        raise ValueError("This XLM-R policy requires the tokenizer's beginning/end special tokens")
    content = tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False)
    if len(content) > budget:
        beginning = (budget + 1) // 2
        ending = budget - beginning
        content = (
            content[:budget]
            if policy == "head"
            else content[:beginning] + (content[-ending:] if ending else [])
        )
    input_ids = [tokenizer.bos_token_id, *content, tokenizer.eos_token_id]
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


def tokenize_records(
    records: pl.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    policy: str,
) -> pl.DataFrame:
    """Tokenize original documents in fixed ID order without assigning chunk labels.

    Args:
        records: Selected ID/text rows, optionally with class labels.
        tokenizer: Matching model tokenizer.
        max_length: Total sequence budget.
        policy: Head or head_tail selection.

    Returns:
        ID-aligned variable-length token arrays and optional document labels.
    """
    records = records.sort("id")
    encoded = []
    for start in range(0, records.height, 1000):
        encoded.extend(
            tokenize_document(text, tokenizer, max_length, policy)
            for text in records["text"][start : start + 1000]
        )
        logger.info("Tokenized %s/%s documents", min(start + 1000, records.height), records.height)
    result = records.select("id").hstack(pl.DataFrame(encoded))
    if "label" in records.columns:
        result = result.with_columns(records["label"].replace_strict(LABEL2ID).alias("labels"))
    return result


class TokenizedDocuments(Dataset):
    """A compact columnar document dataset for dynamic-padding Torch batches.

    Attributes:
        records: ID-aligned unpadded token arrays and optional document labels.
    """

    def __init__(self, records: pl.DataFrame) -> None:
        self.records = records

    def __len__(self) -> int:
        return self.records.height

    def __getitem__(self, index: int) -> dict:
        row = self.records.row(index, named=True)
        return {key: value for key, value in row.items() if key != "id"}


def prediction_table(
    ids: list[str],
    logits: np.ndarray,
    true_labels: list[str] | None = None,
) -> pl.DataFrame:
    """Preserve document IDs and export logits/probabilities in the explicit label order.

    Args:
        ids: IDs in model-inference order.
        logits: One three-class logit row per ID.
        true_labels: Aligned known document labels for labeled evaluation.

    Returns:
        Predictions with named human/ai/fusion logit and probability columns.
    """
    if logits.shape != (len(ids), 3) or len(set(ids)) != len(ids) or not np.isfinite(logits).all():
        raise ValueError("Predictions require unique IDs and finite aligned three-class logits")
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    result = pl.DataFrame(
        {
            "id": ids,
            "prediction": [LABELS[index] for index in logits.argmax(axis=1)],
            **{f"logit_{label}": logits[:, index] for index, label in enumerate(LABELS)},
            **{f"prob_{label}": probabilities[:, index] for index, label in enumerate(LABELS)},
        }
    )
    if true_labels is not None:
        result = result.with_columns(pl.Series("label", true_labels))
    return result
