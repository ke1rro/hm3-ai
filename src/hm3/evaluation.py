import json
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from hm3.validation import LABELS, fingerprint_file


def classification_metrics(truth: list[str], predictions: list[str]) -> dict[str, object]:
    """Evaluate a fixed three-class label order, including absent subgroup classes.

    Args:
        truth: Ground-truth document labels.
        predictions: Aligned predicted labels.

    Returns:
        Aggregate metrics, class supports and raw and row-normalized confusion matrices.
    """
    if not truth or len(truth) != len(predictions):
        raise ValueError("Metrics require nonempty aligned label vectors")
    if not set(truth + predictions) <= set(LABELS):
        raise ValueError("Unknown class label")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=list(LABELS), zero_division=0
    )
    matrix = confusion_matrix(truth, predictions, labels=list(LABELS))
    normalized = np.divide(
        matrix,
        matrix.sum(axis=1, keepdims=True),
        out=np.zeros(matrix.shape, dtype=float),
        where=matrix.sum(axis=1, keepdims=True) != 0,
    )
    return {
        "label_order": list(LABELS),
        "macro_f1": f1_score(
            truth, predictions, labels=list(LABELS), average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            truth, predictions, labels=list(LABELS), average="weighted", zero_division=0
        ),
        "accuracy": accuracy_score(truth, predictions),
        "per_class": [
            {
                "label": label,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        ],
        "confusion_matrix": matrix.tolist(),
        "row_normalized_confusion_matrix": normalized.tolist(),
        "metric_convention": "fixed human, ai, fusion labels; absent classes score zero",
    }


def subgroup_metrics(predictions: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Report fixed-label macro-F1 and support for each requested subgroup.

    Args:
        predictions: Rows containing label, prediction and subgroup columns.
        columns: Joint grouping columns.

    Returns:
        Counts, observed classes, macro-F1 and an explicit small-group flag.
    """
    rows = []
    for group in predictions.partition_by(columns, maintain_order=True):
        truth, predicted = group["label"].to_list(), group["prediction"].to_list()
        rows.append(
            {
                **{name: group[name][0] for name in columns},
                "support": group.height,
                "observed_classes": len(set(truth)),
                "has_sufficient_support": group.height >= 100 and len(set(truth)) == 3,
                "macro_f1": f1_score(
                    truth, predicted, labels=list(LABELS), average="macro", zero_division=0
                ),
                **{f"{label}_support": truth.count(label) for label in LABELS},
            }
        )
    return pl.DataFrame(rows).sort(columns)


def export_submission(
    predictions: pl.DataFrame,
    test_ids: pl.DataFrame,
    sample: pl.DataFrame,
    path: Path,
    metadata: dict[str, object],
    expected_rows: int = 25000,
) -> pl.DataFrame:
    """Align predictions by ID, validate the competition contract and export atomically.

    Args:
        predictions: Predicted id,label pairs in any order.
        test_ids: Complete test identifiers.
        sample: Sample submission defining the required row order.
        path: Destination CSV with a neighboring JSON provenance file.
        metadata: Source, recipe and model fingerprints supplied by inference.
        expected_rows: Required test row count.

    Returns:
        Validated submission in sample order.
    """
    for name, frame in [("predictions", predictions), ("test", test_ids), ("sample", sample)]:
        if frame.height != expected_rows or frame["id"].n_unique() != expected_rows:
            raise ValueError(f"{name} must contain {expected_rows} distinct IDs")
        if frame["id"].null_count():
            raise ValueError(f"{name} contains null IDs")
    if predictions.columns != ["id", "label"] or sample.columns != ["id", "label"]:
        raise ValueError("Predictions and sample must have precisely id,label columns")
    if not set(predictions["label"]) <= set(LABELS):
        raise ValueError("Submission contains missing or invalid labels")
    if set(test_ids["id"]) != set(sample["id"]) or set(predictions["id"]) != set(sample["id"]):
        raise ValueError("Submission IDs do not match test and sample IDs")
    ordered = sample.select("id").join(predictions, on="id", how="left", maintain_order="left")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    ordered.write_csv(temporary)
    temporary.replace(path)
    record = {
        **metadata,
        "rows": expected_rows,
        "columns": ["id", "label"],
        "csv_sha256": fingerprint_file(path),
    }
    temporary_metadata = path.with_suffix(".json.tmp")
    temporary_metadata.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary_metadata.replace(path.with_suffix(".json"))
    return ordered
