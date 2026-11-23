import numpy as np
import polars as pl
from sklearn.metrics import f1_score

from hm3.validation import LABELS

PROBABILITY_COLUMNS = [f"prob_{label}" for label in LABELS]


def predict_with_fusion_offset(probabilities: np.ndarray, offset: float) -> np.ndarray:
    """Predict labels after adding a constant to the fusion log-probability.

    Args:
        probabilities: Rows of class probabilities in `LABELS` order (human, ai, fusion).
        offset: Value added to the log-probability of fusion; zero reproduces argmax.

    Returns:
        String labels from `LABELS`.
    """
    scores = np.log(np.clip(probabilities, 1e-12, 1.0))
    scores[:, LABELS.index("fusion")] += offset
    return np.asarray(LABELS)[scores.argmax(axis=1)]


def select_fusion_offset(
    truth: np.ndarray, probabilities: np.ndarray, grid: np.ndarray
) -> tuple[float, pl.DataFrame]:
    """Choose the offset with the highest macro-F1, breaking ties by the smallest magnitude.

    Args:
        truth: True string labels.
        probabilities: Aligned probabilities in `LABELS` order.
        grid: Candidate offsets.

    Returns:
        Selected offset and the complete macro-F1 curve.
    """
    curve = pl.DataFrame(
        {
            "offset": grid,
            "macro_f1": [
                f1_score(
                    truth,
                    predict_with_fusion_offset(probabilities, offset),
                    labels=list(LABELS),
                    average="macro",
                )
                for offset in grid
            ],
        }
    )
    best = curve.with_columns(pl.col("offset").abs().alias("magnitude")).sort(
        ["macro_f1", "magnitude"], descending=[True, False]
    )
    return float(best["offset"][0]), curve


def blend_probabilities(
    first: pl.DataFrame, second: pl.DataFrame, second_weight: float
) -> pl.DataFrame:
    """Align two probability tables by ID and return their weighted arithmetic mean.

    Args:
        first: Rows with `id` and `PROBABILITY_COLUMNS`.
        second: Rows with `id` and `PROBABILITY_COLUMNS` for the same IDs.
        second_weight: Weight of `second`; `first` receives the complement.

    Returns:
        Rows with `id`, blended `PROBABILITY_COLUMNS` and `prediction`, in `first` order.
    """
    joined = first.select("id", *PROBABILITY_COLUMNS).join(
        second.select("id", *PROBABILITY_COLUMNS),
        on="id",
        how="inner",
        validate="1:1",
        suffix="_second",
        maintain_order="left",
    )
    if joined.height != first.height or joined.height != second.height:
        raise ValueError("Blended probability tables must contain identical IDs")
    blended = joined.select(
        "id",
        *[
            ((1 - second_weight) * pl.col(name) + second_weight * pl.col(f"{name}_second")).alias(
                name
            )
            for name in PROBABILITY_COLUMNS
        ],
    )
    labels = np.asarray(LABELS)[blended.select(PROBABILITY_COLUMNS).to_numpy().argmax(axis=1)]
    return blended.with_columns(pl.Series("prediction", labels))
