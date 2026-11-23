import numpy as np
import polars as pl
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

from hm3.validation import SEED

SENTENCE_SEPARATOR = "\u0001"


def split_sentences(documents: pl.DataFrame, minimum_words: int = 8) -> pl.DataFrame:
    """Split documents into eligible sentences with normalized content hashes.

    A boundary is sentence-final punctuation followed by whitespace. Hashes use lowercase
    text with collapsed whitespace; the original sentence text is preserved for modeling.
    Repeated normalized sentences within one document keep only their first occurrence.

    Args:
        documents: Rows with `id` and `text` columns.
        minimum_words: Minimum whitespace-word count of an eligible sentence.

    Returns:
        Rows with `id`, `position`, `sentence`, `words` and `sentence_hash` in document order.
    """
    return (
        documents.select(
            "id",
            pl.col("text")
            .str.replace_all(r"([.!?])\s+", "$1" + SENTENCE_SEPARATOR)
            .str.split(SENTENCE_SEPARATOR)
            .alias("sentence"),
        )
        .explode("sentence")
        .with_columns(pl.col("sentence").str.strip_chars())
        .with_columns(
            pl.int_range(pl.len()).over("id").alias("position"),
            pl.col("sentence").str.count_matches(r"\S+").alias("words"),
            pl.col("sentence")
            .str.to_lowercase()
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .hash(seed=SEED)
            .alias("sentence_hash"),
        )
        .filter(pl.col("words") >= minimum_words)
        .unique(["id", "sentence_hash"], keep="first", maintain_order=True)
    )


def find_strong_links(
    sentences: pl.DataFrame,
    maximum_documents: int = 20,
    minimum_shared: int = 2,
    minimum_share: float = 0.2,
) -> pl.DataFrame:
    """Return document pairs that share enough verbatim eligible sentences.

    Sentences occurring in more than `maximum_documents` documents are treated as
    boilerplate and ignored. A pair is linked when it shares at least `minimum_shared`
    sentences and at least `minimum_share` of the smaller document's eligible sentences.

    Args:
        sentences: Output of `split_sentences` for the population to link.
        maximum_documents: Highest document frequency of a sentence used for linking.
        minimum_shared: Minimum number of shared sentences.
        minimum_share: Minimum shared fraction of the smaller document.

    Returns:
        Unique unordered pairs with `id`, `id_other` (lexically larger) and `shared` counts.
    """
    sizes = sentences.group_by("id").agg(pl.len().alias("sentences"))
    frequency = sentences.group_by("sentence_hash").agg(pl.len().alias("documents"))
    shared = sentences.select("id", "sentence_hash").join(
        frequency.filter(pl.col("documents").is_between(2, maximum_documents)),
        on="sentence_hash",
    )
    return (
        shared.join(shared, on="sentence_hash", suffix="_other")
        .filter(pl.col("id") < pl.col("id_other"))
        .group_by("id", "id_other")
        .agg(pl.len().alias("shared"))
        .join(sizes, on="id")
        .join(sizes, left_on="id_other", right_on="id", suffix="_other")
        .filter(
            (pl.col("shared") >= minimum_shared)
            & (
                pl.col("shared")
                >= minimum_share * pl.min_horizontal("sentences", "sentences_other")
            )
        )
        .select("id", "id_other", "shared")
        .sort("id", "id_other")
    )


def assign_families(ids: pl.Series, links: pl.DataFrame) -> pl.DataFrame:
    """Assign connected components of the link graph as document families.

    Args:
        ids: Every document identifier in the population, including unlinked documents.
        links: Linked pairs restricted to identifiers in `ids`.

    Returns:
        Rows with `id` and an integer `family` that is stable for the sorted input IDs.
    """
    index = pl.DataFrame({"id": ids.sort()}).with_row_index("node")
    edges = links.join(index, on="id").join(
        index.rename({"id": "id_other", "node": "node_other"}), on="id_other"
    )
    graph = sparse.coo_matrix(
        (
            np.ones(edges.height, dtype=np.int8),
            (edges["node"].to_numpy(), edges["node_other"].to_numpy()),
        ),
        shape=(index.height, index.height),
    )
    _, components = connected_components(graph, directed=False)
    return index.select("id").with_columns(pl.Series("family", components, dtype=pl.Int64))


def assign_family_folds(
    documents: pl.DataFrame, families: pl.DataFrame, folds: int, seed: int = SEED
) -> pl.DataFrame:
    """Allocate whole families to label-stratified folds.

    Args:
        documents: Rows with `id` and `label`.
        families: Family assignment for the same identifiers.
        folds: Number of folds.
        seed: Shuffling seed.

    Returns:
        Rows with `id`, `family` and `family_fold`.
    """
    prepared = documents.select("id", "label").join(families, on="id", validate="1:1").sort("id")
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    assignment = np.full(prepared.height, -1)
    for fold, (_, members) in enumerate(
        splitter.split(
            np.zeros(prepared.height), prepared["label"].to_numpy(), prepared["family"].to_numpy()
        )
    ):
        assignment[members] = fold
    return prepared.select("id", "family").with_columns(pl.Series("family_fold", assignment))


def label_rewritten_sentences(
    sentences: pl.DataFrame, documents: pl.DataFrame, links: pl.DataFrame
) -> pl.DataFrame:
    """Derive sentence training labels from verbatim overlap with linked human documents.

    A sentence of a fusion document is rewritten when its normalized hash is absent from
    every linked human document; otherwise it is retained. All eligible sentences of human
    documents are negatives. Fusion documents without a linked human document and AI
    documents receive no labels. Only the supplied population is used.

    Args:
        sentences: Output of `split_sentences` for the labeled population.
        documents: Rows with `id` and `label` for the same population.
        links: Strong links within the same population.

    Returns:
        Rows with `id`, `position`, `sentence` and boolean `is_rewritten`.
    """
    labels = documents.select("id", "label")
    pairs = pl.concat([links.select("id", "id_other"), links.select(id="id_other", id_other="id")])
    fusion_human = (
        pairs.join(labels, on="id")
        .join(labels.rename({"id": "id_other", "label": "label_other"}), on="id_other")
        .filter((pl.col("label") == "fusion") & (pl.col("label_other") == "human"))
        .select("id", "id_other")
    )
    retained = (
        fusion_human.join(
            sentences.select(id_other="id", sentence_hash="sentence_hash"), on="id_other"
        )
        .select("id", "sentence_hash")
        .unique()
        .with_columns(pl.lit(True).alias("is_retained"))
    )
    fusion_sentences = (
        sentences.join(fusion_human.select("id").unique(), on="id")
        .join(retained, on=["id", "sentence_hash"], how="left")
        .with_columns(pl.col("is_retained").is_null().alias("is_rewritten"))
    )
    human_sentences = sentences.join(
        labels.filter(pl.col("label") == "human").select("id"), on="id"
    ).with_columns(pl.lit(False).alias("is_rewritten"))
    return pl.concat(
        [
            frame.select("id", "position", "sentence", "is_rewritten")
            for frame in [fusion_sentences, human_sentences]
        ]
    ).sort("id", "position")
