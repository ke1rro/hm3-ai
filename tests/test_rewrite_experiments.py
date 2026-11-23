import numpy as np
import polars as pl

SHARED = [
    "The committee reviewed the annual budget proposal in great detail today.",
    "Several members raised concerns about the rising cost of public transport.",
    "The chair promised a written answer before the next plenary session begins.",
]


def test_rewrite_labels_follow_verbatim_overlap_with_linked_human_documents() -> None:
    """Catch labelling retained sentences as rewritten or labelling unrelated documents."""
    from hm3.families import (
        assign_families,
        find_strong_links,
        label_rewritten_sentences,
        split_sentences,
    )

    rewritten = "Members voiced strong reservations regarding escalating public transit expenses."
    documents = pl.DataFrame(
        {
            "id": ["a", "f", "h"],
            "label": ["ai", "fusion", "human"],
            "text": [
                "An entirely generated paragraph discusses weather patterns over the northern sea. "
                "It continues with another long sentence about rainfall and seasonal winds.",
                " ".join([SHARED[0], rewritten, SHARED[2]]),
                " ".join(SHARED),
            ],
        }
    )
    sentences = split_sentences(documents)
    links = find_strong_links(sentences)
    labels = label_rewritten_sentences(sentences, documents, links)

    assert links.select("id", "id_other").rows() == [("f", "h")]
    assert assign_families(documents["id"], links).sort("id")["family"].to_list()[1:] == [1, 1]
    fusion = labels.filter(pl.col("id") == "f").sort("position")
    assert fusion["is_rewritten"].to_list() == [False, True, False]
    assert labels.filter(pl.col("id") == "h")["is_rewritten"].to_list() == [False] * 3
    assert "a" not in labels["id"].to_list()


def test_family_folds_keep_each_family_together() -> None:
    """Catch splitting related documents across cross-fitting folds."""
    from hm3.families import assign_family_folds

    documents = pl.DataFrame(
        {"id": [f"d{i}" for i in range(30)], "label": ["human", "fusion", "ai"] * 10}
    )
    families = documents.select("id").with_columns(pl.Series("family", [i // 3 for i in range(30)]))
    folds = assign_family_folds(documents, families, 3)
    assert folds.group_by("family").agg(pl.col("family_fold").n_unique())["family_fold"].max() == 1
    assert set(folds["family_fold"]) == {0, 1, 2}


def test_aggregates_and_decisions_use_explicit_class_order() -> None:
    """Catch silent feature gaps, wrong probability columns or offset sign errors."""
    from hm3.decision import blend_probabilities, predict_with_fusion_offset
    from hm3.sentence_scores import AGGREGATE_COLUMNS, aggregate_document_scores

    scored = pl.DataFrame({"id": ["x", "x", "x"], "score": [0.1, 0.9, 0.6]})
    features = aggregate_document_scores(scored, pl.Series(["x", "empty"]))
    assert features.columns == ["id", *AGGREGATE_COLUMNS]
    first = features.row(0, named=True)
    assert np.isclose(first["max_score"], 0.9) and np.isclose(first["top2_mean_score"], 0.75)
    assert first["count_over_threshold"] == 2 and first["has_no_sentences"] == 0
    assert features.row(1, named=True)["has_no_sentences"] == 1

    probabilities = np.array([[0.5, 0.1, 0.4], [0.2, 0.7, 0.1]])
    assert predict_with_fusion_offset(probabilities, 0.0).tolist() == ["human", "ai"]
    assert predict_with_fusion_offset(probabilities, 0.5).tolist() == ["fusion", "ai"]
    tables = [
        pl.DataFrame(
            {"id": ids, "prob_human": values[0], "prob_ai": values[1], "prob_fusion": values[2]}
        )
        for ids, values in [
            (["p", "q"], [[0.6, 0.2], [0.1, 0.1], [0.3, 0.7]]),
            (["q", "p"], [[0.9, 0.0], [0.05, 0.1], [0.05, 0.9]]),
        ]
    ]
    blended = blend_probabilities(tables[0], tables[1], 0.5)
    assert blended["id"].to_list() == ["p", "q"]
    assert blended["prediction"].to_list() == ["fusion", "human"]
