import argparse
import json
import logging
import sys
from pathlib import Path

import plotly.express as px
import plotly.graph_objects as go
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hm3.validation import LABELS  # noqa: E402

logger = logging.getLogger("report_figures")
RESULTS = ROOT / "runs/rewrite_experiments"
CLASS_COLORS = {"human": "#2A9D8F", "ai": "#E76F51", "fusion": "#577590"}
LAYOUT = {"template": "plotly_white", "font": {"size": 14}, "margin": {"t": 70}}
KAGGLE_PRIVATE_MACRO_F1 = {
    "xlmr_head_only_40k_sampled.csv": 0.81536,
    "baseline_word_char_logreg_all_train.csv": 0.89524,
    "xlmr_head_tail_40k_sampled.csv": 0.90067,
    "word_char_logreg_sentence_rewrite_all_train.csv": 0.90084,
    "word_char_logreg_xlmr_sentence_rewrite_all_train.csv": 0.91950,
}


def _metrics(stage: str) -> dict:
    return json.loads((RESULTS / stage / "metrics.json").read_text())


def _write(figure: go.Figure, output: Path, width: int = 1000, height: int = 520) -> None:
    figure.update_layout(**LAYOUT)
    figure.write_image(output, width=width, height=height, scale=2)
    logger.info("Wrote %s", output)


def plot_class_share_by_domain(train: Path, output: Path) -> None:
    """Plot the class share of every language and domain in the training data.

    Args:
        train: Path to the competition training CSV.
        output: Destination PNG path.
    """
    shares = (
        pl.read_csv(train, columns=["label", "lang", "domain"])
        .group_by("lang", "domain", "label")
        .len()
        .with_columns(
            (pl.col("len") / pl.col("len").sum().over("lang", "domain")).alias("share"),
            pl.col("len").sum().over("lang", "domain").alias("documents"),
            pl.format("{} ({})", "domain", pl.col("lang").str.slice(0, 2)).alias("group"),
        )
        .sort("lang", "domain", "label")
    )
    figure = px.bar(
        shares,
        x="group",
        y="share",
        color="label",
        hover_data=["documents"],
        color_discrete_map=CLASS_COLORS,
        category_orders={"label": list(LABELS)},
        title="Class share by domain and language in train.csv",
        labels={"group": "Domain (language)", "share": "Share of documents", "label": "Class"},
    )
    figure.update_xaxes(tickangle=-30)
    _write(figure, output, height=560)


def plot_holdout_changes(output: Path) -> None:
    """Plot paired holdout macro-F1 differences with family-cluster 95% intervals.

    Args:
        output: Destination PNG path.
    """
    sentence = _metrics("sentence_holdout")["candidate"]["paired_holdout"]
    offset = _metrics("offset")["schemes"]["family_grouped_folds"]["paired_holdout"]
    blend = _metrics("blend")["paired_holdout"]
    transformer = _metrics("transformer_holdout")["evaluations"]
    rows = [
        ("Character sentence scores vs TF-IDF reference", sentence),
        ("Fusion decision offset vs TF-IDF reference", offset),
        ("Equal-weight blend with XLM-R vs TF-IDF reference", blend),
        (
            "Character plus XLM-R sentence scores vs character only",
            transformer["candidate_char_transformer"]["paired_holdout"],
        ),
        (
            "XLM-R sentence scores vs character only",
            transformer["ablation_transformer"]["paired_holdout"],
        ),
    ]
    changes = pl.DataFrame(
        [
            {
                "comparison": name,
                "difference": values["difference"],
                "error_low": values["difference"] - values["ci95_low"],
                "error_high": values["ci95_high"] - values["difference"],
            }
            for name, values in rows
        ]
    )
    figure = px.scatter(
        changes,
        x="difference",
        y="comparison",
        error_x="error_high",
        error_x_minus="error_low",
        title="Change in fixed-holdout macro-F1 with 95% family-cluster bootstrap interval",
        labels={"difference": "Change in holdout macro-F1", "comparison": "Comparison"},
    )
    figure.add_vline(x=0, line_dash="dash", line_color="#666666")
    figure.update_traces(marker={"size": 11, "color": "#1F3A5F"})
    figure.update_yaxes(autorange="reversed")
    _write(figure, output, width=1100, height=480)


def plot_fusion_recall_by_rewritten_sentences(output: Path) -> None:
    """Plot holdout fusion recall by the number of rewritten eligible sentences.

    Args:
        output: Destination PNG path.
    """
    recall = (
        pl.read_parquet(RESULTS / "sentence_holdout/fusion_recall_by_rewritten_sentences.parquet")
        .with_columns(
            pl.col("rewritten_sentences_capped").replace({"4": "4 or more"}),
        )
        .unpivot(
            index=["rewritten_sentences_capped", "fusion_documents"],
            on=["reference_recall", "candidate_recall"],
            variable_name="model",
            value_name="recall",
        )
        .with_columns(
            pl.col("model").replace(
                {
                    "reference_recall": "TF-IDF reference",
                    "candidate_recall": "Plus character sentence scores",
                }
            )
        )
    )
    figure = px.bar(
        recall,
        x="rewritten_sentences_capped",
        y="recall",
        color="model",
        barmode="group",
        text_auto=".2f",
        hover_data=["fusion_documents"],
        category_orders={
            "rewritten_sentences_capped": ["0", "1", "2", "3", "4 or more", "no linked human"]
        },
        color_discrete_sequence=["#9AA5B1", "#1F3A5F"],
        title="Holdout fusion recall by number of rewritten sentences",
        labels={
            "rewritten_sentences_capped": "Rewritten sentences with at least eight words",
            "recall": "Fusion recall",
            "model": "Model",
        },
    )
    _write(figure, output)


def plot_xlmr_length_buckets(output: Path) -> None:
    """Plot holdout macro-F1 by document length for the XLM-R truncation policies.

    Args:
        output: Destination PNG path.
    """
    bucket_names = {
        "001-128": "1–128",
        "129-512": "129–512",
        "513-1024": "513–1,024",
        "1025+": "over 1,024",
    }
    model_names = {
        "word_char_tfidf_reference": "Word and character TF-IDF reference",
        "head_only": "XLM-R head-only",
        "head_tail": "XLM-R head plus tail",
    }
    lengths = (
        pl.read_parquet(RESULTS / "xlmr_long_text/length_bucket_macro_f1.parquet")
        .filter(pl.col("model").is_in(list(model_names)))
        .with_columns(
            pl.format(
                "{}<br>{} documents",
                pl.col("length_bucket").replace_strict(bucket_names),
                pl.col("support").cast(pl.String).str.replace(r"(\d)(\d{3})$", "$1,$2"),
            ).alias("bucket"),
            pl.col("model").replace_strict(model_names),
        )
        .sort(pl.col("length_bucket").replace_strict(list(bucket_names), range(4)))
    )
    figure = px.bar(
        lengths,
        x="bucket",
        y="macro_f1",
        color="model",
        barmode="group",
        text_auto=".3f",
        category_orders={"model": list(model_names.values())},
        color_discrete_sequence=["#9AA5B1", "#8FB8DE", "#1F3A5F"],
        title="Fixed-holdout macro-F1 by document length",
        labels={
            "bucket": "Document length in words",
            "macro_f1": "Holdout macro-F1",
            "model": "Model",
        },
    )
    figure.update_traces(textangle=0, textposition="outside", textfont_size=12)
    figure.update_yaxes(range=[0, 1])
    _write(figure, output)


def plot_final_confusion(output: Path) -> None:
    """Plot the row-normalized holdout confusion matrix of the selected model.

    Args:
        output: Destination PNG path.
    """
    evaluation = _metrics("transformer_holdout")["evaluations"]["candidate_char_transformer"]
    figure = px.imshow(
        evaluation["holdout"]["row_normalized_confusion_matrix"],
        x=list(LABELS),
        y=list(LABELS),
        text_auto=".3f",
        color_continuous_scale="Blues",
        title="TF-IDF with character and XLM-R sentence scores: holdout confusion",
        labels={"x": "Predicted class", "y": "True class", "color": "Share of true class"},
    )
    _write(figure, output, width=760, height=600)


def plot_model_comparison(output: Path) -> None:
    """Plot holdout, family-clean and Kaggle private macro-F1 for every submitted model.

    Args:
        output: Destination PNG path.
    """
    long_text = _metrics("xlmr_long_text")["evaluations"]
    models = [
        (
            "XLM-R head-only (40k)",
            long_text["xlmr_head_only_original_accumulation"],
            "xlmr_head_only_40k_sampled.csv",
        ),
        (
            "Word + character TF-IDF",
            long_text["word_char_tfidf_reference"],
            "baseline_word_char_logreg_all_train.csv",
        ),
        (
            "XLM-R head plus tail (40k)",
            long_text["xlmr_head_tail"],
            "xlmr_head_tail_40k_sampled.csv",
        ),
        (
            "Plus character sentence scores",
            _metrics("sentence_holdout")["candidate"],
            "word_char_logreg_sentence_rewrite_all_train.csv",
        ),
        (
            "Plus character and XLM-R sentence scores",
            _metrics("transformer_holdout")["evaluations"]["candidate_char_transformer"],
            "word_char_logreg_xlmr_sentence_rewrite_all_train.csv",
        ),
    ]
    offsets = {
        "Fixed holdout": 0.2,
        "Family-clean holdout": 0.0,
        "Kaggle private leaderboard": -0.2,
    }
    scores = pl.DataFrame(
        [
            {
                "model": name,
                "evaluation": evaluation,
                "macro_f1": value,
                "position": index + offsets[evaluation],
            }
            for index, (name, metrics, submission) in enumerate(models)
            for evaluation, value in [
                ("Fixed holdout", metrics["holdout"]["macro_f1"]),
                ("Family-clean holdout", metrics["family_clean"]["macro_f1"]),
                ("Kaggle private leaderboard", KAGGLE_PRIVATE_MACRO_F1[submission]),
            ]
        ]
    )
    figure = px.scatter(
        scores,
        x="macro_f1",
        y="position",
        color="evaluation",
        symbol="evaluation",
        hover_data={"model": True, "macro_f1": ":.4f", "position": False},
        color_discrete_sequence=["#1F3A5F", "#8FB8DE", "#E9A23B"],
        title="Macro-F1 of the submitted models on local and Kaggle evaluations",
        labels={"macro_f1": "Macro-F1", "position": "Model", "evaluation": "Evaluation"},
    )
    figure.update_traces(marker={"size": 12})
    figure.update_yaxes(
        tickvals=list(range(len(models))),
        ticktext=[name for name, _, _ in models],
        showgrid=False,
        zeroline=False,
    )
    _write(figure, output, width=1100, height=480)


def main() -> None:
    """Build every report figure from saved experiment results."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    parser = argparse.ArgumentParser(description="Build report figures from saved results")
    parser.add_argument("--train", type=Path, default=ROOT / "data/train.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "images")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plot_class_share_by_domain(args.train, args.output / "class_share_by_domain.png")
    plot_holdout_changes(args.output / "holdout_macro_f1_changes.png")
    plot_fusion_recall_by_rewritten_sentences(
        args.output / "fusion_recall_by_rewritten_sentences.png"
    )
    plot_xlmr_length_buckets(args.output / "xlmr_macro_f1_by_length.png")
    plot_final_confusion(args.output / "final_model_confusion.png")
    plot_model_comparison(args.output / "model_comparison_macro_f1.png")


if __name__ == "__main__":
    main()
