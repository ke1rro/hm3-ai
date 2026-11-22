from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import (
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
    XLMRobertaTokenizer,
)

from hm3.validation import build_manifest


@pytest.fixture
def frozen_manifest() -> pl.DataFrame:
    """Create representative frozen assignments with an indivisible duplicate pair."""
    rows = [
        {
            "id": f"{label}-{lang}-{domain}-{index}",
            "text": f"{label} {lang} {domain} {index}",
            "label": label,
            "lang": lang,
            "domain": domain,
        }
        for label in ["human", "ai", "fusion"]
        for lang in ["en-EN", "de-DE"]
        for domain in ["news", "books"]
        for index in range(15)
    ]
    rows.append({**rows[0], "id": "duplicate"})
    return build_manifest(pl.DataFrame(rows), "fixture", cv_sample_size=60)[0]


@pytest.fixture
def tiny_tokenizer() -> XLMRobertaTokenizer:
    """Build a local tokenizer with XLM-R special-token behavior and no downloads."""
    vocabulary = {
        "<s>": 0,
        "<pad>": 1,
        "</s>": 2,
        "<unk>": 3,
        "<mask>": 4,
        **{f"word{index}": index + 5 for index in range(30)},
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(
        single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)]
    )
    return XLMRobertaTokenizer(tokenizer_object=backend)


def test_encoder_ids_are_frozen_group_safe_and_holdout_identical(
    frozen_manifest: pl.DataFrame, tmp_path: Path
) -> None:
    """Catch resampling, duplicate-group crossing or holdout use in development selection."""
    from hm3.encoder_data import get_or_create_encoder_manifest

    path = tmp_path / "encoder_ids.parquet"
    first, _ = get_or_create_encoder_manifest(frozen_manifest, path, "parent", 60, 18)
    before = path.read_bytes()
    second, _ = get_or_create_encoder_manifest(frozen_manifest.reverse(), path, "parent", 60, 18)
    assert first.equals(second)
    rebuilt, _ = get_or_create_encoder_manifest(
        frozen_manifest.reverse(), tmp_path / "independent.parquet", "parent", 60, 18
    )
    assert first.equals(rebuilt)
    assert path.read_bytes() == before
    holdout = set(frozen_manifest.filter(pl.col("partition") == "validation")["id"])
    assert set(first.filter(pl.col("role") == "holdout")["id"]) == holdout
    assert set(first["role"]) == {"train", "development_eval", "holdout", "unused"}
    assert first.group_by("group_id").agg(pl.col("role").n_unique())["role"].max() == 1
    with pytest.raises(ValueError, match="incompatible"):
        get_or_create_encoder_manifest(frozen_manifest, path, "different", 60, 18)
    with pytest.raises(ValueError, match="incompatible"):
        get_or_create_encoder_manifest(frozen_manifest, path, "parent", 70, 18)


def test_head_tail_keeps_token_budget_specials_and_short_text(
    tiny_tokenizer: XLMRobertaTokenizer,
) -> None:
    """Catch off-by-two truncation, dropped suffixes and duplicated short documents."""
    from hm3.encoder_data import tokenize_document

    text = " ".join(f"word{index}" for index in range(20))
    raw = tiny_tokenizer.encode(text, add_special_tokens=False)
    head = tokenize_document(text, tiny_tokenizer, 10, "head")
    tail = tokenize_document(text, tiny_tokenizer, 10, "head_tail")
    assert head["input_ids"] == [0, *raw[:8], 2]
    assert tail["input_ids"] == [0, *raw[:4], *raw[-4:], 2]
    assert len(tail["attention_mask"]) == 10
    assert len(tokenize_document(text, tiny_tokenizer, 3, "head_tail")["input_ids"]) == 3
    assert (
        len(tokenize_document(" ".join([text] * 50), tiny_tokenizer, 512, "head_tail")["input_ids"])
        == 512
    )
    short = "word1 word2"
    assert tokenize_document(short, tiny_tokenizer, 512, "head")[
        "input_ids"
    ] == tiny_tokenizer.encode(short)
    assert tokenize_document(short, tiny_tokenizer, 512, "head_tail") == tokenize_document(
        short, tiny_tokenizer, 512, "head"
    )


def test_encoder_prediction_mapping_and_id_alignment() -> None:
    """Catch alphabetic output-head assumptions and positional prediction realignment."""
    from hm3.encoder_data import prediction_table

    result = prediction_table(
        ["b", "a"], np.array([[0.0, 5.0, 1.0], [5.0, 0.0, 1.0]]), ["fusion", "human"]
    )
    assert result["id"].to_list() == ["b", "a"]
    assert result["prediction"].to_list() == ["ai", "human"]
    assert result["label"].to_list() == ["fusion", "human"]
    assert np.allclose(
        result.select("prob_human", "prob_ai", "prob_fusion").to_numpy().sum(axis=1), 1
    )
    with pytest.raises(ValueError):
        prediction_table(["a"], np.zeros((2, 3)))


def test_xlmr_head_has_explicit_three_class_mapping(tmp_path: Path) -> None:
    """Catch a two-class or alphabetically mapped classification head."""
    from hm3.xlmr_training import load_classifier

    config = XLMRobertaConfig(
        vocab_size=40,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=24,
        id2label={0: "human", 1: "ai", 2: "fusion"},
        label2id={"human": 0, "ai": 1, "fusion": 2},
    )
    XLMRobertaForSequenceClassification(config).save_pretrained(tmp_path)
    model = load_classifier(str(tmp_path), revision=None, is_initial=True)
    output = model(
        input_ids=torch.tensor([[0, 5, 6, 2]]), attention_mask=torch.ones((1, 4), dtype=torch.long)
    )
    assert output.logits.shape == (1, 3)
    assert model.config.id2label == {0: "human", 1: "ai", 2: "fusion"}


def test_checkpoint_receipt_rejects_identity_mismatch_and_corruption(tmp_path: Path) -> None:
    """Catch restoring weights from another sampling/truncation recipe or incomplete files."""
    from hm3.xlmr_checkpoints import publish_checkpoint, verify_checkpoint

    (tmp_path / "model.safetensors").write_bytes(b"fixture weights")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "tokenizer.json").write_text("{}")
    identity: dict[str, object] = {
        "sample_sha256": "sample",
        "truncation_policy": "head",
        "labels": ["human", "ai", "fusion"],
    }
    publish_checkpoint(tmp_path, identity, is_training=False)
    verify_checkpoint(tmp_path, identity, is_training=False)
    with pytest.raises(ValueError, match="incompatible"):
        verify_checkpoint(
            tmp_path, {**identity, "truncation_policy": "head_tail"}, is_training=False
        )
    (tmp_path / "model.safetensors").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        verify_checkpoint(tmp_path, identity, is_training=False)


def test_trainer_resume_restores_optimizer_progress(
    tmp_path: Path,
    tiny_tokenizer: XLMRobertaTokenizer,
) -> None:
    """Verify real tiny CPU Trainer state can resume from a published checkpoint."""
    from transformers import (
        DataCollatorWithPadding,
        Trainer,
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )

    from hm3.encoder_data import TokenizedDocuments, tokenize_records
    from hm3.xlmr_checkpoints import verify_checkpoint
    from hm3.xlmr_training import CheckpointReceipts

    class StopAfterFirstStep(TrainerCallback):
        """Interrupt the synthetic training loop after one completed optimizer step."""

        def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs: object,
        ) -> TrainerControl:
            """Stop once to exercise the actual optimizer/RNG restore path."""
            control.should_training_stop = state.global_step == 1
            return control

    model_config = XLMRobertaConfig(
        vocab_size=40,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=24,
        id2label={0: "human", 1: "ai", 2: "fusion"},
        label2id={"human": 0, "ai": 1, "fusion": 2},
    )
    records = pl.DataFrame(
        {
            "id": [str(index) for index in range(6)],
            "text": ["word1 word2"] * 6,
            "label": ["human", "ai", "fusion"] * 2,
        }
    )
    tokens = tokenize_records(records, tiny_tokenizer, 16, "head")
    arguments = TrainingArguments(
        output_dir=str(tmp_path / "checkpoints"),
        use_cpu=True,
        max_steps=2,
        per_device_train_batch_size=2,
        save_steps=1,
        save_total_limit=2,
        report_to="none",
        disable_tqdm=True,
        optim="adamw_torch",
        seed=2026,
    )
    identity: dict[str, object] = {"fixture": "tiny-cpu-resume", "precision": "fp32"}
    trainer = Trainer(
        model=XLMRobertaForSequenceClassification(model_config),
        args=arguments,
        train_dataset=TokenizedDocuments(tokens),
        processing_class=tiny_tokenizer,
        data_collator=DataCollatorWithPadding(tiny_tokenizer),
        callbacks=[CheckpointReceipts(identity, tmp_path), StopAfterFirstStep()],
    )
    trainer.train()
    checkpoint = tmp_path / "checkpoints/checkpoint-1"
    verify_checkpoint(checkpoint, identity, True)
    restored = Trainer(
        model=XLMRobertaForSequenceClassification(model_config),
        args=arguments,
        train_dataset=TokenizedDocuments(tokens),
        processing_class=tiny_tokenizer,
        data_collator=DataCollatorWithPadding(tiny_tokenizer),
        callbacks=[CheckpointReceipts(identity, tmp_path)],
    )
    restored.train(resume_from_checkpoint=str(checkpoint))
    assert restored.state.global_step == 2
    verify_checkpoint(tmp_path / "checkpoints/checkpoint-2", identity, True)
    optimizer = torch.load(tmp_path / "checkpoints/checkpoint-2/optimizer.pt", weights_only=True)
    assert {int(value["step"]) for value in optimizer["state"].values()} == {2}


def test_resume_rejects_corrupted_best_and_recognizes_terminal_state(tmp_path: Path) -> None:
    """Prevent selecting unverified best weights or optimizing after exhausted patience."""
    import json

    from hm3.xlmr_checkpoints import publish_checkpoint
    from hm3.xlmr_training import checkpoint_is_terminal, find_resume_checkpoint

    identity: dict[str, object] = {"fixture": "resume-best", "precision": "fp32"}
    best = tmp_path / "checkpoints/checkpoint-1"
    latest = tmp_path / "checkpoints/checkpoint-2"
    for path, step in [(best, 1), (latest, 2)]:
        path.mkdir(parents=True)
        for name in [
            "config.json",
            "tokenizer.json",
            "model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
        ]:
            (path / name).write_text("fixture")
        state = {
            "global_step": step,
            "max_steps": 10,
            "best_model_checkpoint": str(best),
            "stateful_callbacks": {
                "EarlyStoppingCallback": {
                    "args": {"early_stopping_patience": 2},
                    "attributes": {"early_stopping_patience_counter": 2},
                }
            },
        }
        (path / "trainer_state.json").write_text(json.dumps(state))
        publish_checkpoint(path, identity, True)
    assert checkpoint_is_terminal(latest, is_refit=False)
    assert not checkpoint_is_terminal(latest, is_refit=True)
    assert find_resume_checkpoint(tmp_path, identity) == latest
    (best / "model.safetensors").write_text("corrupted best")
    with pytest.raises(ValueError, match="checksum"):
        find_resume_checkpoint(tmp_path, identity)


def test_attached_run_environment_mismatch_does_not_restart_silently(tmp_path: Path) -> None:
    """Require an explicit recovery decision when saved training uses a different runtime."""
    import json

    from hm3.xlmr_kaggle import _restore_run

    saved = tmp_path / "input/prior_run"
    saved.mkdir(parents=True)
    identity: dict[str, object] = {
        "configuration": {"experiment": "xlmr_head_only_40k"},
        "scope": "sampled",
        "environment": {"gpu": "T4"},
        "tokenizer_sha256": "same",
    }
    (saved / "run_metadata.json").write_text(json.dumps({"identity": identity}))
    changed = {**identity, "environment": {"gpu": "P100"}}
    with pytest.raises(ValueError, match="runtime"):
        _restore_run(tmp_path / "input", tmp_path / "work/run", changed)


def test_xlmr_gradient_accumulation_normalizes_mean_document_loss(tmp_path: Path) -> None:
    """Prevent Trainer from treating XLM-R mean cross-entropy as an accumulation sum."""
    from transformers import Trainer, TrainingArguments

    from hm3.xlmr_training import load_classifier

    config = XLMRobertaConfig(
        vocab_size=40,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=24,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        id2label={0: "human", 1: "ai", 2: "fusion"},
        label2id={"human": 0, "ai": 1, "fusion": 2},
    )
    XLMRobertaForSequenceClassification(config).save_pretrained(tmp_path / "model")
    model = load_classifier(str(tmp_path / "model"), None, False)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path / "output"),
            use_cpu=True,
            gradient_accumulation_steps=8,
            report_to="none",
        ),
    )
    trainer.current_gradient_accumulation_steps = 8
    inputs = {
        "input_ids": torch.tensor([[0, 5, 6, 2], [0, 7, 8, 2]]),
        "labels": torch.tensor([0, 2]),
    }
    expected = model(**inputs).loss.detach() / 8
    actual = trainer.training_step(model, {**inputs}, num_items_in_batch=torch.tensor(16))
    assert torch.allclose(actual, expected)
