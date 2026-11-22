import json
from pathlib import Path

import pytest

from hm3.validation import fingerprint_file


def test_private_bundle_discovery_checks_integrity(tmp_path: Path) -> None:
    """Catch selecting an arbitrary bundle or executing modified uploaded code."""
    from hm3.kaggle_bootstrap import find_bundle, verify_bundle

    root = tmp_path / "random-user-dataset" / "code"
    root.mkdir(parents=True)
    source = root / "example.py"
    source.write_text("value = 2026\n")
    (root / "hm3_bundle.json").write_text(
        json.dumps({"bundle_kind": "code", "files": {"example.py": fingerprint_file(source)}})
    )
    assert find_bundle(tmp_path, "code") == root
    verify_bundle(root, "code")
    source.write_text("value = 0\n")
    with pytest.raises(ValueError, match="checksum"):
        verify_bundle(root, "code")


def test_competition_discovery_requires_verified_train_identity(tmp_path: Path) -> None:
    """Catch using another attached competition dataset with similar filenames."""
    from hm3.kaggle_bootstrap import find_competition_inputs

    source = tmp_path / "unpredictable-competition-slug" / "nested"
    source.mkdir(parents=True)
    for name in ["train.csv", "test.csv", "sample_submission.csv"]:
        (source / name).write_text("synthetic fixture")
    result = find_competition_inputs(tmp_path, fingerprint_file(source / "train.csv"))
    assert result["train"] == source / "train.csv"
    with pytest.raises(ValueError, match="fingerprint"):
        find_competition_inputs(tmp_path, "wrong")


def test_saved_output_restore_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Catch permanently accepting a partially copied saved Kaggle run."""
    from hm3 import kaggle_run

    source = tmp_path / "input/previous/runs/classical_baseline/recipe"
    source.mkdir(parents=True)
    (source / "run_metadata.json").write_text(json.dumps({"recipe_sha256": "recipe"}))
    (source / "model.joblib").write_bytes(b"synthetic checkpoint")
    destination = tmp_path / "output/runs/classical_baseline/recipe"
    original = kaggle_run.shutil.copytree

    def interrupt_copy(source_path: Path, destination_path: Path) -> None:
        destination_path.mkdir(parents=True)
        (destination_path / "run_metadata.json").write_text("partial")
        raise OSError("Simulated interrupted copy")

    monkeypatch.setattr(kaggle_run.shutil, "copytree", interrupt_copy)
    with pytest.raises(OSError, match="interrupted"):
        kaggle_run._restore_saved_run(tmp_path / "input", destination, "recipe")
    assert not destination.exists()
    monkeypatch.setattr(kaggle_run.shutil, "copytree", original)
    kaggle_run._restore_saved_run(tmp_path / "input", destination, "recipe")
    assert (destination / "model.joblib").read_bytes() == b"synthetic checkpoint"
