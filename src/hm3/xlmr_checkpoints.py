import json
from pathlib import Path

from hm3.validation import fingerprint_file


def publish_checkpoint(path: Path, identity: dict[str, object], is_training: bool) -> None:
    """Publish an atomic completion receipt after all checkpoint files exist.

    Args:
        path: Completed Trainer checkpoint or saved inference-model directory.
        identity: Exact data, model, tokenizer, recipe and environment identity.
        is_training: Require optimizer, scheduler, RNG and Trainer state for resume.
    """
    required = ["config.json", "tokenizer.json"]
    if is_training:
        required += ["optimizer.pt", "scheduler.pt", "trainer_state.json", "rng_state.pth"]
        if identity.get("precision") == "fp16":
            required.append("scaler.pt")
    if not all((path / name).is_file() for name in required) or not any(
        (path / name).is_file() for name in ["model.safetensors", "pytorch_model.bin"]
    ):
        raise ValueError("Checkpoint is incomplete and cannot be published")
    files = {
        str(item.relative_to(path)): fingerprint_file(item)
        for item in path.rglob("*")
        if item.is_file()
        and item.name != "checkpoint_manifest.json"
        and not item.name.endswith(".tmp")
    }
    temporary = path / "checkpoint_manifest.json.tmp"
    temporary.write_text(
        json.dumps({"identity": identity, "is_training": is_training, "files": files}, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path / "checkpoint_manifest.json")


def verify_checkpoint(
    path: Path, identity: dict[str, object], is_training: bool
) -> dict[str, object]:
    """Verify a published checkpoint before restoring weights or training state.

    Args:
        path: Completed checkpoint directory.
        identity: Required complete experiment identity.
        is_training: Require a training-resume checkpoint rather than inference weights only.

    Returns:
        Verified completion receipt.
    """
    record = json.loads((path / "checkpoint_manifest.json").read_text())
    if record["identity"] != identity or record["is_training"] != is_training:
        raise ValueError("Checkpoint is incompatible with the experiment identity")
    for name, expected in record["files"].items():
        file_path = (path / name).resolve()
        if not file_path.is_relative_to(path.resolve()) or fingerprint_file(file_path) != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    return record
