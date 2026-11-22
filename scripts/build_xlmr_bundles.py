import argparse
import hashlib
import json
import logging
import sys
import tomllib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

logger = logging.getLogger(__name__)
CODE_FILES = [
    *[
        f"src/hm3/{name}.py"
        for name in [
            "__init__",
            "validation",
            "evaluation",
            "kaggle_bootstrap",
            "encoder_data",
            "xlmr_checkpoints",
            "xlmr_training",
            "xlmr_bootstrap",
            "xlmr_kaggle",
        ]
    ],
    "scripts/run_xlmr_kaggle.py",
    "config/xlmr.toml",
    "config/requirements_xlmr.in",
    "config/requirements_xlmr.txt",
    "experiments/02_xlm_roberta_kaggle.ipynb",
    "experiments/XLMR_README.md",
]
SPLIT_FILES = [
    f"data/metadata/{name}.{suffix}"
    for name in ["validation_manifest", "xlmr_development_40k"]
    for suffix in ["parquet", "json"]
]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_bundle(root: Path, destination: Path, kind: str, names: list[str]) -> None:
    prefix = f"hm3_{kind}"
    record = {"bundle_kind": kind, "files": {name: _sha256(root / name) for name in names}}
    temporary = destination.with_suffix(".zip.tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(root / name, f"{prefix}/{name}")
        archive.writestr(f"{prefix}/hm3_bundle.json", json.dumps(record, indent=2))
    with ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise ValueError("XLM-R bundle archive integrity check failed")
    temporary.replace(destination)
    logger.info(
        "Created %s (%s bytes), SHA-256 %s",
        destination,
        destination.stat().st_size,
        _sha256(destination),
    )


def main() -> None:
    """Build separate private XLM-R archives without rewriting the active baseline bundles."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("dist/xlmr_kaggle"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = tomllib.loads((root / "config/xlmr.toml").read_text())
    frozen = json.loads((root / "data/metadata/validation_manifest.json").read_text())
    selected = json.loads((root / "data/metadata/xlmr_development_40k.json").read_text())
    if (
        _sha256(root / SPLIT_FILES[0]) != config["frozen_manifest_sha256"]
        or _sha256(root / SPLIT_FILES[2]) != config["encoder_manifest_sha256"]
        or frozen["source_sha256"] != config["source_sha256"]
        or selected["parent_manifest_sha256"] != config["frozen_manifest_sha256"]
        or selected["seed"] != config["seed"]
        or selected["train_target"] != config["train_target"]
        or selected["evaluation_target"] != config["evaluation_target"]
    ):
        raise ValueError("XLM-R recipe and frozen source/sampling manifests are incompatible")
    args.output.mkdir(parents=True, exist_ok=True)
    _write_bundle(root, args.output / "xlmr_code.zip", "xlmr_code", CODE_FILES)
    _write_bundle(root, args.output / "xlmr_split.zip", "xlmr_split", SPLIT_FILES)


if __name__ == "__main__":
    main()
