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
            "baseline",
            "baseline_export",
            "evaluation",
            "validation",
            "kaggle_bootstrap",
            "kaggle_run",
        ]
    ],
    "scripts/run_kaggle_baseline.py",
    "scripts/run_bounded_submission.py",
    "config/baseline_kaggle.toml",
    "config/requirements_kaggle.in",
    "config/requirements_kaggle.txt",
    "experiments/01_baseline_kaggle.ipynb",
    "experiments/README.md",
]
SPLIT_FILES = [
    "data/metadata/validation_manifest.parquet",
    "data/metadata/validation_manifest.json",
]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_bundle(root: Path, output: Path, kind: str, names: list[str]) -> None:
    prefix = f"hm3_baseline_{kind}"
    record = {"bundle_kind": kind, "files": {name: _sha256(root / name) for name in names}}
    temporary = output.with_suffix(".zip.tmp")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as archive:
        for name in names:
            archive.write(root / name, f"{prefix}/{name}")
        archive.writestr(f"{prefix}/hm3_bundle.json", json.dumps(record, indent=2))
    with ZipFile(temporary) as archive:
        invalid = archive.testzip()
        if invalid is not None:
            raise ValueError(f"Archive failed verification: {invalid}")
    temporary.replace(output)
    logger.info("Created %s (%s bytes), SHA-256 %s", output, output.stat().st_size, _sha256(output))


def main() -> None:
    """Build separate upload-ready private code and frozen-split ZIP archives."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("dist/kaggle"))
    args = parser.parse_args()
    root = args.root.resolve()
    config = tomllib.loads((root / "config/baseline_kaggle.toml").read_text())
    metadata = json.loads((root / SPLIT_FILES[1]).read_text())
    if (
        metadata["manifest_sha256"] != _sha256(root / SPLIT_FILES[0])
        or metadata["manifest_sha256"] != config["manifest_sha256"]
        or metadata["source_sha256"] != config["source_sha256"]
        or metadata["seed"] != config["seed"]
        or metadata["cv_sample_target"] != config["cv_sample_target"]
    ):
        raise ValueError("Frozen split and hosted configuration are incompatible")
    args.output.mkdir(parents=True, exist_ok=True)
    _write_bundle(root, args.output / "classical_baseline_code.zip", "code", CODE_FILES)
    _write_bundle(root, args.output / "classical_baseline_split.zip", "split", SPLIT_FILES)


if __name__ == "__main__":
    main()
