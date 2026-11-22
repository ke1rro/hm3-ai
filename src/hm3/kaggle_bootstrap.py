import hashlib
import json
import logging
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)
UV_VERSION = "0.13.9"
UV_LINUX_SHA256 = "a7fbaff2b9a271d85c1bdb540941090fd1ce011797b4abeb48e4dcd39c2471d5"


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_bundle(root: Path, kind: str) -> dict[str, object]:
    """Verify every declared file in a private source or split bundle.

    Args:
        root: Unpacked bundle root.
        kind: Expected code or split bundle kind.

    Returns:
        Verified bundle manifest.
    """
    record = json.loads((root / "hm3_bundle.json").read_text())
    if record["bundle_kind"] != kind:
        raise ValueError("Unexpected bundle kind")
    for name, expected in record["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or _sha256(path) != expected:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    return record


def find_bundle(input_root: Path, kind: str) -> Path:
    """Find exactly one unpacked private bundle without assuming a Kaggle slug.

    Args:
        input_root: Root of attached Kaggle inputs.
        kind: Required code or split bundle kind.

    Returns:
        Verified bundle root.
    """
    roots = [
        path.parent
        for path in input_root.rglob("hm3_bundle.json")
        if json.loads(path.read_text()).get("bundle_kind") == kind
    ]
    if len(roots) != 1:
        raise ValueError(f"Expected one {kind} bundle, found {len(roots)}; attach one version")
    verify_bundle(roots[0], kind)
    return roots[0]


def find_competition_inputs(input_root: Path, source_sha256: str) -> dict[str, Path]:
    """Locate attached competition files and verify the labeled source identity.

    Args:
        input_root: Attached Kaggle inputs.
        source_sha256: Expected original train.csv fingerprint.

    Returns:
        Verified train path and neighboring test and sample paths.
    """
    matches = []
    for train in input_root.rglob("train.csv"):
        test = train.parent / "test.csv"
        sample = train.parent / "sample_submission.csv"
        if test.is_file() and sample.is_file() and _sha256(train) == source_sha256:
            matches.append({"train": train, "test": test, "sample": sample})
    if len(matches) != 1:
        raise ValueError("Expected one competition input matching the frozen training fingerprint")
    return matches[0]


def _find_uv(temporary_root: Path) -> Path:
    existing = shutil.which("uv")
    if existing:
        return Path(existing)
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ValueError("Install uv before bootstrap outside the Kaggle Linux x86_64 runtime")
    destination = temporary_root / f"uv-{UV_VERSION}"
    destination.mkdir(parents=True, exist_ok=True)
    executable = destination / "uv"
    if executable.exists():
        return executable
    archive = destination / "uv.tar.gz"
    url = (
        f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/"
        "uv-x86_64-unknown-linux-gnu.tar.gz"
    )
    urllib.request.urlretrieve(url, archive)
    if _sha256(archive) != UV_LINUX_SHA256:
        raise ValueError("uv standalone archive checksum mismatch")
    with tarfile.open(archive) as package:
        stream = package.extractfile("uv-x86_64-unknown-linux-gnu/uv")
        if stream is None:
            raise ValueError("uv executable is missing from its standalone archive")
        with stream, executable.open("wb") as output:
            shutil.copyfileobj(stream, output)
    executable.chmod(0o755)
    archive.unlink()
    return executable


def prepare_runtime(code_root: Path, temporary_root: Path) -> Path:
    """Install pinned CPU dependencies into an isolated hosted Python environment.

    Args:
        code_root: Verified source bundle with its compiled requirements.
        temporary_root: Ephemeral setup/cache directory outside saved notebook outputs.

    Returns:
        Isolated Python executable, leaving kernel, CUDA and PyTorch packages unchanged.
    """
    if not (3, 12) <= sys.version_info[:2] < (3, 15):
        raise ValueError("The hosted requirements support Python 3.12 through 3.14")
    requirements = code_root / "config/requirements_kaggle.txt"
    identity = {"requirements_sha256": _sha256(requirements), "python": platform.python_version()}
    environment = (
        temporary_root / f"env-{identity['requirements_sha256'][:12]}-{identity['python']}"
    )
    python = environment / "bin/python"
    marker = environment / "hm3_runtime.json"
    if marker.exists():
        if json.loads(marker.read_text()) != identity or not python.is_file():
            raise ValueError("Existing hosted environment has incompatible setup metadata")
        return python
    temporary_root.mkdir(parents=True, exist_ok=True)
    uv = _find_uv(temporary_root)
    subprocess.run(
        [str(uv), "venv", "--allow-existing", "--python", sys.executable, str(environment)],
        check=True,
    )
    subprocess.run(
        [
            str(uv),
            "--cache-dir",
            str(temporary_root / "uv-cache"),
            "pip",
            "install",
            "--python",
            str(python),
            "--require-hashes",
            "--only-binary",
            ":all:",
            "--requirement",
            str(requirements),
        ],
        check=True,
    )
    subprocess.run([str(uv), "pip", "check", "--python", str(python)], check=True)
    marker.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    logger.info("Prepared isolated environment at %s", environment)
    return python
