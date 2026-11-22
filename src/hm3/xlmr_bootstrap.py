import hashlib
import json
import logging
import re
import subprocess
import sys
from pathlib import Path

from hm3.kaggle_bootstrap import _find_uv

logger = logging.getLogger(__name__)


def _fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _probe(python: Path, has_auxiliary: bool) -> dict:
    imports = "import torch, json, site, sys; "
    if has_auxiliary:
        imports += "from transformers import Trainer, AutoTokenizer; import accelerate; "
    code = imports + (
        "sys.stdout.write(json.dumps({'torch': str(torch.__version__), 'cuda': torch.version.cuda, "
        "'torch_file': torch.__file__, 'site_packages': site.getsitepackages()}))"
    )
    result = subprocess.run(
        [str(python), "-I", "-c", code], check=True, text=True, capture_output=True
    )
    return json.loads(result.stdout)


def prepare_xlmr_runtime(code_root: Path, temporary_root: Path) -> Path:
    """Install only auxiliary dependencies while retaining the hosted Torch/CUDA stack.

    Args:
        code_root: Verified XLM-R source bundle.
        temporary_root: Separate ephemeral environment/cache location.

    Returns:
        Python executable whose imported Torch build matches the original kernel stack.
    """
    if not (3, 12) <= sys.version_info[:2] < (3, 15):
        raise ValueError("XLM-R bootstrap supports hosted Python 3.12 through 3.14")
    requirements = code_root / "config/requirements_xlmr.txt"
    if re.search(
        r"^(?:torch|torchvision|torchaudio|triton|nvidia[^= ]*)==",
        requirements.read_text(),
        re.MULTILINE,
    ):
        raise ValueError("Auxiliary requirements must not install a different Torch/CUDA stack")
    original = _probe(Path(sys.executable), False)
    torch_version = tuple(int(part) for part in original["torch"].split("+")[0].split(".")[:2])
    if torch_version < (2, 6):
        raise ValueError(
            "Select a Kaggle image with native PyTorch >=2.6 for compatible state loading"
        )
    identity = {
        "requirements_sha256": _fingerprint(requirements),
        "python": sys.version,
        "torch": original,
    }
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    environment = temporary_root / f"env-{identity_hash[:12]}"
    python = environment / "bin/python"
    marker = environment / "xlmr_runtime.json"
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError("Existing XLM-R runtime has incompatible setup provenance")
        _probe(python, True)
        return python
    temporary_root.mkdir(parents=True, exist_ok=True)
    uv = _find_uv(temporary_root)
    subprocess.run(
        [
            str(uv),
            "venv",
            "--allow-existing",
            "--system-site-packages",
            "--python",
            sys.executable,
            str(environment),
        ],
        check=True,
    )
    query = subprocess.run(
        [str(python), "-I", "-c", "import site,sys;sys.stdout.write(site.getsitepackages()[0])"],
        check=True,
        text=True,
        capture_output=True,
    )
    site_path = Path(query.stdout)
    site_path.mkdir(parents=True, exist_ok=True)
    (site_path / "xlmr_host_stack.pth").write_text("\n".join(original["site_packages"]) + "\n")
    subprocess.run(
        [
            str(uv),
            "--cache-dir",
            str(temporary_root / "uv-cache"),
            "pip",
            "install",
            "--python",
            str(python),
            "--no-deps",
            "--require-hashes",
            "--only-binary",
            ":all:",
            "--requirement",
            str(requirements),
        ],
        check=True,
    )
    restored = _probe(python, True)
    if any(restored[key] != original[key] for key in ["torch", "cuda", "torch_file"]):
        raise ValueError("Auxiliary setup changed the native Torch/CUDA build")
    marker.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    logger.info(
        "Prepared XLM-R environment; native Torch %s, CUDA %s retained",
        original["torch"],
        original["cuda"],
    )
    return python
