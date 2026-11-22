import sys
from pathlib import Path


def main() -> None:
    """Import the separate XLM-R source bundle and execute its explicit mode."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hm3.xlmr_kaggle import main as run_main

    run_main()


if __name__ == "__main__":
    main()
