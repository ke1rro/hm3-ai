import sys
from pathlib import Path


def main() -> None:
    """Make the unpacked source importable and run the hosted baseline entry point."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from hm3.kaggle_run import main as run_main

    run_main()


if __name__ == "__main__":
    main()
