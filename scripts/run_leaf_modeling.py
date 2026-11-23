"""Project-local CLI; dependencies may be installed normally or in .vendor."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / ".vendor"), str(ROOT / "src")]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("PYTHONUTF8", "1")
from maize_leaf_v1.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
