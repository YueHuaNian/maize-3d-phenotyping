"""CLI runner for maize_leaf modeling and completion."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / ".vendor"), str(ROOT / "src")]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

from maize_leaf.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
