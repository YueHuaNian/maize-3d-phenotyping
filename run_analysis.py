"""Analyse maize point clouds using the V1.1 nearest-neighbor feature pipeline."""

import argparse
from datetime import datetime
import os
from pathlib import Path

from src.io_utils import locate_data

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version",
                        version=(ROOT / "VERSION").read_text(encoding="utf-8").strip())
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--batch", action="store_true", help="analyse all input pairs (default)")
    selection.add_argument("--sample", help="analyse one plant by filename stem")
    parser.add_argument("--data-dir", type=Path, help="semantic CSV directory")
    parser.add_argument("--txt-dir", type=Path, help="original TXT directory")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output-dir", type=Path, help="new directory for this run")
    destination.add_argument("--results-dir", type=Path, help="parent for timestamped runs")
    args = parser.parse_args()
    try:
        csv_dir = locate_data(args.data_dir)
        configured_txt = os.environ.get("MAIZE_TXT_DIR")
        txt_dir = args.txt_dir or (
            Path(configured_txt) if configured_txt else csv_dir.parent / "0709单天多个材料原始数据"
        )
        if not txt_dir.is_dir():
            raise FileNotFoundError(f"TXT input directory not found: {txt_dir}")
        output = args.output_dir or (args.results_dir or ROOT / "results") / "runs" / (
            f"v1.1_{datetime.now():%Y%m%d_%H%M%S_%f}"
        )
        from src.phenotype_pipeline import run_analysis

        return run_analysis(csv_dir, txt_dir, output, sample=args.sample)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
