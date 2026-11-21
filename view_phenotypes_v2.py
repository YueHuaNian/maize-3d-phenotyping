"""Open the V2 height-ordered per-plant morphotype viewer."""

import argparse
import os
from pathlib import Path

from src.io_utils import locate_data
from src.plant_morphotypes_v2 import LAYERS, SCOPE, classify_results, export_results
from src.viewer_indices import read_index_results
from view_phenotypes import make_viewer, read_results

ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "outputs" / "v1.1"
DEFAULT_INDICES = ROOT / "outputs" / "pai-lmi-baseline"


def prepare_view(source, middle_size="majority"):
    inputs, leaves, _ = read_results(source, "plant")
    rows, summaries = classify_results(source, leaves, middle_size=middle_size)
    for row in rows:
        code = row["plant_cluster_id"] or ""
        name = f"{row['height_layer']}{row['morphotype']}" if code in LAYERS else ""
        leaves[row["plant_id"]][row["leaf_id"]].update(cluster_id=code, morphotype=name)
    policy = "middle majority" if middle_size == "majority" else "free middle size"
    return (inputs, leaves, f"{SCOPE} · {policy}"), rows, summaries


def load_view_indices(source, leaves, directory=None):
    if directory is None:
        directory = source / "indices"
        if not directory.exists() and source == DEFAULT_SOURCE:
            directory = DEFAULT_INDICES
        if not directory.exists():
            return {}, {}
    return read_index_results(directory, source, leaves)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", help="plant to display first")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--indices-dir", type=Path, help="PAI/LMI tables for the same analysis run")
    parser.add_argument("--data-dir", type=Path, help="semantic CSV directory")
    parser.add_argument("--txt-dir", type=Path, help="matching original TXT directory")
    parser.add_argument("--export-dir", type=Path, help="new directory for V2 tables and report")
    parser.add_argument("--middle-size", choices=("majority", "free"), default="majority",
                        help="require a middle-layer majority where feasible (default), or use free cuts")
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--export-only", action="store_true", help="write results without opening the viewer")
    display.add_argument("--save", type=Path, help="save a static preview without opening a window")
    args = parser.parse_args()
    if args.export_only and args.export_dir is None:
        parser.error("--export-only requires --export-dir")
    args.results_dir = args.results_dir.resolve()

    try:
        prepared, rows, summaries = prepare_view(args.results_dir, args.middle_size)
        if args.export_dir is not None:
            export_results(args.export_dir, args.results_dir, rows, summaries,
                           middle_size=args.middle_size)
        ok = sum(row["status"] == "ok" for row in summaries)
        assigned = sum(row["assigned_leaves"] for row in summaries)
        print(f"V2: {ok}/{len(summaries)} plants and {assigned}/{len(rows)} leaves classified", flush=True)
        if not args.export_only:
            import matplotlib
            if args.save:
                matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            csv_dir = locate_data(args.data_dir)
            configured_txt = os.environ.get("MAIZE_TXT_DIR")
            txt_dir = args.txt_dir or (Path(configured_txt) if configured_txt else
                                      csv_dir.parent / "0709单天多个材料原始数据")
            index_results = load_view_indices(args.results_dir, prepared[1], args.indices_dir)
            # Keep the selector alive until the GUI event loop exits.
            fig, selector = make_viewer(args.results_dir, csv_dir, txt_dir, args.sample,
                                        prepared_results=prepared, index_results=index_results)
            if args.save:
                fig.savefig(args.save, dpi=160, facecolor="white")
                plt.close(fig)
            else:
                plt.show()
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")
    return int(ok != len(summaries))


if __name__ == "__main__":
    raise SystemExit(main())
