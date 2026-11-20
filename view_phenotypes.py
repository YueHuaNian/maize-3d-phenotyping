"""Open the V1.1 leaf clusters and nearest pairs in a Matplotlib window."""

import argparse
import csv
from collections import defaultdict
from hashlib import sha256
import os
from pathlib import Path

import numpy as np

from src.io_utils import locate_data
from src.nearest_attachment import load_pair
from src.visualization import CLUSTER_COLORS, UNASSIGNED_COLOR, setup_3d

ROOT = Path(__file__).resolve().parent


def read_results(folder):
    with (folder / "input_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
        inputs = {row["plant_id"]: row for row in csv.DictReader(stream)
                  if row["input_status"] == "ok"}
    groups = defaultdict(dict)
    with (folder / "leaf_features.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            plant = row["plant_id"]
            if plant not in inputs:
                continue
            leaf = int(row["leaf_id"])
            if leaf in groups[plant]:
                raise ValueError(f"{plant}: duplicate leaf {leaf}")
            groups[plant][leaf] = row
    if not groups:
        raise ValueError("No validated plants with V1.1 leaf results")
    return inputs, groups


def _source_bytes(directory, record, kind):
    name = record[f"{kind}_file"]
    if not name or Path(name).name != name:
        raise ValueError(f"Invalid {kind.upper()} filename in the input manifest")
    payload = (directory / name).read_bytes()
    if sha256(payload).hexdigest() != record[f"{kind}_sha256"]:
        raise ValueError(f"{name} differs from the original V1.1 analysis input")
    return payload


def draw_plant(ax, plant, xyz, labels, leaves):
    from matplotlib.lines import Line2D

    visible = np.isfinite(xyz).all(axis=1) & (labels >= 0)
    if not visible.any():
        raise ValueError(f"{plant}: no visible finite stem or leaf coordinates")
    if set(labels[labels > 0]) != set(leaves):
        raise ValueError(f"{plant}: leaf IDs differ between results and point cloud")

    ax.clear()
    stem = xyz[visible & (labels == 0)]
    handles = [ax.scatter(*stem.T, color="#28679f", s=2, alpha=0.6, depthshade=False,
                          linewidths=0, label="Stem")]
    pair_count = 0
    clusters = {row["cluster_id"] for row in leaves.values()} - {""}
    for leaf, row in sorted(leaves.items()):
        name = row["cluster_id"] or "unassigned"
        color = CLUSTER_COLORS.get(name, UNASSIGNED_COLOR)
        points = xyz[visible & (labels == leaf)]
        handles.append(ax.scatter(*points.T, color=color, s=3, alpha=0.62, depthshade=False,
                                  linewidths=0, label=f"Leaf {leaf} · {name}"))
        if row["status"] != "ok":
            continue
        indices = [int(row[f"{side}_point_row"]) for side in ("leaf", "stem")]
        if not all(0 <= i < len(xyz) and visible[i] for i in indices) or (
            labels[indices[0]] != leaf or labels[indices[1]] != 0
        ):
            raise ValueError(f"{plant}: inconsistent nearest pair indices for leaf {leaf}")
        pair = xyz[indices]
        for side, position in zip(("leaf", "stem"), pair):
            recorded = [float(row[f"{side}_{axis}"]) for axis in "xyz"]
            if not np.array_equal(position, recorded):
                raise ValueError(f"{plant}: nearest pair coordinates differ for leaf {leaf}")
        x, y, z = pair.T
        ax.plot(x, y, z, color="#b12635", linewidth=1.2, zorder=5)
        ax.scatter(x[1], y[1], z[1], color="#e8a423", marker="D", s=23, zorder=6,
                   depthshade=False, edgecolors="#362515", linewidths=0.6)
        ax.scatter(x[0], y[0], z[0], facecolors="none", edgecolors="#b12635", s=37,
                   zorder=7, depthshade=False, linewidths=1.2)
        pair_count += 1

    if pair_count:
        handles.extend([
            Line2D([], [], marker="o", markerfacecolor="none", color="#b12635",
                   linestyle="none", label="Nearest leaf point"),
            Line2D([], [], marker="D", color="#e8a423", linestyle="none", label="Nearest stem point"),
            Line2D([], [], color="#b12635", label="Nearest segment"),
        ])
    setup_3d(ax, xyz[visible])
    ax.set_title(f"{plant}  |  {len(leaves)} leaves  |  {len(clusters)} clusters", loc="left")
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False, fontsize=8, markerscale=1.5)
    ax.grid(alpha=0.3)


def make_viewer(results, csv_dir, txt_dir, sample=None):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons

    inputs, leaves = read_results(results)
    plants = sorted(leaves)
    if sample is not None and sample not in plants:
        raise ValueError(f"Plant {sample!r} is missing from the V1.1 results")
    # Prefer a sample containing all three clusters for the initial view.
    initial = sample or max(plants, key=lambda p: len({r["cluster_id"] for r in leaves[p].values()} - {""}))
    fig = plt.figure(figsize=(12.4, 8.1))
    ax = fig.add_axes((0.17, 0.10, 0.63, 0.82), projection="3d", computed_zorder=False)
    list_ax = fig.add_axes((0.025, 0.22, 0.13, 0.60))
    list_ax.set_title("Plant", loc="left", fontsize=10)
    selector = RadioButtons(list_ax, plants, active=plants.index(initial))
    rendered = False
    hint = "Left-drag to rotate. Right-drag to zoom. XYZ use original coordinate units."
    status = fig.text(0.17, 0.039, hint, fontsize=9, color="#475767")

    def show(plant):
        nonlocal rendered
        source = inputs[plant]
        xyz, labels = load_pair(_source_bytes(csv_dir, source, "csv"),
                                _source_bytes(txt_dir, source, "txt"))
        camera = (ax.elev, ax.azim)
        draw_plant(ax, plant, xyz, labels, leaves[plant])
        if rendered:
            ax.view_init(elev=camera[0], azim=camera[1])
        rendered = True
        status.set_text(hint)
        fig.canvas.draw_idle()

    def select(plant):
        try:
            show(plant)
        except (ValueError, OSError) as exc:
            ax.clear()
            status.set_text(str(exc))
            fig.canvas.draw_idle()

    try:
        show(initial)
    except (ValueError, OSError):
        plt.close(fig)
        raise
    selector.on_clicked(select)
    return fig, selector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", help="plant to display first")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "outputs" / "v1.1")
    parser.add_argument("--data-dir", type=Path, help="semantic CSV directory")
    parser.add_argument("--txt-dir", type=Path, help="matching original TXT directory")
    parser.add_argument("--save", type=Path, help="write a static preview instead of opening a window")
    args = parser.parse_args()

    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        csv_dir = locate_data(args.data_dir)
        configured_txt = os.environ.get("MAIZE_TXT_DIR")
        txt_dir = args.txt_dir or (Path(configured_txt) if configured_txt else
                                  csv_dir.parent / "0709单天多个材料原始数据")
        if not txt_dir.is_dir():
            raise FileNotFoundError(f"TXT input directory not found: {txt_dir}")
        # Keep the selector alive while the GUI event loop runs.
        fig, selector = make_viewer(args.results_dir, csv_dir, txt_dir, args.sample)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"{exc}\n")

    if args.save:
        fig.savefig(args.save, dpi=160, facecolor="white")
        plt.close(fig)
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
