"""Open the V1.1 leaf clusters and nearest pairs in a Matplotlib window."""

import argparse
import csv
from collections import defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from textwrap import fill

import numpy as np

from src.io_utils import locate_data
from src.nearest_attachment import load_pair
from src.visualization import (
    CLUSTER_COLORS, PLANT_CLUSTER_COLORS, UNASSIGNED_COLOR, configure, setup_3d,
)

ROOT = Path(__file__).resolve().parent


def read_results(folder, cluster_scope="auto"):
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
    morphology = folder / "plant_morphotypes" / "leaf_clusters.csv"
    if cluster_scope != "batch" and morphology.is_file():
        _read_plant_clusters(morphology, inputs, groups)
        return inputs, groups, "Within-plant morphology K=3"
    within = folder / "plant_clusters" / "leaf_clusters.csv"
    if cluster_scope == "plant" or (cluster_scope == "auto" and within.is_file()):
        _read_plant_clusters(within, inputs, groups)
        return inputs, groups, "Within-plant K=3"
    return inputs, groups, "Batch clusters"


def _read_plant_clusters(path, inputs, groups):
    provenance = json.loads((path.parent / "provenance.json").read_text(encoding="utf-8"))
    for plant in groups:
        reference = provenance.get("inputs", {}).get(plant, {})
        if any(reference.get(key) != inputs[plant][key] for key in ("csv_sha256", "txt_sha256")):
            raise ValueError(f"{plant}: within-plant results belong to a different point cloud")

    labels = {}
    morphotypes = provenance.get("morphotypes", {})
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["plant_id"], int(row["leaf_id"])
            cluster = row["plant_cluster_id"]
            if key in labels or (cluster and cluster not in CLUSTER_COLORS):
                raise ValueError(f"Invalid within-plant cluster row for {key[0]} leaf {key[1]}")
            morphotype = row.get("morphotype", "")
            if morphotypes and morphotype != morphotypes.get(cluster, ""):
                raise ValueError(f"Invalid morphotype for {key[0]} leaf {key[1]}")
            labels[key] = (cluster, morphotype)
    expected = {(plant, leaf) for plant, rows in groups.items() for leaf in rows}
    if labels.keys() != expected:
        raise ValueError("Within-plant cluster rows differ from the V1.1 leaf table")
    for (plant, leaf), (cluster, morphotype) in labels.items():
        groups[plant][leaf].update(cluster_id=cluster, morphotype=morphotype)


def _source_bytes(directory, record, kind):
    name = record[f"{kind}_file"]
    if not name or Path(name).name != name:
        raise ValueError(f"Invalid {kind.upper()} filename in the input manifest")
    payload = (directory / name).read_bytes()
    if sha256(payload).hexdigest() != record[f"{kind}_sha256"]:
        raise ValueError(f"{name} differs from the original V1.1 analysis input")
    return payload


def draw_plant(ax, plant, xyz, labels, leaves, scope):
    from matplotlib.lines import Line2D

    visible = np.isfinite(xyz).all(axis=1) & (labels >= 0)
    if not visible.any():
        raise ValueError(f"{plant}: no visible finite stem or leaf coordinates")
    if set(labels[labels > 0]) != set(leaves):
        raise ValueError(f"{plant}: leaf IDs differ between results and point cloud")

    within = scope.startswith("Within-plant")
    colors = PLANT_CLUSTER_COLORS if within else CLUSTER_COLORS
    stem_color = "#343a46" if within else "#28679f"
    pair_color = "#5f6572" if within else "#b12635"
    stem_point_color = "#343a46" if within else "#e8a423"

    ax.clear()
    stem = xyz[visible & (labels == 0)]
    handles = [ax.scatter(*stem.T, color=stem_color, s=2, alpha=0.5, depthshade=False,
                          linewidths=0, label="Stem")]
    pair_count = 0
    artists = defaultdict(list)
    leaf_ids = defaultdict(list)
    for leaf, row in sorted(leaves.items()):
        name = row["cluster_id"] or "Unassigned"
        color = colors.get(name, UNASSIGNED_COLOR)
        points = xyz[visible & (labels == leaf)]
        leaf_ids[name].append(leaf)
        group = artists[name]
        points_artist = ax.scatter(*points.T, color=color, s=3, alpha=0.88 if within else 0.62,
                                   depthshade=False, linewidths=0, label=f"Leaf {leaf} · {name}")
        group.append(points_artist)
        if not within:
            handles.append(points_artist)

        base = None
        if row["status"] == "ok":
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
            base = pair[0]
            x, y, z = pair.T
            group.extend(ax.plot(x, y, z, color=pair_color, linewidth=1.2, zorder=5))
            group.append(ax.scatter(x[1], y[1], z[1], color=stem_point_color, marker="D",
                                    s=19, zorder=6, depthshade=False, edgecolors="white", linewidths=0.65))
            group.append(ax.scatter(x[0], y[0], z[0], facecolors="none",
                                    edgecolors=color if within else pair_color, s=38,
                                    zorder=7, depthshade=False, linewidths=1.4))
            pair_count += 1

        if within and len(points):
            position = np.median(points, axis=0)
            if base is not None:
                distances = np.linalg.norm(points - base, axis=1)
                position = np.median(points[distances >= np.quantile(distances, 0.9)], axis=0)
            group.append(ax.text(*position, str(leaf), zorder=8, ha="center", va="center",
                                 fontsize=9, fontweight="bold", color=color,
                                 bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                                           edgecolor=color, linewidth=0.8, alpha=0.94)))

    if pair_count:
        handles.extend([
            Line2D([], [], marker="o", markerfacecolor="none", color=pair_color,
                   linestyle="none", label="Nearest leaf point"),
            Line2D([], [], marker="D", color=stem_point_color, linestyle="none",
                   label="Nearest stem point"),
            Line2D([], [], color=pair_color, label="Nearest segment"),
        ])
    setup_3d(ax, xyz[visible])
    cluster_count = sum(name in colors for name in leaf_ids)
    ax.set_title(f"{plant}  |  {len(leaves)} leaves  |  {cluster_count} clusters\n{scope}", loc="left")
    ax.legend(handles=handles, loc="upper left",
              bbox_to_anchor=((1.05, 0.37) if within else (1.01, 1.0)),
              frameon=False, fontsize=8, markerscale=1.5)
    ax.grid(alpha=0.3)
    return artists, leaf_ids


def make_viewer(results, csv_dir, txt_dir, sample=None, cluster_scope="auto"):
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RadioButtons

    inputs, leaves, scope = read_results(results, cluster_scope)
    plants = sorted(leaves)
    if sample is not None and sample not in plants:
        raise ValueError(f"Plant {sample!r} is missing from the V1.1 results")
    within = scope.startswith("Within-plant")
    morphotypes = {row["cluster_id"]: row["morphotype"] for group in leaves.values()
                   for row in group.values() if row.get("morphotype")}
    if morphotypes:
        configure()
    initial = sample or max(plants, key=lambda p: len({r["cluster_id"] for r in leaves[p].values()} - {""}))
    fig = plt.figure(figsize=(12.8, 8.1))
    ax = fig.add_axes((0.17, 0.10, 0.60 if within else 0.63, 0.82),
                      projection="3d", computed_zorder=False)
    list_ax = fig.add_axes((0.025, 0.22, 0.13, 0.60))
    list_ax.set_title("Plant", loc="left", fontsize=10)
    selector = RadioButtons(list_ax, plants, active=plants.index(initial))
    class_selector = None
    codes = ["All", *PLANT_CLUSTER_COLORS]
    if within:
        if any(not r["cluster_id"] for rows in leaves.values() for r in rows.values()):
            codes.append("Unassigned")
        class_ax = fig.add_axes((0.80, 0.48, 0.175, 0.37))
        class_ax.set_title("Show leaf class", loc="left", fontsize=11, fontweight="bold")
        class_selector = RadioButtons(class_ax, codes, activecolor="#303846")
        for code, label in zip(codes, class_selector.labels):
            label.set_color(PLANT_CLUSTER_COLORS.get(code, "#343a46"))
            label.set_fontweight("bold")
            label.set_fontsize(9)
    rendered = False
    artists = {}
    active_class = "All"
    hint = "Left-drag to rotate. Right-drag to zoom. XYZ use original coordinate units."
    status = fig.text(0.17, 0.039, hint, fontsize=9, color="#475767")

    def set_class(label):
        nonlocal active_class
        active_class = label.split()[0]
        for name, group in artists.items():
            for artist in group:
                artist.set_visible(active_class == "All" or name == active_class)
        fig.canvas.draw_idle()

    if class_selector is not None:
        class_selector.on_clicked(set_class)

    def show(plant):
        nonlocal rendered, artists
        source = inputs[plant]
        xyz, labels = load_pair(_source_bytes(csv_dir, source, "csv"),
                                _source_bytes(txt_dir, source, "txt"))
        camera = (ax.elev, ax.azim)
        artists, members = draw_plant(ax, plant, xyz, labels, leaves[plant], scope)
        if rendered:
            ax.view_init(elev=camera[0], azim=camera[1])
        rendered = True
        status.set_text(hint)
        if class_selector is not None:
            class_selector.labels[0].set_text(f"All classes · {len(leaves[plant])} leaves")
            for code, label in zip(codes[1:], class_selector.labels[1:]):
                ids = members.get(code, [])
                count = f"{len(ids)} {'leaf' if len(ids) == 1 else 'leaves'}"
                numbers = ", ".join(map(str, ids)) or "—"
                category = f"{code} {morphotypes[code]}" if code in morphotypes else code
                label.set_text(f"{category} · {count}\n{fill('Leaf IDs: ' + numbers, width=21)}")
            set_class(active_class)
        fig.canvas.draw_idle()

    def select(plant):
        nonlocal artists
        try:
            show(plant)
        except (ValueError, OSError) as exc:
            ax.clear()
            artists = {}
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
    parser.add_argument("--cluster-scope", choices=("auto", "plant", "batch"), default="auto",
                        help="use within-plant clusters when available, or select one scope")
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
        fig, selector = make_viewer(args.results_dir, csv_dir, txt_dir,
                                    args.sample, args.cluster_scope)
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
