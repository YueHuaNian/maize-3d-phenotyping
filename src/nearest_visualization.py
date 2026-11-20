"""Three-dimensional verification views and descriptive batch plots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import proj3d
import numpy as np

STEM = "#2b69a4"
ENDPOINT = "#bb2637"
STEM_ENDPOINT = "#f5bc37"
INK = "#1d2a38"
UNIT = "original coordinate units"
LEAF_COLORS = ("#218574", "#cf7b2c", "#8b579d", "#687e33", "#a65946", "#bd668d",
               "#988417", "#47a0a6", "#795ca9", "#9b6840", "#d28a54", "#7b8896")

plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "pdf.fonttype": 42})


def _label_leaf_bases(ax, pairs, colors):
    # Project the original coordinates, then place labels in two side columns.
    ax.figure.canvas.draw()
    columns = {False: [], True: []}
    for row in pairs:
        position = proj3d.proj_transform(row.leaf_x, row.leaf_y, row.leaf_z, ax.get_proj())
        xy = ax.transAxes.inverted().transform(ax.transData.transform(position[:2]))
        columns[xy[0] >= 0.5].append((xy, row.leaf_id))
    for right, entries in columns.items():
        entries.sort(key=lambda item: (item[0][1], item[1]))
        x = 0.82 if right else 0.06
        step = min(0.047, 0.81 / max(len(entries) - 1, 1))
        next_y = 0.095
        for index, (xy, leaf_id) in enumerate(entries):
            upper = 0.905 - (len(entries) - index - 1) * step
            y = np.clip(xy[1], next_y, upper)
            next_y = y + step
            ax.annotate(
                str(leaf_id), xy=xy, xytext=(x, y), xycoords=ax.transAxes,
                ha="center", va="center", fontsize=9, weight="bold", color=INK,
                bbox=dict(boxstyle="round,pad=0.19", fc="white", ec=colors[leaf_id], lw=1.0),
                arrowprops=dict(arrowstyle="-", color="#596a7a", lw=0.55, shrinkB=7),
                zorder=15,
            )


def plant_figure(plant_id, xyz, labels, attachments):
    fig = plt.figure(figsize=(8.8, 7.6), facecolor="white")
    ax = fig.add_axes([0.08, 0.13, 0.84, 0.78], projection="3d",
                      computed_zorder=False, proj_type="ortho")
    valid = np.isfinite(xyz).all(axis=1)
    ids = [row.leaf_id for row in attachments if row.leaf_id is not None]
    if len(ids) <= len(LEAF_COLORS):
        colors = dict(zip(ids, LEAF_COLORS))
    else:
        palette = plt.get_cmap("tab20" if len(ids) <= 20 else "turbo").resampled(len(ids))
        colors = {leaf_id: palette(index) for index, leaf_id in enumerate(ids)}
    stem = xyz[valid & (labels == 0)]
    ax.scatter(*stem.T, c=STEM, s=1.8, alpha=0.55, edgecolors="none",
               depthshade=False, rasterized=True)
    for leaf_id in ids:
        leaf = xyz[valid & (labels == leaf_id)]
        ax.scatter(*leaf.T, color=colors[leaf_id], s=1.25, alpha=0.48, edgecolors="none",
                   depthshade=False, rasterized=True)

    points = xyz[valid & (labels >= 0)]
    if len(points):
        lower, upper = points.min(axis=0), points.max(axis=0)
        center = lower + (upper - lower) / 2
        spans = upper - lower
        half_width = spans / 2 + (spans.max() * 0.035 or 0.5)
        limits = [(c - w, c + w) for c, w in zip(center, half_width)]
        ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2])
        ax.set_box_aspect(half_width)
    else:
        ax.set(xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))
        ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=20, azim=-57)
    ax.set(xlabel="X", ylabel="Y", zlabel="Z")
    ax.grid(False)
    ax.tick_params(labelsize=8)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.97, 0.98, 0.99, 1.0))
        axis.label.set_size(8)
        axis.set_major_locator(MaxNLocator(nbins=3, prune="both"))

    pairs = [row for row in attachments if row.status == "ok"]
    for row in pairs:
        p = (row.leaf_x, row.leaf_y, row.leaf_z)
        q = (row.stem_x, row.stem_y, row.stem_z)
        ax.plot(*zip(p, q), color=ENDPOINT, lw=1.15, zorder=6)
    if pairs:
        leaf_xyz = np.array([(r.leaf_x, r.leaf_y, r.leaf_z) for r in pairs])
        stem_xyz = np.array([(r.stem_x, r.stem_y, r.stem_z) for r in pairs])
        ax.scatter(*stem_xyz.T, marker="D", c=STEM_ENDPOINT, edgecolors=INK, s=17,
                   linewidths=0.5, depthshade=False, zorder=7)
        ax.scatter(*leaf_xyz.T, marker="o", facecolors="none", edgecolors=ENDPOINT, s=31,
                   linewidths=1.25, depthshade=False, zorder=8)
        _label_leaf_bases(ax, pairs, colors)

    fig.text(0.08, 0.956, f"Plant {plant_id}", fontsize=18, weight="bold", color=INK,
             va="center")
    fig.text(0.08, 0.920, f"{len(pairs)}/{len(ids)} leaf instances localized  |  XYZ in {UNIT}",
             fontsize=10, color="#526476", va="center")
    fig.legend(
        handles=[
            Line2D([], [], marker="o", color=STEM, linestyle="none", label="Stem cloud"),
            Line2D([], [], marker="o", color="#718475", linestyle="none", label="Leaf clouds"),
            Line2D([], [], marker="o", mfc="none", color=ENDPOINT, linestyle="none",
                   label="Leaf point"),
            Line2D([], [], marker="D", mfc=STEM_ENDPOINT, mec=INK, color=INK, linestyle="none",
                   label="Stem point"),
            Line2D([], [], color=ENDPOINT, label="Shortest segment"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=5, frameon=False, fontsize=9,
    )
    return fig


def overview_figure(results, plant_ids, bounds):
    fig = plt.figure(figsize=(11.5, 9.3), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=(1, 1.65),
                           left=0.095, right=0.97, bottom=0.09, top=0.86,
                           wspace=0.30, hspace=0.52)
    histogram = fig.add_subplot(grid[0, 0])
    box = fig.add_subplot(grid[0, 1], sharex=histogram)
    comparison = fig.add_subplot(grid[1, :], sharex=histogram)
    all_distances = np.array([row["nearest_distance"] for row in results if row["status"] == "ok"])
    fig.text(0.095, 0.938, "Leaf-to-stem nearest distances", fontsize=23, weight="bold", color=INK)
    fig.text(0.095, 0.900, f"{len(plant_ids)} plants  |  {len(all_distances)} localized leaves",
             fontsize=11, color="#526476")

    for axis, title in ((histogram, "A  Distance distribution"), (box, "B  Pooled box plot"),
                        (comparison, "C  Distances by plant")):
        axis.set_title(title, loc="left", pad=9, color=INK, fontsize=12, weight="bold")
        axis.set_xlabel(f"Nearest distance ({UNIT})", color=INK)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9, colors=INK)
        axis.set_axisbelow(True)

    color = "#3f7896"
    if len(all_distances):
        histogram.hist(all_distances, bins="auto", color=color, edgecolor="white", linewidth=0.7)
        box.boxplot(all_distances, orientation="horizontal", patch_artist=True,
                    boxprops=dict(facecolor="#dbeaf1", color=color),
                    medianprops=dict(color=INK, lw=1.6),
                    flierprops=dict(marker="o", markersize=4, markeredgecolor=ENDPOINT,
                                    markerfacecolor="none"))
    else:
        for axis in (histogram, box):
            axis.text(0.5, 0.5, "No successful distances", transform=axis.transAxes,
                      ha="center", va="center", color=INK)
    histogram.set_ylabel("Number of leaves", color=INK)
    histogram.grid(axis="y", color="#e7edf0")
    box.set_yticks([])
    box.spines["left"].set_visible(False)

    for position, plant_id in enumerate(plant_ids):
        distances = np.array([row["nearest_distance"] for row in results
                              if row["plant_id"] == plant_id and row["status"] == "ok"])
        if len(distances):
            lower, upper = bounds
            atypical = (distances < lower) | (distances > upper)
            comparison.scatter(distances[~atypical], np.full((~atypical).sum(), position),
                               color=color, s=20, alpha=0.75, linewidths=0)
            comparison.scatter(distances[atypical], np.full(atypical.sum(), position),
                               facecolors="none", edgecolors=ENDPOINT, s=23, linewidths=1)
            comparison.vlines(np.median(distances), position - 0.28, position + 0.28,
                              lw=1.3, color=INK)
    if len(all_distances):
        histogram.set_xlim(-0.002 * all_distances.max(), all_distances.max() * 1.045 or 1)
    comparison.set_yticks(range(len(plant_ids)), plant_ids)
    comparison.invert_yaxis()
    comparison.grid(axis="y", color="#e7edf0")
    comparison.legend(
        handles=[
            Line2D([], [], marker="o", color=color, linestyle="none", label="Leaf"),
            Line2D([], [], marker="o", mfc="none", color=ENDPOINT, linestyle="none",
                   label="Outside pooled 1.5 IQR fences"),
            Line2D([], [], marker="|", markersize=10, color=INK, linestyle="none", label="Plant median"),
        ],
        frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(1, 1.12), fontsize=9,
    )
    return fig


def save_figure(fig, output: Path):
    try:
        fig.savefig(output, dpi=300, facecolor="white")
    finally:
        plt.close(fig)


def plot_plant(output, plant_id, xyz, labels, attachments):
    save_figure(plant_figure(plant_id, xyz, labels, attachments), output)
