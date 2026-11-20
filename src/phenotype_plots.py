"""PCA and distribution figures, plus offline spatial views of derived geometry."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import plotly.graph_objects as go

from .phenotype_features import FEATURES
from .visualization import CLUSTER_COLORS as COLORS, UNASSIGNED_COLOR as UNASSIGNED

INK = "#233143"
FIG_FORMAT = {"font.family": "DejaVu Sans", "pdf.fonttype": 42, "svg.fonttype": "none"}


def _axis_label(index, model):
    return f"PC{index} ({model['pca_explained_variance_ratio'][index-1]:.1%})"


def cluster_figure(leaves, model):
    with plt.rc_context(FIG_FORMAT):
        fig = plt.figure(figsize=(7.2, 4.9))
        ax = fig.add_axes([0.11, 0.13, 0.85, 0.68])
        fig.text(0.11, 0.943, "Leaf morphology clusters", color=INK,
                 weight="bold", fontsize=15, va="center")
        if model["status"] == "ok":
            for cluster, color in COLORS.items():
                group = [r for r in leaves if r["cluster_id"] == cluster]
                ax.scatter([r["pc1"] for r in group], [r["pc2"] for r in group], c=color,
                           label=f"{cluster} ({len(group)} leaves)", s=38, alpha=0.85,
                           edgecolors="white", linewidths=0.5)
            ax.set_xlabel(_axis_label(1, model))
            ax.set_ylabel(_axis_label(2, model))
            handles, labels = ax.get_legend_handles_labels()
            fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.11, 0.867),
                       ncol=3, frameon=False, fontsize=9, borderaxespad=0)
            ax.set_axisbelow(True)
            ax.grid(color="#e6edf2")
        else:
            ax.text(0.5, 0.5, "No batch cluster model", transform=ax.transAxes,
                    ha="center", va="center", color=INK)
            ax.set(xticks=[], yticks=[])
        ax.spines[["top", "right"]].set_visible(False)
        return fig


def distribution_figure(leaves):
    with plt.rc_context(FIG_FORMAT):
        fig = plt.figure(figsize=(10.3, 7.4))
        grid = fig.add_gridspec(2, 3, left=0.075, right=0.97, top=0.80, bottom=0.105,
                               wspace=0.35, hspace=0.57)
        fig.text(0.075, 0.94, "Leaf feature distributions", fontsize=19, weight="bold", color=INK)
        fig.text(0.075, 0.895, f"Quality-screened values from {len(leaves)} observed leaves",
                 fontsize=10, color=UNASSIGNED)
        fig.legend(handles=[Line2D([], [], color="#823c24", linewidth=1.4, label="Median")],
                   loc="center right", bbox_to_anchor=(0.97, 0.903), frameon=False, fontsize=9)
        for index, (field, title) in enumerate(FEATURES.items()):
            ax = fig.add_subplot(grid[index // 3, index % 3])
            values = [r[field] for r in leaves if r[field] is not None]
            ax.set_title(title, loc="left", fontsize=9.5, weight="bold", color=INK, pad=8)
            if values:
                ax.hist(values, bins="auto", color="#38768f", edgecolor="white")
                ax.axvline(np.median(values), color="#823c24", linewidth=1.4)
            ax.set_xlabel(f"{len(values)} leaves", fontsize=8, color=UNASSIGNED)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color="#e6edf2")
            if index % 3 == 0:
                ax.set_ylabel("Leaves")
            ax.tick_params(labelsize=8)
        return fig


def _save_png(fig, path):
    try:
        fig.savefig(path, format="png", dpi=300, facecolor="white")
    finally:
        plt.close(fig)


def _cluster_3d(leaves, model):
    fig = go.Figure()
    if model["status"] == "ok":
        for cluster, color in COLORS.items():
            group = [r for r in leaves if r["cluster_id"] == cluster]
            fig.add_scatter3d(
                x=[r["pc1"] for r in group], y=[r["pc2"] for r in group], z=[r["pc3"] for r in group],
                mode="markers", name=cluster, text=[f"{r['plant_id']} · leaf {r['leaf_id']}" for r in group],
                customdata=[[r[key] for key in FEATURES] for r in group],
                marker={"color": color, "size": 6, "opacity": 0.88},
                hovertemplate=("%{text}<br>" + "<br>".join(
                    f"{label}: %{{customdata[{i}]:.4g}}" for i, label in enumerate(FEATURES.values())
                ) + "<extra>%{fullData.name}</extra>"),
            )
        fig.update_layout(scene=dict(xaxis_title=_axis_label(1, model),
                                     yaxis_title=_axis_label(2, model),
                                     zaxis_title=_axis_label(3, model)))
    else:
        fig.add_annotation(text="No batch cluster model", showarrow=False)
    fig.update_layout(title="V1.1 · leaf clusters in PCA feature space", template="plotly_white")
    return fig


def _plant_3d(leaves, geometry, plant_clusters=None):
    fig = go.Figure()
    grouped = {}
    scope = "Within-plant K=3" if plant_clusters is not None else "Batch clusters"
    for plant, data in geometry.items():
        start = len(fig.data)
        if data["stem"] is not None:
            vertices = data["stem"]
            fig.add_scatter3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                              mode="lines", name="Stem centreline", line=dict(color=INK, width=6))
        for row in (r for r in leaves if r["plant_id"] == plant):
            leaf_id = row["leaf_id"]
            line = data["leaves"].get(leaf_id)
            cluster_id = (row["cluster_id"] if plant_clusters is None
                          else plant_clusters[(plant, leaf_id)])
            cluster = cluster_id or "unassigned"
            title = f"Leaf {leaf_id} · {cluster}"
            color = COLORS.get(cluster, UNASSIGNED)
            label_group = f"{plant}/{leaf_id}"
            if line is not None:
                fig.add_scatter3d(x=line[:, 0], y=line[:, 1], z=line[:, 2], mode="lines",
                                  name=title, legendgroup=label_group,
                                  line=dict(color=color, width=5), hoverinfo="name")
            if row["status"] == "ok":
                coordinates = [[row[f"{tip}_{axis}"] for axis in "xyz"] for tip in ("leaf", "stem")]
                p, q = coordinates
                fig.add_scatter3d(x=[p[0], q[0]], y=[p[1], q[1]], z=[p[2], q[2]],
                                  mode="lines+markers", name=f"{title} nearest pair",
                                  legendgroup=label_group, showlegend=line is None,
                                  text=["Leaf endpoint", "Stem endpoint"],
                                  marker=dict(color=[color if plant_clusters is not None else "#b12635",
                                                     "#e8a423"], size=4),
                                  line=dict(color="#b12635", width=2),
                                  hovertemplate=("%{text}<br>x=%{x:.5g}<br>y=%{y:.5g}<br>z=%{z:.5g}"
                                                 f"<extra>{title}</extra>"))
        grouped[plant] = range(start, len(fig.data))

    options = []
    for plant, indices in grouped.items():
        options.append({"label": plant, "method": "update",
                        "args": [{"visible": [i in indices for i in range(len(fig.data))]},
                                 {"title.text": f"Plant {plant} · {scope}"}]})
    title = f"V1.1 · {scope}"
    if options:
        title = options[0]["args"][1]["title.text"]
        for trace, visible in zip(fig.data, options[0]["args"][0]["visible"]):
            trace.visible = visible
    fig.update_layout(title=title, template="plotly_white",
                      scene=dict(aspectmode="data", xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                      updatemenus=[dict(buttons=options, x=0.02, y=0.99)] if options else [],
                      margin=dict(t=70, b=10, l=10, r=10), legend=dict(groupclick="togglegroup"))
    return fig


def write_figures(path: Path, leaves, model, geometry, plant_cluster_rows):
    _save_png(cluster_figure(leaves, model), path / "cluster_2d.png")
    _save_png(distribution_figure(leaves), path / "feature_distributions.png")
    interactive = path / "interactive"
    interactive.mkdir()
    _cluster_3d(leaves, model).write_html(interactive / "cluster_3d.html", include_plotlyjs="directory",
                                         div_id="maize-cluster-3d")
    _plant_3d(leaves, geometry).write_html(interactive / "plant_geometry_3d.html",
                                          include_plotlyjs="directory", div_id="maize-plant-3d")
    labels = {(r["plant_id"], r["leaf_id"]): r["plant_cluster_id"] for r in plant_cluster_rows}
    _plant_3d(leaves, geometry, labels).write_html(interactive / "plant_clusters_3d.html",
                                                   include_plotlyjs="directory",
                                                   div_id="maize-plant-clusters-3d")
