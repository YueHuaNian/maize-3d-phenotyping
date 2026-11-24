"""Synthetic ground-truth completion demo (hole, edge, and tip withheld)."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".vendor"), str(ROOT / "tests")]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import numpy as np
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Patch
from plotly.subplots import make_subplots
from test_modeling import synthetic
from maize_leaf_v1.model import fit_leaf, Config, COLORS, REGIONS
from maize_leaf_v1.export import scatter, mesh_traces, page, ply, json_write, export_model_metadata, ensure_js


def main():
    base = ROOT / "outputs/leaf_modeling_v1_0"
    out = base / "validation" / "synthetic_demo"
    out.mkdir(parents=True, exist_ok=True)
    ensure_js(base)

    truth, missing, hole, stem = synthetic()
    observed = truth[~missing]
    result = fit_leaf(observed, stem, Config(nu=100, nv=31), 7)
    tree = cKDTree(result["vertices"])
    metrics = {
        "truth_points": len(truth),
        "input_points": len(observed),
        "withheld_points": int(missing.sum()),
        "withheld_hole_points": int(hole.sum()),
        "hole_distance_median": float(np.median(tree.query(truth[hole])[0])),
        "hole_distance_p95": float(np.quantile(tree.query(truth[hole])[0], 0.95)),
        "all_withheld_distance_p95": float(np.quantile(tree.query(truth[missing])[0], 0.95)),
        "fit_metrics": result["metrics"],
    }
    json_write(out / "metrics.json", metrics)
    ply(out / "synthetic_truth.ply", truth)
    ply(out / "synthetic_observed.ply", observed)
    ply(out / "synthetic_surface.ply", result["vertices"], result["faces"], COLORS[result["regions"]], result["regions"], result["face_regions"])
    result["model"].save(out / "synthetic_model.npz")
    export_model_metadata(out, "synthetic", result["model"], result["u"])

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=["Input (hole / edge / tip withheld)", "Completed surface"],
    )
    fig.add_trace(scatter(observed, "Synthetic input", "#334c42"), row=1, col=1)
    for trace in mesh_traces(result):
        fig.add_trace(trace, row=1, col=2)
    fig.add_trace(scatter(observed, "Input overlay", "#334c42", size=1), row=1, col=2)
    fig.update_scenes(aspectmode="data", camera=dict(eye=dict(x=0.5, y=-1.5, z=2)))
    fig.update_layout(height=650, margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=-0.05))
    body = (
        '<header><h1>合成曲面补全对比</h1><a href="../../index.html">返回主页</a></header><main>'
        f'<div class="card"><p>删除 {missing.sum()} / {len(truth)} 点（孔洞 {hole.sum()} 点），孔洞 P95 = {metrics["hole_distance_p95"]:.5f}。</p></div>'
        f'<div class="card">{fig.to_html(full_html=False, include_plotlyjs=False, config={"displaylogo": False})}</div>'
        '<div class="card"><img src="completion_comparison.png"><p><a href="metrics.json">metrics.json</a></p></div></main>'
    )
    page("合成缺失补全对比", body, out / "index.html", "../../assets/plotly.min.js")

    figure = plt.figure(figsize=(13, 6), dpi=140)
    for i in range(2):
        ax = figure.add_subplot(1, 2, i + 1, projection="3d")
        ax.scatter(*observed.T, s=2, color="#334c42", alpha=0.7)
        if i:
            ax.add_collection3d(
                Poly3DCollection(result["vertices"][result["faces"]], facecolors=COLORS[result["face_regions"]] / 255, edgecolor="none", alpha=0.90)
            )
        ax.set(
            xlim=(-0.03, 1.03),
            ylim=(-0.10, 0.10),
            zlim=(-0.02, 0.25),
            xlabel="X",
            ylabel="Y",
            zlabel="Z",
            title="Input" if i == 0 else "Reconstructed surface",
        )
        ax.set_box_aspect([1, 0.32, 0.32])
        ax.view_init(60, -70)
    figure.legend(handles=[Patch(color=COLORS[k] / 255, label=REGIONS[k]) for k in REGIONS], loc="lower center", ncol=3)
    figure.savefig(out / "completion_comparison.png", bbox_inches="tight")
    plt.close(figure)
    print(out)
    print(metrics)


if __name__ == "__main__":
    main()
