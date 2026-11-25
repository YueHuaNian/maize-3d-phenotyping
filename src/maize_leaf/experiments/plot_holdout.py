"""绘制留出误差汇总及逐案例点云与曲面对照。

Render occlusion-error distributions and explicitly ranked diagnostic cases."""
import argparse
import json
import os
from pathlib import Path
import sys
# 验证绘图（重建阶段）：遮挡留出误差分布与失败案例诊断图
ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np
from maize_leaf.validation import SCENARIOS, point_mesh_distances

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 8,
                     "axes.titlesize": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none"})

LABELS = {"hole": "Hole", "margin": "Margin", "tip_10pct": "Tip 10%",
          "tip_20pct": "Tip 20%", "tip_30pct": "Tip 30%"}
BLUE, ORANGE, GREY = "#4477AA", "#CC6677", "#88939E"


def save_figure(fig, path, qa_scripts=None):
    """保存诊断图，并在指定时运行外部图版对齐检查。"""
    fig.canvas.draw()
    if qa_scripts:
        sys.path.insert(0, str(qa_scripts))
        from audit_panel_alignment import require_matplotlib_panel_alignment
        require_matplotlib_panel_alignment(fig, json_out=str(path) + ".alignment.json",
                                           overlay_svg=str(path) + ".alignment.svg", strict=True)
    fig.savefig(str(path) + ".png", dpi=300)
    fig.savefig(str(path) + ".pdf")
    fig.savefig(str(path) + ".svg")
    plt.close(fig)


def case_figure(root, row, destination, qa_scripts=None):
    """绘制单个留出案例的观测点、隐藏点与预测曲面。"""
    prefix = root / row["artifact"]
    with np.load(str(prefix) + "_split.npz") as data:
        points, obs, held = data["points"], data["observed"], data["held"]
        u = data["u"][held]
    center = np.median(points, axis=0)
    _, _, basis = np.linalg.svd(points - center, full_matrices=False)
    projected = (points - center) @ basis.T
    mesh_path = Path(str(prefix) + "_mesh.npz")
    vertices, faces, dist = None, None, None
    if mesh_path.exists():
        with np.load(mesh_path) as data:
            vertices, faces = data["vertices"], data["faces"]
        dist = point_mesh_distances(points[held], vertices, faces)
        surface = (vertices - center) @ basis.T
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 4.5))
    fig.subplots_adjust(left=.065, right=.98, bottom=.22, top=.78, wspace=.36)
    for ax, dims in zip(axes[:2], ((0, 1), (0, 2))):
        ax.scatter(projected[obs, dims[0]], projected[obs, dims[1]], s=2, c=GREY,
                   alpha=.65, rasterized=True)
        if vertices is not None:
            polygons = surface[faces][:, :, dims]
            ax.add_collection(PolyCollection(polygons, facecolor=BLUE, alpha=.20,
                                             edgecolor="none", rasterized=True))
        ax.scatter(projected[held, dims[0]], projected[held, dims[1]], s=5, c=ORANGE,
                   rasterized=True)
        xy = projected[:, dims]
        if vertices is not None:
            xy = np.vstack([xy, surface[:, dims]])
        mid = .5 * (xy.min(axis=0) + xy.max(axis=0))
        half = .55 * max(float(np.ptp(xy, axis=0).max()), 1e-6)
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("PC1 (input units)")
        ax.set_ylabel(f"PC{dims[1]+1} (input units)")
        ax.tick_params(labelsize=6)
        ax.locator_params(nbins=3)
    axes[0].set_title("a  Broad view", loc="left", fontweight="bold")
    axes[1].set_title("b  Side view", loc="left", fontweight="bold")
    ax = axes[2]
    ax.set_box_aspect(1)
    if dist is not None:
        ax.scatter(u, dist / row["ref_max_width"], s=7, c=ORANGE, alpha=.65, rasterized=True)
        threshold = max(6 * row["spacing"], .15 * row["ref_max_width"]) / row["ref_max_width"]
        ax.axhline(threshold, c=GREY, linestyle="--", linewidth=.8)
        text = (f"P95 / width = {row['held_p95_over_width']:.2f}; "
                f"length change = {row['length_change_pct']:.1f}%; area change = {row['area_change_pct']:.1f}%")
    else:
        ax.text(.5, .5, "Fit failed", transform=ax.transAxes, ha="center", color=ORANGE)
        text = "Model fitting failed; the case remains in the error table."
    ax.set_title("c  Hidden-point error", loc="left", fontweight="bold")
    ax.set_xlabel("Frozen longitudinal coordinate")
    ax.set_ylabel("Distance / reference width")
    ax.tick_params(labelsize=6)
    ax.locator_params(nbins=3)
    fig.suptitle(f"Plant {row['plant']} / leaf {row['leaf_id']} / {LABELS[row['scenario']]} / {row['status']}", y=.96)
    fig.text(.5, .88, "Grey: observed scan   Pink: hidden scan   Blue: reconstructed mesh", ha="center", fontsize=8)
    fig.text(.5, .08, text, ha="center", fontsize=8)
    save_figure(fig, destination, qa_scripts)


def summary_figure(report, destination, qa_scripts=None):
    """绘制各遮挡场景的距离误差和表型漂移汇总。"""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.3))
    fig.subplots_adjust(left=.08, right=.98, bottom=.20, top=.84, wspace=.30)
    rows = report["records"]
    for index, scenario in enumerate(SCENARIOS):
        subset = [r for r in rows if r["scenario"] == scenario]
        fitted = [r for r in subset if r["status"] in ("ok", "alert")]
        plants = sorted({r["plant"] for r in fitted})
        values = np.array([np.median([r["held_p95_over_width"] for r in fitted if r["plant"] == p]) for p in plants])
        # Display floor only for the logarithmic axis; raw table values are kept.
        values = np.maximum(values, 1e-8)
        if len(values):
            axes[0].scatter(index + np.linspace(-.13, .13, len(values)), values, c=BLUE, s=14, alpha=.6)
            q25, med, q75 = np.quantile(values, [.25, .5, .75])
            axes[0].errorbar(index, med, yerr=[[med-q25], [q75-med]], fmt="o", c="#172C3C", capsize=4)
        bottom = 0
        for status, color in (("ok", BLUE), ("alert", "#DDCC77"), ("failed", ORANGE), ("skipped", "#DDDDDD")):
            count = sum(r["status"] == status or (status == "failed" and r["status"] == "reference_failed") for r in subset)
            axes[1].bar(index, count, bottom=bottom, color=color, label=status.capitalize() if index == 0 else None)
            bottom += count
    for ax in axes:
        ax.set_xticks(range(len(SCENARIOS)), [LABELS[s] for s in SCENARIOS], rotation=25, ha="right", rotation_mode="anchor")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Plant median hidden-point P95 / width")
    axes[0].set_title("a  Recovery error by plant", loc="left", fontweight="bold")
    axes[1].set_ylabel("Leaf cases")
    axes[1].set_title("b  All cases, including failures", loc="left", fontweight="bold")
    axes[1].legend(ncol=4, loc="lower right", bbox_to_anchor=(1., 1.09), fontsize=7)
    fig.text(.08, .035, "Points: plants; black: median and IQR across plants. Alerts are fixed engineering thresholds.", fontsize=8)
    save_figure(fig, destination, qa_scripts)


def main(argv=None):
    """读取指定留出结果，导出汇总图与案例诊断。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--qa-scripts", type=Path)
    parser.add_argument("--top-per-scenario", type=int, default=2)
    parser.add_argument("--case", action="append", default=[], help="Also show plant:leaf:scenario, including a repaired case")
    args = parser.parse_args(argv)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    report = json.loads((args.root / "holdout_benchmark.json").read_text(encoding="utf-8"))
    folder = args.root / "figures"
    folder.mkdir(exist_ok=True)
    write_contract = {"question": "Where does missing-data reconstruction fail?",
        "archetype": "image plate + quant", "backend": "python",
        "summary_roles": ["plant-level error distribution", "full case outcome counts"],
        "case_roles": ["mask and broad geometry", "orthogonal view", "hidden-point errors"],
        "selection": f"All fit exceptions and {args.top_per_scenario} largest held-point P95/width alerts per scenario, plus explicitly requested cases; all rows retained in statistics",
        "error_bars": "IQR of plant-level leaf medians, not a confidence interval"}
    (folder / "figure_contract.json").write_text(json.dumps(write_contract, indent=2), encoding="utf-8")
    summary_figure(report, folder / "holdout_summary", args.qa_scripts)
    rows = report["records"]
    chosen = [r for r in rows if r["status"] == "failed" and "artifact" in r]
    for selection in args.case:
        plant, lid, scenario = selection.split(":")
        matches = [r for r in rows if (r["plant"], r["leaf_id"], r["scenario"]) == (plant, int(lid), scenario)]
        if not matches or matches[0]["status"] not in ("ok", "alert", "failed"):
            raise ValueError(f"No renderable case: {selection}")
        chosen.append(matches[0])
    for scenario in SCENARIOS:
        subset = [r for r in rows if r["scenario"] == scenario and r["status"] == "alert"]
        chosen.extend(sorted(subset, key=lambda r: r["held_p95_over_width"], reverse=True)[:args.top_per_scenario])
    seen = set()
    entries = []
    for row in chosen:
        key = f"{row['plant']}_leaf_{row['leaf_id']:02d}_{row['scenario']}"
        if key in seen:
            continue
        seen.add(key)
        case_figure(args.root, row, folder / key, args.qa_scripts)
        entries.append(dict(plant=row["plant"], leaf_id=row["leaf_id"], scenario=row["scenario"], status=row["status"], figure=key+".png"))
    (folder / "case_selection.json").write_text(json.dumps(entries, indent=2), encoding="utf-8")
    html = '<meta charset="utf-8"><title>Holdout diagnostics</title><h1>Holdout diagnostics</h1>'
    html += '<p>All cases are in <a href="../per_leaf_errors.csv">per_leaf_errors.csv</a>. Images show fit failures and ranked worst examples.</p>'
    html += '<img src="holdout_summary.png" style="max-width:100%">'
    for entry in entries:
        html += f'<h2>{entry["plant"]} leaf {entry["leaf_id"]} {entry["scenario"]}</h2><img src="{entry["figure"]}" style="max-width:100%">'
    (folder / "index.html").write_text(html, encoding="utf-8")
    print(f"Rendered summary and {len(entries)} failure/alert examples")
