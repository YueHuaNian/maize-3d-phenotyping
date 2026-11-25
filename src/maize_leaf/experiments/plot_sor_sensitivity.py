"""读取已有 SOR 标定结果，生成去噪参数敏感性诊断图。

Generate 4-panel publication-quality SOR parameter sensitivity diagnostic figure."""
import argparse
import csv
import json
import os
from pathlib import Path

# 实验绘图（去噪阶段）：SOR 统计滤波参数敏感性四联图
ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_grid(csv_path):
    """读取参数网格 CSV，并将指标列转换为数值。"""
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in list(r.keys()):
            if k != "method":
                r[k] = float(r[k])
    return rows


def main(argv=None):
    """读取指定标定目录，绘制 SOR 参数诊断图。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/sor_sensitivity", help="验证结果目录")
    out_dir = parser.parse_args(argv).root
    rows = load_grid(out_dir / "sor_grid_results.csv")
    report = json.loads((out_dir / "sor_sensitivity_report.json").read_text(encoding="utf-8"))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 220,
    })

    palette = {
        "pcl_std": "#0072B2",
        "med_mad": "#D55E00",
        "log_mad": "#009E73",
        "rel_med": "#CC79A7",
    }
    labels_m = {
        "pcl_std": r"PCL ($\mu + \alpha\sigma$)",
        "med_mad": r"Robust MAD ($m + \alpha\,\sigma_{\mathrm{MAD}}$)",
        "log_mad": r"Log-MAD ($\log m + \alpha\,\sigma_{\log}$)",
        "rel_med": r"Relative median ($\gamma\,m$)",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6), constrained_layout=True)
    ks = sorted({int(r["k"]) for r in rows if r["method"] in palette})
    cur = next(r for r in rows if r["method"] == "current_impl")

    # Panel (a): Effect of k on plant-mean MCC (best threshold per k) + clustered vs isolated trade-off
    ax = axes[0, 0]
    for m, col in palette.items():
        y_mcc, y_std = [], []
        for kv in ks:
            sub = [r for r in rows if r["method"] == m and int(r["k"]) == kv]
            best = max(sub, key=lambda x: x["l2_mcc_plant_mean"])
            y_mcc.append(best["l2_mcc_plant_mean"])
            y_std.append(best["l2_mcc_plant_std"])
        ax.plot(ks, y_mcc, marker="o", lw=1.8, ms=4.5, color=col, label=labels_m[m])
    ax.axhline(cur["l2_mcc_plant_mean"], color="#444444", ls="--", lw=1.2,
               label=f"Current impl (k=12, MCC={cur['l2_mcc_plant_mean']:.3f})")
    ax.axvspan(8, 12, color="#E69F00", alpha=0.14, label="Recommended k zone (8–12)")
    ax.set_xlabel("Neighborhood size $k$")
    ax.set_ylabel("Plant-mean MCC (Level 2 injection)")
    ax.set_title("a  Effect of $k$ on Outlier Detection (14 Plants / 90 Leaves)")
    ax.set_xticks(ks)
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.legend(fontsize=8, loc="lower left")

    # Panel (b): Trade-off at k=8 — Real-scan Natural Off-surface Recall vs Distal-10% Tip Clean FPR
    ax = axes[0, 1]
    for m, col in palette.items():
        sub = sorted([r for r in rows if r["method"] == m and int(r["k"]) == 8],
                     key=lambda x: x["l1_tip10_clean_fpr_pct"])
        xs = [r["l1_tip10_clean_fpr_pct"] for r in sub]
        ys = [r["l1_nat_offsurf_recall_pct"] for r in sub]
        ax.plot(xs, ys, marker="o", lw=1.8, ms=4.5, color=col, label=labels_m[m])
        # Annotate key operating points
        for r in sub:
            if (m == "med_mad" and r["param"] in (3.5, 4.0)) or \
               (m == "pcl_std" and r["param"] in (1.5, 2.0)) or \
               (m == "rel_med" and r["param"] in (2.0, 2.2)) or \
               (m == "log_mad" and r["param"] == 2.5):
                ax.annotate(f"{r['param']:g}", (r["l1_tip10_clean_fpr_pct"], r["l1_nat_offsurf_recall_pct"]),
                            textcoords="offset points", xytext=(4, -4), fontsize=7.5, color=col)
    ax.scatter([cur["l1_tip10_clean_fpr_pct"]], [cur["l1_nat_offsurf_recall_pct"]],
               marker="X", s=75, color="#111111", zorder=6, label="Current impl (k=12)")
    ax.set_xlabel("Real-scan distal-10% tip clean FPR (%)")
    ax.set_ylabel("Real-scan off-surface (>5 spacing) recall (%)")
    ax.set_title("b  Real-Scan Denoising vs. Tip Erosion Trade-off ($k=8$)")
    ax.set_xlim(0, 9.5)
    ax.set_ylim(20, 75)
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=8, loc="lower right")

    # Panel (c): Isolated vs Clustered recall as a function of k (using med_mad a=3.5 & pcl_std a=1.5)
    ax = axes[1, 0]
    for m, p_fix, col, ls in [
        ("med_mad", 3.5, "#D55E00", "-"),
        ("pcl_std", 1.5, "#0072B2", "-"),
    ]:
        sub = sorted([r for r in rows if r["method"] == m and abs(r["param"] - p_fix) < 1e-6],
                     key=lambda x: x["k"])
        x_k = [int(r["k"]) for r in sub]
        ax.plot(x_k, [100 * r["l2_recall_isolated"] for r in sub], marker="o", lw=1.7, ms=4,
                color=col, ls="-", label=f"{m} ({p_fix}) isolated recall")
        ax.plot(x_k, [100 * r["l2_recall_clustered"] for r in sub], marker="s", lw=1.7, ms=4,
                color=col, ls="--", label=f"{m} ({p_fix}) 3–5pt cluster recall")
    ax.axvspan(8, 12, color="#E69F00", alpha=0.14)
    ax.set_xlabel("Neighborhood size $k$")
    ax.set_ylabel("Injected outlier recall (%)")
    ax.set_title("c  Isolated Halo vs. Clustered Micro-Spike Recall vs. $k$")
    ax.set_xticks(ks)
    ax.set_ylim(0, 95)
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.legend(fontsize=8, loc="upper right")

    # Panel (d): Level 3A End-to-End Reconstruction Phenotype Error (14 plant-stratified noisy leaves)
    ax = axes[1, 1]
    l3a = report.get("level3a_noisy_14leaf", [])
    if l3a:
        # Sort by area error (putting baselines first for clarity)
        order = [r for r in l3a if r["method"] in ("no_sor", "current_impl")] + \
                sorted([r for r in l3a if r["method"] not in ("no_sor", "current_impl")],
                       key=lambda x: (x["failed"], x["med_area_err_pct"] or 999))
        y_pos = np.arange(len(order))
        h = 0.26
        short_names = [
            f"{r['label']} ({r['passed']}/{r['n_leaves']})" for r in order
        ]
        area_e = [r["med_area_err_pct"] or 0 for r in order]
        len_e = [r["med_length_err_pct"] or 0 for r in order]
        wid_e = [r["med_width_err_pct"] or 0 for r in order]

        ax.barh(y_pos - h, area_e, height=h, color="#D55E00", alpha=0.88, label="Area rel. err. (%)")
        ax.barh(y_pos, len_e, height=h, color="#0072B2", alpha=0.88, label="Length rel. err. (%)")
        ax.barh(y_pos + h, wid_e, height=h, color="#009E73", alpha=0.88, label="Width rel. err. (%)")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_names, fontsize=7.8)
        ax.invert_yaxis()
        ax.set_xlabel("Median relative phenotype error vs. clean ref (%)")
        ax.set_title("d  Level 3A End-to-End Noisy Reconstruction (14 Plants)")
        ax.grid(axis="x", alpha=0.25, ls=":")
        ax.legend(fontsize=8, loc="lower right")

    out_png = out_dir / "sor_sensitivity_analysis.png"
    fig.savefig(out_png, dpi=240)
    plt.close(fig)
    print(f"Saved figure: {out_png}")
