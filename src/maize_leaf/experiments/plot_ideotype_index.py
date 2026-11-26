"""绘制阶段 F 理想株型结构符合度指数（I_A / I_L / I）与四项检验四联诊断图。

Four-panel publication-style diagnostic figure for Stage F ideotype conformity indices:
  (A) 14-plant ranking: composite I, architecture I_A, leaf morphology I_L;
  (B) 14 x 7 sub-index decomposition heatmap (S_env, S_light, S_comp, s_angle, s_proj, s_entropy, s_far_dz);
  (C) Monotonicity validation curves (lambda in [0, 1]) + perturbation Spearman rho distribution;
  (D) Weight sensitivity: alpha sweep + 100 Dirichlet weight draws.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def _configure_fonts():
    """优先选择清晰无衬线字体，兼容中英文标注。"""
    plt.rcParams.update({
        "font.sans-serif": ["DejaVu Sans", "Arial", "Microsoft YaHei", "SimHei"],
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
    })


def plot_ideotype_validation(report: dict, out_prefix: Path) -> list[Path]:
    """根据 index_report.json 生成四联诊断图（PNG + PDF + SVG）。"""
    _configure_fonts()
    plants = sorted(report["plant_rankings"], key=lambda r: r["rank"])
    pids = [p["plant_id"] for p in plants]
    n_p = len(plants)

    fig = plt.figure(figsize=(14.0, 9.8), dpi=300, facecolor="white")
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.18, 1.0], height_ratios=[1.12, 1.0],
                  wspace=0.26, hspace=0.34, left=0.115, right=0.96, top=0.90, bottom=0.07)

    # Panel A: 14-plant horizontal ranking bar + I_A / I_L markers
    ax_a = fig.add_subplot(gs[0, 0])
    y = np.arange(n_p)[::-1]
    vals_i = np.array([p["composite_index_i"] for p in plants])
    vals_ia = np.array([p["arch_index_ia"] for p in plants])
    vals_il = np.array([p["leaf_index_il"] for p in plants])

    ax_a.barh(y, vals_i, height=0.60, color="#2a7b5f", alpha=0.82, label="Composite I (0.5 I_A + 0.5 I_L)")
    ax_a.scatter(vals_ia, y, color="#1f77b4", marker="o", s=34, zorder=4, label="Architecture I_A")
    ax_a.scatter(vals_il, y, color="#d95f02", marker="D", s=28, zorder=4, label="Leaf morphology I_L")
    for yi, vi, via, vil in zip(y, vals_i, vals_ia, vals_il):
        x_txt = max(vi, via, vil) + 1.8
        ax_a.text(x_txt, yi, f"I={vi:.1f}", va="center", ha="left", fontsize=7.8, color="#17352b", fontweight="bold")
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([f"#{p['rank']:d} {p['plant_id']} (n={p['leaf_count']})" for p in plants])
    ax_a.set_xlim(0, 104)
    ax_a.set_ylim(-0.7, n_p - 0.1)
    ax_a.set_xlabel("Ideotype Conformity Score (0–100)")
    ax_a.set_title("A. 14-Plant Ideotype Conformity Ranking", loc="left", fontweight="bold", pad=22)
    ax_a.grid(axis="x", linestyle="--", alpha=0.35)
    ax_a.legend(loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=3, frameon=False, fontsize=7.6, handletextpad=0.4, columnspacing=0.9)

    # Panel B: 14 x 7 sub-index heatmap
    ax_b = fig.add_subplot(gs[0, 1])
    sub_keys = ["s_envelope", "s_light", "s_composition", "mean_s_angle", "mean_s_proj", "mean_s_entropy", "mean_s_far_dz"]
    sub_labels = ["Env\nS_env", "Light\nS_light", "Comp\nS_comp", "Angle\ns_θ", "Proj\ns_P", "Entropy\ns_H", "Far Δz\ns_Δz"]
    mat = np.array([[100.0 * p[k] for k in sub_keys] for p in plants], dtype=float)
    im = ax_b.imshow(mat, cmap="YlGn", vmin=20.0, vmax=100.0, aspect="auto")
    ax_b.set_xticks(np.arange(len(sub_keys)))
    ax_b.set_xticklabels(sub_labels, fontsize=8)
    ax_b.set_yticks(np.arange(n_p))
    ax_b.set_yticklabels(pids, fontsize=8)
    ax_b.axvline(2.5, color="#17352b", linewidth=1.3, linestyle="--")
    for r_i in range(n_p):
        for c_i in range(len(sub_keys)):
            v = mat[r_i, c_i]
            ax_b.text(c_i, r_i, f"{v:.0f}", ha="center", va="center", fontsize=7.2,
                      color="white" if v > 78 else "#17352b")
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.042, pad=0.02)
    cbar.set_label("Sub-index Score (0–100)", fontsize=8.5)
    ax_b.set_title("B. Architecture (3) & Leaf Five-Factor (4) Sub-indices", loc="left", fontweight="bold")

    # Panel C: Monotonicity trajectory
    ax_c = fig.add_subplot(gs[1, 0])
    mono = report["validations"]["monotonicity"]
    traj = mono["trajectory"]
    lam = np.asarray(traj["lambda"], dtype=float)
    ax_c.plot(lam, traj["I"], color="#14523d", linewidth=2.4, marker="o", markersize=3.5, label="Composite I")
    ax_c.plot(lam, traj["I_A"], color="#1f77b4", linewidth=1.8, linestyle="--", label="Architecture I_A")
    ax_c.plot(lam, traj["I_L"], color="#d95f02", linewidth=1.8, linestyle="-.", label="Leaf morphology I_L")
    ax_c.set_xlim(0.0, 1.0)
    ax_c.set_ylim(0, 105)
    ax_c.set_xlabel("Trajectory toward Ideal Smart-Canopy Target (λ)")
    ax_c.set_ylabel("Conformity Score (0–100)")
    status_txt = "PASS (all 7 sub-indices monotonic)" if mono["passed"] else "CHECK"
    ax_c.set_title(f"C. Monotonicity Validation — {status_txt}", loc="left", fontweight="bold")
    ax_c.grid(True, linestyle="--", alpha=0.35)
    ax_c.legend(loc="lower right", frameon=True, facecolor="white", fontsize=8.5)

    # Panel D: Robustness & Sensitivity (alpha sweep + MC perturbation + Dirichlet box/violin)
    ax_d = fig.add_subplot(gs[1, 1])
    sens = report["validations"]["sensitivity"]
    pert = report["validations"]["perturbation"]
    boot = report["validations"]["bootstrap"]

    alphas = [a["alpha"] for a in sens["alpha_sweep"]]
    alpha_rhos = [a["spearman_vs_primary"] for a in sens["alpha_sweep"]]
    alpha_taus = [a["kendall_vs_primary"] for a in sens["alpha_sweep"]]
    ax_d.plot(alphas, alpha_rhos, color="#2a7b5f", marker="o", linewidth=2.0, label="α sweep Spearman ρ")
    ax_d.plot(alphas, alpha_taus, color="#7570b3", marker="s", linewidth=1.6, linestyle="--", label="α sweep Kendall τ")
    ax_d.axvline(report["parameters"]["alpha"], color="#8a5700", linestyle=":", linewidth=1.3, label="Primary α = 0.50")

    # Summary callout box for perturbation, bootstrap, Dirichlet & review-leaf exclusion
    mc = pert["monte_carlo"]
    dir_r = sens["dirichlet_random_weights"]
    lolo = boot["leave_one_leaf_out"]
    excl = pert["structural_comparisons"]["exclude_support_review_leaves"]
    summary_text = (
        f"MC Perturbation (n={pert['n_trials']}):  ρ = {mc['spearman_i_median']:.3f} (min {mc['spearman_i_min']:.3f})\n"
        f"Dirichlet Weights (n={dir_r['n_draws']}):  ρ = {dir_r['spearman_i_median']:.3f} (min {dir_r['spearman_i_min']:.3f})\n"
        f"Within-Plant Leaf Boot:  ρ = {boot['within_plant_leaf_bootstrap']['spearman_i_median']:.3f} | LOLO min {lolo['spearman_i_min']:.3f}\n"
        f"Excl. {excl['excluded_leaves']} Review Leaves:  ρ = {excl['spearman_i']:.3f} (max ΔI={excl['max_abs_diff_i']:.2f})"
    )
    ax_d.text(
        0.03, 0.06, summary_text, transform=ax_d.transAxes, fontsize=7.8,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#f7faf8", edgecolor="#c8d8ce", alpha=0.95),
    )
    ax_d.set_xlim(-0.04, 1.04)
    ax_d.set_ylim(0.0, 1.06)
    ax_d.set_xlabel("Primary Architecture Weight α  (I = α I_A + (1 − α) I_L)")
    ax_d.set_ylabel("Rank Correlation vs. Primary (α = 0.50)")
    ax_d.set_title("D. Weight Sensitivity & Stability Summary", loc="left", fontweight="bold")
    ax_d.grid(True, linestyle="--", alpha=0.35)
    ax_d.legend(loc="center left", bbox_to_anchor=(0.01, 0.45), frameon=True, facecolor="white", fontsize=8)

    fig.suptitle(
        "Maize 3D Smart-Canopy Ideotype Conformity Index (Five-Factor Linked, 14 Plants / 90 Leaves)",
        fontsize=12.5, fontweight="bold", y=0.97, color="#123f33",
    )

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for ext in (".png", ".pdf", ".svg"):
        out_file = out_prefix.with_suffix(ext)
        fig.savefig(out_file, dpi=300 if ext == ".png" else None)
        saved.append(out_file)
    plt.close(fig)
    return saved


def main(argv=None):
    """读取 index_report.json 并输出四联科研诊断图。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/index_v2/index_report.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/index_v2/ideotype_index_validation.png")
    args = parser.parse_args(argv)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    saved = plot_ideotype_validation(report, args.output)
    for p in saved:
        print(p)
    return 0
