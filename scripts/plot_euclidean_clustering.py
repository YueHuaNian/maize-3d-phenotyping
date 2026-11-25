"""Generate 4-panel publication-quality Euclidean clustering parameter sensitivity figure."""
import csv
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_grid(csv_path):
    with open(csv_path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in list(r.keys()):
            if k not in ("r_family", "ms_family"):
                r[k] = float(r[k])
    return rows


def main():
    out_dir = ROOT / "outputs" / "euclidean_cluster_sensitivity"
    rows = load_grid(out_dir / "euclidean_cluster_grid.csv")
    report = json.loads((out_dir / "euclidean_cluster_report.json").read_text(encoding="utf-8"))

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.2,
        "axes.titlesize": 10.2,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 220,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.0), constrained_layout=True)

    # Helper to filter fixed min_size = max(20, 1%)
    def f_m20(r):
        return r["ms_family"] == "hybrid_min" and int(r["n_abs"]) == 20 and abs(r["rho_frac"] - 0.01) < 1e-9

    cur = report["current_impl_l1_l2"]

    # ------------------------------------------------------------------
    # Panel (a): Effect of spacing multiplier alpha_sp across radius families
    # ------------------------------------------------------------------
    ax = axes[0, 0]
    series_a = [
        ("pure_sp", None, "#0072B2", "o", "-", r"Pure spacing ($\alpha_{\mathrm{sp}}\,d_{\mathrm{sp}}$)"),
        ("hybrid_robD", 0.02, "#D55E00", "s", "-", r"Robust hybrid ($\max(\alpha_{\mathrm{sp}}d_{\mathrm{sp}},\,0.02\,D_{\mathrm{rob}})$)"),
        ("hybrid_robD", 0.03, "#009E73", "^", "--", r"Robust hybrid ($\max(\alpha_{\mathrm{sp}}d_{\mathrm{sp}},\,0.03\,D_{\mathrm{rob}})$)"),
        ("hybrid", 0.03, "#CC79A7", "D", ":", r"Raw-ptp hybrid ($\max(\alpha_{\mathrm{sp}}d_{\mathrm{sp}},\,0.03\,D_{\mathrm{ptp}})$)"),
    ]
    for fam, ad, col, mk, ls, lbl in series_a:
        sub = [
            r for r in rows
            if r["r_family"] == fam and f_m20(r)
            and (ad is None or abs(r["alpha_d"] - ad) < 1e-9)
            and r["alpha_sp"] <= 12.0
        ]
        sub.sort(key=lambda x: x["alpha_sp"])
        xs = [r["alpha_sp"] for r in sub]
        ys = [r["l2_mcc_plant_mean"] for r in sub]
        ax.plot(xs, ys, marker=mk, lw=1.8, ms=5.0, color=col, ls=ls, label=lbl)

    ax.scatter([5.0], [cur["l2_mcc_plant_mean"]], marker="X", s=90, color="#111111", zorder=7,
               label=f"Current impl (5.0, 0.03D; MCC={cur['l2_mcc_plant_mean']:.3f})")
    ax.axvspan(4.5, 5.5, color="#E69F00", alpha=0.14, label=r"Optimal $\alpha_{\mathrm{sp}} \approx 5$ zone")
    ax.set_xlabel(r"Spacing multiplier $\alpha_{\mathrm{sp}}$  (fixed $N_{\min} = \max(20,\,1\%N)$)")
    ax.set_ylabel("Plant-mean MCC (Level 2 stress test)")
    ax.set_title(r"a  Effect of Radius Multiplier $\alpha_{\mathrm{sp}}$ Across Formulations")
    ax.set_xticks([2.5, 3, 4, 5, 6, 8, 10, 12])
    ax.set_ylim(0.35, 0.95)
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.legend(fontsize=7.4, loc="upper right")

    # ------------------------------------------------------------------
    # Panel (b): Raw ptp D vs Robust D_rob under far-outlier inflation (alpha_sp = 5.0)
    # ------------------------------------------------------------------
    ax = axes[0, 1]
    b_configs = [
        ("Raw-ptp\n(5, 0.03D)\n[current]", "hybrid", 5.0, 0.03),
        ("Raw-ptp\n(5, 0.02D)", "hybrid", 5.0, 0.02),
        ("Robust D\n(5, 0.03D_rob)", "hybrid_robD", 5.0, 0.03),
        ("Robust D\n(5, 0.02D_rob)", "hybrid_robD", 5.0, 0.02),
        ("Pure sp\n(5*sp)", "pure_sp", 5.0, 0.0),
    ]
    b_rows = []
    for lbl, fam, asp, ad in b_configs:
        match = next(
            r for r in rows
            if r["r_family"] == fam and abs(r["alpha_sp"] - asp) < 1e-9
            and abs(r["alpha_d"] - ad) < 1e-9 and f_m20(r)
        )
        b_rows.append((lbl, match))

    x_pos = np.arange(len(b_rows))
    w = 0.20
    ax.bar(x_pos - 1.5 * w, [100 * r["l2_nofar_recall_close"] for _, r in b_rows], width=w,
           color="#56B4E9", edgecolor="white", label="Close patch recall (no far outlier)")
    ax.bar(x_pos - 0.5 * w, [100 * r["l2_recall_close"] for _, r in b_rows], width=w,
           color="#0072B2", edgecolor="white", label="Close patch recall (with far outlier)")
    ax.bar(x_pos + 0.5 * w, [100 * r["l2_nofar_recall_med"] for _, r in b_rows], width=w,
           color="#F0E442", edgecolor="#888888", lw=0.4, label="Medium patch recall (no far outlier)")
    ax.bar(x_pos + 1.5 * w, [100 * r["l2_recall_med"] for _, r in b_rows], width=w,
           color="#D55E00", edgecolor="white", label="Medium patch recall (with far outlier)")

    for i, (_, r) in enumerate(b_rows):
        infl = r["rad_inflation_p95"]
        tag = f"P95 infl\n{infl:.2f}x"
        col_t = "#B22222" if infl > 1.15 else "#2E7D32"
        ax.text(i, 102.0, tag, ha="center", va="bottom", fontsize=7.2, color=col_t, fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([x[0] for x in b_rows], fontsize=8.0)
    ax.set_ylabel("Detached noise-patch recall (%)")
    ax.set_ylim(0, 138)
    ax.set_title(r"b  Diameter-Inflation Ablation ($\alpha_{\mathrm{sp}}=5$, Paired Far Outlier)")
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.legend(fontsize=7.3, loc="upper center", ncol=2, columnspacing=1.0)

    # ------------------------------------------------------------------
    # Panel (c): Secondary cluster min-size rule trade-off at hybrid_robD(5, 0.02)
    # ------------------------------------------------------------------
    ax = axes[1, 0]
    sub_ms = [r for r in rows if r["r_family"] == "hybrid_robD" and r["alpha_sp"] == 5.0 and abs(r["alpha_d"] - 0.02) < 1e-9]
    ms_colors = {
        "main_only": "#CC79A7",
        "abs_only": "#0072B2",
        "frac_only": "#009E73",
        "hybrid_min": "#D55E00",
    }
    ms_markers = {
        "main_only": "X",
        "abs_only": "s",
        "frac_only": "^",
        "hybrid_min": "o",
    }
    ms_labels = {
        "main_only": "Main component only",
        "abs_only": r"Absolute only ($n_{\mathrm{abs}}$)",
        "frac_only": r"Fractional only ($\rho N$)",
        "hybrid_min": r"Hybrid ($\max(n_{\mathrm{abs}},\,\rho N)$)",
    }

    for ms_fam in ("abs_only", "frac_only", "hybrid_min"):
        pts = [r for r in sub_ms if r["ms_family"] == ms_fam]
        xs = [r["l2_det_tip_retention_pct"] for r in pts]
        ys = [r["l2_mcc_plant_mean"] for r in pts]
        ax.scatter(xs, ys, s=64,
                   color=ms_colors[ms_fam], marker=ms_markers[ms_fam],
                   edgecolor="white", lw=0.6, zorder=5, label=ms_labels[ms_fam])

    # Annotate key points clearly across zoomed x in [97.0, 100.0]
    annots = [
        ("abs_only", 10, 0.0, "abs=10", (-46, 2)),
        ("abs_only", 30, 0.0, "abs=30", (-46, 8)),
        ("abs_only", 50, 0.0, "abs=50 (maxLeaf=4.8%)", (-12, 16)),
        ("frac_only", 1, 0.005, "0.5%", (-36, 4)),
        ("frac_only", 1, 0.01, "1%", (-26, 2)),
        ("frac_only", 1, 0.02, "2%", (-28, -2)),
        ("frac_only", 1, 0.03, "3%", (-28, -4)),
        ("frac_only", 1, 0.05, "5%", (8, -6)),
        ("hybrid_min", 10, 0.005, "(10, 0.5%)", (-60, 0)),
        ("hybrid_min", 15, 0.01, "(15, 1%)", (-52, 4)),
        ("hybrid_min", 20, 0.01, "(20, 1%) [peak]", (-88, 6)),
        ("hybrid_min", 30, 0.02, "(30, 2%)", (-12, -18)),
        ("hybrid_min", 50, 0.02, "(50, 2%)", (10, -14)),
    ]
    for mf, na, rf, txt, offs in annots:
        m = [r for r in sub_ms if r["ms_family"] == mf and int(r["n_abs"]) == na and abs(r["rho_frac"] - rf) < 1e-9]
        if m:
            r = m[0]
            ax.annotate(
                txt,
                (r["l2_det_tip_retention_pct"], r["l2_mcc_plant_mean"]),
                textcoords="offset points", xytext=offs,
                fontsize=7.3, color=ms_colors[mf],
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#888888", alpha=0.7),
            )

    mo = next(r for r in sub_ms if r["ms_family"] == "main_only")
    ax.text(
        0.03, 0.05,
        f"Off-scale left — Main component only:\n"
        f"Tip retention = {mo['l2_det_tip_retention_pct']:.1f}%, MCC = {mo['l2_mcc_plant_mean']:.3f}\n"
        f"Real-scan worst-leaf clean loss = {mo['l1_clean_fpr_leaf_max_pct']:.1f}%",
        transform=ax.transAxes, fontsize=7.5, color="#882255",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FFF0F5", edgecolor="#CC79A7", lw=0.8),
    )

    ax.set_xlim(97.0, 100.15)
    ax.set_ylim(0.34, 0.93)
    ax.set_xlabel("Genuine occlusion-detached tip retention (%)")
    ax.set_ylabel("Plant-mean MCC (Level 2 stress test)")
    ax.set_title(r"c  Secondary-Cluster Threshold Trade-Off ($\max(5d_{\mathrm{sp}}, 0.02D_{\mathrm{rob}})$)")
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=7.7, loc="center left")

    # ------------------------------------------------------------------
    # Panel (d): Level 3A End-to-End Perturbed Reconstruction Error (14 Plants)
    # ------------------------------------------------------------------
    ax = axes[1, 1]
    l3a = report.get("level3a_perturbed_14leaf", [])
    keep_order = [
        "main_only_cur",
        "pure_sp_3_m20_1p",
        "no_cluster",
        "robD_5_003_m20_1p",
        "pure_sp_8_m20_1p",
        "current_impl",
        "pure_sp_4_m20_1p",
        "best_hybrid",
        "pure_sp_5_m20_1p",
        "best_hybrid_robD",
    ]
    pretty = {
        "main_only_cur": "Main-only (5sp, 0.03D)",
        "pure_sp_3_m20_1p": "Pure sp = 3.0",
        "no_cluster": "No clustering (SOR only)",
        "robD_5_003_m20_1p": "Robust D (5sp, 0.03D_rob)",
        "pure_sp_8_m20_1p": "Pure sp = 8.0",
        "current_impl": "Current (5sp, 0.03D_ptp)",
        "pure_sp_4_m20_1p": "Pure sp = 4.0",
        "best_hybrid": "Hybrid (5sp, 0.015D_ptp)",
        "pure_sp_5_m20_1p": "Pure sp = 5.0",
        "best_hybrid_robD": "Robust D (5sp, 0.02D_rob)",
    }
    by_lbl = {r["label"]: r for r in l3a}
    ordered = [by_lbl[k] for k in keep_order if k in by_lbl]

    y_pos = np.arange(len(ordered))
    h = 0.26
    area_e = [r["med_area_err_pct"] for r in ordered]
    len_e = [r["med_length_err_pct"] for r in ordered]
    wid_e = [r["med_width_err_pct"] for r in ordered]

    ax.barh(y_pos - h, area_e, height=h, color="#D55E00", alpha=0.88, label="Area rel. err. (%)")
    ax.barh(y_pos, len_e, height=h, color="#0072B2", alpha=0.88, label="Length rel. err. (%)")
    ax.barh(y_pos + h, wid_e, height=h, color="#009E73", alpha=0.88, label="Width rel. err. (%)")

    for i, r in enumerate(ordered):
        ax.text(r["med_area_err_pct"] + 0.18, i - h, f"{r['med_area_err_pct']:.2f}%",
                va="center", fontsize=7.0, color="#8B3A00")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([pretty.get(r["label"], r["label"]) for r in ordered], fontsize=8.0)
    ax.invert_yaxis()
    ax.set_xlim(0, 15.6)
    ax.set_xlabel("Median relative phenotype error vs. clean ref (%)")
    ax.set_title("d  Level 3A End-to-End Perturbed Reconstruction (14 Plants)")
    ax.grid(axis="x", alpha=0.25, ls=":")
    ax.legend(fontsize=7.8, loc="lower right")

    out_png = out_dir / "euclidean_cluster_analysis.png"
    fig.savefig(out_png, dpi=240)
    plt.close(fig)
    print(f"Saved figure: {out_png}")


if __name__ == "__main__":
    main()
