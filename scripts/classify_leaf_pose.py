"""Stage E: Cross-plant 3-class leaf pose classification, LOPO stability, entropy ablation & plots.

Reads frozen Stage D phenotypes from outputs/phenotypes_v2/leaf_traits.csv (strictly read-only),
runs:
  1) Primary model: RobustScaler + K-means (k=3, n_init=50, seed=42) on pure morphology
     features M4 = [stem_leaf_angle_deg, tip_gain, droop_ratio, midrib_curvature_total, projection_ratio]
     (height strictly excluded!).
  2) Comparators: Bezdek FCM (m=2.0) and Diagonal GMM (k=3), plus quality-weighted (q_i) sensitivity.
  3) Strict 14-plant Leave-One-Plant-Out (LOPO) validation + 200-permutation null test (H2).
  4) Feature ablation M1 -> M5 testing whether normal entropy H_N adds stability/gain (H3).
  5) Post-hoc canopy height stratification analysis P(C | h) on norm_base_height (after clustering).
  6) Publication-quality 4-panel diagnostic figure.
"""
import argparse
import csv
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".vendor")]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kruskal, spearmanr

from maize_leaf.classification import (
    ABLATION_MODELS,
    CLASS_NAMES,
    adjusted_rand_index,
    lopo_validate,
    normalized_mutual_info,
)
from maize_leaf.io import write_json


def posthoc_height_analysis(labels, heights, plant_ids):
    """Post-hoc vertical stratification analysis P(C | h) AFTER pure-morphology clustering."""
    labels = np.asarray(labels, dtype=int)
    h = np.asarray(heights, dtype=float)
    plants = np.asarray(plant_ids)

    strata = [
        ("lower_0_33", (h >= 0.0) & (h < 1.0 / 3.0)),
        ("middle_33_67", (h >= 1.0 / 3.0) & (h < 2.0 / 3.0)),
        ("upper_67_100", (h >= 2.0 / 3.0) & (h <= 1.0 + 1e-9)),
    ]
    by_stratum = {}
    for name, mask in strata:
        n = int(np.sum(mask))
        probs = {CLASS_NAMES[c]: (float(np.mean(labels[mask] == c)) if n > 0 else float("nan")) for c in range(3)}
        by_stratum[name] = {"n_leaves": n, **probs}

    # Mean height per class
    class_heights = {
        CLASS_NAMES[c]: {
            "n": int(np.sum(labels == c)),
            "mean_h": float(np.mean(h[labels == c])) if np.any(labels == c) else float("nan"),
            "median_h": float(np.median(h[labels == c])) if np.any(labels == c) else float("nan"),
        }
        for c in range(3)
    }

    groups = [h[labels == c] for c in range(3) if np.sum(labels == c) >= 2]
    kw_stat, kw_p = kruskal(*groups) if len(groups) >= 2 else (float("nan"), float("nan"))
    # Ordinal correlation: convergent=0, expanded=1, drooping=2 vs height (negative => upper more erect)
    sp_res = spearmanr(labels, h)

    # Plant-level independent-unit check (n=14 plants): within-plant Spearman(class, height)
    plant_rhos = []
    for pl in sorted(set(plants)):
        m = plants == pl
        if len(set(labels[m])) >= 2 and np.sum(m) >= 3:
            r = float(spearmanr(labels[m], h[m]).statistic)
            if np.isfinite(r):
                plant_rhos.append(r)

    return {
        "by_height_stratum": by_stratum,
        "height_by_class": class_heights,
        "kruskal_wallis_H": float(kw_stat),
        "kruskal_wallis_p_leaf_level": float(kw_p),
        "spearman_class_vs_height_leaf_level": float(sp_res.statistic),
        "spearman_p_leaf_level": float(sp_res.pvalue),
        "within_plant_spearman_n_plants_with_ge2_classes": len(plant_rhos),
        "within_plant_spearman_median": float(np.median(plant_rhos)) if plant_rhos else float("nan"),
        "within_plant_negative_count": int(sum(r < 0 for r in plant_rhos)),
        "note": "Leaf-level p-values are descriptive (leaves nested within 14 plants); see within-plant summary for plant-level consistency.",
    }


def make_diagnostic_figure(out_png, primary_out, ablation_results, height_summary, heights):
    """Generate 4-panel Stage E classification diagnostic figure."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.2,
        "axes.titlesize": 10.2,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "figure.dpi": 220,
    })

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.0), constrained_layout=True)
    colors = ["#0072B2", "#009E73", "#D55E00"]  # convergent, expanded, drooping
    labels = primary_out["full_labels"]
    Z = primary_out["Z_full"]
    r_ratio = primary_out["distance_ratio"]

    # Panel (a): PCA 2D projection (visualization only!) of M4 scaled features
    ax = axes[0, 0]
    Zc = Z - np.mean(Z, axis=0)
    _, svals, vt = np.linalg.svd(Zc, full_matrices=False)
    var_exp = 100.0 * (svals ** 2) / np.sum(svals ** 2)
    pcs = Zc @ vt[:2].T
    c_pcs = (primary_out["centers_scaled"] - np.mean(Z, axis=0)) @ vt[:2].T

    for c in range(3):
        m = labels == c
        amb = m & (r_ratio > 0.80)
        clr = m & (r_ratio <= 0.80)
        ax.scatter(pcs[clr, 0], pcs[clr, 1], s=42, color=colors[c], alpha=0.88,
                   edgecolor="white", lw=0.5, label=f"{CLASS_NAMES[c]} (n={int(m.sum())})")
        if np.any(amb):
            ax.scatter(pcs[amb, 0], pcs[amb, 1], s=50, facecolors="none", edgecolors=colors[c],
                       lw=1.3, ls="--")
        ax.scatter([c_pcs[c, 0]], [c_pcs[c, 1]], marker="X", s=110, color=colors[c],
                   edgecolor="#111111", lw=0.8, zorder=7)
    ax.scatter([], [], s=50, facecolors="none", edgecolors="#444444", lw=1.2,
               label=f"Ambiguous ($r_i>0.80$, n={int(np.sum(r_ratio>0.80))})")
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1f}% variance)")
    ax.set_title("a  Pure-Morphology K-means ($M_4$, 90 Leaves; PCA Display Only)")
    ax.grid(alpha=0.25, ls=":")
    ax.legend(fontsize=7.7, loc="best")

    # Panel (b): Standardized cluster centroid profiles across M4 features
    ax = axes[0, 1]
    feat_short = [r"$\theta$ (angle)", r"$g$ (tip gain)", r"$d$ (droop)", r"$K_{\mathrm{mid}}$ (curv)", r"$\rho$ (proj)"]
    x_f = np.arange(len(feat_short))
    w = 0.25
    C = primary_out["centers_scaled"]
    for c in range(3):
        ax.bar(x_f + (c - 1) * w, C[c], width=w, color=colors[c], alpha=0.88, label=CLASS_NAMES[c])
    ax.axhline(0.0, color="#444444", lw=0.8)
    ax.set_xticks(x_f)
    ax.set_xticklabels(feat_short, fontsize=8.3)
    ax.set_ylabel("Robust-scaled centroid coordinate")
    ax.set_title("b  Cluster Centroid Profiles (Post-hoc Semantic Anchoring)")
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.legend(fontsize=8.0, loc="upper right")

    # Panel (c): Feature ablation M1..M5 (LOPO OOF ARI, Fold ARI, Silhouette & Switch rate)
    ax = axes[1, 0]
    m_keys = list(ABLATION_MODELS.keys())
    short_m = ["M1\n(θ)", "M2\n(θ,g,d)", "M3\n(+K)", "M4\n(+ρ)\n[main]", "M5\n(+H_N)"]
    xm = np.arange(len(m_keys))
    oof_aris = [ablation_results[k]["lopo_oof_ari"] for k in m_keys]
    fold_aris = [ablation_results[k]["lopo_fold_ari_mean"] for k in m_keys]
    sils = [ablation_results[k]["silhouette"] for k in m_keys]
    rand_p95 = [ablation_results[k]["random_perm_ari_p95"] for k in m_keys]

    ax.plot(xm, oof_aris, "o-", color="#D55E00", lw=2.0, ms=6, label="LOPO OOF ARI (held-out plant)")
    ax.plot(xm, fold_aris, "s--", color="#0072B2", lw=1.7, ms=5.5, label="LOPO fold-mean ARI (all 90)")
    ax.plot(xm, sils, "^:", color="#009E73", lw=1.7, ms=5.5, label="Silhouette score (descriptive)")
    ax.plot(xm, rand_p95, "x-", color="#777777", lw=1.1, ms=5, label="Permutation null P95 ARI")
    for i, val in enumerate(oof_aris):
        ax.annotate(f"{val:.3f}", (i, val), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7.6, color="#8B3A00", fontweight="bold")
    ax.set_xticks(xm)
    ax.set_xticklabels(short_m, fontsize=8.2)
    ax.set_ylim(-0.05, 1.08)
    ax.set_ylabel("Stability / separation score")
    ax.set_title(r"c  Feature Ablation $M_1 \rightarrow M_5$ & Entropy Increment Test ($H_3$)")
    ax.grid(axis="y", alpha=0.25, ls=":")
    ax.legend(fontsize=7.6, loc="lower left")

    # Panel (d): Post-hoc canopy vertical stratification P(C | h) (height NOT used in clustering)
    ax = axes[1, 1]
    st_keys = ["lower_0_33", "middle_33_67", "upper_67_100"]
    st_lbls = [
        f"Lower (0–33%)\nn={height_summary['by_height_stratum']['lower_0_33']['n_leaves']}",
        f"Middle (33–67%)\nn={height_summary['by_height_stratum']['middle_33_67']['n_leaves']}",
        f"Upper (67–100%)\nn={height_summary['by_height_stratum']['upper_67_100']['n_leaves']}",
    ]
    xs = np.arange(3)
    bottom = np.zeros(3)
    for c in range(3):
        vals = np.array([100.0 * height_summary["by_height_stratum"][k][CLASS_NAMES[c]] for k in st_keys])
        ax.bar(xs, vals, bottom=bottom, width=0.52, color=colors[c], alpha=0.88, label=CLASS_NAMES[c])
        for i, v in enumerate(vals):
            if v >= 8.0:
                ax.text(i, bottom[i] + 0.5 * v, f"{v:.0f}%", ha="center", va="center",
                        color="white", fontsize=8.5, fontweight="bold")
        bottom += vals

    ax.set_xticks(xs)
    ax.set_xticklabels(st_lbls, fontsize=8.5)
    ax.set_ylim(0, 114)
    ax.set_ylabel("Pose class proportion (%)")
    rho_h = height_summary["spearman_class_vs_height_leaf_level"]
    w_med = height_summary["within_plant_spearman_median"]
    ax.set_title(f"d  Post-hoc Height Stratification ($\\rho={rho_h:+.2f}$, plant med={w_med:+.2f})", fontsize=9.8)
    ax.legend(fontsize=8.0, loc="upper center", ncol=3)

    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traits-csv", type=Path, default=ROOT / "outputs/phenotypes_v2/leaf_traits.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/classification_v2")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t0 = time.time()
    with args.traits_csv.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    plant_ids = np.array([r["plant_id"] for r in rows])
    heights = np.array([float(r["norm_base_height"]) for r in rows])
    q_weights = np.array([float(r["quality_weight"]) for r in rows])

    def _mat(feat_list):
        return np.column_stack([[float(r[c]) for r in rows] for c in feat_list])

    # 1. Feature ablation M1..M5 with primary K-means (unweighted)
    ablation_summary = {}
    ablation_outs = {}
    print("=== Feature Ablation M1 -> M5 (K-means k=3, 14-plant LOPO) ===", flush=True)
    for m_name, feats in ABLATION_MODELS.items():
        out = lopo_validate(_mat(feats), plant_ids, feats, method="kmeans", seed=args.seed)
        ablation_outs[m_name] = out
        m = out["metrics"]
        ablation_summary[m_name] = m
        print(
            f"  {m_name:16s} | OOF_ARI={m['lopo_oof_ari']:.4f} (p={m['h2_permutation_p_value']:.4f}), "
            f"FoldARI={m['lopo_fold_ari_mean']:.4f}+/-{m['lopo_fold_ari_std']:.4f}, "
            f"Switch={m['lopo_oof_switch_rate_pct']:4.1f}%, Drift={m['centroid_drift_mean']:.3f}, "
            f"Sil={m['silhouette']:.3f}, Amb(0.8)={m['ambiguous_rate_r080_pct']:4.1f}%",
            flush=True,
        )

    # 2. Algorithm comparison on primary M4: K-means (unweighted & weighted) vs FCM vs GMM
    X_m4 = _mat(ABLATION_MODELS["M4_primary"])
    primary_out = ablation_outs["M4_primary"]
    algo_outs = {
        "kmeans_unweighted": primary_out,
        "kmeans_qweighted": lopo_validate(X_m4, plant_ids, ABLATION_MODELS["M4_primary"], method="kmeans", sample_weight=q_weights, seed=args.seed),
        "fcm_unweighted": lopo_validate(X_m4, plant_ids, ABLATION_MODELS["M4_primary"], method="fcm", seed=args.seed),
        "gmm_unweighted": lopo_validate(X_m4, plant_ids, ABLATION_MODELS["M4_primary"], method="gmm", seed=args.seed),
    }
    print("\n=== Algorithm & Quality-Weight Comparison on Primary M4 ===", flush=True)
    algo_summary = {}
    for name, out in algo_outs.items():
        m = dict(out["metrics"])
        m["ari_vs_kmeans_unweighted"] = adjusted_rand_index(primary_out["full_labels"], out["full_labels"])
        m["nmi_vs_kmeans_unweighted"] = normalized_mutual_info(primary_out["full_labels"], out["full_labels"])
        m["agreement_pct_vs_kmeans"] = 100.0 * float(np.mean(primary_out["full_labels"] == out["full_labels"]))
        algo_summary[name] = m
        print(
            f"  {name:18s} | OOF_ARI={m['lopo_oof_ari']:.4f}, FoldARI={m['lopo_fold_ari_mean']:.4f}, "
            f"Switch={m['lopo_oof_switch_rate_pct']:4.1f}%, Sil={m['silhouette']:.3f}, "
            f"vs_KM(ARI={m['ari_vs_kmeans_unweighted']:.3f}, agree={m['agreement_pct_vs_kmeans']:5.1f}%)",
            flush=True,
        )

    # 3. Formal Hypothesis H3 comparison: M5 (with entropy) vs M4 (without entropy)
    m4m = ablation_summary["M4_primary"]
    m5m = ablation_summary["M5_with_entropy"]
    h_m4 = posthoc_height_analysis(ablation_outs["M4_primary"]["full_labels"], heights, plant_ids)
    h_m5 = posthoc_height_analysis(ablation_outs["M5_with_entropy"]["full_labels"], heights, plant_ids)
    h3_delta = {
        "delta_lopo_oof_ari": m5m["lopo_oof_ari"] - m4m["lopo_oof_ari"],
        "delta_lopo_fold_ari": m5m["lopo_fold_ari_mean"] - m4m["lopo_fold_ari_mean"],
        "delta_switch_rate_pct": m5m["lopo_oof_switch_rate_pct"] - m4m["lopo_oof_switch_rate_pct"],
        "delta_silhouette": m5m["silhouette"] - m4m["silhouette"],
        "delta_ambiguous_r080_pct": m5m["ambiguous_rate_r080_pct"] - m4m["ambiguous_rate_r080_pct"],
        "ari_m4_vs_m5": adjusted_rand_index(ablation_outs["M4_primary"]["full_labels"], ablation_outs["M5_with_entropy"]["full_labels"]),
        "agreement_m4_vs_m5_pct": 100.0 * float(np.mean(ablation_outs["M4_primary"]["full_labels"] == ablation_outs["M5_with_entropy"]["full_labels"])),
        "m4_height_kruskal_H": h_m4["kruskal_wallis_H"],
        "m5_height_kruskal_H": h_m5["kruskal_wallis_H"],
    }
    print(
        f"\n=== Hypothesis H3 (Entropy Increment M5 - M4) ===\n"
        f"  dOOF_ARI={h3_delta['delta_lopo_oof_ari']:+.4f}, dFoldARI={h3_delta['delta_lopo_fold_ari']:+.4f}, "
        f"dSwitch={h3_delta['delta_switch_rate_pct']:+.2f}%, dSil={h3_delta['delta_silhouette']:+.4f}, "
        f"dAmb={h3_delta['delta_ambiguous_r080_pct']:+.2f}%, M4-M5 agree={h3_delta['agreement_m4_vs_m5_pct']:.1f}%",
        flush=True,
    )

    # 4. Post-hoc canopy height stratification on primary M4
    print("\n=== Post-hoc Height Stratification (Primary M4, Height Excluded from Clustering) ===", flush=True)
    for k, v in h_m4["by_height_stratum"].items():
        print(f"  {k:14s} (n={v['n_leaves']:2d}): conv={100*v['convergent']:5.1f}%, exp={100*v['expanded']:5.1f}%, droop={100*v['drooping']:5.1f}%", flush=True)
    print(f"  Spearman(class, h)={h_m4['spearman_class_vs_height_leaf_level']:+.3f}, "
          f"within-plant median rho={h_m4['within_plant_spearman_median']:+.3f} "
          f"({h_m4['within_plant_negative_count']}/{h_m4['within_plant_spearman_n_plants_with_ge2_classes']} plants negative)", flush=True)

    # 5. Export per-leaf and per-plant CSVs + JSON report + 4-panel figure
    args.output.mkdir(parents=True, exist_ok=True)
    leaf_out_rows = []
    km_lbl = primary_out["full_labels"]
    km_oof = primary_out["oof_labels"]
    km_r = primary_out["distance_ratio"]
    fcm_p = algo_outs["fcm_unweighted"]["full_proba"]
    fcm_lbl = algo_outs["fcm_unweighted"]["full_labels"]
    gmm_p = algo_outs["gmm_unweighted"]["full_proba"]
    gmm_lbl = algo_outs["gmm_unweighted"]["full_labels"]
    qw_lbl = algo_outs["kmeans_qweighted"]["full_labels"]

    for i, r in enumerate(rows):
        leaf_out_rows.append({
            "plant_id": r["plant_id"],
            "leaf_id": int(r["leaf_id"]),
            "norm_base_height": float(r["norm_base_height"]),
            "quality_weight": float(r["quality_weight"]),
            "stem_leaf_angle_deg": float(r["stem_leaf_angle_deg"]),
            "tip_gain": float(r["tip_gain"]),
            "droop_ratio": float(r["droop_ratio"]),
            "midrib_curvature_total": float(r["midrib_curvature_total"]),
            "projection_ratio": float(r["projection_ratio"]),
            "normal_entropy_b24": float(r["normal_entropy_b24"]),
            "cluster_kmeans": CLASS_NAMES[km_lbl[i]],
            "cluster_kmeans_id": int(km_lbl[i]),
            "cluster_kmeans_lopo_oof": CLASS_NAMES[km_oof[i]],
            "lopo_switched": bool(km_lbl[i] != km_oof[i]),
            "distance_ratio_r": float(km_r[i]),
            "ambiguous_r080": bool(km_r[i] > 0.80),
            "cluster_kmeans_qweighted": CLASS_NAMES[qw_lbl[i]],
            "cluster_fcm": CLASS_NAMES[fcm_lbl[i]],
            "fcm_prob_convergent": float(fcm_p[i, 0]),
            "fcm_prob_expanded": float(fcm_p[i, 1]),
            "fcm_prob_drooping": float(fcm_p[i, 2]),
            "cluster_gmm": CLASS_NAMES[gmm_lbl[i]],
            "gmm_prob_convergent": float(gmm_p[i, 0]),
            "gmm_prob_expanded": float(gmm_p[i, 1]),
            "gmm_prob_drooping": float(gmm_p[i, 2]),
        })

    def _write_csv(path, data):
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            w.writeheader()
            w.writerows(data)

    _write_csv(args.output / "leaf_classifications.csv", leaf_out_rows)

    plant_rows = []
    for pl in sorted(set(plant_ids)):
        sub = [x for x in leaf_out_rows if x["plant_id"] == pl]
        n = len(sub)
        plant_rows.append({
            "plant_id": pl,
            "leaf_count": n,
            "n_convergent": sum(x["cluster_kmeans"] == "convergent" for x in sub),
            "n_expanded": sum(x["cluster_kmeans"] == "expanded" for x in sub),
            "n_drooping": sum(x["cluster_kmeans"] == "drooping" for x in sub),
            "class_fraction_convergent": sum(x["cluster_kmeans"] == "convergent" for x in sub) / n,
            "class_fraction_expanded": sum(x["cluster_kmeans"] == "expanded" for x in sub) / n,
            "class_fraction_drooping": sum(x["cluster_kmeans"] == "drooping" for x in sub) / n,
            "ambiguous_fraction_r080": sum(x["ambiguous_r080"] for x in sub) / n,
            "lopo_switch_fraction": sum(x["lopo_switched"] for x in sub) / n,
        })
    _write_csv(args.output / "plant_class_fractions.csv", plant_rows)

    # Cluster raw physical centroid summaries for interpretation
    centroid_profiles = {}
    for c in range(3):
        m = km_lbl == c
        centroid_profiles[CLASS_NAMES[c]] = {
            "n_leaves": int(np.sum(m)),
            "centroid_raw": {
                fname: float(primary_out["centers_raw"][c, j])
                for j, fname in enumerate(ABLATION_MODELS["M4_primary"])
            },
            "median_raw": {
                fname: float(np.median([float(rows[i][fname]) for i in np.flatnonzero(m)]))
                for fname in ABLATION_MODELS["M4_primary"] + ["normal_entropy_b24", "norm_base_height"]
            },
        }

    report = {
        "traits_csv": str(args.traits_csv),
        "seed": args.seed,
        "n_plants": len(plant_rows),
        "n_leaves": len(rows),
        "primary_model": {
            "algorithm": "kmeans++ (k=3, n_init=50, max_iter=500)",
            "scaler": "RobustScaler (median / IQR)",
            "features": ABLATION_MODELS["M4_primary"],
            "height_excluded_from_clustering": True,
        },
        "primary_centroid_profiles": centroid_profiles,
        "ablation_m1_to_m5": ablation_summary,
        "algorithm_comparison_m4": algo_summary,
        "hypothesis_h3_entropy_increment": h3_delta,
        "posthoc_height_stratification_m4": h_m4,
        "posthoc_height_stratification_m5": h_m5,
        "elapsed_s": round(time.time() - t0, 2),
    }
    write_json(args.output / "classification_report.json", report)
    make_diagnostic_figure(
        args.output / "classification_analysis.png",
        primary_out, ablation_summary, h_m4, heights,
    )
    print(f"\nSaved Stage E outputs to {args.output.resolve()} ({report['elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
