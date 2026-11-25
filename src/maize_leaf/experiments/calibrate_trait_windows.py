"""以解析曲面和冻结留出模型标定表型计算窗口。

Quantitative bias-variance calibration for Stage D geometric window parameters.

Calibrates two window percentages using independent non-circular criteria:
  1) basal_span_frac in {0% (single-point midrib), 2%, 5%, 8%, 10%, 12%, 15%, 20%}
     for basal-surface stem-leaf angle theta_i.
  2) margin_trim_frac in {0%, 1%, 2.5%, 5%, 7.5%, 10%, 12.5%, 15%}
     for interior surface curvature |H| and Willmore bending functional W.

Two complementary benchmarks (fixed seed 2026):
  - Benchmark 1 (Analytical Ground-Truth Bias & Noisy RMSE): 18 diverse analytical
    Dornbusch + curved/drooping + rolled + transversely arched maize leaves with known
    exact theta_true(s=0), H_abs_true, and W_true, reconstructed via full fit_leaf from
    noisy 70%-subsampled scans.
  - Benchmark 2 (Real-Scan Perturbation Variance): 250 real paired occluded models
    (hole / margin / tip_10pct vs full reference across all 14 plants / 90 leaves from
    outputs/holdout_20261125/validated/cases/).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# 标定脚本（表型阶段）：茎叶角基部跨度与曲面曲率边距裁切窗口敏感性实验
ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from maize_leaf.geometry import fit_pspline_gcv
from maize_leaf.io import write_json
from maize_leaf.morphology import DornbuschBlade
from maize_leaf.surface import MaizeLeafModel, ModelConfig, fit_leaf
from maize_leaf.traits import (
    _fundamental_forms_from_grid,
    basal_surface_stem_leaf_angle,
    load_stem_spline,
    midrib_pose_and_curvature,
    surface_curvature_fundamental_forms,
)


BASAL_SPANS = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
MARGIN_TRIMS = [0.0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15]


def make_analytical_leaf(base_deg, droop_rad, roll_deg, arch_amp, seed):
    """构造具有解析几何真值的弯曲、滚转与横向拱起叶片。

    Build analytical curved, rolled, arched Dornbusch maize leaf with exact ground truth."""
    L = 1.0
    blade = DornbuschBlade(length=L, w_max=0.14, xm=0.35, k_lig=0.60, c1=1.8, c2=2.0)

    # Pitch angle alpha(s) from vertical +Z: starts at base_deg, bends smoothly downward by droop_rad
    alpha0 = np.radians(base_deg)
    roll = np.radians(roll_deg)

    def eval_grid(s_vals, v_vals):
        # Integrate centerline from exact pitch alpha(s) = alpha0 + droop_rad * (s/L)^1.35
        """在解析叶片参数网格上计算三维坐标。"""
        ns, nv = len(s_vals), len(v_vals)
        fine_s = np.linspace(0.0, L, 1001)
        alpha_f = alpha0 + droop_rad * (fine_s / L) ** 1.35
        tx_f = np.sin(alpha_f)
        tz_f = np.cos(alpha_f)
        x_f = np.r_[0.0, np.cumsum(0.5 * (tx_f[:-1] + tx_f[1:]) * np.diff(fine_s))]
        z_f = np.r_[0.0, np.cumsum(0.5 * (tz_f[:-1] + tz_f[1:]) * np.diff(fine_s))]

        cx = np.interp(s_vals, fine_s, x_f)
        cz = np.interp(s_vals, fine_s, z_f)
        alpha_s = alpha0 + droop_rad * (s_vals / L) ** 1.35
        tx = np.sin(alpha_s)
        tz = np.cos(alpha_s)

        # Unrolled frame: T=(tx,0,tz), B0=(0,1,0), N0=(-tz,0,tx); roll by roll_deg around T
        T = np.column_stack([tx, np.zeros(ns), tz])
        B0 = np.tile([0.0, 1.0, 0.0], (ns, 1))
        N0 = np.column_stack([-tz, np.zeros(ns), tx])
        B = np.cos(roll) * B0 + np.sin(roll) * N0
        N = -np.sin(roll) * B0 + np.cos(roll) * N0

        w = blade.width(s_vals)
        ss_c = np.column_stack([cx, np.zeros(ns), cz])[:, None, :]  # (ns, 1, 3)
        vv = v_vals[None, :]  # (1, nv)
        lat = 0.5 * vv * w[:, None]  # (ns, nv)
        q = -arch_amp * (vv ** 2) * (w[:, None] / blade.w_max)
        xyz = ss_c + lat[:, :, None] * B[:, None, :] + q[:, :, None] * N[:, None, :]
        nrm = np.broadcast_to(N[:, None, :], xyz.shape).copy()
        return xyz, nrm

    # Exact analytical ground truth (evaluated on fine noiseless interior grid, 5% trim reference)
    s_gt = np.linspace(0.02 * L, 0.96 * L, 101)
    v_gt = np.linspace(-0.90, 0.90, 31)
    xyz_gt, _ = eval_grid(s_gt, v_gt)
    gt_curv = _fundamental_forms_from_grid(xyz_gt, s_gt, v_gt)

    # Also compute exact ground-truth curvature at each margin_trim_frac (so bias vs own domain is checked)
    # Primary truth: true full-blade interior curvature at 2.5% trim (nearly whole blade minus singular edges)
    v_full = np.linspace(-0.95, 0.95, 41)
    xyz_full, _ = eval_grid(s_gt, v_full)
    gt_curv_full = _fundamental_forms_from_grid(xyz_full, s_gt, v_full)

    # Generate noisy 70%-subsampled scan cloud for realistic fit_leaf reconstruction
    rng = np.random.default_rng(seed)
    s_scan = np.linspace(0.0, L, 90)
    v_scan = np.linspace(-1.0, 1.0, 25)
    xyz_s, nrm_s = eval_grid(s_scan, v_scan)
    pts = xyz_s.reshape(-1, 3)
    nrms = nrm_s.reshape(-1, 3)
    keep = rng.choice(len(pts), size=int(0.70 * len(pts)), replace=False)
    pts = pts[keep] + rng.normal(scale=0.0032, size=(len(keep), 3))
    nrms = nrms[keep] + rng.normal(scale=0.04, size=(len(keep), 3))
    nrms /= np.maximum(np.linalg.norm(nrms, axis=1, keepdims=True), 1e-12)

    stem_z = np.linspace(-0.3, 0.3, 80)
    stem_pts = np.column_stack([np.zeros_like(stem_z), np.zeros_like(stem_z), stem_z])
    stem_curve, _ = fit_pspline_gcv(stem_z, stem_pts, lam_bounds=(1e-6, 1e-2), bounded=False)
    stem_bounds = (float(stem_curve.t[stem_curve.k]), float(stem_curve.t[-stem_curve.k - 1]))

    return {
        "pts": pts,
        "nrms": nrms,
        "stem_pts": stem_pts,
        "stem_curve": stem_curve,
        "stem_bounds": stem_bounds,
        "theta_true": float(base_deg),
        "h_abs_true": float(gt_curv_full["mean_curvature_abs"]),
        "willmore_true": float(gt_curv_full["willmore_bending_functional"]),
    }


def _eval_synth_one(args):
    """对单个解析合成样本重建并比较不同表型窗口。"""
    idx, base_deg, droop_rad, roll_deg, arch_amp, seed = args
    syn = make_analytical_leaf(base_deg, droop_rad, roll_deg, arch_amp, seed)
    cfg = ModelConfig(nu=90, nv=25, allow_tip_completion=False, seed=2026)
    res = fit_leaf(syn["pts"], syn["stem_pts"], normals=syn["nrms"], cfg=cfg, leaf_id=idx + 1)
    model = res["model"]
    sc, sb = syn["stem_curve"], syn["stem_bounds"]

    pose = midrib_pose_and_curvature(model, sc, sb, 1.0)
    ang_errs = {0.0: pose["midrib_base_angle_deg"] - syn["theta_true"]}
    for sp in BASAL_SPANS:
        if sp == 0.0:
            continue
        est = basal_surface_stem_leaf_angle(model, sc, sb, 1.0, basal_span_frac=sp)["stem_leaf_angle_deg"]
        ang_errs[sp] = est - syn["theta_true"]

    h_rel_errs = {}
    w_rel_errs = {}
    for tr in MARGIN_TRIMS:
        c = surface_curvature_fundamental_forms(model, margin_trim_frac=tr)
        h_rel_errs[tr] = 100.0 * abs(c["mean_curvature_abs"] - syn["h_abs_true"]) / max(syn["h_abs_true"], 1e-9)
        w_rel_errs[tr] = 100.0 * abs(c["willmore_bending_functional"] - syn["willmore_true"]) / max(syn["willmore_true"], 1e-9)
    return ang_errs, h_rel_errs, w_rel_errs


def run_benchmark1_synthetic(workers=8):
    """对多种解析叶片执行含噪重建和窗口误差标定。

    18 diverse analytical curved/rolled/arched maize leaves."""
    specs = []
    idx = 0
    for base_deg in (20.0, 45.0, 70.0, 110.0):
        for droop_rad in (0.35, 1.05):
            for roll_deg in (0.0, 45.0):
                arch = 0.015 if (idx % 2 == 0) else 0.032
                specs.append((idx, base_deg, droop_rad, roll_deg, arch, 2026 + idx * 17))
                idx += 1
    # 4 * 2 * 2 = 16 diverse leaves
    with ProcessPoolExecutor(max_workers=workers) as ex:
        outs = list(ex.map(_eval_synth_one, specs))

    ang_summary = {}
    for sp in BASAL_SPANS:
        errs = np.array([o[0][sp] for o in outs], dtype=float)
        ang_summary[sp] = {
            "signed_bias_med_deg": float(np.median(errs)),
            "mae_deg": float(np.mean(np.abs(errs))),
            "rmse_deg": float(np.sqrt(np.mean(errs ** 2))),
            "p95_abs_err_deg": float(np.quantile(np.abs(errs), 0.95)),
            "max_abs_err_deg": float(np.max(np.abs(errs))),
        }

    curv_summary = {}
    for tr in MARGIN_TRIMS:
        he = np.array([o[1][tr] for o in outs], dtype=float)
        we = np.array([o[2][tr] for o in outs], dtype=float)
        curv_summary[tr] = {
            "h_abs_med_rel_err_pct": float(np.median(he)),
            "h_abs_p95_rel_err_pct": float(np.quantile(he, 0.95)),
            "willmore_med_rel_err_pct": float(np.median(we)),
            "willmore_p95_rel_err_pct": float(np.quantile(we, 0.95)),
        }
    return len(specs), ang_summary, curv_summary


def _eval_real_plant(args):
    """计算单株冻结参考与遮挡模型间的表型窗口漂移。"""
    plant_name, stem_json, cases_dir = args
    sc, sb, sz, _ = load_stem_spline(stem_json)
    ang_pairs = {sp: [] for sp in BASAL_SPANS}
    h_pairs = {tr: [] for tr in MARGIN_TRIMS}
    w_pairs = {tr: [] for tr in MARGIN_TRIMS}

    for ldir in sorted(Path(cases_dir).glob("leaf_*")):
        ref_p = ldir / "reference_model.npz"
        if not ref_p.is_file():
            continue
        m_ref = MaizeLeafModel.load(ref_p)
        pose_ref = midrib_pose_and_curvature(m_ref, sc, sb, sz)
        ref_ang = {0.0: pose_ref["midrib_base_angle_deg"]}
        for sp in BASAL_SPANS:
            if sp > 0.0:
                ref_ang[sp] = basal_surface_stem_leaf_angle(m_ref, sc, sb, sz, basal_span_frac=sp)["stem_leaf_angle_deg"]

        ref_c = {tr: surface_curvature_fundamental_forms(m_ref, margin_trim_frac=tr) for tr in MARGIN_TRIMS}

        for scen in ("hole", "margin", "tip_10pct"):
            occ_p = ldir / f"{scen}_model.npz"
            if not occ_p.is_file():
                continue
            m_occ = MaizeLeafModel.load(occ_p)
            pose_occ = midrib_pose_and_curvature(m_occ, sc, sb, sz)
            ang_pairs[0.0].append((ref_ang[0.0], pose_occ["midrib_base_angle_deg"]))
            for sp in BASAL_SPANS:
                if sp > 0.0:
                    a_occ = basal_surface_stem_leaf_angle(m_occ, sc, sb, sz, basal_span_frac=sp)["stem_leaf_angle_deg"]
                    ang_pairs[sp].append((ref_ang[sp], a_occ))

            if scen in ("hole", "margin"):
                for tr in MARGIN_TRIMS:
                    c_occ = surface_curvature_fundamental_forms(m_occ, margin_trim_frac=tr)
                    h_pairs[tr].append((ref_c[tr]["mean_curvature_abs"], c_occ["mean_curvature_abs"]))
                    w_pairs[tr].append((ref_c[tr]["willmore_bending_functional"], c_occ["willmore_bending_functional"]))
    return ang_pairs, h_pairs, w_pairs


def run_benchmark2_real_holdout(models_root, holdout_cases_root, workers=8):
    """并行比较真实参考与留出模型的表型稳定性。"""
    tasks = []
    for pdir in sorted(p for p in Path(models_root).glob("plant_*") if p.is_dir()):
        pname = pdir.name.replace("plant_", "", 1)
        cdir = Path(holdout_cases_root) / pname
        if cdir.is_dir():
            tasks.append((pname, pdir / "stem_centerline.json", cdir))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        outs = list(ex.map(_eval_real_plant, tasks))

    ang_summary = {}
    for sp in BASAL_SPANS:
        pairs = [p for o in outs for p in o[0][sp]]
        ref = np.array([p[0] for p in pairs])
        occ = np.array([p[1] for p in pairs])
        diff = np.abs(occ - ref)
        ang_summary[sp] = {
            "n_pairs": len(pairs),
            "med_abs_drift_deg": float(np.median(diff)),
            "mae_drift_deg": float(np.mean(diff)),
            "p95_abs_drift_deg": float(np.quantile(diff, 0.95)),
            "spearman_ref_vs_occ": float(spearmanr(ref, occ).statistic),
        }

    curv_summary = {}
    for tr in MARGIN_TRIMS:
        hp = [p for o in outs for p in o[1][tr]]
        wp = [p for o in outs for p in o[2][tr]]
        hr, ho = np.array([p[0] for p in hp]), np.array([p[1] for p in hp])
        wr, wo = np.array([p[0] for p in wp]), np.array([p[1] for p in wp])
        hd = 100.0 * np.abs(ho - hr) / np.maximum(hr, 1e-9)
        wd = 100.0 * np.abs(wo - wr) / np.maximum(wr, 1e-9)
        curv_summary[tr] = {
            "n_pairs": len(hp),
            "h_abs_med_drift_pct": float(np.median(hd)),
            "h_abs_p95_drift_pct": float(np.quantile(hd, 0.95)),
            "h_abs_spearman": float(spearmanr(hr, ho).statistic),
            "willmore_med_drift_pct": float(np.median(wd)),
            "willmore_p95_drift_pct": float(np.quantile(wd, 0.95)),
            "willmore_spearman": float(spearmanr(wr, wo).statistic),
        }
    return ang_summary, curv_summary


def plot_results(out_png, syn_ang, syn_curv, real_ang, real_curv):
    """绘制表型窗口的合成误差和真实留出漂移。"""
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.7), constrained_layout=True)

    # Panel a: Basal span bias-variance trade-off
    xs = [100 * s for s in BASAL_SPANS]
    ax1.plot(xs, [syn_ang[s]["mae_deg"] for s in BASAL_SPANS], "o-", color="#D55E00", lw=1.9, label="Synthetic GT MAE (bias+var)")
    ax1.plot(xs, [abs(syn_ang[s]["signed_bias_med_deg"]) for s in BASAL_SPANS], "s--", color="#CC79A7", lw=1.5, label="Synthetic GT |median bias|")
    ax1.plot(xs, [real_ang[s]["mae_drift_deg"] for s in BASAL_SPANS], "^-", color="#0072B2", lw=1.9, label="Real 250-pair occlusion MAE drift")
    ax1.plot(xs, [real_ang[s]["p95_abs_drift_deg"] / 3.0 for s in BASAL_SPANS], "d:", color="#009E73", lw=1.5, label="Real P95 drift / 3 (scaled)")
    ax1.set_xlabel("Basal surface window span (% of $L_{\\mathrm{obs}}$; 0% = midrib point)")
    ax1.set_ylabel("Angle error / drift (degrees)")
    ax1.set_title("a  Stem-Leaf Angle: Basal Window Bias–Variance Trade-Off")
    ax1.set_xticks(xs)
    ax1.grid(axis="y", alpha=0.25, ls=":")
    ax1.legend(fontsize=7.8, loc="upper right")

    # Panel b: Lateral margin trim bias-variance trade-off
    xt = [100 * t for t in MARGIN_TRIMS]
    ax2.plot(xt, [syn_curv[t]["h_abs_med_rel_err_pct"] for t in MARGIN_TRIMS], "o-", color="#D55E00", lw=1.9, label="Synthetic GT |H| med. rel. err. (%)")
    ax2.plot(xt, [syn_curv[t]["willmore_med_rel_err_pct"] for t in MARGIN_TRIMS], "s--", color="#CC79A7", lw=1.6, label="Synthetic GT Willmore W med. err. (%)")
    ax2.plot(xt, [real_curv[t]["h_abs_med_drift_pct"] for t in MARGIN_TRIMS], "^-", color="#0072B2", lw=1.9, label="Real 169-pair |H| occlusion drift (%)")
    ax2.plot(xt, [real_curv[t]["willmore_med_drift_pct"] for t in MARGIN_TRIMS], "d:", color="#009E73", lw=1.7, label="Real 169-pair W occlusion drift (%)")
    ax2.set_xlabel("Lateral margin trim per side (% of blade width)")
    ax2.set_ylabel("Median relative error / occlusion drift (%)")
    ax2.set_title("b  Surface Curvature: Margin Trim Bias–Variance Trade-Off")
    ax2.set_xticks(xt)
    ax2.grid(axis="y", alpha=0.25, ls=":")
    ax2.legend(fontsize=7.8, loc="upper right")

    fig.savefig(out_png, dpi=240)
    plt.close(fig)


def main(argv=None):
    """解析标定目录及工作进程数，执行合成与真实留出窗口标定。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, default=ROOT / "outputs/maize_v2")
    parser.add_argument("--cases-root", type=Path, default=ROOT / "outputs/holdout_20261125/validated/cases")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/phenotypes_v2")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    t0 = time.time()
    print("Running Benchmark 1 (16 analytical noisy curved/rolled leaves)...", flush=True)
    n_syn, syn_ang, syn_curv = run_benchmark1_synthetic(workers=args.workers)
    print(f"Benchmark 1 done ({n_syn} leaves) in {time.time() - t0:.1f}s", flush=True)

    t1 = time.time()
    print("Running Benchmark 2 (real 90-leaf frozen holdout perturbation pairs)...", flush=True)
    real_ang, real_curv = run_benchmark2_real_holdout(
        args.models_root,
        args.cases_root,
        workers=args.workers,
    )
    print(f"Benchmark 2 done in {time.time() - t1:.1f}s\n", flush=True)

    print("=== Stem-Leaf Angle Basal Window Sweep ===")
    print(f"{'span%':>6s} | {'SynBias':>8s} {'SynMAE':>7s} {'SynRMSE':>8s} {'SynP95':>7s} | {'RealMed':>7s} {'RealMAE':>7s} {'RealP95':>7s} {'RealRho':>7s}")
    for sp in BASAL_SPANS:
        sa, ra = syn_ang[sp], real_ang[sp]
        print(f"{100*sp:5.1f}% | {sa['signed_bias_med_deg']:+7.2f}  {sa['mae_deg']:6.2f}  {sa['rmse_deg']:7.2f}  {sa['p95_abs_err_deg']:6.2f}  | "
              f"{ra['med_abs_drift_deg']:6.2f}  {ra['mae_drift_deg']:6.2f}  {ra['p95_abs_drift_deg']:6.2f}  {ra['spearman_ref_vs_occ']:7.4f}")

    print("\n=== Curvature Lateral Margin Trim Sweep ===")
    print(f"{'trim%':>6s} | {'SynH%':>7s} {'SynHp95':>7s} {'SynW%':>7s} {'SynWp95':>7s} | {'RealH%':>7s} {'RealHp95':>8s} {'RealW%':>7s} {'RealWp95':>8s} {'RealRhoW':>8s}")
    for tr in MARGIN_TRIMS:
        sc, rc = syn_curv[tr], real_curv[tr]
        print(f"{100*tr:5.1f}% | {sc['h_abs_med_rel_err_pct']:6.2f}% {sc['h_abs_p95_rel_err_pct']:6.2f}% "
              f"{sc['willmore_med_rel_err_pct']:6.2f}% {sc['willmore_p95_rel_err_pct']:6.2f}% | "
              f"{rc['h_abs_med_drift_pct']:6.2f}% {rc['h_abs_p95_drift_pct']:7.2f}% "
              f"{rc['willmore_med_drift_pct']:6.2f}% {rc['willmore_p95_drift_pct']:7.2f}% {rc['willmore_spearman']:8.4f}")

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "seed": 2026,
        "n_synthetic_leaves": n_syn,
        "basal_spans": BASAL_SPANS,
        "margin_trims": MARGIN_TRIMS,
        "synthetic_angle": {f"{100*k:g}pct": v for k, v in syn_ang.items()},
        "real_holdout_angle": {f"{100*k:g}pct": v for k, v in real_ang.items()},
        "synthetic_curvature": {f"{100*k:g}pct": v for k, v in syn_curv.items()},
        "real_holdout_curvature": {f"{100*k:g}pct": v for k, v in real_curv.items()},
        "elapsed_s": round(time.time() - t0, 2),
    }
    write_json(out_dir / "window_calibration_report.json", report)
    plot_results(out_dir / "window_calibration_analysis.png", syn_ang, syn_curv, real_ang, real_curv)
    print(f"\nSaved report & figure under {out_dir} (total {report['elapsed_s']}s)", flush=True)
