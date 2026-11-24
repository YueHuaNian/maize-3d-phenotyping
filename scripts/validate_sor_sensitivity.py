"""SOR (Statistical Outlier Removal) parameter sensitivity benchmark on real maize leaves.

Evaluates k-NN neighborhood sizes and four transparent single-parameter threshold families
across all 14 plants / 90 leaves in dataset/:
  - Level 1: Unperturbed real-scan removal behavior, distal-10% tip retention, and
             natural off-surface vs on-surface separation.
  - Level 2: Controlled noise-injection benchmark (isolated floating halo points +
             3-5 point clustered micro-spikes near/inside the cluster connection radius).
  - Level 3: End-to-end downstream 3D surface reconstruction (fit_leaf) on noisy
             plant-stratified leaves (1 per plant, n=14) and full 90-leaf real-scan audit.

Original dataset/ and outputs/maize_v2/ are strictly read-only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".vendor")]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from maize_leaf.geometry import clean_leaf_points, fit_stem_centerline
from maize_leaf.io import discover, load_pair, write_json
from maize_leaf.pipeline import calibrate_dataset_prior
from maize_leaf.quality import audit_model, surface_support_diagnostics
from maize_leaf.surface import MaizeLeafModel, ModelConfig, fit_leaf, leaf_mesh_grid


K_VALUES = [6, 8, 10, 12, 16, 20, 24, 30, 40, 50]
METHOD_GRIDS = {
    "pcl_std": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
    "med_mad": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0],
    "log_mad": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
    "rel_med": [1.6, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0],
}


def extract_clusters_once(points, normals=None):
    """Exact deduplication + single-pass Euclidean cluster filtering returning fragment labels."""
    _, uniq_idx = np.unique(points, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    p = points[uniq_idx]
    nrm = normals[uniq_idx] if normals is not None else None

    tree = cKDTree(p)
    d2 = tree.query(p, k=min(2, len(p)))[0][:, -1]
    spacing = float(np.median(d2))
    diam = float(np.linalg.norm(np.ptp(p, axis=0)))
    radius = max(spacing * 5.0, diam * 0.03)

    g = tree.sparse_distance_matrix(tree, max_distance=radius, output_type="coo_matrix")
    _, lbl = connected_components(g)
    sizes = np.bincount(lbl)
    main_lbl = int(np.argmax(sizes))
    min_size = max(20, int(0.01 * len(p)))
    keep = (lbl == main_lbl) | (sizes[lbl] >= min_size)
    kept_lbl = lbl[keep]
    # Re-index kept labels to 0..M-1
    _, inv = np.unique(kept_lbl, return_inverse=True)
    return p[keep], (nrm[keep] if nrm is not None else None), inv, spacing, radius


def sor_mask_from_mean_d(mean_d, method, param):
    """Return boolean keep mask for a single fragment given kNN mean distances."""
    if method == "pcl_std":
        thresh = float(np.mean(mean_d) + param * np.std(mean_d))
        return mean_d <= thresh
    if method == "med_mad":
        med = float(np.median(mean_d))
        sigma_rob = 1.4826 * float(np.median(np.abs(mean_d - med)))
        thresh = med + param * sigma_rob
        return mean_d <= thresh
    if method == "log_mad":
        ld = np.log(np.maximum(mean_d, 1e-12))
        med = float(np.median(ld))
        sigma_rob = 1.4826 * float(np.median(np.abs(ld - med)))
        thresh = med + param * sigma_rob
        return ld <= thresh
    if method == "rel_med":
        med = float(np.median(mean_d))
        return mean_d <= param * med
    raise ValueError(f"Unknown SOR method: {method}")


def apply_sor_to_points(points, normals, method, k, param):
    """Full denoising function matching candidate (single-pass clusters + per-fragment SOR)."""
    if method == "current_impl":
        return clean_leaf_points(points, normals)
    p, nrm, frag_lbl, _, _ = extract_clusters_once(points, normals)
    if method == "no_sor":
        return p, nrm
    keep = np.ones(len(p), dtype=bool)
    for lb in np.unique(frag_lbl):
        idx = np.flatnonzero(frag_lbl == lb)
        if len(idx) < 20:
            continue
        k_eff = min(k, len(idx) - 1)
        dists = cKDTree(p[idx]).query(p[idx], k=k_eff + 1)[0][:, 1:]
        mean_d = np.mean(dists, axis=1)
        sub_keep = sor_mask_from_mean_d(mean_d, method, param)
        # Mild safeguard: never wipe out >25% of a coherent fragment
        if int(np.sum(sub_keep)) >= max(15, int(0.75 * len(idx))):
            keep[idx] = sub_keep
    return p[keep], (nrm[keep] if nrm is not None else None)


def inject_realistic_outliers(pts, nrm, spacing, radius, rng, iso_frac=0.02, clus_frac=0.01):
    """Inject isolated near-leaf halo points and 3-5 point clustered spikes.

    Offsets are capped inside ~0.95*radius (or [3.5, 9.5]*spacing) so that Euclidean
    cluster filtering cannot trivially discard them; SOR is tested directly.
    """
    n = len(pts)
    max_off = max(4.5 * spacing, min(9.5 * spacing, 0.90 * radius))
    min_off_iso = min(3.5 * spacing, 0.65 * max_off)
    min_off_clu = min(4.0 * spacing, 0.70 * max_off)

    # Normal directions (fallback to local random unit vectors if normals absent)
    if nrm is not None and len(nrm) == n:
        dirs = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    else:
        dirs = rng.normal(size=(n, 3))
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)

    # Type A: isolated floating halo points
    n_iso = max(4, int(round(iso_frac * n)))
    sel_iso = rng.choice(n, size=n_iso, replace=True)
    signs_iso = rng.choice([-1.0, 1.0], size=n_iso)
    offs_iso = rng.uniform(min_off_iso, max_off, size=n_iso)
    jitter_iso = rng.normal(scale=0.35 * spacing, size=(n_iso, 3))
    pts_iso = pts[sel_iso] + (signs_iso * offs_iso)[:, None] * dirs[sel_iso] + jitter_iso
    nrm_iso = dirs[sel_iso].copy()

    # Type B: clustered 3-5 point spikes
    target_clu = max(4, int(round(clus_frac * n)))
    pts_clu_list, nrm_clu_list = [], []
    made = 0
    while made < target_clu:
        csize = int(rng.integers(3, 6))
        if made + csize > target_clu + 2:
            csize = max(3, target_clu - made)
        anchor = int(rng.integers(n))
        sign = float(rng.choice([-1.0, 1.0]))
        off = float(rng.uniform(min_off_clu, max_off))
        center = pts[anchor] + sign * off * dirs[anchor]
        cluster_pts = center + rng.normal(scale=0.55 * spacing, size=(csize, 3))
        pts_clu_list.append(cluster_pts)
        nrm_clu_list.append(np.tile(dirs[anchor], (csize, 1)))
        made += csize

    pts_clu = np.vstack(pts_clu_list)
    nrm_clu = np.vstack(nrm_clu_list)

    all_pts = np.vstack([pts, pts_iso, pts_clu])
    all_nrm = np.vstack([dirs, nrm_iso, nrm_clu]) if nrm is not None else None
    # Labels: 0=real leaf point, 1=isolated injected outlier, 2=clustered injected outlier
    gt_type = np.concatenate([
        np.zeros(n, dtype=np.int8),
        np.ones(len(pts_iso), dtype=np.int8),
        np.full(len(pts_clu), 2, dtype=np.int8),
    ])
    return all_pts, all_nrm, gt_type


def precompute_leaf_records(input_root, ref_model_root, seed=2026):
    """Load all 90 leaves, compute reference proxies, tip masks, and kNN tables."""
    rng = np.random.default_rng(seed)
    pairs = discover(input_root)
    k_max = max(K_VALUES)
    leaves = []

    for plant_name, txt, csv_path in pairs:
        plant = load_pair(plant_name, txt, csv_path)
        stem_pts = plant.xyz[plant.labels == 0]
        stem, _ = fit_stem_centerline(stem_pts)
        stem_tree = cKDTree(stem)

        for lid in sorted(int(x) for x in set(plant.labels) if x > 0):
            raw_pts = plant.xyz[plant.labels == lid]
            raw_nrm = plant.normals[plant.labels == lid] if plant.normals is not None else None

            # Single-pass cluster extraction (clean baseline)
            p_cl, n_cl, frag_lbl, spacing, radius = extract_clusters_once(raw_pts, raw_nrm)

            # Current implementation mask on p_cl
            p_cur, _ = clean_leaf_points(raw_pts, raw_nrm)
            keep_cur_real = cKDTree(p_cur).query(p_cl)[0] < 1e-12

            # Distal 10% tip mask (model-free: dominant PCA axis oriented away from stem)
            centered = p_cl - np.median(p_cl, axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            proj = centered @ vt[0]
            p_lo = p_cl[int(np.argmin(proj))]
            p_hi = p_cl[int(np.argmax(proj))]
            if stem_tree.query(p_lo)[0] > stem_tree.query(p_hi)[0]:
                proj = -proj
            p90_thr = float(np.quantile(proj, 0.90))
            tip10_mask = proj >= p90_thr

            # Reference surface distance (if validated model exists) for on-surface vs natural-off-surface proxy
            d_surf_sp = None
            mpath = ref_model_root / f"plant_{plant_name}" / "leaves" / f"leaf_{lid:02d}_model.npz"
            if mpath.is_file():
                model = MaizeLeafModel.load(mpath)
                sv, _ = leaf_mesh_grid(model.blade.length, 220, 61)
                s_pts = model.evaluate(sv)
                d_surf_sp = cKDTree(s_pts).query(p_cl)[0] / max(spacing, 1e-12)

            clean_inlier_mask = (d_surf_sp <= 2.0) if d_surf_sp is not None else np.ones(len(p_cl), dtype=bool)
            nat_outlier_mask = (d_surf_sp > 5.0) if d_surf_sp is not None else np.zeros(len(p_cl), dtype=bool)

            # Precompute cumulative kNN distance sums per fragment on REAL points
            real_frags = []
            for lb in np.unique(frag_lbl):
                idx = np.flatnonzero(frag_lbl == lb)
                pts_f = p_cl[idx]
                k_q = min(k_max + 1, len(pts_f))
                dists = cKDTree(pts_f).query(pts_f, k=k_q)[0][:, 1:]
                csum = np.cumsum(dists, axis=1)
                real_frags.append((idx, csum))

            # Level 2: inject realistic isolated + clustered outliers onto p_cl
            noisy_pts, noisy_nrm, gt_type = inject_realistic_outliers(
                p_cl, n_cl, spacing, radius, rng
            )
            # Run single-pass cluster extraction on noisy_pts (tracks which injected points survived clustering)
            # Also evaluate current_impl on noisy_pts directly
            pn_cl, nn_cl, fn_lbl, _, _ = extract_clusters_once(noisy_pts, noisy_nrm)
            # Map pn_cl back to noisy_pts indices (exact match)
            match_d, match_i = cKDTree(noisy_pts).query(pn_cl)
            gt_after_cl = gt_type[match_i]
            is_clean_inlier_n = np.zeros(len(pn_cl), dtype=bool)
            is_tip10_clean_n = np.zeros(len(pn_cl), dtype=bool)
            real_orig_mask = gt_after_cl == 0
            orig_idx = match_i[real_orig_mask]
            is_clean_inlier_n[real_orig_mask] = clean_inlier_mask[orig_idx]
            is_tip10_clean_n[real_orig_mask] = clean_inlier_mask[orig_idx] & tip10_mask[orig_idx]

            # Current impl on noisy_pts (mapped to pn_cl)
            pn_cur, _ = clean_leaf_points(noisy_pts, noisy_nrm)
            keep_cur_noisy = cKDTree(pn_cur).query(pn_cl)[0] < 1e-12

            noisy_frags = []
            for lb in np.unique(fn_lbl):
                idx = np.flatnonzero(fn_lbl == lb)
                pts_f = pn_cl[idx]
                k_q = min(k_max + 1, len(pts_f))
                dists = cKDTree(pts_f).query(pts_f, k=k_q)[0][:, 1:]
                csum = np.cumsum(dists, axis=1)
                noisy_frags.append((idx, csum))

            leaves.append({
                "plant": plant_name,
                "leaf_id": lid,
                "n_raw": len(raw_pts),
                "n_cl": len(p_cl),
                "spacing": spacing,
                "radius": radius,
                "p_cl": p_cl,
                "n_cl_arr": n_cl,
                "raw_pts": raw_pts,
                "raw_nrm": raw_nrm,
                "stem": stem,
                "tip10_mask": tip10_mask,
                "clean_inlier_mask": clean_inlier_mask,
                "nat_outlier_mask": nat_outlier_mask,
                "keep_cur_real": keep_cur_real,
                "real_frags": real_frags,
                "noisy_pts": noisy_pts,
                "noisy_nrm": noisy_nrm,
                "pn_cl": pn_cl,
                "gt_after_cl": gt_after_cl,
                "is_clean_inlier_n": is_clean_inlier_n,
                "is_tip10_clean_n": is_tip10_clean_n,
                "keep_cur_noisy": keep_cur_noisy,
                "noisy_frags": noisy_frags,
            })

    return leaves


def eval_config_level1_level2(leaves, method, k, param):
    """Evaluate one (method, k, param) across all 90 leaves on Level 1 & Level 2."""
    # Level 1 accumulators
    l1_tot = 0
    l1_rem = 0
    l1_tip_tot = 0
    l1_tip_rem = 0
    l1_clean_tot = 0
    l1_clean_rem = 0
    l1_nat_tot = 0
    l1_nat_rem = 0
    l1_leaf_rem_fracs = []
    l1_leaf_tip_rem_fracs = []

    # Level 2 accumulators (clean_inlier = negative 0, injected_outlier = positive 1)
    tp = fp = tn = fn = 0
    iso_tot = iso_det = 0
    clu_tot = clu_det = 0
    tip_n_tot = tip_n_fp = 0
    plant_stats = {}

    for lf in leaves:
        n_cl = lf["n_cl"]
        # --- Level 1 (unperturbed real scan) ---
        if method == "current_impl":
            keep_r = lf["keep_cur_real"]
        elif method == "no_sor":
            keep_r = np.ones(n_cl, dtype=bool)
        else:
            keep_r = np.ones(n_cl, dtype=bool)
            for idx, csum in lf["real_frags"]:
                if len(idx) < 20:
                    continue
                k_eff = min(k, csum.shape[1])
                mean_d = csum[:, k_eff - 1] / k_eff
                sub = sor_mask_from_mean_d(mean_d, method, param)
                if int(np.sum(sub)) >= max(15, int(0.75 * len(idx))):
                    keep_r[idx] = sub

        rem_r = ~keep_r
        l1_tot += n_cl
        l1_rem += int(np.sum(rem_r))
        l1_leaf_rem_fracs.append(float(np.mean(rem_r)))

        tmask = lf["tip10_mask"]
        cmask = lf["clean_inlier_mask"]
        nmask = lf["nat_outlier_mask"]
        t_clean = tmask & cmask
        l1_tip_tot += int(np.sum(t_clean))
        l1_tip_rem += int(np.sum(rem_r & t_clean))
        if np.any(t_clean):
            l1_leaf_tip_rem_fracs.append(float(np.mean(rem_r[t_clean])))

        l1_clean_tot += int(np.sum(cmask))
        l1_clean_rem += int(np.sum(rem_r & cmask))
        l1_nat_tot += int(np.sum(nmask))
        l1_nat_rem += int(np.sum(rem_r & nmask))

        # --- Level 2 (controlled noise injection) ---
        gt = lf["gt_after_cl"]
        n_n = len(gt)
        if method == "current_impl":
            keep_n = lf["keep_cur_noisy"]
        elif method == "no_sor":
            keep_n = np.ones(n_n, dtype=bool)
        else:
            keep_n = np.ones(n_n, dtype=bool)
            for idx, csum in lf["noisy_frags"]:
                if len(idx) < 20:
                    continue
                k_eff = min(k, csum.shape[1])
                mean_d = csum[:, k_eff - 1] / k_eff
                sub = sor_mask_from_mean_d(mean_d, method, param)
                if int(np.sum(sub)) >= max(15, int(0.75 * len(idx))):
                    keep_n[idx] = sub

        pred_out = ~keep_n
        is_inj = gt > 0
        is_iso = gt == 1
        is_clu = gt == 2
        is_cin = lf["is_clean_inlier_n"]
        is_tip = lf["is_tip10_clean_n"]

        l_tp = int(np.sum(pred_out & is_inj))
        l_fn = int(np.sum((~pred_out) & is_inj))
        l_fp = int(np.sum(pred_out & is_cin))
        l_tn = int(np.sum((~pred_out) & is_cin))

        tp += l_tp
        fn += l_fn
        fp += l_fp
        tn += l_tn

        iso_tot += int(np.sum(is_iso))
        iso_det += int(np.sum(pred_out & is_iso))
        clu_tot += int(np.sum(is_clu))
        clu_det += int(np.sum(pred_out & is_clu))
        tip_n_tot += int(np.sum(is_tip))
        tip_n_fp += int(np.sum(pred_out & is_tip))

        pl = lf["plant"]
        ps = plant_stats.setdefault(pl, [0, 0, 0, 0])
        ps[0] += l_tp
        ps[1] += l_fp
        ps[2] += l_tn
        ps[3] += l_fn

    def _f1_mcc(tp_, fp_, tn_, fn_):
        prec = tp_ / max(tp_ + fp_, 1)
        rec = tp_ / max(tp_ + fn_, 1)
        f1 = 2.0 * prec * rec / max(prec + rec, 1e-12)
        denom = float((tp_ + fp_) * (tp_ + fn_) * (tn_ + fp_) * (tn_ + fn_))
        mcc = (tp_ * tn_ - fp_ * fn_) / np.sqrt(denom) if denom > 0 else 0.0
        return prec, rec, f1, float(mcc)

    prec, rec, f1, mcc = _f1_mcc(tp, fp, tn, fn)
    plant_f1s = [_f1_mcc(*v)[2] for v in plant_stats.values()]
    plant_mccs = [_f1_mcc(*v)[3] for v in plant_stats.values()]

    return {
        "method": method,
        "k": k,
        "param": param,
        # Level 1 (real scans)
        "l1_removed_pct": 100.0 * l1_rem / max(l1_tot, 1),
        "l1_leaf_rem_med_pct": 100.0 * float(np.median(l1_leaf_rem_fracs)),
        "l1_leaf_rem_p95_pct": 100.0 * float(np.quantile(l1_leaf_rem_fracs, 0.95)),
        "l1_leaf_rem_max_pct": 100.0 * float(np.max(l1_leaf_rem_fracs)),
        "l1_clean_fpr_pct": 100.0 * l1_clean_rem / max(l1_clean_tot, 1),
        "l1_tip10_clean_fpr_pct": 100.0 * l1_tip_rem / max(l1_tip_tot, 1),
        "l1_tip10_leaf_p95_pct": 100.0 * float(np.quantile(l1_leaf_tip_rem_fracs, 0.95)),
        "l1_nat_offsurf_recall_pct": 100.0 * l1_nat_rem / max(l1_nat_tot, 1),
        # Level 2 (controlled injection)
        "l2_precision": prec,
        "l2_recall_all": rec,
        "l2_recall_isolated": iso_det / max(iso_tot, 1),
        "l2_recall_clustered": clu_det / max(clu_tot, 1),
        "l2_clean_fpr_pct": 100.0 * fp / max(fp + tn, 1),
        "l2_tip10_fpr_pct": 100.0 * tip_n_fp / max(tip_n_tot, 1),
        "l2_f1_pooled": f1,
        "l2_mcc_pooled": mcc,
        "l2_f1_plant_mean": float(np.mean(plant_f1s)),
        "l2_f1_plant_std": float(np.std(plant_f1s)),
        "l2_mcc_plant_mean": float(np.mean(plant_mccs)),
        "l2_mcc_plant_std": float(np.std(plant_mccs)),
    }


def select_best_candidates(grid_rows):
    """Select the top configuration from each threshold family (with tip erosion <= 1.5% constraint)."""
    best_by_method = {}
    for method in METHOD_GRIDS:
        sub = [r for r in grid_rows if r["method"] == method]
        # Primary: maximize plant-mean MCC subject to real clean FPR <= 1.0% and tip10 FPR <= 2.5%
        feasible = [r for r in sub if r["l1_clean_fpr_pct"] <= 1.0 and r["l1_tip10_clean_fpr_pct"] <= 2.5]
        pool = feasible if feasible else sub
        best_by_method[method] = max(pool, key=lambda r: (r["l2_mcc_plant_mean"], r["l2_f1_plant_mean"]))
    return best_by_method


def fit_leaf_with_custom_sor(points, stem, normals, prior, cfg, leaf_id, method, k, param):
    """Run fit_leaf by pre-cleaning with custom SOR and temporarily bypassing internal SOR."""
    import maize_leaf.surface as surf_mod

    cleaned_p, cleaned_n = apply_sor_to_points(points, normals, method, k, param)
    orig_clean = surf_mod.clean_leaf_points
    try:
        surf_mod.clean_leaf_points = lambda pts, nrm=None: (pts, nrm)
        res = fit_leaf(cleaned_p, stem, normals=cleaned_n, prior=prior, cfg=cfg, leaf_id=leaf_id)
    finally:
        surf_mod.clean_leaf_points = orig_clean
    return res, cleaned_p


def _eval_l3a_one(args):
    """Worker for one (config, leaf) in Level 3A."""
    label, method, k, param, noisy_pts, noisy_nrm, stem, leaf_id, clean_pts, ref_len, ref_wid, ref_area, prior, cfg = args
    try:
        res, _ = fit_leaf_with_custom_sor(noisy_pts, stem, noisy_nrm, prior, cfg, leaf_id, method, k, param)
        model = res["model"]
        sv, _ = leaf_mesh_grid(model.blade.length, 220, 61)
        s_pts = model.evaluate(sv)
        d_to_surf = cKDTree(s_pts).query(clean_pts)[0]
        return (
            label, True,
            float(np.quantile(d_to_surf, 0.95)),
            float(np.sqrt(np.mean(d_to_surf ** 2))),
            surface_support_diagnostics(model, clean_pts)["mean"],
            100.0 * abs(res["traits"]["completed_length"] - ref_len) / max(ref_len, 1e-9),
            100.0 * abs(res["traits"]["max_width"] - ref_wid) / max(ref_wid, 1e-9),
            100.0 * abs(res["traits"]["surface_area"] - ref_area) / max(ref_area, 1e-9),
            None,
        )
    except Exception as e:
        return (label, False, None, None, None, None, None, None, str(e))


def _eval_ref_one(args):
    """Worker for clean reference fit of one representative leaf."""
    key, clean_pts, clean_nrm, stem, leaf_id, prior, cfg = args
    res_ref, _ = fit_leaf_with_custom_sor(clean_pts, stem, clean_nrm, prior, cfg, leaf_id, "no_sor", 0, 0.0)
    return key, {
        "length": res_ref["traits"]["completed_length"],
        "obs_length": res_ref["traits"]["observed_length"],
        "max_width": res_ref["traits"]["max_width"],
        "area": res_ref["traits"]["surface_area"],
    }


def _eval_l3b_one(args):
    """Worker for one (config, leaf) in Level 3B (full 90-leaf audit)."""
    label, method, k, param, plant, leaf_id, raw_pts, raw_nrm, stem, clean_ref, prior, cfg = args
    try:
        res, _ = fit_leaf_with_custom_sor(raw_pts, stem, raw_nrm, prior, cfg, leaf_id, method, k, param)
        model = res["model"]
        sv, _ = leaf_mesh_grid(model.blade.length, 220, 61)
        s_pts = model.evaluate(sv)
        s_tree = cKDTree(s_pts)
        d_raw = s_tree.query(raw_pts)[0]
        d_clean = s_tree.query(clean_ref)[0]
        return (
            label, plant, leaf_id, True,
            float(np.quantile(d_raw, 0.95)),
            float(np.quantile(d_clean, 0.95)),
            float(np.median(d_clean)),
            surface_support_diagnostics(model, clean_ref)["mean"],
            float(res["traits"]["observed_length"]),
            float(res["traits"]["max_width"]),
            float(res["traits"]["surface_area"]),
            None,
        )
    except Exception as e:
        return (label, plant, leaf_id, False, None, None, None, None, None, None, None, str(e))


def run_level3_validation(leaves, candidates, input_root, workers=8):
    """Level 3A (noisy plant-stratified 14-leaf reconstruction) in parallel."""
    pairs = discover(input_root)
    prior, n_cal = calibrate_dataset_prior(pairs)
    cfg = ModelConfig(nu=90, nv=25, allow_tip_completion=True)

    # Select 1 representative leaf per plant (largest point count in each plant)
    by_plant = {}
    for lf in leaves:
        by_plant.setdefault(lf["plant"], []).append(lf)
    rep_leaves = [max(by_plant[pl], key=lambda x: x["n_cl"]) for pl in sorted(by_plant)]

    t_ref = time.time()
    ref_jobs = []
    clean_pts_map = {}
    for lf in rep_leaves:
        key = (lf["plant"], lf["leaf_id"])
        c_pts = lf["p_cl"][lf["clean_inlier_mask"]]
        c_nrm = lf["n_cl_arr"][lf["clean_inlier_mask"]] if lf["n_cl_arr"] is not None else None
        clean_pts_map[key] = c_pts
        ref_jobs.append((key, c_pts, c_nrm, lf["stem"], lf["leaf_id"], prior, cfg))

    ref_fits = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for key, val in ex.map(_eval_ref_one, ref_jobs):
            val["clean_pts"] = clean_pts_map[key]
            ref_fits[key] = val
    print(f"  Computed {len(ref_fits)} clean reference fits in {time.time() - t_ref:.1f}s (workers={workers})", flush=True)

    # --- Level 3A: Noisy 14-leaf end-to-end reconstruction ---
    l3a_summary = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for label, method, k, param in candidates:
            t0 = time.time()
            jobs = []
            for lf in rep_leaves:
                key = (lf["plant"], lf["leaf_id"])
                ref = ref_fits[key]
                jobs.append((
                    label, method, k, param,
                    lf["noisy_pts"], lf["noisy_nrm"], lf["stem"], lf["leaf_id"],
                    ref["clean_pts"], ref["length"], ref["max_width"], ref["area"],
                    prior, cfg,
                ))
            results = list(ex.map(_eval_l3a_one, jobs))
            ok = [r for r in results if r[1]]
            passed = len(ok)
            failed = len(results) - passed
            p95s = [r[2] for r in ok]
            rmses = [r[3] for r in ok]
            supports = [r[4] for r in ok]
            len_errs = [r[5] for r in ok]
            wid_errs = [r[6] for r in ok]
            area_errs = [r[7] for r in ok]

            rec = {
                "label": label,
                "method": method,
                "k": k,
                "param": param,
                "n_leaves": len(rep_leaves),
                "passed": passed,
                "failed": failed,
                "med_clean_p95": float(np.median(p95s)) if p95s else None,
                "med_clean_rmse": float(np.median(rmses)) if rmses else None,
                "med_support_mean": float(np.median(supports)) if supports else None,
                "med_length_err_pct": float(np.median(len_errs)) if len_errs else None,
                "med_width_err_pct": float(np.median(wid_errs)) if wid_errs else None,
                "med_area_err_pct": float(np.median(area_errs)) if area_errs else None,
                "p95_area_err_pct": float(np.quantile(area_errs, 0.95)) if area_errs else None,
                "elapsed_s": round(time.time() - t0, 2),
            }
            l3a_summary.append(rec)
            print(f"  [L3A] {label:20s}: pass={passed:2d}/{len(rep_leaves)}, "
                  f"rmse={rec['med_clean_rmse']:.5f}, p95={rec['med_clean_p95']:.5f}, "
                  f"supp={rec['med_support_mean']:.5f}, "
                  f"len={rec['med_length_err_pct']:.2f}%, wid={rec['med_width_err_pct']:.2f}%, "
                  f"area={rec['med_area_err_pct']:.2f}% (p95={rec['p95_area_err_pct']:.2f}%) [{rec['elapsed_s']}s]",
                  flush=True)

    return l3a_summary, prior, cfg


def run_level3b_full90(leaves, finalists, prior, cfg, workers=8):
    """Level 3B: Full 90-leaf unperturbed real-scan audit in parallel."""
    l3b_summary = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for label, method, k, param in finalists:
            t0 = time.time()
            jobs = [
                (label, method, k, param, lf["plant"], lf["leaf_id"],
                 lf["raw_pts"], lf["raw_nrm"], lf["stem"], lf["p_cl"][lf["clean_inlier_mask"]], prior, cfg)
                for lf in leaves
            ]
            results = list(ex.map(_eval_l3b_one, jobs))
            ok = [r for r in results if r[3]]
            failed_list = [f"{r[1]}_L{r[2]}: {r[11]}" for r in results if not r[3]]
            passed = len(ok)
            rec = {
                "label": label,
                "method": method,
                "k": k,
                "param": param,
                "n_leaves": len(leaves),
                "passed": passed,
                "failed": len(failed_list),
                "failed_details": failed_list[:5],
                "med_raw_p95": float(np.median([r[4] for r in ok])) if ok else None,
                "med_clean_p95": float(np.median([r[5] for r in ok])) if ok else None,
                "med_clean_median": float(np.median([r[6] for r in ok])) if ok else None,
                "med_support_clean": float(np.median([r[7] for r in ok])) if ok else None,
                "med_obs_length": float(np.median([r[8] for r in ok])) if ok else None,
                "med_max_width": float(np.median([r[9] for r in ok])) if ok else None,
                "med_surface_area": float(np.median([r[10] for r in ok])) if ok else None,
                "elapsed_s": round(time.time() - t0, 2),
            }
            l3b_summary.append(rec)
            print(f"  [L3B 90-leaf] {label:20s}: pass={passed:2d}/90, "
                  f"raw_p95={rec['med_raw_p95']:.6f}, clean_p95={rec['med_clean_p95']:.6f}, "
                  f"supp={rec['med_support_clean']:.6f}, len={rec['med_obs_length']:.5f}, "
                  f"area={rec['med_surface_area']:.6f} [{rec['elapsed_s']}s]", flush=True)
    return l3b_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--ref-models", type=Path, default=ROOT / "outputs/maize_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/sor_sensitivity")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--skip-l3", action="store_true")
    parser.add_argument("--skip-l3b", action="store_true")
    args = parser.parse_args()

    t_start = time.time()
    print("Loading 14 plants / 90 leaves and precomputing kNN tables...", flush=True)
    leaves = precompute_leaf_records(args.input_root, args.ref_models, seed=args.seed)
    print(f"Loaded {len(leaves)} leaves in {time.time() - t_start:.2f}s", flush=True)

    # --- Level 1 & Level 2 grid sweep ---
    t_grid = time.time()
    grid_rows = []
    # Baselines
    for m in ("no_sor", "current_impl"):
        grid_rows.append(eval_config_level1_level2(leaves, m, 12 if m == "current_impl" else 0, 0.0))

    for method, params in METHOD_GRIDS.items():
        for k in K_VALUES:
            for param in params:
                grid_rows.append(eval_config_level1_level2(leaves, method, k, param))

    print(f"Evaluated {len(grid_rows)} configurations across 90 leaves in {time.time() - t_grid:.2f}s", flush=True)

    best_by_method = select_best_candidates(grid_rows)
    print("\n=== Top Candidate per Threshold Family (L1 clean FPR <= 1.0%, tip10 FPR <= 2.5%) ===", flush=True)
    for m, r in best_by_method.items():
        print(f"  {m:10s} k={r['k']:2d} param={r['param']:4.1f} | "
              f"MCC={r['l2_mcc_plant_mean']:.4f}+/-{r['l2_mcc_plant_std']:.4f}, "
              f"F1={r['l2_f1_plant_mean']:.4f}, "
              f"Rec(all/iso/clu)={r['l2_recall_all']:.3f}/{r['l2_recall_isolated']:.3f}/{r['l2_recall_clustered']:.3f}, "
              f"L1_rem={r['l1_removed_pct']:.2f}%, L1_cleanFPR={r['l1_clean_fpr_pct']:.2f}%, "
              f"L1_tip10FPR={r['l1_tip10_clean_fpr_pct']:.2f}%, L1_natRec={r['l1_nat_offsurf_recall_pct']:.1f}%",
              flush=True)

    cur = next(r for r in grid_rows if r["method"] == "current_impl")
    print(f"  {'current':10s} k=12 (2.2*m)| "
          f"MCC={cur['l2_mcc_plant_mean']:.4f}+/-{cur['l2_mcc_plant_std']:.4f}, "
          f"F1={cur['l2_f1_plant_mean']:.4f}, "
          f"Rec(all/iso/clu)={cur['l2_recall_all']:.3f}/{cur['l2_recall_isolated']:.3f}/{cur['l2_recall_clustered']:.3f}, "
          f"L1_rem={cur['l1_removed_pct']:.2f}%, L1_cleanFPR={cur['l1_clean_fpr_pct']:.2f}%, "
          f"L1_tip10FPR={cur['l1_tip10_clean_fpr_pct']:.2f}%, L1_natRec={cur['l1_nat_offsurf_recall_pct']:.1f}%",
          flush=True)

    # Save Level 1 & 2 outputs immediately
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "sor_grid_results.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        w.writeheader()
        w.writerows(grid_rows)

    l3a_summary = []
    l3b_summary = []
    if not args.skip_l3:
        print("\n=== Level 3A: Noisy 14-Leaf (Plant-Stratified) End-to-End Reconstruction ===", flush=True)
        # Cover baselines + Pareto frontier across all 4 method families and key k values
        l3_candidates = [
            ("no_sor", "no_sor", 0, 0.0),
            ("current_impl", "current_impl", 12, 2.2),
            ("pcl_std_k8_a1.5", "pcl_std", 8, 1.5),
            ("pcl_std_k8_a2.0", "pcl_std", 8, 2.0),
            ("pcl_std_k10_a2.0", "pcl_std", 10, 2.0),
            ("pcl_std_k12_a2.0", "pcl_std", 12, 2.0),
            ("med_mad_k8_a3.5", "med_mad", 8, 3.5),
            ("med_mad_k8_a4.0", "med_mad", 8, 4.0),
            ("log_mad_k8_a2.5", "log_mad", 8, 2.5),
            ("rel_med_k8_g2.0", "rel_med", 8, 2.0),
            ("rel_med_k8_g2.2", "rel_med", 8, 2.2),
        ]

        l3a_summary, prior, cfg = run_level3_validation(leaves, l3_candidates, args.input_root, workers=args.workers)

        if not args.skip_l3b:
            print("\n=== Level 3B: Full 90-Leaf Unperturbed Real-Scan Audit ===", flush=True)
            # Pick top 3 distinct winners from L3A (excluding baselines) + current_impl + no_sor
            non_base = [r for r in l3a_summary if r["method"] not in ("no_sor", "current_impl") and r["passed"] == r["n_leaves"]]
            non_base.sort(key=lambda r: (r["med_clean_rmse"] + r["med_support_mean"], r["med_area_err_pct"]))
            finalists = [
                ("no_sor", "no_sor", 0, 0.0),
                ("current_impl", "current_impl", 12, 2.2),
            ]
            for r in non_base[:3]:
                finalists.append((r["label"], r["method"], r["k"], r["param"]))
            l3b_summary = run_level3b_full90(leaves, finalists, prior, cfg, workers=args.workers)

    report = {
        "seed": args.seed,
        "dataset": str(args.input_root),
        "n_leaves": len(leaves),
        "n_plants": len(set(lf["plant"] for lf in leaves)),
        "k_values": K_VALUES,
        "method_grids": METHOD_GRIDS,
        "best_by_method": best_by_method,
        "current_impl_l1_l2": cur,
        "level3a_noisy_14leaf": l3a_summary,
        "level3b_real_90leaf": l3b_summary,
        "total_elapsed_s": round(time.time() - t_start, 2),
    }
    write_json(args.output / "sor_sensitivity_report.json", report)
    print(f"\nSaved report to {args.output / 'sor_sensitivity_report.json'} (total {report['total_elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
