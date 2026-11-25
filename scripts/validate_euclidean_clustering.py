"""Standalone Euclidean clustering parameter sensitivity benchmark on real maize leaves.

Evaluates radius formulations (pure spacing, pure bounding-box diagonal, hybrid,
and robust-diagonal hybrid) combined with secondary-cluster retention thresholds
across all 14 plants / 90 leaves in dataset/:
  - Level 1: Unperturbed real-scan removal, fragmentation, clean/tip retention,
             and natural off-surface (>5 spacing) recall.
  - Level 2: Controlled structural stress test injecting:
             (a) a genuine occlusion-detached distal blade fragment (to test
                 retention of disconnected leaf tips),
             (b) close (4.5-7.0*sp) and medium (8-14*sp) detached mislabeled
                 noise patches, and
             (c) a far stray artifact (0.6-1.0*D) evaluated in paired
                 without-far vs with-far runs to quantify raw-ptp diameter inflation.
  - Level 3: End-to-end downstream 3D surface reconstruction (fit_leaf, coupled
             with calibrated per-fragment MAD-SOR k=8, alpha=3.5) on:
             - Level 3A: Plant-stratified 14-leaf perturbed stress test.
             - Level 3B: Full 90-leaf unperturbed real-scan quality & fit audit.

Original dataset/, src/, and outputs/maize_v2/ are strictly read-only.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
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

from maize_leaf.geometry import fit_stem_centerline, sor_filter
from maize_leaf.io import discover, load_pair, write_json
from maize_leaf.pipeline import calibrate_dataset_prior
from maize_leaf.quality import surface_support_diagnostics
from maize_leaf.surface import MaizeLeafModel, ModelConfig, fit_leaf, leaf_mesh_grid


def build_radius_configs():
    """Return list of (r_family, alpha_sp, alpha_d) radius definitions."""
    cfgs = []
    # Pure spacing: r = alpha_sp * spacing
    for a_sp in (2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0):
        cfgs.append(("pure_sp", float(a_sp), 0.0))
    # Pure raw bbox diagonal: r = alpha_d * diam
    for a_d in (0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07):
        cfgs.append(("pure_diam", 0.0, float(a_d)))
    # Hybrid raw bbox diagonal (current family): r = max(alpha_sp * spacing, alpha_d * diam)
    for a_sp in (3.0, 4.0, 5.0, 6.0, 8.0):
        for a_d in (0.015, 0.02, 0.03, 0.04, 0.05):
            cfgs.append(("hybrid", float(a_sp), float(a_d)))
    # Hybrid robust diagonal (1%-99% quantile bbox diagonal): immune to far stray outliers
    for a_sp in (3.0, 4.0, 5.0, 6.0, 8.0):
        for a_d in (0.02, 0.03, 0.04):
            cfgs.append(("hybrid_robD", float(a_sp), float(a_d)))
    return cfgs


def build_minsize_configs():
    """Return list of (ms_family, n_abs, rho_frac) secondary-cluster retention rules."""
    cfgs = [
        ("main_only", -1, 1.0),
    ]
    for n_abs in (10, 20, 30, 50):
        cfgs.append(("abs_only", int(n_abs), 0.0))
    for rho in (0.005, 0.01, 0.02, 0.03, 0.05):
        cfgs.append(("frac_only", 1, float(rho)))
    for n_abs, rho in (
        (10, 0.005),
        (15, 0.01),
        (20, 0.005),
        (20, 0.01),   # current production setting
        (20, 0.02),
        (30, 0.01),
        (30, 0.02),
        (50, 0.02),
    ):
        cfgs.append(("hybrid_min", int(n_abs), float(rho)))
    return cfgs


def compute_radius(r_family, alpha_sp, alpha_d, spacing, diam, rob_diam):
    if r_family == "pure_sp":
        return alpha_sp * spacing
    if r_family == "pure_diam":
        return alpha_d * diam
    if r_family == "hybrid":
        return max(alpha_sp * spacing, alpha_d * diam)
    if r_family == "hybrid_robD":
        return max(alpha_sp * spacing, alpha_d * rob_diam)
    raise ValueError(f"Unknown radius family: {r_family}")


def keep_mask_from_components(lbl, sizes, n_pts, ms_family, n_abs, rho_frac):
    main_lbl = int(np.argmax(sizes))
    if ms_family == "main_only":
        return lbl == main_lbl
    min_size = max(n_abs, int(rho_frac * n_pts))
    return (lbl == main_lbl) | (sizes[lbl] >= min_size)


def cluster_and_sor(points, normals, r_family, alpha_sp, alpha_d, ms_family, n_abs, rho_frac):
    """Apply specified Euclidean clustering followed by calibrated per-fragment MAD-SOR (k=8, 3.5)."""
    _, uniq_idx = np.unique(points, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    p = points[uniq_idx]
    nrm = normals[uniq_idx] if normals is not None else None

    if r_family == "no_cluster":
        frag_lbl = np.zeros(len(p), dtype=int)
    else:
        tree = cKDTree(p)
        d2 = tree.query(p, k=min(2, len(p)))[0][:, -1]
        spacing = float(np.median(d2))
        diam = float(np.linalg.norm(np.ptp(p, axis=0)))
        rob_diam = float(np.linalg.norm(np.quantile(p, 0.99, axis=0) - np.quantile(p, 0.01, axis=0)))
        radius = compute_radius(r_family, alpha_sp, alpha_d, spacing, diam, rob_diam)
        g = tree.sparse_distance_matrix(tree, max_distance=radius, output_type="coo_matrix")
        _, lbl = connected_components(g)
        sizes = np.bincount(lbl)
        keep_cl = keep_mask_from_components(lbl, sizes, len(p), ms_family, n_abs, rho_frac)
        p = p[keep_cl]
        nrm = nrm[keep_cl] if nrm is not None else None
        _, frag_lbl = np.unique(lbl[keep_cl], return_inverse=True)

    # Calibrated per-fragment MAD-SOR (k=8, n_sigma=3.5) held strictly constant across all configs
    keep_sor = np.zeros(len(p), dtype=bool)
    for lb in np.unique(frag_lbl):
        idx = np.flatnonzero(frag_lbl == lb)
        part, _ = sor_filter(p[idx], k=8, n_sigma=3.5)
        keep_sor[idx] = cKDTree(part).query(p[idx])[0] < 1e-12
    return p[keep_sor], (nrm[keep_sor] if nrm is not None else None)


def build_level2_perturbed_leaf(p_clean, n_clean, stem_tree, spacing, rng):
    """Build controlled structural perturbation on clean leaf inliers.

    Returns two paired point clouds sharing identical main blade, detached tip,
    and near/medium noise patches:
      - nofar: main blade + genuine occlusion-detached tip + close patch + medium patch
      - withfar: nofar + far stray artifact (tests raw-ptp diameter inflation)

    gt_role codes:
      0 = genuine main blade (negative)
      1 = genuine occlusion-detached terminal fragment (negative)
      2 = close detached mislabeled patch, clearance ~4.5-7.0*sp (positive)
      3 = medium detached mislabeled patch, clearance ~8.0-14.0*sp (positive)
      4 = far stray scanner artifact, offset ~0.65-1.05*D (positive)
    """
    n = len(p_clean)
    centered = p_clean - np.median(p_clean, axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ vt[0]
    if stem_tree.query(p_clean[int(np.argmin(proj))])[0] > stem_tree.query(p_clean[int(np.argmax(proj))])[0]:
        proj = -proj

    # Create a genuine occlusion gap separating the distal blade segment on leaves with >= 400 pts
    if n >= 400:
        q_lo, q_hi = np.quantile(proj, [0.76, 0.85])
        main_mask = proj < q_lo
        tip_mask = proj > q_hi
        tip_idx = np.flatnonzero(tip_mask)
        # Mildly sparsify detached terminal fragment (~68% kept) to mimic realistic tip thinning
        n_keep_tip = max(28, int(round(0.68 * len(tip_idx))))
        if len(tip_idx) > n_keep_tip:
            tip_idx = np.sort(rng.choice(tip_idx, size=n_keep_tip, replace=False))
        main_idx = np.flatnonzero(main_mask)
        leaf_pts = np.vstack([p_clean[main_idx], p_clean[tip_idx]])
        leaf_nrm = np.vstack([n_clean[main_idx], n_clean[tip_idx]]) if n_clean is not None else None
        leaf_role = np.concatenate([
            np.zeros(len(main_idx), dtype=np.int8),
            np.ones(len(tip_idx), dtype=np.int8),
        ])
    else:
        leaf_pts = p_clean.copy()
        leaf_nrm = n_clean.copy() if n_clean is not None else None
        leaf_role = np.zeros(n, dtype=np.int8)

    leaf_tree = cKDTree(leaf_pts)
    d_clean = float(np.linalg.norm(np.ptp(leaf_pts, axis=0)))
    if leaf_nrm is not None and len(leaf_nrm) == len(leaf_pts):
        dirs = leaf_nrm / np.maximum(np.linalg.norm(leaf_nrm, axis=1, keepdims=True), 1e-12)
    else:
        dirs = rng.normal(size=(len(leaf_pts), 3))
        dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)

    def _place_patch(n_min, n_max, off_lo_sp, off_hi_sp, min_clear_sp, max_clear_sp, near_base=False):
        csize = int(rng.integers(n_min, n_max + 1))
        # Candidate anchors: optionally bias close patch near stem/proximal half where mislabels occur
        cand_pool = np.arange(len(leaf_pts))
        if near_base and len(leaf_pts) > 40:
            p_sub = (leaf_pts - np.median(leaf_pts, axis=0)) @ vt[0]
            if stem_tree.query(leaf_pts[int(np.argmin(p_sub))])[0] > stem_tree.query(leaf_pts[int(np.argmax(p_sub))])[0]:
                p_sub = -p_sub
            cand_pool = np.flatnonzero(p_sub <= np.quantile(p_sub, 0.45))
            if len(cand_pool) == 0:
                cand_pool = np.arange(len(leaf_pts))
        best_pts = None
        best_nrm = None
        for _ in range(40):
            anc = int(rng.choice(cand_pool))
            sign = float(rng.choice([-1.0, 1.0]))
            off = float(rng.uniform(off_lo_sp, off_hi_sp)) * spacing
            ctr = leaf_pts[anc] + sign * off * dirs[anc]
            # Compact coherent patch (internal spacing ~0.75*sp so patch itself stays connected)
            cloud = ctr + rng.normal(scale=0.65 * spacing, size=(csize, 3))
            min_d = float(np.min(leaf_tree.query(cloud)[0]))
            if min_clear_sp * spacing <= min_d <= max_clear_sp * spacing:
                best_pts = cloud
                best_nrm = np.tile(dirs[anc], (csize, 1))
                break
        if best_pts is None:
            anc = int(rng.choice(cand_pool))
            off = 0.5 * (off_lo_sp + off_hi_sp) * spacing
            ctr = leaf_pts[anc] + off * dirs[anc]
            best_pts = ctr + rng.normal(scale=0.65 * spacing, size=(csize, 3))
            best_nrm = np.tile(dirs[anc], (csize, 1))
        return best_pts, best_nrm

    # Close detached mislabeled patch (8-15 pts, min clearance 4.5-7.0 * sp, biased near base)
    pts_close, nrm_close = _place_patch(8, 15, 5.5, 7.8, 4.5, 7.2, near_base=True)
    # Medium detached mislabeled patch (10-18 pts, min clearance 8.5-14.0 * sp)
    pts_med, nrm_med = _place_patch(10, 18, 9.5, 14.5, 8.2, 14.5, near_base=False)

    # Far stray scanner artifact (4-6 pts at 0.65-1.05 * D away along random unit direction)
    fsize = int(rng.integers(4, 7))
    fdir = rng.normal(size=3)
    fdir /= max(float(np.linalg.norm(fdir)), 1e-12)
    fctr = np.median(leaf_pts, axis=0) + float(rng.uniform(0.65, 1.05)) * d_clean * fdir
    pts_far = fctr + rng.normal(scale=0.8 * spacing, size=(fsize, 3))
    nrm_far = np.tile(fdir, (fsize, 1))

    pts_nofar = np.vstack([leaf_pts, pts_close, pts_med])
    nrm_nofar = np.vstack([leaf_nrm, nrm_close, nrm_med]) if leaf_nrm is not None else None
    role_nofar = np.concatenate([
        leaf_role,
        np.full(len(pts_close), 2, dtype=np.int8),
        np.full(len(pts_med), 3, dtype=np.int8),
    ])

    pts_withfar = np.vstack([pts_nofar, pts_far])
    nrm_withfar = np.vstack([nrm_nofar, nrm_far]) if nrm_nofar is not None else None
    role_withfar = np.concatenate([
        role_nofar,
        np.full(len(pts_far), 4, dtype=np.int8),
    ])
    return pts_nofar, nrm_nofar, role_nofar, pts_withfar, nrm_withfar, role_withfar


def precompute_leaf_records(input_root, ref_model_root, seed=2026):
    """Load all 14 plants / 90 leaves, reference proxies, and paired L2 stress clouds."""
    rng = np.random.default_rng(seed)
    pairs = discover(input_root)
    leaves = []

    for plant_name, txt, csv_path in pairs:
        plant = load_pair(plant_name, txt, csv_path)
        stem_pts = plant.xyz[plant.labels == 0]
        stem, _ = fit_stem_centerline(stem_pts)
        stem_tree = cKDTree(stem)

        for lid in sorted(int(x) for x in set(plant.labels) if x > 0):
            raw_pts = plant.xyz[plant.labels == lid]
            raw_nrm = plant.normals[plant.labels == lid] if plant.normals is not None else None

            _, uniq_idx = np.unique(raw_pts, axis=0, return_index=True)
            uniq_idx = np.sort(uniq_idx)
            p = raw_pts[uniq_idx]
            nrm = raw_nrm[uniq_idx] if raw_nrm is not None else None

            tree = cKDTree(p)
            spacing = float(np.median(tree.query(p, k=min(2, len(p)))[0][:, -1]))
            diam = float(np.linalg.norm(np.ptp(p, axis=0)))
            rob_diam = float(np.linalg.norm(np.quantile(p, 0.99, axis=0) - np.quantile(p, 0.01, axis=0)))

            # Model-free distal-10% tip mask
            centered = p - np.median(p, axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            proj = centered @ vt[0]
            if stem_tree.query(p[int(np.argmin(proj))])[0] > stem_tree.query(p[int(np.argmax(proj))])[0]:
                proj = -proj
            tip10_mask = proj >= float(np.quantile(proj, 0.90))

            # Reference surface distances (read-only outputs/maize_v2)
            mpath = ref_model_root / f"plant_{plant_name}" / "leaves" / f"leaf_{lid:02d}_model.npz"
            if mpath.is_file():
                model = MaizeLeafModel.load(mpath)
                sv, _ = leaf_mesh_grid(model.blade.length, 220, 61)
                d_surf_sp = cKDTree(model.evaluate(sv)).query(p)[0] / max(spacing, 1e-12)
                clean_inlier_mask = d_surf_sp <= 2.0
                nat_outlier_mask = d_surf_sp > 5.0
            else:
                clean_inlier_mask = np.ones(len(p), dtype=bool)
                nat_outlier_mask = np.zeros(len(p), dtype=bool)

            p_clean = p[clean_inlier_mask]
            n_clean = nrm[clean_inlier_mask] if nrm is not None else None
            p_nf, n_nf, role_nf, p_wf, n_wf, role_wf = build_level2_perturbed_leaf(
                p_clean, n_clean, stem_tree, spacing, rng
            )

            # Spacing & diagonals on L2 clouds
            def _geom_stats(pts):
                t = cKDTree(pts)
                sp_ = float(np.median(t.query(pts, k=min(2, len(pts)))[0][:, -1]))
                d_ = float(np.linalg.norm(np.ptp(pts, axis=0)))
                rd_ = float(np.linalg.norm(np.quantile(pts, 0.99, axis=0) - np.quantile(pts, 0.01, axis=0)))
                return sp_, d_, rd_

            nf_sp, nf_d, nf_rd = _geom_stats(p_nf)
            wf_sp, wf_d, wf_rd = _geom_stats(p_wf)

            leaves.append({
                "plant": plant_name,
                "leaf_id": lid,
                "raw_pts": raw_pts,
                "raw_nrm": raw_nrm,
                "p": p,
                "nrm": nrm,
                "stem": stem,
                "spacing": spacing,
                "diam": diam,
                "rob_diam": rob_diam,
                "tip10_mask": tip10_mask,
                "clean_inlier_mask": clean_inlier_mask,
                "nat_outlier_mask": nat_outlier_mask,
                "p_clean": p_clean,
                "n_clean": n_clean,
                "p_nf": p_nf,
                "n_nf": n_nf,
                "role_nf": role_nf,
                "nf_stats": (nf_sp, nf_d, nf_rd),
                "p_wf": p_wf,
                "n_wf": n_wf,
                "role_wf": role_wf,
                "wf_stats": (wf_sp, wf_d, wf_rd),
            })
    return leaves


def _eval_radius_group(args):
    """Evaluate one radius config across all min-size rules on all 90 leaves (L1 + L2)."""
    r_cfg, minsize_cfgs, leaf_pack = args
    r_family, alpha_sp, alpha_d = r_cfg

    # Precompute connected components once per leaf for this radius config (L1, L2-nofar, L2-withfar)
    l1_comps = []
    nf_comps = []
    wf_comps = []

    for lf in leaf_pack:
        # L1 real cloud
        p = lf["p"]
        t1 = cKDTree(p)
        r1 = compute_radius(r_family, alpha_sp, alpha_d, lf["spacing"], lf["diam"], lf["rob_diam"])
        _, lbl1 = connected_components(t1.sparse_distance_matrix(t1, max_distance=r1, output_type="coo_matrix"))
        sz1 = np.bincount(lbl1)
        l1_comps.append((lbl1, sz1, r1))

        # L2 without far outlier
        pnf = lf["p_nf"]
        tnf = cKDTree(pnf)
        rnf = compute_radius(r_family, alpha_sp, alpha_d, *lf["nf_stats"])
        _, lbl_nf = connected_components(tnf.sparse_distance_matrix(tnf, max_distance=rnf, output_type="coo_matrix"))
        nf_comps.append((lbl_nf, np.bincount(lbl_nf)))

        # L2 with far outlier
        pwf = lf["p_wf"]
        twf = cKDTree(pwf)
        rwf = compute_radius(r_family, alpha_sp, alpha_d, *lf["wf_stats"])
        _, lbl_wf = connected_components(twf.sparse_distance_matrix(twf, max_distance=rwf, output_type="coo_matrix"))
        wf_comps.append((lbl_wf, np.bincount(lbl_wf), rwf / max(rnf, 1e-12)))

    rad_inflation_med = float(np.median([x[2] for x in wf_comps]))
    rad_inflation_p95 = float(np.quantile([x[2] for x in wf_comps], 0.95))

    def _f1_mcc(tp_, fp_, tn_, fn_):
        prec = tp_ / max(tp_ + fp_, 1)
        rec = tp_ / max(tp_ + fn_, 1)
        f1 = 2.0 * prec * rec / max(prec + rec, 1e-12)
        denom = float((tp_ + fp_) * (tp_ + fn_) * (tn_ + fp_) * (tn_ + fn_))
        mcc = (tp_ * tn_ - fp_ * fn_) / np.sqrt(denom) if denom > 0 else 0.0
        return prec, rec, f1, float(mcc)

    rows = []
    for ms_family, n_abs, rho_frac in minsize_cfgs:
        # Level 1 accumulators
        l1_tot = l1_rem = 0
        l1_clean_tot = l1_clean_rem = 0
        l1_tip_tot = l1_tip_rem = 0
        l1_nat_tot = l1_nat_rem = 0
        l1_total_frags = 0
        l1_multi_frag_leaves = 0
        l1_leaf_rem_fracs = []
        l1_leaf_clean_fprs = []

        # Level 2 With-Far accumulators
        wf_tp = wf_fp = wf_tn = wf_fn = 0
        wf_close_tot = wf_close_det = 0
        wf_med_tot = wf_med_det = 0
        wf_far_tot = wf_far_det = 0
        wf_tip_tot = wf_tip_kept = 0
        wf_main_tot = wf_main_rem = 0
        wf_plant = {}

        # Level 2 No-Far accumulators
        nf_tp = nf_fp = nf_tn = nf_fn = 0
        nf_close_tot = nf_close_det = 0
        nf_med_tot = nf_med_det = 0
        nf_plant = {}

        for i, lf in enumerate(leaf_pack):
            # --- Level 1 ---
            lbl1, sz1, _ = l1_comps[i]
            n1 = len(lbl1)
            keep1 = keep_mask_from_components(lbl1, sz1, n1, ms_family, n_abs, rho_frac)
            rem1 = ~keep1
            n_frags = int(len(np.unique(lbl1[keep1])))
            l1_total_frags += n_frags
            if n_frags >= 2:
                l1_multi_frag_leaves += 1

            l1_tot += n1
            l1_rem += int(np.sum(rem1))
            l1_leaf_rem_fracs.append(float(np.mean(rem1)))

            cmask = lf["clean_inlier_mask"]
            tclean = lf["tip10_mask"] & cmask
            nmask = lf["nat_outlier_mask"]
            c_tot = int(np.sum(cmask))
            c_rem = int(np.sum(rem1 & cmask))
            l1_clean_tot += c_tot
            l1_clean_rem += c_rem
            l1_leaf_clean_fprs.append(c_rem / max(c_tot, 1))
            l1_tip_tot += int(np.sum(tclean))
            l1_tip_rem += int(np.sum(rem1 & tclean))
            l1_nat_tot += int(np.sum(nmask))
            l1_nat_rem += int(np.sum(rem1 & nmask))

            # --- Level 2 No-Far ---
            lbl_nf, sz_nf = nf_comps[i]
            role_nf = lf["role_nf"]
            keep_nf = keep_mask_from_components(lbl_nf, sz_nf, len(role_nf), ms_family, n_abs, rho_frac)
            pred_nf = ~keep_nf
            pos_nf = role_nf >= 2
            neg_nf = ~pos_nf
            ltp = int(np.sum(pred_nf & pos_nf))
            lfp = int(np.sum(pred_nf & neg_nf))
            ltn = int(np.sum((~pred_nf) & neg_nf))
            lfn = int(np.sum((~pred_nf) & pos_nf))
            nf_tp += ltp; nf_fp += lfp; nf_tn += ltn; nf_fn += lfn
            nf_close_tot += int(np.sum(role_nf == 2))
            nf_close_det += int(np.sum(pred_nf & (role_nf == 2)))
            nf_med_tot += int(np.sum(role_nf == 3))
            nf_med_det += int(np.sum(pred_nf & (role_nf == 3)))
            ps = nf_plant.setdefault(lf["plant"], [0, 0, 0, 0])
            ps[0] += ltp; ps[1] += lfp; ps[2] += ltn; ps[3] += lfn

            # --- Level 2 With-Far ---
            lbl_wf, sz_wf, _ = wf_comps[i]
            role_wf = lf["role_wf"]
            keep_wf = keep_mask_from_components(lbl_wf, sz_wf, len(role_wf), ms_family, n_abs, rho_frac)
            pred_wf = ~keep_wf
            pos_wf = role_wf >= 2
            neg_wf = ~pos_wf
            wtp = int(np.sum(pred_wf & pos_wf))
            wfp = int(np.sum(pred_wf & neg_wf))
            wtn = int(np.sum((~pred_wf) & neg_wf))
            wfn = int(np.sum((~pred_wf) & pos_wf))
            wf_tp += wtp; wf_fp += wfp; wf_tn += wtn; wf_fn += wfn
            wf_close_tot += int(np.sum(role_wf == 2))
            wf_close_det += int(np.sum(pred_wf & (role_wf == 2)))
            wf_med_tot += int(np.sum(role_wf == 3))
            wf_med_det += int(np.sum(pred_wf & (role_wf == 3)))
            wf_far_tot += int(np.sum(role_wf == 4))
            wf_far_det += int(np.sum(pred_wf & (role_wf == 4)))
            wf_tip_tot += int(np.sum(role_wf == 1))
            wf_tip_kept += int(np.sum(keep_wf & (role_wf == 1)))
            wf_main_tot += int(np.sum(role_wf == 0))
            wf_main_rem += int(np.sum(pred_wf & (role_wf == 0)))
            ps2 = wf_plant.setdefault(lf["plant"], [0, 0, 0, 0])
            ps2[0] += wtp; ps2[1] += wfp; ps2[2] += wtn; ps2[3] += wfn

        wf_prec, wf_rec, wf_f1, wf_mcc = _f1_mcc(wf_tp, wf_fp, wf_tn, wf_fn)
        wf_pf1 = [_f1_mcc(*v)[2] for v in wf_plant.values()]
        wf_pmcc = [_f1_mcc(*v)[3] for v in wf_plant.values()]
        nf_pmcc = [_f1_mcc(*v)[3] for v in nf_plant.values()]

        rows.append({
            "r_family": r_family,
            "alpha_sp": alpha_sp,
            "alpha_d": alpha_d,
            "ms_family": ms_family,
            "n_abs": n_abs,
            "rho_frac": rho_frac,
            "rad_inflation_med": rad_inflation_med,
            "rad_inflation_p95": rad_inflation_p95,
            # Level 1 (90 unperturbed real leaves)
            "l1_removed_pct": 100.0 * l1_rem / max(l1_tot, 1),
            "l1_leaf_rem_med_pct": 100.0 * float(np.median(l1_leaf_rem_fracs)),
            "l1_leaf_rem_p95_pct": 100.0 * float(np.quantile(l1_leaf_rem_fracs, 0.95)),
            "l1_leaf_rem_max_pct": 100.0 * float(np.max(l1_leaf_rem_fracs)),
            "l1_total_frags": l1_total_frags,
            "l1_multi_frag_leaves": l1_multi_frag_leaves,
            "l1_clean_fpr_pct": 100.0 * l1_clean_rem / max(l1_clean_tot, 1),
            "l1_clean_fpr_leaf_p95_pct": 100.0 * float(np.quantile(l1_leaf_clean_fprs, 0.95)),
            "l1_clean_fpr_leaf_max_pct": 100.0 * float(np.max(l1_leaf_clean_fprs)),
            "l1_tip10_clean_fpr_pct": 100.0 * l1_tip_rem / max(l1_tip_tot, 1),
            "l1_nat_offsurf_recall_pct": 100.0 * l1_nat_rem / max(l1_nat_tot, 1),
            # Level 2 With-Far (structural stress test)
            "l2_precision": wf_prec,
            "l2_recall_all": wf_rec,
            "l2_recall_close": wf_close_det / max(wf_close_tot, 1),
            "l2_recall_med": wf_med_det / max(wf_med_tot, 1),
            "l2_recall_far": wf_far_det / max(wf_far_tot, 1),
            "l2_det_tip_retention_pct": 100.0 * wf_tip_kept / max(wf_tip_tot, 1),
            "l2_main_inlier_fpr_pct": 100.0 * wf_main_rem / max(wf_main_tot, 1),
            "l2_f1_pooled": wf_f1,
            "l2_mcc_pooled": wf_mcc,
            "l2_f1_plant_mean": float(np.mean(wf_pf1)),
            "l2_f1_plant_std": float(np.std(wf_pf1)),
            "l2_mcc_plant_mean": float(np.mean(wf_pmcc)),
            "l2_mcc_plant_std": float(np.std(wf_pmcc)),
            # Level 2 No-Far comparison (isolates ptp diameter inflation)
            "l2_nofar_mcc_plant_mean": float(np.mean(nf_pmcc)),
            "l2_nofar_recall_close": nf_close_det / max(nf_close_tot, 1),
            "l2_nofar_recall_med": nf_med_det / max(nf_med_tot, 1),
        })
    return rows


def eval_no_cluster_baseline(leaves):
    """Evaluate no_cluster baseline (0% removed by clustering) on L1 & L2."""
    return {
        "r_family": "no_cluster",
        "alpha_sp": 0.0,
        "alpha_d": 0.0,
        "ms_family": "none",
        "n_abs": 0,
        "rho_frac": 0.0,
        "rad_inflation_med": 1.0,
        "rad_inflation_p95": 1.0,
        "l1_removed_pct": 0.0,
        "l1_leaf_rem_med_pct": 0.0,
        "l1_leaf_rem_p95_pct": 0.0,
        "l1_leaf_rem_max_pct": 0.0,
        "l1_total_frags": len(leaves),
        "l1_multi_frag_leaves": 0,
        "l1_clean_fpr_pct": 0.0,
        "l1_clean_fpr_leaf_p95_pct": 0.0,
        "l1_clean_fpr_leaf_max_pct": 0.0,
        "l1_tip10_clean_fpr_pct": 0.0,
        "l1_nat_offsurf_recall_pct": 0.0,
        "l2_precision": 0.0,
        "l2_recall_all": 0.0,
        "l2_recall_close": 0.0,
        "l2_recall_med": 0.0,
        "l2_recall_far": 0.0,
        "l2_det_tip_retention_pct": 100.0,
        "l2_main_inlier_fpr_pct": 0.0,
        "l2_f1_pooled": 0.0,
        "l2_mcc_pooled": 0.0,
        "l2_f1_plant_mean": 0.0,
        "l2_f1_plant_std": 0.0,
        "l2_mcc_plant_mean": 0.0,
        "l2_mcc_plant_std": 0.0,
        "l2_nofar_mcc_plant_mean": 0.0,
        "l2_nofar_recall_close": 0.0,
        "l2_nofar_recall_med": 0.0,
    }


def fit_leaf_with_custom_cluster(points, stem, normals, prior, cfg, leaf_id,
                                 r_family, alpha_sp, alpha_d, ms_family, n_abs, rho_frac):
    """Run fit_leaf using custom Euclidean clustering + fixed calibrated MAD-SOR."""
    import maize_leaf.surface as surf_mod

    cleaned_p, cleaned_n = cluster_and_sor(
        points, normals, r_family, alpha_sp, alpha_d, ms_family, n_abs, rho_frac
    )
    orig_clean = surf_mod.clean_leaf_points
    try:
        surf_mod.clean_leaf_points = lambda pts, nrm=None: (pts, nrm)
        res = fit_leaf(cleaned_p, stem, normals=cleaned_n, prior=prior, cfg=cfg, leaf_id=leaf_id)
    finally:
        surf_mod.clean_leaf_points = orig_clean
    return res, cleaned_p


def _eval_ref_one(args):
    key, clean_pts, clean_nrm, stem, leaf_id, prior, cfg = args
    res, _ = fit_leaf_with_custom_cluster(
        clean_pts, stem, clean_nrm, prior, cfg, leaf_id,
        "no_cluster", 0.0, 0.0, "none", 0, 0.0
    )
    return key, {
        "length": float(res["traits"]["completed_length"]),
        "obs_length": float(res["traits"]["observed_length"]),
        "max_width": float(res["traits"]["max_width"]),
        "area": float(res["traits"]["surface_area"]),
    }


def _eval_l3a_one(args):
    (label, r_family, alpha_sp, alpha_d, ms_family, n_abs, rho_frac,
     pert_pts, pert_nrm, stem, leaf_id, clean_pts, ref_len, ref_wid, ref_area, prior, cfg) = args
    try:
        res, _ = fit_leaf_with_custom_cluster(
            pert_pts, stem, pert_nrm, prior, cfg, leaf_id,
            r_family, alpha_sp, alpha_d, ms_family, n_abs, rho_frac
        )
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


def _eval_l3b_one(args):
    (label, r_family, alpha_sp, alpha_d, ms_family, n_abs, rho_frac,
     plant, leaf_id, raw_pts, raw_nrm, stem, clean_ref, prior, cfg) = args
    try:
        res, kept_p = fit_leaf_with_custom_cluster(
            raw_pts, stem, raw_nrm, prior, cfg, leaf_id,
            r_family, alpha_sp, alpha_d, ms_family, n_abs, rho_frac
        )
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
            float(res["traits"]["completed_length"]),
            float(res["traits"]["max_width"]),
            float(res["traits"]["surface_area"]),
            len(kept_p),
            None,
        )
    except Exception as e:
        return (label, plant, leaf_id, False, None, None, None, None, None, None, None, None, None, str(e))


def select_best_by_family(grid_rows):
    """Select top configuration per radius family under strict real-scan clean-preservation guards."""
    best = {}
    for fam in ("pure_sp", "pure_diam", "hybrid", "hybrid_robD"):
        sub = [r for r in grid_rows if r["r_family"] == fam]
        feasible = [
            r for r in sub
            if r["l1_clean_fpr_pct"] <= 0.5
            and r["l1_clean_fpr_leaf_max_pct"] <= 5.0
            and r["l2_det_tip_retention_pct"] >= 95.0
        ]
        pool = feasible if feasible else sub
        best[fam] = max(
            pool,
            key=lambda r: (r["l2_mcc_plant_mean"], r["l1_nat_offsurf_recall_pct"], -r["l1_clean_fpr_pct"]),
        )
    return best


def run_level3a(leaves, candidates, input_root, workers=10):
    pairs = discover(input_root)
    prior, _ = calibrate_dataset_prior(pairs)
    cfg = ModelConfig(nu=90, nv=25, allow_tip_completion=True)

    by_plant = {}
    for lf in leaves:
        by_plant.setdefault(lf["plant"], []).append(lf)
    # One representative leaf per plant (largest clean point count with >= 400 pts)
    rep_leaves = [max(by_plant[pl], key=lambda x: len(x["p_clean"])) for pl in sorted(by_plant)]

    t_ref = time.time()
    ref_jobs = []
    clean_map = {}
    for lf in rep_leaves:
        key = (lf["plant"], lf["leaf_id"])
        clean_map[key] = lf["p_clean"]
        ref_jobs.append((key, lf["p_clean"], lf["n_clean"], lf["stem"], lf["leaf_id"], prior, cfg))

    ref_fits = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for key, val in ex.map(_eval_ref_one, ref_jobs):
            val["clean_pts"] = clean_map[key]
            ref_fits[key] = val
    print(f"  Computed {len(ref_fits)} clean reference fits in {time.time() - t_ref:.1f}s", flush=True)

    summary = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for label, r_fam, a_sp, a_d, ms_fam, n_abs, rho in candidates:
            t0 = time.time()
            jobs = []
            for lf in rep_leaves:
                key = (lf["plant"], lf["leaf_id"])
                ref = ref_fits[key]
                jobs.append((
                    label, r_fam, a_sp, a_d, ms_fam, n_abs, rho,
                    lf["p_wf"], lf["n_wf"], lf["stem"], lf["leaf_id"],
                    ref["clean_pts"], ref["length"], ref["max_width"], ref["area"],
                    prior, cfg,
                ))
            results = list(ex.map(_eval_l3a_one, jobs))
            ok = [r for r in results if r[1]]
            failed = len(results) - len(ok)
            rec = {
                "label": label,
                "r_family": r_fam,
                "alpha_sp": a_sp,
                "alpha_d": a_d,
                "ms_family": ms_fam,
                "n_abs": n_abs,
                "rho_frac": rho,
                "n_leaves": len(rep_leaves),
                "passed": len(ok),
                "failed": failed,
                "med_clean_p95": float(np.median([r[2] for r in ok])) if ok else None,
                "med_clean_rmse": float(np.median([r[3] for r in ok])) if ok else None,
                "med_support_mean": float(np.median([r[4] for r in ok])) if ok else None,
                "med_length_err_pct": float(np.median([r[5] for r in ok])) if ok else None,
                "med_width_err_pct": float(np.median([r[6] for r in ok])) if ok else None,
                "med_area_err_pct": float(np.median([r[7] for r in ok])) if ok else None,
                "p95_area_err_pct": float(np.quantile([r[7] for r in ok], 0.95)) if ok else None,
                "elapsed_s": round(time.time() - t0, 2),
            }
            summary.append(rec)
            print(
                f"  [L3A] {label:28s}: pass={rec['passed']:2d}/{rec['n_leaves']}, "
                f"rmse={rec['med_clean_rmse']:.5f}, p95={rec['med_clean_p95']:.5f}, "
                f"supp={rec['med_support_mean']:.5f}, "
                f"len={rec['med_length_err_pct']:.2f}%, wid={rec['med_width_err_pct']:.2f}%, "
                f"area={rec['med_area_err_pct']:.2f}% (p95={rec['p95_area_err_pct']:.2f}%) [{rec['elapsed_s']}s]",
                flush=True,
            )
    return summary, prior, cfg


def run_level3b_full90(leaves, finalists, prior, cfg, workers=10):
    summary = []
    per_leaf_map = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for label, r_fam, a_sp, a_d, ms_fam, n_abs, rho in finalists:
            t0 = time.time()
            jobs = [
                (label, r_fam, a_sp, a_d, ms_fam, n_abs, rho,
                 lf["plant"], lf["leaf_id"], lf["raw_pts"], lf["raw_nrm"],
                 lf["stem"], lf["p_clean"], prior, cfg)
                for lf in leaves
            ]
            results = list(ex.map(_eval_l3b_one, jobs))
            ok = [r for r in results if r[3]]
            failed_list = [f"{r[1]}_L{r[2]}: {r[13]}" for r in results if not r[3]]
            per_leaf_map[label] = {(r[1], r[2]): r for r in ok}
            rec = {
                "label": label,
                "r_family": r_fam,
                "alpha_sp": a_sp,
                "alpha_d": a_d,
                "ms_family": ms_fam,
                "n_abs": n_abs,
                "rho_frac": rho,
                "n_leaves": len(leaves),
                "passed": len(ok),
                "failed": len(failed_list),
                "failed_details": failed_list[:5],
                "med_raw_p95": float(np.median([r[4] for r in ok])) if ok else None,
                "med_clean_p95": float(np.median([r[5] for r in ok])) if ok else None,
                "med_clean_median": float(np.median([r[6] for r in ok])) if ok else None,
                "med_support_clean": float(np.median([r[7] for r in ok])) if ok else None,
                "med_obs_length": float(np.median([r[8] for r in ok])) if ok else None,
                "med_comp_length": float(np.median([r[9] for r in ok])) if ok else None,
                "med_max_width": float(np.median([r[10] for r in ok])) if ok else None,
                "med_surface_area": float(np.median([r[11] for r in ok])) if ok else None,
                "total_kept_points": int(sum(r[12] for r in ok)) if ok else 0,
                "elapsed_s": round(time.time() - t0, 2),
            }
            summary.append(rec)
            print(
                f"  [L3B 90-leaf] {label:28s}: pass={rec['passed']:2d}/90, "
                f"raw_p95={rec['med_raw_p95']:.6f}, clean_p95={rec['med_clean_p95']:.6f}, "
                f"supp={rec['med_support_clean']:.6f}, obsL={rec['med_obs_length']:.5f}, "
                f"area={rec['med_surface_area']:.6f} [{rec['elapsed_s']}s]",
                flush=True,
            )

    # Compare per-leaf area/length shifts against current_impl
    cur_map = per_leaf_map.get("current_impl", {})
    for rec in summary:
        m = per_leaf_map.get(rec["label"], {})
        common = sorted(set(cur_map) & set(m))
        if common and rec["label"] != "current_impl":
            len_diffs = [100.0 * abs(m[k][8] - cur_map[k][8]) / max(cur_map[k][8], 1e-9) for k in common]
            area_diffs = [100.0 * abs(m[k][11] - cur_map[k][11]) / max(cur_map[k][11], 1e-9) for k in common]
            rec["vs_current_len_diff_med_pct"] = float(np.median(len_diffs))
            rec["vs_current_len_diff_max_pct"] = float(np.max(len_diffs))
            rec["vs_current_area_diff_med_pct"] = float(np.median(area_diffs))
            rec["vs_current_area_diff_max_pct"] = float(np.max(area_diffs))
            rec["vs_current_clean_p95_improved"] = int(sum(m[k][5] < cur_map[k][5] - 1e-9 for k in common))
            rec["vs_current_clean_p95_worsened"] = int(sum(m[k][5] > cur_map[k][5] + 1e-9 for k in common))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--ref-models", type=Path, default=ROOT / "outputs/maize_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/euclidean_cluster_sensitivity")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--skip-l3", action="store_true")
    parser.add_argument("--skip-l3b", action="store_true")
    args = parser.parse_args()

    t_start = time.time()
    print("Loading 14 plants / 90 leaves and building Level 1 & Level 2 datasets...", flush=True)
    leaves = precompute_leaf_records(args.input_root, args.ref_models, seed=args.seed)
    n_with_tip = sum(1 for lf in leaves if np.any(lf["role_wf"] == 1))
    print(
        f"Loaded {len(leaves)} leaves (14 plants, {n_with_tip} leaves with synthetic occlusion-detached tips) "
        f"in {time.time() - t_start:.2f}s",
        flush=True,
    )

    # Lightweight pack for L1 & L2 workers
    leaf_pack = [
        {
            "plant": lf["plant"],
            "leaf_id": lf["leaf_id"],
            "p": lf["p"],
            "spacing": lf["spacing"],
            "diam": lf["diam"],
            "rob_diam": lf["rob_diam"],
            "tip10_mask": lf["tip10_mask"],
            "clean_inlier_mask": lf["clean_inlier_mask"],
            "nat_outlier_mask": lf["nat_outlier_mask"],
            "p_nf": lf["p_nf"],
            "role_nf": lf["role_nf"],
            "nf_stats": lf["nf_stats"],
            "p_wf": lf["p_wf"],
            "role_wf": lf["role_wf"],
            "wf_stats": lf["wf_stats"],
        }
        for lf in leaves
    ]

    r_cfgs = build_radius_configs()
    ms_cfgs = build_minsize_configs()
    print(f"Sweeping {len(r_cfgs)} radius configs x {len(ms_cfgs)} min-size rules + no_cluster "
          f"= {len(r_cfgs) * len(ms_cfgs) + 1} configs...", flush=True)

    t_grid = time.time()
    grid_rows = [eval_no_cluster_baseline(leaves)]
    tasks = [(r_cfg, ms_cfgs, leaf_pack) for r_cfg in r_cfgs]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for batch in ex.map(_eval_radius_group, tasks):
            grid_rows.extend(batch)
    print(f"Evaluated {len(grid_rows)} configurations across 90 leaves in {time.time() - t_grid:.2f}s", flush=True)

    cur = next(
        r for r in grid_rows
        if r["r_family"] == "hybrid" and r["alpha_sp"] == 5.0 and r["alpha_d"] == 0.03
        and r["ms_family"] == "hybrid_min" and r["n_abs"] == 20 and r["rho_frac"] == 0.01
    )
    main_only_cur = next(
        r for r in grid_rows
        if r["r_family"] == "hybrid" and r["alpha_sp"] == 5.0 and r["alpha_d"] == 0.03
        and r["ms_family"] == "main_only"
    )
    rob_cur = next(
        r for r in grid_rows
        if r["r_family"] == "hybrid_robD" and r["alpha_sp"] == 5.0 and r["alpha_d"] == 0.03
        and r["ms_family"] == "hybrid_min" and r["n_abs"] == 20 and r["rho_frac"] == 0.01
    )
    best_by_fam = select_best_by_family(grid_rows)

    def _fmt(tag, r):
        return (
            f"  {tag:16s} | {r['r_family']:11s}({r['alpha_sp']:g},{r['alpha_d']:g}) "
            f"{r['ms_family']:10s}({r['n_abs']},{r['rho_frac']:g}) | "
            f"MCC(wf/nf)={r['l2_mcc_plant_mean']:.4f}/{r['l2_nofar_mcc_plant_mean']:.4f}, "
            f"Rec(cl/md/fr)={r['l2_recall_close']:.2f}/{r['l2_recall_med']:.2f}/{r['l2_recall_far']:.2f}, "
            f"TipRet={r['l2_det_tip_retention_pct']:.1f}%, "
            f"L1(rem={r['l1_removed_pct']:.2f}%, cFPR={r['l1_clean_fpr_pct']:.2f}%, "
            f"maxLeaf={r['l1_clean_fpr_leaf_max_pct']:.1f}%, natRec={r['l1_nat_offsurf_recall_pct']:.1f}%, "
            f"frags={r['l1_total_frags']})"
        )

    print("\n=== Key Baselines & Best per Radius Family ===", flush=True)
    print(_fmt("current_impl", cur), flush=True)
    print(_fmt("main_only_cur", main_only_cur), flush=True)
    print(_fmt("robD_same_thr", rob_cur), flush=True)
    for fam, r in best_by_fam.items():
        print(_fmt(f"best_{fam}", r), flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "euclidean_cluster_grid.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        w.writeheader()
        w.writerows(grid_rows)

    l3a_summary = []
    l3b_summary = []
    if not args.skip_l3:
        print("\n=== Level 3A: Perturbed 14-Leaf (Plant-Stratified) End-to-End Reconstruction ===", flush=True)
        raw_cands = [
            ("no_cluster", "no_cluster", 0.0, 0.0, "none", 0, 0.0),
            ("current_impl", "hybrid", 5.0, 0.03, "hybrid_min", 20, 0.01),
            ("main_only_cur", "hybrid", 5.0, 0.03, "main_only", -1, 1.0),
            ("pure_sp_3_m20_1p", "pure_sp", 3.0, 0.0, "hybrid_min", 20, 0.01),
            ("pure_sp_4_m20_1p", "pure_sp", 4.0, 0.0, "hybrid_min", 20, 0.01),
            ("pure_sp_5_m20_1p", "pure_sp", 5.0, 0.0, "hybrid_min", 20, 0.01),
            ("pure_sp_8_m20_1p", "pure_sp", 8.0, 0.0, "hybrid_min", 20, 0.01),
            ("pure_diam_003_m20_1p", "pure_diam", 0.0, 0.03, "hybrid_min", 20, 0.01),
            ("hybrid_4_002_m20_1p", "hybrid", 4.0, 0.02, "hybrid_min", 20, 0.01),
            ("robD_5_003_m20_1p", "hybrid_robD", 5.0, 0.03, "hybrid_min", 20, 0.01),
            ("robD_4_002_m20_1p", "hybrid_robD", 4.0, 0.02, "hybrid_min", 20, 0.01),
            ("robD_4_002_m30_1p", "hybrid_robD", 4.0, 0.02, "hybrid_min", 30, 0.01),
        ]
        # Add data-selected best per family if not already covered
        seen_keys = {c[1:] for c in raw_cands}
        for fam in ("pure_sp", "hybrid", "hybrid_robD"):
            r = best_by_fam[fam]
            key = (r["r_family"], r["alpha_sp"], r["alpha_d"], r["ms_family"], r["n_abs"], r["rho_frac"])
            if key not in seen_keys:
                lbl = f"best_{fam}"
                raw_cands.append((lbl, *key))
                seen_keys.add(key)

        l3a_summary, prior, cfg = run_level3a(leaves, raw_cands, args.input_root, workers=args.workers)

        if not args.skip_l3b:
            print("\n=== Level 3B: Full 90-Leaf Unperturbed Real-Scan Audit ===", flush=True)
            finalists = [
                ("no_cluster", "no_cluster", 0.0, 0.0, "none", 0, 0.0),
                ("main_only_cur", "hybrid", 5.0, 0.03, "main_only", -1, 1.0),
                ("current_impl", "hybrid", 5.0, 0.03, "hybrid_min", 20, 0.01),
                ("robD_5_003_m20_1p", "hybrid_robD", 5.0, 0.03, "hybrid_min", 20, 0.01),
            ]
            seen_f = {c[1:] for c in finalists}
            # Add top 3 distinct L3A winners (passing all 14 leaves)
            non_base = [
                r for r in l3a_summary
                if r["label"] not in ("no_cluster", "main_only_cur", "current_impl")
                and r["passed"] == r["n_leaves"]
            ]
            non_base.sort(key=lambda r: (r["med_clean_rmse"] + r["med_support_mean"], r["med_area_err_pct"]))
            for r in non_base:
                key = (r["r_family"], r["alpha_sp"], r["alpha_d"], r["ms_family"], r["n_abs"], r["rho_frac"])
                if key not in seen_f:
                    finalists.append((r["label"], *key))
                    seen_f.add(key)
                if len(finalists) >= 7:
                    break
            l3b_summary = run_level3b_full90(leaves, finalists, prior, cfg, workers=args.workers)

    # Real-dataset radius dominance & diameter-inflation diagnostics
    diag = {
        "n_leaves": len(leaves),
        "n_plants": len({lf["plant"] for lf in leaves}),
        "raw_points_total": int(sum(len(lf["raw_pts"]) for lf in leaves)),
        "dedup_points_total": int(sum(len(lf["p"]) for lf in leaves)),
        "real_5sp_dominates_leaves": int(sum(5.0 * lf["spacing"] >= 0.03 * lf["diam"] for lf in leaves)),
        "real_003D_dominates_leaves": int(sum(0.03 * lf["diam"] > 5.0 * lf["spacing"] for lf in leaves)),
        "real_diam_to_robD_ratio_med": float(np.median([lf["diam"] / max(lf["rob_diam"], 1e-12) for lf in leaves])),
        "real_diam_to_robD_ratio_p95": float(np.quantile([lf["diam"] / max(lf["rob_diam"], 1e-12) for lf in leaves], 0.95)),
        "real_diam_to_robD_ratio_max": float(np.max([lf["diam"] / max(lf["rob_diam"], 1e-12) for lf in leaves])),
        "real_leaves_diam_gt_1p25_robD": int(sum(lf["diam"] > 1.25 * lf["rob_diam"] for lf in leaves)),
    }

    report = {
        "seed": args.seed,
        "dataset": str(args.input_root),
        "fixed_downstream_sor": {"method": "med_mad", "k": 8, "n_sigma": 3.5},
        "dataset_cluster_diagnostics": diag,
        "n_radius_configs": len(r_cfgs),
        "n_minsize_configs": len(ms_cfgs),
        "n_total_grid_configs": len(grid_rows),
        "current_impl_l1_l2": cur,
        "main_only_cur_l1_l2": main_only_cur,
        "robD_same_thr_l1_l2": rob_cur,
        "best_by_radius_family": best_by_fam,
        "level3a_perturbed_14leaf": l3a_summary,
        "level3b_real_90leaf": l3b_summary,
        "total_elapsed_s": round(time.time() - t_start, 2),
    }
    write_json(args.output / "euclidean_cluster_report.json", report)
    print(f"\nSaved report to {args.output / 'euclidean_cluster_report.json'} (total {report['total_elapsed_s']}s)", flush=True)


if __name__ == "__main__":
    main()
