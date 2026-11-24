"""Real-maize holdout occlusion benchmark on intact leaves from dataset/.

Selects dense, intact-tip real maize leaves, fits full reference surfaces, then masks:
  1. Interior hole (central 15% length x 50% width)
  2. One-sided margin notch (20% length on one side)
  3. Distal tip truncation (15% of leaf length removed)
Evaluates held-out 3D point-to-surface error, length error, and area error.
"""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".vendor")]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import numpy as np
from scipy.spatial import cKDTree
from maize_leaf.geometry import build_normal_aligned_frame, clean_leaf_points, extract_midrib, fit_stem_centerline
from maize_leaf.io import discover, load_pair, write_json
from maize_leaf.morphology import extract_width_profile
from maize_leaf.pipeline import calibrate_dataset_prior
from maize_leaf.surface import ModelConfig, fit_leaf


def evaluate_holdout(input_root, output_dir, max_leaves=8):
    pairs = discover(input_root)
    prior, n_cal = calibrate_dataset_prior(pairs)
    cfg = ModelConfig(nu=90, nv=25, allow_tip_completion=True)

    candidates = []
    for name, txt, csv_path in pairs:
        plant = load_pair(name, txt, csv_path)
        stem, _ = fit_stem_centerline(plant.xyz[plant.labels == 0])
        for lid in sorted(int(x) for x in set(plant.labels) if x > 0):
            pts = plant.xyz[plant.labels == lid]
            if len(pts) < 2000:
                continue
            nrm = plant.normals[plant.labels == lid] if plant.normals is not None else None
            pts, nrm = clean_leaf_points(pts, nrm)
            midrib, l_obs, s_proj, _, _ = extract_midrib(pts, stem)
            frame = build_normal_aligned_frame(midrib, l_obs, pts, s_proj, normals=nrm)
            _, b_p, _ = frame.evaluate(s_proj)
            lat = np.sum((pts - midrib(s_proj)) * b_p, axis=1)
            prof = extract_width_profile(s_proj, lat, l_obs)
            w_max = float(np.quantile(prof["w_obs"], 0.95))
            tip_ratio = float(np.median(prof["w_obs"][-2:]) / max(w_max, 1e-9))
            if tip_ratio < 0.25:
                candidates.append((tip_ratio, name, lid, pts, nrm, stem, l_obs, s_proj, lat, w_max))

    candidates.sort(key=lambda x: x[0])
    selected = candidates[:max_leaves]
    records = []

    for _, name, lid, pts, nrm, stem, l_obs, s_proj, lat, w_max in selected:
        ref = fit_leaf(pts, stem, normals=nrm, prior=prior, cfg=ModelConfig(allow_tip_completion=False), leaf_id=lid)
        ref_area = ref["traits"]["surface_area"]
        ref_len = ref["traits"]["observed_length"]
        u = s_proj / l_obs

        scenarios = {
            "hole": ((u - 0.50) / 0.08) ** 2 + (lat / (0.28 * w_max)) ** 2 < 1.0,
            "margin": (u > 0.30) & (u < 0.52) & (lat > 0.15 * w_max),
            "tip_15pct": u > 0.85,
        }

        leaf_rec = {"plant": name, "leaf_id": lid, "raw_points": len(pts), "ref_length": ref_len, "ref_area": ref_area, "scenarios": {}}
        for sc_name, mask in scenarios.items():
            if np.sum(mask) < 20 or np.sum(~mask) < 200:
                continue
            obs_pts = pts[~mask]
            obs_nrm = nrm[~mask] if nrm is not None else None
            held_pts = pts[mask]
            res = fit_leaf(obs_pts, stem, normals=obs_nrm, prior=prior, cfg=cfg, leaf_id=lid)
            dist = cKDTree(res["vertices"]).query(held_pts)[0]
            len_err_pct = 100.0 * abs(res["traits"]["completed_length"] - ref_len) / ref_len
            area_err_pct = 100.0 * abs(res["traits"]["surface_area"] - ref_area) / ref_area
            leaf_rec["scenarios"][sc_name] = {
                "withheld_points": int(mask.sum()),
                "dist_median": float(np.median(dist)),
                "dist_p95": float(np.quantile(dist, 0.95)),
                "dist_median_over_wmax": float(np.median(dist) / w_max),
                "completed_length": float(res["traits"]["completed_length"]),
                "length_rel_error_pct": float(len_err_pct),
                "area_rel_error_pct": float(area_err_pct),
            }
        records.append(leaf_rec)

    summary = {}
    for sc in ("hole", "margin", "tip_15pct"):
        vals = [r["scenarios"][sc] for r in records if sc in r["scenarios"]]
        if vals:
            summary[sc] = {
                "n_leaves": len(vals),
                "median_3d_error": float(np.median([v["dist_median"] for v in vals])),
                "median_error_over_wmax": float(np.median([v["dist_median_over_wmax"] for v in vals])),
                "median_length_rel_error_pct": float(np.median([v["length_rel_error_pct"] for v in vals])),
                "median_area_rel_error_pct": float(np.median([v["area_rel_error_pct"] for v in vals])),
            }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"prior_calibrated_leaves": n_cal, "evaluated_leaves": len(records), "summary": summary, "records": records}
    write_json(out / "holdout_benchmark.json", report)
    print(f"Saved {out / 'holdout_benchmark.json'}")
    for sc, s in summary.items():
        print(f"  {sc:10s} (n={s['n_leaves']}): med_3d={s['median_3d_error']:.4f} ({s['median_error_over_wmax']:.1%} Wmax), "
              f"len_err={s['median_length_rel_error_pct']:.2f}%, area_err={s['median_area_rel_error_pct']:.2f}%")


def main():
    p = argparse.ArgumentParser(description="Real-maize holdout occlusion benchmark")
    p.add_argument("--input-root", type=Path, default=ROOT / "dataset")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/maize_v2/holdout_validation")
    p.add_argument("--max-leaves", type=int, default=8)
    args = p.parse_args()
    evaluate_holdout(args.input_root, args.output, args.max_leaves)


if __name__ == "__main__":
    main()
