"""Compare saved surfaces against the same raw scans (fit audit, not holdout accuracy)."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".vendor")]

import numpy as np
from scipy.spatial import cKDTree
from maize_leaf.geometry import clean_leaf_points
from maize_leaf.io import discover, load_pair, write_json
from maize_leaf.surface import MaizeLeafModel, leaf_mesh_grid, surface_orientation_diagnostics
from maize_leaf.quality import audit_model


def model_distances(path, points, nu, nv):
    model = MaizeLeafModel.load(path)
    sv, _ = leaf_mesh_grid(model.blade.length, nu, nv)
    return cKDTree(model.evaluate(sv)).query(points)[0], model


def distance_summary(distances):
    return {"median": float(np.median(distances)), "p95": float(np.quantile(distances, 0.95)),
            "rmse": float(np.sqrt(np.mean(distances ** 2)))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nu", type=int, default=220)
    parser.add_argument("--nv", type=int, default=61)
    parser.add_argument("--plant", help="Restrict comparison to one plant")
    parser.add_argument("--leaf", type=int, help="Restrict comparison to one leaf ID")
    args = parser.parse_args()
    records = []
    inputs = []
    for name, txt, csv in discover(args.input_root):
        if args.plant and name != args.plant:
            continue
        plant = load_pair(name, txt, csv)
        inputs.append({"plant": name, "files": [
            {"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in (txt, csv)
        ]})
        for lid in sorted(int(x) for x in set(plant.labels) if x > 0):
            if args.leaf is not None and lid != args.leaf:
                continue
            relative = Path(f"plant_{name}/leaves/leaf_{lid:02d}_model.npz")
            baseline, candidate = args.baseline / relative, args.candidate / relative
            if not baseline.exists() or not candidate.exists():
                raise FileNotFoundError(f"Missing paired models: {relative}")
            points = plant.xyz[plant.labels == lid]
            cleaned, _ = clean_leaf_points(points)
            # Use exactly the same raw and retained points for both model versions.
            keep = cKDTree(cleaned).query(points)[0] < 1e-12
            before, before_model = model_distances(baseline, points, args.nu, args.nv)
            after, after_model = model_distances(candidate, points, args.nu, args.nv)
            quality_before = audit_model(before_model, cleaned)
            quality_after = audit_model(after_model, cleaned)
            spacing = float(np.median(cKDTree(cleaned).query(cleaned, k=2)[0][:, 1]))
            records.append({"plant": name, "leaf": lid, "raw_points": len(points),
                            "retained_points": len(cleaned), "spacing": spacing,
                            "baseline_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
                            "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                            "before": {"raw": distance_summary(before), "retained": distance_summary(before[keep])},
                            "after": {"raw": distance_summary(after), "retained": distance_summary(after[keep])},
                            "quality": {"before": quality_before, "after": quality_after},
                            "surface_support_mean": {"before": quality_before["support"]["mean"],
                                                     "after": quality_after["support"]["mean"]},
                            "max_width": {"before": float(np.max(before_model.width(np.linspace(0, before_model.blade.length, 500)))),
                                          "after": float(np.max(after_model.width(np.linspace(0, after_model.blade.length, 500))))},
                            "orientation": {"before": quality_before["orientation"], "after": quality_after["orientation"]},
                            "coverage_within_3_spacing": {"before": float(np.mean(before <= 3 * spacing)),
                                                          "after": float(np.mean(after <= 3 * spacing))}})
    if not records:
        raise ValueError("No leaves matched the requested comparison")
    summary = {"leaves": len(records), "orientation_reversed_leaves": {
        stage: sum(r["orientation"][stage]["reversed_fraction"] > 0 for r in records)
        for stage in ("before", "after")}}
    summary["quality_passed"] = {stage: sum(r["quality"][stage]["passed"] for r in records)
                                  for stage in ("before", "after")}
    summary["apex_closure_failed"] = {stage: sum(r["quality"][stage]["mesh"]["apex_spread"] > 1e-8 for r in records)
                                       for stage in ("before", "after")}
    summary["surface_support_mean_median"] = {
        stage: float(np.median([r["surface_support_mean"][stage] for r in records]))
        for stage in ("before", "after")}
    summary["surface_support_improved_leaves"] = sum(
        r["surface_support_mean"]["after"] < r["surface_support_mean"]["before"] for r in records)
    summary["width_more_than_doubled_leaves"] = sum(
        r["max_width"]["after"] > 2 * r["max_width"]["before"] for r in records)
    for sample in ("raw", "retained"):
        summary[sample] = {
            stage: {metric: float(np.median([r[stage][sample][metric] for r in records]))
                    for metric in ("median", "p95", "rmse")}
            for stage in ("before", "after")
        }
        summary[sample]["p95_improved_leaves"] = sum(r["after"][sample]["p95"] < r["before"][sample]["p95"] for r in records)
    source_paths = sorted((ROOT / "src/maize_leaf").glob("*.py")) + [Path(__file__)]
    source_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in source_paths}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_sha256": source_hashes,
              "inputs": inputs,
              "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
              "method": "Scan-to-nearest-sampled-surface-vertex distances and observed-surface-area-weighted nearest-scan distance; in-sample fit, not reconstruction ground truth.",
              "distance_units": "input coordinate units", "nu": args.nu, "nv": args.nv,
              "plant_filter": args.plant, "leaf_filter": args.leaf,
              "baseline": str(args.baseline), "candidate": str(args.candidate),
              "summary": summary, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for r in records:
        if (r["plant"], r["leaf"]) in (("1-1-2", 2), ("2-2-2", 4)):
            print(json.dumps(r, ensure_ascii=False))
    return 0 if summary["quality_passed"]["after"] == len(records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
