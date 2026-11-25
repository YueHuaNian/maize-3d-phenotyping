"""All-leaf, leakage-controlled real-scan occlusion benchmark with resumable artifacts."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

# Limit numerical-library threads before importing numpy, also in spawned workers.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[variable] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".vendor")]
import numpy as np
import scipy
from scipy.spatial import cKDTree
from maize_leaf.geometry import clean_leaf_points, fit_stem_centerline, project_points_to_curve
from maize_leaf.io import discover, load_pair, write_json
from maize_leaf.morphology import MaizeShapePrior
from maize_leaf.surface import MaizeLeafModel, ModelConfig, fit_leaf
from maize_leaf.validation import SCENARIOS, occlusion_masks, point_mesh_distances, summarize_distances, holdout_alerts


def source_hashes():
    files = sorted((ROOT / "src/maize_leaf").glob("*.py")) + [Path(__file__)]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def fit_one_leaf(job):
    name, lid, points, normals, stem, folder, cfg, frozen_folder = job
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cached = folder / "records.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    records = []
    try:
        # Fixed engineering prior: no full-data or held-out-leaf calibration.
        prior = MaizeShapePrior()
        if frozen_folder is not None:
            frozen_folder = Path(frozen_folder)
            # A baseline may still be finishing other leaves in a separate run.
            # Never read a partially written split or silently replace it.
            deadline = time.monotonic() + 1200
            while not (frozen_folder / "records.json").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Frozen baseline leaf did not finish within 20 minutes")
                time.sleep(1)
            with np.load(frozen_folder / "hole_split.npz") as data:
                ref = {"vertices": data["reference_vertices"], "faces": data["reference_faces"]}
            ref["model"] = MaizeLeafModel.load(frozen_folder / "reference_model.npz")
            ref["traits"] = json.loads((frozen_folder / "reference_traits.json").read_text(encoding="utf-8"))
        else:
            ref = fit_leaf(points, stem, normals=normals, prior=prior,
                           cfg=replace(cfg, allow_tip_completion=False), leaf_id=lid)
        ref["model"].save(folder / "reference_model.npz")
        write_json(folder / "reference_traits.json", ref["traits"])
        cleaned, _ = clean_leaf_points(points, normals)
        reliable = cKDTree(cleaned).query(points)[0] < 1e-12
        spacing = float(np.median(cKDTree(cleaned).query(cleaned, k=2)[0][:, 1]))
        model = ref["model"]
        s = project_points_to_curve(points, model.midrib, model.length_obs)
        _, b, _ = model.frame.evaluate(s)
        lateral = np.sum((points - model.midrib(s)) * b, axis=1)
        w = np.maximum(model.width(s), .01 * ref["traits"]["max_width"])
        u, v = s / model.length_obs, 2 * lateral / w
        masks = occlusion_masks(u, v)
        tip_width_ratio = float(np.median(model.width(np.linspace(.93, .98, 8) * model.length_obs))
                                / max(ref["traits"]["max_width"], 1e-12))
        for scenario, mask in masks.items():
            start = time.perf_counter()
            hidden = mask & reliable
            if frozen_folder is not None:
                with np.load(frozen_folder / f"{scenario}_split.npz") as data:
                    if not np.array_equal(data["points"], points):
                        raise ValueError("Frozen split point order differs")
                    mask, hidden = ~data["observed"], data["held"]
            record = {"plant": name, "leaf_id": lid, "scenario": scenario,
                      "raw_points": len(points), "withheld_points": int(hidden.sum()),
                      "removed_raw_points": int(mask.sum()), "observed_points": int((~mask).sum()),
                      "ref_length": ref["traits"]["observed_length"],
                      "ref_area": ref["traits"]["surface_area"],
                      "ref_max_width": ref["traits"]["max_width"], "spacing": spacing,
                      "tip_width_ratio": tip_width_ratio,
                      "tip_narrow_reference": bool(tip_width_ratio < .25),
                      "artifact": f"cases/{name}/leaf_{lid:02d}/{scenario}"}
            np.savez_compressed(folder / f"{scenario}_split.npz", points=points,
                                observed=~mask, held=hidden, reference_vertices=ref["vertices"],
                                reference_faces=ref["faces"], u=u, v=v)
            if hidden.sum() < 20 or (~mask).sum() < 200:
                record.update(status="skipped", error="fewer than 20 reliable test points or 200 observed points")
            else:
                try:
                    # Only observed raw points, their supplied normals and stem enter fitting.
                    res = fit_leaf(points[~mask], stem, normals=normals[~mask] if normals is not None else None,
                                   prior=prior, cfg=cfg, leaf_id=lid)
                    res["model"].save(folder / f"{scenario}_model.npz")
                    write_json(folder / f"{scenario}_traits.json", res["traits"])
                    np.savez_compressed(folder / f"{scenario}_mesh.npz", vertices=res["vertices"], faces=res["faces"])
                    distances = point_mesh_distances(points[hidden], res["vertices"], res["faces"])
                    reference_distances = point_mesh_distances(points[hidden], ref["vertices"], ref["faces"])
                    metrics = summarize_distances(distances)
                    record.update({f"held_{key}": value for key, value in metrics.items()})
                    record.update(ref_held_p95=float(np.quantile(reference_distances, .95)),
                                  held_p95_over_width=metrics["p95"] / record["ref_max_width"],
                                  held_rmse_over_width=metrics["rmse"] / record["ref_max_width"],
                                  held_p95_over_spacing=metrics["p95"] / spacing,
                                  length=res["traits"]["completed_length"], area=res["traits"]["surface_area"],
                                  geometry_method=res["traits"]["geometry_method"],
                                  observed_quality_passed=res["quality"]["passed"],
                                  extrapolated_fraction=res["traits"]["area_fractions"]["extrapolated"])
                    record["tip_completion_status"] = res["traits"].get("tip_completion_status", "legacy")
                    record["length_change_pct"] = 100 * abs(record["length"] - record["ref_length"]) / record["ref_length"]
                    record["area_change_pct"] = 100 * abs(record["area"] - record["ref_area"]) / record["ref_area"]
                    record["length_signed_change_pct"] = 100 * (record["length"] / record["ref_length"] - 1)
                    record["area_signed_change_pct"] = 100 * (record["area"] / record["ref_area"] - 1)
                    record["alerts"] = holdout_alerts(record)
                    record["status"] = "alert" if record["alerts"] else "ok"
                except Exception as exc:
                    record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            record["elapsed_s"] = round(time.perf_counter() - start, 3)
            write_json(folder / f"{scenario}_record.json", record)
            records.append(record)
    except Exception as exc:
        records = [{"plant": name, "leaf_id": lid, "scenario": sc, "status": "reference_failed",
                    "raw_points": len(points), "error": f"{type(exc).__name__}: {exc}"} for sc in SCENARIOS]
    write_json(cached, records)
    return records


def summarize(records):
    summary = {}
    for scenario in SCENARIOS:
        subset = [r for r in records if r["scenario"] == scenario]
        fitted = [r for r in subset if r["status"] in ("ok", "alert")]
        rec = {"total": len(subset), "fitted": len(fitted),
               "ok": sum(r["status"] == "ok" for r in subset),
               "alerts": sum(r["status"] == "alert" for r in subset),
               "failed": sum(r["status"] in ("failed", "reference_failed") for r in subset),
               "skipped": sum(r["status"] == "skipped" for r in subset)}
        for metric in ("held_median", "held_p95", "held_rmse", "held_p95_over_width",
                       "held_p95_over_spacing", "length_change_pct", "area_change_pct"):
            rec[f"leaf_median_{metric}"] = float(np.median([r[metric] for r in fitted])) if fitted else None
            plant_medians = [np.median([r[metric] for r in fitted if r["plant"] == pl])
                             for pl in sorted({r["plant"] for r in fitted})]
            rec[f"plant_mean_of_leaf_medians_{metric}"] = float(np.mean(plant_medians)) if fitted else None
        summary[scenario] = rec
    return summary


def evaluate_holdout(input_root, output_dir, max_leaves=None, workers=6, plants=None, split_root=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    hashes = source_hashes()
    pairs = [p for p in discover(input_root) if not plants or p[0] in plants]
    inputs = [{"plant": name, "files": [{"name": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                                         for p in (txt, csv_path)]} for name, txt, csv_path in pairs]
    cfg = ModelConfig()
    manifest = {"source_sha256": hashes, "inputs": inputs, "config": asdict(cfg),
                "prior": asdict(MaizeShapePrior()), "scenarios": list(SCENARIOS), "max_leaves": max_leaves,
                "minimum_points": {"held": 20, "observed": 200},
                "alerts": {"held_p95": "max(6*spacing, .15*reference_max_width)",
                           "length_change_pct": 15, "area_change_pct": 25}}
    if split_root is not None:
        manifest["frozen_split_fingerprint"] = json.loads((Path(split_root) / "manifest.json").read_text(encoding="utf-8"))["fingerprint"]
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old["fingerprint"] != fingerprint:
            raise ValueError("Input/source/config changed; choose a new output directory to preserve this run")
    else:
        write_json(manifest_path, dict(manifest, fingerprint=fingerprint))
    jobs = []
    for name, txt, csv_path in pairs:
        plant = load_pair(name, txt, csv_path)
        stem, _ = fit_stem_centerline(plant.xyz[plant.labels == 0])
        for lid in sorted(int(x) for x in set(plant.labels) if x > 0):
            mask = plant.labels == lid
            jobs.append((name, lid, plant.xyz[mask], plant.normals[mask] if plant.normals is not None else None,
                         stem, out / f"cases/{name}/leaf_{lid:02d}", cfg,
                         Path(split_root) / f"cases/{name}/leaf_{lid:02d}" if split_root is not None else None))
    if max_leaves is not None:
        jobs = jobs[:max_leaves]
    records = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fit_one_leaf, job): (job[0], job[1]) for job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            records.extend(result)
            name, lid = futures[future]
            print(f"[{index}/{len(jobs)}] {name} leaf {lid}: "
                  + ", ".join(f"{r['scenario']}={r['status']}" for r in result), flush=True)
    records.sort(key=lambda r: (r["plant"], r["leaf_id"], SCENARIOS.index(r["scenario"])))
    fields = list(dict.fromkeys(k for r in records for k in r))
    with (out / "per_leaf_errors.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: (";".join(v) if isinstance(v, list) else v) for k, v in r.items()} for r in records)
    summary = summarize(records)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(),
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
              "fingerprint": fingerprint, "plants": len({job[0] for job in jobs}), "leaves": len(jobs),
              "prior_calibrated_leaves": 0, "elapsed_s": round(time.perf_counter() - started, 2),
              "distance": "exact nearest triangle on 90x25 exported mesh; input-coordinate units",
              "reference": "no-tip-completion full-scan fit; length/area changes are model consistency, not manual truth",
              "split": "full-scan coordinates define masks; hidden points and full-reference model never enter masked fits",
              "summary": summary, "records": records}
    write_json(out / "holdout_benchmark.json", report)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"DONE {len(jobs)} leaves / {len(records)} cases in {report['elapsed_s']} s", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/holdout_20261125/baseline")
    parser.add_argument("--max-leaves", type=int, default=None, help="Optional deterministic subset for debugging")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--plant", action="append", help="Repeat to select plants for diagnosis")
    parser.add_argument("--split-root", type=Path, help="Reuse this run's frozen masks and full-scan reference")
    args = parser.parse_args()
    evaluate_holdout(args.input_root, args.output, args.max_leaves, args.workers, args.plant, args.split_root)


if __name__ == "__main__":
    main()
