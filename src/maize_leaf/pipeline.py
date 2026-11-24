"""Batch modeling pipeline with empirical prior calibration and export."""
import csv
from pathlib import Path
import numpy as np
from .geometry import build_normal_aligned_frame, clean_leaf_points, extract_midrib, fit_stem_centerline
from .io import discover, load_pair, write_json, write_obj, write_ply
from .morphology import MaizeShapePrior, calibrate_prior_from_leaves, extract_width_profile, fit_dornbusch_blade
from .surface import COLORS, REGIONS, ModelConfig, fit_leaf
from .viz import ensure_plotly_js, export_index_html, export_leaf_html, export_plant_html, save_static_plot


def calibrate_dataset_prior(pairs):
    """Calibrate Dornbusch (2011) shape prior from intact-tip leaves in the dataset."""
    intact_blades = []
    base_prior = MaizeShapePrior()
    for name, txt, csv_path in pairs:
        plant = load_pair(name, txt, csv_path)
        stem, _ = fit_stem_centerline(plant.xyz[plant.labels == 0])
        for lid in sorted(int(x) for x in set(plant.labels) if x > 0):
            pts = plant.xyz[plant.labels == lid]
            if len(pts) < 800:
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
            if tip_ratio < 0.22:
                blade = fit_dornbusch_blade(prof["s"], prof["w_obs"], l_obs, prior=base_prior, allow_tip_completion=False)
                intact_blades.append(blade)
    return calibrate_prior_from_leaves(intact_blades), len(intact_blades)


def export_leaf(folder, plant, leaf_id, result):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    key = f"leaf_{leaf_id:02d}"
    m, v, f, reg = result["model"], result["vertices"], result["faces"], result["regions"]
    result["traits"]["plant"] = plant.name

    write_ply(folder / f"{key}_points.ply", result["points"], colors=plant.rgb[plant.labels == leaf_id])
    write_ply(folder / f"{key}_surface.ply", v, f, COLORS[reg], reg, result["face_regions"], np.full(len(v), leaf_id))
    write_obj(folder / f"{key}_surface.obj", v, f, result["face_regions"], REGIONS, COLORS)
    m.save(folder / f"{key}_model.npz")
    write_json(folder / f"{key}_midrib.json", {"s": result["s"].tolist(), "xyz": result["midrib"].tolist()})
    write_json(folder / f"{key}_blade.json", result["blade"].to_dict())
    write_json(folder / f"{key}_traits.json", result["traits"])

    png = folder / f"{key}_visualization.png"
    save_static_plot(png, result["points"], [result], title=f"Plant {plant.name} / Leaf {leaf_id}")
    export_leaf_html(folder / f"{key}_visualization.html", plant.name, leaf_id, result, png.name)


def export_plant(folder, plant, stem, results):
    folder = Path(folder)
    verts = np.vstack([r["vertices"] for r in results.values()])
    offsets = np.cumsum([0] + [len(r["vertices"]) for r in results.values()])[:-1]
    faces = np.vstack([r["faces"] + off for r, off in zip(results.values(), offsets)])
    regions = np.concatenate([r["regions"] for r in results.values()])
    freg = np.concatenate([r["face_regions"] for r in results.values()])
    ids = np.concatenate([np.full(len(r["vertices"]), lid) for lid, r in results.items()])

    write_ply(folder / "whole_plant_mesh.ply", verts, faces, COLORS[regions], regions, freg, ids)
    write_obj(folder / "whole_plant_mesh.obj", verts, faces, freg, REGIONS, COLORS)
    write_ply(folder / "stem_centerline.ply", stem)
    save_static_plot(folder / "whole_plant.png", plant.xyz, list(results.values()), stem=stem, title=f"Plant {plant.name}")
    export_plant_html(folder / "whole_plant.html", plant, stem, results)


def run(input_root, output, plant_name=None, leaf_id=None, cfg=None, calibrate_prior=True):
    cfg = cfg or ModelConfig()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    ensure_plotly_js(out)

    all_pairs = discover(input_root)
    if calibrate_prior and len(all_pairs) > 1:
        prior, n_cal = calibrate_dataset_prior(all_pairs)
    else:
        prior, n_cal = MaizeShapePrior(), 0
    write_json(out / "species_prior.json", {"calibrated_from_intact_leaves": n_cal, **prior.__dict__})

    pairs = [p for p in all_pairs if not plant_name or p[0] == plant_name]
    plants, all_traits = [], []

    for name, txt, csv_path in pairs:
        plant = load_pair(name, txt, csv_path)
        leaf_ids = [leaf_id] if leaf_id is not None else sorted(int(x) for x in set(plant.labels) if x > 0)
        folder = out / f"plant_{name}"
        folder.mkdir(parents=True, exist_ok=True)

        stem, stem_json = fit_stem_centerline(plant.xyz[plant.labels == 0])
        write_json(folder / "stem_centerline.json", stem_json)

        results = {}
        for lid in leaf_ids:
            print(f"Modeling {name} leaf {lid}", flush=True)
            pts = plant.xyz[plant.labels == lid]
            nrm = plant.normals[plant.labels == lid] if plant.normals is not None else None
            res = fit_leaf(pts, stem, normals=nrm, prior=prior, cfg=cfg, leaf_id=lid)
            export_leaf(folder / "leaves", plant, lid, res)
            results[lid] = res
            all_traits.append(res["traits"])

        export_plant(folder, plant, stem, results)
        plants.append({"plant": name, "leaf_count": len(results)})

    fields = [
        "plant", "leaf_id", "raw_points", "observed_length", "completed_length",
        "tip_completion_length", "max_width", "xm_rel_pos", "k_lig", "c1", "c2",
        "form_factor_alpha", "surface_area", "observed", "interpolated", "extrapolated",
        "completion_ratio", "base_inclination_deg", "fit_rmse", "tps_gcv_lambda", "tps_edf",
    ]
    with (out / "traits.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for t in all_traits:
            row = {k: t.get(k) for k in fields}
            row.update(t["area_fractions"])
            w.writerow(row)

    export_index_html(out / "index.html", plants, all_traits)
    print(f"DONE {out.resolve()} | plants={len(plants)}, leaves={len(all_traits)}, prior_cal={n_cal}", flush=True)
    return out
