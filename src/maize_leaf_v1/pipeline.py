"""Minimal batch pipeline: discover -> load -> fit -> export."""
import csv
from pathlib import Path
from .data import discover, load_pair
from .model import Config, stem_model, fit_leaf
from .export import json_write, ensure_js, export_leaf, export_plant, export_index


def run(input_root, output, plant_name=None, leaf_id=None, cfg=None):
    cfg = cfg or Config()
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    ensure_js(out)

    pairs = discover(input_root)
    if plant_name:
        pairs = [p for p in pairs if p[0] == plant_name]

    plants, metrics = [], []
    for name, txt, csv_path in pairs:
        plant = load_pair(name, txt, csv_path)
        leaf_ids = [leaf_id] if leaf_id is not None else sorted(int(x) for x in set(plant.labels) if x > 0)
        folder = out / f"plant_{name}"
        folder.mkdir(parents=True, exist_ok=True)

        stem, stem_json = stem_model(plant.xyz[plant.labels == 0])
        json_write(folder / "stem_centerline.json", stem_json)

        results = {}
        for lid in leaf_ids:
            print(f"Fitting {name} leaf {lid}", flush=True)
            result = fit_leaf(plant.xyz[plant.labels == lid], stem, cfg, lid)
            export_leaf(folder / "leaves", plant, lid, result)
            results[lid] = result
            metrics.append(result["metrics"])

        export_plant(folder, plant, stem, results)
        plants.append(dict(plant=name, leaf_count=len(results)))

    fields = [
        "plant",
        "leaf_id",
        "raw_points",
        "completion_ratio",
        "observed",
        "interpolated",
        "extrapolated",
        "fit_rmse",
        "surface_area",
        "centerline_length",
        "max_width",
    ]
    with (out / "metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in metrics:
            row = {k: m.get(k) for k in fields}
            row.update(m["area_fractions"])
            w.writerow(row)

    export_index(out, plants, metrics)
    print(f"RESULT {out.resolve()} | leaves={len(metrics)}", flush=True)
    return out
