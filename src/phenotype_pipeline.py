"""Run validated point clouds through the V1.1 feature and clustering pipeline."""

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from threadpoolctl import threadpool_limits

from .nearest_attachment import InputError, load_pair
from .phenotype_clusters import cluster_leaves
from .phenotype_features import ENTROPY_EDGES, PARAMS, RULES, VARIANTS, analyse_plant
from .phenotype_plots import write_figures
from .phenotype_report import write_outputs
from .plant_clustering import cluster_plant, plant_cluster_summary, write_plant_clusters

ROOT = Path(__file__).resolve().parents[1]
TIMEZONE = timezone(timedelta(hours=8))
SOURCES = [
    "VERSION", "requirements.txt", "run_analysis.py", "src/geometry.py", "src/robust_geometry.py",
    "src/leaf.py", "src/stem.py", "src/io_utils.py", "src/nearest_attachment.py", "src/tables.py",
    "src/plant_clustering.py", "src/visualization.py",
    *[f"src/phenotype_{name}.py" for name in ("features", "clusters", "plots", "report", "pipeline")],
]


def _provenance():
    try:
        git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        git_head = "unavailable"
    return {
        "run_at": datetime.now(TIMEZONE).isoformat(timespec="seconds"),
        "version": (ROOT / "VERSION").read_text(encoding="utf-8").strip(),
        "git_head": git_head, "python": platform.python_version(),
        "packages": {name: version(name) for name in
                     ("numpy", "scipy", "matplotlib", "scikit-learn", "plotly", "threadpoolctl")},
        "point_row_origin": 0, "coordinate_unit": "original input coordinate units; physical unit unconfirmed",
        "sha256_text_source_lf": {
            name: sha256((ROOT / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in SOURCES
        },
        "stem_and_leaf_parameters": asdict(PARAMS),
        "geodesic_variants": VARIANTS, "quality_rules": RULES,
        "inclination_entropy_bin_edges_deg": ENTROPY_EDGES,
    }


def _read_inputs(plant, csv_files, txt_files, record):
    blobs = {}
    for kind, sources in (("csv", csv_files), ("txt", txt_files)):
        path = sources.get(plant)
        if path is not None:
            record[f"{kind}_file"] = path.name
            blobs[kind] = path.read_bytes()
            record[f"{kind}_sha256"] = sha256(blobs[kind]).hexdigest()
    if len(blobs) != 2:
        missing = "CSV" if "csv" not in blobs else "TXT"
        raise InputError(f"missing matching {missing}")
    return load_pair(blobs["csv"], blobs["txt"])


def run_analysis(csv_dir: Path, txt_dir: Path, output: Path, sample: str | None = None) -> int:
    csv_files = {p.stem: p for p in csv_dir.glob("*.csv") if p.is_file()}
    txt_files = {p.stem: p for p in txt_dir.glob("*.txt") if p.is_file()}
    plants = [sample] if sample is not None else sorted(csv_files.keys() | txt_files.keys())
    if not plants:
        raise ValueError("no paired input file names found")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    provenance = _provenance()
    leaves, manifest, quality, geometry = [], [], {}, {}
    plant_cluster_rows, plant_cluster_summaries, plant_cluster_centers = [], [], []
    with threadpool_limits(limits=1):
        for plant in plants:
            record = {
                "plant_id": plant, "csv_file": None, "csv_sha256": None, "txt_file": None,
                "txt_sha256": None, "source_points": None, "nonfinite_xyz_rows": None,
                "input_status": "ok", "analysis_status": "ok", "detail": "",
            }
            manifest.append(record)
            try:
                xyz, labels = _read_inputs(plant, csv_files, txt_files, record)
            except (InputError, OSError) as exc:
                record.update(input_status="failed", analysis_status="INPUT_FAILED",
                              detail=str(exc) if isinstance(exc, InputError) else "input read failed")
                plant_cluster_summaries.append(plant_cluster_summary(plant, None, "INPUT_FAILED"))
                print(f"{plant}: {record['detail']}", flush=True)
                continue
            record.update(source_points=len(xyz),
                          nonfinite_xyz_rows=int((~np.isfinite(xyz).all(axis=1)).sum()))
            rows, stem, audits, geometry[plant] = analyse_plant(plant, xyz, labels)
            leaves.extend(rows)
            classified, summary, centroids = cluster_plant(plant, xyz, labels, rows)
            plant_cluster_rows.extend(classified)
            plant_cluster_summaries.append(summary)
            plant_cluster_centers.extend(centroids)
            quality[plant] = {"stem": stem, "leaves": audits}
            located = sum(r["status"] == "ok" for r in rows)
            if not rows:
                record["analysis_status"] = "NO_LEAVES"
            elif not located:
                record["analysis_status"] = "NO_NEAREST_PAIRS"
            print(f"{plant}: {len(rows)} leaves, {located} nearest pairs, "
                  f"stem fit {stem['status']}", flush=True)

    model, centers, folds = cluster_leaves(leaves, plants)
    with TemporaryDirectory(prefix=".phenotype-", dir=output.parent) as work:
        staged = Path(work) / "result"
        staged.mkdir()
        write_outputs(staged, leaves, manifest, quality, provenance, model, centers, folds)
        write_plant_clusters(staged, plant_cluster_rows, plant_cluster_summaries,
                             plant_cluster_centers, manifest, provenance)
        write_figures(staged, leaves, model, geometry, plant_cluster_rows)
        staged.rename(output)
    ok_count = sum(r["input_status"] == "ok" for r in manifest)
    classified = sum(row["assigned_leaves"] for row in plant_cluster_summaries)
    print(f"{ok_count}/{len(plants)} plants validated, {len(leaves)} leaves, "
          f"{classified} clustered within plants, {model['assigned_leaves']} clustered across plants. "
          f"Outputs: {output}", flush=True)
    return int(any(r["status"] != "ok" for r in plant_cluster_summaries))
