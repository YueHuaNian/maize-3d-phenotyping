"""Height-constrained V2 clustering using verified V1.1 leaf descriptors."""

from collections import Counter
import csv
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.preprocessing import StandardScaler

from .io_utils import dump_json
from .plant_clustering import FIELDS, MORPHOTYPES
from .tables import save_csv

HEIGHT = "attachment_height_ratio"
V2_FIELDS = (HEIGHT, *FIELDS)
WEIGHTS = np.array([0.65, 0.20, 0.05, 0.05, 0.05])
BOTTOM_TO_TOP = ("C3", "C2", "C1")
LAYERS = {"C3": "下层", "C2": "中层", "C1": "上层"}
SCOPE = "Within-plant V2 · height-ordered K=3"
INPUTS = (
    "input_manifest.csv",
    "leaf_features.csv",
    "plant_morphotypes/leaf_clusters.csv",
    "plant_morphotypes/plant_summary.csv",
    "plant_morphotypes/provenance.json",
)


def _height_partition(matrix):
    """Find the minimum weighted SSE over two cuts at distinct adjacent heights."""
    cuts = np.flatnonzero(np.diff(matrix[:, 0]) > 0) + 1
    if len(cuts) < 2:
        raise ValueError("fewer than three distinct attachment heights")
    weighted = StandardScaler().fit_transform(matrix) * np.sqrt(WEIGHTS)
    sums = np.vstack((np.zeros(matrix.shape[1]), np.cumsum(weighted, axis=0)))
    norms = np.r_[0.0, np.cumsum(np.sum(weighted ** 2, axis=1))]

    def loss(a, b):
        total = sums[b] - sums[a]
        return max(0.0, float(norms[b] - norms[a] - (total @ total) / (b - a)))

    n = len(matrix)
    return min((loss(0, a) + loss(a, b) + loss(b, n), int(a), int(b))
               for index, a in enumerate(cuts) for b in cuts[index + 1:])


def _number(value):
    if value is None or value == "":
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def cluster_plant(plant, leaves, descriptors, lower, upper):
    """Keep every input leaf in the table, even when three valid bands are impossible."""
    valid_stem = lower is not None and upper is not None and upper > lower
    stem_span = upper - lower if valid_stem else None
    rows = []
    for leaf, row in sorted(leaves.items()):
        z = _number(row["stem_z"]) if row["status"] == "ok" else None
        height = (z - lower) / stem_span if valid_stem and z is not None else None
        rows.append({"plant_id": plant, "leaf_id": leaf, "plant_cluster_id": None,
                     "morphotype": None, "height_layer": None, "status": None,
                     "nearest_stem_z": z, HEIGHT: height,
                     **{field: _number(descriptors[leaf][field]) for field in FIELDS}})
    summary = {"plant_id": plant, "leaf_count": len(rows), "assigned_leaves": 0, "status": "ok",
               "C1": 0, "C2": 0, "C3": 0, "lower_cut_stem_z": None, "upper_cut_stem_z": None,
               "weighted_sse": None}
    if len(rows) < 3:
        summary["status"] = "TOO_FEW_LEAVES"
    elif not valid_stem:
        summary["status"] = "MISSING_STEM_HEIGHT_RANGE"
    elif any(row[field] is None for row in rows for field in V2_FIELDS):
        summary["status"] = "MISSING_HEIGHT_OR_MORPHOLOGY"
    elif len({row[HEIGHT] for row in rows}) < 3:
        summary["status"] = "TOO_FEW_DISTINCT_ATTACHMENT_HEIGHTS"
    else:
        ordered = sorted(rows, key=lambda row: (row[HEIGHT], row["leaf_id"]))
        matrix = np.array([[row[field] for field in V2_FIELDS] for row in ordered])
        loss, lower_cut, upper_cut = _height_partition(matrix)
        for code, start, stop in zip(BOTTOM_TO_TOP, (0, lower_cut, upper_cut),
                                     (lower_cut, upper_cut, len(rows))):
            for row in ordered[start:stop]:
                row.update(plant_cluster_id=code, morphotype=MORPHOTYPES[code],
                           height_layer=LAYERS[code])
        summary.update(assigned_leaves=len(rows), weighted_sse=loss,
                       lower_cut_stem_z=(ordered[lower_cut - 1]["nearest_stem_z"]
                                         + ordered[lower_cut]["nearest_stem_z"]) / 2,
                       upper_cut_stem_z=(ordered[upper_cut - 1]["nearest_stem_z"]
                                         + ordered[upper_cut]["nearest_stem_z"]) / 2)
        summary.update(Counter(row["plant_cluster_id"] for row in rows))
    for row in rows:
        row["status"] = summary["status"]
    return rows, summary


def _read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def classify_results(source, leaves):
    descriptors = {}
    for row in _read_csv(source / "plant_morphotypes/leaf_clusters.csv"):
        key = row["plant_id"], int(row["leaf_id"])
        if key in descriptors:
            raise ValueError(f"Duplicate morphology descriptor: {key}")
        descriptors[key] = row
    if descriptors.keys() != {(p, leaf) for p, rows in leaves.items() for leaf in rows}:
        raise ValueError("V1.1 morphology descriptor rows do not match the verified leaf table")
    stem_ranges = {row["plant_id"]: (_number(row["stem_z_q01"]), _number(row["stem_z_q99"]))
                   for row in _read_csv(source / "plant_morphotypes/plant_summary.csv")}

    output, summaries = [], []
    for plant, records in sorted(leaves.items()):
        lower, upper = stem_ranges.get(plant, (None, None))
        observed = {leaf: descriptors[plant, leaf] for leaf in records}
        rows, summary = cluster_plant(plant, records, observed, lower, upper)
        output.extend(rows)
        summaries.append(summary)
    return output, summaries


def export_results(destination, source, leaves, plants):
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                           cwd=Path(__file__).resolve().parents[1]).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unavailable"
    provenance = {
        "git_head": revision, "method": "minimum weighted SSE with contiguous attachment height bands",
        "k": 3, "squared_distance_weights": dict(zip(V2_FIELDS, WEIGHTS)),
        "class_order_bottom_to_top": BOTTOM_TO_TOP, "morphotypes": MORPHOTYPES,
        "numpy": version("numpy"), "scikit-learn": version("scikit-learn"),
        "v1_1_input_sha256": {name: sha256((source / name).read_bytes()).hexdigest() for name in INPUTS},
    }
    ok = sum(row["status"] == "ok" for row in plants)
    assigned = sum(row["assigned_leaves"] for row in plants)
    report = [
        "# V2 高度优先的逐株三分类", "",
        f"{ok}/{len(plants)} 株得到三个连续高度组，{assigned}/{len(leaves)} 片叶获得标签。", "",
        "按最近茎点的 Z 高度排序，各株分别标准化。以加权簇内平方和最小为目标，"
        "在不同高度之间选择两个切点。叶基高度权重 65%，观测方向 Shannon 熵 20%，"
        "水平伸展、末端落差和下行段占比各 5%。", "",
        "上层为 C1 聚拢型，中层为 C2 伸展型，下层为 C3 耷拉型。"
        "高度分层依据叶基，叶尖仍可跨越分层边界。", "",
        "| 植株 | 叶片数 | 上层 C1 | 中层 C2 | 下层 C3 | 状态 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    report.extend(f"| {r['plant_id']} | {r['leaf_count']} | {r['C1']} | {r['C2']} | "
                  f"{r['C3']} | {r['status']} |" for r in plants)
    report.extend(["", "[逐叶高度与分组](leaf_layers.csv) · [逐株切点](plant_summary.csv) · "
                       "[复算参数与输入哈希](provenance.json)", ""])
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".height-v2-", dir=destination.parent) as temporary:
        staged = Path(temporary) / "result"
        staged.mkdir()
        save_csv(staged / "leaf_layers.csv", leaves)
        save_csv(staged / "plant_summary.csv", plants)
        dump_json(staged / "provenance.json", provenance)
        (staged / "HEIGHT_ORDERED_REPORT.md").write_text("\n".join(report), encoding="utf-8", newline="\n")
        staged.rename(destination)
