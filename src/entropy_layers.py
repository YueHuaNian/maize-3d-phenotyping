"""Reproducible height-ordered, entropy-informed leaf clustering report."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .plant_clustering import FIELDS
from .plant_morphotypes_v2 import V2_FIELDS, WEIGHTS, classify_results
from .tables import save_csv

TARGETS = {"C1": "上层直立", "C2": "中层展开", "C3": "下层下垂"}


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def median(values):
    valid = [float(v) for v in values if v not in (None, "") and np.isfinite(float(v))]
    return float(np.median(valid)) if valid else None


def text_sha256(path):
    """Hash text with portable LF line endings, matching repository conventions."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    raw = read_rows(source / "leaf_features.csv")
    leaves = {}
    for row in raw:
        plant, leaf = row["plant_id"], int(row["leaf_id"])
        if leaf in leaves.setdefault(plant, {}):
            raise ValueError(f"Duplicate leaf: {plant}, {leaf}")
        leaves[plant][leaf] = row
    classified, plants = classify_results(source, leaves, middle_size="free")
    for row in classified:
        original = leaves[row["plant_id"]][row["leaf_id"]]
        row["target_description"] = TARGETS.get(row["plant_cluster_id"])
        # Position is known; the target morphology is not a measured label.
        row.pop("morphotype", None)
        for name in ("leaf_angle_deg", "leaf_angle_deg_status", "curvature_rms_norm",
                     "curvature_rms_norm_status", "droop_index", "spread_index"):
            row[name] = original.get(name, "")

    summaries = []
    for plant in plants:
        for code in ("C1", "C2", "C3"):
            group = [r for r in classified if r["plant_id"] == plant["plant_id"]
                     and r["plant_cluster_id"] == code]
            if not group:
                continue
            summary = {"plant_id": plant["plant_id"], "cluster_id": code,
                       "height_layer": group[0]["height_layer"], "target_description": TARGETS[code],
                       "leaf_count": len(group),
                       "valid_angle_count": sum(r["leaf_angle_deg_status"] == "ok"
                                                and r["leaf_angle_deg"] != "" for r in group)}
            for field in ("attachment_height_ratio", *FIELDS, "leaf_angle_deg", "curvature_rms_norm"):
                summary[field + "_median"] = median([r.get(field) for r in group])
            summaries.append(summary)

    inputs = ("leaf_features.csv", "plant_morphotypes/leaf_clusters.csv",
              "plant_morphotypes/plant_summary.csv")
    metadata = {"method": "per-plant exact contiguous three-cluster weighted SSE minimization",
                "middle_size": "free", "features": dict(zip(V2_FIELDS, WEIGHTS.tolist())),
                "text_sha256_lf": {name: text_sha256(source / name) for name in inputs},
                "code_sha256_lf": {name: text_sha256(Path(__file__).parent / name)
                                   for name in ("entropy_layers.py", "plant_morphotypes_v2.py")},
                "warning": "Target descriptions are hypotheses, not verified morphology labels."}
    ok = sum(p["status"] == "ok" for p in plants)
    assigned = sum(p["assigned_leaves"] for p in plants)
    report = ["# 基于信息熵的高度有序聚类", "",
              f"{ok}/{len(plants)} 株完成三层，{assigned}/{len(classified)} 片叶获得分层。", "",
              "每株叶片按最近茎点高度排序，并在高度相同处不切分。对高度、观测方向 Shannon 熵、"
              "水平伸展、末端落差、下行段占比分别标准化，按平方距离权重 65%、20%、5%、5%、5%，"
              "枚举两个切点，使三层加权簇内平方和最小；三层均非空，不强制中层多数。", "",
              "上层直立、中层展开、下层下垂是各层的目标描述，不表示已验证叶片达到对应形态。"
              "信息熵表示方向分散，不能单独确定弯曲方向；叶茎角和曲率沿用原质量筛查，缺失不填补。", "",
              "| 植株 | 叶片数 | 上层 | 中层 | 下层 | 状态 |",
              "|---|---:|---:|---:|---:|---|"]
    for p in plants:
        report.append(f"| {p['plant_id']} | {p['leaf_count']} | {p['C1']} | {p['C2']} | {p['C3']} | {p['status']} |")
    report += ["", "逐叶特征见 leaf_layers.csv；逐层特征中位数及角度有效数量见 layer_summary.csv。",
               "每株叶片少且固定三组，聚类与叶基均尚需人工和独立样本验证。", ""]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".entropy-layers-", dir=destination.parent) as tmp:
        stage = Path(tmp) / "result"
        stage.mkdir()
        save_csv(stage / "leaf_layers.csv", classified)
        save_csv(stage / "plant_summary.csv", plants)
        save_csv(stage / "layer_summary.csv", summaries)
        (stage / "provenance.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (stage / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
        stage.rename(destination)
    return classified, plants, summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/v1.1"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    leaves, plants, _ = run(args.results_dir, args.output_dir)
    print(f"Assigned {sum(p['assigned_leaves'] for p in plants)}/{len(leaves)} leaves")


if __name__ == "__main__":
    main()
