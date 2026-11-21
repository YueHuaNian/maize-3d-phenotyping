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
COMPARISONS = (
    # name, first layer, second layer, metric, desired relationship
    ("upper_angle_vs_middle", "C1", "C2", "leaf_angle_deg", "less"),
    ("middle_reach_vs_upper", "C2", "C1", "horizontal_reach_q95_ratio", "greater"),
    ("lower_downward_vs_middle", "C3", "C2", "downward_path_fraction", "greater"),
    ("lower_drop_vs_middle", "C3", "C2", "distal_drop_ratio", "greater"),
)


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def median(values):
    valid = [float(v) for v in values if v not in (None, "") and np.isfinite(float(v))]
    return float(np.median(valid)) if valid else None


def text_sha256(path):
    """Hash text with portable LF line endings, matching repository conventions."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def compare_layers(plant_id, layers):
    """Compare layer medians, keeping missing data, ties and coverage explicit."""
    result = []
    for name, first, second, feature, direction in COMPARISONS:
        a, b = layers.get(first, {}), layers.get(second, {})
        left, right = a.get(feature + "_median"), b.get(feature + "_median")
        ac, bc = a.get(feature + "_valid_count", 0), b.get(feature + "_valid_count", 0)
        if left is None or right is None:
            relation, supports = "insufficient_data", None
        elif left == right:
            relation, supports = "equal", None
        else:
            relation = "less" if left < right else "greater"
            supports = relation == direction
        coverage = ("insufficient" if relation == "insufficient_data" else
                    "complete" if ac == a.get("leaf_count") and bc == b.get("leaf_count") else "partial")
        result.append({"plant_id": plant_id, "comparison": name, "feature": feature,
                       "first_layer": first, "second_layer": second, "desired_relation": direction,
                       "first_median": left, "second_median": right,
                       "first_valid_count": ac, "first_leaf_count": a.get("leaf_count", 0),
                       "second_valid_count": bc, "second_leaf_count": b.get("leaf_count", 0),
                       "relation": relation, "supports_expected_order": supports, "coverage": coverage})
    return result


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
                       "leaf_count": len(group)}
            for field in ("attachment_height_ratio", *FIELDS, "leaf_angle_deg", "curvature_rms_norm"):
                values = [r.get(field) for r in group if field not in ("leaf_angle_deg", "curvature_rms_norm")
                          or r[field + "_status"] == "ok"]
                valid = [float(v) for v in values if v not in (None, "") and np.isfinite(float(v))]
                summary[field + "_median"] = median(valid)
                summary[field + "_valid_count"] = len(valid)
            summaries.append(summary)

    comparisons = []
    for plant in plants:
        by_layer = {r["cluster_id"]: r for r in summaries if r["plant_id"] == plant["plant_id"]}
        comparisons.extend(compare_layers(plant["plant_id"], by_layer))

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
               "每株叶片少且固定三组，聚类与叶基均尚需人工和独立样本验证。", "",
               "## 层间形态对照", "",
               "按各层有效叶片的中位数，比较上层夹角是否小于中层、中层伸展是否大于上层，"
               "以及下层的下行段占比和末端落差是否分别大于中层。比较只反映相对次序，不是合格判定；"
               "两个中位数相等或缺测时不判定支持。角度数据可能部分覆盖，详情见 comparisons.csv。", "",
               "| 对照 | 支持预期次序 | 不支持 | 相等 | 数据不足 |", "|---|---:|---:|---:|---:|"]
    labels = {"upper_angle_vs_middle": "上层夹角小于中层", "middle_reach_vs_upper": "中层伸展大于上层",
              "lower_downward_vs_middle": "下层下行段占比大于中层", "lower_drop_vs_middle": "下层末端落差大于中层"}
    for name, *_ in COMPARISONS:
        items = [r for r in comparisons if r["comparison"] == name]
        report.append(f"| {labels[name]} | {sum(r['supports_expected_order'] is True for r in items)} | "
                      f"{sum(r['supports_expected_order'] is False for r in items)} | "
                      f"{sum(r['relation'] == 'equal' for r in items)} | "
                      f"{sum(r['relation'] == 'insufficient_data' for r in items)} |")
    report += ["", "| 植株 | 上层夹角较小 | 中层伸展较大 | 下层下行段较多 | 下层落差较大 |",
               "|---|---|---|---|---|"]
    def display(item):
        if item["relation"] == "insufficient_data":
            return "数据不足"
        if item["relation"] == "equal":
            value = "相等"
        else:
            value = "支持" if item["supports_expected_order"] else "不支持"
        return value + ("（部分覆盖）" if item["coverage"] == "partial" else "")
    for plant in plants:
        by_name = {r["comparison"]: r for r in comparisons if r["plant_id"] == plant["plant_id"]}
        cells = " | ".join(display(by_name[name]) for name, *_ in COMPARISONS)
        report.append(f"| {plant['plant_id']} | {cells} |")
    report += ["", "单叶分层、点云缺失和少量有效角度都可能影响对照结果。分层和对照使用了部分相同特征，"
               "因此对照不是独立验证；这也不是对植株好坏或产量的判断。", ""]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".entropy-layers-", dir=destination.parent) as tmp:
        stage = Path(tmp) / "result"
        stage.mkdir()
        save_csv(stage / "leaf_layers.csv", classified)
        save_csv(stage / "plant_summary.csv", plants)
        save_csv(stage / "layer_summary.csv", summaries)
        save_csv(stage / "comparisons.csv", comparisons)
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
