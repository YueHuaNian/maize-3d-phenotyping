"""Audit recorded angle quality decisions without changing accepted measurements."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .phenotype_features import RULES
from .tables import save_csv


def number(value):
    if value in (None, ""):
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def audit_leaf(row, detail):
    variants = detail.get("variants") or []
    primary = variants[0] if variants else {}
    length, spacing, gap = (number(primary.get(k)) for k in ("length", "graph_spacing", "root_gap"))
    root_limit = (max(RULES["anchor_gap_length_fraction"] * length,
                      RULES["anchor_gap_spacing_factor"] * spacing)
                  if length is not None and spacing is not None else None)
    distance, pair_limit = number(row.get("nearest_distance")), number(detail.get("pair_gap_gate"))
    return {
        "plant_id": row["plant_id"], "leaf_id": row["leaf_id"],
        "angle_status": row["leaf_angle_deg_status"], "leaf_point_count": row["leaf_point_count"],
        "accepted_angle_deg": row["leaf_angle_deg"],
        # Never substitute this unvalidated estimate for the accepted angle.
        "unvalidated_diagnostic_angle_deg": detail.get("diagnostic_values", {}).get("leaf_angle_deg"),
        "stem_projection_status": detail.get("stem_projection_status"),
        "nearest_distance": distance, "pair_gap_limit": pair_limit,
        "pair_gap_exceeded": distance > pair_limit if distance is not None and pair_limit is not None else None,
        "root_gap": gap, "root_gap_limit": root_limit,
        "root_gap_to_limit": gap / root_limit if gap is not None and root_limit else None,
        "retained_fraction": primary.get("retained_fraction"),
        "proximal_window_span_deg": primary.get("proximal_window_span_deg"),
        "variant_angle_max_deg": detail.get("variant_angle_max_deg"),
        "primary_flags": "|".join(primary.get("flags", [])),
        "supported_variant_count": detail.get("supported_variant_count"),
    }


def run(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    with (source / "leaf_features.csv").open(encoding="utf-8-sig", newline="") as f:
        leaves = list(csv.DictReader(f))
    with (source / "quality_details.json").open(encoding="utf-8") as f:
        quality = json.load(f)
    lookup = {(plant, int(leaf["leaf_id"])): leaf for plant, info in quality.items()
              for leaf in info.get("leaves", [])}
    keys = [(r["plant_id"], int(r["leaf_id"])) for r in leaves]
    if len(set(keys)) != len(keys) or set(keys) != set(lookup):
        raise ValueError("Leaf feature and quality rows are duplicated or do not match")
    audited = [audit_leaf(row, lookup[key]) for row, key in zip(leaves, keys)]
    counts = Counter(r["angle_status"] for r in audited)
    flagged = [r for r in audited if "ANCHOR_CURVE_GAP" in r["angle_status"]]
    ratios = [r["root_gap_to_limit"] for r in flagged if r["root_gap_to_limit"] is not None]
    reasons = {"ok": "有效", "ANCHOR_CURVE_GAP": "叶基与中线起点距离过大",
               "LEAF_COMPONENT_FRAGMENTED|ANCHOR_CURVE_GAP": "点云断连且叶基与中线起点距离过大",
               "LEAF_COMPONENT_FRAGMENTED": "叶点主连通分量不足", "LEAF_POINTS_TOO_FEW": "叶点少于 200",
               "LEAF_STEM_GAP_LARGE": "最近茎叶点距离超过门槛", "STEM_PROJECTION_AT_BOUNDARY": "茎线投影落在边界",
               "STEM_PROJECTION_UNSUPPORTED": "茎线投影缺少支持", "PROXIMAL_DIRECTION_UNSTABLE": "叶基方向不稳定",
               "MIDLINE_VARIANTS_UNSTABLE": "中线拟合扰动不稳定"}
    lines = ["# 叶茎夹角缺失诊断", "", f"共 {len(audited)} 片叶，{counts['ok']} 片有效，"
             f"{len(audited) - counts['ok']} 片缺失。状态为程序首个阻断阶段，不能理解为唯一原因。", "",
             "| 状态 | 解释 | 叶片数 |", "|---|---|---:|"]
    for status, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {status} | {reasons.get(status, '其他或组合状态')} | {count} |")
    if ratios:
        stem_issues = sum(r["stem_projection_status"] not in (None, "ok") for r in flagged)
        pair_issues = sum(r["pair_gap_exceeded"] is True for r in flagged)
        window_issues = sum(number(r["proximal_window_span_deg"]) is not None and
                            float(r["proximal_window_span_deg"]) > RULES["max_window_angle_deg"] for r in flagged)
        lines += ["", f"叶基与中线起点距离过大的叶片共 {len(flagged)} 片（含组合状态）。"
                  f"距离除以门槛的中位数为 {np.median(ratios):.2f}，范围 {min(ratios):.2f}–{max(ratios):.2f}。",
                  f"其中 {stem_issues} 片另有茎线投影边界或支持问题，{pair_issues} 片的茎叶距离也超门槛，"
                  f"{window_issues} 片的近端窗口方向变化超过 15°；这些数量相互重叠。", ""]
    lines += ["状态按程序顺序产生；早期失败后，后续扰动稳定性可能未执行，空白表示未计算而非通过。",
              "诊断角度只是中间计算值，未通过质量筛查，不能替代正式夹角。",
              "结果不能单独区分遮挡、分割问题、叶基误定位或中线方法偏差；需查看原始点云及人工标注。", "",
              "## 建议的排查顺序", "",
              "1. 对叶基到中线起点距离超限的叶片，叠加点云、叶基点和中线，人工核验叶基与近端拟合。",
              "2. 对茎叶距离超限和茎线边界问题，核验语义分割、茎线覆盖和最近点对位置。",
              "3. 对方向或拟合不稳定的叶片，比较不同窗口和邻域参数的局部方向及点云支持。",
              "4. 在人工标注样本上对比改进方法，记录误差和覆盖率后，再考虑更改正式质量规则。", ""]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".angle-audit-", dir=destination.parent) as tmp:
        stage = Path(tmp) / "report"
        stage.mkdir()
        save_csv(stage / "angle_diagnostics.csv", audited)
        (stage / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
        stage.rename(destination)
    return audited


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/v1.1"))
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    rows = run(args.results_dir, args.output_dir)
    print(f"Audited {len(rows)} leaves")


if __name__ == "__main__":
    main()
