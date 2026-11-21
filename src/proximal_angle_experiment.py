"""Explore raw point PCA near the nearest leaf point; never changes formal angles."""
import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .angle_diagnostics import audit_leaf
from .io_utils import load_cloud, sha256
from .leaf import angle
from .phenotype_features import PARAMS, _project_stem
from .robust_geometry import robust_stem
from .stem import tangent_at
from .tables import save_csv

RADII = (0.10, 0.15, 0.20)  # fraction of the 95th percentile of distances from anchor
MIN_POINTS = 30
MAX_WINDOW_SPAN_DEG = 15.0
MIN_AXIS_RATIO = 1.5


def local_directions(points, anchor):
    """Fit directed PCA axes in three nested radial windows, recording failures."""
    offsets = points - anchor
    distances = np.linalg.norm(offsets, axis=1)
    scale = float(np.quantile(distances, .95))
    if not np.isfinite(scale) or scale <= 0:
        return [], [], [], scale
    directions, counts, ratios = [], [], []
    for fraction in RADII:
        local = offsets[distances <= fraction * scale]
        counts.append(len(local))
        if len(local) < MIN_POINTS:
            directions.append(None)
            ratios.append(None)
            continue
        _, s, axes = np.linalg.svd(local - local.mean(axis=0), full_matrices=False)
        if len(s) < 2 or s[0] <= 0:
            directions.append(None)
            ratios.append(None)
            continue
        direction = axes[0]
        outward = local.mean(axis=0)
        if np.dot(direction, outward) < 0:
            direction = -direction
        directions.append(direction)
        ratios.append(float((s[0] / max(s[1], 1e-12)) ** 2))
    return directions, counts, ratios, scale


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def run(source, data_dir, destination):
    source, data_dir, destination = Path(source), Path(data_dir), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    leaves = read_csv(source / "leaf_features.csv")
    manifest = {r["plant_id"]: r for r in read_csv(source / "input_manifest.csv")}
    with (source / "quality_details.json").open(encoding="utf-8") as f:
        quality = json.load(f)
    output = []
    for plant in sorted({r["plant_id"] for r in leaves}):
        record = manifest[plant]
        path = data_dir / record["csv_file"]
        if sha256(path) != record["csv_sha256"]:
            raise ValueError(f"CSV hash does not match source results: {plant}")
        xyz, labels = load_cloud(path)
        stem = robust_stem(xyz[labels == 0], PARAMS)
        details = {int(d["leaf_id"]): d for d in quality[plant]["leaves"]}
        for row in (r for r in leaves if r["plant_id"] == plant):
            leaf = int(row["leaf_id"])
            points = xyz[labels == leaf]
            base = np.array([float(row["leaf_" + axis]) for axis in "xyz"])
            stem_point = np.array([float(row["stem_" + axis]) for axis in "xyz"])
            audit = audit_leaf(row, details[leaf])
            station, projection = _project_stem(stem_point, stem)
            stem_axis = tangent_at(stem, np.asarray(station))
            vectors, counts, ratios, scale = local_directions(points, base)
            usable = len(vectors) == len(RADII) and all(v is not None for v in vectors)
            values = [angle(v, stem_axis) for v in vectors] if usable else []
            span = max((angle(a, b) for a in vectors for b in vectors), default=None) if usable else None
            failures = []
            if not usable:
                failures.append("INSUFFICIENT_WINDOW_POINTS_OR_DEGENERATE")
            if usable and span > MAX_WINDOW_SPAN_DEG:
                failures.append("WINDOW_DIRECTION_UNSTABLE")
            if any(r is not None and r < MIN_AXIS_RATIO for r in ratios):
                failures.append("LOCAL_AXIS_AMBIGUOUS")
            if projection != "ok":
                failures.append("STEM_PROJECTION_" + projection.upper())
            if audit["pair_gap_exceeded"] is not False:
                failures.append("STEM_LEAF_GAP_FAILED_OR_UNKNOWN")
            if int(row["leaf_point_count"]) < 200:
                failures.append("TOO_FEW_LEAF_POINTS")
            if audit["retained_fraction"] is None or audit["retained_fraction"] < .94:
                failures.append("LEAF_COMPONENT_FRAGMENTED_OR_UNKNOWN")
            # This is only an exploratory shortlist, not an acceptance/accuracy claim.
            shortlist = not failures
            item = {"plant_id": plant, "leaf_id": leaf, "original_status": row["leaf_angle_deg_status"],
                    "formal_angle_deg": row["leaf_angle_deg"], "stem_projection_status": projection,
                    "pair_gap_exceeded": audit["pair_gap_exceeded"], "retained_fraction": audit["retained_fraction"],
                    "radius_scale_q95": scale, "direction_span_deg": span,
                    "passes_exploratory_screen": bool(shortlist), "screen_failures": "|".join(failures)}
            for index, fraction in enumerate(RADII):
                key = str(round(fraction * 100))
                item["points_" + key] = counts[index] if counts else 0
                item["axis_ratio_" + key] = ratios[index] if ratios else None
                item["angle_" + key + "_deg"] = values[index] if values else None
            item["candidate_angle_deg"] = values[1] if usable else None
            item["difference_to_formal_deg"] = (values[1] - float(row["leaf_angle_deg"])) if usable and row["leaf_angle_deg"] else None
            output.append(item)
    target = [r for r in output if "ANCHOR_CURVE_GAP" in r["original_status"]]
    reference = [r for r in output if r["original_status"] == "ok" and r["difference_to_formal_deg"] is not None]
    filtered = [r for r in reference if r["passes_exploratory_screen"]]
    def failure_count(name):
        return sum(name in r["screen_failures"].split("|") for r in target)
    def stats(rows):
        delta = np.abs([r["difference_to_formal_deg"] for r in rows])
        return f"{len(rows)} 片；绝对差中位数 {np.median(delta):.1f}°、90% 分位数 {np.quantile(delta, .9):.1f}°" if len(delta) else "0 片"
    lines = ["# 基于近端原始点的夹角对照实验", "",
             "使用已有最近叶点作为锚点，按该叶点到叶点距离的 95% 分位数设定尺度；"
             "在尺度的 10%、15%、20% 内对原始叶点分别拟合 PCA 主轴，并与原有茎线切向计算夹角。"
             "与正式方法使用同一估计叶基和茎线；该方法无法验证它们的正确性。", "",
             f"问题组 {len(target)} 片，其中 {sum(r['candidate_angle_deg'] is not None for r in target)} 片三个窗口均可计算；"
             f"{sum(r['passes_exploratory_screen'] for r in target)} 片通过探索性筛选。", "",
             f"问题组中 {failure_count('INSUFFICIENT_WINDOW_POINTS_OR_DEGENERATE')} 片有窗口点数不足或退化，"
             f"{failure_count('WINDOW_DIRECTION_UNSTABLE')} 片的窗口方向差超过 15°，"
             f"{failure_count('LOCAL_AXIS_AMBIGUOUS')} 片有窗口主轴不清晰；原因可同时出现。", "",
             "探索性筛选：每个窗口至少 30 点、各窗口最大方向差不超过 15°、每个窗口前两主方差比至少 1.5、"
             "原有茎线投影和茎叶距离检查通过、点数至少 200、主连通分量占比至少 94%。"
             "这些门槛仅用于挑选待人工检查案例，不是精度验证或正式质量标准。", "",
             f"正式夹角有效叶片中可比较：{stats(reference)}；其中通过探索性筛选：{stats(filtered)}。"
             "正式角度也并非人工真值，且此对照非独立标注，差异不等于测量误差。", "",
             "角度候选、窗口点数、形状比、窗口差异和原始质量状态见 experiment.csv；"
             "26 片问题叶单独列在 problem_leaves.csv。"
             "即使通过筛选，也可能由于锚点错误、叶片宽度、叶鞘、遮挡或分割问题而偏差，不能填入正式结果。", ""]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".proximal-pca-", dir=destination.parent) as tmp:
        stage = Path(tmp) / "results"
        stage.mkdir()
        save_csv(stage / "experiment.csv", output)
        save_csv(stage / "problem_leaves.csv", target)
        (stage / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
        (stage / "parameters.json").write_text(json.dumps({"radii": RADII, "min_points": MIN_POINTS,
            "max_window_span_deg": MAX_WINDOW_SPAN_DEG, "min_axis_ratio": MIN_AXIS_RATIO,
            "reference": "existing accepted estimates, not ground truth"}, indent=2) + "\n", encoding="utf-8")
        stage.rename(destination)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/v1.1"))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(f"Examined {len(run(args.results_dir, args.data_dir, args.output_dir))} leaves")


if __name__ == "__main__":
    main()
