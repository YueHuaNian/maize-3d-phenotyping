"""阶段 1（辅助入口）：基于已保存模型导出保守观测支撑域，无需重跑 TPS 拟合。

功能说明：
    - 严格只读原模型目录，在新目录导出收缩后的模型与整株网格；
    - 沿观测跨度独立限制左右边缘，并撤下未观测叶尖延长（截断端保留开放横截面）；
    - 对每片叶执行修正前后的贴合、拓扑与远离点云面积占比审计，生成 support_audit.json；
    - 注意：本脚本仅限制既有曲面的有效支撑域，不推断缺失叶片的真实形态。

常用命令：
    python scripts/refine_surface_support.py --source outputs/archive/... --output outputs/maize_supported
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
# 将运行生成的字节码集中到缓存区，源码目录保持整洁。
sys.pycache_prefix = str(ROOT / ".cache" / "python")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".vendor")]
import numpy as np
from scipy.spatial import cKDTree, Delaunay
from maize_leaf.geometry import clean_leaf_points, project_points_to_curve
from maize_leaf.io import read_ply, write_ply, write_obj, write_json
from maize_leaf.surface import MaizeLeafModel, COLORS, REGIONS, restrict_to_scan_support, triangle_areas
from maize_leaf.quality import audit_model


def finalize_output(output):
    """汇总导出整株拼接网格（PLY/OBJ），并将旧版 HTML 入口重定向到共用三维查看器。"""
    for plant_dir in sorted(Path(output).glob("plant_*")):
        vertices, faces, regions, face_regions, ids = [], [], [], [], []
        offset = 0
        plant = plant_dir.name.removeprefix("plant_")
        for path in sorted((plant_dir / "leaves").glob("leaf_*_surface.ply")):
            leaf = int(path.name.split("_")[1])
            v, f = read_ply(path)
            vertices.append(v[:, :3])
            regions.append(v[:, 6].astype(int))
            faces.append(f[:, 1:4] + offset)
            face_regions.append(f[:, 4])
            ids.append(np.full(len(v), leaf))
            offset += len(v)
            (path.parent / f"leaf_{leaf:02d}_visualization.html").write_text(
                f'<meta charset="utf-8"><meta http-equiv="refresh" content="0;url=../../../classification_v2/viewer/plant_{plant}.html?leaf={leaf}">', encoding="utf-8")
        if not vertices:
            continue
        vertices, faces = np.vstack(vertices), np.vstack(faces)
        regions, face_regions = np.concatenate(regions), np.concatenate(face_regions)
        write_ply(plant_dir / "whole_plant_mesh.ply", vertices, faces, COLORS[regions], regions, face_regions, np.concatenate(ids))
        write_obj(plant_dir / "whole_plant_mesh.obj", vertices, faces, face_regions, REGIONS, COLORS)
        (plant_dir / "whole_plant.html").write_text(
            f'<meta charset="utf-8"><meta http-equiv="refresh" content="0;url=../../classification_v2/viewer/plant_{plant}.html">', encoding="utf-8")
    (Path(output) / "index.html").write_text('<meta charset="utf-8"><meta http-equiv="refresh" content="0;url=../classification_v2/viewer/index.html">', encoding="utf-8")


def export_supported_leaf(source, target, plant, leaf):
    """对单叶应用保守观测支撑域限制、重新划分观测/插值/外推区域并执行质量审计。"""
    key = f"leaf_{leaf:02d}"
    path = source / f"{key}_model.npz"
    model = MaizeLeafModel.load(path)
    # 1. 读取原始点云并复现标准去噪流程，记录修正前质量基线
    raw = read_ply(source / f"{key}_points.ply")[0][:, :3]
    points, _ = clean_leaf_points(raw)
    before = audit_model(model, points)
    prior_traits = json.loads((source / f"{key}_traits.json").read_text(encoding="utf-8"))
    # 2. 独立收紧左右边缘并撤下未观测端部外推，核验拓扑与正向贴合不退化
    policy = restrict_to_scan_support(model, points)
    after = audit_model(model, points)
    if not after["passed"]:
        raise ValueError(f"{plant}/L{leaf}: supported candidate failed topology/orientation/fit checks")
    if after["fit"]["p95"] > before["fit"]["p95"] + 2 * after["fit"]["spacing"]:
        raise ValueError(f"{plant}/L{leaf}: point coverage worsened beyond two scan spacings")
    # 3. 在新活动域上重建网格，并依据扫描凸包与近邻距离标记观测/插值/外推面元
    sv, faces = model.mesh_grid()
    vertices = model.evaluate(sv)
    projection = project_points_to_curve(points, model.midrib, model.length_obs)
    _, b, _ = model.frame.evaluate(projection)
    lateral = np.sum((points - model.midrib(projection)) * b, axis=1)
    hull = Delaunay(np.column_stack([projection, lateral]))
    inside = hull.find_simplex(np.column_stack([sv[:, 0], model.lateral_coordinate(sv[:, 0], sv[:, 1])])) >= 0
    distances = cKDTree(points).query(vertices)[0]
    support_radius = max(2.5 * policy["spacing"], .75 * max(model.length / 89, np.max(model.width(sv[:, 0])) / 24))
    regions = np.where(inside, np.where(distances <= support_radius, 0, 1), 2).astype(np.uint8)
    face_regions = regions[faces].max(axis=1)
    areas = triangle_areas(vertices, faces)
    fractions = {REGIONS[k]: float(areas[face_regions == k].sum() / areas.sum()) for k in REGIONS}
    s = np.linspace(0., model.length, 500)
    midrib = model.evaluate(np.column_stack([s, np.zeros(len(s))]))
    length = float(np.sum(np.linalg.norm(np.diff(midrib, axis=0), axis=1)))
    fit_v = np.clip(model.lateral_parameter(projection, lateral), -1., 1.)
    residual = model.evaluate(np.column_stack([projection, fit_v])) - points
    traits = {**prior_traits, "surface_area": float(areas.sum()), "area_fractions": fractions,
              "completion_ratio": fractions["interpolated"] + fractions["extrapolated"],
              "observed_length": length, "completed_length": length, "tip_completion_length": 0.,
              "parameter_completed_length": model.length, "max_width": float(np.max(model.width(s))),
              "fit_rmse": float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))),
              "scan_distance_median": after["fit"]["median"], "scan_distance_p95": after["fit"]["p95"],
              "surface_support_mean": after["support"]["mean"], "quality_passed": after["passed"],
              "support_far_area_fraction": after["support"]["far_area_fraction"],
              "support_review_required": bool(after["support"]["far_area_fraction"] > .05),
              "support_radius": support_radius, "support_policy": policy,
              "tip_completion_status": "unobserved_extension_withheld" if model.open_tip else "observed_terminal",
              "baseline_model_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source / f"{key}_points.ply", target / f"{key}_points.ply")
    model.save(target / f"{key}_model.npz")
    write_ply(target / f"{key}_surface.ply", vertices, faces, COLORS[regions], regions, face_regions)
    write_obj(target / f"{key}_surface.obj", vertices, faces, face_regions, REGIONS, COLORS)
    write_json(target / f"{key}_traits.json", traits)
    write_json(target / f"{key}_quality.json", after)
    sample_s = np.linspace(0., model.length, 90)
    write_json(target / f"{key}_midrib.json", {"s": sample_s.tolist(), "xyz": model.evaluate(np.column_stack([sample_s, np.zeros(len(sample_s))])).tolist()})
    write_json(target / f"{key}_blade.json", {**model.blade.to_dict(), "active_parameter_length": model.length})
    return {"plant": plant, "leaf": leaf, "before": before, "after": after,
            "area_before": prior_traits["surface_area"], "area_after": float(areas.sum()),
            "domain_policy": policy, "input_sha256": traits["baseline_model_sha256"]}, traits


def main():
    # 解析命令行参数：禁止原地覆盖源目录，确保旧实验可追溯
    """将已有模型限制至保守观测支撑域并另存审计结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/maize_v2", help="已保存模型的只读源目录")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/maize_supported", help="新输出目录（不可已存在）")
    parser.add_argument("--leaf-key", action="append", help="可选筛选指定叶片，格式 plant:leaf，可重复传入")
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        raise ValueError("Original models must remain read-only")
    if args.output.exists():
        raise FileExistsError("Use a new output directory to preserve prior experiments")
    selected = set(args.leaf_key or [])
    records, traits = [], []
    args.output.mkdir(parents=True)
    for plant_dir in sorted(args.source.glob("plant_*")):
        plant = plant_dir.name.removeprefix("plant_")
        for model in sorted((plant_dir / "leaves").glob("leaf_*_model.npz")):
            leaf = int(model.name.split("_")[1])
            if selected and f"{plant}:{leaf}" not in selected:
                continue
            target = args.output / plant_dir.name
            target.mkdir(parents=True, exist_ok=True)
            for name in ("stem_centerline.json", "stem_centerline.ply"):
                if (plant_dir / name).exists() and not (target / name).exists():
                    shutil.copy2(plant_dir / name, target / name)
            record, leaf_traits = export_supported_leaf(plant_dir / "leaves", target / "leaves", plant, leaf)
            records.append(record)
            traits.append(leaf_traits)
            print(f"{plant}/L{leaf}: far area {record['before']['support']['far_area_fraction']:.3f} -> "
                  f"{record['after']['support']['far_area_fraction']:.3f}; area {record['area_before']:.4f} -> {record['area_after']:.4f}", flush=True)
            write_json(args.output / "support_audit.json", {"source": os.path.relpath(args.source, ROOT).replace("\\", "/"), "records": records})
    if not records:
        raise ValueError("No matching leaves")
    finalize_output(args.output)
    flat = [{**{k: v for k, v in t.items() if not isinstance(v, (dict, list))}, **t["area_fractions"]} for t in traits]
    with (args.output / "traits.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in flat))))
        writer.writeheader()
        writer.writerows(flat)
    print(f"Exported {len(records)} supported models to {args.output}", flush=True)


if __name__ == "__main__":
    main()
