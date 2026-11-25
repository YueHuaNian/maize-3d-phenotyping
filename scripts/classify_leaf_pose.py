"""阶段 3（主入口）：基于三项形态特征与相对着生高度的玉米叶片姿态三分类。

功能说明：
    读取阶段 2 表型表（默认 outputs/phenotypes_v2/leaf_traits.csv），执行：
    1. 特征构造：茎叶角 θ、同株相对投影面积 P、纵轴俯仰角信息熵 H_pitch 与相对茎高 h；
    2. 标准化与加权：在每个 LOPO 训练折内拟合 median/IQR，再按平方距离权重 [1, 1, 1, w_h] 缩放；
    3. 聚类与语义判定：全局 K-means++（k=3）学习三类原型，结合顶部小型幼叶与端点距离比规则输出
       紧缩 / 舒展 / 下垂标签（每株不设强制类别配额）；
    4. 稳定性与敏感性评估：14 株留一株交叉验证（LOPO）、特征消融、高度权重扫描与距离比敏感性；
    5. 导出逐叶分类表、逐株类别占比表、完整 JSON 报告与交互式四联诊断页。

注意：
    LOPO ARI 衡量相对全数据划分的稳定性，而非无人工标签条件下的分类准确率；
    因高度已作为聚类输入，其高度分层结果仅作描述统计，不作独立验证。

常用命令：
    python scripts/classify_leaf_pose.py --height-weight 0.5
"""
import argparse
import csv
import hashlib
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
# 将运行生成的字节码集中到缓存区，源码目录保持整洁。
sys.pycache_prefix = str(ROOT / ".cache" / "python")
sys.path[:0] = [str(ROOT / "src"), str(ROOT / ".vendor")]

import numpy as np
import scipy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from maize_leaf.classification import (
    ABLATION_MODELS, CLASS_NAMES, DEFAULT_HEIGHT_WEIGHT, DEFAULT_POSE_POLICY, PRIMARY_FEATURES,
    adjusted_rand_index, lopo_validate, contextual_predict, KMeansModel,
)
from maize_leaf.io import write_json
from maize_leaf.viz import ensure_plotly_js

COLORS = ["#1f77b4", "#1b9e77", "#d95f02"]
HEIGHT_DEFINITION = "clip((leaf_base_z - stem_min_z) / (stem_max_z - stem_min_z), 0, 1)"


def load_traits(path):
    """读取并校验阶段 2 表型 CSV：检查重复键、数值有限性及相对茎高 [0, 1] 边界。"""
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("Trait table is empty")
    seen = set()
    for row in rows:
        key = (row["plant_id"], int(row["leaf_id"]))
        if key in seen:
            raise ValueError(f"Duplicate plant/leaf key: {key}")
        seen.add(key)
        row["relative_height"] = float(row["stem_rel_height"])
        for name in PRIMARY_FEATURES:
            if not np.isfinite(float(row[name])):
                raise ValueError(f"Non-finite {name} for {key}")
        if not 0 <= row["relative_height"] <= 1:
            raise ValueError(f"Relative height outside [0, 1] for {key}")
    return rows


def height_distribution(labels, heights):
    """按相对高度三等分统计各类别占比（纯描述统计：高度已参与聚类，不作独立验证）。"""
    result = {}
    for name, lo, hi in [("lower", 0, 1/3), ("middle", 1/3, 2/3), ("upper", 2/3, 1.000001)]:
        mask = (heights >= lo) & (heights < hi)
        result[name] = {
            "n_leaves": int(mask.sum()),
            **{CLASS_NAMES[c]: float(np.mean(labels[mask] == c)) if mask.any() else None for c in range(3)},
        }
    return result


def write_csv(path, rows):
    """按记录字段写入带 UTF-8 标记的 CSV 表。"""
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_diagnostics(output, primary, ablations, sensitivity, rows):
    """生成交互式四联诊断 HTML（PCA 仅用于二维可视化投影，不参与聚类距离计算）。"""
    fig = make_subplots(rows=2, cols=2, subplot_titles=(
        "Weighted feature space (PCA display)", "Cluster centroids (weighted robust scale)",
        "Feature ablation: partition stability", "Height weight sensitivity (not accuracy)",
    ), vertical_spacing=0.18, horizontal_spacing=0.12)
    Z = primary["Z_full"]
    centered = Z - Z.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pcs = centered @ vt[:2].T
    for c, name in enumerate(CLASS_NAMES):
        mask = primary["full_labels"] == c
        ids = np.flatnonzero(mask)
        hover = [f"{rows[i]['plant_id']} / L{rows[i]['leaf_id']}<br>h={float(rows[i]['relative_height']):.3f}" for i in ids]
        fig.add_trace(go.Scatter(x=pcs[mask, 0], y=pcs[mask, 1], mode="markers", name=name,
                                text=hover, hovertemplate="%{text}<extra>%{fullData.name}</extra>",
                                marker=dict(color=COLORS[c], size=8)), row=1, col=1)
        fig.add_trace(go.Bar(x=["Angle", "Projection", "Entropy", "Height"],
                            y=primary["centers_scaled"][c], name=name, marker_color=COLORS[c],
                            showlegend=False), row=1, col=2)
    keys = list(ablations)
    for metric, title, color in [("lopo_oof_ari", "Held-out predictions vs full partition", "#1f77b4"),
                                  ("silhouette", "Silhouette", "#1b9e77")]:
        fig.add_trace(go.Scatter(x=keys, y=[ablations[k][metric] for k in keys], mode="lines+markers",
                                name=title, line=dict(color=color)), row=2, col=1)
    fig.add_trace(go.Scatter(x=[r["height_weight"] for r in sensitivity],
                            y=[r["lopo_oof_ari"] for r in sensitivity], mode="lines+markers",
                            name="Height sweep: OOF ARI", line=dict(color="#d95f02")), row=2, col=2)
    fig.update_layout(template="plotly_white", height=800, width=1200, barmode="group",
                      margin=dict(t=85, b=100), title="Three morphology factors + relative insertion height",
                      legend=dict(orientation="h", y=-0.14))
    fig.update_xaxes(title_text="Squared-distance height weight", row=2, col=2)
    fig.write_html(str(output / "classification_analysis.html"), include_plotlyjs="assets/plotly.min.js")


def run(traits_csv, output, seed=42, height_weight=DEFAULT_HEIGHT_WEIGHT):
    """执行完整的四因素三分类、特征消融、权重/规则敏感性分析并导出报告与图表。"""
    if not np.isfinite(height_weight) or height_weight < 0:
        raise ValueError("Height weight must be finite and nonnegative")
    t0 = time.time()
    # 1. 加载表型表并准备植株分组与高度数组
    rows = load_traits(traits_csv)
    plants = np.array([r["plant_id"] for r in rows])
    heights = np.array([r["relative_height"] for r in rows])

    def fit(features, weight=height_weight):
        # 仅主模型使用显式顶部幼叶/端点语义规则，消融对照使用原始原型聚类
        """按指定特征与高度权重拟合全局原型并应用统一语义规则。"""
        X = np.array([[float(r[name]) for name in features] for r in rows])
        weights = [weight if name == "relative_height" else 1.0 for name in features]
        return lopo_validate(X, plants, features, seed=seed, feature_weights=weights,
                             decision_policy=DEFAULT_POSE_POLICY if features == PRIMARY_FEATURES else None)

    # 2. 运行 M1–M5 消融及不同高度权重（0、0.25、0.5、1、2）敏感性扫描
    outcomes = {}
    for name, features in ABLATION_MODELS.items():
        outcomes[name] = fit(features)
        metrics = outcomes[name]["metrics"]
        print(f"{name}: OOF ARI={metrics['lopo_oof_ari']:.4f}; "
              f"switch={metrics['lopo_oof_switch_rate_pct']:.1f}%; counts={metrics['class_counts']}", flush=True)
    primary = outcomes["M5_primary"]
    no_height = outcomes["M3_morphology"]
    sensitivity = []
    for weight in sorted({0.0, 0.25, 0.5, 1.0, 2.0, float(height_weight)}):
        trial = primary if weight == height_weight else fit(PRIMARY_FEATURES, weight)
        sensitivity.append({"height_weight": weight, **trial["metrics"],
                            "ari_vs_primary": adjusted_rand_index(primary["full_labels"], trial["full_labels"])})

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    ensure_plotly_js(output)
    leaf_rows = []
    for i, row in enumerate(rows):
        lbl = int(primary["full_labels"][i])
        oof = int(primary["oof_labels"][i])
        record = {"plant_id": row["plant_id"], "leaf_id": int(row["leaf_id"])}
        for name in ["stem_leaf_angle_deg", "projection_ratio", "rel_projected_area", "pitch_entropy_b12", "relative_height",
                     "stem_rel_height", "norm_base_height", "projected_area_element", "surface_area",
                     "quality_weight", "coverage_observed", "coverage_extrapolated", "tip_gain", "droop_ratio"]:
            record[name] = float(row[name])
        record.update(cluster_kmeans=CLASS_NAMES[lbl], cluster_kmeans_id=lbl,
                      cluster_kmeans_lopo_oof=CLASS_NAMES[oof], lopo_switched=bool(lbl != oof),
                      distance_ratio_r=float(primary["distance_ratio"][i]),
                      ambiguous_r080=bool(primary["distance_ratio"][i] > 0.80),
                      cluster_raw_kmeans=CLASS_NAMES[int(primary["raw_labels"][i])],
                      semantic_reassigned=bool(lbl != primary["raw_labels"][i]),
                      cluster_without_height=CLASS_NAMES[int(no_height["full_labels"][i])],
                      changed_by_height=bool(lbl != no_height["full_labels"][i]))
        leaf_rows.append(record)
    write_csv(output / "leaf_classifications.csv", leaf_rows)
    plant_rows = []
    for plant in sorted(set(plants)):
        sub = [r for r in leaf_rows if r["plant_id"] == plant]
        plant_rows.append({"plant_id": plant, "leaf_count": len(sub),
                           **{f"n_{c}": sum(r["cluster_kmeans"] == c for r in sub) for c in CLASS_NAMES},
                           "n_ambiguous": sum(r["ambiguous_r080"] for r in sub),
                           "n_lopo_switched": sum(r["lopo_switched"] for r in sub),
                           "n_changed_by_height": sum(r["changed_by_height"] for r in sub)})
    write_csv(output / "plant_class_fractions.csv", [
        {**r, **{f"class_fraction_{c}": r[f"n_{c}"] / r["leaf_count"] for c in CLASS_NAMES}}
        for r in plant_rows
    ])
    centroids = {
        name: {"n_leaves": int(np.sum(primary["full_labels"] == c)),
               "prototype_raw": dict(zip(PRIMARY_FEATURES, primary["centers_raw"][c].tolist())),
               "final_class_mean": {f: float(np.mean([float(rows[i][f]) for i in np.flatnonzero(primary["full_labels"] == c)])) if np.any(primary["full_labels"] == c) else None for f in PRIMARY_FEATURES}}
        for c, name in enumerate(CLASS_NAMES)
    }
    sources = ["src/maize_leaf/classification.py", "scripts/classify_leaf_pose.py"]
    prototype_model = KMeansModel()
    prototype_model.centers_ = primary["centers_scaled"]
    policy_sensitivity = []
    for ratio in (0.5, 0.6, 2/3, 0.7, 0.8):
        labels, _, _ = contextual_predict(
            prototype_model, primary["Z_full"],
            np.array([[float(r[f]) for f in PRIMARY_FEATURES] for r in rows]), plants,
            PRIMARY_FEATURES, {**DEFAULT_POSE_POLICY, "endpoint_distance_ratio": ratio})
        policy_sensitivity.append({"endpoint_distance_ratio": ratio,
                                   "class_counts": {CLASS_NAMES[c]: int(np.sum(labels == c)) for c in range(3)}})
    report = {
        "schema_version": "four_factor_apical_young_v2",
        "traits_csv": os.path.relpath(traits_csv, ROOT).replace("\\", "/"),
        "traits_sha256": hashlib.sha256(Path(traits_csv).read_bytes()).hexdigest(),
        "source_sha256": {s: hashlib.sha256((ROOT / s).read_bytes()).hexdigest() for s in sources},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "scipy": scipy.__version__},
        "seed": seed, "n_plants": len(plant_rows), "n_leaves": len(rows),
        "primary_model": {
            "scheme": "three_morphology_plus_height_apical_young", "features": PRIMARY_FEATURES,
            "algorithm": "kmeans++ with explicit apical-young endpoint policy",
            "decision_policy": DEFAULT_POSE_POLICY,
            "per_plant_class_quota": None, "n_clusters": 3, "n_init": 50, "max_iter": 500,
            "scaler": "training-fold median/IQR, then multiply by sqrt(distance weight)",
            "height_excluded_from_clustering": False, "height_source_column": "stem_rel_height",
            "height_definition": HEIGHT_DEFINITION, "height_weight": height_weight,
            "feature_distance_weights": primary["feature_weights"].tolist(),
            "weight_selection": "Fixed initial weight; sensitivity sweep does not change the primary choice",
            "semantic_labels": "Small apical candidates may be compact; confident drooping endpoints remain drooping; transitions are expanded",
            "scaler_median": primary["scaler_median"].tolist(), "scaler_iqr": primary["scaler_iqr"].tolist(),
            "centers_weighted_scaled": primary["centers_scaled"].tolist(),
        },
        "primary_metrics": primary["metrics"], "primary_centroid_profiles": centroids,
        "ablation": {k: v["metrics"] for k, v in outcomes.items()},
        "height_weight_sensitivity": sensitivity,
        "decision_policy_sensitivity": policy_sensitivity,
        "height_distribution_descriptive": height_distribution(primary["full_labels"], heights),
        "interpretation": {
            "stability": "LOPO OOF ARI compares held-out predictions with the full-data partition; no external truth labels",
            "height": "Height is an input. Its class distribution is descriptive, not independent biological validation",
            "projection": "Element horizontal projection / maximum element projection in the same plant; retains relative leaf size; geometry proxy, not photosynthetic capacity",
            "entropy": "Area-weighted longitudinal tangent pitch entropy, 12 fixed bins; not a signed droop measure",
            "development_policy": "Apical small-leaf limits and distance margin are expert-guided initial settings, not validated biological constants; no forced class counts",
            "ablation": "Only the full-feature primary uses the explicit endpoint policy; ablations are raw prototype clustering and are not controlled policy-effect estimates",
            "comparison_flag": "changed_by_height compares contextual labels with raw no-height clustering; it includes policy differences and is not an isolated height effect",
            "uncertainty": "All leaves retained; coverage and extrapolation exported for review; missing-tip robustness remains unresolved",
            "senescence": "Drooping is a geometric label, not a diagnosis of leaf senescence",
        },
        "elapsed_s": round(time.time() - t0, 2),
    }
    write_json(output / "classification_report.json", report)
    make_diagnostics(output, primary, report["ablation"], sensitivity, rows)
    print(f"Saved {len(rows)} leaves / {len(plant_rows)} plants to {output}", flush=True)
    return report


def main():
    # 解析命令行参数（默认读取 outputs/phenotypes_v2，输出至 outputs/classification_v2）
    """解析四因素分类、留一株验证及敏感性分析参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traits-csv", type=Path, default=ROOT / "outputs/phenotypes_v2/leaf_traits.csv", help="阶段 2 逐叶表型 CSV 路径")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/classification_v2", help="分类结果与诊断页输出目录")
    parser.add_argument("--seed", type=int, default=42, help="K-means++ 与置换检验随机种子")
    parser.add_argument("--height-weight", type=float, default=DEFAULT_HEIGHT_WEIGHT,
                        help="标准化后相对着生高度的平方距离权重（设为 0 表示屏蔽高度特征）")
    args = parser.parse_args()
    run(args.traits_csv, args.output, args.seed, args.height_weight)


if __name__ == "__main__":
    main()
