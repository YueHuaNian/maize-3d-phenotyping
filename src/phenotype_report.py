"""Batch output and a data-driven Markdown report."""

from collections import Counter
from statistics import fmean, stdev

import numpy as np

from .io_utils import dump_json
from .phenotype_features import FEATURES
from .tables import save_csv

FEATURE_ZH = {
    "leaf_angle_deg": "叶茎夹角（度）", "height_ratio": "归一化着生高度",
    "droop_index": "下垂指数", "curvature_rms_norm": "归一化 RMS 曲率",
    "spread_index": "水平延展指数", "shannon_entropy_bits": "倾角 Shannon 熵（bit）",
}


def distribution(values):
    if not values:
        return dict(n=0, **dict.fromkeys(("min", "q1", "median", "q3", "max", "mean", "sample_sd")))
    q1, median, q3 = np.quantile(values, (0.25, 0.5, 0.75))
    return dict(n=len(values), min=min(values), q1=float(q1), median=float(median),
                q3=float(q3), max=max(values), mean=fmean(values),
                sample_sd=stdev(values) if len(values) > 1 else None)


def _plant_summary(leaves, inputs):
    output = []
    for inp in inputs:
        plant = inp["plant_id"]
        rows = [r for r in leaves if r["plant_id"] == plant]
        counts = Counter(r["cluster_id"] for r in rows)
        summary = {
            "plant_id": plant, "input_status": inp["input_status"], "source_points": inp["source_points"],
            "leaf_count": len(rows) if inp["input_status"] == "ok" else None,
            "nearest_pairs": sum(r["status"] == "ok" for r in rows),
            "complete_leaves": sum(all(r[f] is not None for f in FEATURES) for r in rows),
            "clustered_leaves": sum(counts[c] for c in ("C1", "C2", "C3")),
            **{c: counts[c] for c in ("C1", "C2", "C3")},
        }
        for feature in FEATURES:
            stats = distribution([r[feature] for r in rows if r[feature] is not None])
            summary.update({f"{feature}_{k}": stats[k] for k in ("n", "q1", "median", "q3")})
        output.append(summary)
    return output


def write_outputs(out, leaves, inputs, checks, provenance, model, centers, folds):
    plant_rows = _plant_summary(leaves, inputs)
    features = [
        {"feature": name, **distribution([r[name] for r in leaves if r[name] is not None]),
         "missing": sum(r[name] is None for r in leaves)}
        for name in FEATURES
    ]
    save_csv(out / "leaf_features.csv", leaves,
             ("plant_id", "leaf_id", "status", *FEATURES, "cluster_id", "cluster_status"))
    save_csv(out / "plant_summary.csv", plant_rows)
    save_csv(out / "feature_summary.csv", features)
    save_csv(out / "input_manifest.csv", inputs)
    save_csv(out / "cluster_centers.csv", centers, ("cluster_id", "leaves", "plants", *FEATURES))
    save_csv(out / "cluster_stability.csv", folds,
             ("held_out_plant", "train_leaves", "train_plants", "held_out_leaves", "status",
              "train_ari", "held_out_ari", "held_out_matching_leaves", "held_out_agreement"))
    dump_json(out / "quality_details.json", checks)
    dump_json(out / "run_provenance.json", provenance)
    dump_json(out / "cluster_model.json", model)
    _report(out, plant_rows, features, model, centers, folds, provenance, leaves)


def _report(out, plants, features, model, centers, folds, provenance, leaves):
    def fmt(value):
        return f"{value:.5g}" if value is not None else "—"

    total_points = sum(r["source_points"] or 0 for r in plants)
    ok = sum(r["input_status"] == "ok" for r in plants)
    localized = sum(r["nearest_pairs"] for r in plants)
    lines = [
        "# V1.1 玉米叶片几何与聚类分析", "",
        f"运行于 {provenance['run_at']}。运行代码提交 {provenance['git_head']}。",
        f"通过 TXT/CSV 逐行核验 {ok}/{len(plants)} 株，共 {total_points:,} 行原始点数据。"
        f"识别 {len(leaves)} 片叶，其中 {localized} 片获得 V1.0 最近点对。", "",
        "原始坐标物理单位待确认。角度以度计，熵以 bit 计，其他主特征均为无量纲量。",
        "每片叶的保留状态和零基源行号见 [逐叶 CSV](leaf_features.csv)。"
        "各项分布基于通过对应质量门槛的叶片，植株为独立观测单位。", "",
        "## 特征与 Shannon 熵", "",
        "| 特征 | 有效叶数 | 缺失 | 最小值 | Q1 | 中位数 | Q3 | 最大值 | 均值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in features:
        values = [row[key] for key in ("n", "missing", "min", "q1", "median", "q3", "max", "mean")]
        lines.append("| " + " | ".join([FEATURE_ZH[row["feature"]], *map(fmt, values)]) + " |")
    lines.extend([
        "",
        "倾角 Shannon 熵用 18 个等宽角度箱统计可见中线各段的弧长占比。"
        "较高值表示弧长分布于更多倾角范围。缺失原因、几何诊断值和各箱概率见 "
        "[质量明细](quality_details.json)。", "",
        "![六项叶片特征分布](feature_distributions.png)", "",
        "## K-means 与按株稳定性", "",
        f"六项特征齐全的叶片共 {model['complete_leaves']} 片，来自 "
        f"{model['complete_plants']} 株。{model['assigned_leaves']} 片获得聚类标签。",
        "模型在标准化后的六维特征空间中拟合。二维和三维 PCA 仅用于展示。",
        "类别按簇中心的归一化着生高度从低到高编号。", "",
    ])
    if model["status"] == "ok":
        metrics, stability = model["metrics"], model["stability"]
        lines.extend([
            f"轮廓系数 {fmt(metrics['silhouette'])}，Calinski–Harabasz 指数 "
            f"{fmt(metrics['calinski_harabasz'])}，Davies–Bouldin 指数 "
            f"{fmt(metrics['davies_bouldin'])}，簇内平方和 {fmt(metrics['inertia'])}。", "",
            "| 簇 | 叶片数 | 植株数 | 平均高度比 | 平均倾角熵（bit） |",
            "|---|---:|---:|---:|---:|",
            *(f"| {r['cluster_id']} | {r['leaves']} | {r['plants']} | {fmt(r['height_ratio'])} | "
              f"{fmt(r['shannon_entropy_bits'])} |" for r in centers),
            "",
            f"逐株留出完成 {stability['evaluated_plants']}/{len(folds)} 折。"
            f"训练重叠叶的 ARI 中位数 {fmt(stability['train_ari_median'])}，"
            f"最小值 {fmt(stability['train_ari_min'])}，"
            f"按叶数加权的留出标签一致率 {fmt(stability['held_out_leaf_agreement'])}，"
            f"各株等权的一致率 {fmt(stability['held_out_plant_mean_agreement'])}。"
            f"至少有 2 片留出叶的折的 ARI 中位数为 {fmt(stability['held_out_ari_median'])}。",
            "每折只用余下植株重新训练缩放器与 K-means，用训练重叠部分匹配簇标签。"
            "整批模型参与了参照标签生成，因此这些指标衡量组内复算稳定性。", "",
            "[簇中心](cluster_centers.csv) · [逐株留出结果](cluster_stability.csv) · [模型参数](cluster_model.json)",
        ])
    else:
        lines.append(f"本次聚类状态为 {model['status']}。共同模型要求至少两株有完整叶片特征。")
    lines.extend([
        "", "![二维 PCA 聚类图](cluster_2d.png)", "",
        "本机交互输出位于 interactive/。cluster_3d.html 可旋转 PCA 簇图并查阅逐叶特征。"
        "plant_geometry_3d.html 可逐株旋转拟合茎线、叶片可见中线和最近点对。"
        "plant_clusters_3d.html 按独立逐株模型给各叶着色。", "",
        "[逐株三分类报告](plant_clusters/PLANT_CLUSTER_REPORT.md)", "",
        "## 逐株结果", "",
        "| 植株 | 输入状态 | 叶片数 | 最近点对 | 完整特征 | C1 | C2 | C3 | 倾角熵中位数（bit） |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        *(f"| {r['plant_id']} | {r['input_status']} | {fmt(r['leaf_count'])} | {r['nearest_pairs']} | "
          f"{r['complete_leaves']} | {r['C1']} | {r['C2']} | {r['C3']} | "
          f"{fmt(r['shannon_entropy_bits_median'])} |" for r in plants), "",
        "[逐株完整统计](plant_summary.csv) · [汇总特征统计](feature_summary.csv) · "
        "[原始输入 SHA-256](input_manifest.csv) · [软件版本和运行参数](run_provenance.json)", "",
        "## 解释范围", "",
        "各项结果描述通过质量筛查的可见点云几何。最近点对仍是对叶基的初始估计，"
        "遮挡或贴茎叶段会造成方向歧义。模型覆盖与缺失情况应结合逐株结果判读。"
        "聚类与倾角熵的生物学含义需要独立数据验证。", "",
    ])
    (out / "V1.1_ANALYSIS_REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
