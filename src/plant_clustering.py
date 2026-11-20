"""Independent K=3 models using observed point-cloud geometry from each plant."""

from collections import Counter

import numpy as np
from sklearn.metrics import silhouette_score

from .io_utils import dump_json
from .phenotype_clusters import K, N_INIT, SEED, fit_standardized_kmeans
from .tables import save_csv

STEM_QUANTILES = (0.01, 0.99)
LEAF_QUANTILES = (0.05, 0.95)
FIELDS = (
    "nearest_stem_z_ratio",
    "horizontal_reach_q95_ratio",
    "median_rise_ratio",
    "vertical_span_q90_ratio",
)


def plant_cluster_summary(plant, leaf_count, status="ok"):
    return {"plant_id": plant, "leaf_count": leaf_count, "assigned_leaves": 0,
            "status": status, "C1": 0, "C2": 0, "C3": 0, "singleton_clusters": 0,
            "silhouette": None, "inertia": None, "stem_z_q01": None, "stem_z_q99": None}


def cluster_plant(plant, xyz, labels, leaf_rows):
    """Fit one model only when all observed leaf instances can participate."""
    leaf_rows = sorted(leaf_rows, key=lambda row: row["leaf_id"])
    sheets = [{"plant_id": plant, "leaf_id": row["leaf_id"], "plant_cluster_id": None,
               "plant_cluster_status": None, **dict.fromkeys(FIELDS)} for row in leaf_rows]
    counts = plant_cluster_summary(plant, len(sheets))
    centers = []

    valid = np.isfinite(xyz).all(axis=1)
    stem_z = xyz[valid & (labels == 0), 2]
    if len(stem_z):
        lower, upper = np.quantile(stem_z, STEM_QUANTILES)
        counts.update(stem_z_q01=float(lower), stem_z_q99=float(upper))
        stem_span = upper - lower
        if stem_span > 0:
            for row, entry in zip(leaf_rows, sheets):
                if row["status"] != "ok":
                    continue
                points = xyz[valid & (labels == row["leaf_id"])]
                if not len(points):
                    continue
                base = np.array([row[f"leaf_{c}"] for c in "xyz"])
                relative = points - base
                reach = np.quantile(np.linalg.norm(relative[:, :2], axis=1), LEAF_QUANTILES[1])
                low, high = np.quantile(relative[:, 2], LEAF_QUANTILES)
                entry.update(nearest_stem_z_ratio=float((row["stem_z"] - lower) / stem_span),
                             horizontal_reach_q95_ratio=float(reach / stem_span),
                             median_rise_ratio=float(np.median(relative[:, 2]) / stem_span),
                             vertical_span_q90_ratio=float((high - low) / stem_span))

    matrix = np.array([[entry[field] for field in FIELDS] for entry in sheets],
                      dtype=float).reshape(-1, len(FIELDS))
    if len(sheets) < K:
        counts["status"] = "TOO_FEW_LEAVES"
    elif not np.isfinite(matrix).all():
        counts["status"] = "MISSING_OBSERVED_GEOMETRY"
    elif len(np.unique(matrix, axis=0)) < K:
        counts["status"] = "TOO_FEW_DISTINCT_GEOMETRIES"
    else:
        scaler, standardized, model = fit_standardized_kmeans(matrix)
        if len(set(model.labels_)) != K:
            counts["status"] = "DEGENERATE_CLUSTERS"
        else:
            original_centers = scaler.inverse_transform(model.cluster_centers_)
            order = sorted(range(K), key=lambda i: tuple(original_centers[i]))
            lookup = {label: f"C{index}" for index, label in enumerate(order, start=1)}
            sizes = Counter(model.labels_)
            for entry, label in zip(sheets, model.labels_):
                entry["plant_cluster_id"] = lookup[label]
            for label in order:
                name = lookup[label]
                counts[name] = sizes[label]
                centers.append({"plant_id": plant, "plant_cluster_id": name, "leaf_count": sizes[label],
                                **dict(zip(FIELDS, original_centers[label]))})
            counts.update(assigned_leaves=len(sheets), inertia=float(model.inertia_),
                          singleton_clusters=sum(n == 1 for n in sizes.values()))
            if len(sheets) > K:
                counts["silhouette"] = float(silhouette_score(standardized, model.labels_))

    for entry in sheets:
        entry["plant_cluster_status"] = counts["status"]
    return sheets, counts, centers


def write_plant_clusters(out, leaves, plants, centers, manifest, provenance):
    directory = out / "plant_clusters"
    directory.mkdir()
    save_csv(directory / "leaf_clusters.csv", leaves,
             ("plant_id", "leaf_id", "plant_cluster_id", "plant_cluster_status", *FIELDS))
    save_csv(directory / "plant_summary.csv", plants)
    save_csv(directory / "cluster_centers.csv", centers,
             ("plant_id", "plant_cluster_id", "leaf_count", *FIELDS))
    dump_json(directory / "provenance.json", {
        "scope": "independent model per plant",
        "run_at": provenance["run_at"], "git_head": provenance["git_head"],
        "packages": provenance["packages"],
        "sha256_text_source_lf": provenance["sha256_text_source_lf"],
        "k": K, "seed": SEED, "n_init": N_INIT, "feature_order": FIELDS,
        "stem_quantiles": STEM_QUANTILES, "leaf_quantiles": LEAF_QUANTILES,
        "inputs": {row["plant_id"]: {key: row[key] for key in ("csv_sha256", "txt_sha256")}
                   for row in manifest if row["input_status"] == "ok"},
    })
    successful = sum(row["status"] == "ok" for row in plants)
    assigned = sum(row["assigned_leaves"] for row in plants)
    lines = [
        "# 逐株叶片三分类", "",
        f"{successful}/{len(plants)} 株各自得到三个非空簇。"
        f"{assigned}/{len(leaves)} 片已识别的叶片获得逐株簇编号。", "",
        "每株使用自己的 StandardScaler 和 K-means（K=3）。输入为原始点云的四项可观测特征。"
        "C1、C2、C3 依本株簇中心的最近茎点高度从低到高编号，相同编号在不同株间可对应不同形态。"
        "整批六特征聚类及其缺失记录仍保存在原表中。", "",
        "| 植株 | 叶片数 | C1 | C2 | C3 | 轮廓系数 | 状态 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in plants:
        score = "—" if row["silhouette"] is None else f"{row['silhouette']:.3f}"
        total = row["leaf_count"] if row["leaf_count"] is not None else "—"
        lines.append(f"| {row['plant_id']} | {total} | {row['C1']} | {row['C2']} | "
                     f"{row['C3']} | {score} | {row['status']} |")
    lines.extend([
        "", "单叶簇保留。K=3 固定，轮廓系数描述本株的几何分离程度。"
        "三片叶各自成簇时轮廓系数未定义。", "",
        "[逐叶分类](leaf_clusters.csv) · [逐株统计](plant_summary.csv) · "
        "[簇中心](cluster_centers.csv) · [参数及数据哈希](provenance.json)", "",
    ])
    (directory / "PLANT_CLUSTER_REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
