"""Entropy-first K=3 models on directly observed leaf profiles."""

from collections import Counter

import numpy as np
from sklearn.metrics import silhouette_score

from .io_utils import dump_json
from .phenotype_clusters import K, N_INIT, SEED, fit_standardized_kmeans
from .phenotype_features import ENTROPY_EDGES, inclination_entropy
from .tables import save_csv

STEM_QUANTILES = (0.01, 0.99)
PROFILE_RADIUS_QUANTILE = 0.99
PROFILE_SECTIONS = 16
MIN_PROFILE_SECTIONS = 3
MIN_SECTION_POINTS = 3
DOWNWARD_INCLINATION_DEG = 100
HORIZONTAL_REACH_QUANTILE = 0.95
FIELDS = (
    "observed_shannon_entropy_bits",
    "horizontal_reach_q95_ratio",
    "distal_drop_ratio",
    "downward_path_fraction",
)
FEATURE_WEIGHTS = np.array([0.55, 0.15, 0.15, 0.15])
MORPHOTYPES = {"C1": "聚拢型", "C2": "伸展型", "C3": "耷拉型"}


def _observed_profile_features(relative):
    result = {"profile_sections": 0, **dict.fromkeys(FIELDS)}
    if not len(relative):
        return result
    horizontal = np.linalg.norm(relative[:, :2], axis=1)
    result["horizontal_reach_q95_ratio"] = float(np.quantile(horizontal, HORIZONTAL_REACH_QUANTILE))
    radial = np.linalg.norm(relative, axis=1)
    limit = np.quantile(radial, PROFILE_RADIUS_QUANTILE)
    if not np.isfinite(limit) or limit <= 0:
        return result
    inside = radial <= limit
    bins = np.minimum((radial[inside] / limit * PROFILE_SECTIONS).astype(int), PROFILE_SECTIONS - 1)
    points = relative[inside]
    centers = [np.median(points[bins == index], axis=0) for index in range(PROFILE_SECTIONS)
               if np.count_nonzero(bins == index) >= MIN_SECTION_POINTS]
    result["profile_sections"] = len(centers)
    if len(centers) < MIN_PROFILE_SECTIONS:
        return result
    # Anchor this observed profile at the existing nearest leaf point.
    profile = np.vstack(([0.0, 0.0, 0.0], centers))
    if not np.any(np.diff(profile, axis=0)):
        return result
    entropy, fractions = inclination_entropy(profile, np.array([0.0, 0.0, 1.0]))
    downward_start = np.searchsorted(ENTROPY_EDGES, DOWNWARD_INCLINATION_DEG)
    result.update(observed_shannon_entropy_bits=max(0.0, entropy),
                  distal_drop_ratio=float(profile[:, 2].max() - profile[-1, 2]),
                  downward_path_fraction=float(np.clip(fractions[downward_start:].sum(), 0.0, 1.0)))
    return result


def _morphotype_order(centers, standardized):
    """Return fitted cluster indices in compact, extended, drooping order."""
    horizontal = FIELDS.index("horizontal_reach_q95_ratio")
    downward = [FIELDS.index(key) for key in ("distal_drop_ratio", "downward_path_fraction")]
    droop = max(range(K), key=lambda i: (sum(standardized[i, downward]), *centers[i, downward], -i))
    compact, extended = sorted((i for i in range(K) if i != droop),
                               key=lambda i: (centers[i, horizontal], *centers[i], i))
    return compact, extended, droop


def plant_cluster_summary(plant, leaf_count, status="ok"):
    return {"plant_id": plant, "leaf_count": leaf_count, "assigned_leaves": 0,
            "status": status, "C1": 0, "C2": 0, "C3": 0, "singleton_clusters": 0,
            "silhouette": None, "inertia": None, "stem_z_q01": None, "stem_z_q99": None}


def cluster_plant(plant, xyz, labels, leaf_rows):
    """Fit one model only when all observed leaf instances can participate."""
    leaf_rows = sorted(leaf_rows, key=lambda row: row["leaf_id"])
    sheets = [{"plant_id": plant, "leaf_id": row["leaf_id"], "plant_cluster_id": None,
               "morphotype": None, "plant_cluster_status": None,
               "profile_sections": 0, **dict.fromkeys(FIELDS)} for row in leaf_rows]
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
                base = np.array([row[f"leaf_{axis}"] for axis in "xyz"])
                entry.update(_observed_profile_features((points - base) / stem_span))

    matrix = np.array([[entry[field] for field in FIELDS] for entry in sheets],
                      dtype=float).reshape(-1, len(FIELDS))
    if len(sheets) < K:
        counts["status"] = "TOO_FEW_LEAVES"
    elif not np.isfinite(matrix).all():
        counts["status"] = "MISSING_OBSERVED_GEOMETRY"
    elif len(np.unique(matrix, axis=0)) < K:
        counts["status"] = "TOO_FEW_DISTINCT_GEOMETRIES"
    else:
        scaler, weighted, model = fit_standardized_kmeans(matrix, FEATURE_WEIGHTS)
        if len(set(model.labels_)) != K:
            counts["status"] = "DEGENERATE_CLUSTERS"
        else:
            standardized_centers = model.cluster_centers_ / np.sqrt(FEATURE_WEIGHTS)
            original_centers = scaler.inverse_transform(standardized_centers)
            order = _morphotype_order(original_centers, standardized_centers)
            lookup = dict(zip(order, MORPHOTYPES))
            sizes = Counter(model.labels_)
            for entry, label in zip(sheets, model.labels_):
                code = lookup[label]
                entry.update(plant_cluster_id=code, morphotype=MORPHOTYPES[code])
            for label in order:
                code = lookup[label]
                counts[code] = sizes[label]
                centers.append({"plant_id": plant, "plant_cluster_id": code, "morphotype": MORPHOTYPES[code],
                                "leaf_count": sizes[label], **dict(zip(FIELDS, original_centers[label]))})
            counts.update(assigned_leaves=len(sheets), inertia=float(model.inertia_),
                          singleton_clusters=sum(n == 1 for n in sizes.values()))
            if len(sheets) > K:
                counts["silhouette"] = float(silhouette_score(weighted, model.labels_))

    for entry in sheets:
        entry["plant_cluster_status"] = counts["status"]
    return sheets, counts, centers


def write_plant_clusters(out, leaves, plants, centers, manifest, provenance):
    directory = out / "plant_morphotypes"
    directory.mkdir()
    save_csv(directory / "leaf_clusters.csv", leaves,
             ("plant_id", "leaf_id", "plant_cluster_id", "morphotype", "plant_cluster_status",
              "profile_sections", *FIELDS))
    save_csv(directory / "plant_summary.csv", plants)
    save_csv(directory / "cluster_centers.csv", centers,
             ("plant_id", "plant_cluster_id", "morphotype", "leaf_count", *FIELDS))
    dump_json(directory / "provenance.json", {
        "scope": "independent entropy-first morphology model per plant",
        "run_at": provenance["run_at"], "git_head": provenance["git_head"],
        "packages": provenance["packages"],
        "sha256_text_source_lf": provenance["sha256_text_source_lf"],
        "k": K, "seed": SEED, "n_init": N_INIT, "feature_order": FIELDS,
        "squared_distance_weights": dict(zip(FIELDS, FEATURE_WEIGHTS)),
        "morphotypes": MORPHOTYPES, "stem_quantiles": STEM_QUANTILES,
        "profile_radius_quantile": PROFILE_RADIUS_QUANTILE, "profile_sections": PROFILE_SECTIONS,
        "min_profile_sections": MIN_PROFILE_SECTIONS, "min_section_points": MIN_SECTION_POINTS,
        "inclination_bin_edges_deg": ENTROPY_EDGES, "downward_inclination_deg": DOWNWARD_INCLINATION_DEG,
        "horizontal_reach_quantile": HORIZONTAL_REACH_QUANTILE,
        "inputs": {row["plant_id"]: {key: row[key] for key in ("csv_sha256", "txt_sha256")}
                   for row in manifest if row["input_status"] == "ok"},
    })
    successful = sum(row["status"] == "ok" for row in plants)
    assigned = sum(row["assigned_leaves"] for row in plants)
    entropies = [row[FIELDS[0]] for row in leaves if row[FIELDS[0]] is not None]
    lines = [
        "# 逐株叶片熵形态聚类", "",
        f"{successful}/{len(plants)} 株各自得到三个非空簇。"
        f"{assigned}/{len(leaves)} 片已识别的叶片获得逐株形态标签。", "",
    ]
    if entropies:
        lines.extend([f"{len(entropies)} 片叶获得观测方向 Shannon 熵，"
                      f"中位数 {np.median(entropies):.3f} bit，"
                      f"范围 {min(entropies):.3f}–{max(entropies):.3f} bit。", ""])
    lines.extend([
        "每株独立标准化后执行 K-means（K=3）。观测方向 Shannon 熵占平方距离权重的 55%，"
        "水平伸展、末端落差和下行段占比分别占 15%。",
        "C3 耷拉型对应下行特征最明显的簇。其余两簇按水平伸展区别为 C1 聚拢型和 C2 伸展型。"
        "类型表示本株内的相对形态，需结合各簇中心和点云核验。", "",
        "| 植株 | 叶片数 | 聚拢型 C1 | 伸展型 C2 | 耷拉型 C3 | 轮廓系数 | 状态 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in plants:
        score = "—" if row["silhouette"] is None else f"{row['silhouette']:.3f}"
        total = row["leaf_count"] if row["leaf_count"] is not None else "—"
        lines.append(f"| {row['plant_id']} | {total} | {row['C1']} | {row['C2']} | "
                     f"{row['C3']} | {score} | {row['status']} |")
    lines.extend([
        "", "单叶簇保留。三片叶各自成簇时轮廓系数未定义。", "",
        "[逐叶分类](leaf_clusters.csv) · [逐株统计](plant_summary.csv) · "
        "[熵与其他特征的簇中心](cluster_centers.csv) · [参数及数据哈希](provenance.json)", "",
    ])
    (directory / "PLANT_CLUSTER_REPORT.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
