"""Complete-case K-means and leave-one-plant-out agreement for V1.1."""

from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .phenotype_features import FEATURES

K = 3
SEED = 20261120
N_INIT = 50


def _sufficient(matrix):
    return len(matrix) > K and len(np.unique(matrix, axis=0)) >= K


def fit_standardized_kmeans(matrix):
    scaler = StandardScaler()
    standardized = scaler.fit_transform(matrix)
    model = KMeans(n_clusters=K, n_init=N_INIT, random_state=SEED,
                   max_iter=300, tol=1e-4, algorithm="lloyd").fit(standardized)
    return scaler, standardized, model


def _stability(matrix, groups, reference, requested_plants):
    output = []
    for plant in requested_plants:
        held = groups == plant
        train = ~held
        fold = {"held_out_plant": plant, "train_leaves": int(train.sum()),
                "train_plants": len(set(groups[train])), "held_out_leaves": int(held.sum()),
                "status": "ok", "train_ari": None, "held_out_ari": None,
                "held_out_matching_leaves": None, "held_out_agreement": None}
        output.append(fold)
        if not held.any():
            fold["status"] = "NO_COMPLETE_HELD_OUT_LEAVES"
            continue
        if not _sufficient(matrix[train]) or fold["train_plants"] < 2:
            fold["status"] = "INSUFFICIENT_TRAINING_DATA"
            continue

        scaler, _, model = fit_standardized_kmeans(matrix[train])
        if len(set(model.labels_)) != K:
            fold["status"] = "DEGENERATE_TRAINING_CLUSTERS"
            continue
        observed = reference[train]
        confusion = np.zeros((K, K), dtype=int)
        np.add.at(confusion, (model.labels_, observed), 1)
        old, canonical = linear_sum_assignment(confusion, maximize=True)
        mapping = dict(zip(old, canonical))
        prediction = model.predict(scaler.transform(matrix[held]))
        aligned = np.array([mapping[label] for label in prediction])
        matches = int(np.sum(aligned == reference[held]))
        fold.update(train_ari=float(adjusted_rand_score(observed, model.labels_)),
                    held_out_matching_leaves=matches,
                    held_out_agreement=matches / int(held.sum()),
                    held_out_ari=(float(adjusted_rand_score(reference[held], aligned))
                                  if held.sum() > 1 else None))
    return output


def cluster_leaves(leaves, plant_ids):
    """Annotate rows in place and return model, centroids and grouped evaluations."""
    complete = []
    fields = list(FEATURES)
    for row in leaves:
        reasons = [f"{key}={row[f'{key}_status']}" for key in fields
                   if row[key] is None or not np.isfinite(row[key])]
        row.update(cluster_id=None, cluster_status="FEATURES_MISSING" if reasons else "ready",
                   cluster_exclusion_reasons="; ".join(reasons), pc1=None, pc2=None, pc3=None)
        if not reasons:
            complete.append(row)
    matrix = np.array([[r[k] for k in fields] for r in complete], dtype=float).reshape(-1, len(fields))
    groups = np.asarray([r["plant_id"] for r in complete], dtype=str)
    metadata = {
        "status": "ok", "k": K, "seed": SEED, "n_init": N_INIT, "feature_order": fields,
        "total_leaves": len(leaves), "complete_leaves": len(complete),
        "complete_plants": len(set(groups)), "assigned_leaves": 0, "metrics": {},
        "cluster_sizes": {}, "stability": {},
    }
    if len(plant_ids) == 1:
        metadata["status"] = "SINGLE_PLANT"
    elif not _sufficient(matrix) or len(set(groups)) < 2:
        metadata["status"] = "INSUFFICIENT_COMPLETE_CASES"
    centroids, folds = [], []
    if metadata["status"] == "ok":
        with threadpool_limits(limits=1):
            scaler, standardized, model = fit_standardized_kmeans(matrix)
            if len(set(model.labels_)) < K:
                metadata["status"] = "DEGENERATE_CLUSTERS"
            else:
                original_centers = scaler.inverse_transform(model.cluster_centers_)
                height_key = fields.index("height_ratio")
                order = sorted(range(K), key=lambda i: (original_centers[i, height_key],
                                                        *original_centers[i]))
                canonical = {model_id: i for i, model_id in enumerate(order)}
                assigned = np.array([canonical[i] for i in model.labels_])
                pca = PCA(n_components=3, svd_solver="full")
                pcs = pca.fit_transform(standardized)
                for row, label, point in zip(complete, assigned, pcs):
                    row.update(cluster_status="ok", cluster_id=f"C{label+1}",
                               pc1=float(point[0]), pc2=float(point[1]), pc3=float(point[2]))
                counts = Counter(row["cluster_id"] for row in complete)
                for i, model_id in enumerate(order, start=1):
                    name = f"C{i}"
                    members = {r["plant_id"] for r in complete if r["cluster_id"] == name}
                    centroids.append({
                        "cluster_id": name, "leaves": counts[name], "plants": len(members),
                        **dict(zip(fields, original_centers[model_id])),
                    })
                folds = _stability(matrix, groups, assigned, plant_ids)
                evaluated = [r for r in folds if r["status"] == "ok"]
                train_ari = [r["train_ari"] for r in evaluated]
                held_ari = [r["held_out_ari"] for r in evaluated if r["held_out_ari"] is not None]
                held_n = sum(r["held_out_leaves"] for r in evaluated)
                metadata.update(
                    assigned_leaves=len(assigned), cluster_sizes=dict(sorted(counts.items())),
                    metrics={"silhouette": float(silhouette_score(standardized, assigned)),
                             "calinski_harabasz": float(calinski_harabasz_score(standardized, assigned)),
                             "davies_bouldin": float(davies_bouldin_score(standardized, assigned)),
                             "inertia": float(model.inertia_)},
                    scaling_mean=scaler.mean_, scaling_scale=scaler.scale_,
                    constant_features=[k for k, v in zip(fields, scaler.var_) if v == 0],
                    pca_explained_variance_ratio=pca.explained_variance_ratio_,
                    pca_components=pca.components_,
                    stability={
                        "evaluated_plants": len(evaluated), "requested_plants": len(plant_ids),
                        "train_ari_median": float(np.median(train_ari)) if train_ari else None,
                        "train_ari_min": min(train_ari) if train_ari else None,
                        "held_out_ari_median": float(np.median(held_ari)) if held_ari else None,
                        "held_out_leaf_agreement": (
                            sum(r["held_out_matching_leaves"] for r in evaluated) / held_n
                            if held_n else None
                        ),
                        "held_out_plant_mean_agreement": (
                            float(np.mean([r["held_out_agreement"] for r in evaluated])) if evaluated else None
                        ),
                        "evaluated_held_out_leaves": held_n,
                    },
                )
    if metadata["status"] != "ok":
        for row in complete:
            row["cluster_status"] = metadata["status"]
    return metadata, centroids, folds
