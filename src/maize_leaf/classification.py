"""五因素叶姿聚类、语义规则与严格留一株稳定性验证。

Weighted leaf pose clustering with morphology features, relative insertion height, and far-end height difference.

Mathematical components (pure NumPy/SciPy, zero extra dependencies):
  - Primary vector: [stem_leaf_angle_deg, rel_projected_area, pitch_entropy_b12,
    relative_height, far_end_height_diff].
    relative_height is leaf-base height normalized by the observed stem extent.
    far_end_height_diff is z_far - z_base for the leaf surface point farthest from the stem.
  - RobustScaler: x_tilde = (x - median(x)) / IQR(x), fitted strictly on training plants inside LOPO.
  - Classifier: K-means++ (Arthur & Vassilvitskii 2007; k=3, n_init=50, max_iter=500, seed=42).
  - Post-hoc semantic anchoring (never feeds back into clustering):
      S_c = -z_theta (entropy is not assumed to be monotonic with drooping).
      Highest S_c -> convergent (0), middle -> expanded (1), lowest -> drooping (2).
  - Ambiguity score: r_i = d_(1) / d_(2) (ratio of nearest to second-nearest centroid distance).
  - Exact Hubert-Arabie Adjusted Rand Index (ARI), Normalized Mutual Information (NMI),
    Rousseeuw Silhouette score, and Leave-One-Plant-Out (LOPO) stability.
"""
import numpy as np
from scipy.spatial.distance import cdist


CLASS_NAMES = ("convergent", "expanded", "drooping")
CLASS_NAMES_ZH = ("紧缩", "舒展", "下垂")

PRIMARY_FEATURES = [
    "stem_leaf_angle_deg",
    "rel_projected_area",
    "pitch_entropy_b12",
    "relative_height",
    "far_end_height_diff",
]
DEFAULT_HEIGHT_WEIGHT = 1.0  # 初始平方距离权重，尚未标定为生物学常数。
DEFAULT_FAR_HEIGHT_DIFF_WEIGHT = 1.0  # 初始平方距离权重，尚未标定为生物学常数。
DEFAULT_POSE_POLICY = {"compact_max_relative_area": 0.5, "apical_height_span": 0.15,
                       "endpoint_distance_ratio": 2 / 3}
ABLATION_MODELS = {
    "M1_angle": ["stem_leaf_angle_deg"],
    "M2_angle_proj": ["stem_leaf_angle_deg", "rel_projected_area"],
    "M3_morphology": PRIMARY_FEATURES[:3],
    "M4_no_entropy": ["stem_leaf_angle_deg", "rel_projected_area", "relative_height", "far_end_height_diff"],
    "M5_no_far_diff": PRIMARY_FEATURES[:4],
    "M6_primary": PRIMARY_FEATURES,
}


class RobustScaler:
    """按中位数和四分位距缩放特征；退化列回退到标准差。

    Median / IQR scaler resistant to heavy tails; falls back to std if IQR degenerates."""

    def __init__(self, feature_weights=None):
        """初始化稳健缩放器的特征距离权重。"""
        self.median_ = None
        self.scale_ = None
        self.feature_weights = feature_weights
        self.multiplier_ = None

    def fit(self, X):
        """在当前训练样本估计每列的中位数和稳健尺度。"""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not len(X) or not np.all(np.isfinite(X)):
            raise ValueError("Features must be a non-empty finite 2D matrix")
        weights = np.ones(X.shape[1]) if self.feature_weights is None else np.asarray(self.feature_weights, dtype=float)
        if weights.shape != (X.shape[1],) or not np.all(np.isfinite(weights)) or np.any(weights < 0) or not np.any(weights > 0):
            raise ValueError("Feature weights must match columns and be finite, nonnegative, and not all zero")
        self.multiplier_ = np.sqrt(weights)
        self.median_ = np.median(X, axis=0)
        q75 = np.quantile(X, 0.75, axis=0)
        q25 = np.quantile(X, 0.25, axis=0)
        iqr = q75 - q25
        std = np.std(X, axis=0)
        self.scale_ = np.where(iqr > 1e-12, iqr, np.where(std > 1e-12, std, 1.0))
        return self

    def transform(self, X):
        """使用已经拟合的尺度参数变换输入特征。"""
        return (np.asarray(X, dtype=float) - self.median_) / self.scale_ * self.multiplier_

    def fit_transform(self, X):
        """拟合当前数据的缩放参数并返回变换后的特征。"""
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        """将缩放后的特征还原到原始量纲。"""
        unweighted = np.divide(np.asarray(Z, dtype=float), self.multiplier_,
                               out=np.zeros_like(np.asarray(Z, dtype=float)), where=self.multiplier_ > 0)
        return unweighted * self.scale_ + self.median_


def _kmeans_pp_init(X, k, rng):
    """按距离平方抽样初始化聚类中心，降低初始点依赖。

    Arthur & Vassilvitskii (2007) k-means++ seeding."""
    n = len(X)
    centers = np.empty((k, X.shape[1]), dtype=float)
    centers[0] = X[int(rng.integers(n))]
    min_d2 = np.sum((X - centers[0]) ** 2, axis=1)
    for c in range(1, k):
        total = float(np.sum(min_d2))
        if total <= 1e-18:
            centers[c] = X[int(rng.integers(n))]
        else:
            centers[c] = X[int(rng.choice(n, p=min_d2 / total))]
        d2 = np.sum((X - centers[c]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
    return centers


class KMeansModel:
    """保存多次初始化择优的 K-means 原型及预测方法。

    Exact multi-start k-means++ clustering."""

    def __init__(self, n_clusters=3, n_init=50, max_iter=500, tol=1e-6, seed=42):
        """配置聚类数、随机种子、重复初始化与收敛限制。"""
        self.n_clusters = int(n_clusters)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.seed = int(seed)
        self.centers_ = None
        self.inertia_ = float("inf")

    def fit(self, X):
        """执行多次 K-means++ 初始化，保留簇内平方和最小的解。"""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not np.all(np.isfinite(X)) or len(np.unique(X, axis=0)) < self.n_clusters:
            raise ValueError("K-means requires finite features and at least k distinct samples")
        rng = np.random.default_rng(self.seed)

        best_centers = None
        best_inertia = float("inf")

        for _ in range(self.n_init):
            centers = _kmeans_pp_init(X, self.n_clusters, rng)
            for _it in range(self.max_iter):
                d2 = cdist(X, centers, metric="sqeuclidean")
                labels = np.argmin(d2, axis=1)
                new_centers = centers.copy()
                for c in range(self.n_clusters):
                    mask = labels == c
                    if np.any(mask):
                        new_centers[c] = np.mean(X[mask], axis=0)
                    else:
                        far = int(np.argmax(np.min(d2, axis=1)))
                        new_centers[c] = X[far]
                shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
                centers = new_centers
                if shift <= self.tol:
                    break
            d2_final = cdist(X, centers, metric="sqeuclidean")
            inertia = float(np.sum(np.min(d2_final, axis=1)))
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()

        self.centers_ = best_centers
        self.inertia_ = best_inertia
        return self

    def predict(self, X):
        """返回各样本距离最近的聚类中心编号。"""
        d = cdist(np.asarray(X, dtype=float), self.centers_, metric="euclidean")
        return np.argmin(d, axis=1)

    def distance_ratio(self, X):
        """返回最近与次近原型的距离比，接近一表示边界模糊。

        Return r_i = d_(1) / d_(2) in [0, 1] (values near 1 indicate boundary ambiguity)."""
        d = np.sort(cdist(np.asarray(X, dtype=float), self.centers_, metric="euclidean"), axis=1)
        return d[:, 0] / np.maximum(d[:, 1], 1e-12)


def semantic_anchor_order(raw_centroids_scaled, feature_names):
    """按原型形态将无序簇编号映射到紧缩、舒展和下垂。

    Map raw cluster indices -> canonical semantic order (0=convergent, 1=expanded, 2=drooping).

    Uses stem-leaf angle alone to name clusters from small to large basal angle.
    (purely post-hoc semantic relabeling; never changes clustering).
    """
    signs = {
        "stem_leaf_angle_deg": -1.0,
    }
    C = np.asarray(raw_centroids_scaled, dtype=float)
    c_std = np.std(C, axis=0)
    Z_c = (C - np.mean(C, axis=0)) / np.where(c_std > 1e-12, c_std, 1.0)

    scores = np.zeros(len(C), dtype=float)
    for j, fname in enumerate(feature_names):
        if fname in signs:
            scores += signs[fname] * Z_c[:, j]

    order = np.argsort(-scores)
    remap = np.empty(len(C), dtype=int)
    for new_lbl, old_lbl in enumerate(order):
        remap[old_lbl] = new_lbl
    return remap, scores[order]


def adjusted_rand_index(labels_true, labels_pred):
    """计算校正随机一致性的兰德指数 ARI。

    Exact Hubert & Arabie (1985) Adjusted Rand Index."""
    a = np.asarray(labels_true, dtype=int)
    b = np.asarray(labels_pred, dtype=int)
    n = len(a)
    if n < 2:
        return 1.0
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    ka, kb = int(ia.max()) + 1, int(ib.max()) + 1
    cont = np.zeros((ka, kb), dtype=np.int64)
    np.add.at(cont, (ia, ib), 1)

    def _comb2(arr):
        """计算计数中不重复样本对的数量，供 ARI 列联表使用。"""
        return np.sum(arr * (arr - 1) // 2)

    sum_ij = float(_comb2(cont))
    sum_a = float(_comb2(cont.sum(axis=1)))
    sum_b = float(_comb2(cont.sum(axis=0)))
    tot = float(n * (n - 1) // 2)
    expected = (sum_a * sum_b) / max(tot, 1.0)
    max_idx = 0.5 * (sum_a + sum_b)
    denom = max_idx - expected
    if abs(denom) < 1e-12:
        return 1.0 if abs(sum_ij - expected) < 1e-12 else 0.0
    return float((sum_ij - expected) / denom)


def normalized_mutual_info(labels_true, labels_pred):
    """用两组标签的平均熵归一化互信息 NMI。

    Arithmetic-mean Normalized Mutual Information: 2 * I(U; V) / (H(U) + H(V))."""
    a = np.asarray(labels_true, dtype=int)
    b = np.asarray(labels_pred, dtype=int)
    n = float(len(a))
    if n <= 0:
        return 0.0
    _, ia = np.unique(a, return_inverse=True)
    _, ib = np.unique(b, return_inverse=True)
    ka, kb = int(ia.max()) + 1, int(ib.max()) + 1
    cont = np.zeros((ka, kb), dtype=float)
    np.add.at(cont, (ia, ib), 1.0)
    pij = cont / n
    pi = pij.sum(axis=1)
    pj = pij.sum(axis=0)
    hi = -float(np.sum(pi[pi > 0] * np.log(pi[pi > 0])))
    hj = -float(np.sum(pj[pj > 0] * np.log(pj[pj > 0])))
    if hi + hj <= 1e-15:
        return 1.0
    nz = pij > 0
    outer = np.outer(pi, pj)
    mi = float(np.sum(pij[nz] * np.log(pij[nz] / outer[nz])))
    return float(2.0 * mi / (hi + hj))


def silhouette_score(X, labels):
    """按欧氏距离计算全样本平均轮廓系数。

    Exact Rousseeuw (1987) mean Silhouette coefficient over Euclidean distances."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels, dtype=int)
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(X) <= len(uniq):
        return float("nan")
    D = cdist(X, X, metric="euclidean")
    n = len(X)
    s = np.zeros(n, dtype=float)
    for i in range(n):
        same = labels == labels[i]
        cnt_same = int(np.sum(same))
        if cnt_same <= 1:
            s[i] = 0.0
            continue
        a_i = float(np.sum(D[i, same])) / (cnt_same - 1)
        b_i = min(float(np.mean(D[i, labels == u])) for u in uniq if u != labels[i])
        denom = max(a_i, b_i)
        s[i] = (b_i - a_i) / denom if denom > 1e-12 else 0.0
    return float(np.mean(s))


def fit_and_anchor(X_raw, feature_names, seed=42, feature_weights=None):
    """拟合稳健缩放器和三个聚类原型，并统一语义编号。

    Fit RobustScaler + K-means++ (k=3) and remap cluster IDs to canonical semantic order."""
    if np.asarray(X_raw).ndim != 2 or np.asarray(X_raw).shape[1] != len(feature_names):
        raise ValueError("Feature names must match matrix columns")
    scaler = RobustScaler(feature_weights).fit(X_raw)
    Z = scaler.transform(X_raw)
    model = KMeansModel(n_clusters=3, n_init=50, max_iter=500, seed=seed).fit(Z)

    remap, _ = semantic_anchor_order(model.centers_, feature_names)
    inv_order = np.argsort(remap)
    model.centers_ = model.centers_[inv_order]
    labels = model.predict(Z)
    return scaler, model, Z, labels


def contextual_predict(model, Z, X_raw, plant_ids, feature_names, policy):
    """结合顶部小叶与原型距离优势解释类别，不要求每株具备三类。

    Apply explicit expert-guided endpoint semantics to globally learned prototypes.

    Compact candidates must be small apical leaves, rather than all low-angle leaves.
    Endpoint assignments require a clear distance margin; transitions are expanded.
    There are no per-plant quotas and no requirement that any class occur in a plant.
    """
    raw = model.predict(Z)
    distances = cdist(Z, model.centers_)
    ratio = np.sort(distances, axis=1)[:, 0] / np.maximum(np.sort(distances, axis=1)[:, 1], 1e-12)
    if not policy:
        return raw, raw, ratio
    if not set(PRIMARY_FEATURES).issubset(feature_names):
        raise ValueError("Contextual policy requires all four primary features")
    area = np.asarray(X_raw)[:, feature_names.index("rel_projected_area")]
    height = np.asarray(X_raw)[:, feature_names.index("relative_height")]
    plants = np.asarray(plant_ids)
    top = np.empty(len(plants))
    for plant in set(plants):
        mask = plants == plant
        top[mask] = np.max(height[mask])
    eligible = ((area <= policy["compact_max_relative_area"])
                & (top - height <= policy["apical_height_span"]))
    labels = np.ones(len(raw), dtype=int)
    compact_margin = distances[:, 0] / np.maximum(np.min(distances[:, 1:], axis=1), 1e-12)
    droop_margin = distances[:, 2] / np.maximum(distances[:, 1], 1e-12)
    labels[(raw == 0) & eligible & (compact_margin <= policy["endpoint_distance_ratio"])] = 0
    labels[(raw == 2) & (droop_margin <= policy["endpoint_distance_ratio"])] = 2
    return labels, raw, ratio


def lopo_validate(X_raw, plant_ids, feature_names, seed=42, n_perm=200, feature_weights=None,
                  decision_policy=None):
    """逐株留出并仅用训练株拟合缩放器和聚类，评估标签稳定性。

    Strict Leave-One-Plant-Out validation + permutation baseline for K-means++ (k=3)."""
    X_raw = np.asarray(X_raw, dtype=float)
    plants = np.asarray(plant_ids)
    uniq_plants = sorted(set(plants))
    if len(plants) != len(X_raw) or len(uniq_plants) < 2:
        raise ValueError("LOPO requires matching plant IDs from at least two plants")

    full_scaler, full_model, Z_full, full_labels = fit_and_anchor(X_raw, feature_names, seed=seed, feature_weights=feature_weights)
    full_labels, raw_labels, r_ratio = contextual_predict(full_model, Z_full, X_raw, plants, feature_names, decision_policy)
    full_sil = silhouette_score(Z_full, full_labels)

    fold_aris = []
    fold_nmis = []
    fold_drifts = []
    fold_switch_rates = []
    oof_labels = np.empty(len(X_raw), dtype=int)

    for p in uniq_plants:
        tr = plants != p
        te = plants == p
        sc_f, mdl_f, _, _ = fit_and_anchor(X_raw[tr], feature_names, seed=seed, feature_weights=feature_weights)
        Z_all_f = sc_f.transform(X_raw)
        pred_all, _, _ = contextual_predict(mdl_f, Z_all_f, X_raw, plants, feature_names, decision_policy)
        oof_labels[te] = pred_all[te]

        fold_aris.append(adjusted_rand_index(full_labels, pred_all))
        fold_nmis.append(normalized_mutual_info(full_labels, pred_all))
        fold_switch_rates.append(float(np.mean(pred_all != full_labels)))
        c_raw_f = sc_f.inverse_transform(mdl_f.centers_)
        c_in_full = full_scaler.transform(c_raw_f)
        fold_drifts.append(float(np.mean(np.linalg.norm(c_in_full - full_model.centers_, axis=1))))

    oof_ari = adjusted_rand_index(full_labels, oof_labels)
    oof_nmi = normalized_mutual_info(full_labels, oof_labels)
    oof_switch = float(np.mean(oof_labels != full_labels))

    rng = np.random.default_rng(seed)
    perm_aris = [adjusted_rand_index(full_labels, rng.permutation(full_labels)) for _ in range(n_perm)]
    perm_p = float((1 + np.sum(np.asarray(perm_aris) >= oof_ari)) / (n_perm + 1))

    return {
        "method": "kmeans",
        "features": list(feature_names),
        "full_labels": full_labels,
        "raw_labels": raw_labels,
        "decision_policy": decision_policy,
        "oof_labels": oof_labels,
        "distance_ratio": r_ratio,
        "Z_full": Z_full,
        "centers_scaled": full_model.centers_,
        "centers_raw": full_scaler.inverse_transform(full_model.centers_),
        "scaler_median": full_scaler.median_,
        "scaler_iqr": full_scaler.scale_,
        "feature_weights": full_scaler.multiplier_ ** 2,
        "metrics": {
            "silhouette": full_sil,
            "lopo_fold_ari_mean": float(np.mean(fold_aris)),
            "lopo_fold_ari_std": float(np.std(fold_aris)),
            "lopo_fold_ari_min": float(np.min(fold_aris)),
            "lopo_fold_nmi_mean": float(np.mean(fold_nmis)),
            "lopo_oof_ari": oof_ari,
            "lopo_oof_nmi": oof_nmi,
            "lopo_oof_switch_rate_pct": 100.0 * oof_switch,
            "lopo_fold_switch_rate_mean_pct": 100.0 * float(np.mean(fold_switch_rates)),
            "centroid_drift_mean": float(np.mean(fold_drifts)),
            "centroid_drift_max": float(np.max(fold_drifts)),
            "random_perm_ari_mean": float(np.mean(perm_aris)),
            "random_perm_ari_p95": float(np.quantile(perm_aris, 0.95)),
            "h2_permutation_p_value": perm_p,
            "ambiguous_rate_r075_pct": 100.0 * float(np.mean(r_ratio > 0.75)),
            "ambiguous_rate_r080_pct": 100.0 * float(np.mean(r_ratio > 0.80)),
            "ambiguous_rate_r085_pct": 100.0 * float(np.mean(r_ratio > 0.85)),
            "class_counts": {CLASS_NAMES[c]: int(np.sum(full_labels == c)) for c in range(3)},
        },
    }
