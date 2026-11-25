"""Pure-morphology 3-class maize leaf pose classification & LOPO validation (V2.0 Stage E).

Mathematical components (pure NumPy/SciPy, zero extra dependencies):
  - RobustScaler: x_tilde = (x - median(x)) / IQR(x), fitted strictly on training plants inside LOPO.
  - Primary classifier: K-means++ (Arthur & Vassilvitskii 2007; k=3, n_init=50, max_iter=500, seed=42).
  - Soft-classification comparators:
      * Bezdek (1981) Fuzzy C-Means (FCM, fuzzifier m=2.0, multi-start)
      * EM Gaussian Mixture Model (GMM, full/diag regularized covariance, multi-start)
  - Post-hoc semantic anchoring (never feeds back into clustering):
      S_c = -z_theta + z_g - z_d - z_K  (computed from standardized cluster centroids)
      Highest S_c -> convergent (0), middle -> expanded (1), lowest -> drooping (2).
  - Ambiguity score: r_i = d_(1) / d_(2) (ratio of nearest to second-nearest centroid distance).
  - Exact Hubert-Arabie Adjusted Rand Index (ARI), Normalized Mutual Information (NMI),
    Rousseeuw Silhouette score, and strict Leave-One-Plant-Out (LOPO) stability & M1..M5 ablation.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp


CLASS_NAMES = ("convergent", "expanded", "drooping")
CLASS_NAMES_ZH = ("聚拢", "伸展", "下垂")

# Ablation feature hierarchy (pure single-leaf morphology; height & canopy occlusion strictly excluded!)
# Four physical dimensions in M4_primary:
#   1) Basal emergence angle: stem_leaf_angle_deg (theta)
#   2) Horizontal projected area (efficiency + within-plant scale): projection_ratio (rho), rel_projected_area (Ah_rel)
#   3) Vertical terminal pose: tip_gain (g), droop_ratio (d)
#   4) Longitudinal orientation complexity: pitch_entropy_b12 (H_pit)
ABLATION_MODELS = {
    "M1_angle": ["stem_leaf_angle_deg"],
    "M2_angle_proj": ["stem_leaf_angle_deg", "projection_ratio", "rel_projected_area"],
    "M3_angle_proj_pose": [
        "stem_leaf_angle_deg", "projection_ratio", "rel_projected_area", "tip_gain", "droop_ratio",
    ],
    "M4_primary": [
        "stem_leaf_angle_deg", "projection_ratio", "rel_projected_area",
        "tip_gain", "droop_ratio", "pitch_entropy_b12",
    ],
    "M5_alt_normal_entropy": [
        "stem_leaf_angle_deg", "projection_ratio", "rel_projected_area",
        "tip_gain", "droop_ratio", "normal_entropy_b24",
    ],
}


class RobustScaler:
    """Median / IQR scaler resistant to heavy tails; falls back to std if IQR degenerates."""

    def __init__(self):
        self.median_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.median_ = np.median(X, axis=0)
        q75 = np.quantile(X, 0.75, axis=0)
        q25 = np.quantile(X, 0.25, axis=0)
        iqr = q75 - q25
        std = np.std(X, axis=0)
        self.scale_ = np.where(iqr > 1e-12, iqr, np.where(std > 1e-12, std, 1.0))
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.median_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, Z):
        return np.asarray(Z, dtype=float) * self.scale_ + self.median_


def _kmeans_pp_init(X, k, rng, weights=None):
    """Arthur & Vassilvitskii (2007) k-means++ seeding."""
    n = len(X)
    w = np.ones(n, dtype=float) if weights is None else np.maximum(np.asarray(weights, dtype=float), 1e-12)
    centers = np.empty((k, X.shape[1]), dtype=float)
    p0 = w / np.sum(w)
    centers[0] = X[int(rng.choice(n, p=p0))]
    min_d2 = np.sum((X - centers[0]) ** 2, axis=1)
    for c in range(1, k):
        probs = w * min_d2
        total = float(np.sum(probs))
        if total <= 1e-18:
            centers[c] = X[int(rng.integers(n))]
        else:
            centers[c] = X[int(rng.choice(n, p=probs / total))]
        d2 = np.sum((X - centers[c]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
    return centers


class KMeansModel:
    """Exact multi-start k-means++ with optional sample weights."""

    def __init__(self, n_clusters=3, n_init=50, max_iter=500, tol=1e-6, seed=42):
        self.n_clusters = int(n_clusters)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.seed = int(seed)
        self.centers_ = None
        self.inertia_ = float("inf")

    def fit(self, X, sample_weight=None):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        w = np.ones(n, dtype=float) if sample_weight is None else np.maximum(np.asarray(sample_weight, dtype=float), 1e-12)
        rng = np.random.default_rng(self.seed)

        best_centers = None
        best_inertia = float("inf")

        for _ in range(self.n_init):
            centers = _kmeans_pp_init(X, self.n_clusters, rng, weights=w)
            for _it in range(self.max_iter):
                d2 = cdist(X, centers, metric="sqeuclidean")
                labels = np.argmin(d2, axis=1)
                new_centers = centers.copy()
                for c in range(self.n_clusters):
                    mask = labels == c
                    if np.any(mask):
                        wc = w[mask]
                        new_centers[c] = np.sum(X[mask] * wc[:, None], axis=0) / np.sum(wc)
                    else:
                        # Re-seed empty cluster to farthest point
                        far = int(np.argmax(np.min(d2, axis=1)))
                        new_centers[c] = X[far]
                shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
                centers = new_centers
                if shift <= self.tol:
                    break
            d2_final = cdist(X, centers, metric="sqeuclidean")
            inertia = float(np.sum(w * np.min(d2_final, axis=1)))
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers.copy()

        self.centers_ = best_centers
        self.inertia_ = best_inertia
        return self

    def predict(self, X):
        d = cdist(np.asarray(X, dtype=float), self.centers_, metric="euclidean")
        return np.argmin(d, axis=1)

    def distance_ratio(self, X):
        """Return r_i = d_(1) / d_(2) in [0, 1] (values near 1 indicate boundary ambiguity)."""
        d = np.sort(cdist(np.asarray(X, dtype=float), self.centers_, metric="euclidean"), axis=1)
        return d[:, 0] / np.maximum(d[:, 1], 1e-12)

    def predict_proba(self, X):
        """Inverse-distance soft confidence proxy for reporting consistency."""
        d2 = cdist(np.asarray(X, dtype=float), self.centers_, metric="sqeuclidean")
        inv = 1.0 / np.maximum(d2, 1e-12)
        return inv / np.sum(inv, axis=1, keepdims=True)


class FuzzyCMeansModel:
    """Bezdek (1981) Fuzzy C-Means clustering (fuzzifier m=2.0) with optional sample weights."""

    def __init__(self, n_clusters=3, m=2.0, n_init=30, max_iter=500, tol=1e-6, seed=42):
        self.n_clusters = int(n_clusters)
        self.m = float(m)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.seed = int(seed)
        self.centers_ = None
        self.objective_ = float("inf")

    def _memberships(self, X, centers):
        d = cdist(X, centers, metric="euclidean")
        power = 2.0 / (self.m - 1.0)
        U = np.zeros_like(d)
        zero_hit = d < 1e-12
        any_zero = np.any(zero_hit, axis=1)
        if np.any(any_zero):
            U[any_zero] = zero_hit[any_zero] / np.sum(zero_hit[any_zero], axis=1, keepdims=True)
        nz = ~any_zero
        if np.any(nz):
            d_nz = d[nz]
            # u_ik = (d_ik^-power) / sum_j (d_ij^-power)
            inv = d_nz ** (-power)
            U[nz] = inv / np.sum(inv, axis=1, keepdims=True)
        return U

    def fit(self, X, sample_weight=None):
        X = np.asarray(X, dtype=float)
        n = len(X)
        w = np.ones(n, dtype=float) if sample_weight is None else np.maximum(np.asarray(sample_weight, dtype=float), 1e-12)
        rng = np.random.default_rng(self.seed)

        best_centers = None
        best_obj = float("inf")
        for _ in range(self.n_init):
            centers = _kmeans_pp_init(X, self.n_clusters, rng, weights=w)
            for _it in range(self.max_iter):
                U = self._memberships(X, centers)
                Um = (U ** self.m) * w[:, None]
                denom = np.maximum(np.sum(Um, axis=0), 1e-12)
                new_centers = (Um.T @ X) / denom[:, None]
                shift = float(np.max(np.linalg.norm(new_centers - centers, axis=1)))
                centers = new_centers
                if shift <= self.tol:
                    break
            U = self._memberships(X, centers)
            d2 = cdist(X, centers, metric="sqeuclidean")
            obj = float(np.sum(w[:, None] * (U ** self.m) * d2))
            if obj < best_obj:
                best_obj = obj
                best_centers = centers.copy()

        self.centers_ = best_centers
        self.objective_ = best_obj
        return self

    def predict_proba(self, X):
        return self._memberships(np.asarray(X, dtype=float), self.centers_)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


class DiagonalGMMModel:
    """Regularized EM Gaussian Mixture Model (diagonal covariance for small-n stability)."""

    def __init__(self, n_clusters=3, n_init=30, max_iter=300, tol=1e-5, reg_covar=1e-3, seed=42):
        self.n_clusters = int(n_clusters)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.reg_covar = float(reg_covar)
        self.seed = int(seed)
        self.centers_ = None
        self.vars_ = None
        self.weights_ = None
        self.log_likelihood_ = -float("inf")

    def _log_prob(self, X, means, variances, pi):
        d = X.shape[1]
        log_p = np.empty((len(X), self.n_clusters), dtype=float)
        for c in range(self.n_clusters):
            v = np.maximum(variances[c], self.reg_covar)
            diff = X - means[c]
            maha = np.sum((diff * diff) / v[None, :], axis=1)
            log_det = float(np.sum(np.log(v)))
            log_p[:, c] = np.log(max(pi[c], 1e-12)) - 0.5 * (d * np.log(2.0 * np.pi) + log_det + maha)
        return log_p

    def fit(self, X, sample_weight=None):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        w = np.ones(n, dtype=float) if sample_weight is None else np.maximum(np.asarray(sample_weight, dtype=float), 1e-12)
        w_sum = float(np.sum(w))
        rng = np.random.default_rng(self.seed)

        best_ll = -float("inf")
        best_params = None
        global_var = np.maximum(np.var(X, axis=0), self.reg_covar)

        for _ in range(self.n_init):
            means = _kmeans_pp_init(X, self.n_clusters, rng, weights=w)
            variances = np.tile(global_var, (self.n_clusters, 1))
            pi = np.full(self.n_clusters, 1.0 / self.n_clusters)
            prev_ll = -float("inf")

            for _it in range(self.max_iter):
                log_p = self._log_prob(X, means, variances, pi)
                log_norm = logsumexp(log_p, axis=1)
                ll = float(np.sum(w * log_norm))
                resp = np.exp(log_p - log_norm[:, None])

                rw = resp * w[:, None]
                nk = np.maximum(np.sum(rw, axis=0), 1e-8)
                pi = nk / w_sum
                means = (rw.T @ X) / nk[:, None]
                for c in range(self.n_clusters):
                    diff = X - means[c]
                    variances[c] = np.sum(rw[:, c][:, None] * (diff * diff), axis=0) / nk[c] + self.reg_covar

                if abs(ll - prev_ll) <= self.tol * max(1.0, abs(ll)):
                    prev_ll = ll
                    break
                prev_ll = ll

            if prev_ll > best_ll:
                best_ll = prev_ll
                best_params = (means.copy(), variances.copy(), pi.copy())

        self.centers_, self.vars_, self.weights_ = best_params
        self.log_likelihood_ = best_ll
        return self

    def predict_proba(self, X):
        log_p = self._log_prob(np.asarray(X, dtype=float), self.centers_, self.vars_, self.weights_)
        log_norm = logsumexp(log_p, axis=1, keepdims=True)
        return np.exp(log_p - log_norm)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def semantic_anchor_order(raw_centroids_scaled, feature_names):
    """Map raw cluster indices -> canonical semantic order (0=convergent, 1=expanded, 2=drooping).

    Uses pose score S_c = -z_theta + z_g - z_d - z_K over whichever core pose features are
    present in feature_names (purely post-hoc semantic relabeling; never changes clustering).
    """
    signs = {
        "stem_leaf_angle_deg": -1.0,
        "tip_gain": +1.0,
        "droop_ratio": -1.0,
        "pitch_entropy_b12": -0.35,
        "normal_entropy_b24": -0.35,
        "midrib_curvature_total": -1.0,
    }
    C = np.asarray(raw_centroids_scaled, dtype=float)
    # Standardize centroids across the k clusters for equal-scale semantic scoring
    c_std = np.std(C, axis=0)
    Z_c = (C - np.mean(C, axis=0)) / np.where(c_std > 1e-12, c_std, 1.0)

    scores = np.zeros(len(C), dtype=float)
    for j, fname in enumerate(feature_names):
        if fname in signs:
            scores += signs[fname] * Z_c[:, j]

    # Highest score -> 0 (convergent), middle -> 1 (expanded), lowest -> 2 (drooping)
    order = np.argsort(-scores)
    remap = np.empty(len(C), dtype=int)
    for new_lbl, old_lbl in enumerate(order):
        remap[old_lbl] = new_lbl
    return remap, scores[order]


def adjusted_rand_index(labels_true, labels_pred):
    """Exact Hubert & Arabie (1985) Adjusted Rand Index."""
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
    """Arithmetic-mean Normalized Mutual Information: 2 * I(U; V) / (H(U) + H(V))."""
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
    """Exact Rousseeuw (1987) mean Silhouette coefficient over Euclidean distances."""
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


def fit_and_anchor(X_raw, feature_names, method="kmeans", sample_weight=None, seed=42):
    """Fit RobustScaler + clustering model and remap cluster IDs to canonical semantic order."""
    scaler = RobustScaler().fit(X_raw)
    Z = scaler.transform(X_raw)
    if method == "kmeans":
        model = KMeansModel(n_clusters=3, n_init=50, max_iter=500, seed=seed).fit(Z, sample_weight=sample_weight)
    elif method == "fcm":
        model = FuzzyCMeansModel(n_clusters=3, m=2.0, n_init=30, max_iter=500, seed=seed).fit(Z, sample_weight=sample_weight)
    elif method == "gmm":
        model = DiagonalGMMModel(n_clusters=3, n_init=30, max_iter=300, seed=seed).fit(Z, sample_weight=sample_weight)
    else:
        raise ValueError(f"Unknown method: {method}")

    remap, _ = semantic_anchor_order(model.centers_, feature_names)
    inv_order = np.argsort(remap)
    # Reorder model centroids/parameters in place so label 0=convergent, 1=expanded, 2=drooping
    model.centers_ = model.centers_[inv_order]
    if method == "gmm":
        model.vars_ = model.vars_[inv_order]
        model.weights_ = model.weights_[inv_order]

    labels = model.predict(Z)
    proba = model.predict_proba(Z)
    return scaler, model, Z, labels, proba


def lopo_validate(X_raw, plant_ids, feature_names, method="kmeans", sample_weight=None,
                  seed=42, n_perm=200):
    """Strict Leave-One-Plant-Out validation + permutation baseline for Hypothesis H2."""
    X_raw = np.asarray(X_raw, dtype=float)
    plants = np.asarray(plant_ids)
    uniq_plants = sorted(set(plants))
    w_all = None if sample_weight is None else np.asarray(sample_weight, dtype=float)

    full_scaler, full_model, Z_full, full_labels, full_proba = fit_and_anchor(
        X_raw, feature_names, method=method, sample_weight=w_all, seed=seed
    )
    full_sil = silhouette_score(Z_full, full_labels)

    # Distance ratio ambiguity on full fit (in scaled feature space)
    d_sorted = np.sort(cdist(Z_full, full_model.centers_, metric="euclidean"), axis=1)
    r_ratio = d_sorted[:, 0] / np.maximum(d_sorted[:, 1], 1e-12)

    fold_aris = []
    fold_nmis = []
    fold_drifts = []
    fold_switch_rates = []
    oof_labels = np.empty(len(X_raw), dtype=int)

    for p in uniq_plants:
        tr = plants != p
        te = plants == p
        w_tr = None if w_all is None else w_all[tr]
        sc_f, mdl_f, _, _, _ = fit_and_anchor(
            X_raw[tr], feature_names, method=method, sample_weight=w_tr, seed=seed
        )
        # Predict all 90 leaves and specifically out-of-sample held-out plant leaves
        Z_all_f = sc_f.transform(X_raw)
        pred_all = mdl_f.predict(Z_all_f)
        oof_labels[te] = pred_all[te]

        fold_aris.append(adjusted_rand_index(full_labels, pred_all))
        fold_nmis.append(normalized_mutual_info(full_labels, pred_all))
        fold_switch_rates.append(float(np.mean(pred_all != full_labels)))
        # Centroid drift measured in full-fit scaled space for direct comparability
        c_raw_f = sc_f.inverse_transform(mdl_f.centers_)
        c_in_full = full_scaler.transform(c_raw_f)
        fold_drifts.append(float(np.mean(np.linalg.norm(c_in_full - full_model.centers_, axis=1))))

    oof_ari = adjusted_rand_index(full_labels, oof_labels)
    oof_nmi = normalized_mutual_info(full_labels, oof_labels)
    oof_switch = float(np.mean(oof_labels != full_labels))

    # Random permutation null baseline for H2: ARI_LOPO > ARI_random
    rng = np.random.default_rng(seed)
    perm_aris = [adjusted_rand_index(full_labels, rng.permutation(full_labels)) for _ in range(n_perm)]
    perm_p = float((1 + np.sum(np.asarray(perm_aris) >= oof_ari)) / (n_perm + 1))

    return {
        "method": method,
        "features": list(feature_names),
        "weighted": w_all is not None,
        "full_labels": full_labels,
        "oof_labels": oof_labels,
        "full_proba": full_proba,
        "distance_ratio": r_ratio,
        "Z_full": Z_full,
        "centers_scaled": full_model.centers_,
        "centers_raw": full_scaler.inverse_transform(full_model.centers_),
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
