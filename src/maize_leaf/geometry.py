"""点云去噪、茎秆与中脉拟合及无扭转移动标架。

Differential geometry primitives for maize stems and leaves.

References:
- Eilers & Marx (1996) Statistical Science: Penalized B-splines (P-splines)
- Wahba (1990): Generalized Cross-Validation (GCV) for smoothing parameter selection
- Hastie & Stuetzle (1989) JASA: Principal Curves (iterative projection + smoothing)
- Piegl & Tiller (1997) The NURBS Book: Newton orthogonal point projection onto B-splines
- Bishop (1975); Wang et al. (2008) ACM TOG: Double Reflection Rotation Minimizing Frames
"""
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import lsq_linear, minimize_scalar
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


def unit(v):
    """按最后一维归一化向量，并用小量避免零向量除零。"""
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def _euclidean_clusters(points, normals=None):
    """精确去重后进行单次欧氏连通分量筛选，同时保留断片编号。

    Exact deduplication + single-pass Euclidean cluster filtering returning fragment labels.

    Uses the 1%-99% quantile bounding-box diagonal D_rob so far stray outliers cannot inflate
    the connection radius r = max(5.0 * spacing, 0.02 * D_rob) and absorb nearby detached noise.
    """
    _, uniq_idx = np.unique(points, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    p = points[uniq_idx]
    nrm = normals[uniq_idx] if normals is not None else None

    tree = cKDTree(p)
    d2 = tree.query(p, k=min(2, len(p)))[0][:, -1]
    spacing = float(np.median(d2))
    rob_diam = float(np.linalg.norm(np.quantile(p, 0.99, axis=0) - np.quantile(p, 0.01, axis=0)))
    radius = max(spacing * 5.0, rob_diam * 0.02)

    g = tree.sparse_distance_matrix(tree, max_distance=radius, output_type="coo_matrix")
    _, lbl = connected_components(g)
    sizes = np.bincount(lbl)
    main_lbl = int(np.argmax(sizes))
    min_size = max(20, int(0.01 * len(p)))
    keep = (lbl == main_lbl) | (sizes[lbl] >= min_size)
    _, frag_lbl = np.unique(lbl[keep], return_inverse=True)
    return p[keep], (nrm[keep] if nrm is not None else None), frag_lbl


def filter_noise_clusters(points, normals=None):
    """保留最大连通簇和足够大的断片，去除小型离散误标簇。

    Euclidean cluster filtering (Rusu & Cousins 2011) to remove tiny detached mislabeled fragments.

    Retains the largest connected component plus coherent fragments (>= max(20, 1% of points)),
    preventing small scanner artifacts near the stem from hijacking leaf root selection.
    """
    p, nrm, _ = _euclidean_clusters(points, normals)
    return p, nrm


def sor_filter(points, normals=None, k=8, n_sigma=3.5):
    """按近邻平均距离及稳健 MAD 阈值筛除单个断片的离群点。

    Robust MAD Statistical Outlier Removal (Rusu et al. 2008; calibrated k=8, n_sigma=3.5)."""
    if len(points) < 20:
        return points, normals
    k_eff = min(k, len(points) - 1)
    tree = cKDTree(points)
    dists = tree.query(points, k=k_eff + 1)[0][:, 1:]
    mean_d = np.mean(dists, axis=1)
    med = float(np.median(mean_d))
    mad = float(np.median(np.abs(mean_d - med)))
    sigma_rob = 1.4826 * mad
    if sigma_rob <= 1e-12:
        sigma_rob = float(np.std(mean_d))
    thresh = med + n_sigma * sigma_rob
    keep = mean_d <= thresh
    if int(np.sum(keep)) < max(15, int(0.75 * len(points))):
        return points, normals
    return points[keep], (normals[keep] if normals is not None else None)


def clean_leaf_points(points, normals=None):
    """先筛连通簇再逐断片去噪，避免稀疏叶尖与密集叶身直接比较。

    Dual-stage denoising: single-pass Euclidean cluster filtering + per-fragment robust MAD SOR."""
    p, nrm, labels = _euclidean_clusters(points, normals)
    # 在各保留断片内部估计噪声，不再收缩连接半径：
    # 避免将稀疏末端断片与密集叶身直接比较。
    keep = np.zeros(len(p), dtype=bool)
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        part, _ = sor_filter(p[idx])
        keep[idx] = cKDTree(part).query(p[idx])[0] < 1e-12
    return p[keep], (nrm[keep] if nrm is not None else None)


def fit_pspline_gcv(t, y, n_basis=10, degree=3, weights=None, lam_bounds=(1e-3, 1e4), bounded=True):
    """拟合带二阶差分惩罚的 B 样条，通过 GCV 自动选择平滑强度。

    Penalized B-spline (Eilers & Marx 1996) with automatic GCV smoothing (Wahba 1990)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    is_1d = y.ndim == 1
    Y = y[:, None] if is_1d else y

    k = min(degree, len(t) - 1)
    nc = min(n_basis, max(k + 1, len(t) // 3))
    knots = np.r_[
        np.repeat(t[0], k + 1),
        np.linspace(t[0], t[-1], nc - k + 1)[1:-1],
        np.repeat(t[-1], k + 1),
    ]
    B = BSpline.design_matrix(t, knots, k).toarray()
    D = np.diff(np.eye(nc), n=2, axis=0)
    P = D.T @ D

    w = np.ones(len(t)) if weights is None else np.asarray(weights, dtype=float)
    sw = np.sqrt(w)
    Bw = B * sw[:, None]
    Yw = Y * sw[:, None]
    BtB = Bw.T @ Bw
    BtY = Bw.T @ Yw
    n = len(t)

    def gcv(log_lam):
        """计算给定对数平滑参数下的广义交叉验证目标。"""
        lam = 10.0 ** log_lam
        A = BtB + lam * P + 1e-12 * np.eye(nc)
        Ainv_Bt = np.linalg.solve(A, Bw.T)
        C = Ainv_Bt @ Yw
        resid = Yw - Bw @ C
        rss = float(np.mean(resid ** 2))
        edf = float(np.sum(Bw * Ainv_Bt.T))
        denom = max(1.0 - edf / n, 1e-4)
        return rss / (denom ** 2)

    res = minimize_scalar(gcv, bounds=np.log10(lam_bounds), method="bounded")
    lam_opt = 10.0 ** res.x
    if bounded:
        A_stack = np.vstack([Bw, np.sqrt(lam_opt) * D])
        cols_c = []
        for j in range(Y.shape[1]):
            col = Y[:, j]
            span = max(float(np.ptp(col)), 1e-5)
            lo, hi = float(col.min() - 0.02 * span), float(col.max() + 0.02 * span)
            rhs = np.r_[Yw[:, j], np.zeros(len(D))]
            cols_c.append(lsq_linear(A_stack, rhs, bounds=(lo, hi), tol=1e-10).x)
        C = np.asarray(cols_c).T
    else:
        C = np.linalg.solve(BtB + lam_opt * P + 1e-12 * np.eye(nc), BtY)
    spline = BSpline(knots, C[:, 0] if is_1d else C, k)
    return spline, float(lam_opt)


def spline_dict(s):
    """将样条节点、系数和阶数转为可保存的字典。"""
    return {"knots": s.t.tolist(), "coefficients": s.c.tolist(), "degree": int(s.k)}


def fit_stem_centerline(points):
    """沿高度分箱取茎秆中位点，并用平滑样条拟合中心线。

    Z-binned median stem centerline fitted with GCV P-spline."""
    z = points[:, 2]
    nb = min(50, max(8, len(points) // 40))
    edges = np.linspace(z.min(), z.max(), nb + 1)
    nodes = np.asarray([
        np.median(points[(z >= a) & (z <= b)], axis=0)
        for a, b in zip(edges[:-1], edges[1:])
        if np.sum((z >= a) & (z <= b)) >= 3
    ])
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))]
    curve, lam = fit_pspline_gcv(s, nodes)
    ss = np.linspace(0.0, s[-1], 160)
    sampled = curve(ss)
    return sampled, {"s": ss.tolist(), "xyz": sampled.tolist(), "spline": spline_dict(curve), "gcv_lambda": lam}


def geodesic_order(points, stem, k_nn=12):
    """从靠近茎秆的叶基出发，用近邻图最短路近似叶片测地次序。

    Approximate intrinsic leaf geodesic coordinate from stem attachment via kNN + Dijkstra."""
    tree = cKDTree(points)
    dist, idx = tree.query(points, k=min(k_nn + 1, len(points)))
    spacing = float(np.median(dist[:, 1]))
    rows = np.repeat(np.arange(len(points)), dist.shape[1] - 1)
    cols, weights = idx[:, 1:].ravel(), dist[:, 1:].ravel()
    graph = coo_matrix((weights, (rows, cols)), shape=(len(points), len(points))).tocsr()
    graph = graph.maximum(graph.T)

    ncomp, labels = connected_components(graph)
    if ncomp > 1:
        graph = graph.tolil()
        active = labels == np.argmax(np.bincount(labels))
        while not np.all(active):
            ia, ib = np.flatnonzero(active), np.flatnonzero(~active)
            d, j = cKDTree(points[ia]).query(points[ib])
            k = int(np.argmin(d))
            a, b = int(ia[j[k]]), int(ib[k])
            graph[a, b] = graph[b, a] = float(d[k])
            active |= labels == labels[b]
        graph = graph.tocsr()

    root = int(np.argmin(cKDTree(stem).query(points)[0]))
    geo = dijkstra(graph, directed=False, indices=root)
    return geo, spacing


def reparameterize_arc_length(curve, n_samples=400):
    """以采样弦长近似弧长，重新拟合曲线参数。

    Approximate arc-length reparameterization by sampled chord lengths and spline refit."""
    t0, t1 = float(curve.t[curve.k]), float(curve.t[-curve.k - 1])
    ts = np.linspace(t0, t1, n_samples)
    pts = curve(ts)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    arc_curve, _ = fit_pspline_gcv(s, pts, n_basis=min(18, len(curve.c) + 2), lam_bounds=(1e-8, 1e-1))
    return arc_curve, float(s[-1])


def project_points_to_curve(points, curve, s_max, n_init=300, n_newton=5):
    """以离散最近点初始化，再有限次牛顿迭代求曲线正交投影。

    Approximate nearest-curve projection with sampled initialization and finite Newton steps."""
    grid_s = np.linspace(0.0, s_max, n_init)
    grid_p = curve(grid_s)
    _, idx = cKDTree(grid_p).query(points)
    s = grid_s[idx].copy()

    d1 = curve.derivative(1)
    d2 = curve.derivative(2)
    for _ in range(n_newton):
        c = curve(s)
        c1 = d1(s)
        c2 = d2(s)
        diff = c - points
        f = np.sum(diff * c1, axis=1)
        fp = np.sum(c1 * c1, axis=1) + np.sum(diff * c2, axis=1)
        step = np.where(np.abs(fp) > 1e-12, f / fp, 0.0)
        s = np.clip(s - np.clip(step, -0.05 * s_max, 0.05 * s_max), 0.0, s_max)
    return s


def _farthest_point_sample(points, m, seed=2026):
    """确定性最远点采样，为骨架提供分布均匀的初始节点。

    Deterministic greedy Farthest Point Sampling (FPS) for uniform skeleton seeding."""
    n = len(points)
    if n <= m:
        return points.copy()
    rng = np.random.default_rng(seed)
    sel = np.empty(m, dtype=int)
    sel[0] = int(rng.integers(n))
    min_d2 = np.sum((points - points[sel[0]]) ** 2, axis=1)
    for i in range(1, m):
        sel[i] = int(np.argmax(min_d2))
        d2 = np.sum((points - points[sel[i]]) ** 2, axis=1)
        min_d2 = np.minimum(min_d2, d2)
    return points[sel].copy()


def l1_medial_skeleton(points, n_samples=180, n_iters=10, mu=0.28, seed=2026):
    """通过 L1 中值收缩估计叶片骨架节点。

    L1-Medial Skeleton contraction (Huang et al. 2013, ACM TOG).

    Contracts the noisy, incomplete blade point cloud toward its 1D L1-median spine using
    locally weighted Weiszfeld attraction combined with directional linearity repulsion.
    """
    rng = np.random.default_rng(seed)
    ref_idx = np.sort(rng.choice(len(points), min(len(points), 2400), replace=False))
    Q = points[ref_idx]
    m = min(n_samples, max(24, len(Q) // 10))
    X = _farthest_point_sample(Q, m, seed=seed)

    d_nn = cKDTree(Q).query(Q, k=2)[0][:, 1]
    spacing = float(np.median(d_nn))
    diam = float(np.linalg.norm(np.ptp(Q, axis=0)))
    h = max(0.12 * diam, 6.0 * spacing)
    h_max = max(0.24 * diam, h)

    for it in range(n_iters):
        if it > 0 and it % 2 == 0:
            h = min(h * 1.15, h_max)
        inv_r2 = 4.0 / (h * h)

        D_xq = cdist(X, Q)
        theta = np.exp(-inv_r2 * (D_xq ** 2))
        alpha = theta / np.maximum(D_xq, 1e-4 * h)
        w_sum = np.sum(alpha, axis=1, keepdims=True)
        A = np.where(w_sum > 1e-12, (alpha @ Q) / np.maximum(w_sum, 1e-12), X)

        D_xx = cdist(X, X)
        np.fill_diagonal(D_xx, np.inf)
        phi = np.exp(-inv_r2 * (D_xx ** 2))
        beta = phi / np.maximum(D_xx ** 2, (0.06 * h) ** 2)
        np.fill_diagonal(beta, 0.0)

        # 局部方向线性度 sigma_i = lambda_max / sum(lambda)，见 Huang 等 2013 式 5。
        repulsion = np.zeros_like(X)
        b_sum = np.sum(beta, axis=1)
        for i in range(m):
            if b_sum[i] < 1e-12:
                continue
            nb_idx = phi[i] > 1e-3
            if np.sum(nb_idx) >= 3:
                diff = X[nb_idx] - X[i]
                cov = (diff * phi[i, nb_idx, None]).T @ diff
                eig = np.linalg.eigvalsh(cov)
                sigma = float(eig[-1] / max(float(np.sum(eig)), 1e-12))
            else:
                sigma = 0.5
            r_vec = np.sum(beta[i, :, None] * (X[i] - X), axis=0) / b_sum[i]
            r_norm = float(np.linalg.norm(r_vec))
            if r_norm > 0.25 * h:
                r_vec = r_vec * (0.25 * h / r_norm)
            repulsion[i] = mu * sigma * r_vec

        X = A + repulsion

    return X, spacing


def _axis_midrib(points, stem, axis, regularized=False):
    """骨架丢失叶身范围时，以数据支撑的主轴坐标回退估计中脉。

    Data-supported fallback when contracted skeleton nodes lose blade extent.

    Section medians follow the dominant spatial coordinate, so an incomplete
    skeleton cannot introduce a loop inside an otherwise unfolded blade.
    """
    coordinate = points @ axis
    edges = np.linspace(coordinate.min(), coordinate.max(), 25 if regularized else 33)
    nodes = np.asarray([
        np.median(points[(coordinate >= a) & (coordinate <= b)], axis=0)
        for a, b in zip(edges[:-1], edges[1:])
        if np.sum((coordinate >= a) & (coordinate <= b)) >= (6 if regularized else 3)
    ])
    if len(nodes) < 4:
        return None
    endpoint_distances = cKDTree(stem).query(nodes[[0, -1]])[0]
    if endpoint_distances[0] > endpoint_distances[1]:
        nodes = nodes[::-1]
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))]
    curve, lam = fit_pspline_gcv(
        s, nodes, n_basis=8 if regularized else 16,
        lam_bounds=(1.0, 1e4) if regularized else (1e-5, 1e2),
    )
    curve, length = reparameterize_arc_length(curve)
    return curve, length, lam


def extract_midrib(points, stem, max_graph_points=4500, n_principal_iters=2, seed=2026, stabilize=False):
    """组合骨架、测地排序和主曲线迭代，提取叶片中脉。

    L1-Medial Skeleton (Huang et al. 2013) + Geodesic ordering + Principal Curve refinement."""
    if stabilize:
        _, _, axes = np.linalg.svd(points - np.median(points, axis=0), full_matrices=False)
        fallback = _axis_midrib(points, stem, axes[0], regularized=True)
        if fallback is not None:
            curve, length, lam = fallback
            spacing = float(np.median(cKDTree(points).query(points, k=2)[0][:, 1]))
            return curve, length, project_points_to_curve(points, curve, length), spacing, lam
    if len(points) >= 60:
        skel, spacing = l1_medial_skeleton(points, seed=seed)
        geo, _ = geodesic_order(skel, stem, k_nn=8)
        order = np.argsort(geo)
        skel_ord = skel[order]
        geo_ord = geo[order]
        # 沿测地坐标对排序后的 L1 骨架分箱，避免几乎重合的样条节点。
        nb_s = min(28, max(8, len(skel_ord) // 4))
        edges_s = np.linspace(geo_ord[0], geo_ord[-1], nb_s + 1)
        nodes = np.asarray([
            np.median(skel_ord[(geo_ord >= a) & (geo_ord <= b)], axis=0)
            for a, b in zip(edges_s[:-1], edges_s[1:])
            if np.sum((geo_ord >= a) & (geo_ord <= b)) >= 1
        ])
    else:
        geo, spacing = geodesic_order(points, stem)
        edges = np.linspace(0.0, geo.max(), 11)
        nodes = np.asarray([
            np.median(points[(geo >= a) & (geo <= b)], axis=0)
            for a, b in zip(edges[:-1], edges[1:])
            if np.sum((geo >= a) & (geo <= b)) >= 2
        ])

    s0 = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))]
    curve, lam = fit_pspline_gcv(s0, nodes)
    curve, length = reparameterize_arc_length(curve)

    # 沿端点切向延长两端，避免 L1 收缩裁掉终端叶点。
    t_start = unit(curve(0.0, nu=1))
    t_end = unit(curve(length, nu=1))
    p_ext = np.vstack([
        curve(0.0) - 0.15 * length * t_start,
        curve(np.linspace(0.0, length, 40)),
        curve(length) + 0.15 * length * t_end,
    ])
    s_ext = np.r_[-0.15 * length, np.linspace(0.0, length, 40), 1.15 * length]
    ext_curve, _ = fit_pspline_gcv(s_ext, p_ext, n_basis=14, lam_bounds=(1e-6, 1e-1), bounded=False)
    grid_s = np.linspace(-0.15 * length, 1.15 * length, 350)
    s_proj = grid_s[cKDTree(ext_curve(grid_s)).query(points)[1]]
    s_lo, s_hi = float(np.quantile(s_proj, 0.005)), float(np.quantile(s_proj, 0.995))

    nb = min(32, max(10, len(points) // 80))
    for _ in range(n_principal_iters):
        edges = np.linspace(s_lo, s_hi, nb + 1)
        s_nodes, p_nodes = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            mask = (s_proj >= a) & (s_proj <= b)
            if np.sum(mask) >= 3:
                s_nodes.append(float(np.median(s_proj[mask])))
                p_nodes.append(np.median(points[mask], axis=0))
        if len(s_nodes) < 6:
            break
        s_nodes = np.asarray(s_nodes)
        p_nodes = np.asarray(p_nodes)
        ord_idx = np.argsort(s_nodes)
        curve, lam = fit_pspline_gcv(s_nodes[ord_idx], p_nodes[ord_idx])
        curve, length = reparameterize_arc_length(curve)
        s_proj = project_points_to_curve(points, curve, length)
        s_lo, s_hi = 0.0, length

    # 骨架收缩可能丢失稀疏断开的叶尖，或令宽叶片折成环。
    # 因此需对照原始输入范围检查，不能只检查骨架本身。
    _, _, axes = np.linalg.svd(points - np.median(points, axis=0), full_matrices=False)
    axis = axes[0]
    axial_points = points @ axis
    axial_curve = curve(np.linspace(0.0, length, 250)) @ axis
    span = float(np.diff(np.quantile(axial_points, [0.01, 0.99]))[0])
    steps = np.diff(axial_curve)
    backtrack = min(float(np.maximum(steps, 0).sum()), float(np.maximum(-steps, 0).sum()))
    if np.ptp(axial_curve) < 0.95 * span or backtrack > 0.08 * span:
        fallback = _axis_midrib(points, stem, axis)
        if fallback is not None:
            curve, length, lam = fallback
            s_proj = project_points_to_curve(points, curve, length)

    return curve, length, s_proj, spacing, lam


def double_reflection_rmf(points, tangents, n0=None):
    """用双反射构造沿曲线平行运输的最小旋转标架。

    Rotation Minimizing Frame via Double Reflection (Wang et al. 2008, ACM TOG)."""
    m = len(points)
    T = unit(tangents)
    N = np.zeros_like(T)
    if n0 is None:
        ref = np.array([0.0, 0.0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        N[0] = unit(ref - np.dot(ref, T[0]) * T[0])
    else:
        N[0] = unit(n0 - np.dot(n0, T[0]) * T[0])

    for i in range(m - 1):
        v1 = points[i + 1] - points[i]
        c1 = float(np.dot(v1, v1))
        if c1 < 1e-16:
            N[i + 1] = unit(N[i] - np.dot(N[i], T[i + 1]) * T[i + 1])
            continue
        rL = N[i] - (2.0 / c1) * np.dot(v1, N[i]) * v1
        tL = T[i] - (2.0 / c1) * np.dot(v1, T[i]) * v1
        v2 = T[i + 1] - tL
        c2 = float(np.dot(v2, v2))
        if c2 < 1e-16:
            N[i + 1] = unit(rL - np.dot(rL, T[i + 1]) * T[i + 1])
        else:
            N[i + 1] = unit(rL - (2.0 / c2) * np.dot(v2, rL) * v2)

    B = unit(np.cross(N, T))
    return T, B, N


class MovingFrame:
    """保存沿弧长变化的无扭转局部坐标系。

    Twist-free RMF aligned to measured point-cloud surface normals."""

    def __init__(self, s_grid, T_grid, B_grid, N_grid):
        """保存采样弧长、标架向量与滚转角插值器。"""
        self.s_grid = np.asarray(s_grid, dtype=float)
        self.T_grid = np.asarray(T_grid, dtype=float)
        self.B_grid = np.asarray(B_grid, dtype=float)
        self.N_grid = np.asarray(N_grid, dtype=float)

    def evaluate(self, s):
        """在指定弧长插值并返回局部切向、横向和法向。"""
        s = np.atleast_1d(np.asarray(s, dtype=float))
        s_clamped = np.clip(s, self.s_grid[0], self.s_grid[-1])
        t = unit(np.column_stack([np.interp(s_clamped, self.s_grid, self.T_grid[:, j]) for j in range(3)]))
        n = np.column_stack([np.interp(s_clamped, self.s_grid, self.N_grid[:, j]) for j in range(3)])
        n = unit(n - np.sum(n * t, axis=1, keepdims=True) * t)
        b = unit(np.cross(n, t))
        return t, b, n


def build_normal_aligned_frame(curve, length, points, s_proj, normals=None, n_grid=200):
    """构造最小旋转标架，并按扫描法向对齐横向滚转。

    Build RMF (Wang et al. 2008) and align roll angle to measured leaf normals."""
    s_grid = np.linspace(0.0, length, n_grid)
    c_grid = curve(s_grid)
    t_grid = unit(curve(s_grid, nu=1))
    T0, B0, N0 = double_reflection_rmf(c_grid, t_grid)

    # 将扫描法向投影到垂直 T0 的截面，估计分段叶面法向；
    # 无法向时使用截面协方差的最小特征轴。
    n_bins = min(30, max(10, len(points) // 100))
    bin_s = np.linspace(0.0, length, n_bins)
    win = max(length / (n_bins - 1), length * 0.06)

    raw_n = []
    valid_s = []
    for sb in bin_s:
        sel = np.abs(s_proj - sb) <= win
        if np.sum(sel) < 10:
            sel = np.argsort(np.abs(s_proj - sb))[: min(30, len(points))]
        tb = unit(curve(sb, nu=1))
        if normals is not None:
            nb = normals[sel]
            nb = nb - (nb @ tb)[:, None] * tb
            # 由散布矩阵求无定向法向主方向，对正负翻转不敏感。
            _, _, vt = np.linalg.svd(nb, full_matrices=False)
            n_est = vt[0]
        else:
            res = points[sel] - curve(sb)
            res = res - (res @ tb)[:, None] * tb
            _, _, vt = np.linalg.svd(res, full_matrices=False)
            n_est = unit(np.cross(tb, vt[0]))
        if raw_n:
            if np.dot(n_est, raw_n[-1]) < 0:
                n_est = -n_est
        elif n_est[2] < 0:
            n_est = -n_est
        raw_n.append(unit(n_est))
        valid_s.append(sb)

    valid_s = np.asarray(valid_s)
    raw_n = np.asarray(raw_n)

    # 以相对无扭转标架 (N0, B0) 的平滑滚转角 phi(s) 表示对齐法向。
    N0_b = np.column_stack([np.interp(valid_s, s_grid, N0[:, j]) for j in range(3)])
    B0_b = np.column_stack([np.interp(valid_s, s_grid, B0[:, j]) for j in range(3)])
    cos_phi = np.sum(raw_n * N0_b, axis=1)
    sin_phi = np.sum(raw_n * B0_b, axis=1)
    # 使用二倍角消除相差 180 度的法向符号歧义。
    phi_raw = 0.5 * np.unwrap(2.0 * np.arctan2(sin_phi, cos_phi))
    # 先做三点中值滤波去除孤立截面异常，再拟合 GCV 平滑样条。
    phi_clean = phi_raw.copy()
    for i in range(1, len(phi_raw) - 1):
        phi_clean[i] = float(np.median(phi_raw[i - 1 : i + 2]))
    phi_spline, _ = fit_pspline_gcv(valid_s, phi_clean, n_basis=min(8, max(4, len(valid_s) // 3)), lam_bounds=(1e-2, 1e4), bounded=False)
    phi_grid = phi_spline(s_grid)

    N_aligned = unit(np.cos(phi_grid)[:, None] * N0 + np.sin(phi_grid)[:, None] * B0)
    B_aligned = unit(np.cross(N_aligned, T0))
    return MovingFrame(s_grid, T0, B_aligned, N_aligned)
