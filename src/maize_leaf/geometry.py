"""Differential geometry primitives for maize stems and leaves.

References:
- Eilers & Marx (1996) Statistical Science: Penalized B-splines (P-splines)
- Wahba (1990): Generalized Cross-Validation (GCV) for smoothing parameter selection
- Hastie & Stuetzle (1989) JASA: Principal Curves (iterative projection + smoothing)
- Piegl & Tiller (1997) The NURBS Book: Newton orthogonal point projection onto B-splines
- Bishop (1975); Wang et al. (2008) ACM TOG: Double Reflection Rotation Minimizing Frames
"""
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra


def unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


def fit_pspline_gcv(t, y, n_basis=10, degree=3, weights=None, lam_bounds=(1e-3, 1e4)):
    """Penalized B-spline (Eilers & Marx 1996) with automatic GCV smoothing (Wahba 1990)."""
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
    C = np.linalg.solve(BtB + lam_opt * P + 1e-12 * np.eye(nc), BtY)
    spline = BSpline(knots, C[:, 0] if is_1d else C, k)
    return spline, float(lam_opt)


def spline_dict(s):
    return {"knots": s.t.tolist(), "coefficients": s.c.tolist(), "degree": int(s.k)}


def fit_stem_centerline(points):
    """Z-binned median stem centerline fitted with GCV P-spline."""
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
    """Approximate intrinsic leaf geodesic coordinate from stem attachment via kNN + Dijkstra."""
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
    """Reparameterize a 3D B-spline by true physical arc length s in [0, L]."""
    t0, t1 = float(curve.t[curve.k]), float(curve.t[-curve.k - 1])
    ts = np.linspace(t0, t1, n_samples)
    pts = curve(ts)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    arc_curve, _ = fit_pspline_gcv(s, pts, n_basis=min(18, len(curve.c) + 2), lam_bounds=(1e-8, 1e-1))
    return arc_curve, float(s[-1])


def project_points_to_curve(points, curve, s_max, n_init=300, n_newton=5):
    """Exact orthogonal closest-point projection via Newton iteration (Piegl & Tiller 1997)."""
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


def extract_midrib(points, stem, max_graph_points=4500, n_principal_iters=2, seed=2026):
    """Geodesic-initialized Principal Curve (Hastie & Stuetzle 1989) with arc-length P-spline."""
    rng = np.random.default_rng(seed)
    idx_g = np.sort(rng.choice(len(points), min(len(points), max_graph_points), replace=False))
    gp = points[idx_g]
    geo, spacing = geodesic_order(gp, stem)

    nb = min(32, max(10, len(gp) // 80))
    edges = np.linspace(0.0, geo.max(), nb + 1)
    nodes = np.asarray([
        np.median(gp[(geo >= a) & (geo <= b)], axis=0)
        for a, b in zip(edges[:-1], edges[1:])
        if np.sum((geo >= a) & (geo <= b)) >= 3
    ])
    s0 = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))]
    curve, lam = fit_pspline_gcv(s0, nodes)
    curve, length = reparameterize_arc_length(curve)

    # Principal curve refinement: project all points continuously, re-estimate centers by arc-length bins.
    for _ in range(n_principal_iters):
        s_proj = project_points_to_curve(points, curve, length)
        edges = np.linspace(0.0, length, nb + 1)
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
        order = np.argsort(s_nodes)
        curve, lam = fit_pspline_gcv(s_nodes[order], p_nodes[order])
        curve, length = reparameterize_arc_length(curve)

    s_proj = project_points_to_curve(points, curve, length)
    return curve, length, s_proj, spacing, lam


def double_reflection_rmf(points, tangents, n0=None):
    """Rotation Minimizing Frame via Double Reflection (Wang et al. 2008, ACM TOG)."""
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
    """Twist-free RMF aligned to measured point-cloud surface normals."""

    def __init__(self, s_grid, T_grid, B_grid, N_grid):
        self.s_grid = np.asarray(s_grid, dtype=float)
        self.T_grid = np.asarray(T_grid, dtype=float)
        self.B_grid = np.asarray(B_grid, dtype=float)
        self.N_grid = np.asarray(N_grid, dtype=float)

    def evaluate(self, s):
        s = np.atleast_1d(np.asarray(s, dtype=float))
        s_clamped = np.clip(s, self.s_grid[0], self.s_grid[-1])
        t = unit(np.column_stack([np.interp(s_clamped, self.s_grid, self.T_grid[:, j]) for j in range(3)]))
        n = np.column_stack([np.interp(s_clamped, self.s_grid, self.N_grid[:, j]) for j in range(3)])
        n = unit(n - np.sum(n * t, axis=1, keepdims=True) * t)
        b = unit(np.cross(n, t))
        return t, b, n


def build_normal_aligned_frame(curve, length, points, s_proj, normals=None, n_grid=200):
    """Build RMF (Wang et al. 2008) and align roll angle to measured leaf normals."""
    s_grid = np.linspace(0.0, length, n_grid)
    c_grid = curve(s_grid)
    t_grid = unit(curve(s_grid, nu=1))
    T0, B0, N0 = double_reflection_rmf(c_grid, t_grid)

    # Estimate per-section blade normal from supplied point normals (projected perpendicular to T0),
    # or fallback to smallest cross-section covariance axis when normals are absent.
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
            # Unoriented normal principal direction via scatter matrix (invariant to +/- flips).
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

    # Express aligned normal as smooth roll angle phi(s) relative to the twist-free RMF (N0, B0).
    N0_b = np.column_stack([np.interp(valid_s, s_grid, N0[:, j]) for j in range(3)])
    B0_b = np.column_stack([np.interp(valid_s, s_grid, B0[:, j]) for j in range(3)])
    cos_phi = np.sum(raw_n * N0_b, axis=1)
    sin_phi = np.sum(raw_n * B0_b, axis=1)
    # Use 2*phi (period pi) to resolve 180-deg normal sign ambiguity smoothly across noisy sections.
    phi_raw = 0.5 * np.unwrap(2.0 * np.arctan2(sin_phi, cos_phi))
    # Reject isolated section glitches with 3-point median before smooth GCV spline.
    phi_clean = phi_raw.copy()
    for i in range(1, len(phi_raw) - 1):
        phi_clean[i] = float(np.median(phi_raw[i - 1 : i + 2]))
    phi_spline, _ = fit_pspline_gcv(valid_s, phi_clean, n_basis=min(8, max(4, len(valid_s) // 3)), lam_bounds=(1e-2, 1e4))
    phi_grid = phi_spline(s_grid)

    N_aligned = unit(np.cos(phi_grid)[:, None] * N0 + np.sin(phi_grid)[:, None] * B0)
    B_aligned = unit(np.cross(N_aligned, T0))
    return MovingFrame(s_grid, T0, B_aligned, N_aligned)
