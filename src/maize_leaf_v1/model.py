"""B-spline midrib + moving frame + width envelope + thin-plate RBF surface."""
from dataclasses import dataclass
import numpy as np
from scipy.interpolate import BSpline, splrep, RBFInterpolator
from scipy.optimize import lsq_linear
from scipy.ndimage import gaussian_filter1d, maximum_filter1d
from scipy.spatial import cKDTree, Delaunay
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra

REGIONS = {0: "observed", 1: "interpolated", 2: "extrapolated"}
COLORS = np.array([[40, 158, 123], [247, 185, 63], [218, 79, 116]], dtype=np.uint8)


@dataclass
class Config:
    nu: int = 90
    nv: int = 25
    max_graph_points: int = 4500
    rbf_points: int = 350
    rbf_smoothing: float = 0.003
    tip_extension: float = 0.04
    edge_expansion: float = 0.06
    support_radius_factor: float = 1.8
    seed: int = 20261123


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def curve_fit(u, xyz, penalty_weight=0.05):
    u, xyz = np.asarray(u), np.asarray(xyz)
    k = min(3, len(u) - 1)
    nc = min(14, max(k + 1, len(u) // 2))
    knots = np.r_[
        np.repeat(u[0], k + 1),
        np.linspace(u[0], u[-1], nc - k + 1)[1:-1],
        np.repeat(u[-1], k + 1),
    ]
    basis = BSpline.design_matrix(u, knots, k).toarray()
    weights = np.ones(len(u))
    weights[[0, -1]] = 3
    penalty = np.diff(np.eye(nc), n=2, axis=0)
    a = np.vstack([basis * weights[:, None], np.sqrt(penalty_weight) * penalty])
    coefs = []
    for col in xyz.T:
        span = max(float(np.ptp(col)), 1e-5)
        lo, hi = float(col.min() - 0.02 * span), float(col.max() + 0.02 * span)
        target = np.r_[col * weights, np.zeros(len(penalty))]
        coefs.append(lsq_linear(a, target, bounds=(lo, hi), tol=1e-10).x)
    return BSpline(knots, np.asarray(coefs).T, k)


def spline_dict(s):
    return {"knots": s.t.tolist(), "coefficients": s.c.tolist(), "degree": s.k}


def stem_model(points):
    z = points[:, 2]
    edges = np.linspace(z.min(), z.max(), min(55, max(8, len(points) // 35)))
    nodes = np.asarray([
        np.median(points[(z >= a) & (z <= b)], axis=0)
        for a, b in zip(edges[:-1], edges[1:])
        if np.sum((z >= a) & (z <= b)) >= 3
    ])
    u = (nodes[:, 2] - nodes[0, 2]) / (nodes[-1, 2] - nodes[0, 2])
    curve = curve_fit(u, nodes)
    us = np.linspace(0, 1, 160)
    sampled = curve(us)
    return sampled, {"spline": spline_dict(curve), "u": us.tolist(), "xyz": sampled.tolist()}


def graph_coordinate(p, stem):
    tree = cKDTree(p)
    dist, idx = tree.query(p, k=min(15, len(p)))
    spacing = float(np.median(dist[:, 1]))
    rows = np.repeat(np.arange(len(p)), dist.shape[1] - 1)
    cols, weights = idx[:, 1:].ravel(), dist[:, 1:].ravel()
    graph = coo_matrix((weights, (rows, cols)), shape=(len(p), len(p))).tocsr()
    graph = graph.maximum(graph.T)
    ncomp, labels = connected_components(graph)
    if ncomp > 1:
        graph = graph.tolil()
        active = labels == np.argmax(np.bincount(labels))
        while not np.all(active):
            ia, ib = np.flatnonzero(active), np.flatnonzero(~active)
            d, j = cKDTree(p[ia]).query(p[ib])
            k = int(np.argmin(d))
            a, b = int(ia[j[k]]), int(ib[k])
            graph[a, b] = graph[b, a] = float(d[k])
            active |= labels == labels[b]
        graph = graph.tocsr()
    root = int(np.argmin(cKDTree(stem).query(p)[0]))
    s = dijkstra(graph, directed=False, indices=root)
    return s / s.max(), spacing


def faces_grid(nu, nv):
    f = []
    for a in range(nu - 1):
        for b in range(nv - 1):
            k = a * nv + b
            f.extend(((k, k + nv, k + 1), (k + 1, k + nv, k + nv + 1)))
    return np.asarray(f, dtype=np.int32)


def triangle_areas(vertices, faces):
    tri = vertices[faces]
    return np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2


class LeafModel:
    def frame(self, u):
        t = unit(self.center(u, nu=1))
        guide = self.lateral(u)
        b = unit(guide - np.sum(guide * t, axis=-1, keepdims=True) * t)
        n = unit(np.cross(t, b))
        return t, b, n

    def evaluate(self, uv):
        uv = np.atleast_2d(uv)
        u, v = uv[:, 0], uv[:, 1]
        _, b, n = self.frame(u)
        width = np.maximum(self.width(np.minimum(u, 1)), self.width_floor)
        if self.extension > 0:
            e = np.clip((u - 1) / self.extension, 0, 1)
            width *= 1 - (1 - self.width_floor / self.tip_width) * (3 * e**2 - 2 * e**3)
        pred = self.rbf(np.column_stack([np.minimum(u, 1), v])) * self.q_scale
        q = self.q_cap * np.tanh(pred / self.q_cap)
        q *= np.minimum(1, width / self.tip_width)
        return self.center(u) + (v * width / 2)[:, None] * b + q[:, None] * n

    def save(self, path):
        np.savez_compressed(
            path,
            center_t=self.center.t,
            center_c=self.center.c,
            center_k=self.center.k,
            lateral_t=self.lateral.t,
            lateral_c=self.lateral.c,
            lateral_k=self.lateral.k,
            width_t=self.width.t,
            width_c=self.width.c,
            width_k=self.width.k,
            rbf_xy=self.rbf_xy,
            rbf_d=self.rbf_d,
            smoothing=self.cfg.rbf_smoothing,
            q_scale=self.q_scale,
            q_cap=self.q_cap,
            width_floor=self.width_floor,
            tip_width=self.tip_width,
            extension=self.extension,
        )

    @classmethod
    def load(cls, path):
        a = np.load(path)
        m = cls()
        for key in ("center", "lateral", "width"):
            setattr(m, key, BSpline(a[key + "_t"], a[key + "_c"], int(a[key + "_k"])))
        for key in ("q_scale", "q_cap", "width_floor", "tip_width", "extension"):
            setattr(m, key, float(a[key]))
        m.cfg = Config(rbf_smoothing=float(a["smoothing"]))
        m.rbf_xy, m.rbf_d = a["rbf_xy"], a["rbf_d"]
        m.rbf = RBFInterpolator(
            m.rbf_xy,
            m.rbf_d,
            kernel="thin_plate_spline",
            smoothing=float(a["smoothing"]),
            degree=1,
        )
        return m


def fit_leaf(points, stem, cfg=None, leaf_id=None):
    cfg = cfg or Config()
    m = LeafModel()
    m.cfg = cfg
    rng = np.random.default_rng(cfg.seed)

    unique = np.unique(points, axis=0)
    d = cKDTree(unique).query(unique, k=min(7, len(unique)))[0][:, -1]
    keep = d <= max(np.median(d) * 6, np.linalg.norm(np.ptp(unique, axis=0)) * 0.02)
    p = unique[keep]

    graph_p = p[np.sort(rng.choice(len(p), min(len(p), cfg.max_graph_points), replace=False))]
    g, spacing = graph_coordinate(graph_p, stem)

    nb = min(36, max(10, len(graph_p) // 65))
    edges = np.linspace(0, 1, nb + 1)
    nodes = np.asarray([
        np.median(graph_p[(g >= a) & (g <= b)], axis=0)
        for a, b in zip(edges[:-1], edges[1:])
        if np.sum((g >= a) & (g <= b)) >= 3
    ])
    u = np.r_[0, np.cumsum(np.linalg.norm(np.diff(nodes, axis=0), axis=1))]
    length = float(u[-1])
    u /= length
    m.center = curve_fit(u, nodes)

    dense_u = np.linspace(0, 1, 650)
    pu = dense_u[cKDTree(m.center(dense_u)).query(p)[1]]

    guide_u = np.linspace(0, 1, 24)
    tangent = unit(m.center(guide_u, nu=1))
    guides = []
    for i, x in enumerate(guide_u):
        j = np.argsort(np.abs(pu - x))[: max(25, len(p) // 14)]
        residual = p[j] - m.center(x)
        residual -= (residual @ tangent[i])[:, None] * tangent[i]
        _, _, vt = np.linalg.svd(residual, full_matrices=False)
        b = vt[0]
        if guides and b @ guides[-1] < 0:
            b = -b
        guides.append(b)
    guides = gaussian_filter1d(np.asarray(guides), 1.3, axis=0, mode="nearest")
    m.lateral = curve_fit(guide_u, guides)

    _, b, n = m.frame(pu)
    residual = p - m.center(pu)
    lateral = np.sum(residual * b, axis=1)
    normal = np.sum(residual * n, axis=1)

    wu = np.linspace(0, 1, 36)
    half = np.asarray([
        np.quantile(np.abs(lateral[np.argsort(np.abs(pu - x))[: max(18, len(p) // 24)]]), 0.96)
        for x in wu
    ])
    half = gaussian_filter1d(0.65 * half + 0.35 * maximum_filter1d(half, 5), 1.4)
    width = np.maximum(2 * half * (1 + cfg.edge_expansion), spacing * 2)
    m.width = BSpline(*splrep(wu, width, k=3, s=len(wu) * (width.max() * 0.018) ** 2))
    m.width_floor = float(width.max() * 0.006)
    m.tip_width = float(max(m.width(1), m.width_floor))
    m.extension = cfg.tip_extension if m.tip_width > width.max() * 0.18 else 0.0

    v = 2 * lateral / np.maximum(m.width(pu), m.width_floor)
    uv = np.column_stack([pu, v])
    bins = np.column_stack([np.round(pu * 45), np.round(v * 12)]).astype(int)
    _, ix = np.unique(bins, axis=0, return_index=True)
    if len(ix) > cfg.rbf_points:
        ix = ix[np.sort(rng.choice(len(ix), cfg.rbf_points, replace=False))]
    m.rbf_xy = uv[ix]
    m.q_scale = float(max(np.quantile(np.abs(normal), 0.9), spacing))
    m.q_cap = float(max(width.max() * 0.32, m.q_scale))
    m.rbf_d = normal[ix] / m.q_scale
    m.rbf = RBFInterpolator(
        m.rbf_xy, m.rbf_d, kernel="thin_plate_spline", smoothing=cfg.rbf_smoothing, degree=1
    )

    us, vs = np.linspace(0, 1 + m.extension, cfg.nu), np.linspace(-1, 1, cfg.nv)
    uu, vv = np.meshgrid(us, vs, indexing="ij")
    grid_uv = np.column_stack([uu.ravel(), vv.ravel()])
    vertices = m.evaluate(grid_uv)
    faces = faces_grid(cfg.nu, cfg.nv)

    phys = np.column_stack([pu * length, lateral])
    wg = np.maximum(m.width(np.minimum(grid_uv[:, 0], 1)), m.width_floor)
    grid_phys = np.column_stack([grid_uv[:, 0] * length, grid_uv[:, 1] * wg / 2])
    tree = cKDTree(phys)
    density = tree.query(phys, k=2)[0][:, 1]
    support_radius = max(
        float(np.median(density[density > 0])) * cfg.support_radius_factor, spacing * 0.8
    )
    inside = Delaunay(phys).find_simplex(grid_phys, tol=1e-10) >= 0
    region = np.where(inside, 1, 2).astype(np.uint8)
    region[inside & (tree.query(grid_phys)[0] <= support_radius)] = 0
    region[grid_uv[:, 0] > 1 + 1e-10] = 2

    face_regions = region[faces].max(axis=1)
    area = triangle_areas(vertices, faces)
    total_area = float(area.sum())
    fractions = {REGIONS[i]: float(area[face_regions == i].sum() / total_area) for i in REGIONS}

    fitted = m.evaluate(uv)
    fit_rmse = float(np.sqrt(np.mean(np.sum((fitted - p) ** 2, axis=1))))
    mesh_dist = cKDTree(vertices).query(p)[0]

    metrics = dict(
        leaf_id=int(leaf_id) if leaf_id is not None else None,
        raw_points=len(points),
        fitted_points=len(p),
        centerline_length=length,
        max_width=float(width.max()),
        surface_area=total_area,
        area_fractions=fractions,
        completion_ratio=fractions["interpolated"] + fractions["extrapolated"],
        fit_rmse=fit_rmse,
        point_to_grid_median=float(np.median(mesh_dist)),
        point_to_grid_p95=float(np.quantile(mesh_dist, 0.95)),
        tip_extension_fraction=m.extension,
        support_radius=support_radius,
    )
    return dict(
        model=m,
        vertices=vertices,
        faces=faces,
        regions=region,
        face_regions=face_regions,
        uv=grid_uv,
        points=points,
        metrics=metrics,
        midrib=m.center(us),
        u=us,
        width=np.maximum(m.width(np.minimum(us, 1)), m.width_floor),
    )
