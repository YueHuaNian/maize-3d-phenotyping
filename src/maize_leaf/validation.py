"""Deterministic occlusion splits and exact point-to-triangle benchmark distances."""
import numpy as np

SCENARIOS = ("hole", "margin", "tip_10pct", "tip_20pct", "tip_30pct")


def occlusion_masks(u, v):
    """Use frozen full-scan coordinates only to define the test split."""
    u, v = np.asarray(u), np.asarray(v)
    return {
        "hole": ((u - 0.50) / 0.08) ** 2 + (v / 0.56) ** 2 < 1.0,
        "margin": (u > 0.30) & (u < 0.52) & (v > 0.30),
        "tip_10pct": u > 0.90,
        "tip_20pct": u > 0.80,
        "tip_30pct": u > 0.70,
    }


def point_mesh_distances(points, vertices, faces, chunk_size=32):
    """Euclidean distance to triangle interiors/edges, including degenerate faces.

    All triangles are examined in bounded point batches. The result is exact
    for the supplied mesh, rather than a nearest sampled-vertex approximation.
    """
    points = np.asarray(points, dtype=float)
    tri = np.asarray(vertices, dtype=float)[np.asarray(faces, dtype=int)]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac = b - a, c - a
    dot = lambda x, y: np.sum(x * y, axis=-1)
    aa, bb, cc = dot(ab, ab), dot(ab, ac), dot(ac, ac)
    determinant = aa * cc - bb * bb
    normal = np.cross(ab, ac)
    normal2 = dot(normal, normal)
    result = np.empty(len(points))
    for start in range(0, len(points), chunk_size):
        q = points[start:start + chunk_size, None, :]
        ap = q - a
        d, e = dot(ap, ab), dot(ap, ac)
        safe = np.where(determinant > 0, determinant, 1.0)
        u, v = (cc * d - bb * e) / safe, (aa * e - bb * d) / safe
        inside = (determinant > 0) & (u >= 0) & (v >= 0) & (u + v <= 1)
        plane2 = dot(ap, normal) ** 2 / np.maximum(normal2, 1e-300)
        minimum = np.where(inside, plane2, np.inf)
        for x, y in ((a, b), (b, c), (c, a)):
            edge = y - x
            t = np.clip(dot(q - x, edge) / np.maximum(dot(edge, edge), 1e-300), 0, 1)
            residual = q - x - t[..., None] * edge
            minimum = np.minimum(minimum, dot(residual, residual))
        result[start:start + chunk_size] = np.sqrt(np.maximum(minimum.min(axis=1), 0.0))
    return result


def summarize_distances(distances):
    d = np.asarray(distances)
    return {"median": float(np.median(d)), "p95": float(np.quantile(d, .95)),
            "rmse": float(np.sqrt(np.mean(d ** 2)))}


def holdout_alerts(record):
    """Fixed engineering alerts, recorded before the benchmark; not acceptance truth."""
    alerts = []
    if record["held_p95"] > max(6 * record["spacing"], .15 * record["ref_max_width"]):
        alerts.append("held_p95")
    if record["length_change_pct"] > 15:
        alerts.append("length_change")
    if record["area_change_pct"] > 25:
        alerts.append("area_change")
    return alerts
