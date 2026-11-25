"""基于曲面面积加权的法向熵和纵向俯仰熵计算。

Area-weighted equal-solid-angle hemisphere normal Shannon entropy (standalone toggleable module).

Mathematical basis:
  - Leaf blades are thin two-sided surfaces whose outward normal sign may flip; normals are
    canonicalized onto the upper hemisphere (n_z >= 0, with deterministic equatorial tie-break
    and signed-zero cleanup).
  - Uses an exact equal-solid-angle zonal partition with a single circular polar cap at the
    zenith (Leopardi 2006, ETNA 25:309-327): on the unit hemisphere, dOmega = d(cos theta) dphi,
    so a ring with m_r cells spanning Delta(n_z) = m_r / B and Delta(phi) = 2*pi / m_r gives
    every cell the exact solid angle Delta_Omega = 2*pi / B without any azimuthal singularity
    at the zenith pole (so nearly horizontal leaves stay inside the single polar cap).
  - Bin probabilities are weighted by triangle surface area A_f (invariant to scan point density),
    and normalized Shannon entropy H_N = -sum(p_b ln p_b) / ln(B) lies strictly in [0, 1].
"""
import numpy as np


# 从赤道到天顶的环带分箱数，总和等于 B。
ZONAL_RING_COUNTS = {
    12: (6, 5, 1),
    24: (10, 8, 5, 1),
    48: (16, 14, 11, 6, 1),
}


def canonicalize_hemisphere_normals(normals):
    """将正负等价的法向统一映射至上半球。

    Map unit normals n ~ -n canonically onto the upper hemisphere (n_z >= 0)."""
    n = np.asarray(normals, dtype=float)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    u = n / np.maximum(norms, 1e-12)
    flip = (
        (u[:, 2] < -1e-12)
        | ((np.abs(u[:, 2]) <= 1e-12) & (u[:, 0] < -1e-12))
        | ((np.abs(u[:, 2]) <= 1e-12) & (np.abs(u[:, 0]) <= 1e-12) & (u[:, 1] < 0.0))
    )
    out = u.copy()
    out[flip] *= -1.0
    out[np.abs(out) < 1e-15] = 0.0
    out[:, 2] = np.clip(out[:, 2], 0.0, 1.0)
    return out


def triangle_normals_and_areas(vertices, faces):
    """过滤退化三角面，计算单位法向与面积。

    Return unit normals and areas for non-degenerate mesh triangles."""
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    areas = 0.5 * twice_area
    valid = areas > 1e-15
    normals = cross[valid] / twice_area[valid, None]
    return normals, areas[valid]


def equal_area_bin_centers(n_bins=24):
    """生成等立体角法向分箱的代表方向。

    Return representative unit normal vectors at the center of each equal-solid-angle cell."""
    if n_bins not in ZONAL_RING_COUNTS:
        raise ValueError(f"Unsupported n_bins={n_bins}; choose from {sorted(ZONAL_RING_COUNTS)}")
    rings = ZONAL_RING_COUNTS[n_bins]
    centers = []
    cum = 0
    for m in rings:
        z_lo = cum / float(n_bins)
        z_hi = (cum + m) / float(n_bins)
        z_mid = 0.5 * (z_lo + z_hi)
        if m == 1:
            centers.append([0.0, 0.0, 1.0])
        else:
            r = np.sqrt(max(0.0, 1.0 - z_mid * z_mid))
            for j in range(m):
                phi = -np.pi + (j + 0.5) * (2.0 * np.pi / m)
                centers.append([r * np.cos(phi), r * np.sin(phi), z_mid])
        cum += m
    return np.asarray(centers, dtype=float)


def normal_hemisphere_entropy(normals, areas, n_bins=24):
    """按面元面积累计等立体角法向分布的归一化香农熵。

    Compute area-weighted normalized Shannon entropy H_N in [0, 1] over equal-solid-angle cells."""
    if n_bins not in ZONAL_RING_COUNTS:
        raise ValueError(f"Unsupported n_bins={n_bins}; choose from {sorted(ZONAL_RING_COUNTS)}")
    rings = ZONAL_RING_COUNTS[n_bins]

    areas = np.asarray(areas, dtype=float)
    valid = np.isfinite(areas) & (areas > 0.0)
    total_area = float(np.sum(areas[valid]))
    if total_area <= 0.0:
        return float("nan")

    u = canonicalize_hemisphere_normals(np.asarray(normals, dtype=float)[valid])
    w = areas[valid]

    # 将各法向映射到等立体角环带及方位角扇区。
    z_scaled = u[:, 2] * float(n_bins)
    phi = np.where(np.hypot(u[:, 0], u[:, 1]) < 1e-12, 0.0, np.arctan2(u[:, 1], u[:, 0]))
    phi_norm = (phi + np.pi) / (2.0 * np.pi)  # [0, 1)

    bin_id = np.empty(len(u), dtype=int)
    cum = 0
    for r_idx, m in enumerate(rings):
        lo = float(cum)
        hi = float(cum + m)
        if r_idx == len(rings) - 1:
            sel = z_scaled >= lo
        else:
            sel = (z_scaled >= lo) & (z_scaled < hi)
        if np.any(sel):
            if m == 1:
                bin_id[sel] = cum
            else:
                sector = np.minimum((phi_norm[sel] * m).astype(int), m - 1)
                bin_id[sel] = cum + sector
        cum += m

    mass = np.bincount(bin_id, weights=w, minlength=n_bins)
    p = mass[mass > 0.0] / total_area
    h = -float(np.sum(p * np.log(p))) / np.log(float(n_bins))
    return float(np.clip(h, 0.0, 1.0))


def mesh_normal_entropy(vertices, faces, n_bins=24):
    """由网格面元直接计算面积加权半球法向熵。

    Convenience wrapper computing area-weighted hemisphere normal entropy directly from a mesh."""
    normals, areas = triangle_normals_and_areas(vertices, faces)
    return normal_hemisphere_entropy(normals, areas, n_bins=n_bins)


def surface_pitch_entropy_from_grid(xyz, s_vals, v_vals, n_bins=12):
    """对纵向切向俯仰角进行面积加权分箱，计算归一化熵。

    Area-weighted 1D Shannon entropy of longitudinal tangent pitch angle phi in [-90, +90] deg.

    Mathematical properties:
      - Longitudinal tangent S_s is tangent to the midrib/blade growth direction, so its vertical
        pitch phi = arcsin(hat(S_s)_z) is strictly invariant to lateral blade roll around the
        longitudinal axis and to leaf azimuth around the stem.
      - Weighted by exact parametric surface area element dA = ||S_s x S_v|| ds dv (invariant to
        scan point density).
      - Normalized by ln(n_bins) so H_pit lies strictly in [0, 1]: 0 for a straight constant-pitch
        blade (e.g., tightly upright young leaf), higher for strongly arched/bending blades.
    """
    ds = np.gradient(xyz, s_vals, axis=0, edge_order=2)
    dv = np.gradient(xyz, v_vals, axis=1, edge_order=2)
    jac = np.linalg.norm(np.cross(ds, dv), axis=2)

    ws = np.empty(len(s_vals), dtype=float)
    ws[0] = 0.5 * (s_vals[1] - s_vals[0])
    ws[-1] = 0.5 * (s_vals[-1] - s_vals[-2])
    ws[1:-1] = 0.5 * (s_vals[2:] - s_vals[:-2])

    wv = np.empty(len(v_vals), dtype=float)
    wv[0] = 0.5 * (v_vals[1] - v_vals[0])
    wv[-1] = 0.5 * (v_vals[-1] - v_vals[-2])
    wv[1:-1] = 0.5 * (v_vals[2:] - v_vals[:-2])

    dA = jac * ws[:, None] * wv[None, :]
    total_a = float(np.sum(dA))
    if total_a <= 1e-15:
        return float("nan"), float("nan")

    ds_norm = np.maximum(np.linalg.norm(ds, axis=2), 1e-12)
    pitch_deg = np.degrees(np.arcsin(np.clip(ds[:, :, 2] / ds_norm, -1.0, 1.0)))
    edges = np.linspace(-90.0, 90.0, int(n_bins) + 1)
    mass, _ = np.histogram(pitch_deg.ravel(), bins=edges, weights=dA.ravel())
    p = mass[mass > 0.0] / total_a
    h = -float(np.sum(p * np.log(p))) / np.log(float(n_bins))
    mean_pitch = float(np.sum(pitch_deg * dA) / total_a)
    return float(np.clip(h, 0.0, 1.0)), mean_pitch


def model_pitch_entropy(model, n_bins=12, ns=60, nv=17, v_lim=0.90):
    """在模型活动域采样并计算纵向俯仰角信息熵。

    Compute area-weighted longitudinal pitch-angle Shannon entropy directly from a MaizeLeafModel."""
    s_vals = np.linspace(0.0, model.length, ns)
    v_vals = np.linspace(-float(v_lim), float(v_lim), nv)
    ss, vv = np.meshgrid(s_vals, v_vals, indexing="ij")
    xyz = model.evaluate(np.column_stack([ss.ravel(), vv.ravel()])).reshape(ns, nv, 3)
    return surface_pitch_entropy_from_grid(xyz, s_vals, v_vals, n_bins=n_bins)
