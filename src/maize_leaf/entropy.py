"""Area-weighted equal-solid-angle hemisphere normal Shannon entropy (standalone toggleable module).

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


# Ring cell counts from equator (n_z = 0) to zenith polar cap (n_z = 1): sum == B.
ZONAL_RING_COUNTS = {
    12: (6, 5, 1),
    24: (10, 8, 5, 1),
    48: (16, 14, 11, 6, 1),
}


def canonicalize_hemisphere_normals(normals):
    """Map unit normals n ~ -n canonically onto the upper hemisphere (n_z >= 0)."""
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
    """Return unit normals and areas for non-degenerate mesh triangles."""
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    twice_area = np.linalg.norm(cross, axis=1)
    areas = 0.5 * twice_area
    valid = areas > 1e-15
    normals = cross[valid] / twice_area[valid, None]
    return normals, areas[valid]


def equal_area_bin_centers(n_bins=24):
    """Return representative unit normal vectors at the center of each equal-solid-angle cell."""
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
    """Compute area-weighted normalized Shannon entropy H_N in [0, 1] over equal-solid-angle cells."""
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

    # Map each normal to its equal-solid-angle zonal ring and azimuthal sector
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
    """Convenience wrapper computing area-weighted hemisphere normal entropy directly from a mesh."""
    normals, areas = triangle_normals_and_areas(vertices, faces)
    return normal_hemisphere_entropy(normals, areas, n_bins=n_bins)
