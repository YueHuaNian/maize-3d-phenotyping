"""Thin-Plate Spline (TPS) surface with exact GCV regularization and leaf model.

References:
- Duchon (1977): Splines minimizing rotation-invariant semi-norms (Thin-Plate Splines).
- Wahba (1990): Spline Models for Observational Data (exact QR-spectral GCV for TPS).
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree, Delaunay
from scipy.spatial.distance import cdist
from .geometry import MovingFrame, build_normal_aligned_frame, extract_midrib, filter_noise_clusters, project_points_to_curve
from .morphology import (
    DornbuschBlade,
    MaizeShapePrior,
    extrapolate_midrib_and_frame,
    extract_width_profile,
    fit_dornbusch_blade,
)

REGIONS = {0: "observed", 1: "interpolated", 2: "extrapolated"}
COLORS = np.array([[40, 158, 123], [247, 185, 63], [218, 79, 116]], dtype=np.uint8)


@dataclass
class ModelConfig:
    nu: int = 90
    nv: int = 25
    max_tps_points: int = 350
    allow_tip_completion: bool = True
    seed: int = 2026


def tps_kernel(r):
    """2D Thin-Plate Spline Green's function phi(r) = r^2 * log(r) (Duchon 1977)."""
    r2 = r * r
    return np.where(r > 1e-12, r2 * np.log(np.maximum(r, 1e-12)), 0.0)


class ThinPlateSplineGCV:
    """2D Thin-Plate Spline with exact Generalized Cross-Validation smoothing (Wahba 1990)."""

    def __init__(self, xy, w, a, lam, edf):
        self.xy = np.asarray(xy, dtype=float)
        self.w = np.asarray(w, dtype=float)
        self.a = np.asarray(a, dtype=float)
        self.lam = float(lam)
        self.edf = float(edf)

    @classmethod
    def fit(cls, xy, values, lam_bounds=(1e-7, 1e2)):
        xy = np.asarray(xy, dtype=float)
        y = np.asarray(values, dtype=float)
        m = len(xy)
        K = tps_kernel(cdist(xy, xy))
        P = np.column_stack([np.ones(m), xy])

        # QR decomposition of P = [Q1, Q2] [R; 0]; project into null space of P^T (Wahba 1990).
        Q, R = np.linalg.qr(P, mode="complete")
        Q1, Q2 = Q[:, :3], Q[:, 3:]
        R1 = R[:3, :]

        # Symmetric eigendecomposition of M = Q2^T K Q2 (positive definite on P^T w = 0).
        M = Q2.T @ K @ Q2
        M = 0.5 * (M + M.T)
        eigvals, U = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 1e-12)
        z = U.T @ (Q2.T @ y)

        # Generalized Maximum Likelihood / REML criterion (Wahba 1985, Annals of Statistics):
        # theoretically avoids finite-sample GCV undersmoothing as lambda -> 0.
        m_eff = len(eigvals)

        def gml(log_lam):
            lam = 10.0 ** log_lam
            denom = eigvals + lam
            num = float(np.sum((z ** 2) / denom))
            log_det = float(np.mean(np.log(denom)))
            return num * np.exp(log_det)

        res = minimize_scalar(gml, bounds=np.log10(lam_bounds), method="bounded")
        lam_opt = 10.0 ** res.x

        c = U @ (z / (eigvals + lam_opt))
        w = Q2 @ c
        a = np.linalg.solve(R1, Q1.T @ (y - K @ w - lam_opt * w))
        edf = float(m - np.sum(lam_opt / (eigvals + lam_opt)))
        return cls(xy, w, a, lam_opt, edf)

    def evaluate(self, xy_query):
        xy_query = np.atleast_2d(np.asarray(xy_query, dtype=float))
        K = tps_kernel(cdist(xy_query, self.xy))
        P = np.column_stack([np.ones(len(xy_query)), xy_query])
        return K @ self.w + P @ self.a


def leaf_mesh_grid(length, nu, nv):
    """Non-degenerate mesh grid with a single apex vertex at s = L (where w(L) = 0)."""
    s_rows = np.linspace(0.0, length, nu)[:-1]
    v_cols = np.linspace(-1.0, 1.0, nv)
    ss, vv = np.meshgrid(s_rows, v_cols, indexing="ij")
    sv = np.vstack([np.column_stack([ss.ravel(), vv.ravel()]), [[length, 0.0]]])

    faces = []
    for a in range(nu - 2):
        for b in range(nv - 1):
            k = a * nv + b
            faces.extend(((k, k + nv, k + 1), (k + 1, k + nv, k + nv + 1)))
    # Fan triangulation closing the last cross-section cleanly to the apex vertex.
    tip_idx = (nu - 1) * nv
    base_row = (nu - 2) * nv
    for b in range(nv - 1):
        faces.append((base_row + b, tip_idx, base_row + b + 1))
    return sv, np.asarray(faces, dtype=np.int32)


def triangle_areas(vertices, faces):
    tri = vertices[faces]
    return np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2.0


class MaizeLeafModel:
    """Continuous parametric maize leaf surface:
    S(s, v) = C(s) + (v * w(s) / 2) * b(s) + q(s, v) * n(s),  s in [0, L], v in [-1, 1].
    """

    def __init__(self, midrib, frame, blade, tps, length_obs):
        self.midrib = midrib
        self.frame = frame
        self.blade = blade
        self.tps = tps
        self.length_obs = float(length_obs)

    def evaluate(self, sv):
        sv = np.atleast_2d(np.asarray(sv, dtype=float))
        s, v = sv[:, 0], sv[:, 1]
        _, b, n = self.frame.evaluate(s)
        w = self.blade.width(s)
        # TPS trained on normalized (s / L_obs, v); smoothly tapers with blade width toward tip.
        u_tps = np.clip(s / self.length_obs, 0.0, 1.0)
        q = self.tps.evaluate(np.column_stack([u_tps, v]))
        if self.blade.length > self.length_obs + 1e-6:
            w_end = max(float(self.blade.width(self.length_obs)), 1e-6)
            taper = np.where(s > self.length_obs, np.clip(w / w_end, 0.0, 1.0), 1.0)
            q = q * taper
        return self.midrib(np.clip(s, 0.0, self.blade.length)) + (0.5 * v * w)[:, None] * b + q[:, None] * n

    def save(self, path):
        np.savez_compressed(
            path,
            midrib_t=self.midrib.t,
            midrib_c=self.midrib.c,
            midrib_k=self.midrib.k,
            frame_s=self.frame.s_grid,
            frame_T=self.frame.T_grid,
            frame_B=self.frame.B_grid,
            frame_N=self.frame.N_grid,
            blade=np.array([self.blade.length, self.blade.w_max, self.blade.xm, self.blade.k_lig, self.blade.c1, self.blade.c2]),
            tps_xy=self.tps.xy,
            tps_w=self.tps.w,
            tps_a=self.tps.a,
            tps_lam=self.tps.lam,
            tps_edf=self.tps.edf,
            length_obs=self.length_obs,
        )

    @classmethod
    def load(cls, path):
        d = np.load(path)
        midrib = BSpline(d["midrib_t"], d["midrib_c"], int(d["midrib_k"]))
        frame = MovingFrame(d["frame_s"], d["frame_T"], d["frame_B"], d["frame_N"])
        blade = DornbuschBlade(*[float(x) for x in d["blade"]])
        tps = ThinPlateSplineGCV(d["tps_xy"], d["tps_w"], d["tps_a"], float(d["tps_lam"]), float(d["tps_edf"]))
        return cls(midrib, frame, blade, tps, float(d["length_obs"]))


def _select_spatial_subset(u, v, max_points, rng):
    bins = np.column_stack([np.round(u * 40), np.round(v * 12)]).astype(int)
    _, ix = np.unique(bins, axis=0, return_index=True)
    if len(ix) > max_points:
        ix = ix[np.sort(rng.choice(len(ix), max_points, replace=False))]
    return ix


def fit_leaf(points, stem, normals=None, prior=None, cfg=None, leaf_id=None):
    """Full literature-backed maize leaf reconstruction and completion."""
    cfg = cfg or ModelConfig()
    prior = prior or MaizeShapePrior()
    rng = np.random.default_rng(cfg.seed)

    # Deduplicate and remove tiny detached noise fragments (Euclidean cluster filtering).
    p, nrm = filter_noise_clusters(points, normals)

    # 1. Geodesic-initialized Principal Curve midrib (arc-length parameterized).
    midrib_obs, length_obs, s_proj, spacing, midrib_lam = extract_midrib(p, stem, seed=cfg.seed)

    # 2. Twist-free RMF aligned with measured point-cloud surface normals.
    frame_obs = build_normal_aligned_frame(midrib_obs, length_obs, p, s_proj, normals=nrm)

    # 3. Express points in local physical coordinates (s, lateral r, normal h).
    _, b_p, n_p = frame_obs.evaluate(s_proj)
    res_p = p - midrib_obs(s_proj)
    lateral = np.sum(res_p * b_p, axis=1)
    normal_h = np.sum(res_p * n_p, axis=1)

    # 4. Left/right width profile & Dornbusch (2011) analytical blade model.
    prof = extract_width_profile(s_proj, lateral, length_obs)
    blade = fit_dornbusch_blade(
        prof["s"], prof["w_obs"], length_obs, prior=prior, allow_tip_completion=cfg.allow_tip_completion
    )

    # 5. Midrib + RMF extrapolation for truncated tips (Prévot 1991 / ADEL-Maize).
    midrib_full, frame_full = extrapolate_midrib_and_frame(midrib_obs, frame_obs, length_obs, blade.length)

    # 6. Thin-Plate Spline (Duchon 1977) with exact GCV smoothing (Wahba 1990).
    w_at_p = np.maximum(blade.width(s_proj), blade.w_max * 0.01)
    u_norm = s_proj / length_obs
    v_norm = np.clip(2.0 * lateral / w_at_p, -1.25, 1.25)
    ix = _select_spatial_subset(u_norm, v_norm, cfg.max_tps_points, rng)
    tps = ThinPlateSplineGCV.fit(np.column_stack([u_norm[ix], v_norm[ix]]), normal_h[ix])

    model = MaizeLeafModel(midrib_full, frame_full, blade, tps, length_obs)

    # 7. Evaluate continuous surface on non-degenerate fan-apex (s, v) mesh.
    s_grid = np.linspace(0.0, blade.length, cfg.nu)
    grid_sv, faces = leaf_mesh_grid(blade.length, cfg.nu, cfg.nv)
    vertices = model.evaluate(grid_sv)

    # 8. Region classification in continuous physical (s, r) blade coordinates + 3D point spacing.
    phys_obs = np.column_stack([s_proj, lateral])
    w_grid = blade.width(grid_sv[:, 0])
    phys_grid = np.column_stack([grid_sv[:, 0], 0.5 * grid_sv[:, 1] * w_grid])
    hull_inside = Delaunay(phys_obs).find_simplex(phys_grid, tol=1e-10) >= 0

    # Support radius grounded in 3D local spacing and mesh cell size (avoids discretization artifact).
    du = blade.length / max(cfg.nu - 1, 1)
    dv = blade.w_max / max(cfg.nv - 1, 1)
    support_radius = max(2.5 * spacing, 0.75 * max(du, dv))
    dist_3d = cKDTree(p).query(vertices)[0]

    regions = np.where(hull_inside, 1, 2).astype(np.uint8)
    regions[hull_inside & (dist_3d <= support_radius)] = 0
    regions[grid_sv[:, 0] > length_obs + 1e-8] = 2
    face_regions = regions[faces].max(axis=1)

    areas = triangle_areas(vertices, faces)
    total_area = float(areas.sum())
    fractions = {REGIONS[k]: float(areas[face_regions == k].sum() / total_area) for k in REGIONS}

    # In-sample fit residual & phenotypic traits.
    fitted_p = model.evaluate(np.column_stack([s_proj, np.clip(2.0 * lateral / w_at_p, -1.0, 1.0)]))
    fit_rmse = float(np.sqrt(np.mean(np.sum((fitted_p - p) ** 2, axis=1))))
    t_base = midrib_full(0.0, nu=1)
    t_base = t_base / max(float(np.linalg.norm(t_base)), 1e-12)
    base_inclination_deg = float(np.degrees(np.arccos(np.clip(t_base[2], -1.0, 1.0))))

    traits = dict(
        leaf_id=int(leaf_id) if leaf_id is not None else None,
        raw_points=len(points),
        fitted_points=len(p),
        observed_length=float(length_obs),
        completed_length=float(blade.length),
        tip_completion_length=float(max(0.0, blade.length - length_obs)),
        max_width=float(blade.w_max),
        xm_rel_pos=float(blade.xm),
        k_lig=float(blade.k_lig),
        c1=float(blade.c1),
        c2=float(blade.c2),
        form_factor_alpha=float(blade.form_factor),
        surface_area=total_area,
        area_fractions=fractions,
        completion_ratio=fractions["interpolated"] + fractions["extrapolated"],
        base_inclination_deg=base_inclination_deg,
        fit_rmse=fit_rmse,
        midrib_gcv_lambda=float(midrib_lam),
        tps_gcv_lambda=float(tps.lam),
        tps_edf=float(tps.edf),
        support_radius=float(support_radius),
    )
    return dict(
        model=model,
        vertices=vertices,
        faces=faces,
        regions=regions,
        face_regions=face_regions,
        sv=grid_sv,
        points=points,
        midrib=midrib_full(s_grid),
        s=s_grid,
        width=blade.width(s_grid),
        blade=blade,
        traits=traits,
    )
