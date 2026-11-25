"""Thin-Plate Spline (TPS) surface with spectral GML smoothing and leaf model.

References:
- Duchon (1977): Splines minimizing rotation-invariant semi-norms (Thin-Plate Splines).
- Wahba (1985): GML smoothing parameter selection; QR-spectral implementation.
"""
from dataclasses import dataclass, asdict, replace
import numpy as np
from scipy.interpolate import BSpline, PchipInterpolator
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree, Delaunay
from scipy.spatial.distance import cdist
from .geometry import (
    MovingFrame,
    build_normal_aligned_frame,
    clean_leaf_points,
    extract_midrib,
    fit_pspline_gcv,
    project_points_to_curve,
    reparameterize_arc_length,
)
from .joint import joint_optimize_geometry
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
    refine_tps_points: int = 700
    allow_tip_completion: bool = True
    seed: int = 2026


def tps_kernel(r):
    """2D Thin-Plate Spline Green's function phi(r) = r^2 * log(r) (Duchon 1977)."""
    r2 = r * r
    return np.where(r > 1e-12, r2 * np.log(np.maximum(r, 1e-12)), 0.0)


class ThinPlateSplineGCV:
    """2D TPS with spectral GML smoothing; the class name is retained for NPZ/API compatibility."""

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
        # Select smoothing from the data; neither GML nor GCV guarantees against overfitting.
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


def surface_orientation_diagnostics(model, nu=121, nv=31):
    """Detect local chart reversals against the moving-frame normal.

    A small point-to-surface distance or nonzero triangle area does not rule out
    a folded strip. This samples its signed Jacobian away from the apex; it is
    not a global triangle-intersection certificate.
    """
    s = np.linspace(0.01, 0.99, nu) * model.blade.length
    v = np.linspace(-0.98, 0.98, nv)
    ss, vv = np.meshgrid(s, v, indexing="ij")
    xyz = model.evaluate(np.column_stack([ss.ravel(), vv.ravel()])).reshape(nu, nv, 3)
    ds = np.gradient(xyz, s, axis=0, edge_order=2)
    dv = np.gradient(xyz, v, axis=1, edge_order=2)
    normal = np.cross(ds, dv)
    _, _, reference = model.frame.evaluate(s)
    signed = np.sum(normal * reference[:, None, :], axis=2)
    cosine = signed / np.maximum(np.linalg.norm(normal, axis=2), 1e-12)
    return {"reversed_fraction": float(np.mean(cosine < -1e-6)),
            "minimum_normal_cosine": float(np.min(cosine)), "nu": nu, "nv": nv}


class MaizeLeafModel:
    """Continuous parametric maize leaf surface:
    S(s, v) = C(s) + (v * w(s) / 2) * b(s) + q(s, v) * n(s),  s in [0, L], v in [-1, 1].
    """

    def __init__(self, midrib, frame, blade, tps, length_obs, width_s=None, width_extra=None,
                 apex_taper=True, surface_tip_extension=True):
        self.midrib = midrib
        self.frame = frame
        self.blade = blade
        self.tps = tps
        self.length_obs = float(length_obs)
        self.apex_taper = bool(apex_taper)
        self.surface_tip_extension = bool(surface_tip_extension)
        self._tip_center = None
        self.width_s = np.asarray(width_s if width_s is not None else [0.0, length_obs], dtype=float)
        self.width_extra = np.asarray(width_extra if width_extra is not None else [0.0, 0.0], dtype=float)
        self._width_extra = PchipInterpolator(self.width_s, self.width_extra, extrapolate=False)

    def width(self, s):
        """Morphology prior plus smooth observed-envelope support on the scanned span."""
        extra = np.nan_to_num(self._width_extra(np.asarray(s, dtype=float)), nan=0.0)
        return self.blade.width(s) + np.maximum(extra, 0.0)

    def evaluate(self, sv):
        sv = np.atleast_2d(np.asarray(sv, dtype=float))
        s, v = sv[:, 0], sv[:, 1]
        _, b, n = self.frame.evaluate(s)
        w = self.width(s)
        # TPS trained on normalized (s / L_obs, v); smoothly tapers with blade width toward tip.
        u_tps = np.clip(s / self.length_obs, 0.0, 1.0)
        q = self.tps.evaluate(np.column_stack([u_tps, v]))
        if self.apex_taper:
            # A zero-width tip is one point, not a transverse TPS height curve.
            # Preserve the center height while collapsing its transverse relief.
            q_center = self.tps.evaluate(np.column_stack([u_tps, np.zeros(len(v))]))
            transverse_taper = np.clip(w / max(0.1 * self.blade.w_max, 1e-12), 0.0, 1.0)
            transverse_taper = transverse_taper ** 2 * (3.0 - 2.0 * transverse_taper)
            q = q_center + transverse_taper * (q - q_center)
        if self.blade.length > self.length_obs + 1e-6:
            w_end = max(float(self.blade.width(self.length_obs)), 1e-6)
            taper = np.where(s > self.length_obs, np.clip(w / w_end, 0.0, 1.0), 1.0)
            q = q * taper
        xyz = self.midrib(np.clip(s, 0.0, self.blade.length)) + (0.5 * v * w)[:, None] * b + q[:, None] * n
        outside = s > self.length_obs
        if self.surface_tip_extension and np.any(outside):
            # Continue the actual surface center S(s,0), whose height may encode
            # most longitudinal bending when the reference curve is linear.
            # Fading its absolute TPS height to zero bends the tip toward the
            # reference line instead of continuing the measured blade.
            if self._tip_center is None:
                self._build_tip_center()
            end_v = self.tps.evaluate(np.column_stack([np.ones(outside.sum()), v[outside]]))
            end_zero = float(self.tps.evaluate([[1.0, 0.0]])[0])
            end_width = max(float(self.width(self.length_obs)), 1e-12)
            relief_taper = np.clip(w[outside] / end_width, 0.0, 1.0)
            # Include the same transverse apex factor at the observed seam.
            end_factor = np.clip(end_width / max(.1 * self.blade.w_max, 1e-12), 0., 1.)
            end_factor = end_factor ** 2 * (3 - 2 * end_factor) if self.apex_taper else 1.
            xyz[outside] = (self._tip_center(s[outside])
                            + (.5 * v[outside] * w[outside])[:, None] * b[outside]
                            + (end_factor * (end_v - end_zero) * relief_taper)[:, None] * n[outside])
        return xyz

    def _build_tip_center(self):
        ss = np.linspace(.75 * self.length_obs, self.length_obs, 64)
        center = self.evaluate(np.column_stack([ss, np.zeros(len(ss))]))
        velocity = np.gradient(center, ss, axis=0, edge_order=2)
        speed = np.linalg.norm(velocity, axis=1)
        tangent = velocity / np.maximum(speed[:, None], 1e-12)
        theta = np.arcsin(np.clip(tangent[:, 2], -1., 1.))
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(center, axis=0), axis=1))]
        curvature = min(float(np.polyfit(arc, theta, 1)[0]), 0.)
        azimuth = tangent[-1, :2]
        azimuth = azimuth / max(float(np.linalg.norm(azimuth)), 1e-12)
        ext_s = np.linspace(self.length_obs, self.blade.length, 129)
        delta = .5 * (ext_s[1:] + ext_s[:-1]) - self.length_obs
        pitch = np.maximum(min(theta[-1], -.48 * np.pi),
            theta[-1] + curvature * speed[-1] * delta * np.exp(-.5 * delta / self.length_obs))
        direction = np.column_stack([np.cos(pitch)[:, None] * azimuth, np.sin(pitch)])
        steps = np.diff(ext_s) * speed[-1]
        xyz = center[-1] + np.vstack([np.zeros(3), np.cumsum(steps[:, None] * direction, axis=0)])
        self._tip_center = PchipInterpolator(ext_s, xyz, axis=0)

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
            width_s=self.width_s,
            width_extra=self.width_extra,
            apex_taper=self.apex_taper,
            surface_tip_extension=self.surface_tip_extension,
        )

    @classmethod
    def load(cls, path):
        d = np.load(path)
        midrib = BSpline(d["midrib_t"], d["midrib_c"], int(d["midrib_k"]))
        frame = MovingFrame(d["frame_s"], d["frame_T"], d["frame_B"], d["frame_N"])
        blade = DornbuschBlade(*[float(x) for x in d["blade"]])
        tps = ThinPlateSplineGCV(d["tps_xy"], d["tps_w"], d["tps_a"], float(d["tps_lam"]), float(d["tps_edf"]))
        return cls(midrib, frame, blade, tps, float(d["length_obs"]),
                   d["width_s"] if "width_s" in d else None,
                   d["width_extra"] if "width_extra" in d else None,
                   bool(d["apex_taper"]) if "apex_taper" in d else False,
                   bool(d["surface_tip_extension"]) if "surface_tip_extension" in d else False)


def _aggregate_tps_bins(u, v, h, max_points, rng):
    """Aggregate TPS training samples by 2D bin median to suppress residual scatter spikes."""
    bins = np.column_stack([np.round(u * 40), np.round(v * 12)]).astype(int)
    _, inv = np.unique(bins, axis=0, return_inverse=True)
    n_bins = int(inv.max()) + 1
    uv_med = np.empty((n_bins, 2), dtype=float)
    h_med = np.empty(n_bins, dtype=float)
    for idx in range(n_bins):
        sel = inv == idx
        uv_med[idx, 0] = float(np.median(u[sel]))
        uv_med[idx, 1] = float(np.median(v[sel]))
        h_med[idx] = float(np.median(h[sel]))
    if n_bins > max_points:
        pick = np.sort(rng.choice(n_bins, max_points, replace=False))
        return uv_med[pick], h_med[pick]
    return uv_med, h_med


def fit_leaf(points, stem, normals=None, prior=None, cfg=None, leaf_id=None,
             _stable_geometry=False, _geometry_mode=None, _geometry_cache=None):
    """Fit and audit multiple geometric charts, select a non-reversed scan-fitting surface."""
    cfg = cfg or ModelConfig()
    prior = prior or MaizeShapePrior()
    # This cache is local to one observed point set. It never crosses a leaf or
    # a holdout split; TPS density does not change its deterministic geometry.
    cache = {} if _geometry_cache is None else _geometry_cache
    if "clean" not in cache:
        cache["clean"] = clean_leaf_points(points, normals)
    if _geometry_mode is None and not _stable_geometry:
        from .quality import audit_model, surface_support_diagnostics
        # Compare regular surfaces on identical retained scan points. Retrying
        # only a single heavily smoothed curve can fix folds while harming fit.
        retained, _ = cache["clean"]
        candidates = []
        for mode in ("joint", "initial", "smoothed", "axis", "linear"):
            candidate = fit_leaf(points, stem, normals=normals, prior=prior, cfg=cfg,
                                 leaf_id=leaf_id, _geometry_mode=mode, _geometry_cache=cache)
            sv, _ = leaf_mesh_grid(candidate["blade"].length, 220, 61)
            distances = cKDTree(candidate["model"].evaluate(sv)).query(retained)[0]
            median, p95 = np.quantile(distances, [0.5, 0.95])
            candidate["traits"]["scan_distance_median"] = float(median)
            candidate["traits"]["scan_distance_p95"] = float(p95)
            candidate["traits"]["surface_support_mean"] = surface_support_diagnostics(
                candidate["model"], retained)["mean"]
            candidates.append(candidate)
        valid = [r for r in candidates if r["traits"]["orientation_reversed_fraction"] == 0.0]
        if not valid:
            raise ValueError(f"Leaf {leaf_id}: no candidate passed the local orientation check")
        ordered = sorted(valid, key=lambda r: (
            r["traits"]["scan_distance_p95"] + 0.5 * r["traits"]["scan_distance_median"]
            + r["traits"]["surface_support_mean"],
        ))
        best = ordered[0]
        rejected = []
        for candidate in ordered:
            candidate["quality"] = audit_model(candidate["model"], retained)
            if candidate["quality"]["passed"]:
                best = candidate
                break
            rejected.append({"geometry_method": candidate["traits"]["geometry_method"],
                             "quality": candidate["quality"]})
        best["traits"]["candidate_quality_rejections"] = rejected
        best["traits"]["geometry_candidates"] = [
            {k: r["traits"][k] for k in ("geometry_method", "orientation_reversed_fraction",
                                         "scan_distance_median", "scan_distance_p95",
                                         "surface_support_mean")}
            for r in candidates
        ]
        spacing = float(np.median(cKDTree(retained).query(retained, k=2)[0][:, 1]))
        if ((best["traits"]["scan_distance_p95"] > 3.0 * spacing or not best["quality"]["passed"])
                and cfg.max_tps_points < cfg.refine_tps_points):
            refinement = {"initial_limit": cfg.max_tps_points, "refined_limit": cfg.refine_tps_points,
                          "accepted": False, "initial_p95": best["traits"]["scan_distance_p95"]}
            try:
                dense = fit_leaf(points, stem, normals=normals, prior=prior,
                                 cfg=replace(cfg, max_tps_points=cfg.refine_tps_points), leaf_id=leaf_id,
                                 _geometry_cache=cache)
                accepted = (not best["quality"]["passed"] or (
                    dense["traits"]["scan_distance_p95"] <= best["traits"]["scan_distance_p95"]
                    and dense["traits"]["scan_distance_median"] <= best["traits"]["scan_distance_median"]
                    and dense["traits"]["surface_support_mean"] <= best["traits"]["surface_support_mean"]))
                refinement.update(accepted=bool(accepted), refined_p95=dense["traits"]["scan_distance_p95"])
                if accepted:
                    best = dense
            except (ValueError, np.linalg.LinAlgError) as exc:
                # Refinement is optional; its failure cannot invalidate an
                # already audited coarse surface. Never relax quality gates.
                refinement["error"] = str(exc)
            best["traits"]["tps_refinement"] = refinement
        best["traits"]["quality_passed"] = best["quality"]["passed"]
        if not best["quality"]["passed"]:
            raise ValueError(f"Leaf {leaf_id}: selected surface failed the quality checks: {best['quality']}")
        return best

    mode = _geometry_mode or "axis"
    _stable_geometry = mode == "axis"
    rng = np.random.default_rng(cfg.seed)

    # 1. Dual-stage denoising (Euclidean clusters + SOR).
    p, nrm = cache["clean"]

    # 2. L1-medial skeleton + Principal Curve midrib & normal-aligned RMF initialization.
    geometry_key = ("axis" if _stable_geometry else "initial", cfg.seed)
    if geometry_key not in cache:
        cache[geometry_key] = extract_midrib(p, stem, seed=cfg.seed, stabilize=_stable_geometry)
    midrib_0, length_0, s_proj_0, spacing, midrib_lam = cache[geometry_key]
    if mode == "linear":
        # Monotone axial chart: longitudinal curvature is carried by TPS height.
        # Its constant reference tangent prevents a strip from turning inside out.
        center = np.median(p, axis=0)
        _, _, axes = np.linalg.svd(p - center, full_matrices=False)
        axial = (p - center) @ axes[0]
        ends = center + np.array([axial.min(), axial.max()])[:, None] * axes[0]
        if cKDTree(stem).query(ends)[0][0] > cKDTree(stem).query(ends)[0][1]:
            ends = ends[::-1]
        length_0 = float(np.linalg.norm(ends[1] - ends[0]))
        controls = ends[0] + np.linspace(0.0, 1.0, 4)[:, None] * (ends[1] - ends[0])
        midrib_0 = BSpline(np.r_[np.zeros(4), np.full(4, length_0)], controls, 3)
        s_proj_0 = project_points_to_curve(p, midrib_0, length_0)
        midrib_lam = 0.0
    frame_key = ("frame", geometry_key)
    if mode == "linear":
        frame_0 = build_normal_aligned_frame(midrib_0, length_0, p, s_proj_0, normals=nrm)
    else:
        if frame_key not in cache:
            cache[frame_key] = build_normal_aligned_frame(midrib_0, length_0, p, s_proj_0, normals=nrm)
        frame_0 = cache[frame_key]

    if mode == "smoothed":
        samples = np.linspace(0.0, length_0, 80)
        midrib_0, midrib_lam = fit_pspline_gcv(samples, midrib_0(samples), n_basis=12,
                                             lam_bounds=(0.1, 1e2))
        midrib_0, length_0 = reparameterize_arc_length(midrib_0)
        s_proj_0 = project_points_to_curve(p, midrib_0, length_0)
        frame_0 = build_normal_aligned_frame(midrib_0, length_0, p, s_proj_0, normals=nrm)

    # 3. Joint differentiable geometric optimization (fixes one-sided margin bias & frame roll).
    if mode != "joint":
        # The unconstrained joint shifts can reintroduce the tight bends that
        # caused a strip reversal. Keep the regularized geometry in this retry.
        midrib_obs, frame_obs, length_obs, s_proj = midrib_0, frame_0, length_0, s_proj_0
    else:
        if "joint" not in cache:
            cache["joint"] = joint_optimize_geometry(p, nrm, midrib_0, frame_0, length_0, s_proj_0, prior=prior)
        midrib_obs, frame_obs, length_obs, s_proj = cache["joint"]

    # 4. Express points in refined local physical coordinates (s, lateral r, normal h).
    _, b_p, n_p = frame_obs.evaluate(s_proj)
    res_p = p - midrib_obs(s_proj)
    lateral = np.sum(res_p * b_p, axis=1)
    normal_h = np.sum(res_p * n_p, axis=1)

    # 5. Dornbusch (2011) blade model with BIC tip-truncation selection.
    prof = extract_width_profile(s_proj, lateral, length_obs)
    blade = fit_dornbusch_blade(
        prof["s"], prof["w_obs"], length_obs, prior=prior, allow_tip_completion=cfg.allow_tip_completion
    )

    # 6. Midrib + RMF extrapolation for truncated tips (Prévot 1991 / ADEL-Maize).
    midrib_full, frame_full = extrapolate_midrib_and_frame(midrib_obs, frame_obs, length_obs, blade.length)

    # 7. Thin-Plate Spline (Duchon 1977) on bin-median representatives + GML/REML smoothing (Wahba 1985).
    # A global species outline must not clip a surviving measured margin. Preserve
    # local envelopes; retain the prior alone where observations are missing.
    width_s = np.r_[0.0, prof["s"], length_obs]
    observed_width = 2.0 * np.maximum(prof["w_left"], prof["w_right"])
    width_extra = np.maximum(observed_width - blade.width(prof["s"]), 0.0)
    if mode in ("axis", "linear") and len(width_extra) >= 6:
        # Do not turn noisy section quantiles into a scalloped blade boundary.
        smooth_extra, _ = fit_pspline_gcv(prof["s"], width_extra, n_basis=8,
                                         lam_bounds=(0.1, 1e2))
        width_extra = np.maximum(smooth_extra(prof["s"]), 0.0)
    width_extra = np.r_[width_extra[0], width_extra, 0.0]
    extra_at_p = PchipInterpolator(width_s, width_extra)(s_proj)
    w_at_p = np.maximum(blade.width(s_proj) + extra_at_p, blade.w_max * 0.01)
    u_norm = s_proj / length_obs
    v_norm = np.clip(2.0 * lateral / w_at_p, -1.25, 1.25)
    tps_xy, tps_h = _aggregate_tps_bins(u_norm, v_norm, normal_h, cfg.max_tps_points, rng)
    tps = ThinPlateSplineGCV.fit(tps_xy, tps_h)

    model = MaizeLeafModel(midrib_full, frame_full, blade, tps, length_obs, width_s, width_extra)
    orientation = surface_orientation_diagnostics(model, 241, 61)

    # 7. Evaluate continuous surface on non-degenerate fan-apex (s, v) mesh.
    s_grid = np.linspace(0.0, blade.length, cfg.nu)
    grid_sv, faces = leaf_mesh_grid(blade.length, cfg.nu, cfg.nv)
    vertices = model.evaluate(grid_sv)

    # 8. Region classification in continuous physical (s, r) blade coordinates + 3D point spacing.
    phys_obs = np.column_stack([s_proj, lateral])
    w_grid = model.width(grid_sv[:, 0])
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
    center_s = np.unique(np.r_[np.linspace(0.0, blade.length, 500), length_obs])
    center_xyz = model.evaluate(np.column_stack([center_s, np.zeros(len(center_s))]))
    center_arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(center_xyz, axis=0), axis=1))]
    observed_center_length = float(np.interp(length_obs, center_s, center_arc))
    completed_center_length = float(center_arc[-1])
    t_base = center_xyz[1] - center_xyz[0]
    t_base = t_base / max(float(np.linalg.norm(t_base)), 1e-12)
    base_inclination_deg = float(np.degrees(np.arccos(np.clip(t_base[2], -1.0, 1.0))))

    traits = dict(
        leaf_id=int(leaf_id) if leaf_id is not None else None,
        raw_points=len(points),
        fitted_points=len(p),
        observed_length=observed_center_length,
        completed_length=completed_center_length,
        tip_completion_length=max(0.0, completed_center_length - observed_center_length),
        parameter_observed_length=float(length_obs),
        parameter_completed_length=float(blade.length),
        max_width=float(np.max(model.width(s_grid))),
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
        tps_training_points=len(tps.xy),
        tip_completion_status=getattr(blade, "completion_status", "legacy"),
        support_radius=float(support_radius),
        geometry_method=mode,
        orientation_reversed_fraction=orientation["reversed_fraction"],
        minimum_normal_cosine=orientation["minimum_normal_cosine"],
    )
    return dict(
        model=model,
        vertices=vertices,
        faces=faces,
        regions=regions,
        face_regions=face_regions,
        sv=grid_sv,
        points=points,
        midrib=model.evaluate(np.column_stack([s_grid, np.zeros(len(s_grid))])),
        s=s_grid,
        width=model.width(s_grid),
        blade=blade,
        traits=traits,
    )
