"""Joint robust nonlinear fitting of midrib, roll frame, width, and camber.

Simultaneously optimizes:
  - Midrib lateral & normal B-spline offsets (delta_b, delta_n)
  - Moving-frame roll angle B-spline (delta_phi)
  - Smooth bivariate camber modes q(u, v) = a1(u)*v + a2(u)*v^2
  - Dornbusch (2011) blade shape parameters (w_max, xm, k_lig, c1, c2)
against 3D point-to-surface residuals, scanner surface-normal consistency,
one-sided-robust margin envelope loss, and midrib bending energy.
"""
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from .geometry import (
    build_normal_aligned_frame,
    fit_pspline_gcv,
    project_points_to_curve,
    reparameterize_arc_length,
)
from .morphology import DornbuschBlade, MaizeShapePrior, extract_width_profile, fit_dornbusch_blade


def _bspline_basis(u, n_basis, degree):
    u = np.clip(np.asarray(u, dtype=float), 0.0, 1.0)
    k = min(degree, max(1, n_basis - 1))
    knots = np.r_[
        np.repeat(0.0, k + 1),
        np.linspace(0.0, 1.0, n_basis - k + 1)[1:-1],
        np.repeat(1.0, k + 1),
    ]
    B = BSpline.design_matrix(u, knots, k).toarray()
    # Analytic derivative basis with respect to u via spline basis differentiation.
    I = np.eye(n_basis)
    dB = np.column_stack([BSpline(knots, I[:, j], k).derivative(1)(u) for j in range(n_basis)])
    return B, dB, knots


def joint_optimize_geometry(points, normals, midrib0, frame0, length0, s_proj0, prior=None):
    """Jointly refine midrib, roll frame, and blade width over observed span [0, length0]."""
    prior = prior or MaizeShapePrior()
    if len(points) < 60 or length0 <= 1e-8:
        return midrib0, frame0, length0, s_proj0

    # Initial projection in frame0 for width initialization & anchor edge points.
    _, b_all, n_all = frame0.evaluate(s_proj0)
    res_all = points - midrib0(s_proj0)
    lat_all = np.sum(res_all * b_all, axis=1)
    prof0 = extract_width_profile(s_proj0, lat_all, length0)
    if len(prof0["s"]) < 5:
        return midrib0, frame0, length0, s_proj0
    blade0 = fit_dornbusch_blade(prof0["s"], prof0["w_obs"], length0, prior=prior, allow_tip_completion=False)
    w_scale = max(blade0.w_max, 1e-6)

    # Stratified subsample along s_proj0 for fast vectorized optimization.
    order = np.argsort(s_proj0)
    n_opt = min(len(points), 1200)
    sel = order[np.linspace(0, len(points) - 1, n_opt, dtype=int)]
    p_s = points[sel]
    n_scan = normals[sel] if normals is not None else None
    u_s = np.clip(s_proj0[sel] / length0, 0.0, 1.0)
    c0_s = midrib0(u_s * length0)
    t0_s, b0_s, n0_s = frame0.evaluate(u_s * length0)

    # Fixed 3D edge anchors for piecewise-smooth margin residuals.
    nb = 22
    edges = np.linspace(0.0, length0, nb + 1)
    pL_list, pR_list, u_bin_list = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        mask = (s_proj0 >= a) & (s_proj0 <= b)
        if np.sum(mask) < 8:
            continue
        pts_b = points[mask]
        lats_b = lat_all[mask]
        ord_b = np.argsort(lats_b)
        m_b = len(ord_b)
        i_l = ord_b[max(0, int(0.02 * m_b)) : max(1, int(0.05 * m_b))]
        i_r = ord_b[min(m_b - 1, int(0.95 * m_b)) : max(min(m_b, int(0.98 * m_b)), min(m_b - 1, int(0.95 * m_b)) + 1)]
        pL_list.append(np.mean(pts_b[i_l], axis=0))
        pR_list.append(np.mean(pts_b[i_r], axis=0))
        u_bin_list.append(float(np.median(s_proj0[mask]) / length0))
    if len(u_bin_list) < 5:
        return midrib0, frame0, length0, s_proj0

    pL = np.asarray(pL_list)
    pR = np.asarray(pR_list)
    u_bin = np.asarray(u_bin_list)
    c0_bin = midrib0(u_bin * length0)
    _, b0_bin, n0_bin = frame0.evaluate(u_bin * length0)

    # Basis matrices (constant throughout optimization).
    km, kp, kq = 8, 6, 5
    Bm_s, _, _ = _bspline_basis(u_s, km, 3)
    Bm_bin, _, _ = _bspline_basis(u_bin, km, 3)
    Bp_s, _, _ = _bspline_basis(u_s, kp, 3)
    Bp_bin, _, _ = _bspline_basis(u_bin, kp, 3)
    Bq_s, dBq_s, _ = _bspline_basis(u_s, kq, 2)

    u_grid = np.linspace(0.0, 1.0, 40)
    Bm_g, _, _ = _bspline_basis(u_grid, km, 3)
    c0_g = midrib0(u_grid * length0)
    _, b0_g, n0_g = frame0.evaluate(u_grid * length0)
    D2_g = np.diff(np.eye(len(u_grid)), n=2, axis=0)
    D2_p = np.diff(np.eye(kp), n=2, axis=0)
    D2_q = np.diff(np.eye(kq), n=2, axis=0)

    # Parameter vector: [db(8), dn(8), dphi(6), q_v(5), q_v2(5), w_max, xm, k_lig, c1, c2] (37 params).
    x0 = np.r_[
        np.zeros(km),
        np.zeros(km),
        np.zeros(kp),
        np.zeros(kq),
        np.zeros(kq),
        blade0.w_max,
        blade0.xm,
        blade0.k_lig,
        blade0.c1,
        blade0.c2,
    ]
    shift_lim = 0.75 * w_scale
    lb = np.r_[
        np.full(km, -shift_lim),
        np.full(km, -0.45 * w_scale),
        np.full(kp, -0.55),
        np.full(kq, -0.40 * w_scale),
        np.full(kq, -0.40 * w_scale),
        0.70 * w_scale,
        0.15,
        0.15,
        1.10,
        1.10,
    ]
    ub = np.r_[
        np.full(km, shift_lim),
        np.full(km, 0.45 * w_scale),
        np.full(kp, 0.55),
        np.full(kq, 0.40 * w_scale),
        np.full(kq, 0.40 * w_scale),
        1.45 * w_scale,
        0.60,
        0.98,
        3.60,
        4.20,
    ]
    x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)

    # Balance data residuals (sqrt(n_opt)) vs bin-level margin and smoothness priors.
    w_pt = 1.0 / np.sqrt(max(n_opt, 1))
    w_nrm = 0.35 / np.sqrt(max(n_opt, 1))
    w_bin = 1.25 / np.sqrt(max(len(u_bin), 1))
    s_bin = u_bin * length0

    def residuals(x):
        db = x[:km]
        dn = x[km : 2 * km]
        dp = x[2 * km : 2 * km + kp]
        qv = x[2 * km + kp : 2 * km + kp + kq]
        qv2 = x[2 * km + kp + kq : 2 * km + kp + 2 * kq]
        w_max, xm, k_lig, c1, c2 = x[-5:]

        blade = DornbuschBlade(length0, w_max, xm, k_lig, c1, c2)

        # 1. Surface height & normal residuals over sampled points.
        off_b = Bm_s @ db
        off_n = Bm_s @ dn
        c_s = c0_s + off_b[:, None] * b0_s + off_n[:, None] * n0_s
        phi_s = Bp_s @ dp
        cp, sp = np.cos(phi_s)[:, None], np.sin(phi_s)[:, None]
        b_s = cp * b0_s + sp * n0_s
        n_s = -sp * b0_s + cp * n0_s

        diff_s = p_s - c_s
        r_s = np.sum(diff_s * b_s, axis=1)
        h_s = np.sum(diff_s * n_s, axis=1)
        w_s = np.maximum(blade.width(u_s * length0), 0.02 * w_scale)
        v_s = np.clip(2.0 * r_s / w_s, -1.25, 1.25)

        a1 = Bq_s @ qv
        a2 = Bq_s @ qv2
        q_pred = a1 * v_s + a2 * (v_s ** 2)
        res_surf = w_pt * (h_s - q_pred) / w_scale

        # Analytical normal alignment using scan unit normals (invariant to +/- sign flips).
        if n_scan is not None:
            dq_dv = a1 + 2.0 * a2 * v_s
            dq_du = (dBq_s @ qv) * v_s + (dBq_s @ qv2) * (v_s ** 2)
            slope_b = 2.0 * dq_dv / w_s
            slope_t = dq_du / length0
            n_surf = n_s - slope_b[:, None] * b_s - slope_t[:, None] * t0_s
            n_surf = n_surf / np.maximum(np.linalg.norm(n_surf, axis=1, keepdims=True), 1e-12)
            # ||n_surf x n_scan|| = |sin(angle)|; invariant to normal sign.
            res_nrm = w_nrm * np.linalg.norm(np.cross(n_surf, n_scan), axis=1)
        else:
            res_nrm = np.empty(0)

        # 2. Asymmetric differentiable margin envelope residuals across longitudinal bins.
        c_bin = c0_bin + (Bm_bin @ db)[:, None] * b0_bin + (Bm_bin @ dn)[:, None] * n0_bin
        phi_b = Bp_bin @ dp
        b_bin = np.cos(phi_b)[:, None] * b0_bin + np.sin(phi_b)[:, None] * n0_bin
        rL = np.sum((pL - c_bin) * b_bin, axis=1)
        rR = np.sum((pR - c_bin) * b_bin, axis=1)
        half_w = 0.5 * blade.width(s_bin)

        # Enclosure penalty: observed points must not lie outside +/- w/2.
        viol_L = np.maximum(0.0, -half_w - rL) / w_scale
        viol_R = np.maximum(0.0, rR - half_w) / w_scale
        # Surviving-side envelope match: whichever margin extends farther out anchors half_w.
        surv_err = np.where(-rL >= rR, rL + half_w, half_w - rR) / w_scale
        # Robust saturated bilateral balance: centers midrib where both margins exist,
        # but saturates on one-sided notches so notch doesn't pull midrib inward.
        bilat_L = 0.12 * np.tanh(((rL + half_w) / w_scale) / 0.12)
        bilat_R = 0.12 * np.tanh(((half_w - rR) / w_scale) / 0.12)
        res_edges = w_bin * np.r_[1.6 * viol_L, 1.6 * viol_R, 1.2 * surv_err, 0.35 * bilat_L, 0.35 * bilat_R]

        # 3. Midrib bending smoothness, gentle shift anchor, roll smoothness, camber prior, Dornbusch prior.
        c_g = c0_g + (Bm_g @ db)[:, None] * b0_g + (Bm_g @ dn)[:, None] * n0_g
        res_bend = (1.6 / w_scale) * (D2_g @ c_g).ravel()
        res_anchor = np.r_[0.10 * db / w_scale, 0.25 * dn / w_scale, 0.20 * dp, 0.40 * (D2_p @ dp)]
        res_camber = np.r_[
            0.35 * qv / w_scale,
            0.10 * qv2 / w_scale,
            0.35 * (D2_q @ qv) / w_scale,
            0.35 * (D2_q @ qv2) / w_scale,
        ]
        res_prior = 0.18 * np.array([
            (xm - prior.xm_mean) / prior.xm_std,
            (k_lig - prior.klig_mean) / prior.klig_std,
            (c1 - prior.c1_mean) / prior.c1_std,
            (c2 - prior.c2_mean) / prior.c2_std,
        ])
        return np.r_[res_surf, res_nrm, res_edges, res_bend, res_anchor, res_camber, res_prior]

    sol = least_squares(residuals, x0, bounds=(lb, ub), loss="soft_l1", f_scale=0.08, max_nfev=65)
    db = sol.x[:km]
    dn = sol.x[km : 2 * km]

    c_refined = c0_g + (Bm_g @ db)[:, None] * b0_g + (Bm_g @ dn)[:, None] * n0_g
    s_g = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(c_refined, axis=0), axis=1))]
    midrib_opt, _ = fit_pspline_gcv(s_g, c_refined, n_basis=min(12, len(midrib0.c) + 2), lam_bounds=(1e-5, 1e2))
    midrib_opt, length_opt = reparameterize_arc_length(midrib_opt)
    s_proj_opt = project_points_to_curve(points, midrib_opt, length_opt)
    frame_opt = build_normal_aligned_frame(midrib_opt, length_opt, points, s_proj_opt, normals=normals)
    return midrib_opt, frame_opt, length_opt, s_proj_opt
