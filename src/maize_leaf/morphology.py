"""Maize leaf blade morphology priors and analytical completion models.

References:
- Dornbusch et al. (2011) Annals of Botany 107(5): 865-873:
  Empirical piecewise power-law cereal leaf shape model (distal + basal segments).
- Stewart & Dwyer (1999) Crop Science 39: 422-427; Montgomery (1911):
  Maize leaf area form factor alpha = A / (L * W_max) ~= 0.734 - 0.75.
- Prévot, Aries & Monestiez (1991) Agronomie 11(6): 491-503;
  Fournier & Andrieu (1998) ADEL-Maize: Vertical-plane midrib inclination trajectory.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import least_squares
from .geometry import MovingFrame, double_reflection_rmf, fit_pspline_gcv, unit


@dataclass
class MaizeShapePrior:
    """Species prior on dimensionless Dornbusch (2011) maize leaf shape parameters.

    Configured engineering defaults can be empirically shrunk toward intact-leaf fits;
    they are not a claim that each constant was directly measured in a cited paper.
    """
    xm_mean: float = 0.36
    xm_std: float = 0.10
    klig_mean: float = 0.62
    klig_std: float = 0.16
    c1_mean: float = 1.85
    c1_std: float = 0.45
    c2_mean: float = 2.00
    c2_std: float = 0.55


@dataclass
class DornbuschBlade:
    """C1 piecewise power-law leaf width model (Dornbusch et al. 2011)."""
    length: float
    w_max: float
    xm: float
    k_lig: float
    c1: float
    c2: float

    def width(self, s):
        """Blade width w(s) at physical arc length s in [0, length]."""
        s = np.asarray(s, dtype=float)
        u = np.clip(s / self.length, 0.0, 1.0)
        basal = self.w_max * (1.0 - (1.0 - self.k_lig) * ((self.xm - np.minimum(u, self.xm)) / self.xm) ** self.c2)
        distal = self.w_max * (1.0 - ((np.maximum(u, self.xm) - self.xm) / (1.0 - self.xm)) ** self.c1)
        return np.where(u <= self.xm, basal, distal)

    @property
    def form_factor(self):
        """Exact integral form factor alpha = integral(w(s) ds) / (L * w_max)."""
        a_basal = self.xm * (1.0 - (1.0 - self.k_lig) / (self.c2 + 1.0))
        a_distal = (1.0 - self.xm) * (self.c1 / (self.c1 + 1.0))
        return float(a_basal + a_distal)

    def to_dict(self):
        d = asdict(self)
        d["form_factor_alpha"] = self.form_factor
        return d


def extract_width_profile(s_proj, lateral, length_obs, n_bins=32):
    """Extract independent left/right half-width envelopes along arc length."""
    edges = np.linspace(0.0, length_obs, n_bins + 1)
    s_mid, w_left, w_right, counts = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (s_proj >= a) & (s_proj <= b)
        if np.sum(sel) < 6:
            continue
        lat = lateral[sel]
        s_mid.append(0.5 * (a + b))
        w_left.append(float(-np.quantile(lat, 0.02)))
        w_right.append(float(np.quantile(lat, 0.98)))
        counts.append(int(np.sum(sel)))
    s_mid = np.asarray(s_mid)
    w_left = np.maximum(np.asarray(w_left), 0.0)
    w_right = np.maximum(np.asarray(w_right), 0.0)
    # Full observed width; only double the intact side when the opposite side is nearly absent (>75% deficit).
    w_sum = w_left + w_right
    asym = np.abs(w_left - w_right) / np.maximum(np.maximum(w_left, w_right), 1e-9)
    w_obs = np.where(asym > 0.75, 2.0 * np.maximum(w_left, w_right), w_sum)
    center_shift = 0.5 * (w_right - w_left)
    return dict(s=s_mid, w_left=w_left, w_right=w_right, w_obs=w_obs, center_shift=center_shift, counts=np.asarray(counts))


def fit_dornbusch_blade(s_obs, w_obs, length_obs, prior=None, allow_tip_completion=True):
    """Fit Dornbusch (2011) blade model via MAP soft-L1 least squares + identifiable tip inversion.

    To avoid (L, c1) collinearity blow-up when the distal tip is truncated, the truncated model M1
    anchors dimensionless shape parameters (xm, k_lig, c1, c2) tightly to the maize species prior
    while solving for physical total length L >= L_obs from the observed distal width ratio, and
    accepts M1 only under positive BIC evidence (delta BIC > 4, Kass & Raftery 1995).
    """
    prior = prior or MaizeShapePrior()
    n = len(s_obs)
    w_max0 = float(np.quantile(w_obs, 0.95))
    scale = max(w_max0, 1e-6)
    xm_obs = float(s_obs[int(np.argmax(w_obs))] / max(length_obs, 1e-9))
    tip_ratio = float(np.median(w_obs[-min(3, n):]) / scale)

    # Analytical upper bound on missing tip length implied by Dornbusch distal power law at observed tip_ratio:
    # w(u_obs)/w_max = 1 - ((u_obs - xm)/(1 - xm))^c1  =>  u_obs = xm + (1 - xm)*(1 - tip_ratio)^(1/c1)
    u_obs_est = prior.xm_mean + (1.0 - prior.xm_mean) * (max(1.0 - min(tip_ratio, 0.85), 0.05) ** (1.0 / prior.c1_mean))
    L_prior = float(np.clip(length_obs / max(u_obs_est, 0.65), length_obs, length_obs * 1.45))

    def _fit(free_length):
        if free_length:
            L_hi = min(length_obs * 1.45, L_prior * 1.15)
            x0 = np.array([L_prior, w_max0, prior.xm_mean, np.clip(float(w_obs[0] / scale), 0.25, 0.95),
                           prior.c1_mean, prior.c2_mean])
            lb = np.array([length_obs, w_max0 * 0.85, max(0.20, prior.xm_mean - 0.12), 0.15,
                           max(1.25, prior.c1_mean - 0.45), max(1.25, prior.c2_mean - 0.55)])
            ub = np.array([L_hi, w_max0 * 1.20, min(0.55, prior.xm_mean + 0.12), 0.98,
                           min(3.0, prior.c1_mean + 0.45), min(3.5, prior.c2_mean + 0.55)])
            prior_w = 0.45
        else:
            x0 = np.array([w_max0, np.clip(xm_obs, 0.18, 0.58),
                           np.clip(float(w_obs[0] / scale), 0.25, 0.95), prior.c1_mean, prior.c2_mean])
            lb = np.array([w_max0 * 0.80, 0.14, 0.15, 1.05, 1.05])
            ub = np.array([w_max0 * 1.25, 0.65, 0.98, 3.8, 4.5])
            prior_w = 0.08
        x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)

        def residuals(params):
            if free_length:
                L, w_max, xm, k_lig, c1, c2 = params
            else:
                L = length_obs
                w_max, xm, k_lig, c1, c2 = params
            pred = DornbuschBlade(L, w_max, xm, k_lig, c1, c2).width(s_obs)
            data_res = (pred - w_obs) / scale
            reg = prior_w * np.array([
                (xm - prior.xm_mean) / prior.xm_std,
                (k_lig - prior.klig_mean) / prior.klig_std,
                (c1 - prior.c1_mean) / prior.c1_std,
                (c2 - prior.c2_mean) / prior.c2_std,
            ])
            return np.r_[data_res, reg]

        sol = least_squares(residuals, x0, bounds=(lb, ub), loss="soft_l1", f_scale=0.12, max_nfev=400)
        if free_length:
            blade = DornbuschBlade(*[float(v) for v in sol.x])
            k_params = 6
        else:
            blade = DornbuschBlade(float(length_obs), *[float(v) for v in sol.x])
            k_params = 5
        err = (blade.width(s_obs) - w_obs) / scale
        rss = max(float(np.sum(err ** 2)), 1e-12)
        bic = n * np.log(rss / n) + k_params * np.log(n)
        # An optimum pressed against the imposed length cap does not identify a
        # missing tip. Returning that cap as a measured length creates large,
        # unstable extensions after even an interior margin is removed.
        blade.completion_at_length_bound = bool(free_length and
            (L_hi - blade.length) <= 1e-3 * (L_hi - length_obs))
        return blade, bic

    blade_intact, bic_intact = _fit(free_length=False)
    blade_intact.completion_status = "disabled" if not allow_tip_completion else "not_supported"
    if not allow_tip_completion or tip_ratio <= 0.22:
        return blade_intact

    blade_ext, bic_ext = _fit(free_length=True)
    # Require positive BIC evidence (delta BIC > 4, Kass & Raftery 1995) before extending L.
    if bic_ext + 4.0 < bic_intact:
        blade_ext.completion_status = "boundary_limited" if blade_ext.completion_at_length_bound else "estimated"
        return blade_ext
    return blade_intact


def calibrate_prior_from_leaves(blades, base_prior=None, min_samples=10):
    """Empirical Bayes calibration shrinking toward literature maize prior (Dornbusch 2011)."""
    lit = base_prior or MaizeShapePrior()
    valid = [b for b in blades if 0.20 < b.xm < 0.55 and 1.2 < b.c1 < 3.2 and 0.65 <= b.form_factor <= 0.82]
    n = len(valid)
    if n < 3:
        return lit
    # Shrinkage weight w = n / (n + min_samples) prevents small noisy subsets from collapsing the prior.
    w = n / (n + min_samples)
    xm = np.array([b.xm for b in valid])
    klig = np.array([b.k_lig for b in valid])
    c1 = np.array([b.c1 for b in valid])
    c2 = np.array([b.c2 for b in valid])
    return MaizeShapePrior(
        xm_mean=float(w * np.median(xm) + (1.0 - w) * lit.xm_mean),
        xm_std=float(max(w * np.std(xm) + (1.0 - w) * lit.xm_std, 0.06)),
        klig_mean=float(w * np.median(klig) + (1.0 - w) * lit.klig_mean),
        klig_std=float(max(w * np.std(klig) + (1.0 - w) * lit.klig_std, 0.10)),
        c1_mean=float(w * np.median(c1) + (1.0 - w) * lit.c1_mean),
        c1_std=float(max(w * np.std(c1) + (1.0 - w) * lit.c1_std, 0.30)),
        c2_mean=float(w * np.median(c2) + (1.0 - w) * lit.c2_mean),
        c2_std=float(max(w * np.std(c2) + (1.0 - w) * lit.c2_std, 0.35)),
    )


def extrapolate_midrib_and_frame(curve, frame, length_obs, length_total, n_grid=240):
    """Engineering damped-inclination extension from L_obs to L_total.

    Fit the distal 25% inclination slope, limit upward bending, and integrate
    unit tangents. This is not a verbatim implementation of Prevot or ADEL-Maize.
    """
    if length_total <= length_obs + 1e-6:
        return curve, frame

    s_obs_grid = np.linspace(0.0, length_obs, 200)
    p_obs = curve(s_obs_grid)
    t_obs = unit(curve(s_obs_grid, nu=1))

    # Estimate distal inclination and curvature rate on last 25% of observed arc length.
    tail_mask = s_obs_grid >= 0.75 * length_obs
    s_tail = s_obs_grid[tail_mask] - length_obs
    t_tail = t_obs[tail_mask]
    theta_tail = np.arcsin(np.clip(t_tail[:, 2], -1.0, 1.0))
    horiz_tail = t_tail[:, :2]
    az_vec = unit(np.mean(horiz_tail, axis=0))

    # Linear fit of inclination theta(s) = theta_end + kappa * (s - L_obs), capped at vertical droop (-pi/2).
    kappa = float(np.polyfit(s_tail, theta_tail, 1)[0]) if len(s_tail) >= 3 else 0.0
    # Maize tips droop downward under gravity; prevent unphysical upward curl in extrapolation.
    kappa = min(kappa, 0.0)
    theta_end = float(theta_tail[-1])

    ext_len = length_total - length_obs
    n_ext = max(12, int(np.ceil(ext_len / max(length_obs / 200.0, 1e-6))))
    ds = ext_len / n_ext
    s_ext = length_obs + np.arange(1, n_ext + 1) * ds

    p_ext = []
    t_ext = []
    p_curr = p_obs[-1].copy()
    for s_val in s_ext:
        delta = s_val - length_obs
        # Exponentially damped downward curvature toward vertical droop.
        theta = max(-0.48 * np.pi, theta_end + kappa * delta * np.exp(-0.5 * delta / max(length_obs, 1e-6)))
        t_vec = unit(np.array([np.cos(theta) * az_vec[0], np.cos(theta) * az_vec[1], np.sin(theta)]))
        p_curr = p_curr + ds * t_vec
        p_ext.append(p_curr.copy())
        t_ext.append(t_vec)

    s_all = np.r_[s_obs_grid, s_ext]
    p_all = np.vstack([p_obs, np.asarray(p_ext)])
    ext_curve, _ = fit_pspline_gcv(s_all, p_all, n_basis=min(22, len(curve.c) + 6), lam_bounds=(1e-8, 1e-2))

    # Extend twist-free frame smoothly from the observed endpoint frame via Double Reflection RMF.
    t_end, _, n_end = frame.evaluate(length_obs)
    p_ext_full = np.vstack([p_obs[-1:], np.asarray(p_ext)])
    t_ext_full = np.vstack([t_end, np.asarray(t_ext)])
    T_e, B_e, N_e = double_reflection_rmf(p_ext_full, t_ext_full, n0=n_end[0])

    s_f_all = np.r_[frame.s_grid, s_ext]
    T_all = np.vstack([frame.T_grid, T_e[1:]])
    B_all = np.vstack([frame.B_grid, B_e[1:]])
    N_all = np.vstack([frame.N_grid, N_e[1:]])
    return ext_curve, MovingFrame(s_f_all, T_all, B_all, N_all)
