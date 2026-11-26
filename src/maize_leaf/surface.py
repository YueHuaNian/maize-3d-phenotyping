"""连续叶面拟合、观测支撑域限制及兼容旧模型的序列化。

Thin-Plate Spline (TPS) surface with spectral GML smoothing and leaf model.

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
    """集中保存去噪、几何拟合、采样和补全策略参数。"""
    nu: int = 90
    nv: int = 25
    max_tps_points: int = 700
    refine_tps_points: int = 700
    allow_tip_completion: bool = True
    seed: int = 2026
    conservative_support: bool = False
    include_joint_candidate: bool = False


def tps_kernel(r):
    """计算二维薄板样条径向基核，零距离处连续取零。

    2D Thin-Plate Spline Green's function phi(r) = r^2 * log(r) (Duchon 1977)."""
    r2 = r * r
    return np.where(r > 1e-12, r2 * np.log(np.maximum(r, 1e-12)), 0.0)


class ThinPlateSplineGCV:
    """以谱分解 GML 选择平滑强度的二维薄板样条。

    2D TPS with spectral GML smoothing; the class name is retained for NPZ/API compatibility."""

    def __init__(self, xy, w, a, lam, edf):
        """初始化薄板样条的采样与平滑参数。"""
        self.xy = np.asarray(xy, dtype=float)
        self.w = np.asarray(w, dtype=float)
        self.a = np.asarray(a, dtype=float)
        self.lam = float(lam)
        self.edf = float(edf)

    @classmethod
    def fit(cls, xy, values, lam_bounds=(1e-7, 1e2)):
        """构造薄板基函数并以谱分解 GML 拟合平滑叶面高度。"""
        xy = np.asarray(xy, dtype=float)
        y = np.asarray(values, dtype=float)
        m = len(xy)
        K = tps_kernel(cdist(xy, xy))
        P = np.column_stack([np.ones(m), xy])

        # 对 P 做 QR 分解，投影到 P 转置的零空间，参见 Wahba 1990。
        Q, R = np.linalg.qr(P, mode="complete")
        Q1, Q2 = Q[:, :3], Q[:, 3:]
        R1 = R[:3, :]

        # 对 M = Q2.T @ K @ Q2 做对称特征分解；其在约束空间正定。
        M = Q2.T @ K @ Q2
        M = 0.5 * (M + M.T)
        eigvals, U = np.linalg.eigh(M)
        eigvals = np.maximum(eigvals, 1e-12)
        z = U.T @ (Q2.T @ y)

        # 使用广义最大似然 GML / REML 准则，参见 Wahba 1985：
        # 由数据选择平滑强度；GML 和 GCV 均不能保证避免过拟合。
        m_eff = len(eigvals)

        def gml(log_lam):
            """计算谱分解后的广义最大似然平滑目标。"""
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
        """在二维参数坐标计算薄板样条预测高度。"""
        xy_query = np.atleast_2d(np.asarray(xy_query, dtype=float))
        K = tps_kernel(cdist(xy_query, self.xy))
        P = np.column_stack([np.ones(len(xy_query)), xy_query])
        return K @ self.w + P @ self.a


def leaf_mesh_grid(length, nu, nv, close_tip=True):
    """生成叶片三角网格；未观测叶尖以开放横截面结尾。

    Fan-apex mesh, or an open terminal cross-section when close_tip is false."""
    s_rows = np.linspace(0.0, length, nu)
    if close_tip:
        s_rows = s_rows[:-1]
    v_cols = np.linspace(-1.0, 1.0, nv)
    ss, vv = np.meshgrid(s_rows, v_cols, indexing="ij")
    sv = np.column_stack([ss.ravel(), vv.ravel()])
    if close_tip:
        sv = np.vstack([sv, [[length, 0.0]]])

    faces = []
    for a in range(len(s_rows) - 1):
        for b in range(nv - 1):
            k = a * nv + b
            faces.extend(((k, k + nv, k + 1), (k + 1, k + nv, k + nv + 1)))
    # 以扇形三角面将最后横截面收束到单个叶尖顶点。
    tip_idx = (nu - 1) * nv
    base_row = (nu - 2) * nv
    if close_tip:
        for b in range(nv - 1):
            faces.append((base_row + b, tip_idx, base_row + b + 1))
    return sv, np.asarray(faces, dtype=np.int32)


def triangle_areas(vertices, faces):
    """用三角形两边的叉积计算各面元面积。"""
    tri = vertices[faces]
    return np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1) / 2.0


def surface_orientation_diagnostics(model, nu=121, nv=31):
    """对照局部标架法向检查曲面坐标翻转。

    Detect local chart reversals against the moving-frame normal.

    A small point-to-surface distance or nonzero triangle area does not rule out
    a folded strip. This samples its signed Jacobian away from the apex; it is
    not a global triangle-intersection certificate.
    """
    s = np.linspace(0.01, 0.99, nu) * model.length
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
    """封装连续参数叶面、活动域、支撑包络及模型存取。

    Continuous parametric maize leaf surface:
    S(s, v) = C(s) + (v * w(s) / 2) * b(s) + q(s, v) * n(s),  s in [0, L], v in [-1, 1].
    """

    def __init__(self, midrib, frame, blade, tps, length_obs, width_s=None, width_extra=None,
                 apex_taper=True, surface_tip_extension=True, support_s=None,
                 support_left=None, support_right=None, domain_end=None):
        """组合中脉、移动标架、叶宽和残差曲面。"""
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
        self.domain_end = float(blade.length if domain_end is None else domain_end)
        self.support_s = np.asarray([0., length_obs] if support_s is None else support_s, dtype=float)
        self.support_left = np.asarray(np.ones(len(self.support_s)) if support_left is None else support_left, dtype=float)
        self.support_right = np.asarray(np.ones(len(self.support_s)) if support_right is None else support_right, dtype=float)
        self._support_left = PchipInterpolator(self.support_s, self.support_left)
        self._support_right = PchipInterpolator(self.support_s, self.support_right)

    @property
    def length(self):
        """返回当前活动参数范围；历史先验叶长仍保存在 blade 中。

        Active parameter extent; the original blade prior remains available separately."""
        return self.domain_end

    @property
    def open_tip(self):
        """判断活动域末端是否应保留开放横截面。"""
        return self.length < self.blade.length - 1e-8

    def mesh_grid(self, nu=90, nv=25):
        """按当前活动域与叶尖状态生成一致的网格。"""
        return leaf_mesh_grid(self.length, nu, nv, close_tip=not self.open_tip)

    def _reference_width(self, s):
        """计算参考叶宽，用于活动横向域的归一化。"""
        extra = np.nan_to_num(self._width_extra(np.asarray(s, dtype=float)), nan=0.0)
        return self.blade.width(s) + np.maximum(extra, 0.0)

    def _lateral_scales(self, s):
        """插值左右半宽缩放，保留观测不对称性。"""
        s = np.clip(np.asarray(s, dtype=float), self.support_s[0], self.support_s[-1])
        return np.clip(self._support_left(s), .01, 1.), np.clip(self._support_right(s), .01, 1.)

    def lateral_coordinate(self, s, v):
        """将归一化横向参数转换为物理横向坐标。"""
        left, right = self._lateral_scales(s)
        return .5 * self._reference_width(s) * v * np.where(np.asarray(v) < 0, left, right)

    def lateral_parameter(self, s, lateral):
        """将物理横向坐标转换为归一化参数。"""
        left, right = self._lateral_scales(s)
        return 2 * lateral / np.maximum(self._reference_width(s) * np.where(lateral < 0, left, right), 1e-12)

    def width(self, s):
        """返回形态先验经左右观测包络约束后的实际叶宽。

        Morphology prior plus smooth observed-envelope support on the scanned span."""
        left, right = self._lateral_scales(s)
        return self._reference_width(s) * .5 * (left + right)

    def evaluate(self, sv):
        """将纵向与归一化横向参数映射为三维曲面点。"""
        sv = np.atleast_2d(np.asarray(sv, dtype=float))
        s, v = sv[:, 0], sv[:, 1]
        if np.any(s < -1e-8) or np.any(s > self.length + 1e-8):
            raise ValueError("Surface query outside its active supported parameter domain")
        left, right = self._lateral_scales(s)
        v = v * np.where(v < 0, left, right)
        _, b, n = self.frame.evaluate(s)
        w = self._reference_width(s)
        # TPS 在归一化 (s/L_obs, v) 上训练，向叶尖随叶宽平滑衰减。
        u_tps = np.clip(s / self.length_obs, 0.0, 1.0)
        q = self.tps.evaluate(np.column_stack([u_tps, v]))
        if self.apex_taper:
            # 零宽叶尖必须是一个点，不能保留横向 TPS 高度曲线。
            # 保留中心高度，同时收束横向起伏。
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
            # 延长实际曲面中心 S(s,0)；参考曲线为直线时，
            # 大部分纵向弯曲可能由该中心的 TPS 高度承载。
            # 若将绝对 TPS 高度衰减为零，叶尖会偏向参考直线，
            # 从而无法延续已测得的叶身走向。
            if self._tip_center is None:
                self._build_tip_center()
            end_v = self.tps.evaluate(np.column_stack([np.ones(outside.sum()), v[outside]]))
            end_zero = float(self.tps.evaluate([[1.0, 0.0]])[0])
            end_width = max(float(self.width(self.length_obs)), 1e-12)
            relief_taper = np.clip(w[outside] / end_width, 0.0, 1.0)
            # 在观测接缝处使用相同的横向叶尖收束因子。
            end_factor = np.clip(end_width / max(.1 * self.blade.w_max, 1e-12), 0., 1.)
            end_factor = end_factor ** 2 * (3 - 2 * end_factor) if self.apex_taper else 1.
            xyz[outside] = (self._tip_center(s[outside])
                            + (.5 * v[outside] * w[outside])[:, None] * b[outside]
                            + (end_factor * (end_v - end_zero) * relief_taper)[:, None] * n[outside])
        return xyz

    def _build_tip_center(self):
        """由最终曲面中心轨迹构造连续的叶尖中心延长。"""
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
        """将连续模型参数和支撑域状态保存为 NPZ。"""
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
            support_s=self.support_s,
            support_left=self.support_left,
            support_right=self.support_right,
            domain_end=self.domain_end,
        )

    @classmethod
    def load(cls, path):
        """读取模型并兼容缺少新字段的历史 NPZ 几何。"""
        d = np.load(path)
        midrib = BSpline(d["midrib_t"], d["midrib_c"], int(d["midrib_k"]))
        frame = MovingFrame(d["frame_s"], d["frame_T"], d["frame_B"], d["frame_N"])
        blade = DornbuschBlade(*[float(x) for x in d["blade"]])
        tps = ThinPlateSplineGCV(d["tps_xy"], d["tps_w"], d["tps_a"], float(d["tps_lam"]), float(d["tps_edf"]))
        return cls(midrib, frame, blade, tps, float(d["length_obs"]),
                   d["width_s"] if "width_s" in d else None,
                   d["width_extra"] if "width_extra" in d else None,
                   bool(d["apex_taper"]) if "apex_taper" in d else False,
                   bool(d["surface_tip_extension"]) if "surface_tip_extension" in d else False,
                   d["support_s"] if "support_s" in d else None,
                   d["support_left"] if "support_left" in d else None,
                   d["support_right"] if "support_right" in d else None,
                   float(d["domain_end"]) if "domain_end" in d else None)


def restrict_to_scan_support(model, points, padding_spacing=2.0):
    """按独立左右观测包络收紧活动域，并撤下未观测叶尖延长。

    Keep the fitted chart only within independent measured side envelopes.

    This is a conservative observed-domain model, not recovery of a missing margin/tip.
    Retain an open terminal cross-section when the tip is unobserved.
    """
    points = np.asarray(points, dtype=float)
    spacing = float(np.median(cKDTree(points).query(points, k=2)[0][:, 1]))
    s = project_points_to_curve(points, model.midrib, model.length_obs)
    _, b, _ = model.frame.evaluate(s)
    lateral = np.sum((points - model.midrib(s)) * b, axis=1)
    profile = extract_width_profile(s, lateral, model.length_obs)
    if len(profile["s"]) < 4:
        raise ValueError("Insufficient supported cross-sections for a conservative domain")
    width = np.maximum(model._reference_width(profile["s"]), spacing)
    left = np.clip(2 * (profile["w_left"] + padding_spacing * spacing) / width, .03, 1.)
    right = np.clip(2 * (profile["w_right"] + padding_spacing * spacing) / width, .03, 1.)
    model.support_s = np.r_[0., profile["s"], model.length_obs]
    model.support_left = np.r_[left[0], left, left[-1]]
    model.support_right = np.r_[right[0], right, right[-1]]
    model._support_left = PchipInterpolator(model.support_s, model.support_left)
    model._support_right = PchipInterpolator(model.support_s, model.support_right)
    model.domain_end = min(model.length_obs, model.blade.length)
    model._tip_center = None
    return {"policy": "independent_scan_side_envelopes_and_observed_extent", "padding_spacing": padding_spacing,
            "spacing": spacing, "open_terminal_cross_section": model.open_tip,
            "prior_parameter_length": model.blade.length, "active_parameter_length": model.length}


def _aggregate_tps_bins(u, v, h, max_points, rng):
    """在二维参数分箱内取中位点，抑制残余扫描毛刺。

    Aggregate TPS training samples by 2D bin median to suppress residual scatter spikes."""
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
    """拟合多个几何坐标候选，经方向和贴合审计选择连续叶面。

    Fit and audit multiple geometric charts, select a non-reversed scan-fitting surface."""
    cfg = cfg or ModelConfig()
    prior = prior or MaizeShapePrior()
    # 该缓存仅属于当前观测点集，不能跨叶片或
    # 跨留出划分复用；TPS 密度不应改变确定性几何。
    cache = {} if _geometry_cache is None else _geometry_cache
    if "clean" not in cache:
        cache["clean"] = clean_leaf_points(points, normals)
    if _geometry_mode is None and not _stable_geometry:
        from .quality import audit_model, surface_support_diagnostics
        # 在同一组保留扫描点上比较规则曲面；仅重试
        # 一条过度平滑中脉可能消除折叠却恶化贴合。
        retained, _ = cache["clean"]
        candidates = []
        candidate_modes = (
            ("joint", "initial", "smoothed", "axis", "linear")
            if cfg.include_joint_candidate
            else ("initial", "smoothed", "axis", "linear")
        )
        for mode in candidate_modes:
            candidate = fit_leaf(points, stem, normals=normals, prior=prior, cfg=cfg,
                                 leaf_id=leaf_id, _geometry_mode=mode, _geometry_cache=cache)
            sv, _ = candidate["model"].mesh_grid(220, 61)
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
                # 细化是可选步骤；细化失败不能使已通过审计的
                # 粗曲面失效，也不能放宽质量门禁。
                refinement["error"] = str(exc)
            best["traits"]["tps_refinement"] = refinement
        best["traits"]["quality_passed"] = best["quality"]["passed"]
        best["traits"]["support_far_area_fraction"] = best["quality"]["support"]["far_area_fraction"]
        best["traits"]["support_review_required"] = best["quality"]["support"]["far_area_fraction"] > .05
        if not best["quality"]["passed"]:
            raise ValueError(f"Leaf {leaf_id}: selected surface failed the quality checks: {best['quality']}")
        return best

    mode = _geometry_mode or "axis"
    _stable_geometry = mode == "axis"
    rng = np.random.default_rng(cfg.seed)

    # 第一步：欧氏连通簇与 SOR 双阶段去噪。
    p, nrm = cache["clean"]

    # 第二步：L1 骨架、主曲线中脉与法向对齐 RMF 初始化。
    geometry_key = ("axis" if _stable_geometry else "initial", cfg.seed)
    if geometry_key not in cache:
        cache[geometry_key] = extract_midrib(p, stem, seed=cfg.seed, stabilize=_stable_geometry)
    midrib_0, length_0, s_proj_0, spacing, midrib_lam = cache[geometry_key]
    if mode == "linear":
        # 单调主轴坐标的纵向曲率由 TPS 高度承载；
        # 恒定参考切向避免参数带发生内外翻转。
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

    # 第三步：联合可微几何优化，校正单侧叶缘偏差与标架滚转。
    if mode != "joint":
        # 无约束联合位移可能重新引入急弯，
        # 因此在此轮重试中保留正则化后的几何。
        midrib_obs, frame_obs, length_obs, s_proj = midrib_0, frame_0, length_0, s_proj_0
    else:
        if "joint" not in cache:
            cache["joint"] = joint_optimize_geometry(p, nrm, midrib_0, frame_0, length_0, s_proj_0, prior=prior)
        midrib_obs, frame_obs, length_obs, s_proj = cache["joint"]

    # 第四步：将点表达为局部物理坐标 (s, 横向 r, 法向 h)。
    _, b_p, n_p = frame_obs.evaluate(s_proj)
    res_p = p - midrib_obs(s_proj)
    lateral = np.sum(res_p * b_p, axis=1)
    normal_h = np.sum(res_p * n_p, axis=1)

    # 第五步：拟合 Dornbusch 叶形，并用 BIC 选择截尖模型。
    prof = extract_width_profile(s_proj, lateral, length_obs)
    blade = fit_dornbusch_blade(
        prof["s"], prof["w_obs"], length_obs, prior=prior, allow_tip_completion=cfg.allow_tip_completion
    )

    # 第六步：历史补全模式下延长中脉与 RMF，参见 Prévot 1991 / ADEL-Maize。
    midrib_full, frame_full = extrapolate_midrib_and_frame(midrib_obs, frame_obs, length_obs, blade.length)

    # 第七步：分箱中位代表点拟合薄板样条，使用 GML/REML 选择平滑强度。
    # 全局物种轮廓不应裁掉仍被扫描观测到的叶缘，
    # 因此保留局部包络；缺少观测时才仅使用先验。
    width_s = np.r_[0.0, prof["s"], length_obs]
    observed_width = 2.0 * np.maximum(prof["w_left"], prof["w_right"])
    width_extra = np.maximum(observed_width - blade.width(prof["s"]), 0.0)
    if mode in ("axis", "linear") and len(width_extra) >= 6:
        # 平滑分段分位数，避免噪声造成锯齿状叶缘。
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
    support_policy = restrict_to_scan_support(model, p) if cfg.conservative_support else None
    orientation = surface_orientation_diagnostics(model, 241, 61)

    # 在非退化的叶尖扇形参数网格上计算连续曲面。
    s_grid = np.linspace(0.0, model.length, cfg.nu)
    grid_sv, faces = model.mesh_grid(cfg.nu, cfg.nv)
    vertices = model.evaluate(grid_sv)

    # 第八步：结合物理叶面坐标与三维点间距划分观测、插值和外推区域。
    phys_obs = np.column_stack([s_proj, lateral])
    phys_grid = np.column_stack([grid_sv[:, 0], model.lateral_coordinate(grid_sv[:, 0], grid_sv[:, 1])])
    hull_inside = Delaunay(phys_obs).find_simplex(phys_grid, tol=1e-10) >= 0

    # 支撑半径兼顾三维局部点间距与网格单元尺寸，减少离散化影响。
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

    # 记录扫描内贴合残差与几何表型。
    fitted_p = model.evaluate(np.column_stack([s_proj, np.clip(model.lateral_parameter(s_proj, lateral), -1.0, 1.0)]))
    fit_rmse = float(np.sqrt(np.mean(np.sum((fitted_p - p) ** 2, axis=1))))
    center_s = np.unique(np.r_[np.linspace(0.0, model.length, 500), length_obs])
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
        parameter_completed_length=float(model.length),
        support_policy=support_policy,
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
        tip_completion_status=("unobserved_extension_withheld" if model.open_tip
                               else getattr(blade, "completion_status", "legacy")),
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
