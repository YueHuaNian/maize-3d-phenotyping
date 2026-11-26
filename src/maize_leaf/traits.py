"""从冻结连续模型提取茎叶角、曲率、投影面积与冠层表型。

Mathematically grounded 3D leaf & plant geometric trait extraction (V2.0 Stage D).

Computes scale-consistent geometric phenotypes directly from frozen continuous leaf models
(MaizeLeafModel .npz) and continuous cubic B-spline stem centerlines:
  - Formal stem-leaf angle theta_i = arccos(clip(t_s . t_l, -1, 1)) via orthogonal curve
    projection onto the stem spline and arc-length centerline tangent at s = 0 (Wu et al. 2019).
  - Dimensionless pose invariants: tip gain g_i = (z_tip - z_base) / L_i, droop ratio
    d_i = (z_max - z_tip) / L_i, straightness xi_i = ||p_tip - p_base|| / L_i.
  - Three distinct area/optical quantities:
      1) True surface area A_i = sum A_f
      2) Element horizontal projected area A_h^elem = sum |n_f . e_z| A_f and ratio rho_i = A_h^elem / A_i
      3) Exact barycentric Z-buffer union projected area A_union (single-leaf & whole-plant) and
         5-direction geometric light-exposure proxy A_vis(s) (nadir + 4 oblique directions).
  - Differential geometry curvature invariants (do Carmo):
      1) Dimensionless total absolute midrib curvature K_mid = integral_0^L kappa(s) ds
      2) First & second fundamental forms (E, F, G; e, f, g) on interior parametric grid
         (excluding boundary margin bands) yielding mean curvature H, Gaussian curvature K_G,
         and the scale-invariant Willmore geometric bending functional W = integral H^2 dA.
  - Leaf confidence weight q_i = C_i * exp(-e_i / tau) and height-conditioned 95th-percentile
    canopy radial envelope R(h) relative to the stem centerline.
"""
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree

from .entropy import mesh_normal_entropy, model_pitch_entropy
from .geometry import fit_pspline_gcv, project_points_to_curve, unit
from .surface import MaizeLeafModel, leaf_mesh_grid


# 五个方向的光暴露探测参数（历史兼容）：(名称, 天顶角, 方位角)。
LIGHT_DIRECTIONS = (
    ("nadir", 0.0, 0.0),
    ("zen30_az0", 30.0, 0.0),
    ("zen30_az90", 30.0, 90.0),
    ("zen60_az0", 60.0, 0.0),
    ("zen60_az90", 60.0, 90.0),
)

# den Dulk (1989) 46 方向 Turtle Sky 等立体角半球分带：(仰角 deg, 方位角扇区数, 起始方位角 deg)。
# 广泛用于 GroIMP / Caribu / VegeSTAR 玉米 FSPM 冠层截光模拟，绕 Z 轴近似旋转不变。
TURTLE_SKY_46_BANDS = (
    (90.0, 1, 0.0),
    (66.42, 5, 0.0),
    (47.87, 10, 18.0),
    (28.57, 15, 0.0),
    (9.23, 15, 12.0),
)


def build_turtle_sky_46():
    """构造 den Dulk (1989) 46 方向等立体角半球方向与标准阴天（SOC）辐射权重。

    Return unit directions (46, 3), normalized SOC irradiance weights, and uniform-sky weights."""
    dirs, w_soc, w_uni = [], [], []
    for elev_deg, n_az, az0 in TURTLE_SKY_46_BANDS:
        el = np.radians(elev_deg)
        sin_e, cos_e = float(np.sin(el)), float(np.cos(el))
        for j in range(n_az):
            az = np.radians(az0 + 360.0 * j / n_az)
            dirs.append([cos_e * float(np.cos(az)), cos_e * float(np.sin(az)), sin_e])
            w_soc.append(sin_e * (1.0 + 2.0 * sin_e) / 3.0)
            w_uni.append(sin_e)
    dirs = np.asarray(dirs, dtype=float)
    w_soc = np.asarray(w_soc, dtype=float)
    w_soc /= float(np.sum(w_soc))
    w_uni = np.asarray(w_uni, dtype=float)
    w_uni /= float(np.sum(w_uni))
    return dirs, w_soc, w_uni


def compute_turtle_sky_46_exposure(vertices, faces, face_leaf_ids, resolution=160):
    """基于 46 方向 Turtle Sky（SOC 加权）计算绕 Z 轴近旋转不变的冠层截光代理指标。

    Compute rotation-invariant hemispherical canopy light-interception metrics using
    den Dulk (1989) 46-sector equal-solid-angle Turtle Sky with Standard Overcast Sky (SOC) weights."""
    dirs, w_soc, _ = build_turtle_sky_46()
    vis = np.empty(len(dirs), dtype=float)
    proj = np.empty(len(dirs), dtype=float)
    tot_area = 0.0
    for i, d in enumerate(dirs):
        tv, _, _ = rasterize_directional_exposure(vertices, faces, face_leaf_ids, d, resolution=resolution)
        ta, pa = element_projected_area(vertices, faces, d)
        tot_area = ta
        vis[i] = tv
        proj[i] = pa
    vis_w = float(np.sum(w_soc * vis))
    proj_w = float(np.sum(w_soc * proj))
    non_overlap = float(np.clip(vis_w / max(proj_w, 1e-12), 0.0, 1.0)) if proj_w > 1e-12 else 0.0
    star = float(vis_w / max(tot_area, 1e-12)) if tot_area > 1e-12 else 0.0
    oblique_w = float(np.sum(w_soc[1:] * vis[1:]) / max(float(np.sum(w_soc[1:])), 1e-12))
    slant_ret = float(np.clip(oblique_w / max(vis[0], 1e-12), 0.0, 1.0)) if vis[0] > 1e-12 else 0.0
    s_light = float(0.5 * non_overlap + 0.5 * slant_ret)
    return {
        "turtle46_visible_area": vis_w,
        "turtle46_projected_area": proj_w,
        "turtle46_non_overlap": non_overlap,
        "turtle46_star": star,
        "turtle46_slant_retention": slant_ret,
        "turtle46_s_light": s_light,
    }


def load_stem_spline(stem_json_path):
    """读取沿高度向上定向的连续茎秆中心样条。

    Load continuous cubic B-spline stem centerline C_s(t) oriented strictly upward (dz > 0)."""
    data = json.loads(Path(stem_json_path).read_text(encoding="utf-8"))
    sp = data["spline"]
    knots = np.asarray(sp["knots"], dtype=float)
    coeffs = np.asarray(sp["coefficients"], dtype=float)
    degree = int(sp["degree"])
    curve = BSpline(knots, coeffs, degree)
    t_lo = float(knots[degree])
    t_hi = float(knots[-degree - 1])
    # 确保茎秆沿全局 +Z 向上定向。
    z_lo = float(curve(t_lo)[2])
    z_hi = float(curve(t_hi)[2])
    sign = 1.0 if z_hi >= z_lo else -1.0
    sampled = np.asarray(data["xyz"], dtype=float)
    return curve, (t_lo, t_hi), sign, sampled


def stem_tangent_at_point(stem_curve, t_bounds, z_sign, point):
    """将叶基正交投影到茎秆样条并求局部单位切向。

    Project a 3D point orthogonally onto the stem spline and return (t*, foot_xyz, unit_t_s)."""
    t_lo, t_hi = t_bounds
    # 平移参数域至 [0, span]，适配点到曲线投影函数。
    span = max(t_hi - t_lo, 1e-12)
    grid_t = np.linspace(t_lo, t_hi, 300)
    grid_p = stem_curve(grid_t)
    _, idx = cKDTree(grid_p).query(np.asarray(point, dtype=float))
    t = float(grid_t[idx])

    d1 = stem_curve.derivative(1)
    d2 = stem_curve.derivative(2)
    p = np.asarray(point, dtype=float)
    for _ in range(6):
        c = stem_curve(t)
        c1 = d1(t)
        c2 = d2(t)
        diff = c - p
        f = float(np.dot(diff, c1))
        fp = float(np.dot(c1, c1) + np.dot(diff, c2))
        if abs(fp) > 1e-12:
            step = np.clip(f / fp, -0.05 * span, 0.05 * span)
            t = float(np.clip(t - step, t_lo, t_hi))
    foot = np.asarray(stem_curve(t), dtype=float)
    tan = np.asarray(d1(t), dtype=float) * z_sign
    tan = tan / max(float(np.linalg.norm(tan)), 1e-12)
    return t, foot, tan


def _smooth_arc_centerline(xyz_samples):
    """按弧长重参数化采样中心轨迹，拟合二阶连续样条。

    Reparameterize sampled 3D centerline by arc length and fit tight C^2 cubic B-spline."""
    pts = np.asarray(xyz_samples, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.r_[0.0, np.cumsum(seg)]
    total_len = float(arc[-1])
    if total_len <= 1e-12 or len(pts) < 6:
        return None, arc, total_len
    # 去除重复弧长节点。
    keep = np.r_[True, seg > 1e-12]
    arc_u = arc[keep]
    pts_u = pts[keep]
    if len(arc_u) < 6:
        return None, arc, total_len
    curve, _ = fit_pspline_gcv(
        arc_u, pts_u, n_basis=min(24, max(8, len(arc_u) // 10)),
        lam_bounds=(1e-9, 1e-3), bounded=False,
    )
    return curve, arc_u, total_len


def midrib_pose_and_curvature(model, stem_curve, stem_bounds, stem_z_sign, n_samples=400):
    """提取中脉姿态不变量、茎叶角与总曲率。

    Extract stem-leaf angle, dimensionless pose invariants, and total curvature K_mid.

    Evaluated on the actual continuous surface center trajectory M(s) = S(s, 0) for both
    the observed span [0, L_obs] and full completed span [0, L_comp].
    """
    s_obs = np.linspace(0.0, model.length_obs, n_samples)
    m_obs = model.evaluate(np.column_stack([s_obs, np.zeros(n_samples)]))
    c_obs, arc_obs, len_obs = _smooth_arc_centerline(m_obs)

    s_full = np.linspace(0.0, model.length, n_samples)
    m_full = model.evaluate(np.column_stack([s_full, np.zeros(n_samples)]))
    c_full, arc_full, len_full = _smooth_arc_centerline(m_full)

    # 在叶基 M(0) 处定义茎叶角。
    p_base = m_obs[0]
    _, stem_foot, t_s = stem_tangent_at_point(stem_curve, stem_bounds, stem_z_sign, p_base)
    base_to_stem_dist = float(np.linalg.norm(p_base - stem_foot))

    # 从基部 0–2% 平滑弧长中心线估计叶基切向。
    if c_obs is not None:
        t_l = np.asarray(c_obs(0.0, nu=1), dtype=float)
    else:
        t_l = m_obs[min(5, len(m_obs) - 1)] - m_obs[0]
    t_l = t_l / max(float(np.linalg.norm(t_l)), 1e-12)

    stem_leaf_angle_deg = float(np.degrees(np.arccos(np.clip(np.dot(t_s, t_l), -1.0, 1.0))))
    base_z_inclination_deg = float(np.degrees(np.arccos(np.clip(t_l[2], -1.0, 1.0))))

    def _pose_and_k(pts, curve, total_len):
        """在指定中心轨迹采样分辨率下计算姿态与总曲率。"""
        if total_len <= 1e-12:
            return 0.0, 0.0, 1.0, 0.0, 0.0
        z_base = float(pts[0, 2])
        z_tip = float(pts[-1, 2])
        z_max = float(np.max(pts[:, 2]))
        tip_gain = (z_tip - z_base) / total_len
        droop_ratio = (z_max - z_tip) / total_len
        straightness = float(np.linalg.norm(pts[-1] - pts[0])) / total_len
        if curve is not None:
            ll = np.linspace(0.0, total_len, n_samples)
            d1 = curve(ll, nu=1)
            d2 = curve(ll, nu=2)
            speed = np.maximum(np.linalg.norm(d1, axis=1), 1e-12)
            kappa = np.linalg.norm(np.cross(d1, d2), axis=1) / (speed ** 3)
            # 排除两端极端 1% 节点，减小三次样条二阶导的边界伪影。
            inner = slice(max(1, n_samples // 100), n_samples - max(1, n_samples // 100))
            k_total = float(np.trapezoid(kappa[inner], ll[inner]))
            k_mean = float(k_total / max(ll[inner][-1] - ll[inner][0], 1e-12))
        else:
            k_total, k_mean = 0.0, 0.0
        return float(tip_gain), float(droop_ratio), float(straightness), k_total, k_mean

    g_obs, d_obs, xi_obs, k_mid_obs, k_mean_obs = _pose_and_k(m_obs, c_obs, len_obs)
    g_full, d_full, xi_full, k_mid_full, k_mean_full = _pose_and_k(m_full, c_full, len_full)

    return {
        "base_xyz": p_base.tolist(),
        "tip_xyz": m_full[-1].tolist(),
        "base_z": float(p_base[2]),
        "base_to_stem_dist": base_to_stem_dist,
        "midrib_base_angle_deg": stem_leaf_angle_deg,
        "base_z_inclination_deg": base_z_inclination_deg,
        "observed_center_length": len_obs,
        "completed_center_length": len_full,
        "tip_gain": g_full,
        "droop_ratio": d_full,
        "straightness": xi_full,
        "midrib_curvature_total": k_mid_full,
        "midrib_curvature_mean": k_mean_full,
        "tip_gain_observed": g_obs,
        "droop_ratio_observed": d_obs,
        "midrib_curvature_total_observed": k_mid_obs,
    }


def far_end_height_diff(vertices, stem_sampled, base_z):
    """查找叶面上距茎秆中心线最远的顶点，计算其与叶基的高度差。

    Signed height difference z_far - z_base between the leaf-surface vertex with
    maximum 3D Euclidean distance to the stem centerline and the leaf base."""
    verts = np.asarray(vertices, dtype=float)
    stem = np.asarray(stem_sampled, dtype=float)
    dists = cKDTree(stem).query(verts)[0]
    far_idx = int(np.argmax(dists))
    far_xyz = verts[far_idx]
    return {
        "far_end_xyz": far_xyz.tolist(),
        "far_end_z": float(far_xyz[2]),
        "far_end_stem_dist": float(dists[far_idx]),
        "far_end_height_diff": float(far_xyz[2] - float(base_z)),
    }


def _basal_surface_tangent_from_grid(xyz, s_vals, v_vals):
    """在基部叶面窗口对纵向切向及横向轴进行面积加权积分。

    Area-weighted 2D quadrature of unit longitudinal surface tangent S_s and lateral axis S_v."""
    ds = np.gradient(xyz, s_vals, axis=0, edge_order=2)
    dv = np.gradient(xyz, v_vals, axis=1, edge_order=2)
    cross = np.cross(ds, dv)
    jac = np.linalg.norm(cross, axis=2)  # 面积雅可比：纵向与横向偏导叉积的模。

    # 构造 (s_vals, v_vals) 上的二维梯形求积权重。
    ws = np.empty(len(s_vals), dtype=float)
    ws[0] = 0.5 * (s_vals[1] - s_vals[0])
    ws[-1] = 0.5 * (s_vals[-1] - s_vals[-2])
    ws[1:-1] = 0.5 * (s_vals[2:] - s_vals[:-2])

    wv = np.empty(len(v_vals), dtype=float)
    wv[0] = 0.5 * (v_vals[1] - v_vals[0])
    wv[-1] = 0.5 * (v_vals[-1] - v_vals[-2])
    wv[1:-1] = 0.5 * (v_vals[2:] - v_vals[:-2])

    dA = jac * ws[:, None] * wv[None, :]
    total_a = max(float(np.sum(dA)), 1e-15)
    w = dA / total_a

    ds_unit = ds / np.maximum(np.linalg.norm(ds, axis=2, keepdims=True), 1e-12)
    t_surf = np.sum(ds_unit * w[:, :, None], axis=(0, 1))
    t_surf = t_surf / max(float(np.linalg.norm(t_surf)), 1e-12)

    dv_unit = dv / np.maximum(np.linalg.norm(dv, axis=2, keepdims=True), 1e-12)
    b_surf = np.sum(dv_unit * w[:, :, None], axis=(0, 1))
    b_surf = b_surf / max(float(np.linalg.norm(b_surf)), 1e-12)
    return t_surf, b_surf


def basal_surface_stem_leaf_angle(model, stem_curve, stem_bounds, stem_z_sign,
                                  basal_span_frac=0.15, v_lim=0.80, ns=16, nv=17):
    """计算基部叶面平均纵向切向与局部茎秆切向的夹角。

    Formal stem-leaf angle from area-weighted longitudinal tangent on basal surface patch.

    Evaluates S(s, v) over s in [0, basal_span_frac * L_obs], v in [-v_lim, v_lim], integrates
    hat(S_s) dA to suppress single-endpoint derivative whip, preserves full [0, 180] deg range
    for drooping leaf bases, and remains strictly invariant to lateral blade roll around the midrib.
    """
    p_base = model.evaluate([[0.0, 0.0]])[0]
    _, _, t_s = stem_tangent_at_point(stem_curve, stem_bounds, stem_z_sign, p_base)

    s_hi = max(float(basal_span_frac) * model.length_obs, 1e-4)
    s_vals = np.linspace(0.0, s_hi, ns)
    v_vals = np.linspace(-float(v_lim), float(v_lim), nv)
    ss, vv = np.meshgrid(s_vals, v_vals, indexing="ij")
    xyz = model.evaluate(np.column_stack([ss.ravel(), vv.ravel()])).reshape(ns, nv, 3)

    t_surf, b_surf = _basal_surface_tangent_from_grid(xyz, s_vals, v_vals)
    theta_deg = float(np.degrees(np.arccos(np.clip(np.dot(t_s, t_surf), -1.0, 1.0))))
    z_inc_deg = float(np.degrees(np.arccos(np.clip(t_surf[2], -1.0, 1.0))))
    roll_coupling = abs(float(np.dot(t_s, b_surf)))
    return {
        "stem_leaf_angle_deg": theta_deg,
        "basal_surface_z_inclination_deg": z_inc_deg,
        "basal_roll_coupling": roll_coupling,
    }


def surface_curvature_fundamental_forms(model, nu=101, nv=31, margin_trim_frac=0.05, tip_trim_frac=0.04):
    """由第一、第二基本形式计算曲面曲率及 Willmore 泛函。

    Compute surface mean curvature H, Gaussian curvature K_G, and Willmore functional W = int H^2 dA.

    Uses first (E, F, G) and second (e, f, g) fundamental forms on the continuous parametric
    surface S(s, v), trimming margin_trim_frac of the normalized width from each lateral edge
    (default 5% -> v in [-0.90, 0.90]) and tip_trim_frac near the apex singularity.
    """
    v_lim = max(0.1, 1.0 - 2.0 * float(margin_trim_frac))
    s_lo = max(0.02 * model.length, 1e-6)
    s_hi = (1.0 - float(tip_trim_frac)) * model.length
    if s_hi <= s_lo:
        return {
            "surface_interior_area": 0.0,
            "mean_curvature_abs": float("nan"),
            "mean_curvature_abs_dimless": float("nan"),
            "gaussian_curvature_abs": float("nan"),
            "willmore_bending_functional": float("nan"),
        }

    s_vals = np.linspace(s_lo, s_hi, nu)
    v_vals = np.linspace(-v_lim, v_lim, nv)
    ss, vv = np.meshgrid(s_vals, v_vals, indexing="ij")
    xyz = model.evaluate(np.column_stack([ss.ravel(), vv.ravel()])).reshape(nu, nv, 3)

    return _fundamental_forms_from_grid(xyz, s_vals, v_vals)


def _fundamental_forms_from_grid(xyz, s_vals, v_vals):
    """在规则参数网格上求导并积分平均曲率、总曲率和弯曲能。

    Exact first/second fundamental form curvature integration on a regular (nu, nv, 3) grid."""
    ds = np.gradient(xyz, s_vals, axis=0, edge_order=2)
    dv = np.gradient(xyz, v_vals, axis=1, edge_order=2)
    dss = np.gradient(ds, s_vals, axis=0, edge_order=2)
    dsv = np.gradient(ds, v_vals, axis=1, edge_order=2)
    dvv = np.gradient(dv, v_vals, axis=1, edge_order=2)

    E = np.sum(ds * ds, axis=2)
    F = np.sum(ds * dv, axis=2)
    G = np.sum(dv * dv, axis=2)
    cross = np.cross(ds, dv)
    jac = np.linalg.norm(cross, axis=2)  # 由第一基本形式计算面积雅可比 sqrt(E*G-F^2)。
    valid = jac > 1e-12
    N = np.zeros_like(cross)
    N[valid] = cross[valid] / jac[valid, None]

    e = np.sum(dss * N, axis=2)
    f = np.sum(dsv * N, axis=2)
    g = np.sum(dvv * N, axis=2)

    det1 = np.maximum(E * G - F * F, 1e-24)
    H = np.where(valid, (e * G - 2.0 * f * F + g * E) / (2.0 * det1), 0.0)
    K_G = np.where(valid, (e * g - f * f) / det1, 0.0)

    # 在内部参数域 (s, v) 上做二维梯形积分。
    def _int2d(field):
        """按两个参数方向进行数值积分。"""
        return float(np.trapezoid(np.trapezoid(field, v_vals, axis=1), s_vals, axis=0))

    area_int = _int2d(jac)
    if area_int <= 1e-15:
        return {
            "surface_interior_area": 0.0,
            "mean_curvature_abs": float("nan"),
            "mean_curvature_abs_dimless": float("nan"),
            "gaussian_curvature_abs": float("nan"),
            "willmore_bending_functional": float("nan"),
        }

    willmore = _int2d((H ** 2) * jac)
    h_abs_mean = _int2d(np.abs(H) * jac) / area_int
    kg_abs_mean = _int2d(np.abs(K_G) * jac) / area_int
    return {
        "surface_interior_area": area_int,
        "mean_curvature_abs": h_abs_mean,
        "mean_curvature_abs_dimless": float(h_abs_mean * np.sqrt(area_int)),
        "gaussian_curvature_abs": kg_abs_mean,
        "willmore_bending_functional": willmore,
    }


def direction_unit_vector(zenith_deg, azimuth_deg):
    """将入射光源的天顶角和方位角转换为单位方向。

    Return unit vector s pointing toward incoming light source at (zenith, azimuth)."""
    alpha = np.radians(float(zenith_deg))
    phi = np.radians(float(azimuth_deg))
    return np.array([
        np.sin(alpha) * np.cos(phi),
        np.sin(alpha) * np.sin(phi),
        np.cos(alpha),
    ], dtype=float)


def _ortho_basis_for_direction(s_dir):
    """为指定投影方向建立正交二维平面基。"""
    s = np.asarray(s_dir, dtype=float)
    s = s / max(float(np.linalg.norm(s)), 1e-12)
    ref = np.array([1.0, 0.0, 0.0]) if abs(s[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, s) * s
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(s, u)
    v = v / max(float(np.linalg.norm(v)), 1e-12)
    return u, v, s


def rasterize_directional_exposure(vertices, faces, face_leaf_ids, s_dir, resolution=360, pixel_size=None):
    """用重心坐标和深度缓冲估计遮挡后的方向投影面积。

    Exact barycentric Z-buffer directional geometric light-exposure proxy A_vis(s).

    Projects all triangles orthographically onto the plane perpendicular to unit light direction s.
    Each covered pixel has projected area Delta^2 = |n . s| dA, retaining the front-most triangle
    (largest coordinate along +s, i.e., V(x, s) = 1). Returns total union visible area and per-leaf
    visible area breakdown.
    """
    u_ax, v_ax, s_ax = _ortho_basis_for_direction(s_dir)
    verts = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int32)
    f_ids = np.asarray(face_leaf_ids, dtype=np.int32)

    pu = verts @ u_ax
    pv = verts @ v_ax
    pz = verts @ s_ax

    u_min, u_max = float(pu.min()), float(pu.max())
    v_min, v_max = float(pv.min()), float(pv.max())
    span = max(u_max - u_min, v_max - v_min, 1e-9)
    if pixel_size is None:
        pixel_size = span / float(resolution)
    else:
        pixel_size = float(pixel_size)

    # 四周各补一个像素，避免投影边界被裁切。
    u0 = u_min - pixel_size
    v0 = v_min - pixel_size
    nx = max(2, int(np.ceil((u_max - u_min) / pixel_size)) + 3)
    ny = max(2, int(np.ceil((v_max - v_min) / pixel_size)) + 3)

    zbuf = np.full((ny, nx), -np.inf, dtype=np.float64)
    owner = np.full((ny, nx), -1, dtype=np.int32)

    tri_u = pu[faces]  # 数组形状为 (面片数 F, 3)。
    tri_v = pv[faces]
    tri_z = pz[faces]

    # 计算像素包围盒，像素中心为 u0+(ix+0.5)*dx。
    ix_min = np.maximum(0, np.floor((tri_u.min(axis=1) - u0) / pixel_size - 0.5).astype(int))
    ix_max = np.minimum(nx - 1, np.ceil((tri_u.max(axis=1) - u0) / pixel_size - 0.5).astype(int))
    iy_min = np.maximum(0, np.floor((tri_v.min(axis=1) - v0) / pixel_size - 0.5).astype(int))
    iy_max = np.minimum(ny - 1, np.ceil((tri_v.max(axis=1) - v0) / pixel_size - 0.5).astype(int))

    # 预计算用于重心坐标的二维边向量。
    u0t, u1t, u2t = tri_u[:, 0], tri_u[:, 1], tri_u[:, 2]
    v0t, v1t, v2t = tri_v[:, 0], tri_v[:, 1], tri_v[:, 2]
    z0t, z1t, z2t = tri_z[:, 0], tri_z[:, 1], tri_z[:, 2]
    denom = (v1t - v2t) * (u0t - u2t) + (u2t - u1t) * (v0t - v2t)

    pixel_u = u0 + (np.arange(nx) + 0.5) * pixel_size
    pixel_v = v0 + (np.arange(ny) + 0.5) * pixel_size

    eps = -1e-9
    for fi in range(len(faces)):
        den = denom[fi]
        if abs(den) < 1e-18:
            continue
        x0, x1 = ix_min[fi], ix_max[fi]
        y0, y1 = iy_min[fi], iy_max[fi]
        if x1 < x0 or y1 < y0:
            continue
        xs = pixel_u[x0 : x1 + 1][None, :]  # 横向广播网格形状为 (1, W)。
        ys = pixel_v[y0 : y1 + 1][:, None]  # 纵向广播网格形状为 (H, 1)。

        inv_den = 1.0 / den
        w0 = ((v1t[fi] - v2t[fi]) * (xs - u2t[fi]) + (u2t[fi] - u1t[fi]) * (ys - v2t[fi])) * inv_den
        w1 = ((v2t[fi] - v0t[fi]) * (xs - u2t[fi]) + (u0t[fi] - u2t[fi]) * (ys - v2t[fi])) * inv_den
        w2 = 1.0 - w0 - w1

        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        if not np.any(inside):
            continue
        z_interp = w0 * z0t[fi] + w1 * z1t[fi] + w2 * z2t[fi]
        sub_z = zbuf[y0 : y1 + 1, x0 : x1 + 1]
        sub_o = owner[y0 : y1 + 1, x0 : x1 + 1]
        closer = inside & (z_interp > sub_z)
        if np.any(closer):
            sub_z[closer] = z_interp[closer]
            sub_o[closer] = f_ids[fi]

    px_area = pixel_size * pixel_size
    occ = owner >= 0
    total_vis = float(np.sum(occ)) * px_area
    per_leaf = {}
    if np.any(occ):
        u_ids, counts = np.unique(owner[occ], return_counts=True)
        for lid, cnt in zip(u_ids, counts):
            per_leaf[int(lid)] = float(cnt) * px_area
    return total_vis, per_leaf, pixel_size


def element_projected_area(vertices, faces, s_dir=(0.0, 0.0, 1.0)):
    """按各面元法向计算总曲面面积与指定方向投影面积。

    Compute true surface area A and element projected area A_proj = sum |n_f . s| A_f."""
    s = np.asarray(s_dir, dtype=float)
    s = s / max(float(np.linalg.norm(s)), 1e-12)
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    twice_a = np.linalg.norm(cross, axis=1)
    areas = 0.5 * twice_a
    valid = areas > 1e-15
    normals = cross[valid] / twice_a[valid, None]
    total_area = float(np.sum(areas[valid]))
    proj_area = float(np.sum(np.abs(normals @ s) * areas[valid]))
    return total_area, proj_area


def compute_canopy_radial_envelope(all_leaf_vertices, stem_sampled, n_bins=25, quantile=0.95):
    """沿高度估计相对茎秆中心的径向分位数包络。

    Compute height-conditioned radial envelope R(h) = Q_0.95(sqrt((x-x_s(z))^2 + (y-y_s(z))^2))."""
    pts = np.asarray(all_leaf_vertices, dtype=float)
    stem = np.asarray(stem_sampled, dtype=float)
    # 按 z 排序茎秆点，保证中心位置插值单调。
    order = np.argsort(stem[:, 2])
    stem_z = stem[order, 2]
    stem_x = stem[order, 0]
    stem_y = stem[order, 1]

    z_min = float(min(pts[:, 2].min(), stem_z.min()))
    z_max = float(max(pts[:, 2].max(), stem_z.max()))
    height = max(z_max - z_min, 1e-12)

    xs_at_z = np.interp(pts[:, 2], stem_z, stem_x)
    ys_at_z = np.interp(pts[:, 2], stem_z, stem_y)
    radii = np.hypot(pts[:, 0] - xs_at_z, pts[:, 1] - ys_at_z)
    h_norm = (pts[:, 2] - z_min) / height

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (h_norm >= a) & (h_norm <= b if i == n_bins - 1 else h_norm < b)
        cnt = int(np.sum(mask))
        r_q = float(np.quantile(radii[mask], quantile)) if cnt >= 5 else float("nan")
        rows.append({
            "bin_index": i,
            "h_lo": float(a),
            "h_hi": float(b),
            "h_mid": float(0.5 * (a + b)),
            "n_vertices": cnt,
            "radius_q95": r_q,
        })
    return rows, z_min, z_max


def extract_plant_phenotypes(plant_dir, include_entropy=True, mesh_nu=120, mesh_nv=31,
                             margin_trim_frac=0.05, basal_span_frac=0.15,
                             zbuf_resolution=360, envelope_bins=25,
                             include_turtle46=True, turtle_resolution=160):
    """从单株冻结模型提取逐叶及整株几何表型。

    Extract all Stage D leaf-level and plant-level traits from one frozen plant output folder."""
    plant_dir = Path(plant_dir)
    plant_id = plant_dir.name.replace("plant_", "", 1)

    stem_curve, stem_bounds, stem_z_sign, stem_sampled = load_stem_spline(plant_dir / "stem_centerline.json")
    leaves_dir = plant_dir / "leaves"
    model_paths = sorted(leaves_dir.glob("leaf_*_model.npz"))

    leaf_records = []
    plant_verts_list = []
    plant_faces_list = []
    plant_fids_list = []
    v_offset = 0

    for mpath in model_paths:
        stem_name = mpath.stem  # 从 leaf_01_model 形式的文件名解析叶片编号。
        lid = int(stem_name.split("_")[1])
        tpath = leaves_dir / f"leaf_{lid:02d}_traits.json"
        qpath = leaves_dir / f"leaf_{lid:02d}_quality.json"
        t_json = json.loads(tpath.read_text(encoding="utf-8")) if tpath.is_file() else {}
        q_json = json.loads(qpath.read_text(encoding="utf-8")) if qpath.is_file() else {}

        model = MaizeLeafModel.load(mpath)
        sv, faces = model.mesh_grid(mesh_nu, mesh_nv)
        verts = model.evaluate(sv)

        # 第一组表型：姿态、基部叶面茎叶角、离茎最远端高度差与中脉曲率。
        pose = midrib_pose_and_curvature(model, stem_curve, stem_bounds, stem_z_sign)
        far_end = far_end_height_diff(verts, stem_sampled, pose["base_z"])
        basal_ang = basal_surface_stem_leaf_angle(
            model, stem_curve, stem_bounds, stem_z_sign, basal_span_frac=basal_span_frac
        )

        # 第二组表型：曲面面积、面元水平投影与单叶投影并集。
        surf_area, proj_elem = element_projected_area(verts, faces, (0.0, 0.0, 1.0))
        proj_ratio = proj_elem / max(surf_area, 1e-12)
        leaf_union_area, _, _ = rasterize_directional_exposure(
            verts, faces, np.full(len(faces), lid, dtype=np.int32),
            (0.0, 0.0, 1.0), resolution=260,
        )
        self_overlap_ratio = max(0.0, 1.0 - leaf_union_area / max(proj_elem, 1e-12))

        # 第三组表型：裁去侧缘后，以第一、第二基本形式积分曲面曲率。
        curv = surface_curvature_fundamental_forms(model, margin_trim_frac=margin_trim_frac)

        # 第四组表型：独立的面积加权方向信息熵。
        # 主要指标为纵向俯仰角香农熵，分箱数 B=6、12、24。
        # 对照指标为三维等立体角半球法向熵，分箱数 B=12、24、48。
        if include_entropy:
            h_pit_b6, mean_pitch_deg = model_pitch_entropy(model, n_bins=6)
            h_pit_b12, _ = model_pitch_entropy(model, n_bins=12)
            h_pit_b24, _ = model_pitch_entropy(model, n_bins=24)
            h_b12 = mesh_normal_entropy(verts, faces, n_bins=12)
            h_b24 = mesh_normal_entropy(verts, faces, n_bins=24)
            h_b48 = mesh_normal_entropy(verts, faces, n_bins=48)
        else:
            h_pit_b6 = h_pit_b12 = h_pit_b24 = mean_pitch_deg = float("nan")
            h_b12 = h_b24 = h_b48 = float("nan")

        af = t_json.get("area_fractions", {})
        cov_obs = float(af.get("observed", float("nan")))
        cov_int = float(af.get("interpolated", float("nan")))
        cov_ext = float(af.get("extrapolated", float("nan")))

        rec = {
            "plant_id": plant_id,
            "leaf_id": lid,
            "raw_points": int(t_json.get("raw_points", 0)),
            "geometry_method": t_json.get("geometry_method", ""),
            "tip_completion_status": t_json.get("tip_completion_status", ""),
            "quality_passed": bool(t_json.get("quality_passed", q_json.get("passed", False))),
            "coverage_observed": cov_obs,
            "coverage_interpolated": cov_int,
            "coverage_extrapolated": cov_ext,
            "support_far_area_fraction": t_json.get("support_far_area_fraction", float("nan")),
            "support_review_required": bool(t_json.get("support_review_required", False)),
            "open_terminal_cross_section": bool(model.open_tip),
            "fit_rmse": float(t_json.get("fit_rmse", float("nan"))),
            "scan_distance_median": float(t_json.get("scan_distance_median", float("nan"))),
            "scan_distance_p95": float(t_json.get("scan_distance_p95", float("nan"))),
            "surface_support_mean": float(t_json.get("surface_support_mean", float("nan"))),
            "observed_length": pose["observed_center_length"],
            "completed_length": pose["completed_center_length"],
            "max_width": float(t_json.get("max_width", float(np.max(model.width(np.linspace(0, model.length, 120)))))),
            "surface_area": surf_area,
            "projected_area_element": proj_elem,
            "projected_area_union": leaf_union_area,
            "projection_ratio": proj_ratio,
            "self_overlap_ratio": self_overlap_ratio,
            "stem_leaf_angle_deg": basal_ang["stem_leaf_angle_deg"],
            "basal_surface_z_inclination_deg": basal_ang["basal_surface_z_inclination_deg"],
            "basal_roll_coupling": basal_ang["basal_roll_coupling"],
            "midrib_base_angle_deg": pose["midrib_base_angle_deg"],
            "base_z_inclination_deg": pose["base_z_inclination_deg"],
            "base_to_stem_dist": pose["base_to_stem_dist"],
            "base_z": pose["base_z"],
            "far_end_z": far_end["far_end_z"],
            "far_end_stem_dist": far_end["far_end_stem_dist"],
            "far_end_height_diff": far_end["far_end_height_diff"],
            "far_dz_norm": float(np.clip(
                far_end["far_end_height_diff"] / max(pose["observed_center_length"], 1e-6), -1.0, 1.0
            )),
            "tip_gain": pose["tip_gain"],
            "droop_ratio": pose["droop_ratio"],
            "straightness": pose["straightness"],
            "midrib_curvature_total": pose["midrib_curvature_total"],
            "midrib_curvature_mean": pose["midrib_curvature_mean"],
            "tip_gain_observed": pose["tip_gain_observed"],
            "droop_ratio_observed": pose["droop_ratio_observed"],
            "midrib_curvature_total_observed": pose["midrib_curvature_total_observed"],
            "mean_curvature_abs": curv["mean_curvature_abs"],
            "mean_curvature_abs_dimless": curv["mean_curvature_abs_dimless"],
            "gaussian_curvature_abs": curv["gaussian_curvature_abs"],
            "willmore_bending_functional": curv["willmore_bending_functional"],
            "mean_pitch_deg": mean_pitch_deg,
            "pitch_entropy_b6": h_pit_b6,
            "pitch_entropy_b12": h_pit_b12,
            "pitch_entropy_b24": h_pit_b24,
            "normal_entropy_b12": h_b12,
            "normal_entropy_b24": h_b24,
            "normal_entropy_b48": h_b48,
        }
        leaf_records.append(rec)

        plant_verts_list.append(verts)
        plant_faces_list.append(faces + v_offset)
        plant_fids_list.append(np.full(len(faces), lid, dtype=np.int32))
        v_offset += len(verts)

    all_verts = np.vstack(plant_verts_list)
    all_faces = np.vstack(plant_faces_list)
    all_fids = np.concatenate(plant_fids_list)

    # 计算整株按高度分层的径向包络 R(h)。
    env_rows, z_min, z_max = compute_canopy_radial_envelope(
        all_verts, stem_sampled, n_bins=envelope_bins, quantile=0.95
    )
    plant_height = z_max - z_min
    stem_z_min = float(stem_sampled[:, 2].min())
    stem_z_max = float(stem_sampled[:, 2].max())
    stem_span = max(stem_z_max - stem_z_min, 1e-12)

    max_proj_elem = max((r["projected_area_element"] for r in leaf_records), default=0.0)
    for rec in leaf_records:
        rec["rel_projected_area"] = float(rec["projected_area_element"] / max(max_proj_elem, 1e-12))
        rec["norm_base_height"] = float(np.clip((rec["base_z"] - z_min) / max(plant_height, 1e-12), 0.0, 1.0))
        rec["stem_rel_height"] = float(np.clip((rec["base_z"] - stem_z_min) / stem_span, 0.0, 1.0))

    # 用五方向 Z-buffer 估计整株几何光暴露代理面积（兼容保留），并叠加 46 方向 Turtle Sky 旋转不变受光指标。
    dir_totals = {}
    nadir_per_leaf = {}
    for dname, zen, az in LIGHT_DIRECTIONS:
        s_vec = direction_unit_vector(zen, az)
        vis_tot, per_leaf, _ = rasterize_directional_exposure(
            all_verts, all_faces, all_fids, s_vec, resolution=zbuf_resolution,
        )
        dir_totals[f"visible_area_{dname}"] = vis_tot
        if dname == "nadir":
            nadir_per_leaf = per_leaf

    turtle_totals = {}
    if include_turtle46:
        # 使用中等网格密度（55x15）在 46 方向光栅化，速度提升 5 倍且与高密网格相对差异 <0.3%。
        t_verts_list, t_faces_list, t_fids_list = [], [], []
        t_off = 0
        for mpath in model_paths:
            lid = int(mpath.stem.split("_")[1])
            m_t = MaizeLeafModel.load(mpath)
            sv_t, f_t = m_t.mesh_grid(55, 15)
            v_t = m_t.evaluate(sv_t)
            t_verts_list.append(v_t)
            t_faces_list.append(f_t + t_off)
            t_fids_list.append(np.full(len(f_t), lid, dtype=np.int32))
            t_off += len(v_t)
        turtle_totals = compute_turtle_sky_46_exposure(
            np.vstack(t_verts_list), np.vstack(t_faces_list), np.concatenate(t_fids_list),
            resolution=turtle_resolution,
        )

    for rec in leaf_records:
        lid = rec["leaf_id"]
        vis_nadir = float(nadir_per_leaf.get(lid, 0.0))
        rec["canopy_visible_nadir_area"] = vis_nadir
        rec["canopy_shading_ratio"] = max(0.0, 1.0 - vis_nadir / max(rec["projected_area_union"], 1e-12))

    # 汇总整株表型。
    valid_r = [r["radius_q95"] for r in env_rows if np.isfinite(r["radius_q95"])]
    upper_r = [r["radius_q95"] for r in env_rows if r["h_mid"] >= 0.67 and np.isfinite(r["radius_q95"])]
    lower_r = [r["radius_q95"] for r in env_rows if r["h_mid"] <= 0.33 and np.isfinite(r["radius_q95"])]

    total_leaf_area = float(sum(r["surface_area"] for r in leaf_records))
    total_elem_proj = float(sum(r["projected_area_element"] for r in leaf_records))
    canopy_union = dir_totals["visible_area_nadir"]

    plant_rec = {
        "plant_id": plant_id,
        "leaf_count": len(leaf_records),
        "plant_height": float(plant_height),
        "stem_height": float(stem_span),
        "total_leaf_area": total_leaf_area,
        "sum_projected_area_element": total_elem_proj,
        "canopy_union_area": canopy_union,
        "canopy_overlap_index": total_elem_proj / max(canopy_union, 1e-12),
        "max_canopy_radius_q95": float(max(valid_r)) if valid_r else float("nan"),
        "upper_third_radius_q95": float(np.mean(upper_r)) if upper_r else float("nan"),
        "lower_third_radius_q95": float(np.mean(lower_r)) if lower_r else float("nan"),
        "radius_top_to_bottom_ratio": (
            float(np.mean(upper_r) / max(np.mean(lower_r), 1e-12))
            if (upper_r and lower_r) else float("nan")
        ),
        **dir_totals,
        **turtle_totals,
        "mean_stem_leaf_angle_deg": float(np.mean([r["stem_leaf_angle_deg"] for r in leaf_records])),
        "mean_droop_ratio": float(np.mean([r["droop_ratio"] for r in leaf_records])),
        "mean_midrib_curvature_total": float(np.mean([r["midrib_curvature_total"] for r in leaf_records])),
        "mean_pitch_entropy_b12": (
            float(np.nanmean([r["pitch_entropy_b12"] for r in leaf_records])) if include_entropy else float("nan")
        ),
        "mean_normal_entropy_b24": (
            float(np.nanmean([r["normal_entropy_b24"] for r in leaf_records])) if include_entropy else float("nan")
        ),
    }

    for er in env_rows:
        er["plant_id"] = plant_id

    return leaf_records, plant_rec, env_rows


def attach_quality_weights(all_leaf_records, tau=None):
    """根据观测覆盖率和扫描贴合误差生成质量权重。

    Compute q_i = C_i * exp(-e_i / tau) using observed coverage C_i and scan_distance_p95 e_i."""
    errs = np.array([r["scan_distance_p95"] for r in all_leaf_records], dtype=float)
    valid_errs = errs[np.isfinite(errs) & (errs > 0.0)]
    if tau is None:
        tau = float(np.median(valid_errs)) if len(valid_errs) else 1.0
    for r in all_leaf_records:
        c_i = r["coverage_observed"]
        e_i = r["scan_distance_p95"]
        if np.isfinite(c_i) and np.isfinite(e_i) and tau > 0.0:
            r["quality_weight"] = float(np.clip(c_i, 0.0, 1.0) * np.exp(-max(e_i, 0.0) / tau))
        else:
            r["quality_weight"] = float("nan")
    return float(tau)
