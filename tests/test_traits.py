"""使用解析几何检验表型定义与尺度、旋转不变性。

Analytical mathematical verification tests for V2.0 Stage D traits and normal entropy."""
import unittest
import numpy as np
from scipy.interpolate import BSpline

from maize_leaf.entropy import (
    equal_area_bin_centers,
    normal_hemisphere_entropy,
    surface_pitch_entropy_from_grid,
)
from maize_leaf.geometry import fit_pspline_gcv
from maize_leaf.traits import (
    _basal_surface_tangent_from_grid,
    _fundamental_forms_from_grid,
    _smooth_arc_centerline,
    compute_canopy_radial_envelope,
    direction_unit_vector,
    element_projected_area,
    far_end_height_diff,
    rasterize_directional_exposure,
    stem_tangent_at_point,
)


class StageDMathTraitsTests(unittest.TestCase):
    """使用圆柱、球冠和圆弧检查表型数值公式。"""
    def test_cylinder_fundamental_forms_and_willmore_scale_invariance(self):
        # Cylinder patch of radius R, angle span theta_span, height H_len:
        # Exact H = 1/(2R), K_G = 0, A = R * theta_span * H_len, W = int H^2 dA = A / (4 R^2).
        """检查圆柱曲率与 Willmore 泛函的尺度不变性。"""
        R = 0.25
        theta_vals = np.linspace(0.0, 0.8, 61)
        z_vals = np.linspace(0.0, 1.0, 51)
        tt, zz = np.meshgrid(theta_vals, z_vals, indexing="ij")
        xyz = np.stack([R * np.cos(tt), R * np.sin(tt), zz], axis=2)

        res = _fundamental_forms_from_grid(xyz, theta_vals, z_vals)
        area_exact = R * 0.8 * 1.0
        w_exact = area_exact / (4.0 * R * R)
        self.assertAlmostEqual(res["surface_interior_area"], area_exact, delta=1e-4)
        self.assertAlmostEqual(res["mean_curvature_abs"], 1.0 / (2.0 * R), delta=1e-3)
        self.assertAlmostEqual(res["gaussian_curvature_abs"], 0.0, delta=1e-4)
        self.assertAlmostEqual(res["willmore_bending_functional"], w_exact, delta=2e-3)

        # Uniform 10x scaling must leave Willmore bending functional strictly invariant.
        res_scaled = _fundamental_forms_from_grid(xyz * 10.0, theta_vals, z_vals)
        self.assertAlmostEqual(res_scaled["willmore_bending_functional"], res["willmore_bending_functional"], delta=1e-6)
        self.assertAlmostEqual(res_scaled["mean_curvature_abs_dimless"], res["mean_curvature_abs_dimless"], delta=1e-6)

    def test_sphere_cap_fundamental_forms(self):
        # Sphere patch of radius R: exact |H| = 1/R, |K_G| = 1/R^2, W = A / R^2.
        """检查球冠第一、第二基本形式导出的曲率。"""
        R = 0.5
        phi = np.linspace(0.4, 1.1, 61)
        lam = np.linspace(0.0, 0.9, 51)
        pp, ll = np.meshgrid(phi, lam, indexing="ij")
        xyz = np.stack([R * np.sin(pp) * np.cos(ll), R * np.sin(pp) * np.sin(ll), R * np.cos(pp)], axis=2)

        res = _fundamental_forms_from_grid(xyz, phi, lam)
        self.assertAlmostEqual(res["mean_curvature_abs"], 1.0 / R, delta=3e-3)
        self.assertAlmostEqual(res["gaussian_curvature_abs"], 1.0 / (R * R), delta=1e-2)
        self.assertAlmostEqual(
            res["willmore_bending_functional"], res["surface_interior_area"] / (R * R), delta=5e-3
        )

    def test_quarter_circle_total_midrib_curvature_and_scale_invariance(self):
        # Quarter circle arc of radius R has exact total absolute curvature K_mid = pi / 2 (independent of R).
        """检查四分之一圆弧总曲率及尺度不变性。"""
        for R in (0.4, 4.0):
            ang = np.linspace(0.0, 0.5 * np.pi, 300)
            pts = np.column_stack([R * np.sin(ang), np.zeros_like(ang), R * np.cos(ang)])
            curve, _, total_len = _smooth_arc_centerline(pts)
            ll = np.linspace(0.0, total_len, 300)
            d1 = curve(ll, nu=1)
            d2 = curve(ll, nu=2)
            kappa = np.linalg.norm(np.cross(d1, d2), axis=1) / np.maximum(np.linalg.norm(d1, axis=1), 1e-12) ** 3
            k_tot = float(np.trapezoid(kappa[3:-3], ll[3:-3]))
            expected = 0.5 * np.pi * (ll[-4] - ll[3]) / total_len
            self.assertAlmostEqual(k_tot, expected, delta=0.02)

    def test_tilted_stem_leaf_angle_uses_stem_tangent_not_global_z(self):
        # Stem tilted 20 deg from +Z toward +X.
        """检查倾斜茎秆的茎叶角使用局部茎切向。"""
        tilt = np.radians(20.0)
        t_stem_true = np.array([np.sin(tilt), 0.0, np.cos(tilt)])
        s = np.linspace(0.0, 1.0, 40)
        stem_pts = s[:, None] * t_stem_true[None, :]
        stem_curve, _ = fit_pspline_gcv(s, stem_pts, lam_bounds=(1e-6, 1e-2), bounded=False)
        t_bounds = (float(stem_curve.t[stem_curve.k]), float(stem_curve.t[-stem_curve.k - 1]))

        _, _, t_s = stem_tangent_at_point(stem_curve, t_bounds, 1.0, np.array([0.1, 0.0, 0.5]))
        # Leaf pointing vertically along +Z: angle to global Z is 0 deg, angle to stem is 20 deg.
        t_leaf = np.array([0.0, 0.0, 1.0])
        ang_deg = float(np.degrees(np.arccos(np.clip(np.dot(t_s, t_leaf), -1.0, 1.0))))
        self.assertAlmostEqual(ang_deg, 20.0, delta=0.1)

    def test_basal_surface_tangent_roll_invariance_and_obtuse_preservation(self):
        # Vertical stem t_s = (0, 0, 1). Test planar blades at exact true angles 45 deg and 125 deg
        # with strong 60 deg lateral blade roll around the longitudinal axis.
        """检查基部叶面角不受滚转影响且保留钝角。"""
        t_s = np.array([0.0, 0.0, 1.0])
        roll = np.radians(60.0)
        s_vals = np.linspace(0.0, 0.12, 16)
        v_vals = np.linspace(-0.8, 0.8, 17)
        ss, vv = np.meshgrid(s_vals, v_vals, indexing="ij")

        for true_deg in (45.0, 125.0):
            rad = np.radians(true_deg)
            t_true = np.array([np.sin(rad), 0.0, np.cos(rad)])
            b0 = np.array([0.0, 1.0, 0.0])
            n0 = np.cross(t_true, b0)
            b_rolled = np.cos(roll) * b0 + np.sin(roll) * n0
            xyz = ss[:, :, None] * t_true[None, None, :] + (0.05 * vv)[:, :, None] * b_rolled[None, None, :]
            t_surf, b_surf = _basal_surface_tangent_from_grid(xyz, s_vals, v_vals)
            est_deg = float(np.degrees(np.arccos(np.clip(np.dot(t_s, t_surf), -1.0, 1.0))))
            self.assertAlmostEqual(est_deg, true_deg, delta=1e-6)
            # Verify roll coupling is indeed large (proving normal line-to-plane angle would fail)
            self.assertGreater(abs(float(np.dot(t_s, b_surf))), 0.4)

    def test_zbuffer_directional_exposure_and_occlusion_union(self):
        # Two stacked 1x1 horizontal squares at z=0 (leaf 1) and z=1 (leaf 2).
        """检查方向投影的深度遮挡与面积并集。"""
        v_bot = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=float)
        v_top = np.array([[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], dtype=float)
        verts = np.vstack([v_bot, v_top])
        faces = np.array([[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]], dtype=np.int32)
        fids = np.array([1, 1, 2, 2], dtype=np.int32)

        # Element projection sums both squares = 2.0
        area_tot, elem_proj = element_projected_area(verts, faces, (0, 0, 1))
        self.assertAlmostEqual(area_tot, 2.0, places=6)
        self.assertAlmostEqual(elem_proj, 2.0, places=6)

        # Nadir Z-buffer union = 1.0; top leaf gets 1.0, bottom leaf is completely shaded (0.0)
        vis_nadir, per_leaf, _ = rasterize_directional_exposure(verts, faces, fids, (0, 0, 1), resolution=200)
        self.assertAlmostEqual(vis_nadir, 1.0, delta=0.02)
        self.assertAlmostEqual(per_leaf.get(2, 0.0), 1.0, delta=0.02)
        self.assertEqual(per_leaf.get(1, 0.0), 0.0)

        # Single top square at 60 deg zenith: exact projected exposure = cos(60 deg) * 1.0 = 0.5
        s60 = direction_unit_vector(60.0, 0.0)
        vis_60, _, _ = rasterize_directional_exposure(v_top, faces[:2], fids[:2], s60, resolution=200)
        self.assertAlmostEqual(vis_60, 0.5, delta=0.015)

    def test_hemisphere_normal_entropy_bounds_and_sign_invariance(self):
        # 1. Flat horizontal surface with arbitrary +/- normal flips => entropy = 0.0
        """检查法向熵取值边界及正负法向等价性。"""
        n_flat = np.vstack([np.tile([0.0, 0.0, 1.0], (50, 1)), np.tile([0.0, 0.0, -1.0], (50, 1))])
        a_flat = np.ones(100)
        for nb in (12, 24, 48):
            self.assertAlmostEqual(normal_hemisphere_entropy(n_flat, a_flat, n_bins=nb), 0.0, places=10)

        # 2. Equal-solid-angle uniform centers across all bins => entropy = 1.0
        for nb in (12, 24, 48):
            vecs = equal_area_bin_centers(n_bins=nb)
            self.assertEqual(len(vecs), nb)
            self.assertAlmostEqual(normal_hemisphere_entropy(vecs, np.ones(nb), n_bins=nb), 1.0, places=10)

    def test_pitch_entropy_roll_invariance_and_arch_growth(self):
        # Straight planar blade at exact +40 deg pitch with strong 60 deg lateral roll -> H_pit == 0.0
        """检查俯仰熵对滚转不变及随纵向拱曲变化。"""
        pitch_rad = np.radians(40.0)
        roll_rad = np.radians(60.0)
        t_dir = np.array([np.cos(pitch_rad), 0.0, np.sin(pitch_rad)])
        b0 = np.array([0.0, 1.0, 0.0])
        n0 = np.cross(t_dir, b0)
        b_rolled = np.cos(roll_rad) * b0 + np.sin(roll_rad) * n0

        s_vals = np.linspace(0.0, 1.0, 40)
        v_vals = np.linspace(-0.9, 0.9, 15)
        ss, vv = np.meshgrid(s_vals, v_vals, indexing="ij")
        xyz_flat = ss[:, :, None] * t_dir[None, None, :] + (0.08 * vv)[:, :, None] * b_rolled[None, None, :]
        h_flat, mean_p = surface_pitch_entropy_from_grid(xyz_flat, s_vals, v_vals, n_bins=12)
        self.assertAlmostEqual(h_flat, 0.0, places=10)
        self.assertAlmostEqual(mean_p, 40.0, delta=1e-4)

        # Semicircular arch spanning pitch +90 deg down to -90 deg -> uniform across all 12 bins -> H_pit ~ 1.0
        ang = np.linspace(0.0, np.pi, 180)
        aa, vv2 = np.meshgrid(ang, v_vals, indexing="ij")
        xyz_arch = np.stack([1.0 - np.cos(aa), 0.08 * vv2, np.sin(aa)], axis=2)
        h_arch, mean_arch = surface_pitch_entropy_from_grid(xyz_arch, ang, v_vals, n_bins=12)
        self.assertGreater(h_arch, 0.95)
        self.assertAlmostEqual(mean_arch, 0.0, delta=1.5)

    def test_canopy_radial_envelope_exact_radius(self):
        # Cylindrical point shell of radius 0.3 around a tilted stem x_s(z) = 0.2 * z
        """检查解析冠层的径向包络半径。"""
        zs = np.linspace(0.0, 1.0, 60)
        stem = np.column_stack([0.2 * zs, np.zeros_like(zs), zs])
        ang = np.linspace(0.0, 2.0 * np.pi, 36, endpoint=False)
        zz, aa = np.meshgrid(zs, ang, indexing="ij")
        verts = np.column_stack([
            (0.2 * zz + 0.3 * np.cos(aa)).ravel(),
            (0.3 * np.sin(aa)).ravel(),
            zz.ravel(),
        ])
        rows, z_min, z_max = compute_canopy_radial_envelope(verts, stem, n_bins=10, quantile=0.95)
        self.assertAlmostEqual(z_max - z_min, 1.0, places=6)
        for r in rows:
            self.assertAlmostEqual(r["radius_q95"], 0.3, delta=1e-4)

    def test_far_end_height_diff_upward_and_drooping(self):
        """检查离茎最远端高度差对上翘叶为正、下垂叶为负。"""
        stem = np.column_stack([np.zeros(101), np.zeros(101), np.linspace(0.0, 1.0, 101)])
        up_verts = np.array([[0.05, 0.0, 0.40], [0.25, 0.0, 0.55], [0.50, 0.0, 0.72]])
        res_up = far_end_height_diff(up_verts, stem, base_z=0.40)
        self.assertAlmostEqual(res_up["far_end_stem_dist"], 0.50, places=6)
        self.assertAlmostEqual(res_up["far_end_height_diff"], 0.32, places=6)

        droop_verts = np.array([[0.05, 0.0, 0.40], [0.30, 0.0, 0.48], [0.45, 0.0, 0.22]])
        res_down = far_end_height_diff(droop_verts, stem, base_z=0.40)
        self.assertAlmostEqual(res_down["far_end_stem_dist"], 0.45, places=6)
        self.assertAlmostEqual(res_down["far_end_height_diff"], -0.18, places=6)


if __name__ == "__main__":
    unittest.main()
