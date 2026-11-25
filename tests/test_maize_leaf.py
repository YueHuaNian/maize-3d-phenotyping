"""Unit tests for literature-backed maize leaf modeling components."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree
from maize_leaf.geometry import clean_leaf_points, double_reflection_rmf, extract_midrib, fit_pspline_gcv, l1_medial_skeleton, project_points_to_curve, reparameterize_arc_length
from maize_leaf.morphology import DornbuschBlade, MaizeShapePrior, fit_dornbusch_blade
from maize_leaf.pipeline import run
from maize_leaf.surface import MaizeLeafModel, ModelConfig, ThinPlateSplineGCV, fit_leaf, leaf_mesh_grid, surface_orientation_diagnostics, triangle_areas
from maize_leaf.geometry import MovingFrame
from maize_leaf.quality import mesh_diagnostics, surface_support_diagnostics


def make_synthetic_maize_leaf():
    """Analytical maize blade following Dornbusch (2011) width + parabolic droop + transverse arch."""
    L = 1.0
    blade = DornbuschBlade(length=L, w_max=0.14, xm=0.35, k_lig=0.60, c1=1.8, c2=2.0)
    s_vals = np.linspace(0.0, L, 95)
    v_vals = np.linspace(-1.0, 1.0, 27)
    ss, vv = np.meshgrid(s_vals, v_vals, indexing="ij")
    w = blade.width(ss)

    # Vertical-plane midrib C(s) = (x(s), 0, z(s))
    x = ss
    z = 0.22 * np.sin(0.95 * np.pi * ss)
    y = 0.5 * vv * w
    # Normal arch
    dz = 0.22 * 0.95 * np.pi * np.cos(0.95 * np.pi * ss)
    tnorm = np.sqrt(1.0 + dz ** 2)
    tx, tz = 1.0 / tnorm, dz / tnorm
    nx, nz = -tz, tx
    q = -0.018 * (vv ** 2) * (w / blade.w_max)

    px = x + q * nx
    py = y
    pz = z + q * nz
    xyz = np.column_stack([px.ravel(), py.ravel(), pz.ravel()])
    normals = np.column_stack([nx.ravel(), np.zeros(px.size), nz.ravel()])
    stem = np.column_stack([np.zeros(80), np.zeros(80), np.linspace(-0.3, 0.2, 80)])

    hole = ((ss - 0.50) / 0.08) ** 2 + (vv / 0.45) ** 2 < 1.0
    tip = ss > 0.85
    margin = (ss > 0.25) & (ss < 0.42) & (vv > 0.55)
    return xyz, normals, stem, blade, hole.ravel(), margin.ravel(), tip.ravel()


class MaizeLeafV2Tests(unittest.TestCase):
    def test_dornbusch_boundaries_and_form_factor(self):
        b = DornbuschBlade(length=1.2, w_max=0.15, xm=0.35, k_lig=0.62, c1=1.8, c2=2.0)
        self.assertAlmostEqual(float(b.width(0.0)), 0.15 * 0.62, places=10)
        self.assertAlmostEqual(float(b.width(1.2 * 0.35)), 0.15, places=10)
        self.assertAlmostEqual(float(b.width(1.2)), 0.0, places=10)
        # Numerical integral matches closed-form alpha.
        s = np.linspace(0.0, 1.2, 5000)
        num_alpha = float(np.trapezoid(b.width(s), s) / (1.2 * 0.15))
        self.assertAlmostEqual(num_alpha, b.form_factor, places=4)
        self.assertTrue(0.65 < b.form_factor < 0.80)

    def test_dornbusch_tip_length_inversion(self):
        truth = DornbuschBlade(length=1.0, w_max=0.14, xm=0.35, k_lig=0.60, c1=1.85, c2=2.0)
        s_obs = np.linspace(0.02, 0.82, 30)
        w_obs = truth.width(s_obs)
        fitted = fit_dornbusch_blade(s_obs, w_obs, length_obs=0.82, prior=MaizeShapePrior(), allow_tip_completion=True)
        self.assertAlmostEqual(fitted.length, 1.0, delta=0.06)
        self.assertAlmostEqual(fitted.w_max, 0.14, delta=0.01)

    def test_length_cap_is_explicitly_flagged_without_forcing_short_tip(self):
        s = np.linspace(.02, .98, 30)
        fitted = fit_dornbusch_blade(s, np.full(len(s), .14), 1.)
        self.assertGreater(fitted.length, 1.)
        self.assertLessEqual(fitted.length, 1.45)
        self.assertEqual(fitted.completion_status, "boundary_limited")

    def test_extension_follows_final_surface_center_instead_of_reference_line(self):
        s = np.linspace(0., 1.2, 60)
        controls = np.column_stack([np.linspace(0., 1.2, 4), np.zeros(4), np.zeros(4)])
        curve = BSpline(np.r_[np.zeros(4), np.full(4, 1.2)], controls, 3)
        t, b, n = double_reflection_rmf(curve(s), curve(s, nu=1), n0=np.array([0., 0., 1.]))
        tps = ThinPlateSplineGCV(np.array([[0, 0], [1, 0], [0, 1]]), np.zeros(3), np.array([.2, .1, 0.]), 1., 3.)
        model = MaizeLeafModel(curve, MovingFrame(s, t, b, n),
                               DornbuschBlade(1.2, .14, .35, .6, 1.8, 2.), tps, 1.)
        np.testing.assert_allclose(model.evaluate([[1.2, 0.]])[0], [1.2, 0., .32], atol=1e-5)
        near = model.evaluate([[1.-1e-6, 0.], [1.+1e-6, 0.]])
        self.assertLess(np.linalg.norm(near[1]-near[0]), 3e-6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "extended.npz"
            model.save(path)
            np.testing.assert_allclose(MaizeLeafModel.load(path).evaluate([[1.2, 0.]]),
                                       model.evaluate([[1.2, 0.]]), atol=1e-12)
            with np.load(path) as saved:
                legacy = {key: saved[key] for key in saved.files if key != "surface_tip_extension"}
            np.savez_compressed(path, **legacy)
            self.assertFalse(MaizeLeafModel.load(path).surface_tip_extension)

    def test_rmf_orthonormality_and_twist_free(self):
        t = np.linspace(0.0, 2.0, 100)
        pts = np.column_stack([t, 0.2 * np.sin(t), 0.3 * np.cos(t)])
        tan = np.column_stack([np.ones_like(t), 0.2 * np.cos(t), -0.3 * np.sin(t)])
        T, B, N = double_reflection_rmf(pts, tan)
        for v in (T, B, N):
            np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-10)
        np.testing.assert_allclose(np.sum(T * B, axis=1), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.sum(T * N, axis=1), 0.0, atol=1e-10)
        np.testing.assert_allclose(np.sum(B * N, axis=1), 0.0, atol=1e-10)

    def test_pspline_gcv_and_exact_projection(self):
        rng = np.random.default_rng(0)
        s = np.linspace(0.0, 1.0, 60)
        pts = np.column_stack([s, np.zeros_like(s), 0.2 * np.sin(np.pi * s)])
        noisy = pts + rng.normal(0, 0.002, pts.shape)
        curve, lam = fit_pspline_gcv(s, noisy)
        self.assertGreater(lam, 0.0)
        arc_c, L = reparameterize_arc_length(curve)
        s_proj = project_points_to_curve(pts, arc_c, L)
        self.assertTrue(np.all(np.diff(s_proj) >= -1e-4))

    def test_tps_gcv_surface_recovery(self):
        rng = np.random.default_rng(1)
        xy = rng.uniform(-1.0, 1.0, (120, 2))
        y = 0.2 * xy[:, 0] ** 2 - 0.15 * xy[:, 1] ** 2 + rng.normal(0, 0.003, 120)
        tps = ThinPlateSplineGCV.fit(xy, y)
        self.assertGreater(tps.lam, 0.0)
        pred = tps.evaluate(xy)
        self.assertLess(float(np.sqrt(np.mean((pred - y) ** 2))), 0.01)

    def test_shared_geometry_cache_preserves_candidate_surfaces(self):
        xyz, normals, stem, _, hole, margin, _ = make_synthetic_maize_leaf()
        visible = ~(hole | margin)
        points, nrm = xyz[visible], normals[visible]
        cache = {}
        cfg = ModelConfig(max_tps_points=60, refine_tps_points=60, nu=30, nv=11)
        for mode in ("joint", "initial", "smoothed", "axis", "linear"):
            shared = fit_leaf(points, stem, normals=nrm, cfg=cfg,
                              _geometry_mode=mode, _geometry_cache=cache)
            independent = fit_leaf(points, stem, normals=nrm, cfg=cfg, _geometry_mode=mode)
            np.testing.assert_allclose(shared["vertices"], independent["vertices"], atol=1e-12)

    def test_full_leaf_completion_and_serialization(self):
        xyz, normals, stem, _, hole, margin, tip = make_synthetic_maize_leaf()
        missing = hole | margin | tip
        res = fit_leaf(xyz[~missing], stem, normals=normals[~missing], cfg=ModelConfig(nu=80, nv=21), leaf_id=3)
        self.assertTrue(np.isfinite(res["vertices"]).all())
        self.assertTrue(np.all(triangle_areas(res["vertices"], res["faces"]) > 0))
        np.testing.assert_allclose(res["midrib"], res["model"].evaluate(np.column_stack([res["s"], np.zeros(len(res["s"]))])))
        self.assertTrue(res["quality"]["passed"])
        # Tip length should be extended beyond truncated observation.
        self.assertGreater(res["traits"]["completed_length"], res["traits"]["observed_length"])
        # Hole & margin reconstruction accuracy.
        err_hole = cKDTree(res["vertices"]).query(xyz[hole])[0]
        err_margin = cKDTree(res["vertices"]).query(xyz[margin])[0]
        self.assertLess(float(np.median(err_hole)), 0.008)
        self.assertLess(float(np.median(err_margin)), 0.012)
        # Model round-trip.
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "leaf.npz"
            res["model"].save(p)
            reloaded = MaizeLeafModel.load(p)
            sv = np.array([[0.2, -0.5], [0.5, 0.0], [0.8, 0.4]])
            np.testing.assert_allclose(res["model"].evaluate(sv), reloaded.evaluate(sv), atol=1e-10)

    def test_sor_denoising_and_l1_skeleton(self):
        xyz, normals, _, _, _, _, _ = make_synthetic_maize_leaf()
        rng = np.random.default_rng(7)
        outliers = np.array([[0.5, 0.4, 0.3], [0.2, -0.45, -0.1], [0.8, 0.35, 0.5]])
        out_nrm = rng.normal(size=(3, 3))
        out_nrm /= np.linalg.norm(out_nrm, axis=1, keepdims=True)
        pts_c, nrm_c = clean_leaf_points(np.vstack([xyz, outliers]), np.vstack([normals, out_nrm]))
        expected_n = len(np.unique(xyz, axis=0))
        self.assertEqual(len(pts_c), expected_n)
        self.assertEqual(len(nrm_c), expected_n)
        skel, _ = l1_medial_skeleton(xyz, n_samples=120)
        self.assertLess(float(np.std(skel[:, 1])), float(np.std(xyz[:, 1])) * 0.45)

    def test_end_to_end_pipeline(self):
        xyz, normals, stem, _, hole, _, _ = make_synthetic_maize_leaf()
        pts = np.vstack([stem, xyz[~hole]])
        nrm = np.vstack([np.tile([1.0, 0.0, 0.0], (len(stem), 1)), normals[~hole]])
        lbl = np.r_[np.zeros(len(stem), dtype=int), np.full(int((~hole).sum()), 2, dtype=int)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            in_dir = root / "in"
            in_dir.mkdir()
            with (in_dir / "p1.txt").open("w", encoding="utf-8") as f:
                f.write(f"//X Y Z R G B Constant Nx Ny Nz\n{len(pts)}\n")
                for p, n, l in zip(pts, nrm, lbl):
                    f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 40 150 90 {l} {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            with (in_dir / "p1.csv").open("w", encoding="utf-8") as f:
                f.write("X,Y,Z,semantic\n")
                for p, l in zip(pts, lbl):
                    f.write(f"{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},{l}\n")
            out = run(in_dir, root / "out", cfg=ModelConfig(nu=35, nv=13), calibrate_prior=False)
            self.assertTrue((out / "index.html").is_file())
            self.assertTrue((out / "traits.csv").is_file())
            self.assertTrue((out / "plant_p1/whole_plant_mesh.ply").is_file())
            self.assertTrue((out / "plant_p1/leaves/leaf_02_model.npz").is_file())

    def test_sparse_disconnected_terminal_fragment_is_preserved(self):
        # A coherent terminal section with <5% of the scan remains leaf data,
        # even when its spacing is greater than that of the main blade.
        x, y = np.meshgrid(np.linspace(0.0, 0.75, 160), np.linspace(-0.04, 0.04, 21))
        main = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
        x, y = np.meshgrid(np.linspace(0.90, 1.0, 16), np.linspace(-0.02, 0.02, 5))
        tip = np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])
        points = np.vstack([main, tip, [[0.4, 0.6, 0.5]]])
        normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
        cleaned, nrm = clean_leaf_points(points, normals)
        self.assertGreaterEqual(int(np.sum(cleaned[:, 0] >= 0.90)), 70)
        self.assertLess(float(np.max(np.abs(cleaned[:, 1]))), 0.1)
        np.testing.assert_allclose(nrm, np.tile([0.0, 0.0, 1.0], (len(cleaned), 1)))

    def test_broad_blade_midrib_reaches_both_ends(self):
        x, v = np.meshgrid(np.linspace(0.0, 0.30, 75), np.linspace(-1.0, 1.0, 25))
        width = 0.05 + 0.14 * np.sin(np.pi * x / 0.30)
        points = np.column_stack([x.ravel(), (v * width / 2).ravel(), np.zeros(x.size)])
        stem = np.column_stack([np.zeros(40), np.zeros(40), np.linspace(-0.2, 0, 40)])
        curve, length, _, _, _ = extract_midrib(points, stem)
        sampled = curve(np.linspace(0.0, length, 300))
        self.assertLess(float(cKDTree(sampled).query([[0.0, 0.0, 0.0], [0.30, 0.0, 0.0]])[0].max()), 0.015)
        self.assertTrue(np.all(np.diff(sampled[:, 0]) > -1e-4))

    def test_surface_check_detects_folded_strip_with_nonzero_triangles(self):
        # A strip wider than its centerline's bend radius folds over itself,
        # although every sampled triangle can still have positive area.
        radius = 0.05
        s = np.linspace(0.0, 0.14, 100)
        theta = s / radius
        points = np.column_stack([radius * np.sin(theta), radius * (1 - np.cos(theta)), np.zeros(len(s))])
        curve, _ = fit_pspline_gcv(s, points, n_basis=16, lam_bounds=(1e-8, 1e-6))
        T, B, N = double_reflection_rmf(curve(s), curve(s, nu=1), n0=np.array([0.0, 0.0, 1.0]))
        frame = MovingFrame(s, T, B, N)
        blade = DornbuschBlade(s[-1], 0.22, 0.35, 0.6, 1.8, 2.0)
        zero_tps = ThinPlateSplineGCV(np.array([[0, 0], [1, 0], [0, 1]]), np.zeros(3), np.zeros(3), 1.0, 3)
        folded = MaizeLeafModel(curve, frame, blade, zero_tps, s[-1])
        sv, faces = leaf_mesh_grid(s[-1], 90, 25)
        self.assertTrue(np.all(triangle_areas(folded.evaluate(sv), faces) > 0))
        self.assertGreater(surface_orientation_diagnostics(folded)["reversed_fraction"], 0.05)

        straight_points = np.column_stack([s, np.zeros(len(s)), np.zeros(len(s))])
        straight_curve, _ = fit_pspline_gcv(s, straight_points)
        T, B, N = double_reflection_rmf(straight_curve(s), straight_curve(s, nu=1), n0=np.array([0.0, 0.0, 1.0]))
        straight = MaizeLeafModel(straight_curve, MovingFrame(s, T, B, N), blade, zero_tps, s[-1])
        self.assertEqual(surface_orientation_diagnostics(straight)["reversed_fraction"], 0.0)

    def test_tip_is_one_point_and_legacy_loading_preserves_old_geometry(self):
        s = np.linspace(0.0, 1.0, 60)
        curve, _ = fit_pspline_gcv(s, np.column_stack([s, np.zeros(len(s)), np.zeros(len(s))]))
        T, B, N = double_reflection_rmf(curve(s), curve(s, nu=1), n0=np.array([0., 0., 1.]))
        frame = MovingFrame(s, T, B, N)
        blade = DornbuschBlade(1.0, 0.14, 0.35, 0.6, 1.8, 2.0)
        # A transverse height slope previously left a line of different tip points
        # even though the width was exactly zero.
        tps = ThinPlateSplineGCV(np.array([[0, 0], [1, 0], [0, 1]]), np.zeros(3), np.array([0., .01, .04]), 1., 3.)
        model = MaizeLeafModel(curve, frame, blade, tps, 1.0)
        sv = np.column_stack([np.ones(31), np.linspace(-1, 1, 31)])
        np.testing.assert_allclose(model.evaluate(sv), np.tile(model.evaluate([[1., 0.]]), (31, 1)), atol=1e-12)
        self.assertTrue(mesh_diagnostics(model)["passed"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "current.npz"
            model.save(path)
            np.testing.assert_allclose(MaizeLeafModel.load(path).evaluate(sv), model.evaluate(sv), atol=1e-12)
            with np.load(path) as saved:
                legacy = {key: saved[key] for key in saved.files if key != "apex_taper"}
            old_path = Path(folder) / "old.npz"
            np.savez_compressed(old_path, **legacy)
            old_model = MaizeLeafModel.load(old_path)
            self.assertGreater(float(np.ptp(old_model.evaluate(sv)[:, 2])), 0.07)
            self.assertFalse(mesh_diagnostics(old_model)["passed"])

    def test_reverse_support_penalizes_unsupported_wide_surface(self):
        s = np.linspace(0.0, 1.0, 60)
        curve, _ = fit_pspline_gcv(s, np.column_stack([s, np.zeros(len(s)), np.zeros(len(s))]))
        t, b, n = double_reflection_rmf(curve(s), curve(s, nu=1), n0=np.array([0., 0., 1.]))
        frame = MovingFrame(s, t, b, n)
        tps = ThinPlateSplineGCV(np.array([[0, 0], [1, 0], [0, 1]]), np.zeros(3), np.zeros(3), 1., 3.)
        narrow = MaizeLeafModel(curve, frame, DornbuschBlade(1., .10, .35, .6, 1.8, 2.), tps, 1.)
        inflated = MaizeLeafModel(curve, frame, DornbuschBlade(1., .40, .35, .6, 1.8, 2.), tps, 1.)
        scan = narrow.evaluate(leaf_mesh_grid(1., 100, 21)[0])
        self.assertGreater(surface_support_diagnostics(inflated, scan)["mean"],
                           5 * surface_support_diagnostics(narrow, scan)["mean"])


    def test_sor_single_pass_clustering_removes_near_surface_outliers(self):
        # Single-pass clustering must not re-shrink the radius graph after removing a far
        # detached outlier and split a near-leaf halo point into a tiny fragment that bypasses SOR.
        xyz, normals, _, _, _, _, _ = make_synthetic_maize_leaf()
        near_halo = np.array([[0.5, 0.0, 0.085]])
        far_outlier = np.array([[4.0, 4.0, 4.0]])
        pts = np.vstack([xyz, near_halo, far_outlier])
        nrm = np.vstack([normals, [[0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0]]])
        cleaned, _ = clean_leaf_points(pts, nrm)
        self.assertGreater(float(cKDTree(cleaned).query(near_halo[0])[0]), 0.02)
        self.assertGreater(float(cKDTree(cleaned).query(far_outlier[0])[0]), 1.0)

    def test_euclidean_clustering_robust_diagonal_resists_far_outlier_inflation(self):
        # A far stray cluster must not inflate the bounding-box diagonal and pull a nearby
        # detached mislabeled patch into the main blade component.
        xyz, normals, _, _, _, _, _ = make_synthetic_maize_leaf()
        rng = np.random.default_rng(2026)
        # Compact 10-point detached mislabeled patch offset ~0.09 above mid-blade (> 5 * spacing)
        patch = np.array([0.5, 0.0, 0.31]) + rng.normal(scale=0.004, size=(10, 3))
        far_stray = np.array([4.5, 4.5, 4.5]) + rng.normal(scale=0.005, size=(4, 3))
        extra_nrm = np.tile([0.0, 0.0, 1.0], (len(patch) + len(far_stray), 1))
        pts = np.vstack([xyz, patch, far_stray])
        nrm = np.vstack([normals, extra_nrm])
        cleaned, _ = clean_leaf_points(pts, nrm)
        tree = cKDTree(cleaned)
        self.assertGreater(float(tree.query(patch)[0].min()), 0.05)
        self.assertGreater(float(tree.query(far_stray)[0].min()), 1.0)
        self.assertGreaterEqual(len(cleaned), int(0.98 * len(np.unique(xyz, axis=0))))


if __name__ == "__main__":
    unittest.main()
