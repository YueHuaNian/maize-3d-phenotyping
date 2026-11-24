"""Unit tests for literature-backed maize leaf modeling components."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial import cKDTree
from maize_leaf.geometry import double_reflection_rmf, fit_pspline_gcv, project_points_to_curve, reparameterize_arc_length
from maize_leaf.morphology import DornbuschBlade, MaizeShapePrior, fit_dornbusch_blade
from maize_leaf.pipeline import run
from maize_leaf.surface import MaizeLeafModel, ModelConfig, ThinPlateSplineGCV, fit_leaf, triangle_areas


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

    def test_full_leaf_completion_and_serialization(self):
        xyz, normals, stem, _, hole, margin, tip = make_synthetic_maize_leaf()
        missing = hole | margin | tip
        res = fit_leaf(xyz[~missing], stem, normals=normals[~missing], cfg=ModelConfig(nu=80, nv=21), leaf_id=3)
        self.assertTrue(np.isfinite(res["vertices"]).all())
        self.assertTrue(np.all(triangle_areas(res["vertices"], res["faces"]) > 0))
        # Tip length should be extended beyond truncated observation.
        self.assertGreater(res["traits"]["completed_length"], res["traits"]["observed_length"])
        # Hole reconstruction accuracy.
        err_hole = cKDTree(res["vertices"]).query(xyz[hole])[0]
        self.assertLess(float(np.median(err_hole)), 0.008)
        # Model round-trip.
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "leaf.npz"
            res["model"].save(p)
            reloaded = MaizeLeafModel.load(p)
            sv = np.array([[0.2, -0.5], [0.5, 0.0], [0.8, 0.4]])
            np.testing.assert_allclose(res["model"].evaluate(sv), reloaded.evaluate(sv), atol=1e-10)

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


if __name__ == "__main__":
    unittest.main()
