"""Core modeling, data loading, and pipeline tests."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial import cKDTree
from maize_leaf_v1.data import discover, load_pair
from maize_leaf_v1.model import Config, LeafModel, curve_fit, fit_leaf, triangle_areas
from maize_leaf_v1.pipeline import run


def synthetic():
    u, v = np.meshgrid(np.linspace(0, 1, 90), np.linspace(-1, 1, 25), indexing="ij")
    width = 0.16 * np.sin(np.pi * (0.10 + 0.8 * u))
    xyz = np.column_stack([u.ravel(), (v * width / 2).ravel(), (0.20 * np.sin(np.pi * u) + 0.025 * v * v).ravel()])
    hole = ((u - 0.50) / 0.09) ** 2 + (v / 0.48) ** 2 < 1
    missing_tip = u > 0.94
    missing_edge = (u > 0.27) & (u < 0.40) & (v > 0.6)
    missing = (hole | missing_tip | missing_edge).ravel()
    stem = np.column_stack([np.zeros(80), np.zeros(80), np.linspace(-0.2, 0.1, 80)])
    return xyz, missing, hole.ravel(), stem


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.truth, cls.missing, cls.hole, cls.stem = synthetic()
        cls.result = fit_leaf(cls.truth[~cls.missing], cls.stem, Config(nu=100, nv=31), 7)

    def test_continuous_model_serialization(self):
        r = self.result
        uv = np.array([[0.2, 0.3], [0.55, -0.2], [1.02, 0.7]])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "model.npz"
            r["model"].save(p)
            reloaded = LeafModel.load(p)
            np.testing.assert_allclose(r["model"].evaluate(uv), reloaded.evaluate(uv), rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(reloaded.evaluate([0.55, -0.2]), reloaded.evaluate([[0.55, -0.2]]))

    def test_surface_and_orthonormal_frames(self):
        r = self.result
        self.assertTrue(np.isfinite(r["vertices"]).all())
        self.assertTrue(np.all(triangle_areas(r["vertices"], r["faces"]) > 0))
        t, b, n = r["model"].frame(np.linspace(0, 1.02, 100))
        for x in (t, b, n):
            np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1, atol=1e-10)
        np.testing.assert_allclose(np.sum(t * b, axis=1), 0, atol=1e-10)
        np.testing.assert_allclose(np.cross(t, b), n, atol=1e-10)

    def test_support_labels_and_area(self):
        r = self.result
        self.assertEqual(set(r["regions"]), {0, 1, 2})
        self.assertTrue(np.all(r["regions"][r["uv"][:, 0] > 1] == 2))
        np.testing.assert_array_equal(r["face_regions"], r["regions"][r["faces"]].max(axis=1))
        self.assertAlmostEqual(sum(r["metrics"]["area_fractions"].values()), 1, places=10)

    def test_withheld_hole_geometry(self):
        r = self.result
        error = cKDTree(r["vertices"]).query(self.truth[self.hole])[0]
        self.assertLess(float(np.quantile(error, 0.95)), 0.035)
        idx = cKDTree(r["vertices"]).query(self.truth[self.hole])[1]
        self.assertGreater(float(np.mean(r["regions"][idx] == 1)), 0.5)

    def test_original_leaf_id_preserved(self):
        self.assertEqual(self.result["metrics"]["leaf_id"], 7)

    def test_curve_does_not_overshoot_near_duplicate_nodes(self):
        u = np.r_[0, 0.00001, 0.00002, np.linspace(0.05, 1, 25)]
        xyz = np.column_stack([u, 0.1 * np.sin(u * 5), u * 0.2])
        xyz[1] = xyz[0] + [1e-7, -1e-7, 1e-7]
        c = curve_fit(u, xyz)
        sampled = c(np.linspace(0, 1, 500))
        self.assertTrue(np.all(sampled >= xyz.min(axis=0) - np.ptp(xyz, axis=0) * 0.021))
        self.assertTrue(np.all(sampled <= xyz.max(axis=0) + np.ptp(xyz, axis=0) * 0.021))

    def test_discover_load_and_pipeline(self):
        observed = self.truth[~self.missing]
        pts = np.vstack([self.stem, observed])
        labels = np.r_[np.zeros(len(self.stem), dtype=int), np.full(len(observed), 7, dtype=int)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            in_dir = root / "in"
            in_dir.mkdir()
            txt = in_dir / "p1.txt"
            csv_path = in_dir / "p1.csv"
            with txt.open("w", encoding="utf-8") as f:
                f.write(f"//X Y Z R G B Constant\n{len(pts)}\n")
                for p, lbl in zip(pts, labels):
                    f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 40 150 90 {lbl}\n")
            with csv_path.open("w", encoding="utf-8") as f:
                f.write("X,Y,Z,semantic\n")
                for p, lbl in zip(pts, labels):
                    f.write(f"{p[0]:.6f},{p[1]:.6f},{p[2]:.6f},{lbl}\n")

            pairs = discover(in_dir)
            self.assertEqual(len(pairs), 1)
            plant = load_pair(*pairs[0])
            self.assertEqual(len(plant.xyz), len(pts))

            out = run(in_dir, root / "out", cfg=Config(nu=30, nv=11))
            self.assertTrue((out / "index.html").is_file())
            self.assertTrue((out / "metrics.csv").is_file())
            self.assertTrue((out / "plant_p1/whole_plant.html").is_file())
            self.assertTrue((out / "plant_p1/leaves/leaf_07_surface.ply").is_file())
            self.assertTrue((out / "plant_p1/leaves/leaf_07_model.npz").is_file())


if __name__ == "__main__":
    unittest.main()
