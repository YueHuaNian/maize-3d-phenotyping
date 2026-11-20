import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.geometry import arc_lengths
from src.phenotype_clusters import cluster_leaves
from src.phenotype_features import FEATURES, analyse_plant, curve_features, inclination_entropy
from src.phenotype_pipeline import run_analysis


class FeatureTests(unittest.TestCase):
    axis = np.array([0.0, 0.0, 1.0])

    def test_entropy_one_direction_is_zero(self):
        curve = np.array([[0., 0., 0.], [0., 0., 2.], [0., 0., 3.]])
        entropy, p = inclination_entropy(curve, self.axis)
        self.assertEqual(entropy, 0)
        self.assertEqual(p[0], 1)

    def test_entropy_uses_length_and_keeps_180_degree_endpoint(self):
        curve = np.array([[0., 0., 0.], [0., 0., 3.], [0., 0., 2.]])
        entropy, p = inclination_entropy(curve, self.axis)
        self.assertAlmostEqual(entropy, 0.8112781244591328)
        np.testing.assert_allclose(p[[0, -1]], [0.75, 0.25])
        self.assertAlmostEqual(p.sum(), 1.0)

    def test_droop_spread_and_normalized_curvature(self):
        curve = np.array([[0., 0., 0.], [3., 0., 0.], [3., 0., -1.]])
        fields, _ = curve_features({"curve": curve, "arc": arc_lengths(curve),
                                    "curvature": np.full(3, 0.25)}, self.axis)
        self.assertAlmostEqual(fields["droop_index"], 0.25)
        self.assertAlmostEqual(fields["spread_index"], 0.75)
        self.assertAlmostEqual(fields["curvature_rms_norm"], 1.0)

    def test_degenerate_entropy_has_no_value(self):
        with self.assertRaisesRegex(ValueError, "zero-length"):
            inclination_entropy(np.zeros((3, 3)), self.axis)

    def test_unavailable_stem_retains_leaf_and_explicit_missing_values(self):
        xyz = np.array([[0., 0., 1.], [0., 1., 1.]])
        rows, stem, _, _ = analyse_plant("plant", xyz, np.array([-1, 9]))
        self.assertEqual(stem["status"], "STEM_FIT_UNAVAILABLE")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["leaf_id"], 9)
        self.assertEqual(rows[0]["status"], "no_valid_stem_points")
        for name in FEATURES:
            self.assertIsNone(rows[0][name])
            self.assertEqual(rows[0][f"{name}_status"], "no_valid_stem_points")

    def test_nearest_pair_sets_height_but_sparse_leaf_cannot_get_angle(self):
        xyz = np.array([[0., 0., x] for x in np.linspace(0, 2, 100)] + [[0.01, 0., 1.]])
        labels = np.array([0] * 100 + [10])
        curve = np.array([[0., 0., 0.], [0., 0., 1.], [0., 0., 2.]])
        stem = {"arc": np.array([0., 1., 2.]), "curve": curve, "curve_supported_mask": np.ones(3, bool),
                "method": "test-curve", "radius": 0.1, "selected": {}, "axis": self.axis,
                "unsupported_curve_fraction": 0.0}
        with patch("src.phenotype_features.robust_stem", return_value=stem):
            rows, _, _, _ = analyse_plant("plant", xyz, labels)
        self.assertEqual(rows[0]["leaf_point_row"], 100)
        expected_height = xyz[rows[0]["stem_point_row"], 2] / 2
        self.assertAlmostEqual(rows[0]["height_ratio"], expected_height)
        self.assertIsNone(rows[0]["leaf_angle_deg"])
        self.assertEqual(rows[0]["leaf_angle_deg_status"], "LEAF_POINTS_TOO_FEW")


def cohort():
    rows = []
    for p in range(4):
        for i in range(3):
            profile = [20 + 25 * i, 0.15 + 0.3 * i, 0.04 + 0.15 * i,
                       0.6 + 1.3 * i, 0.1 + 0.3 * i, 0.5 + i]
            variation = [0.035, .001, .002, .012, .003, .013]
            values = [n + p * delta for n, delta in zip(profile, variation)]
            rows.append({"plant_id": f"P{p}", "leaf_id": i + 1, **dict(zip(FEATURES, values)),
                         **{f"{name}_status": "ok" for name in FEATURES}})
    return rows


class ClusterTests(unittest.TestCase):
    def test_three_clusters_and_group_holdouts(self):
        rows = cohort()
        model, centers, folds = cluster_leaves(rows, ["P0", "P1", "P2", "P3"])
        self.assertEqual(model["cluster_sizes"], {"C1": 4, "C2": 4, "C3": 4})
        self.assertEqual([r["cluster_id"] for r in rows[:3]], ["C1", "C2", "C3"])
        self.assertEqual(sorted(c["height_ratio"] for c in centers), [c["height_ratio"] for c in centers])
        for r in folds:
            self.assertEqual((r["train_plants"], r["train_leaves"], r["held_out_leaves"]), (3, 9, 3))
            self.assertEqual((r["train_ari"], r["held_out_ari"], r["held_out_agreement"]),
                             (1.0, 1.0, 1.0))
        self.assertGreater(model["metrics"]["silhouette"], 0.9)

    def test_scaler_and_model_receive_training_rows_only_in_each_fold(self):
        from src.phenotype_clusters import _fit

        rows = cohort()
        with patch("src.phenotype_clusters._fit", wraps=_fit) as spy:
            cluster_leaves(rows, ["P0", "P1", "P2", "P3"])
        feature = list(FEATURES)[0]
        for index in range(4):
            observed = spy.call_args_list[index + 1].args[0][:, 0].tolist()
            self.assertEqual(observed, [r[feature] for r in rows if r["plant_id"] != f"P{index}"])

    def test_missing_data_keeps_reason_and_reports_unevaluable_plant(self):
        rows = cohort()
        for row in rows[:3]:
            row.update(shannon_entropy_bits=None, shannon_entropy_bits_status="CURVE_SUPPORT_LOW")
        model, _, folds = cluster_leaves(rows, ["P0", "P1", "P2", "P3"])
        self.assertEqual(model["assigned_leaves"], 9)
        self.assertIsNone(rows[0]["cluster_id"])
        self.assertIn("CURVE_SUPPORT_LOW", rows[0]["cluster_exclusion_reasons"])
        self.assertEqual(folds[0]["status"], "NO_COMPLETE_HELD_OUT_LEAVES")

    def test_single_plant_retains_features_without_clustering(self):
        rows = cohort()[:3]
        model, _, folds = cluster_leaves(rows, ["P0"])
        self.assertEqual(model["status"], "SINGLE_PLANT")
        self.assertEqual(model["complete_leaves"], 3)
        self.assertEqual(folds, [])


class PipelineTests(unittest.TestCase):
    def test_mismatched_input_is_reported_and_existing_output_is_protected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            csv_dir, txt_dir, out = (root / item for item in ("csv", "txt", "out"))
            csv_dir.mkdir()
            txt_dir.mkdir()
            (csv_dir / "P.csv").write_text("X,Y,Z,semantic\n1,0,0,9\n")
            (txt_dir / "P.txt").write_text("//X Y Z Constant\n1\n2 0 0 9\n")
            with patch("src.phenotype_pipeline.write_figures"):
                self.assertEqual(run_analysis(csv_dir, txt_dir, out), 1)
            self.assertIn("XYZ differ at point row 0", (out / "input_manifest.csv").read_text(encoding="utf-8-sig"))
            leaf_csv = (out / "leaf_features.csv").read_text(encoding="utf-8-sig").splitlines()
            self.assertEqual(len(leaf_csv), 1)
            with self.assertRaises(FileExistsError):
                run_analysis(csv_dir, txt_dir, out)


if __name__ == "__main__":
    unittest.main()
