import csv
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from src.nearest_attachment import locate_attachments
from src.phenotype_clusters import fit_standardized_kmeans
from src.plant_clustering import FIELDS, FEATURE_WEIGHTS, MORPHOTYPES, _observed_profile_features, cluster_plant
from view_phenotypes import read_results


def sample(profiles):
    xyz = [[0., 0., float(z)] for z in range(11)]
    labels = [0] * len(xyz)
    t = np.linspace(0, 1, 161)
    for leaf, height, x, y in profiles:
        xyz.extend(np.column_stack((0.01 + x * t, np.zeros_like(t), height + y * t)))
        labels.extend([leaf] * len(t))
    xyz, labels = np.asarray(xyz, float), np.asarray(labels)
    return xyz, labels, [asdict(r) for r in locate_attachments("P", xyz, labels)]


class PlantClusteringTests(unittest.TestCase):
    def classify(self, profiles):
        with threadpool_limits(limits=1):
            return cluster_plant("P", *sample(profiles))

    def test_three_nonempty_groups_with_all_leaf_ids(self):
        rows, summary, centers = self.classify([
            (8, 8, .40, .16), (3, 8, .41, .164),
            (12, 2, 2.0, .10), (1, 2, 2.05, .1025),
            (9, 5, 1.0, -.8), (20, 5, 1.02, -.816),
        ])
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["assigned_leaves"], 6)
        self.assertEqual([summary[c] for c in ("C1", "C2", "C3")], [2, 2, 2])
        by_leaf = {r["leaf_id"]: r["plant_cluster_id"] for r in rows}
        self.assertEqual(by_leaf, {3: "C1", 8: "C1", 1: "C2", 12: "C2", 9: "C3", 20: "C3"})
        self.assertEqual([c["morphotype"] for c in centers], list(MORPHOTYPES.values()))
        self.assertLess(centers[0]["horizontal_reach_q95_ratio"], centers[1]["horizontal_reach_q95_ratio"])
        self.assertGreater(centers[2]["downward_path_fraction"], centers[1]["downward_path_fraction"])
        self.assertTrue(all(r["profile_sections"] >= 3 for r in rows))

    def test_entropy_has_largest_squared_distance_weight(self):
        matrix = np.arange(24.0).reshape(6, 4)
        with threadpool_limits(limits=1):
            scaler, weighted, _ = fit_standardized_kmeans(matrix, FEATURE_WEIGHTS)
        np.testing.assert_allclose(weighted ** 2, scaler.transform(matrix) ** 2 * FEATURE_WEIGHTS)
        self.assertEqual(FEATURE_WEIGHTS[0], 0.55)

    def test_distal_drop_can_occur_above_leaf_base(self):
        t = np.linspace(0, 1, 257)
        profile = np.column_stack((2 * t, np.zeros_like(t), 0.4 * np.sin(2.6 * t)))
        self.assertGreater(profile[-1, 2], 0)
        measured = _observed_profile_features(profile)
        self.assertGreater(measured["observed_shannon_entropy_bits"], 0)
        self.assertGreater(measured["distal_drop_ratio"], 0)
        self.assertGreater(measured["downward_path_fraction"], 0)

    def test_exactly_three_leaves_have_three_clusters_without_silhouette(self):
        _, summary, _ = self.classify([(1, 2, .5, 0), (5, 5, 1, .2), (9, 8, 2, .4)])
        self.assertEqual([summary[k] for k in ("C1", "C2", "C3")], [1, 1, 1])
        self.assertEqual(summary["singleton_clusters"], 3)
        self.assertIsNone(summary["silhouette"])

    def test_too_few_or_identical_leaves_are_reported(self):
        first = [(1, 2, .5, 0), (2, 2, .5, 0), (3, 2, .5, 0)]
        for profile, status in ((first[:2], "TOO_FEW_LEAVES"),
                                (first, "TOO_FEW_DISTINCT_GEOMETRIES")):
            with self.subTest(status=status):
                rows, summary, _ = self.classify(profile)
                self.assertEqual(summary["status"], status)
                self.assertEqual(len(rows), len(profile))
                self.assertTrue(all(r["plant_cluster_id"] is None for r in rows))

    def test_missing_point_geometry_is_not_fabricated(self):
        xyz, labels, leaves = sample([(1, 2, .5, 0), (5, 5, 1, .2), (9, 8, 2, .4)])
        leaves[0]["status"] = "no_valid_leaf_points"
        rows, summary, centers = cluster_plant("P", xyz, labels, leaves)
        self.assertEqual(summary["status"], "MISSING_OBSERVED_GEOMETRY")
        self.assertIsNone(rows[0][FIELDS[0]])
        self.assertIsNotNone(rows[1][FIELDS[0]])
        self.assertTrue(all(r["plant_cluster_id"] is None for r in rows))
        self.assertEqual(centers, [])


class ViewerScopeTests(unittest.TestCase):
    def test_scope_preserves_old_results_and_checks_new_data_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = {"plant_id": "P", "input_status": "ok", "csv_sha256": "csv", "txt_sha256": "txt"}
            with (root / "input_manifest.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=inputs)
                writer.writeheader()
                writer.writerow(inputs)
            (root / "leaf_features.csv").write_text("plant_id,leaf_id,cluster_id\nP,7,C1\n",
                                                    encoding="utf-8")
            self.assertEqual(read_results(root)[2], "Batch clusters")
            within = root / "plant_clusters"
            within.mkdir()
            (within / "leaf_clusters.csv").write_text(
                "plant_id,leaf_id,plant_cluster_id\nP,7,C3\n", encoding="utf-8")
            provenance = within / "provenance.json"
            provenance.write_text(json.dumps({"inputs": {"P": inputs}}), encoding="utf-8")
            _, auto, scope = read_results(root)
            self.assertEqual((auto["P"][7]["cluster_id"], scope), ("C3", "Within-plant K=3"))
            self.assertEqual(read_results(root, "batch")[1]["P"][7]["cluster_id"], "C1")
            provenance.write_text(json.dumps({"inputs": {"P": {**inputs, "csv_sha256": "other"}}}),
                                  encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different point cloud"):
                read_results(root)

            newest = root / "plant_morphotypes"
            newest.mkdir()
            (newest / "leaf_clusters.csv").write_text(
                "plant_id,leaf_id,plant_cluster_id,morphotype\nP,7,C2,伸展型\n", encoding="utf-8")
            (newest / "provenance.json").write_text(
                json.dumps({"inputs": {"P": inputs}, "morphotypes": MORPHOTYPES}), encoding="utf-8")
            _, morph, scope = read_results(root)
            self.assertEqual((morph["P"][7]["morphotype"], scope),
                             ("伸展型", "Within-plant morphology K=3"))
            self.assertEqual(read_results(root, "batch")[1]["P"][7]["cluster_id"], "C1")


if __name__ == "__main__":
    unittest.main()
