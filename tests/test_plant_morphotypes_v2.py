import itertools
import unittest

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.plant_clustering import FIELDS
from src.plant_morphotypes_v2 import BOTTOM_TO_TOP, V2_FIELDS, WEIGHTS, _height_partition, cluster_plant


def inputs(heights):
    leaves = {i: {"status": "ok", "stem_z": z} for i, z in enumerate(heights, start=1)}
    features = {i: dict(zip(FIELDS, [5 - 0.3 * i, (i % 3) / 4, (i % 2) / 7, i / 12]))
                for i in leaves}
    return leaves, features


class HeightOrderedMorphotypeTests(unittest.TestCase):
    def test_every_leaf_is_assigned_and_layers_are_strictly_ordered(self):
        leaves, features = inputs([8, 4, 1, 4, 6, 2, 9])
        rows, summary = cluster_plant("P", leaves, features, 0, 10)
        self.assertEqual(summary["status"], "ok")
        self.assertEqual(summary["assigned_leaves"], len(leaves))
        self.assertTrue(summary["middle_majority_met"])
        self.assertGreater(summary["middle_share"], 0.5)
        self.assertEqual({r["plant_cluster_id"] for r in rows}, set(BOTTOM_TO_TOP))
        ranges = [[r["nearest_stem_z"] for r in rows if r["plant_cluster_id"] == code]
                  for code in BOTTOM_TO_TOP]
        self.assertLess(max(ranges[0]), min(ranges[1]))
        self.assertLess(max(ranges[1]), min(ranges[2]))
        self.assertEqual(len({r["plant_cluster_id"] for r in rows if r["nearest_stem_z"] == 4}), 1)
        self.assertEqual({r["plant_cluster_id"]: r["morphotype"] for r in rows},
                         {"C1": "聚拢型", "C2": "伸展型", "C3": "耷拉型"})

    def test_three_distinct_heights_are_required(self):
        leaves, features = inputs([1, 1, 3, 3])
        rows, summary = cluster_plant("P", leaves, features, 0, 10)
        self.assertEqual(summary["status"], "TOO_FEW_DISTINCT_ATTACHMENT_HEIGHTS")
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(r["plant_cluster_id"] is None for r in rows))

    def test_missing_entropy_keeps_an_unassigned_record(self):
        leaves, features = inputs([1, 2, 3, 4])
        features[2][FIELDS[0]] = ""
        rows, summary = cluster_plant("P", leaves, features, 0, 10)
        self.assertEqual(summary["status"], "MISSING_HEIGHT_OR_MORPHOLOGY")
        self.assertIsNone(rows[1][FIELDS[0]])

    def test_partition_minimizes_the_weighted_variance_without_splitting_ties(self):
        heights = [0.0, 0.1, 0.1, 0.24, 0.57, 0.68, 1.0]
        descriptions = [[1.2, 0.1, 0.8, 0.1], [0.5, 0.3, 0.2, 0.2], [0.8, 0.3, 0.4, 0.4],
                        [2.0, 0.2, 0.6, 0.0], [2.5, 0.2, 0.2, 0.9], [2.6, 0.1, 0.3, 0.4],
                        [1.8, 0.3, 0.5, 0.5]]
        matrix = np.column_stack((heights, descriptions))
        self.assertEqual(len(V2_FIELDS), matrix.shape[1])
        self.assertEqual(WEIGHTS[0], 0.65)
        self.assertAlmostEqual(WEIGHTS.sum(), 1)
        transformed = StandardScaler().fit_transform(matrix) * np.sqrt(WEIGHTS)
        scores = []
        for p, q in itertools.combinations(np.where(np.diff(heights) > 0)[0] + 1, 2):
            parts = np.split(transformed, (p, q))
            scores.append((sum(((group - group.mean(axis=0)) ** 2).sum() for group in parts), p, q))
        for policy, candidates in (("free", scores),
                                   ("majority", [s for s in scores if s[2] - s[1] > len(heights) / 2])):
            with self.subTest(policy=policy):
                objective, a, b = _height_partition(matrix, policy)
                expected = min(candidates)
                self.assertEqual((a, b), expected[1:])
                self.assertAlmostEqual(objective, expected[0])

    def test_middle_is_maximized_when_a_majority_is_infeasible(self):
        for heights, expected_middle in (([1, 2, 3, 4], 2), ([1, 1, 2, 3, 4, 4], 2)):
            with self.subTest(heights=heights):
                leaves, features = inputs(heights)
                _, summary = cluster_plant("P", leaves, features, 0, 10)
                self.assertEqual(summary["status"], "ok")
                self.assertEqual(summary["assigned_leaves"], len(heights))
                self.assertEqual(summary["C2"], expected_middle)
                self.assertFalse(summary["middle_majority_met"])


if __name__ == "__main__":
    unittest.main()
