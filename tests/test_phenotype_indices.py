import unittest

from src.phenotype_indices import ENTROPY_MAX, evaluate_plant
from src.plant_clustering import FIELDS


class PhenotypeIndexTests(unittest.TestCase):
    def setUp(self):
        self.record = {"plant_id": "P", "input_status": "ok"}
        self.leaves = [{"plant_id": "P", "leaf_id": i, "status": "ok", "stem_z": i}
                       for i in (0, 1, 2)]
        self.stem = {"stem_z_q01": 0, "stem_z_q99": 2}
        self.profiles = {i: {"profile_sections": 5,
                             **dict(zip(FIELDS, (ENTROPY_MAX / 2, 1, 1, 0.5)))}
                         for i in (0, 1, 2)}

    def test_scores_match_hand_calculation_and_ignore_cluster_labels(self):
        for profile in self.profiles.values():
            profile.update(plant_cluster_status="TOO_FEW_DISTINCT_GEOMETRIES", plant_cluster_id=None)
        rows, summary = evaluate_plant(self.record, self.leaves, self.profiles, self.stem)
        self.assertEqual([row["lmi"] for row in rows], [50, 50, 50])
        self.assertAlmostEqual(summary["pai"], 50 * (0.5 + 0.8 / 1.8))
        self.assertEqual(summary["lmi_median"], 50)
        self.assertEqual(summary["lmi_coverage"], 1)

    def test_partial_lmi_can_leave_pai_valid(self):
        self.profiles[0].update(profile_sections=2, observed_shannon_entropy_bits=None)
        rows, summary = evaluate_plant(self.record, self.leaves, self.profiles, self.stem)
        self.assertIsNone(rows[0]["lmi"])
        self.assertEqual(rows[0]["lmi_status"], "INSUFFICIENT_PROFILE_SECTIONS")
        self.assertEqual(summary["pai_status"], "ok")
        self.assertEqual(summary["lmi_summary_status"], "partial")
        self.assertAlmostEqual(summary["lmi_coverage"], 2 / 3)

    def test_missing_attachment_leaves_pai_blank(self):
        self.leaves[0].update(status="NO_STEM", stem_z=None)
        self.profiles.pop(0)
        rows, summary = evaluate_plant(self.record, self.leaves, self.profiles, self.stem)
        self.assertEqual(rows[0]["height_status"], "NO_NEAREST_STEM_POINT")
        self.assertEqual(rows[0]["lmi_issue"], "NO_STEM")
        self.assertIsNone(summary["pai"])
        self.assertEqual(summary["height_valid_leaves"], 2)
        self.assertEqual(summary["pai_status"], "INCOMPLETE_HEIGHT_OR_REACH")

    def test_failed_input_and_small_plant(self):
        _, failed = evaluate_plant({"plant_id": "P", "input_status": "failed"}, [], {},
                                  {"stem_z_q01": None, "stem_z_q99": None})
        self.assertEqual(failed["pai_status"], "INPUT_FAILED")
        self.assertIsNone(failed["leaf_count"])
        rows, small = evaluate_plant(self.record, self.leaves[:1], self.profiles, self.stem)
        self.assertEqual(rows[0]["lmi"], 50)
        self.assertEqual(small["pai_status"], "TOO_FEW_LEAVES")
