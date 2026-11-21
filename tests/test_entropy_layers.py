import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.entropy_layers import compare_layers, run


class EntropyLayersIntegrationTests(unittest.TestCase):
    def test_comparisons_report_missing_ties_and_partial_coverage(self):
        upper = {"leaf_count": 2, "leaf_angle_deg_median": 20, "leaf_angle_deg_valid_count": 1,
                 "horizontal_reach_q95_ratio_median": .2, "horizontal_reach_q95_ratio_valid_count": 2}
        middle = {"leaf_count": 2, "leaf_angle_deg_median": 40, "leaf_angle_deg_valid_count": 2,
                  "horizontal_reach_q95_ratio_median": .2, "horizontal_reach_q95_ratio_valid_count": 2,
                  "downward_path_fraction_median": .3, "downward_path_fraction_valid_count": 2}
        lower = {"leaf_count": 1, "downward_path_fraction_median": .1,
                 "downward_path_fraction_valid_count": 1}
        comparisons = {r["comparison"]: r for r in compare_layers("P", {"C1": upper, "C2": middle, "C3": lower})}
        self.assertIs(comparisons["upper_angle_vs_middle"]["supports_expected_order"], True)
        self.assertEqual(comparisons["upper_angle_vs_middle"]["coverage"], "partial")
        self.assertEqual(comparisons["middle_reach_vs_upper"]["relation"], "equal")
        self.assertIsNone(comparisons["middle_reach_vs_upper"]["supports_expected_order"])
        self.assertIs(comparisons["lower_downward_vs_middle"]["supports_expected_order"], False)
        self.assertEqual(comparisons["lower_drop_vs_middle"]["relation"], "insufficient_data")

    def test_published_batch_keeps_features_and_reproduces_free_partition(self):
        root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "layers"
            leaves, plants, groups = run(root / "outputs/v1.1", output)
            with (root / "outputs/v2/leaf_layers.csv").open(encoding="utf-8-sig", newline="") as f:
                expected = {(r["plant_id"], int(r["leaf_id"])): r["plant_cluster_id"]
                            for r in csv.DictReader(f)}
            self.assertEqual({(r["plant_id"], r["leaf_id"]): r["plant_cluster_id"]
                              for r in leaves}, expected)
            self.assertEqual(len(plants), 14)
            self.assertEqual(len(groups), 42)
            self.assertTrue(all("morphotype" not in row for row in leaves))
            self.assertTrue(any(row["leaf_angle_deg"] == "" for row in leaves))
            with self.assertRaises(FileExistsError):
                run(root / "outputs/v1.1", output)


if __name__ == "__main__":
    unittest.main()
