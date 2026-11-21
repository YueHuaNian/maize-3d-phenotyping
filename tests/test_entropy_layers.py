import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.entropy_layers import run


class EntropyLayersIntegrationTests(unittest.TestCase):
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
