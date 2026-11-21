from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.angle_diagnostics import audit_leaf, run


class AngleDiagnosticsTests(unittest.TestCase):
    def test_early_failure_does_not_claim_uncomputed_stability_passed(self):
        row = {"plant_id": "P", "leaf_id": "1", "leaf_angle_deg_status": "ANCHOR_CURVE_GAP",
               "leaf_point_count": "300", "leaf_angle_deg": "", "nearest_distance": "0.1"}
        detail = {"pair_gap_gate": .05, "diagnostic_values": {"leaf_angle_deg": 42},
                  "variants": [{"length": 1, "graph_spacing": .01, "root_gap": .24,
                                "flags": ["ANCHOR_CURVE_GAP"]}]}
        result = audit_leaf(row, detail)
        self.assertEqual(result["root_gap_to_limit"], 2)
        self.assertTrue(result["pair_gap_exceeded"])
        self.assertEqual(result["accepted_angle_deg"], "")
        self.assertEqual(result["unvalidated_diagnostic_angle_deg"], 42)
        self.assertIsNone(result["variant_angle_max_deg"])

    def test_real_batch_and_no_overwrite(self):
        source = Path(__file__).resolve().parents[1] / "outputs/v1.1"
        with TemporaryDirectory() as tmp:
            destination = Path(tmp) / "audit"
            rows = run(source, destination)
            self.assertEqual(len(rows), 90)
            self.assertEqual(sum(r["angle_status"] == "ok" for r in rows), 33)
            self.assertEqual(sum("ANCHOR_CURVE_GAP" in r["angle_status"] for r in rows), 26)
            with self.assertRaises(FileExistsError):
                run(source, destination)


if __name__ == "__main__":
    unittest.main()
