"""Benchmark geometry checks independent of the reconstruction implementation."""
import unittest
import numpy as np
from maize_leaf.validation import point_mesh_distances, occlusion_masks, holdout_alerts


class HoldoutGeometryTests(unittest.TestCase):
    def test_triangle_distance_uses_face_interior_and_edges(self):
        vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        queries = np.array([[.25, .25, 1.], [.5, -1., 0.], [2., 2., 0.], [.25, .25, 0.]])
        actual = point_mesh_distances(queries, vertices, [[0, 1, 2]], chunk_size=2)
        np.testing.assert_allclose(actual, [1., 1., np.sqrt(4.5), 0.], atol=1e-12)

    def test_triangle_distance_handles_degenerate_face(self):
        actual = point_mesh_distances([[.5, 1., 0.]], [[0., 0., 0.], [1., 0., 0.], [1., 0., 0.]], [[0, 1, 2]])
        np.testing.assert_allclose(actual, [1.], atol=1e-12)

    def test_tip_masks_are_nested_and_leave_root_visible(self):
        masks = occlusion_masks(np.linspace(0, 1, 101), np.zeros(101))
        self.assertTrue(np.all(~masks["tip_10pct"] | masks["tip_20pct"]))
        self.assertTrue(np.all(~masks["tip_20pct"] | masks["tip_30pct"]))
        self.assertFalse(any(m[0] for m in masks.values()))

    def test_alerts_use_frozen_reference_width(self):
        result = holdout_alerts(dict(held_p95=.02, spacing=.001, ref_max_width=.1,
                                    length_change_pct=16., area_change_pct=26.))
        self.assertEqual(result, ["held_p95", "length_change", "area_change"])


if __name__ == "__main__":
    unittest.main()
