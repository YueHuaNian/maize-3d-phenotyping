import unittest

import numpy as np

from src.leaf import angle
from src.proximal_angle_experiment import local_directions


class ProximalExperimentTests(unittest.TestCase):
    def test_line_direction_remains_outward_when_anchor_translated(self):
        rng = np.random.default_rng(42)
        t = np.linspace(0, 1, 1500)
        true = np.array([0.6, 0.0, 0.8])
        anchor = np.array([2., -1., 3.])
        points = anchor + t[:, None] * true + rng.normal(0, .001, (len(t), 3))
        directions, counts, ratios, _ = local_directions(points, anchor)
        self.assertTrue(all(n >= 30 for n in counts))
        self.assertTrue(all(r > 1.5 for r in ratios))
        self.assertTrue(all(angle(direction, true) < 3 for direction in directions))

    def test_sparse_and_degenerate_points_do_not_produce_direction(self):
        directions, counts, _, _ = local_directions(np.zeros((100, 3)), np.zeros(3))
        self.assertEqual(directions, [])
        self.assertEqual(counts, [])
        directions, counts, _, _ = local_directions(np.array([[0., 0., 0.], [1., 0., 0.]]), np.zeros(3))
        self.assertEqual(counts, [1, 1, 1])
        self.assertTrue(all(direction is None for direction in directions))


if __name__ == "__main__":
    unittest.main()
