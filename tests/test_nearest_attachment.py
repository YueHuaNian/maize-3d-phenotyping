import unittest

import numpy as np

from src.nearest_attachment import InputError, load_pair, locate_attachments


class NearestAttachmentTests(unittest.TestCase):
    def locate(self, xyz, labels):
        return locate_attachments("test-plant", np.asarray(xyz, dtype=float).reshape(-1, 3),
                                  np.asarray(labels, dtype=object))

    def test_one_leaf_point_multiple_stem_points(self):
        row, = self.locate([(0, 0, 0), (4, 0, 0), (3, 1, 0)], [0, 0, 1])
        self.assertEqual((row.leaf_point_row, row.stem_point_row), (2, 1))
        self.assertAlmostEqual(row.nearest_distance, 2 ** 0.5)
        self.assertEqual((row.leaf_x, row.leaf_y, row.leaf_z), (3, 1, 0))
        self.assertEqual((row.stem_x, row.stem_y, row.stem_z), (4, 0, 0))

    def test_global_minimum_over_entire_leaf(self):
        row, = self.locate([(0, 0, 0), (9, 1, 0), (1, 2, 0), (0, 1, 0), (10, 1, 0)],
                           [0, 2, 2, 2, 0])
        self.assertEqual((row.leaf_point_row, row.stem_point_row, row.nearest_distance), (1, 4, 1))

    def test_zero_distance(self):
        row, = self.locate([(1, 2, 3), (100, 0, 0), (1, 2, 3)], [0, 0, 1])
        self.assertEqual((row.status, row.nearest_distance, row.stem_point_row), ("ok", 0.0, 0))

    def test_empty_leaf_cloud(self):
        row, = self.locate([(0, 0, 0), (2, 0, 0)], [0, -1])
        self.assertIsNone(row.leaf_id)
        self.assertEqual((row.leaf_point_count, row.status), (0, "no_leaf_points"))

    def test_empty_stem_cloud_preserves_leaves(self):
        rows = self.locate([(0, 2, 0), (2, 0, 0), (3, 0, 0)], [1, -1, 12])
        self.assertEqual([r.leaf_id for r in rows], [1, 12])
        for row in rows:
            self.assertEqual((row.stem_point_count, row.status), (0, "no_valid_stem_points"))
            self.assertIsNone(row.nearest_distance)

    def test_ties_prefer_earliest_leaf_then_earliest_stem(self):
        cloud = [(-1, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 0), (0, 0, 0),
                 (0, -1, 0), (-1, 0, 0)]
        labels = [0, 0, 0, 1, 1, 0, 0]
        for _ in range(3):
            row, = self.locate(cloud, labels)
            self.assertEqual((row.leaf_point_row, row.stem_point_row, row.nearest_distance),
                             (3, 0, 1.0))

    def test_original_rows_include_background_and_invalid_rows(self):
        csv = b"X,Y,Z,semantic\n1,2,3,-1\n0,0,0,0\nNaN,0,0,9\n0,0,1,9\n"
        txt = (b"//X Y Z Constant Constant_#2\n4\n1 2 3 -1 nan\n0 0 0 0 nan\n"
               b"nan 0 0 1 9\n0 0 1 1 9\n")
        points, labels = load_pair(csv, txt)
        row, = locate_attachments("paired-plant", points, labels)
        self.assertEqual((row.leaf_point_row, row.stem_point_row), (3, 1))
        self.assertEqual((row.leaf_point_count, row.leaf_raw_point_count), (1, 2))

    def test_all_positive_ids_are_preserved(self):
        huge = 10 ** 22 + 7
        xyz = [(0, 0, 0)] * 7
        labels = [0, -1, huge, 8, 9, 7, 1000]
        rows = self.locate(xyz, labels)
        self.assertEqual([r.leaf_id for r in rows], [7, 8, 9, 1000, huge])

    def test_invalid_leaf_coordinates_retain_leaf_id(self):
        row, = self.locate([(0, 0, 0), (float("nan"), 0, 0)], [0, 7])
        self.assertEqual((row.leaf_id, row.leaf_point_count, row.leaf_raw_point_count), (7, 0, 1))
        self.assertEqual(row.status, "no_valid_leaf_points")

    def test_row_and_coordinate_mismatches_are_rejected(self):
        csv = b"X,Y,Z,semantic\n1,2,3,8\n"
        for txt, message in (
            (b"//X Y Z Constant\n0\n", "point counts differ"),
            (b"//X Y Z Constant\n1\n1 2 4 8\n", "XYZ differ at point row 0"),
            (b"//X Y Z Constant\n1\n1 2 3 7\n", "semantic labels differ at point row 0"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(InputError, message):
                load_pair(csv, txt)


if __name__ == "__main__":
    unittest.main()
