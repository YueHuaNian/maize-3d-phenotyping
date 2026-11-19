"""Cover dynamic label scope, strict leaf order, and V5 comparison values."""
import unittest
import numpy as np
from src.pipeline import failure_codes, leaf_instance_ids, order_diagnostics
from src.report import same_number


class LeafInstanceTests(unittest.TestCase):
    def test_sparse_positive_ids_preserve_original_labels(self):
        self.assertEqual(leaf_instance_ids(np.array([10,2,-1,0,7,2,105,105])),[2,7,10,105])
        self.assertEqual(leaf_instance_ids(np.array([-1,0,0])),[])

    def test_order_uses_all_real_ids_and_distinguishes_ties_and_missing(self):
        rows = [{'leaf_id':leaf,'height_ratio_raw':height} for leaf,height in
                [(2,.7),(10,.7),(8,.9),(25,np.nan),(80,.2)]]
        pairs = {(r['upper_leaf_id'],r['lower_leaf_id']):r for r in order_diagnostics(rows,'test')}
        self.assertEqual(len(pairs),10)
        self.assertEqual(pairs[(2,8)]['height_order_severity'],'SEVERE')
        self.assertEqual(pairs[(2,10)]['height_order_severity'],'TIE')
        self.assertIs(pairs[(2,10)]['height_order_conflict'],True)
        self.assertIsNone(pairs[(10,25)]['height_order_conflict'])
        self.assertTrue(np.isnan(pairs[(10,25)]['height_order_inversion']))
        self.assertEqual([r['leaf_id'] for r in rows],[2,10,8,25,80])
        self.assertEqual(rows[1]['height_ratio_raw'],.7)

    def test_failure_reason_is_geometric_for_any_positive_id(self):
        feature = {'leaf_id':23,'point_count':360,'base_confidence':'LOW','midline_confidence':'HIGH',
                   'leaf_angle_deg':np.nan,'leaf_length':np.nan,'curvature_rms_norm':np.nan}
        issues = failure_codes({'feature':feature},{'selected':None})
        self.assertEqual(issues['v4_formal_angle'],['V4_BASE_GATE_NOT_MET'])
        self.assertEqual(issues['v4_formal_curvature'],['V4_NO_SUPPORTED_FULL_LENGTH'])

    def test_baseline_comparison_distinguishes_zero_from_missing(self):
        self.assertTrue(same_number('0.0',0.))
        self.assertFalse(same_number('nan',0.))
        self.assertTrue(same_number('',np.nan))
