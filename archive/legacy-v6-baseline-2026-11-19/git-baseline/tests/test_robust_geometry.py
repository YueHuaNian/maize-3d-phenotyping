"""Numerical/real-data safety checks, not biological validation."""
from pathlib import Path
from dataclasses import replace
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.io_utils import load_cloud, locate_data
from src.robust_geometry import ParametersV2,robust_stem


class RealDataV2Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.source = locate_data() / '1-1-2.csv'
        except FileNotFoundError as error:
            raise unittest.SkipTest('配套的真实点云未配置') from error
        if not cls.source.is_file():
            raise unittest.SkipTest('缺少测试样本 1-1-2.csv')
        cls.xyz,cls.labels = load_cloud(cls.source)
        cls.params = ParametersV2()

    def test_stem_records_every_source_point_and_all_variants(self):
        stem = robust_stem(self.xyz[self.labels == 0], self.params)
        self.assertEqual(len(stem['core_mask']),int(np.sum(self.labels==0)))
        self.assertEqual(len(stem['parameter_comparison']),15)
        self.assertTrue(np.all(stem['d1'] @ stem['axis'] > 0))
        again = robust_stem(self.xyz[self.labels==0],self.params)
        np.testing.assert_allclose(stem['curve'],again['curve'],rtol=0,atol=1e-12)






if __name__ == '__main__':
    unittest.main(verbosity=2)
