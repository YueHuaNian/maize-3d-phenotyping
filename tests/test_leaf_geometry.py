"""Analytic checks for the automatic surface and feature gates."""
import unittest
import numpy as np
from src.robust_geometry import ParametersV2
from src.leaf import extract_leaf
from src.stem import fit_stem_surface, surface_query


class V4AutomaticGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        z,phi = np.meshgrid(np.linspace(0,1,80),np.linspace(0,2*np.pi,18,endpoint=False))
        cls.stem_points = np.column_stack([.021*np.cos(phi.ravel()),.021*np.sin(phi.ravel()),z.ravel()])
        cls.params = ParametersV2()
        cls.stem,cls.surface,cls.quality = fit_stem_surface(cls.stem_points,cls.params)
        x,w = np.meshgrid(np.linspace(.021,.48,95),np.linspace(-.02,.02,11))
        v = w*(.12+.88*np.sin(np.pi*(x-.021)/.459))
        cls.leaf = np.column_stack([x.ravel(),v.ravel(),(.48+.3*(x-.021)).ravel()])

    def test_radius_and_contact_without_annotation(self):
        self.assertAlmostEqual(float(np.median(self.surface['radius'])),.021,places=3)
        query = surface_query(np.array([[.021,0,.48]]),self.stem,self.surface)
        self.assertTrue(query['supported'][0])
        self.assertLess(query['distance'][0],.003)
        result = extract_leaf(self.leaf,np.arange(len(self.leaf)),1,self.stem,self.surface,self.params)
        row = result['feature']
        self.assertAlmostEqual(row['leaf_angle_deg'],np.degrees(np.arctan2(1,.3)),delta=3)
        self.assertGreater(result['base_candidate_selected']['support_count'],0)
        self.assertAlmostEqual(row['leaf_length'],np.hypot(.459,.459*.3),delta=.025)
        self.assertNotEqual(row['leaf_length'],row['visible_midline_length'])


    def test_sparse_leaf_cannot_pass_formal_gates(self):
        leaf = self.leaf[np.linspace(0,len(self.leaf)-1,148,dtype=int)]
        result = extract_leaf(leaf,np.arange(len(leaf)),5,self.stem,self.surface,self.params)
        for field in ('height_ratio','leaf_angle_deg','leaf_length','curvature_rms_norm','droop_index'):
            self.assertTrue(np.isnan(result['feature'][field]),field)
        self.assertTrue(any('148 < 200' in warning for warning in result['warnings']))
