"""Analytic fixtures test formulas only; never replace the real sample analysis."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.geometry import Parameters, spline_fit, tangent_angle, project_to_curve
from src.io_utils import locate_data, audit_inputs


class FormulaChecks(unittest.TestCase):
    def test_directed_angle_and_straight_curvature(self):
        s = np.linspace(0,1,20)
        stem = spline_fit(np.column_stack([s*0,s*0,s]),.006)
        for degrees in (30,120):
            theta = np.radians(degrees)
            curve = np.column_stack([s*np.sin(theta),s*0,s*np.cos(theta)])
            fit = spline_fit(curve,.006)
            measurement = tangent_angle(fit,stem,.2,.12)
            self.assertAlmostEqual(measurement['angle_deg'],degrees,places=5)
            self.assertLess(float(np.max(fit['curvature'])),1e-8)
            length = fit['arc'][-1]
            droop = np.maximum(-fit['curve'][:,2],0).max()/length
            self.assertAlmostEqual(droop,max(0,-np.cos(theta)),places=5)

    def test_rigid_rotation_preserves_angle(self):
        s = np.linspace(0,1,20)
        stem_points = np.column_stack([0*s,0*s,s])
        leaf_points = np.column_stack([s*.8,0*s,s*.6])
        angle = .3
        rotation = np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        before = tangent_angle(spline_fit(leaf_points,.006),spline_fit(stem_points,.006),.2,.12)['angle_deg']
        after = tangent_angle(spline_fit(leaf_points @ rotation.T,.006),spline_fit(stem_points @ rotation.T,.006),.2,.12)['angle_deg']
        self.assertAlmostEqual(before,after,places=6)

    def test_projection_uses_stem_arclength(self):
        curve = np.array([[0.,0.,0.],[0.,0.,1.],[1.,0.,1.]])
        distance,at,foot = project_to_curve(np.array([[.5,.2,1.]]),curve)
        self.assertAlmostEqual(distance[0],.2)
        self.assertAlmostEqual(at[0],1.5)
        np.testing.assert_allclose(foot,[[.5,0,1]])


class RealSampleChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            folder = locate_data()
        except FileNotFoundError as error:
            raise unittest.SkipTest('配套的真实点云未配置') from error
        if not (folder / '1-1-2.csv').is_file():
            raise unittest.SkipTest('缺少测试样本 1-1-2.csv')
        txt = folder.parent / '0709单天多个材料原始数据' / '1-1-2.txt'
        if not txt.is_file():
            raise unittest.SkipTest('缺少配套测试 TXT')
        cls.xyz, cls.labels, cls.audit = audit_inputs(folder / '1-1-2.csv', txt)

    def test_actual_csv_txt_correspondence(self):
        self.assertEqual(self.audit['csv_rows'],13340)
        self.assertTrue(self.audit['xyz_rowwise_exact'])
        self.assertTrue(self.audit['merged_labels_rowwise_exact'])





if __name__ == '__main__':
    unittest.main(verbosity=2)
