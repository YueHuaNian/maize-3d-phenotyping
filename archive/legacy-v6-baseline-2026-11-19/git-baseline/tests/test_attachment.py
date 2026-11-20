"""Check candidate provenance, source rows, and order flags."""
import unittest
import numpy as np
from src.attachment import attachment_diagnostics, contact_root
from src.robust_geometry import ParametersV2


class AttachmentDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        s = np.linspace(0,1,220)
        zeros = np.zeros_like(s)
        curve = np.column_stack([zeros,zeros,s])
        self.stem = {'curve':curve,'arc':s,'d1':np.tile([0.,0.,1.],(len(s),1))}
        self.surface = {'arc':s,'center':curve,'center_offset_xyz':np.zeros_like(curve),
                        'radius':np.full_like(s,.02),'supported':np.ones(len(s),bool)}
        u,v = np.meshgrid(np.linspace(.02,.30,45),np.linspace(-.006,.006,9))
        self.points = np.column_stack([u.ravel(),v.ravel(),(.5+.2*(u-.02)).ravel()])
        design = np.linspace(.028,.29,220)
        self.leaf = {'geometry':{'curve':np.column_stack([design,zeros,.5+.2*(design-.02)])},
                     'midline_quality':{'spacing':.004}}

    def test_observations_and_extrapolations_have_distinct_provenance(self):
        ids = np.arange(len(self.points))+120
        record = attachment_diagnostics(self.points,ids,self.leaf,self.stem,self.surface,ParametersV2())
        by_method = {item['method']:item for item in record['candidates']}
        closest = by_method['surface_nearest']
        self.assertEqual(record['closest_supported_surface_row_0based'],closest['observed_row_0based'])
        np.testing.assert_array_equal(closest['observed_xyz'],self.points[closest['observed_row_0based']-120])
        for method in ('local_tangent_backward','quadratic_midline_backward'):
            candidate = by_method[method]
            self.assertIsNone(candidate['observed_row_0based'])
            self.assertEqual(candidate['coordinate_source'],'bounded_extrapolation')
            self.assertLess(candidate['extrapolation_length'],.04)

    def test_surface_hit_requires_supported_ring(self):
        path = np.array([[.03,0,.5],[.01,0,.5]])
        self.assertIsNotNone(contact_root(path,np.array([0,.02]),self.stem,self.surface))
        self.surface['supported'][:] = False
        self.assertIsNone(contact_root(path,np.array([0,.02]),self.stem,self.surface))
