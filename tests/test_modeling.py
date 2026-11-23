"""Numerical acceptance tests: withheld holes, reproducibility and label safety."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from scipy.spatial import cKDTree
from maize_leaf_v1.data import load_pair, discover
from maize_leaf_v1.model import Config, fit_leaf, LeafModel, triangle_areas, curve_fit


def synthetic():
    u,v=np.meshgrid(np.linspace(0,1,90),np.linspace(-1,1,25),indexing="ij")
    width=.16*np.sin(np.pi*(.10+.8*u))
    xyz=np.column_stack([u.ravel(),(v*width/2).ravel(),(.20*np.sin(np.pi*u)+.025*v*v).ravel()])
    hole=((u-.50)/.09)**2+(v/.48)**2<1
    missing_tip=u>.94
    missing_edge=(u>.27)&(u<.40)&(v>.6)
    missing=(hole|missing_tip|missing_edge).ravel()
    return xyz,missing,hole.ravel(),np.column_stack([np.zeros(80),np.zeros(80),np.linspace(-.2,.1,80)])


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.truth,cls.missing,cls.hole,stem=synthetic()
        cls.result=fit_leaf(cls.truth[~cls.missing],stem,Config(nu=100,nv=31),7)

    def test_continuous_model_serialization(self):
        r=self.result
        uv=np.array([[.2,.3],[.55,-.2],[1.02,.7]])
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"model.npz"
            r["model"].save(p)
            reloaded=LeafModel.load(p)
            np.testing.assert_allclose(r["model"].evaluate(uv),reloaded.evaluate(uv),rtol=1e-11,atol=1e-11)
            reloaded.save(Path(d)/"saved_again.npz")

    def test_surface_and_orthonormal_frames(self):
        r=self.result
        self.assertTrue(np.isfinite(r["vertices"]).all())
        self.assertTrue(np.all(triangle_areas(r["vertices"],r["faces"])>0))
        t,b,n=r["model"].frame(np.linspace(0,1.02,100))
        for x in (t,b,n):
            np.testing.assert_allclose(np.linalg.norm(x,axis=1),1,atol=1e-10)
        np.testing.assert_allclose(np.sum(t*b,axis=1),0,atol=1e-10)
        np.testing.assert_allclose(np.cross(t,b),n,atol=1e-10)

    def test_support_labels_and_area(self):
        r=self.result
        self.assertEqual(set(r["regions"]),{0,1,2})
        self.assertTrue(np.all(r["regions"][r["uv"][:,0]>1]==2))
        np.testing.assert_array_equal(r["face_regions"],r["regions"][r["faces"]].max(axis=1))
        self.assertAlmostEqual(sum(r["metrics"]["area_fractions"].values()),1,places=10)

    def test_withheld_hole_geometry(self):
        # Known synthetic ground truth only; does not establish real-leaf accuracy.
        r=self.result
        error=cKDTree(r["vertices"]).query(self.truth[self.hole])[0]
        self.assertLess(float(np.quantile(error,.95)),.035)
        # Interior withheld points should be predominantly assigned internal filling.
        idx=cKDTree(r["vertices"]).query(self.truth[self.hole])[1]
        self.assertGreater(float(np.mean(r["regions"][idx]==1)),.5)

    def test_sparse_failure_is_explicit(self):
        with self.assertRaisesRegex(ValueError,">=35"):
            fit_leaf(np.zeros((10,3)),np.zeros((20,3)))

    def test_original_leaf_id_preserved(self):
        self.assertEqual(self.result["metrics"]["leaf_id"],7)

    def test_curve_does_not_overshoot_near_duplicate_nodes(self):
        u=np.r_[0,.00001,.00002,np.linspace(.05,1,25)]
        xyz=np.column_stack([u,.1*np.sin(u*5),u*.2])
        xyz[1]=xyz[0]+[1e-7,-1e-7,1e-7]
        c=curve_fit(u,xyz)
        sampled=c(np.linspace(0,1,500))
        self.assertTrue(np.all(sampled>=xyz.min(axis=0)-np.ptp(xyz,axis=0)*.021))
        self.assertTrue(np.all(sampled<=xyz.max(axis=0)+np.ptp(xyz,axis=0)*.021))


class InputTests(unittest.TestCase):
    def test_label_header_permutation_and_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            t=root/"p.txt"; c=root/"p.csv"
            t.write_text("//X Y Z R G B Constant Constant_#2 Original_cloud_index\n3\n0 0 0 1 2 3 0 nan 1\n1 0 0 1 2 3 2 7 0\n2 0 0 1 2 3 2 nan 1\n")
            c.write_text("X,Y,Z,semantic\n0,0,0,0\n1,0,0,7\n2,0,0,2\n")
            plant=load_pair("p",t,c)
            self.assertEqual(plant.labels.tolist(),[0,7,2])
            c.write_text("X,Y,Z,semantic\n0,0,0,0\n1.1,0,0,7\n2,0,0,2\n")
            with self.assertRaisesRegex(ValueError,"coordinate/order mismatch"):
                load_pair("p",t,c)

    def test_missing_pair_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/"p.csv").write_text("X,Y,Z,semantic\n")
            with self.assertRaisesRegex(ValueError,"unmatched"):
                discover(d)

    def test_existing_output_preserved(self):
        from maize_leaf_v1.pipeline import fresh_output
        with tempfile.TemporaryDirectory() as d:
            marker=Path(d)/"old.txt"
            marker.write_text("keep")
            out=fresh_output(d)
            self.assertEqual(marker.read_text(),"keep")
            self.assertNotEqual(out,Path(d))

    def test_failed_stem_records_each_leaf(self):
        from unittest.mock import patch
        from maize_leaf_v1.data import Plant
        from maize_leaf_v1.pipeline import run
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            plant=Plant("p",np.zeros((40,3)),np.r_[np.zeros(20,dtype=int),np.full(20,9)],
                        np.zeros((40,3),dtype=np.uint8),{"plant":"p"})
            with patch("maize_leaf_v1.pipeline.discover",return_value=[("p",root/"p.txt",root/"p.csv")]),patch("maize_leaf_v1.pipeline.load_pair",return_value=plant),patch("maize_leaf_v1.pipeline.ensure_js"):
                code,out=run(root,root/"out",expect_plants=1)
            self.assertEqual(code,1)
            metric=json.loads((out/"plant_p/leaves/leaf_09_metrics.json").read_text())
            self.assertEqual(metric["status"],"failed")
            self.assertEqual(json.loads((out/"summary.json").read_text())["failed_leaves"],1)


if __name__=="__main__":
    unittest.main()
