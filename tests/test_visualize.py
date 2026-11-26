"""验证查看器载荷、离线模板与接口输入校验。

Regression checks for one-leaf scenes, offline parity and viewer HTTP errors."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("maize_viewer_script", ROOT / "scripts/visualize.py")
viewer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viewer)


class ViewerTests(unittest.TestCase):
    """覆盖离线页面、分类输入与 HTTP 参数校验的回归集。"""
    def scene(self):
        """构造查看器测试使用的最小场景载荷。"""
        xyz = np.array([[0., 0., 0.], [1., 0., .5], [0., 1., 1.]])
        return {"plant": "p1", "leaf": None, "title": "p1", "stem": xyz,
                "leaves": [{"leaf_id": 2, "points": xyz, "rgb": None, "vertices": xyz,
                            "faces": np.array([[0, 1, 2]]), "face_regions": np.array([0]),
                            "midrib": xyz, "traits": {"surface_area": 1.}}]}

    def test_single_leaf_summary_is_uniform_and_offline_colors_are_available(self):
        """检查单叶汇总格式一致且离线类别配色完整。"""
        rec = {"cluster_kmeans": "convergent", "relative_height": .8}
        payload = viewer.scene_to_plotly_payload(self.scene(), {("p1", 2): rec}, "kmeans")
        self.assertEqual(payload["info"]["leaf_count"], 1)
        self.assertEqual(payload["info"]["leaves"][0]["leaf_id"], 2)
        labels = next(t for t in payload["traces"] if t["group"] == "labels")
        self.assertEqual(labels["leaf_ids"], [2])
        surface = next(t for t in payload["traces"] if t["group"] == "surface")
        self.assertEqual(len(surface["region_facecolors"]), 1)
        json.dumps(payload, allow_nan=False)

    def test_offline_template_escapes_script_end_and_has_no_unfilled_tokens(self):
        """检查离线 JSON 防止提前结束脚本且模板占位符已填充。"""
        html = viewer.viewer_page({"label": "</script>"}, offline=True)
        self.assertIn('<script src="assets/plotly.min.js">', html)
        self.assertIn('<\\/script>', html)
        self.assertNotIn("__BOOTSTRAP_JSON__", html)
        self.assertNotIn("__PLOTLY_SRC__", html)

    def test_invalid_scene_query_returns_json_400(self):
        """检查非法场景查询返回 JSON 格式的 400 响应。"""
        handler = viewer.make_handler(ROOT, {"p1": [2]}, "p1", None, {}, "region")
        # Exercise request dispatch without depending on host loopback permissions.
        for query in ("plant=missing", "plant=p1&leaf=oops", "plant=p1&color_by=bad"):
            instance = handler.__new__(handler)
            instance.path = f"/api/scene?{query}"
            responses = []
            instance._send = lambda *args: responses.append(args)
            instance.do_GET()
            self.assertEqual(responses[0][0], 400)
            self.assertIn("error", json.loads(responses[0][2]))

    def test_duplicate_classification_rows_are_rejected(self):
        """检查分类表的重复植株叶片键被拒绝。"""
        temp_root = ROOT / ".cache/test_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=temp_root) as temp:
            path = Path(temp) / "classes.csv"
            path.write_text("plant_id,leaf_id\np1,2\np1,2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                viewer.load_classifications(path)

    def test_ideotype_indices_attach_to_payload_and_hover(self):
        """检查整株与单叶指数安全注入查看器载荷且缺失时不报错。"""
        rec = {"cluster_kmeans": "convergent", "relative_height": .8}
        pidx = {"p1": {"plant_id": "p1", "rank": 1, "composite_index_i": 80.4, "arch_index_ia": 74.8,
                       "leaf_index_il": 86.0, "s_envelope": 0.47, "s_light": 0.78, "s_composition": 1.0,
                       "mean_s_angle": 0.98, "mean_s_proj": 0.86, "mean_s_entropy": 0.71, "mean_s_far_dz": 0.90}}
        lidx = {("p1", 2): {"plant_id": "p1", "leaf_id": 2, "leaf_index_il": 88.5,
                            "s_angle": 0.95, "s_proj": 0.85, "s_entropy": 0.82, "s_far_dz": 0.92}}
        payload = viewer.scene_to_plotly_payload(
            self.scene(), {("p1", 2): rec}, "kmeans", plant_indices=pidx, leaf_indices=lidx
        )
        self.assertEqual(payload["info"]["plant_index"]["rank"], 1)
        self.assertAlmostEqual(payload["info"]["leaves"][0]["ideotype"]["leaf_index_il"], 88.5)
        surface = next(t for t in payload["traces"] if t["group"] == "surface")
        self.assertIn("I_L=88.5", surface["hovertext"])
        json.dumps(payload, allow_nan=False)
        self.assertEqual(viewer.load_ideotype_indices(ROOT / ".cache/nonexistent_index_dir"), ({}, {}))


if __name__ == "__main__":
    unittest.main()
