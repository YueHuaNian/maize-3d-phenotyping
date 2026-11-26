"""阶段 F 五因素联动理想株型符合度指数（I_A / I_L / I）与四项验证回归测试。"""
import math
import unittest

import numpy as np

from maize_leaf.ideotype import (
    DEFAULT_ALPHA,
    compute_leaf_ideotype_scores,
    compute_plant_ideotype_scores,
    ideal_envelope_radius,
    ideal_leaf_targets_at_height,
    normalize_far_height_diff,
    run_bootstrap_validation,
    run_monotonicity_validation,
    run_perturbation_validation,
    run_sensitivity_validation,
    score_plant_composition,
    score_plant_envelope,
    score_plant_light,
    score_single_leaf,
)


def _synthetic_dataset(n_plants=6):
    """构造可复现的合成多株数据集，用于验证四项方法学检验。"""
    plant_rows, env_rows, leaf_rows = [], [], []
    h_bins = np.linspace(0.02, 0.98, 25)
    r_ideal = ideal_envelope_radius(h_bins)
    r_bad = np.clip(0.15 + 0.85 * h_bins, 0.05, 1.0)
    r_bad /= np.max(r_bad)

    for p_idx in range(n_plants):
        pid = f"p{p_idx + 1}"
        lam = p_idx / max(1, n_plants - 1)
        r_cur = (1.0 - lam) * r_bad + lam * r_ideal
        for hb, rv in zip(h_bins, r_cur):
            env_rows.append({"plant_id": pid, "h_mid": float(hb), "radius_q95": float(rv)})

        non_ov = 0.55 + 0.42 * lam
        slant_ret = 0.50 + 0.45 * lam
        plant_rows.append({
            "plant_id": pid,
            "sum_projected_area_element": 1.0 / non_ov,
            "canopy_union_area": 1.0,
            "visible_area_nadir": 1.0,
            "visible_area_zen30_az0": slant_ret,
            "visible_area_zen30_az90": slant_ret,
            "visible_area_zen60_az0": slant_ret,
            "visible_area_zen60_az90": slant_ret,
        })

        classes = (
            ["convergent", "expanded", "expanded", "expanded", "drooping"]
            if lam >= 0.5 else
            ["expanded", "expanded", "drooping", "drooping", "drooping"]
        )
        for lid, (lh, cls_name) in enumerate(zip(np.linspace(0.9, 0.15, 5), classes), start=1):
            t = ideal_leaf_targets_at_height(lh)
            leaf_rows.append({
                "plant_id": pid,
                "leaf_id": lid,
                "cluster_kmeans": cls_name,
                "relative_height": float(lh),
                "stem_leaf_angle_deg": float(t["theta_target_deg"] + (1.0 - lam) * 35.0),
                "rel_projected_area": float(np.clip(t["proj_target"] - (1.0 - lam) * 0.25, 0.05, 1.0)),
                "pitch_entropy_b12": float(np.clip(t["entropy_target"] + (1.0 - lam) * 0.22, 0.05, 0.99)),
                "far_end_height_diff": float((t["far_dz_norm_target"] - (1.0 - lam) * 0.45) * 0.6),
                "observed_length": 0.6,
                "surface_area": 0.05 + 0.02 * lid,
                "quality_weight": 0.5 + 0.05 * lid,
                "support_review_required": (lid == 5 and p_idx % 2 == 0),
            })
    return plant_rows, env_rows, leaf_rows


class StageFIdeotypeIndexTests(unittest.TestCase):
    """覆盖尺度不变性、精确峰值、单调性及四项方法学检验的单元测试。"""

    def test_ideal_leaf_and_plant_achieve_exact_100(self):
        """检查严格匹配目标的合成单叶与整株得分均为 100。"""
        h = 0.72
        t = ideal_leaf_targets_at_height(h)
        rec = score_single_leaf({
            "plant_id": "p1", "leaf_id": 1, "cluster_kmeans": "expanded",
            "relative_height": h,
            "stem_leaf_angle_deg": t["theta_target_deg"],
            "rel_projected_area": t["proj_target"],
            "pitch_entropy_b12": t["entropy_target"],
            "far_end_height_diff": t["far_dz_norm_target"] * 0.8,
            "observed_length": 0.8,
        })
        self.assertAlmostEqual(rec["leaf_index_il"], 100.0, places=9)
        for k in ("s_angle", "s_proj", "s_entropy", "s_far_dz"):
            self.assertAlmostEqual(rec[k], 1.0, places=9)

        h_bins = np.linspace(0.02, 0.98, 25)
        r_ideal = ideal_envelope_radius(h_bins)
        env = score_plant_envelope([{"h_mid": float(hb), "radius_q95": float(rv)} for hb, rv in zip(h_bins, r_ideal)])
        self.assertAlmostEqual(env["s_envelope"], 1.0, places=9)

        light = score_plant_light({
            "sum_projected_area_element": 2.0, "canopy_union_area": 2.0, "visible_area_nadir": 2.0,
            "visible_area_zen30_az0": 2.0, "visible_area_zen30_az90": 2.0,
            "visible_area_zen60_az0": 2.0, "visible_area_zen60_az90": 2.0,
        })
        self.assertAlmostEqual(light["s_light"], 1.0, places=9)

        comp = score_plant_composition(["convergent", "expanded", "expanded", "expanded", "drooping"])
        self.assertAlmostEqual(comp["s_composition"], 1.0, places=9)

    def test_global_coordinate_scale_invariance(self):
        """检查点云整体缩放（如米换毫米）不改变 I_A、I_L 与 I。"""
        plant_rows, env_rows, leaf_rows = _synthetic_dataset(n_plants=4)
        base_l = compute_leaf_ideotype_scores(leaf_rows)
        base_p = compute_plant_ideotype_scores(plant_rows, env_rows, base_l)

        scale = 1000.0
        scaled_env = [{**r, "radius_q95": r["radius_q95"] * scale} for r in env_rows]
        scaled_plants = []
        for r in plant_rows:
            sr = dict(r)
            for k in ("sum_projected_area_element", "canopy_union_area", "visible_area_nadir",
                      "visible_area_zen30_az0", "visible_area_zen30_az90",
                      "visible_area_zen60_az0", "visible_area_zen60_az90"):
                sr[k] = r[k] * (scale ** 2)
            scaled_plants.append(sr)
        scaled_leaves = [{
            **r,
            "far_end_height_diff": r["far_end_height_diff"] * scale,
            "observed_length": r["observed_length"] * scale,
            "surface_area": r["surface_area"] * (scale ** 2),
        } for r in leaf_rows]

        sc_l = compute_leaf_ideotype_scores(scaled_leaves)
        sc_p = compute_plant_ideotype_scores(scaled_plants, scaled_env, sc_l)
        for a, b in zip(base_l, sc_l):
            self.assertAlmostEqual(a["leaf_index_il"], b["leaf_index_il"], places=9)
        for a, b in zip(base_p, sc_p):
            self.assertAlmostEqual(a["composite_index_i"], b["composite_index_i"], places=9)
            self.assertAlmostEqual(a["arch_index_ia"], b["arch_index_ia"], places=9)
            self.assertAlmostEqual(a["leaf_index_il"], b["leaf_index_il"], places=9)

    def test_far_height_diff_normalization_clamps_extremes(self):
        """检查无量纲远端高差裁剪到 [-1, 1] 并防零叶长除零。"""
        self.assertEqual(normalize_far_height_diff(5.0, 1.0), 1.0)
        self.assertEqual(normalize_far_height_diff(-5.0, 1.0), -1.0)
        self.assertEqual(normalize_far_height_diff(0.0, 0.0), 0.0)

    def test_monotonicity_validation_passes_all_subindices(self):
        """检查单调性检验在 I、I_A、I_L 及全部 7 个分项上严格通过。"""
        mono = run_monotonicity_validation(n_steps=15)
        self.assertTrue(mono["passed"])
        self.assertAlmostEqual(mono["endpoint_ideal_i"], 100.0, delta=0.1)
        for name, item in mono["checks"].items():
            self.assertTrue(item["passed"], f"Monotonicity failed for {name}")
            self.assertGreaterEqual(item["min_step_delta"], -1e-12)

    def test_perturbation_bootstrap_and_sensitivity_are_reproducible(self):
        """检查扰动、Bootstrap/LOLO 与敏感性检验可复现且输出有限有效指标。"""
        plant_rows, env_rows, leaf_rows = _synthetic_dataset(n_plants=6)
        p1 = run_perturbation_validation(plant_rows, env_rows, leaf_rows, n_trials=20, seed=123)
        p2 = run_perturbation_validation(plant_rows, env_rows, leaf_rows, n_trials=20, seed=123)
        self.assertEqual(p1["monte_carlo"]["spearman_i_median"], p2["monte_carlo"]["spearman_i_median"])
        self.assertGreater(p1["monte_carlo"]["spearman_i_median"], 0.8)
        self.assertIn("exclude_support_review_leaves", p1["structural_comparisons"])

        boot = run_bootstrap_validation(plant_rows, env_rows, leaf_rows, n_boot=40, seed=123)
        lo, hi = boot["plant_level_bootstrap"]["mean_i_ci95"]
        self.assertLessEqual(lo, hi)
        self.assertGreater(boot["leave_one_leaf_out"]["spearman_i_median"], 0.8)

        sens = run_sensitivity_validation(plant_rows, env_rows, leaf_rows, n_dirichlet=20, seed=123)
        primary_alpha = next(x for x in sens["alpha_sweep"] if math.isclose(x["alpha"], DEFAULT_ALPHA))
        self.assertAlmostEqual(primary_alpha["spearman_vs_primary"], 1.0, places=9)
        self.assertGreater(sens["dirichlet_random_weights"]["spearman_i_median"], 0.8)

    def test_score_plant_light_prefers_turtle_sky_46_when_available(self):
        """检查存在 46 向 Turtle Sky 列时优先采用旋转不变受光代理，缺省回退 5 方向。"""
        res_46 = score_plant_light({
            "turtle46_non_overlap": 0.86,
            "turtle46_slant_retention": 0.90,
            "sum_projected_area_element": 2.0,
            "canopy_union_area": 1.0,
            "visible_area_nadir": 1.0,
        })
        self.assertEqual(res_46["light_source"], "turtle_sky_46_soc")
        self.assertAlmostEqual(res_46["s_light"], 0.88, places=9)

        res_5 = score_plant_light({
            "sum_projected_area_element": 2.0,
            "canopy_union_area": 1.6,
            "visible_area_nadir": 1.6,
            "visible_area_zen30_az0": 1.28,
            "visible_area_zen30_az90": 1.28,
            "visible_area_zen60_az0": 1.28,
            "visible_area_zen60_az90": 1.28,
        })
        self.assertEqual(res_5["light_source"], "zbuffer_5dir")
        self.assertAlmostEqual(res_5["s_light"], 0.80, places=9)


if __name__ == "__main__":
    unittest.main()
