"""验证四因素聚类、留一株隔离与不强制逐株类别配额。

Numerical regression tests for weighted leaf pose classification and LOPO."""
import unittest
from unittest.mock import patch
import numpy as np

from maize_leaf.classification import (
    ABLATION_MODELS,
    PRIMARY_FEATURES,
    KMeansModel,
    RobustScaler,
    adjusted_rand_index,
    lopo_validate,
    normalized_mutual_info,
    silhouette_score,
    fit_and_anchor,
    contextual_predict,
    DEFAULT_POSE_POLICY,
)


class StageEClassificationTests(unittest.TestCase):
    """覆盖聚类数学性质、特征权重及逐株语义约束的测试集。"""
    def _make_well_separated_plants(self, seed=7):
        """生成按植株分组且类别分离的合成四因素样本。"""
        rng = np.random.default_rng(seed)
        # 10 synthetic plants, each with 6 leaves spanning 3 clean S1 pose clusters:
        # [stem_leaf_angle_deg, projection_ratio, pitch_entropy_b12]
        #   0=convergent (small angle ~20, low rho 0.40, low H_pit 0.50)
        #   1=expanded   (mid angle ~45, high rho 0.65, high H_pit 0.78)
        #   2=drooping   (large angle ~72, mid rho 0.58, high H_pit 0.75)
        prototypes = np.array([
            [20.0, 0.40, 0.50],
            [45.0, 0.65, 0.78],
            [72.0, 0.58, 0.75],
        ])
        scales = np.array([3.5, 0.03, 0.04])
        X, y_true, plants = [], [], []
        for p in range(10):
            for c in (0, 0, 1, 1, 2, 2):
                X.append(prototypes[c] + rng.normal(scale=scales))
                y_true.append(c)
                plants.append(f"p{p:02d}")
        return np.asarray(X), np.asarray(y_true), np.asarray(plants)

    def test_ari_nmi_and_silhouette_identities(self):
        """检查相同标签和分离样本的 ARI、NMI、轮廓系数恒等性质。"""
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y_perm = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])
        self.assertAlmostEqual(adjusted_rand_index(y, y_perm), 1.0, places=10)
        self.assertAlmostEqual(normalized_mutual_info(y, y_perm), 1.0, places=10)
        X = np.vstack([np.zeros((3, 2)), np.full((3, 2), 5.0), np.full((3, 2), 10.0)])
        self.assertAlmostEqual(silhouette_score(X, y), 1.0, places=10)

    def test_kmeans_recovers_semantic_order_and_lopo_stability(self):
        """检查可分离合成样本的语义顺序和留一株稳定性。"""
        X, y_true, plants = self._make_well_separated_plants()
        feats = ABLATION_MODELS["M3_morphology"]
        out = lopo_validate(X, plants, feats, seed=42, n_perm=50)
        self.assertGreater(adjusted_rand_index(y_true, out["full_labels"]), 0.95)
        self.assertEqual(int(np.sum(out["full_labels"] != y_true)), 0)
        self.assertGreater(out["metrics"]["lopo_oof_ari"], 0.95)
        self.assertLess(out["metrics"]["h2_permutation_p_value"], 0.05)

    def test_robust_scaler_is_resistant_to_extreme_outlier(self):
        """检查极端离群值对中位数与四分位距缩放的影响受限。"""
        X = np.column_stack([np.linspace(0, 10, 41), np.linspace(-1, 1, 41)])
        X_out = np.vstack([X, [[1e5, -1e5]]])
        s_clean = RobustScaler().fit(X)
        s_out = RobustScaler().fit(X_out)
        self.assertLess(abs(s_clean.median_[0] - s_out.median_[0]), 0.3)
        self.assertLess(abs(s_clean.scale_[0] - s_out.scale_[0]), 0.3)

    def test_zero_height_weight_reproduces_three_feature_partition(self):
        """检查高度和远端高差权重为零时复现三形态特征的划分。"""
        X, _, plants = self._make_well_separated_plants()
        h = np.linspace(0, 1, len(X))
        far_dz = np.linspace(0.5, -0.3, len(X))
        baseline = lopo_validate(X, plants, PRIMARY_FEATURES[:3], n_perm=10)
        augmented = lopo_validate(np.column_stack([X, h, far_dz]), plants, PRIMARY_FEATURES,
                                  feature_weights=[1, 1, 1, 0, 0], n_perm=10)
        np.testing.assert_array_equal(baseline["full_labels"], augmented["full_labels"])
        np.testing.assert_array_equal(baseline["oof_labels"], augmented["oof_labels"])

    def test_height_changes_distance_and_scaler_round_trips(self):
        """检查高度参与距离及尺度变换可逆。"""
        X = np.array([[0., 0.], [1., .5], [2., 1.]])
        scaler = RobustScaler([1., .25]).fit(X)
        Z = scaler.transform(X)
        np.testing.assert_allclose(scaler.inverse_transform(Z), X)
        # One IQR of height contributes precisely 0.25 to squared distance.
        delta = scaler.transform([[1., 1.]]) - scaler.transform([[1., .5]])
        self.assertAlmostEqual(float(np.sum(delta ** 2)), .25)
        with self.assertRaises(ValueError):
            RobustScaler([1., -1.]).fit(X)
        with self.assertRaises(ValueError):
            RobustScaler([1.]).fit(X)

    def test_lopo_fits_scaler_without_held_out_plant(self):
        """检查留一株验证的缩放器仅在训练植株拟合。"""
        X, _, plants = self._make_well_separated_plants()
        X = np.column_stack([X, np.linspace(0, 1, len(X)), np.linspace(0.4, -0.2, len(X))])
        fitted = []
        original = RobustScaler.fit

        def record_fit(scaler, data):
            """记录缩放器拟合输入，检查留出植株没有进入训练。"""
            fitted.append(np.asarray(data).copy())
            return original(scaler, data)

        with patch.object(RobustScaler, "fit", record_fit):
            lopo_validate(X, plants, PRIMARY_FEATURES, feature_weights=[1, 1, 1, 1, 1], n_perm=5)
        self.assertEqual(len(fitted), 11)
        for i, plant in enumerate(sorted(set(plants))):
            np.testing.assert_array_equal(fitted[i + 1], X[plants != plant])

    def test_height_is_available_as_an_actual_clustering_feature(self):
        """检查相对高度作为实际聚类特征影响划分。"""
        heights = np.repeat([.1, .5, .9], 8)
        X = np.column_stack([np.zeros((24, 3)), heights, np.zeros(24)])
        _, _, _, labels = fit_and_anchor(X, PRIMARY_FEATURES, feature_weights=[1, 1, 1, 1, 1])
        self.assertAlmostEqual(adjusted_rand_index(np.repeat([0, 1, 2], 8), labels), 1.)

    def test_far_height_diff_separates_upward_and_drooping_clusters(self):
        """检查离茎最远端高度差可作为实际聚类特征区分上下。"""
        far_dz = np.repeat([.5, .1, -.3], 8)
        X = np.column_stack([np.zeros((24, 4)), far_dz])
        _, _, _, labels = fit_and_anchor(X, PRIMARY_FEATURES, feature_weights=[1, 1, 1, 1, 1])
        self.assertAlmostEqual(adjusted_rand_index(np.repeat([0, 1, 2], 8), labels), 1.)

    def test_compact_requires_small_apical_candidate_without_per_plant_quotas(self):
        """检查紧缩限于顶部小叶，且各株可缺少该类别。"""
        model = KMeansModel()
        model.centers_ = np.array([[0., 0., 0., 1., 1.], [2., 1., 1., .5, .5], [5., .4, 1., 0., -1.]])
        Z = np.vstack([model.centers_[0], model.centers_[0], model.centers_[0], model.centers_[1]])
        X = np.array([[20., .25, .4, .9, .5], [20., .25, .4, .3, .5],
                      [20., .9, .4, .9, .5], [40., 1., .7, .4, .2]])
        labels, raw, _ = contextual_predict(model, Z, X, ['p1', 'p1', 'p2', 'p2'],
                                            PRIMARY_FEATURES, DEFAULT_POSE_POLICY)
        np.testing.assert_array_equal(labels, [0, 1, 1, 1])
        self.assertEqual(len(set(labels[2:])), 1)  # p2 legitimately has only expanded leaves.
        self.assertEqual(raw[2], 0)  # An upright but large mature leaf is not a compact young leaf.

    def test_five_factor_dimless_feature_is_scale_invariant(self):
        """检查无量纲化第 5 特征 far_dz_norm 满足严格坐标缩放不变性。"""
        self.assertEqual(PRIMARY_FEATURES[-1], "far_dz_norm")
        X, _, plants = self._make_well_separated_plants()
        h = np.linspace(0.05, 0.95, len(X))
        obs_len = np.linspace(0.3, 0.9, len(X))
        raw_dz = np.linspace(0.25, -0.35, len(X))
        far_dz_norm_1x = np.clip(raw_dz / obs_len, -1.0, 1.0)
        far_dz_norm_1000x = np.clip((raw_dz * 1000.0) / (obs_len * 1000.0), -1.0, 1.0)
        X1 = np.column_stack([X, h, far_dz_norm_1x])
        X2 = np.column_stack([X, h, far_dz_norm_1000x])
        sc, mdl, Z1, _ = fit_and_anchor(X1, PRIMARY_FEATURES, seed=42)
        l1, _, _ = contextual_predict(mdl, Z1, X1, plants, PRIMARY_FEATURES, DEFAULT_POSE_POLICY)
        l2, _, _ = contextual_predict(mdl, sc.transform(X2), X2, plants, PRIMARY_FEATURES, DEFAULT_POSE_POLICY)
        np.testing.assert_array_equal(l1, l2)


if __name__ == "__main__":
    unittest.main()
