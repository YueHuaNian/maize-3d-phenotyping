"""Unit tests for Stage E pure-morphology leaf pose classification & LOPO metrics."""
import unittest
import numpy as np

from maize_leaf.classification import (
    ABLATION_MODELS,
    DiagonalGMMModel,
    FuzzyCMeansModel,
    KMeansModel,
    RobustScaler,
    adjusted_rand_index,
    fit_and_anchor,
    lopo_validate,
    normalized_mutual_info,
    silhouette_score,
)


class StageEClassificationTests(unittest.TestCase):
    def _make_well_separated_plants(self, seed=7):
        rng = np.random.default_rng(seed)
        # 10 synthetic plants, each with 6 leaves spanning 3 clean pose clusters:
        #   0=convergent (small angle ~20, high tip_gain +0.75, low droop 0.02, low K 1.2, low rho 0.35)
        #   1=expanded   (mid angle ~50, mid tip_gain +0.20, mid droop 0.12, mid K 2.5, mid rho 0.62)
        #   2=drooping   (large angle ~80, neg tip_gain -0.30, high droop 0.38, high K 4.2, mid-high rho 0.68)
        prototypes = np.array([
            [20.0, +0.75, 0.02, 1.2, 0.35],
            [50.0, +0.20, 0.12, 2.5, 0.62],
            [80.0, -0.30, 0.38, 4.2, 0.68],
        ])
        scales = np.array([4.0, 0.06, 0.02, 0.25, 0.04])
        X, y_true, plants = [], [], []
        for p in range(10):
            for c in (0, 0, 1, 1, 2, 2):
                X.append(prototypes[c] + rng.normal(scale=scales))
                y_true.append(c)
                plants.append(f"p{p:02d}")
        return np.asarray(X), np.asarray(y_true), np.asarray(plants)

    def test_ari_nmi_and_silhouette_identities(self):
        y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        y_perm = np.array([2, 2, 2, 0, 0, 0, 1, 1, 1])  # label permutation = identical partition
        self.assertAlmostEqual(adjusted_rand_index(y, y_perm), 1.0, places=10)
        self.assertAlmostEqual(normalized_mutual_info(y, y_perm), 1.0, places=10)
        X = np.vstack([np.zeros((3, 2)), np.full((3, 2), 5.0), np.full((3, 2), 10.0)])
        self.assertAlmostEqual(silhouette_score(X, y), 1.0, places=10)

    def test_kmeans_fcm_gmm_recover_semantic_order_and_lopo_stability(self):
        X, y_true, plants = self._make_well_separated_plants()
        feats = ABLATION_MODELS["M4_primary"]
        for method in ("kmeans", "fcm", "gmm"):
            out = lopo_validate(X, plants, feats, method=method, seed=42, n_perm=50)
            # Semantic anchoring must match exact canonical order 0=convergent, 1=expanded, 2=drooping
            self.assertGreater(adjusted_rand_index(y_true, out["full_labels"]), 0.95)
            self.assertEqual(int(np.sum(out["full_labels"] != y_true)), 0)
            self.assertGreater(out["metrics"]["lopo_oof_ari"], 0.95)
            self.assertLess(out["metrics"]["h2_permutation_p_value"], 0.05)

    def test_robust_scaler_is_resistant_to_extreme_outlier(self):
        X = np.column_stack([np.linspace(0, 10, 41), np.linspace(-1, 1, 41)])
        X_out = np.vstack([X, [[1e5, -1e5]]])
        s_clean = RobustScaler().fit(X)
        s_out = RobustScaler().fit(X_out)
        self.assertLess(abs(s_clean.median_[0] - s_out.median_[0]), 0.3)
        self.assertLess(abs(s_clean.scale_[0] - s_out.scale_[0]), 0.3)


if __name__ == "__main__":
    unittest.main()
