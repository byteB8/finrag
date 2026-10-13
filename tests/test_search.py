import unittest

import numpy as np

from core import metrics
from core.search import rrf, top_k


class Fusion(unittest.TestCase):
    def test_rrf_rewards_agreement(self):
        # 2 is second in both lists, 1 and 3 are first in one list and absent from the other
        self.assertEqual(rrf([[1, 2], [3, 2]], k=60)[0], 2)

    def test_rrf_ties_break_by_id(self):
        self.assertEqual(rrf([[5], [4]], k=60), [4, 5])

    def test_top_k_with_mask(self):
        s = np.array([0.1, 0.9, 0.5, 0.7])
        self.assertEqual(top_k(s, 2), [1, 3])
        self.assertEqual(top_k(s, 2, mask=np.array([True, False, True, False])), [2, 0])
        self.assertEqual(top_k(s, 10, mask=np.array([False, False, True, False])), [2])
        self.assertEqual(top_k(s, 3, mask=np.zeros(4, dtype=bool)), [])


class Metrics(unittest.TestCase):
    def test_ranks(self):
        ranks = [metrics.first_hit(["a", "b", "c"], {"b"}), metrics.first_hit(["a"], {"z"}), 1]
        self.assertEqual(ranks, [2, None, 1])
        np.testing.assert_array_equal(metrics.hits_at(ranks, 1), [0, 0, 1])
        np.testing.assert_array_equal(metrics.hits_at(ranks, 2), [1, 0, 1])
        np.testing.assert_allclose(metrics.reciprocal_ranks(ranks, 5), [0.5, 0, 1])
        np.testing.assert_allclose(metrics.reciprocal_ranks(ranks, 1), [0, 0, 1])

    def test_bootstrap(self):
        m, lo, hi = metrics.bootstrap_ci([1.0] * 20)
        self.assertEqual((m, lo, hi), (1.0, 1.0, 1.0))
        m, lo, hi = metrics.bootstrap_ci([0, 1] * 50)
        self.assertAlmostEqual(m, 0.5)
        self.assertTrue(lo < 0.5 < hi)

    def test_paired_diff(self):
        d = metrics.paired_diff([1] * 30, [0] * 30)
        self.assertEqual((d["diff"], d["p"]), (1.0, 0.0))


if __name__ == "__main__":
    unittest.main()
