import math
import unittest

from calibrlab.metrics import logits_nll, nll
from calibrlab.mlp import softmax
from calibrlab.temp import fit_temperature
from calibrlab.calibrate import calibrate


class TestLogitNLL(unittest.TestCase):
    def test_matches_probability_loss_in_ordinary_range(self):
        rows, ys = [[1, 2, -3], [-2, 3, 1]], [0, 1]
        self.assertAlmostEqual(logits_nll(rows, ys), nll([softmax(r) for r in rows], ys))

    def test_no_probability_floor_at_large_margin(self):
        self.assertAlmostEqual(logits_nll([[1000, 0]], [1]), 1000)
        self.assertAlmostEqual(logits_nll([[1000, 0]], [1], 10), 100)
        self.assertEqual(fit_temperature([[1000, 0]], [1], 1, 10, 1)[0], 10)

    def test_translation_invariant_and_uniform(self):
        self.assertAlmostEqual(logits_nll([[1002, 1000]], [1]), logits_nll([[2, 0]], [1]))
        self.assertAlmostEqual(logits_nll([[1e308, 1e308]], [0]), math.log(2))

    def test_opposite_extremes_and_unrepresentable_loss(self):
        self.assertAlmostEqual(logits_nll([[1e308, -1e308]], [1], 10), 2e307)
        with self.assertRaises(ValueError):
            logits_nll([[1e308, -1e308]], [1], .5)

    def test_mean_avoids_intermediate_sum_overflow(self):
        self.assertEqual(logits_nll([[1e308, 0]] * 4, [1] * 4), 1e308)

    def test_softmax_opposite_extremes_at_large_temperature(self):
        p = softmax([1e308, -1e308], 1e308)
        self.assertAlmostEqual(p[1], 1 / (1 + math.exp(2)))

    def test_grid_overflow_is_clean_value_error(self):
        with self.assertRaises(ValueError):
            fit_temperature([[1, 0]], [0], 1e-308, 1e308, 1e-308)

    def test_caller_report_uses_unclipped_logit_nll(self):
        split = {'logits': [[1000, 0]], 'labels': [1]}
        result = calibrate({'validation': split, 'test': split}, lo=1, hi=10, step=1)
        self.assertEqual(result['temperature'], 10)
        self.assertAlmostEqual(result['test_before']['nll'], 1000)
        self.assertAlmostEqual(result['test_after']['nll'], 100)
