import math
import unittest

from calibrlab.metrics import accuracy, ece, mce, brier, nll
from calibrlab.mlp import softmax


def _perfect(n_classes=3):
    probs, ys = [], []
    for k in range(n_classes):
        p = [0.0] * n_classes
        p[k] = 1.0
        probs.append(p)
        ys.append(k)
    return probs, ys


class TestMetrics(unittest.TestCase):
    def test_perfect_calibration_has_zero_ece(self):
        probs, ys = _perfect()
        self.assertAlmostEqual(ece(probs, ys), 0.0, places=9)
        self.assertAlmostEqual(mce(probs, ys), 0.0, places=9)
        self.assertAlmostEqual(brier(probs, ys), 0.0, places=9)

    def test_nll_of_uniform_is_log_classes(self):
        n = 4
        probs = [[1.0 / n] * n for _ in range(8)]
        ys = [i % n for i in range(8)]
        self.assertAlmostEqual(nll(probs, ys), math.log(n), places=9)

    def test_accuracy_counts_argmax_hits(self):
        probs = [[0.7, 0.3], [0.4, 0.6], [0.9, 0.1]]
        self.assertAlmostEqual(accuracy(probs, [0, 1, 1]), 2 / 3, places=9)

    def test_temperature_scaling_preserves_argmax(self):
        logits = [0.3, -1.2, 2.5, 0.1]
        base = softmax(logits)
        for t in (0.5, 1.0, 3.85, 10.0):
            scaled = softmax(logits, t)
            self.assertEqual(max(range(4), key=lambda i: base[i]),
                             max(range(4), key=lambda i: scaled[i]))
            self.assertAlmostEqual(sum(scaled), 1.0, places=9)

    def test_higher_temperature_moves_toward_uniform(self):
        logits = [0.0, 1.0, 2.0]
        hot = softmax(logits, 10.0)
        cold = softmax(logits, 0.5)
        self.assertLess(max(hot), max(cold))


if __name__ == "__main__":
    unittest.main()
