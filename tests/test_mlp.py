import unittest

from calibrlab.data import make_fixture
from calibrlab.mlp import MLP, finite_difference_check, loss_and_grads, train


class TestMLP(unittest.TestCase):
    def test_gradients_finite_difference(self):
        fixture = make_fixture(seed=13)
        xs, ys = fixture["train"]
        model = MLP(fixture["dim"], hidden=24, n_classes=fixture["n_classes"], seed=7)
        worst = finite_difference_check(model, xs[:12], ys[:12])
        self.assertLess(worst, 1e-6)

    def test_training_reduces_loss(self):
        fixture = make_fixture(seed=13)
        xs, ys = fixture["train"]
        model = MLP(fixture["dim"], hidden=24, n_classes=fixture["n_classes"], seed=7)
        before, _ = loss_and_grads(model, xs, ys)
        train(model, xs, ys, epochs=50, lr=0.5)
        after, _ = loss_and_grads(model, xs, ys)
        self.assertLess(after, before * 0.5)

    def test_fixture_is_deterministic(self):
        a = make_fixture(seed=13)
        b = make_fixture(seed=13)
        self.assertEqual(a["train"][1], b["train"][1])
        self.assertEqual(a["train"][0][0], b["train"][0][0])


if __name__ == "__main__":
    unittest.main()
