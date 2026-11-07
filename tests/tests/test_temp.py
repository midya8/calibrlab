import copy
import unittest
from unittest.mock import patch

from calibrlab import mlp
from calibrlab.temp import fit_temperature


class TestTemperatureAndGradients(unittest.TestCase):
    def test_grid_rejects_invalid_bounds(self):
        for lo, hi, step in [(0, 1, .1), (2, 1, .1), (1, 2, 0),
                             (1, 2, -.1), (1, float('inf'), .1),
                             (float('nan'), 2, .1), (1, 2, 1e-9)]:
            with self.subTest(lo=lo, hi=hi, step=step):
                with self.assertRaises(ValueError):
                    fit_temperature([[1, 0]], [0], lo, hi, step)

    def test_grid_rejects_invalid_data(self):
        for logits, ys in [([], []), ([[1, 0]], []), ([[]], [0]),
                           ([[1, 0], [1]], [0, 0]),
                           ([[float('nan'), 0]], [0]), ([[1, 0]], [2]),
                           ([[1, 0]], [-1]), ([[1, 0]], [.5])]:
            with self.subTest(logits=logits, ys=ys):
                with self.assertRaises(ValueError):
                    fit_temperature(logits, ys)

    def test_grid_tie_uses_first_candidate(self):
        t, _ = fit_temperature([[0, 0]], [0], lo=.5, hi=1, step=.1)
        self.assertEqual(t, .5)

    def test_grid_includes_aligned_upper_endpoint(self):
        t, _ = fit_temperature([[5, 0]], [1], lo=.5, hi=1, step=.1)
        self.assertEqual(t, 1.0)

    def test_gradient_check_covers_every_parameter_group(self):
        model = mlp.MLP(2, 2, 2)
        xs, ys = [[.4, -.2], [-.1, .7]], [0, 1]
        self.assertLess(mlp.finite_difference_check(model, xs, ys, param_cap=None), 1e-6)
        original = mlp.loss_and_grads
        for group in range(4):
            def corrupted(*args):
                loss, grads = original(*args)
                if group in (0, 2):
                    grads[group][0][0] += .1
                else:
                    grads[group][0] += .1
                return loss, grads
            with self.subTest(group=group), patch.object(mlp, 'loss_and_grads', corrupted):
                self.assertGreater(mlp.finite_difference_check(model, xs, ys, param_cap=4), .09)

    def test_gradient_check_restores_parameters_on_error(self):
        model = mlp.MLP(2, 2, 2)
        saved = copy.deepcopy(model.__dict__)
        original = mlp.loss_and_grads
        calls = 0
        def failing(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('synthetic loss failure')
            return original(*args)
        with patch.object(mlp, 'loss_and_grads', failing):
            with self.assertRaises(RuntimeError):
                mlp.finite_difference_check(model, [[1, 0]], [0])
        self.assertEqual(model.__dict__, saved)
