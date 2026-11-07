import copy
import io
import json
import random
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from calibrlab.calibrate import calibrate
from calibrlab.cli import main
from calibrlab.uncertainty import _percentile


def payload():
    return {'validation': {'logits': [[3, 0], [0, 3], [3, 0]], 'labels': [0, 1, 1]},
            'test': {'logits': [[2, 0], [0, 4], [1, 0], [0, 1]], 'labels': [0, 1, 1, 0]}}


class TestBootstrap(unittest.TestCase):
    def test_disabled_default_and_invalid_options(self):
        self.assertNotIn('bootstrap', calibrate(payload()))
        for count in [-1, 10001, True, .5]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                calibrate(payload(), bootstrap=count)
        with self.assertRaises(ValueError):
            calibrate(payload(), bootstrap=1, seed=True)
        with self.assertRaises(ValueError):
            calibrate(payload(), bootstrap=1, seed='1')

    def test_seeded_repeatable_and_no_input_or_global_rng_mutation(self):
        data = payload(); saved = copy.deepcopy(data); state = random.getstate()
        a = calibrate(data, bootstrap=40, seed=22)
        self.assertEqual(a, calibrate(data, bootstrap=40, seed=22))
        self.assertEqual(data, saved)
        self.assertEqual(random.getstate(), state)
        json.dumps(a, allow_nan=False)

    def test_accuracy_delta_zero_and_fixed_unit_t_all_deltas_zero(self):
        report = calibrate(payload(), lo=1, hi=1, bootstrap=30)['bootstrap']
        for metric in report['intervals'].values():
            self.assertEqual(metric['delta_after_minus_before'], [0., 0.])
        report = calibrate(payload(), bootstrap=30)['bootstrap']
        self.assertEqual(report['intervals']['accuracy']['delta_after_minus_before'], [0., 0.])

    def test_single_row_intervals_collapse_to_point(self):
        data = payload(); data['test'] = {'logits': [[2, 0]], 'labels': [1]}
        report = calibrate(data, bootstrap=20)
        for metric, interval in report['bootstrap']['intervals'].items():
            for kind, point in [('before', report['test_before'][metric]),
                                ('after', report['test_after'][metric])]:
                self.assertAlmostEqual(interval[kind][0], point)
                self.assertAlmostEqual(interval[kind][1], point)

    def test_test_only_bootstrap_does_not_refit_temperature(self):
        from calibrlab.temp import fit_temperature
        with patch('calibrlab.calibrate.fit_temperature', wraps=fit_temperature) as fit:
            a = calibrate(payload(), bootstrap=25)
            fit.assert_called_once()
        self.assertEqual(a['temperature'], calibrate(payload())['temperature'])
        self.assertIn('No validation refit', a['bootstrap']['scope'])

    def test_quantile_interpolation_and_sorted_bounds(self):
        self.assertAlmostEqual(_percentile([10, 0], .025), .25)
        self.assertAlmostEqual(_percentile([10, 0], .975), 9.75)
        report = calibrate(payload(), bootstrap=30)['bootstrap']
        for kinds in report['intervals'].values():
            for lo, hi in kinds.values():
                self.assertLessEqual(lo, hi)

    def test_quantile_avoids_opposite_sign_overflow(self):
        self.assertAlmostEqual(_percentile([-1e308, 1e308], .5), 0.)
        self.assertTrue(_percentile([-1e308, 1e308], .025) < 0)

    def test_work_limit_and_cli_round_trip(self):
        data = payload(); data['test'] = {'logits': [[2, 0]] * 201, 'labels': [0] * 201}
        with self.assertRaises(ValueError):
            calibrate(data, bootstrap=10000)
        out = io.StringIO()
        with patch('sys.stdin', io.StringIO(json.dumps(payload()))), redirect_stdout(out):
            self.assertEqual(main(['calibrate', '--input', '-', '--bootstrap', '12', '--seed', '7']), 0)
        self.assertEqual(json.loads(out.getvalue())['bootstrap']['draws'], 12)
