import copy
import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr
from unittest.mock import patch

from calibrlab.calibrate import calibrate
from calibrlab.cli import main
from calibrlab.mlp import softmax


def sample():
    return {"validation": {"logits": [[4., 0.], [0., 4.], [4., 0.], [0., 4.]],
                           "labels": [0, 1, 1, 0]},
            "test": {"logits": [[3., 0.], [0., 2.]], "labels": [0, 1]}}


class TestCallerLogits(unittest.TestCase):
    def test_report_and_no_mutation(self):
        data = sample()
        saved = copy.deepcopy(data)
        report = calibrate(data)
        self.assertEqual(data, saved)
        self.assertEqual(report['samples'], {'validation': 4, 'test': 2})
        self.assertEqual(report['prediction_changes'], 0)
        self.assertEqual(report['test_before']['accuracy'], report['test_after']['accuracy'])
        json.dumps(report, allow_nan=False)

    def test_test_labels_cannot_change_temperature(self):
        data = sample()
        first = calibrate(data)
        data['test']['labels'] = [1, 0]
        second = calibrate(data)
        self.assertEqual(first['temperature'], second['temperature'])
        self.assertNotEqual(first['test_after']['accuracy'], second['test_after']['accuracy'])

    def test_bad_schemas_fail(self):
        cases = [{}, {'validation': {}}, sample()]
        cases[-1]['test']['logits'] = [[1, 0, 1], [0, 1, 0]]
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                calibrate(data)

    def test_invalid_test_values_and_labels_fail(self):
        for value in [float('nan'), float('inf'), '1', True]:
            data = sample(); data['test']['logits'][0][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                calibrate(data)
        data = sample(); data['test']['labels'][0] = True
        with self.assertRaises(ValueError):
            calibrate(data)

    def test_cli_stdin_is_json_and_avoids_fixture_training(self):
        out = io.StringIO()
        with patch('sys.stdin', io.StringIO(json.dumps(sample()))), redirect_stdout(out), \
             patch('calibrlab.cli.run_evaluation', side_effect=AssertionError('unexpected training')):
            self.assertEqual(main(['calibrate', '--input', '-']), 0)
        self.assertIn('test_after', json.loads(out.getvalue()))

    def test_cli_bad_input_has_exit_two_and_no_report(self):
        out, err = io.StringIO(), io.StringIO()
        with patch('sys.stdin', io.StringIO('{')), redirect_stdout(out), redirect_stderr(err):
            self.assertEqual(main(['calibrate', '--input', '-']), 2)
        self.assertEqual(out.getvalue(), '')
        self.assertIn('input error', err.getvalue())

    def test_softmax_extreme_logits_and_temperature_validation(self):
        self.assertEqual(softmax([1e308, -1e308], .5), [1., 0.])
        for t in [0, -1, float('nan'), float('inf')]:
            with self.subTest(t=t), self.assertRaises(ValueError):
                softmax([1, 0], t)
