"""Regression controls for decisions hidden by rounded softmax ties."""
import copy
import json
import math
import random
import subprocess
import sys
import unittest

from calibrlab.calibrate import calibrate
from calibrlab.metrics import accuracy, ece, mce
from calibrlab.mlp import softmax


def payload(rows, labels):
    return {'validation': {'logits': [[0., 1., 0.]], 'labels': [1]},
            'test': {'logits': rows, 'labels': labels}}


def reference(rows, labels, temperature, indices=None):
    indices = list(range(len(rows))) if indices is None else indices
    # Decisions are from supplied logits; confidence is still a rounded
    # softmax value. No epsilon, tie-breaking perturbation or probability fix.
    pairs = []
    for i in indices:
        k = max(range(len(rows[i])), key=rows[i].__getitem__)
        pairs.append((softmax(rows[i], temperature)[k], int(k == labels[i])))
    gaps = []
    for b in range(10):
        members = [(c, a) for c, a in pairs
                   if b / 10 <= c < (b + 1) / 10 or (b == 9 and c == 1)]
        if members:
            gaps.append((len(members), abs(sum(a - c for c, a in members) / len(members))))
    return {'accuracy': sum(a for c, a in pairs) / len(pairs),
            'ece': sum(n * gap for n, gap in gaps) / len(pairs),
            'mce': max(gap for n, gap in gaps)}


def percentile(values, q):
    values = sorted(values)
    x = (len(values) - 1) * q
    lo = int(x)
    fraction = x - lo
    return values[lo] * (1 - fraction) + values[math.ceil(x)] * fraction


class LogitDecisionTests(unittest.TestCase):
    def test_new_rounded_tie_is_not_a_prediction_change(self):
        p = payload([[0., 1e-16, 0.]], [1])
        self.assertEqual(softmax(p['test']['logits'][0], 10), [1 / 3] * 3)
        report = calibrate(p, lo=10, hi=10)
        self.assertEqual(report['prediction_changes'], 0)
        for side in ['test_before', 'test_after']:
            self.assertEqual(report[side]['accuracy'], 1)
            self.assertAlmostEqual(report[side]['ece'], 2 / 3)
            self.assertAlmostEqual(report[side]['mce'], 2 / 3)
        # The probability-only API still reports its own first-max decision.
        self.assertEqual(accuracy([softmax(p['test']['logits'][0], 10)], [1]), 0)

    def test_tiny_gaps_true_ties_labels_and_temperatures(self):
        rows = [[0., d, 0.] for d in [1e-16, 1e-18, math.ulp(0.), 0., -1e-16]]
        rows += [[1e-16, 0., 0.], [0., 0., 1e-16], [1., 1., 0.], [0., 0., 0.]]
        for shift in range(3):
            labels = [(i + shift) % 3 for i in range(len(rows))]
            p = payload(rows, labels)
            saved = copy.deepcopy(p)
            for t in [.1, .5, 1., 2., 10., 20.]:
                with self.subTest(shift=shift, temperature=t):
                    report = calibrate(p, lo=t, hi=t)
                    self.assertEqual(report['prediction_changes'], 0)
                    for side, temp in [('test_before', 1.), ('test_after', t)]:
                        for metric, value in reference(rows, labels, temp).items():
                            self.assertAlmostEqual(report[side][metric], value, places=14)
                    json.dumps(report, allow_nan=False)
            self.assertEqual(p, saved)

    def test_paired_bootstrap_matches_direct_resampled_logit_decisions(self):
        rows = [[0., 1e-16, 0.], [0., 0., 1e-18], [0., 0., 0.], [0., 1., 0.]]
        labels = [1, 2, 1, 0]
        report = calibrate(payload(rows, labels), lo=10, hi=10, bootstrap=47, seed=23)
        rng = random.Random(23)
        expected = {k: {side: [] for side in ['before', 'after', 'delta_after_minus_before']}
                    for k in ['accuracy', 'ece', 'mce']}
        for _ in range(47):
            indices = [rng.randrange(4) for i in range(4)]
            a, b = reference(rows, labels, 1., indices), reference(rows, labels, 10., indices)
            for k in expected:
                expected[k]['before'].append(a[k])
                expected[k]['after'].append(b[k])
                expected[k]['delta_after_minus_before'].append(b[k] - a[k])
        for k, kinds in expected.items():
            for kind, values in kinds.items():
                got = report['bootstrap']['intervals'][k][kind]
                for actual, q in zip(got, [.025, .975]):
                    self.assertAlmostEqual(actual, percentile(values, q), places=14)
        self.assertEqual(report['bootstrap']['intervals']['accuracy']['delta_after_minus_before'], [0., 0.])

    def test_explicit_decisions_are_optional_validated_and_do_not_edit_probabilities(self):
        probs = [[1 / 3] * 3, [.5, .5, 0.]]
        saved = copy.deepcopy(probs)
        labels, decisions = [2, 1], [2, 1]
        self.assertEqual(accuracy(probs, labels), 0.)
        self.assertEqual(accuracy(probs, labels, predictions=decisions), 1.)
        self.assertAlmostEqual(ece(probs, labels, predictions=decisions), 7 / 12)
        self.assertAlmostEqual(mce(probs, labels, predictions=decisions), 2 / 3)
        for bad in [[], [1], [True, 1], [0., 1], [3, 1], [-1, 1]]:
            for fn in [accuracy, ece, mce]:
                with self.subTest(fn=fn.__name__, predictions=bad), self.assertRaises(ValueError):
                    fn(probs, labels, predictions=bad)
        self.assertEqual(probs, saved)
        self.assertEqual(decisions, [2, 1])

    def test_cli_and_singleton_bootstrap_keep_original_decision(self):
        p = payload([[0., 1e-16, 0.]], [1])
        run = subprocess.run([sys.executable, '-m', 'calibrlab', 'calibrate', '--input', '-',
                              '--lo', '10', '--hi', '10', '--bootstrap', '13'],
                             input=json.dumps(p), text=True, capture_output=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        r = json.loads(run.stdout)
        self.assertEqual(r['prediction_changes'], 0)
        self.assertEqual(r['test_after']['accuracy'], 1.)
        self.assertEqual(r['bootstrap']['intervals']['accuracy']['after'], [1., 1.])
        self.assertAlmostEqual(r['bootstrap']['intervals']['ece']['after'][0], 2 / 3)

    def test_probability_losses_and_validation_fit_are_not_changed(self):
        p = payload([[0., 1e-16, 0.], [0., 0., 0.]], [1, 2])
        a = calibrate(p, lo=10, hi=10)
        p['test']['labels'] = [0, 1]
        b = calibrate(p, lo=10, hi=10)
        self.assertEqual(a['temperature'], b['temperature'])
        self.assertEqual(a['validation_nll'], b['validation_nll'])
        self.assertNotEqual(a['test_after']['accuracy'], b['test_after']['accuracy'])
        from calibrlab.metrics import brier, logits_nll
        for side, t in [('test_before', 1.), ('test_after', 10.)]:
            probs = [softmax(row, t) for row in p['test']['logits']]
            self.assertEqual(b[side]['brier'], brier(probs, p['test']['labels']))
            self.assertEqual(b[side]['nll'], logits_nll(p['test']['logits'], p['test']['labels'], t))
