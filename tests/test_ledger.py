import copy
import decimal
import json
import math
import random
import subprocess
import sys
import unittest
from pathlib import Path
from calibrlab.ledger import ledger, _rows, _summary, bin_index
from calibrlab.ledger_fixtures import fixtures, TEMPERATURES
from calibrlab.ledger_export import project, verify
from calibrlab.calibrate import calibrate


class LedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = fixtures()
        cls.cases = {c['id']: c['payload'] for c in cls.data['cases']}

    def test_original_pipeline_and_caller_retained(self):
        from calibrlab.eval import run_evaluation
        verify(run_evaluation(), self.data['original_report'])
        self.assertEqual(self.cases['adverse'], json.loads(Path('examples/logits.json').read_text()))
        for name, p in self.cases.items():
            r = ledger(p, 1)
            c = calibrate(p)
            self.assertAlmostEqual(r['fit']['temperature'], c['temperature'])
            for key, v in c['test_before'].items():
                self.assertAlmostEqual(r['before'][key], v, places=12)
        adverse = ledger(self.cases['adverse'], 10)
        self.assertGreater(adverse['delta']['nll'], 0)

    def test_all_95_archives_strict_numeric(self):
        root = Path('calibrlab/ledger_data')
        n = 0
        for path in sorted(root.glob('score-*.json')):
            for w in json.loads(path.read_text()):
                verify(project(ledger(self.cases[w['case']], w['temperature'])), w['witness'])
                n += 1
        self.assertEqual(n, 95)

    def test_all_rows_and_bin_terms_reconstruct(self):
        for name, p in self.cases.items():
            for t in [.1, 1, 3.85, 20]:
                r = ledger(p, t)
                n = len(r['rows'])
                for side in ['before', 'selected']:
                    rows = [x[side] for x in r['rows']]
                    self.assertEqual(sum(b['count'] for b in r[side]['bins']), n)
                    self.assertEqual(sorted(i for b in r[side]['bins'] for i in b['positions']), list(range(n)))
                    self.assertAlmostEqual(math.fsum(b['ece_contribution'] for b in r[side]['bins']), r[side]['ece'])
                    self.assertAlmostEqual(sum(x['nll'] for x in rows) / n, r[side]['nll'])
                    for row in rows:
                        self.assertAlmostEqual(sum(row['probabilities']), 1)
                        self.assertEqual(row['brier'], sum(row['brier_terms']))
                self.assertEqual(sum(map(sum, r['migration'])), n)
                self.assertEqual([sum(row) for row in r['migration']], [b['count'] for b in r['before']['bins']])
                self.assertEqual([sum(row[j] for row in r['migration']) for j in range(10)], [b['count'] for b in r['selected']['bins']])
                self.assertEqual(r['delta']['accuracy'], 0)

    def test_all_position_deletion_direct_recompute(self):
        for name, p in self.cases.items():
            for t in [.1, 1, 3.85, 20]:
                r = ledger(p, t, 7)
                rows = [v['selected'] for v in r['rows']]
                for d in r['deletions']:
                    remain = [x for x in rows if x['position'] != d['position']]
                    ref = _summary(remain, 7)
                    expected = None if ref is None else {k: ref[k] for k in ['accuracy','nll','brier','ece','mce']}
                    verify(d['scores'], expected)

    def test_160_independent_decimal_probability_games(self):
        # Decimal exp/log independent of binary64 softmax/logits_nll arithmetic.
        rng = random.Random(619)
        with decimal.localcontext() as ctx:
            ctx.prec = 70
            D = decimal.Decimal
            for seed in range(160):
                k = 2 + seed % 5
                z = [rng.randrange(-30, 31) / 4 for _ in range(k)]
                y = seed % k
                t = [.25, .5, 1, 2, 10][seed % 5]
                p = {'validation': {'logits': [z], 'labels': [y]}, 'test': {'logits': [z], 'labels': [y]}}
                r = ledger(p, t, 2 + seed % 19)['rows'][0]['selected']
                ex = [(D(str(v))/D(str(t))).exp() for v in z]
                probs = [v / sum(ex) for v in ex]
                nll = -probs[y].ln()
                brier = sum((v - int(i == y)) ** 2 for i,v in enumerate(probs))
                for a,b in zip(r['probabilities'],probs):self.assertAlmostEqual(a,float(b),places=12)
                self.assertAlmostEqual(r['nll'],float(nll),places=12)
                self.assertAlmostEqual(r['brier'],float(brier),places=12)

    def test_boundaries_ties_and_cancellation(self):
        self.assertEqual([bin_index(v,10) for v in [0,.1,.3,.5,.9,1]], [0,1,3,5,9,9])
        tied = ledger(self.cases['ties'], 20)
        self.assertTrue(all(x['selected']['prediction'] == 0 for x in tied['rows']))
        self.assertEqual(tied['delta']['nll'], 0)
        cancel = ledger(self.cases['cancellation'], 1)
        self.assertEqual(cancel['selected']['ece'], 0)
        self.assertEqual(cancel['selected']['accuracy'], .75)
        self.assertEqual(sum(x['selected']['correct'] == 0 for x in cancel['rows']), 1)

    def test_underflow_stable_nll_singleton_and_bounds(self):
        r = ledger(self.cases['extreme'], 1)
        self.assertEqual(r['selected']['nll'], 1000)
        self.assertEqual(r['rows'][0]['selected']['probabilities'][1], 0)
        self.assertIsNone(r['deletions'][0]['scores'])
        for t in [True,'1',0,.09,20.1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):ledger(self.cases['ties'],t)
        for b in [True,'10',1,21,1.5]:
            with self.assertRaises(ValueError):ledger(self.cases['ties'],1,b)
        for edit in ['extra','too_many','huge','bool','classes']:
            p = copy.deepcopy(self.cases['adverse'])
            if edit=='extra':p['extra']=1
            if edit=='too_many':p['test']={'logits':[[0,1]]*513,'labels':[0]*513}
            if edit=='huge':p['test']['logits'][0][0]=10001
            if edit=='bool':p['test']['labels'][0]=True
            if edit=='classes':p['test']['logits']=[[0]*17]*2
            with self.assertRaises(ValueError):ledger(p)

    def test_inputs_immutable_and_test_labels_never_fit(self):
        p = copy.deepcopy(self.cases['original']);before=copy.deepcopy(p)
        a = ledger(p,2,10)
        self.assertEqual(p,before)
        p['test']['labels']=[(y+1)%6 for y in p['test']['labels']]
        b=ledger(p,2,10)
        self.assertEqual(a['fit'], b['fit'])
        self.assertNotEqual(a['selected']['nll'],b['selected']['nll'])
        self.assertEqual([x['selected']['probabilities'] for x in a['rows']], [x['selected']['probabilities'] for x in b['rows']])

    def test_temperature_path_and_bin_sensitivity(self):
        p=self.cases['adverse']
        r=ledger(p,3.85,3)
        self.assertEqual([x['temperature'] for x in r['path']],TEMPERATURES)
        for point in r['path']:
            direct=_summary(_rows(p['test']['logits'],p['test']['labels'],point['temperature'],3),3)
            verify(point['test'], {k:direct[k] for k in point['test']})
        for bins in [2,5,10,15,20]:
            other=ledger(p,3.85,bins)
            self.assertEqual(other['selected']['nll'],r['selected']['nll'])
            self.assertEqual(other['selected']['brier'],r['selected']['brier'])

    def test_cli_duplicate_rejected_only_for_audit(self):
        bad='{"validation":{},"validation":{},"test":{}}'
        out=subprocess.run([sys.executable,'-m','calibrlab','score-audit','--input','-'],input=bad,text=True,capture_output=True)
        self.assertEqual(out.returncode,2);self.assertEqual(out.stdout,'');self.assertIn('duplicate',out.stderr)
        p=subprocess.run([sys.executable,'-m','calibrlab','score-audit','--input','examples/logits.json','--temperature','10'],text=True,capture_output=True)
        self.assertEqual(p.returncode,0);self.assertEqual(json.loads(p.stdout)['fit']['temperature'],10)
        out=subprocess.run([sys.executable,'-m','calibrlab','score-demo'],text=True,capture_output=True)
        self.assertEqual(out.returncode,0);self.assertEqual(len(json.loads(out.stdout)['cases']),5)

    def test_witness_comparator_rejects_hidden_changes(self):
        for a,b in [(True,1),(float('nan'),1.),({'a':1},{'a':1,'b':2}),([1],[1,2]),(1.1,1.)]:
            with self.assertRaises(AssertionError):verify(a,b)
