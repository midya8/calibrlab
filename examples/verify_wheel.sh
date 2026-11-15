#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
python -m pip wheel --no-deps "$root" --wheel-dir "$tmp/wheels" >/dev/null
python -m venv "$tmp/venv"
"$tmp/venv/bin/python" -m pip install --no-index "$tmp"/wheels/*.whl >/dev/null
cd "$tmp"
"$tmp/venv/bin/python" - <<'PY'
import importlib,json,pathlib
for name in ['calibrate','temp','metrics','eval','ledger','ledger_fixtures','ledger_export']:importlib.import_module('calibrlab.'+name)
from calibrlab.ledger_fixtures import build_bundle,fixtures
from calibrlab.ledger_export import verify,project
from calibrlab.ledger import ledger
import calibrlab
expected=fixtures();verify(build_bundle(),expected)
root=pathlib.Path(calibrlab.__file__).parent/'ledger_data'
cases={c['id']:c['payload']for c in expected['cases']};n=0
for p in sorted(root.glob('score-*.json')):
 for w in json.loads(p.read_text()):
  verify(project(ledger(cases[w['case']],w['temperature'])),w['witness']);n+=1
assert n==95
pathlib.Path('input.json').write_text(json.dumps(cases['adverse']))
print('7modules import; original full training/data/report and95witnesses reproduce outside checkout')
PY
"$tmp/venv/bin/python" -m calibrlab evaluate > evaluate.txt
"$tmp/venv/bin/python" -m calibrlab gates --min-ece-reduction .1 --max-nll-after 1.4 > gates.txt
"$tmp/venv/bin/python" -m calibrlab calibrate --input input.json > calibrate.json
"$tmp/venv/bin/python" -m calibrlab score-audit --input input.json --temperature 10 > audit.json
"$tmp/venv/bin/python" -m calibrlab score-demo > demo.json
"$tmp/venv/bin/python" - <<'PY'
import json
c=json.load(open('calibrate.json'));a=json.load(open('audit.json'));d=json.load(open('demo.json'))
assert c['temperature']==a['fit']['temperature']==10
assert a['delta']['nll']>0
assert len(d['cases'])==5
assert '1.0000' in open('evaluate.txt').read()
assert 'all gates passed' in open('gates.txt').read()
from calibrlab.calibrate import calibrate
near_tie = {'validation': {'logits': [[0., 1., 0.]], 'labels': [1]},
            'test': {'logits': [[0., 1e-16, 0.]], 'labels': [1]}}
r = calibrate(near_tie, lo=10, hi=10, bootstrap=13)
assert r['prediction_changes'] == 0
assert r['test_before']['accuracy'] == r['test_after']['accuracy'] == 1
assert abs(r['test_after']['ece'] - 2 / 3) < 1e-15
assert r['bootstrap']['intervals']['accuracy']['after'] == [1., 1.]
assert r['bootstrap']['intervals']['accuracy']['delta_after_minus_before'] == [0., 0.]
print('All5commands execute; adverse caller outcome, validation-only fit and rounded-tie bootstrap checked')
PY
