# Score, bin and migration ledger

This is an inspection layer over standard temperature scaling and proper-score
losses. It is not a new calibration method, uncertainty interval or deployment
rule. Existing model/data/training/evaluation/metrics/temperature-fit/bootstraps
and the original 45 tests are unchanged.

## Native API and commands

```python
from calibrlab.ledger import ledger
report = ledger(payload, temperature=3.85, bins=10)
```

`payload` has exactly `validation` and `test`, each with `logits` and `labels`,
matching the original `calibrate` interface. Each split is bounded to 1..512
rows, 2..16 classes, finite numeric logits within [-10000,10000], and integer
class indices. Class dimensions must agree; semantic class order and genuinely
separate splits remain the caller's responsibility. Booleans are not numbers
in the schema. Temperature is finite in [0.1,20]; bins are integers 2..20.

```bash
python -m calibrlab score-audit --input examples/logits.json --temperature 10 --bins 10
python -m calibrlab score-demo
python -m calibrlab.ledger_export
python -m unittest discover -s tests -v
node --test tests/*.cjs
bash examples/verify_wheel.sh
```

`score-audit` accepts `--input -` for stdin, rejects duplicate JSON keys and
returns exit2 with no report on invalid input. This duplicate-key rule applies
only to the new command; existing command parsing is unchanged. `score-demo`
returns the checked-in controls and original report, not new model training.
`ledger_export` rebuilds the 19 witness chunks from those controls. To regenerate
the original fixture with the untouched trainer, use
`calibrlab.ledger_fixtures.build_bundle()` and compare with `fixtures()` using
`calibrlab.ledger_export.verify`.

The browser computes locally. It exports the full report plus a `ui` object
(selected position, case description and unchanged original report). For native
reproduction, call `ledger(exported['input'], exported['temperature'],
exported['bins'])`. Native and browser numerical fields use the same schema;
`project` strips only input/scope/path for the compact checked-in witnesses.
The full 19-point path is tested separately against all native witness states.

## Equations and conventions

For row logits z, true label y, maximum m and positive temperature T:

- p[k] = exp((z[k]-m)/T) / sum_j exp((z[j]-m)/T).
- Prediction is the first maximum of the original logits. Positive scaling
  preserves this argmax in real arithmetic, including tied maxima. Binary64
  probabilities can acquire rounded ties or underflow, so predictions do not
  get recomputed by sorting rounded probabilities.
- Row NLL = (m-z[y])/T + log(sum_k exp((z[k]-m)/T)), in nats. No probability
  floor is used. A wrong margin1000 has NLL1000 at T1 even when p[y] rounds to0.
- Row multiclass Brier = sum_k (p[k]-1[k=y])², with no 1/K or 1/2 scaling.
- Row confidence c is the probability of the selected first-maximum class;
  a=1[prediction=y]. Original ECE convention is retained: equal-width [b/B,
  (b+1)/B), except the last bin includes1. Bin assignment compares endpoints
  directly rather than using floor(c*B), whose rounding can differ.
- For occupied bin b: C_b=sum(c), A_b=sum(a), n_b=count. Signed gap is
  (C_b-A_b)/n_b. ECE contribution is |C_b-A_b|/N, so ECE is the sum of these
  contributions. MCE is max_b |(C_b-A_b)/n_b| over occupied bins.
- Empty-bin confidence, accuracy and signed gap are null, not zero. Empty-bin
  ECE contribution is0. All bins still appear.
- Accuracy/NLL/Brier are averages over test rows. Migration M[b,c] counts
  positions in before-bin b and selected-bin c. Its row/column sums equal the
  two bin-count vectors exactly.

Temperature fitting is the untouched original 191-candidate grid
T=.5+i*.05, i=0..190. The incumbent changes only when validation NLL improves
by more than1e-12; ties keep the earlier grid point. No test label is used in
fitting. The displayed 19 sampled temperatures are not this optimizer grid and
are not joined/interpolated. Selecting T by inspecting test paths would leak
information; the path is descriptive, not a recommendation.

For each test position, deletion holds temperature/logits fixed and removes
that row. Counts, confidence sums and correctness sums change only in its bin;
ECE renormalizes by N-1, and MCE is recomputed across remaining occupied bins.
NLL/Brier/accuracy are recomputed on the remaining rows. A singleton deletion
has zero rows and null scores, never zero scores. These are not validation
refits, training influence functions, causal effects or confidence intervals.

## Controls and checked evidence

- Original seed13 six-class MLP: all240validation and480test logits from the
  unchanged trainer, plus the unchanged original evaluation report. Training
  is not rerun by the browser. At fitted T3.85, original ten-bin ECE goes
  .2765 to .0651 and NLL2.0030 to1.2060 at zero accuracy change; the test-oracle
  T3.30 remains an unattainable reference, never a fit target.
- Original caller example: validation picks upper boundary T10 and test NLL
  worsens. It remains a failure example, not a calibration win.
- Six uniform three-class ties: first maximum class0, fixed probabilities at
  every temperature. Shared synthetic values are an arithmetic control.
- Four repeated binary logits with p(max)=.75 at T1: three correct/one wrong.
  Signed within-bin gaps cancel to ECE0 even though accuracy is.75, illustrating
  why per-row error and bin calibration are different objects. This does not
  claim that the population is calibrated or miscalibrated.
- One wrong margin1000 row: underflow, finite logit NLL and undefined deletion.

There are95case/temperature native witnesses,9,367row records and9,367
fixed-temperature deletions, with55,328class Brier terms. Tests additionally
check160independent70-digit Decimal probability/loss games and80independent
integer/uniform-probability browser games, direct all-position deletion
recomputation, input immutability, test-label noninterference in the fit,
bin-count sensitivity, CLI invalid/duplicate input, copy/download failure and
success, keyboard input and responsive320/390/1280layouts. Actual Chrome runs
all95full native witnesses and DOM-checks all493control positions and50bins.

Comparison requires exact structure, keys, booleans, strings, nulls and integer
fields, with1e-12absolute/relative tolerance for finite float fields. Python
uses math.fsum where relevant; browser independently uses compensated sums.
Counts/ties/indices are exact under the documented binary64 inputs, not a
claim of arbitrary-precision transcendental functions. Numerical boundaries
can differ on other runtimes or after rounded input edits.

The original README gap0.0076 comes from subtracting its rounded displayed
NLLs. The unchanged full-precision gap0.007543134492483228 rounds to0.0075 in
the new ledger. This is not a changed fit or result.

## Statistical and practical limits

ECE/MCE depend on bins, finite samples and only the top class. A small ECE does
not establish calibration of every class probability or a deployment domain;
within-bin cancellation and sparse-bin noise remain. NLL and Brier are proper
forecast losses, but a smaller value is not evidence of calibration alone.
Temperature scaling cannot repair incorrect class ranking or guarantee
improvement on a new test set. The synthetic original is descriptive, not a
representative model benchmark or mechanism-isolating experiment. The four
small controls reuse/shared values deliberately and are not independent
validation/test experiments. Test path inspection and row deletion provide
no uncertainty estimates. Existing paired-bootstrap APIs retain their
conditional IID test-row limitations and are not added to this browser ledger.
No data leaves the page, no paid service is called, and model training is not
performed by the demo.

## Sources and attribution

Standard methods and limitations, not original methodological claims:

1. Guo, Pleiss, Sun and Weinberger (2017), *On Calibration of Modern Neural
   Networks*, section4.2, equation9: temperature scaling, validation NLL fit,
   argmax invariance. https://arxiv.org/pdf/1706.04599
2. Roelofs, Cain, Shlens and Mozer (2022), *Mitigating Bias in Calibration Error
   Estimation*: finite-sample/binning bias and equal-mass alternatives. This
   ledger does not implement their debiasing estimator or ECEsweep.
   https://proceedings.mlr.press/v151/roelofs22a/roelofs22a.pdf
3. Nixon et al. (2019), *Measuring Calibration in Deep Learning*, sections3.1-3.4:
   top-class-only limitations, binning tradeoffs and cancellation. This ledger
   is not ACE/SCE/TACE. https://openaccess.thecvf.com/content_CVPRW_2019/papers/Uncertainty%20and%20Robustness%20in%20Deep%20Visual%20Learning/Nixon_Measuring_Calibration_in_Deep_Learning_CVPRW_2019_paper.pdf
4. Gneiting and Raftery (2007), *Strictly Proper Scoring Rules, Prediction, and
   Estimation*: proper scoring rules and score aggregation on the same forecast
   situations. https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jasa.pdf
5. scikit-learn, *Probability calibration*: Brier/log loss assess reliability,
   resolution and data uncertainty together; loss improvements do not isolate
   calibration. https://scikit-learn.org/stable/modules/calibration.html
6. SciPy, *logsumexp*: stable log of a sum of exponentials. The implementation
   here is dependency-free and does not call SciPy.
   https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.logsumexp.html
