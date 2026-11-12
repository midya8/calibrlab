# CalibrLab

Fit a temperature on validation logits, then inspect what changed on a separate
test split. CalibrLab reports accuracy, ECE, MCE, Brier score, logit-space NLL,
reliability bins and paired bootstrap intervals. A separate score ledger shows
row losses and bin migration.

Python uses only the standard library. The included synthetic MLP fixture
shows a useful validation fit; a separate caller example gets worse. They are
development controls, not a promise that calibration improves new data.

## Use your own classifier logits

```bash
python -m calibrlab calibrate --input examples/logits.json
# Or pipe the same JSON schema to stdin:
cat examples/logits.json | python -m calibrlab calibrate --input -
```

Input has exactly `validation` and `test` splits, each containing `logits`
(a list of numeric rows) and `labels` (integer class indices). Both must share
the same class order. `--lo`, `--hi` and `--step` configure the temperature grid.
Output is JSON: fitted T, sample counts, validation NLL, test metrics before/
after and prediction-change count. Invalid inputs exit 2, with an error on
stderr and no JSON report. This command never trains the synthetic MLP or fits
on test labels. You must supply genuinely separate splits; schema checks cannot
detect leaked examples or a mismatched semantic class order.

The tiny JSON example demonstrates the interface, not a successful calibration:
it selects the upper grid boundary T=10 and worsens test NLL (0.0878 to 0.5762).
A validation fit can hurt on a different test sample. There is no promised
improvement on caller data; inspect the reported metrics and grid range.

```python
from calibrlab.calibrate import calibrate
report = calibrate(payload)  # same validation/test schema as the CLI
```

## Paired bootstrap intervals

```bash
python -m calibrlab calibrate --input examples/logits.json --bootstrap 1000 --seed 13
```

The Python API accepts the same `bootstrap=1000, seed=13` options. The JSON
`bootstrap` object includes nominal 95% percentile intervals for accuracy,
ECE, MCE, Brier and NLL before/after plus paired after-minus-before deltas.
Each draw resamples identical test indices before and after. T stays fixed
at its original validation fit. The seeded local RNG does not change global
random state. Zero draws (the default) omit the object.

These are conditional IID test-row bootstrap intervals, not guaranteed
coverage or uncertainty from validation fitting/model training. Do not use
them unchanged for correlated/grouped rows. Tiny samples and binned ECE/MCE
can produce misleading or degenerate intervals; the two-row example is only
an interface demonstration. A negative loss/error delta means improvement,
but an interval is not a claim about a mechanism or future deployment.
Draws must be integers from 0 to 10000, with at most 2000000 sampled test rows.
Quantiles use linear interpolation of sorted bootstrap values.

## Numerical scoring

Temperature fitting and reported fixture/caller NLL use stable log-sum-exp
from logits, without a probability floor. A wrong-class margin of 1000 has
NLL 1000, not a clipped ~27.6. Unrepresentable losses fail with ValueError
(CLI exit 2) rather than a capped objective or Infinity JSON. The low-level
probability-only `metrics.nll` retains its documented 1e-12 floor; it is not
used for fitting or pipeline reports.

## The included training fixture

- Builds a deterministic classification fixture: six Gaussian classes in 16 dimensions with enough overlap (noise 0.6) that true uncertainty exists.
- Trains a one-hidden-layer MLP (hand-derived backprop, finite-difference verified in the suite) on only 20 samples per class - few enough that it memorizes the training set and its confidence drifts out of calibration.
- Measures calibration on held-out test data: accuracy, ECE, MCE, Brier score, NLL, plus a per-bin reliability table showing over/under-confidence.
- Fits a single temperature by grid search on the validation split's NLL - the test split never influences T - then re-measures everything at the fitted T.
- Pins the one guarantee temperature scaling makes: accuracy is exactly invariant (argmax is preserved by positive scaling), asserted bit-for-bit in the gates.
- Reports the oracle reference: the T that would minimize TEST NLL, as an honest bound on what the validation fit leaves on the table.

## Measured results (checked-in fixture, pinned in tests)

```
CalibrLab evaluation (checked-in fixture, 6 classes x dim 16, noise 0.6)

Train accuracy:                 1.0000   (overfit by design)
Test accuracy:                  0.5563

Uncalibrated (T=1) vs temperature-scaled (T fit on validation):
  ECE:    0.2765 -> 0.0651   (reduction +0.2114)
  MCE:    0.3487 -> 0.1216
  Brier:  0.7149 -> 0.5964
  NLL:    2.0030 -> 1.2060
  Accuracy drift from scaling: 0.000000   (exactly 0 by construction)

Fitted temperature (validation):  T = 3.85   (val NLL 1.3118)
Oracle temperature (on test):     T = 3.30   (test NLL 1.1984)
The oracle row is an unattainable reference: fitting on test would leak.

Reliability before:
  confidence    n    mean_conf  accuracy
  --------------------------------------------
  [0.0, 0.1)    0
  [0.1, 0.2)    0
  [0.2, 0.3)    1    0.2675     0.0000  OVER
  [0.3, 0.4)    8    0.3662     0.2500  OVER
  [0.4, 0.5)   28    0.4656     0.3929  OVER
  [0.5, 0.6)   37    0.5471     0.4324  OVER
  [0.6, 0.7)   46    0.6387     0.4130  OVER
  [0.7, 0.8)   34    0.7423     0.4412  OVER
  [0.8, 0.9)   70    0.8487     0.5000  OVER
  [0.9, 1.0)  256    0.9735     0.6602  OVER

Reliability after:
  confidence    n    mean_conf  accuracy
  --------------------------------------------
  [0.0, 0.1)    0
  [0.1, 0.2)    0
  [0.2, 0.3)   26    0.2631     0.1923  OVER
  [0.3, 0.4)  112    0.3486     0.4107  under
  [0.4, 0.5)  130    0.4471     0.5308  under
  [0.5, 0.6)   96    0.5467     0.5833  under
  [0.6, 0.7)   51    0.6431     0.7647  under
  [0.7, 0.8)   47    0.7446     0.7660  under
  [0.8, 0.9)   17    0.8369     0.8824  under
  [0.9, 1.0)    1    0.9073     1.0000  under
```

## Reading the fixture results

- The overfit model is badly overconfident: its top confidence bin averages 0.9735 confidence at 0.6602 accuracy, and ECE is 0.2765.
- Validation-fitted temperature scaling (T = 3.85) cuts ECE to 0.0651 and test NLL from 2.0030 to 1.2060, at exactly zero accuracy change - that invariance is a property of the method, not a measurement.
- The test-grid oracle temperature (3.30) would give NLL 1.1984, giving a rounded oracle gap of 0.0076.
- After scaling the model is mildly under-confident in most bins - one scalar cannot match a per-bin curve, and MCE stays higher (0.1216) because small bins are noisy. Both are reported, not smoothed over.

## Gates

`bash examples/demo.sh` runs the evaluation, asserts the achievable gate set passes (ECE reduction >= 0.10, calibrated NLL <= 1.40, oracle gap <= 0.05, zero accuracy drift), and asserts an impossible gate (ECE reduction >= 0.9) fails - CI fails in both directions.

## Layout

- `calibrlab/data.py` - deterministic 6-class Gaussian fixture (train/val/test)
- `calibrlab/mlp.py` - one-hidden-layer MLP, hand-derived gradients, finite-difference check
- `calibrlab/metrics.py` - accuracy, ECE, MCE, Brier, NLL, reliability table
- `calibrlab/temp.py` - temperature scaling by validation-NLL grid search (+ oracle reference)
- `calibrlab/eval.py` - the measured pipeline; `calibrlab/cli.py` - evaluate/gates
- `tests/` - 56 tests, retaining the earlier 45 metric/gradient/fixture tests and adding score-ledger checks

## Limits

A single scalar temperature is the weakest useful calibrator: it cannot fix a model whose confidence ranking is wrong, only one that is systematically over/under-confident, and it does nothing for accuracy. The fixture is synthetic and small; real calibration work needs per-domain validation data and richer methods (histogram binning, isotonic regression, ensemble recalibration). The oracle row is a reference bound, never a fit target.

## Numerical checks and input safety

The gradient checker samples all four parameter groups (W1, b1, W2, b2) even
with a cap; `param_cap=None` checks every parameter on a tiny model. Tests
inject a wrong gradient into each group to prove detection, and a loss failure
to verify exact parameter restoration. Temperature grids reject zero/negative
steps, nonfinite bounds, oversized grids, misaligned rows and invalid labels.
The CLI rejects any nonzero measured accuracy drift.


## v0.2: score and bin-migration ledger

The [live score ledger](https://midya8.github.io/calibrlab/) exposes every
row's stable logit NLL and class Brier terms, all confidence-bin gaps and ECE
contributions, before/selected bin migration, and every fixed-temperature
row deletion. Choose 2..20 bins or a temperature in 0.1..20; the validation fit
still uses only the original 191-point NLL grid. A 19-point descriptive path
shows validation/test NLL without recommending a test-chosen temperature.

Five public synthetic controls retain all original 240 validation / 480 test
logits and the unchanged report, the adverse caller example, uniform ties,
within-bin cancellation and singleton underflow. All 493 positions are
inspectable. Copy/download includes the full input and native-compatible
report, not just a selected row. Inputs stay in the page; no paid API.

```bash
python -m calibrlab score-audit --input examples/logits.json --temperature 10
python -m calibrlab score-demo
node --test tests/*.cjs
bash examples/verify_wheel.sh
```

56 Python tests (original 45 unchanged) and four Node/actual-Chrome tests check
95 native/browser witnesses, 9,367 row/deletion records, 55,328 class Brier terms,
160 independent 70-digit Decimal games and 80 integer/uniform controls. Installed
wheel reproduction rebuilds the original full training/logits/report and all
95 witnesses outside checkout, imports 7 modules and runs all 5 commands.

These are standard temperature scaling and proper-score formulas, not a new
calibration method. ECE/MCE depend on bins and only the top class; NLL/Brier
are not calibration-only measures. Fixed-temperature deletion is not refitting,
causality or uncertainty. Binary64/endpoint limits remain explicit. The
original 45 tests, model/data/trainer/metrics/grid/bootstraps and all original
results above are unchanged. See [LEDGER.md](LEDGER.md) for equations, input
bounds, schema, reproduction, six primary/reference sources and limits.

The original overview's oracle gap 0.0076 subtracts its displayed rounded NLLs.
The unchanged full-precision gap is 0.007543134492483228, shown as 0.0075 in the
new ledger; this is a display-precision distinction, not a changed result.
