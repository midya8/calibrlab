# CalibrLab design notes

## The claim being tested

Temperature scaling (Guo et al. 2017, "On Calibration of Modern Neural
Networks") rescales logits by one scalar T fit on held-out data. Two properties
make it a clean subject for a from-scratch implementation:

1. It has an exact guarantee: positive scaling preserves argmax, so accuracy
   cannot change. The gates assert drift == 0 bit-for-bit, which catches any
   implementation that accidentally couples T into the prediction path.
2. Its failure modes are knowable: one scalar cannot fix a miscalibrated
   ranking, and a validation fit can only approximate the test-optimal T.

## Fixture

Six classes in 16 dimensions; class centers are random unit vectors, samples
are center + 0.6 * Gaussian noise (seed 13). Overlap makes Bayes accuracy well
below 1.0, so uncertainty is real. Train 20/class (120 samples) forces the
24-hidden-unit MLP to memorize: train accuracy 1.0000, test 0.5563, top
reliability bin at 0.9735 mean confidence with 0.6602 accuracy.

## Metrics

ECE/MCE use 10 equal-width confidence bins (top-bin inclusive at 1.0).
Brier is the multiclass squared error; NLL is mean negative log-likelihood of
the true class. The reliability table prints per-bin mean confidence vs
accuracy so over/under-confidence is visible, not just summarized.

## Temperature fit

Grid search T in [0.5, 10.0] at 0.05 steps minimizing validation NLL. This reports
the best temperature on that finite grid, not a certified continuous optimum. The oracle row fits T
on the test split and is labeled unattainable - it exists to quantify what the
validation fit leaves on the table (0.0076 NLL), not to be used.

## Measured reference points (checked-in fixture)

ECE 0.2765 -> 0.0651 (T = 3.85 fit on validation); MCE 0.3487 -> 0.1216;
Brier 0.7149 -> 0.5964; NLL 2.0030 -> 1.2060 (oracle T = 3.30, NLL 1.1984);
accuracy 0.5563 unchanged exactly. Pinned in tests as fixture regression.
Post-scaling bins skew slightly under-confident: one scalar cannot match a
per-bin curve, which is the honest limitation of the method.

## Development scope

Noise levels 0.5, 0.6, 0.7 and 0.8 were explored during fixture development
and 0.6 was selected for this demonstration. The test split does not fit T,
but these are development-fixture results, not an untouched final benchmark.
No Bayes-optimal accuracy or sampling-uncertainty estimate is computed.

## Verification coverage

Round-robin finite differences cover both matrices and both bias vectors.
Tests check every parameter on a tiny model, inject errors into each group,
and ensure a raised loss restores the original parameters. Grid-search inputs
are validated before iteration, and integer indexing avoids accumulated step
rounding. Current suite: 29 tests.

## Caller-supplied logits

The calibrate API and JSON CLI fit on validation only, then score test. They
never invoke fixture training or compute the test-grid oracle. Tests change
only test labels and verify T remains unchanged. Schema checks reject invalid
values and class dimensions; callers own split independence and class order.
Softmax subtracts the largest logit before division, avoiding positive overflow
when a finite logit is near the float limit and T is below one.
