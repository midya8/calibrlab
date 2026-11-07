"""Measured pipeline on the checked-in fixture:

1. Train a small MLP on 20 samples/class - few enough that it overfits and
   its confidence drifts out of calibration on held-out data.
2. Measure test-set calibration at T=1: ECE, MCE, Brier, NLL, accuracy.
3. Fit a temperature on the VALIDATION split only (never the test split).
4. Re-measure on test at the fitted T. Accuracy must not move: scaling all
   logits by a positive T preserves argmax, so this is exact, not empirical.
5. Report the oracle reference: the T that would minimize TEST NLL, as an
   honest bound on how much the validation fit leaves on the table.
"""
from .data import make_fixture
from .mlp import MLP, train, softmax
from .metrics import accuracy, ece, mce, brier, logits_nll, reliability_table
from .temp import fit_temperature, fit_temperature_oracle


def run_evaluation(seed=13):
    fixture = make_fixture(seed=seed)
    xtr, ytr = fixture["train"]
    xva, yva = fixture["val"]
    xte, yte = fixture["test"]

    model = MLP(fixture["dim"], hidden=24, n_classes=fixture["n_classes"], seed=7)
    train(model, xtr, ytr, epochs=400, lr=0.5)

    train_acc = accuracy([softmax(z) for z in model.predict_logits(xtr)], ytr)
    te_logits = model.predict_logits(xte)
    va_logits = model.predict_logits(xva)
    te_raw = [softmax(z) for z in te_logits]

    t_val, val_nll = fit_temperature(va_logits, yva)
    te_cal = [softmax(z, t_val) for z in te_logits]
    t_oracle, oracle_nll = fit_temperature_oracle(te_logits, yte)

    r = {
        "train_acc": train_acc,
        "test_acc_raw": accuracy(te_raw, yte),
        "test_acc_cal": accuracy(te_cal, yte),
        "ece_raw": ece(te_raw, yte),
        "ece_cal": ece(te_cal, yte),
        "mce_raw": mce(te_raw, yte),
        "mce_cal": mce(te_cal, yte),
        "brier_raw": brier(te_raw, yte),
        "brier_cal": brier(te_cal, yte),
        "nll_raw": logits_nll(te_logits, yte),
        "nll_cal": logits_nll(te_logits, yte, t_val),
        "t_val": t_val,
        "val_nll_at_t": val_nll,
        "t_oracle": t_oracle,
        "test_nll_oracle": oracle_nll,
        "table_raw": reliability_table(te_raw, yte),
        "table_cal": reliability_table(te_cal, yte),
    }
    r["ece_reduction"] = r["ece_raw"] - r["ece_cal"]
    r["acc_drift"] = abs(r["test_acc_cal"] - r["test_acc_raw"])
    return r


def render(r):
    return "\n".join([
        "CalibrLab evaluation (checked-in fixture, 6 classes x dim 16, noise 0.6)",
        "",
        f"Train accuracy:                 {r['train_acc']:.4f}   (overfit by design)",
        f"Test accuracy:                  {r['test_acc_raw']:.4f}",
        "",
        "Uncalibrated (T=1) vs temperature-scaled (T fit on validation):",
        f"  ECE:    {r['ece_raw']:.4f} -> {r['ece_cal']:.4f}   (reduction {r['ece_reduction']:+.4f})",
        f"  MCE:    {r['mce_raw']:.4f} -> {r['mce_cal']:.4f}",
        f"  Brier:  {r['brier_raw']:.4f} -> {r['brier_cal']:.4f}",
        f"  NLL:    {r['nll_raw']:.4f} -> {r['nll_cal']:.4f}",
        f"  Accuracy drift from scaling: {r['acc_drift']:.6f}   (exactly 0 by construction)",
        "",
        f"Fitted temperature (validation):  T = {r['t_val']:.2f}   (val NLL {r['val_nll_at_t']:.4f})",
        f"Oracle temperature (on test):     T = {r['t_oracle']:.2f}   (test NLL {r['test_nll_oracle']:.4f})",
        "The oracle row is an unattainable reference: fitting on test would leak.",
        "",
        "Reliability before:", r["table_raw"], "",
        "Reliability after:", r["table_cal"],
    ])
