"""Validation-fitted temperature scaling for caller-supplied classifier logits.

No model training or test-set temperature search happens in this API. Labels
are integer indices into a common class order shared by both splits.
"""
import math

from .metrics import accuracy, ece, mce, brier, logits_nll
from .mlp import softmax
from .temp import fit_temperature


def _validate_split(split, name):
    if not isinstance(split, dict) or set(split) != {"logits", "labels"}:
        raise ValueError(f"{name} must have exactly logits and labels")
    logits, labels = split["logits"], split["labels"]
    if not isinstance(logits, list) or not isinstance(labels, list) or not logits:
        raise ValueError(f"{name} logits/labels must be nonempty lists")
    if len(logits) != len(labels):
        raise ValueError(f"{name} logits and labels must align")
    if not isinstance(logits[0], list) or len(logits[0]) < 2:
        raise ValueError(f"{name} requires at least two classes")
    width = len(logits[0])
    for row in logits:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError(f"{name} logits must have equal-width rows")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in row):
            raise ValueError(f"{name} logits must be finite numbers")
    if any(type(y) is not int or not 0 <= y < width for y in labels):
        raise ValueError(f"{name} labels must be valid integer class indices")
    return logits, labels, width


def _scores(probs, labels, logits, temperature=1.0):
    return {"accuracy": accuracy(probs, labels), "ece": ece(probs, labels),
            "mce": mce(probs, labels), "brier": brier(probs, labels),
            "nll": logits_nll(logits, labels, temperature)}


def calibrate(payload, lo=.5, hi=10.0, step=.05):
    """Fit only on validation labels and score on separate test labels.

    The caller owns data provenance: this checks schema and class dimensions,
    not whether examples or labels leaked between the supplied splits.
    """
    if not isinstance(payload, dict) or set(payload) != {"validation", "test"}:
        raise ValueError("input must have exactly validation and test splits")
    va, vy, vk = _validate_split(payload["validation"], "validation")
    te, ty, tk = _validate_split(payload["test"], "test")
    if vk != tk:
        raise ValueError("validation and test must share a class dimension/order")
    temperature, validation_nll = fit_temperature(va, vy, lo, hi, step)
    before = [softmax(row) for row in te]
    after = [softmax(row, temperature) for row in te]
    return {
        "temperature": temperature,
        "validation_nll": validation_nll,
        "grid": {"lo": lo, "hi": hi, "step": step},
        "samples": {"validation": len(vy), "test": len(ty)},
        "classes": vk,
        "test_before": _scores(before, ty, te),
        "test_after": _scores(after, ty, te, temperature),
        "prediction_changes": sum(max(range(tk), key=row.__getitem__) !=
                                  max(range(tk), key=scaled.__getitem__)
                                  for row, scaled in zip(before, after)),
        "scope": "Temperature fit on validation only; test used for scoring. "
                 "Caller must ensure disjoint splits and a common class order. "
                 "ECE/MCE use ten equal-width top-confidence bins.",
    }
