"""Temperature scaling: fit a single scalar T by minimizing negative
log-likelihood on held-out validation data. Scaling logits by T recalibrates
confidence without changing argmax, so accuracy is invariant by construction."""
import math

from .mlp import softmax
from .metrics import nll


def fit_temperature(logits_list, ys, lo=0.5, hi=10.0, step=0.05):
    """Grid search T in [lo, hi] minimizing validation NLL. Returns (T, nll)."""
    if not all(math.isfinite(v) for v in (lo, hi, step)):
        raise ValueError("grid bounds and step must be finite")
    if lo <= 0 or hi < lo or step <= 0:
        raise ValueError("require 0 < lo <= hi and step > 0")
    if not logits_list or len(logits_list) != len(ys):
        raise ValueError("logits and labels must be nonempty and aligned")
    width = len(logits_list[0])
    if not width or any(len(z) != width or not all(math.isfinite(v) for v in z)
                        for z in logits_list):
        raise ValueError("logits must be finite, nonempty, equal-width rows")
    if any(not isinstance(y, int) or not 0 <= y < width for y in ys):
        raise ValueError("labels must be valid integer class indices")
    best_t, best_n = None, float("inf")
    count = int(math.floor((hi - lo) / step + 1e-9)) + 1
    if count > 100000:
        raise ValueError("grid exceeds 100000 candidates")
    for index in range(count):
        t = lo + index * step
        probs = [softmax(z, t) for z in logits_list]
        cur = nll(probs, ys)
        if cur < best_n - 1e-12:
            best_t, best_n = t, cur
    return best_t, best_n


def fit_temperature_oracle(logits_list, ys, lo=0.5, hi=10.0, step=0.05):
    """The same fit on TEST data: the unattainable reference point, reported
    to show how close validation-fitted T lands to the test-optimal T."""
    return fit_temperature(logits_list, ys, lo, hi, step)
