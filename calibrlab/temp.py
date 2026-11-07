"""Temperature scaling: fit a single scalar T by minimizing negative
log-likelihood on held-out validation data. Scaling logits by T recalibrates
confidence without changing argmax, so accuracy is invariant by construction."""
from .mlp import softmax
from .metrics import nll


def fit_temperature(logits_list, ys, lo=0.5, hi=10.0, step=0.05):
    """Grid search T in [lo, hi] minimizing validation NLL. Returns (T, nll)."""
    best_t, best_n = None, float("inf")
    t = lo
    while t <= hi + 1e-9:
        probs = [softmax(z, t) for z in logits_list]
        cur = nll(probs, ys)
        if cur < best_n - 1e-12:
            best_t, best_n = t, cur
        t += step
    return best_t, best_n


def fit_temperature_oracle(logits_list, ys, lo=0.5, hi=10.0, step=0.05):
    """The same fit on TEST data: the unattainable reference point, reported
    to show how close validation-fitted T lands to the test-optimal T."""
    return fit_temperature(logits_list, ys, lo, hi, step)
