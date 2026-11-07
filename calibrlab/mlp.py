"""A one-hidden-layer MLP with hand-derived backprop. No ML frameworks."""
import math
import random


def softmax(logits, temperature=1.0):
    if not logits or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("logits must be nonempty and temperature finite/positive")
    if not all(math.isfinite(v) for v in logits):
        raise ValueError("logits must be finite")
    m = max(logits)
    # Subtract before dividing to avoid positive overflow at small T.
    shifted = [(v - m) / temperature for v in logits]
    if temperature >= 1 and any(not math.isfinite(v) for v in shifted):
        shifted = [v / temperature - m / temperature for v in logits]
    e = [math.exp(v) for v in shifted]
    s = sum(e)
    return [v / s for v in e]


class MLP:
    def __init__(self, dim, hidden, n_classes, seed=7):
        rng = random.Random(seed)
        self.w1 = [[rng.gauss(0.0, 1.0 / math.sqrt(dim)) for _ in range(dim)] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.w2 = [[rng.gauss(0.0, 1.0 / math.sqrt(hidden)) for _ in range(hidden)] for _ in range(n_classes)]
        self.b2 = [0.0] * n_classes

    def forward(self, x):
        h = [math.tanh(sum(w * xi for w, xi in zip(row, x)) + b)
             for row, b in zip(self.w1, self.b1)]
        logits = [sum(w * hi for w, hi in zip(row, h)) + b
                  for row, b in zip(self.w2, self.b2)]
        return h, logits

    def predict_logits(self, xs):
        return [self.forward(x)[1] for x in xs]


def cross_entropy(probs, y):
    return -math.log(max(probs[y], 1e-12))


def loss_and_grads(model, xs, ys):
    """Mean softmax cross-entropy and exact analytic gradients."""
    n = len(xs)
    hidden, n_classes = len(model.w1), len(model.w2)
    gw1 = [[0.0] * len(model.w1[0]) for _ in range(hidden)]
    gb1 = [0.0] * hidden
    gw2 = [[0.0] * hidden for _ in range(n_classes)]
    gb2 = [0.0] * n_classes
    total = 0.0
    for x, y in zip(xs, ys):
        h, logits = model.forward(x)
        p = softmax(logits)
        total += cross_entropy(p, y)
        dlogits = p[:]
        dlogits[y] -= 1.0
        for c in range(n_classes):
            for j in range(hidden):
                gw2[c][j] += dlogits[c] * h[j] / n
            gb2[c] += dlogits[c] / n
        dh = [sum(dlogits[c] * model.w2[c][j] for c in range(n_classes)) for j in range(hidden)]
        for j in range(hidden):
            d = dh[j] * (1.0 - h[j] * h[j]) / n
            gb1[j] += d
            for d_in in range(len(x)):
                gw1[j][d_in] += d * x[d_in]
    return total / n, (gw1, gb1, gw2, gb2)


def sgd_step(model, grads, lr):
    gw1, gb1, gw2, gb2 = grads
    for j in range(len(model.w1)):
        for d in range(len(model.w1[0])):
            model.w1[j][d] -= lr * gw1[j][d]
        model.b1[j] -= lr * gb1[j]
    for c in range(len(model.w2)):
        for j in range(len(model.w2[0])):
            model.w2[c][j] -= lr * gw2[c][j]
        model.b2[c] -= lr * gb2[c]


def train(model, xs, ys, epochs=400, lr=0.5):
    for _ in range(epochs):
        _, grads = loss_and_grads(model, xs, ys)
        sgd_step(model, grads, lr)
    return model


def finite_difference_check(model, xs, ys, eps=1e-5, param_cap=40):
    """Max gradient error with coverage across all four parameter groups.

    Round-robin sampling prevents a cap from testing only the first matrix.
    param_cap=None checks every parameter. Perturbations restore exact saved
    values, including when a loss computation raises.
    """
    if eps <= 0 or not math.isfinite(eps):
        raise ValueError("eps must be finite and positive")
    if param_cap is not None and param_cap < 4:
        raise ValueError("param_cap must be at least 4 or None")
    _, (gw1, gb1, gw2, gb2) = loss_and_grads(model, xs, ys)
    groups = [
        [(model.w1, gw1, i, j) for i in range(len(gw1)) for j in range(len(gw1[0]))],
        [(model.b1, gb1, i, None) for i in range(len(gb1))],
        [(model.w2, gw2, i, j) for i in range(len(gw2)) for j in range(len(gw2[0]))],
        [(model.b2, gb2, i, None) for i in range(len(gb2))],
    ]
    selected = [group[k] for k in range(max(map(len, groups)))
                for group in groups if k < len(group)]
    if param_cap is not None:
        selected = selected[:param_cap]
    worst = 0.0
    for ref, grad, i, j in selected:
        row, idx = (ref, i) if j is None else (ref[i], j)
        ana = grad[i] if j is None else grad[i][j]
        saved = row[idx]
        try:
            row[idx] = saved + eps
            lp, _ = loss_and_grads(model, xs, ys)
            row[idx] = saved - eps
            lm, _ = loss_and_grads(model, xs, ys)
        finally:
            row[idx] = saved
        worst = max(worst, abs(ana - (lp - lm) / (2 * eps)))
    return worst
