"""Deterministic synthetic classification fixture.

Six classes in 16 dimensions: each class has a random center, and a sample is
its center plus isotropic Gaussian noise. Class overlap is real (noise 0.6 on
unit-spaced centers), so Bayes accuracy is well below 1.0 and a trained model
has genuine uncertainty to calibrate - not a toy where everything is separable.
"""
import math
import random

DIM = 16
N_CLASSES = 6
NOISE = 0.6


def _centers(rng):
    centers = []
    for _ in range(N_CLASSES):
        v = [rng.gauss(0.0, 1.0) for _ in range(DIM)]
        n = math.sqrt(sum(x * x for x in v))
        centers.append([x / n for x in v])
    return centers


def make_split(rng, centers, n_per_class):
    xs, ys = [], []
    for c in range(N_CLASSES):
        for _ in range(n_per_class):
            xs.append([centers[c][d] + NOISE * rng.gauss(0.0, 1.0) for d in range(DIM)])
            ys.append(c)
    order = list(range(len(xs)))
    rng.shuffle(order)
    return [xs[i] for i in order], [ys[i] for i in order]


def make_fixture(seed=13, train_per_class=20, val_per_class=40, test_per_class=80):
    """Small train split (20/class) so the model overfits and its confidence
    drifts out of calibration; val for fitting temperature, test untouched."""
    rng = random.Random(seed)
    centers = _centers(rng)
    xtr, ytr = make_split(rng, centers, train_per_class)
    xva, yva = make_split(rng, centers, val_per_class)
    xte, yte = make_split(rng, centers, test_per_class)
    return {"train": (xtr, ytr), "val": (xva, yva), "test": (xte, yte),
            "dim": DIM, "n_classes": N_CLASSES}
