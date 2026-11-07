"""Seeded paired test-row bootstrap, conditional on an already fitted T."""
import math
import random

from .metrics import accuracy, ece, mce, brier, logits_nll


def _percentile(values, q):
    """Linear interpolation between sorted empirical order statistics."""
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return math.fsum((ordered[lower] * (1 - weight), ordered[upper] * weight))


def paired_bootstrap(logits, labels, before, after, temperature, draws, seed):
    """Internal helper; calibrate() validates inputs and work limits first.

    Each replicate uses identical resampled indices for before and after.
    T is never refitted. Percentiles are descriptive IID row-bootstrap
    intervals; no finite-sample or distribution-free coverage is promised.
    """
    rng = random.Random(seed)
    n = len(labels)
    metrics = {'accuracy': accuracy, 'ece': ece, 'mce': mce, 'brier': brier}
    samples = {name: {'before': [], 'after': [], 'delta_after_minus_before': []}
               for name in (*metrics, 'nll')}
    loss_before = [logits_nll([row], [y]) for row, y in zip(logits, labels)]
    loss_after = [logits_nll([row], [y], temperature) for row, y in zip(logits, labels)]
    for _ in range(draws):
        indices = [rng.randrange(n) for _ in range(n)]
        ys = [labels[i] for i in indices]
        raw = [before[i] for i in indices]
        scaled = [after[i] for i in indices]
        for name, function in metrics.items():
            a, b = function(raw, ys), function(scaled, ys)
            samples[name]['before'].append(a)
            samples[name]['after'].append(b)
            samples[name]['delta_after_minus_before'].append(b - a)
        a = math.fsum(loss_before[i] / n for i in indices)
        b = math.fsum(loss_after[i] / n for i in indices)
        samples['nll']['before'].append(a)
        samples['nll']['after'].append(b)
        samples['nll']['delta_after_minus_before'].append(b - a)
    intervals = {name: {kind: [_percentile(values, .025), _percentile(values, .975)]
                        for kind, values in kinds.items()}
                 for name, kinds in samples.items()}
    return {
        'method': 'paired percentile bootstrap, linear empirical quantiles',
        'nominal_level': .95, 'draws': draws, 'seed': seed,
        'intervals': intervals,
        'scope': 'IID test-row resampling only, conditional on fixed fitted T. '
                 'No validation refit, training uncertainty, or coverage guarantee. '
                 'Correlated/grouped rows need a different sampling design; '
                 'small samples and binned ECE/MCE can give unreliable intervals. '
                 'Deltas are after minus before; negative loss/error delta is better.'
    }
