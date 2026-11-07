"""Calibration metrics: accuracy, ECE, MCE, Brier score, NLL, and an ASCII
reliability table. All computed from probability vectors and true labels."""
import math


def accuracy(probs_list, ys):
    correct = sum(1 for p, y in zip(probs_list, ys) if max(range(len(p)), key=lambda i: p[i]) == y)
    return correct / len(ys)


def _conf_correct(probs_list, ys):
    pairs = []
    for p, y in zip(probs_list, ys):
        k = max(range(len(p)), key=lambda i: p[i])
        pairs.append((p[k], 1.0 if k == y else 0.0))
    return pairs


def ece(probs_list, ys, n_bins=10):
    """Expected Calibration Error with equal-width bins over [0, 1]."""
    pairs = _conf_correct(probs_list, ys)
    n = len(pairs)
    total = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        bin_pairs = [(c, a) for c, a in pairs if lo <= c < hi or (b == n_bins - 1 and c == 1.0)]
        if not bin_pairs:
            continue
        conf = sum(c for c, _ in bin_pairs) / len(bin_pairs)
        acc = sum(a for _, a in bin_pairs) / len(bin_pairs)
        total += len(bin_pairs) / n * abs(acc - conf)
    return total


def mce(probs_list, ys, n_bins=10):
    """Maximum Calibration Error across occupied equal-width bins."""
    pairs = _conf_correct(probs_list, ys)
    worst = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        bin_pairs = [(c, a) for c, a in pairs if lo <= c < hi or (b == n_bins - 1 and c == 1.0)]
        if not bin_pairs:
            continue
        conf = sum(c for c, _ in bin_pairs) / len(bin_pairs)
        acc = sum(a for _, a in bin_pairs) / len(bin_pairs)
        worst = max(worst, abs(acc - conf))
    return worst


def brier(probs_list, ys):
    """Multiclass Brier score: mean squared error of the probability vector."""
    total = 0.0
    for p, y in zip(probs_list, ys):
        total += sum((pi - (1.0 if i == y else 0.0)) ** 2 for i, pi in enumerate(p))
    return total / len(probs_list)


def nll(probs_list, ys):
    """Probability-only NLL, clipped at 1e-12; use logits_nll for fitting."""
    total = 0.0
    for p, y in zip(probs_list, ys):
        total -= math.log(max(p[y], 1e-12))
    return total / len(probs_list)


def reliability_table(probs_list, ys, n_bins=10):
    """ASCII reliability table: bin range, count, mean confidence, accuracy."""
    pairs = _conf_correct(probs_list, ys)
    lines = ["  confidence    n    mean_conf  accuracy", "  " + "-" * 44]
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        bin_pairs = [(c, a) for c, a in pairs if lo <= c < hi or (b == n_bins - 1 and c == 1.0)]
        if not bin_pairs:
            lines.append(f"  [{lo:.1f}, {hi:.1f})    0")
            continue
        conf = sum(c for c, _ in bin_pairs) / len(bin_pairs)
        acc = sum(a for _, a in bin_pairs) / len(bin_pairs)
        gap = "OVER" if conf > acc else "under"
        lines.append(f"  [{lo:.1f}, {hi:.1f}) {len(bin_pairs):4d}    {conf:.4f}     {acc:.4f}  {gap}")
    return "\n".join(lines)


def logits_nll(logits_list, ys, temperature=1.0):
    """Stable mean cross-entropy without probability clipping.

    Rejects losses that cannot be represented as finite Python floats rather
    than silently capping the objective or emitting nonstandard JSON Infinity.
    """
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not logits_list or len(logits_list) != len(ys):
        raise ValueError("logits and labels must be nonempty and aligned")
    width = len(logits_list[0])
    if width < 2:
        raise ValueError("at least two classes required")
    losses = []
    for row, y in zip(logits_list, ys):
        if len(row) != width or any(type(v) not in (int, float) or not math.isfinite(v) for v in row):
            raise ValueError("logits must be finite, equal-width rows")
        if type(y) is not int or not 0 <= y < width:
            raise ValueError("labels must be valid integer class indices")
        peak = max(row)
        shifted = [(v - peak) / temperature for v in row]
        gap = (peak - row[y]) / temperature
        # Opposite-sign subtraction can overflow even if division first is safe.
        if not math.isfinite(gap) and temperature >= 1:
            shifted = [v / temperature - peak / temperature for v in row]
            gap = peak / temperature - row[y] / temperature
        loss = gap + math.log(math.fsum(math.exp(v) for v in shifted))
        if not math.isfinite(loss):
            raise ValueError("NLL exceeds finite float range; rescale logits or grid")
        losses.append(loss / len(ys))
    try:
        result = math.fsum(losses)
    except OverflowError as exc:
        raise ValueError("mean NLL exceeds finite float range") from exc
    if not math.isfinite(result):
        raise ValueError("mean NLL exceeds finite float range")
    return result
