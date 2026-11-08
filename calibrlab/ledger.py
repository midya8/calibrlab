"""Bounded descriptive score/bin ledger. Standard scores, not a calibrator."""
import math
from .calibrate import _validate_split
from .temp import fit_temperature
from .mlp import softmax
from .metrics import logits_nll

SCOPE = ("Temperature is selected only on validation NLL using the original 191-point "
         "0.5..10 step 0.05 grid (first strict improvement beyond 1e-12). Test paths "
         "and fixed-temperature row deletions are descriptive, not tuning, refitting, "
         "uncertainty, causality or deployment guarantees. Ten-bin original scores "
         "are retained separately. Top-confidence ECE/MCE are bin-dependent finite-sample "
         "summaries, not classwise calibration tests; Brier/NLL are proper-score losses, "
         "not calibration-only measures. Caller owns split provenance/class order. "
         "Positive scaling preserves real-arithmetic argmax; reported predictions use "
         "first-maximum original logits. Probabilities and losses use binary64; "
         "underflow, rounded ties and numeric bin boundaries remain possible.")


def validate(payload):
    if not isinstance(payload, dict) or set(payload) != {"validation", "test"}:
        raise ValueError("input must have exactly validation and test splits")
    result = {}
    widths = []
    for name in ("validation", "test"):
        z, y, width = _validate_split(payload[name], name)
        if len(y) > 512 or width > 16:
            raise ValueError("ledger limit: 512 rows per split, 2..16 classes")
        if any(abs(v) > 10000 for row in z for v in row):
            raise ValueError("ledger logits must be within [-10000,10000]")
        result[name] = (z, y)
        widths.append(width)
    if widths[0] != widths[1]:
        raise ValueError("splits must share class dimensions/order")
    return result, widths[0]


def bin_index(confidence, bins):
    # Match original [lo,hi), last bin including 1, rather than floor(c*bins).
    for b in range(bins):
        if b / bins <= confidence < (b + 1) / bins or b == bins - 1 and confidence == 1:
            return b
    raise ValueError("unrepresentable confidence bin")


def _summary(rows, bins):
    n = len(rows)
    if not n:
        return None
    groups = [[r for r in rows if r["bin"] == b] for b in range(bins)]
    table = []
    for b, group in enumerate(groups):
        count = len(group)
        conf = math.fsum(r["confidence"] for r in group)
        correct = sum(r["correct"] for r in group)
        signed = (conf - correct) / count if count else None
        table.append({"index": b, "lower": b / bins, "upper": (b + 1) / bins,
                      "upper_inclusive": b == bins - 1, "count": count,
                      "positions": [r["position"] for r in group],
                      "confidence_sum": conf, "correct_sum": correct,
                      "mean_confidence": conf / count if count else None,
                      "accuracy": correct / count if count else None,
                      "signed_gap": signed,
                      "ece_contribution": abs(conf - correct) / n})
    return {"samples": n, "accuracy": sum(r["correct"] for r in rows) / n,
            "nll": math.fsum(r["nll"] / n for r in rows),
            "brier": math.fsum(r["brier"] / n for r in rows),
            "ece": math.fsum(b["ece_contribution"] for b in table),
            "mce": max(abs(b["signed_gap"]) for b in table if b["count"]),
            "bins": table}


def _rows(logits, labels, temperature, bins):
    rows = []
    for i, (z, y) in enumerate(zip(logits, labels)):
        p = softmax(z, temperature)
        pred = max(range(len(z)), key=z.__getitem__)
        terms = [(v - int(k == y)) ** 2 for k, v in enumerate(p)]
        rows.append({"position": i, "label": y, "prediction": pred,
                     "correct": int(pred == y), "probabilities": p,
                     "confidence": p[pred], "bin": bin_index(p[pred], bins),
                     "nll": logits_nll([z], [y], temperature),
                     "brier_terms": terms, "brier": sum(terms)})
    return rows


def _deletions(rows, summary):
    # Hold logits and temperature fixed. Only the affected bin changes.
    n = len(rows)
    out = []
    for r in rows:
        if n == 1:
            out.append({"position": r["position"], "remaining": 0, "scores": None})
            continue
        contributions, gaps = [], []
        for b in summary["bins"]:
            hit = b["index"] == r["bin"]
            count = b["count"] - int(hit)
            conf = b["confidence_sum"] - (r["confidence"] if hit else 0)
            correct = b["correct_sum"] - (r["correct"] if hit else 0)
            if count:
                contributions.append(abs(conf - correct) / (n - 1))
                gaps.append(abs(conf - correct) / count)
        out.append({"position": r["position"], "remaining": n - 1,
                    "scores": {"accuracy": (sum(x["correct"] for x in rows) - r["correct"]) / (n - 1),
                               "nll": math.fsum(x["nll"] / (n - 1) for x in rows if x is not r),
                               "brier": math.fsum(x["brier"] / (n - 1) for x in rows if x is not r),
                               "ece": math.fsum(contributions), "mce": max(gaps)}})
    return out


def ledger(payload, temperature=1.0, bins=10):
    data, classes = validate(payload)
    if type(temperature) not in (float, int) or not math.isfinite(temperature) or not .1 <= temperature <= 20:
        raise ValueError("temperature must be finite in [0.1,20]")
    if type(bins) is not int or not 2 <= bins <= 20:
        raise ValueError("bins must be an integer from 2 to 20")
    va, vy = data["validation"]
    te, ty = data["test"]
    fitted, validation_nll = fit_temperature(va, vy)
    before, after = _rows(te, ty, 1, bins), _rows(te, ty, temperature, bins)
    raw, selected = _summary(before, bins), _summary(after, bins)
    migration = [[0 for _ in range(bins)] for _ in range(bins)]
    for r, s in zip(before, after):
        migration[r["bin"]][s["bin"]] += 1
    return {"schema": "calibrlab-score-ledger-v1", "classes": classes,
            "input": payload, "temperature": temperature, "bins": bins,
            "fit": {"temperature": fitted, "validation_nll": validation_nll,
                    "samples": len(vy), "grid": {"lo": .5, "hi": 10., "step": .05, "count": 191}},
            "validation_nll_at_selected": logits_nll(va, vy, temperature),
            "rows": [{"before": r, "selected": s, "nll_delta": s["nll"] - r["nll"],
                      "brier_delta": s["brier"] - r["brier"]} for r, s in zip(before, after)],
            "before": raw, "selected": selected,
            "delta": {k: selected[k] - raw[k] for k in ("accuracy", "nll", "brier", "ece", "mce")},
            "migration": migration, "deletions": _deletions(after, selected),
            "path": temperature_path(data, bins), "scope": SCOPE}


def temperature_path(data, bins):
    from .ledger_fixtures import TEMPERATURES
    va, vy = data["validation"]
    te, ty = data["test"]
    out = []
    for t in TEMPERATURES:
        scores = _summary(_rows(te, ty, t, bins), bins)
        out.append({"temperature": t, "validation_nll": logits_nll(va, vy, t),
                    "test": {k: scores[k] for k in ("accuracy", "nll", "brier", "ece", "mce")}})
    return out
