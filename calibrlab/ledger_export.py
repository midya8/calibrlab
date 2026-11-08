"""Reproducible native/browser ledger witnesses, no fixture-specific arithmetic."""
import json
import math
from pathlib import Path
from .ledger import ledger
from .ledger_fixtures import fixtures, TEMPERATURES


def verify(actual, expected, path="root"):
    if type(actual) != type(expected) and not (type(actual) in (int, float) and type(expected) in (int, float)):
        raise AssertionError(f"{path}: type differs")
    if type(expected) is float:
        if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"{path}: {actual} != {expected}")
    elif type(expected) is int:
        if type(actual) is not int or actual != expected:
            raise AssertionError(f"{path}: integer differs")
    elif type(expected) is dict:
        if set(actual) != set(expected):
            raise AssertionError(f"{path}: keys differ")
        for k in expected:
            verify(actual[k], expected[k], f"{path}.{k}")
    elif type(expected) is list:
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: length differs")
        for i, (a, b) in enumerate(zip(actual, expected)):
            verify(a, b, f"{path}[{i}]")
    elif actual != expected:
        raise AssertionError(f"{path}: exact value differs")


def project(report):
    return {k: v for k, v in report.items() if k not in ("input", "scope", "path")}


def export(root=None):
    root = Path(root) if root else Path(__file__).with_name("ledger_data")
    data = fixtures()
    for i, t in enumerate(TEMPERATURES):
        witnesses = [{"case": c["id"], "temperature": t, "witness": project(ledger(c["payload"], t))}
                     for c in data["cases"]]
        (root / f"score-{i:02}.json").write_text(json.dumps(witnesses, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    return 95


if __name__ == "__main__":
    print(f"{export()} score witnesses exported")
