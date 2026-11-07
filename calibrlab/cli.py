"""Command line: evaluate prints the measured report; gates assert on it."""
import argparse
import json
import sys

from .eval import run_evaluation, render


def main(argv=None):
    parser = argparse.ArgumentParser(prog="calibrlab")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("evaluate")
    g = sub.add_parser("gates")
    g.add_argument("--min-ece-reduction", type=float, default=None)
    g.add_argument("--max-nll-after", type=float, default=None)
    g.add_argument("--max-oracle-gap", type=float, default=None,
                   help="max |test NLL at fitted T - test NLL at oracle T|")
    c = sub.add_parser("calibrate", help="fit on supplied validation logits and score test")
    c.add_argument("--input", required=True, help="JSON file, or - for stdin")
    c.add_argument("--lo", type=float, default=.5)
    c.add_argument("--hi", type=float, default=10.0)
    c.add_argument("--step", type=float, default=.05)
    args = parser.parse_args(argv)

    if args.cmd == "calibrate":
        from .calibrate import calibrate
        try:
            if args.input == "-":
                payload = json.load(sys.stdin)
            else:
                with open(args.input, encoding="utf-8") as source:
                    payload = json.load(source)
            report = calibrate(payload, args.lo, args.hi, args.step)
        except (OSError, ValueError, TypeError, OverflowError) as exc:
            print(f"calibration input error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
        return 0

    r = run_evaluation()
    if args.cmd == "evaluate":
        print(render(r))
        return 0

    failures = []
    if args.min_ece_reduction is not None and r["ece_reduction"] < args.min_ece_reduction:
        failures.append(f"ece_reduction {r['ece_reduction']:.4f} < {args.min_ece_reduction}")
    if r["acc_drift"] != 0.0:
        failures.append(f"accuracy drifted {r['acc_drift']} (must be exactly 0)")
    if args.max_nll_after is not None and r["nll_cal"] > args.max_nll_after:
        failures.append(f"calibrated NLL {r['nll_cal']:.4f} > {args.max_nll_after}")
    if args.max_oracle_gap is not None:
        gap = abs(r["nll_cal"] - r["test_nll_oracle"])
        if gap > args.max_oracle_gap:
            failures.append(f"oracle gap {gap:.4f} > {args.max_oracle_gap}")
    if failures:
        print("GATES FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("all gates passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
