#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== evaluation ==="
python -m calibrlab evaluate

echo
echo "=== gates (achievable set: must pass) ==="
python -m calibrlab gates --min-ece-reduction 0.10 --max-nll-after 1.40 --max-oracle-gap 0.05

echo
echo "=== gates (impossible ECE reduction: must fail) ==="
if python -m calibrlab gates --min-ece-reduction 0.9; then
    echo "ERROR: impossible gate set passed"
    exit 1
else
    echo "impossible gate set failed as expected"
fi

echo
echo "=== caller-supplied logits (separate validation/test) ==="
python -m calibrlab calibrate --input examples/logits.json --bootstrap 200 --seed 13
