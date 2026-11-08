# Public synthetic score-ledger fixtures

`cases.json` contains the original240validation/480test model logits and unchanged
report, plus four small arithmetic controls. No private data. The original
trainer can regenerate all data with `ledger_fixtures.build_bundle()`.

`score-00.json` through `score-18.json` hold95complete native report witnesses
for the19sampled temperatures, excluding only input/scope/path to avoid repeating
fixed metadata. `python -m calibrlab.ledger_export` rebuilds them. Tests check
strict structure/types/counts and1e-12finite float agreement, not exact real
arithmetic. See the repository's LEDGER.md for all limits and source attribution.
