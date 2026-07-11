# SHD temporal-horizon screening

This directory contains the small, curated result artifacts from the legacy
seed-42 sweep. Checkpoints, logs, generated configs, profiler output, and dataset
files are intentionally excluded from Git.

## Current coverage

| T | C3 minus strongest C1/C2 | coverage |
|---:|---:|---|
| 25 | +0.4108 pp | C0-C3, seed 42 only |
| 50 | -1.7667 pp | C0-C3, seed 42 only |
| 100 | +0.6846 pp | C0-C3, seed 42 only |
| 200 | unavailable | C0 only; C1 incomplete, C2/C3 OOM |

These results are screening evidence, not confirmatory evidence:

- The old data pipeline used the official SHD test split as validation.
- Random temporal shift was active during validation/test.
- C3 used batch 64 at T=25 and batch 160 at T=50 while the baselines used 128.
- Every completed condition has one seed, so a 95% confidence interval and a
  PASS/PIVOT/FAIL verdict are unavailable.
- The stored C3 gate-parameter metadata was corrected from 1,048 to 1,304 to
  include the two LayerNorm affine parameter sets. Accuracy metrics were not
  changed.

Do not append corrected-protocol results to this directory. Use a new result
root and the MGA-v2 protocol in `docs/MGA_V2_SPEC.md`.
