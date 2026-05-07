# LIS Test Fixtures

This directory contains synthetic and/or minimized test fixtures used for
parser, view, and compatibility tests in `tools/tests/`.

## Provenance

- These fixtures are **synthetic and/or minimized test fixtures**.
- They are used for **parser/view compatibility tests** only.
- They are **not benchmark evidence**.
- They are **not model quality evidence**.
- They contain **no private data**.
- They contain **no private prompt text**.
- They contain **no confidential generated text**.
- Specific model-family labels, if present (e.g. `qwen3_1p7b`), are
  **compatibility labels only** and do not imply model endorsement or quality
  claims.

## Files

| File | Purpose |
|---|---|
| `qwen3_1p7b_run.{json,stderr,md}` | Minimal baseline — no per-token data, no token ID list, retention-minimized |
| `qwen3_1p7b_run_full.{json,stderr,md}` | Fuller fixture — adds `perf-per-token` lines and explicit token IDs |
| `qwen3_1p7b_run_compare.{json,stderr}` | Second run for two-run compare validation |

All fixtures are schema-valid against `lis.execution_artifact/v1` and are
derived from grounded reference parameters, not captured real production runs.
