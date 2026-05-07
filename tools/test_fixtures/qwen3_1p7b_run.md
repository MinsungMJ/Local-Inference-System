# qwen3_1p7b_run fixture

Synthetic but schema-valid fixture used to exercise the LIS Inspect parser
and derivation logic.

These values are **synthesized** — they were not produced by a real LIS
execution. They exist only to drive UI layout development and unit-test
parsing/derivation logic.

| Aspect           | Value                       |
|------------------|-----------------------------|
| Model family     | `qwen3_dense_decoder`       |
| Backend          | `avx2`                      |
| Context / batch  | 128 / 1                     |
| Generation limit | 10                          |
| Threads          | 1                           |
| Status           | OK                          |
| Stop reason      | `decode_limit`              |
| Dominant stage   | `decode_steady_state`       |
| Run profile      | decode-heavy                |

Files:

- `qwen3_1p7b_run.json` — `lis.execution_artifact/v1` `run_report`
- `qwen3_1p7b_run.stderr` — `--perf` stderr capture (no `--perf-per-token`)
- `qwen3_1p7b_run.md` — this description

Replace with a real captured run as soon as one is available. The synthetic
fixture must continue to round-trip through the parser and produce the same
derived metrics.
