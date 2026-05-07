# qwen3_1p7b_run_full fixture

Fuller fixture based on `qwen3_1p7b_run.{json,stderr}`. Adds:

- `selected_token_ids` and `emitted_token_ids` (10 IDs each, identical) so the
  Artifact tab's bounded token preview can render an actual head/tail slice.
- `runtime_settings.perf_per_token_enabled = true` so the Overview reflects
  per-token capture.
- 10 `perf-per-token` lines in the stderr (steps 0..9). Their per-step `ms`
  values were chosen so that:
  - step 0 (`first_decode` warm-up) matches the minimal baseline first_decode value
    (206.777 ms), and
  - steps 1..9 sum to 1853.223 ms with mean 205.914 ms — chosen so that the
    derived `decode_steady_state` total and `itl_ms` match the synthetic
    reference numbers used elsewhere in this fixture set.

| Aspect           | Value                       |
|------------------|-----------------------------|
| Per-token steps  | 10 (0..9)                   |
| Step 0 (ms)      | 206.777                     |
| Mean steps 1..9  | 205.914                     |
| Token ID list    | 10 IDs, head-only preview   |

These values are **synthesized** to match documented reference numbers — they
were not produced by a real LIS execution.
