# Reproducibility and Execution Artifacts

LIS provides one bounded, versioned execution-artifact contract:

- CLI runs may request `--report-json PATH`.
- The emitted schema string is `lis.execution_artifact/v1`.
- The core object kind is `run_report`.
- Verification and benchmark snapshots reuse that same core run report instead of defining disconnected schema families.

## Core Run Report

Each run report contains:

- `manifest`
  - retention policy (`absolute_paths`, `raw_prompt_text`, `generated_text` all omitted by default)
  - binary fingerprint
  - model format/family plus concrete model-artifact fingerprint
  - config fingerprint
  - tokenizer-or-token-input fingerprint and input mode
  - runtime settings plus runtime fingerprint
  - resolved backend name plus backend fingerprint
- `report`
  - `execution_status`
  - `status_code`
  - `stop_reason`
  - `output_mode`
  - per-sequence prompt token counts plus token-ID digests
  - `selected_token_ids` and digest
  - `emitted_token_ids` and digest
  - deterministic KV cache accounting under `report.kv_cache`
  - optional bounded perf summary when `--perf` is enabled, emitted under
    `report.perf`

The prompt identity is bounded and deterministic: LIS records token counts and
token-ID digests, not raw prompt text. The generated-text payload is also
omitted by default; emitted token IDs are the canonical deterministic output
surface.

`report.perf` is the canonical JSON key for the bounded perf object. It must
not be renamed to `report.perf_summary` during compatibility work.

## KV Cache Accounting

`report.kv_cache` is part of the existing `kind:"run_report"` artifact.
The schema string remains `lis.execution_artifact/v1`; no new artifact kind or
schema string is introduced. Existing `run_report` artifacts without
`report.kv_cache` remain valid compatibility inputs.

The KV block is deterministic structural accounting derived from the current
runtime KV layout and logical batch positions:

```json
"kv_cache": {
  "scope": "run_local",
  "policy": {
    "eviction_free": true,
    "monotonic_growth": true,
    "paging": false,
    "offload": false,
    "sliding_window": false,
    "prefix_reuse": false
  },
  "storage_dtype": "f32",
  "max_tokens": 128,
  "used_tokens": 12,
  "bytes_per_token": 64,
  "allocated_bytes": 8192,
  "used_bytes": 768,
  "shape": {
    "layer_count": 1,
    "batch_size": 2,
    "kv_head_count": 1,
    "head_dim": 4,
    "element_size": 4
  }
}
```

Accounting formulas:

- `bytes_per_token = layer_count * batch_size * 2 * kv_head_count * head_dim * element_size`
- `allocated_bytes = bytes_per_token * max_tokens`
- `used_bytes = bytes_per_token * used_tokens`
- `max_tokens = lis_kv_cache_layout.context_length`
- `storage_dtype = lis_kv_cache_layout.dtype`

`used_tokens` is logical populated token positions, not per-layer write count.
For the current static-batch runtime report it is derived from the maximum
runtime batch position at artifact emission time.

Compatibility note: LIS Inspect accepts and preserves `report.kv_cache` in
the raw payload, but current Inspect support does not add rendering,
interpretation, comparison, graphing, or visualization for KV cache data.

## Human-Readable Markdown Companion

`--report-md PATH` writes a bounded human-readable Markdown companion report
with the same identity and boundedness policy as the JSON artifact. The Markdown
report is organized by reviewer priority and is intended for terminal reading
and GitHub-style Markdown rendering.

Sections include:

- Outcome — status (`OK`/`error`) and stop reason
- Identity — schema, kind, model format/family, backend
- Retention Policy — omission notes by default
- Runtime — configured context, batch size, generation limit, threads, diagnostics/perf flags
- KV Cache — scope, policy, storage dtype, max/used tokens, structural byte
  accounting, and shape summary
- Token Accounting — prompt sequence count and digests, selected/emitted token counts with bounded ID previews (up to 16 IDs in backticks, remainder as count) and `fnv1a64:`-prefixed digests
- Performance Summary (when `--perf` enabled) — TTFT, mean ITL, steady-state and end-to-end TPS, plus a per-stage timing table
- Fingerprints — `fnv1a64:`-prefixed binary, model, config, input, runtime, and backend digests
- Notes — reminder that JSON is canonical; that token IDs and digests are retained; and that raw text and paths are omitted by default

The Markdown companion does not replace the JSON artifact. The JSON remains the
canonical machine-readable source of truth. Markdown is emitted only when
requested and does not require `--report-json`.

## Fail-Closed Behavior

Artifact requests are fail-closed:

- if required identity cannot be captured, LIS reports an artifact error
  explicitly instead of silently emitting a partial JSON file
- if report writing fails, the CLI reports the artifact failure
- unsupported combinations are rejected explicitly; current LIS rejects
  `--report-json` together with `--forced-prefix`

This contract is intentionally narrow. It does not add:

- per-layer trace output
- logits dumps
- activation dumps
- telemetry upload or a dashboard
- raw prompt retention by default
- raw generated-text retention by default

## Snapshot Reuse

The core run report is reused in:

- `tests/verification/token_parity_snapshot.json`
- `tests/verification/qwen3_sanity_snapshot.json`
- `tests/perf/results.json`
- `tests/verification/perf_smoke.json`

These wrapper files keep the same schema string and embed one or more complete
core run reports together with bounded mode-specific comparison or summary
fields.

## Reproducibility Boundary

Execution artifacts identify the exact resolved execution path that ran:

- concrete local binary
- concrete model artifact
- concrete config and tokenizer-or-token-input artifact
- configured context/batch/generate/thread settings
- resolved backend/path

They do not imply cross-backend or cross-thread-count bitwise equivalence unless
that equivalence is separately validated and documented.

## Markdown Companion Sample Excerpt

```markdown
# LIS Execution Report

> This Markdown report is a human-readable companion. The JSON artifact remains the canonical machine-readable source of truth.

## Outcome

- Status: OK
- Stop reason: decode_limit

## Identity

- Schema: `lis.execution_artifact/v1`
- Kind: `run_report`
- Model format: safetensors
- Model family: llama3_decoder
- Backend: avx2

## Runtime

- Configured context: 512
- Batch size: 1
- Generation limit: 16
- Thread count: 4
- Diagnostics: disabled
- Perf: enabled

## Token Accounting

- Prompt sequences: 1
  - seq 1: 18 token(s), digest `fnv1a64:a1b2c3d4e5f60718`
- Selected tokens: 16
  - digest: `fnv1a64:1234567890abcdef`
  - IDs: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15`
- Emitted tokens: 16
  - digest: `fnv1a64:fedcba0987654321`
  - IDs: `0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15`

## Fingerprints

- Binary: `fnv1a64:abc123def4567890` (123456 bytes)
- Model: `fnv1a64:1112223334445556` (9876543 bytes)
- Config: `fnv1a64:9998887776665554` (1024 bytes)
- Input: `fnv1a64:aaaabbbbccccddd` (32 bytes)
- Runtime: `fnv1a64:1a2b3c4d5e6f7a8b`
- Backend: `fnv1a64:0f1e2d3c4b5a6978`
```

## Decode-Step Trace Artifact

LIS supports a bounded `decode_trace` artifact emitted via `--trace-json PATH`:

- The schema string is `lis.execution_artifact/v1`, the same as `run_report`.
- The kind is `decode_trace`.
- The artifact contains the same `manifest` object as a `run_report` (binary, model, config, input, runtime, and backend identity through bounded fingerprints).
- The artifact carries a `decode_trace` array instead of a `report` object.

### Decode-Trace Entries

Each entry in the `decode_trace` array contains:

- `step` — zero-based decode step index
- `phase` — `first_decode` (step 0) or `decode` (step > 0)
- `selected_token_id` — greedy-selected token for this step
- `raw_score_selected` — logit score before penalty or suppression
- `adjusted_score_selected` — effective score after penalty and suppression
- `runner_up_token_id` — second-highest adjusted-score token (`null` when no runner-up exists)
- `runner_up_adjusted_score` — adjusted score of the runner-up (`null` when no runner-up exists)
- `decision_margin` — `adjusted_score_selected - runner_up_adjusted_score` (`null` when no runner-up exists)
- `structural_suppression_affected` — whether structural-token suppression affected any candidate
- `repetition_penalty_changed_selection` — whether the repetition penalty changed the greedy winner
- `selected_token_penalized` — whether the selected token itself was penalized
- `suppressed_token_count` — number of tokens suppressed by structural suppression
- `penalized_token_count` — number of tokens penalized by repetition penalty
- `decision_class` — one of `greedy`, `structural_suppression`, `repetition_penalty_shifted`. When both structural suppression and repetition penalty affect the step, `structural_suppression` takes priority
- `topk` — array of up to 5 entries (capped at `LIS_TRACE_TOPK_SIZE`), each with `token_id`, `raw_score`, `adjusted_score`, `is_selected`
- `stop_reason` — optional string (emitted only on the final step when a stop condition is reached; omitted from intermediate steps)

### Boundedness and Retention

The decode-trace artifact follows the same boundedness policy as `run_report`:

- Raw prompt text, generated text, and absolute file paths are omitted.
- Full-vocab rankings, raw logits arrays, and activation dumps are not included.
- Timestamps and wall-clock times are not included.
- The top-k list is capped at 5 entries (`LIS_TRACE_TOPK_SIZE`); this is not user-configurable. Fewer than 5 entries may appear when the vocabulary is smaller or fewer candidates are available.

### Fail-Closed Behavior

Trace artifact writing is fail-closed:

- If the output file cannot be opened, LIS reports the artifact error explicitly.
- If `fclose` fails after writing, LIS reports the artifact error explicitly. A partial file may remain on disk when the close fails.

### Independence from Existing Surfaces

- When `--trace-json` is absent, all existing behavior (stdout, stderr, `--report-json`, `--report-md`, `--diagnostics`, `--perf`) is unchanged.
- The trace artifact does not depend on `--diagnostics` or `--perf` being enabled.
- `--report-json` output is identical whether or not `--trace-json` is also present.

## Layer-Trace Artifact

LIS supports a compact `layer_trace` artifact emitted via `--layer-trace-json PATH`
when `--layer-checkpoints` is also enabled:

- The schema string is `lis.execution_artifact/v1`, the same as `run_report`
  and `decode_trace`.
- The kind is `layer_trace`.
- The top-level object order is `schema`, `kind`, `manifest`, then
  `layer_trace`.
- The artifact contains the same manifest shape as `decode_trace` and
  `run_report`: retention policy, binary, model, config, input, runtime, and
  backend identity through bounded fingerprints.
- `manifest.runtime.precision_path` is present and uses the same canonical
  semicolon-delimited value as the other trace artifacts.
- The artifact carries a `layer_trace` array instead of a `report` object or a
  `decode_trace` array.

### Layer-Trace Entries

Each entry in the `layer_trace` array represents one stat-bearing
tensor-summary `lis: layer-checkpoint` line and contains:

- `step` — generation/checkpoint step
- `phase` — checkpoint phase such as `prefill` or `decode`
- `name` — checkpoint tensor name
- `shape` — tensor shape dimensions
- `min` — minimum value, or `null` for non-finite values
- `max` — maximum value, or `null` for non-finite values
- `mean` — mean value, or `null` for non-finite values
- `l2` — L2 norm, or `null` for non-finite values
- `nan` — `0` or `1` companion flag
- `inf` — `0` or `1` companion flag

The scalar value-only `attn_scale` checkpoint remains stderr-only. It lacks
shape/min/max/mean/l2 tensor-summary semantics, is not routed through the
layer-trace record, and must not appear in `layer_trace[]`. This is a
documented and tested exclusion, not a silent omission.

### Boundedness and Retention

The layer-trace artifact follows the same bounded retention policy as the other
execution artifacts:

- Raw prompt text, generated text, and absolute file paths are omitted.
- Full activation dumps, logits dumps, and full tensor payloads are not
  included.
- Only compact checkpoint summaries already surfaced through stat-bearing
  `lis: layer-checkpoint` lines are captured.
- Capture is bounded by `LIS_LAYER_TRACE_HARD_MAX`; overflow sets a sticky
  failure flag and suppresses artifact emission instead of writing a partial
  trace.

### Fail-Closed Behavior

Layer-trace artifact writing is fail-closed:

- `--layer-trace-json PATH` without `--layer-checkpoints` is rejected before
  inference.
- If capture overflows, LIS reports the artifact error and leaves the requested
  path untouched for that run.
- If the output file cannot be opened or `fclose` fails, LIS reports the
  artifact error explicitly. A partial file may remain on disk when the close
  fails, consistent with the other artifact writers.

## LIS Inspect Compatibility Protection

Current LIS Inspect compatibility protects supported inputs: `run_report` JSON and
stderr perf logs. `decode_trace` and `layer_trace` are valid LIS artifacts using
schema `lis.execution_artifact/v1`, but current LIS Inspect is not required to
parse or display those artifact kinds. Trace/layer Inspect support
is deferred to future Inspect-owned work.

Compatibility rules:

- Existing `--report-json` keys are not removed or renamed.
- The top-level `run_report` shape remains `schema`, `kind`, `manifest`, then
  `report`.
- `report.perf` remains canonical for JSON perf data.
- `manifest.runtime.precision_path` is additive and does not replace existing
  runtime fields.
- Existing stderr prefixes are preserved for `lis: perf-stage `,
  `lis: perf-summary `, `lis: perf-per-token `,
  `lis: generation-diagnostic `,
  `lis: generation-diagnostic-candidate `, `lis: layer-checkpoint `, and
  `lis: simd backend=`.
- Additive stderr lines, including `lis: generation-diagnostic-reasoning `
  and `lis: precision path=`, must not mutate existing lines.

Current LIS Inspect limitation:

- Current LIS Inspect JSON parsing is scoped to `run_report`.
  Rejection of `decode_trace` or `layer_trace` by current Inspect is documented
  behavior, not a compatibility failure.

Accepted runtime fingerprint nuance:

- `runtime_fingerprint.size_bytes` intentionally changed from
  `sizeof(lis_cli_options)` to a stable semantic sum of hashed runtime inputs.
  Runtime fingerprint digest identity remains unchanged for unchanged hashed
  inputs.

## KV Cache Inspect Compatibility Protection

Current LIS Inspect compatibility protects supported inputs without broadening Inspect.
`run_report` JSON with additive `report.kv_cache` remains accepted and
older `run_report` JSON without `report.kv_cache` remains accepted. The raw
payload is preserved for compatibility/debugging. The optional
`lis: kv-cache:` diagnostics line is ignored by the existing perf stderr
parser.

Current LIS Inspect limitation:

- Current LIS Inspect does not render, interpret, compare, graph, or
  visualize KV cache data. That is deferred to a separate Inspect-owned future
  work item.

The approved design contract for differential verification is documented in
`docs/differential_verification.md`. It defines a planned `verification_report`
artifact kind under `lis.execution_artifact/v1`. The feature remains planned and
is not yet part of the currently implemented runtime, CLI, verification, or
artifact surface.
