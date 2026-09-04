# Verification Framework

LIS verification is a documented, repeatable engineering surface. It does not
change inference semantics, add decode policy, introduce new model families, or
expand precision behavior. The CPU reference path remains the correctness
baseline; optimized and precision-aware paths must either match it under the
documented policy or report a documented unsupported result.

## Verification Layers

| Layer | Entry Point | Purpose | Routine |
|---|---|---|---|
| Kernel verification | `make verify-kernels` | Compare backend kernel outputs against the reference behavior at focused shapes. The initial implementation reuses `srcs/libs/test_cpu_avx`, but the entry point is intentionally backend-neutral so future non-SIMD kernel checks can live under the same target. | Yes |
| CLI regression | `make verify-cli` | Reuse the existing bounded CLI integration tests, including diagnostics and end-to-end validation fixtures. | Yes |
| Token parity | `make verify-token-parity` | Run a fixed Llama-family prompt through the reference path and the default dispatch path, then compare the exact generated token ID sequence. The harness writes a verification snapshot JSON that embeds both core run reports. Requires the documented local model artifacts. | Model-backed routine |
| Performance smoke | `make verify-perf-smoke` | Exercise the existing perf parser on a tiny default-dispatch run and write CSV/Markdown plus a bounded benchmark snapshot JSON reusing the core run-report contract. This is protocol verification, not a benchmark expansion. | Optional |

`make verify` runs the no-model routine checks: `verify-kernels` and
`verify-cli`. Model-backed parity and perf smoke checks are separate because
they require local HuggingFace artifacts and can be slow on the scalar reference
path.

## Result Classes

Verification results must be classified with one of these meanings:

| Result Class | Meaning |
|---|---|
| `pass` | The check completed and satisfied the documented comparison policy. |
| `documented_unsupported` | The requested artifact, backend, or runtime path is outside the current LIS support envelope. This is not a regression. |
| `numeric_regression` | A kernel or numeric comparison exceeded the active tolerance policy. |
| `token_parity_regression` | A selected fixed-prompt case produced a different generated token ID sequence between the reference path and the path under test. |
| `benchmark_protocol_regression` | A performance/profiling check failed to emit or parse the expected protocol lines, or a documented regression threshold failed in later authorized work. |
| `harness_configuration_error` | The verification command could not run because of missing files, missing binaries, malformed arguments, or other harness setup problems. |

Unsupported scope and harness configuration errors must not be reported as
generic correctness failures. The output should make it clear whether to fix
LIS behavior, install/provide artifacts, or skip an unsupported path.

## Numeric Policy

Kernel-level comparisons use the active kernel suite's documented tolerance.
The current initial kernel-verification implementation is the reference-vs-AVX
suite:

- absolute tolerance: `1e-6`
- relative tolerance: `1e-4`
- reduction-noise cap: up to 2% of outputs may exceed the pointwise gate for
  lane-parallel reduction shapes where the difference is documented as valid
  IEEE-754 rounding-order drift

Future kernel checks may add tighter or different tolerances only when the
kernel contract justifies them and the policy is documented next to the check.
Exact token parity supersedes acceptable numeric noise: if a tolerated numeric
diff changes greedy token selection in a selected parity case, the result is a
token-parity regression and the kernel/path must be investigated.

## Token-Parity Case

The initial bounded token-parity fixture lives in
`tests/verification/run_token_parity.py` and is exposed through
`make verify-token-parity`.

Defaults:

- `make verify-token-parity` requires `VERIFY_MODEL`. Set:
  `VERIFY_MODEL=/path/to/plain-rope-llama`
- config: `VERIFY_CONFIG=/path/to/config.json` (defaults to
  `$(VERIFY_MODEL)/config.json` when `VERIFY_MODEL` is set)
- tokenizer: `VERIFY_HF_TOKENIZER=/path/to/tokenizer.json` (defaults to
  `$(VERIFY_MODEL)/tokenizer.json` when `VERIFY_MODEL` is set)
- prompt: `"Write one short sentence about the sea."`
- context: `128`
- batch: `1`
- generate: `8`
- threads: `1`

`make verify` is the routine no-model gate. Model-backed parity and perf smoke
checks are optional local/manual gates because they require local HuggingFace
artifacts and can be slow on the scalar reference path.

The harness runs two paths:

1. reference path: `LIS_SIMD=0`
2. candidate path: default dispatch with `LIS_SIMD` unset

Both runs request `--report-json` and compare the emitted
`report.selected_token_ids` arrays for exact equality. The harness reads the
observed backend from `manifest.backend.name` in each run report and writes a
verification snapshot JSON that embeds both run reports. If the candidate path
resolves to `reference`, the result is `documented_unsupported` for
optimized-backend parity on that host rather than a false optimized-path pass.

## Backend And Path Observability

Verification still uses existing LIS stderr conventions where appropriate. The
CLI already emits:

```text
lis: simd backend=reference|avx2|avx512
```

when `--diagnostics` or `--perf` is enabled. Verification does not add a logging or
telemetry subsystem. Verification and benchmark snapshots also carry
the resolved backend in the embedded core run report, so backend/path
observability no longer depends only on parsing stderr.

## Entry Point Details

```sh
make verify
make verify-kernels
make verify-cli
make verify-token-parity
make verify-qwen3-sanity
make verify-perf-smoke
```

`verify-perf-smoke` runs a tiny default-dispatch perf matrix cell and writes
scratch artifacts under `tests/verification/`, including
`perf_smoke.csv`, `perf_smoke.md`, and `perf_smoke.json`. It exists to verify
protocol and backend/path capture, not to establish performance claims.

## Extension Discipline

Precision-aware runtime checks and Qwen3 Dense support must reuse this
classification and entry-point discipline. Those areas may add cases under the
existing layers, but they should not redefine the core result classes,
observability rules, or fixture-vs-deferred-work separation established here.

For precision policy specifically, `make verify` remains the routine no-model gate,
`make verify-kernels` is the routine precision-aware numeric/kernel gate, and
`make verify-token-parity` is a stronger, slower, model-backed release validation
gate. Precision unsupported scope, such as mixed per-tensor dtype artifacts, is
reported as `documented_unsupported`; numeric drift and token changes map to
`numeric_regression` and `token_parity_regression` respectively.

For Qwen3 Dense support, `make verify` remains the routine no-model gate. `make
verify-qwen3-sanity` is a stronger, slower, model-backed release validation gate
for the documented Qwen3 Dense path. It requires `VERIFY_QWEN3_MODEL`; set:
`VERIFY_QWEN3_MODEL=/path/to/qwen3-dense`.
It uses direct token IDs, records the artifact/family path, writes a verification
snapshot JSON reusing the core run report, and requires bounded Qwen3-specific
checkpoint evidence without adding chat-template/Jinja scope.

## Framework Non-Goals

- The verification framework does not own precision semantics or
  Qwen3 model-family semantics; those areas plug model-backed cases into the
  existing result-class and entry-point discipline.
- No new serving/API surface.
- No decode-policy, sampling, prompt-template, tokenizer, loader, KV-cache, or
  inference-semantic changes from verification entry points themselves.
- No broad prompt-quality evaluation suite.
- No telemetry platform or new benchmark framework.

The approved design contract for differential verification is documented in
`docs/differential_verification.md`. Passes 0–4 are implemented as bounded
diagnostic library and artifact surfaces. The customer-facing LIS Verify
contract is frozen in `docs/lis_verify_contract.md`. Its M1 installable CLI,
unified report model/production, state-machine orchestration spine, private
workspace/ledger, deterministic renderers, and bounded execution primitives are
implemented. The M2 seeded `demo` adapter now exercises Pass 0–4 through the
unified report. Real backend/runtime adapters, `verify-diff`, and the public
model CI workflow remain later milestones.
