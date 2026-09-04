# LIS (Local Inference System)

[![CI](https://github.com/MinsungMJ/Local-Inference-System/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MinsungMJ/Local-Inference-System/actions/workflows/ci.yml)

> **Correctness first. Transparency always.**

LIS is a CPU-only local inference runtime for causal decoder-only models, built for engineers and researchers who need a system they can inspect, validate, reproduce, and optimise with confidence. It prioritises correctness, clear diagnostics, reproducibility, differential verification, and performance transparency over broad feature coverage.

LIS is an independent personal project. The initial codebase is personally authored.

## Key Properties

- **Correctness-first** — reference execution path with verified token parity and token mismatch localisation
- **Inspectability** — opt-in machine-readable execution artifacts and diagnostics
- **Differential verification** — verified mismatch-boundary reproduction and coverage-scoped Llama layer and intra-layer localisation
- **Reproducibility** — bounded, versioned, source-bound execution artifacts with canonical content identities
- **Performance transparency** — opt-in per-stage and per-token wall-clock instrumentation
- **Artifact-friendly execution** — structured JSON reports, Markdown companions, and diagnostic traces without telemetry or uploads
- **Conservative support boundaries** — documented subset, explicit rejection of unsupported inputs

## Differential Verification

The workflow localises a selected-token mismatch, verifies its prompt, generated-prefix, decode-policy, context, and target-checkpoint boundary, then compares source-bound Llama layer-output summaries from two verified comparable executions.

LIS can identify the earliest observed mismatching Llama layer-output checkpoint within common captured coverage and report a bounded suspect interval. Sparse capture remains explicit, so the interval may contain uncaptured layers.

For an observed mismatching Llama layer output, LIS can narrow that interval to the earliest observed mismatching intra-layer checkpoint within validated common captured coverage. This opt-in diagnostic is decode-only, targets one selected layer, and preserves sparse or missing local coverage in the reported interval.

This is bounded digest-based diagnostic evidence. It does not prove tensor equality, confirm the first numeric or operation-level divergence, identify a root cause, establish complete intra-layer coverage, or localise an uncaptured operation. See [Differential Verification](docs/differential_verification.md) for the detailed contract and [Reproducibility and Execution Artifacts](docs/repro_execution_artifacts.md) for artifact generation and handling.

## Supported Scope

### Runtime and Model Support

- CPU-only local execution
- Causal decoder-only models within the documented plain-RoPE Llama-family scope
- A narrow Qwen3 Dense BF16 merged-safetensors path (does not imply broad Qwen-family support)
  - Prompts are passed as raw tokenizer text. LIS does not apply model-specific chat templates or expose thinking-mode controls, so reasoning-oriented models may produce extended explanatory output even for short prompts.
- Local HuggingFace-style directories containing `config.json`, a merged `model.safetensors`, and a compatible `tokenizer.json`
- Supported floating dtypes:
  - Llama-family path: F32, F16, BF16
  - Qwen3 Dense path: BF16 only
- HuggingFace BPE `tokenizer.json` subset, `LIS_VOCAB_V1`, and direct token IDs
- Greedy decode only
- Opt-in artifact and diagnostic outputs
- LIS Inspect currently supports `run_report` JSON and optional perf stderr logs

### Differential-Verification Support

- Token mismatch localisation and mismatch-boundary reproduction require semantically compatible, source-bound execution artifacts.
- Coverage-scoped layer localisation supports the versioned Llama layer-output trace layout with matching precision paths.
- Coverage-scoped intra-layer localisation supports Llama decode evidence for one previously identified mismatching layer, using the fixed 17-stage semantic capture profile and an authoritative revalidated layer-output boundary.
- The C CLI captures the required evidence with `--intra-layer-checkpoints LAYER`; localization remains available through the `lis_verify` Python library. The installable `lis-verify` product spine exists, while its seeded and real workflow adapters remain gated to Pass 5 M2 and M3 respectively.
- Qwen3 inference support does not imply layer- or intra-layer-localisation support. Qwen3 and legacy layer-trace layouts are unsupported for these comparisons.

## Unsupported / Non-Goals

- GPU backend
- Serving / HTTP endpoint
- Distributed inference
- Continuous batching
- Sampling frameworks (temperature, top-p, top-k, beam search, speculative decoding)
- GGUF / GGML
- PyTorch `.bin`, `.pt`, `.pth`
- Index-only sharded safetensors loading
- LoRA / QLoRA / adapters
- Quantised formats beyond current floating dtype scope
- Broad Qwen-family support, Qwen2/Qwen2.5, Qwen3 MoE, multimodal/VL
- Mistral, GPT-2, and other model families unless separately implemented
- RoPE scaling, YaRN, sliding window, long-context variants
- Chat-template / Jinja execution
- LIS Inspect rendering for `decode_trace`, `layer_trace`, or KV visualisation (deferred)

## Build

LIS requires a C11 compiler, standard library, and POSIX threads (`pthreads`). No external dependencies.

```bash
git clone <repo-url> LIS
cd LIS
make build
```

The built binary is `srcs/libs/lis`. `make build` requires no private model artifacts.

## Test

```bash
make test
```

`make test` requires no private model artifacts. It builds the binary and runs the core, loader, backend, runtime, CLI, tokenizer, and threading test suites.

## LIS Verify Product Spine

The Pass 5 M1 product spine is installable without UI dependencies:

```bash
python -m pip install --no-deps .
lis-verify --help
lis-verify --version
```

It provides the frozen `demo`, `backend`, and `runtime` parsers, canonical
`lis.verification_report/v1` model and serializer, deterministic summaries,
private attempt workspaces and ledgers, an explicit stage machine, and bounded
subprocess primitives. The base `lis-verify` path has no required dependency on
Textual. Install `.[inspect]` only when the optional LIS Inspect TUI is wanted.

M1 is a product-spine milestone, not the customer Alpha. Until M2 connects the
seeded fixture, `lis-verify demo` fails before attempt creation with exit 2 and
does not fabricate a report or source identity. `backend` and `runtime` remain
likewise unconnected until M3. See [LIS Verify Product Spine](docs/lis_verify_product_spine.md)
and [LIS Verify Product Contract](docs/lis_verify_contract.md).

## First Run

Model-backed execution requires a user-supplied local model. The examples below use a placeholder path; replace it with your own plain-RoPE Llama-family model directory.

```bash
MODEL_DIR=/path/to/plain-rope-llama

./srcs/libs/lis \
  --model "$MODEL_DIR" \
  --config "$MODEL_DIR/config.json" \
  --hf-tokenizer "$MODEL_DIR/tokenizer.json" \
  --prompt "Write one short sentence about the sea." \
  --context 128 \
  --batch 1 \
  --generate 8 \
  --threads 1 \
  --report-json /tmp/lis_run.json
```

`/tmp/lis_*` paths are example output locations; you may choose any writable path.

## Optional Model-Backed Validation

Model-backed targets require explicit environment variables. Unset variables yield a clear error message; no target falls back to a private path.

```bash
make verify-token-parity VERIFY_MODEL=/path/to/plain-rope-llama
make verify-qwen3-sanity VERIFY_QWEN3_MODEL=/path/to/qwen3-dense
make bench BENCH_MODEL=/path/to/plain-rope-llama
```

`VERIFY_CONFIG` and `VERIFY_HF_TOKENIZER` may be supplied explicitly when the default derived paths are not suitable.

## Artifacts and Diagnostics

All artifact and diagnostic surfaces are opt-in.

### CLI Flags

| Flag | Purpose |
|---|---|
| `--report-json PATH` | Canonical machine-readable `run_report` execution artifact (`lis.execution_artifact/v1`) |
| `--report-md PATH` | Human-readable Markdown companion report |
| `--trace-json PATH` | Bounded `decode_trace` artifact with decode-step token evidence |
| `--layer-trace-json PATH` | Bounded `layer_trace` artifact; the supported Llama layout records checkpoint coverage, execution order, and representation digests (requires `--layer-checkpoints`) |
| `--diagnostics` | Opt-in generation diagnostics to stderr |
| `--perf` | Per-stage wall-clock timings and summary to stderr |
| `--perf-per-token` | Implies `--perf`; adds per-decode-step latency lines |
| `--forced-prefix "ID ..."` | Forced token IDs for diagnostic comparison |
| `--layer-checkpoints STEP` | Capture bounded layer checkpoint summaries at runtime checkpoint `STEP` (`0` = prefill) |

### Stderr Surfaces

- `lis: perf-stage` / `lis: perf-summary` / `lis: perf-per-token` — performance instrumentation
- `lis: generation-diagnostic*` — token-selection diagnostics
- `lis: precision path=` — resolved precision path summary
- `lis: kv-cache:` — KV cache diagnostics

### Artifact Keys

- `report.kv_cache` — deterministic KV cache structural accounting
- `manifest.runtime.precision_path` — run precision summary in `f32_accum;weights=<dtype>;kv=<dtype>` form
- `artifact_set_id` — probabilistic association shared by sibling artifacts from one CLI execution. It is not a content hash.
- `checkpoint_layout.requested_coordinates`, `captured_coordinates`, and `missing_coordinates` — explicit Llama checkpoint coverage, including missing-state metadata
- `checkpoint_layout.ordering_semantics` and `digest_contract` — declared execution ordering and bounded observed-representation digest rules

The JSON `run_report` is the canonical machine-readable source of truth. The Markdown report is a human-readable companion. Canonical SHA-256 identifies artifact content, while semantic manifest identity establishes execution and configuration compatibility. The verification tooling validates these roles and each artifact's same-execution association before comparing checkpoint summaries.

`decode_trace` and `layer_trace` are bounded artifact outputs. The resulting `layer_localization` artifact records common coverage, digest decisions, and any suspect interval without embedding full tensor payloads. LIS Inspect currently supports `run_report` only; it has no dedicated `decode_trace`, `layer_trace`, or `layer_localization` view. Artifact generation and machine-readable consumption do not depend on LIS Inspect visualisation.

## LIS Inspect

LIS Inspect is a post-execution TUI inspector (Textual-based) that reads the canonical `--report-json` artifact and optional captured stderr from a `--perf` run. It provides Overview, Perf, Per-Token, Artifact, Raw, and Issues tabs. With two report inputs it launches a two-run compare view.

```bash
PYTHONPATH=tools python -m lis_inspect \
  --report-json /tmp/lis_run.json \
  --stderr-log /tmp/lis_run.stderr
```

Currently supports `run_report` JSON and optional perf stderr logs. Trace, layer, and KV rendering are deferred.

## Documentation

- [Differential Verification](docs/differential_verification.md)
- [LIS Verify Product Spine](docs/lis_verify_product_spine.md)
- [LIS Verify Product Contract](docs/lis_verify_contract.md)
- [Reproducibility and Execution Artifacts](docs/repro_execution_artifacts.md)
- [Precision Policy](docs/precision_policy.md)
- [HuggingFace tokenizer.json Compatibility](docs/hf_tokenizer_compat.md)
- [HuggingFace Llama Compatibility](docs/huggingface_llama_compatibility.md)
- [Loader Format Scope](docs/loader_format_scope.md)

## Security

See [SECURITY.md](SECURITY.md) for vulnerability reporting. Please use GitHub private vulnerability reporting / GitHub Security Advisories.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding style, compatibility expectations, and pull request guidelines.

## License

Licensed under the Apache License, Version 2.0 ([LICENSE](LICENSE)). SPDX identifier: `Apache-2.0`.

See [NOTICE](NOTICE) for attribution, including third-party dependency attribution.
