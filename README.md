# LIS (Local Inference System)

[![CI](https://github.com/MinsungMJ/Local-Inference-System/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MinsungMJ/Local-Inference-System/actions/workflows/ci.yml)

> **Correctness first. Transparency always.**

LIS is a CPU-only local inference runtime for causal decoder-only models, built for engineers and researchers who need a system they can inspect, validate, reproduce, and optimise with confidence. It prioritises correctness, clear diagnostics, reproducibility, and performance transparency over broad feature coverage.

LIS is an independent personal project. The initial codebase is personally authored.

## Key Properties

- **Correctness-first** — reference execution path with verified token parity
- **Inspectability** — opt-in machine-readable execution artifacts and diagnostics
- **Reproducibility** — bounded, versioned run reports with content-addressable fingerprints
- **Performance transparency** — opt-in per-stage and per-token wall-clock instrumentation
- **Artifact-friendly execution** — structured JSON reports, Markdown companions, and diagnostic traces without telemetry or uploads
- **Conservative support boundaries** — documented subset, explicit rejection of unsupported inputs

## Supported Scope

- CPU-only local execution
- Causal decoder-only models within the documented plain-RoPE Llama-family scope
- A narrow Qwen3 Dense BF16 merged-safetensors path (does not imply broad Qwen-family support)
- Local HuggingFace-style directories containing `config.json`, a merged `model.safetensors`, and a compatible `tokenizer.json`
- Supported floating dtypes:
  - Llama-family path: F32, F16, BF16
  - Qwen3 Dense path: BF16 only
- HuggingFace BPE `tokenizer.json` subset, `LIS_VOCAB_V1`, and direct token IDs
- Greedy decode only
- Opt-in artifact and diagnostic outputs
- LIS Inspect currently supports `run_report` JSON and optional perf stderr logs

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
| `--report-json PATH` | Canonical machine-readable execution artifact (`lis.execution_artifact/v1`) |
| `--report-md PATH` | Human-readable Markdown companion report |
| `--trace-json PATH` | Bounded decode-step trace artifact |
| `--layer-trace-json PATH` | Compact per-layer / per-stage trace artifact (requires `--layer-checkpoints`) |
| `--diagnostics` | Opt-in generation diagnostics to stderr |
| `--perf` | Per-stage wall-clock timings and summary to stderr |
| `--perf-per-token` | Implies `--perf`; adds per-decode-step latency lines |
| `--forced-prefix "ID ..."` | Forced token IDs for diagnostic comparison |
| `--layer-checkpoints STEP` | Layer checkpoint stats at the given step |

### Stderr Surfaces

- `lis: perf-stage` / `lis: perf-summary` / `lis: perf-per-token` — performance instrumentation
- `lis: generation-diagnostic*` — token-selection diagnostics
- `lis: precision path=` — resolved precision path summary
- `lis: kv-cache:` — KV cache diagnostics

### Artifact Keys

- `report.kv_cache` — deterministic KV cache structural accounting
- `manifest.runtime.precision_path` — run precision summary in `f32_accum;weights=<dtype>;kv=<dtype>` form

The JSON `run_report` is the canonical machine-readable source of truth. The Markdown report is a human-readable companion. `decode_trace` and `layer_trace` are bounded artifact outputs; current LIS Inspect is not required to render them.

## LIS Inspect

LIS Inspect is a post-execution TUI inspector (Textual-based) that reads the canonical `--report-json` artifact and optional captured stderr from a `--perf` run. It provides Overview, Perf, Per-Token, Artifact, Raw, and Issues tabs. With two report inputs it launches a two-run compare view.

```bash
PYTHONPATH=tools python -m lis_inspect \
  --report-json /tmp/lis_run.json \
  --stderr-log /tmp/lis_run.stderr
```

Currently supports `run_report` JSON and optional perf stderr logs. Trace, layer, and KV rendering are deferred.

## Documentation

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
