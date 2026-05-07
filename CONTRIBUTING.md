# Contributing to LIS

Thank you for your interest in contributing to LIS. This document describes how to contribute effectively.

## Project Scope and Contribution Fit

LIS is a CPU-only local inference runtime for causal decoder-only models. It prioritises correctness, inspectability, reproducibility, and performance transparency over broad feature coverage.

Contributions that align with the project's scope are welcome. Contributions that expand scope beyond documented boundaries — such as GPU backends, serving endpoints, sampling frameworks, or unsupported model families — will require explicit maintainer discussion before acceptance.

## Before Opening an Issue or PR

- Search existing issues and pull requests to avoid duplicates.
- Open an issue for discussion before starting significant work.
- Keep pull requests narrow and reviewable.
- Do not include secrets, model weights, private prompts, private generated text, local or private file paths, or other confidential data in issues, pull requests, documentation, fixtures, or examples.
- Do not overclaim model, tokenizer, runtime, artifact, LIS Inspect, or performance support.
- Security-sensitive reports must follow [SECURITY.md](SECURITY.md), not public issues.

## Development Setup

LIS requires:

- A C11 compiler
- POSIX threads (`pthreads`)
- Standard C library
- No external dependencies for the core build and test targets

## Build and Test

Default build and test require no private model artifacts:

```bash
make build
make test
```

`make build` produces `srcs/libs/lis`. `make test` builds and runs the core, loader, backend, runtime, CLI, tokenizer, and threading test suites.

## Optional Python Tooling Tests

Python tooling tests require dependencies from `tools/requirements.txt`:

```bash
PYTHONPATH=tools python3 -m unittest discover tools/tests
```

Missing optional dependencies (such as `textual`) may cause some tests to skip or error. Python tooling tests are not required for the core C build and test gate.

## Optional Model-Backed Validation

Model-backed targets require explicit environment variables. Unset variables produce a clear error message. No target falls back to a private or hard-coded path.

```bash
make verify-token-parity VERIFY_MODEL=/path/to/plain-rope-llama
make verify-qwen3-sanity VERIFY_QWEN3_MODEL=/path/to/qwen3-dense
make bench BENCH_MODEL=/path/to/plain-rope-llama
```

`VERIFY_CONFIG` and `VERIFY_HF_TOKENIZER` may be supplied explicitly when the default derived paths are not suitable.

## Coding Style

LIS follows a C11 baseline with explicit contracts and reviewable code. Key expectations:

- Target C11 or newer; prefer widely practical language features.
- Use fixed-width integer types from `<stdint.h>` when width matters. Use `size_t` for sizes, lengths, and capacities.
- Make ownership, lifetime, mutability, nullability, size/capacity expectations, and failure models clear through signatures, naming, or comments.
- Pass buffer pointer and length/capacity together. Bound all parse, copy, and encode operations explicitly.
- Use consistent explicit error/status conventions within a module. Do not collapse distinct failure modes into an ambiguous boolean.
- Allocate with `sizeof(*ptr)`. Check allocation results. Keep cleanup paths consistent and leak-resistant.
- Use `_Static_assert` for layout, size, alignment, and invariant checks rather than leaving assumptions as comments only.
- Prefer `static inline` over function-like macros. Use macros only where the language is insufficient (include guards, conditional compilation, carefully justified generic wrappers, token/string generation).
- Use `<stdatomic.h>` for lock-free shared-state synchronisation when needed. Do not use `volatile` as a thread-synchronisation primitive.
- Use `const` aggressively for read-only data. Use designated initialisers for nontrivial structs.
- Make environment-specific assumptions explicit. Do not present environment-specific code as portable ISO C.
- Generated or submitted code should compile warning-clean under strong warnings (`-Wall -Wextra -Wpedantic -Werror`).
- Prefer small, explicit helper functions over clever macros or terse one-liners.
- Prefer enums for states and status codes.
- Prefer a single cleanup path when resource management is nontrivial.

See [`docs/development_style.md`](docs/development_style.md) for the full public style guide.

## Documentation Expectations

- Update documentation when code behaviour, CLI flags, artifact schemas, or supported surfaces change.
- Every `.c` file should eventually have a matching explanation document under `docs/`.
- Documentation must not include local or private paths, private prompts, private generated text, secrets, or model weights.
- Per-source documentation is for source files that exist, not for files planned but not yet created.

## Artifact and Diagnostic Compatibility Expectations

The following are compatibility-sensitive surfaces. Changes to these surfaces require explicit documentation and tests:

- CLI flags
- Artifact schema strings and kind names (e.g. `lis.execution_artifact/v1`, `run_report`)
- Documented `stderr` prefixes (`lis: perf-stage`, `lis: perf-summary`, `lis: perf-per-token`, `lis: generation-diagnostic*`, `lis: precision path=`, `lis: kv-cache:`, `lis: simd backend=`)
- LIS Inspect accepted inputs (`run_report` JSON, optional perf stderr logs)
- `run_report`, `decode_trace`, `layer_trace`, `report.kv_cache`, `precision_path` field naming and semantics

### Compatibility Rules

- Changes to compatibility-sensitive surfaces must include tests and documentation updates.
- Backward-incompatible changes require explicit maintainer decision and release notes.
- Prefer additive fields over renames or removals.
- Avoid accidental schema, stderr, CLI, or LIS Inspect drift.

## Supported and Unsupported Scope Preservation

Contributions must not expand the documented supported scope without maintainer discussion:

- No GPU backend implementation without explicit authorisation.
- No new model-family, quantisation, or serving features beyond documented scope.
- No sampling frameworks, continuous batching, or chat-template execution.
- No GGUF, GGML, or PyTorch `.bin`/`.pt`/`.pth` format support beyond documented boundaries.
- No overclaim of model, tokenizer, runtime, artifact, or LIS Inspect support.
- No local or private paths in any committed file.

## Security Reporting

Report suspected vulnerabilities through [SECURITY.md](SECURITY.md) using GitHub private vulnerability reporting or GitHub Security Advisories. Do not report security issues through public GitHub issues.

Do not include secrets, private keys, credentials, proprietary model weights, or confidential data in any report.

## Benchmark and Performance Claim Policy

New benchmark or performance claims require:

- The exact command used to produce the measurement
- The model, config, and tokenizer used
- Hardware description (CPU, cores, memory)
- Date of the measurement
- Context length, number of generated tokens, and thread count
- Explicit limitations or caveats

Undocumented, unqualified, or irreproducible performance claims will not be accepted.

## Pull Request Checklist

Before submitting a pull request, verify:

- [ ] `make build` and `make test` pass with no model artifacts.
- [ ] No local or private paths are included in any file.
- [ ] No secrets, model weights, private prompts, or private generated text are included.
- [ ] Compatibility-sensitive surface changes (CLI flags, artifact schemas, stderr prefixes, LIS Inspect inputs) are documented and tested.
- [ ] New `.c` files have a matching explanation document under `docs/`.
- [ ] Documentation is consistent with code behaviour.
- [ ] No scope expansion beyond documented supported boundaries without maintainer discussion.
- [ ] Benchmark or performance claims, if any, include command, model, hardware, date, context, generated-token count, and limitations.

## License

By contributing, you agree that your contributions are licensed under the
Apache License, Version 2.0 (`Apache-2.0`), the same licence as the project,
unless project governance explicitly states otherwise.
