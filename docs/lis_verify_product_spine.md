# LIS Verify Product Spine

- Milestone: Pass 5 M1–M4 product execution and CI gating
- Status: Implemented
- Product contract: `lis.verify.product_contract/v1`
- Report schema: `lis.verification_report/v1`

M1 provides the reusable lifecycle framework. M2 connects the model-free seeded
demonstration. M3 connects real local-model `backend` and `runtime` execution,
including source-bound forced-prefix reproduction and bounded Pass 3/4
recapture.
M4 adds a pinned public-model manifest, strict local-material verification,
acceptance-classified CI consumption, and the `make verify-diff` wrapper.

## Installation

The base package has no runtime dependency:

```bash
python -m pip install --no-deps .
lis-verify --help
lis-verify --version
```

The equivalent source-tree entry point is:

```bash
PYTHONPATH=tools python -m lis_verify --help
```

Textual is not imported by the normal `lis-verify` path. The existing
`lis-inspect` TUI is packaged behind the optional `inspect` extra:

```bash
python -m pip install '.[inspect]'
```

## Current command boundary

The parser exposes the frozen modes and only their customer options:

```text
demo
backend --model MODEL
runtime --reference-bin BIN --candidate-bin BIN --model MODEL
```

Common options are `--out`, `--require-supported`, `--debug-retain`,
`--stage-timeout-seconds`, and `--verbose`. Pass numbers, forced prefixes,
checkpoint steps, target layers, intermediate artifact paths, and artifact-set
IDs are not customer options.

M2 registers the production `demo` runner. M3 registers the production
`backend` and `runtime` runners. `backend` resolves the checkout binary before
`PATH`, compares `LIS_SIMD=0` with normal optimized dispatch, and refuses to call
reference fallback an optimized-path pass. `runtime` requires two distinct
explicit binary identities and preserves each role across reproduction.

Both real modes use the packaged `plain_rope_llama_direct_token_v1` profile:
direct token ID `[1]`, context 128, batch 1, generation limit 8, and one thread.
The local model must be a merged `model.safetensors` plain-RoPE Llama directory.
The generic profile remains distinct from the M4 public-model revision
manifest in `lis_verify.golden_models`.

Every eligible binary requires a binary-adjacent `<binary>.lis-build.json`
manifest with schema `lis.build_provenance/v1`. Its canonical source-tree and
binary SHA-256 values are verified before inference. A missing sidecar yields
`INCONCLUSIVE`; a stale, tampered, malformed, or symlinked sidecar fails as an
integrity error. Neither path invents source authority.

## Implemented components

- immutable validated report values backed by the M0 validator;
- exhaustive Pass-status and policy aggregation helpers;
- UTF-8 sorted compact canonical JSON with one trailing newline;
- deterministic terminal and Markdown report-only renderers;
- atomic, private, no-overwrite report bundle publication;
- `verification_report.json` as the last bundle commit marker;
- canonical stage-order and dependency state machine;
- mode-0700 attempt workspaces and mode-0600 artifacts;
- 128-bit random attempt identity;
- append-only, fsync-backed ledger with mutation detection;
- shell-free streaming subprocess capture with a combined 1 MiB cap;
- per-stage timeout, process-group termination, signal classification, and
  bounded grace;
- observed cleanup/residue reporting and explicit debug retention;
- real LIS model/profile and binary provenance preflight;
- source-bound non-empty forced-prefix reproduction without normal raw-prefix
  retention;
- role-aware backend/runtime binary continuity; and
- Pass 3A discovery, fresh bounded recapture, Pass 3B revalidation, and Pass 4
  intra-layer localization for reproduced mismatches;
- immutable public-model material, license, configuration, and expected-result
  manifest validation;
- clean-state acceptance-manifest loading for CI without adding a public CLI
  mode or Pass control; and
- canonical report/summary/ledger CI validation with bounded step summaries.

## Output lifecycle

An injected or future production runner follows this order:

```text
request validation
  -> runner availability preflight
  -> private attempt and ledger
  -> canonical stage machine
  -> aggregation
  -> runtime cleanup observation
  -> report and summary preparation
  -> ledger finish
  -> summary publication
  -> verification_report.json publication
```

An equal real pair stops after Pass 1 and records Pass 2–4 as
`not_applicable`. A mismatch runs an independent paired reproduction. For a
non-empty prefix, the C producer recomputes the applied count/digest and binds
the run to Pass 0, the role-specific original report, Pass 1, and localization.
Only verified reproduction can enter Pass 3A. A discovered layer then triggers
one fresh bounded recapture generation before Pass 3B and Pass 4.

The final report is the only customer-result source of truth. `summary.md` and
terminal output are deterministic projections and cannot strengthen evidence.
Expected final report, summary, and ledger files are published outputs rather
than residue. Unreported processes, runtime files, sockets, or partial files are
residue.

## Safety and non-claims

- No network access, telemetry, or model download.
- No raw prompt/generated text or raw tensor values in the canonical report.
- No silent retry or attempt reuse.
- Timeout blocks dependent stages and never implies health.
- Unknown cleanup state is not reported as zero residue.
- Bounded digests are not tensor equality or numeric confirmation.
- A suspect interval is not a confirmed first divergence.
- M1 synthetic fixtures are not production or acceptance evidence.

Implementation/debugging attempts and verification/acceptance attempts use
separate identities and evidence. A clean one-shot acceptance begins only after
debug workspaces and processes are scoped-cleaned and the final source and
package inputs are frozen.

Normal interactive commands remain `development_debugging`. CI supplies the
private `LIS_VERIFY_ACCEPTANCE_MANIFEST` authority only after clean state,
source revision/tree, dependency files, and command files have been frozen.
Malformed or non-private authority fails before an attempt is started.
