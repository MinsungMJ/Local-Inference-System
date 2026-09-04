# LIS Verify Product Spine

- Milestone: Pass 5 M1
- Status: Implemented
- Product contract: `lis.verify.product_contract/v1`
- Report schema: `lis.verification_report/v1`

M1 provides the reusable, model-free lifecycle framework for later LIS Verify
workflows. It does not run a model or claim that a customer comparison occurred.

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

M1 intentionally registers no production mode runner. Calling a valid mode
before its adapter is implemented exits 2 before attempt creation and does not
write a report. This avoids inventing required source identities. M2 connects
`demo`; M3 connects `backend` and `runtime` after the source-bound forced-prefix
report prerequisite is implemented.

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
  bounded grace; and
- observed cleanup/residue reporting and explicit debug retention.

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
