# LIS Verify Product Contract

- Contract status: Approved / frozen for Pass 5 M0
- Contract version: `lis.verify.product_contract/v1`
- Customer report schema: `lis.verification_report/v1`
- Implementation status: M0 contract frozen; M1 product spine implemented;
  seeded and real evidence adapters remain pending M2/M3

This document is the normative human-readable contract for the Pass 5 customer
product. Machine-readable conformance facts live under
`tools/test_fixtures/lis_verify_contract/`. Disagreement between this document,
those fixtures, and `tools/lis_verify/product_contract.py` is a contract failure;
none silently overrides another.

M0 freezes semantics and readiness only. It does not add an executable,
packaging, subprocess orchestration, report serializer, model workflow, C CLI
behavior, or inference behavior.

## 1. Product boundary

LIS Verify will turn the implemented Pass 0 through Pass 4 libraries and
producer artifacts into one report-driven customer workflow. Pass-local status,
reason, and disposition values remain available in detailed evidence. They are
not the primary customer decision.

The customer verdict enum is exactly:

```text
PASS
REGRESSION
INCONCLUSIVE
UNSUPPORTED
HARNESS_ERROR
```

- `PASS` means no selected-token difference was observed within the explicit
  comparison scope. It is never a whole-runtime or mathematical-equivalence
  claim.
- `REGRESSION` requires a source-bound selected-token mismatch in core v1.
  Later localization failure does not erase that independently valid result.
- `INCONCLUSIVE` means the requested supported workflow did not obtain enough
  trusted evidence for a pass or regression decision.
- `UNSUPPORTED` is a known capability boundary, not a regression.
- `HARNESS_ERROR` is an input, execution, artifact, identity, integrity, or
  report-production failure that prevents a trustworthy comparison decision.

## 2. Public commands

The frozen modes are `demo`, `backend`, and `runtime`.

| Mode | Required inputs | Behavior |
|---|---|---|
| `demo` | none | Offline model-free seeded fixture |
| `backend` | `--model` | Current binary reference backend versus resolved optimized backend |
| `runtime` | `--reference-bin`, `--candidate-bin`, `--model` | Two separately identified binary/source revisions |

The common output default is `.lis/verify`. Each attempt receives a new private
directory; existing attempts are never silently overwritten. Network access,
model download, telemetry, raw-text retention, raw-tensor retention, and debug
retention are disabled by default.

The common public options are `--out`, `--require-supported`,
`--debug-retain`, `--stage-timeout-seconds`, and `--verbose`. Core defaults are
batch size 1, one thread, eight generated tokens, and the current
`lis_policy_modified_greedy_v1` selection policy. `demo` uses its seeded
model-free fixture; `backend` and `runtime` obtain fixed direct-token input from
the supported-model profile. There is no implicit prompt or tokenizer choice.

The normal customer surface does not accept a Pass number, forced prefix,
runtime checkpoint step, target layer, intermediate artifact path,
artifact-set ID, or recapture sequence. The orchestrator owns those internal
choices.

`--require-supported` is a CI policy switch, not a comparison mode. Help and
version exit zero without starting an attempt. A syntax or argument error exits
2 before an attempt and need not produce a report.

## 3. Aggregation contract

Every current Pass 0 through Pass 4 status has one explicit aggregation action
in `pass_status_mapping_v1.json`. The mapping covers 46 statuses and has no
default branch. Adding or renaming a Pass-local status without extending the
mapping is a contract-test failure.

The finite actions are:

```text
continue
stop
inherit
retain_proven_regression_and_block_localization
not_applicable
```

Pass 0 `comparison_blocked` uses an exhaustive reason partition:

| Blocking reason | Customer verdict |
|---|---|
| `incompatible_decode_policy` | `UNSUPPORTED` |
| `input_token_divergence` | `HARNESS_ERROR` |
| `incompatible_model_family` | `UNSUPPORTED` |
| `config_fingerprint_mismatch` | `HARNESS_ERROR` |
| `external_oracle_ineligible` | `UNSUPPORTED` |
| `prompt_token_array_missing` | `INCONCLUSIVE` |
| `prompt_token_identity_unverified` | `INCONCLUSIVE` |

Pass 3 `observable_mismatch_found` uses the stage role: Pass 3A proceeds to
bounded recapture, while authoritative Pass 3B proceeds to Pass 4.

Source authority must exist before `PASS` or `REGRESSION`. A valid Pass 1
selected-token mismatch remains `REGRESSION` if Pass 2, Pass 3, or Pass 4 cannot
complete localization. A later failure may replace that verdict only when it
invalidates the comparison source, canonical report trust, or confidentiality.

## 4. Exit and CI policy

| Semantic verdict | Default exit | `--require-supported` exit |
|---|---:|---:|
| `PASS` | 0 | 0 |
| `REGRESSION` | 4 | 4 |
| `INCONCLUSIVE` | 3 | 3 |
| `UNSUPPORTED` | 0 | 6 |
| `HARNESS_ERROR` | 2 | 2 |

The report contains separate `verdict` and `policy_result` fields. Strict policy
can fail a mandatory unsupported check but cannot rewrite `UNSUPPORTED` as a
regression or harness error. Handled `SIGINT` and `SIGTERM` use process exits 130
and 143 respectively while preserving the report's semantic result when a
trustworthy report can be emitted.

The original planned base contract's numeric-regression exit 5 remains frozen
in its legacy fixture. It is not the LIS Verify product exit mapping.

## 5. Canonical report

`verification_report.json` is the sole customer-result source of truth.
Terminal and Markdown summaries must be deterministic renderings of it. The
report identity is:

```text
schema = lis.verification_report/v1
kind = verification_report
report_version = 1.0
```

The exact top-level fields are:

```text
schema
kind
report_version
attempt
command
verdict
reason_codes
policy_result
identities
token_comparison
localization
coverage
numeric_confirmation
evidence
stages
next_action
warnings
cleanup
```

Reference and candidate each record separate SHA-256 identities for source,
binary, model, configuration, input, runtime, and backend. Relative paths,
artifact-set IDs, or self-declared labels cannot replace those identities.

The report uses UTF-8, lexicographically sorted keys, compact separators, a
single trailing newline, and rejects NaN and infinity. Unknown fields are
rejected. The total JSON size is at most 1 MiB and the Markdown summary at most
64 KiB.

Other v1 bounds are:

- identifiers: 128 UTF-8 bytes;
- reason/detail/warning: 256 UTF-8 bytes per item;
- next-action summary: 512 UTF-8 bytes;
- warnings and reason codes: 32 each;
- selected-token or prefix preview: 64 IDs;
- layer collections: 4096 entries and indices 0–4095;
- intra-layer stage collections: 17 entries; and
- ledger events: 64 KiB each.

Execution is also bounded: the default per-stage timeout is 1800 seconds and
cannot exceed 7200 seconds; captured subprocess output is at most 1 MiB;
temporary disk use is at most 1 GiB; in-memory artifact buffering is at most
64 MiB; and handled termination receives a 10-second grace window. Unbounded
overrides are prohibited and limit exhaustion fails closed.

`next_action` is null for `PASS` and exactly one bounded object for every other
verdict. It is guidance, never evidence. Numeric confirmation is explicitly
`not_performed` or `incomplete`; absence is not inferred. Both
`confirmed_divergence_at_checkpoint` and `confirmed_first_divergence` are always
null in core v1. Conditional M6 numeric confirmation requires an explicit
versioned contract amendment and cannot be enabled by implementation code alone.

Raw prompt/generated text, raw tensor values, full forced-prefix arrays,
absolute model paths, and absolute temporary paths are prohibited.

## 6. Stage state

The final report contains exactly these stages in this order:

```text
preflight
reference_original_execution
candidate_original_execution
pass0_calibration
pass1_token_localization
pass2_prefix_policy_reproduction
pass3a_discovery
bounded_recapture
pass3b_authoritative_localization
pass4_intra_layer_localization
aggregation
cleanup
```

Each stage has exactly one terminal state: `executed`, `not_applicable`,
`blocked`, or `failed`. `pending` and `running` are ledger-only transient facts
and cannot appear in a final report.

- `executed` requires a canonical result digest and evidence tier.
- `not_applicable` requires a bounded reason.
- `blocked` requires a canonical blocker stage and bounded reason.
- `failed` requires a bounded failure class and reason.

An executed diagnostic stage cannot depend on a failed or blocked stage.
Aggregation is the exception that must execute over terminal success and
failure states so that a bounded customer result can be emitted. Equal-token
early stop marks Pass 2 through Pass 4 and recapture `not_applicable`.

## 7. Attempts and append-only ledger

Every product execution is classified before it starts as
`development_debugging` or `verification_acceptance`. Interactive commands
default to the former. Acceptance requires an explicitly frozen manifest and
clean-state preflight.

Attempt IDs use `lisa1:<32 lowercase hex>` with 128 bits of OS randomness.
Retries always receive a new ID. A retry may name `supersedes_attempt_id`, but
an attempt is never resumed or reused. Acceptance does not consume debugging
results as acceptance evidence.

The private mode-0600 JSON Lines ledger is append-only and has contiguous
sequence numbers starting at zero. Its events are `attempt_started`,
`stage_started`, `stage_finished`, `cleanup_observed`, and `attempt_finished`.
Each event has exactly `sequence`, `attempt_id`, `workflow_classification`,
`event`, `timestamp_utc`, and `payload`; timestamps use RFC 3339 UTC seconds.
The containing run directory is mode 0700. The ledger contains no raw text, raw
tensors, credentials, or absolute private paths. Silent retry and line
replacement are prohibited.

## 8. Cleanup, timeout, interruption, and residue

Cleanup is mandatory on normal and handled-error paths. Timeout fails the
active stage and blocks dependent stages. When a trustworthy report remains
possible, timeout yields `INCONCLUSIVE`. Handled interruption records the
active stage as failed and performs scoped cleanup before report publication.

After `SIGKILL`, host crash, power loss, kernel failure, or filesystem failure,
cleanup is best effort only and residue is `unknown` unless observed otherwise.
Unknown is never inferred as zero or healthy.

Cleanup status is one of `success`, `failed`, `partial`, `retained_debug`, or
`not_applicable`. Residue status is one of `none_observed`, `present`, `unknown`,
or `retained_debug`. A normal cleanup warning does not alter a valid semantic
verdict. Report-emission failure, confidentiality hard failure, or invalidated
required evidence produces `HARNESS_ERROR`.

Stale runs are detected and warned about on startup but never deleted
automatically. Explicit cleanup must first validate owner, mode, lock, and
heartbeat state.

## 9. Forced-prefix run-report design

The forced-prefix artifact channel is design-frozen by M0 and remains
unimplemented until M3. The current C CLI rejection of `--forced-prefix` with
`--report-json` and Pass 0's `artifact_supported = false` therefore remain
accurate after M0.

M3 may allow the combination only when the orchestrator supplies complete
source-binding metadata. The C run report must compute the digest and count from
the prefix actually applied. The additive `forced_prefix` object contains:

```text
mode
applied
token_count
token_ids_sha256
prefix_start_generated_step
prefix_end_generated_step_exclusive
target_generated_token_step
runtime_checkpoint_step
prompt_token_count
context_position
selection_policy
selection_policy_sha256
source_pass0_artifact_sha256
source_original_run_report_sha256
source_pass1_artifact_sha256
source_localization_ref_sha256
```

V1 requires a non-empty prefix of at most 64 IDs, a policy identity of
`raw_greedy` or `lis_policy_modified_greedy_v1` plus its SHA-256, generated
range `[0, token_count)`, target generated step equal to `token_count`, runtime
checkpoint step `N + 1`, and context position equal to prompt count plus prefix
count. The exact Pass 0 artifact is also bound. Raw token IDs are omitted by
default.

The policy digest is SHA-256 over UTF-8 sorted compact JSON with one trailing
newline and domain `lis.selection_policy/v1`. The profile records selection
mode, repetition-penalty decimal text, and structural-token-suppression state;
therefore a policy label alone cannot establish policy continuity.

The Python binder must cross-check the actual digest/count, Pass 1 prefix,
original role-specific run report, localization reference, step mapping,
context, input, configuration, binary, and policy. Missing SHA, mismatch,
wrong-side binding, request-only evidence, and a prefill construction standing
in for the decode boundary all fail closed. `artifact_set_id` remains
association evidence only.

## 10. Evidence ceiling and non-claims

The following report values are mandatory `false` in core v1:

```text
tensor_equality
numeric_divergence_confirmed
first_divergence_confirmed
whole_runtime_equivalence
```

Specifically:

- bounded digest equality is not tensor equality;
- bounded digest mismatch is not numeric confirmation;
- a suspect interval is not confirmed first divergence;
- partial coverage is not whole-runtime equivalence;
- request-only evidence is not an independent rerun;
- a prefill construction is not the required decode boundary;
- `next_action` is not evidence; and
- missing coverage is not inferred equal.

Pass 3 and Pass 4 local statuses and non-claims remain authoritative. Product
aggregation cannot strengthen them.

## 11. Machine-readable contract index

The following block mirrors the corresponding fields in
`product_contract_v1.json` and is checked value-for-value.

<!-- LIS-VERIFY-CONTRACT-INDEX-BEGIN -->
```json
{
  "contract_status": "approved",
  "implementation_status": "contract_only",
  "contract_version": "lis.verify.product_contract/v1",
  "schema": "lis.verification_report/v1",
  "kind": "verification_report",
  "report_version": "1.0",
  "customer_verdicts": ["PASS", "REGRESSION", "INCONCLUSIVE", "UNSUPPORTED", "HARNESS_ERROR"],
  "execution_policies": ["default", "require_supported"],
  "workflow_classifications": ["development_debugging", "verification_acceptance"],
  "stage_states": ["executed", "not_applicable", "blocked", "failed"],
  "cleanup_statuses": ["success", "failed", "partial", "retained_debug", "not_applicable"],
  "residue_statuses": ["none_observed", "present", "unknown", "retained_debug"],
  "aggregation_actions": ["continue", "stop", "inherit", "retain_proven_regression_and_block_localization", "not_applicable"],
  "report_top_level_fields": ["schema", "kind", "report_version", "attempt", "command", "verdict", "reason_codes", "policy_result", "identities", "token_comparison", "localization", "coverage", "numeric_confirmation", "evidence", "stages", "next_action", "warnings", "cleanup"],
  "identity_fields": ["source_sha256", "binary_sha256", "model_sha256", "config_sha256", "input_sha256", "runtime_sha256", "backend_sha256"],
  "evidence_nonclaims": ["tensor_equality", "numeric_divergence_confirmed", "first_divergence_confirmed", "whole_runtime_equivalence"],
  "bounds": {
    "max_report_bytes": 1048576,
    "max_summary_bytes": 65536,
    "max_identifier_bytes": 128,
    "max_detail_bytes": 256,
    "max_next_action_bytes": 512,
    "max_warnings": 32,
    "max_reason_codes": 32,
    "max_token_id_preview": 64,
    "max_layer_collection": 4096,
    "max_intra_layer_stages": 17,
    "max_ledger_event_bytes": 65536
  },
  "resource_limits": {
    "default_stage_timeout_seconds": 1800,
    "max_stage_timeout_seconds": 7200,
    "max_subprocess_output_bytes": 1048576,
    "max_temp_disk_bytes": 1073741824,
    "max_in_memory_artifact_bytes": 67108864,
    "termination_grace_seconds": 10,
    "unbounded_override_allowed": false,
    "limit_exhaustion_fails_closed": true
  }
}
```
<!-- LIS-VERIFY-CONTRACT-INDEX-END -->

## 12. Implementation transition

M1 has consumed this contract to implement packaging, parsing, report models,
canonical bundle publication, deterministic rendering, state-machine
scaffolding, private workspace/ledger handling, and bounded execution. M2 will
connect the seeded model-free evidence adapter. M3 consumes the forced-prefix
design and connects the real backend/runtime adapters. No milestone may relax
or reinterpret M0 values in implementation code. A required semantic change
needs an explicit contract version amendment, debugging verification, and a new
clean acceptance attempt.

The M1 production runner registry is deliberately empty. A valid but not-yet-
connected mode exits 2 before attempt creation and emits no report; it does not
invent unavailable source, binary, model, or input identities. Model-free M1
integration tests use an explicitly injected runner with synthetic source-bound
fixture identities. This development boundary is replaced by the M2/M3 mode
adapters, not treated as a customer verdict.
