# Differential Verification & First-Divergence Locator — Design Contract

- Contract status: Approved
- Implemented diagnostic boundary: Pass 0 calibration, Pass 1 selected-token
  localization, Pass 2 prefix and policy reproduction, and Pass 3
  coverage-scoped Llama layer localization
- Frozen later-stage boundary: Pass 4 intra-layer contract only (P4-1);
  producer, runtime capture, parsing, localization execution, and result
  serialization are not implemented
- Base contract — Implementation status: Planned / Not yet implemented
- Contract version: 1.0

This document combines the original differential-verification design contract
with additive implemented diagnostic stages. The base `verification_report`,
exhaustive tensor comparison, intra-layer localization, and confirmatory
verdict surfaces remain unimplemented. Their approved contracts are retained
for future work.

The implemented boundary includes token mismatch localization,
mismatch-boundary reproduction, source-bound producer artifacts, and
coverage-scoped Llama layer-output localization. These results are bounded
diagnostic evidence. They do not prove tensor equality or confirm the first
numeric divergence. Qwen3 inference remains supported only within its
documented runtime scope and does not provide the layer-output layout required
for layer localization.

The Pass 4 intra-layer contract is frozen separately so later producer and
Python work share one taxonomy, coordinate grammar, digest domain, and evidence
ceiling. This freeze is not an implemented localization capability: no
intra-layer producer fields are emitted, runtime capture is unavailable, and
no Pass 4 localization can currently be executed.

## 1. Status and Scope

This document defines the approved contracts for LIS differential verification.
It specifies comparison modes, checkpoint identity, evidence tiers, coverage
and divergence semantics, source binding, token localization,
mismatch-boundary reproduction, producer checkpoint artifacts, and
coverage-scoped layer localization.

Implemented capabilities are documented in the additive staged sections below.
They include model-free compatibility calibration, selected-token mismatch
localization, prefix and policy reproduction, the bounded C checkpoint-artifact
producer, and coverage-scoped Llama layer-output localization.

The following base or later-stage surfaces remain outside the implemented
boundary:

- binary full-tensor checkpoint capture,
- exhaustive tensor comparison,
- intra-layer capture, artifact parsing, and localization execution (the P4-1
  contract alone is frozen),
- confirmatory numeric or first-divergence verdicts,
- production of the base `verification_report` artifact,
- new differential-verification CLI commands or confirm-checkpoint flags,
- a `verify-diff` make target,
- LIS Inspect verification views,
- an external semantic adapter.

Comparison Modes A and B remain the intended minimum scope of the original base
contract. Mode C is deferred. Implemented localization remains bounded to
compatible, source-bound evidence and does not expand the documented LIS
runtime support envelope.

## 2. Authority and Fixture Role

This Markdown specification is the normative human-readable contract.
`tools/test_fixtures/differential_verification_contract.json` is the
machine-readable conformance oracle. Disagreement between the Markdown
specification and the JSON fixture is a contract-validation failure; neither
file silently overrides the other.

The top-level machine-readable index describes the original base
`verification_report` contract, whose implementation remains planned. Additive
namespaces later in the same fixture and document record the implemented staged
contracts. Their local status fields and marker blocks are authoritative for
those capabilities.

The marker-delimited JSON block below must remain byte-for-value identical to
the corresponding top-level fields in
`tools/test_fixtures/differential_verification_contract.json`. It is not an
implementation artifact and is unchanged by this documentation consistency
update.

<!-- CONTRACT-INDEX-BEGIN -->
```json
{
  "contract_status": "approved",
  "implementation_status": "planned",
  "comparison_modes": ["backend_differential", "runtime_differential", "external_semantic"],
  "mvp_modes": ["backend_differential", "runtime_differential"],
  "deferred_modes": ["external_semantic"],
  "mode_b_submodes": ["runtime_regression", "determinism_check", "configuration_equivalence", "precision_policy_comparison"],
  "phase_enum": ["prefill", "decode"],
  "dtype_enum": ["fp32", "bf16"],
  "model_family_enum": ["llama3_decoder", "qwen3_dense_decoder"],
  "canonical_stage_enum": ["embedding", "layer_input", "attention_norm_out", "q_projection", "k_projection", "v_projection", "qk_norm", "rope_out", "kv_append", "attention_output", "mlp_output", "final_norm", "logits"],
  "digest_algorithm_enum": ["fnv1a64", "sha256", "none"],
  "token_identity_levels": ["hash", "strong_digest", "array_equal"],
  "evidence_tier_enum": ["tier0_structural", "tier1_bounded", "tier2_exhaustive"],
  "evidence_completeness_enum": ["complete", "partial", "incomplete"],
  "divergence_state_enum": ["no_divergence_observed", "suspected_divergence", "confirmed_divergence_at_checkpoint", "confirmed_first_divergence", "inconclusive_partial_coverage", "inconclusive_prefix_reproduction_failure"],
  "comparison_outcome_enum": ["no_divergence_observed_on_coverage", "suspected_divergence_on_coverage", "divergence_confirmed_at_checkpoint", "first_divergence_confirmed", "bitwise_equal", "byte_difference_only", "inconclusive"],
  "overall_verdict_enum": ["verified_equivalent", "divergence_confirmed_at_checkpoint", "first_divergence_confirmed", "inconclusive"],
  "first_divergence_scope_enum": ["captured_checkpoints_only", "full_runtime"],
  "cleanup_status_enum": ["success", "failed", "partial", "retained_debug", "not_applicable"],
  "result_class_enum": ["pass", "documented_unsupported", "numeric_regression", "token_parity_regression", "benchmark_protocol_regression", "harness_configuration_error", "verification_inconclusive"],
  "reason_code_enum": ["equivalent_full_coverage", "equivalent_on_observed_coverage", "bitwise_identical", "unsupported_comparison", "incompatible_model_family", "unsupported_mode", "unsupported_model", "input_token_divergence", "prefix_reproduction_failed", "asymmetric_checkpoint", "requested_step_absent", "truncated_artifact", "malformed_artifact", "oversized_capture", "byte_difference_only", "temp_cleanup_failed", "structural_divergence", "nonfinite_divergence", "suspected_numeric_divergence", "confirmed_numeric_divergence_at_checkpoint", "confirmed_first_divergence", "token_selection_divergence", "reference_run_failed", "candidate_run_failed", "incomplete_exhaustive_comparison", "interrupted_verification", "partial_evidence_inconclusive"],
  "tolerance_profile_names": ["exact", "f32-reference", "f32-simd", "bf16-simd", "external-fp32", "external-bf16"],
  "threshold_status_enum": ["uncalibrated_default", "calibrated", "deprecated"],
  "coverage_fields": ["target_generated_token_step", "runtime_checkpoint_step", "total_layers", "captured_layers", "captured_stages", "layer_coverage_complete", "stage_coverage_complete", "element_comparison_complete", "prefix_reproduction_verified", "first_divergence_scope"],
  "exit_code_mapping": {"pass": 0, "documented_unsupported": 0, "harness_configuration_error": 2, "verification_inconclusive": 3, "token_parity_regression": 4, "numeric_regression": 5}
}
```
<!-- CONTRACT-INDEX-END -->

## 3. Project Constraints

This contract is bound by the existing LIS support envelope:

- CPU-only, single-process, offline inference runtime.
- Causal decoder-only models: plain-RoPE Llama 3.x subset and narrow Qwen3
  Dense BF16 path only.
- Greedy decode only, batch 1 on real-forward paths.
- No GPU backends, serving, sampling frameworks, chat templates, or broad
  Qwen-family support.
- Existing artifact schema string: `lis.execution_artifact/v1`.
- Existing public result classes are preserved. The contract adds exactly one
  additive class, `verification_inconclusive`.
- `llama.c` and `qwen3.c` remain architecturally parallel and independent.
- The C runtime is the implemented bounded-evidence producer for
  source-associated execution and Llama checkpoint artifacts. Implemented
  Python tooling interprets those artifacts for token localization, boundary
  reproduction, and coverage-scoped layer localization. The base
  `verification_report` producer and exhaustive comparator remain
  unimplemented.

## 4. Comparison Modes

| Mode | Name | Definition | Scope |
|---|---|---|---|
| A | `backend_differential` | scalar (`LIS_SIMD=0`) vs optimized dispatch, same binary | Minimum viable |
| B | `runtime_differential` | build/commit/config A vs B | Minimum viable |
| C | `external_semantic` | LIS vs offline external implementation over identical tokens | Deferred |

Modes A and B must pass the compatibility and prompt-identity gates before any
numeric comparison. Mode C requires explicit token-prefix injection and external
framework provenance and is outside the minimum viable scope.

Mode B submodes:

- `runtime_regression`
- `determinism_check`
- `configuration_equivalence`
- `precision_policy_comparison`

`precision_policy_comparison` allows dtype-path differences within one
compatible model family, using a precision-appropriate tolerance profile. It is
not ordinary exact regression equivalence.

## 5. Compatibility Gate

Differential verification never numerically compares across model families or
incompatible configurations/fingerprints. The canonical stage taxonomy is
reporting-only and does not make two runs compatible.

Rejected numeric-comparison cases include:

- different model families,
- incompatible architecture constants,
- unrelated model fingerprints,
- incompatible prompt/input identity,
- unsupported precision-path combinations.

Cross-family canonical-stage equality is not compatibility evidence.

## 6. Token and Checkpoint Step Semantics

Generated token indexing and runtime checkpoint indexing are distinct.

Rules:

- prefill `runtime_checkpoint_step = 0`,
- `generated_token_step` is zero-based for decode/generated token arrays,
- decode `runtime_checkpoint_step = generated_token_step + 1`,
- Pass-1 first mismatching generated token index `N` maps to decode
  `runtime_checkpoint_step = N + 1`,
- reports for decode checkpoint comparison must carry both fields,
- checkpoint identity must distinguish `generated_token_step` from
  `runtime_checkpoint_step`,
- avoid ambiguous use of `step` where either explicit field is meant.

Examples:

| Phase | generated_token_step | runtime_checkpoint_step |
|---|---:|---:|
| prefill | null | 0 |
| decode | 0 | 1 |
| decode | 17 | 18 |

Pass-2 prefix verification for a target generated token `N` verifies generated
prefix `[0..N-1]`, prior generated-token count `N`, context/position alignment,
and checkpoint capture at runtime checkpoint step `N + 1`.

## 7. Checkpoint Identity

Checkpoint identity fields:

- `mode`
- `submode`
- `phase`
- `generated_token_step` (nullable for prefill)
- `runtime_checkpoint_step`
- `model_family`
- `layer` (nullable in JSON; binary header uses a global sentinel)
- `canonical_stage`
- `source_checkpoint_name`
- `execution_ordinal`
- `dtype`
- `shape`

Planned exhaustive confirmation targets one existing emitted checkpoint from the
current runtime. Arbitrary-layer/stage capture is planned for later slices and
is not yet implemented.

## 8. Current Checkpoint Coverage and Planned Taxonomy

Current emitted checkpoint coverage is sparse and is separate from the
family-neutral planned taxonomy.

Current Llama coverage includes bounded diagnostics from current
`lis_checkpoint_diagnostic` call sites, including:

- `embedding`,
- selected layer inputs and layernorm outputs for layers 0, 1, 7, and 8 as
  currently emitted,
- selected Q/K/V projection outputs, RoPE outputs, attention outputs, residuals,
  MLP outputs, and final layer outputs for the currently hardcoded layers,
- additional diagnostic names such as per-head Q/K/V views, attention scores,
  attention probabilities, attention context, layer-7 weights, dynamic
  `layer.<n>.output`, `final_norm`, and `logits`.

Current Qwen3 coverage is intentionally sparse:

- `qwen3.embedding`,
- `qwen3.layer.0.q_after_q_norm`,
- `qwen3.layer.0.k_after_k_norm`,
- `qwen3.final_norm`,
- `qwen3.logits`.

Current Qwen3 does not emit arbitrary-layer projection, RoPE, attention-output,
or MLP checkpoints. Those are planned for later slices and must not be treated
as current functionality.

Canonical stages remain family-neutral for reporting, ordering, schema, and
future Inspect views:

- `embedding`
- `layer_input`
- `attention_norm_out`
- `q_projection`
- `k_projection`
- `v_projection`
- `qk_norm`
- `rope_out`
- `kv_append`
- `attention_output`
- `mlp_output`
- `final_norm`
- `logits`

Source names remain family-specific. Non-emitted canonical stages are planned
coverage, not current coverage. The taxonomy does not permit cross-family
numeric comparison.

## 9. Autoregressive and Prefix-Reproduction Semantics

Comparison order:

1. artifact and input compatibility,
2. prompt token identity,
3. Pass-2 prefix reproduction,
4. prefill checkpoint evidence when relevant,
5. decode checkpoint evidence at `runtime_checkpoint_step = N + 1`,
6. final norm and logits for the target generated token,
7. selected token for the target generated token,
8. stop ordinary comparison after the first mismatching generated token.

Pass-2 gate failure:

- result class: `verification_inconclusive` when the invocation was well formed,
- reason code: `prefix_reproduction_failed`,
- comparison outcome: `inconclusive`,
- overall verdict: `inconclusive`,
- divergence state: `inconclusive_prefix_reproduction_failure`,
- numeric comparison is prohibited.

If the prefix failure was caused by malformed arguments or missing files, use
`harness_configuration_error` instead.

## 10. Token Identity and Digest Semantics

Prompt identity levels:

| Level | Meaning | Strongest supported use |
|---|---|---|
| `hash` | count plus existing `fnv1a64` digest | checkpoint confirmation only |
| `strong_digest` | `sha256` digest over the ordered token ID array | strong identity evidence, not exact array equality |
| `array_equal` | element-wise equality of the ordered token ID array | required for `confirmed_first_divergence` |

The contract adopts `sha256` as the only strong digest.

Representation:

```text
sha256:<64 lowercase hexadecimal characters>
```

Existing FNV-1a remains a compatibility fingerprint. SHA-256 equality is strong
digest identity evidence, not element-wise token-array equality. SHA-256
mismatch is not tolerance-aware numeric evidence. Hash-only mismatch cannot
produce `confirmed_divergence_at_checkpoint`.

## 11. Evidence Tiers and Divergence States

Evidence tiers:

- `tier0_structural`
- `tier1_bounded`
- `tier2_exhaustive`

Tier 0 structural evidence covers identity, phase, generated/runtime step
identity, layer, canonical stage, source name, shape, dtype, missing evidence,
and nonfinite flags.

Tier 1 bounded evidence includes summary statistics, deterministic samples, and
optional hashes. Bounded samples never confirm numeric divergence.

Tier 2 exhaustive selected-checkpoint comparison compares every aligned element
of the whole selected checkpoint tensor and emits bounded aggregates only.
Incomplete exhaustive comparison cannot produce a confirmed verdict.

Divergence states:

- `no_divergence_observed`
- `suspected_divergence`
- `confirmed_divergence_at_checkpoint`
- `confirmed_first_divergence`
- `inconclusive_partial_coverage`
- `inconclusive_prefix_reproduction_failure`

`confirmed_divergence_at_checkpoint` requires complete exhaustive numeric
comparison, all elements compared, active tolerance applied, at least one
beyond-tolerance element, compatible inputs, verified prefix reproduction, and
no malformed or incomplete evidence.

`confirmed_first_divergence` additionally requires exhaustive predecessor
equivalence, complete layer/stage coverage, prompt token `array_equal`, and
deep-verification resource controls. Under the minimum viable coverage,
`confirmed_first_divergence` is always `null`.

## 12. Coverage and Verdict Semantics

Coverage fields:

- `target_generated_token_step`
- `runtime_checkpoint_step`
- `total_layers`
- `captured_layers`
- `captured_stages`
- `layer_coverage_complete`
- `stage_coverage_complete`
- `element_comparison_complete`
- `prefix_reproduction_verified`
- `first_divergence_scope`

Allowed first-divergence scopes:

- `captured_checkpoints_only`
- `full_runtime`

Under the minimum viable partial coverage:

- layer and stage coverage are incomplete,
- first-divergence scope is `captured_checkpoints_only`,
- `confirmed_first_divergence` is null,
- whole-runtime equivalence is not claimable.

## 13. Result Classes, Reason Codes, and Exit Codes

Existing public result classes are preserved:

- `pass`
- `documented_unsupported`
- `numeric_regression`
- `token_parity_regression`
- `benchmark_protocol_regression`
- `harness_configuration_error`

The contract adds exactly one additive result class:

- `verification_inconclusive`

Use `verification_inconclusive` for verification that completed sufficiently to
report an inconclusive outcome but does not fit the existing classes. Examples:
hash-only byte mismatch without numeric confirmation, incomplete exhaustive
comparison, partial evidence that cannot support a stronger verdict,
prefix-reproduction failure not caused by malformed invocation, and interrupted
verification with a valid partial report.

`verification_inconclusive` reason codes must map to overall verdict
`inconclusive`; they must not map to confirmed numeric outcomes.

Differential-verification exit codes:

| Exit code | Meaning |
|---:|---|
| 0 | pass or documented supported skip |
| 2 | harness configuration error |
| 3 | verification inconclusive |
| 4 | token parity regression |
| 5 | numeric regression |

`documented_unsupported` may exit 0 when the harness completed correctly and
produced a documented skip. Cleanup warnings do not alter an otherwise valid
result. Cleanup failure may change exit status only if report emission failed,
confidentiality policy requires hard failure, or required evidence cannot be
trusted.

## 14. Tolerance Profiles

This contract specifies profile names, required metadata fields, and versioning
rules. It does not validate numeric thresholds; threshold calibration is planned
later work.

Profile names:

- `exact`
- `f32-reference`
- `f32-simd`
- `bf16-simd`
- `external-fp32`
- `external-bf16`

Required profile fields:

- `name`
- `profile_version`
- `threshold_status`
- `abs_floor`
- `rel_band`
- `dtype_policy`
- `stage_overrides`
- `reduction_noise_policy`
- `calibration_provenance`
- `calibration_dataset_id`
- `holdout_validation`
- `created_by`
- `created_at`

Allowed `threshold_status` values:

- `uncalibrated_default`
- `calibrated`
- `deprecated`

Current numeric values are uncalibrated defaults. A profile cannot claim
`calibrated` without provenance and holdout validation. Calibrated profile
changes require a new profile version. Each report stores the selected profile
plus resolved thresholds.

## 15. Temporary Tensor Binary Header

This contract does not implement binary serialization. The Version-1 temporary
tensor header is fixed-width and little-endian.

Header properties:

- magic: `LISB`
- major version: `1`
- explicit `header_length`
- explicit `flags`
- explicit dtype enum
- explicit phase enum
- rank limited to 4
- fixed `shape[4]`
- element count
- payload byte count
- generated token step
- runtime checkpoint step
- layer index or `UINT64_MAX` global sentinel
- execution ordinal
- mode enum
- Mode-B submode enum or `none` sentinel
- model-family identifier
- canonical stage identifier
- source checkpoint name in a 96-byte fixed field
- digest algorithm enum
- optional SHA-256 payload digest
- reserved extension bytes

Validation rules:

- reject wrong magic,
- reject unsupported major version,
- reject invalid header length,
- reject unknown required flags,
- reject invalid rank,
- reject nonzero unused shape dimensions where prohibited,
- reject element-count overflow,
- reject payload-byte-count mismatch,
- reject file length not exactly equal to header plus payload,
- reject truncated files,
- reject oversized files,
- reject checkpoint identity mismatch,
- reject dtype or shape mismatch across comparison inputs,
- reject malformed string fields,
- reserved fields must be zero for version 1; future readers may define
  extension semantics by version and flag, but current readers must reject
  nonzero reserved bytes unless explicitly supported.

## 16. Secure Temporary Transport and Cleanup

Threat model: local single-user minimum viable scope. The design does not
attempt to defend against a compromised same-UID producer or kernel/filesystem
failure.

Path contract:

- Python harness creates an unpredictable per-run directory.
- Directory mode is `0700`.
- Directory is validated as a directory owned by the current UID.
- Harness chooses unpredictable, distinct reference and candidate basenames.
- C producer receives the validated directory plus basename.
- C joins directory and basename without accepting traversal components.
- Reject `..`, absolute basenames, directory separators, and any path outside
  the run directory.

Secure creation:

- producer uses exclusive creation,
- no silent overwrite,
- no symlink following where supported,
- file mode `0600`,
- producer validates the opened file with `fstat`,
- comparator validates regular file type, owner UID, mode, hard-link count
  `st_nlink == 1`, expected header, expected identity, and exact file length.

`O_NOFOLLOW` portability rule: use it where available. Where unavailable, the
private owned `0700` directory and post-open metadata checks are mandatory.
Validation is never silently skipped.

Cleanup:

- mandatory on all normal and handled-error paths,
- best effort only after abnormal termination,
- no guarantee after `SIGKILL`, host crash, power loss, kernel failure, or
  filesystem failure,
- numeric verdict remains valid if cleanup fails after comparison,
- report includes cleanup operational status,
- cleanup failure emits a warning and reason code such as `temp_cleanup_failed`.

Stale-run policy:

- default stale threshold: 24 hours,
- destructive cleanup is an explicit cleanup command,
- startup may detect and warn but must not delete by default,
- run directory contains a metadata marker with format version, UID, PID,
  creation time, last heartbeat time, and debug-retain flag,
- active-run protection uses a lock or heartbeat,
- owner and mode are validated before deletion,
- retained debug directories are distinguished,
- every deleted or skipped directory is represented in an audit summary,
- warnings use safe run identifiers and do not include model/input/temp paths.

## 17. Verification Report Schema

Planned artifact shape:

```text
schema = "lis.execution_artifact/v1"
kind = "verification_report"
report_version = "1.0"
```

Required top-level fields:

- `schema`
- `kind`
- `report_version`
- `mode`
- `submode`
- `result_class`
- `reason_code`
- `comparison_outcome`
- `overall_verdict`
- `evidence_completeness`
- `tolerance_profile`
- `sources`
- `prompt_token_identity`
- `prefix_reproduction`
- `coverage`
- `comparison`
- `suspected_first_divergence`
- `confirmed_divergence_at_checkpoint`
- `confirmed_first_divergence`
- `warnings`
- `cleanup_status`

Nullable fields are explicit in the fixture. `confirmed_first_divergence` is
always null in the minimum viable scope. Reports do not contain raw tensor
values, full prompt token arrays by default, raw text, absolute temporary paths,
model paths, or temporary artifact paths.

## 18. Deep Verification and Mode C Deferred Contracts

`confirmed_first_divergence` is outside the minimum viable scope and requires
deep-verification mode. Resource controls:

- deep-verification mode required,
- explicit user acknowledgment required,
- `max_tensor_bytes`,
- `max_checkpoints`,
- `max_temp_disk_bytes`,
- sequential predecessor recapture by default,
- no unbounded retention,
- early stop on first confirmed divergence,
- fail closed when resource limits are exceeded,
- the minimum viable scope always sets `confirmed_first_divergence = null`.

Mode C remains deferred and cannot appear in the minimum viable capability list.
Future Mode-C reports require:

- explicit injected token prefix identity,
- target generated-token step,
- runtime checkpoint step,
- token position alignment,
- external framework name/version,
- adapter name/version,
- model provenance,
- config provenance,
- tokenizer provenance,
- dependency-environment identity,
- dtype/accumulation policy,
- deterministic-execution settings,
- hook/checkpoint mapping version,
- confidentiality policy,
- no raw prompt/model leakage by default.

## 19. Minimum Viable Scope Boundaries

The minimum viable scope includes:

- Mode A backend differential,
- Mode B runtime differential,
- compatibility and prompt-identity gates,
- first mismatching-token localization,
- Pass-2 prefix verification,
- Tier-0/Tier-1 comparison over currently captured checkpoints,
- exhaustive Tier-2 comparison of one existing checkpoint,
- bounded verification report,
- secure temporary transport,
- model-free contract/comparator tests,
- model-backed manual verification.

The minimum viable scope excludes:

- Mode C,
- whole-runtime equivalence claims,
- cross-family numeric comparison,
- numeric confirmation from hash-only mismatch,
- `confirmed_first_divergence`.

Strongest minimum-viable verdict: `confirmed_divergence_at_checkpoint`.

## 20. Compatibility and Confidentiality

Implemented differential-verification surfaces are additive. Existing
`run_report` keys, `report.perf`, stderr prefixes, LIS Inspect `run_report`
support, Make targets, CI workflows, and prior artifact readers remain
compatible. The base `verification_report`, exhaustive temporary-tensor
transport, and later confirmatory surfaces remain governed by their approved
but unimplemented contracts.

Default reports contain bounded summaries and fingerprints only. Exhaustive
Tier-2 uses temporary full-tensor binaries that are created securely, validated,
streamed, and removed on normal and handled-error paths. Retention requires an
explicit debug option and warning.

## 21. Implementation Status

Implementation is staged. The original base `verification_report` and
exhaustive-confirmation contract remains planned, while the additive diagnostic
stages below are implemented.

Implemented:

- Pass 0 calibration preflight,
- Pass 1 selected-token mismatch localization,
- Pass 2 prefix and policy reproduction,
- producer-vNext source-associated artifacts and the versioned Llama
  layer-output checkpoint layout,
- Pass 3 coverage-scoped Llama layer localization.

Frozen contract only:

- Pass 4 P4-1 Llama/decode intra-layer taxonomy, coordinate, coverage,
  parent, interval, digest, artifact-identity, status/reason, and evidence
  ceiling contracts.

The implemented layer-localization result identifies the earliest observed
digest mismatch within validated common captured coverage and may report a
sparse suspect interval. It is bounded representation evidence, not exhaustive
numeric confirmation. Matching digests do not prove tensor equality. Qwen3 and
legacy layer-trace layouts remain unsupported for Pass 3.

Not implemented:

- binary full-tensor checkpoint capture,
- exhaustive tensor comparison,
- the Pass 4 producer, runtime intra-layer capture, runtime artifact parsing,
  localization algorithm/execution, result serializer, or public execution
  API,
- `confirmed_divergence_at_checkpoint` or `confirmed_first_divergence`
  production,
- production of the base `verification_report` artifact,
- runtime support for the base `verification_inconclusive` result class,
- new differential-verification CLI commands,
- a `verify-diff` make target,
- LIS Inspect verification views,
- the Mode-C external semantic adapter.

## 22. Calibration Preflight (Pass 0)

- Pass 0 status: Implemented (model-free Python, `tools/lis_verify/`).
- Artifact kind: `calibration_preflight` under schema `lis.execution_artifact/v1`.
- Artifact `contract_version` identifier: `differential_verification_contract_v1`
  (distinct from this contract's semantic version `1.0`).

Pass 0 is the calibration gate of the P1 differential-verification system. It
decides whether two LIS executions are semantically comparable *before* Pass 1
attempts token localization or any numeric comparison, so later passes never
mistake a decode-policy, prompt-boundary, config-binding, or numeric-policy
difference for a runtime tensor divergence. Pass 0 reads existing `run_report`
(and optional `decode_trace`) artifacts plus a checked-in LIS build calibration
profile; it runs no model and reads no tensors.

Calibration vocabulary is an additive namespace. It does **not** modify the
frozen `verification_report` enums (`reason_code_enum`, `result_class_enum`, and
their mappings); Pass 0 block reasons map *into* existing report reason codes via
`calibration_preflight.block_reason_to_report_reason_code`.

Key invariants (see also Corrections 1–4 in the implementation plan):

- `ComparisonMode` values are contract-owned: they must equal entries in
  `comparison_modes`. Mode C is exactly `external_semantic`.
- The reason-code registry stores base severity only; mode-specific escalation
  (e.g. `external_oracle_ineligible` blocking only in Mode C) is performed by the
  aggregator, listed in `aggregator_escalated_codes`.
- Forced-token runtime oracle is potential-only:
  `hf_forced_token_runtime.artifact_supported = false` while `--forced-prefix`
  and `--report-json` are mutually exclusive.
- Prompt-token array equality (`array_equal`) is distinct from digest-only
  identity (`digest_only`); only `array_equal` enables HF default-greedy
  eligibility.
- `verdict_strength_limit` has no first-divergence member; Pass 0 must never
  enable `confirmed_first_divergence`. The strongest Pass 0 verdict is
  `comparison_allowed`; the strongest downstream ceiling it authorizes is
  `checkpoint_confirmation_allowed`.

The machine-readable conformance facts live in
`tools/test_fixtures/differential_verification_contract.json` under the
`calibration_preflight` key. The block below mirrors that fixture for
Markdown/fixture consistency checks (see
`tools/tests/test_calibration_contract.py`).

<!-- CALIBRATION-INDEX-BEGIN -->
```json
{
  "schema": "lis.execution_artifact/v1",
  "kind": "calibration_preflight",
  "contract_version": "differential_verification_contract_v1",
  "comparison_eligibility_enum": [
    "comparable",
    "limited_comparison",
    "incompatible"
  ],
  "pass0_verdict_enum": [
    "comparison_allowed",
    "limited_comparison_allowed",
    "comparison_blocked"
  ],
  "verdict_strength_limit_enum": [
    "no_comparison",
    "token_localization_only",
    "checkpoint_confirmation_allowed"
  ],
  "selection_mode_enum": [
    "raw_greedy",
    "policy_modified_greedy"
  ],
  "prompt_boundary_enum": [
    "direct_token_ids",
    "text"
  ],
  "prompt_identity_evidence_enum": [
    "array_equal",
    "digest_only",
    "unverified",
    "divergent"
  ],
  "oracle_scope_enum": [
    "internal_lis_only",
    "internal_lis_and_runtime",
    "external_semantic"
  ],
  "calibration_domain_enum": [
    "decode_policy",
    "tokenizer_boundary",
    "config_semantics",
    "numeric_policy",
    "comparison_mode",
    "oracle_scope"
  ],
  "reason_severity_enum": [
    "block",
    "downgrade",
    "informational"
  ],
  "calibration_reason_code_enum": [
    "incompatible_decode_policy",
    "policy_modified_greedy",
    "decode_policy_not_raw",
    "decode_policy_uncalibrated",
    "tokenizer_boundary_uncalibrated",
    "prompt_token_array_missing",
    "prompt_token_identity_unverified",
    "input_token_divergence",
    "confidence_downgrade_text_prompt_boundary",
    "config_semantics_uncalibrated",
    "rms_norm_eps_runtime_unbound",
    "config_fingerprint_mismatch",
    "runtime_config_fingerprint_missing",
    "requires_fix_or_guard",
    "incompatible_model_family",
    "numeric_policy_uncalibrated",
    "kv_write_rounding_unverified",
    "fma_policy_backend_defined",
    "reduction_order_backend_defined",
    "tolerance_caveat",
    "external_oracle_ineligible",
    "hf_default_greedy_ineligible",
    "hf_forced_token_runtime_eligible",
    "internal_lis_differential_only",
    "oracle_scope_limited",
    "forced_prefix_report_json_channel_missing"
  ],
  "aggregator_escalated_codes": [
    "config_fingerprint_mismatch",
    "external_oracle_ineligible",
    "prompt_token_array_missing",
    "prompt_token_identity_unverified"
  ],
  "mvp": {
    "strongest_pass0_verdict": "comparison_allowed",
    "strongest_downstream_strength_limit": "checkpoint_confirmation_allowed",
    "enables_confirmed_first_divergence": false,
    "external_semantic_mode_blocked_in_mvp": true,
    "hf_forced_token_runtime_artifact_supported": false,
    "hf_default_greedy_requires_array_equal": true
  }
}
```
<!-- CALIBRATION-INDEX-END -->

## 23. Calibrated Compatibility and Token Localization (Pass 1)

- Pass 1 status: Implemented (model-free Python, `tools/lis_verify/`).
- Artifact kind: `token_localization` under schema
  `lis.execution_artifact/v1`.
- Artifact `contract_version` identifier:
  `differential_verification_contract_v1`.

Pass 1 consumes the authoritative `Pass0GateDecision`, verifies an immutable
per-side `run_report_sha256` source binding, and compares explicit generated
`selected_token_ids` arrays. It stops before selected-token extraction when
the source binding fails or Pass 0 blocks comparison.

The source digest is SHA-256 over deterministic compact UTF-8 JSON produced
from the parsed object with sorted object keys. It is not a hash of raw file
bytes. Duplicate JSON keys are malformed and are rejected before
canonicalization.

Pass 1 localizes either an exact token-ID mismatch or the boundary where one
exact sequence is a strict prefix of the other. Generated-token steps are
zero-based and decode runtime checkpoint steps use
`runtime_checkpoint_step = generated_token_step + 1`. Prompt tokens are never
part of this index.

Digest-only selected-token evidence cannot produce a concrete mismatch step or
Pass 2 reproduction prefix. The default serialized prefix cap is 64 token IDs;
longer exact prefixes remain in memory but serialize only count, range, and
SHA-256 with `availability: "exact_source_required"`.

`first_mismatch_found` is local output-selection evidence. It has no frozen
verification-report reason mapping in Pass 1. In particular, Pass 1 does not
use `token_selection_divergence`, whose frozen mapping implies checkpoint
confirmation, and does not claim `confirmed_divergence_at_checkpoint` or
`confirmed_first_divergence`.

The machine-readable conformance facts live in
`tools/test_fixtures/differential_verification_contract.json` under the
`token_localization` key. The block below mirrors that fixture for
Markdown/fixture consistency checks.

<!-- TOKEN-LOCALIZATION-INDEX-BEGIN -->
```json
{
  "schema": "lis.execution_artifact/v1",
  "kind": "token_localization",
  "contract_version": "differential_verification_contract_v1",
  "pass1_status_enum": [
    "comparison_blocked_by_pass0",
    "token_equivalent_on_observed_range",
    "first_mismatch_found",
    "input_token_divergence",
    "selected_token_array_missing",
    "selected_token_identity_unverified",
    "unsupported_comparison",
    "inconclusive"
  ],
  "mismatch_kind_enum": [
    "token_id_mismatch",
    "length_mismatch_or_early_termination"
  ],
  "selected_token_evidence_level_enum": [
    "array_exact",
    "digest_only",
    "metadata_only",
    "missing"
  ],
  "pass2_disposition_enum": [
    "ready",
    "not_required",
    "blocked_by_pass0",
    "blocked_by_evidence",
    "blocked_by_strength_limit"
  ],
  "prefix_availability_enum": [
    "embedded",
    "exact_source_required",
    "not_applicable"
  ],
  "pass1_reason_code_enum": [
    "pass1.selected_token_array_missing",
    "pass1.selected_token_identity_unverified",
    "pass1.selected_token_metadata_inconsistent",
    "pass1.gate_run_identity_inconsistent",
    "pass1.unsupported_run_artifact",
    "pass1.unsupported_batch_shape"
  ],
  "source_binding": {
    "mandatory": true,
    "mvp_transport": "Pass0SourceBinding",
    "identity_field": "run_report_sha256",
    "digest_algorithm": "sha256",
    "digest_prefix": "sha256:",
    "canonicalization": "parsed_json_sorted_keys_compact_utf8",
    "raw_file_bytes_hashed": false,
    "duplicate_json_keys": "malformed",
    "verify_before_selected_token_access": true
  },
  "step_mapping": {
    "generated_token_step_base": 0,
    "runtime_checkpoint_step_formula": "generated_token_step + 1",
    "equal_arrays_generated_token_step": null,
    "equal_arrays_runtime_checkpoint_step": null
  },
  "prefix_policy": {
    "default_embedded_token_cap": 64,
    "long_prefix_availability": "exact_source_required",
    "long_prefix_fields": [
      "generated_prefix_token_count",
      "generated_prefix_sha256",
      "prefix_start_generated_step",
      "prefix_end_generated_step_exclusive"
    ],
    "exact_prefix_retained_in_memory": true,
    "redacted_prefix_is_reproduction_material": false
  },
  "calibration_reference": {
    "default_full_embedding": false,
    "canonical_sha256_required": true,
    "selected_summary_fields": [
      "comparison_mode",
      "pass0_verdict",
      "comparison_eligibility",
      "prompt_identity_evidence",
      "verdict_strength_limit",
      "blocking_reasons",
      "oracle_scope"
    ]
  },
  "report_boundary": {
    "first_mismatch_is_local_evidence_only": true,
    "first_mismatch_report_reason_code": null,
    "prohibited_first_mismatch_reason_codes": [
      "token_selection_divergence"
    ],
    "confirmed_divergence_at_checkpoint": null,
    "confirmed_first_divergence": null,
    "frozen_verification_report_enums_modified": false
  },
  "artifact_required_fields": [
    "schema",
    "kind",
    "contract_version",
    "comparison_mode",
    "pass0_verdict",
    "comparison_eligibility",
    "source_binding",
    "pass1_status",
    "evidence_scope",
    "evidence_completeness",
    "compatibility",
    "selected_token_evidence",
    "token_localization",
    "prefix_for_reproduction",
    "pass2_disposition",
    "calibration_ref",
    "verdict_strength_limit",
    "reason_codes",
    "inherited_pass0_reason_codes",
    "blocking_reasons",
    "warnings"
  ]
}
```
<!-- TOKEN-LOCALIZATION-INDEX-END -->

## 24. Prefix and Policy Reproduction Verification (Pass 2)

- Pass 2 status: Implemented (model-free Python, `tools/lis_verify/`).
- Artifact kind: `prefix_policy_reproduction` under schema
  `lis.execution_artifact/v1`.
- Artifact `contract_version` identifier:
  `differential_verification_contract_v1`.

Pass 2 consumes `Pass1Result` as its primary semantic input and requires the
exact original reference and candidate `CanonicalRunReport`s as evidence
inputs. It verifies both original canonical report hashes against
`pass1.source_binding` before materializing either report or reading selected
tokens, binary identity, prompt count, batch size, thread count, runtime
metadata, or model/config/input identity.

The pass verifies the exact shared generated prefix `[0..N-1]`, binary/build
continuity, `runtime_checkpoint_step = generated_token_step + 1`, and derived
context/batch/sequence alignment. A thread count greater than one is recorded
as `thread_count_gt_1_determinism_caveat` but is not a local Pass 2 blocker.

`reproduction_verified` is evidence-tier-aware. At
`original_pair_boundary_consistent`, it means the source-bound original pair
is internally boundary-consistent; it does not claim a fresh rerun occurred.
Fresh paired rerun reports are required for `independent_rerun_verified`.
`reproduction_request_only` can never accompany `reproduction_verified`.

Computed checkpoint-step evidence only validates the Pass 1 step formula. It
does not establish that a materialized checkpoint artifact exists. Pass 2
does not read tensors, logits, activations, or numeric checkpoint values, and
does not emit confirmed-divergence or confirmed-first-divergence fields.

The block below is byte-for-value identical to the
`prefix_policy_reproduction` namespace in the machine-readable contract.

<!-- PREFIX-POLICY-REPRODUCTION-INDEX-BEGIN -->
```json
{
  "status": "implemented",
  "schema": "lis.execution_artifact/v1",
  "kind": "prefix_policy_reproduction",
  "contract_version": "differential_verification_contract_v1",
  "package": "tools/lis_verify",
  "model_free": true,
  "pass2_status_enum": [
    "reproduction_verified",
    "comparison_blocked_by_pass0",
    "token_localization_not_available",
    "no_mismatch_to_reproduce",
    "source_binding_inconsistent",
    "prefix_material_unavailable",
    "prefix_reproduction_failed",
    "decode_policy_reproduction_failed",
    "checkpoint_step_mapping_mismatch",
    "context_position_mismatch",
    "unsupported_reproduction_mode",
    "inconclusive"
  ],
  "pass2_reason_code_enum": [
    "pass2.source_binding_inconsistent",
    "pass2.pass1_status_not_reproducible",
    "pass2.prefix_material_unavailable",
    "pass2.prefix_digest_mismatch",
    "pass2.prefix_token_mismatch",
    "pass2.decode_policy_reproduction_failed",
    "pass2.checkpoint_step_mapping_mismatch",
    "pass2.context_position_mismatch",
    "pass2.unsupported_reproduction_mode",
    "pass2.reproduction_artifact_malformed",
    "pass2.verdict_strength_limit_blocks_reproduction"
  ],
  "reproduction_evidence_tier_enum": [
    "independent_rerun_verified",
    "original_pair_boundary_consistent",
    "reproduction_request_only"
  ],
  "pass3_disposition_enum": [
    "ready",
    "not_required",
    "blocked_by_pass0",
    "blocked_by_pass1_evidence",
    "blocked_by_reproduction"
  ],
  "checkpoint_step_evidence_enum": [
    "computed_from_pass1_step_mapping",
    "corroborated_by_trace_artifact"
  ],
  "source_binding": {
    "pass1_is_primary_semantic_input": true,
    "required_original_inputs": [
      "reference_original",
      "candidate_original"
    ],
    "original_identity_field": "run_report_sha256",
    "verify_both_originals_before_materialization": true,
    "verify_before_metadata_access": [
      "selected_token_ids",
      "binary_fingerprint",
      "prompt_token_count",
      "batch_size",
      "thread_count",
      "runtime_metadata",
      "model_config_input_fingerprints"
    ],
    "required_original_metadata": [
      "manifest.binary.fingerprint",
      "report.prompt_sequences[0].token_count",
      "manifest.runtime.batch_size",
      "manifest.runtime.thread_count",
      "model_fingerprint",
      "config_fingerprint",
      "input_fingerprint"
    ],
    "reproduction_inputs_are_paired": true,
    "reproduction_identity_fields": [
      "model_fingerprint",
      "config_fingerprint",
      "input_fingerprint"
    ],
    "missing_or_malformed_metadata_fails_closed": true
  },
  "prefix_policy": {
    "exact_token_ids_required": true,
    "redacted_prefix_requires_exact_source": true,
    "digest_only_is_not_exact_material": true,
    "prefix_start_generated_step": 0
  },
  "checkpoint_step_evidence": {
    "runtime_checkpoint_step_formula": "generated_token_step + 1",
    "default": "computed_from_pass1_step_mapping",
    "optional_trace": "corroborated_by_trace_artifact",
    "computed_implies_materialized_checkpoint": false,
    "numeric_trace_values_accessed": false
  },
  "determinism_caveats": {
    "thread_count_gt_1_warning": "thread_count_gt_1_determinism_caveat",
    "applies_to_original_or_reproduction": true,
    "pass2_blocking": false
  },
  "downstream_readiness": {
    "primary_gate": "pass3_disposition",
    "stronger_claims_also_check": "reproduction_evidence_tier",
    "ready_implies_independent_rerun": false
  },
  "report_boundary": {
    "reproduction_verified_is_local_evidence_only": true,
    "reproduction_request_only_may_be_verified": false,
    "prohibited_status_fields": [
      "confirmed_divergence_at_checkpoint",
      "confirmed_first_divergence"
    ],
    "frozen_verification_report_enums_modified": false
  },
  "artifact_required_fields": [
    "schema",
    "kind",
    "contract_version",
    "comparison_mode",
    "pass0_verdict",
    "comparison_eligibility",
    "pass1_status",
    "pass1_pass2_disposition",
    "source_binding",
    "source_binding_verified",
    "pass2_status",
    "reproduction_evidence_tier",
    "reproduction_verified_semantics",
    "target",
    "prompt_reproduction",
    "prefix_reproduction",
    "policy_reproduction",
    "checkpoint_step_reproduction",
    "context_reproduction",
    "pass3_disposition",
    "localization_ref",
    "verdict_strength_limit",
    "reason_codes",
    "inherited_pass1_reason_codes",
    "inherited_pass0_reason_codes",
    "blocking_reasons",
    "warnings"
  ],
  "mvp": {
    "actual_model_rerun_execution": false,
    "optional_trace_metadata_input": false,
    "numeric_checkpoint_comparison": false,
    "confirmation_verdict_fields_emitted": false
  }
}
```
<!-- PREFIX-POLICY-REPRODUCTION-INDEX-END -->

## 25. Producer-vNext Checkpoint Artifact Contract

The producer prerequisite is frozen before C/runtime and Pass 3 core
implementation. One CLI inference execution samples exactly 16 unmodified
bytes from the operating-system CSPRNG before inference or related artifact
emission and serializes them as `aset1:<32 lowercase hexadecimal
characters>`. The value is propagated unchanged to every related artifact
successfully emitted by that execution. Entropy failure aborts without a
fallback.

`artifact_set_id` is probabilistic same-execution association evidence only.
Canonical SHA-256 is artifact content identity; the semantic manifest is
execution/configuration compatibility evidence. A matching ID never bypasses
another source-binding link, including under a forced collision. FNV-1a
remains compatibility or bounded content-link evidence only.

Producer-vNext Llama traces use the explicit
`llama_layer_output_summary` version-1 layout. Coordinates, requested,
captured, missing-by-state coverage, execution ordinals, observed dtype,
element counts, available fields, and digest envelopes are serialized
directly. Semantic coordinates are never inferred only from `name`.
Duplicate logical coordinates fail before writing and are independently
rejected by Pass 3.

The checkpoint digest is SHA-256 over the frozen
`lis.checkpoint.fp32le/v1` stream: domain tag and zero byte, 64-bit
little-endian byte-length-prefixed version and role, 64-bit rank and shape,
length-prefixed observed dtype and byte order, element count, then logical
row-major binary32 little-endian observations. Finite and infinity bits are
preserved, signed zero is preserved, and all NaNs normalize to `0x7fc00000`.
It is computed only in explicit checkpoint diagnostics. Digest mismatch is
bounded observed-representation evidence; digest match is not mathematical
tensor equality.

Legacy layer traces remain valid execution artifacts but are unsupported
Pass 3 layouts. Qwen3 is `unsupported_checkpoint_layout`; its Q/K
normalization summaries may not masquerade as layer output.

The block below is byte-for-value identical to the
`producer_checkpoint_artifact` namespace in the machine-readable contract.

<!-- PRODUCER-CHECKPOINT-ARTIFACT-INDEX-BEGIN -->
```json
{
  "status": "frozen",
  "schema": "lis.execution_artifact/v1",
  "contract_version": "differential_verification_contract_v1",
  "related_artifact_kinds": [
    "run_report",
    "layer_trace",
    "decode_trace"
  ],
  "artifact_set_id": {
    "field": "artifact_set_id",
    "format": "aset1:<32 lowercase hexadecimal characters>",
    "regex": "^aset1:[0-9a-f]{32}$",
    "random_source": "operating_system_csprng",
    "random_bytes": 16,
    "bytes_transformed_before_hex": false,
    "generation_point": "once_at_cli_inference_execution_start_before_inference_or_related_artifact_emission",
    "propagation": "unchanged_to_every_related_artifact_successfully_emitted_by_that_cli_execution",
    "separate_execution_policy": "independently_sampled",
    "failure_policy": "fail_closed_before_inference_and_related_artifact_emission",
    "fallback_allowed": false,
    "prohibited_fallbacks": [
      "time",
      "pid",
      "counter",
      "fnv1a64",
      "settings",
      "predictable_data"
    ],
    "evidence_semantics": "probabilistic_same_cli_execution_association",
    "absolute_uniqueness_claim": false,
    "content_identity": false,
    "configuration_compatibility": false,
    "matching_id_sufficient_for_source_binding": false
  },
  "evidence_roles": {
    "artifact_set_id": "probabilistic_same_cli_execution_association_evidence",
    "canonical_sha256": "artifact_content_identity",
    "semantic_manifest": "execution_configuration_compatibility_evidence",
    "fnv1a64": "compatibility_or_bounded_content_link_evidence_only"
  },
  "source_binding": {
    "all_links_mandatory": true,
    "chain": [
      "supplied_canonical_pass2_artifact",
      "canonical_pass2_artifact_sha256",
      "typed_pass2_result_artifact_coherence",
      "pass2_bound_role_run_report_sha256",
      "supplied_run_report_canonical_sha256",
      "matching_artifact_set_id",
      "matching_semantic_manifest_identity",
      "matching_target_runtime_checkpoint_step",
      "canonical_layer_trace_sha256"
    ],
    "forced_id_collision_bypasses_other_links": false,
    "fnv1a64_satisfies_chain": false
  },
  "checkpoint_layout": {
    "outer_kind": "layer_trace",
    "layout_name": "llama_layer_output_summary",
    "layout_version": 1,
    "supported_model_family": "llama3_decoder",
    "tensor_role": "layer_output",
    "stage_order": 0,
    "ordering_semantics": "runtime_step_layer_stage_ordinal",
    "duplicate_coordinate_policy": "reject_artifact_before_write",
    "logical_coordinate_fields": [
      "runtime_checkpoint_step",
      "layer_index",
      "tensor_role",
      "batch_index",
      "sequence_index",
      "stage_order"
    ],
    "order_validation_fields": [
      "runtime_checkpoint_step",
      "layer_index",
      "stage_order",
      "execution_ordinal"
    ],
    "required_layout_fields": [
      "layout_name",
      "layout_version",
      "runtime_checkpoint_step",
      "tensor_role",
      "stage_order",
      "ordering_semantics",
      "total_layer_count",
      "requested_coordinates",
      "captured_coordinates",
      "missing_coordinates",
      "available_summary_fields",
      "digest_contract",
      "duplicate_coordinate_policy"
    ],
    "required_entry_fields": [
      "runtime_checkpoint_step",
      "layer_index",
      "tensor_role",
      "batch_index",
      "sequence_index",
      "stage_order",
      "execution_ordinal",
      "observed_dtype",
      "element_count",
      "phase",
      "name",
      "shape",
      "available_summary_fields",
      "min",
      "max",
      "mean",
      "l2",
      "nan",
      "inf",
      "digest"
    ],
    "coverage_state_enum": [
      "captured",
      "not_captured",
      "unsupported",
      "malformed",
      "unexpectedly_absent"
    ],
    "coverage_rules": {
      "requested_is_explicit": true,
      "captured_is_explicit": true,
      "missing_is_explicit_and_stateful": true,
      "dictionary_absence_is_coverage_evidence": false,
      "captured_coordinates_equal_summary_coordinates": true,
      "requested_partition": "captured_coordinates union missing_coordinates",
      "requested_partition_disjoint": true,
      "coordinates_unique": true,
      "ordinals_strictly_increasing": true,
      "malformed_requested_coordinate_fails_artifact": true
    },
    "source_name_conformity": "layer.<layer_index>.output",
    "batch_index": 0,
    "sequence_index": 0,
    "observed_dtype": "fp32",
    "full_tensor_payload_allowed": false
  },
  "digest_contract": {
    "algorithm": "sha256",
    "version": "lis.checkpoint.fp32le/v1",
    "domain_tag_utf8": "LIS_CHECKPOINT_DIGEST",
    "domain_terminator_hex": "00",
    "length_prefix_encoding": "unsigned_64_bit_little_endian_byte_count",
    "integer_encoding": "unsigned_64_bit_little_endian",
    "stream_fields_in_order": [
      "domain_tag_and_zero_byte",
      "length_prefixed_digest_version_utf8",
      "length_prefixed_tensor_role_utf8",
      "rank",
      "shape_dimensions",
      "length_prefixed_observed_dtype_utf8",
      "length_prefixed_byte_order_utf8",
      "element_count",
      "row_major_fp32_values"
    ],
    "tensor_role": "layer_output",
    "observed_dtype": "fp32",
    "byte_order": "little",
    "row_order": "logical_row_major",
    "canonicalization": "ieee754-binary32-le;canonical-qnan;preserve-signed-zero",
    "canonical_qnan_bits_hex": "7fc00000",
    "finite_bits": "preserved",
    "infinity_bits": "preserved",
    "signed_zero_bits": "preserved",
    "nan_bits": "all_nan_encodings_normalized_to_7fc00000",
    "digest_value_regex": "^sha256:[0-9a-f]{64}$",
    "input_representation": "post_observation_fp32",
    "diagnostic_mode_only": true,
    "normal_inference_digest_work": false,
    "failure_suppresses_partial_artifact": true,
    "match_semantics": "no_digest_difference_observed_for_aligned_representation",
    "mismatch_semantics": "bounded_observed_representation_digest_mismatch",
    "mathematical_tensor_equality_claim": false,
    "collision_free_claim": false,
    "precision_path_policy": "underlying_precision_paths_must_match_exactly_for_mvp"
  },
  "compatibility": {
    "outer_schema_additive": true,
    "legacy_artifacts_remain_valid_execution_artifacts": true,
    "legacy_pass3_behavior": "unsupported_checkpoint_layout",
    "semantic_coordinates_may_be_inferred_from_name": false,
    "unknown_layout_version_behavior": "unsupported_checkpoint_layout",
    "qwen3_mvp_behavior": "unsupported_checkpoint_layout",
    "qwen3_qk_norm_adaptation_allowed": false
  },
  "required_schema_examples": [
    "llama_layer_trace_vnext_schema_examples.json",
    "legacy_layer_trace_without_binding.json"
  ],
  "required_digest_vectors": "checkpoint_digest_test_vectors.json",
  "dependency_gate": {
    "p3_p1_green_unlocks": [
      "P3-P2-through-P3-P7",
      "P3-C1-through-P3-C8"
    ],
    "p3_i_requires": [
      "full_P3-P_green",
      "full_P3-C_green",
      "actual_C_artifact_validation"
    ],
    "post_freeze_change_requires_coordinated_contract_revision": true
  }
}
```
<!-- PRODUCER-CHECKPOINT-ARTIFACT-INDEX-END -->

## 26. Coverage-Scoped Layer Localization Contract

Pass 3 consumes both the typed `Pass2Result` and its supplied canonical
`prefix_policy_reproduction` artifact. The supplied artifact, rendered
under the existing canonical JSON contract, is the only Pass 2 content
identity used by Pass 3. Typed/artifact coherence and both complete source
binding chains pass before any trace summary is accessed.

The MVP compares only explicit producer-vNext Llama `layer_output`
coordinates in validated common comparable coverage. SHA-256 checkpoint
digests are bounded observed-representation evidence: mismatch localizes an
earliest observable suspect interval, while match proves neither tensor
equality nor whole-runtime equivalence. Sparse intervals preserve unobserved
layers. Qwen3 and legacy layouts are unsupported.

Pass 3 emits only local conservative downstream dispositions. It does not
certify Pass 4 or Pass 5 readiness and has no automatic frozen success
mapping.

The block below is byte-for-value identical to the
`coverage_scoped_layer_localization` namespace in the machine-readable
contract.

<!-- COVERAGE-SCOPED-LAYER-LOCALIZATION-INDEX-BEGIN -->
```json
{
  "status": "frozen",
  "schema": "lis.execution_artifact/v1",
  "kind": "layer_localization",
  "contract_version": "differential_verification_contract_v1",
  "pass3_status_enum": [
    "observable_mismatch_found",
    "no_mismatch_in_captured_coverage",
    "comparison_blocked_by_pass2",
    "insufficient_common_coverage",
    "source_binding_inconsistent",
    "checkpoint_alignment_inconsistent",
    "checkpoint_artifact_missing",
    "checkpoint_summary_malformed",
    "comparison_policy_unavailable",
    "unsupported_checkpoint_layout",
    "inconclusive"
  ],
  "pass3_reason_code_enum": [
    "pass3.pass2_not_ready",
    "pass3.reproduction_request_only",
    "pass3.source_binding_inconsistent",
    "pass3.pass2_artifact_identity_inconsistent",
    "pass3.pass2_object_artifact_inconsistent",
    "pass3.run_report_canonical_sha_inconsistent",
    "pass3.artifact_set_id_inconsistent",
    "pass3.binding_metadata_missing",
    "pass3.runtime_checkpoint_step_mismatch",
    "pass3.insufficient_common_coverage",
    "pass3.reference_checkpoint_missing",
    "pass3.candidate_checkpoint_missing",
    "pass3.checkpoint_alignment_inconsistent",
    "pass3.duplicate_checkpoint_coordinate",
    "pass3.checkpoint_summary_malformed",
    "pass3.summary_field_missing",
    "pass3.checkpoint_digest_incompatible",
    "pass3.comparison_policy_unavailable",
    "pass3.unsupported_checkpoint_layout",
    "pass3.asymmetric_coverage",
    "pass3.observable_mismatch_found",
    "pass3.no_mismatch_in_captured_coverage"
  ],
  "coverage_state_enum": [
    "captured",
    "not_captured",
    "unsupported",
    "malformed",
    "unexpectedly_absent"
  ],
  "alignment_status_enum": [
    "aligned",
    "shape_mismatch",
    "dtype_mismatch",
    "precision_path_mismatch",
    "stage_mismatch",
    "batch_mismatch",
    "sequence_mismatch",
    "model_family_mismatch",
    "duplicate_coordinate"
  ],
  "summary_field_disposition_enum": [
    "exact",
    "tolerance_aware",
    "informational_only",
    "unsupported"
  ],
  "summary_evidence_level_enum": [
    "tier0_structural",
    "tier1_bounded_exact",
    "tier1_bounded_calibrated",
    "tier1_bounded_digest",
    "unavailable"
  ],
  "downstream_disposition_enum": [
    "blocked",
    "exploratory_localization_only",
    "suspect_interval_available"
  ],
  "pass2_evidence": {
    "required_inputs": [
      "typed_pass2_result",
      "supplied_canonical_pass2_artifact"
    ],
    "canonical_identity_field": "pass2_artifact_sha256",
    "coherence_field": "pass2_object_artifact_coherence_verified",
    "canonical_identity_source": "existing_pass2_artifact_serializer_and_canonical_json_contract",
    "typed_object_has_independent_identity": false,
    "coherence_scope": "every_pass3_relevant_serialized_field",
    "prohibited_identity_mechanisms": [
      "hash_repr",
      "pickle_or_object_memory_hash",
      "ad_hoc_pass3_serialization",
      "selected_field_identity_hash",
      "pass3_local_pass2_result_serializer"
    ]
  },
  "gate_order": [
    "A1_typed_pass2_readiness",
    "A2_canonical_pass2_artifact_validation_hashing_and_coherence",
    "B_both_complete_source_binding_chains",
    "C_trace_headers_layout_coordinates_coverage_and_order",
    "D_bounded_checkpoint_summaries_and_digest_evidence"
  ],
  "source_binding": {
    "all_links_mandatory": true,
    "chain": [
      "supplied_canonical_pass2_artifact",
      "canonical_pass2_artifact_sha256",
      "typed_pass2_result_artifact_coherence",
      "pass2_bound_role_run_report_sha256",
      "supplied_run_report_canonical_sha256",
      "matching_artifact_set_id",
      "matching_semantic_manifest_identity",
      "matching_target_runtime_checkpoint_step",
      "canonical_layer_trace_sha256"
    ],
    "summary_access_before_complete_binding": false,
    "matching_artifact_set_id_alone_sufficient": false,
    "fnv1a64_satisfies_chain": false
  },
  "input_contract": {
    "trace_kind": "layer_trace",
    "layout_name": "llama_layer_output_summary",
    "layout_version": 1,
    "model_family": "llama3_decoder",
    "tensor_role": "layer_output",
    "stage_order": 0,
    "batch_index": 0,
    "sequence_index": 0,
    "ordering_semantics": "runtime_step_layer_stage_ordinal",
    "full_tensor_payload_allowed": false,
    "legacy_layout_status": "unsupported_checkpoint_layout",
    "qwen3_mvp_status": "unsupported_checkpoint_layout"
  },
  "coordinate_contract": {
    "join_key_fields": [
      "runtime_checkpoint_step",
      "layer_index",
      "tensor_role",
      "batch_index",
      "sequence_index",
      "stage_order"
    ],
    "alignment_check_fields": [
      "shape",
      "element_count",
      "observed_dtype",
      "precision_path",
      "model_family",
      "source_name_conformity",
      "execution_ordinal"
    ],
    "validated_order_fields": [
      "runtime_checkpoint_step",
      "layer_index",
      "stage_order",
      "execution_ordinal"
    ],
    "malformed_order_may_be_silently_sorted": false,
    "duplicate_logical_coordinates_allowed": false,
    "alignment_failure_is_numeric_mismatch": false
  },
  "coverage_contract": {
    "sets": [
      "reference_requested",
      "reference_captured",
      "candidate_requested",
      "candidate_captured",
      "common_captured",
      "reference_only",
      "candidate_only",
      "common_comparable",
      "missing_by_state"
    ],
    "asymmetric_coverage_is_mismatch": false,
    "asymmetric_coverage_is_metadata": true,
    "empty_common_comparable_status": "insufficient_common_coverage",
    "dictionary_absence_is_coverage_evidence": false
  },
  "decision_policy": {
    "mvp_decision_field": "checkpoint_digest",
    "digest_algorithm": "sha256",
    "digest_version": "lis.checkpoint.fp32le/v1",
    "digest_canonicalization": "ieee754-binary32-le;canonical-qnan;preserve-signed-zero",
    "precision_path_eligibility": "exact_match_required",
    "digest_mismatch_semantics": "observed_representation_digest_mismatch",
    "digest_match_semantics": "no_digest_difference_observed_for_aligned_representation",
    "digest_match_proves_tensor_equality": false,
    "digest_collision_free_claim": false,
    "uncalibrated_default_allowed": false,
    "missing_compatible_decision_field_status": "comparison_policy_unavailable",
    "decision_semantics_field": "observed_representation_digest_mismatch",
    "evidence_level": "tier1_bounded_digest"
  },
  "localization_contract": {
    "fields": [
      "last_observed_equivalent_layer",
      "first_observed_mismatching_layer",
      "earliest_observable_suspect_layer",
      "suspect_interval"
    ],
    "dense_example": "(7, 8]",
    "sparse_example": "(4, 8]",
    "entry_example": "[entry, L]",
    "sparse_unobserved_layers_are_explicit": true,
    "earliest_observable_is_confirmed_first_divergence": false,
    "no_mismatch_scope": "captured_common_comparable_coverage_only"
  },
  "artifact_required_fields": [
    "schema",
    "kind",
    "contract_version",
    "pass2_artifact_sha256",
    "pass2_object_artifact_coherence_verified",
    "pass2_evidence",
    "source_binding",
    "checkpoint_artifact_binding_verified",
    "target",
    "coverage",
    "comparisons",
    "localization",
    "evidence",
    "pass3_status",
    "downstream_disposition",
    "reason_codes",
    "inherited_pass2_reason_codes",
    "inherited_pass1_reason_codes",
    "inherited_pass0_reason_codes",
    "warnings",
    "semantic_limits"
  ],
  "downstream_boundary": {
    "success_has_automatic_frozen_mapping": false,
    "blocked_mapping_requires_exact_semantic_equivalence": true,
    "pass4_or_pass5_readiness_certified": false,
    "frozen_verification_report_enums_modified": false
  },
  "prohibited_claims": [
    "confirmed_first_divergent_layer",
    "confirmed_divergence_at_checkpoint",
    "confirmed_first_divergence",
    "mathematical_tensor_equality",
    "whole_runtime_equivalence",
    "pass4_ready",
    "pass5_ready"
  ],
  "semantic_limits": {
    "mismatch_is_bounded_localization_evidence_only": true,
    "match_is_representation_scoped_collision_limited_evidence": true,
    "full_tensor_comparison_performed": false,
    "stage_localization_performed": false,
    "numeric_confirmation_performed": false
  }
}
```
<!-- COVERAGE-SCOPED-LAYER-LOCALIZATION-INDEX-END -->

## 27. Coverage-Scoped Intra-Layer Localization Contract Freeze

Pass 4 P4-1 is frozen as a contract-only dependency for later implementation.
It does not add an implemented producer or localization surface. In particular:

- the C producer is not implemented;
- runtime intra-layer capture is unavailable;
- `intra_layer_checkpoint_layout` and `intra_layer_trace` are not emitted;
- no runtime-artifact parser or localization algorithm is implemented;
- localization execution and Pass 4 result serialization are unavailable;
- no CLI flag or public Pass 4 execution API exists.

The future outer evidence artifact retains schema
`lis.execution_artifact/v1` and kind `layer_trace`. The additive producer-side
field names are `intra_layer_checkpoint_layout` and `intra_layer_trace`.
Enabled recapture manifests will conditionally bind
`intra_layer_checkpoints_enabled`, `intra_layer_target_layer`, and capture
profile `semantic_layer_and_intra_v1`. The future local result retains the same
schema and uses kind `intra_layer_localization`, contract version
`differential_verification_contract_v1`, and namespace
`coverage_scoped_intra_layer_localization`.

### Frozen Llama stage order

The v1 layout identity is:

```text
layout_name = llama_intra_layer_summary
layout_version = 1
stage_taxonomy = lis.llama.intra_layer_stages/v1
model_family = llama3_decoder
phase = decode
ordering_semantics = runtime_step_layer_stage_ordinal
duplicate_coordinate_policy = reject_artifact_before_write
```

The exact logical stage order is:

| Order | Stage ID and tensor role |
|---:|---|
| 0 | `layer_input` |
| 1 | `attention_norm_output` |
| 2 | `query_projection_output` |
| 3 | `key_projection_output` |
| 4 | `value_projection_output` |
| 5 | `rope_query_output` |
| 6 | `rope_key_output` |
| 7 | `attention_scores` |
| 8 | `attention_probabilities` |
| 9 | `attention_context` |
| 10 | `attention_output_projection` |
| 11 | `post_attention_residual` |
| 12 | `mlp_norm_output` |
| 13 | `mlp_gate_projection` |
| 14 | `mlp_up_projection` |
| 15 | `mlp_gated_activation` |
| 16 | `mlp_down_projection` |

For v1, `stage_id == tensor_role`, `execution_ordinal == stage_order`, and
batch and sequence indices are zero. The order is a contracted logical
execution/dependency order, not proof of causal order among sibling Q/K/V
branches.

`layer_output` is deliberately absent from this local taxonomy. It remains the
separate inherited boundary `parent:layer_output`, with evidence origin
`authoritative_pass3`. It is never part of requested, captured, missing, or
common local coverage and is never rehashed as a Pass 4 checkpoint.

### Coordinates, coverage, and intervals

The immutable local coordinate contains runtime checkpoint step, layer index,
stage ID, tensor role, batch index, sequence index, token position, stage
order, and execution ordinal. The step is at least 1; layer and token position
are non-negative; all integers reject booleans and implicit coercion. Unknown
stages, role/order mismatches, duplicates, and out-of-order input fail closed.
Malformed order is never silently sorted.

Each side declares the exact ordered 17-stage requested list. Captured
coordinates are an ordered unique subset and missing coordinates are their
stateful ordered complement, using the existing `CoverageState` values
`captured`, `not_captured`, `unsupported`, `malformed`, and
`unexpectedly_absent` unchanged. Common captured coverage preserves reference
order; common comparable coverage is an aligned subset; and one-sided fields
are exact captured-set differences. Different requested lists are
`unsupported_intra_layer_layout`, not ordinary sparse coverage.

The suspect interval is tagged. Its start is either the inclusive virtual
`selected_layer_entry` or an exclusive local coordinate. Its inclusive end is
either the first local mismatch with origin `pass4_local`, or the exact Pass 3B
layer-output coordinate with origin `authoritative_pass3`. The interval lists
every requested local stage between its bounds that is absent from common
comparable coverage. Local and inherited endpoint fields are mutually
exclusive.

### Pass 3 parent and source identity

Pass 3A is discovery provenance only and never authorizes Pass 4 evidence.
After recapture and upstream rebuild, Pass 3B is the authoritative parent
binding the exact extended traces. A valid observed mismatch is eligible; a
valid no-mismatch parent is `not_applicable`; valid blocked or malformed
parents are `comparison_blocked_by_pass3`; unsupported family, layout, or
evidence policy is `unsupported_parent`; and Pass 3A/Pass 3B target or semantic
drift is `parent_revalidation_inconsistent`.

The future canonical Pass 3 wrapper must call the existing Pass 3 serializer,
existing canonical JSON and SHA-256 helpers, and strict duplicate-key-rejecting
JSON parser. It may not hash `repr` or pickle, hash selected fields, define a
separate Pass 4 serializer for Pass 3, or rerun Pass 3 while parsing an
artifact.

`artifact_set_id` remains association evidence only. It cannot substitute for
the canonical Pass 3 artifact, Pass 2 artifact, per-side run report, exact
layer trace, or semantic-manifest identities.

### Contextual digest domain

The new frozen digest identity is:

```text
algorithm = sha256
version = lis.checkpoint.intra_layer.fp32le/v1
domain_tag = LIS_INTRA_LAYER_CHECKPOINT_DIGEST
observed_dtype = fp32
byte_order = little
canonicalization = ieee754-binary32-le;canonical-qnan;preserve-signed-zero
```

The existing Pass 3 version `lis.checkpoint.fp32le/v1` is unchanged.
The canonical byte stream is the domain tag followed by one zero byte, then:

```text
digest_version
layout_name
layout_version
stage_taxonomy
model_family
precision_path
phase
runtime_checkpoint_step
layer_index
stage_id
tensor_role
batch_index
sequence_index
token_position
stage_order
execution_ordinal
rank
shape_dimensions
observed_dtype
byte_order
element_count
logical_row_major_FP32_bytes
```

Every string after the domain tag is
`u64le(UTF-8 byte length) || UTF-8 bytes`. Every integer is unsigned 64-bit
little-endian. Finite and infinity bits are preserved, signed zeros remain
distinct, and all NaNs canonicalize to `0x7fc00000`. Rank zero, empty tensors,
zero dimensions, shape overflow, element-count mismatch, and malformed
strided views are rejected.

The committed literal vectors cover finite values, signed zero, infinities,
NaNs, shape/stage-role/layer/step/token/phase/precision domain separation,
string framing, row-major order, strided equivalence, overflow, and malformed
input. Their expected canonical streams and SHA-256 values are fixed fixture
literals rather than generated approvals.

### Local status algebra and evidence ceiling

The frozen local statuses are:

```text
observable_intra_layer_mismatch_found
mismatch_bounded_to_inherited_closing_boundary
not_applicable
comparison_blocked_by_pass3
insufficient_common_intra_layer_coverage
source_binding_inconsistent
checkpoint_alignment_inconsistent
checkpoint_summary_malformed
comparison_policy_unavailable
unsupported_parent
unsupported_intra_layer_layout
parent_revalidation_inconsistent
```

Both bounded mismatch statuses map to `suspect_interval_available`;
`not_applicable` maps to itself; unsupported parent/layout statuses map to
`unsupported`; parent revalidation inconsistency maps to `inconclusive`; and
all other failures map to `blocked`. Every `pass4.*` reason code and its allowed
status set is frozen in the authoritative P4-1 fixture and parity-tested.
Successful Pass 4 results have no automatic mapping into frozen global
verification success enums.

Comparison evidence uses `tier1_bounded_digest`. Every result must serialize
all of these as false:

```text
numeric_divergence_confirmed
true_first_divergence_confirmed
root_cause_identified
tensor_equality_proved
complete_intra_layer_coverage_proved
operation_level_localization_performed
exhaustive_confirmation_performed
automatic_frozen_success_mapping
```

The authoritative machine-readable P4-1 contract is
`tools/test_fixtures/intra_layer_localization/pass4_contract.json`. Its compact
documentation index is mirrored below.

<!-- COVERAGE-SCOPED-INTRA-LAYER-LOCALIZATION-INDEX-BEGIN -->
```json
{
  "status": "frozen",
  "scope": "P4-1_contract_only",
  "layout_name": "llama_intra_layer_summary",
  "layout_version": 1,
  "stage_taxonomy": "lis.llama.intra_layer_stages/v1",
  "stage_ids": [
    "layer_input",
    "attention_norm_output",
    "query_projection_output",
    "key_projection_output",
    "value_projection_output",
    "rope_query_output",
    "rope_key_output",
    "attention_scores",
    "attention_probabilities",
    "attention_context",
    "attention_output_projection",
    "post_attention_residual",
    "mlp_norm_output",
    "mlp_gate_projection",
    "mlp_up_projection",
    "mlp_gated_activation",
    "mlp_down_projection"
  ],
  "inherited_boundary": "parent:layer_output",
  "digest_version": "lis.checkpoint.intra_layer.fp32le/v1",
  "result_kind": "intra_layer_localization",
  "evidence_level": "tier1_bounded_digest",
  "producer_implemented": false,
  "runtime_capture_available": false,
  "localization_execution_available": false
}
```
<!-- COVERAGE-SCOPED-INTRA-LAYER-LOCALIZATION-INDEX-END -->
