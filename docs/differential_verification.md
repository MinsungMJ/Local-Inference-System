# Differential Verification & First-Divergence Locator — Design Contract

- Contract status: Approved
- Implementation status: Planned / Not yet implemented
- Contract version: 1.0

This document is the approved technical design contract for LIS Differential
Verification & First-Divergence Locator. The contract is approved and stable as
a design specification. The feature itself is not yet implemented: no runtime,
CLI, comparator, verification, or artifact surface described here currently
exists. Every mechanism, command, artifact, and runtime capability below is an
approved design requirement, a planned interface, or explicitly not yet
implemented, as marked.

## 1. Status and Scope

This document defines the approved contract for LIS differential verification.
It specifies comparison modes, checkpoint identity, divergence semantics,
evidence tiers, tolerance-profile structure, hash-only mode, secure
temporary-tensor transport, and the planned `verification_report` artifact
shape.

The contract is approved. The feature is planned and not yet implemented. This
document does not implement, and the current runtime does not provide:

- binary checkpoint capture,
- an exhaustive tensor comparator,
- a differential harness,
- new differential-verification CLI flags,
- a `verify-diff` make target,
- LIS Inspect verification views,
- an external semantic adapter.

Scope of the design: comparison Modes A and B are the minimum viable scope.
Mode C (external semantic) is deferred. The contract is a design specification,
not a current support claim, and does not change the currently documented LIS
support envelope.

## 2. Authority and Fixture Role

This Markdown specification is the normative human-readable contract.
`tools/test_fixtures/differential_verification_contract.json` is the
machine-readable conformance oracle. Disagreement between the Markdown
specification and the JSON fixture is a contract-validation failure; neither
file silently overrides the other.

The approved status applies to the design contract only. It is not a claim that
any differential-verification runtime, CLI, comparator, or artifact surface is
implemented.

The marker-delimited JSON block below exists only to make Markdown/fixture
consistency checks deterministic. It is not an implementation artifact.

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
- The C runtime is the planned bounded-evidence producer; the planned Python
  comparison code interprets evidence and writes `kind:"verification_report"`.

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

All surfaces in this contract are additive and planned for future
implementation. Existing `run_report` keys, `report.perf`, existing stderr
prefixes, LIS Inspect `run_report` support, Make targets, CI workflows, and
README support claims are unchanged by this design contract.

Default reports contain bounded summaries and fingerprints only. Exhaustive
Tier-2 uses temporary full-tensor binaries that are created securely, validated,
streamed, and removed on normal and handled-error paths. Retention requires an
explicit debug option and warning.

## 21. Implementation Status

The contract is approved as a design specification. The feature is planned and
not yet implemented. The following are defined by this contract but do not
currently exist in the runtime, CLI, build, or artifact surface:

- binary checkpoint capture and the temporary tensor binary writer,
- the exhaustive tensor comparator,
- production of the `verification_report` artifact,
- runtime support for the `verification_inconclusive` result class,
- new differential-verification CLI flags (such as confirm-checkpoint flags),
- a `verify-diff` make target,
- LIS Inspect verification views,
- the Mode-C external semantic adapter.

Pass 0 calibration preflight and Pass 1 selected-token localization are
implemented as model-free Python tooling. Passes 2–6 and the runtime/numeric
surfaces listed above remain planned.

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
