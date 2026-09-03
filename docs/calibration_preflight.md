# Calibration Preflight (P1 Pass 0)

- Status: Implemented (model-free Python, `tools/lis_verify/`).
- Artifact: `kind: "calibration_preflight"` under schema `lis.execution_artifact/v1`.
- Scope: Pass 0 of the P1 differential-verification system. Passes 1–4 are
  implemented bounded diagnostic stages. The Pass 5 product contract is frozen,
  while its unified report and orchestration remain unimplemented; numeric
  confirmation remains conditional future work. See
  `docs/differential_verification.md` and `docs/lis_verify_contract.md`.

## What Pass 0 is

Pass 0 is the **calibration gate**. Given two LIS executions (a reference and a
candidate) and a declared comparison mode, it decides whether they are
*semantically comparable* before Pass 1 attempts token localization or any
numeric comparison. It exists so later passes never mistake a **decode-policy**,
**prompt-boundary**, **model-config-binding**, or **numeric-policy** difference
for a runtime tensor divergence, and never overclaim an external (HuggingFace)
oracle that LIS cannot currently satisfy.

It is **model-free**: it reads existing `run_report` (and optional
`decode_trace`) artifacts plus a checked-in LIS build calibration profile. It
runs no model, reads no tensors, and emits no divergence verdict.

### Why it is needed (F1/F3)

Three properties of the current LIS build make naïve comparison unsafe; Pass 0
encodes them as machine-readable calibration facts:

- **F1 — always-on repetition penalty.** `srcs/cli/driver.c` hard-codes
  `LIS_CLI_REPETITION_PENALTY = 1.2f`, applied every decode step with no flag to
  disable it. Structural token suppression is likewise always on. LIS "greedy"
  is really *policy-modified greedy*.
- **F3 — `rms_norm_eps` not runtime-bound on the Llama path.** `lis_rms_norm`
  has no eps parameter and the reference kernel hard-codes `1.0e-5f`; only Qwen3
  binds `config.rms_norm_eps`.
- **Numeric policy** — BF16/F16 KV write rounding is not verified round-to-
  nearest-even, and FMA/reduction order are backend-defined.

These facts live in the build calibration profile
(`tools/test_fixtures/calibration/lis_build_profile.json`), because none of them
are recoverable from a `run_report` artifact.

## Outputs

Pass 0 produces a `calibration_preflight` artifact with `comparison_mode`,
`comparison_eligibility` (`comparable` / `limited_comparison` / `incompatible`),
`pass0_verdict` (`comparison_allowed` / `limited_comparison_allowed` /
`comparison_blocked`), per-domain `calibration_status`, the four domain objects
(`decode_policy_identity`, `tokenizer_boundary`, `config_semantics`,
`numeric_policy`), `oracle_eligibility`, `reason_codes`, `warnings`,
`blocking_reasons`, and `verdict_strength_limit`. It also yields a
`Pass0GateDecision` consumed by Pass 1.

## Calibration domains

| Domain | Key questions | Representative reason codes |
|---|---|---|
| Decode policy | Same policy on both sides? Raw greedy or policy-modified? | `incompatible_decode_policy`, `policy_modified_greedy`, `decode_policy_not_raw` |
| Tokenizer / prompt boundary | Direct token IDs or text? Array-equal vs digest-only? | `confidence_downgrade_text_prompt_boundary`, `prompt_token_array_missing`, `input_token_divergence` |
| Config semantics | Is `rms_norm_eps` runtime-bound? Config fingerprints match? | `rms_norm_eps_runtime_unbound`, `config_semantics_uncalibrated`, `config_fingerprint_mismatch` |
| Numeric policy | KV write rounding verified? FMA/reduction backend-defined? | `kv_write_rounding_unverified`, `numeric_policy_uncalibrated`, `tolerance_caveat` |
| Oracle scope | Internal differential vs HF oracle eligibility | `hf_default_greedy_ineligible`, `external_oracle_ineligible`, `forced_prefix_report_json_channel_missing` |

## Reason-code severity and the aggregator

The reason-code registry (`reason_codes.py`) stores `code -> (domain,
base_severity)` and is **context-free**. Mode-specific escalation is performed by
the aggregator (`pass0.py`):

- `external_oracle_ineligible` is `informational` by default and escalates to a
  **block** only when the declared mode is `external_semantic` (Mode C).
- `config_fingerprint_mismatch` is `downgrade` by default and escalates to a
  **block** outside the Mode B `configuration_equivalence` submode (where a
  config difference is the intended subject).
- `prompt_token_array_missing` / `prompt_token_identity_unverified` escalate to a
  **block** in Mode C (external oracle requires array-strong identity).

Verdict resolution: any block-effective code → `comparison_blocked`; otherwise
any downgrade-effective code → `limited_comparison_allowed`; otherwise
`comparison_allowed`. Oracle-scope codes are informational with respect to the
internal eligibility verdict — they bound oracle scope without downgrading the
internal LIS differential.

## Comparison modes

- **Mode A — `backend_differential`** (scalar vs optimized dispatch). MVP.
- **Mode B — `runtime_differential`** (build/commit/config A vs B), with submodes
  including `configuration_equivalence`. MVP.
- **Mode C — `external_semantic`** (LIS vs external implementation). **Deferred**:
  blocked in the MVP.

`ComparisonMode` string spellings are owned by the contract fixture
(`comparison_modes`); the implementation must not rename them. Mode C is exactly
`external_semantic`.

## Oracle distinctions (do not conflate)

- `lis_internal_backend_differential` — the MVP's primary capability.
- `hf_default_greedy` — eligible only for a raw-greedy, fully calibrated run with
  array-equal prompt identity. **Always false** for the current LIS build.
- `hf_forced_token_runtime` — a structured object: `potentially_eligible` but
  `artifact_supported = false`, because `--forced-prefix` and `--report-json` are
  mutually exclusive today. Pass 5 M0 freezes an additive source-bound report
  design for later M3 implementation; M0 does not change this current false
  implementation status.
- `oracle_scope` — capped at `internal_lis_only` in the MVP.

Internal differential success is never serialized as external semantic
correctness.

## MVP ceiling

`verdict_strength_limit` has no first-divergence member. The strongest Pass 0
verdict is `comparison_allowed`; the strongest downstream ceiling it authorizes
is `checkpoint_confirmation_allowed`. Pass 0 must never enable
`confirmed_first_divergence`.

## Integration

- **Pass 1** consumes `Pass0GateDecision`. It must trust `comparison_mode`,
  `decode_policy_identity`, `tokenizer_boundary`, `oracle_eligibility`, and
  `verdict_strength_limit` and must not re-infer them. The Pass 1 MVP also
  requires an immutable `Pass0SourceBinding` created alongside the gate. If
  the binding fails or `gate.proceed` is false, Pass 1 stops before
  selected-token extraction.
- **Pass 6** embeds the serialized artifact and maps Pass 0 block reasons into
  existing `verification_report` reason codes via
  `calibration_preflight.block_reason_to_report_reason_code`. The calibration
  namespace does **not** modify the frozen `verification_report` enums.

## Tests

- `tools/tests/test_pass0_calibration.py` — behavior (decision matrix, domains,
  oracle eligibility, golden artifact, gate, report mapping).
- `tools/tests/test_calibration_contract.py` — fixture ↔ Markdown ↔ package
  parity for the additive `calibration_preflight` contract namespace.

Run with: `PYTHONPATH=tools python3 -m unittest discover -s tools/tests`.
