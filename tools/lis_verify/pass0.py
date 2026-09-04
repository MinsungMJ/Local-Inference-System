"""Pass 0 aggregator: ``run_calibration_preflight``.

Runs the four domain calibrators, computes oracle eligibility, applies
mode-specific severity escalation (Correction 2), and resolves the
eligibility / verdict / verdict-strength-limit per the decision matrix. Never
fails open: missing metadata downgrades or blocks, never upgrades to
``comparable``.
"""

from __future__ import annotations

from typing import Optional

from .domains import (
    calibrate_config,
    calibrate_decode_policy,
    calibrate_numeric,
    calibrate_tokenizer,
)
from .inputs import PreflightInputs
from .model import (
    CalibrationPreflightArtifact,
    CalibrationReasonCode as C,
    CalibrationStatus,
    ComparisonEligibility,
    ComparisonMode,
    ForcedTokenRuntimeEligibility,
    ModeBSubmode,
    OracleEligibility,
    OracleScope,
    Pass0Verdict,
    PromptIdentityEvidence,
    ReasonSeverity,
    VerdictStrengthLimit,
)
from .reason_codes import AGGREGATOR_ESCALATED, base_severity, order_codes


_FORCED_TOKEN_STATUS = "eligible_with_source_bound_artifact_channel"


def effective_severity(
    code: C, mode: ComparisonMode, submode: Optional[ModeBSubmode]
) -> ReasonSeverity:
    """Base severity with mode-specific escalation applied (Correction 2)."""
    base = base_severity(code)
    if code not in AGGREGATOR_ESCALATED:
        return base
    if code == C.EXTERNAL_ORACLE_INELIGIBLE:
        return ReasonSeverity.BLOCK if mode == ComparisonMode.EXTERNAL_SEMANTIC else base
    if code == C.CONFIG_FINGERPRINT_MISMATCH:
        if (
            mode == ComparisonMode.RUNTIME_DIFFERENTIAL
            and submode == ModeBSubmode.CONFIGURATION_EQUIVALENCE
        ):
            return ReasonSeverity.INFORMATIONAL  # intended subject, not a block
        return ReasonSeverity.BLOCK
    if code in (C.PROMPT_TOKEN_ARRAY_MISSING, C.PROMPT_TOKEN_IDENTITY_UNVERIFIED):
        # Rule 4: external oracle needs array-strong prompt identity.
        return ReasonSeverity.BLOCK if mode == ComparisonMode.EXTERNAL_SEMANTIC else base
    return base


def run_calibration_preflight(inputs: PreflightInputs) -> CalibrationPreflightArtifact:
    ref, cand, profile = inputs.reference, inputs.candidate, inputs.build_profile
    mode, submode = inputs.declared_mode, inputs.declared_submode

    codes: list = []
    warnings: list = []

    decode_obj, decode_cal, c1, w1 = calibrate_decode_policy(ref, cand, profile)
    tok_obj, tok_cal, c2, w2 = calibrate_tokenizer(ref, cand)
    cfg_obj, cfg_cal, c3, w3 = calibrate_config(ref, cand, profile)
    num_obj, num_cal, c4, w4 = calibrate_numeric(ref, cand, profile)
    codes += c1 + c2 + c3 + c4
    warnings += w1 + w2 + w3 + w4

    # Compatibility pre-check: differing model families never compare.
    if (
        ref.model_family is not None
        and cand.model_family is not None
        and ref.model_family != cand.model_family
    ):
        codes.append(C.INCOMPATIBLE_MODEL_FAMILY)

    # --- Oracle eligibility -------------------------------------------------
    raw_greedy = decode_obj.raw_greedy_equivalent
    array_equal_evidence = (
        tok_obj.prompt_identity_evidence == PromptIdentityEvidence.ARRAY_EQUAL
    )
    # HF default greedy is eligible only when LIS is raw-greedy and every domain
    # is calibrated with array-strong prompt identity (Corrections 1/4, Rule 1/2).
    hf_default_greedy = bool(
        raw_greedy and cfg_cal and num_cal and tok_cal and array_equal_evidence
    )
    if not hf_default_greedy:
        codes.append(C.HF_DEFAULT_GREEDY_INELIGIBLE)

    # The M3 source-bound report channel makes forced-token runtime evidence
    # artifact-backed. The Python binder still has to verify every source SHA.
    forced = ForcedTokenRuntimeEligibility(
        potentially_eligible=True,
        artifact_supported=True,
        status=_FORCED_TOKEN_STATUS,
    )
    codes.append(C.HF_FORCED_TOKEN_RUNTIME_ELIGIBLE)

    family_block = C.INCOMPATIBLE_MODEL_FAMILY in codes
    policy_block = C.INCOMPATIBLE_DECODE_POLICY in codes
    input_block = C.INPUT_TOKEN_DIVERGENCE in codes
    lis_internal = (mode != ComparisonMode.EXTERNAL_SEMANTIC) and not (
        family_block or policy_block or input_block
    )
    if lis_internal:
        codes.append(C.INTERNAL_LIS_DIFFERENTIAL_ONLY)

    # MVP caps oracle scope at internal_lis_only (Mode C never wired; Rule 7).
    oracle_scope = OracleScope.INTERNAL_LIS_ONLY
    codes.append(C.ORACLE_SCOPE_LIMITED)
    codes.append(C.EXTERNAL_ORACLE_INELIGIBLE)

    oracle = OracleEligibility(
        lis_internal_backend_differential=lis_internal,
        hf_default_greedy=hf_default_greedy,
        hf_forced_token_runtime=forced,
        oracle_scope=oracle_scope,
    )

    # --- Verdict aggregation ------------------------------------------------
    eff = {code: effective_severity(code, mode, submode) for code in set(codes)}
    blocking = [code for code in codes if eff[code] == ReasonSeverity.BLOCK]
    has_block = len(blocking) > 0
    has_downgrade = any(eff[code] == ReasonSeverity.DOWNGRADE for code in codes)

    if has_block:
        verdict = Pass0Verdict.COMPARISON_BLOCKED
        eligibility = ComparisonEligibility.INCOMPATIBLE
        vsl = VerdictStrengthLimit.NO_COMPARISON
    elif has_downgrade:
        verdict = Pass0Verdict.LIMITED_COMPARISON_ALLOWED
        eligibility = ComparisonEligibility.LIMITED_COMPARISON
        vsl = VerdictStrengthLimit.CHECKPOINT_CONFIRMATION_ALLOWED
    else:
        verdict = Pass0Verdict.COMPARISON_ALLOWED
        eligibility = ComparisonEligibility.COMPARABLE
        vsl = VerdictStrengthLimit.CHECKPOINT_CONFIRMATION_ALLOWED

    status = CalibrationStatus(
        decode_policy_calibrated=decode_cal,
        config_semantics_calibrated=cfg_cal,
        numeric_policy_calibrated=num_cal,
        tokenizer_boundary_calibrated=tok_cal,
    )

    return CalibrationPreflightArtifact(
        comparison_mode=mode,
        comparison_eligibility=eligibility,
        pass0_verdict=verdict,
        calibration_status=status,
        decode_policy_identity=decode_obj,
        tokenizer_boundary=tok_obj,
        config_semantics=cfg_obj,
        numeric_policy=num_obj,
        oracle_eligibility=oracle,
        reason_codes=order_codes(codes),
        warnings=warnings,
        blocking_reasons=order_codes(blocking),
        verdict_strength_limit=vsl,
    )
