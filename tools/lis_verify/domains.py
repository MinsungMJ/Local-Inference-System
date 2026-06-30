"""Pure, mode-agnostic calibration domains for Pass 0.

Each calibrator returns ``(domain_object, calibrated, codes, warnings)``. All
mode-specific severity decisions are deferred to the aggregator (``pass0.py``);
these functions never consult the comparison mode.
"""

from __future__ import annotations

from typing import Optional

from .build_profile import BuildCalibrationProfile
from .inputs import RunSide
from .model import (
    CalibrationReasonCode as C,
    ConfigSemantics,
    ConfigSemanticsStatus,
    Confidence,
    DecodePolicyIdentity,
    KvWriteStatus,
    NumericPolicy,
    PromptBoundary,
    PromptIdentityEvidence,
    SelectionMode,
    TokenizerBoundary,
)


def _resolve_policy(side: RunSide, profile: BuildCalibrationProfile):
    """Resolve a side's effective (penalty, enabled, suppression)."""
    if side.repetition_penalty is not None:
        penalty = float(side.repetition_penalty)
        enabled = penalty != 1.0
    else:
        penalty = float(profile.repetition_penalty)
        enabled = bool(profile.repetition_penalty_enabled)
    if side.structural_token_suppression is not None:
        suppression = bool(side.structural_token_suppression)
    else:
        suppression = bool(profile.structural_token_suppression)
    return penalty, enabled, suppression


def calibrate_decode_policy(ref: RunSide, cand: RunSide, profile: BuildCalibrationProfile):
    codes: list = []
    warnings: list = []

    r_pen, r_en, r_supp = _resolve_policy(ref, profile)
    c_pen, c_en, c_supp = _resolve_policy(cand, profile)
    policies_match = (r_pen == c_pen and r_en == c_en and r_supp == c_supp)

    penalty, enabled, suppression = r_pen, r_en, r_supp
    non_neutral = enabled and penalty != 1.0
    raw_greedy_equivalent = (not non_neutral) and (not suppression)
    selection_mode = (
        SelectionMode.RAW_GREEDY
        if raw_greedy_equivalent
        else SelectionMode.POLICY_MODIFIED_GREEDY
    )

    identity = DecodePolicyIdentity(
        selection_mode=selection_mode,
        repetition_penalty=penalty,
        repetition_penalty_enabled=enabled,
        structural_token_suppression=suppression,
        raw_greedy_equivalent=raw_greedy_equivalent,
    )

    if not policies_match:
        # Rule 3: different decode policies block comparison.
        codes.append(C.INCOMPATIBLE_DECODE_POLICY)
        return identity, False, codes, warnings

    if non_neutral:
        codes.append(C.POLICY_MODIFIED_GREEDY)  # Rule 1
    if suppression:
        codes.append(C.DECODE_POLICY_NOT_RAW)  # Rule 2

    # Calibrated = we know the policy and both sides match it.
    return identity, True, codes, warnings


def calibrate_tokenizer(ref: RunSide, cand: RunSide):
    codes: list = []
    warnings: list = []

    direct = ref.input_mode == "tokens" and cand.input_mode == "tokens"
    boundary = PromptBoundary.DIRECT_TOKEN_IDS if direct else PromptBoundary.TEXT

    arrays_available = (
        ref.prompt_token_array is not None and cand.prompt_token_array is not None
    )
    array_equal: Optional[bool] = None
    if arrays_available:
        array_equal = ref.prompt_token_array == cand.prompt_token_array

    digest_known = (
        ref.prompt_token_digest is not None
        and cand.prompt_token_digest is not None
        and ref.prompt_token_count is not None
        and cand.prompt_token_count is not None
    )
    digest_equal: Optional[bool] = None
    if digest_known:
        digest_equal = (
            ref.prompt_token_digest == cand.prompt_token_digest
            and ref.prompt_token_count == cand.prompt_token_count
        )

    # Correction 4: array equality is distinct from digest-only evidence.
    if arrays_available and array_equal is False:
        evidence = PromptIdentityEvidence.DIVERGENT
    elif arrays_available and array_equal is True:
        evidence = PromptIdentityEvidence.ARRAY_EQUAL
    elif digest_equal is True:
        evidence = PromptIdentityEvidence.DIGEST_ONLY
    else:
        evidence = PromptIdentityEvidence.UNVERIFIED

    # Confidence reflects both the identity evidence and the prompt boundary: a
    # text boundary carries an inherent text->token tokenization risk, so it
    # caps confidence below `high` even when the supplied arrays are equal.
    if evidence == PromptIdentityEvidence.ARRAY_EQUAL:
        confidence = Confidence.HIGH if direct else Confidence.MEDIUM
    elif evidence == PromptIdentityEvidence.DIGEST_ONLY:
        confidence = Confidence.MEDIUM if direct else Confidence.LOW
    else:
        confidence = Confidence.LOW

    boundary_obj = TokenizerBoundary(
        prompt_boundary=boundary,
        prompt_token_array_available=arrays_available,
        prompt_token_array_equal=array_equal,
        prompt_token_digest_equal=digest_equal,
        prompt_identity_evidence=evidence,
        confidence=confidence,
    )

    if not direct:
        codes.append(C.CONFIDENCE_DOWNGRADE_TEXT_PROMPT_BOUNDARY)

    if evidence == PromptIdentityEvidence.DIVERGENT:
        # Rule 4: explicit arrays present and unequal -> block.
        codes.append(C.INPUT_TOKEN_DIVERGENCE)
        return boundary_obj, False, codes, warnings

    if not arrays_available:
        codes.append(C.PROMPT_TOKEN_ARRAY_MISSING)
    if evidence != PromptIdentityEvidence.ARRAY_EQUAL:
        codes.append(C.PROMPT_TOKEN_IDENTITY_UNVERIFIED)

    # Calibrated for external-grade identity only with direct IDs + array equality.
    calibrated = (
        boundary == PromptBoundary.DIRECT_TOKEN_IDS
        and evidence == PromptIdentityEvidence.ARRAY_EQUAL
    )
    if not calibrated:
        codes.append(C.TOKENIZER_BOUNDARY_UNCALIBRATED)

    return boundary_obj, calibrated, codes, warnings


def calibrate_config(ref: RunSide, cand: RunSide, profile: BuildCalibrationProfile):
    codes: list = []
    warnings: list = []

    eps_bound = profile.eps_bound_for(ref.model_family)

    fp_present = (
        ref.config_fingerprint is not None and cand.config_fingerprint is not None
    )
    fp_equal = fp_present and ref.config_fingerprint == cand.config_fingerprint

    if not fp_present:
        codes.append(C.RUNTIME_CONFIG_FINGERPRINT_MISSING)
    elif not fp_equal:
        # Base downgrade; the aggregator escalates to block outside the
        # configuration_equivalence submode (Correction 2).
        codes.append(C.CONFIG_FINGERPRINT_MISMATCH)

    if not eps_bound:
        # Rule 5: rms_norm_eps not runtime-bound -> uncalibrated config semantics.
        codes.append(C.RMS_NORM_EPS_RUNTIME_UNBOUND)
        codes.append(C.CONFIG_SEMANTICS_UNCALIBRATED)
        codes.append(C.REQUIRES_FIX_OR_GUARD)
        status = ConfigSemanticsStatus.REQUIRES_FIX_OR_GUARD
        calibrated = False
    elif fp_present and fp_equal:
        status = ConfigSemanticsStatus.CALIBRATED
        calibrated = True
    else:
        status = ConfigSemanticsStatus.UNKNOWN
        calibrated = False
        codes.append(C.CONFIG_SEMANTICS_UNCALIBRATED)

    obj = ConfigSemantics(rms_norm_eps_runtime_bound=eps_bound, status=status)
    return obj, calibrated, codes, warnings


def calibrate_numeric(ref: RunSide, cand: RunSide, profile: BuildCalibrationProfile):
    codes: list = []
    warnings: list = []

    rne = bool(profile.kv_write_round_to_nearest_even)
    kv_status = (
        KvWriteStatus.ROUND_TO_NEAREST_EVEN if rne else KvWriteStatus.TRUNCATE_OR_UNVERIFIED
    )
    obj = NumericPolicy(
        kv_bf16_write=kv_status, kv_f16_write=kv_status, round_to_nearest_even=rne
    )

    calibrated = True
    if not rne:
        # Rule 6: KV write rounding unverified -> numeric policy uncalibrated.
        codes.append(C.KV_WRITE_ROUNDING_UNVERIFIED)
        codes.append(C.NUMERIC_POLICY_UNCALIBRATED)
        calibrated = False

    backends_differ = (
        ref.backend is not None
        and cand.backend is not None
        and ref.backend != cand.backend
    )
    if backends_differ or profile.fma_contraction_backend_defined:
        codes.append(C.FMA_POLICY_BACKEND_DEFINED)
    if backends_differ or profile.reduction_order_backend_defined:
        codes.append(C.REDUCTION_ORDER_BACKEND_DEFINED)
    if backends_differ:
        codes.append(C.TOLERANCE_CAVEAT)
        calibrated = False

    return obj, calibrated, codes, warnings
