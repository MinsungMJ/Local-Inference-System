"""P1 Pass 1: calibrated compatibility and selected-token localization.

Pass 0 remains authoritative.  This module performs only structural
compatibility/coherence checks and exact selected-token comparison; it has no
numeric, tensor, logit, layer-trace, or checkpoint-comparison dependency.
"""

from __future__ import annotations

from typing import Optional

from .artifact import serialize as serialize_pass0
from .gate import Pass0GateDecision
from .model import (
    CalibrationReasonCode as C,
    Pass0Verdict,
    PromptIdentityEvidence,
    VerdictStrengthLimit,
)
from .pass1_inputs import (
    CanonicalRunReport,
    RunReportInputError,
    UnsupportedRunReport,
    canonical_json,
    extract_run_metadata,
    extract_selected_token_sequence,
    sha256_text,
    token_ids_sha256,
)
from .pass1_model import (
    DEFAULT_EMBEDDED_PREFIX_CAP,
    CalibrationReference,
    CompatibilityResult,
    EvidenceCompleteness,
    MismatchKind,
    Pass0SourceBinding,
    Pass1ReasonCode,
    Pass1Result,
    Pass1RunInput,
    Pass1Status,
    Pass2Disposition,
    PrefixAvailability,
    PrefixForReproduction,
    SelectedTokenEvidenceLevel,
    TokenLocalization,
)


SUPPORTED_MODEL_FAMILIES = frozenset({
    "llama3_decoder",
    "qwen3_dense_decoder",
})


def runtime_checkpoint_step_for_generated(generated_token_step: int) -> int:
    if (
        isinstance(generated_token_step, bool)
        or not isinstance(generated_token_step, int)
        or generated_token_step < 0
    ):
        raise ValueError("generated_token_step must be a non-negative integer")
    return generated_token_step + 1


def locate_first_selected_token_mismatch(
    reference_tokens: tuple[int, ...] | list[int],
    candidate_tokens: tuple[int, ...] | list[int],
) -> TokenLocalization:
    """Locate the first exact selected-token or sequence-boundary mismatch."""
    ref = tuple(reference_tokens)
    cand = tuple(candidate_tokens)
    min_len = min(len(ref), len(cand))

    for index in range(min_len):
        if ref[index] != cand[index]:
            return TokenLocalization(
                generated_token_step=index,
                runtime_checkpoint_step=runtime_checkpoint_step_for_generated(index),
                reference_selected_token_id=ref[index],
                candidate_selected_token_id=cand[index],
                matched_generated_prefix_length=index,
                observed_reference_generated_length=len(ref),
                observed_candidate_generated_length=len(cand),
                mismatch_kind=MismatchKind.TOKEN_ID_MISMATCH,
            )

    if len(ref) == len(cand):
        return TokenLocalization(
            generated_token_step=None,
            runtime_checkpoint_step=None,
            reference_selected_token_id=None,
            candidate_selected_token_id=None,
            matched_generated_prefix_length=len(ref),
            observed_reference_generated_length=len(ref),
            observed_candidate_generated_length=len(cand),
            mismatch_kind=None,
        )

    index = min_len
    return TokenLocalization(
        generated_token_step=index,
        runtime_checkpoint_step=runtime_checkpoint_step_for_generated(index),
        reference_selected_token_id=ref[index] if index < len(ref) else None,
        candidate_selected_token_id=cand[index] if index < len(cand) else None,
        matched_generated_prefix_length=index,
        observed_reference_generated_length=len(ref),
        observed_candidate_generated_length=len(cand),
        mismatch_kind=MismatchKind.LENGTH_MISMATCH_OR_EARLY_TERMINATION,
    )


def _calibration_reference(gate: Pass0GateDecision) -> CalibrationReference:
    payload = serialize_pass0(gate.artifact)
    rendered = canonical_json(payload)
    return CalibrationReference(
        sha256=sha256_text(rendered),
        comparison_mode=gate.artifact.comparison_mode.value,
        pass0_verdict=gate.verdict.value,
        comparison_eligibility=gate.eligibility.value,
        prompt_identity_evidence=(
            gate.artifact.tokenizer_boundary.prompt_identity_evidence.value
        ),
        verdict_strength_limit=gate.verdict_strength_limit.value,
        blocking_reasons=tuple(
            reason.value for reason in gate.blocking_reasons
        ),
        oracle_scope=gate.oracle_eligibility.oracle_scope.value,
        canonical_json=rendered,
    )


def _gate_is_coherent(gate: Pass0GateDecision) -> bool:
    artifact = gate.artifact
    return bool(
        gate.proceed == (gate.verdict != Pass0Verdict.COMPARISON_BLOCKED)
        and gate.eligibility == artifact.comparison_eligibility
        and gate.verdict == artifact.pass0_verdict
        and gate.verdict_strength_limit == artifact.verdict_strength_limit
        and gate.oracle_eligibility == artifact.oracle_eligibility
        and list(gate.blocking_reasons) == list(artifact.blocking_reasons)
    )


def _binding_matches(
    source: CanonicalRunReport, expected_identity
) -> bool:
    return source.identity == expected_identity


def _inherited_codes(gate: Pass0GateDecision) -> tuple[str, ...]:
    return tuple(reason.value for reason in gate.artifact.reason_codes)


def _blocking_codes(gate: Pass0GateDecision) -> tuple[str, ...]:
    return tuple(reason.value for reason in gate.blocking_reasons)


def _compatibility(
    gate: Pass0GateDecision,
    *,
    blocked: bool,
    boundary: Optional[str],
    model_family_state: str = "not_evaluated",
    config_state: str = "not_evaluated",
) -> CompatibilityResult:
    return CompatibilityResult(
        comparison_blocked=blocked,
        boundary=boundary,
        model_family_state=model_family_state,
        config_state=config_state,
        prompt_identity_evidence=(
            gate.artifact.tokenizer_boundary.prompt_identity_evidence.value
        ),
    )


def _result(
    gate: Pass0GateDecision,
    source_binding: Pass0SourceBinding,
    *,
    binding_verified: bool,
    status: Pass1Status,
    completeness: EvidenceCompleteness,
    compatibility: CompatibilityResult,
    reference: Optional[Pass1RunInput] = None,
    candidate: Optional[Pass1RunInput] = None,
    localization: Optional[TokenLocalization] = None,
    prefix: Optional[PrefixForReproduction] = None,
    pass2: Pass2Disposition = Pass2Disposition.BLOCKED_BY_EVIDENCE,
    reasons: tuple[Pass1ReasonCode, ...] = (),
    warnings: tuple[str, ...] = (),
) -> Pass1Result:
    return Pass1Result(
        comparison_mode=gate.artifact.comparison_mode,
        pass0_verdict=gate.verdict,
        comparison_eligibility=gate.eligibility,
        source_binding=source_binding,
        source_binding_verified=binding_verified,
        status=status,
        evidence_completeness=completeness,
        compatibility=compatibility,
        reference=reference,
        candidate=candidate,
        localization=localization,
        prefix_for_reproduction=prefix,
        pass2_disposition=pass2,
        calibration_ref=_calibration_reference(gate),
        verdict_strength_limit=gate.verdict_strength_limit,
        reason_codes=reasons,
        inherited_pass0_reason_codes=_inherited_codes(gate),
        blocking_reasons=_blocking_codes(gate),
        warnings=warnings,
    )


def _blocked_boundary(gate: Pass0GateDecision) -> str:
    reasons = set(gate.blocking_reasons)
    if C.INPUT_TOKEN_DIVERGENCE in reasons:
        return "prompt_identity"
    if C.INCOMPATIBLE_MODEL_FAMILY in reasons:
        return "model_family"
    if C.CONFIG_FINGERPRINT_MISMATCH in reasons:
        return "config"
    if C.INCOMPATIBLE_DECODE_POLICY in reasons:
        return "decode_policy"
    return "pass0"


def _metadata_failure(
    gate: Pass0GateDecision,
    source_binding: Pass0SourceBinding,
    error: RunReportInputError,
    role: str,
) -> Pass1Result:
    status = (
        Pass1Status.UNSUPPORTED_COMPARISON
        if isinstance(error, UnsupportedRunReport)
        else Pass1Status.INCONCLUSIVE
    )
    return _result(
        gate,
        source_binding,
        binding_verified=True,
        status=status,
        completeness=EvidenceCompleteness.INCOMPLETE,
        compatibility=_compatibility(
            gate, blocked=True, boundary="run_artifact"
        ),
        reasons=(error.reason,),
        warnings=(f"{role}: {error}",),
    )


def _validate_run_compatibility(
    gate: Pass0GateDecision, reference, candidate
) -> tuple[str, str]:
    for metadata in (reference, candidate):
        if (
            metadata.execution_status is None
            or metadata.execution_status.lower() != "ok"
        ):
            return "failed_execution", metadata.role
        if metadata.model_family not in SUPPORTED_MODEL_FAMILIES:
            return "unsupported_model_family", metadata.role
        if metadata.batch_size != 1:
            return "unsupported_batch", metadata.role

    if reference.model_family != candidate.model_family:
        return "incompatible_model_family", "comparison"
    if (
        reference.model_fingerprint is not None
        and candidate.model_fingerprint is not None
        and reference.model_fingerprint != candidate.model_fingerprint
    ):
        return "incompatible_model_fingerprint", "comparison"

    pass0_codes = set(gate.artifact.reason_codes)
    config_mismatch = reference.config_fingerprint != candidate.config_fingerprint
    pass0_recorded_mismatch = C.CONFIG_FINGERPRINT_MISMATCH in pass0_codes
    pass0_recorded_missing = C.RUNTIME_CONFIG_FINGERPRINT_MISSING in pass0_codes
    config_missing = (
        reference.config_fingerprint is None
        or candidate.config_fingerprint is None
    )
    if config_mismatch and not (pass0_recorded_mismatch or pass0_recorded_missing):
        return "gate_config_inconsistent", "comparison"
    if not config_mismatch and pass0_recorded_mismatch:
        return "gate_config_inconsistent", "comparison"
    if config_missing and not pass0_recorded_missing:
        return "gate_config_inconsistent", "comparison"

    config_state = (
        "accepted_by_pass0"
        if config_mismatch or config_missing
        else "compatible"
    )
    return "compatible", config_state


def run_token_localization(
    gate: Pass0GateDecision,
    reference: CanonicalRunReport,
    candidate: CanonicalRunReport,
    source_binding: Pass0SourceBinding,
) -> Pass1Result:
    """Run Pass 1 over an explicitly source-bound pair of run reports."""
    if not isinstance(source_binding, Pass0SourceBinding):
        raise TypeError("source_binding must be a Pass0SourceBinding")

    # Step 1: gate coherence. No report materialization or selected-token access.
    if not _gate_is_coherent(gate):
        return _result(
            gate,
            source_binding,
            binding_verified=False,
            status=Pass1Status.INCONCLUSIVE,
            completeness=EvidenceCompleteness.INCOMPLETE,
            compatibility=_compatibility(
                gate, blocked=True, boundary="pass0_gate"
            ),
            reasons=(Pass1ReasonCode.GATE_RUN_IDENTITY_INCONSISTENT,),
            warnings=("Pass0GateDecision is internally inconsistent",),
        )

    # Steps 2/3: verify immutable identities before report materialization.
    if not (
        _binding_matches(reference, source_binding.reference)
        and _binding_matches(candidate, source_binding.candidate)
    ):
        return _result(
            gate,
            source_binding,
            binding_verified=False,
            status=Pass1Status.INCONCLUSIVE,
            completeness=EvidenceCompleteness.INCOMPLETE,
            compatibility=_compatibility(
                gate, blocked=True, boundary="source_binding"
            ),
            reasons=(Pass1ReasonCode.GATE_RUN_IDENTITY_INCONSISTENT,),
            warnings=("supplied run reports do not match Pass 0 source binding",),
        )

    # Step 4: Pass 0 is authoritative and blocks before selected-token access.
    if not gate.proceed:
        input_divergence = C.INPUT_TOKEN_DIVERGENCE in set(
            gate.blocking_reasons
        )
        return _result(
            gate,
            source_binding,
            binding_verified=True,
            status=(
                Pass1Status.INPUT_TOKEN_DIVERGENCE
                if input_divergence
                else Pass1Status.COMPARISON_BLOCKED_BY_PASS0
            ),
            completeness=EvidenceCompleteness.INCOMPLETE,
            compatibility=_compatibility(
                gate, blocked=True, boundary=_blocked_boundary(gate)
            ),
            pass2=Pass2Disposition.BLOCKED_BY_PASS0,
        )

    # Steps 5-8: materialize structure and validate compatibility without
    # reading the selected-token fields.
    try:
        reference_raw = reference.materialize()
        candidate_raw = candidate.materialize()
        reference_metadata = extract_run_metadata(
            reference_raw, reference.identity, "reference"
        )
        candidate_metadata = extract_run_metadata(
            candidate_raw, candidate.identity, "candidate"
        )
    except RunReportInputError as error:
        return _metadata_failure(
            gate, source_binding, error, "run report"
        )

    compatibility_state, compatibility_detail = _validate_run_compatibility(
        gate, reference_metadata, candidate_metadata
    )
    if compatibility_state != "compatible":
        if compatibility_state == "unsupported_batch":
            reason = Pass1ReasonCode.UNSUPPORTED_BATCH_SHAPE
            status = Pass1Status.UNSUPPORTED_COMPARISON
        elif compatibility_state == "gate_config_inconsistent":
            reason = Pass1ReasonCode.GATE_RUN_IDENTITY_INCONSISTENT
            status = Pass1Status.INCONCLUSIVE
        else:
            reason = Pass1ReasonCode.UNSUPPORTED_RUN_ARTIFACT
            status = (
                Pass1Status.INCONCLUSIVE
                if compatibility_state == "failed_execution"
                else Pass1Status.UNSUPPORTED_COMPARISON
            )
        return _result(
            gate,
            source_binding,
            binding_verified=True,
            status=status,
            completeness=EvidenceCompleteness.INCOMPLETE,
            compatibility=_compatibility(
                gate,
                blocked=True,
                boundary=compatibility_state,
            ),
            reasons=(reason,),
            warnings=(f"{compatibility_detail}: {compatibility_state}",),
        )

    prompt_evidence = (
        gate.artifact.tokenizer_boundary.prompt_identity_evidence
    )
    if prompt_evidence == PromptIdentityEvidence.DIVERGENT:
        return _result(
            gate,
            source_binding,
            binding_verified=True,
            status=Pass1Status.INPUT_TOKEN_DIVERGENCE,
            completeness=EvidenceCompleteness.INCOMPLETE,
            compatibility=_compatibility(
                gate,
                blocked=True,
                boundary="prompt_identity",
                model_family_state="compatible",
                config_state=compatibility_detail,
            ),
            pass2=Pass2Disposition.BLOCKED_BY_PASS0,
        )

    # Steps 9/10: selected-token evidence is first accessed here.
    try:
        reference_sequence = extract_selected_token_sequence(
            reference_raw, "reference"
        )
        candidate_sequence = extract_selected_token_sequence(
            candidate_raw, "candidate"
        )
    except RunReportInputError as error:
        return _metadata_failure(
            gate, source_binding, error, "selected-token metadata"
        )

    reference_input = Pass1RunInput(
        reference_metadata, reference.identity, reference_sequence
    )
    candidate_input = Pass1RunInput(
        candidate_metadata, candidate.identity, candidate_sequence
    )
    compatibility = _compatibility(
        gate,
        blocked=False,
        boundary=None,
        model_family_state="compatible",
        config_state=compatibility_detail,
    )

    if (
        reference_sequence.token_ids is None
        or candidate_sequence.token_ids is None
    ):
        both_digest_only = (
            reference_sequence.evidence_level
            == SelectedTokenEvidenceLevel.DIGEST_ONLY
            and candidate_sequence.evidence_level
            == SelectedTokenEvidenceLevel.DIGEST_ONLY
        )
        reasons = [Pass1ReasonCode.SELECTED_TOKEN_ARRAY_MISSING]
        if both_digest_only:
            reasons.append(Pass1ReasonCode.SELECTED_TOKEN_IDENTITY_UNVERIFIED)
        return _result(
            gate,
            source_binding,
            binding_verified=True,
            status=(
                Pass1Status.SELECTED_TOKEN_IDENTITY_UNVERIFIED
                if both_digest_only
                else Pass1Status.SELECTED_TOKEN_ARRAY_MISSING
            ),
            completeness=EvidenceCompleteness.INCOMPLETE,
            compatibility=compatibility,
            reference=reference_input,
            candidate=candidate_input,
            reasons=tuple(reasons),
            pass2=Pass2Disposition.BLOCKED_BY_EVIDENCE,
        )

    # Step 11: exact selected-token localization.
    localization = locate_first_selected_token_mismatch(
        reference_sequence.token_ids, candidate_sequence.token_ids
    )
    if localization.generated_token_step is None:
        return _result(
            gate,
            source_binding,
            binding_verified=True,
            status=Pass1Status.TOKEN_EQUIVALENT_ON_OBSERVED_RANGE,
            completeness=EvidenceCompleteness.COMPLETE,
            compatibility=compatibility,
            reference=reference_input,
            candidate=candidate_input,
            localization=localization,
            pass2=Pass2Disposition.NOT_REQUIRED,
        )

    prefix_ids = reference_sequence.token_ids[
        : localization.generated_token_step
    ]
    availability = (
        PrefixAvailability.EMBEDDED
        if len(prefix_ids) <= DEFAULT_EMBEDDED_PREFIX_CAP
        else PrefixAvailability.EXACT_SOURCE_REQUIRED
    )
    prefix = PrefixForReproduction(
        exact_token_ids=prefix_ids,
        availability=availability,
        prefix_start_generated_step=0,
        prefix_end_generated_step_exclusive=len(prefix_ids),
        token_count=len(prefix_ids),
        sha256=token_ids_sha256(prefix_ids),
    )
    pass2 = (
        Pass2Disposition.READY
        if gate.verdict_strength_limit
        == VerdictStrengthLimit.CHECKPOINT_CONFIRMATION_ALLOWED
        else Pass2Disposition.BLOCKED_BY_STRENGTH_LIMIT
    )
    return _result(
        gate,
        source_binding,
        binding_verified=True,
        status=Pass1Status.FIRST_MISMATCH_FOUND,
        completeness=EvidenceCompleteness.COMPLETE,
        compatibility=compatibility,
        reference=reference_input,
        candidate=candidate_input,
        localization=localization,
        prefix=prefix,
        pass2=pass2,
    )
