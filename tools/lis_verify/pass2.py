"""P1 Pass 2: source-bound prefix and policy reproduction verification.

Pass 0 and Pass 1 remain authoritative.  This module verifies only report
identity, exact generated-prefix equality, build continuity, checkpoint-step
mapping, and execution-boundary metadata.  It has no tensor, logit, activation,
or numeric checkpoint comparison dependency.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

from .pass1_artifact import serialize as serialize_pass1
from .pass1_inputs import (
    CanonicalRunReport,
    canonical_json,
    sha256_text,
    token_ids_sha256,
)
from .pass1_model import (
    Pass1Result,
    Pass1Status,
    Pass2Disposition,
    PrefixAvailability,
)
from .pass2_inputs import (
    Pass2InputError,
    RunEvidence,
    extract_run_evidence,
    original_binding_matches,
    report_sha256,
    reproduction_identity_matches,
    selected_tokens,
)
from .pass2_model import (
    COMPUTED_STEP_EVIDENCE,
    THREAD_COUNT_CAVEAT,
    CheckpointStepGateResult,
    ContextGateResult,
    LocalizationReference,
    Pass2ReasonCode,
    Pass2Result,
    Pass2SourceBinding,
    Pass2Status,
    Pass3Disposition,
    PolicyGateResult,
    PrefixGateResult,
    PromptGateResult,
    ReproductionEvidenceTier,
    RunBuildEvidence,
    RunContextEvidence,
    TargetCheckpoint,
)


_NOT_EVALUATED = "not_evaluated"
_VERIFIED = "verified"
_FAILED = "failed"
_UNAVAILABLE = "unavailable"


def _localization_reference(pass1: Pass1Result) -> LocalizationReference:
    payload = serialize_pass1(pass1)
    rendered = canonical_json(payload)
    localization = pass1.localization
    prefix = pass1.prefix_for_reproduction
    return LocalizationReference(
        sha256=sha256_text(rendered),
        pass1_status=pass1.status.value,
        generated_token_step=(
            localization.generated_token_step
            if localization is not None
            else None
        ),
        runtime_checkpoint_step=(
            localization.runtime_checkpoint_step
            if localization is not None
            else None
        ),
        matched_generated_prefix_length=(
            localization.matched_generated_prefix_length
            if localization is not None
            else None
        ),
        prefix_availability=(
            prefix.availability.value
            if prefix is not None
            else PrefixAvailability.NOT_APPLICABLE.value
        ),
        pass2_disposition=pass1.pass2_disposition.value,
        verdict_strength_limit=pass1.verdict_strength_limit.value,
        canonical_json=rendered,
    )


def _target(pass1: Pass1Result) -> TargetCheckpoint:
    localization = pass1.localization
    if localization is None:
        return TargetCheckpoint(None, None, None)
    expected = (
        localization.generated_token_step + 1
        if localization.generated_token_step is not None
        else None
    )
    return TargetCheckpoint(
        localization.generated_token_step,
        expected,
        localization.matched_generated_prefix_length,
    )


def _initial_binding(
    localization_ref: LocalizationReference,
    reference_original,
    candidate_original,
    reference_reproduction,
    candidate_reproduction,
) -> Pass2SourceBinding:
    return Pass2SourceBinding(
        pass1_artifact_sha256=localization_ref.sha256,
        reference_original_run_report_sha256=report_sha256(
            reference_original
        ),
        candidate_original_run_report_sha256=report_sha256(
            candidate_original
        ),
        reference_original_verified=False,
        candidate_original_verified=False,
        reference_reproduction_sha256=report_sha256(
            reference_reproduction
        ),
        candidate_reproduction_sha256=report_sha256(
            candidate_reproduction
        ),
        reproduction_verified=None,
        verified=False,
    )


def _unique(values) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _result(
    pass1: Pass1Result,
    localization_ref: LocalizationReference,
    binding: Pass2SourceBinding,
    *,
    status: Pass2Status,
    tier: ReproductionEvidenceTier,
    pass3: Pass3Disposition,
    reasons: tuple[Pass2ReasonCode, ...] = (),
    prompt: Optional[PromptGateResult] = None,
    prefix: Optional[PrefixGateResult] = None,
    policy: Optional[PolicyGateResult] = None,
    checkpoint: Optional[CheckpointStepGateResult] = None,
    context: Optional[ContextGateResult] = None,
    warnings: tuple[str, ...] = (),
) -> Pass2Result:
    target = _target(pass1)
    prompt = prompt or PromptGateResult(
        _NOT_EVALUATED, pass1.compatibility.prompt_identity_evidence
    )
    prefix = prefix or PrefixGateResult(_NOT_EVALUATED)
    policy = policy or PolicyGateResult(
        _NOT_EVALUATED,
        False,
        pass1.calibration_ref.sha256,
    )
    checkpoint = checkpoint or CheckpointStepGateResult(
        _NOT_EVALUATED,
        target.generated_token_step,
        (
            pass1.localization.runtime_checkpoint_step
            if pass1.localization is not None
            else None
        ),
        target.expected_runtime_checkpoint_step,
        COMPUTED_STEP_EVIDENCE,
    )
    context = context or ContextGateResult(_NOT_EVALUATED)
    local_blocking = tuple(reason.value for reason in reasons)
    return Pass2Result(
        comparison_mode=pass1.comparison_mode,
        pass0_verdict=pass1.pass0_verdict,
        comparison_eligibility=pass1.comparison_eligibility,
        pass1_status=pass1.status,
        pass1_pass2_disposition=pass1.pass2_disposition,
        source_binding=binding,
        source_binding_verified=binding.verified,
        status=status,
        reproduction_evidence_tier=tier,
        target=target,
        prompt_reproduction=prompt,
        prefix_reproduction=prefix,
        policy_reproduction=policy,
        checkpoint_step_reproduction=checkpoint,
        context_reproduction=context,
        pass3_disposition=pass3,
        localization_ref=localization_ref,
        verdict_strength_limit=pass1.verdict_strength_limit,
        reason_codes=reasons,
        inherited_pass1_reason_codes=tuple(
            reason.value for reason in pass1.reason_codes
        ),
        inherited_pass0_reason_codes=pass1.inherited_pass0_reason_codes,
        blocking_reasons=_unique(
            tuple(pass1.blocking_reasons) + local_blocking
        ),
        warnings=_unique(tuple(pass1.warnings) + tuple(warnings)),
    )


def _early_disposition(
    pass1: Pass1Result,
    localization_ref: LocalizationReference,
    binding: Pass2SourceBinding,
) -> Optional[Pass2Result]:
    disposition = pass1.pass2_disposition
    if disposition == Pass2Disposition.READY:
        if pass1.status == Pass1Status.FIRST_MISMATCH_FOUND:
            return None
        return _result(
            pass1,
            localization_ref,
            binding,
            status=Pass2Status.INCONCLUSIVE,
            tier=ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY,
            pass3=Pass3Disposition.BLOCKED_BY_REPRODUCTION,
            reasons=(Pass2ReasonCode.PASS1_STATUS_NOT_REPRODUCIBLE,),
        )
    if disposition == Pass2Disposition.NOT_REQUIRED:
        return _result(
            pass1,
            localization_ref,
            binding,
            status=Pass2Status.NO_MISMATCH_TO_REPRODUCE,
            tier=ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY,
            pass3=Pass3Disposition.NOT_REQUIRED,
        )
    if disposition == Pass2Disposition.BLOCKED_BY_PASS0:
        return _result(
            pass1,
            localization_ref,
            binding,
            status=Pass2Status.COMPARISON_BLOCKED_BY_PASS0,
            tier=ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY,
            pass3=Pass3Disposition.BLOCKED_BY_PASS0,
        )
    if disposition == Pass2Disposition.BLOCKED_BY_EVIDENCE:
        return _result(
            pass1,
            localization_ref,
            binding,
            status=Pass2Status.TOKEN_LOCALIZATION_NOT_AVAILABLE,
            tier=ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY,
            pass3=Pass3Disposition.BLOCKED_BY_PASS1_EVIDENCE,
        )
    if disposition == Pass2Disposition.BLOCKED_BY_STRENGTH_LIMIT:
        return _result(
            pass1,
            localization_ref,
            binding,
            status=Pass2Status.UNSUPPORTED_REPRODUCTION_MODE,
            tier=ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY,
            pass3=Pass3Disposition.BLOCKED_BY_REPRODUCTION,
            reasons=(
                Pass2ReasonCode.VERDICT_STRENGTH_LIMIT_BLOCKS_REPRODUCTION,
            ),
        )
    return _result(
        pass1,
        localization_ref,
        binding,
        status=Pass2Status.INCONCLUSIVE,
        tier=ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY,
        pass3=Pass3Disposition.BLOCKED_BY_REPRODUCTION,
        reasons=(Pass2ReasonCode.PASS1_STATUS_NOT_REPRODUCIBLE,),
    )


def _run_builds(runs: Sequence[RunEvidence]) -> tuple[RunBuildEvidence, ...]:
    return tuple(
        RunBuildEvidence(run.role, run.binary_fingerprint) for run in runs
    )


def _policy(
    pass1: Pass1Result,
    runs: Sequence[RunEvidence],
    *,
    status: str,
    verified: bool,
) -> PolicyGateResult:
    return PolicyGateResult(
        status,
        verified,
        pass1.calibration_ref.sha256,
        _run_builds(runs),
    )


def _thread_warnings(runs: Sequence[RunEvidence]) -> tuple[str, ...]:
    if any(run.thread_count > 1 for run in runs):
        return (THREAD_COUNT_CAVEAT,)
    return ()


def _normalize_prefix_source(
    source: Sequence[int],
) -> Optional[tuple[int, ...]]:
    if isinstance(source, (str, bytes)):
        return None
    try:
        values = tuple(source)
    except TypeError:
        return None
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            return None
    return values


def _resolve_expected_prefix(
    pass1: Pass1Result,
    exact_prefix_source: Optional[Sequence[int]],
) -> tuple[
    Optional[tuple[int, ...]],
    Optional[Pass2ReasonCode],
    Optional[str],
]:
    localization = pass1.localization
    prefix = pass1.prefix_for_reproduction
    if (
        localization is None
        or localization.generated_token_step is None
        or prefix is None
    ):
        return None, Pass2ReasonCode.PASS1_STATUS_NOT_REPRODUCIBLE, None
    generated_step = localization.generated_token_step
    if (
        generated_step < 0
        or localization.matched_generated_prefix_length != generated_step
        or prefix.prefix_start_generated_step != 0
        or prefix.prefix_end_generated_step_exclusive != generated_step
        or prefix.token_count != generated_step
    ):
        return (
            None,
            Pass2ReasonCode.PREFIX_TOKEN_MISMATCH,
            "length_mismatch",
        )

    if exact_prefix_source is not None:
        values = _normalize_prefix_source(exact_prefix_source)
    elif len(prefix.exact_token_ids) == prefix.token_count:
        values = _normalize_prefix_source(prefix.exact_token_ids)
    elif (
        pass1.reference is not None
        and pass1.reference.selected_tokens.token_ids is not None
        and len(pass1.reference.selected_tokens.token_ids)
        >= prefix.token_count
    ):
        values = _normalize_prefix_source(
            pass1.reference.selected_tokens.token_ids[: prefix.token_count]
        )
    else:
        values = None

    if values is None:
        if (
            prefix.availability == PrefixAvailability.EXACT_SOURCE_REQUIRED
            and exact_prefix_source is None
        ):
            return (
                None,
                Pass2ReasonCode.PREFIX_MATERIAL_UNAVAILABLE,
                None,
            )
        return (
            None,
            Pass2ReasonCode.PREFIX_TOKEN_MISMATCH,
            "malformed_token_array",
        )
    if len(values) != prefix.token_count:
        return (
            None,
            Pass2ReasonCode.PREFIX_TOKEN_MISMATCH,
            "length_mismatch",
        )
    if token_ids_sha256(values) != prefix.sha256:
        return (
            None,
            Pass2ReasonCode.PREFIX_DIGEST_MISMATCH,
            "digest_mismatch",
        )
    return values, None, None


def _prefix_failure_result(
    pass1: Pass1Result,
    localization_ref: LocalizationReference,
    binding: Pass2SourceBinding,
    reason: Pass2ReasonCode,
    mismatch_kind: Optional[str],
    *,
    policy: Optional[PolicyGateResult] = None,
    warnings: tuple[str, ...] = (),
) -> Pass2Result:
    prefix = pass1.prefix_for_reproduction
    unavailable = reason == Pass2ReasonCode.PREFIX_MATERIAL_UNAVAILABLE
    return _result(
        pass1,
        localization_ref,
        binding,
        status=(
            Pass2Status.PREFIX_MATERIAL_UNAVAILABLE
            if unavailable
            else Pass2Status.PREFIX_REPRODUCTION_FAILED
        ),
        tier=ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY,
        pass3=Pass3Disposition.BLOCKED_BY_REPRODUCTION,
        reasons=(reason,),
        prefix=PrefixGateResult(
            _UNAVAILABLE if unavailable else _FAILED,
            prefix.token_count if prefix is not None else None,
            prefix.sha256 if prefix is not None else None,
            mismatch_kind=mismatch_kind,
        ),
        policy=policy,
        warnings=warnings,
    )


def _compare_prefixes(
    expected: tuple[int, ...],
    runs: Sequence[RunEvidence],
    expected_sha256: str,
) -> tuple[PrefixGateResult, Optional[Pass2ReasonCode]]:
    if not expected:
        return (
            PrefixGateResult(
                _VERIFIED,
                0,
                expected_sha256,
                tuple(run.role for run in runs),
            ),
            None,
        )
    verified = []
    for run in runs:
        values = selected_tokens(run)
        if values is None:
            return (
                PrefixGateResult(
                    _UNAVAILABLE,
                    len(expected),
                    expected_sha256,
                    tuple(verified),
                    failed_side=run.role,
                    mismatch_kind="exact_array_missing",
                ),
                Pass2ReasonCode.PREFIX_MATERIAL_UNAVAILABLE,
            )
        if len(values) < len(expected):
            return (
                PrefixGateResult(
                    _FAILED,
                    len(expected),
                    expected_sha256,
                    tuple(verified),
                    failed_side=run.role,
                    first_diff_index=len(values),
                    mismatch_kind="length_mismatch",
                ),
                Pass2ReasonCode.PREFIX_TOKEN_MISMATCH,
            )
        for index, token_id in enumerate(expected):
            if values[index] != token_id:
                return (
                    PrefixGateResult(
                        _FAILED,
                        len(expected),
                        expected_sha256,
                        tuple(verified),
                        failed_side=run.role,
                        first_diff_index=index,
                        mismatch_kind="token_mismatch",
                    ),
                    Pass2ReasonCode.PREFIX_TOKEN_MISMATCH,
                )
        verified.append(run.role)
    return (
        PrefixGateResult(
            _VERIFIED,
            len(expected),
            expected_sha256,
            tuple(verified),
        ),
        None,
    )


def _context_gate(
    runs: Sequence[RunEvidence], generated_step: int
) -> ContextGateResult:
    evidence = tuple(
        RunContextEvidence(
            role=run.role,
            prompt_token_count=run.prompt_token_count,
            context_position=run.prompt_token_count + generated_step,
            batch_size=run.batch_size,
            sequence_index=0,
            thread_count=run.thread_count,
        )
        for run in runs
    )
    positions = {item.context_position for item in evidence}
    valid = (
        len(positions) == 1
        and all(item.batch_size == 1 for item in evidence)
        and all(item.sequence_index == 0 for item in evidence)
    )
    return ContextGateResult(_VERIFIED if valid else _FAILED, evidence)


def run_prefix_policy_reproduction(
    pass1: Pass1Result,
    reference_original: CanonicalRunReport,
    candidate_original: CanonicalRunReport,
    *,
    exact_prefix_source: Sequence[int] | None = None,
    reference_reproduction: CanonicalRunReport | None = None,
    candidate_reproduction: CanonicalRunReport | None = None,
) -> Pass2Result:
    """Verify Pass 1's mismatch boundary using source-bound run reports."""
    if not isinstance(pass1, Pass1Result):
        raise TypeError("pass1 must be a Pass1Result")

    localization_ref = _localization_reference(pass1)
    binding = _initial_binding(
        localization_ref,
        reference_original,
        candidate_original,
        reference_reproduction,
        candidate_reproduction,
    )

    # Gate 0: preserve Pass 1's five-way provenance before report access.
    early = _early_disposition(pass1, localization_ref, binding)
    if early is not None:
        return early

    # Gate 1: immutable original hashes only; no materialization before this.
    reference_matches, candidate_matches = original_binding_matches(
        pass1, reference_original, candidate_original
    )
    has_reproductions = (
        reference_reproduction is not None
        or candidate_reproduction is not None
    )
    binding = replace(
        binding,
        reference_original_verified=reference_matches,
        candidate_original_verified=candidate_matches,
        verified=(
            reference_matches
            and candidate_matches
            and not has_reproductions
        ),
    )
    if not (reference_matches and candidate_matches):
        return _result(
            pass1,
            localization_ref,
            binding,
            status=Pass2Status.SOURCE_BINDING_INCONSISTENT,
            tier=ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY,
            pass3=Pass3Disposition.BLOCKED_BY_REPRODUCTION,
            reasons=(Pass2ReasonCode.SOURCE_BINDING_INCONSISTENT,),
        )

    # Gate 2: source-bound original metadata.
    try:
        reference = extract_run_evidence(
            reference_original, "reference_original"
        )
        candidate = extract_run_evidence(
            candidate_original, "candidate_original"
        )
    except (Pass2InputError, AttributeError, TypeError, ValueError) as exc:
        return _result(
            pass1,
            localization_ref,
            binding,
            status=Pass2Status.INCONCLUSIVE,
            tier=ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY,
            pass3=Pass3Disposition.BLOCKED_BY_REPRODUCTION,
            reasons=(Pass2ReasonCode.REPRODUCTION_ARTIFACT_MALFORMED,),
            warnings=(str(exc),),
        )
    runs: list[RunEvidence] = [reference, candidate]
    warnings = _thread_warnings(runs)

    # Gate 3: original-pair build continuity.
    original_policy = _policy(
        pass1,
        runs,
        status=(
            _VERIFIED
            if reference.binary_fingerprint == candidate.binary_fingerprint
            else _FAILED
        ),
        verified=(
            reference.binary_fingerprint == candidate.binary_fingerprint
        ),
    )
    if not original_policy.build_continuity_verified:
        return _result(
            pass1,
            localization_ref,
            binding,
            status=Pass2Status.DECODE_POLICY_REPRODUCTION_FAILED,
            tier=ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY,
            pass3=Pass3Disposition.BLOCKED_BY_REPRODUCTION,
            reasons=(
                Pass2ReasonCode.DECODE_POLICY_REPRODUCTION_FAILED,
            ),
            policy=original_policy,
            warnings=warnings,
        )

    # Gate 4: exact expected prefix, without relocalizing the mismatch.
    expected_prefix, prefix_reason, mismatch_kind = (
        _resolve_expected_prefix(pass1, exact_prefix_source)
    )
    if prefix_reason is not None:
        return _prefix_failure_result(
            pass1,
            localization_ref,
            binding,
            prefix_reason,
            mismatch_kind,
            policy=original_policy,
            warnings=warnings,
        )
    assert expected_prefix is not None

    # Gate 5: optional reruns are paired and identity-bound to originals.
    if has_reproductions:
        if (
            reference_reproduction is None
            or candidate_reproduction is None
        ):
            binding = replace(
                binding, reproduction_verified=False, verified=False
            )
            return _result(
                pass1,
                localization_ref,
                binding,
                status=Pass2Status.UNSUPPORTED_REPRODUCTION_MODE,
                tier=ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY,
                pass3=Pass3Disposition.BLOCKED_BY_REPRODUCTION,
                reasons=(
                    Pass2ReasonCode.UNSUPPORTED_REPRODUCTION_MODE,
                ),
                policy=original_policy,
                warnings=warnings,
            )
        ref_repro_matches = reproduction_identity_matches(
            reference_reproduction, reference
        )
        cand_repro_matches = reproduction_identity_matches(
            candidate_reproduction, candidate
        )
        binding = replace(
            binding,
            reproduction_verified=(
                ref_repro_matches and cand_repro_matches
            ),
            verified=(
                reference_matches
                and candidate_matches
                and ref_repro_matches
                and cand_repro_matches
            ),
        )
        if not (ref_repro_matches and cand_repro_matches):
            return _result(
                pass1,
                localization_ref,
                binding,
                status=Pass2Status.SOURCE_BINDING_INCONSISTENT,
                tier=ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY,
                pass3=Pass3Disposition.BLOCKED_BY_REPRODUCTION,
                reasons=(Pass2ReasonCode.SOURCE_BINDING_INCONSISTENT,),
                policy=original_policy,
                warnings=warnings,
            )
        try:
            reference_rerun = extract_run_evidence(
                reference_reproduction, "reference_reproduction"
            )
            candidate_rerun = extract_run_evidence(
                candidate_reproduction, "candidate_reproduction"
            )
        except (Pass2InputError, AttributeError, TypeError, ValueError) as exc:
            return _result(
                pass1,
                localization_ref,
                binding,
                status=Pass2Status.INCONCLUSIVE,
                tier=ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY,
                pass3=Pass3Disposition.BLOCKED_BY_REPRODUCTION,
                reasons=(
                    Pass2ReasonCode.REPRODUCTION_ARTIFACT_MALFORMED,
                ),
                policy=original_policy,
                warnings=warnings + (str(exc),),
            )
        runs.extend((reference_rerun, candidate_rerun))
        warnings = _thread_warnings(runs)

    # Gate 6: inherited prompt identity and exact [0..N-1] prefix.
    prompt = PromptGateResult(
        (
            _FAILED
            if pass1.compatibility.prompt_identity_evidence == "divergent"
            else _VERIFIED
        ),
        pass1.compatibility.prompt_identity_evidence,
    )
    if prompt.status == _FAILED:
        return _result(
            pass1,
            localization_ref,
            binding,
            status=Pass2Status.COMPARISON_BLOCKED_BY_PASS0,
            tier=ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY,
            pass3=Pass3Disposition.BLOCKED_BY_PASS0,
            reasons=(Pass2ReasonCode.PASS1_STATUS_NOT_REPRODUCIBLE,),
            prompt=prompt,
            policy=original_policy,
            warnings=warnings,
        )
    try:
        prefix_gate, prefix_gate_reason = _compare_prefixes(
            expected_prefix,
            runs,
            pass1.prefix_for_reproduction.sha256,
        )
    except Pass2InputError as exc:
        return _result(
            pass1,
            localization_ref,
            binding,
            status=Pass2Status.INCONCLUSIVE,
            tier=ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY,
            pass3=Pass3Disposition.BLOCKED_BY_REPRODUCTION,
            reasons=(Pass2ReasonCode.REPRODUCTION_ARTIFACT_MALFORMED,),
            prompt=prompt,
            policy=original_policy,
            warnings=warnings + (str(exc),),
        )
    if prefix_gate_reason is not None:
        unavailable = (
            prefix_gate_reason
            == Pass2ReasonCode.PREFIX_MATERIAL_UNAVAILABLE
        )
        return _result(
            pass1,
            localization_ref,
            binding,
            status=(
                Pass2Status.PREFIX_MATERIAL_UNAVAILABLE
                if unavailable
                else Pass2Status.PREFIX_REPRODUCTION_FAILED
            ),
            tier=ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY,
            pass3=Pass3Disposition.BLOCKED_BY_REPRODUCTION,
            reasons=(prefix_gate_reason,),
            prompt=prompt,
            prefix=prefix_gate,
            policy=original_policy,
            warnings=warnings,
        )

    # Gate 7: policy build continuity across originals and reruns.
    common_binary = reference.binary_fingerprint
    policy_verified = all(
        run.binary_fingerprint == common_binary for run in runs
    )
    policy = _policy(
        pass1,
        runs,
        status=_VERIFIED if policy_verified else _FAILED,
        verified=policy_verified,
    )
    if not policy_verified:
        return _result(
            pass1,
            localization_ref,
            binding,
            status=Pass2Status.DECODE_POLICY_REPRODUCTION_FAILED,
            tier=ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY,
            pass3=Pass3Disposition.BLOCKED_BY_REPRODUCTION,
            reasons=(
                Pass2ReasonCode.DECODE_POLICY_REPRODUCTION_FAILED,
            ),
            prompt=prompt,
            prefix=prefix_gate,
            policy=policy,
            warnings=warnings,
        )

    # Gate 8: exact Pass 1 step mapping, with no checkpoint-artifact claim.
    localization = pass1.localization
    assert localization is not None
    generated_step = localization.generated_token_step
    assert generated_step is not None
    expected_checkpoint = generated_step + 1
    checkpoint_verified = bool(
        localization.runtime_checkpoint_step == expected_checkpoint
        and localization.matched_generated_prefix_length == generated_step
    )
    checkpoint = CheckpointStepGateResult(
        _VERIFIED if checkpoint_verified else _FAILED,
        generated_step,
        localization.runtime_checkpoint_step,
        expected_checkpoint,
        COMPUTED_STEP_EVIDENCE,
        materialized_checkpoint_artifact_verified=False,
    )
    if not checkpoint_verified:
        return _result(
            pass1,
            localization_ref,
            binding,
            status=Pass2Status.CHECKPOINT_STEP_MAPPING_MISMATCH,
            tier=ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY,
            pass3=Pass3Disposition.BLOCKED_BY_REPRODUCTION,
            reasons=(
                Pass2ReasonCode.CHECKPOINT_STEP_MAPPING_MISMATCH,
            ),
            prompt=prompt,
            prefix=prefix_gate,
            policy=policy,
            checkpoint=checkpoint,
            warnings=warnings,
        )

    # Gate 9: derived context position, batch size, sequence index.
    context = _context_gate(runs, generated_step)
    if context.status != _VERIFIED:
        return _result(
            pass1,
            localization_ref,
            binding,
            status=Pass2Status.CONTEXT_POSITION_MISMATCH,
            tier=ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY,
            pass3=Pass3Disposition.BLOCKED_BY_REPRODUCTION,
            reasons=(Pass2ReasonCode.CONTEXT_POSITION_MISMATCH,),
            prompt=prompt,
            prefix=prefix_gate,
            policy=policy,
            checkpoint=checkpoint,
            context=context,
            warnings=warnings,
        )

    # Gate 10: successful local readiness plus explicit evidence strength.
    tier = (
        ReproductionEvidenceTier.INDEPENDENT_RERUN_VERIFIED
        if has_reproductions
        else ReproductionEvidenceTier.ORIGINAL_PAIR_BOUNDARY_CONSISTENT
    )
    return _result(
        pass1,
        localization_ref,
        binding,
        status=Pass2Status.REPRODUCTION_VERIFIED,
        tier=tier,
        pass3=Pass3Disposition.READY,
        prompt=prompt,
        prefix=prefix_gate,
        policy=policy,
        checkpoint=checkpoint,
        context=context,
        warnings=warnings,
    )
