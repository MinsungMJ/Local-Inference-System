"""Coverage-scoped, model-free Pass 3 layer localization."""

from __future__ import annotations

from typing import Optional

from .pass1_inputs import CanonicalRunReport
from .pass2_model import (
    Pass2Result,
    Pass2Status,
    Pass3Disposition,
    ReproductionEvidenceTier,
    THREAD_COUNT_CAVEAT,
)
from .pass3_inputs import (
    CanonicalLayerTrace,
    CanonicalPass2Artifact,
    CheckpointSummaryArtifact,
    Pass3InputError,
    normalize_layer_trace,
    validate_both_source_bindings,
    validate_pass2_artifact_coherence,
)
from .pass3_model import (
    AlignmentStatus,
    AlignedCheckpointPair,
    CoverageAnalysis,
    DIGEST_DECISION_FIELD,
    DIGEST_DECISION_SEMANTICS,
    DIGEST_VERSION,
    FieldComparison,
    Pass2Evidence,
    Pass3DownstreamDisposition,
    Pass3ReasonCode,
    Pass3Result,
    Pass3Status,
    SummaryComparisonResult,
    SummaryEvidenceLevel,
    SummaryFieldDisposition,
    SuspectInterval,
)


def _unique(values):
    return tuple(dict.fromkeys(values))


def _reason(value: str) -> Pass3ReasonCode:
    try:
        return Pass3ReasonCode(value)
    except ValueError:
        return Pass3ReasonCode.CHECKPOINT_SUMMARY_MALFORMED


def _blocked(
    pass2: Pass2Result,
    status: Pass3Status,
    reason: Pass3ReasonCode,
    *,
    pass2_sha: Optional[str] = None,
    coherence: bool = False,
    pass2_evidence: Optional[Pass2Evidence] = None,
    reference_binding=None,
    candidate_binding=None,
) -> Pass3Result:
    return Pass3Result(
        status,
        Pass3DownstreamDisposition.BLOCKED,
        pass2_artifact_sha256=pass2_sha,
        pass2_object_artifact_coherence_verified=coherence,
        pass2_evidence=pass2_evidence,
        reference_binding=reference_binding,
        candidate_binding=candidate_binding,
        checkpoint_artifact_binding_verified=(
            reference_binding is not None and candidate_binding is not None
        ),
        target_runtime_checkpoint_step=(
            pass2.target.expected_runtime_checkpoint_step
            if isinstance(pass2, Pass2Result)
            else None
        ),
        reason_codes=(reason,),
        inherited_pass2_reason_codes=tuple(
            item.value for item in pass2.reason_codes
        ),
        inherited_pass1_reason_codes=pass2.inherited_pass1_reason_codes,
        inherited_pass0_reason_codes=pass2.inherited_pass0_reason_codes,
        warnings=pass2.warnings,
    )


def _typed_pass2_ready(pass2: Pass2Result) -> Optional[Pass3ReasonCode]:
    if (
        pass2.reproduction_evidence_tier
        == ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY
    ):
        return Pass3ReasonCode.REPRODUCTION_REQUEST_ONLY
    target = pass2.target
    checkpoint = pass2.checkpoint_step_reproduction
    coherent_target = (
        isinstance(target.expected_runtime_checkpoint_step, int)
        and not isinstance(target.expected_runtime_checkpoint_step, bool)
        and target.expected_runtime_checkpoint_step >= 0
        and isinstance(target.generated_token_step, int)
        and not isinstance(target.generated_token_step, bool)
        and target.expected_runtime_checkpoint_step
        == target.generated_token_step + 1
        and checkpoint.expected_runtime_checkpoint_step
        == target.expected_runtime_checkpoint_step
        and checkpoint.generated_token_step == target.generated_token_step
    )
    gates_verified = (
        pass2.prompt_reproduction.status == "verified"
        and pass2.prefix_reproduction.status == "verified"
        and pass2.policy_reproduction.status == "verified"
        and pass2.context_reproduction.status == "verified"
        and checkpoint.status == "verified"
    )
    if not (
        pass2.status == Pass2Status.REPRODUCTION_VERIFIED
        and pass2.pass3_disposition == Pass3Disposition.READY
        and pass2.source_binding_verified
        and pass2.source_binding.verified
        and coherent_target
        and gates_verified
    ):
        return Pass3ReasonCode.PASS2_NOT_READY
    return None


def _pass2_evidence(pass2: Pass2Result) -> Pass2Evidence:
    target = pass2.target
    binding = pass2.source_binding
    return Pass2Evidence(
        pass2.reproduction_evidence_tier.value,
        target.generated_token_step,
        target.expected_runtime_checkpoint_step,
        pass2.localization_ref.sha256,
        binding.reference_original_run_report_sha256,
        binding.candidate_original_run_report_sha256,
        binding.reference_reproduction_sha256,
        binding.candidate_reproduction_sha256,
        pass2.checkpoint_step_reproduction.evidence,
        pass2.verdict_strength_limit.value,
        THREAD_COUNT_CAVEAT in pass2.warnings,
    )


def _summary_map(artifact: CheckpointSummaryArtifact):
    return {item.coordinate.logical_key: item for item in artifact.summaries}


def _coordinate_map(coordinates):
    return {item.logical_key: item for item in coordinates}


def _coordinate_basis_conflicts(reference, candidate) -> bool:
    ref = {
        (item.runtime_checkpoint_step, item.layer_index): item.logical_key
        for item in reference.coverage.captured_coordinates
    }
    cand = {
        (item.runtime_checkpoint_step, item.layer_index): item.logical_key
        for item in candidate.coverage.captured_coordinates
    }
    return any(ref[key] != cand[key] for key in ref.keys() & cand.keys())


def _coverage(reference, candidate) -> CoverageAnalysis:
    ref_requested = _coordinate_map(reference.coverage.requested_coordinates)
    cand_requested = _coordinate_map(candidate.coverage.requested_coordinates)
    ref_captured = _coordinate_map(reference.coverage.captured_coordinates)
    cand_captured = _coordinate_map(candidate.coverage.captured_coordinates)
    common_keys = ref_captured.keys() & cand_captured.keys()
    ref_only_keys = ref_captured.keys() - cand_captured.keys()
    cand_only_keys = cand_captured.keys() - ref_captured.keys()
    common = tuple(
        coordinate
        for coordinate in reference.coverage.captured_coordinates
        if coordinate.logical_key in common_keys
    )
    return CoverageAnalysis(
        reference_requested=tuple(ref_requested.values()),
        reference_captured=tuple(ref_captured.values()),
        candidate_requested=tuple(cand_requested.values()),
        candidate_captured=tuple(cand_captured.values()),
        common_captured=common,
        reference_only=tuple(
            coordinate
            for coordinate in reference.coverage.captured_coordinates
            if coordinate.logical_key in ref_only_keys
        ),
        candidate_only=tuple(
            coordinate
            for coordinate in candidate.coverage.captured_coordinates
            if coordinate.logical_key in cand_only_keys
        ),
        common_comparable=common,
        reference_missing=reference.coverage.missing_coordinates,
        candidate_missing=candidate.coverage.missing_coordinates,
    )


def _alignment_failure(reference, candidate, coverage) -> Optional[str]:
    if _coordinate_basis_conflicts(reference, candidate):
        return "logical_coordinate_mismatch"
    if reference.model_family != candidate.model_family:
        return AlignmentStatus.MODEL_FAMILY_MISMATCH.value
    ref = _summary_map(reference)
    cand = _summary_map(candidate)
    for coordinate in coverage.common_captured:
        left = ref[coordinate.logical_key]
        right = cand[coordinate.logical_key]
        if left.shape != right.shape:
            return AlignmentStatus.SHAPE_MISMATCH.value
        if left.element_count != right.element_count:
            return "element_count_mismatch"
        if left.observed_dtype != right.observed_dtype:
            return AlignmentStatus.DTYPE_MISMATCH.value
        if left.precision_path != right.precision_path:
            return AlignmentStatus.PRECISION_PATH_MISMATCH.value
        if left.source_checkpoint_name != right.source_checkpoint_name:
            return "source_name_mismatch"
        if left.coordinate.execution_ordinal != right.coordinate.execution_ordinal:
            return "execution_ordinal_mismatch"
    return None


def _interval(last_equal, mismatch) -> SuspectInterval:
    if last_equal is None:
        return SuspectInterval(
            "runtime_entry",
            None,
            mismatch,
            False,
            True,
            tuple(range(0, mismatch.layer_index)),
            f"[entry, {mismatch.layer_index}]",
        )
    return SuspectInterval(
        "observed_checkpoint",
        last_equal,
        mismatch,
        True,
        True,
        tuple(range(last_equal.layer_index + 1, mismatch.layer_index)),
        f"({last_equal.layer_index}, {mismatch.layer_index}]",
    )


def run_coverage_scoped_layer_localization(
    pass2: Pass2Result,
    pass2_artifact: CanonicalPass2Artifact,
    reference_trace: CanonicalLayerTrace,
    candidate_trace: CanonicalLayerTrace,
    *,
    reference_source_report: CanonicalRunReport,
    candidate_source_report: CanonicalRunReport,
    calibrated_aggregate_policy=None,
) -> Pass3Result:
    """Localize the earliest observable digest mismatch on common coverage."""
    if not isinstance(pass2, Pass2Result):
        raise TypeError("pass2 must be a Pass2Result")
    if calibrated_aggregate_policy is not None:
        raise TypeError("calibrated aggregate policies are not implemented")

    # Gate A1: typed state only.
    readiness_reason = _typed_pass2_ready(pass2)
    if readiness_reason is not None:
        return _blocked(
            pass2, Pass3Status.COMPARISON_BLOCKED_BY_PASS2,
            readiness_reason,
        )

    # Gate A2: supplied canonical bytes and complete typed/artifact coherence.
    try:
        _, pass2_sha = validate_pass2_artifact_coherence(
            pass2, pass2_artifact
        )
    except Pass3InputError as exc:
        return _blocked(
            pass2, Pass3Status.COMPARISON_BLOCKED_BY_PASS2,
            _reason(exc.reason),
        )
    evidence = _pass2_evidence(pass2)

    # Gate B: both role/hash/set/manifest/step/trace-hash chains. Neither trace
    # summary accessor is called in this block.
    try:
        reference_binding, candidate_binding = validate_both_source_bindings(
            pass2,
            reference_source_report,
            candidate_source_report,
            reference_trace,
            candidate_trace,
        )
    except Pass3InputError as exc:
        if exc.reason == "pass3.runtime_checkpoint_step_mismatch":
            status = Pass3Status.CHECKPOINT_ALIGNMENT_INCONSISTENT
        elif exc.reason == "pass3.unsupported_checkpoint_layout":
            status = Pass3Status.UNSUPPORTED_CHECKPOINT_LAYOUT
        else:
            status = Pass3Status.SOURCE_BINDING_INCONSISTENT
        return _blocked(
            pass2, status, _reason(exc.reason), pass2_sha=pass2_sha,
            coherence=True, pass2_evidence=evidence,
        )

    # Gates C and D: strict layout/coverage/order then bounded summaries.
    try:
        reference = normalize_layer_trace(reference_trace)
        candidate = normalize_layer_trace(candidate_trace)
    except Pass3InputError as exc:
        if exc.reason == "pass3.unsupported_checkpoint_layout":
            status = Pass3Status.UNSUPPORTED_CHECKPOINT_LAYOUT
        elif exc.reason in (
            "pass3.checkpoint_alignment_inconsistent",
            "pass3.duplicate_checkpoint_coordinate",
            "pass3.runtime_checkpoint_step_mismatch",
        ):
            status = Pass3Status.CHECKPOINT_ALIGNMENT_INCONSISTENT
        elif exc.reason in (
            "pass3.summary_field_missing",
            "pass3.checkpoint_digest_incompatible",
        ):
            status = Pass3Status.COMPARISON_POLICY_UNAVAILABLE
        else:
            status = Pass3Status.CHECKPOINT_SUMMARY_MALFORMED
        return _blocked(
            pass2, status, _reason(exc.reason), pass2_sha=pass2_sha,
            coherence=True, pass2_evidence=evidence,
            reference_binding=reference_binding,
            candidate_binding=candidate_binding,
        )

    coverage = _coverage(reference, candidate)
    inherited = dict(
        pass2_artifact_sha256=pass2_sha,
        pass2_object_artifact_coherence_verified=True,
        pass2_evidence=evidence,
        reference_binding=reference_binding,
        candidate_binding=candidate_binding,
        checkpoint_artifact_binding_verified=True,
        target_runtime_checkpoint_step=(
            pass2.target.expected_runtime_checkpoint_step
        ),
        coverage=coverage,
        inherited_pass2_reason_codes=tuple(
            item.value for item in pass2.reason_codes
        ),
        inherited_pass1_reason_codes=pass2.inherited_pass1_reason_codes,
        inherited_pass0_reason_codes=pass2.inherited_pass0_reason_codes,
        warnings=pass2.warnings,
    )
    if not coverage.common_captured:
        reasons = [Pass3ReasonCode.INSUFFICIENT_COMMON_COVERAGE]
        if not coverage.reference_captured:
            reasons.append(Pass3ReasonCode.REFERENCE_CHECKPOINT_MISSING)
        if not coverage.candidate_captured:
            reasons.append(Pass3ReasonCode.CANDIDATE_CHECKPOINT_MISSING)
        if coverage.reference_only or coverage.candidate_only:
            reasons.append(Pass3ReasonCode.ASYMMETRIC_COVERAGE)
        return Pass3Result(
            Pass3Status.INSUFFICIENT_COMMON_COVERAGE,
            Pass3DownstreamDisposition.BLOCKED,
            reason_codes=_unique(reasons),
            **inherited,
        )
    alignment = _alignment_failure(reference, candidate, coverage)
    if alignment is not None:
        return Pass3Result(
            Pass3Status.CHECKPOINT_ALIGNMENT_INCONSISTENT,
            Pass3DownstreamDisposition.BLOCKED,
            reason_codes=(
                Pass3ReasonCode.CHECKPOINT_ALIGNMENT_INCONSISTENT,
            ),
            warnings=_unique(pass2.warnings + (alignment,)),
            **{key: value for key, value in inherited.items() if key != "warnings"},
        )

    ref_summaries = _summary_map(reference)
    cand_summaries = _summary_map(candidate)
    comparisons = []
    last_equal = None
    mismatch = None
    for coordinate in coverage.common_comparable:
        left = ref_summaries[coordinate.logical_key]
        right = cand_summaries[coordinate.logical_key]
        if left.digest is None or right.digest is None:
            return Pass3Result(
                Pass3Status.COMPARISON_POLICY_UNAVAILABLE,
                Pass3DownstreamDisposition.BLOCKED,
                comparisons=tuple(comparisons),
                reason_codes=(Pass3ReasonCode.COMPARISON_POLICY_UNAVAILABLE,),
                **inherited,
            )
        equivalent = left.digest.value == right.digest.value
        comparisons.append(
            SummaryComparisonResult(
                coordinate,
                equivalent,
                () if equivalent else (DIGEST_DECISION_FIELD,),
                (
                    FieldComparison(
                        DIGEST_DECISION_FIELD,
                        SummaryFieldDisposition.EXACT,
                        equivalent,
                    ),
                ),
                SummaryEvidenceLevel.TIER1_BOUNDED_DIGEST,
            )
        )
        if equivalent:
            last_equal = coordinate
            continue
        mismatch = coordinate
        break

    common_kwargs = dict(
        comparisons=tuple(comparisons),
        last_observed_equivalent_coordinate=last_equal,
        decision_field=DIGEST_DECISION_FIELD,
        decision_semantics=DIGEST_DECISION_SEMANTICS,
        evidence_level=SummaryEvidenceLevel.TIER1_BOUNDED_DIGEST,
        digest_contract_identity=DIGEST_VERSION,
        **inherited,
    )
    asymmetric = bool(coverage.reference_only or coverage.candidate_only)
    if mismatch is not None:
        interval = _interval(last_equal, mismatch)
        reasons = [Pass3ReasonCode.OBSERVABLE_MISMATCH_FOUND]
        if asymmetric:
            reasons.append(Pass3ReasonCode.ASYMMETRIC_COVERAGE)
        return Pass3Result(
            Pass3Status.OBSERVABLE_MISMATCH_FOUND,
            Pass3DownstreamDisposition.SUSPECT_INTERVAL_AVAILABLE,
            first_observed_mismatch_coordinate=mismatch,
            earliest_observable_suspect_layer=mismatch.layer_index,
            suspect_interval=interval,
            reason_codes=tuple(reasons),
            **common_kwargs,
        )
    reasons = [Pass3ReasonCode.NO_MISMATCH_IN_CAPTURED_COVERAGE]
    if asymmetric:
        reasons.append(Pass3ReasonCode.ASYMMETRIC_COVERAGE)
    return Pass3Result(
        Pass3Status.NO_MISMATCH_IN_CAPTURED_COVERAGE,
        Pass3DownstreamDisposition.EXPLORATORY_LOCALIZATION_ONLY,
        reason_codes=tuple(reasons),
        **common_kwargs,
    )
