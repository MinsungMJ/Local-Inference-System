"""Coverage-scoped intra-layer localization and public orchestration.

The P4-9 core consumes the binding-first P4-8 typed outcome, applies the frozen
coverage algebra, validates shared-entry alignment, checks digest-policy
availability, compares every bounded common entry, and constructs one coherent
immutable Pass4Result.  P4-10 adds only a thin public P4-3→P4-8→P4-9 wrapper.

Neither path recomputes checkpoint digests, serializes results, maps successful
results to global enums, or performs real-run orchestration.  Serialization is
owned by ``pass4_artifact`` and real integration remains a P4-12 responsibility.
"""

from __future__ import annotations

from typing import Optional

from .pass1_inputs import CanonicalRunReport
from .pass3_inputs import CanonicalLayerTrace, CanonicalPass2Artifact
from .pass3_model import Pass3Result
from .pass4_contract import (
    DIGEST_VERSION,
    EVIDENCE_LEVEL,
    INHERITED_BOUNDARY_EVIDENCE_ORIGIN,
    INTRA_LAYER_STAGES,
    LOCAL_EVIDENCE_ORIGIN,
    STATUS_TO_DISPOSITION,
    IntraLayerCoordinate,
    Pass4ReasonCode,
    Pass4Status,
    Pass4SuspectInterval,
    analyze_coverage,
)
from .pass4_inputs import (
    IntraLayerTraceEntry,
    ParsedIntraLayerSource,
    Pass4TraceParsingOutcome,
    parse_pass4_intra_layer_inputs,
)
from .pass4_model import (
    FROZEN_EVIDENCE_CEILING,
    MAX_WARNINGS,
    Pass4ClosingBoundaryDecision,
    Pass4Comparison,
    Pass4ComparisonDecision,
    Pass4CoverageAnalysis,
    Pass4LocalCoverageOutcome,
    Pass4Result,
)
from .pass4_parent import (
    CanonicalPass3Artifact,
    bind_pass4_parent_inputs,
)


def _inherited_kwargs(outcome: Pass4TraceParsingOutcome) -> dict:
    parent = outcome.parent_outcome
    return {
        "inherited_pass3_reason_codes": (
            parent.inherited_pass3_reason_codes
        ),
        "inherited_pass2_reason_codes": (
            parent.inherited_pass2_reason_codes
        ),
        "inherited_pass1_reason_codes": (
            parent.inherited_pass1_reason_codes
        ),
        "inherited_pass0_reason_codes": (
            parent.inherited_pass0_reason_codes
        ),
    }


def _merged_warnings(
    outcome: Pass4TraceParsingOutcome,
) -> Optional[tuple[str, ...]]:
    values = tuple(
        dict.fromkeys(
            outcome.parent_outcome.inherited_parent_warnings
            + outcome.warnings
        )
    )
    return values if len(values) <= MAX_WARNINGS else None


def _warning_overflow_result(
    outcome: Pass4TraceParsingOutcome,
) -> Pass4Result:
    return Pass4Result(
        Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
        STATUS_TO_DISPOSITION[Pass4Status.COMPARISON_BLOCKED_BY_PASS3],
        FROZEN_EVIDENCE_CEILING,
        (Pass4ReasonCode.PARENT_TYPED_ARTIFACT_INCOHERENT,),
        **_inherited_kwargs(outcome),
    )


def _terminal_result(
    outcome: Pass4TraceParsingOutcome,
    warnings: tuple[str, ...],
) -> Pass4Result:
    parent = outcome.parent_outcome
    return Pass4Result(
        outcome.status,
        outcome.disposition,
        FROZEN_EVIDENCE_CEILING,
        outcome.reason_codes,
        parent_pass3=parent.parent,
        reference_binding=parent.reference_binding,
        candidate_binding=parent.candidate_binding,
        warnings=warnings,
        **_inherited_kwargs(outcome),
    )


def _target_kwargs(
    outcome: Pass4TraceParsingOutcome,
    reference: ParsedIntraLayerSource,
) -> dict:
    parent = outcome.parent_outcome
    header = reference.header
    return {
        "target_runtime_checkpoint_step": (
            parent.target_runtime_checkpoint_step
        ),
        "target_layer": parent.target_layer,
        "target_token_position": header.token_position,
        "phase": header.phase,
        "model_family": parent.model_family,
        "precision_path": parent.precision_path,
        "layout_name": header.layout_name,
        "layout_version": header.layout_version,
        "stage_taxonomy": header.stage_taxonomy,
    }


def _coverage(
    reference: ParsedIntraLayerSource,
    candidate: ParsedIntraLayerSource,
    comparable: tuple[IntraLayerCoordinate, ...],
) -> Pass4CoverageAnalysis:
    derived = analyze_coverage(
        reference.coverage,
        candidate.coverage,
        common_comparable=comparable,
    )
    return Pass4CoverageAnalysis(
        reference.coverage,
        candidate.coverage,
        derived.common_captured,
        derived.common_comparable,
        derived.reference_only,
        derived.candidate_only,
    )


def _entry_index(
    source: ParsedIntraLayerSource,
) -> dict[tuple[object, ...], IntraLayerTraceEntry]:
    return {
        entry.coordinate.logical_key: entry for entry in source.entries
    }


def _alignment_reason(
    reference_source: ParsedIntraLayerSource,
    candidate_source: ParsedIntraLayerSource,
    reference: IntraLayerTraceEntry,
    candidate: IntraLayerTraceEntry,
) -> Optional[Pass4ReasonCode]:
    left = reference.coordinate
    right = candidate.coordinate
    if left.runtime_checkpoint_step != right.runtime_checkpoint_step:
        return Pass4ReasonCode.STEP_ALIGNMENT_MISMATCH
    if left.layer_index != right.layer_index:
        return Pass4ReasonCode.LAYER_ALIGNMENT_MISMATCH
    if (
        left.stage_id != right.stage_id
        or left.tensor_role != right.tensor_role
        or reference.public_name != candidate.public_name
        or left.stage_order != right.stage_order
        or left.execution_ordinal != right.execution_ordinal
    ):
        return Pass4ReasonCode.STAGE_ROLE_OR_ORDER_ALIGNMENT_MISMATCH
    if (
        reference.shape != candidate.shape
        or reference.element_count != candidate.element_count
    ):
        return Pass4ReasonCode.SHAPE_OR_COUNT_ALIGNMENT_MISMATCH
    if (
        reference.observed_dtype != candidate.observed_dtype
        or reference.precision_path != candidate.precision_path
    ):
        return Pass4ReasonCode.DTYPE_OR_PRECISION_ALIGNMENT_MISMATCH
    left_header = reference_source.header
    right_header = candidate_source.header
    if (
        reference.phase != candidate.phase
        or left_header.phase != right_header.phase
        or left.batch_index != right.batch_index
        or left.sequence_index != right.sequence_index
        or left.token_position != right.token_position
        or left_header.token_position != right_header.token_position
    ):
        return Pass4ReasonCode.PHASE_OR_POSITION_ALIGNMENT_MISMATCH
    if left_header.runtime_checkpoint_step != right_header.runtime_checkpoint_step:
        return Pass4ReasonCode.STEP_ALIGNMENT_MISMATCH
    if left_header.target_layer != right_header.target_layer:
        return Pass4ReasonCode.LAYER_ALIGNMENT_MISMATCH
    if (
        left_header.model_family != right_header.model_family
        or left_header.layout_name != right_header.layout_name
        or left_header.layout_version != right_header.layout_version
        or left_header.stage_taxonomy != right_header.stage_taxonomy
    ):
        return Pass4ReasonCode.STAGE_ROLE_OR_ORDER_ALIGNMENT_MISMATCH
    return None


def _noncomparison_result(
    outcome: Pass4TraceParsingOutcome,
    status: Pass4Status,
    reason: Pass4ReasonCode,
    warnings: tuple[str, ...],
    *,
    coverage: Pass4CoverageAnalysis,
    target: dict,
) -> Pass4Result:
    parent = outcome.parent_outcome
    return Pass4Result(
        status,
        STATUS_TO_DISPOSITION[status],
        FROZEN_EVIDENCE_CEILING,
        (reason,),
        parent_pass3=parent.parent,
        reference_binding=parent.reference_binding,
        candidate_binding=parent.candidate_binding,
        coverage=coverage,
        warnings=warnings,
        **target,
        **_inherited_kwargs(outcome),
    )


def _comparison(
    coordinate: IntraLayerCoordinate,
    reference: IntraLayerTraceEntry,
    candidate: IntraLayerTraceEntry,
) -> Pass4Comparison:
    equivalent = reference.digest.value == candidate.digest.value
    decision = (
        Pass4ComparisonDecision.EQUIVALENT_OBSERVED_REPRESENTATION_DIGEST
        if equivalent
        else Pass4ComparisonDecision.MISMATCHING_OBSERVED_REPRESENTATION_DIGEST
    )
    return Pass4Comparison(
        coordinate,
        reference.shape,
        reference.digest.value,
        candidate.digest.value,
        decision,
    )


def _missing_stage_ids(
    start: Optional[IntraLayerCoordinate],
    end: Optional[IntraLayerCoordinate],
    common: tuple[IntraLayerCoordinate, ...],
) -> tuple[str, ...]:
    start_order = -1 if start is None else start.stage_order
    end_order = len(INTRA_LAYER_STAGES) if end is None else end.stage_order
    common_orders = {coordinate.stage_order for coordinate in common}
    return tuple(
        stage.stage_id
        for stage in INTRA_LAYER_STAGES
        if start_order < stage.stage_order <= end_order
        and stage.stage_order not in common_orders
    )


def _local_interval(
    start: Optional[IntraLayerCoordinate],
    end: IntraLayerCoordinate,
    common: tuple[IntraLayerCoordinate, ...],
) -> Pass4SuspectInterval:
    missing = _missing_stage_ids(start, end, common)
    if start is None:
        return Pass4SuspectInterval(
            "selected_layer_entry",
            None,
            True,
            "local_coordinate",
            end,
            None,
            LOCAL_EVIDENCE_ORIGIN,
            True,
            missing,
            f"[selected_layer_entry, {end.stage_id}]",
        )
    return Pass4SuspectInterval(
        "local_coordinate",
        start,
        False,
        "local_coordinate",
        end,
        None,
        LOCAL_EVIDENCE_ORIGIN,
        True,
        missing,
        f"({start.stage_id}, {end.stage_id}]",
    )


def _inherited_interval(
    start: IntraLayerCoordinate,
    common: tuple[IntraLayerCoordinate, ...],
    parent_coordinate,
) -> Pass4SuspectInterval:
    return Pass4SuspectInterval(
        "local_coordinate",
        start,
        False,
        "inherited_parent_boundary",
        None,
        parent_coordinate,
        INHERITED_BOUNDARY_EVIDENCE_ORIGIN,
        True,
        _missing_stage_ids(start, None, common),
        f"({start.stage_id}, parent:layer_output]",
    )


def _success_reasons(
    primary: Pass4ReasonCode,
    coverage: Pass4CoverageAnalysis,
) -> tuple[Pass4ReasonCode, ...]:
    if coverage.asymmetric:
        return (primary, Pass4ReasonCode.ASYMMETRIC_COVERAGE_RETAINED)
    return (primary,)


def localize_bound_intra_layer_inputs(
    parsed_inputs: Pass4TraceParsingOutcome,
) -> Pass4Result:
    """Localize one already-bound P4-8 outcome without artifact I/O."""

    if not isinstance(parsed_inputs, Pass4TraceParsingOutcome):
        raise TypeError(
            "parsed_inputs must be a Pass4TraceParsingOutcome"
        )

    warnings = _merged_warnings(parsed_inputs)
    if warnings is None:
        return _warning_overflow_result(parsed_inputs)
    if not parsed_inputs.proceed:
        return _terminal_result(parsed_inputs, warnings)

    reference = parsed_inputs.reference
    candidate = parsed_inputs.candidate
    parent = parsed_inputs.parent_outcome
    target = _target_kwargs(parsed_inputs, reference)

    empty_comparable: tuple[IntraLayerCoordinate, ...] = ()
    initial_coverage = _coverage(
        reference, candidate, empty_comparable
    )
    if not initial_coverage.common_captured:
        return _noncomparison_result(
            parsed_inputs,
            Pass4Status.INSUFFICIENT_COMMON_INTRA_LAYER_COVERAGE,
            Pass4ReasonCode.NO_COMMON_CAPTURED_COORDINATES,
            warnings,
            coverage=initial_coverage,
            target=target,
        )

    reference_entries = _entry_index(reference)
    candidate_entries = _entry_index(candidate)
    for coordinate in initial_coverage.common_captured:
        left = reference_entries[coordinate.logical_key]
        right = candidate_entries[coordinate.logical_key]
        reason = _alignment_reason(reference, candidate, left, right)
        if reason is not None:
            return _noncomparison_result(
                parsed_inputs,
                Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT,
                reason,
                warnings,
                coverage=initial_coverage,
                target=target,
            )

    comparable = initial_coverage.common_captured
    coverage = _coverage(reference, candidate, comparable)
    left_policy = reference.header.digest_contract
    right_policy = candidate.header.digest_contract
    if (
        left_policy != right_policy
        or not left_policy.frozen_policy_supported
        or not right_policy.frozen_policy_supported
    ):
        return _noncomparison_result(
            parsed_inputs,
            Pass4Status.COMPARISON_POLICY_UNAVAILABLE,
            Pass4ReasonCode.DIGEST_CONTRACT_UNKNOWN,
            warnings,
            coverage=coverage,
            target=target,
        )

    comparisons = tuple(
        _comparison(
            coordinate,
            reference_entries[coordinate.logical_key],
            candidate_entries[coordinate.logical_key],
        )
        for coordinate in coverage.common_comparable
    )
    mismatches = tuple(
        index
        for index, comparison in enumerate(comparisons)
        if not comparison.equivalent
    )
    common = coverage.common_comparable
    shared = {
        "parent_pass3": parent.parent,
        "reference_binding": parent.reference_binding,
        "candidate_binding": parent.candidate_binding,
        "coverage": coverage,
        "comparisons": comparisons,
        "evidence_level": EVIDENCE_LEVEL,
        "digest_contract_identity": DIGEST_VERSION,
        "warnings": warnings,
        **target,
        **_inherited_kwargs(parsed_inputs),
    }
    if mismatches:
        first = mismatches[0]
        mismatch = comparisons[first].coordinate
        last_equivalent = comparisons[first - 1].coordinate if first else None
        status = Pass4Status.OBSERVABLE_INTRA_LAYER_MISMATCH_FOUND
        return Pass4Result(
            status,
            STATUS_TO_DISPOSITION[status],
            FROZEN_EVIDENCE_CEILING,
            _success_reasons(
                Pass4ReasonCode.LOCAL_DIGEST_MISMATCH, coverage
            ),
            local_coverage_outcome=(
                Pass4LocalCoverageOutcome.LOCAL_MISMATCH_FOUND
            ),
            last_observed_equivalent_coordinate=last_equivalent,
            first_observed_local_mismatch_coordinate=mismatch,
            suspect_interval=_local_interval(
                last_equivalent, mismatch, common
            ),
            **shared,
        )

    status = Pass4Status.MISMATCH_BOUNDED_TO_INHERITED_CLOSING_BOUNDARY
    last_equivalent = comparisons[-1].coordinate
    closing = Pass4ClosingBoundaryDecision(
        parent.parent.parent_first_mismatch_coordinate
    )
    return Pass4Result(
        status,
        STATUS_TO_DISPOSITION[status],
        FROZEN_EVIDENCE_CEILING,
        _success_reasons(
            Pass4ReasonCode.NO_LOCAL_MISMATCH_BEFORE_INHERITED_BOUNDARY,
            coverage,
        ),
        local_coverage_outcome=(
            Pass4LocalCoverageOutcome
            .NO_MISMATCH_IN_COMMON_INTRA_LAYER_COVERAGE
        ),
        last_observed_equivalent_coordinate=last_equivalent,
        closing_boundary_decision=closing,
        suspect_interval=_inherited_interval(
            last_equivalent,
            common,
            closing.parent_coordinate,
        ),
        **shared,
    )


def run_coverage_scoped_intra_layer_localization(
    discovery_pass3: Pass3Result,
    discovery_pass3_artifact: CanonicalPass3Artifact,
    authoritative_pass3: Pass3Result,
    authoritative_pass3_artifact: CanonicalPass3Artifact,
    pass2_artifact: CanonicalPass2Artifact,
    *,
    discovery_reference_report: CanonicalRunReport,
    discovery_candidate_report: CanonicalRunReport,
    discovery_reference_trace: CanonicalLayerTrace,
    discovery_candidate_trace: CanonicalLayerTrace,
    authoritative_reference_report: CanonicalRunReport,
    authoritative_candidate_report: CanonicalRunReport,
    authoritative_reference_trace: CanonicalLayerTrace,
    authoritative_candidate_trace: CanonicalLayerTrace,
    discovery_pass2_artifact: Optional[CanonicalPass2Artifact] = None,
) -> Pass4Result:
    """Run the frozen binding, parsing, and localization gates in order."""

    parent = bind_pass4_parent_inputs(
        discovery_pass3,
        discovery_pass3_artifact,
        authoritative_pass3,
        authoritative_pass3_artifact,
        pass2_artifact,
        discovery_reference_report=discovery_reference_report,
        discovery_candidate_report=discovery_candidate_report,
        discovery_reference_trace=discovery_reference_trace,
        discovery_candidate_trace=discovery_candidate_trace,
        authoritative_reference_report=authoritative_reference_report,
        authoritative_candidate_report=authoritative_candidate_report,
        authoritative_reference_trace=authoritative_reference_trace,
        authoritative_candidate_trace=authoritative_candidate_trace,
        discovery_pass2_artifact=discovery_pass2_artifact,
    )
    parsed = parse_pass4_intra_layer_inputs(
        parent,
        authoritative_reference_trace,
        authoritative_candidate_trace,
    )
    return localize_bound_intra_layer_inputs(parsed)
