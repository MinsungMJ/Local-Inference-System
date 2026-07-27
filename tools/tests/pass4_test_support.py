#!/usr/bin/env python3
"""Bounded builders for Pass 4 result-model tests.

Every builder constructs typed model objects directly.  Nothing here parses a
runtime artifact, hashes bytes, compares digests, or localizes a mismatch.
"""

from __future__ import annotations

from typing import Optional

from lis_verify.pass3_model import (
    CheckpointCoordinate,
    CoverageState,
    DIGEST_DECISION_FIELD,
    DIGEST_DECISION_SEMANTICS,
    SummaryEvidenceLevel,
    SuspectInterval,
)
from lis_verify.pass4_contract import (
    DIGEST_VERSION,
    EVIDENCE_LEVEL,
    INTRA_LAYER_LAYOUT_NAME,
    INTRA_LAYER_LAYOUT_VERSION,
    MODEL_FAMILY,
    PASS3_DIGEST_VERSION,
    PHASE,
    STAGE_TAXONOMY,
    IntraLayerCoordinate,
    IntraLayerSideCoverage,
    MissingIntraLayerCoordinate,
    ParentSourceIdentity,
    Pass3ParentBinding,
    Pass3ParentClassification,
    Pass3ParentRole,
    Pass4Disposition,
    Pass4ReasonCode,
    Pass4Status,
    Pass4SuspectInterval,
    STATUS_TO_DISPOSITION,
    requested_coordinates,
)
from lis_verify.pass4_model import (
    FROZEN_EVIDENCE_CEILING,
    Pass3ParentEvidence,
    Pass4ClosingBoundaryDecision,
    Pass4Comparison,
    Pass4ComparisonDecision,
    Pass4CoverageAnalysis,
    Pass4LocalCoverageOutcome,
    Pass4Result,
    Pass4SourceBinding,
)


TARGET_STEP = 3
TARGET_LAYER = 8
TARGET_TOKEN = 11
PRECISION_PATH = "f32"
PASS2_TIER = "independent_rerun_verified"

PRIMARY_REASON_FOR_STATUS = {
    Pass4Status.OBSERVABLE_INTRA_LAYER_MISMATCH_FOUND:
        Pass4ReasonCode.LOCAL_DIGEST_MISMATCH,
    Pass4Status.MISMATCH_BOUNDED_TO_INHERITED_CLOSING_BOUNDARY:
        Pass4ReasonCode.NO_LOCAL_MISMATCH_BEFORE_INHERITED_BOUNDARY,
    Pass4Status.NOT_APPLICABLE:
        Pass4ReasonCode.PARENT_HAS_NO_OBSERVED_MISMATCH,
    Pass4Status.COMPARISON_BLOCKED_BY_PASS3:
        Pass4ReasonCode.PARENT_STATUS_BLOCKED,
    Pass4Status.INSUFFICIENT_COMMON_INTRA_LAYER_COVERAGE:
        Pass4ReasonCode.NO_COMMON_CAPTURED_COORDINATES,
    Pass4Status.SOURCE_BINDING_INCONSISTENT:
        Pass4ReasonCode.TRACE_SHA_MISMATCH,
    Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT:
        Pass4ReasonCode.SHAPE_OR_COUNT_ALIGNMENT_MISMATCH,
    Pass4Status.CHECKPOINT_SUMMARY_MALFORMED:
        Pass4ReasonCode.DIGEST_FIELD_MALFORMED,
    Pass4Status.COMPARISON_POLICY_UNAVAILABLE:
        Pass4ReasonCode.DIGEST_CONTRACT_UNKNOWN,
    Pass4Status.UNSUPPORTED_PARENT:
        Pass4ReasonCode.PARENT_CONTRACT_OR_FAMILY_UNSUPPORTED,
    Pass4Status.UNSUPPORTED_INTRA_LAYER_LAYOUT:
        Pass4ReasonCode.INTRA_LAYOUT_OR_TAXONOMY_UNSUPPORTED,
    Pass4Status.PARENT_REVALIDATION_INCONSISTENT:
        Pass4ReasonCode.DISCOVERY_REBOUND_LAYER_CHANGED,
}


def stages() -> tuple[IntraLayerCoordinate, ...]:
    return requested_coordinates(TARGET_STEP, TARGET_LAYER, TARGET_TOKEN)


def digest(seed: str) -> str:
    return "sha256:" + seed * 64


def parent_coordinate() -> CheckpointCoordinate:
    return CheckpointCoordinate(
        TARGET_STEP, TARGET_LAYER, "layer_output", 0, 0, 0, TARGET_LAYER
    )


def parent_interval() -> SuspectInterval:
    return SuspectInterval(
        "runtime_entry",
        None,
        parent_coordinate(),
        False,
        True,
        tuple(range(0, TARGET_LAYER)),
        f"[entry, {TARGET_LAYER}]",
    )


def source_identity(seed: str = "a") -> ParentSourceIdentity:
    return ParentSourceIdentity(
        digest(seed),
        digest("b" if seed != "b" else "c"),
        digest("c" if seed != "c" else "d"),
        "aset1:" + "d" * 32,
    )


def parent_binding(role: Pass3ParentRole) -> Pass3ParentBinding:
    return Pass3ParentBinding(
        role,
        digest("1"),
        digest("2"),
        source_identity("a"),
        source_identity("e"),
        role == Pass3ParentRole.AUTHORITATIVE_PASS3B,
    )


def eligible_parent(**overrides) -> Pass3ParentEvidence:
    values = dict(
        classification=Pass3ParentClassification.ELIGIBLE,
        discovery=parent_binding(Pass3ParentRole.DISCOVERY_PASS3A),
        authoritative=parent_binding(
            Pass3ParentRole.AUTHORITATIVE_PASS3B
        ),
        typed_artifact_coherence_verified=True,
        source_binding_verified=True,
        cross_generation_semantic_coherence_verified=True,
        discovery_selected_layer=TARGET_LAYER,
        authoritative_selected_layer=TARGET_LAYER,
        target_runtime_checkpoint_step=TARGET_STEP,
        parent_first_mismatch_coordinate=parent_coordinate(),
        parent_suspect_interval=parent_interval(),
        parent_evidence_level=SummaryEvidenceLevel.TIER1_BOUNDED_DIGEST,
        parent_decision_field=DIGEST_DECISION_FIELD,
        parent_decision_semantics=DIGEST_DECISION_SEMANTICS,
        parent_digest_contract_identity=PASS3_DIGEST_VERSION,
        pass2_reproduction_evidence_tier=PASS2_TIER,
    )
    values.update(overrides)
    return Pass3ParentEvidence(**values)


def classified_parent(
    classification: Pass3ParentClassification, **overrides
) -> Pass3ParentEvidence:
    values = dict(
        classification=classification,
        discovery=parent_binding(Pass3ParentRole.DISCOVERY_PASS3A),
        authoritative=parent_binding(
            Pass3ParentRole.AUTHORITATIVE_PASS3B
        ),
        typed_artifact_coherence_verified=True,
        source_binding_verified=False,
        cross_generation_semantic_coherence_verified=False,
        discovery_selected_layer=TARGET_LAYER,
        target_runtime_checkpoint_step=TARGET_STEP,
    )
    if classification == Pass3ParentClassification.PARENT_REVALIDATION_INCONSISTENT:
        values["authoritative_selected_layer"] = TARGET_LAYER + 1
    values.update(overrides)
    return Pass3ParentEvidence(**values)


def source_binding(role: str) -> Pass4SourceBinding:
    return Pass4SourceBinding(role, source_identity("a"), True)


def bindings() -> dict:
    return {
        "reference_binding": source_binding("reference_reproduction"),
        "candidate_binding": source_binding("candidate_reproduction"),
    }


def target_identity() -> dict:
    return {
        "target_runtime_checkpoint_step": TARGET_STEP,
        "target_layer": TARGET_LAYER,
        "target_token_position": TARGET_TOKEN,
        "phase": PHASE,
        "model_family": MODEL_FAMILY,
        "precision_path": PRECISION_PATH,
        "layout_name": INTRA_LAYER_LAYOUT_NAME,
        "layout_version": INTRA_LAYER_LAYOUT_VERSION,
        "stage_taxonomy": STAGE_TAXONOMY,
    }


def side_coverage(
    captured: tuple[IntraLayerCoordinate, ...],
    requested: Optional[tuple[IntraLayerCoordinate, ...]] = None,
) -> IntraLayerSideCoverage:
    requested = stages() if requested is None else requested
    keys = {item.logical_key for item in captured}
    missing = tuple(
        MissingIntraLayerCoordinate(
            item, CoverageState.NOT_CAPTURED, "fixture sparse coverage"
        )
        for item in requested
        if item.logical_key not in keys
    )
    return IntraLayerSideCoverage(requested, captured, missing)


def coverage(
    reference_captured: tuple[IntraLayerCoordinate, ...],
    candidate_captured: Optional[tuple[IntraLayerCoordinate, ...]] = None,
    *,
    common_comparable: Optional[tuple[IntraLayerCoordinate, ...]] = None,
) -> Pass4CoverageAnalysis:
    if candidate_captured is None:
        candidate_captured = reference_captured
    reference = side_coverage(reference_captured)
    candidate = side_coverage(candidate_captured)
    candidate_keys = {item.logical_key for item in candidate_captured}
    reference_keys = {item.logical_key for item in reference_captured}
    common = tuple(
        item
        for item in reference_captured
        if item.logical_key in candidate_keys
    )
    reference_only = tuple(
        item
        for item in reference_captured
        if item.logical_key not in candidate_keys
    )
    candidate_only = tuple(
        item
        for item in candidate_captured
        if item.logical_key not in reference_keys
    )
    return Pass4CoverageAnalysis(
        reference,
        candidate,
        common,
        common if common_comparable is None else common_comparable,
        reference_only,
        candidate_only,
    )


def comparison(
    coordinate: IntraLayerCoordinate, *, equivalent: bool = True
) -> Pass4Comparison:
    return Pass4Comparison(
        coordinate,
        (2, 2),
        digest("a"),
        digest("a") if equivalent else digest("b"),
        Pass4ComparisonDecision.EQUIVALENT_OBSERVED_REPRESENTATION_DIGEST
        if equivalent
        else Pass4ComparisonDecision.MISMATCHING_OBSERVED_REPRESENTATION_DIGEST,
    )


def local_interval(
    start: Optional[IntraLayerCoordinate],
    end: IntraLayerCoordinate,
    missing: tuple[str, ...] = (),
) -> Pass4SuspectInterval:
    if start is None:
        return Pass4SuspectInterval(
            "selected_layer_entry",
            None,
            True,
            "local_coordinate",
            end,
            None,
            "pass4_local",
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
        "pass4_local",
        True,
        missing,
        f"({start.stage_id}, {end.stage_id}]",
    )


def inherited_interval(
    start: Optional[IntraLayerCoordinate],
    missing: tuple[str, ...] = (),
) -> Pass4SuspectInterval:
    notation_start = (
        "[selected_layer_entry" if start is None else f"({start.stage_id}"
    )
    return Pass4SuspectInterval(
        "selected_layer_entry" if start is None else "local_coordinate",
        start,
        start is None,
        "inherited_parent_boundary",
        None,
        parent_coordinate(),
        "authoritative_pass3",
        True,
        missing,
        f"{notation_start}, parent:layer_output]",
    )


def local_mismatch_result(**overrides) -> Pass4Result:
    coordinates = stages()
    captured = (coordinates[10], coordinates[13])
    values = dict(
        parent_pass3=eligible_parent(),
        coverage=coverage(captured),
        comparisons=(
            comparison(coordinates[10]),
            comparison(coordinates[13], equivalent=False),
        ),
        local_coverage_outcome=(
            Pass4LocalCoverageOutcome.LOCAL_MISMATCH_FOUND
        ),
        last_observed_equivalent_coordinate=coordinates[10],
        first_observed_local_mismatch_coordinate=coordinates[13],
        suspect_interval=local_interval(
            coordinates[10],
            coordinates[13],
            ("post_attention_residual", "mlp_norm_output"),
        ),
        evidence_level=EVIDENCE_LEVEL,
        digest_contract_identity=DIGEST_VERSION,
        **bindings(),
        **target_identity(),
    )
    values.update(overrides)
    disposition = values.pop(
        "disposition", Pass4Disposition.SUSPECT_INTERVAL_AVAILABLE
    )
    reason_codes = values.pop(
        "reason_codes", (Pass4ReasonCode.LOCAL_DIGEST_MISMATCH,)
    )
    ceiling = values.pop("evidence_ceiling", FROZEN_EVIDENCE_CEILING)
    return Pass4Result(
        Pass4Status.OBSERVABLE_INTRA_LAYER_MISMATCH_FOUND,
        disposition,
        ceiling,
        reason_codes,
        **values,
    )


def inherited_boundary_result(**overrides) -> Pass4Result:
    coordinates = stages()
    captured = (coordinates[16],)
    values = dict(
        parent_pass3=eligible_parent(),
        coverage=coverage(captured),
        comparisons=(comparison(coordinates[16]),),
        local_coverage_outcome=(
            Pass4LocalCoverageOutcome
            .NO_MISMATCH_IN_COMMON_INTRA_LAYER_COVERAGE
        ),
        last_observed_equivalent_coordinate=coordinates[16],
        closing_boundary_decision=Pass4ClosingBoundaryDecision(
            parent_coordinate()
        ),
        suspect_interval=inherited_interval(coordinates[16]),
        evidence_level=EVIDENCE_LEVEL,
        digest_contract_identity=DIGEST_VERSION,
        **bindings(),
        **target_identity(),
    )
    values.update(overrides)
    disposition = values.pop(
        "disposition", Pass4Disposition.SUSPECT_INTERVAL_AVAILABLE
    )
    reason_codes = values.pop(
        "reason_codes",
        (Pass4ReasonCode.NO_LOCAL_MISMATCH_BEFORE_INHERITED_BOUNDARY,),
    )
    ceiling = values.pop("evidence_ceiling", FROZEN_EVIDENCE_CEILING)
    return Pass4Result(
        Pass4Status.MISMATCH_BOUNDED_TO_INHERITED_CLOSING_BOUNDARY,
        disposition,
        ceiling,
        reason_codes,
        **values,
    )


def _noncomparison_values(status: Pass4Status) -> dict:
    coordinates = stages()
    if status == Pass4Status.NOT_APPLICABLE:
        return {
            "parent_pass3": classified_parent(
                Pass3ParentClassification.NOT_APPLICABLE
            )
        }
    if status == Pass4Status.COMPARISON_BLOCKED_BY_PASS3:
        return {
            "parent_pass3": classified_parent(
                Pass3ParentClassification.COMPARISON_BLOCKED_BY_PASS3
            )
        }
    if status == Pass4Status.UNSUPPORTED_PARENT:
        return {
            "parent_pass3": classified_parent(
                Pass3ParentClassification.UNSUPPORTED_PARENT
            )
        }
    if status == Pass4Status.PARENT_REVALIDATION_INCONSISTENT:
        return {
            "parent_pass3": classified_parent(
                Pass3ParentClassification.PARENT_REVALIDATION_INCONSISTENT
            )
        }
    if status == Pass4Status.SOURCE_BINDING_INCONSISTENT:
        return {"parent_pass3": eligible_parent()}
    if status == Pass4Status.UNSUPPORTED_INTRA_LAYER_LAYOUT:
        return {"parent_pass3": eligible_parent(), **bindings()}
    if status == Pass4Status.CHECKPOINT_SUMMARY_MALFORMED:
        return {
            "parent_pass3": eligible_parent(),
            **bindings(),
            **target_identity(),
        }
    if status == Pass4Status.INSUFFICIENT_COMMON_INTRA_LAYER_COVERAGE:
        return {
            "parent_pass3": eligible_parent(),
            "coverage": coverage(
                (coordinates[0],), (coordinates[1],), common_comparable=()
            ),
            **bindings(),
            **target_identity(),
        }
    if status == Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT:
        return {
            "parent_pass3": eligible_parent(),
            "coverage": coverage(
                (coordinates[10], coordinates[13]), common_comparable=()
            ),
            **bindings(),
            **target_identity(),
        }
    if status == Pass4Status.COMPARISON_POLICY_UNAVAILABLE:
        return {
            "parent_pass3": eligible_parent(),
            "coverage": coverage((coordinates[10], coordinates[13])),
            **bindings(),
            **target_identity(),
        }
    raise AssertionError(f"unsupported non-comparison status {status}")


def result_for(status: Pass4Status, **overrides) -> Pass4Result:
    """Build one valid result for any frozen Pass 4 status."""
    if status == Pass4Status.OBSERVABLE_INTRA_LAYER_MISMATCH_FOUND:
        return local_mismatch_result(**overrides)
    if status == Pass4Status.MISMATCH_BOUNDED_TO_INHERITED_CLOSING_BOUNDARY:
        return inherited_boundary_result(**overrides)
    values = _noncomparison_values(status)
    values.update(overrides)
    reason_codes = values.pop(
        "reason_codes", (PRIMARY_REASON_FOR_STATUS[status],)
    )
    disposition = values.pop(
        "disposition", STATUS_TO_DISPOSITION[status]
    )
    ceiling = values.pop("evidence_ceiling", FROZEN_EVIDENCE_CEILING)
    return Pass4Result(status, disposition, ceiling, reason_codes, **values)
