"""Bounded deterministic serialization for Pass 4 localization evidence."""

from __future__ import annotations

import json

from .pass1_inputs import canonical_json
from .pass4_contract import EVIDENCE_LEVEL
from .pass4_model import Pass4Result


MAX_CANONICAL_ARTIFACT_BYTES = 256 * 1024


def _parent_coordinate(value):
    if value is None:
        return None
    return {
        "runtime_checkpoint_step": value.runtime_checkpoint_step,
        "layer_index": value.layer_index,
        "tensor_role": value.tensor_role,
        "batch_index": value.batch_index,
        "sequence_index": value.sequence_index,
        "stage_order": value.stage_order,
        "execution_ordinal": value.execution_ordinal,
    }


def _local_coordinate(value):
    if value is None:
        return None
    return {
        "runtime_checkpoint_step": value.runtime_checkpoint_step,
        "layer_index": value.layer_index,
        "stage_id": value.stage_id,
        "tensor_role": value.tensor_role,
        "batch_index": value.batch_index,
        "sequence_index": value.sequence_index,
        "token_position": value.token_position,
        "stage_order": value.stage_order,
        "execution_ordinal": value.execution_ordinal,
    }


def _identity(value):
    return {
        "run_report_sha256": value.run_report_sha256,
        "layer_trace_sha256": value.layer_trace_sha256,
        "semantic_manifest_sha256": value.semantic_manifest_sha256,
        "artifact_set_id": value.artifact_set_id,
    }


def _parent_binding(value):
    return {
        "role": value.role.value,
        "canonical_pass3_artifact_sha256": (
            value.canonical_pass3_artifact_sha256
        ),
        "pass2_artifact_sha256": value.pass2_artifact_sha256,
        "reference": _identity(value.reference),
        "candidate": _identity(value.candidate),
        "authorizes_pass4_evidence": value.authorizes_pass4_evidence,
    }


def _parent_interval(value):
    if value is None:
        return None
    return {
        "start_boundary": value.start_boundary,
        "last_observed_equivalent_coordinate": _parent_coordinate(
            value.last_observed_equivalent_coordinate
        ),
        "first_observed_mismatch_coordinate": _parent_coordinate(
            value.first_observed_mismatch_coordinate
        ),
        "start_exclusive": value.start_exclusive,
        "end_inclusive": value.end_inclusive,
        "unobserved_layer_indices": list(value.unobserved_layer_indices),
        "notation": value.notation,
    }


def _parent_evidence(value):
    if value is None:
        return None
    return {
        "classification": value.classification.value,
        "discovery": _parent_binding(value.discovery),
        "authoritative": _parent_binding(value.authoritative),
        "typed_artifact_coherence_verified": (
            value.typed_artifact_coherence_verified
        ),
        "source_binding_verified": value.source_binding_verified,
        "cross_generation_semantic_coherence_verified": (
            value.cross_generation_semantic_coherence_verified
        ),
        "discovery_selected_layer": value.discovery_selected_layer,
        "authoritative_selected_layer": (
            value.authoritative_selected_layer
        ),
        "target_runtime_checkpoint_step": (
            value.target_runtime_checkpoint_step
        ),
        "parent_first_mismatch_coordinate": _parent_coordinate(
            value.parent_first_mismatch_coordinate
        ),
        "parent_last_observed_equivalent_coordinate": _parent_coordinate(
            value.parent_last_observed_equivalent_coordinate
        ),
        "parent_suspect_interval": _parent_interval(
            value.parent_suspect_interval
        ),
        "parent_evidence_level": (
            value.parent_evidence_level.value
            if value.parent_evidence_level is not None
            else None
        ),
        "parent_decision_field": value.parent_decision_field,
        "parent_decision_semantics": value.parent_decision_semantics,
        "parent_digest_contract_identity": (
            value.parent_digest_contract_identity
        ),
        "pass2_reproduction_evidence_tier": (
            value.pass2_reproduction_evidence_tier
        ),
    }


def _source_binding(value):
    if value is None:
        return None
    identity = _identity(value.identity)
    return {
        "role": value.role,
        **identity,
        "parent_recorded_trace_binding_verified": (
            value.parent_recorded_trace_binding_verified
        ),
    }


def _missing(value):
    return {
        "coordinate": _local_coordinate(value.coordinate),
        "state": value.state.value,
        "detail": value.detail,
    }


def _side_coverage(value):
    if value is None:
        return {"requested": [], "captured": [], "missing": []}
    return {
        "requested": [
            _local_coordinate(item) for item in value.requested_coordinates
        ],
        "captured": [
            _local_coordinate(item) for item in value.captured_coordinates
        ],
        "missing": [_missing(item) for item in value.missing_coordinates],
    }


def _coverage(value):
    if value is None:
        return {
            "reference": _side_coverage(None),
            "candidate": _side_coverage(None),
            "common_captured": [],
            "common_comparable": [],
            "reference_only": [],
            "candidate_only": [],
        }
    return {
        "reference": _side_coverage(value.reference),
        "candidate": _side_coverage(value.candidate),
        "common_captured": [
            _local_coordinate(item) for item in value.common_captured
        ],
        "common_comparable": [
            _local_coordinate(item) for item in value.common_comparable
        ],
        "reference_only": [
            _local_coordinate(item) for item in value.reference_only
        ],
        "candidate_only": [
            _local_coordinate(item) for item in value.candidate_only
        ],
    }


def _comparison(value):
    return {
        "coordinate": _local_coordinate(value.coordinate),
        "shape": list(value.shape),
        "reference_digest": value.reference_digest,
        "candidate_digest": value.candidate_digest,
        "decision": value.decision.value,
        "equivalent": value.equivalent,
        "decision_semantics": value.decision_semantics,
    }


def _closing_boundary(value):
    if value is None:
        return None
    return {
        "boundary_id": value.boundary_id,
        "evidence_origin": value.evidence_origin,
        "parent_coordinate": _parent_coordinate(value.parent_coordinate),
        "parent_decision_field": value.parent_decision_field,
        "parent_decision_semantics": value.parent_decision_semantics,
        "parent_digest_contract_identity": (
            value.parent_digest_contract_identity
        ),
        "parent_evidence_level": value.parent_evidence_level.value,
    }


def _interval(value):
    if value is None:
        return None
    return {
        "start_kind": value.start_kind,
        "start_local_coordinate": _local_coordinate(
            value.start_local_coordinate
        ),
        "start_inclusive": value.start_inclusive,
        "end_kind": value.end_kind,
        "end_local_coordinate": _local_coordinate(
            value.end_local_coordinate
        ),
        "end_parent_coordinate": _parent_coordinate(
            value.end_parent_coordinate
        ),
        "end_evidence_origin": value.end_evidence_origin,
        "end_inclusive": value.end_inclusive,
        "missing_local_stage_ids": list(value.missing_local_stage_ids),
        "notation": value.notation,
    }


def _localization(result: Pass4Result):
    if (
        result.local_coverage_outcome is None
        and result.suspect_interval is None
        and result.closing_boundary_decision is None
    ):
        return None
    parent_coordinate = (
        result.parent_pass3.parent_first_mismatch_coordinate
        if result.parent_pass3 is not None
        else None
    )
    return {
        "local_coverage_outcome": (
            result.local_coverage_outcome.value
            if result.local_coverage_outcome is not None
            else None
        ),
        "last_observed_equivalent_coordinate": _local_coordinate(
            result.last_observed_equivalent_coordinate
        ),
        "first_observed_local_mismatch_coordinate": _local_coordinate(
            result.first_observed_local_mismatch_coordinate
        ),
        "authoritative_parent_coordinate": _parent_coordinate(
            parent_coordinate
        ),
        "closing_boundary_decision": _closing_boundary(
            result.closing_boundary_decision
        ),
        "suspect_interval": _interval(result.suspect_interval),
    }


def _canonical_size(value: dict) -> int:
    return len(canonical_json(value).encode("utf-8"))


def serialize(result: Pass4Result) -> dict:
    """Serialize the complete bounded result without sorting or truncating it."""

    if not isinstance(result, Pass4Result):
        raise TypeError("result must be a Pass4Result")
    comparisons = [_comparison(value) for value in result.comparisons]
    nonclaims = result.evidence_ceiling.as_mapping()
    artifact = {
        "schema": result.schema,
        "kind": result.kind,
        "contract_version": result.contract_version,
        "contract_namespace": result.contract_namespace,
        "parent_pass3": _parent_evidence(result.parent_pass3),
        "source_binding": {
            "reference": _source_binding(result.reference_binding),
            "candidate": _source_binding(result.candidate_binding),
        },
        "target": {
            "runtime_checkpoint_step": (
                result.target_runtime_checkpoint_step
            ),
            "layer": result.target_layer,
            "token_position": result.target_token_position,
            "phase": result.phase,
            "model_family": result.model_family,
            "precision_path": result.precision_path,
        },
        "layout": {
            "layout_name": result.layout_name,
            "layout_version": result.layout_version,
            "stage_taxonomy": result.stage_taxonomy,
        },
        "coverage": _coverage(result.coverage),
        "comparisons": {
            "items": comparisons,
            "total_count": len(result.comparisons),
            "serialized_count": len(comparisons),
            "truncated": False,
        },
        "localization": _localization(result),
        "status": result.status.value,
        "disposition": result.disposition.value,
        "reason_codes": [value.value for value in result.reason_codes],
        "inherited_reason_codes": {
            "pass3": list(result.inherited_pass3_reason_codes),
            "pass2": list(result.inherited_pass2_reason_codes),
            "pass1": list(result.inherited_pass1_reason_codes),
            "pass0": list(result.inherited_pass0_reason_codes),
        },
        "warnings": list(result.warnings),
        "evidence_level": result.evidence_level,
        "digest_contract_identity": result.digest_contract_identity,
        "evidence_ceiling": {
            "evidence_level": EVIDENCE_LEVEL,
            "nonclaims": dict(nonclaims),
        },
        "nonclaims": dict(nonclaims),
    }
    if _canonical_size(artifact) > MAX_CANONICAL_ARTIFACT_BYTES:
        raise ValueError(
            "canonical Pass 4 artifact exceeds the 256 KiB bound"
        )
    return artifact


def to_json(result: Pass4Result, indent: int = 2) -> str:
    return json.dumps(
        serialize(result),
        indent=indent,
        ensure_ascii=False,
        allow_nan=False,
    )
