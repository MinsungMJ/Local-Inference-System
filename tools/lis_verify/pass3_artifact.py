"""Bounded deterministic serialization for Pass 3 localization evidence."""

from __future__ import annotations

import json

from .pass3_model import Pass3Result


MAX_SERIALIZED_COMPARISONS = 256


def _coordinate(value):
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


def _coverage_entry(value):
    return {
        "coordinate": _coordinate(value.coordinate),
        "state": value.state.value,
        "detail": value.detail,
    }


def _binding(value):
    if value is None:
        return None
    return {
        "role": value.role,
        "run_report_sha256": value.run_report_sha256,
        "layer_trace_sha256": value.trace_sha256,
        "artifact_set_id": value.artifact_set_id,
        "semantic_manifest_sha256": value.semantic_manifest_sha256,
    }


def _pass2_evidence(value):
    if value is None:
        return None
    return {
        "reproduction_evidence_tier": value.reproduction_evidence_tier,
        "generated_token_step": value.generated_token_step,
        "runtime_checkpoint_step": value.runtime_checkpoint_step,
        "localization_ref_sha256": value.localization_ref_sha256,
        "reference_original_run_report_sha256": (
            value.reference_original_run_report_sha256
        ),
        "candidate_original_run_report_sha256": (
            value.candidate_original_run_report_sha256
        ),
        "reference_reproduction_sha256": (
            value.reference_reproduction_sha256
        ),
        "candidate_reproduction_sha256": (
            value.candidate_reproduction_sha256
        ),
        "checkpoint_step_evidence": value.checkpoint_step_evidence,
        "verdict_strength_limit": value.verdict_strength_limit,
        "thread_count_caveat": value.thread_count_caveat,
    }


def _comparison(value):
    return {
        "coordinate": _coordinate(value.coordinate),
        "equivalent": value.equivalent,
        "mismatching_fields": list(value.mismatching_fields),
        "field_results": [
            {
                "field_name": field.field_name,
                "disposition": field.disposition.value,
                "equivalent": field.equivalent,
                "abs_diff": field.abs_diff,
                "resolved_abs_floor": field.resolved_abs_floor,
                "resolved_rel_band": field.resolved_rel_band,
            }
            for field in value.field_results
        ],
        "evidence_level": value.evidence_level.value,
        "warnings": list(value.warnings),
    }


def serialize(result: Pass3Result) -> dict:
    """Serialize only bounded summaries, binding identities, and coverage."""
    if not isinstance(result, Pass3Result):
        raise TypeError("result must be a Pass3Result")
    coverage = result.coverage
    interval = result.suspect_interval
    comparisons = result.comparisons[:MAX_SERIALIZED_COMPARISONS]
    return {
        "schema": result.schema,
        "kind": result.kind,
        "contract_version": result.contract_version,
        "pass2_artifact_sha256": result.pass2_artifact_sha256,
        "pass2_object_artifact_coherence_verified": (
            result.pass2_object_artifact_coherence_verified
        ),
        "pass2_evidence": _pass2_evidence(result.pass2_evidence),
        "source_binding": {
            "reference": _binding(result.reference_binding),
            "candidate": _binding(result.candidate_binding),
        },
        "checkpoint_artifact_binding_verified": (
            result.checkpoint_artifact_binding_verified
        ),
        "target": {
            "runtime_checkpoint_step": result.target_runtime_checkpoint_step
        },
        "coverage": {
            "reference_requested": [
                _coordinate(value) for value in coverage.reference_requested
            ],
            "reference_captured": [
                _coordinate(value) for value in coverage.reference_captured
            ],
            "candidate_requested": [
                _coordinate(value) for value in coverage.candidate_requested
            ],
            "candidate_captured": [
                _coordinate(value) for value in coverage.candidate_captured
            ],
            "common_captured": [
                _coordinate(value) for value in coverage.common_captured
            ],
            "reference_only": [
                _coordinate(value) for value in coverage.reference_only
            ],
            "candidate_only": [
                _coordinate(value) for value in coverage.candidate_only
            ],
            "common_comparable": [
                _coordinate(value) for value in coverage.common_comparable
            ],
            "reference_missing": [
                _coverage_entry(value) for value in coverage.reference_missing
            ],
            "candidate_missing": [
                _coverage_entry(value) for value in coverage.candidate_missing
            ],
        },
        "comparisons": {
            "items": [_comparison(value) for value in comparisons],
            "total_count": len(result.comparisons),
            "serialized_count": len(comparisons),
            "truncated": len(comparisons) != len(result.comparisons),
        },
        "localization": {
            "last_observed_equivalent_coordinate": _coordinate(
                result.last_observed_equivalent_coordinate
            ),
            "last_observed_equivalent_layer": (
                result.last_observed_equivalent_coordinate.layer_index
                if result.last_observed_equivalent_coordinate is not None
                else None
            ),
            "first_observed_mismatch_coordinate": _coordinate(
                result.first_observed_mismatch_coordinate
            ),
            "first_observed_mismatching_layer": (
                result.first_observed_mismatch_coordinate.layer_index
                if result.first_observed_mismatch_coordinate is not None
                else None
            ),
            "earliest_observable_suspect_layer": (
                result.earliest_observable_suspect_layer
            ),
            "suspect_interval": (
                {
                    "start_boundary": interval.start_boundary,
                    "last_observed_equivalent_coordinate": _coordinate(
                        interval.last_observed_equivalent_coordinate
                    ),
                    "first_observed_mismatch_coordinate": _coordinate(
                        interval.first_observed_mismatch_coordinate
                    ),
                    "start_exclusive": interval.start_exclusive,
                    "end_inclusive": interval.end_inclusive,
                    "unobserved_layer_indices": list(
                        interval.unobserved_layer_indices
                    ),
                    "notation": interval.notation,
                }
                if interval is not None
                else None
            ),
        },
        "evidence": {
            "decision_field": result.decision_field,
            "decision_semantics": result.decision_semantics,
            "evidence_level": result.evidence_level.value,
            "digest_contract_identity": result.digest_contract_identity,
        },
        "pass3_status": result.status.value,
        "downstream_disposition": result.downstream_disposition.value,
        "reason_codes": [value.value for value in result.reason_codes],
        "inherited_pass2_reason_codes": list(
            result.inherited_pass2_reason_codes
        ),
        "inherited_pass1_reason_codes": list(
            result.inherited_pass1_reason_codes
        ),
        "inherited_pass0_reason_codes": list(
            result.inherited_pass0_reason_codes
        ),
        "warnings": list(result.warnings),
        "semantic_limits": {
            "bounded_equality_semantics": result.equality_semantics,
            "non_confirmation_semantics": result.non_confirmation_semantics,
            "full_tensor_comparison_performed": False,
            "stage_localization_performed": False,
            "numeric_confirmation_performed": False,
            "pass4_or_pass5_readiness_certified": False,
            "automatic_frozen_success_mapping": False,
        },
    }


def to_json(result: Pass3Result, indent: int = 2) -> str:
    return json.dumps(serialize(result), indent=indent)
