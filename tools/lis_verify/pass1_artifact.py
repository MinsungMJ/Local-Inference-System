"""Bounded serialization for the Pass 1 ``token_localization`` artifact."""

from __future__ import annotations

import json

from .pass1_model import (
    DEFAULT_EMBEDDED_PREFIX_CAP,
    Pass1Result,
    PrefixAvailability,
)


def _digest(value):
    return {
        "algorithm": value.algorithm,
        "value": value.value,
        "size_bytes": value.size_bytes,
        "valid": value.valid,
        "matches_array": value.matches_array,
    }


def _selected(value):
    if value is None:
        return None
    sequence = value.selected_tokens
    return {
        "level": sequence.evidence_level.value,
        "source_locations": list(sequence.source_locations),
        "declared_count": sequence.declared_count,
        "observed_array_length": sequence.observed_length,
        "digest": _digest(sequence.digest),
        "stop_reason": sequence.stop_reason,
        "generation_limit": sequence.generation_limit,
    }


def _localization(value):
    if value is None:
        return None
    return {
        "generated_token_step": value.generated_token_step,
        "runtime_checkpoint_step": value.runtime_checkpoint_step,
        "reference_selected_token_id": value.reference_selected_token_id,
        "candidate_selected_token_id": value.candidate_selected_token_id,
        "matched_generated_prefix_length": (
            value.matched_generated_prefix_length
        ),
        "observed_reference_generated_length": (
            value.observed_reference_generated_length
        ),
        "observed_candidate_generated_length": (
            value.observed_candidate_generated_length
        ),
        "mismatch_kind": (
            value.mismatch_kind.value if value.mismatch_kind is not None else None
        ),
    }


def _prefix(value):
    if value is None:
        return {
            "availability": PrefixAvailability.NOT_APPLICABLE.value,
            "generated_prefix_token_ids": None,
            "generated_prefix_token_count": 0,
            "generated_prefix_sha256": None,
            "prefix_start_generated_step": 0,
            "prefix_end_generated_step_exclusive": 0,
        }
    embedded = (
        list(value.exact_token_ids)
        if value.availability == PrefixAvailability.EMBEDDED
        and value.token_count <= DEFAULT_EMBEDDED_PREFIX_CAP
        else None
    )
    return {
        "availability": value.availability.value,
        "generated_prefix_token_ids": embedded,
        "generated_prefix_token_count": value.token_count,
        "generated_prefix_sha256": value.sha256,
        "prefix_start_generated_step": value.prefix_start_generated_step,
        "prefix_end_generated_step_exclusive": (
            value.prefix_end_generated_step_exclusive
        ),
    }


def serialize(
    result: Pass1Result, *, embed_calibration: bool = False
) -> dict:
    calibration = result.calibration_ref
    return {
        "schema": result.schema,
        "kind": result.kind,
        "contract_version": result.contract_version,
        "comparison_mode": result.comparison_mode.value,
        "pass0_verdict": result.pass0_verdict.value,
        "comparison_eligibility": result.comparison_eligibility.value,
        "source_binding": {
            "reference_run_report_sha256": (
                result.source_binding.reference.run_report_sha256
            ),
            "candidate_run_report_sha256": (
                result.source_binding.candidate.run_report_sha256
            ),
            "verified": result.source_binding_verified,
        },
        "pass1_status": result.status.value,
        "evidence_scope": result.evidence_scope,
        "evidence_completeness": result.evidence_completeness.value,
        "compatibility": {
            "comparison_blocked": result.compatibility.comparison_blocked,
            "boundary": result.compatibility.boundary,
            "model_family_state": result.compatibility.model_family_state,
            "config_state": result.compatibility.config_state,
            "prompt_identity_evidence": (
                result.compatibility.prompt_identity_evidence
            ),
        },
        "selected_token_evidence": {
            "reference": _selected(result.reference),
            "candidate": _selected(result.candidate),
        },
        "token_localization": _localization(result.localization),
        "prefix_for_reproduction": _prefix(
            result.prefix_for_reproduction
        ),
        "pass2_disposition": result.pass2_disposition.value,
        "calibration_ref": {
            "sha256": calibration.sha256,
            "summary": {
                "comparison_mode": calibration.comparison_mode,
                "pass0_verdict": calibration.pass0_verdict,
                "comparison_eligibility": calibration.comparison_eligibility,
                "prompt_identity_evidence": (
                    calibration.prompt_identity_evidence
                ),
                "verdict_strength_limit": calibration.verdict_strength_limit,
                "blocking_reasons": list(calibration.blocking_reasons),
                "oracle_scope": calibration.oracle_scope,
            },
            "embedded": (
                json.loads(calibration.canonical_json)
                if embed_calibration
                else None
            ),
        },
        "verdict_strength_limit": result.verdict_strength_limit.value,
        "reason_codes": [reason.value for reason in result.reason_codes],
        "inherited_pass0_reason_codes": list(
            result.inherited_pass0_reason_codes
        ),
        "blocking_reasons": list(result.blocking_reasons),
        "warnings": list(result.warnings),
    }


def to_json(
    result: Pass1Result,
    indent: int = 2,
    *,
    embed_calibration: bool = False,
) -> str:
    return json.dumps(
        serialize(result, embed_calibration=embed_calibration),
        indent=indent,
    )
