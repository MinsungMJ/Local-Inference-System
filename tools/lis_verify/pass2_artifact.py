"""Bounded serialization for the Pass 2 reproduction artifact."""

from __future__ import annotations

import json

from .pass2_model import (
    Pass2Result,
    REPRODUCTION_VERIFIED_SEMANTICS,
)


def _prefix(value):
    return {
        "status": value.status,
        "expected_token_count": value.expected_token_count,
        "expected_sha256": value.expected_sha256,
        "verified_sides": list(value.verified_sides),
        "failed_side": value.failed_side,
        "first_diff_index": value.first_diff_index,
        "mismatch_kind": value.mismatch_kind,
    }


def _policy(value):
    return {
        "status": value.status,
        "build_continuity_verified": value.build_continuity_verified,
        "inherited_calibration_sha256": (
            value.inherited_calibration_sha256
        ),
        "runs": [
            {
                "role": run.role,
                "binary_fingerprint": run.binary_fingerprint,
            }
            for run in value.runs
        ],
    }


def _context(value):
    return {
        "status": value.status,
        "runs": [
            {
                "role": run.role,
                "prompt_token_count": run.prompt_token_count,
                "context_position": run.context_position,
                "batch_size": run.batch_size,
                "sequence_index": run.sequence_index,
                "thread_count": run.thread_count,
            }
            for run in value.runs
        ],
    }


def serialize(
    result: Pass2Result, *, embed_localization: bool = False
) -> dict:
    binding = result.source_binding
    target = result.target
    checkpoint = result.checkpoint_step_reproduction
    localization = result.localization_ref
    return {
        "schema": result.schema,
        "kind": result.kind,
        "contract_version": result.contract_version,
        "comparison_mode": result.comparison_mode.value,
        "pass0_verdict": result.pass0_verdict.value,
        "comparison_eligibility": result.comparison_eligibility.value,
        "pass1_status": result.pass1_status.value,
        "pass1_pass2_disposition": (
            result.pass1_pass2_disposition.value
        ),
        "source_binding": {
            "pass1_artifact_sha256": binding.pass1_artifact_sha256,
            "reference_original_run_report_sha256": (
                binding.reference_original_run_report_sha256
            ),
            "candidate_original_run_report_sha256": (
                binding.candidate_original_run_report_sha256
            ),
            "reference_original_verified": (
                binding.reference_original_verified
            ),
            "candidate_original_verified": (
                binding.candidate_original_verified
            ),
            "reference_reproduction_sha256": (
                binding.reference_reproduction_sha256
            ),
            "candidate_reproduction_sha256": (
                binding.candidate_reproduction_sha256
            ),
            "reproduction_verified": binding.reproduction_verified,
            "verified": binding.verified,
        },
        "source_binding_verified": result.source_binding_verified,
        "pass2_status": result.status.value,
        "reproduction_evidence_tier": (
            result.reproduction_evidence_tier.value
        ),
        "reproduction_verified_semantics": (
            REPRODUCTION_VERIFIED_SEMANTICS
        ),
        "target": {
            "generated_token_step": target.generated_token_step,
            "expected_runtime_checkpoint_step": (
                target.expected_runtime_checkpoint_step
            ),
            "matched_generated_prefix_length": (
                target.matched_generated_prefix_length
            ),
        },
        "prompt_reproduction": {
            "status": result.prompt_reproduction.status,
            "inherited_prompt_identity_evidence": (
                result.prompt_reproduction.inherited_prompt_identity_evidence
            ),
        },
        "prefix_reproduction": _prefix(result.prefix_reproduction),
        "policy_reproduction": _policy(result.policy_reproduction),
        "checkpoint_step_reproduction": {
            "status": checkpoint.status,
            "generated_token_step": checkpoint.generated_token_step,
            "pass1_runtime_checkpoint_step": (
                checkpoint.pass1_runtime_checkpoint_step
            ),
            "expected_runtime_checkpoint_step": (
                checkpoint.expected_runtime_checkpoint_step
            ),
            "evidence": checkpoint.evidence,
            "trace_runtime_checkpoint_step": (
                checkpoint.trace_runtime_checkpoint_step
            ),
            "materialized_checkpoint_artifact_verified": (
                checkpoint.materialized_checkpoint_artifact_verified
            ),
        },
        "context_reproduction": _context(result.context_reproduction),
        "pass3_disposition": result.pass3_disposition.value,
        "localization_ref": {
            "sha256": localization.sha256,
            "summary": {
                "pass1_status": localization.pass1_status,
                "generated_token_step": (
                    localization.generated_token_step
                ),
                "runtime_checkpoint_step": (
                    localization.runtime_checkpoint_step
                ),
                "matched_generated_prefix_length": (
                    localization.matched_generated_prefix_length
                ),
                "prefix_availability": (
                    localization.prefix_availability
                ),
                "pass2_disposition": localization.pass2_disposition,
                "verdict_strength_limit": (
                    localization.verdict_strength_limit
                ),
            },
            "embedded": (
                json.loads(localization.canonical_json)
                if embed_localization
                else None
            ),
        },
        "verdict_strength_limit": result.verdict_strength_limit.value,
        "reason_codes": [reason.value for reason in result.reason_codes],
        "inherited_pass1_reason_codes": list(
            result.inherited_pass1_reason_codes
        ),
        "inherited_pass0_reason_codes": list(
            result.inherited_pass0_reason_codes
        ),
        "blocking_reasons": list(result.blocking_reasons),
        "warnings": list(result.warnings),
    }


def to_json(
    result: Pass2Result,
    indent: int = 2,
    *,
    embed_localization: bool = False,
) -> str:
    return json.dumps(
        serialize(result, embed_localization=embed_localization),
        indent=indent,
    )
