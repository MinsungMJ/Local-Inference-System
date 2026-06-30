"""Serialization for the ``calibration_preflight`` artifact.

Stable key order (matching the plan §4.4); enums emit their string values;
the artifact carries no raw prompt text, token arrays, paths, or tensor values.
"""

from __future__ import annotations

import json

from .model import CalibrationPreflightArtifact


def serialize(a: CalibrationPreflightArtifact) -> dict:
    return {
        "schema": a.schema,
        "kind": a.kind,
        "contract_version": a.contract_version,
        "comparison_mode": a.comparison_mode.value,
        "comparison_eligibility": a.comparison_eligibility.value,
        "pass0_verdict": a.pass0_verdict.value,
        "calibration_status": {
            "decode_policy_calibrated": a.calibration_status.decode_policy_calibrated,
            "config_semantics_calibrated": a.calibration_status.config_semantics_calibrated,
            "numeric_policy_calibrated": a.calibration_status.numeric_policy_calibrated,
            "tokenizer_boundary_calibrated": a.calibration_status.tokenizer_boundary_calibrated,
        },
        "decode_policy_identity": {
            "selection_mode": a.decode_policy_identity.selection_mode.value,
            "repetition_penalty": a.decode_policy_identity.repetition_penalty,
            "repetition_penalty_enabled": a.decode_policy_identity.repetition_penalty_enabled,
            "structural_token_suppression": a.decode_policy_identity.structural_token_suppression,
            "raw_greedy_equivalent": a.decode_policy_identity.raw_greedy_equivalent,
        },
        "tokenizer_boundary": {
            "prompt_boundary": a.tokenizer_boundary.prompt_boundary.value,
            "prompt_token_array_available": a.tokenizer_boundary.prompt_token_array_available,
            "prompt_token_array_equal": a.tokenizer_boundary.prompt_token_array_equal,
            "prompt_token_digest_equal": a.tokenizer_boundary.prompt_token_digest_equal,
            "prompt_identity_evidence": a.tokenizer_boundary.prompt_identity_evidence.value,
            "confidence": a.tokenizer_boundary.confidence.value,
        },
        "config_semantics": {
            "rms_norm_eps_runtime_bound": a.config_semantics.rms_norm_eps_runtime_bound,
            "status": a.config_semantics.status.value,
        },
        "numeric_policy": {
            "kv_bf16_write": a.numeric_policy.kv_bf16_write.value,
            "kv_f16_write": a.numeric_policy.kv_f16_write.value,
            "round_to_nearest_even": a.numeric_policy.round_to_nearest_even,
        },
        "oracle_eligibility": {
            "lis_internal_backend_differential": a.oracle_eligibility.lis_internal_backend_differential,
            "hf_default_greedy": a.oracle_eligibility.hf_default_greedy,
            "hf_forced_token_runtime": {
                "potentially_eligible": a.oracle_eligibility.hf_forced_token_runtime.potentially_eligible,
                "artifact_supported": a.oracle_eligibility.hf_forced_token_runtime.artifact_supported,
                "status": a.oracle_eligibility.hf_forced_token_runtime.status,
            },
            "oracle_scope": a.oracle_eligibility.oracle_scope.value,
        },
        "reason_codes": [code.value for code in a.reason_codes],
        "warnings": list(a.warnings),
        "blocking_reasons": [code.value for code in a.blocking_reasons],
        "verdict_strength_limit": a.verdict_strength_limit.value,
    }


def to_json(a: CalibrationPreflightArtifact, indent: int = 2) -> str:
    return json.dumps(serialize(a), indent=indent)
