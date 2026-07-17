#!/usr/bin/env python3
"""Strict revalidation over independently emitted actual C artifacts.

The probe derives a real Pass 1 mismatch boundary from two same-input CLI
reports, the candidate of which uses a test-only selected-token override.
Pass 2 then verifies independent same-input reproduction reports. Pass 3 is
exercised once with two clean reproductions and once with a test-only layer-4
observation perturbation. The old different-input pair is required to fail
Pass 2 reproduction identity rather than masquerade as same-boundary evidence.
"""

from __future__ import annotations

import sys

from lis_verify import (
    CanonicalLayerTrace,
    CanonicalPass2Artifact,
    CanonicalRunReport,
    ComparisonMode,
    PreflightInputs,
    RunSide,
    build_gate,
    build_source_binding,
    run_calibration_preflight,
    run_coverage_scoped_layer_localization,
    run_prefix_policy_reproduction,
    run_token_localization,
)
from lis_verify.pass1_model import Pass1Status
from lis_verify.pass2_model import (
    Pass2Status,
    Pass3Disposition,
    ReproductionEvidenceTier,
)
from lis_verify.pass3_artifact import serialize as serialize_pass3
from lis_verify.pass3_inputs import LayerTraceIdentity
from lis_verify.pass3_model import (
    DIGEST_DECISION_FIELD,
    DIGEST_DECISION_SEMANTICS,
    Pass3ReasonCode,
    Pass3Status,
    SummaryEvidenceLevel,
)


PROMPT_TOKEN_IDS = (0,)
TARGET_GENERATED_STEP = 0
TARGET_RUNTIME_CHECKPOINT_STEP = 1
CONTROLLED_LAYER = 4
CONTROLLED_ELEMENT = 0
CONTROLLED_DELTA = 0.25
PROHIBITED_CONFIRMATION_KEYS = {
    "confirmed_first_divergent_layer",
    "confirmed_divergence_at_checkpoint",
    "confirmed_first_divergence",
}
PROHIBITED_TENSOR_KEYS = {"tensor_payload", "tensor_values", "values"}


class SummarySentinelTrace(CanonicalLayerTrace):
    def materialize(self):
        raise AssertionError("actual trace summary accessed before binding")


def _raw(report: CanonicalRunReport) -> dict:
    return report.materialize()


def _artifact_set_id(report: CanonicalRunReport) -> str:
    value = _raw(report)["artifact_set_id"]
    assert isinstance(value, str)
    return value


def _semantic_snapshot(report: CanonicalRunReport) -> dict:
    raw = _raw(report)
    manifest = raw["manifest"]
    body = raw["report"]
    return {
        "binary": manifest["binary"],
        "model": manifest["model"],
        "config": manifest["config"],
        "input": manifest["input"],
        "runtime": manifest["runtime"],
        "backend": manifest["backend"],
        "output_mode": body["output_mode"],
        "prompt_sequences": body["prompt_sequences"],
        "selected_token_ids": body["selected_token_ids"],
    }


def _assert_same_semantics(
    reference: CanonicalRunReport,
    candidate: CanonicalRunReport,
) -> dict:
    left = _semantic_snapshot(reference)
    right = _semantic_snapshot(candidate)
    for field in (
        "binary",
        "model",
        "config",
        "input",
        "runtime",
        "backend",
        "output_mode",
        "prompt_sequences",
        "selected_token_ids",
    ):
        assert left[field] == right[field], field
    assert reference.identity.model_fingerprint == candidate.identity.model_fingerprint
    assert reference.identity.config_fingerprint == candidate.identity.config_fingerprint
    assert reference.identity.input_fingerprint == candidate.identity.input_fingerprint
    return left


def _pass1_for_actual_originals(
    reference: CanonicalRunReport,
    candidate: CanonicalRunReport,
):
    reference_side = RunSide.from_run_report(
        _raw(reference), "reference", prompt_token_array=PROMPT_TOKEN_IDS
    )
    candidate_side = RunSide.from_run_report(
        _raw(candidate), "candidate", prompt_token_array=PROMPT_TOKEN_IDS
    )
    preflight = run_calibration_preflight(
        PreflightInputs(
            reference=reference_side,
            candidate=candidate_side,
            declared_mode=ComparisonMode.BACKEND_DIFFERENTIAL,
        )
    )
    pass1 = run_token_localization(
        build_gate(preflight),
        reference,
        candidate,
        build_source_binding(reference, candidate),
    )
    assert pass1.status == Pass1Status.FIRST_MISMATCH_FOUND
    assert pass1.localization is not None
    assert pass1.localization.generated_token_step == TARGET_GENERATED_STEP
    assert (
        pass1.localization.runtime_checkpoint_step
        == TARGET_RUNTIME_CHECKPOINT_STEP
    )
    assert pass1.localization.matched_generated_prefix_length == 0
    assert pass1.prefix_for_reproduction is not None
    assert pass1.prefix_for_reproduction.token_count == 0
    return pass1


def _actual_pass2(
    original_reference: CanonicalRunReport,
    original_candidate: CanonicalRunReport,
    reproduction_reference: CanonicalRunReport,
    reproduction_candidate: CanonicalRunReport,
):
    pass1 = _pass1_for_actual_originals(
        original_reference, original_candidate
    )
    return run_prefix_policy_reproduction(
        pass1,
        original_reference,
        original_candidate,
        reference_reproduction=reproduction_reference,
        candidate_reproduction=reproduction_candidate,
    )


def _ready_actual_pass2(*args):
    pass2 = _actual_pass2(*args)
    assert pass2.status == Pass2Status.REPRODUCTION_VERIFIED
    assert pass2.pass3_disposition == Pass3Disposition.READY
    assert (
        pass2.reproduction_evidence_tier
        == ReproductionEvidenceTier.INDEPENDENT_RERUN_VERIFIED
    )
    assert pass2.source_binding_verified
    assert pass2.source_binding.verified
    assert pass2.prompt_reproduction.status == "verified"
    assert pass2.prefix_reproduction.status == "verified"
    assert pass2.prefix_reproduction.expected_token_count == 0
    assert pass2.policy_reproduction.status == "verified"
    assert pass2.policy_reproduction.build_continuity_verified
    assert pass2.context_reproduction.status == "verified"
    assert pass2.checkpoint_step_reproduction.status == "verified"
    assert (
        pass2.target.expected_runtime_checkpoint_step
        == TARGET_RUNTIME_CHECKPOINT_STEP
    )
    return pass2


def _run_pass3(
    pass2,
    reference_report,
    reference_trace,
    candidate_report,
    candidate_trace,
):
    artifact = CanonicalPass2Artifact.from_result(pass2)
    return run_coverage_scoped_layer_localization(
        pass2,
        artifact,
        reference_trace,
        candidate_trace,
        reference_source_report=reference_report,
        candidate_source_report=candidate_report,
    )


def _semantic_entries(trace: CanonicalLayerTrace) -> dict[int, dict]:
    raw = trace.materialize()
    return {
        entry["layer_index"]: entry
        for entry in raw["layer_trace"]
        if entry.get("tensor_role") == "layer_output"
    }


def _all_keys(value) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for child in value.values():
            keys.update(_all_keys(child))
        return keys
    if isinstance(value, list):
        keys = set()
        for child in value:
            keys.update(_all_keys(child))
        return keys
    return set()


def _sentinel(trace: CanonicalLayerTrace, identity: LayerTraceIdentity):
    return SummarySentinelTrace(identity, trace.canonical_text)


def _test_independent_same_semantics(
    original_reference,
    original_candidate,
    reference_report,
    reference_trace,
    candidate_report,
    candidate_trace,
    unrelated_trace,
):
    semantics = _assert_same_semantics(reference_report, candidate_report)
    reference_id = _artifact_set_id(reference_report)
    candidate_id = _artifact_set_id(candidate_report)
    assert reference_id != candidate_id
    assert reference_trace.identity.artifact_set_id == reference_id
    assert candidate_trace.identity.artifact_set_id == candidate_id
    assert reference_trace.identity.trace_sha256 != candidate_trace.identity.trace_sha256
    assert reference_trace.canonical_text != candidate_trace.canonical_text
    assert reference_trace.identity.semantic_manifest_sha256 == candidate_trace.identity.semantic_manifest_sha256

    pass2 = _ready_actual_pass2(
        original_reference,
        original_candidate,
        reference_report,
        candidate_report,
    )
    result = _run_pass3(
        pass2,
        reference_report,
        reference_trace,
        candidate_report,
        candidate_trace,
    )
    assert result.status == Pass3Status.NO_MISMATCH_IN_CAPTURED_COVERAGE
    assert result.checkpoint_artifact_binding_verified
    assert result.reference_binding.artifact_set_id == reference_id
    assert result.candidate_binding.artifact_set_id == candidate_id
    assert all(item.equivalent for item in result.comparisons)
    assert result.coverage.reference_captured == result.coverage.candidate_captured
    assert result.coverage.common_captured == result.coverage.reference_captured
    captured_layers = [
        item.layer_index for item in result.coverage.common_captured
    ]
    assert captured_layers == [0, 1, 2, 3, 4, 6, 9, 11]

    # Cross-run artifact-set equality is intentionally not a binding gate.
    assert reference_id != candidate_id and result.status == Pass3Status.NO_MISMATCH_IN_CAPTURED_COVERAGE

    # An unrelated trace cannot be paired with the candidate report, and its
    # summary accessor must remain untouched.
    swapped = _run_pass3(
        pass2,
        reference_report,
        _sentinel(reference_trace, reference_trace.identity),
        candidate_report,
        _sentinel(unrelated_trace, unrelated_trace.identity),
    )
    assert swapped.reason_codes == (
        Pass3ReasonCode.ARTIFACT_SET_ID_INCONSISTENT,
    )

    # Within-run report/trace equality remains mandatory.
    identity = candidate_trace.identity
    broken_id = LayerTraceIdentity(
        identity.trace_sha256,
        identity.schema,
        identity.kind,
        "aset1:00000000000000000000000000000000",
        identity.semantic_manifest_sha256,
        identity.runtime_checkpoint_step,
    )
    broken = _run_pass3(
        pass2,
        reference_report,
        _sentinel(reference_trace, reference_trace.identity),
        candidate_report,
        _sentinel(candidate_trace, broken_id),
    )
    assert broken.reason_codes == (
        Pass3ReasonCode.ARTIFACT_SET_ID_INCONSISTENT,
    )

    # Target-step disagreement fails before trace summary access.
    wrong_step = LayerTraceIdentity(
        identity.trace_sha256,
        identity.schema,
        identity.kind,
        identity.artifact_set_id,
        identity.semantic_manifest_sha256,
        TARGET_RUNTIME_CHECKPOINT_STEP + 1,
    )
    blocked_step = _run_pass3(
        pass2,
        reference_report,
        _sentinel(reference_trace, reference_trace.identity),
        candidate_report,
        _sentinel(candidate_trace, wrong_step),
    )
    assert blocked_step.reason_codes == (
        Pass3ReasonCode.RUNTIME_CHECKPOINT_STEP_MISMATCH,
    )
    return pass2, result, semantics, reference_id, candidate_id


def _test_same_boundary_controlled_mismatch(
    original_reference,
    original_candidate,
    reference_report,
    reference_trace,
    controlled_report,
    controlled_trace,
):
    semantics = _assert_same_semantics(reference_report, controlled_report)
    reference_id = _artifact_set_id(reference_report)
    controlled_id = _artifact_set_id(controlled_report)
    assert reference_id != controlled_id
    assert reference_trace.identity.artifact_set_id == reference_id
    assert controlled_trace.identity.artifact_set_id == controlled_id

    pass2 = _ready_actual_pass2(
        original_reference,
        original_candidate,
        reference_report,
        controlled_report,
    )
    result = _run_pass3(
        pass2,
        reference_report,
        reference_trace,
        controlled_report,
        controlled_trace,
    )
    assert result.status == Pass3Status.OBSERVABLE_MISMATCH_FOUND
    assert result.checkpoint_artifact_binding_verified
    assert result.reference_binding.artifact_set_id == reference_id
    assert result.candidate_binding.artifact_set_id == controlled_id
    assert result.decision_field == DIGEST_DECISION_FIELD
    assert result.decision_semantics == DIGEST_DECISION_SEMANTICS
    assert result.evidence_level == SummaryEvidenceLevel.TIER1_BOUNDED_DIGEST
    assert result.first_observed_mismatch_coordinate is not None
    assert result.first_observed_mismatch_coordinate.layer_index == CONTROLLED_LAYER
    assert result.suspect_interval is not None
    assert result.suspect_interval.notation == "(3, 4]"
    assert [item.coordinate.layer_index for item in result.comparisons] == [
        0, 1, 2, 3, 4
    ]
    assert all(item.equivalent for item in result.comparisons[:-1])
    assert not result.comparisons[-1].equivalent

    reference_entries = _semantic_entries(reference_trace)
    controlled_entries = _semantic_entries(controlled_trace)
    assert set(reference_entries) == set(controlled_entries)
    for layer in (0, 1, 2, 3):
        assert reference_entries[layer]["digest"] == controlled_entries[layer]["digest"]
    assert reference_entries[CONTROLLED_LAYER]["digest"] != controlled_entries[CONTROLLED_LAYER]["digest"]
    for layer in (6, 9, 11):
        assert reference_entries[layer]["digest"] == controlled_entries[layer]["digest"]

    serialized = serialize_pass3(result)
    keys = _all_keys(serialized)
    assert not (_all_keys(reference_trace.materialize()) & PROHIBITED_TENSOR_KEYS)
    assert not (_all_keys(controlled_trace.materialize()) & PROHIBITED_TENSOR_KEYS)
    assert not (keys & PROHIBITED_CONFIRMATION_KEYS)
    assert not (keys & PROHIBITED_TENSOR_KEYS)
    assert serialized["semantic_limits"]["full_tensor_comparison_performed"] is False
    assert serialized["semantic_limits"]["numeric_confirmation_performed"] is False
    assert serialized["semantic_limits"]["pass4_or_pass5_readiness_certified"] is False
    assert serialized["semantic_limits"]["automatic_frozen_success_mapping"] is False
    return pass2, result, semantics, controlled_id


def _test_different_input_blocked(
    original_reference,
    original_candidate,
    reference_report,
    different_report,
):
    assert reference_report.identity.input_fingerprint != different_report.identity.input_fingerprint
    blocked = _actual_pass2(
        original_reference,
        original_candidate,
        reference_report,
        different_report,
    )
    assert blocked.status == Pass2Status.SOURCE_BINDING_INCONSISTENT
    assert blocked.pass3_disposition == Pass3Disposition.BLOCKED_BY_REPRODUCTION
    assert not blocked.source_binding_verified
    return blocked


def main(argv):
    if len(argv) != 11:
        raise SystemExit(
            "usage: probe ORIG_REF_REPORT ORIG_CAND_REPORT "
            "SAME_REF_REPORT SAME_REF_TRACE SAME_CAND_REPORT SAME_CAND_TRACE "
            "CONTROL_REPORT CONTROL_TRACE DIFF_REPORT DIFF_TRACE"
        )
    original_reference = CanonicalRunReport.load(argv[1])
    original_candidate = CanonicalRunReport.load(argv[2])
    same_reference_report = CanonicalRunReport.load(argv[3])
    same_reference_trace = CanonicalLayerTrace.load(argv[4])
    same_candidate_report = CanonicalRunReport.load(argv[5])
    same_candidate_trace = CanonicalLayerTrace.load(argv[6])
    controlled_report = CanonicalRunReport.load(argv[7])
    controlled_trace = CanonicalLayerTrace.load(argv[8])
    different_report = CanonicalRunReport.load(argv[9])
    different_trace = CanonicalLayerTrace.load(argv[10])

    # The original reports establish a real same-input Pass 1 token boundary.
    original_semantics = _semantic_snapshot(original_reference)
    candidate_original_semantics = _semantic_snapshot(original_candidate)
    for field in (
        "binary", "model", "config", "input", "runtime", "backend",
        "output_mode", "prompt_sequences",
    ):
        assert original_semantics[field] == candidate_original_semantics[field]
    assert original_semantics["selected_token_ids"][0] != candidate_original_semantics["selected_token_ids"][0]

    pass2_a, result_a, semantics_a, reference_id, candidate_id = (
        _test_independent_same_semantics(
            original_reference,
            original_candidate,
            same_reference_report,
            same_reference_trace,
            same_candidate_report,
            same_candidate_trace,
            different_trace,
        )
    )
    pass2_b, result_b, semantics_b, controlled_id = (
        _test_same_boundary_controlled_mismatch(
            original_reference,
            original_candidate,
            same_reference_report,
            same_reference_trace,
            controlled_report,
            controlled_trace,
        )
    )
    blocked_different = _test_different_input_blocked(
        original_reference,
        original_candidate,
        same_reference_report,
        different_report,
    )

    assert pass2_a.status == pass2_b.status == Pass2Status.REPRODUCTION_VERIFIED
    assert semantics_a == semantics_b
    assert result_a.status == Pass3Status.NO_MISMATCH_IN_CAPTURED_COVERAGE
    assert result_b.status == Pass3Status.OBSERVABLE_MISMATCH_FOUND
    print(
        "pass3-real-integration-revalidation: "
        f"test_a={result_a.status.value} "
        f"ids={reference_id},{candidate_id} "
        f"coverage={[item.layer_index for item in result_a.coverage.common_captured]} "
        f"test_b={result_b.status.value} "
        f"controlled_id={controlled_id} "
        "boundary_hook=selected_token_step0_to1 "
        f"perturb=layer{CONTROLLED_LAYER}:element{CONTROLLED_ELEMENT}:delta{CONTROLLED_DELTA} "
        f"first_layer={result_b.first_observed_mismatch_coordinate.layer_index} "
        f"interval={result_b.suspect_interval.notation} "
        f"model={same_reference_report.identity.model_fingerprint} "
        f"input={same_reference_report.identity.input_fingerprint} "
        f"different_input={blocked_different.status.value}:reclassified_blocked"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
