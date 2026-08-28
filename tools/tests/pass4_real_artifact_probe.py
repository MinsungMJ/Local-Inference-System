#!/usr/bin/env python3
"""P4-12 two-generation verification over actual C CLI artifacts."""

from __future__ import annotations

import copy
import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from unittest import mock

from lis_verify import pass4 as pass4_module
from lis_verify import (
    CanonicalLayerTrace,
    CanonicalPass2Artifact,
    CanonicalRunReport,
    Pass4LocalCoverageOutcome,
    Pass4ReasonCode,
    Pass4Status,
    run_coverage_scoped_intra_layer_localization,
    serialize_pass4,
)
from lis_verify.pass3_model import Pass3Status
from lis_verify.pass3_inputs import LayerTraceIdentity
from lis_verify.pass4_parent import CanonicalPass3Artifact

import pass3_real_artifact_probe as pass3_probe


TARGET_LAYER = 4
TARGET_STEP = 1
TARGET_STAGE = "mlp_gate_projection"
STAGE_IDS = (
    "layer_input",
    "attention_norm_output",
    "query_projection_output",
    "key_projection_output",
    "value_projection_output",
    "rope_query_output",
    "rope_key_output",
    "attention_scores",
    "attention_probabilities",
    "attention_context",
    "attention_output_projection",
    "post_attention_residual",
    "mlp_norm_output",
    "mlp_gate_projection",
    "mlp_up_projection",
    "mlp_gated_activation",
    "mlp_down_projection",
)
PROHIBITED_KEYS = {
    "tensor_payload",
    "tensor_values",
    "values",
    "samples",
    "prompt",
    "prompt_text",
    "generated_text",
    "absolute_path",
    "model_path",
}


@dataclass(frozen=True)
class Generation:
    original_reference: CanonicalRunReport
    original_candidate: CanonicalRunReport
    reference_report: CanonicalRunReport
    candidate_report: CanonicalRunReport
    reference_decode: dict
    candidate_decode: dict
    reference_trace: CanonicalLayerTrace
    candidate_trace: CanonicalLayerTrace
    pass2: object
    pass2_artifact: CanonicalPass2Artifact
    pass3: object
    pass3_artifact: CanonicalPass3Artifact


def _path(directory: Path, name: str, kind: str) -> Path:
    return directory / f"{name}_{kind}.json"


def _load_generation(directory: Path, prefix: str) -> Generation:
    original_reference = CanonicalRunReport.load(
        _path(directory, f"{prefix}_original_reference", "report")
    )
    original_candidate = CanonicalRunReport.load(
        _path(directory, f"{prefix}_original_candidate", "report")
    )
    reference_report = CanonicalRunReport.load(
        _path(directory, f"{prefix}_reproduction_reference", "report")
    )
    candidate_report = CanonicalRunReport.load(
        _path(directory, f"{prefix}_reproduction_candidate", "report")
    )
    reference_decode = json.loads(
        _path(directory, f"{prefix}_reproduction_reference", "decode")
        .read_text(encoding="utf-8")
    )
    candidate_decode = json.loads(
        _path(directory, f"{prefix}_reproduction_candidate", "decode")
        .read_text(encoding="utf-8")
    )
    assert reference_decode["decode_trace"] == candidate_decode["decode_trace"]
    reference_trace = CanonicalLayerTrace.load(
        _path(directory, f"{prefix}_reproduction_reference", "layer")
    )
    candidate_trace = CanonicalLayerTrace.load(
        _path(directory, f"{prefix}_reproduction_candidate", "layer")
    )
    pass2 = pass3_probe._ready_actual_pass2(
        original_reference,
        original_candidate,
        reference_report,
        candidate_report,
    )
    pass2_artifact = CanonicalPass2Artifact.from_result(pass2)
    pass3 = pass3_probe._run_pass3(
        pass2,
        reference_report,
        reference_trace,
        candidate_report,
        candidate_trace,
    )
    assert pass3.status == Pass3Status.OBSERVABLE_MISMATCH_FOUND
    assert pass3.first_observed_mismatch_coordinate is not None
    assert pass3.first_observed_mismatch_coordinate.layer_index == TARGET_LAYER
    assert pass3.suspect_interval is not None
    assert pass3.suspect_interval.notation == "(3, 4]"
    return Generation(
        original_reference,
        original_candidate,
        reference_report,
        candidate_report,
        reference_decode,
        candidate_decode,
        reference_trace,
        candidate_trace,
        pass2,
        pass2_artifact,
        pass3,
        CanonicalPass3Artifact.from_result(pass3),
    )


def _keys(value) -> set[str]:
    if isinstance(value, dict):
        result = set(value)
        for child in value.values():
            result.update(_keys(child))
        return result
    if isinstance(value, list):
        result = set()
        for child in value:
            result.update(_keys(child))
        return result
    return set()


def _intra_entries(trace: CanonicalLayerTrace) -> list[dict]:
    raw = trace.materialize()
    entries = raw["intra_layer_trace"]
    assert [entry["stage_id"] for entry in entries] == list(STAGE_IDS)
    assert len(entries) == 17
    return entries


def _run_pass4(
    discovery: Generation,
    authoritative: Generation,
    *,
    discovery_pass3=None,
    discovery_pass3_artifact: CanonicalPass3Artifact | None = None,
    authoritative_pass3_artifact: CanonicalPass3Artifact | None = None,
):
    return run_coverage_scoped_intra_layer_localization(
        discovery_pass3 or discovery.pass3,
        discovery_pass3_artifact or discovery.pass3_artifact,
        authoritative.pass3,
        authoritative_pass3_artifact or authoritative.pass3_artifact,
        authoritative.pass2_artifact,
        discovery_reference_report=discovery.reference_report,
        discovery_candidate_report=discovery.candidate_report,
        discovery_reference_trace=discovery.reference_trace,
        discovery_candidate_trace=discovery.candidate_trace,
        authoritative_reference_report=authoritative.reference_report,
        authoritative_candidate_report=authoritative.candidate_report,
        authoritative_reference_trace=authoritative.reference_trace,
        authoritative_candidate_trace=authoritative.candidate_trace,
        discovery_pass2_artifact=discovery.pass2_artifact,
    )


def _with_trace_mutations(
    generation: Generation,
    *,
    mutate_reference=None,
    mutate_candidate=None,
) -> Generation:
    reference_raw = copy.deepcopy(generation.reference_trace.materialize())
    candidate_raw = copy.deepcopy(generation.candidate_trace.materialize())
    if mutate_reference is not None:
        mutate_reference(reference_raw)
    if mutate_candidate is not None:
        mutate_candidate(candidate_raw)
    reference_trace = CanonicalLayerTrace.from_object(reference_raw)
    candidate_trace = CanonicalLayerTrace.from_object(candidate_raw)
    pass3 = pass3_probe._run_pass3(
        generation.pass2,
        generation.reference_report,
        reference_trace,
        generation.candidate_report,
        candidate_trace,
    )
    return replace(
        generation,
        reference_trace=reference_trace,
        candidate_trace=candidate_trace,
        pass3=pass3,
        pass3_artifact=CanonicalPass3Artifact.from_result(pass3),
    )


def _mutate_all_intra_coordinates(raw: dict, field: str, value) -> None:
    layout = raw["intra_layer_checkpoint_layout"]
    layout[field] = value
    for collection in (
        layout["requested_coordinates"],
        layout["captured_coordinates"],
        raw["intra_layer_trace"],
    ):
        for item in collection:
            item[field] = value


def _assert_negative(
    discovery: Generation,
    authoritative: Generation,
    expected_status: Pass4Status,
    expected_reason: Pass4ReasonCode,
    *,
    digest_comparison_forbidden: bool = False,
) -> None:
    if digest_comparison_forbidden:
        context = mock.patch.object(
            pass4_module,
            "_comparison",
            side_effect=AssertionError("digest comparison reached"),
        )
    else:
        context = mock.patch.object(pass4_module, "_comparison", wraps=pass4_module._comparison)
    with context:
        result = _run_pass4(discovery, authoritative)
    assert result.status == expected_status, (
        result.status,
        expected_status,
        result.reason_codes,
    )
    assert result.reason_codes == (expected_reason,), result.reason_codes
    if digest_comparison_forbidden:
        assert result.comparisons == ()


def _assert_generation_rebind(discovery: Generation,
                              authoritative: Generation) -> None:
    assert discovery.pass3_artifact.artifact_sha256 != (
        authoritative.pass3_artifact.artifact_sha256
    )
    for left, right in (
        (discovery.reference_report, authoritative.reference_report),
        (discovery.candidate_report, authoritative.candidate_report),
    ):
        assert left.identity.run_report_sha256 != right.identity.run_report_sha256
    for left, right in (
        (discovery.reference_trace, authoritative.reference_trace),
        (discovery.candidate_trace, authoritative.candidate_trace),
    ):
        assert left.identity.trace_sha256 != right.identity.trace_sha256
    assert discovery.reference_decode["decode_trace"] == (
        authoritative.reference_decode["decode_trace"]
    )
    assert discovery.candidate_decode["decode_trace"] == (
        authoritative.candidate_decode["decode_trace"]
    )

    ids = {
        pass3_probe._artifact_set_id(report)
        for report in (
            discovery.original_reference,
            discovery.original_candidate,
            discovery.reference_report,
            discovery.candidate_report,
            authoritative.original_reference,
            authoritative.original_candidate,
            authoritative.reference_report,
            authoritative.candidate_report,
        )
    }
    assert len(ids) == 8


def _assert_case_a(discovery: Generation, authoritative: Generation) -> None:
    result = _run_pass4(discovery, authoritative)
    assert result.status == (
        Pass4Status.MISMATCH_BOUNDED_TO_INHERITED_CLOSING_BOUNDARY
    )
    assert result.local_coverage_outcome == (
        Pass4LocalCoverageOutcome.NO_MISMATCH_IN_COMMON_INTRA_LAYER_COVERAGE
    )
    assert result.suspect_interval is not None
    assert result.suspect_interval.notation == (
        "(mlp_down_projection, parent:layer_output]"
    )
    assert len(result.comparisons) == 17
    assert all(item.equivalent for item in result.comparisons)
    assert result.target_runtime_checkpoint_step == TARGET_STEP
    assert result.target_layer == TARGET_LAYER

    reference_entries = _intra_entries(authoritative.reference_trace)
    candidate_entries = _intra_entries(authoritative.candidate_trace)
    assert all(
        left["digest"]["value"] == right["digest"]["value"]
        for left, right in zip(reference_entries, candidate_entries)
    )
    serialized = serialize_pass4(result)
    assert not (_keys(serialized) & PROHIBITED_KEYS)
    assert serialized["nonclaims"] == {
        "automatic_frozen_success_mapping": False,
        "complete_intra_layer_coverage_proved": False,
        "exhaustive_confirmation_performed": False,
        "numeric_divergence_confirmed": False,
        "operation_level_localization_performed": False,
        "root_cause_identified": False,
        "tensor_equality_proved": False,
        "true_first_divergence_confirmed": False,
    }


def _assert_case_b(discovery: Generation, authoritative: Generation) -> None:
    result = _run_pass4(discovery, authoritative)
    assert result.status == Pass4Status.OBSERVABLE_INTRA_LAYER_MISMATCH_FOUND
    assert result.local_coverage_outcome == Pass4LocalCoverageOutcome.LOCAL_MISMATCH_FOUND
    assert result.first_observed_local_mismatch_coordinate is not None
    assert result.first_observed_local_mismatch_coordinate.stage_id == TARGET_STAGE
    assert result.suspect_interval is not None
    assert result.suspect_interval.notation == (
        "(mlp_norm_output, mlp_gate_projection]"
    )
    assert len(result.comparisons) == 17
    assert [item.equivalent for item in result.comparisons] == (
        [True] * 13 + [False] + [True] * 3
    )

    reference_entries = _intra_entries(authoritative.reference_trace)
    candidate_entries = _intra_entries(authoritative.candidate_trace)
    for index, (left, right) in enumerate(
        zip(reference_entries, candidate_entries)
    ):
        assert (left["digest"]["value"] == right["digest"]["value"]) == (
            index != 13
        )


def _assert_different_input_blocked(directory: Path,
                                    authoritative: Generation) -> None:
    report = CanonicalRunReport.load(
        _path(directory, "authoritative_different_input", "report")
    )
    blocked = pass3_probe._actual_pass2(
        authoritative.original_reference,
        authoritative.original_candidate,
        authoritative.reference_report,
        report,
    )
    assert blocked.status.value == "source_binding_inconsistent"


def _assert_actual_binding_negatives(discovery: Generation,
                                     authoritative: Generation) -> None:
    swapped = run_coverage_scoped_intra_layer_localization(
        discovery.pass3,
        discovery.pass3_artifact,
        authoritative.pass3,
        authoritative.pass3_artifact,
        authoritative.pass2_artifact,
        discovery_reference_report=discovery.reference_report,
        discovery_candidate_report=discovery.candidate_report,
        discovery_reference_trace=discovery.reference_trace,
        discovery_candidate_trace=discovery.candidate_trace,
        authoritative_reference_report=authoritative.reference_report,
        authoritative_candidate_report=authoritative.candidate_report,
        authoritative_reference_trace=authoritative.candidate_trace,
        authoritative_candidate_trace=authoritative.reference_trace,
        discovery_pass2_artifact=discovery.pass2_artifact,
    )
    assert swapped.status == Pass4Status.SOURCE_BINDING_INCONSISTENT

    legacy = run_coverage_scoped_intra_layer_localization(
        discovery.pass3,
        discovery.pass3_artifact,
        authoritative.pass3,
        authoritative.pass3_artifact,
        authoritative.pass2_artifact,
        discovery_reference_report=discovery.reference_report,
        discovery_candidate_report=discovery.candidate_report,
        discovery_reference_trace=discovery.reference_trace,
        discovery_candidate_trace=discovery.candidate_trace,
        authoritative_reference_report=authoritative.reference_report,
        authoritative_candidate_report=authoritative.candidate_report,
        authoritative_reference_trace=discovery.reference_trace,
        authoritative_candidate_trace=authoritative.candidate_trace,
        discovery_pass2_artifact=discovery.pass2_artifact,
    )
    assert legacy.status == Pass4Status.SOURCE_BINDING_INCONSISTENT

    identity = authoritative.reference_trace.identity
    wrong_sha = LayerTraceIdentity(
        discovery.reference_trace.identity.trace_sha256,
        identity.schema,
        identity.kind,
        identity.artifact_set_id,
        identity.semantic_manifest_sha256,
        identity.runtime_checkpoint_step,
    )
    wrong_sha_trace = pass3_probe._sentinel(
        authoritative.reference_trace, wrong_sha
    )
    wrong_sha_result = run_coverage_scoped_intra_layer_localization(
        discovery.pass3,
        discovery.pass3_artifact,
        authoritative.pass3,
        authoritative.pass3_artifact,
        authoritative.pass2_artifact,
        discovery_reference_report=discovery.reference_report,
        discovery_candidate_report=discovery.candidate_report,
        discovery_reference_trace=discovery.reference_trace,
        discovery_candidate_trace=discovery.candidate_trace,
        authoritative_reference_report=authoritative.reference_report,
        authoritative_candidate_report=authoritative.candidate_report,
        authoritative_reference_trace=wrong_sha_trace,
        authoritative_candidate_trace=authoritative.candidate_trace,
        discovery_pass2_artifact=discovery.pass2_artifact,
    )
    assert wrong_sha_result.reason_codes == (
        Pass4ReasonCode.TRACE_SHA_MISMATCH,
    )

    wrong_manifest = LayerTraceIdentity(
        identity.trace_sha256,
        identity.schema,
        identity.kind,
        identity.artifact_set_id,
        "sha256:" + "0" * 64,
        identity.runtime_checkpoint_step,
    )
    wrong_manifest_result = run_coverage_scoped_intra_layer_localization(
        discovery.pass3,
        discovery.pass3_artifact,
        authoritative.pass3,
        authoritative.pass3_artifact,
        authoritative.pass2_artifact,
        discovery_reference_report=discovery.reference_report,
        discovery_candidate_report=discovery.candidate_report,
        discovery_reference_trace=discovery.reference_trace,
        discovery_candidate_trace=discovery.candidate_trace,
        authoritative_reference_report=authoritative.reference_report,
        authoritative_candidate_report=authoritative.candidate_report,
        authoritative_reference_trace=pass3_probe._sentinel(
            authoritative.reference_trace, wrong_manifest
        ),
        authoritative_candidate_trace=authoritative.candidate_trace,
        discovery_pass2_artifact=discovery.pass2_artifact,
    )
    assert wrong_manifest_result.reason_codes == (
        Pass4ReasonCode.SEMANTIC_MANIFEST_BINDING_MISMATCH,
    )

    wrong_set = LayerTraceIdentity(
        identity.trace_sha256,
        identity.schema,
        identity.kind,
        "aset1:" + "0" * 32,
        identity.semantic_manifest_sha256,
        identity.runtime_checkpoint_step,
    )
    wrong_set_result = run_coverage_scoped_intra_layer_localization(
        discovery.pass3,
        discovery.pass3_artifact,
        authoritative.pass3,
        authoritative.pass3_artifact,
        authoritative.pass2_artifact,
        discovery_reference_report=discovery.reference_report,
        discovery_candidate_report=discovery.candidate_report,
        discovery_reference_trace=discovery.reference_trace,
        discovery_candidate_trace=discovery.candidate_trace,
        authoritative_reference_report=authoritative.reference_report,
        authoritative_candidate_report=authoritative.candidate_report,
        authoritative_reference_trace=pass3_probe._sentinel(
            authoritative.reference_trace, wrong_set
        ),
        authoritative_candidate_trace=authoritative.candidate_trace,
        discovery_pass2_artifact=discovery.pass2_artifact,
    )
    assert wrong_set_result.reason_codes == (
        Pass4ReasonCode.ARTIFACT_SET_BINDING_MISMATCH,
    )


def _assert_actual_alignment_negatives(
    discovery: Generation, authoritative: Generation
) -> None:
    def step(raw):
        _mutate_all_intra_coordinates(raw, "runtime_checkpoint_step", 2)

    def layer(raw):
        _mutate_all_intra_coordinates(raw, "layer_index", 5)
        raw["intra_layer_checkpoint_layout"]["target_layer"] = 5

    def phase(raw):
        raw["intra_layer_trace"][7]["phase"] = "prefill"

    def position(raw):
        raw["intra_layer_trace"][7]["token_position"] = 3

    def precision(raw):
        raw["intra_layer_trace"][7]["precision_path"] = (
            "f32_accum;weights=f32;kv=f32;drift"
        )

    def shape(raw):
        entry = raw["intra_layer_trace"][7]
        entry["shape"] = [2]
        entry["element_count"] = 2
        entry["digest"]["shape"] = [2]

    cases = (
        ("step", step, Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT,
         Pass4ReasonCode.STEP_ALIGNMENT_MISMATCH, True),
        ("layer", layer, Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT,
         Pass4ReasonCode.LAYER_ALIGNMENT_MISMATCH, True),
        ("phase", phase, Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT,
         Pass4ReasonCode.PHASE_OR_POSITION_ALIGNMENT_MISMATCH, False),
        ("position", position, Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
         Pass4ReasonCode.COVERAGE_PARTITION_MALFORMED, False),
        ("precision", precision, Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT,
         Pass4ReasonCode.DTYPE_OR_PRECISION_ALIGNMENT_MISMATCH, False),
        ("shape", shape, Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT,
         Pass4ReasonCode.SHAPE_OR_COUNT_ALIGNMENT_MISMATCH, False),
    )
    for label, mutation, status, reason, mutate_both in cases:
        changed = _with_trace_mutations(
            authoritative,
            mutate_reference=mutation if mutate_both else None,
            mutate_candidate=mutation,
        )
        try:
            _assert_negative(
                discovery,
                changed,
                status,
                reason,
                digest_comparison_forbidden=True,
            )
        except AssertionError as exc:
            raise AssertionError(f"alignment negative {label}: {exc}") from exc


def _assert_actual_structure_negatives(
    discovery: Generation, authoritative: Generation
) -> None:
    def layout_version(raw):
        raw["intra_layer_checkpoint_layout"]["layout_version"] = 2

    def taxonomy(raw):
        raw["intra_layer_checkpoint_layout"]["stage_taxonomy"] = (
            "lis.llama.intra_layer_stages/v2"
        )

    for mutation in (layout_version, taxonomy):
        changed = _with_trace_mutations(
            authoritative, mutate_candidate=mutation
        )
        _assert_negative(
            discovery,
            changed,
            Pass4Status.UNSUPPORTED_INTRA_LAYER_LAYOUT,
            Pass4ReasonCode.INTRA_LAYOUT_OR_TAXONOMY_UNSUPPORTED,
            digest_comparison_forbidden=True,
        )

    def duplicate_requested(raw):
        layout = raw["intra_layer_checkpoint_layout"]
        layout["requested_coordinates"][1] = copy.deepcopy(
            layout["requested_coordinates"][0]
        )

    def duplicate_entry(raw):
        raw["intra_layer_trace"][1] = copy.deepcopy(
            raw["intra_layer_trace"][0]
        )

    for mutation in (duplicate_requested, duplicate_entry):
        changed = _with_trace_mutations(
            authoritative, mutate_candidate=mutation
        )
        result = _run_pass4(discovery, changed)
        assert result.status == Pass4Status.CHECKPOINT_SUMMARY_MALFORMED
        assert result.reason_codes == (
            Pass4ReasonCode.DUPLICATE_OR_OUT_OF_ORDER_COORDINATE,
        )


def _assert_actual_parent_revalidation_negatives(
    discovery: Generation, authoritative: Generation
) -> None:
    def selected_layer(raw):
        reference_entries = pass3_probe._semantic_entries(
            authoritative.reference_trace
        )
        entries = {
            entry["layer_index"]: entry for entry in raw["layer_trace"]
            if entry.get("tensor_role") == "layer_output"
        }
        entries[TARGET_LAYER]["digest"] = copy.deepcopy(
            reference_entries[TARGET_LAYER]["digest"]
        )
        entries[6]["digest"]["value"] = "sha256:" + "f" * 64

    layer_drift = _with_trace_mutations(
        authoritative, mutate_candidate=selected_layer
    )
    assert layer_drift.pass3.first_observed_mismatch_coordinate.layer_index == 6
    _assert_negative(
        discovery,
        layer_drift,
        Pass4Status.PARENT_REVALIDATION_INCONSISTENT,
        Pass4ReasonCode.DISCOVERY_REBOUND_LAYER_CHANGED,
        digest_comparison_forbidden=True,
    )

    def coverage(raw):
        layout = raw["checkpoint_layout"]
        removed = layout["requested_coordinates"].pop()
        layout["captured_coordinates"] = [
            item for item in layout["captured_coordinates"]
            if item != removed
        ]
        raw["layer_trace"] = [
            item for item in raw["layer_trace"]
            if not (
                item.get("runtime_checkpoint_step")
                == removed["runtime_checkpoint_step"]
                and item.get("layer_index") == removed["layer_index"]
                and item.get("tensor_role") == removed["tensor_role"]
            )
        ]

    coverage_drift = _with_trace_mutations(
        authoritative,
        mutate_reference=coverage,
        mutate_candidate=coverage,
    )
    _assert_negative(
        discovery,
        coverage_drift,
        Pass4Status.PARENT_REVALIDATION_INCONSISTENT,
        Pass4ReasonCode.DISCOVERY_REBOUND_SEMANTICS_CHANGED,
        digest_comparison_forbidden=True,
    )


def _assert_actual_parent_artifact_negatives(
    discovery: Generation, authoritative: Generation
) -> None:
    truncated = replace(
        discovery.pass3,
        comparisons=discovery.pass3.comparisons * 90,
    )
    truncated_artifact = CanonicalPass3Artifact.from_result(truncated)
    with mock.patch.object(
        pass4_module,
        "_comparison",
        side_effect=AssertionError("truncated parent reached digest compare"),
    ):
        result = _run_pass4(
            discovery,
            authoritative,
            discovery_pass3=truncated,
            discovery_pass3_artifact=truncated_artifact,
        )
    assert result.status == Pass4Status.COMPARISON_BLOCKED_BY_PASS3
    assert result.reason_codes == (
        Pass4ReasonCode.PARENT_COMPARISONS_TRUNCATED,
    ), result.reason_codes


def _assert_private_directory(directory: Path) -> None:
    assert directory.is_dir() and not directory.is_symlink()
    assert directory.stat().st_mode & 0o777 == 0o700
    for path in directory.iterdir():
        if path.name.endswith(("_report.json", "_layer.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            retention = raw["manifest"]["retention_policy"]
            assert retention == {
                "absolute_paths": "omitted",
                "raw_prompt_text": "omitted",
                "generated_text": "omitted",
            }
            if path.name.endswith("_layer.json") and (
                "intra_layer_checkpoint_layout" in raw
            ):
                blocks = {
                    "intra_layer_checkpoint_layout": raw[
                        "intra_layer_checkpoint_layout"
                    ],
                    "intra_layer_trace": raw["intra_layer_trace"],
                }
                assert not (_keys(blocks) & PROHIBITED_KEYS)


def _run_case(directory: Path, *, local_mismatch: bool) -> None:
    _assert_private_directory(directory)
    discovery = _load_generation(directory, "discovery")
    authoritative = _load_generation(directory, "authoritative")
    _assert_generation_rebind(discovery, authoritative)
    pass3_probe._assert_same_semantics(
        discovery.reference_report, discovery.candidate_report
    )
    pass3_probe._assert_same_semantics(
        authoritative.reference_report, authoritative.candidate_report
    )
    if local_mismatch:
        _assert_case_b(discovery, authoritative)
    else:
        _assert_case_a(discovery, authoritative)
    _assert_different_input_blocked(directory, authoritative)
    _assert_actual_binding_negatives(discovery, authoritative)
    if not local_mismatch:
        _assert_actual_alignment_negatives(discovery, authoritative)
        _assert_actual_structure_negatives(discovery, authoritative)
        _assert_actual_parent_revalidation_negatives(
            discovery, authoritative
        )
        _assert_actual_parent_artifact_negatives(
            discovery, authoritative
        )


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("usage: probe CASE_A_PRIVATE_DIR CASE_B_PRIVATE_DIR")
    case_a = Path(argv[1])
    case_b = Path(argv[2])
    _run_case(case_a, local_mismatch=False)
    _run_case(case_b, local_mismatch=True)
    print(
        "pass4-real-integration: "
        "test_a=mismatch_bounded_to_inherited_closing_boundary "
        "test_b=observable_intra_layer_mismatch_found "
        "generations=4+4-per-case actual_cli_processes=18 "
        "negative_gates=binding,parent-revalidation,layout,alignment,summary"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
