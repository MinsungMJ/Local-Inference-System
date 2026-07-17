"""Synthetic, model-free Pass 3 fixture builders."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from lis_verify import (
    CanonicalRunReport,
    ComparisonMode,
    PreflightInputs,
    RunSide,
    build_gate,
    build_source_binding,
    run_calibration_preflight,
    run_prefix_policy_reproduction,
    run_token_localization,
)
from lis_verify.pass3_inputs import CanonicalLayerTrace, CanonicalPass2Artifact


ROOT = Path(__file__).resolve().parents[2]
PASS2_FIXTURES = ROOT / "tools" / "test_fixtures" / "prefix_policy_reproduction"


def _base_report(name: str, artifact_set_id: str) -> CanonicalRunReport:
    raw = json.loads((PASS2_FIXTURES / name).read_text(encoding="utf-8"))
    raw["artifact_set_id"] = artifact_set_id
    runtime = raw["manifest"]["runtime"]
    runtime.setdefault(
        "precision_path", "f32_accum;weights=bf16;kv=bf16"
    )
    runtime.update(
        layer_checkpoints_enabled=True,
        layer_checkpoint_step=18,
        diagnostics_enabled=False,
        perf_enabled=False,
        perf_per_token_enabled=False,
    )
    return CanonicalRunReport.from_object(raw)


def _pass1(reference: CanonicalRunReport, candidate: CanonicalRunReport):
    reference_side = RunSide.from_run_report(
        reference.materialize(), "reference", prompt_token_array=[1, 2, 3]
    )
    candidate_side = RunSide.from_run_report(
        candidate.materialize(), "candidate", prompt_token_array=[1, 2, 3]
    )
    preflight = run_calibration_preflight(
        PreflightInputs(
            reference=reference_side,
            candidate=candidate_side,
            declared_mode=ComparisonMode.BACKEND_DIFFERENTIAL,
        )
    )
    return run_token_localization(
        build_gate(preflight),
        reference,
        candidate,
        build_source_binding(reference, candidate),
    )


def trace_object(
    report: CanonicalRunReport,
    *,
    layers=(0, 4, 8, 12),
    captured=None,
    digest_overrides=None,
    total_layer_count=13,
):
    captured = tuple(layers if captured is None else captured)
    digest_overrides = digest_overrides or {}
    requested_coordinates = []
    captured_coordinates = []
    missing_coordinates = []
    entries = []
    for ordinal, layer in enumerate(layers):
        coordinate = {
            "runtime_checkpoint_step": 18,
            "layer_index": layer,
            "tensor_role": "layer_output",
            "batch_index": 0,
            "sequence_index": 0,
            "stage_order": 0,
            "execution_ordinal": ordinal,
        }
        requested_coordinates.append(copy.deepcopy(coordinate))
        if layer not in captured:
            missing_coordinates.append(
                {
                    "coordinate": copy.deepcopy(coordinate),
                    "state": "not_captured",
                    "detail": "target_checkpoint_not_observed",
                }
            )
            continue
        captured_coordinates.append(copy.deepcopy(coordinate))
        digest_value = digest_overrides.get(
            layer, f"sha256:{layer + 1:064x}"
        )
        entries.append(
            {
                "step": 18,
                **copy.deepcopy(coordinate),
                "phase": "decode",
                "name": f"layer.{layer}.output",
                "observed_dtype": "fp32",
                "shape": [1, 1, 3],
                "element_count": 3,
                "available_summary_fields": [
                    "min", "max", "mean", "l2", "nan", "inf", "digest"
                ],
                "min": -1.0,
                "max": 1.0,
                "mean": 0.0,
                "l2": 1.0,
                "nan": 0,
                "inf": 0,
                "digest": {
                    "algorithm": "sha256",
                    "version": "lis.checkpoint.fp32le/v1",
                    "tensor_role": "layer_output",
                    "shape": [1, 1, 3],
                    "observed_dtype": "fp32",
                    "byte_order": "little",
                    "canonicalization": (
                        "ieee754-binary32-le;canonical-qnan;preserve-signed-zero"
                    ),
                    "value": digest_value,
                },
            }
        )
    return {
        "schema": "lis.execution_artifact/v1",
        "kind": "layer_trace",
        "artifact_set_id": report.materialize()["artifact_set_id"],
        "manifest": copy.deepcopy(report.materialize()["manifest"]),
        "checkpoint_layout": {
            "layout_name": "llama_layer_output_summary",
            "layout_version": 1,
            "runtime_checkpoint_step": 18,
            "tensor_role": "layer_output",
            "stage_order": 0,
            "ordering_semantics": "runtime_step_layer_stage_ordinal",
            "total_layer_count": total_layer_count,
            "requested_coordinates": requested_coordinates,
            "captured_coordinates": captured_coordinates,
            "missing_coordinates": missing_coordinates,
            "available_summary_fields": [
                "min", "max", "mean", "l2", "nan", "inf", "digest"
            ],
            "digest_contract": {
                "algorithm": "sha256",
                "version": "lis.checkpoint.fp32le/v1",
                "observed_dtype": "fp32",
                "byte_order": "little",
                "canonicalization": (
                    "ieee754-binary32-le;canonical-qnan;preserve-signed-zero"
                ),
            },
            "duplicate_coordinate_policy": "reject_artifact_before_write",
        },
        "layer_trace": entries,
    }


def ready_case(
    *,
    reference_layers=(0, 4, 8, 12),
    candidate_layers=(0, 4, 8, 12),
    reference_captured=None,
    candidate_captured=None,
    candidate_digest_overrides=None,
    independent=False,
):
    reference_report = _base_report(
        "reference_original_bound.json",
        "aset1:11111111111111111111111111111111",
    )
    candidate_report = _base_report(
        "candidate_original_bound.json",
        "aset1:22222222222222222222222222222222",
    )
    reference_source = reference_report
    candidate_source = candidate_report
    reproduction_args = {}
    if independent:
        reference_source = _base_report(
            "reference_reproduction_verified.json",
            "aset1:33333333333333333333333333333333",
        )
        candidate_source = _base_report(
            "candidate_reproduction_verified.json",
            "aset1:44444444444444444444444444444444",
        )
        reproduction_args = {
            "reference_reproduction": reference_source,
            "candidate_reproduction": candidate_source,
        }
    pass2 = run_prefix_policy_reproduction(
        _pass1(reference_report, candidate_report),
        reference_report,
        candidate_report,
        **reproduction_args,
    )
    pass2_artifact = CanonicalPass2Artifact.from_result(pass2)
    reference_raw = trace_object(
        reference_source,
        layers=reference_layers,
        captured=reference_captured,
    )
    candidate_raw = trace_object(
        candidate_source,
        layers=candidate_layers,
        captured=candidate_captured,
        digest_overrides=candidate_digest_overrides,
    )
    return {
        "pass2": pass2,
        "pass2_artifact": pass2_artifact,
        "reference_report": reference_source,
        "candidate_report": candidate_source,
        "reference_raw": reference_raw,
        "candidate_raw": candidate_raw,
        "reference_trace": CanonicalLayerTrace.from_object(reference_raw),
        "candidate_trace": CanonicalLayerTrace.from_object(candidate_raw),
    }


def run_case(case):
    from lis_verify import run_coverage_scoped_layer_localization

    return run_coverage_scoped_layer_localization(
        case["pass2"],
        case["pass2_artifact"],
        case["reference_trace"],
        case["candidate_trace"],
        reference_source_report=case["reference_report"],
        candidate_source_report=case["candidate_report"],
    )
