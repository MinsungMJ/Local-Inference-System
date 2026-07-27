#!/usr/bin/env python3
"""Two-generation synthetic builders for P4-3 parent-binding tests."""

from __future__ import annotations

import copy

from lis_verify import run_prefix_policy_reproduction
from lis_verify.pass1_inputs import CanonicalRunReport
from lis_verify.pass3 import run_coverage_scoped_layer_localization
from lis_verify.pass3_inputs import CanonicalLayerTrace, CanonicalPass2Artifact
from lis_verify.pass4_parent import (
    CanonicalPass3Artifact,
    bind_pass4_parent_inputs,
)

from . import pass3_test_support


TARGET_LAYER = 8
TARGET_STEP = 18
MISMATCH_DIGEST = "sha256:" + "f" * 64


def _report(
    fixture_name: str,
    artifact_set_id: str,
    *,
    authoritative_capture: bool,
    target_layer: int,
    runtime_overrides: dict | None = None,
) -> CanonicalRunReport:
    base = pass3_test_support._base_report(fixture_name, artifact_set_id)
    raw = base.materialize()
    runtime = raw["manifest"]["runtime"]
    if authoritative_capture:
        runtime.update(
            intra_layer_checkpoints_enabled=True,
            intra_layer_target_layer=target_layer,
            diagnostic_capture_profile="semantic_layer_and_intra_v1",
        )
    else:
        runtime.pop("intra_layer_checkpoints_enabled", None)
        runtime.pop("intra_layer_target_layer", None)
        runtime.pop("diagnostic_capture_profile", None)
    if runtime_overrides:
        runtime.update(runtime_overrides)
    return CanonicalRunReport.from_object(raw)


def _generation(
    *,
    identifiers: tuple[str, str, str, str],
    authoritative_capture: bool,
    mismatch_layer: int | None = TARGET_LAYER,
    layers: tuple[int, ...] = (0, 4, 8, 12),
    runtime_overrides: dict | None = None,
) -> dict:
    capture_layer = TARGET_LAYER if mismatch_layer is None else mismatch_layer
    original_reference = _report(
        "reference_original_bound.json",
        identifiers[0],
        authoritative_capture=authoritative_capture,
        target_layer=capture_layer,
        runtime_overrides=runtime_overrides,
    )
    original_candidate = _report(
        "candidate_original_bound.json",
        identifiers[1],
        authoritative_capture=authoritative_capture,
        target_layer=capture_layer,
        runtime_overrides=runtime_overrides,
    )
    source_reference = _report(
        "reference_reproduction_verified.json",
        identifiers[2],
        authoritative_capture=authoritative_capture,
        target_layer=capture_layer,
        runtime_overrides=runtime_overrides,
    )
    source_candidate = _report(
        "candidate_reproduction_verified.json",
        identifiers[3],
        authoritative_capture=authoritative_capture,
        target_layer=capture_layer,
        runtime_overrides=runtime_overrides,
    )
    pass1 = pass3_test_support._pass1(
        original_reference, original_candidate
    )
    pass2 = run_prefix_policy_reproduction(
        pass1,
        original_reference,
        original_candidate,
        reference_reproduction=source_reference,
        candidate_reproduction=source_candidate,
    )
    pass2_artifact = CanonicalPass2Artifact.from_result(pass2)
    reference_raw = pass3_test_support.trace_object(
        source_reference, layers=layers
    )
    candidate_raw = pass3_test_support.trace_object(
        source_candidate,
        layers=layers,
        digest_overrides=(
            {mismatch_layer: MISMATCH_DIGEST}
            if mismatch_layer is not None
            else {}
        ),
    )
    if authoritative_capture:
        for raw in (reference_raw, candidate_raw):
            raw["intra_layer_checkpoint_layout"] = {
                "sentinel": "P4-3 must not parse this structure"
            }
            raw["intra_layer_trace"] = [
                {"sentinel": "P4-3 must not parse this structure"}
            ]
    reference_trace = CanonicalLayerTrace.from_object(reference_raw)
    candidate_trace = CanonicalLayerTrace.from_object(candidate_raw)
    pass3 = run_coverage_scoped_layer_localization(
        pass2,
        pass2_artifact,
        reference_trace,
        candidate_trace,
        reference_source_report=source_reference,
        candidate_source_report=source_candidate,
    )
    return {
        "pass2": pass2,
        "pass2_artifact": pass2_artifact,
        "reference_report": source_reference,
        "candidate_report": source_candidate,
        "reference_raw": reference_raw,
        "candidate_raw": candidate_raw,
        "reference_trace": reference_trace,
        "candidate_trace": candidate_trace,
        "pass3": pass3,
        "pass3_artifact": CanonicalPass3Artifact.from_result(pass3),
    }


def two_generation_case(
    *,
    authoritative_capture: bool = True,
    authoritative_mismatch_layer: int | None = TARGET_LAYER,
    authoritative_layers: tuple[int, ...] = (0, 4, 8, 12),
    authoritative_runtime_overrides: dict | None = None,
) -> dict:
    """Build real Pass 0→3 discovery and authoritative generations."""
    discovery = _generation(
        identifiers=(
            "aset1:11111111111111111111111111111111",
            "aset1:22222222222222222222222222222222",
            "aset1:33333333333333333333333333333333",
            "aset1:44444444444444444444444444444444",
        ),
        authoritative_capture=False,
    )
    authoritative = _generation(
        identifiers=(
            "aset1:55555555555555555555555555555555",
            "aset1:66666666666666666666666666666666",
            "aset1:77777777777777777777777777777777",
            "aset1:88888888888888888888888888888888",
        ),
        authoritative_capture=authoritative_capture,
        mismatch_layer=authoritative_mismatch_layer,
        layers=authoritative_layers,
        runtime_overrides=authoritative_runtime_overrides,
    )
    return {
        "discovery": discovery,
        "authoritative": authoritative,
    }


def bind_kwargs(case: dict) -> dict:
    discovery = case["discovery"]
    authoritative = case["authoritative"]
    return {
        "discovery_pass3": discovery["pass3"],
        "discovery_pass3_artifact": discovery["pass3_artifact"],
        "authoritative_pass3": authoritative["pass3"],
        "authoritative_pass3_artifact": authoritative["pass3_artifact"],
        "pass2_artifact": authoritative["pass2_artifact"],
        "discovery_reference_report": discovery["reference_report"],
        "discovery_candidate_report": discovery["candidate_report"],
        "discovery_reference_trace": discovery["reference_trace"],
        "discovery_candidate_trace": discovery["candidate_trace"],
        "authoritative_reference_report": authoritative["reference_report"],
        "authoritative_candidate_report": authoritative["candidate_report"],
        "authoritative_reference_trace": authoritative["reference_trace"],
        "authoritative_candidate_trace": authoritative["candidate_trace"],
    }


def bind_case(case: dict, **overrides):
    values = bind_kwargs(case)
    values.update(overrides)
    positional = (
        values.pop("discovery_pass3"),
        values.pop("discovery_pass3_artifact"),
        values.pop("authoritative_pass3"),
        values.pop("authoritative_pass3_artifact"),
        values.pop("pass2_artifact"),
    )
    return bind_pass4_parent_inputs(*positional, **values)


def cloned(case: dict) -> dict:
    """Return a mutable outer copy; immutable wrappers are safe to share."""
    return {
        "discovery": dict(case["discovery"]),
        "authoritative": dict(case["authoritative"]),
    }


def wrapper_from_mutated_parent(
    case: dict,
    generation: str,
    mutate,
) -> CanonicalPass3Artifact:
    raw = copy.deepcopy(
        case[generation]["pass3_artifact"].materialize_verified()
    )
    mutate(raw)
    return CanonicalPass3Artifact.from_object(raw)
