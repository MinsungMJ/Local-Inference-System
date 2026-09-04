"""Real Pass 0–4 execution over the packaged seeded replay evidence."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
from typing import Any

from .artifact import serialize as serialize_pass0
from .build_profile import BuildCalibrationProfile
from .demo_fixture import DemoFixture, ResourceReader, load_demo_fixture
from .gate import build_gate
from .inputs import PreflightInputs, RunSide
from .model import CalibrationPreflightArtifact, ComparisonMode
from .pass0 import run_calibration_preflight
from .pass1 import run_token_localization
from .pass1_artifact import serialize as serialize_pass1
from .pass1_inputs import CanonicalRunReport, build_source_binding
from .pass1_model import Pass1Result
from .pass2 import run_prefix_policy_reproduction
from .pass2_artifact import serialize as serialize_pass2
from .pass2_model import Pass2Result
from .pass3 import run_coverage_scoped_layer_localization
from .pass3_artifact import serialize as serialize_pass3
from .pass3_inputs import CanonicalLayerTrace, CanonicalPass2Artifact
from .pass3_model import Pass3Result
from .pass4 import run_coverage_scoped_intra_layer_localization
from .pass4_artifact import serialize as serialize_pass4
from .pass4_model import Pass4Result
from .pass4_parent import CanonicalPass3Artifact
from .product_contract import EVIDENCE_NONCLAIMS, canonical_json_bytes
from .report_model import VerificationReport


PLACEHOLDER_ATTEMPT_ID = "lisa1:00000000000000000000000000000000"


class DemoPipelineError(RuntimeError):
    """The bounded fixture did not produce the frozen demonstration result."""


@dataclass(frozen=True)
class DemoGeneration:
    reference_original: CanonicalRunReport
    candidate_original: CanonicalRunReport
    reference_reproduction: CanonicalRunReport
    candidate_reproduction: CanonicalRunReport
    pass0: CalibrationPreflightArtifact
    pass1: Pass1Result
    pass2: Pass2Result
    pass2_artifact: CanonicalPass2Artifact
    reference_trace: CanonicalLayerTrace
    candidate_trace: CanonicalLayerTrace
    pass3: Pass3Result
    pass3_artifact: CanonicalPass3Artifact


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _report(
    fixture: DemoFixture,
    resource_name: str,
    artifact_set_id: str,
    *,
    authoritative: bool,
    target_layer: int,
) -> CanonicalRunReport:
    raw = fixture.value(resource_name)
    raw["artifact_set_id"] = artifact_set_id
    runtime = raw["manifest"]["runtime"]
    runtime.update(
        precision_path="f32_accum;weights=bf16;kv=bf16",
        layer_checkpoints_enabled=True,
        layer_checkpoint_step=18,
        diagnostics_enabled=False,
        perf_enabled=False,
        perf_per_token_enabled=False,
    )
    if authoritative:
        runtime.update(
            intra_layer_checkpoints_enabled=True,
            intra_layer_target_layer=target_layer,
            diagnostic_capture_profile="semantic_layer_and_intra_v1",
        )
    return CanonicalRunReport.from_object(raw)


def _pass0_and_pass1(
    reference: CanonicalRunReport,
    candidate: CanonicalRunReport,
    profile: dict[str, Any],
) -> tuple[CalibrationPreflightArtifact, Pass1Result]:
    prompt = profile["prompt_token_ids"]
    reference_side = RunSide.from_run_report(
        reference.materialize(), "reference", prompt_token_array=prompt
    )
    candidate_side = RunSide.from_run_report(
        candidate.materialize(), "candidate", prompt_token_array=prompt
    )
    pass0 = run_calibration_preflight(
        PreflightInputs(
            reference=reference_side,
            candidate=candidate_side,
            declared_mode=ComparisonMode(profile["comparison_mode"]),
            build_profile=BuildCalibrationProfile.from_dict(
                profile["build_calibration"]
            ),
        )
    )
    pass1 = run_token_localization(
        build_gate(pass0),
        reference,
        candidate,
        build_source_binding(reference, candidate),
    )
    return pass0, pass1


def _layer_trace(
    report: CanonicalRunReport,
    profile: dict[str, Any],
    *,
    candidate: bool,
) -> dict[str, Any]:
    layers = profile["layer_indices"]
    mismatch_layer = profile["target_layer"]
    requested: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    for ordinal, layer in enumerate(layers):
        coordinate = {
            "runtime_checkpoint_step": profile["runtime_checkpoint_step"],
            "layer_index": layer,
            "tensor_role": "layer_output",
            "batch_index": 0,
            "sequence_index": 0,
            "stage_order": 0,
            "execution_ordinal": ordinal,
        }
        requested.append(copy.deepcopy(coordinate))
        digest = f"sha256:{layer + 1:064x}"
        if candidate and layer == mismatch_layer:
            digest = profile["candidate_layer_digest"]
        entries.append(
            {
                "step": profile["runtime_checkpoint_step"],
                **copy.deepcopy(coordinate),
                "phase": "decode",
                "name": f"layer.{layer}.output",
                "observed_dtype": "fp32",
                "shape": [1, 1, 3],
                "element_count": 3,
                "available_summary_fields": [
                    "min",
                    "max",
                    "mean",
                    "l2",
                    "nan",
                    "inf",
                    "digest",
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
                    "value": digest,
                },
            }
        )
    raw = {
        "schema": "lis.execution_artifact/v1",
        "kind": "layer_trace",
        "artifact_set_id": report.materialize()["artifact_set_id"],
        "manifest": copy.deepcopy(report.materialize()["manifest"]),
        "checkpoint_layout": {
            "layout_name": "llama_layer_output_summary",
            "layout_version": 1,
            "runtime_checkpoint_step": profile["runtime_checkpoint_step"],
            "tensor_role": "layer_output",
            "stage_order": 0,
            "ordering_semantics": "runtime_step_layer_stage_ordinal",
            "total_layer_count": profile["total_layer_count"],
            "requested_coordinates": requested,
            "captured_coordinates": copy.deepcopy(requested),
            "missing_coordinates": [],
            "available_summary_fields": [
                "min",
                "max",
                "mean",
                "l2",
                "nan",
                "inf",
                "digest",
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
    return raw


def _add_intra_layer(
    raw: dict[str, Any],
    fixture: DemoFixture,
    profile: dict[str, Any],
    *,
    candidate: bool,
) -> None:
    blocks = fixture.value("intra_layer_trace")
    raw["intra_layer_checkpoint_layout"] = blocks[
        "intra_layer_checkpoint_layout"
    ]
    raw["intra_layer_trace"] = blocks["intra_layer_trace"]
    if candidate:
        mismatch_order = profile["candidate_intra_layer_mismatch_stage_order"]
        matches = [
            entry
            for entry in raw["intra_layer_trace"]
            if entry["stage_order"] == mismatch_order
        ]
        if len(matches) != 1:
            raise DemoPipelineError("intra-layer mismatch seed is not unique")
        matches[0]["digest"]["value"] = profile["candidate_layer_digest"]


def _build_generation(
    fixture: DemoFixture,
    profile: dict[str, Any],
    name: str,
    *,
    authoritative: bool,
) -> DemoGeneration:
    ids = profile["generations"][name]
    target_layer = profile["target_layer"]
    reference_original = _report(
        fixture,
        "reference_original",
        ids["reference_original_artifact_set_id"],
        authoritative=authoritative,
        target_layer=target_layer,
    )
    candidate_original = _report(
        fixture,
        "candidate_original",
        ids["candidate_original_artifact_set_id"],
        authoritative=authoritative,
        target_layer=target_layer,
    )
    reference_reproduction = _report(
        fixture,
        "reference_reproduction",
        ids["reference_reproduction_artifact_set_id"],
        authoritative=authoritative,
        target_layer=target_layer,
    )
    candidate_reproduction = _report(
        fixture,
        "candidate_reproduction",
        ids["candidate_reproduction_artifact_set_id"],
        authoritative=authoritative,
        target_layer=target_layer,
    )
    pass0, pass1 = _pass0_and_pass1(
        reference_original, candidate_original, profile
    )
    pass2 = run_prefix_policy_reproduction(
        pass1,
        reference_original,
        candidate_original,
        reference_reproduction=reference_reproduction,
        candidate_reproduction=candidate_reproduction,
    )
    pass2_artifact = CanonicalPass2Artifact.from_result(pass2)
    reference_raw = _layer_trace(
        reference_reproduction, profile, candidate=False
    )
    candidate_raw = _layer_trace(
        candidate_reproduction, profile, candidate=True
    )
    if authoritative:
        _add_intra_layer(reference_raw, fixture, profile, candidate=False)
        _add_intra_layer(candidate_raw, fixture, profile, candidate=True)
    reference_trace = CanonicalLayerTrace.from_object(reference_raw)
    candidate_trace = CanonicalLayerTrace.from_object(candidate_raw)
    pass3 = run_coverage_scoped_layer_localization(
        pass2,
        pass2_artifact,
        reference_trace,
        candidate_trace,
        reference_source_report=reference_reproduction,
        candidate_source_report=candidate_reproduction,
    )
    return DemoGeneration(
        reference_original,
        candidate_original,
        reference_reproduction,
        candidate_reproduction,
        pass0,
        pass1,
        pass2,
        pass2_artifact,
        reference_trace,
        candidate_trace,
        pass3,
        CanonicalPass3Artifact.from_result(pass3),
    )


def _validate_expected(
    profile: dict[str, Any],
    discovery: DemoGeneration,
    authoritative: DemoGeneration,
    pass4: Pass4Result,
) -> None:
    expected = profile["expected"]
    for generation in (discovery, authoritative):
        actual = (
            generation.pass0.pass0_verdict.value,
            generation.pass1.status.value,
            generation.pass2.status.value,
            generation.pass3.status.value,
        )
        wanted = (
            expected["pass0_status"],
            expected["pass1_status"],
            expected["pass2_status"],
            expected["pass3_status"],
        )
        if actual != wanted:
            raise DemoPipelineError("Pass 0–3 result disagrees with demo profile")
    if pass4.status.value != expected["pass4_status"]:
        raise DemoPipelineError("Pass 4 result disagrees with demo profile")
    localization = authoritative.pass1.localization
    if (
        localization is None
        or localization.generated_token_step != expected["generated_token_step"]
        or authoritative.pass3.suspect_interval is None
        or authoritative.pass3.suspect_interval.notation != expected["layer_interval"]
        or pass4.suspect_interval is None
        or pass4.suspect_interval.notation != expected["intra_layer_interval"]
    ):
        raise DemoPipelineError("demo localization result drifted")
    if (
        discovery.pass3.earliest_observable_suspect_layer
        != authoritative.pass3.earliest_observable_suspect_layer
        or authoritative.pass3.earliest_observable_suspect_layer
        != profile["target_layer"]
    ):
        raise DemoPipelineError("demo Pass 3A/3B target continuity failed")


def _component_identity(field: str, value: Any) -> str:
    return _digest(
        {
            "domain": "lis.demo_fixture_identity/v1",
            "field": field,
            "value": value,
        }
    )


def _identities(
    fixture: DemoFixture, generation: DemoGeneration
) -> dict[str, dict[str, str]]:
    identities: dict[str, dict[str, str]] = {}
    for role, report in (
        ("reference", generation.reference_original),
        ("candidate", generation.candidate_original),
    ):
        manifest = report.materialize()["manifest"]
        identities[role] = {
            "source_sha256": fixture.manifest_sha256,
            "binary_sha256": _component_identity("binary", manifest["binary"]),
            "model_sha256": _component_identity("model", manifest["model"]),
            "config_sha256": _component_identity("config", manifest["config"]),
            "input_sha256": _component_identity("input", manifest["input"]),
            "runtime_sha256": _component_identity("runtime", manifest["runtime"]),
            "backend_sha256": _component_identity("backend", manifest["backend"]),
        }
    return identities


def _stage(name: str, result_ref: str, tier: str) -> dict[str, Any]:
    return {
        "name": name,
        "state": "executed",
        "result_ref": result_ref,
        "evidence_tier": tier,
        "failure_class": None,
        "reason": None,
        "blocker": None,
    }


def _ordered_unique(values: list[Any]) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def build_demo_report(*, reader: ResourceReader | None = None) -> VerificationReport:
    """Run every production Pass consumer and build one canonical report."""

    fixture = load_demo_fixture(reader)
    profile = fixture.value("profile")
    discovery = _build_generation(
        fixture, profile, "discovery", authoritative=False
    )
    authoritative = _build_generation(
        fixture, profile, "authoritative", authoritative=True
    )
    pass4 = run_coverage_scoped_intra_layer_localization(
        discovery.pass3,
        discovery.pass3_artifact,
        authoritative.pass3,
        authoritative.pass3_artifact,
        authoritative.pass2_artifact,
        discovery_reference_report=discovery.reference_reproduction,
        discovery_candidate_report=discovery.candidate_reproduction,
        discovery_reference_trace=discovery.reference_trace,
        discovery_candidate_trace=discovery.candidate_trace,
        authoritative_reference_report=authoritative.reference_reproduction,
        authoritative_candidate_report=authoritative.candidate_reproduction,
        authoritative_reference_trace=authoritative.reference_trace,
        authoritative_candidate_trace=authoritative.candidate_trace,
        discovery_pass2_artifact=discovery.pass2_artifact,
    )
    _validate_expected(profile, discovery, authoritative, pass4)

    localization = authoritative.pass1.localization
    reference = authoritative.pass1.reference
    candidate = authoritative.pass1.candidate
    layer_coverage = authoritative.pass3.coverage
    intra_coverage = pass4.coverage
    assert localization is not None
    assert reference is not None and candidate is not None
    assert authoritative.pass3.suspect_interval is not None
    assert pass4.suspect_interval is not None and intra_coverage is not None

    recapture_ref = _digest(
        {
            "reference_original": authoritative.reference_original.identity.run_report_sha256,
            "candidate_original": authoritative.candidate_original.identity.run_report_sha256,
            "reference_reproduction": (
                authoritative.reference_reproduction.identity.run_report_sha256
            ),
            "candidate_reproduction": (
                authoritative.candidate_reproduction.identity.run_report_sha256
            ),
            "reference_trace": authoritative.reference_trace.identity.trace_sha256,
            "candidate_trace": authoritative.candidate_trace.identity.trace_sha256,
        }
    )
    decision = {
        "verdict": "REGRESSION",
        "generated_token_step": localization.generated_token_step,
        "layer_interval": authoritative.pass3.suspect_interval.notation,
        "intra_layer_interval": pass4.suspect_interval.notation,
    }
    stages = [
        _stage("preflight", fixture.manifest_sha256, "seeded_fixture_manifest"),
        _stage(
            "reference_original_execution",
            discovery.reference_original.identity.run_report_sha256,
            "bounded_replay_run_report",
        ),
        _stage(
            "candidate_original_execution",
            discovery.candidate_original.identity.run_report_sha256,
            "bounded_replay_run_report",
        ),
        _stage("pass0_calibration", _digest(serialize_pass0(discovery.pass0)), "tier0_structural"),
        _stage(
            "pass1_token_localization",
            _digest(serialize_pass1(discovery.pass1)),
            "selected_token_array_exact",
        ),
        _stage(
            "pass2_prefix_policy_reproduction",
            _digest(serialize_pass2(discovery.pass2)),
            "independent_rerun_replay_verified",
        ),
        _stage(
            "pass3a_discovery",
            _digest(serialize_pass3(discovery.pass3)),
            "tier1_bounded_digest",
        ),
        _stage("bounded_recapture", recapture_ref, "source_bound_replay_recapture"),
        _stage(
            "pass3b_authoritative_localization",
            _digest(serialize_pass3(authoritative.pass3)),
            "tier1_bounded_digest",
        ),
        _stage(
            "pass4_intra_layer_localization",
            _digest(serialize_pass4(pass4)),
            "tier1_bounded_digest",
        ),
        _stage("aggregation", _digest(decision), "product_contract_v1"),
        _stage("cleanup", _digest({"state": "pending_orchestrator_cleanup"}), "pending_cleanup"),
    ]
    raw = {
        "schema": "lis.verification_report/v1",
        "kind": "verification_report",
        "report_version": "1.0",
        "attempt": {
            "id": PLACEHOLDER_ATTEMPT_ID,
            "workflow_classification": "development_debugging",
        },
        "command": {
            "mode": "demo",
            "require_supported": False,
            "output_path_redacted": True,
        },
        "verdict": "REGRESSION",
        "reason_codes": ["selected_token_mismatch"],
        "policy_result": {
            "policy": "default",
            "satisfied": False,
            "exit_code": 4,
            "reason": "seeded_selected_token_regression",
        },
        "identities": _identities(fixture, authoritative),
        "token_comparison": {
            "status": "mismatch",
            "reference_observed_count": reference.selected_tokens.observed_length,
            "candidate_observed_count": candidate.selected_tokens.observed_length,
            "first_mismatch": {
                "generated_token_step": localization.generated_token_step,
                "reference_token_id": localization.reference_selected_token_id,
                "candidate_token_id": localization.candidate_selected_token_id,
            },
        },
        "localization": {
            "layer_suspect_interval": authoritative.pass3.suspect_interval.notation,
            "intra_layer_suspect_interval": pass4.suspect_interval.notation,
        },
        "coverage": {
            "scope": "seeded_selected_tokens_and_bounded_llama_decode",
            "common_layers": _ordered_unique(
                [item.layer_index for item in layer_coverage.common_captured]
            ),
            "missing_reference_layers": _ordered_unique(
                [item.coordinate.layer_index for item in layer_coverage.reference_missing]
            ),
            "missing_candidate_layers": _ordered_unique(
                [item.coordinate.layer_index for item in layer_coverage.candidate_missing]
            ),
            "common_intra_layer_stages": _ordered_unique(
                [item.stage_id for item in intra_coverage.common_captured]
            ),
            "missing_reference_intra_layer_stages": _ordered_unique(
                [item.coordinate.stage_id for item in intra_coverage.reference_missing]
            ),
            "missing_candidate_intra_layer_stages": _ordered_unique(
                [item.coordinate.stage_id for item in intra_coverage.candidate_missing]
            ),
        },
        "numeric_confirmation": {
            "status": "not_performed",
            "confirmed_divergence_at_checkpoint": None,
            "confirmed_first_divergence": None,
        },
        "evidence": {
            "tier": "tier1_bounded_digest",
            "ceiling": "bounded_suspect_interval",
            "nonclaims": {name: False for name in EVIDENCE_NONCLAIMS},
        },
        "stages": stages,
        "next_action": {
            "code": "inspect_seeded_suspect_interval",
            "summary": (
                "Inspect the seeded layer (4, 8] and intra-layer "
                "(rope_key_output, attention_scores] suspect intervals."
            ),
        },
        "warnings": [
            "Seeded synthetic replay; no model or LIS binary was executed.",
            "Numeric confirmation was not performed.",
        ],
        "cleanup": {
            "status": "success",
            "residue_status": "none_observed",
            "observed": True,
            "retained_debug": False,
        },
    }
    return VerificationReport.from_dict(raw)


def semantic_projection(report: VerificationReport) -> dict[str, Any]:
    """Normalize only per-attempt fields for repeatability comparison."""

    value = report.to_dict()
    value["attempt"]["id"] = PLACEHOLDER_ATTEMPT_ID
    cleanup = value["stages"][-1]
    if cleanup["name"] != "cleanup":
        raise DemoPipelineError("cleanup stage is not canonical")
    cleanup["result_ref"] = "sha256:" + "0" * 64
    return value


def semantic_digest(report: VerificationReport) -> str:
    return _digest(semantic_projection(report))
