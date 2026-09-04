"""Pass 0–4 orchestration for real backend and runtime comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

from .artifact import serialize as serialize_pass0
from .forced_prefix import (
    forced_prefix_binding_bytes,
    validate_paired_forced_prefix_reproductions,
)
from .gate import build_gate
from .inputs import PreflightInputs, RunSide
from .model import ComparisonMode, Pass0Verdict
from .model_profile import (
    ModelExecutionProfile,
    ResolvedModel,
    load_model_profile,
    resolve_model,
)
from .orchestrator import RunContext
from .pass0 import run_calibration_preflight
from .pass1 import run_token_localization
from .pass1_artifact import serialize as serialize_pass1
from .pass1_inputs import CanonicalRunReport, build_source_binding
from .pass1_model import Pass1Result, Pass1Status
from .pass2 import run_prefix_policy_reproduction
from .pass2_artifact import serialize as serialize_pass2
from .pass2_inputs import extract_run_evidence
from .pass2_model import Pass2Result, Pass2Status
from .pass3 import run_coverage_scoped_layer_localization
from .pass3_artifact import serialize as serialize_pass3
from .pass3_inputs import CanonicalPass2Artifact
from .pass3_model import Pass3Result, Pass3Status
from .pass4 import run_coverage_scoped_intra_layer_localization
from .pass4_artifact import serialize as serialize_pass4
from .pass4_contract import Pass4Status
from .pass4_parent import CanonicalPass3Artifact
from .product_contract import (
    CANONICAL_STAGES,
    EVIDENCE_NONCLAIMS,
    CustomerVerdict,
    StageState,
    canonical_json_bytes,
)
from .aggregation import policy_result_for
from .real_execution import (
    RealExecutionError,
    RealLISExecutor,
    ResolvedBinary,
    RunCapture,
    resolve_backend_binary,
    resolve_binary,
    role_identity,
)


PLACEHOLDER_ATTEMPT_ID = "lisa1:00000000000000000000000000000000"


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _binary_fingerprint(report: CanonicalRunReport, role: str) -> str:
    return extract_run_evidence(report, role).binary_fingerprint


def _backend_name(report: CanonicalRunReport) -> str:
    manifest = report.materialize().get("manifest")
    backend = manifest.get("backend") if isinstance(manifest, dict) else None
    name = backend.get("name") if isinstance(backend, dict) else None
    if not isinstance(name, str) or not name:
        raise RealExecutionError("LIS backend identity is unavailable")
    return name


def _artifact_set_id(report: CanonicalRunReport) -> str:
    value = report.materialize().get("artifact_set_id")
    if not isinstance(value, str) or not value:
        raise RealExecutionError("LIS artifact-set identity is unavailable")
    return value


def _require_optimized_backend(report: CanonicalRunReport) -> None:
    if _backend_name(report) not in {"avx2", "avx512"}:
        raise RealExecutionError(
            "optimized backend is unavailable or unsupported",
            classification="unsupported",
        )


def _unavailable_identities(seed: str) -> dict[str, dict[str, str]]:
    fields = (
        "source_sha256",
        "binary_sha256",
        "model_sha256",
        "config_sha256",
        "input_sha256",
        "runtime_sha256",
        "backend_sha256",
    )
    return {
        role: {
            field: _digest(
                {
                    "domain": "lis.verify.unavailable_identity/v1",
                    "mode": seed,
                    "role": role,
                    "field": field,
                }
            )
            for field in fields
        }
        for role in ("reference", "candidate")
    }


def _empty_coverage(scope: str) -> dict[str, Any]:
    return {
        "scope": scope,
        "common_layers": [],
        "missing_reference_layers": [],
        "missing_candidate_layers": [],
        "common_intra_layer_stages": [],
        "missing_reference_intra_layer_stages": [],
        "missing_candidate_intra_layer_stages": [],
    }


def _cleanup_placeholder() -> dict[str, Any]:
    return {
        "name": "cleanup",
        "state": "executed",
        "result_ref": _digest({"state": "pending_orchestrator_cleanup"}),
        "evidence_tier": "pending_cleanup",
        "failure_class": None,
        "reason": None,
        "blocker": None,
    }


def _finish_executed(context: RunContext, result: Any, tier: str) -> str:
    identity = _digest(result)
    context.finish_stage(
        StageState.EXECUTED,
        result_ref=identity,
        evidence_tier=tier,
    )
    return identity


def _finish_until_aggregation(
    context: RunContext,
    *,
    state: StageState,
    reason: str,
    blocker: str | None = None,
) -> None:
    while context.state_machine.next_stage not in {"aggregation", None}:
        stage = context.state_machine.next_stage
        assert stage is not None
        context.start_stage(stage)
        if state == StageState.NOT_APPLICABLE:
            context.finish_stage(state, reason=reason)
        else:
            assert blocker is not None
            context.finish_stage(state, reason=reason, blocker=blocker)


def _token_projection(pass1: Pass1Result | None) -> dict[str, Any]:
    if pass1 is None or pass1.reference is None or pass1.candidate is None:
        return {
            "status": "unavailable",
            "reference_observed_count": 0,
            "candidate_observed_count": 0,
            "first_mismatch": None,
        }
    reference = pass1.reference.selected_tokens
    candidate = pass1.candidate.selected_tokens
    reference_count = reference.observed_length or 0
    candidate_count = candidate.observed_length or 0
    if pass1.status == Pass1Status.TOKEN_EQUIVALENT_ON_OBSERVED_RANGE:
        return {
            "status": "equal",
            "reference_observed_count": reference_count,
            "candidate_observed_count": candidate_count,
            "first_mismatch": None,
        }
    localization = pass1.localization
    if pass1.status == Pass1Status.FIRST_MISMATCH_FOUND and localization is not None:
        return {
            "status": "mismatch",
            "reference_observed_count": reference_count,
            "candidate_observed_count": candidate_count,
            "first_mismatch": {
                "generated_token_step": localization.generated_token_step,
                "reference_token_id": localization.reference_selected_token_id,
                "candidate_token_id": localization.candidate_selected_token_id,
            },
        }
    return {
        "status": "unavailable",
        "reference_observed_count": reference_count,
        "candidate_observed_count": candidate_count,
        "first_mismatch": None,
    }


def _report(
    context: RunContext,
    *,
    verdict: CustomerVerdict,
    reason_codes: list[str],
    policy_reason: str,
    identities: dict[str, dict[str, str]],
    pass1: Pass1Result | None,
    layer_interval: str | None = None,
    intra_interval: str | None = None,
    coverage: dict[str, Any] | None = None,
    evidence_tier: str = "tier0_structural",
    evidence_ceiling: str = "no_comparison",
    next_action: dict[str, str] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    if context.state_machine.next_stage != "aggregation":
        raise RuntimeError("real pipeline did not reach aggregation")
    context.start_stage("aggregation")
    _finish_executed(
        context,
        {
            "verdict": verdict.value,
            "reason_codes": reason_codes,
            "evidence_ceiling": evidence_ceiling,
        },
        "product_contract_v1",
    )
    stages = [item.to_dict() for item in context.state_machine.results]
    stages.append(_cleanup_placeholder())
    return {
        "schema": "lis.verification_report/v1",
        "kind": "verification_report",
        "report_version": "1.0",
        "attempt": {
            "id": PLACEHOLDER_ATTEMPT_ID,
            "workflow_classification": context.request.workflow.value,
        },
        "command": {
            "mode": context.request.mode,
            "require_supported": context.request.require_supported,
            "output_path_redacted": True,
        },
        "verdict": verdict.value,
        "reason_codes": list(dict.fromkeys(reason_codes)),
        "policy_result": policy_result_for(
            verdict,
            require_supported=context.request.require_supported,
            reason=policy_reason,
        ),
        "identities": identities,
        "token_comparison": _token_projection(pass1),
        "localization": {
            "layer_suspect_interval": layer_interval,
            "intra_layer_suspect_interval": intra_interval,
        },
        "coverage": coverage or _empty_coverage("preflight_only"),
        "numeric_confirmation": {
            "status": "not_performed",
            "confirmed_divergence_at_checkpoint": None,
            "confirmed_first_divergence": None,
        },
        "evidence": {
            "tier": evidence_tier,
            "ceiling": evidence_ceiling,
            "nonclaims": {name: False for name in EVIDENCE_NONCLAIMS},
        },
        "stages": stages,
        "next_action": next_action,
        "warnings": list(dict.fromkeys(warnings or [])),
        "cleanup": {
            "status": "success",
            "residue_status": "none_observed",
            "observed": True,
            "retained_debug": False,
        },
    }


def _blocked_report(
    context: RunContext,
    *,
    blocker: str,
    verdict: CustomerVerdict,
    reason_code: str,
    policy_reason: str,
    identities: dict[str, dict[str, str]],
    pass1: Pass1Result | None,
    warning: str,
) -> dict[str, Any]:
    _finish_until_aggregation(
        context,
        state=StageState.BLOCKED,
        reason="Upstream evidence did not authorize this stage.",
        blocker=blocker,
    )
    return _report(
        context,
        verdict=verdict,
        reason_codes=[reason_code],
        policy_reason=policy_reason,
        identities=identities,
        pass1=pass1,
        coverage=_empty_coverage(
            "selected_tokens" if pass1 is not None else "preflight_only"
        ),
        evidence_tier=(
            "selected_token_array_exact"
            if pass1 is not None and pass1.status == Pass1Status.FIRST_MISMATCH_FOUND
            else "tier0_structural"
        ),
        evidence_ceiling=(
            "token_localization_only"
            if pass1 is not None and pass1.status == Pass1Status.FIRST_MISMATCH_FOUND
            else "no_comparison"
        ),
        next_action={
            "code": "repair_or_recapture_evidence",
            "summary": "Repair the reported evidence boundary and start a new verification attempt.",
        },
        warnings=[warning],
    )


@dataclass(frozen=True)
class RealSetup:
    profile: ModelExecutionProfile
    model: ResolvedModel
    reference_binary: ResolvedBinary
    candidate_binary: ResolvedBinary
    reference_environment: Mapping[str, str]
    candidate_environment: Mapping[str, str]
    mode: ComparisonMode


@dataclass(frozen=True)
class Generation:
    reference_original: RunCapture
    candidate_original: RunCapture
    pass1: Pass1Result
    pass2: Pass2Result
    reference_reproduction: RunCapture
    candidate_reproduction: RunCapture


def _setup(context: RunContext) -> RealSetup:
    profile = load_model_profile()
    assert context.request.model is not None
    model = resolve_model(context.request.model, profile)
    if context.request.mode == "backend":
        binary = resolve_backend_binary()
        _validate_acceptance_candidate(context, binary)
        return RealSetup(
            profile,
            model,
            binary,
            binary,
            {"LIS_SIMD": "0"},
            {},
            ComparisonMode.BACKEND_DIFFERENTIAL,
        )
    if context.request.mode != "runtime":
        raise RealExecutionError("real execution mode is unsupported", classification="unsupported")
    assert context.request.reference_bin is not None
    assert context.request.candidate_bin is not None
    reference = resolve_binary(context.request.reference_bin)
    candidate = resolve_binary(context.request.candidate_bin)
    _validate_acceptance_candidate(context, candidate)
    if reference.provenance.binary_sha256 == candidate.provenance.binary_sha256:
        raise RealExecutionError(
            "runtime comparison requires two distinct binary identities",
            classification="unsupported",
        )
    return RealSetup(
        profile,
        model,
        reference,
        candidate,
        {},
        {},
        ComparisonMode.RUNTIME_DIFFERENTIAL,
    )


def _validate_acceptance_candidate(
    context: RunContext, candidate: ResolvedBinary
) -> None:
    manifest = context.request.acceptance_manifest
    if manifest is None:
        return
    provenance = candidate.provenance
    if (
        provenance.revision != manifest.source_revision
        or provenance.source_sha256 != manifest.source_tree_sha256
        or provenance.dirty is not False
    ):
        raise RealExecutionError(
            "acceptance manifest does not bind the candidate source",
            classification="harness_error",
        )


def _run_pass0(
    setup: RealSetup,
    reference: CanonicalRunReport,
    candidate: CanonicalRunReport,
) -> Any:
    tokens = list(setup.profile.direct_token_ids)
    pass0 = run_calibration_preflight(
        PreflightInputs(
            reference=RunSide.from_run_report(
                reference.materialize(), "reference", prompt_token_array=tokens
            ),
            candidate=RunSide.from_run_report(
                candidate.materialize(), "candidate", prompt_token_array=tokens
            ),
            declared_mode=setup.mode,
        )
    )
    return pass0


def _run_pass1(
    pass0: Any,
    reference: CanonicalRunReport,
    candidate: CanonicalRunReport,
) -> Pass1Result:
    return run_token_localization(
        build_gate(pass0),
        reference,
        candidate,
        build_source_binding(reference, candidate),
    )


def _reproduction_pair(
    adapter: RealLISExecutor,
    setup: RealSetup,
    pass1: Pass1Result,
    reference_original: CanonicalRunReport,
    candidate_original: CanonicalRunReport,
    *,
    intra_layer: int | None,
) -> tuple[RunCapture, RunCapture, Pass2Result]:
    localization = pass1.localization
    prefix = pass1.prefix_for_reproduction
    if localization is None or prefix is None:
        raise RealExecutionError("Pass 1 did not produce a reproducible boundary")
    checkpoint_step = localization.runtime_checkpoint_step
    if checkpoint_step is None:
        raise RealExecutionError("Pass 1 checkpoint mapping is unavailable")
    forced = tuple(prefix.exact_token_ids) if prefix.token_count else None
    ref_binding = (
        forced_prefix_binding_bytes(pass1, reference_original, role="reference")
        if forced is not None
        else None
    )
    cand_binding = (
        forced_prefix_binding_bytes(pass1, candidate_original, role="candidate")
        if forced is not None
        else None
    )
    reference = adapter.run(
        setup.reference_binary,
        role="reference-reproduction",
        environment=setup.reference_environment,
        expected_backend=_backend_name(reference_original),
        expected_binary_fingerprint=_binary_fingerprint(reference_original, "reference"),
        forced_prefix=forced,
        forced_binding=ref_binding,
        checkpoint_step=checkpoint_step,
        intra_layer=intra_layer,
    )
    candidate = adapter.run(
        setup.candidate_binary,
        role="candidate-reproduction",
        environment=setup.candidate_environment,
        expected_backend=_backend_name(candidate_original),
        expected_binary_fingerprint=_binary_fingerprint(candidate_original, "candidate"),
        forced_prefix=forced,
        forced_binding=cand_binding,
        checkpoint_step=checkpoint_step,
        intra_layer=intra_layer,
    )
    if forced is not None:
        validate_paired_forced_prefix_reproductions(
            pass1,
            reference_original,
            candidate_original,
            reference.report,
            candidate.report,
        )
    pass2 = run_prefix_policy_reproduction(
        pass1,
        reference_original,
        candidate_original,
        exact_prefix_source=prefix.exact_token_ids,
        reference_reproduction=reference.report,
        candidate_reproduction=candidate.report,
    )
    return reference, candidate, pass2


def _new_generation(
    adapter: RealLISExecutor,
    setup: RealSetup,
    *,
    intra_layer: int | None,
) -> Generation:
    reference_original = adapter.run(
        setup.reference_binary,
        role="reference-recapture-original",
        environment=setup.reference_environment,
        expected_backend="reference" if setup.mode == ComparisonMode.BACKEND_DIFFERENTIAL else None,
    )
    candidate_original = adapter.run(
        setup.candidate_binary,
        role="candidate-recapture-original",
        environment=setup.candidate_environment,
    )
    if (
        setup.mode == ComparisonMode.BACKEND_DIFFERENTIAL
        and _backend_name(candidate_original.report) not in {"avx2", "avx512"}
    ):
        _require_optimized_backend(candidate_original.report)
    pass0 = _run_pass0(
        setup, reference_original.report, candidate_original.report
    )
    pass1 = _run_pass1(
        pass0, reference_original.report, candidate_original.report
    )
    if pass1.status != Pass1Status.FIRST_MISMATCH_FOUND:
        raise RealExecutionError("selected-token mismatch did not reproduce during recapture")
    reference_reproduction, candidate_reproduction, pass2 = _reproduction_pair(
        adapter,
        setup,
        pass1,
        reference_original.report,
        candidate_original.report,
        intra_layer=intra_layer,
    )
    return Generation(
        reference_original,
        candidate_original,
        pass1,
        pass2,
        reference_reproduction,
        candidate_reproduction,
    )


def run_real_pipeline(
    context: RunContext,
    *,
    executor=None,
    signal_exit: list[int] | None = None,
) -> dict[str, Any]:
    """Drive one real mode attempt and always return a canonical report."""

    identities = _unavailable_identities(context.request.mode)
    pass1: Pass1Result | None = None
    adapter: RealLISExecutor | None = None
    context.start_stage("preflight")
    try:
        setup = _setup(context)
        adapter = RealLISExecutor(
            runtime_dir=context.workspace.runtime_dir,
            model=setup.model,
            profile=setup.profile,
            timeout_seconds=context.request.stage_timeout_seconds,
            executor=executor,
        )
        _finish_executed(
            context,
            {
                "mode": setup.mode.value,
                "profile_sha256": setup.profile.identity_sha256,
                "model_sha256": setup.model.model_sha256,
                "config_sha256": setup.model.config_sha256,
                "reference_source_sha256": setup.reference_binary.provenance.source_sha256,
                "reference_binary_sha256": setup.reference_binary.provenance.binary_sha256,
                "candidate_source_sha256": setup.candidate_binary.provenance.source_sha256,
                "candidate_binary_sha256": setup.candidate_binary.provenance.binary_sha256,
            },
            "source_bound_preflight",
        )

        context.start_stage("reference_original_execution")
        reference_original = adapter.run(
            setup.reference_binary,
            role="reference-original",
            environment=setup.reference_environment,
            expected_backend=(
                "reference"
                if setup.mode == ComparisonMode.BACKEND_DIFFERENTIAL
                else None
            ),
        )
        _finish_executed(
            context,
            {"run_report_sha256": reference_original.report.identity.run_report_sha256},
            "run_report",
        )

        context.start_stage("candidate_original_execution")
        candidate_original = adapter.run(
            setup.candidate_binary,
            role="candidate-original",
            environment=setup.candidate_environment,
        )
        if (
            setup.mode == ComparisonMode.BACKEND_DIFFERENTIAL
            and _backend_name(candidate_original.report) not in {"avx2", "avx512"}
        ):
            _require_optimized_backend(candidate_original.report)
        _finish_executed(
            context,
            {"run_report_sha256": candidate_original.report.identity.run_report_sha256},
            "run_report",
        )
        identities = {
            "reference": role_identity(
                setup.reference_binary,
                setup.model,
                setup.profile,
                adapter.input_sha256,
                reference_original.report,
            ),
            "candidate": role_identity(
                setup.candidate_binary,
                setup.model,
                setup.profile,
                adapter.input_sha256,
                candidate_original.report,
            ),
        }

        context.start_stage("pass0_calibration")
        pass0 = _run_pass0(
            setup, reference_original.report, candidate_original.report
        )
        _finish_executed(context, serialize_pass0(pass0), "tier0_structural")
        if pass0.pass0_verdict == Pass0Verdict.COMPARISON_BLOCKED:
            return _blocked_report(
                context,
                blocker="pass0_calibration",
                verdict=CustomerVerdict.UNSUPPORTED,
                reason_code="unsupported_comparison",
                policy_reason="pass0_comparison_blocked",
                identities=identities,
                pass1=None,
                warning="Pass 0 did not authorize selected-token comparison.",
            )

        context.start_stage("pass1_token_localization")
        pass1 = _run_pass1(
            pass0, reference_original.report, candidate_original.report
        )
        _finish_executed(
            context,
            serialize_pass1(pass1),
            "selected_token_array_exact",
        )
        if pass1.status == Pass1Status.TOKEN_EQUIVALENT_ON_OBSERVED_RANGE:
            _finish_until_aggregation(
                context,
                state=StageState.NOT_APPLICABLE,
                reason="No selected-token mismatch was observed.",
            )
            return _report(
                context,
                verdict=CustomerVerdict.PASS,
                reason_codes=["selected_tokens_equal"],
                policy_reason="semantic_pass",
                identities=identities,
                pass1=pass1,
                coverage=_empty_coverage("selected_tokens"),
                evidence_tier="selected_token_array_exact",
                evidence_ceiling="equivalent_on_observed_selected_token_range",
                next_action=None,
            )
        if pass1.status != Pass1Status.FIRST_MISMATCH_FOUND:
            return _blocked_report(
                context,
                blocker="pass1_token_localization",
                verdict=CustomerVerdict.INCONCLUSIVE,
                reason_code="partial_evidence_inconclusive",
                policy_reason="pass1_evidence_inconclusive",
                identities=identities,
                pass1=pass1,
                warning="Selected-token evidence was insufficient for localization.",
            )

        context.start_stage("pass2_prefix_policy_reproduction")
        reference_reproduction, candidate_reproduction, pass2 = _reproduction_pair(
            adapter,
            setup,
            pass1,
            reference_original.report,
            candidate_original.report,
            intra_layer=None,
        )
        _finish_executed(
            context,
            serialize_pass2(pass2),
            pass2.reproduction_evidence_tier.value,
        )
        if pass2.status != Pass2Status.REPRODUCTION_VERIFIED:
            return _blocked_report(
                context,
                blocker="pass2_prefix_policy_reproduction",
                verdict=CustomerVerdict.REGRESSION,
                reason_code="selected_token_mismatch",
                policy_reason="regression_retained_reproduction_incomplete",
                identities=identities,
                pass1=pass1,
                warning="The selected-token regression was proven, but reproduction evidence was incomplete.",
            )

        context.start_stage("pass3a_discovery")
        if reference_reproduction.layer_trace is None or candidate_reproduction.layer_trace is None:
            raise RealExecutionError("Pass 3A layer trace is unavailable")
        discovery_pass2_artifact = CanonicalPass2Artifact.from_result(pass2)
        discovery_pass3 = run_coverage_scoped_layer_localization(
            pass2,
            discovery_pass2_artifact,
            reference_reproduction.layer_trace,
            candidate_reproduction.layer_trace,
            reference_source_report=reference_reproduction.report,
            candidate_source_report=candidate_reproduction.report,
        )
        _finish_executed(
            context,
            serialize_pass3(discovery_pass3),
            "tier1_bounded_digest",
        )
        if discovery_pass3.status != Pass3Status.OBSERVABLE_MISMATCH_FOUND:
            return _blocked_report(
                context,
                blocker="pass3a_discovery",
                verdict=CustomerVerdict.REGRESSION,
                reason_code="selected_token_mismatch",
                policy_reason="regression_retained_layer_localization_incomplete",
                identities=identities,
                pass1=pass1,
                warning="The selected-token regression was proven, but no layer mismatch was localized in captured coverage.",
            )
        target_layer = discovery_pass3.earliest_observable_suspect_layer
        if target_layer is None:
            raise RealExecutionError("Pass 3A did not select a bounded target layer")

        context.start_stage("bounded_recapture")
        authoritative = _new_generation(
            adapter, setup, intra_layer=target_layer
        )
        if authoritative.pass2.status != Pass2Status.REPRODUCTION_VERIFIED:
            raise RealExecutionError("authoritative recapture did not reproduce Pass 2")
        discovery_localization = pass1.localization
        authoritative_localization = authoritative.pass1.localization
        if (
            discovery_localization is None
            or authoritative_localization is None
            or authoritative_localization.generated_token_step
            != discovery_localization.generated_token_step
            or authoritative_localization.reference_selected_token_id
            != discovery_localization.reference_selected_token_id
            or authoritative_localization.candidate_selected_token_id
            != discovery_localization.candidate_selected_token_id
        ):
            raise RealExecutionError(
                "authoritative recapture selected-token boundary drifted"
            )
        artifact_sets = {
            _artifact_set_id(reference_original.report),
            _artifact_set_id(candidate_original.report),
            _artifact_set_id(reference_reproduction.report),
            _artifact_set_id(candidate_reproduction.report),
            _artifact_set_id(authoritative.reference_original.report),
            _artifact_set_id(authoritative.candidate_original.report),
            _artifact_set_id(authoritative.reference_reproduction.report),
            _artifact_set_id(authoritative.candidate_reproduction.report),
        }
        if len(artifact_sets) != 8:
            raise RealExecutionError(
                "authoritative recapture reused an earlier artifact set"
            )
        assert authoritative.reference_reproduction.layer_trace is not None
        assert authoritative.candidate_reproduction.layer_trace is not None
        recapture_ref = {
            "reference_original": authoritative.reference_original.report.identity.run_report_sha256,
            "candidate_original": authoritative.candidate_original.report.identity.run_report_sha256,
            "reference_reproduction": authoritative.reference_reproduction.report.identity.run_report_sha256,
            "candidate_reproduction": authoritative.candidate_reproduction.report.identity.run_report_sha256,
            "reference_trace": authoritative.reference_reproduction.layer_trace.identity.trace_sha256,
            "candidate_trace": authoritative.candidate_reproduction.layer_trace.identity.trace_sha256,
        }
        _finish_executed(context, recapture_ref, "source_bound_real_recapture")

        context.start_stage("pass3b_authoritative_localization")
        authoritative_pass2_artifact = CanonicalPass2Artifact.from_result(
            authoritative.pass2
        )
        authoritative_pass3 = run_coverage_scoped_layer_localization(
            authoritative.pass2,
            authoritative_pass2_artifact,
            authoritative.reference_reproduction.layer_trace,
            authoritative.candidate_reproduction.layer_trace,
            reference_source_report=authoritative.reference_reproduction.report,
            candidate_source_report=authoritative.candidate_reproduction.report,
        )
        _finish_executed(
            context,
            serialize_pass3(authoritative_pass3),
            "tier1_bounded_digest",
        )
        if (
            authoritative_pass3.status != Pass3Status.OBSERVABLE_MISMATCH_FOUND
            or authoritative_pass3.earliest_observable_suspect_layer != target_layer
        ):
            return _blocked_report(
                context,
                blocker="pass3b_authoritative_localization",
                verdict=CustomerVerdict.REGRESSION,
                reason_code="selected_token_mismatch",
                policy_reason="regression_retained_recapture_inconsistent",
                identities=identities,
                pass1=pass1,
                warning="The selected-token regression was proven, but authoritative layer recapture drifted.",
            )

        context.start_stage("pass4_intra_layer_localization")
        pass4 = run_coverage_scoped_intra_layer_localization(
            discovery_pass3,
            CanonicalPass3Artifact.from_result(discovery_pass3),
            authoritative_pass3,
            CanonicalPass3Artifact.from_result(authoritative_pass3),
            authoritative_pass2_artifact,
            discovery_reference_report=reference_reproduction.report,
            discovery_candidate_report=candidate_reproduction.report,
            discovery_reference_trace=reference_reproduction.layer_trace,
            discovery_candidate_trace=candidate_reproduction.layer_trace,
            authoritative_reference_report=authoritative.reference_reproduction.report,
            authoritative_candidate_report=authoritative.candidate_reproduction.report,
            authoritative_reference_trace=authoritative.reference_reproduction.layer_trace,
            authoritative_candidate_trace=authoritative.candidate_reproduction.layer_trace,
            discovery_pass2_artifact=discovery_pass2_artifact,
        )
        _finish_executed(
            context,
            serialize_pass4(pass4),
            "tier1_bounded_digest",
        )
        layer_interval = (
            authoritative_pass3.suspect_interval.notation
            if authoritative_pass3.suspect_interval is not None
            else None
        )
        intra_interval = (
            pass4.suspect_interval.notation
            if pass4.suspect_interval is not None
            else None
        )
        layer_coverage = authoritative_pass3.coverage
        intra_coverage = pass4.coverage
        coverage = {
            "scope": "selected_tokens_and_bounded_llama_decode",
            "common_layers": list(
                dict.fromkeys(item.layer_index for item in layer_coverage.common_captured)
            ) if layer_coverage is not None else [],
            "missing_reference_layers": list(
                dict.fromkeys(item.coordinate.layer_index for item in layer_coverage.reference_missing)
            ) if layer_coverage is not None else [],
            "missing_candidate_layers": list(
                dict.fromkeys(item.coordinate.layer_index for item in layer_coverage.candidate_missing)
            ) if layer_coverage is not None else [],
            "common_intra_layer_stages": list(
                dict.fromkeys(item.stage_id for item in intra_coverage.common_captured)
            ) if intra_coverage is not None else [],
            "missing_reference_intra_layer_stages": list(
                dict.fromkeys(item.coordinate.stage_id for item in intra_coverage.reference_missing)
            ) if intra_coverage is not None else [],
            "missing_candidate_intra_layer_stages": list(
                dict.fromkeys(item.coordinate.stage_id for item in intra_coverage.candidate_missing)
            ) if intra_coverage is not None else [],
        }
        successful_pass4 = {
            Pass4Status.OBSERVABLE_INTRA_LAYER_MISMATCH_FOUND,
            Pass4Status.MISMATCH_BOUNDED_TO_INHERITED_CLOSING_BOUNDARY,
        }
        warning = []
        if pass4.status not in successful_pass4:
            warning.append(
                "The selected-token regression was proven, but intra-layer localization remained incomplete."
            )
        return _report(
            context,
            verdict=CustomerVerdict.REGRESSION,
            reason_codes=["selected_token_mismatch"],
            policy_reason="selected_token_regression",
            identities=identities,
            pass1=pass1,
            layer_interval=layer_interval,
            intra_interval=intra_interval,
            coverage=coverage,
            evidence_tier="tier1_bounded_digest",
            evidence_ceiling="bounded_suspect_interval",
            next_action={
                "code": "inspect_bounded_suspect_interval",
                "summary": "Inspect the reported bounded layer and intra-layer suspect intervals.",
            },
            warnings=warning,
        )
    except Exception as raw_exc:
        exc = (
            raw_exc
            if isinstance(raw_exc, RealExecutionError)
            else RealExecutionError(
                "real verification evidence was malformed or inconsistent",
                classification=(
                    "unsupported"
                    if raw_exc.__class__.__name__
                    in {"ModelProfileError", "ProvenanceError"}
                    else "harness_error"
                ),
            )
        )
        active = context.state_machine.active_stage
        if active is None:
            raise
        if exc.execution is not None and signal_exit is not None:
            signal_code = exc.execution.signal_exit_code
            if signal_code is not None:
                signal_exit.append(signal_code)
        context.finish_stage(
            StageState.FAILED,
            failure_class=exc.classification,
            reason="The real execution stage failed closed.",
        )
        proven_regression = (
            pass1 is not None and pass1.status == Pass1Status.FIRST_MISMATCH_FOUND
        )
        if proven_regression:
            verdict = CustomerVerdict.REGRESSION
            reason_code = "selected_token_mismatch"
            policy_reason = "regression_retained_after_localization_failure"
        elif exc.classification == "unsupported":
            verdict = CustomerVerdict.UNSUPPORTED
            reason_code = "unsupported_comparison"
            policy_reason = "real_execution_unsupported"
        elif exc.classification == "inconclusive":
            verdict = CustomerVerdict.INCONCLUSIVE
            reason_code = "partial_evidence_inconclusive"
            policy_reason = "real_execution_inconclusive"
        else:
            verdict = CustomerVerdict.HARNESS_ERROR
            reason_code = "malformed_artifact"
            policy_reason = "real_execution_integrity_failure"
        return _blocked_report(
            context,
            blocker=active,
            verdict=verdict,
            reason_code=reason_code,
            policy_reason=policy_reason,
            identities=identities,
            pass1=pass1,
            warning=str(exc),
        )
