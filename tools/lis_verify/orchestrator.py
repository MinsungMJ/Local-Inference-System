"""Report-first lifecycle coordinator for M1 and later mode adapters."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Callable

from .aggregation import policy_result_for
from .ledger import AppendOnlyLedger
from .product_contract import (
    CLI_MODES,
    MAX_STAGE_TIMEOUT_SECONDS,
    SHA256_RE,
    CustomerVerdict,
    StageState,
    WorkflowClassification,
    canonical_json_bytes,
)
from .report_artifact import (
    PreparedReportBundle,
    prepare_report_bundle,
    publish_report_bundle,
)
from .report_model import VerificationReport
from .state_machine import StageMachine
from .summary import render_markdown, render_terminal
from .workspace import AttemptWorkspace, CleanupObservation


class OrchestrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class AcceptanceManifest:
    source_revision: str
    source_tree_sha256: str
    dependency_sha256: str
    commands_sha256: str
    clean_state_observed: bool

    def validate(self) -> None:
        if (
            not isinstance(self.source_revision, str)
            or len(self.source_revision) != 40
            or any(
                character not in "0123456789abcdef"
                for character in self.source_revision
            )
        ):
            raise OrchestrationError("acceptance source revision is not frozen")
        for value in (
            self.source_tree_sha256,
            self.dependency_sha256,
            self.commands_sha256,
        ):
            if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
                raise OrchestrationError("acceptance manifest digest is not canonical")
        if self.clean_state_observed is not True:
            raise OrchestrationError("acceptance requires observed clean state")


@dataclass(frozen=True)
class CommandRequest:
    mode: str
    output_root: Path
    require_supported: bool = False
    debug_retain: bool = False
    stage_timeout_seconds: int = 1800
    verbose: bool = False
    model: Path | None = None
    reference_bin: Path | None = None
    candidate_bin: Path | None = None
    workflow: WorkflowClassification = WorkflowClassification.DEVELOPMENT_DEBUGGING
    acceptance_manifest: AcceptanceManifest | None = None


@dataclass
class RunContext:
    request: CommandRequest
    workspace: AttemptWorkspace
    ledger: AppendOnlyLedger
    state_machine: StageMachine

    def start_stage(self, stage: str) -> None:
        self.state_machine.start(stage)
        self.ledger.start_stage(stage)

    def finish_stage(self, state: StageState, **fields: Any) -> None:
        active = self.state_machine.active_stage
        result = self.state_machine.finish(state, **fields)
        assert active == result.name
        self.ledger.finish_stage(result.name, result.state)


ModeRunner = Callable[[RunContext], dict[str, Any]]


@dataclass(frozen=True)
class OrchestrationResult:
    report: VerificationReport
    report_path: Path
    summary_path: Path
    terminal_summary: str
    exit_code: int


def _cleanup_stage(observation: CleanupObservation) -> dict[str, Any]:
    digest = "sha256:" + hashlib.sha256(
        canonical_json_bytes(observation.to_dict())
    ).hexdigest()
    return {
        "name": "cleanup",
        "state": "executed",
        "result_ref": digest,
        "evidence_tier": (
            "explicit_debug_retention"
            if observation.retained_debug
            else "observed_cleanup"
        ),
        "failure_class": None,
        "reason": None,
        "blocker": None,
    }


def _validate_request(request: CommandRequest) -> None:
    if not isinstance(request.workflow, WorkflowClassification):
        raise OrchestrationError("workflow classification is not frozen")
    if not isinstance(request.require_supported, bool) or not isinstance(
        request.debug_retain, bool
    ):
        raise OrchestrationError("command policy flags must be boolean")
    if request.mode not in CLI_MODES:
        raise OrchestrationError("command mode is not frozen")
    if (
        isinstance(request.stage_timeout_seconds, bool)
        or request.stage_timeout_seconds <= 0
        or request.stage_timeout_seconds > MAX_STAGE_TIMEOUT_SECONDS
    ):
        raise OrchestrationError("stage timeout is outside the frozen bound")
    required = set(CLI_MODES[request.mode]["required_options"])
    supplied = {
        name
        for name, value in (
            ("model", request.model),
            ("reference_bin", request.reference_bin),
            ("candidate_bin", request.candidate_bin),
        )
        if value is not None
    }
    if not required.issubset(supplied):
        raise OrchestrationError("command is missing a required mode input")
    if supplied - required:
        raise OrchestrationError("command contains an input forbidden for its mode")
    if request.workflow == WorkflowClassification.VERIFICATION_ACCEPTANCE:
        if request.acceptance_manifest is None:
            raise OrchestrationError("acceptance requires a frozen manifest")
        request.acceptance_manifest.validate()
    elif request.acceptance_manifest is not None:
        raise OrchestrationError("debugging request cannot consume acceptance authority")


def _finalize_report(
    raw: dict[str, Any],
    *,
    request: CommandRequest,
    workspace: AttemptWorkspace,
    cleanup: CleanupObservation,
) -> VerificationReport:
    value = VerificationReport.from_dict(raw).to_dict()
    value["attempt"] = {
        "id": workspace.attempt_id,
        "workflow_classification": request.workflow.value,
    }
    value["command"] = {
        "mode": request.mode,
        "require_supported": request.require_supported,
        "output_path_redacted": True,
    }
    verdict = CustomerVerdict(value["verdict"])
    value["policy_result"] = policy_result_for(
        verdict,
        require_supported=request.require_supported,
        reason=value["policy_result"]["reason"],
    )
    value["cleanup"] = cleanup.to_dict()
    value["stages"][-1] = _cleanup_stage(cleanup)
    return VerificationReport.from_dict(value)


def run_orchestration(request: CommandRequest, runner: ModeRunner) -> OrchestrationResult:
    _validate_request(request)
    workspace = AttemptWorkspace.create(request.output_root)
    ledger = AppendOnlyLedger(
        workspace.ledger_path,
        attempt_id=workspace.attempt_id,
        workflow=request.workflow,
    )
    bundle: PreparedReportBundle | None = None
    ledger_finished = False
    cleanup: CleanupObservation | None = None
    context: RunContext | None = None
    try:
        ledger.start_attempt(request.mode)
        context = RunContext(request, workspace, ledger, StageMachine())
        raw = runner(context)
        if len(context.state_machine.results) != len(raw.get("stages", ())) - 1:
            raise OrchestrationError(
                "mode runner did not drive every pre-cleanup canonical stage"
            )
        expected = [item.to_dict() for item in context.state_machine.results]
        if expected != raw["stages"][:-1] or context.state_machine.next_stage != "cleanup":
            raise OrchestrationError("mode runner stage evidence disagrees with state machine")
        cleanup = workspace.cleanup_runtime(debug_retain=request.debug_retain)
        cleanup_result = _cleanup_stage(cleanup)
        context.start_stage("cleanup")
        context.finish_stage(
            StageState.EXECUTED,
            result_ref=cleanup_result["result_ref"],
            evidence_tier=cleanup_result["evidence_tier"],
        )
        context.state_machine.finalize()
        report = _finalize_report(
            raw,
            request=request,
            workspace=workspace,
            cleanup=cleanup,
        )
        markdown = render_markdown(report)
        terminal = render_terminal(report)
        bundle = prepare_report_bundle(
            report,
            markdown,
            report_path=workspace.report_path,
            summary_path=workspace.summary_path,
        )
        ledger.observe_cleanup(cleanup.residue_status)
        ledger.finish_attempt(report.verdict)
        ledger_finished = True
        publish_report_bundle(bundle)
        return OrchestrationResult(
            report=report,
            report_path=workspace.report_path,
            summary_path=workspace.summary_path,
            terminal_summary=terminal,
            exit_code=report.exit_code,
        )
    except Exception as exc:
        if bundle is not None:
            bundle.discard()
        if cleanup is None:
            cleanup = workspace.cleanup_runtime(debug_retain=request.debug_retain)
        if not ledger_finished:
            try:
                if context is not None and context.state_machine.active_stage is not None:
                    context.finish_stage(
                        StageState.FAILED,
                        failure_class="runner_failure",
                        reason="mode runner failed before producing a canonical result",
                    )
                if not ledger.cleanup_recorded:
                    ledger.observe_cleanup(cleanup.residue_status)
                ledger.finish_attempt(CustomerVerdict.HARNESS_ERROR)
            except Exception:
                ledger.abort()
        raise OrchestrationError("verification orchestration failed closed") from exc
