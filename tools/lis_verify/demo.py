"""Customer adapter for the model-free seeded demonstration."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import sys
from typing import Any

from .demo_pipeline import PLACEHOLDER_ATTEMPT_ID
from .execution import BoundedExecutor, ExecutionResult
from .orchestrator import (
    CommandRequest,
    OrchestrationResult,
    RunContext,
    run_orchestration,
)
from .product_contract import (
    CANONICAL_STAGES,
    EVIDENCE_NONCLAIMS,
    SIGNAL_EXIT_CODES,
    StageState,
    canonical_json_bytes,
)
from .report_artifact import load_report
from .report_model import VerificationReport


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _finish_from_stage(context: RunContext, raw: dict[str, Any]) -> None:
    state = StageState(raw["state"])
    context.finish_stage(
        state,
        result_ref=raw["result_ref"],
        evidence_tier=raw["evidence_tier"],
        failure_class=raw["failure_class"],
        reason=raw["reason"],
        blocker=raw["blocker"],
    )


def _drive_success(context: RunContext, report: VerificationReport) -> dict[str, Any]:
    raw = report.to_dict()
    expected_result = {
        "verdict": "REGRESSION",
        "mode": "demo",
        "token_status": "mismatch",
        "generated_token_step": 17,
        "layer_interval": "(4, 8]",
        "intra_layer_interval": "(rope_key_output, attention_scores]",
        "evidence_tier": "tier1_bounded_digest",
        "evidence_ceiling": "bounded_suspect_interval",
    }
    actual_result = {
        "verdict": raw["verdict"],
        "mode": raw["command"]["mode"],
        "token_status": raw["token_comparison"]["status"],
        "generated_token_step": (
            raw["token_comparison"]["first_mismatch"] or {}
        ).get("generated_token_step"),
        "layer_interval": raw["localization"]["layer_suspect_interval"],
        "intra_layer_interval": raw["localization"][
            "intra_layer_suspect_interval"
        ],
        "evidence_tier": raw["evidence"]["tier"],
        "evidence_ceiling": raw["evidence"]["ceiling"],
    }
    if actual_result != expected_result:
        raise ValueError("demo worker returned the wrong product result")
    stages = raw["stages"][:-1]
    if tuple(stage["name"] for stage in stages) != CANONICAL_STAGES[:-1]:
        raise ValueError("demo worker stage list is not canonical")
    if any(stage["state"] != "executed" for stage in stages):
        raise ValueError("demo worker did not execute every expected stage")
    _finish_from_stage(context, stages[0])
    for stage in stages[1:]:
        context.start_stage(stage["name"])
        _finish_from_stage(context, stage)
    return raw


def _unavailable_identities() -> dict[str, dict[str, str]]:
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
                    "domain": "lis.demo_unavailable_identity/v1",
                    "role": role,
                    "field": field,
                }
            )
            for field in fields
        }
        for role in ("reference", "candidate")
    }


def _drive_failure(
    context: RunContext,
    *,
    verdict: str,
    failure_class: str,
    reason_code: str,
    policy_reason: str,
    next_action_code: str,
    next_action_summary: str,
) -> dict[str, Any]:
    context.finish_stage(
        StageState.FAILED,
        failure_class=failure_class,
        reason="Seeded demo evidence did not complete.",
    )
    for stage in CANONICAL_STAGES[1:-2]:
        context.start_stage(stage)
        context.finish_stage(
            StageState.BLOCKED,
            reason="Seeded demo preflight did not complete.",
            blocker="preflight",
        )
    context.start_stage("aggregation")
    context.finish_stage(
        StageState.EXECUTED,
        result_ref=_digest(
            {
                "verdict": verdict,
                "failure_class": failure_class,
                "reason_code": reason_code,
            }
        ),
        evidence_tier="product_contract_v1",
    )
    stages = [item.to_dict() for item in context.state_machine.results]
    stages.append(
        {
            "name": "cleanup",
            "state": "executed",
            "result_ref": _digest({"state": "pending_orchestrator_cleanup"}),
            "evidence_tier": "pending_cleanup",
            "failure_class": None,
            "reason": None,
            "blocker": None,
        }
    )
    exit_code = 3 if verdict == "INCONCLUSIVE" else 2
    return {
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
        "verdict": verdict,
        "reason_codes": [reason_code],
        "policy_result": {
            "policy": "default",
            "satisfied": False,
            "exit_code": exit_code,
            "reason": policy_reason,
        },
        "identities": _unavailable_identities(),
        "token_comparison": {
            "status": "unavailable",
            "reference_observed_count": 0,
            "candidate_observed_count": 0,
            "first_mismatch": None,
        },
        "localization": {
            "layer_suspect_interval": None,
            "intra_layer_suspect_interval": None,
        },
        "coverage": {
            "scope": "seeded_demo_unavailable",
            "common_layers": [],
            "missing_reference_layers": [],
            "missing_candidate_layers": [],
            "common_intra_layer_stages": [],
            "missing_reference_intra_layer_stages": [],
            "missing_candidate_intra_layer_stages": [],
        },
        "numeric_confirmation": {
            "status": "not_performed",
            "confirmed_divergence_at_checkpoint": None,
            "confirmed_first_divergence": None,
        },
        "evidence": {
            "tier": "unavailable",
            "ceiling": "no_comparison",
            "nonclaims": {name: False for name in EVIDENCE_NONCLAIMS},
        },
        "stages": stages,
        "next_action": {
            "code": next_action_code,
            "summary": next_action_summary,
        },
        "warnings": [
            "No seeded mismatch result was accepted from partial evidence.",
            "Failure identities are deterministic unavailable sentinels.",
        ],
        "cleanup": {
            "status": "success",
            "residue_status": "none_observed",
            "observed": True,
            "retained_debug": False,
        },
    }


def _failure_parameters(execution: ExecutionResult) -> dict[str, str]:
    if execution.status == "timeout":
        return {
            "verdict": "INCONCLUSIVE",
            "failure_class": "stage_timeout",
            "reason_code": "verification_timeout",
            "policy_reason": "seeded_demo_timed_out",
            "next_action_code": "rerun_seeded_demo",
            "next_action_summary": "Rerun the seeded demo after checking local process capacity.",
        }
    if execution.status == "interrupted":
        return {
            "verdict": "INCONCLUSIVE",
            "failure_class": "handled_interruption",
            "reason_code": "verification_interrupted",
            "policy_reason": "seeded_demo_interrupted",
            "next_action_code": "rerun_seeded_demo",
            "next_action_summary": "Rerun the seeded demo when interruption is no longer required.",
        }
    return {
        "verdict": "HARNESS_ERROR",
        "failure_class": "demo_worker_failure",
        "reason_code": "malformed_artifact",
        "policy_reason": "seeded_demo_integrity_failure",
        "next_action_code": "repair_demo_installation",
        "next_action_summary": "Reinstall LIS Verify and rerun the seeded demo.",
    }


def _mode_runner(
    context: RunContext,
    executor: BoundedExecutor,
    signal_exit: list[int],
) -> dict[str, Any]:
    context.start_stage("preflight")
    worker_path = context.workspace.runtime_dir / "demo_worker_result.json"
    package_root = Path(__file__).resolve().parents[1]
    execution = executor.run(
        [
            sys.executable,
            "-m",
            "lis_verify.demo_worker",
            "--output",
            os.fspath(worker_path.absolute()),
        ],
        environment={
            "PATH": os.defpath,
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONPATH": os.fspath(package_root),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        timeout_seconds=context.request.stage_timeout_seconds,
    )
    if execution.ok and not execution.stdout and not execution.stderr:
        try:
            return _drive_success(context, load_report(worker_path))
        except Exception:
            execution = ExecutionResult(
                status="malformed_result",
                returncode=execution.returncode,
                stdout=execution.stdout,
                stderr=execution.stderr,
            )
    if execution.signal_exit_code in SIGNAL_EXIT_CODES.values():
        signal_exit.append(execution.signal_exit_code)
    return _drive_failure(context, **_failure_parameters(execution))


def run_demo(
    request: CommandRequest,
    *,
    executor: BoundedExecutor | None = None,
) -> OrchestrationResult:
    """Run the public demo with bounded worker and canonical failure results."""

    bounded = BoundedExecutor() if executor is None else executor
    signal_exit: list[int] = []
    result = run_orchestration(
        request,
        lambda context: _mode_runner(context, bounded, signal_exit),
    )
    if signal_exit:
        return replace(result, exit_code=signal_exit[0])
    return result
