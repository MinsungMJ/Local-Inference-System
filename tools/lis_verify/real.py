"""Customer adapters for real backend and runtime verification."""

from __future__ import annotations

from dataclasses import replace

from .execution import BoundedExecutor
from .orchestrator import CommandRequest, OrchestrationResult, run_orchestration
from .real_pipeline import run_real_pipeline


def run_real(
    request: CommandRequest,
    *,
    executor: BoundedExecutor | None = None,
) -> OrchestrationResult:
    if request.mode not in {"backend", "runtime"}:
        raise ValueError("real adapter requires backend or runtime mode")
    signal_exit: list[int] = []
    result = run_orchestration(
        request,
        lambda context: run_real_pipeline(
            context,
            executor=executor,
            signal_exit=signal_exit,
        ),
    )
    if signal_exit:
        return replace(result, exit_code=signal_exit[0])
    return result


def run_backend(request: CommandRequest) -> OrchestrationResult:
    return run_real(request)


def run_runtime(request: CommandRequest) -> OrchestrationResult:
    return run_real(request)
