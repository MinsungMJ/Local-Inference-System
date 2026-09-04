"""Explicit final-state machine for the frozen canonical stage graph."""

from __future__ import annotations

from dataclasses import dataclass

from .product_contract import CANONICAL_STAGES, STAGE_DEPENDENCIES, StageState
from .report_model import StageResult


class StageTransitionError(ValueError):
    pass


@dataclass
class StageMachine:
    _active: str | None = None

    def __post_init__(self) -> None:
        self._results: list[StageResult] = []

    @property
    def next_stage(self) -> str | None:
        if len(self._results) == len(CANONICAL_STAGES):
            return None
        return CANONICAL_STAGES[len(self._results)]

    @property
    def results(self) -> tuple[StageResult, ...]:
        return tuple(self._results)

    @property
    def active_stage(self) -> str | None:
        return self._active

    def start(self, stage: str) -> None:
        if self._active is not None:
            raise StageTransitionError("a stage is already running")
        if stage != self.next_stage:
            raise StageTransitionError("stages must start in canonical order")
        self._active = stage

    def finish(
        self,
        state: StageState,
        *,
        result_ref: str | None = None,
        evidence_tier: str | None = None,
        failure_class: str | None = None,
        reason: str | None = None,
        blocker: str | None = None,
    ) -> StageResult:
        if self._active is None:
            raise StageTransitionError("no stage is running")
        stage = self._active
        if state == StageState.EXECUTED:
            if result_ref is None or evidence_tier is None:
                raise StageTransitionError("executed stage requires result evidence")
            if any(value is not None for value in (failure_class, reason, blocker)):
                raise StageTransitionError("executed stage carries failure fields")
            if stage != "aggregation":
                prior = {item.name: item.state for item in self._results}
                for dependency in STAGE_DEPENDENCIES[stage]:
                    if prior.get(dependency) != StageState.EXECUTED:
                        raise StageTransitionError(
                            "executed stage has a non-executed dependency"
                        )
        elif state == StageState.NOT_APPLICABLE:
            if not reason or any(
                value is not None
                for value in (result_ref, evidence_tier, failure_class, blocker)
            ):
                raise StageTransitionError("not_applicable fields are incoherent")
        elif state == StageState.BLOCKED:
            if not reason or blocker not in CANONICAL_STAGES:
                raise StageTransitionError("blocked stage requires reason and blocker")
            if blocker not in {item.name for item in self._results}:
                raise StageTransitionError("blocked stage requires a prior blocker")
            if any(
                value is not None
                for value in (result_ref, evidence_tier, failure_class)
            ):
                raise StageTransitionError("blocked stage carries result evidence")
        elif state == StageState.FAILED:
            if not reason or not failure_class:
                raise StageTransitionError("failed stage requires class and reason")
            if any(value is not None for value in (result_ref, evidence_tier, blocker)):
                raise StageTransitionError("failed stage carries result evidence")
        else:
            raise StageTransitionError("unknown final stage state")

        result = StageResult(
            name=stage,
            state=state,
            result_ref=result_ref,
            evidence_tier=evidence_tier,
            failure_class=failure_class,
            reason=reason,
            blocker=blocker,
        )
        self._results.append(result)
        self._active = None
        return result

    def finalize(self) -> tuple[StageResult, ...]:
        if self._active is not None or self.next_stage is not None:
            raise StageTransitionError("state machine is not terminal")
        return self.results
