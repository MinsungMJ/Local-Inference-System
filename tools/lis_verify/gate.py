"""Pass 0 -> Pass 1 gating interface.

Pass 0 ships this dataclass and ``build_gate``; it does not implement Pass 1.
Pass 1 must treat the gate as authoritative and must not re-infer calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

from .model import (
    CalibrationPreflightArtifact,
    ComparisonEligibility,
    OracleEligibility,
    Pass0Verdict,
    VerdictStrengthLimit,
)


@dataclass
class Pass0GateDecision:
    proceed: bool
    eligibility: ComparisonEligibility
    verdict: Pass0Verdict
    verdict_strength_limit: VerdictStrengthLimit
    oracle_eligibility: OracleEligibility
    blocking_reasons: list
    artifact: CalibrationPreflightArtifact


def build_gate(artifact: CalibrationPreflightArtifact) -> Pass0GateDecision:
    return Pass0GateDecision(
        proceed=artifact.pass0_verdict != Pass0Verdict.COMPARISON_BLOCKED,
        eligibility=artifact.comparison_eligibility,
        verdict=artifact.pass0_verdict,
        verdict_strength_limit=artifact.verdict_strength_limit,
        oracle_eligibility=artifact.oracle_eligibility,
        blocking_reasons=list(artifact.blocking_reasons),
        artifact=artifact,
    )
