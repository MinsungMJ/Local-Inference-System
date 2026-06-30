"""LIS verification core — P1 Pass 0 (Calibration Preflight).

Model-free Python package that decides whether two LIS executions are
semantically comparable before any token localization or numeric comparison.
See ``docs/calibration_preflight.md`` and the approved implementation plan.
"""

from __future__ import annotations

from .artifact import serialize, to_json
from .build_profile import BuildCalibrationProfile, default_build_profile
from .gate import Pass0GateDecision, build_gate
from .inputs import DecodeTraceSummary, PreflightInputs, RunSide
from .model import (
    CalibrationPreflightArtifact,
    CalibrationReasonCode,
    ComparisonEligibility,
    ComparisonMode,
    ModeBSubmode,
    OracleScope,
    Pass0Verdict,
    PromptIdentityEvidence,
    VerdictStrengthLimit,
)
from .pass0 import run_calibration_preflight
from .report_mapping import map_block_reason

__all__ = [
    "BuildCalibrationProfile",
    "default_build_profile",
    "CalibrationPreflightArtifact",
    "CalibrationReasonCode",
    "ComparisonEligibility",
    "ComparisonMode",
    "ModeBSubmode",
    "OracleScope",
    "Pass0Verdict",
    "PromptIdentityEvidence",
    "VerdictStrengthLimit",
    "PreflightInputs",
    "RunSide",
    "DecodeTraceSummary",
    "Pass0GateDecision",
    "build_gate",
    "run_calibration_preflight",
    "serialize",
    "to_json",
    "map_block_reason",
]
