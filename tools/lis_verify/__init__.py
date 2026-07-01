"""LIS verification core — P1 Pass 0, Pass 1, and Pass 2.

Model-free calibration gating, exact selected-token localization, and
source-bound mismatch-boundary reproduction. Numeric comparison and runtime
checkpoint confirmation are intentionally out of scope.
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
from .pass1 import (
    locate_first_selected_token_mismatch,
    run_token_localization,
    runtime_checkpoint_step_for_generated,
)
from .pass1_artifact import (
    serialize as serialize_pass1,
    to_json as pass1_to_json,
)
from .pass1_inputs import (
    CanonicalRunReport,
    build_source_binding,
    canonical_json_sha256,
    strict_json_loads,
)
from .pass1_model import (
    DEFAULT_EMBEDDED_PREFIX_CAP,
    MismatchKind,
    Pass0SourceBinding,
    Pass1ReasonCode,
    Pass1Result,
    Pass1Status,
    Pass2Disposition,
    PrefixAvailability,
    SelectedTokenEvidenceLevel,
)
from .pass1_report_mapping import map_pass1_reason, map_pass1_status
from .pass2 import run_prefix_policy_reproduction
from .pass2_artifact import (
    serialize as serialize_pass2,
    to_json as pass2_to_json,
)
from .pass2_model import (
    COMPUTED_STEP_EVIDENCE,
    TRACE_STEP_EVIDENCE,
    THREAD_COUNT_CAVEAT,
    Pass2ReasonCode,
    Pass2Result,
    Pass2Status,
    Pass3Disposition,
    ReproductionEvidenceTier,
)
from .pass2_report_mapping import map_pass2_reason, map_pass2_status
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
    "CanonicalRunReport",
    "build_source_binding",
    "canonical_json_sha256",
    "strict_json_loads",
    "Pass0SourceBinding",
    "Pass1Result",
    "Pass1Status",
    "MismatchKind",
    "SelectedTokenEvidenceLevel",
    "Pass2Disposition",
    "PrefixAvailability",
    "Pass1ReasonCode",
    "DEFAULT_EMBEDDED_PREFIX_CAP",
    "locate_first_selected_token_mismatch",
    "runtime_checkpoint_step_for_generated",
    "run_token_localization",
    "serialize_pass1",
    "pass1_to_json",
    "map_pass1_reason",
    "map_pass1_status",
    "Pass2Result",
    "Pass2Status",
    "Pass2ReasonCode",
    "ReproductionEvidenceTier",
    "Pass3Disposition",
    "COMPUTED_STEP_EVIDENCE",
    "TRACE_STEP_EVIDENCE",
    "THREAD_COUNT_CAVEAT",
    "run_prefix_policy_reproduction",
    "serialize_pass2",
    "pass2_to_json",
    "map_pass2_reason",
    "map_pass2_status",
]
