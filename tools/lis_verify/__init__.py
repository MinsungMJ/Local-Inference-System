"""LIS verification core — P1 Pass 0 through coverage-scoped Pass 4.

Model-free calibration gating, exact selected-token localization, and
source-bound layer and intra-layer mismatch localization. Numeric confirmation
and root-cause identification are intentionally out of scope.
"""

from __future__ import annotations

__version__ = "0.1.0a1"

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
from .pass3 import run_coverage_scoped_layer_localization
from .pass3_artifact import serialize as serialize_pass3, to_json as pass3_to_json
from .pass3_inputs import CanonicalLayerTrace, CanonicalPass2Artifact
from .pass3_model import (
    AlignmentStatus,
    CheckpointCoordinate,
    CoverageState,
    Pass3DownstreamDisposition,
    Pass3ReasonCode,
    Pass3Result,
    Pass3Status,
    SummaryEvidenceLevel,
    SummaryFieldDisposition,
)
from .pass3_report_mapping import map_pass3_reason, map_pass3_status
from .pass4 import run_coverage_scoped_intra_layer_localization
from .pass4_artifact import (
    serialize as serialize_pass4,
    to_json as pass4_to_json,
)
from .pass4_contract import (
    IntraLayerCoordinate,
    Pass4Disposition,
    Pass4ReasonCode,
    Pass4Status,
)
from .pass4_model import (
    Pass3ParentEvidence,
    Pass4ClosingBoundaryDecision,
    Pass4Comparison,
    Pass4ComparisonDecision,
    Pass4CoverageAnalysis,
    Pass4EvidenceCeiling,
    Pass4LocalCoverageOutcome,
    Pass4Result,
    Pass4SourceBinding,
)
from .pass4_parent import CanonicalPass3Artifact
from .pass4_report_mapping import map_pass4_reason, map_pass4_status
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
    "Pass3Result",
    "Pass3Status",
    "Pass3ReasonCode",
    "Pass3DownstreamDisposition",
    "CheckpointCoordinate",
    "CoverageState",
    "AlignmentStatus",
    "SummaryFieldDisposition",
    "SummaryEvidenceLevel",
    "CanonicalPass2Artifact",
    "CanonicalLayerTrace",
    "run_coverage_scoped_layer_localization",
    "serialize_pass3",
    "pass3_to_json",
    "map_pass3_reason",
    "map_pass3_status",
    "Pass4Result",
    "Pass4Status",
    "Pass4Disposition",
    "Pass4ReasonCode",
    "Pass4ComparisonDecision",
    "Pass4LocalCoverageOutcome",
    "Pass4Comparison",
    "Pass4CoverageAnalysis",
    "Pass4SourceBinding",
    "Pass3ParentEvidence",
    "Pass4ClosingBoundaryDecision",
    "Pass4EvidenceCeiling",
    "IntraLayerCoordinate",
    "CanonicalPass3Artifact",
    "run_coverage_scoped_intra_layer_localization",
    "serialize_pass4",
    "pass4_to_json",
    "map_pass4_reason",
    "map_pass4_status",
]
