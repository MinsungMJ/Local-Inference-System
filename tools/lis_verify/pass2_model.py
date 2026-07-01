"""Typed data model for P1 Pass 2 prefix/policy reproduction.

Pass 2 records source-bound token-prefix and execution-boundary evidence.  It
does not carry tensor values, logits, numeric checkpoint comparisons, or a
confirmed-divergence verdict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .model import (
    ComparisonEligibility,
    ComparisonMode,
    Pass0Verdict,
    VerdictStrengthLimit,
)
from .pass1_model import Pass1Status, Pass2Disposition


SCHEMA = "lis.execution_artifact/v1"
KIND = "prefix_policy_reproduction"
CONTRACT_VERSION = "differential_verification_contract_v1"
COMPUTED_STEP_EVIDENCE = "computed_from_pass1_step_mapping"
TRACE_STEP_EVIDENCE = "corroborated_by_trace_artifact"
THREAD_COUNT_CAVEAT = "thread_count_gt_1_determinism_caveat"
REPRODUCTION_VERIFIED_SEMANTICS = (
    "When reproduction_evidence_tier is not independent_rerun_verified, "
    "reproduction_verified means the available boundary evidence is "
    "internally consistent and sufficient for the declared downstream "
    "disposition. It does not mean a fresh independent rerun reproduced "
    "the prefix."
)


class Pass2Status(str, Enum):
    REPRODUCTION_VERIFIED = "reproduction_verified"
    COMPARISON_BLOCKED_BY_PASS0 = "comparison_blocked_by_pass0"
    TOKEN_LOCALIZATION_NOT_AVAILABLE = "token_localization_not_available"
    NO_MISMATCH_TO_REPRODUCE = "no_mismatch_to_reproduce"
    SOURCE_BINDING_INCONSISTENT = "source_binding_inconsistent"
    PREFIX_MATERIAL_UNAVAILABLE = "prefix_material_unavailable"
    PREFIX_REPRODUCTION_FAILED = "prefix_reproduction_failed"
    DECODE_POLICY_REPRODUCTION_FAILED = (
        "decode_policy_reproduction_failed"
    )
    CHECKPOINT_STEP_MAPPING_MISMATCH = (
        "checkpoint_step_mapping_mismatch"
    )
    CONTEXT_POSITION_MISMATCH = "context_position_mismatch"
    UNSUPPORTED_REPRODUCTION_MODE = "unsupported_reproduction_mode"
    INCONCLUSIVE = "inconclusive"


class Pass2ReasonCode(str, Enum):
    SOURCE_BINDING_INCONSISTENT = "pass2.source_binding_inconsistent"
    PASS1_STATUS_NOT_REPRODUCIBLE = (
        "pass2.pass1_status_not_reproducible"
    )
    PREFIX_MATERIAL_UNAVAILABLE = "pass2.prefix_material_unavailable"
    PREFIX_DIGEST_MISMATCH = "pass2.prefix_digest_mismatch"
    PREFIX_TOKEN_MISMATCH = "pass2.prefix_token_mismatch"
    DECODE_POLICY_REPRODUCTION_FAILED = (
        "pass2.decode_policy_reproduction_failed"
    )
    CHECKPOINT_STEP_MAPPING_MISMATCH = (
        "pass2.checkpoint_step_mapping_mismatch"
    )
    CONTEXT_POSITION_MISMATCH = "pass2.context_position_mismatch"
    UNSUPPORTED_REPRODUCTION_MODE = (
        "pass2.unsupported_reproduction_mode"
    )
    REPRODUCTION_ARTIFACT_MALFORMED = (
        "pass2.reproduction_artifact_malformed"
    )
    VERDICT_STRENGTH_LIMIT_BLOCKS_REPRODUCTION = (
        "pass2.verdict_strength_limit_blocks_reproduction"
    )


class ReproductionEvidenceTier(str, Enum):
    INDEPENDENT_RERUN_VERIFIED = "independent_rerun_verified"
    ORIGINAL_PAIR_BOUNDARY_CONSISTENT = (
        "original_pair_boundary_consistent"
    )
    REPRODUCTION_REQUEST_ONLY = "reproduction_request_only"


class Pass3Disposition(str, Enum):
    READY = "ready"
    NOT_REQUIRED = "not_required"
    BLOCKED_BY_PASS0 = "blocked_by_pass0"
    BLOCKED_BY_PASS1_EVIDENCE = "blocked_by_pass1_evidence"
    BLOCKED_BY_REPRODUCTION = "blocked_by_reproduction"


@dataclass(frozen=True)
class Pass2SourceBinding:
    pass1_artifact_sha256: str
    reference_original_run_report_sha256: Optional[str]
    candidate_original_run_report_sha256: Optional[str]
    reference_original_verified: bool
    candidate_original_verified: bool
    reference_reproduction_sha256: Optional[str] = None
    candidate_reproduction_sha256: Optional[str] = None
    reproduction_verified: Optional[bool] = None
    verified: bool = False


@dataclass(frozen=True)
class TargetCheckpoint:
    generated_token_step: Optional[int]
    expected_runtime_checkpoint_step: Optional[int]
    matched_generated_prefix_length: Optional[int]


@dataclass(frozen=True)
class PromptGateResult:
    status: str
    inherited_prompt_identity_evidence: str


@dataclass(frozen=True)
class PrefixGateResult:
    status: str
    expected_token_count: Optional[int] = None
    expected_sha256: Optional[str] = None
    verified_sides: tuple[str, ...] = ()
    failed_side: Optional[str] = None
    first_diff_index: Optional[int] = None
    mismatch_kind: Optional[str] = None


@dataclass(frozen=True)
class RunBuildEvidence:
    role: str
    binary_fingerprint: str


@dataclass(frozen=True)
class PolicyGateResult:
    status: str
    build_continuity_verified: bool
    inherited_calibration_sha256: str
    runs: tuple[RunBuildEvidence, ...] = ()


@dataclass(frozen=True)
class CheckpointStepGateResult:
    status: str
    generated_token_step: Optional[int]
    pass1_runtime_checkpoint_step: Optional[int]
    expected_runtime_checkpoint_step: Optional[int]
    evidence: str
    trace_runtime_checkpoint_step: Optional[int] = None
    materialized_checkpoint_artifact_verified: bool = False


@dataclass(frozen=True)
class RunContextEvidence:
    role: str
    prompt_token_count: int
    context_position: int
    batch_size: int
    sequence_index: int
    thread_count: int


@dataclass(frozen=True)
class ContextGateResult:
    status: str
    runs: tuple[RunContextEvidence, ...] = ()


@dataclass(frozen=True)
class LocalizationReference:
    sha256: str
    pass1_status: str
    generated_token_step: Optional[int]
    runtime_checkpoint_step: Optional[int]
    matched_generated_prefix_length: Optional[int]
    prefix_availability: str
    pass2_disposition: str
    verdict_strength_limit: str
    canonical_json: str = field(repr=False)


@dataclass(frozen=True)
class Pass2Result:
    comparison_mode: ComparisonMode
    pass0_verdict: Pass0Verdict
    comparison_eligibility: ComparisonEligibility
    pass1_status: Pass1Status
    pass1_pass2_disposition: Pass2Disposition
    source_binding: Pass2SourceBinding
    source_binding_verified: bool
    status: Pass2Status
    reproduction_evidence_tier: ReproductionEvidenceTier
    target: TargetCheckpoint
    prompt_reproduction: PromptGateResult
    prefix_reproduction: PrefixGateResult
    policy_reproduction: PolicyGateResult
    checkpoint_step_reproduction: CheckpointStepGateResult
    context_reproduction: ContextGateResult
    pass3_disposition: Pass3Disposition
    localization_ref: LocalizationReference
    verdict_strength_limit: VerdictStrengthLimit
    reason_codes: tuple[Pass2ReasonCode, ...] = ()
    inherited_pass1_reason_codes: tuple[str, ...] = ()
    inherited_pass0_reason_codes: tuple[str, ...] = ()
    blocking_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    schema: str = SCHEMA
    kind: str = KIND
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self):
        request_only_verified = (
            self.reproduction_evidence_tier
            == ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY
            and self.status == Pass2Status.REPRODUCTION_VERIFIED
        )
        if request_only_verified:
            raise ValueError(
                "reproduction_request_only cannot be reproduction_verified"
            )
        ready = self.pass3_disposition == Pass3Disposition.READY
        verified = self.status == Pass2Status.REPRODUCTION_VERIFIED
        if ready != verified:
            raise ValueError(
                "pass3 ready must exactly match reproduction_verified"
            )
