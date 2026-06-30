"""Typed data model for P1 Pass 1 token localization.

Pass 1 carries output-selection evidence only.  It deliberately has no fields
for numeric checkpoint confirmation or confirmed first divergence.
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


SCHEMA = "lis.execution_artifact/v1"
KIND = "token_localization"
CONTRACT_VERSION = "differential_verification_contract_v1"
DEFAULT_EMBEDDED_PREFIX_CAP = 64
EVIDENCE_SCOPE = "generated_selected_token_sequences"


class Pass1Status(str, Enum):
    COMPARISON_BLOCKED_BY_PASS0 = "comparison_blocked_by_pass0"
    TOKEN_EQUIVALENT_ON_OBSERVED_RANGE = "token_equivalent_on_observed_range"
    FIRST_MISMATCH_FOUND = "first_mismatch_found"
    INPUT_TOKEN_DIVERGENCE = "input_token_divergence"
    SELECTED_TOKEN_ARRAY_MISSING = "selected_token_array_missing"
    SELECTED_TOKEN_IDENTITY_UNVERIFIED = "selected_token_identity_unverified"
    UNSUPPORTED_COMPARISON = "unsupported_comparison"
    INCONCLUSIVE = "inconclusive"


class MismatchKind(str, Enum):
    TOKEN_ID_MISMATCH = "token_id_mismatch"
    LENGTH_MISMATCH_OR_EARLY_TERMINATION = (
        "length_mismatch_or_early_termination"
    )


class SelectedTokenEvidenceLevel(str, Enum):
    ARRAY_EXACT = "array_exact"
    DIGEST_ONLY = "digest_only"
    METADATA_ONLY = "metadata_only"
    MISSING = "missing"


class Pass2Disposition(str, Enum):
    READY = "ready"
    NOT_REQUIRED = "not_required"
    BLOCKED_BY_PASS0 = "blocked_by_pass0"
    BLOCKED_BY_EVIDENCE = "blocked_by_evidence"
    BLOCKED_BY_STRENGTH_LIMIT = "blocked_by_strength_limit"


class EvidenceCompleteness(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    INCOMPLETE = "incomplete"


class PrefixAvailability(str, Enum):
    EMBEDDED = "embedded"
    EXACT_SOURCE_REQUIRED = "exact_source_required"
    NOT_APPLICABLE = "not_applicable"


class Pass1ReasonCode(str, Enum):
    SELECTED_TOKEN_ARRAY_MISSING = "pass1.selected_token_array_missing"
    SELECTED_TOKEN_IDENTITY_UNVERIFIED = (
        "pass1.selected_token_identity_unverified"
    )
    SELECTED_TOKEN_METADATA_INCONSISTENT = (
        "pass1.selected_token_metadata_inconsistent"
    )
    GATE_RUN_IDENTITY_INCONSISTENT = (
        "pass1.gate_run_identity_inconsistent"
    )
    UNSUPPORTED_RUN_ARTIFACT = "pass1.unsupported_run_artifact"
    UNSUPPORTED_BATCH_SHAPE = "pass1.unsupported_batch_shape"


@dataclass(frozen=True)
class RunReportIdentity:
    run_report_sha256: str
    schema: Optional[str]
    kind: Optional[str]
    model_fingerprint: Optional[str] = None
    config_fingerprint: Optional[str] = None
    input_fingerprint: Optional[str] = None


@dataclass(frozen=True)
class Pass0SourceBinding:
    reference: RunReportIdentity
    candidate: RunReportIdentity


@dataclass(frozen=True)
class NormalizedDigest:
    algorithm: Optional[str]
    value: Optional[str]
    size_bytes: Optional[int]
    valid: bool
    matches_array: Optional[bool] = None


@dataclass(frozen=True)
class SelectedTokenSequence:
    role: str
    token_ids: Optional[tuple[int, ...]]
    source_locations: tuple[str, ...]
    declared_count: Optional[int]
    digest: NormalizedDigest
    stop_reason: Optional[str]
    generation_limit: Optional[int]
    evidence_level: SelectedTokenEvidenceLevel

    @property
    def observed_length(self) -> Optional[int]:
        return len(self.token_ids) if self.token_ids is not None else None


@dataclass(frozen=True)
class RunReportMetadata:
    role: str
    schema: Optional[str]
    kind: Optional[str]
    execution_status: Optional[str]
    model_family: Optional[str]
    model_fingerprint: Optional[str]
    config_fingerprint: Optional[str]
    input_fingerprint: Optional[str]
    batch_size: Optional[int]


@dataclass(frozen=True)
class Pass1RunInput:
    metadata: RunReportMetadata
    identity: RunReportIdentity
    selected_tokens: SelectedTokenSequence


@dataclass(frozen=True)
class TokenLocalization:
    generated_token_step: Optional[int]
    runtime_checkpoint_step: Optional[int]
    reference_selected_token_id: Optional[int]
    candidate_selected_token_id: Optional[int]
    matched_generated_prefix_length: int
    observed_reference_generated_length: int
    observed_candidate_generated_length: int
    mismatch_kind: Optional[MismatchKind]


@dataclass(frozen=True)
class PrefixForReproduction:
    exact_token_ids: tuple[int, ...]
    availability: PrefixAvailability
    prefix_start_generated_step: int
    prefix_end_generated_step_exclusive: int
    token_count: int
    sha256: str


@dataclass(frozen=True)
class CompatibilityResult:
    comparison_blocked: bool
    boundary: Optional[str]
    model_family_state: str
    config_state: str
    prompt_identity_evidence: str


@dataclass(frozen=True)
class CalibrationReference:
    sha256: str
    comparison_mode: str
    pass0_verdict: str
    comparison_eligibility: str
    prompt_identity_evidence: str
    verdict_strength_limit: str
    blocking_reasons: tuple[str, ...]
    oracle_scope: str
    canonical_json: str = field(repr=False)


@dataclass(frozen=True)
class Pass1Result:
    comparison_mode: ComparisonMode
    pass0_verdict: Pass0Verdict
    comparison_eligibility: ComparisonEligibility
    source_binding: Pass0SourceBinding
    source_binding_verified: bool
    status: Pass1Status
    evidence_completeness: EvidenceCompleteness
    compatibility: CompatibilityResult
    reference: Optional[Pass1RunInput]
    candidate: Optional[Pass1RunInput]
    localization: Optional[TokenLocalization]
    prefix_for_reproduction: Optional[PrefixForReproduction]
    pass2_disposition: Pass2Disposition
    calibration_ref: CalibrationReference
    verdict_strength_limit: VerdictStrengthLimit
    reason_codes: tuple[Pass1ReasonCode, ...] = ()
    inherited_pass0_reason_codes: tuple[str, ...] = ()
    blocking_reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    schema: str = SCHEMA
    kind: str = KIND
    contract_version: str = CONTRACT_VERSION
    evidence_scope: str = EVIDENCE_SCOPE
