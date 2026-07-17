"""Immutable data model for coverage-scoped Pass 3 localization.

The model intentionally represents bounded, observed checkpoint evidence.  It
has no field that can certify numeric divergence, tensor equality, or later-
pass readiness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import prod
from typing import Optional


SCHEMA = "lis.execution_artifact/v1"
KIND = "layer_localization"
CONTRACT_VERSION = "differential_verification_contract_v1"
DIGEST_ALGORITHM = "sha256"
DIGEST_VERSION = "lis.checkpoint.fp32le/v1"
DIGEST_DTYPE = "fp32"
DIGEST_BYTE_ORDER = "little"
DIGEST_CANONICALIZATION = (
    "ieee754-binary32-le;canonical-qnan;preserve-signed-zero"
)
DIGEST_DECISION_FIELD = "checkpoint_digest"
DIGEST_DECISION_SEMANTICS = "observed_representation_digest_mismatch"
DIGEST_MATCH_SEMANTICS = (
    "no_digest_difference_observed_for_aligned_representation"
)
BOUNDED_EQUALITY_SEMANTICS = (
    "No digest difference was observed only for aligned representations in "
    "captured common comparable coverage; this is not mathematical tensor "
    "equality or whole-runtime equivalence."
)
NON_CONFIRMATION_SEMANTICS = (
    "An observable digest mismatch is bounded localization evidence only; "
    "it does not confirm the first divergent layer or numeric divergence."
)


class Pass3Status(str, Enum):
    OBSERVABLE_MISMATCH_FOUND = "observable_mismatch_found"
    NO_MISMATCH_IN_CAPTURED_COVERAGE = (
        "no_mismatch_in_captured_coverage"
    )
    COMPARISON_BLOCKED_BY_PASS2 = "comparison_blocked_by_pass2"
    INSUFFICIENT_COMMON_COVERAGE = "insufficient_common_coverage"
    SOURCE_BINDING_INCONSISTENT = "source_binding_inconsistent"
    CHECKPOINT_ALIGNMENT_INCONSISTENT = (
        "checkpoint_alignment_inconsistent"
    )
    CHECKPOINT_ARTIFACT_MISSING = "checkpoint_artifact_missing"
    CHECKPOINT_SUMMARY_MALFORMED = "checkpoint_summary_malformed"
    COMPARISON_POLICY_UNAVAILABLE = "comparison_policy_unavailable"
    UNSUPPORTED_CHECKPOINT_LAYOUT = "unsupported_checkpoint_layout"
    INCONCLUSIVE = "inconclusive"


class Pass3ReasonCode(str, Enum):
    PASS2_NOT_READY = "pass3.pass2_not_ready"
    REPRODUCTION_REQUEST_ONLY = "pass3.reproduction_request_only"
    SOURCE_BINDING_INCONSISTENT = "pass3.source_binding_inconsistent"
    PASS2_ARTIFACT_IDENTITY_INCONSISTENT = (
        "pass3.pass2_artifact_identity_inconsistent"
    )
    PASS2_OBJECT_ARTIFACT_INCONSISTENT = (
        "pass3.pass2_object_artifact_inconsistent"
    )
    RUN_REPORT_CANONICAL_SHA_INCONSISTENT = (
        "pass3.run_report_canonical_sha_inconsistent"
    )
    ARTIFACT_SET_ID_INCONSISTENT = (
        "pass3.artifact_set_id_inconsistent"
    )
    BINDING_METADATA_MISSING = "pass3.binding_metadata_missing"
    RUNTIME_CHECKPOINT_STEP_MISMATCH = (
        "pass3.runtime_checkpoint_step_mismatch"
    )
    INSUFFICIENT_COMMON_COVERAGE = (
        "pass3.insufficient_common_coverage"
    )
    REFERENCE_CHECKPOINT_MISSING = "pass3.reference_checkpoint_missing"
    CANDIDATE_CHECKPOINT_MISSING = "pass3.candidate_checkpoint_missing"
    CHECKPOINT_ALIGNMENT_INCONSISTENT = (
        "pass3.checkpoint_alignment_inconsistent"
    )
    DUPLICATE_CHECKPOINT_COORDINATE = (
        "pass3.duplicate_checkpoint_coordinate"
    )
    CHECKPOINT_SUMMARY_MALFORMED = (
        "pass3.checkpoint_summary_malformed"
    )
    SUMMARY_FIELD_MISSING = "pass3.summary_field_missing"
    CHECKPOINT_DIGEST_INCOMPATIBLE = (
        "pass3.checkpoint_digest_incompatible"
    )
    COMPARISON_POLICY_UNAVAILABLE = (
        "pass3.comparison_policy_unavailable"
    )
    UNSUPPORTED_CHECKPOINT_LAYOUT = (
        "pass3.unsupported_checkpoint_layout"
    )
    ASYMMETRIC_COVERAGE = "pass3.asymmetric_coverage"
    OBSERVABLE_MISMATCH_FOUND = "pass3.observable_mismatch_found"
    NO_MISMATCH_IN_CAPTURED_COVERAGE = (
        "pass3.no_mismatch_in_captured_coverage"
    )


class CoverageState(str, Enum):
    CAPTURED = "captured"
    NOT_CAPTURED = "not_captured"
    UNSUPPORTED = "unsupported"
    MALFORMED = "malformed"
    UNEXPECTEDLY_ABSENT = "unexpectedly_absent"


class AlignmentStatus(str, Enum):
    ALIGNED = "aligned"
    SHAPE_MISMATCH = "shape_mismatch"
    DTYPE_MISMATCH = "dtype_mismatch"
    PRECISION_PATH_MISMATCH = "precision_path_mismatch"
    STAGE_MISMATCH = "stage_mismatch"
    BATCH_MISMATCH = "batch_mismatch"
    SEQUENCE_MISMATCH = "sequence_mismatch"
    MODEL_FAMILY_MISMATCH = "model_family_mismatch"
    DUPLICATE_COORDINATE = "duplicate_coordinate"


class SummaryFieldDisposition(str, Enum):
    EXACT = "exact"
    TOLERANCE_AWARE = "tolerance_aware"
    INFORMATIONAL_ONLY = "informational_only"
    UNSUPPORTED = "unsupported"


class SummaryEvidenceLevel(str, Enum):
    TIER0_STRUCTURAL = "tier0_structural"
    TIER1_BOUNDED_EXACT = "tier1_bounded_exact"
    TIER1_BOUNDED_CALIBRATED = "tier1_bounded_calibrated"
    TIER1_BOUNDED_DIGEST = "tier1_bounded_digest"
    UNAVAILABLE = "unavailable"


class Pass3DownstreamDisposition(str, Enum):
    BLOCKED = "blocked"
    EXPLORATORY_LOCALIZATION_ONLY = "exploratory_localization_only"
    SUSPECT_INTERVAL_AVAILABLE = "suspect_interval_available"


def _nonnegative_int(value: int, label: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


@dataclass(frozen=True)
class CheckpointCoordinate:
    runtime_checkpoint_step: int
    layer_index: int
    tensor_role: str
    batch_index: int
    sequence_index: int
    stage_order: int
    execution_ordinal: int

    def __post_init__(self):
        for name in (
            "runtime_checkpoint_step",
            "layer_index",
            "batch_index",
            "sequence_index",
            "stage_order",
            "execution_ordinal",
        ):
            _nonnegative_int(getattr(self, name), name)
        if not isinstance(self.tensor_role, str) or not self.tensor_role:
            raise ValueError("tensor_role must be a non-empty string")

    @property
    def logical_key(self) -> tuple[object, ...]:
        return (
            self.runtime_checkpoint_step,
            self.layer_index,
            self.tensor_role,
            self.batch_index,
            self.sequence_index,
            self.stage_order,
        )

    @property
    def order_key(self) -> tuple[int, ...]:
        return (
            self.runtime_checkpoint_step,
            self.layer_index,
            self.stage_order,
            self.execution_ordinal,
        )


@dataclass(frozen=True)
class CheckpointDigest:
    algorithm: str
    version: str
    tensor_role: str
    shape: tuple[int, ...]
    observed_dtype: str
    byte_order: str
    canonicalization: str
    value: str

    def __post_init__(self):
        if not self.shape:
            raise ValueError("digest shape must be non-empty")
        for dimension in self.shape:
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
            ):
                raise ValueError("digest shape dimensions must be positive")


@dataclass(frozen=True)
class CheckpointSummary:
    coordinate: CheckpointCoordinate
    source_checkpoint_name: str
    phase: str
    shape: tuple[int, ...]
    observed_dtype: str
    precision_path: str
    element_count: Optional[int]
    min_value: Optional[float]
    max_value: Optional[float]
    mean_value: Optional[float]
    l1_norm: Optional[float]
    l2_norm: Optional[float]
    finite_count: Optional[int]
    nan_count: Optional[int]
    inf_count: Optional[int]
    nan_present: Optional[bool]
    inf_present: Optional[bool]
    digest: Optional[CheckpointDigest]
    available_fields: frozenset[str]

    def __post_init__(self):
        if not self.shape:
            raise ValueError("checkpoint shape must be non-empty")
        for dimension in self.shape:
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension <= 0
            ):
                raise ValueError("checkpoint shape dimensions must be positive")
        if self.element_count is not None:
            _nonnegative_int(self.element_count, "element_count")
            if self.element_count != prod(self.shape):
                raise ValueError("element_count contradicts checkpoint shape")


@dataclass(frozen=True)
class CoverageEntry:
    coordinate: CheckpointCoordinate
    state: CoverageState
    detail: Optional[str] = None


@dataclass(frozen=True)
class CheckpointCoverage:
    requested_coordinates: tuple[CheckpointCoordinate, ...]
    captured_coordinates: tuple[CheckpointCoordinate, ...]
    missing_coordinates: tuple[CoverageEntry, ...]
    ordered_coordinates: tuple[CheckpointCoordinate, ...]
    total_layer_count: Optional[int]
    layout_name: str


@dataclass(frozen=True)
class AlignedCheckpointPair:
    coordinate: CheckpointCoordinate
    reference_summary: CheckpointSummary
    candidate_summary: CheckpointSummary
    alignment_status: AlignmentStatus


@dataclass(frozen=True)
class FieldComparison:
    field_name: str
    disposition: SummaryFieldDisposition
    equivalent: Optional[bool]
    abs_diff: Optional[float] = None
    resolved_abs_floor: Optional[float] = None
    resolved_rel_band: Optional[float] = None


@dataclass(frozen=True)
class SummaryComparisonResult:
    coordinate: CheckpointCoordinate
    equivalent: Optional[bool]
    mismatching_fields: tuple[str, ...]
    field_results: tuple[FieldComparison, ...]
    evidence_level: SummaryEvidenceLevel
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class SuspectInterval:
    start_boundary: str
    last_observed_equivalent_coordinate: Optional[CheckpointCoordinate]
    first_observed_mismatch_coordinate: CheckpointCoordinate
    start_exclusive: bool
    end_inclusive: bool
    unobserved_layer_indices: tuple[int, ...]
    notation: str

    def __post_init__(self):
        if self.start_boundary not in ("runtime_entry", "observed_checkpoint"):
            raise ValueError("unsupported suspect interval boundary")
        if not self.end_inclusive:
            raise ValueError("suspect interval end must be inclusive")
        if self.start_boundary == "runtime_entry":
            if self.last_observed_equivalent_coordinate is not None:
                raise ValueError("entry interval cannot have an equal coordinate")
            if self.start_exclusive or not self.notation.startswith("[entry, "):
                raise ValueError("entry interval must use [entry, L] semantics")
        else:
            if self.last_observed_equivalent_coordinate is None:
                raise ValueError("observed interval requires an equal coordinate")
            if not self.start_exclusive or not self.notation.startswith("("):
                raise ValueError("observed interval start must be exclusive")


@dataclass(frozen=True)
class Pass2Evidence:
    reproduction_evidence_tier: str
    generated_token_step: Optional[int]
    runtime_checkpoint_step: int
    localization_ref_sha256: str
    reference_original_run_report_sha256: Optional[str]
    candidate_original_run_report_sha256: Optional[str]
    reference_reproduction_sha256: Optional[str]
    candidate_reproduction_sha256: Optional[str]
    checkpoint_step_evidence: str
    verdict_strength_limit: str
    thread_count_caveat: bool


@dataclass(frozen=True)
class SourceArtifactBinding:
    role: str
    run_report_sha256: str
    trace_sha256: str
    artifact_set_id: str
    semantic_manifest_sha256: str


@dataclass(frozen=True)
class CoverageAnalysis:
    reference_requested: tuple[CheckpointCoordinate, ...] = ()
    reference_captured: tuple[CheckpointCoordinate, ...] = ()
    candidate_requested: tuple[CheckpointCoordinate, ...] = ()
    candidate_captured: tuple[CheckpointCoordinate, ...] = ()
    common_captured: tuple[CheckpointCoordinate, ...] = ()
    reference_only: tuple[CheckpointCoordinate, ...] = ()
    candidate_only: tuple[CheckpointCoordinate, ...] = ()
    common_comparable: tuple[CheckpointCoordinate, ...] = ()
    reference_missing: tuple[CoverageEntry, ...] = ()
    candidate_missing: tuple[CoverageEntry, ...] = ()


@dataclass(frozen=True)
class Pass3Result:
    status: Pass3Status
    downstream_disposition: Pass3DownstreamDisposition
    pass2_artifact_sha256: Optional[str] = None
    pass2_object_artifact_coherence_verified: bool = False
    pass2_evidence: Optional[Pass2Evidence] = None
    reference_binding: Optional[SourceArtifactBinding] = None
    candidate_binding: Optional[SourceArtifactBinding] = None
    checkpoint_artifact_binding_verified: bool = False
    target_runtime_checkpoint_step: Optional[int] = None
    coverage: CoverageAnalysis = field(default_factory=CoverageAnalysis)
    comparisons: tuple[SummaryComparisonResult, ...] = ()
    last_observed_equivalent_coordinate: Optional[CheckpointCoordinate] = None
    first_observed_mismatch_coordinate: Optional[CheckpointCoordinate] = None
    earliest_observable_suspect_layer: Optional[int] = None
    suspect_interval: Optional[SuspectInterval] = None
    decision_field: Optional[str] = None
    decision_semantics: Optional[str] = None
    evidence_level: SummaryEvidenceLevel = SummaryEvidenceLevel.UNAVAILABLE
    digest_contract_identity: Optional[str] = None
    reason_codes: tuple[Pass3ReasonCode, ...] = ()
    inherited_pass2_reason_codes: tuple[str, ...] = ()
    inherited_pass1_reason_codes: tuple[str, ...] = ()
    inherited_pass0_reason_codes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    equality_semantics: str = BOUNDED_EQUALITY_SEMANTICS
    non_confirmation_semantics: str = NON_CONFIRMATION_SEMANTICS
    schema: str = SCHEMA
    kind: str = KIND
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self):
        mismatch = self.status == Pass3Status.OBSERVABLE_MISMATCH_FOUND
        no_mismatch = (
            self.status == Pass3Status.NO_MISMATCH_IN_CAPTURED_COVERAGE
        )
        if mismatch:
            if (
                self.first_observed_mismatch_coordinate is None
                or self.suspect_interval is None
                or self.earliest_observable_suspect_layer
                != self.first_observed_mismatch_coordinate.layer_index
            ):
                raise ValueError("mismatch result requires a coherent interval")
            if (
                self.downstream_disposition
                != Pass3DownstreamDisposition.SUSPECT_INTERVAL_AVAILABLE
            ):
                raise ValueError("mismatch result requires suspect disposition")
        elif any(
            value is not None
            for value in (
                self.first_observed_mismatch_coordinate,
                self.earliest_observable_suspect_layer,
                self.suspect_interval,
            )
        ):
            raise ValueError("non-mismatch result cannot claim localization")
        if no_mismatch:
            if (
                self.downstream_disposition
                != Pass3DownstreamDisposition.EXPLORATORY_LOCALIZATION_ONLY
            ):
                raise ValueError("no-mismatch result is exploratory only")
        elif not mismatch and (
            self.downstream_disposition
            != Pass3DownstreamDisposition.BLOCKED
        ):
            raise ValueError("blocked/error result requires blocked disposition")
        if (
            self.pass2_evidence is not None
            and self.pass2_evidence.reproduction_evidence_tier
            == "reproduction_request_only"
            and (mismatch or no_mismatch)
        ):
            raise ValueError("request-only evidence cannot complete comparison")
        if self.checkpoint_artifact_binding_verified and (
            not self.pass2_object_artifact_coherence_verified
            or self.reference_binding is None
            or self.candidate_binding is None
        ):
            raise ValueError("artifact binding requires all prior evidence")
