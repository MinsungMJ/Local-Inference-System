"""Immutable result model for coverage-scoped Pass 4 intra-layer localization.

The model represents one bounded intra-layer localization outcome built from
facts that earlier work units already established.  It validates supplied
values for internal coherence only.  It performs no file I/O, JSON parsing,
artifact hashing, SHA verification, Pass 3 execution, parent classification,
runtime artifact parsing, coverage derivation, digest calculation or
comparison, mismatch selection, or serialization; those belong to the later
Pass 4 input, core, and artifact work units.

No field can certify numeric divergence, tensor equality, root cause, or
mapping into a frozen global verification-success enum.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from .pass3_model import (
    CheckpointCoordinate,
    DIGEST_DECISION_FIELD,
    DIGEST_DECISION_SEMANTICS,
    DIGEST_MATCH_SEMANTICS,
    SummaryEvidenceLevel,
    SuspectInterval,
)
from .pass4_contract import (
    CONTRACT_NAMESPACE,
    CONTRACT_VERSION,
    DIGEST_VERSION,
    EVIDENCE_LEVEL,
    INHERITED_BOUNDARY_EVIDENCE_ORIGIN,
    INHERITED_BOUNDARY_ID,
    INTRA_LAYER_LAYOUT_NAME,
    INTRA_LAYER_LAYOUT_VERSION,
    INTRA_LAYER_STAGES,
    LOCAL_EVIDENCE_ORIGIN,
    MODEL_FAMILY,
    NONCLAIMS,
    PASS3_DIGEST_VERSION,
    PHASE,
    RESULT_KIND,
    SCHEMA,
    STAGE_TAXONOMY,
    UINT64_MAX,
    IntraLayerCoordinate,
    IntraLayerSideCoverage,
    ParentSourceIdentity,
    Pass3ParentBinding,
    Pass3ParentClassification,
    Pass4Disposition,
    Pass4ReasonCode,
    Pass4Status,
    Pass4SuspectInterval,
    _SHA256,
    _strict_int,
    _strict_string,
    _strict_tuple,
    analyze_coverage,
    requested_coordinates,
    validate_interval_against_coverage,
    validate_nonclaims,
    validate_pass3_parent_pair,
    validate_status_algebra,
)


__all__ = [
    "Pass4ComparisonDecision",
    "Pass4LocalCoverageOutcome",
    "Pass4Comparison",
    "Pass4CoverageAnalysis",
    "Pass4SourceBinding",
    "Pass3ParentEvidence",
    "Pass4ClosingBoundaryDecision",
    "Pass4EvidenceCeiling",
    "Pass4Result",
]


MAX_COMPARISONS = len(INTRA_LAYER_STAGES)
MAX_REQUESTED_COORDINATES = len(INTRA_LAYER_STAGES)
MAX_WARNINGS = 32
MAX_INHERITED_REASON_CODES = 32
MAX_DETAIL_BYTES = 256
MAX_IDENTIFIER_BYTES = 128

REPRODUCTION_REQUEST_ONLY = "reproduction_request_only"


def _strict_bool(value: Any, label: str) -> bool:
    """Accept only a real boolean; integers never stand in for a flag."""
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be an explicit boolean")
    return value


def _bounded_string(value: Any, label: str, limit: int) -> str:
    _strict_string(value, label)
    if len(value.encode("utf-8")) > limit:
        raise ValueError(f"{label} exceeds {limit} UTF-8 bytes")
    return value


def _bounded_strings(
    values: Any,
    label: str,
    *,
    max_items: int,
    limit: int = MAX_DETAIL_BYTES,
) -> tuple[str, ...]:
    _strict_tuple(values, label)
    if len(values) > max_items:
        raise ValueError(f"{label} exceeds {max_items} entries")
    for item in values:
        _bounded_string(item, f"{label} entry", limit)
    return values


def _digest_value(value: Any, label: str) -> str:
    """Accept only the frozen canonical `sha256:<64 lowercase hex>` form."""
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a canonical SHA-256 identity")
    return value


def _optional_int(value: Any, label: str, *, minimum: int = 0) -> Optional[int]:
    if value is None:
        return None
    return _strict_int(value, label, minimum=minimum, maximum=UINT64_MAX)


def _optional_identifier(value: Any, label: str) -> Optional[str]:
    if value is None:
        return None
    return _bounded_string(value, label, MAX_IDENTIFIER_BYTES)


def _exact(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise ValueError(f"{label} must be exactly {expected!r}")


class Pass4ComparisonDecision(str, Enum):
    EQUIVALENT_OBSERVED_REPRESENTATION_DIGEST = (
        "equivalent_observed_representation_digest"
    )
    MISMATCHING_OBSERVED_REPRESENTATION_DIGEST = (
        "mismatching_observed_representation_digest"
    )


class Pass4LocalCoverageOutcome(str, Enum):
    LOCAL_MISMATCH_FOUND = "local_mismatch_found"
    NO_MISMATCH_IN_COMMON_INTRA_LAYER_COVERAGE = (
        "no_mismatch_in_common_intra_layer_coverage"
    )


COMPARISON_DECISION_SEMANTICS = {
    Pass4ComparisonDecision.EQUIVALENT_OBSERVED_REPRESENTATION_DIGEST:
        DIGEST_MATCH_SEMANTICS,
    Pass4ComparisonDecision.MISMATCHING_OBSERVED_REPRESENTATION_DIGEST:
        DIGEST_DECISION_SEMANTICS,
}


LOCAL_COVERAGE_OUTCOME_FOR_STATUS = {
    Pass4Status.OBSERVABLE_INTRA_LAYER_MISMATCH_FOUND:
        Pass4LocalCoverageOutcome.LOCAL_MISMATCH_FOUND,
    Pass4Status.MISMATCH_BOUNDED_TO_INHERITED_CLOSING_BOUNDARY:
        Pass4LocalCoverageOutcome.NO_MISMATCH_IN_COMMON_INTRA_LAYER_COVERAGE,
}


COMPARISON_STATUSES = frozenset(LOCAL_COVERAGE_OUTCOME_FOR_STATUS)


@dataclass(frozen=True)
class Pass4Comparison:
    """One observed digest decision for one common comparable coordinate."""

    coordinate: IntraLayerCoordinate
    shape: tuple[int, ...]
    reference_digest: str
    candidate_digest: str
    decision: Pass4ComparisonDecision

    def __post_init__(self):
        if not isinstance(self.coordinate, IntraLayerCoordinate):
            raise TypeError("coordinate must be an IntraLayerCoordinate")
        _strict_tuple(self.shape, "comparison shape")
        if not self.shape:
            raise ValueError("comparison shape must be non-empty")
        for dimension in self.shape:
            _strict_int(
                dimension,
                "comparison shape dimension",
                minimum=1,
                maximum=UINT64_MAX,
            )
        _digest_value(self.reference_digest, "reference_digest")
        _digest_value(self.candidate_digest, "candidate_digest")
        if not isinstance(self.decision, Pass4ComparisonDecision):
            raise ValueError("unknown Pass 4 comparison decision")
        mismatching = (
            self.decision
            == Pass4ComparisonDecision.MISMATCHING_OBSERVED_REPRESENTATION_DIGEST
        )
        if mismatching != (self.reference_digest != self.candidate_digest):
            raise ValueError(
                "comparison decision contradicts the observed digests"
            )

    @property
    def equivalent(self) -> bool:
        return (
            self.decision
            == Pass4ComparisonDecision.EQUIVALENT_OBSERVED_REPRESENTATION_DIGEST
        )

    @property
    def decision_semantics(self) -> str:
        return COMPARISON_DECISION_SEMANTICS[self.decision]


@dataclass(frozen=True)
class Pass4CoverageAnalysis:
    """Result-side coverage that verifies, never recomputes, its algebra."""

    reference: IntraLayerSideCoverage
    candidate: IntraLayerSideCoverage
    common_captured: tuple[IntraLayerCoordinate, ...]
    common_comparable: tuple[IntraLayerCoordinate, ...]
    reference_only: tuple[IntraLayerCoordinate, ...]
    candidate_only: tuple[IntraLayerCoordinate, ...]

    def __post_init__(self):
        if not isinstance(self.reference, IntraLayerSideCoverage):
            raise TypeError("reference must be IntraLayerSideCoverage")
        if not isinstance(self.candidate, IntraLayerSideCoverage):
            raise TypeError("candidate must be IntraLayerSideCoverage")
        for label in (
            "common_captured",
            "common_comparable",
            "reference_only",
            "candidate_only",
        ):
            _strict_tuple(getattr(self, label), label)
        # Frozen P4-1 algebra is the only implementation; the supplied values
        # are verified against it and never replaced, sorted, or filled in.
        derived = analyze_coverage(
            self.reference,
            self.candidate,
            common_comparable=self.common_comparable,
        )
        if self.common_captured != derived.common_captured:
            raise ValueError(
                "common_captured contradicts the frozen coverage algebra"
            )
        if self.reference_only != derived.reference_only:
            raise ValueError(
                "reference_only is not the exact captured-set difference"
            )
        if self.candidate_only != derived.candidate_only:
            raise ValueError(
                "candidate_only is not the exact captured-set difference"
            )
        if len(self.reference.requested_coordinates) != (
            MAX_REQUESTED_COORDINATES
        ):
            raise ValueError("requested coverage is not the frozen v1 request")
        for side in (self.reference, self.candidate):
            for missing in side.missing_coordinates:
                if missing.detail is not None:
                    _bounded_string(
                        missing.detail,
                        "missing coordinate detail",
                        MAX_DETAIL_BYTES,
                    )

    @property
    def requested_coordinates(self) -> tuple[IntraLayerCoordinate, ...]:
        return self.reference.requested_coordinates

    @property
    def reference_captured(self) -> tuple[IntraLayerCoordinate, ...]:
        return self.reference.captured_coordinates

    @property
    def candidate_captured(self) -> tuple[IntraLayerCoordinate, ...]:
        return self.candidate.captured_coordinates

    @property
    def reference_missing(self) -> tuple:
        return self.reference.missing_coordinates

    @property
    def candidate_missing(self) -> tuple:
        return self.candidate.missing_coordinates

    @property
    def asymmetric(self) -> bool:
        return bool(self.reference_only or self.candidate_only)


@dataclass(frozen=True)
class Pass4SourceBinding:
    """One already-verified per-side artifact identity chain."""

    role: str
    identity: ParentSourceIdentity
    parent_recorded_trace_binding_verified: bool

    def __post_init__(self):
        _bounded_string(self.role, "source binding role", MAX_IDENTIFIER_BYTES)
        if not isinstance(self.identity, ParentSourceIdentity):
            raise TypeError("identity must be a ParentSourceIdentity")
        _strict_bool(
            self.parent_recorded_trace_binding_verified,
            "parent_recorded_trace_binding_verified",
        )


@dataclass(frozen=True)
class Pass3ParentEvidence:
    """Already-validated Pass 3A discovery and Pass 3B authoritative facts."""

    classification: Pass3ParentClassification
    discovery: Pass3ParentBinding
    authoritative: Pass3ParentBinding
    typed_artifact_coherence_verified: bool
    source_binding_verified: bool
    cross_generation_semantic_coherence_verified: bool
    discovery_selected_layer: Optional[int] = None
    authoritative_selected_layer: Optional[int] = None
    target_runtime_checkpoint_step: Optional[int] = None
    parent_first_mismatch_coordinate: Optional[CheckpointCoordinate] = None
    parent_last_observed_equivalent_coordinate: Optional[
        CheckpointCoordinate
    ] = None
    parent_suspect_interval: Optional[SuspectInterval] = None
    parent_evidence_level: Optional[SummaryEvidenceLevel] = None
    parent_decision_field: Optional[str] = None
    parent_decision_semantics: Optional[str] = None
    parent_digest_contract_identity: Optional[str] = None
    pass2_reproduction_evidence_tier: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.classification, Pass3ParentClassification):
            raise ValueError("unknown Pass 3 parent classification")
        validate_pass3_parent_pair(self.discovery, self.authoritative)
        for label in (
            "typed_artifact_coherence_verified",
            "source_binding_verified",
            "cross_generation_semantic_coherence_verified",
        ):
            _strict_bool(getattr(self, label), label)
        _optional_int(
            self.discovery_selected_layer, "discovery_selected_layer"
        )
        _optional_int(
            self.authoritative_selected_layer, "authoritative_selected_layer"
        )
        _optional_int(
            self.target_runtime_checkpoint_step,
            "target_runtime_checkpoint_step",
            minimum=1,
        )
        for label in (
            "parent_first_mismatch_coordinate",
            "parent_last_observed_equivalent_coordinate",
        ):
            value = getattr(self, label)
            if value is not None and not isinstance(value, CheckpointCoordinate):
                raise TypeError(f"{label} must be a CheckpointCoordinate")
        if self.parent_suspect_interval is not None and not isinstance(
            self.parent_suspect_interval, SuspectInterval
        ):
            raise TypeError(
                "parent_suspect_interval must be a Pass 3 SuspectInterval"
            )
        if self.parent_evidence_level is not None and not isinstance(
            self.parent_evidence_level, SummaryEvidenceLevel
        ):
            raise ValueError("unknown parent evidence level")
        for label in (
            "parent_decision_field",
            "parent_decision_semantics",
            "parent_digest_contract_identity",
            "pass2_reproduction_evidence_tier",
        ):
            _optional_identifier(getattr(self, label), label)
        if self.parent_decision_field is not None:
            _exact(
                self.parent_decision_field,
                DIGEST_DECISION_FIELD,
                "parent_decision_field",
            )
        if self.parent_decision_semantics is not None:
            _exact(
                self.parent_decision_semantics,
                DIGEST_DECISION_SEMANTICS,
                "parent_decision_semantics",
            )
        if self.parent_digest_contract_identity is not None:
            _exact(
                self.parent_digest_contract_identity,
                PASS3_DIGEST_VERSION,
                "parent_digest_contract_identity",
            )
        if self.pass2_reproduction_evidence_tier == REPRODUCTION_REQUEST_ONLY:
            raise ValueError(
                "request-only Pass 2 evidence cannot support Pass 4 evidence"
            )
        self._validate_coordinates()
        self._validate_classification()

    def _validate_coordinates(self) -> None:
        mismatch = self.parent_first_mismatch_coordinate
        equivalent = self.parent_last_observed_equivalent_coordinate
        if mismatch is not None and mismatch.tensor_role != "layer_output":
            raise ValueError(
                "the parent mismatch coordinate must be a layer output"
            )
        if (
            mismatch is not None
            and equivalent is not None
            and equivalent.order_key >= mismatch.order_key
        ):
            raise ValueError(
                "the parent equivalent coordinate must precede the mismatch"
            )
        interval = self.parent_suspect_interval
        if interval is not None:
            if interval.first_observed_mismatch_coordinate != mismatch:
                raise ValueError(
                    "the parent interval does not end at the parent mismatch"
                )
            if interval.last_observed_equivalent_coordinate != equivalent:
                raise ValueError(
                    "the parent interval start contradicts the parent evidence"
                )

    def _validate_classification(self) -> None:
        eligible = self.classification == Pass3ParentClassification.ELIGIBLE
        drift = (
            self.classification
            == Pass3ParentClassification.PARENT_REVALIDATION_INCONSISTENT
        )
        layers = (
            self.discovery_selected_layer,
            self.authoritative_selected_layer,
        )
        both_layers = all(layer is not None for layer in layers)
        if not drift and both_layers and layers[0] != layers[1]:
            raise ValueError(
                "discovery and authoritative layer drift is only representable "
                "as parent_revalidation_inconsistent"
            )
        if eligible:
            self._validate_eligible(both_layers)
            return
        if self.parent_first_mismatch_coordinate is not None:
            raise ValueError(
                "only an eligible parent may carry parent localization"
            )
        if self.parent_suspect_interval is not None:
            raise ValueError(
                "only an eligible parent may carry a parent suspect interval"
            )
        if drift and self.cross_generation_semantic_coherence_verified and (
            not both_layers or layers[0] == layers[1]
        ):
            raise ValueError(
                "parent_revalidation_inconsistent requires observed drift"
            )
        if (
            self.classification == Pass3ParentClassification.NOT_APPLICABLE
        ):
            if not self.typed_artifact_coherence_verified:
                raise ValueError(
                    "a not-applicable parent must be fully validated"
                )
            if self.authoritative_selected_layer is not None:
                raise ValueError(
                    "a no-mismatch parent cannot select a target layer"
                )

    def _validate_eligible(self, both_layers: bool) -> None:
        if not (
            self.typed_artifact_coherence_verified
            and self.source_binding_verified
            and self.cross_generation_semantic_coherence_verified
        ):
            raise ValueError(
                "an eligible parent requires every verification flag"
            )
        if not both_layers:
            raise ValueError(
                "an eligible parent requires both selected layers"
            )
        if self.target_runtime_checkpoint_step is None:
            raise ValueError(
                "an eligible parent requires the runtime checkpoint step"
            )
        if self.parent_first_mismatch_coordinate is None:
            raise ValueError(
                "an eligible parent requires its first mismatch coordinate"
            )
        if self.parent_suspect_interval is None:
            raise ValueError(
                "an eligible parent requires its suspect interval"
            )
        if (
            self.parent_first_mismatch_coordinate.layer_index
            != self.authoritative_selected_layer
        ):
            raise ValueError(
                "the selected layer must be the parent mismatch layer"
            )
        if (
            self.parent_first_mismatch_coordinate.runtime_checkpoint_step
            != self.target_runtime_checkpoint_step
        ):
            raise ValueError(
                "the parent mismatch step must equal the target step"
            )
        if (
            self.parent_evidence_level
            != SummaryEvidenceLevel.TIER1_BOUNDED_DIGEST
        ):
            raise ValueError(
                "an eligible parent requires tier1_bounded_digest evidence"
            )
        for label in (
            "parent_decision_field",
            "parent_decision_semantics",
            "parent_digest_contract_identity",
            "pass2_reproduction_evidence_tier",
        ):
            if getattr(self, label) is None:
                raise ValueError(f"an eligible parent requires {label}")

    @property
    def authorizes_pass4_evidence(self) -> bool:
        return (
            self.classification == Pass3ParentClassification.ELIGIBLE
            and self.authoritative.authorizes_pass4_evidence
        )


@dataclass(frozen=True)
class Pass4ClosingBoundaryDecision:
    """The inherited authoritative Pass 3B terminal decision."""

    parent_coordinate: CheckpointCoordinate
    parent_decision_field: str = DIGEST_DECISION_FIELD
    parent_decision_semantics: str = DIGEST_DECISION_SEMANTICS
    parent_digest_contract_identity: str = PASS3_DIGEST_VERSION
    parent_evidence_level: SummaryEvidenceLevel = (
        SummaryEvidenceLevel.TIER1_BOUNDED_DIGEST
    )
    boundary_id: str = INHERITED_BOUNDARY_ID
    evidence_origin: str = INHERITED_BOUNDARY_EVIDENCE_ORIGIN

    def __post_init__(self):
        if not isinstance(self.parent_coordinate, CheckpointCoordinate):
            raise TypeError(
                "the closing boundary must use a Pass 3 CheckpointCoordinate"
            )
        if self.parent_coordinate.tensor_role != "layer_output":
            raise ValueError(
                "the closing boundary must be the parent layer output"
            )
        _exact(self.boundary_id, INHERITED_BOUNDARY_ID, "boundary_id")
        _exact(
            self.evidence_origin,
            INHERITED_BOUNDARY_EVIDENCE_ORIGIN,
            "evidence_origin",
        )
        _exact(
            self.parent_decision_field,
            DIGEST_DECISION_FIELD,
            "parent_decision_field",
        )
        _exact(
            self.parent_decision_semantics,
            DIGEST_DECISION_SEMANTICS,
            "parent_decision_semantics",
        )
        _exact(
            self.parent_digest_contract_identity,
            PASS3_DIGEST_VERSION,
            "parent_digest_contract_identity",
        )
        if (
            self.parent_evidence_level
            != SummaryEvidenceLevel.TIER1_BOUNDED_DIGEST
        ):
            raise ValueError(
                "the closing boundary requires tier1_bounded_digest evidence"
            )


@dataclass(frozen=True)
class Pass4EvidenceCeiling:
    """Every frozen nonclaim, mandatory, defaultless, and exactly false."""

    numeric_divergence_confirmed: bool
    true_first_divergence_confirmed: bool
    root_cause_identified: bool
    tensor_equality_proved: bool
    complete_intra_layer_coverage_proved: bool
    operation_level_localization_performed: bool
    exhaustive_confirmation_performed: bool
    automatic_frozen_success_mapping: bool

    def __post_init__(self):
        for label in NONCLAIMS:
            if getattr(self, label) is not False:
                raise ValueError(
                    f"{label} must be present and exactly false"
                )
        validate_nonclaims(self.as_mapping())

    def as_mapping(self) -> dict[str, bool]:
        return {label: getattr(self, label) for label in NONCLAIMS}


if tuple(Pass4EvidenceCeiling.__dataclass_fields__) != tuple(NONCLAIMS):
    raise RuntimeError(
        "Pass4EvidenceCeiling fields must mirror the frozen nonclaims"
    )


FROZEN_EVIDENCE_CEILING = Pass4EvidenceCeiling(
    False, False, False, False, False, False, False, False
)


_REQUIRED = "required"
_FORBIDDEN = "forbidden"
_OPTIONAL = "optional"

_POLICY_FIELDS = (
    "parent_pass3",
    "bindings",
    "target",
    "coverage",
    "comparisons",
    "local_coverage_outcome",
    "first_observed_local_mismatch_coordinate",
    "last_observed_equivalent_coordinate",
    "closing_boundary_decision",
    "suspect_interval",
    "evidence_level",
    "digest_contract_identity",
)

_TARGET_FIELDS = (
    "target_runtime_checkpoint_step",
    "target_layer",
    "target_token_position",
    "phase",
    "model_family",
    "precision_path",
    "layout_name",
    "layout_version",
    "stage_taxonomy",
)


def _policy(*values: str) -> dict[str, str]:
    return dict(zip(_POLICY_FIELDS, values))


# Derived from the frozen status_algebra.failure_precedence: a status may
# retain exactly the evidence produced by the gates it passed.
FIELD_POLICY = {
    Pass4Status.OBSERVABLE_INTRA_LAYER_MISMATCH_FOUND: _policy(
        _REQUIRED, _REQUIRED, _REQUIRED, _REQUIRED, _REQUIRED, _REQUIRED,
        _REQUIRED, _OPTIONAL, _FORBIDDEN, _REQUIRED, _REQUIRED, _REQUIRED,
    ),
    Pass4Status.MISMATCH_BOUNDED_TO_INHERITED_CLOSING_BOUNDARY: _policy(
        _REQUIRED, _REQUIRED, _REQUIRED, _REQUIRED, _REQUIRED, _REQUIRED,
        _FORBIDDEN, _REQUIRED, _REQUIRED, _REQUIRED, _REQUIRED, _REQUIRED,
    ),
    Pass4Status.NOT_APPLICABLE: _policy(
        _REQUIRED, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN,
        _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN,
    ),
    Pass4Status.COMPARISON_BLOCKED_BY_PASS3: _policy(
        _OPTIONAL, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN,
        _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN,
    ),
    Pass4Status.UNSUPPORTED_PARENT: _policy(
        _REQUIRED, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN,
        _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN,
    ),
    Pass4Status.PARENT_REVALIDATION_INCONSISTENT: _policy(
        _REQUIRED, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN,
        _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN,
    ),
    Pass4Status.SOURCE_BINDING_INCONSISTENT: _policy(
        _REQUIRED, _OPTIONAL, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN,
        _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN,
    ),
    Pass4Status.UNSUPPORTED_INTRA_LAYER_LAYOUT: _policy(
        _REQUIRED, _REQUIRED, _OPTIONAL, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN,
        _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN,
    ),
    Pass4Status.CHECKPOINT_SUMMARY_MALFORMED: _policy(
        _REQUIRED, _REQUIRED, _REQUIRED, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN,
        _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN,
    ),
    Pass4Status.INSUFFICIENT_COMMON_INTRA_LAYER_COVERAGE: _policy(
        _REQUIRED, _REQUIRED, _REQUIRED, _REQUIRED, _FORBIDDEN, _FORBIDDEN,
        _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN,
    ),
    Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT: _policy(
        _REQUIRED, _REQUIRED, _REQUIRED, _REQUIRED, _FORBIDDEN, _FORBIDDEN,
        _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN,
    ),
    Pass4Status.COMPARISON_POLICY_UNAVAILABLE: _policy(
        _REQUIRED, _REQUIRED, _REQUIRED, _REQUIRED, _FORBIDDEN, _FORBIDDEN,
        _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN, _FORBIDDEN,
    ),
}


PARENT_CLASSIFICATION_FOR_STATUS = {
    Pass4Status.NOT_APPLICABLE: Pass3ParentClassification.NOT_APPLICABLE,
    Pass4Status.COMPARISON_BLOCKED_BY_PASS3:
        Pass3ParentClassification.COMPARISON_BLOCKED_BY_PASS3,
    Pass4Status.UNSUPPORTED_PARENT:
        Pass3ParentClassification.UNSUPPORTED_PARENT,
    Pass4Status.PARENT_REVALIDATION_INCONSISTENT:
        Pass3ParentClassification.PARENT_REVALIDATION_INCONSISTENT,
}


@dataclass(frozen=True)
class Pass4Result:
    """One coherent, bounded Pass 4 intra-layer localization outcome."""

    status: Pass4Status
    disposition: Pass4Disposition
    evidence_ceiling: Pass4EvidenceCeiling
    reason_codes: tuple[Pass4ReasonCode, ...]
    parent_pass3: Optional[Pass3ParentEvidence] = None
    reference_binding: Optional[Pass4SourceBinding] = None
    candidate_binding: Optional[Pass4SourceBinding] = None
    target_runtime_checkpoint_step: Optional[int] = None
    target_layer: Optional[int] = None
    target_token_position: Optional[int] = None
    phase: Optional[str] = None
    model_family: Optional[str] = None
    precision_path: Optional[str] = None
    layout_name: Optional[str] = None
    layout_version: Optional[int] = None
    stage_taxonomy: Optional[str] = None
    coverage: Optional[Pass4CoverageAnalysis] = None
    comparisons: tuple[Pass4Comparison, ...] = ()
    local_coverage_outcome: Optional[Pass4LocalCoverageOutcome] = None
    last_observed_equivalent_coordinate: Optional[IntraLayerCoordinate] = None
    first_observed_local_mismatch_coordinate: Optional[
        IntraLayerCoordinate
    ] = None
    closing_boundary_decision: Optional[Pass4ClosingBoundaryDecision] = None
    suspect_interval: Optional[Pass4SuspectInterval] = None
    evidence_level: Optional[str] = None
    digest_contract_identity: Optional[str] = None
    inherited_pass3_reason_codes: tuple[str, ...] = ()
    inherited_pass2_reason_codes: tuple[str, ...] = ()
    inherited_pass1_reason_codes: tuple[str, ...] = ()
    inherited_pass0_reason_codes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    schema: str = SCHEMA
    kind: str = RESULT_KIND
    contract_version: str = CONTRACT_VERSION
    contract_namespace: str = CONTRACT_NAMESPACE

    def __post_init__(self):
        self._validate_types()
        validate_status_algebra(
            self.status, self.disposition, self.reason_codes
        )
        self._validate_identity()
        self._validate_bounded_text()
        self._validate_presence()
        self._validate_parent_classification()
        if self.status in COMPARISON_STATUSES:
            self._validate_comparison_result()
        else:
            self._validate_noncomparison_result()
        self._validate_secondary_reasons()

    # -- structural typing -------------------------------------------------

    def _validate_types(self) -> None:
        if not isinstance(self.evidence_ceiling, Pass4EvidenceCeiling):
            raise TypeError(
                "evidence_ceiling must be a Pass4EvidenceCeiling"
            )
        validate_nonclaims(self.evidence_ceiling.as_mapping())
        typed = (
            ("parent_pass3", Pass3ParentEvidence),
            ("reference_binding", Pass4SourceBinding),
            ("candidate_binding", Pass4SourceBinding),
            ("coverage", Pass4CoverageAnalysis),
            ("closing_boundary_decision", Pass4ClosingBoundaryDecision),
            ("suspect_interval", Pass4SuspectInterval),
            ("last_observed_equivalent_coordinate", IntraLayerCoordinate),
            (
                "first_observed_local_mismatch_coordinate",
                IntraLayerCoordinate,
            ),
        )
        for label, expected in typed:
            value = getattr(self, label)
            if value is not None and not isinstance(value, expected):
                raise TypeError(f"{label} must be a {expected.__name__}")
        _strict_tuple(self.comparisons, "comparisons")
        for comparison in self.comparisons:
            if not isinstance(comparison, Pass4Comparison):
                raise TypeError("comparisons must contain Pass4Comparison")
        if len(self.comparisons) > MAX_COMPARISONS:
            raise ValueError(
                f"comparisons exceed the frozen bound of {MAX_COMPARISONS}"
            )
        if self.local_coverage_outcome is not None and not isinstance(
            self.local_coverage_outcome, Pass4LocalCoverageOutcome
        ):
            raise ValueError("unknown Pass 4 local coverage outcome")
        _optional_int(
            self.target_runtime_checkpoint_step,
            "target_runtime_checkpoint_step",
            minimum=1,
        )
        _optional_int(self.target_layer, "target_layer")
        _optional_int(self.target_token_position, "target_token_position")
        _optional_int(self.layout_version, "layout_version")
        for label in (
            "phase",
            "model_family",
            "precision_path",
            "layout_name",
            "stage_taxonomy",
            "evidence_level",
            "digest_contract_identity",
        ):
            _optional_identifier(getattr(self, label), label)

    def _validate_identity(self) -> None:
        _exact(self.schema, SCHEMA, "schema")
        _exact(self.kind, RESULT_KIND, "kind")
        _exact(self.contract_version, CONTRACT_VERSION, "contract_version")
        _exact(
            self.contract_namespace,
            CONTRACT_NAMESPACE,
            "contract_namespace",
        )

    def _validate_bounded_text(self) -> None:
        _bounded_strings(self.warnings, "warnings", max_items=MAX_WARNINGS)
        for label in (
            "inherited_pass3_reason_codes",
            "inherited_pass2_reason_codes",
            "inherited_pass1_reason_codes",
            "inherited_pass0_reason_codes",
        ):
            _bounded_strings(
                getattr(self, label),
                label,
                max_items=MAX_INHERITED_REASON_CODES,
            )

    # -- status-dependent field presence -----------------------------------

    def _present(self, key: str) -> bool:
        if key == "bindings":
            return (
                self.reference_binding is not None
                or self.candidate_binding is not None
            )
        if key == "target":
            return any(
                getattr(self, label) is not None for label in _TARGET_FIELDS
            )
        value = getattr(self, key)
        if isinstance(value, tuple):
            return bool(value)
        return value is not None

    def _validate_presence(self) -> None:
        policy = FIELD_POLICY[self.status]
        for key, rule in policy.items():
            present = self._present(key)
            if rule == _REQUIRED and not present:
                raise ValueError(
                    f"{self.status.value} requires {key}"
                )
            if rule == _FORBIDDEN and present:
                raise ValueError(
                    f"{self.status.value} cannot carry {key}"
                )
        if policy["bindings"] == _REQUIRED and (
            self.reference_binding is None or self.candidate_binding is None
        ):
            raise ValueError(
                f"{self.status.value} requires both source bindings"
            )
        supplied = tuple(
            label
            for label in _TARGET_FIELDS
            if getattr(self, label) is not None
        )
        if policy["target"] == _REQUIRED and len(supplied) != len(
            _TARGET_FIELDS
        ):
            raise ValueError(
                f"{self.status.value} requires the complete target identity"
            )
        if policy["target"] == _OPTIONAL and supplied and len(supplied) != len(
            _TARGET_FIELDS
        ):
            raise ValueError("the target identity must be complete or absent")

    def _validate_parent_classification(self) -> None:
        if self.parent_pass3 is None:
            return
        expected = PARENT_CLASSIFICATION_FOR_STATUS.get(
            self.status, Pass3ParentClassification.ELIGIBLE
        )
        if self.parent_pass3.classification != expected:
            raise ValueError(
                f"{self.status.value} requires a {expected.value} parent"
            )

    # -- comparison results ------------------------------------------------

    def _validate_target_identity(self) -> None:
        _exact(self.phase, PHASE, "phase")
        _exact(self.model_family, MODEL_FAMILY, "model_family")
        _exact(self.layout_name, INTRA_LAYER_LAYOUT_NAME, "layout_name")
        _exact(
            self.layout_version, INTRA_LAYER_LAYOUT_VERSION, "layout_version"
        )
        _exact(self.stage_taxonomy, STAGE_TAXONOMY, "stage_taxonomy")
        parent = self.parent_pass3
        if parent is not None and parent.target_runtime_checkpoint_step not in (
            None,
            self.target_runtime_checkpoint_step,
        ):
            raise ValueError(
                "the target step contradicts the authoritative parent"
            )
        if parent is not None and parent.authoritative_selected_layer not in (
            None,
            self.target_layer,
        ):
            raise ValueError(
                "the target layer contradicts the authoritative parent"
            )
        if self.coverage is not None:
            expected = requested_coordinates(
                self.target_runtime_checkpoint_step,
                self.target_layer,
                self.target_token_position,
            )
            if self.coverage.requested_coordinates != expected:
                raise ValueError(
                    "coverage requested coordinates contradict the target"
                )

    def _validate_comparison_result(self) -> None:
        self._validate_target_identity()
        parent = self.parent_pass3
        if not parent.authorizes_pass4_evidence:
            raise ValueError(
                "a comparison result requires an authorizing Pass 3B parent"
            )
        for binding in (self.reference_binding, self.candidate_binding):
            if not binding.parent_recorded_trace_binding_verified:
                raise ValueError(
                    "a comparison result requires verified source bindings"
                )
        _exact(self.evidence_level, EVIDENCE_LEVEL, "evidence_level")
        _exact(
            self.digest_contract_identity,
            DIGEST_VERSION,
            "digest_contract_identity",
        )
        _exact(
            self.local_coverage_outcome,
            LOCAL_COVERAGE_OUTCOME_FOR_STATUS[self.status],
            "local_coverage_outcome",
        )
        coordinates = tuple(item.coordinate for item in self.comparisons)
        if coordinates != self.coverage.common_comparable:
            raise ValueError(
                "comparisons must be exactly the declared common comparable "
                "coverage in order"
            )
        for coordinate in coordinates:
            if (
                coordinate.runtime_checkpoint_step
                != self.target_runtime_checkpoint_step
                or coordinate.layer_index != self.target_layer
                or coordinate.token_position != self.target_token_position
            ):
                raise ValueError(
                    "a comparison coordinate contradicts the target"
                )
        mismatches = tuple(
            index
            for index, item in enumerate(self.comparisons)
            if not item.equivalent
        )
        if self.status == Pass4Status.OBSERVABLE_INTRA_LAYER_MISMATCH_FOUND:
            self._validate_local_mismatch(mismatches)
        else:
            self._validate_inherited_boundary(mismatches)

    def _validate_local_mismatch(self, mismatches: tuple[int, ...]) -> None:
        if not mismatches:
            raise ValueError(
                "a local mismatch result requires a mismatching comparison"
            )
        first = mismatches[0]
        if (
            self.first_observed_local_mismatch_coordinate
            != self.comparisons[first].coordinate
        ):
            raise ValueError(
                "the first observed local mismatch is not the earliest "
                "mismatching comparison"
            )
        expected_equivalent = (
            self.comparisons[first - 1].coordinate if first else None
        )
        if self.last_observed_equivalent_coordinate != expected_equivalent:
            raise ValueError(
                "the last observed equivalent coordinate is inconsistent"
            )
        interval = self.suspect_interval
        if interval.end_kind != "local_coordinate":
            raise ValueError("a local mismatch requires a local interval end")
        if interval.end_evidence_origin != LOCAL_EVIDENCE_ORIGIN:
            raise ValueError("a local interval end must be local evidence")
        self._validate_interval_start(expected_equivalent)
        validate_interval_against_coverage(
            interval,
            self.coverage.common_comparable,
            first_local_mismatch=(
                self.first_observed_local_mismatch_coordinate
            ),
            authoritative_parent_coordinate=(
                self.parent_pass3.parent_first_mismatch_coordinate
            ),
        )

    def _validate_inherited_boundary(
        self, mismatches: tuple[int, ...]
    ) -> None:
        if mismatches:
            raise ValueError(
                "an inherited closing boundary requires no local mismatch"
            )
        if not self.comparisons:
            raise ValueError(
                "an inherited closing boundary requires local comparisons"
            )
        if (
            self.last_observed_equivalent_coordinate
            != self.comparisons[-1].coordinate
        ):
            raise ValueError(
                "the last observed equivalent coordinate is inconsistent"
            )
        closing = self.closing_boundary_decision
        parent_coordinate = self.parent_pass3.parent_first_mismatch_coordinate
        if closing.parent_coordinate != parent_coordinate:
            raise ValueError(
                "the closing boundary must be the authoritative Pass 3B "
                "terminal coordinate"
            )
        interval = self.suspect_interval
        if interval.end_kind != "inherited_parent_boundary":
            raise ValueError(
                "an inherited closing boundary requires an inherited interval"
            )
        if interval.end_evidence_origin != INHERITED_BOUNDARY_EVIDENCE_ORIGIN:
            raise ValueError(
                "an inherited interval end must be authoritative Pass 3"
            )
        if interval.end_parent_coordinate != closing.parent_coordinate:
            raise ValueError(
                "the interval end contradicts the closing boundary"
            )
        self._validate_interval_start(
            self.last_observed_equivalent_coordinate
        )
        validate_interval_against_coverage(
            interval,
            self.coverage.common_comparable,
            first_local_mismatch=None,
            authoritative_parent_coordinate=closing.parent_coordinate,
        )

    def _validate_interval_start(
        self, last_equivalent: Optional[IntraLayerCoordinate]
    ) -> None:
        interval = self.suspect_interval
        if interval.start_kind == "selected_layer_entry":
            if last_equivalent is not None:
                raise ValueError(
                    "an entry interval cannot follow an equivalent coordinate"
                )
            return
        if interval.start_local_coordinate != last_equivalent:
            raise ValueError(
                "the interval start must be the last equivalent coordinate"
            )

    # -- non-comparison results --------------------------------------------

    def _validate_noncomparison_result(self) -> None:
        if self.coverage is None:
            return
        self._validate_target_identity()
        if (
            self.status
            == Pass4Status.INSUFFICIENT_COMMON_INTRA_LAYER_COVERAGE
            and self.coverage.common_captured
        ):
            raise ValueError(
                "insufficient common coverage cannot report common captured "
                "coordinates"
            )
        if (
            self.status == Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT
            and not self.coverage.common_captured
        ):
            raise ValueError(
                "an alignment failure requires common captured coverage"
            )

    def _validate_secondary_reasons(self) -> None:
        if Pass4ReasonCode.ASYMMETRIC_COVERAGE_RETAINED not in (
            self.reason_codes
        ):
            return
        if self.coverage is None or not self.coverage.asymmetric:
            raise ValueError(
                "retained asymmetric coverage requires one-sided coordinates"
            )
