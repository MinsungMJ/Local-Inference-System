"""Frozen Pass 4 intra-layer contract primitives.

This module is the P4-1 contract freeze, not a Pass 4 implementation.  It
provides immutable coordinate/coverage/interval values, the local enum algebra,
and a Python reference encoder for committed digest vectors.  It deliberately
does not parse runtime artifacts, localize mismatches, serialize Pass 4
results, rerun Pass 3, or expose a Pass 4 execution entry point.
"""

from __future__ import annotations

import hashlib
import re
import struct
from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import Any, Optional

from .pass3_model import CheckpointCoordinate, CoverageState


SCHEMA = "lis.execution_artifact/v1"
OUTER_TRACE_KIND = "layer_trace"
RESULT_KIND = "intra_layer_localization"
CONTRACT_VERSION = "differential_verification_contract_v1"
CONTRACT_NAMESPACE = "coverage_scoped_intra_layer_localization"

INTRA_LAYER_LAYOUT_NAME = "llama_intra_layer_summary"
INTRA_LAYER_LAYOUT_VERSION = 1
STAGE_TAXONOMY = "lis.llama.intra_layer_stages/v1"
MODEL_FAMILY = "llama3_decoder"
PHASE = "decode"
ORDERING_SEMANTICS = "runtime_step_layer_stage_ordinal"
DUPLICATE_COORDINATE_POLICY = "reject_artifact_before_write"

INTRA_LAYER_CHECKPOINT_LAYOUT_FIELD = "intra_layer_checkpoint_layout"
INTRA_LAYER_TRACE_FIELD = "intra_layer_trace"
DIAGNOSTIC_CAPTURE_PROFILE = "semantic_layer_and_intra_v1"

INHERITED_BOUNDARY_ID = "parent:layer_output"
INHERITED_BOUNDARY_EVIDENCE_ORIGIN = "authoritative_pass3"
LOCAL_EVIDENCE_ORIGIN = "pass4_local"

DIGEST_ALGORITHM = "sha256"
DIGEST_VERSION = "lis.checkpoint.intra_layer.fp32le/v1"
DIGEST_DOMAIN_TAG = "LIS_INTRA_LAYER_CHECKPOINT_DIGEST"
DIGEST_OBSERVED_DTYPE = "fp32"
DIGEST_BYTE_ORDER = "little"
DIGEST_CANONICALIZATION = (
    "ieee754-binary32-le;canonical-qnan;preserve-signed-zero"
)
DIGEST_CANONICAL_QNAN_BITS = 0x7FC00000
PASS3_DIGEST_VERSION = "lis.checkpoint.fp32le/v1"

EVIDENCE_LEVEL = "tier1_bounded_digest"
UINT64_MAX = (1 << 64) - 1
UINT32_MAX = (1 << 32) - 1


@dataclass(frozen=True)
class IntraLayerStage:
    stage_order: int
    stage_id: str
    tensor_role: str


INTRA_LAYER_STAGES = (
    IntraLayerStage(0, "layer_input", "layer_input"),
    IntraLayerStage(
        1, "attention_norm_output", "attention_norm_output"
    ),
    IntraLayerStage(
        2, "query_projection_output", "query_projection_output"
    ),
    IntraLayerStage(
        3, "key_projection_output", "key_projection_output"
    ),
    IntraLayerStage(
        4, "value_projection_output", "value_projection_output"
    ),
    IntraLayerStage(5, "rope_query_output", "rope_query_output"),
    IntraLayerStage(6, "rope_key_output", "rope_key_output"),
    IntraLayerStage(7, "attention_scores", "attention_scores"),
    IntraLayerStage(
        8, "attention_probabilities", "attention_probabilities"
    ),
    IntraLayerStage(9, "attention_context", "attention_context"),
    IntraLayerStage(
        10,
        "attention_output_projection",
        "attention_output_projection",
    ),
    IntraLayerStage(
        11, "post_attention_residual", "post_attention_residual"
    ),
    IntraLayerStage(12, "mlp_norm_output", "mlp_norm_output"),
    IntraLayerStage(
        13, "mlp_gate_projection", "mlp_gate_projection"
    ),
    IntraLayerStage(14, "mlp_up_projection", "mlp_up_projection"),
    IntraLayerStage(
        15, "mlp_gated_activation", "mlp_gated_activation"
    ),
    IntraLayerStage(
        16, "mlp_down_projection", "mlp_down_projection"
    ),
)
STAGE_BY_ID = {stage.stage_id: stage for stage in INTRA_LAYER_STAGES}
STAGE_IDS = tuple(stage.stage_id for stage in INTRA_LAYER_STAGES)


NONCLAIMS = {
    "numeric_divergence_confirmed": False,
    "true_first_divergence_confirmed": False,
    "root_cause_identified": False,
    "tensor_equality_proved": False,
    "complete_intra_layer_coverage_proved": False,
    "operation_level_localization_performed": False,
    "exhaustive_confirmation_performed": False,
    "automatic_frozen_success_mapping": False,
}


class UnsupportedIntraLayerLayoutError(ValueError):
    """Requested coordinates do not identify the exact frozen v1 layout."""

    status = "unsupported_intra_layer_layout"


class Pass4Status(str, Enum):
    OBSERVABLE_INTRA_LAYER_MISMATCH_FOUND = (
        "observable_intra_layer_mismatch_found"
    )
    MISMATCH_BOUNDED_TO_INHERITED_CLOSING_BOUNDARY = (
        "mismatch_bounded_to_inherited_closing_boundary"
    )
    NOT_APPLICABLE = "not_applicable"
    COMPARISON_BLOCKED_BY_PASS3 = "comparison_blocked_by_pass3"
    INSUFFICIENT_COMMON_INTRA_LAYER_COVERAGE = (
        "insufficient_common_intra_layer_coverage"
    )
    SOURCE_BINDING_INCONSISTENT = "source_binding_inconsistent"
    CHECKPOINT_ALIGNMENT_INCONSISTENT = (
        "checkpoint_alignment_inconsistent"
    )
    CHECKPOINT_SUMMARY_MALFORMED = "checkpoint_summary_malformed"
    COMPARISON_POLICY_UNAVAILABLE = "comparison_policy_unavailable"
    UNSUPPORTED_PARENT = "unsupported_parent"
    UNSUPPORTED_INTRA_LAYER_LAYOUT = "unsupported_intra_layer_layout"
    PARENT_REVALIDATION_INCONSISTENT = (
        "parent_revalidation_inconsistent"
    )


class Pass4Disposition(str, Enum):
    BLOCKED = "blocked"
    NOT_APPLICABLE = "not_applicable"
    UNSUPPORTED = "unsupported"
    SUSPECT_INTERVAL_AVAILABLE = "suspect_interval_available"
    INCONCLUSIVE = "inconclusive"


class Pass4ReasonCode(str, Enum):
    LOCAL_DIGEST_MISMATCH = "pass4.local_digest_mismatch"
    NO_LOCAL_MISMATCH_BEFORE_INHERITED_BOUNDARY = (
        "pass4.no_local_mismatch_before_inherited_boundary"
    )
    ASYMMETRIC_COVERAGE_RETAINED = (
        "pass4.asymmetric_coverage_retained"
    )
    PARENT_HAS_NO_OBSERVED_MISMATCH = (
        "pass4.parent_has_no_observed_mismatch"
    )
    PARENT_STATUS_BLOCKED = "pass4.parent_status_blocked"
    PARENT_ARTIFACT_MALFORMED = "pass4.parent_artifact_malformed"
    PARENT_TYPED_ARTIFACT_INCOHERENT = (
        "pass4.parent_typed_artifact_incoherent"
    )
    PARENT_COMPARISONS_TRUNCATED = (
        "pass4.parent_comparisons_truncated"
    )
    PARENT_LOCALIZATION_INCOHERENT = (
        "pass4.parent_localization_incoherent"
    )
    PARENT_UPSTREAM_EVIDENCE_INCOHERENT = (
        "pass4.parent_upstream_evidence_incoherent"
    )
    DISCOVERY_REBOUND_LAYER_CHANGED = (
        "pass4.discovery_rebound_layer_changed"
    )
    DISCOVERY_REBOUND_SEMANTICS_CHANGED = (
        "pass4.discovery_rebound_semantics_changed"
    )
    TRACE_SHA_MISMATCH = "pass4.trace_sha_mismatch"
    SOURCE_ROLE_MISMATCH = "pass4.source_role_mismatch"
    RUN_REPORT_BINDING_MISMATCH = (
        "pass4.run_report_binding_mismatch"
    )
    ARTIFACT_SET_BINDING_MISMATCH = (
        "pass4.artifact_set_binding_mismatch"
    )
    SEMANTIC_MANIFEST_BINDING_MISMATCH = (
        "pass4.semantic_manifest_binding_mismatch"
    )
    RUNTIME_CAPTURE_IDENTITY_MISMATCH = (
        "pass4.runtime_capture_identity_mismatch"
    )
    NO_COMMON_CAPTURED_COORDINATES = (
        "pass4.no_common_captured_coordinates"
    )
    STEP_ALIGNMENT_MISMATCH = "pass4.step_alignment_mismatch"
    LAYER_ALIGNMENT_MISMATCH = "pass4.layer_alignment_mismatch"
    PHASE_OR_POSITION_ALIGNMENT_MISMATCH = (
        "pass4.phase_or_position_alignment_mismatch"
    )
    SHAPE_OR_COUNT_ALIGNMENT_MISMATCH = (
        "pass4.shape_or_count_alignment_mismatch"
    )
    DTYPE_OR_PRECISION_ALIGNMENT_MISMATCH = (
        "pass4.dtype_or_precision_alignment_mismatch"
    )
    STAGE_ROLE_OR_ORDER_ALIGNMENT_MISMATCH = (
        "pass4.stage_role_or_order_alignment_mismatch"
    )
    COVERAGE_PARTITION_MALFORMED = (
        "pass4.coverage_partition_malformed"
    )
    DUPLICATE_OR_OUT_OF_ORDER_COORDINATE = (
        "pass4.duplicate_or_out_of_order_coordinate"
    )
    SUMMARY_FIELD_MALFORMED = "pass4.summary_field_malformed"
    DIGEST_FIELD_MALFORMED = "pass4.digest_field_malformed"
    PROHIBITED_PAYLOAD_PRESENT = "pass4.prohibited_payload_present"
    DIGEST_CONTRACT_UNKNOWN = "pass4.digest_contract_unknown"
    PARENT_CONTRACT_OR_FAMILY_UNSUPPORTED = (
        "pass4.parent_contract_or_family_unsupported"
    )
    PARENT_PHASE_UNSUPPORTED = "pass4.parent_phase_unsupported"
    PARENT_DIGEST_POLICY_UNSUPPORTED = (
        "pass4.parent_digest_policy_unsupported"
    )
    INTRA_LAYOUT_OR_TAXONOMY_UNSUPPORTED = (
        "pass4.intra_layout_or_taxonomy_unsupported"
    )
    REQUESTED_STAGE_SET_UNSUPPORTED = (
        "pass4.requested_stage_set_unsupported"
    )


STATUS_TO_DISPOSITION = {
    Pass4Status.OBSERVABLE_INTRA_LAYER_MISMATCH_FOUND:
        Pass4Disposition.SUSPECT_INTERVAL_AVAILABLE,
    Pass4Status.MISMATCH_BOUNDED_TO_INHERITED_CLOSING_BOUNDARY:
        Pass4Disposition.SUSPECT_INTERVAL_AVAILABLE,
    Pass4Status.NOT_APPLICABLE: Pass4Disposition.NOT_APPLICABLE,
    Pass4Status.COMPARISON_BLOCKED_BY_PASS3: Pass4Disposition.BLOCKED,
    Pass4Status.INSUFFICIENT_COMMON_INTRA_LAYER_COVERAGE:
        Pass4Disposition.BLOCKED,
    Pass4Status.SOURCE_BINDING_INCONSISTENT: Pass4Disposition.BLOCKED,
    Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT:
        Pass4Disposition.BLOCKED,
    Pass4Status.CHECKPOINT_SUMMARY_MALFORMED: Pass4Disposition.BLOCKED,
    Pass4Status.COMPARISON_POLICY_UNAVAILABLE:
        Pass4Disposition.BLOCKED,
    Pass4Status.UNSUPPORTED_PARENT: Pass4Disposition.UNSUPPORTED,
    Pass4Status.UNSUPPORTED_INTRA_LAYER_LAYOUT:
        Pass4Disposition.UNSUPPORTED,
    Pass4Status.PARENT_REVALIDATION_INCONSISTENT:
        Pass4Disposition.INCONCLUSIVE,
}


REASON_ALLOWED_STATUSES = {
    Pass4ReasonCode.LOCAL_DIGEST_MISMATCH: (
        Pass4Status.OBSERVABLE_INTRA_LAYER_MISMATCH_FOUND,
    ),
    Pass4ReasonCode.NO_LOCAL_MISMATCH_BEFORE_INHERITED_BOUNDARY: (
        Pass4Status.MISMATCH_BOUNDED_TO_INHERITED_CLOSING_BOUNDARY,
    ),
    Pass4ReasonCode.ASYMMETRIC_COVERAGE_RETAINED: (
        Pass4Status.OBSERVABLE_INTRA_LAYER_MISMATCH_FOUND,
        Pass4Status.MISMATCH_BOUNDED_TO_INHERITED_CLOSING_BOUNDARY,
    ),
    Pass4ReasonCode.PARENT_HAS_NO_OBSERVED_MISMATCH: (
        Pass4Status.NOT_APPLICABLE,
    ),
    Pass4ReasonCode.PARENT_STATUS_BLOCKED: (
        Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
    ),
    Pass4ReasonCode.PARENT_ARTIFACT_MALFORMED: (
        Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
    ),
    Pass4ReasonCode.PARENT_TYPED_ARTIFACT_INCOHERENT: (
        Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
    ),
    Pass4ReasonCode.PARENT_COMPARISONS_TRUNCATED: (
        Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
    ),
    Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT: (
        Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
    ),
    Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT: (
        Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
    ),
    Pass4ReasonCode.DISCOVERY_REBOUND_LAYER_CHANGED: (
        Pass4Status.PARENT_REVALIDATION_INCONSISTENT,
    ),
    Pass4ReasonCode.DISCOVERY_REBOUND_SEMANTICS_CHANGED: (
        Pass4Status.PARENT_REVALIDATION_INCONSISTENT,
    ),
    Pass4ReasonCode.TRACE_SHA_MISMATCH: (
        Pass4Status.SOURCE_BINDING_INCONSISTENT,
    ),
    Pass4ReasonCode.SOURCE_ROLE_MISMATCH: (
        Pass4Status.SOURCE_BINDING_INCONSISTENT,
    ),
    Pass4ReasonCode.RUN_REPORT_BINDING_MISMATCH: (
        Pass4Status.SOURCE_BINDING_INCONSISTENT,
    ),
    Pass4ReasonCode.ARTIFACT_SET_BINDING_MISMATCH: (
        Pass4Status.SOURCE_BINDING_INCONSISTENT,
    ),
    Pass4ReasonCode.SEMANTIC_MANIFEST_BINDING_MISMATCH: (
        Pass4Status.SOURCE_BINDING_INCONSISTENT,
    ),
    Pass4ReasonCode.RUNTIME_CAPTURE_IDENTITY_MISMATCH: (
        Pass4Status.SOURCE_BINDING_INCONSISTENT,
    ),
    Pass4ReasonCode.NO_COMMON_CAPTURED_COORDINATES: (
        Pass4Status.INSUFFICIENT_COMMON_INTRA_LAYER_COVERAGE,
    ),
    Pass4ReasonCode.STEP_ALIGNMENT_MISMATCH: (
        Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT,
    ),
    Pass4ReasonCode.LAYER_ALIGNMENT_MISMATCH: (
        Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT,
    ),
    Pass4ReasonCode.PHASE_OR_POSITION_ALIGNMENT_MISMATCH: (
        Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT,
    ),
    Pass4ReasonCode.SHAPE_OR_COUNT_ALIGNMENT_MISMATCH: (
        Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT,
    ),
    Pass4ReasonCode.DTYPE_OR_PRECISION_ALIGNMENT_MISMATCH: (
        Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT,
    ),
    Pass4ReasonCode.STAGE_ROLE_OR_ORDER_ALIGNMENT_MISMATCH: (
        Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT,
    ),
    Pass4ReasonCode.COVERAGE_PARTITION_MALFORMED: (
        Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
    ),
    Pass4ReasonCode.DUPLICATE_OR_OUT_OF_ORDER_COORDINATE: (
        Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
    ),
    Pass4ReasonCode.SUMMARY_FIELD_MALFORMED: (
        Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
    ),
    Pass4ReasonCode.DIGEST_FIELD_MALFORMED: (
        Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
    ),
    Pass4ReasonCode.PROHIBITED_PAYLOAD_PRESENT: (
        Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
    ),
    Pass4ReasonCode.DIGEST_CONTRACT_UNKNOWN: (
        Pass4Status.COMPARISON_POLICY_UNAVAILABLE,
    ),
    Pass4ReasonCode.PARENT_CONTRACT_OR_FAMILY_UNSUPPORTED: (
        Pass4Status.UNSUPPORTED_PARENT,
    ),
    Pass4ReasonCode.PARENT_PHASE_UNSUPPORTED: (
        Pass4Status.UNSUPPORTED_PARENT,
    ),
    Pass4ReasonCode.PARENT_DIGEST_POLICY_UNSUPPORTED: (
        Pass4Status.UNSUPPORTED_PARENT,
    ),
    Pass4ReasonCode.INTRA_LAYOUT_OR_TAXONOMY_UNSUPPORTED: (
        Pass4Status.UNSUPPORTED_INTRA_LAYER_LAYOUT,
    ),
    Pass4ReasonCode.REQUESTED_STAGE_SET_UNSUPPORTED: (
        Pass4Status.UNSUPPORTED_INTRA_LAYER_LAYOUT,
    ),
}


PRIMARY_REASONS = {
    status: tuple(
        reason
        for reason, allowed in REASON_ALLOWED_STATUSES.items()
        if status in allowed
        and reason != Pass4ReasonCode.ASYMMETRIC_COVERAGE_RETAINED
    )
    for status in Pass4Status
}


class Pass3ParentRole(str, Enum):
    DISCOVERY_PASS3A = "pass3a_discovery_provenance"
    AUTHORITATIVE_PASS3B = "pass3b_authoritative_parent"


class Pass3ParentClassification(str, Enum):
    ELIGIBLE = "eligible"
    NOT_APPLICABLE = "not_applicable"
    COMPARISON_BLOCKED_BY_PASS3 = "comparison_blocked_by_pass3"
    UNSUPPORTED_PARENT = "unsupported_parent"
    PARENT_REVALIDATION_INCONSISTENT = (
        "parent_revalidation_inconsistent"
    )


def _strict_int(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer without coercion")
    if value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} exceeds unsigned 64-bit range")
    return value


def _strict_tuple(value: Any, label: str) -> tuple[Any, ...]:
    if not isinstance(value, tuple):
        raise ValueError(f"{label} must be an immutable tuple")
    return value


def _strict_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    encoded = value.encode("utf-8")
    if len(encoded) > UINT64_MAX:
        raise ValueError(f"{label} UTF-8 byte length exceeds u64")
    return value


@dataclass(frozen=True)
class IntraLayerCoordinate:
    runtime_checkpoint_step: int
    layer_index: int
    stage_id: str
    tensor_role: str
    batch_index: int
    sequence_index: int
    token_position: int
    stage_order: int
    execution_ordinal: int

    def __post_init__(self):
        _strict_int(
            self.runtime_checkpoint_step,
            "runtime_checkpoint_step",
            minimum=1,
        )
        _strict_int(self.layer_index, "layer_index")
        _strict_int(self.token_position, "token_position")
        _strict_int(self.batch_index, "batch_index")
        _strict_int(self.sequence_index, "sequence_index")
        _strict_int(self.stage_order, "stage_order")
        _strict_int(self.execution_ordinal, "execution_ordinal")
        _strict_string(self.stage_id, "stage_id")
        _strict_string(self.tensor_role, "tensor_role")
        if self.batch_index != 0 or self.sequence_index != 0:
            raise ValueError("v1 requires batch_index == sequence_index == 0")
        stage = STAGE_BY_ID.get(self.stage_id)
        if stage is None:
            raise ValueError("unknown intra-layer stage_id")
        if self.tensor_role != stage.tensor_role:
            raise ValueError("stage_id and tensor_role violate the taxonomy")
        if self.stage_order != stage.stage_order:
            raise ValueError("stage_order violates the frozen taxonomy")
        if self.execution_ordinal != self.stage_order:
            raise ValueError("execution_ordinal must equal stage_order")

    @property
    def logical_key(self) -> tuple[object, ...]:
        return (
            self.runtime_checkpoint_step,
            self.layer_index,
            self.stage_id,
            self.tensor_role,
            self.batch_index,
            self.sequence_index,
            self.token_position,
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


COORDINATE_FIELDS = tuple(IntraLayerCoordinate.__dataclass_fields__)


def coordinate_from_mapping(raw: Any) -> IntraLayerCoordinate:
    """Strict fixture/schema conversion with no defaulting or coercion."""
    if not isinstance(raw, dict) or set(raw) != set(COORDINATE_FIELDS):
        raise ValueError("coordinate must contain exactly the frozen fields")
    return IntraLayerCoordinate(**{field: raw[field] for field in COORDINATE_FIELDS})


def coordinate_to_mapping(value: IntraLayerCoordinate) -> dict[str, Any]:
    if not isinstance(value, IntraLayerCoordinate):
        raise TypeError("value must be an IntraLayerCoordinate")
    return {field: getattr(value, field) for field in COORDINATE_FIELDS}


def requested_coordinates(
    runtime_checkpoint_step: int,
    layer_index: int,
    token_position: int,
) -> tuple[IntraLayerCoordinate, ...]:
    """Return the exact ordered 17-coordinate request for one target."""
    return tuple(
        IntraLayerCoordinate(
            runtime_checkpoint_step,
            layer_index,
            stage.stage_id,
            stage.tensor_role,
            0,
            0,
            token_position,
            stage.stage_order,
            stage.stage_order,
        )
        for stage in INTRA_LAYER_STAGES
    )


def validate_coordinate_sequence(
    coordinates: tuple[IntraLayerCoordinate, ...],
    label: str,
) -> None:
    """Reject malformed, duplicate, or non-increasing input without sorting."""
    _strict_tuple(coordinates, label)
    seen = set()
    previous = None
    for coordinate in coordinates:
        if not isinstance(coordinate, IntraLayerCoordinate):
            raise ValueError(f"{label} contains an invalid coordinate")
        if coordinate.logical_key in seen:
            raise ValueError(f"{label} contains a duplicate coordinate")
        if previous is not None and coordinate.order_key <= previous:
            raise ValueError(f"{label} is out of contracted order")
        seen.add(coordinate.logical_key)
        previous = coordinate.order_key


def validate_requested_coordinates(
    coordinates: tuple[IntraLayerCoordinate, ...],
) -> None:
    """Require one exact target and the complete ordered v1 stage request."""
    validate_coordinate_sequence(coordinates, "requested_coordinates")
    if not coordinates:
        raise UnsupportedIntraLayerLayoutError(
            "requested coordinates omit the frozen stage list"
        )
    first = coordinates[0]
    expected = requested_coordinates(
        first.runtime_checkpoint_step,
        first.layer_index,
        first.token_position,
    )
    if coordinates != expected:
        raise UnsupportedIntraLayerLayoutError(
            "requested coordinates are not the exact ordered 17-stage list"
        )


@dataclass(frozen=True)
class MissingIntraLayerCoordinate:
    coordinate: IntraLayerCoordinate
    state: CoverageState
    detail: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.coordinate, IntraLayerCoordinate):
            raise ValueError("missing coordinate is malformed")
        if not isinstance(self.state, CoverageState):
            raise ValueError("missing coordinate state is malformed")
        if self.state == CoverageState.CAPTURED:
            raise ValueError("captured is not a missing-coordinate state")
        if self.detail is not None and not isinstance(self.detail, str):
            raise ValueError("missing coordinate detail must be a string")


@dataclass(frozen=True)
class IntraLayerSideCoverage:
    requested_coordinates: tuple[IntraLayerCoordinate, ...]
    captured_coordinates: tuple[IntraLayerCoordinate, ...]
    missing_coordinates: tuple[MissingIntraLayerCoordinate, ...]

    def __post_init__(self):
        validate_requested_coordinates(self.requested_coordinates)
        validate_coordinate_sequence(
            self.captured_coordinates, "captured_coordinates"
        )
        _strict_tuple(self.missing_coordinates, "missing_coordinates")
        missing_local = tuple(item.coordinate for item in self.missing_coordinates)
        validate_coordinate_sequence(missing_local, "missing_coordinates")
        requested_keys = tuple(
            item.logical_key for item in self.requested_coordinates
        )
        captured_keys = tuple(
            item.logical_key for item in self.captured_coordinates
        )
        missing_keys = tuple(item.logical_key for item in missing_local)
        requested_set = set(requested_keys)
        captured_set = set(captured_keys)
        missing_set = set(missing_keys)
        if not captured_set.issubset(requested_set):
            raise ValueError("captured coordinates must be requested")
        if captured_set & missing_set:
            raise ValueError("captured and missing coordinates overlap")
        if captured_set | missing_set != requested_set:
            raise ValueError("captured and missing must partition requested")
        if captured_keys != tuple(
            key for key in requested_keys if key in captured_set
        ):
            raise ValueError("captured coordinates must preserve requested order")
        complement = tuple(
            key for key in requested_keys if key not in captured_set
        )
        if missing_keys != complement:
            raise ValueError("missing coordinates must be the ordered complement")


@dataclass(frozen=True)
class IntraLayerCoverageAnalysis:
    requested_coordinates: tuple[IntraLayerCoordinate, ...]
    captured_coordinates: tuple[IntraLayerCoordinate, ...]
    missing_coordinates: tuple[MissingIntraLayerCoordinate, ...]
    common_captured: tuple[IntraLayerCoordinate, ...]
    common_comparable: tuple[IntraLayerCoordinate, ...]
    reference_only: tuple[IntraLayerCoordinate, ...]
    candidate_only: tuple[IntraLayerCoordinate, ...]
    reference_missing: tuple[MissingIntraLayerCoordinate, ...]
    candidate_missing: tuple[MissingIntraLayerCoordinate, ...]


def analyze_coverage(
    reference: IntraLayerSideCoverage,
    candidate: IntraLayerSideCoverage,
    *,
    common_comparable: Optional[tuple[IntraLayerCoordinate, ...]] = None,
) -> IntraLayerCoverageAnalysis:
    """Apply only frozen set algebra; no alignment or digest decisions."""
    if not isinstance(reference, IntraLayerSideCoverage):
        raise TypeError("reference must be IntraLayerSideCoverage")
    if not isinstance(candidate, IntraLayerSideCoverage):
        raise TypeError("candidate must be IntraLayerSideCoverage")
    if reference.requested_coordinates != candidate.requested_coordinates:
        raise UnsupportedIntraLayerLayoutError(
            "reference and candidate requested stages differ"
        )
    ref_keys = {item.logical_key for item in reference.captured_coordinates}
    cand_keys = {item.logical_key for item in candidate.captured_coordinates}
    common = tuple(
        item
        for item in reference.captured_coordinates
        if item.logical_key in cand_keys
    )
    ref_only = tuple(
        item
        for item in reference.captured_coordinates
        if item.logical_key not in cand_keys
    )
    cand_only = tuple(
        item
        for item in candidate.captured_coordinates
        if item.logical_key not in ref_keys
    )
    comparable = common if common_comparable is None else common_comparable
    validate_coordinate_sequence(comparable, "common_comparable")
    common_keys = {item.logical_key for item in common}
    if any(item.logical_key not in common_keys for item in comparable):
        raise ValueError("common_comparable must be a subset of common_captured")
    comparable_keys = {item.logical_key for item in comparable}
    if comparable != tuple(
        item for item in common if item.logical_key in comparable_keys
    ):
        raise ValueError("common_comparable must preserve common order")
    return IntraLayerCoverageAnalysis(
        reference.requested_coordinates,
        reference.captured_coordinates,
        reference.missing_coordinates,
        common,
        comparable,
        ref_only,
        cand_only,
        reference.missing_coordinates,
        candidate.missing_coordinates,
    )


@dataclass(frozen=True)
class Pass4SuspectInterval:
    start_kind: str
    start_local_coordinate: Optional[IntraLayerCoordinate]
    start_inclusive: bool
    end_kind: str
    end_local_coordinate: Optional[IntraLayerCoordinate]
    end_parent_coordinate: Optional[CheckpointCoordinate]
    end_evidence_origin: str
    end_inclusive: bool
    missing_local_stage_ids: tuple[str, ...]
    notation: str

    def __post_init__(self):
        if self.start_kind not in ("selected_layer_entry", "local_coordinate"):
            raise ValueError("unsupported interval start_kind")
        if self.end_kind not in (
            "local_coordinate",
            "inherited_parent_boundary",
        ):
            raise ValueError("unsupported interval end_kind")
        if self.start_kind == "selected_layer_entry":
            if self.start_local_coordinate is not None or not self.start_inclusive:
                raise ValueError("selected layer entry must be an inclusive virtual start")
        elif (
            not isinstance(self.start_local_coordinate, IntraLayerCoordinate)
            or self.start_inclusive
        ):
            raise ValueError("local interval start must be present and exclusive")
        if not self.end_inclusive:
            raise ValueError("interval end must be inclusive")
        if self.end_kind == "local_coordinate":
            if (
                not isinstance(
                    self.end_local_coordinate, IntraLayerCoordinate
                )
                or self.end_parent_coordinate is not None
                or self.end_evidence_origin != LOCAL_EVIDENCE_ORIGIN
            ):
                raise ValueError("local interval end is incoherent")
        elif (
            self.end_local_coordinate is not None
            or not isinstance(self.end_parent_coordinate, CheckpointCoordinate)
            or self.end_parent_coordinate.tensor_role != "layer_output"
            or self.end_evidence_origin
            != INHERITED_BOUNDARY_EVIDENCE_ORIGIN
        ):
            raise ValueError("inherited interval end is incoherent")
        _strict_tuple(
            self.missing_local_stage_ids, "missing_local_stage_ids"
        )
        if (
            len(self.missing_local_stage_ids)
            != len(set(self.missing_local_stage_ids))
            or any(stage not in STAGE_BY_ID for stage in self.missing_local_stage_ids)
            or tuple(
                sorted(
                    self.missing_local_stage_ids,
                    key=lambda item: STAGE_BY_ID[item].stage_order,
                )
            )
            != self.missing_local_stage_ids
        ):
            raise ValueError("missing local stage IDs are invalid or unordered")
        expected_notation = _interval_notation(self)
        if self.notation != expected_notation:
            raise ValueError("interval notation is not canonical")
        if (
            self.start_local_coordinate is not None
            and self.end_local_coordinate is not None
            and (
                self.start_local_coordinate.runtime_checkpoint_step,
                self.start_local_coordinate.layer_index,
                self.start_local_coordinate.token_position,
            )
            != (
                self.end_local_coordinate.runtime_checkpoint_step,
                self.end_local_coordinate.layer_index,
                self.end_local_coordinate.token_position,
            )
        ):
            raise ValueError("local interval endpoints target different coordinates")
        if (
            self.start_local_coordinate is not None
            and self.end_local_coordinate is not None
            and self.start_local_coordinate.stage_order
            >= self.end_local_coordinate.stage_order
        ):
            raise ValueError("local interval endpoints are out of order")


def _interval_notation(interval: Pass4SuspectInterval) -> str:
    if interval.start_kind == "selected_layer_entry":
        start = "[selected_layer_entry, "
    else:
        start = f"({interval.start_local_coordinate.stage_id}, "
    if interval.end_kind == "local_coordinate":
        end = interval.end_local_coordinate.stage_id
    else:
        end = INHERITED_BOUNDARY_ID
    return f"{start}{end}]"


def validate_interval_against_coverage(
    interval: Pass4SuspectInterval,
    common_comparable: tuple[IntraLayerCoordinate, ...],
    *,
    first_local_mismatch: Optional[IntraLayerCoordinate],
    authoritative_parent_coordinate: CheckpointCoordinate,
) -> None:
    """Validate endpoint identity and the exact missing-stage list."""
    if not isinstance(interval, Pass4SuspectInterval):
        raise TypeError("interval must be a Pass4SuspectInterval")
    validate_coordinate_sequence(common_comparable, "common_comparable")
    if not isinstance(authoritative_parent_coordinate, CheckpointCoordinate):
        raise TypeError("authoritative parent coordinate is required")
    if interval.end_kind == "local_coordinate":
        if (
            first_local_mismatch is None
            or interval.end_local_coordinate != first_local_mismatch
            or not any(
                item.logical_key == first_local_mismatch.logical_key
                for item in common_comparable
            )
        ):
            raise ValueError("local interval does not end at the first mismatch")
        end_order = first_local_mismatch.stage_order
    else:
        if (
            first_local_mismatch is not None
            or interval.end_parent_coordinate
            != authoritative_parent_coordinate
        ):
            raise ValueError("inherited interval does not bind Pass 3B exactly")
        end_order = len(INTRA_LAYER_STAGES)
    start_order = (
        -1
        if interval.start_local_coordinate is None
        else interval.start_local_coordinate.stage_order
    )
    common_orders = {
        coordinate.stage_order for coordinate in common_comparable
    }
    expected_missing = tuple(
        stage.stage_id
        for stage in INTRA_LAYER_STAGES
        if start_order < stage.stage_order <= end_order
        and stage.stage_order not in common_orders
    )
    if interval.missing_local_stage_ids != expected_missing:
        raise ValueError("missing stage list contradicts common comparable coverage")


_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_ARTIFACT_SET_ID = re.compile(r"^aset1:[0-9a-f]{32}$")


@dataclass(frozen=True)
class ParentSourceIdentity:
    run_report_sha256: str
    layer_trace_sha256: str
    semantic_manifest_sha256: str
    artifact_set_id: str

    def __post_init__(self):
        for label in (
            "run_report_sha256",
            "layer_trace_sha256",
            "semantic_manifest_sha256",
        ):
            value = getattr(self, label)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"{label} must be a canonical SHA-256 identity")
        if (
            not isinstance(self.artifact_set_id, str)
            or not _ARTIFACT_SET_ID.fullmatch(self.artifact_set_id)
        ):
            raise ValueError("artifact_set_id is malformed")


@dataclass(frozen=True)
class Pass3ParentBinding:
    role: Pass3ParentRole
    canonical_pass3_artifact_sha256: str
    pass2_artifact_sha256: str
    reference: ParentSourceIdentity
    candidate: ParentSourceIdentity
    authorizes_pass4_evidence: bool

    def __post_init__(self):
        if not isinstance(self.role, Pass3ParentRole):
            raise ValueError("Pass 3 parent role is malformed")
        for label in (
            "canonical_pass3_artifact_sha256",
            "pass2_artifact_sha256",
        ):
            value = getattr(self, label)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"{label} must be a canonical SHA-256 identity")
        if not isinstance(self.reference, ParentSourceIdentity) or not isinstance(
            self.candidate, ParentSourceIdentity
        ):
            raise ValueError("both role-bound parent source identities are required")
        if not isinstance(self.authorizes_pass4_evidence, bool):
            raise ValueError("parent authorization must be an explicit boolean")
        expected_authority = (
            self.role == Pass3ParentRole.AUTHORITATIVE_PASS3B
        )
        if self.authorizes_pass4_evidence != expected_authority:
            raise ValueError("Pass 3A is discovery-only and Pass 3B is authoritative")


def validate_pass3_parent_pair(
    discovery: Pass3ParentBinding,
    authoritative: Pass3ParentBinding,
) -> None:
    if not isinstance(discovery, Pass3ParentBinding) or not isinstance(
        authoritative, Pass3ParentBinding
    ):
        raise TypeError("both Pass 3 parent bindings are required")
    if discovery.role != Pass3ParentRole.DISCOVERY_PASS3A:
        raise ValueError("first parent must be Pass 3A discovery provenance")
    if authoritative.role != Pass3ParentRole.AUTHORITATIVE_PASS3B:
        raise ValueError("second parent must be authoritative Pass 3B")


def validate_status_algebra(
    status: Pass4Status,
    disposition: Pass4Disposition,
    reason_codes: tuple[Pass4ReasonCode, ...],
) -> None:
    if not isinstance(status, Pass4Status):
        raise ValueError("unknown Pass 4 status")
    if disposition != STATUS_TO_DISPOSITION[status]:
        raise ValueError("Pass 4 status and disposition are incoherent")
    _strict_tuple(reason_codes, "reason_codes")
    if (
        not reason_codes
        or len(reason_codes) > 32
        or len(reason_codes) != len(set(reason_codes))
    ):
        raise ValueError("reason codes must be nonempty, unique, and bounded")
    for reason in reason_codes:
        if (
            not isinstance(reason, Pass4ReasonCode)
            or status not in REASON_ALLOWED_STATUSES[reason]
        ):
            raise ValueError("reason code is not allowed for the status")
    if reason_codes[0] not in PRIMARY_REASONS[status]:
        raise ValueError("the first reason is not a valid primary reason")


def validate_nonclaims(value: Any) -> None:
    if not isinstance(value, dict) or value != NONCLAIMS:
        raise ValueError("every frozen Pass 4 nonclaim must be present and false")


def _u64(value: Any, label: str) -> bytes:
    return struct.pack(
        "<Q", _strict_int(value, label, maximum=UINT64_MAX)
    )


def _framed(value: Any, label: str) -> bytes:
    encoded = _strict_string(value, label).encode("utf-8")
    return _u64(len(encoded), f"{label} length") + encoded


def _shape_product(shape: tuple[int, ...]) -> int:
    _strict_tuple(shape, "shape")
    if not shape:
        raise ValueError("rank zero and empty tensors are rejected")
    count = 1
    for dimension in shape:
        dimension = _strict_int(
            dimension, "shape dimension", minimum=1, maximum=UINT64_MAX
        )
        if count > UINT64_MAX // dimension:
            raise ValueError("shape product overflows unsigned 64-bit")
        count *= dimension
    return count


def _canonical_fp32_bytes(bits: Any) -> bytes:
    bits = _strict_int(bits, "FP32 bits", maximum=UINT32_MAX)
    exponent = bits & 0x7F800000
    mantissa = bits & 0x007FFFFF
    if exponent == 0x7F800000 and mantissa:
        bits = DIGEST_CANONICAL_QNAN_BITS
    return struct.pack("<I", bits)


def logical_fp32_bits_from_view(
    shape: tuple[int, ...],
    physical_fp32_bits: tuple[int, ...],
    element_strides: tuple[int, ...],
) -> tuple[int, ...]:
    """Traverse a bounded physical view in logical row-major index order."""
    count = _shape_product(shape)
    _strict_tuple(physical_fp32_bits, "physical_fp32_bits")
    _strict_tuple(element_strides, "element_strides")
    if len(element_strides) != len(shape):
        raise ValueError("stride rank must equal shape rank")
    for bits in physical_fp32_bits:
        _strict_int(bits, "physical FP32 bits", maximum=UINT32_MAX)
    strides = tuple(
        _strict_int(
            stride,
            "element stride",
            minimum=1,
            maximum=UINT64_MAX,
        )
        for stride in element_strides
    )
    max_offset = 0
    for dimension, stride in zip(shape, strides):
        extent = dimension - 1
        if extent and stride > UINT64_MAX // extent:
            raise ValueError("strided maximum offset overflows unsigned 64-bit")
        contribution = extent * stride
        if max_offset > UINT64_MAX - contribution:
            raise ValueError("strided maximum offset overflows unsigned 64-bit")
        max_offset += contribution
    if not physical_fp32_bits or max_offset >= len(physical_fp32_bits):
        raise ValueError("logical view exceeds the physical element span")
    logical = tuple(
        physical_fp32_bits[
            sum(index * stride for index, stride in zip(indices, strides))
        ]
        for indices in product(*(range(dimension) for dimension in shape))
    )
    if len(logical) != count:
        raise ValueError("logical element count contradicts shape")
    return logical


def canonical_intra_layer_digest_stream(
    *,
    coordinate: IntraLayerCoordinate,
    precision_path: str,
    phase: str,
    shape: tuple[int, ...],
    logical_fp32_bits: tuple[int, ...],
    element_count: Optional[int] = None,
) -> bytes:
    """Encode the exact frozen digest byte grammar for contract tests."""
    if not isinstance(coordinate, IntraLayerCoordinate):
        raise TypeError("coordinate must be an IntraLayerCoordinate")
    _strict_string(precision_path, "precision_path")
    _strict_string(phase, "phase")
    count = _shape_product(shape)
    _strict_tuple(logical_fp32_bits, "logical_fp32_bits")
    if element_count is None:
        element_count = len(logical_fp32_bits)
    element_count = _strict_int(
        element_count, "element_count", maximum=UINT64_MAX
    )
    if (
        element_count != count
        or len(logical_fp32_bits) != count
    ):
        raise ValueError("element count does not match shape and logical values")
    stream = bytearray(DIGEST_DOMAIN_TAG.encode("ascii"))
    stream.append(0)
    stream.extend(_framed(DIGEST_VERSION, "digest_version"))
    stream.extend(_framed(INTRA_LAYER_LAYOUT_NAME, "layout_name"))
    stream.extend(_u64(INTRA_LAYER_LAYOUT_VERSION, "layout_version"))
    stream.extend(_framed(STAGE_TAXONOMY, "stage_taxonomy"))
    stream.extend(_framed(MODEL_FAMILY, "model_family"))
    stream.extend(_framed(precision_path, "precision_path"))
    stream.extend(_framed(phase, "phase"))
    stream.extend(
        _u64(
            coordinate.runtime_checkpoint_step,
            "runtime_checkpoint_step",
        )
    )
    stream.extend(_u64(coordinate.layer_index, "layer_index"))
    stream.extend(_framed(coordinate.stage_id, "stage_id"))
    stream.extend(_framed(coordinate.tensor_role, "tensor_role"))
    stream.extend(_u64(coordinate.batch_index, "batch_index"))
    stream.extend(_u64(coordinate.sequence_index, "sequence_index"))
    stream.extend(_u64(coordinate.token_position, "token_position"))
    stream.extend(_u64(coordinate.stage_order, "stage_order"))
    stream.extend(
        _u64(coordinate.execution_ordinal, "execution_ordinal")
    )
    stream.extend(_u64(len(shape), "rank"))
    for dimension in shape:
        stream.extend(_u64(dimension, "shape dimension"))
    stream.extend(_framed(DIGEST_OBSERVED_DTYPE, "observed_dtype"))
    stream.extend(_framed(DIGEST_BYTE_ORDER, "byte_order"))
    stream.extend(_u64(element_count, "element_count"))
    for bits in logical_fp32_bits:
        stream.extend(_canonical_fp32_bytes(bits))
    return bytes(stream)


def intra_layer_digest_sha256(stream: bytes) -> str:
    if not isinstance(stream, bytes):
        raise TypeError("stream must be bytes")
    return "sha256:" + hashlib.sha256(stream).hexdigest()
