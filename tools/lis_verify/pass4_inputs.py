"""Strict bound intra-layer trace parsing for Pass 4.

This module implements only P4-8.  It consumes an already-classified P4-3
parent-binding outcome and the two authoritative Pass 3B layer traces.  Both
trace identities are checked against the parent outcome before either additive
intra-layer block is materialized.

The parser validates the frozen layout and the declared per-side coverage,
then produces bounded immutable entries.  It does not compute common coverage,
align sources, decide digest-policy availability, compare digests, localize a
mismatch, construct a Pass4Result, serialize an artifact, or expose a public
package entry point.  Those remain P4-9/P4-10 responsibilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

from .pass3_inputs import CanonicalLayerTrace, Pass3InputError
from .pass3_model import CoverageState
from .pass4_contract import (
    COORDINATE_FIELDS,
    DIGEST_ALGORITHM,
    DIGEST_BYTE_ORDER,
    DIGEST_CANONICALIZATION,
    DIGEST_OBSERVED_DTYPE,
    DIGEST_VERSION,
    DUPLICATE_COORDINATE_POLICY,
    INTRA_LAYER_LAYOUT_NAME,
    INTRA_LAYER_LAYOUT_VERSION,
    MODEL_FAMILY,
    ORDERING_SEMANTICS,
    PHASE,
    STAGE_BY_ID,
    STAGE_TAXONOMY,
    STATUS_TO_DISPOSITION,
    UINT64_MAX,
    IntraLayerCoordinate,
    IntraLayerSideCoverage,
    MissingIntraLayerCoordinate,
    Pass4Disposition,
    Pass4ReasonCode,
    Pass4Status,
    UnsupportedIntraLayerLayoutError,
    _SHA256,
    coordinate_from_mapping,
    requested_coordinates,
    validate_coordinate_sequence,
    validate_status_algebra,
)
from .pass4_model import MAX_DETAIL_BYTES, MAX_IDENTIFIER_BYTES
from .pass4_parent import Pass4ParentBindingOutcome


MAX_INTRA_LAYER_ENTRIES = len(STAGE_BY_ID)
MAX_INTRA_LAYER_RANK = 4
MAX_DIAGNOSTICS = 32

AVAILABLE_SUMMARY_FIELDS = (
    "min",
    "max",
    "mean",
    "l2",
    "nan",
    "inf",
    "digest",
)

_LAYOUT_FIELDS = frozenset(
    (
        "layout_name",
        "layout_version",
        "model_family",
        "stage_taxonomy",
        "runtime_checkpoint_step",
        "phase",
        "target_layer",
        "batch_index",
        "sequence_index",
        "token_position",
        "ordering_semantics",
        "duplicate_coordinate_policy",
        "requested_coordinates",
        "captured_coordinates",
        "missing_coordinates",
        "available_summary_fields",
        "digest_contract",
        "full_tensor_payload_allowed",
    )
)

_LAYOUT_DIGEST_FIELDS = frozenset(
    (
        "algorithm",
        "version",
        "observed_dtype",
        "byte_order",
        "canonicalization",
    )
)

_ENTRY_FIELDS = frozenset(
    (
        "runtime_checkpoint_step",
        "phase",
        "layer_index",
        "stage_id",
        "tensor_role",
        "public_name",
        "batch_index",
        "sequence_index",
        "token_position",
        "stage_order",
        "execution_ordinal",
        "shape",
        "observed_dtype",
        "precision_path",
        "element_count",
        "available_summary_fields",
        "min",
        "max",
        "mean",
        "l2",
        "nan",
        "inf",
        "digest",
    )
)

_ENTRY_DIGEST_FIELDS = frozenset(
    (
        "algorithm",
        "version",
        "tensor_role",
        "shape",
        "observed_dtype",
        "byte_order",
        "canonicalization",
        "value",
    )
)

_PUBLIC_NAME_BY_STAGE_ID = {
    "layer_input": "Layer input",
    "attention_norm_output": "Pre-attention RMSNorm output",
    "query_projection_output": "Q projection output",
    "key_projection_output": "K projection output",
    "value_projection_output": "V projection output",
    "rope_query_output": "RoPE-applied Q",
    "rope_key_output": "RoPE-applied K",
    "attention_scores": "Attention pre-softmax scores",
    "attention_probabilities": "Attention softmax output",
    "attention_context": "Attention context",
    "attention_output_projection": "Attention output projection",
    "post_attention_residual": "Post-attention residual",
    "mlp_norm_output": "Pre-MLP RMSNorm output",
    "mlp_gate_projection": "MLP gate projection",
    "mlp_up_projection": "MLP up projection",
    "mlp_gated_activation": "MLP gated activation",
    "mlp_down_projection": "MLP down projection",
}

_PROHIBITED_KEYS = frozenset(
    (
        "tensor_payload",
        "tensor_values",
        "values",
        "samples",
        "tensor_samples",
        "prompt",
        "prompt_text",
        "generated_text",
        "absolute_path",
        "model_path",
    )
)


class Pass4InputError(ValueError):
    """A frozen classified P4-8 failure with privacy-safe detail."""

    def __init__(
        self,
        status: Pass4Status,
        reason: Pass4ReasonCode,
        detail: str,
    ):
        super().__init__(detail)
        self.status = status
        self.reason = reason
        self.detail = _bounded_detail(detail)


@dataclass(frozen=True)
class IntraLayerDigestContract:
    algorithm: str
    version: str
    observed_dtype: str
    byte_order: str
    canonicalization: str

    def __post_init__(self):
        for label in (
            "algorithm",
            "version",
            "observed_dtype",
            "byte_order",
            "canonicalization",
        ):
            _identifier(getattr(self, label), f"digest contract {label}")

    @property
    def frozen_policy_supported(self) -> bool:
        """A fact for P4-9; P4-8 never turns it into a policy decision."""

        return self == IntraLayerDigestContract(
            DIGEST_ALGORITHM,
            DIGEST_VERSION,
            DIGEST_OBSERVED_DTYPE,
            DIGEST_BYTE_ORDER,
            DIGEST_CANONICALIZATION,
        )


@dataclass(frozen=True)
class IntraLayerDigestEnvelope:
    algorithm: str
    version: str
    tensor_role: str
    shape: tuple[int, ...]
    observed_dtype: str
    byte_order: str
    canonicalization: str
    value: str

    def __post_init__(self):
        for label in (
            "algorithm",
            "version",
            "tensor_role",
            "observed_dtype",
            "byte_order",
            "canonicalization",
        ):
            _identifier(getattr(self, label), f"digest {label}")
        _validate_shape_tuple(self.shape, "digest shape")
        if not isinstance(self.value, str) or not _SHA256.fullmatch(self.value):
            raise ValueError("digest value must be canonical SHA-256")


@dataclass(frozen=True)
class IntraLayerTraceHeader:
    layout_name: str
    layout_version: int
    model_family: str
    stage_taxonomy: str
    runtime_checkpoint_step: int
    phase: str
    target_layer: int
    batch_index: int
    sequence_index: int
    token_position: int
    ordering_semantics: str
    duplicate_coordinate_policy: str
    available_summary_fields: tuple[str, ...]
    digest_contract: IntraLayerDigestContract
    full_tensor_payload_allowed: bool

    def __post_init__(self):
        for label in (
            "layout_name",
            "model_family",
            "stage_taxonomy",
            "phase",
            "ordering_semantics",
            "duplicate_coordinate_policy",
        ):
            _identifier(getattr(self, label), label)
        _u64(self.layout_version, "layout_version")
        _u64(self.runtime_checkpoint_step, "runtime_checkpoint_step", minimum=1)
        _u64(self.target_layer, "target_layer")
        _u64(self.batch_index, "batch_index")
        _u64(self.sequence_index, "sequence_index")
        _u64(self.token_position, "token_position")
        if not isinstance(self.available_summary_fields, tuple):
            raise ValueError("available_summary_fields must be immutable")
        if not isinstance(self.digest_contract, IntraLayerDigestContract):
            raise TypeError("digest_contract must be IntraLayerDigestContract")
        if not isinstance(self.full_tensor_payload_allowed, bool):
            raise ValueError("full_tensor_payload_allowed must be boolean")


@dataclass(frozen=True)
class IntraLayerTraceEntry:
    coordinate: IntraLayerCoordinate
    phase: str
    public_name: str
    shape: tuple[int, ...]
    observed_dtype: str
    precision_path: str
    element_count: int
    available_summary_fields: tuple[str, ...]
    min_value: Optional[float]
    max_value: Optional[float]
    mean_value: Optional[float]
    l2_norm: Optional[float]
    nan_present: bool
    inf_present: bool
    digest: IntraLayerDigestEnvelope

    def __post_init__(self):
        if not isinstance(self.coordinate, IntraLayerCoordinate):
            raise TypeError("coordinate must be IntraLayerCoordinate")
        _identifier(self.phase, "entry phase")
        _identifier(self.public_name, "public_name")
        _validate_shape_tuple(self.shape, "entry shape")
        _identifier(self.observed_dtype, "observed_dtype")
        _identifier(self.precision_path, "precision_path")
        _u64(self.element_count, "element_count", minimum=1)
        if self.element_count != _shape_product(self.shape, "entry shape"):
            raise ValueError("element_count contradicts entry shape")
        if self.available_summary_fields != AVAILABLE_SUMMARY_FIELDS:
            raise ValueError("available summary fields are not frozen v1")
        for label in (
            "min_value",
            "max_value",
            "mean_value",
            "l2_norm",
        ):
            value = getattr(self, label)
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, float)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{label} must be finite or null")
        if self.l2_norm is not None and self.l2_norm < 0.0:
            raise ValueError("l2_norm cannot be negative")
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError("min_value cannot exceed max_value")
        if not isinstance(self.nan_present, bool) or not isinstance(
            self.inf_present, bool
        ):
            raise ValueError("nonfinite markers must be boolean facts")
        if any(
            value is None
            for value in (
                self.min_value,
                self.max_value,
                self.mean_value,
                self.l2_norm,
            )
        ) and not (self.nan_present or self.inf_present):
            raise ValueError("null aggregate lacks nonfinite evidence")
        if not isinstance(self.digest, IntraLayerDigestEnvelope):
            raise TypeError("digest must be IntraLayerDigestEnvelope")


@dataclass(frozen=True)
class ParsedIntraLayerSource:
    header: IntraLayerTraceHeader
    coverage: IntraLayerSideCoverage
    entries: tuple[IntraLayerTraceEntry, ...]

    def __post_init__(self):
        if not isinstance(self.header, IntraLayerTraceHeader):
            raise TypeError("header must be IntraLayerTraceHeader")
        if not isinstance(self.coverage, IntraLayerSideCoverage):
            raise TypeError("coverage must be IntraLayerSideCoverage")
        if not isinstance(self.entries, tuple):
            raise ValueError("entries must be immutable")
        if len(self.entries) > MAX_INTRA_LAYER_ENTRIES:
            raise ValueError("entries exceed the frozen bound")
        if any(not isinstance(item, IntraLayerTraceEntry) for item in self.entries):
            raise TypeError("entries contain an invalid value")
        entry_coordinates = tuple(item.coordinate for item in self.entries)
        if entry_coordinates != self.coverage.captured_coordinates:
            raise ValueError("entries disagree with captured coordinates")


@dataclass(frozen=True)
class Pass4TraceParsingOutcome:
    """P4-8 parsed evidence, or a frozen terminal classification."""

    parent_outcome: Pass4ParentBindingOutcome
    status: Optional[Pass4Status]
    disposition: Optional[Pass4Disposition]
    reason_codes: tuple[Pass4ReasonCode, ...]
    reference: Optional[ParsedIntraLayerSource]
    candidate: Optional[ParsedIntraLayerSource]
    warnings: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.parent_outcome, Pass4ParentBindingOutcome):
            raise TypeError("parent_outcome must be Pass4ParentBindingOutcome")
        _bounded_messages(self.warnings, "warnings")
        _bounded_messages(self.diagnostics, "diagnostics")
        if self.status is None:
            if (
                self.disposition is not None
                or self.reason_codes
                or not self.parent_outcome.proceed
                or not isinstance(self.reference, ParsedIntraLayerSource)
                or not isinstance(self.candidate, ParsedIntraLayerSource)
            ):
                raise ValueError("proceed outcome requires complete parsed sources")
        else:
            if self.reference is not None or self.candidate is not None:
                raise ValueError(
                    "terminal parser outcome cannot retain partial sources"
                )
            validate_status_algebra(
                self.status, self.disposition, self.reason_codes
            )
            if not self.parent_outcome.proceed and (
                self.status != self.parent_outcome.status
                or self.disposition != self.parent_outcome.disposition
                or self.reason_codes != self.parent_outcome.reason_codes
            ):
                raise ValueError("parent terminal classification was not preserved")

    @property
    def proceed(self) -> bool:
        return self.status is None


@dataclass(frozen=True)
class _ParsedLayout:
    header: IntraLayerTraceHeader
    coverage: IntraLayerSideCoverage
    raw: dict[str, Any]


def _bounded_detail(value: str) -> str:
    if not isinstance(value, str) or not value:
        return "intra-layer validation failed"
    encoded = value.encode("utf-8")
    if len(encoded) <= MAX_DETAIL_BYTES:
        return value
    return encoded[:MAX_DETAIL_BYTES].decode("utf-8", errors="ignore")


def _bounded_messages(values: Any, label: str) -> tuple[str, ...]:
    if (
        not isinstance(values, tuple)
        or len(values) > MAX_DIAGNOSTICS
        or len(values) != len(set(values))
    ):
        raise ValueError(f"{label} must be an immutable bounded unique tuple")
    for value in values:
        if (
            not isinstance(value, str)
            or not value
            or len(value.encode("utf-8")) > MAX_DETAIL_BYTES
            or "/" in value
            or "sha256:" in value
            or "aset1:" in value
        ):
            raise ValueError(f"{label} contains unsafe detail")
    return values


def _identifier(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > MAX_IDENTIFIER_BYTES
        or any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value)
    ):
        raise ValueError(f"{label} must be a bounded identifier")
    return value


def _detail(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > MAX_DETAIL_BYTES
        or any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value)
    ):
        raise ValueError(f"{label} must be bounded text")
    return value


def _u64(value: Any, label: str, *, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > UINT64_MAX
    ):
        raise ValueError(f"{label} must be an unsigned 64-bit integer")
    return value


def _shape_product(shape: tuple[int, ...], label: str) -> int:
    product = 1
    for dimension in shape:
        if product > UINT64_MAX // dimension:
            raise ValueError(f"{label} product overflows u64")
        product *= dimension
    return product


def _validate_shape_tuple(shape: Any, label: str) -> tuple[int, ...]:
    if (
        not isinstance(shape, tuple)
        or not shape
        or len(shape) > MAX_INTRA_LAYER_RANK
    ):
        raise ValueError(f"{label} rank is outside the frozen bound")
    for dimension in shape:
        _u64(dimension, f"{label} dimension", minimum=1)
    _shape_product(shape, label)
    return shape


def _shape(raw: Any, label: str) -> tuple[int, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{label} must be an array")
    return _validate_shape_tuple(tuple(raw), label)


def _string_tuple(raw: Any, label: str) -> tuple[str, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"{label} must be an array")
    values = tuple(_identifier(value, f"{label} entry") for value in raw)
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate entries")
    return values


def _mapping_fields(raw: Any, expected: frozenset[str], label: str) -> dict:
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError(f"{label} must contain exactly the frozen fields")
    return raw


def _coordinate(raw: Any, label: str) -> IntraLayerCoordinate:
    try:
        return coordinate_from_mapping(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is malformed") from exc


def _coverage_sequence(
    raw: Any,
    label: str,
) -> tuple[IntraLayerCoordinate, ...]:
    if not isinstance(raw, list) or len(raw) > MAX_INTRA_LAYER_ENTRIES:
        raise ValueError(f"{label} must be a bounded array")
    coordinates = tuple(
        _coordinate(item, f"{label} entry") for item in raw
    )
    try:
        validate_coordinate_sequence(coordinates, label)
    except ValueError as exc:
        raise Pass4InputError(
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
            Pass4ReasonCode.DUPLICATE_OR_OUT_OF_ORDER_COORDINATE,
            f"{label} coordinate order",
        ) from exc
    return coordinates


def _missing_coordinates(raw: Any, label: str) -> tuple:
    if not isinstance(raw, list) or len(raw) > MAX_INTRA_LAYER_ENTRIES:
        raise ValueError(f"{label} must be a bounded array")
    parsed = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {
            "coordinate",
            "state",
            "detail",
        }:
            raise ValueError(f"{label} entry is malformed")
        try:
            state = CoverageState(item["state"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} state is malformed") from exc
        detail = item["detail"]
        if detail is not None:
            _detail(detail, f"{label} detail")
        try:
            parsed.append(
                MissingIntraLayerCoordinate(
                    _coordinate(item["coordinate"], f"{label} coordinate"),
                    state,
                    detail,
                )
            )
        except ValueError as exc:
            raise ValueError(f"{label} entry is malformed") from exc
    coordinates = tuple(item.coordinate for item in parsed)
    try:
        validate_coordinate_sequence(coordinates, label)
    except ValueError as exc:
        raise Pass4InputError(
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
            Pass4ReasonCode.DUPLICATE_OR_OUT_OF_ORDER_COORDINATE,
            f"{label} coordinate order",
        ) from exc
    return tuple(parsed)


def _parse_layout_digest(raw: Any, label: str) -> IntraLayerDigestContract:
    try:
        value = _mapping_fields(raw, _LAYOUT_DIGEST_FIELDS, label)
        return IntraLayerDigestContract(
            _identifier(value["algorithm"], f"{label} algorithm"),
            _identifier(value["version"], f"{label} version"),
            _identifier(value["observed_dtype"], f"{label} dtype"),
            _identifier(value["byte_order"], f"{label} byte order"),
            _identifier(
                value["canonicalization"], f"{label} canonicalization"
            ),
        )
    except (TypeError, ValueError) as exc:
        raise Pass4InputError(
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
            Pass4ReasonCode.DIGEST_FIELD_MALFORMED,
            f"{label} malformed",
        ) from exc


def _parse_layout(raw: Any, side: str) -> _ParsedLayout:
    label = f"{side} intra layout"
    if not isinstance(raw, dict):
        raise Pass4InputError(
            Pass4Status.UNSUPPORTED_INTRA_LAYER_LAYOUT,
            Pass4ReasonCode.INTRA_LAYOUT_OR_TAXONOMY_UNSUPPORTED,
            f"{label} unavailable",
        )
    if (
        raw.get("layout_name") != INTRA_LAYER_LAYOUT_NAME
        or isinstance(raw.get("layout_version"), bool)
        or raw.get("layout_version") != INTRA_LAYER_LAYOUT_VERSION
        or raw.get("model_family") != MODEL_FAMILY
        or raw.get("stage_taxonomy") != STAGE_TAXONOMY
        or raw.get("phase") != PHASE
        or raw.get("ordering_semantics") != ORDERING_SEMANTICS
        or raw.get("duplicate_coordinate_policy")
        != DUPLICATE_COORDINATE_POLICY
        or isinstance(raw.get("batch_index"), bool)
        or raw.get("batch_index") != 0
        or isinstance(raw.get("sequence_index"), bool)
        or raw.get("sequence_index") != 0
        or raw.get("full_tensor_payload_allowed") is not False
    ):
        raise Pass4InputError(
            Pass4Status.UNSUPPORTED_INTRA_LAYER_LAYOUT,
            Pass4ReasonCode.INTRA_LAYOUT_OR_TAXONOMY_UNSUPPORTED,
            f"{label} identity",
        )
    try:
        step = _u64(
            raw.get("runtime_checkpoint_step"),
            f"{label} step",
            minimum=1,
        )
        target_layer = _u64(raw.get("target_layer"), f"{label} layer")
        token_position = _u64(
            raw.get("token_position"), f"{label} token position"
        )
        requested = _coverage_sequence(
            raw.get("requested_coordinates"), f"{side} requested"
        )
        captured = _coverage_sequence(
            raw.get("captured_coordinates"), f"{side} captured"
        )
        missing = _missing_coordinates(
            raw.get("missing_coordinates"), f"{side} missing"
        )
        available = _string_tuple(
            raw.get("available_summary_fields"),
            f"{label} available fields",
        )
    except Pass4InputError:
        raise
    except (TypeError, ValueError) as exc:
        raise Pass4InputError(
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
            Pass4ReasonCode.COVERAGE_PARTITION_MALFORMED,
            f"{label} coverage",
        ) from exc
    expected_requested = requested_coordinates(
        step, target_layer, token_position
    )
    if requested != expected_requested:
        raise Pass4InputError(
            Pass4Status.UNSUPPORTED_INTRA_LAYER_LAYOUT,
            Pass4ReasonCode.REQUESTED_STAGE_SET_UNSUPPORTED,
            f"{label} requested stages",
        )
    if available != AVAILABLE_SUMMARY_FIELDS:
        raise Pass4InputError(
            Pass4Status.UNSUPPORTED_INTRA_LAYER_LAYOUT,
            Pass4ReasonCode.INTRA_LAYOUT_OR_TAXONOMY_UNSUPPORTED,
            f"{label} summary fields",
        )
    digest_contract = _parse_layout_digest(
        raw.get("digest_contract"), f"{label} digest contract"
    )
    try:
        coverage = IntraLayerSideCoverage(requested, captured, missing)
    except UnsupportedIntraLayerLayoutError as exc:
        raise Pass4InputError(
            Pass4Status.UNSUPPORTED_INTRA_LAYER_LAYOUT,
            Pass4ReasonCode.REQUESTED_STAGE_SET_UNSUPPORTED,
            f"{label} requested stages",
        ) from exc
    except ValueError as exc:
        raise Pass4InputError(
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
            Pass4ReasonCode.COVERAGE_PARTITION_MALFORMED,
            f"{label} partition",
        ) from exc
    try:
        header = IntraLayerTraceHeader(
            INTRA_LAYER_LAYOUT_NAME,
            INTRA_LAYER_LAYOUT_VERSION,
            MODEL_FAMILY,
            STAGE_TAXONOMY,
            step,
            PHASE,
            target_layer,
            0,
            0,
            token_position,
            ORDERING_SEMANTICS,
            DUPLICATE_COORDINATE_POLICY,
            available,
            digest_contract,
            False,
        )
    except (TypeError, ValueError) as exc:
        raise Pass4InputError(
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
            Pass4ReasonCode.SUMMARY_FIELD_MALFORMED,
            f"{label} header",
        ) from exc
    return _ParsedLayout(header, coverage, raw)


def _reject_prohibited(value: Any, side: str) -> None:
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, child in current.items():
                if key in _PROHIBITED_KEYS:
                    raise Pass4InputError(
                        Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
                        Pass4ReasonCode.PROHIBITED_PAYLOAD_PRESENT,
                        f"{side} prohibited payload",
                    )
                stack.append(child)
        elif isinstance(current, list):
            stack.extend(current)
        elif isinstance(current, str) and current.startswith("/"):
            raise Pass4InputError(
                Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
                Pass4ReasonCode.PROHIBITED_PAYLOAD_PRESENT,
                f"{side} absolute path payload",
            )


def _aggregate(raw: Any, label: str) -> Optional[float]:
    if raw is None:
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ValueError(f"{label} is malformed")
    try:
        value = float(raw)
    except OverflowError as exc:
        raise ValueError(f"{label} is outside the finite range") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite or null")
    return value


def _flag(raw: Any, label: str) -> bool:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw not in (0, 1):
        raise ValueError(f"{label} must be integer zero or one")
    return bool(raw)


def _parse_entry_digest(
    raw: Any,
    entry_shape: tuple[int, ...],
    coordinate: IntraLayerCoordinate,
    observed_dtype: str,
    contract: IntraLayerDigestContract,
    label: str,
) -> IntraLayerDigestEnvelope:
    try:
        value = _mapping_fields(raw, _ENTRY_DIGEST_FIELDS, label)
        digest_shape = _shape(value["shape"], f"{label} shape")
        digest = IntraLayerDigestEnvelope(
            _identifier(value["algorithm"], f"{label} algorithm"),
            _identifier(value["version"], f"{label} version"),
            _identifier(value["tensor_role"], f"{label} role"),
            digest_shape,
            _identifier(value["observed_dtype"], f"{label} dtype"),
            _identifier(value["byte_order"], f"{label} byte order"),
            _identifier(
                value["canonicalization"], f"{label} canonicalization"
            ),
            value["value"],
        )
    except (TypeError, ValueError) as exc:
        raise Pass4InputError(
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
            Pass4ReasonCode.DIGEST_FIELD_MALFORMED,
            f"{label} malformed",
        ) from exc
    if (
        digest.algorithm != contract.algorithm
        or digest.version != contract.version
        or digest.observed_dtype != contract.observed_dtype
        or digest.byte_order != contract.byte_order
        or digest.canonicalization != contract.canonicalization
        or digest.tensor_role != coordinate.tensor_role
        or digest.shape != entry_shape
        or digest.observed_dtype != observed_dtype
    ):
        raise Pass4InputError(
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
            Pass4ReasonCode.DIGEST_FIELD_MALFORMED,
            f"{label} inconsistent",
        )
    return digest


def _parse_entry(
    raw: Any,
    layout: _ParsedLayout,
    side: str,
) -> IntraLayerTraceEntry:
    label = f"{side} intra entry"
    try:
        value = _mapping_fields(raw, _ENTRY_FIELDS, label)
        coordinate = _coordinate(
            {field: value[field] for field in COORDINATE_FIELDS},
            f"{label} coordinate",
        )
        phase = _identifier(value["phase"], f"{label} phase")
        public_name = _identifier(value["public_name"], f"{label} name")
        shape = _shape(value["shape"], f"{label} shape")
        observed_dtype = _identifier(
            value["observed_dtype"], f"{label} dtype"
        )
        precision_path = _identifier(
            value["precision_path"], f"{label} precision"
        )
        element_count = _u64(
            value["element_count"], f"{label} count", minimum=1
        )
        available = _string_tuple(
            value["available_summary_fields"], f"{label} fields"
        )
        min_value = _aggregate(value["min"], f"{label} min")
        max_value = _aggregate(value["max"], f"{label} max")
        mean_value = _aggregate(value["mean"], f"{label} mean")
        l2_norm = _aggregate(value["l2"], f"{label} l2")
        nan_present = _flag(value["nan"], f"{label} nan")
        inf_present = _flag(value["inf"], f"{label} inf")
    except Pass4InputError:
        raise
    except (TypeError, ValueError) as exc:
        raise Pass4InputError(
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
            Pass4ReasonCode.SUMMARY_FIELD_MALFORMED,
            f"{label} malformed",
        ) from exc
    if phase != layout.header.phase:
        raise Pass4InputError(
            Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT,
            Pass4ReasonCode.PHASE_OR_POSITION_ALIGNMENT_MISMATCH,
            f"{label} phase",
        )
    if public_name != _PUBLIC_NAME_BY_STAGE_ID[coordinate.stage_id]:
        raise Pass4InputError(
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
            Pass4ReasonCode.SUMMARY_FIELD_MALFORMED,
            f"{label} public name",
        )
    if element_count != _shape_product(shape, f"{label} shape"):
        raise Pass4InputError(
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
            Pass4ReasonCode.SUMMARY_FIELD_MALFORMED,
            f"{label} shape count",
        )
    if available != AVAILABLE_SUMMARY_FIELDS:
        raise Pass4InputError(
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
            Pass4ReasonCode.SUMMARY_FIELD_MALFORMED,
            f"{label} available fields",
        )
    digest = _parse_entry_digest(
        value["digest"],
        shape,
        coordinate,
        observed_dtype,
        layout.header.digest_contract,
        f"{label} digest",
    )
    try:
        return IntraLayerTraceEntry(
            coordinate,
            phase,
            public_name,
            shape,
            observed_dtype,
            precision_path,
            element_count,
            available,
            min_value,
            max_value,
            mean_value,
            l2_norm,
            nan_present,
            inf_present,
            digest,
        )
    except (TypeError, ValueError) as exc:
        raise Pass4InputError(
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
            Pass4ReasonCode.SUMMARY_FIELD_MALFORMED,
            f"{label} summary",
        ) from exc


def _parse_entries(
    raw: dict[str, Any],
    layout: _ParsedLayout,
    side: str,
) -> ParsedIntraLayerSource:
    _reject_prohibited(
        {
            "intra_layer_checkpoint_layout": layout.raw,
            "intra_layer_trace": raw.get("intra_layer_trace"),
        },
        side,
    )
    if set(layout.raw) != _LAYOUT_FIELDS:
        raise Pass4InputError(
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
            Pass4ReasonCode.SUMMARY_FIELD_MALFORMED,
            f"{side} intra layout fields",
        )
    entries_raw = raw.get("intra_layer_trace")
    if (
        not isinstance(entries_raw, list)
        or len(entries_raw) > MAX_INTRA_LAYER_ENTRIES
    ):
        raise Pass4InputError(
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
            Pass4ReasonCode.SUMMARY_FIELD_MALFORMED,
            f"{side} intra entries",
        )
    entries = tuple(
        _parse_entry(item, layout, side) for item in entries_raw
    )
    coordinates = tuple(item.coordinate for item in entries)
    try:
        validate_coordinate_sequence(coordinates, f"{side} intra entries")
    except ValueError as exc:
        raise Pass4InputError(
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
            Pass4ReasonCode.DUPLICATE_OR_OUT_OF_ORDER_COORDINATE,
            f"{side} intra entry order",
        ) from exc
    if coordinates != layout.coverage.captured_coordinates:
        raise Pass4InputError(
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
            Pass4ReasonCode.COVERAGE_PARTITION_MALFORMED,
            f"{side} coverage entry disagreement",
        )
    try:
        return ParsedIntraLayerSource(layout.header, layout.coverage, entries)
    except (TypeError, ValueError) as exc:
        raise Pass4InputError(
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
            Pass4ReasonCode.COVERAGE_PARTITION_MALFORMED,
            f"{side} parsed source",
        ) from exc


def _trace_expected_sha(
    parent: Pass4ParentBindingOutcome,
    side: str,
) -> str:
    binding = (
        parent.reference_binding if side == "reference" else parent.candidate_binding
    )
    identities = parent.artifact_identities
    recorded = (
        identities.authoritative_reference
        if side == "reference"
        else identities.authoritative_candidate
    )
    if binding is None or identities is None:
        raise Pass4InputError(
            Pass4Status.SOURCE_BINDING_INCONSISTENT,
            Pass4ReasonCode.TRACE_SHA_MISMATCH,
            f"{side} bound trace identity",
        )
    if binding.identity.layer_trace_sha256 != recorded.layer_trace_sha256:
        raise Pass4InputError(
            Pass4Status.SOURCE_BINDING_INCONSISTENT,
            Pass4ReasonCode.TRACE_SHA_MISMATCH,
            f"{side} parent trace identity",
        )
    return recorded.layer_trace_sha256


def _verify_trace_identity(
    parent: Pass4ParentBindingOutcome,
    trace: CanonicalLayerTrace,
    side: str,
) -> None:
    expected = _trace_expected_sha(parent, side)
    if trace.identity.trace_sha256 != expected:
        raise Pass4InputError(
            Pass4Status.SOURCE_BINDING_INCONSISTENT,
            Pass4ReasonCode.TRACE_SHA_MISMATCH,
            f"{side} supplied trace identity",
        )


def _materialize(trace: CanonicalLayerTrace, side: str) -> dict[str, Any]:
    try:
        raw = trace.materialize()
    except Pass3InputError as exc:
        raise Pass4InputError(
            Pass4Status.SOURCE_BINDING_INCONSISTENT,
            Pass4ReasonCode.TRACE_SHA_MISMATCH,
            f"{side} canonical trace identity",
        ) from exc
    if not isinstance(raw, dict):
        raise Pass4InputError(
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
            Pass4ReasonCode.SUMMARY_FIELD_MALFORMED,
            f"{side} trace root",
        )
    return raw


def _terminal(
    parent: Pass4ParentBindingOutcome,
    status: Pass4Status,
    reason_codes: tuple[Pass4ReasonCode, ...],
    *,
    diagnostic: Optional[str] = None,
) -> Pass4TraceParsingOutcome:
    diagnostics = parent.diagnostics
    if diagnostic:
        detail = _bounded_detail(diagnostic)
        if detail not in diagnostics and len(diagnostics) < MAX_DIAGNOSTICS:
            diagnostics = diagnostics + (detail,)
    return Pass4TraceParsingOutcome(
        parent,
        status,
        STATUS_TO_DISPOSITION[status],
        reason_codes,
        None,
        None,
        parent.warnings,
        diagnostics,
    )


def parse_pass4_intra_layer_inputs(
    parent_outcome: Pass4ParentBindingOutcome,
    reference_trace: CanonicalLayerTrace,
    candidate_trace: CanonicalLayerTrace,
) -> Pass4TraceParsingOutcome:
    """Parse P4-8 evidence only after the frozen P4-3 binding boundary."""

    if not isinstance(parent_outcome, Pass4ParentBindingOutcome):
        raise TypeError("parent_outcome must be Pass4ParentBindingOutcome")
    if not isinstance(reference_trace, CanonicalLayerTrace):
        raise TypeError("reference_trace must be CanonicalLayerTrace")
    if not isinstance(candidate_trace, CanonicalLayerTrace):
        raise TypeError("candidate_trace must be CanonicalLayerTrace")

    if not parent_outcome.proceed:
        return Pass4TraceParsingOutcome(
            parent_outcome,
            parent_outcome.status,
            parent_outcome.disposition,
            parent_outcome.reason_codes,
            None,
            None,
            parent_outcome.warnings,
            parent_outcome.diagnostics,
        )

    try:
        # Both immutable identities are verified before either summary block is
        # materialized.  Do not combine these loops.
        _verify_trace_identity(parent_outcome, reference_trace, "reference")
        _verify_trace_identity(parent_outcome, candidate_trace, "candidate")

        reference_raw = _materialize(reference_trace, "reference")
        candidate_raw = _materialize(candidate_trace, "candidate")

        # Complete gate 6 on both sides before gate 7 entries on either side.
        reference_layout = _parse_layout(
            reference_raw.get("intra_layer_checkpoint_layout"), "reference"
        )
        candidate_layout = _parse_layout(
            candidate_raw.get("intra_layer_checkpoint_layout"), "candidate"
        )
        if (
            reference_layout.coverage.requested_coordinates
            != candidate_layout.coverage.requested_coordinates
        ):
            raise Pass4InputError(
                Pass4Status.UNSUPPORTED_INTRA_LAYER_LAYOUT,
                Pass4ReasonCode.REQUESTED_STAGE_SET_UNSUPPORTED,
                "requested stage lists differ",
            )
        target_step = parent_outcome.target_runtime_checkpoint_step
        target_layer = parent_outcome.target_layer
        for side, layout in (
            ("reference", reference_layout),
            ("candidate", candidate_layout),
        ):
            if layout.header.runtime_checkpoint_step != target_step:
                raise Pass4InputError(
                    Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT,
                    Pass4ReasonCode.STEP_ALIGNMENT_MISMATCH,
                    f"{side} layout target step",
                )
            if layout.header.target_layer != target_layer:
                raise Pass4InputError(
                    Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT,
                    Pass4ReasonCode.LAYER_ALIGNMENT_MISMATCH,
                    f"{side} layout target layer",
                )

        reference = _parse_entries(
            reference_raw, reference_layout, "reference"
        )
        candidate = _parse_entries(
            candidate_raw, candidate_layout, "candidate"
        )
        return Pass4TraceParsingOutcome(
            parent_outcome,
            None,
            None,
            (),
            reference,
            candidate,
            parent_outcome.warnings,
            parent_outcome.diagnostics,
        )
    except Pass4InputError as exc:
        return _terminal(
            parent_outcome,
            exc.status,
            (exc.reason,),
            diagnostic=exc.detail,
        )
