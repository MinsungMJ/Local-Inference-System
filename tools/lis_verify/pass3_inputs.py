"""Strict canonical inputs and source binding for Pass 3.

Trace summaries are materialized only after typed Pass 2 readiness, supplied
Pass 2 artifact coherence, and both report/trace binding chains succeed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from math import prod
from typing import Any, Optional

from .pass1_inputs import (
    CanonicalRunReport,
    MalformedRunReport,
    canonical_json,
    canonical_json_sha256,
    sha256_text,
    strict_json_loads,
)
from .pass2_artifact import serialize as serialize_pass2
from .pass2_model import Pass2Result, ReproductionEvidenceTier
from .pass3_model import (
    CheckpointCoordinate,
    CheckpointCoverage,
    CheckpointDigest,
    CheckpointSummary,
    CoverageEntry,
    CoverageState,
    DIGEST_ALGORITHM,
    DIGEST_BYTE_ORDER,
    DIGEST_CANONICALIZATION,
    DIGEST_DTYPE,
    DIGEST_VERSION,
    SourceArtifactBinding,
)


_ARTIFACT_SET_ID = re.compile(r"^aset1:[0-9a-f]{32}$")


class Pass3InputError(ValueError):
    """Strict Pass 3 input failure carrying a local reason string."""

    def __init__(self, message: str, reason: str):
        super().__init__(message)
        self.reason = reason


def _strict_object(text: str, label: str) -> dict[str, Any]:
    try:
        return strict_json_loads(text)
    except MalformedRunReport as exc:
        raise Pass3InputError(
            f"{label}: {exc}", "pass3.checkpoint_summary_malformed"
        ) from exc


@dataclass(frozen=True)
class CanonicalPass2Artifact:
    """Supplied canonical Pass 2 bytes and their only Pass 3 identity."""

    artifact_sha256: str
    canonical_text: str

    @classmethod
    def from_json(cls, text: str) -> "CanonicalPass2Artifact":
        raw = _strict_object(text, "Pass 2 artifact")
        rendered = canonical_json(raw)
        return cls(sha256_text(rendered), rendered)

    @classmethod
    def from_object(cls, raw: dict[str, Any]) -> "CanonicalPass2Artifact":
        return cls.from_json(canonical_json(raw))

    @classmethod
    def from_result(cls, result: Pass2Result) -> "CanonicalPass2Artifact":
        """Use the existing Pass 2 serializer; define no Pass 3 serializer."""
        return cls.from_object(serialize_pass2(result))

    @classmethod
    def load(cls, path) -> "CanonicalPass2Artifact":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def materialize_verified(self) -> dict[str, Any]:
        raw = _strict_object(self.canonical_text, "Pass 2 artifact")
        rendered = canonical_json(raw)
        if rendered != self.canonical_text or sha256_text(rendered) != self.artifact_sha256:
            raise Pass3InputError(
                "Pass 2 artifact canonical identity is inconsistent",
                "pass3.pass2_artifact_identity_inconsistent",
            )
        return raw


@dataclass(frozen=True)
class LayerTraceIdentity:
    trace_sha256: str
    schema: Optional[str]
    kind: Optional[str]
    artifact_set_id: Optional[str]
    semantic_manifest_sha256: Optional[str]
    runtime_checkpoint_step: Optional[int]


@dataclass(frozen=True)
class CanonicalLayerTrace:
    """Canonical layer trace with immutable header identity.

    `materialize` is the summary access boundary.  Binding code uses only the
    identity populated during strict construction.
    """

    identity: LayerTraceIdentity
    canonical_text: str

    @classmethod
    def from_json(cls, text: str) -> "CanonicalLayerTrace":
        raw = _strict_object(text, "layer trace")
        rendered = canonical_json(raw)
        manifest = raw.get("manifest")
        layout = raw.get("checkpoint_layout")
        manifest_sha = (
            canonical_json_sha256(manifest) if isinstance(manifest, dict) else None
        )
        target = layout.get("runtime_checkpoint_step") if isinstance(layout, dict) else None
        if isinstance(target, bool) or not isinstance(target, int) or target < 0:
            target = None
        identity = LayerTraceIdentity(
            trace_sha256=sha256_text(rendered),
            schema=raw.get("schema") if isinstance(raw.get("schema"), str) else None,
            kind=raw.get("kind") if isinstance(raw.get("kind"), str) else None,
            artifact_set_id=(
                raw.get("artifact_set_id")
                if isinstance(raw.get("artifact_set_id"), str)
                else None
            ),
            semantic_manifest_sha256=manifest_sha,
            runtime_checkpoint_step=target,
        )
        return cls(identity, rendered)

    @classmethod
    def from_object(cls, raw: dict[str, Any]) -> "CanonicalLayerTrace":
        return cls.from_json(canonical_json(raw))

    @classmethod
    def load(cls, path) -> "CanonicalLayerTrace":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def materialize(self) -> dict[str, Any]:
        raw = _strict_object(self.canonical_text, "layer trace")
        rendered = canonical_json(raw)
        if rendered != self.canonical_text or sha256_text(rendered) != self.identity.trace_sha256:
            raise Pass3InputError(
                "layer trace canonical identity is inconsistent",
                "pass3.checkpoint_summary_malformed",
            )
        return raw


def validate_pass2_artifact_coherence(
    pass2: Pass2Result, artifact: CanonicalPass2Artifact
) -> tuple[dict[str, Any], str]:
    """Validate the complete existing Pass 2 serialization representation."""
    if not isinstance(artifact, CanonicalPass2Artifact):
        raise TypeError("pass2_artifact must be a CanonicalPass2Artifact")
    raw = artifact.materialize_verified()
    if (
        raw.get("schema") != "lis.execution_artifact/v1"
        or raw.get("kind") != "prefix_policy_reproduction"
        or raw.get("contract_version")
        != "differential_verification_contract_v1"
    ):
        raise Pass3InputError(
            "unsupported canonical Pass 2 artifact identity",
            "pass3.pass2_artifact_identity_inconsistent",
        )
    # This is intentionally the authoritative existing serializer.  Exact
    # object equality covers every serialized Pass 3-relevant field and avoids
    # a selected-field identity shortcut.
    if raw != serialize_pass2(pass2):
        raise Pass3InputError(
            "typed Pass2Result disagrees with its supplied canonical artifact",
            "pass3.pass2_object_artifact_inconsistent",
        )
    return raw, artifact.artifact_sha256


def selected_source_hashes(pass2: Pass2Result) -> tuple[tuple[str, str], tuple[str, str]]:
    """Select the role-bound report identities authorized by Pass 2's tier."""
    binding = pass2.source_binding
    if (
        pass2.reproduction_evidence_tier
        == ReproductionEvidenceTier.INDEPENDENT_RERUN_VERIFIED
    ):
        reference = binding.reference_reproduction_sha256
        candidate = binding.candidate_reproduction_sha256
        roles = ("reference_reproduction", "candidate_reproduction")
    else:
        reference = binding.reference_original_run_report_sha256
        candidate = binding.candidate_original_run_report_sha256
        roles = ("reference_original", "candidate_original")
    if not reference or not candidate:
        raise Pass3InputError(
            "Pass 2 source role binding is incomplete",
            "pass3.binding_metadata_missing",
        )
    return (roles[0], reference), (roles[1], candidate)


def _validated_run_report_binding(
    role: str,
    expected_sha256: str,
    report: CanonicalRunReport,
    trace: CanonicalLayerTrace,
    target_runtime_checkpoint_step: int,
) -> SourceArtifactBinding:
    try:
        actual_sha256 = report.identity.run_report_sha256
    except AttributeError as exc:
        raise Pass3InputError(
            f"{role}: canonical run report is required",
            "pass3.run_report_canonical_sha_inconsistent",
        ) from exc
    if actual_sha256 != expected_sha256:
        raise Pass3InputError(
            f"{role}: run-report canonical SHA-256 does not match Pass 2",
            "pass3.run_report_canonical_sha_inconsistent",
        )

    identity = trace.identity
    if identity.schema != "lis.execution_artifact/v1" or identity.kind != "layer_trace":
        raise Pass3InputError(
            f"{role}: supplied trace is not a layer_trace artifact",
            "pass3.binding_metadata_missing",
        )
    if not identity.artifact_set_id or not _ARTIFACT_SET_ID.fullmatch(
        identity.artifact_set_id
    ):
        raise Pass3InputError(
            f"{role}: trace artifact_set_id is missing or malformed",
            "pass3.binding_metadata_missing",
        )
    if identity.runtime_checkpoint_step is None:
        raise Pass3InputError(
            f"{role}: versioned checkpoint layout is unavailable",
            "pass3.unsupported_checkpoint_layout",
        )
    if identity.runtime_checkpoint_step != target_runtime_checkpoint_step:
        raise Pass3InputError(
            f"{role}: trace runtime checkpoint step does not match Pass 2",
            "pass3.runtime_checkpoint_step_mismatch",
        )

    # Run-report materialization is permitted at Gate B. Trace summary
    # materialization remains forbidden until both calls return successfully.
    raw = report.materialize()
    if (
        not isinstance(raw, dict)
        or raw.get("schema") != "lis.execution_artifact/v1"
        or raw.get("kind") != "run_report"
    ):
        raise Pass3InputError(
            f"{role}: unsupported run report",
            "pass3.binding_metadata_missing",
        )
    report_set_id = raw.get("artifact_set_id")
    manifest = raw.get("manifest")
    if (
        not isinstance(report_set_id, str)
        or not _ARTIFACT_SET_ID.fullmatch(report_set_id)
        or not isinstance(manifest, dict)
        or identity.semantic_manifest_sha256 is None
    ):
        raise Pass3InputError(
            f"{role}: complete report/trace binding metadata is required",
            "pass3.binding_metadata_missing",
        )
    if report_set_id != identity.artifact_set_id:
        raise Pass3InputError(
            f"{role}: report and trace artifact_set_id differ",
            "pass3.artifact_set_id_inconsistent",
        )
    manifest_sha = canonical_json_sha256(manifest)
    if manifest_sha != identity.semantic_manifest_sha256:
        raise Pass3InputError(
            f"{role}: report and trace semantic manifests differ",
            "pass3.source_binding_inconsistent",
        )
    runtime = manifest.get("runtime")
    if (
        not isinstance(runtime, dict)
        or runtime.get("layer_checkpoint_step") != target_runtime_checkpoint_step
    ):
        raise Pass3InputError(
            f"{role}: report runtime target does not match Pass 2",
            "pass3.runtime_checkpoint_step_mismatch",
        )
    return SourceArtifactBinding(
        role=role,
        run_report_sha256=actual_sha256,
        trace_sha256=identity.trace_sha256,
        artifact_set_id=identity.artifact_set_id,
        semantic_manifest_sha256=manifest_sha,
    )


def validate_both_source_bindings(
    pass2: Pass2Result,
    reference_report: CanonicalRunReport,
    candidate_report: CanonicalRunReport,
    reference_trace: CanonicalLayerTrace,
    candidate_trace: CanonicalLayerTrace,
) -> tuple[SourceArtifactBinding, SourceArtifactBinding]:
    """Complete both role chains without accessing either trace summary."""
    target = pass2.target.expected_runtime_checkpoint_step
    if isinstance(target, bool) or not isinstance(target, int) or target < 0:
        raise Pass3InputError(
            "Pass 2 target runtime checkpoint step is unavailable",
            "pass3.pass2_not_ready",
        )
    reference_source, candidate_source = selected_source_hashes(pass2)
    # Validate both immutable canonical hashes before materializing either
    # report. This mirrors the existing Pass 2 source-before-metadata rule.
    for role_and_hash, report in (
        (reference_source, reference_report),
        (candidate_source, candidate_report),
    ):
        try:
            actual = report.identity.run_report_sha256
        except AttributeError as exc:
            raise Pass3InputError(
                f"{role_and_hash[0]}: canonical run report is required",
                "pass3.run_report_canonical_sha_inconsistent",
            ) from exc
        if actual != role_and_hash[1]:
            raise Pass3InputError(
                f"{role_and_hash[0]}: run-report canonical SHA-256 does not match Pass 2",
                "pass3.run_report_canonical_sha_inconsistent",
            )
    reference = _validated_run_report_binding(
        reference_source[0], reference_source[1], reference_report,
        reference_trace, target,
    )
    candidate = _validated_run_report_binding(
        candidate_source[0], candidate_source[1], candidate_report,
        candidate_trace, target,
    )
    return reference, candidate


@dataclass(frozen=True)
class CheckpointSummaryArtifact:
    model_family: str
    precision_path: str
    coverage: CheckpointCoverage
    summaries: tuple[CheckpointSummary, ...]


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Pass3InputError(
            f"{label} must be a non-negative integer",
            "pass3.checkpoint_summary_malformed",
        )
    return value


def _positive_int(value: Any, label: str) -> int:
    value = _nonnegative_int(value, label)
    if value == 0:
        raise Pass3InputError(
            f"{label} must be positive",
            "pass3.checkpoint_summary_malformed",
        )
    return value


def _coordinate(raw: Any, label: str) -> CheckpointCoordinate:
    if not isinstance(raw, dict):
        raise Pass3InputError(
            f"{label} coordinate must be an object",
            "pass3.checkpoint_summary_malformed",
        )
    required = (
        "runtime_checkpoint_step",
        "layer_index",
        "tensor_role",
        "batch_index",
        "sequence_index",
        "stage_order",
        "execution_ordinal",
    )
    if any(name not in raw for name in required):
        raise Pass3InputError(
            f"{label} coordinate is incomplete",
            "pass3.checkpoint_summary_malformed",
        )
    role = raw["tensor_role"]
    if not isinstance(role, str) or not role:
        raise Pass3InputError(
            f"{label} tensor_role is malformed",
            "pass3.checkpoint_summary_malformed",
        )
    try:
        return CheckpointCoordinate(
            _nonnegative_int(raw["runtime_checkpoint_step"], f"{label}.step"),
            _nonnegative_int(raw["layer_index"], f"{label}.layer"),
            role,
            _nonnegative_int(raw["batch_index"], f"{label}.batch"),
            _nonnegative_int(raw["sequence_index"], f"{label}.sequence"),
            _nonnegative_int(raw["stage_order"], f"{label}.stage"),
            _nonnegative_int(raw["execution_ordinal"], f"{label}.ordinal"),
        )
    except ValueError as exc:
        raise Pass3InputError(
            f"{label}: {exc}", "pass3.checkpoint_summary_malformed"
        ) from exc


def _validate_coordinate_order(
    coordinates: tuple[CheckpointCoordinate, ...], label: str
) -> None:
    previous = None
    logical = set()
    for coordinate in coordinates:
        if coordinate.logical_key in logical:
            raise Pass3InputError(
                f"{label} contains a duplicate logical coordinate",
                "pass3.duplicate_checkpoint_coordinate",
            )
        logical.add(coordinate.logical_key)
        if previous is not None and coordinate.order_key <= previous:
            raise Pass3InputError(
                f"{label} is not in contracted execution order",
                "pass3.checkpoint_alignment_inconsistent",
            )
        previous = coordinate.order_key


def _shape(raw: Any, label: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise Pass3InputError(
            f"{label} shape must be a non-empty array",
            "pass3.checkpoint_summary_malformed",
        )
    return tuple(_positive_int(value, f"{label}.shape") for value in raw)


def _digest(raw: Any, summary_shape: tuple[int, ...], role: str) -> CheckpointDigest:
    if not isinstance(raw, dict):
        raise Pass3InputError(
            "checkpoint digest is required",
            "pass3.summary_field_missing",
        )
    digest_shape = _shape(raw.get("shape"), "digest")
    value = raw.get("value")
    fields = (
        raw.get("algorithm"),
        raw.get("version"),
        raw.get("tensor_role"),
        raw.get("observed_dtype"),
        raw.get("byte_order"),
        raw.get("canonicalization"),
    )
    if not all(isinstance(item, str) for item in fields) or not isinstance(value, str):
        raise Pass3InputError(
            "checkpoint digest envelope is malformed",
            "pass3.checkpoint_summary_malformed",
        )
    if (
        fields
        != (
            DIGEST_ALGORITHM,
            DIGEST_VERSION,
            role,
            DIGEST_DTYPE,
            DIGEST_BYTE_ORDER,
            DIGEST_CANONICALIZATION,
        )
        or digest_shape != summary_shape
        or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
    ):
        raise Pass3InputError(
            "checkpoint digest contract is incompatible",
            "pass3.checkpoint_digest_incompatible",
        )
    return CheckpointDigest(
        fields[0], fields[1], fields[2], digest_shape,
        fields[3], fields[4], fields[5], value,
    )


def _has_prohibited_tensor_payload(value: Any) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ("tensor_payload", "tensor_values", "values"):
                return True
            if _has_prohibited_tensor_payload(child):
                return True
    elif isinstance(value, list):
        return any(_has_prohibited_tensor_payload(item) for item in value)
    return False


def normalize_layer_trace(trace: CanonicalLayerTrace) -> CheckpointSummaryArtifact:
    """Gate C/D parser; caller must complete both bindings first."""
    if not isinstance(trace, CanonicalLayerTrace):
        raise TypeError("trace must be a CanonicalLayerTrace")
    raw = trace.materialize()
    if _has_prohibited_tensor_payload(raw):
        raise Pass3InputError(
            "full tensor payloads are prohibited",
            "pass3.checkpoint_summary_malformed",
        )
    layout = raw.get("checkpoint_layout")
    if not isinstance(layout, dict):
        raise Pass3InputError(
            "versioned checkpoint layout is unavailable",
            "pass3.unsupported_checkpoint_layout",
        )
    expected_layout = (
        layout.get("layout_name") == "llama_layer_output_summary"
        and layout.get("layout_version") == 1
        and layout.get("tensor_role") == "layer_output"
        and layout.get("stage_order") == 0
        and layout.get("ordering_semantics")
        == "runtime_step_layer_stage_ordinal"
        and layout.get("duplicate_coordinate_policy")
        == "reject_artifact_before_write"
    )
    if not expected_layout:
        raise Pass3InputError(
            "unsupported checkpoint layout",
            "pass3.unsupported_checkpoint_layout",
        )
    target = _nonnegative_int(
        layout.get("runtime_checkpoint_step"), "layout target step"
    )
    if target != trace.identity.runtime_checkpoint_step:
        raise Pass3InputError(
            "layout target contradicts bound trace identity",
            "pass3.runtime_checkpoint_step_mismatch",
        )
    total_layer_count = _positive_int(
        layout.get("total_layer_count"), "total_layer_count"
    )
    requested_raw = layout.get("requested_coordinates")
    captured_raw = layout.get("captured_coordinates")
    missing_raw = layout.get("missing_coordinates")
    if not all(isinstance(item, list) for item in (requested_raw, captured_raw, missing_raw)):
        raise Pass3InputError(
            "explicit requested, captured, and missing coverage is required",
            "pass3.checkpoint_summary_malformed",
        )
    requested = tuple(
        _coordinate(item, f"requested[{index}]")
        for index, item in enumerate(requested_raw)
    )
    captured = tuple(
        _coordinate(item, f"captured[{index}]")
        for index, item in enumerate(captured_raw)
    )
    missing = []
    for index, item in enumerate(missing_raw):
        if not isinstance(item, dict):
            raise Pass3InputError(
                f"missing[{index}] must be an object",
                "pass3.checkpoint_summary_malformed",
            )
        try:
            state = CoverageState(item.get("state"))
        except ValueError as exc:
            raise Pass3InputError(
                f"missing[{index}] has an unsupported state",
                "pass3.checkpoint_summary_malformed",
            ) from exc
        detail = item.get("detail")
        if detail is not None and not isinstance(detail, str):
            raise Pass3InputError(
                f"missing[{index}] detail must be a string",
                "pass3.checkpoint_summary_malformed",
            )
        missing.append(
            CoverageEntry(_coordinate(item.get("coordinate"), f"missing[{index}]"), state, detail)
        )
    missing_tuple = tuple(missing)
    _validate_coordinate_order(requested, "requested coverage")
    _validate_coordinate_order(captured, "captured coverage")
    _validate_coordinate_order(
        tuple(item.coordinate for item in missing_tuple), "missing coverage"
    )
    requested_keys = {item.logical_key for item in requested}
    captured_keys = {item.logical_key for item in captured}
    missing_keys = {item.coordinate.logical_key for item in missing_tuple}
    if (
        captured_keys & missing_keys
        or requested_keys != captured_keys | missing_keys
        or not captured_keys.issubset(requested_keys)
        or any(item.layer_index >= total_layer_count for item in requested)
    ):
        raise Pass3InputError(
            "checkpoint coverage declarations are inconsistent",
            "pass3.checkpoint_summary_malformed",
        )
    for coordinate in requested:
        if (
            coordinate.runtime_checkpoint_step != target
            or coordinate.tensor_role != "layer_output"
            or coordinate.batch_index != 0
            or coordinate.sequence_index != 0
            or coordinate.stage_order != 0
        ):
            raise Pass3InputError(
                "requested coordinate violates the MVP layout",
                "pass3.checkpoint_alignment_inconsistent",
            )

    layout_fields = layout.get("available_summary_fields")
    digest_contract = layout.get("digest_contract")
    if (
        not isinstance(layout_fields, list)
        or not all(isinstance(item, str) for item in layout_fields)
        or "digest" not in layout_fields
        or not isinstance(digest_contract, dict)
        or digest_contract
        != {
            "algorithm": DIGEST_ALGORITHM,
            "version": DIGEST_VERSION,
            "observed_dtype": DIGEST_DTYPE,
            "byte_order": DIGEST_BYTE_ORDER,
            "canonicalization": DIGEST_CANONICALIZATION,
        }
    ):
        raise Pass3InputError(
            "layout digest contract is incompatible",
            "pass3.checkpoint_digest_incompatible",
        )

    manifest = raw.get("manifest")
    model = manifest.get("model") if isinstance(manifest, dict) else None
    runtime = manifest.get("runtime") if isinstance(manifest, dict) else None
    if not isinstance(model, dict) or not isinstance(runtime, dict):
        raise Pass3InputError(
            "trace semantic manifest is malformed",
            "pass3.binding_metadata_missing",
        )
    family = model.get("family")
    precision_path = runtime.get("precision_path")
    if family != "llama3_decoder":
        raise Pass3InputError(
            "only the Llama layer-output layout is supported",
            "pass3.unsupported_checkpoint_layout",
        )
    if not isinstance(precision_path, str) or not precision_path:
        raise Pass3InputError(
            "precision path is required",
            "pass3.checkpoint_summary_malformed",
        )

    entries = raw.get("layer_trace")
    if not isinstance(entries, list):
        raise Pass3InputError(
            "layer_trace must be an array",
            "pass3.checkpoint_summary_malformed",
        )
    summaries = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise Pass3InputError(
                f"layer_trace[{index}] must be an object",
                "pass3.checkpoint_summary_malformed",
            )
        semantic_markers = {
            "runtime_checkpoint_step", "layer_index", "tensor_role",
            "batch_index", "sequence_index", "stage_order",
            "execution_ordinal", "observed_dtype", "element_count",
        }
        present = semantic_markers.intersection(entry)
        if not present:
            continue
        if present != semantic_markers:
            raise Pass3InputError(
                f"layer_trace[{index}] has partial semantic coordinates",
                "pass3.checkpoint_summary_malformed",
            )
        coordinate = _coordinate(entry, f"layer_trace[{index}]")
        shape = _shape(entry.get("shape"), f"layer_trace[{index}]")
        element_count = _nonnegative_int(
            entry.get("element_count"), f"layer_trace[{index}].element_count"
        )
        if element_count != prod(shape):
            raise Pass3InputError(
                f"layer_trace[{index}] element count contradicts shape",
                "pass3.checkpoint_summary_malformed",
            )
        name = entry.get("name")
        phase = entry.get("phase")
        observed_dtype = entry.get("observed_dtype")
        available = entry.get("available_summary_fields")
        if (
            name != f"layer.{coordinate.layer_index}.output"
            or not isinstance(phase, str)
            or observed_dtype != DIGEST_DTYPE
            or not isinstance(available, list)
            or not all(isinstance(item, str) for item in available)
            or set(available)
            not in (
                {"min", "max", "mean", "l2", "nan", "inf"},
                {"min", "max", "mean", "l2", "nan", "inf", "digest"},
            )
        ):
            raise Pass3InputError(
                f"layer_trace[{index}] semantic metadata is malformed",
                "pass3.checkpoint_summary_malformed",
            )
        numeric = []
        for field in ("min", "max", "mean", "l2"):
            value = entry.get(field)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                raise Pass3InputError(
                    f"layer_trace[{index}].{field} is malformed",
                    "pass3.checkpoint_summary_malformed",
                )
            numeric.append(float(value) if value is not None else None)
        nan = entry.get("nan")
        inf = entry.get("inf")
        if nan not in (0, 1, False, True) or inf not in (0, 1, False, True):
            raise Pass3InputError(
                f"layer_trace[{index}] nonfinite flags are malformed",
                "pass3.checkpoint_summary_malformed",
            )
        if any(value is None for value in numeric) and not (bool(nan) or bool(inf)):
            raise Pass3InputError(
                f"layer_trace[{index}] null aggregate lacks nonfinite metadata",
                "pass3.checkpoint_summary_malformed",
            )
        digest = _digest(entry.get("digest"), shape, coordinate.tensor_role)
        summaries.append(
            CheckpointSummary(
                coordinate, name, phase, shape, observed_dtype,
                precision_path, element_count, numeric[0], numeric[1],
                numeric[2], None, numeric[3], None, None, None,
                bool(nan), bool(inf), digest, frozenset(available),
            )
        )
    summary_tuple = tuple(summaries)
    summary_coordinates = tuple(item.coordinate for item in summary_tuple)
    _validate_coordinate_order(summary_coordinates, "layer summaries")
    if {item.logical_key for item in summary_coordinates} != captured_keys:
        raise Pass3InputError(
            "captured coverage does not equal semantic summary coordinates",
            "pass3.checkpoint_summary_malformed",
        )
    if summary_coordinates != captured:
        raise Pass3InputError(
            "captured coordinates and summaries disagree on order metadata",
            "pass3.checkpoint_alignment_inconsistent",
        )
    coverage = CheckpointCoverage(
        requested, captured, missing_tuple, requested,
        total_layer_count, "llama_layer_output_summary",
    )
    return CheckpointSummaryArtifact(
        family, precision_path, coverage, summary_tuple
    )
