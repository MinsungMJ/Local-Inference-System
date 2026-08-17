"""Strict Pass 4 parent loading, classification, and source binding.

This module implements only P4-3.  It validates the two Pass 3 generations and
their already-produced source artifacts, then returns frozen P4-2 parent/source
models.  It does not parse local checkpoint payloads, derive coverage, compute
or compare digests, localize a mismatch, build a Pass4Result, or serialize a
Pass 4 artifact.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from typing import Any, Optional

from .pass1_inputs import (
    CanonicalRunReport,
    MalformedRunReport,
    canonical_json,
    canonical_json_sha256,
    sha256_text,
    strict_json_loads,
)
from .pass3_artifact import serialize as serialize_pass3
from .pass3_inputs import (
    CanonicalLayerTrace,
    CanonicalPass2Artifact,
    Pass3InputError,
)
from .pass3_model import (
    BOUNDED_EQUALITY_SEMANTICS,
    CONTRACT_VERSION as PASS3_CONTRACT_VERSION,
    DIGEST_DECISION_FIELD,
    DIGEST_DECISION_SEMANTICS,
    DIGEST_VERSION as PASS3_MODEL_DIGEST_VERSION,
    NON_CONFIRMATION_SEMANTICS,
    CheckpointCoordinate,
    CoverageState,
    Pass3DownstreamDisposition,
    Pass3Result,
    Pass3Status,
    SummaryEvidenceLevel,
)
from .pass4_contract import (
    DIAGNOSTIC_CAPTURE_PROFILE,
    INTRA_LAYER_STAGES,
    MODEL_FAMILY,
    NONCLAIMS,
    OUTER_TRACE_KIND,
    PASS3_DIGEST_VERSION,
    PRIMARY_REASONS,
    REASON_ALLOWED_STATUSES,
    SCHEMA,
    STATUS_TO_DISPOSITION,
    ParentSourceIdentity,
    Pass3ParentBinding,
    Pass3ParentClassification,
    Pass3ParentRole,
    Pass4Disposition,
    Pass4ReasonCode,
    Pass4Status,
    _ARTIFACT_SET_ID,
    _SHA256,
    validate_pass3_parent_pair,
    validate_status_algebra,
)
from .pass4_model import (
    FIELD_POLICY,
    MAX_DETAIL_BYTES,
    MAX_IDENTIFIER_BYTES,
    MAX_WARNINGS,
    PARENT_CLASSIFICATION_FOR_STATUS,
    REPRODUCTION_REQUEST_ONLY,
    Pass3ParentEvidence,
    Pass4SourceBinding,
)


MAX_PASS3_ARTIFACT_BYTES = 4 * 1024 * 1024
MAX_PASS2_ARTIFACT_BYTES = 1 * 1024 * 1024
MAX_RUN_REPORT_BYTES = 4 * 1024 * 1024
MAX_LAYER_TRACE_BYTES = 16 * 1024 * 1024
MAX_JSON_DEPTH = 64
MAX_OBJECT_KEYS = 4096
MAX_ARRAY_ITEMS = 65536
MAX_PATH_BYTES = 4096
MAX_IDENTITY_BYTES = MAX_IDENTIFIER_BYTES

P4_3_TERMINAL_STATUSES = frozenset(
    (
        Pass4Status.NOT_APPLICABLE,
        Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
        Pass4Status.UNSUPPORTED_PARENT,
        Pass4Status.PARENT_REVALIDATION_INCONSISTENT,
        Pass4Status.SOURCE_BINDING_INCONSISTENT,
    )
)

_PASS3_BLOCKED_STATUSES = frozenset(
    (
        Pass3Status.COMPARISON_BLOCKED_BY_PASS2,
        Pass3Status.INSUFFICIENT_COMMON_COVERAGE,
        Pass3Status.SOURCE_BINDING_INCONSISTENT,
        Pass3Status.CHECKPOINT_ALIGNMENT_INCONSISTENT,
        Pass3Status.CHECKPOINT_ARTIFACT_MISSING,
        Pass3Status.CHECKPOINT_SUMMARY_MALFORMED,
        Pass3Status.INCONCLUSIVE,
    )
)

_COORDINATE_FIELDS = (
    "runtime_checkpoint_step",
    "layer_index",
    "tensor_role",
    "batch_index",
    "sequence_index",
    "stage_order",
    "execution_ordinal",
)

_SEMANTIC_LIMITS = {
    "bounded_equality_semantics": BOUNDED_EQUALITY_SEMANTICS,
    "non_confirmation_semantics": NON_CONFIRMATION_SEMANTICS,
    "full_tensor_comparison_performed": False,
    "stage_localization_performed": False,
    "numeric_confirmation_performed": False,
    "pass4_or_pass5_readiness_certified": False,
    "automatic_frozen_success_mapping": False,
}

_CROSS_GENERATION_PATHS = (
    ("manifest.model.format", ("model", "format")),
    ("manifest.model.family", ("model", "family")),
    ("manifest.model.fingerprint", ("model", "fingerprint")),
    ("manifest.config.fingerprint", ("config", "fingerprint")),
    ("manifest.input.mode", ("input", "mode")),
    ("manifest.input.fingerprint", ("input", "fingerprint")),
    ("manifest.tokenizer", ("tokenizer",)),
    ("manifest.binary.fingerprint", ("binary", "fingerprint")),
    ("manifest.backend.name", ("backend", "name")),
    ("manifest.backend.fingerprint", ("backend", "fingerprint")),
    (
        "manifest.runtime.backend.name",
        ("runtime", "backend", "name"),
    ),
    (
        "manifest.runtime.backend.fingerprint",
        ("runtime", "backend", "fingerprint"),
    ),
    ("manifest.runtime.precision_path", ("runtime", "precision_path")),
    (
        "manifest.runtime.configured_context",
        ("runtime", "configured_context"),
    ),
    ("manifest.runtime.batch_size", ("runtime", "batch_size")),
    (
        "manifest.runtime.generation_limit",
        ("runtime", "generation_limit"),
    ),
    ("manifest.runtime.thread_count", ("runtime", "thread_count")),
    (
        "manifest.runtime.layer_checkpoints_enabled",
        ("runtime", "layer_checkpoints_enabled"),
    ),
    (
        "manifest.runtime.layer_checkpoint_step",
        ("runtime", "layer_checkpoint_step"),
    ),
    (
        "manifest.runtime.diagnostics_enabled",
        ("runtime", "diagnostics_enabled"),
    ),
    ("manifest.runtime.perf_enabled", ("runtime", "perf_enabled")),
    (
        "manifest.runtime.perf_per_token_enabled",
        ("runtime", "perf_per_token_enabled"),
    ),
)


def _bounded_detail(value: Any, fallback: str) -> str:
    if not isinstance(value, str) or not value:
        value = fallback
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= MAX_DETAIL_BYTES:
        return value
    return encoded[:MAX_DETAIL_BYTES].decode("utf-8", errors="ignore")


class Pass4ParentInputError(ValueError):
    """Internal evidence failure carrying its frozen classification."""

    def __init__(
        self,
        message: str,
        *,
        status: Pass4Status,
        reason: Pass4ReasonCode,
        detail: str = "",
    ):
        super().__init__(message)
        if not isinstance(status, Pass4Status):
            raise TypeError("status must be a Pass4Status")
        if not isinstance(reason, Pass4ReasonCode):
            raise TypeError("reason must be a Pass4ReasonCode")
        self.status = status
        self.reason = reason
        self.detail = _bounded_detail(detail, "artifact validation failed")


class Pass4ArtifactLoadError(Pass4ParentInputError):
    """A file could not be safely acquired or strictly parsed."""

    def __init__(
        self,
        label: str,
        detail: str,
        *,
        status: Pass4Status = Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
        reason: Pass4ReasonCode = Pass4ReasonCode.PARENT_ARTIFACT_MALFORMED,
    ):
        safe_label = (
            label
            if isinstance(label, str)
            and label
            and len(label.encode("utf-8")) <= MAX_IDENTIFIER_BYTES
            else "artifact"
        )
        safe_detail = _bounded_detail(detail, "could not load artifact")
        super().__init__(
            f"{safe_label}: {safe_detail}",
            status=status,
            reason=reason,
            detail=f"{safe_label}: {safe_detail}",
        )
        self.label = safe_label


def _fail(
    status: Pass4Status,
    reason: Pass4ReasonCode,
    detail: str,
) -> None:
    raise Pass4ParentInputError(
        detail,
        status=status,
        reason=reason,
        detail=detail,
    )


def _require_int(
    value: Any,
    label: str,
    *,
    minimum: int = 0,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
            label,
        )
    return value


def _require_string(
    value: Any,
    label: str,
    *,
    limit: int = MAX_IDENTIFIER_BYTES,
) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > limit
    ):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_ARTIFACT_MALFORMED,
            label,
        )
    return value


def _require_sha(
    value: Any,
    label: str,
    *,
    reason: Pass4ReasonCode = Pass4ReasonCode.PARENT_ARTIFACT_MALFORMED,
) -> str:
    if (
        not isinstance(value, str)
        or len(value.encode("utf-8")) > MAX_IDENTITY_BYTES
        or not _SHA256.fullmatch(value)
    ):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            reason,
            label,
        )
    return value


def _validate_path_shape(path: Any, label: str) -> str:
    try:
        value = os.fspath(path)
    except TypeError as exc:
        raise TypeError("path must be a string or path-like object") from exc
    if isinstance(value, bytes):
        try:
            value = os.fsdecode(value)
        except UnicodeError as exc:
            raise Pass4ArtifactLoadError(label, "path is not decodable") from exc
    if not isinstance(value, str):
        raise TypeError("path must resolve to a string")
    try:
        encoded = os.fsencode(value)
    except UnicodeError as exc:
        raise Pass4ArtifactLoadError(label, "path is not encodable") from exc
    if not encoded:
        raise Pass4ArtifactLoadError(label, "path is empty")
    if b"\0" in encoded:
        raise Pass4ArtifactLoadError(label, "path contains a NUL byte")
    if len(encoded) > MAX_PATH_BYTES:
        raise Pass4ArtifactLoadError(label, "path exceeds the byte bound")
    return value


def load_bounded_text(path, *, limit: int, label: str) -> str:
    """Read one regular, non-symlink UTF-8 file under an explicit byte cap."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise TypeError("limit must be a positive integer")
    if not isinstance(label, str) or not label:
        raise TypeError("label must be a non-empty string")
    path_value = _validate_path_shape(path, label)
    try:
        before = os.lstat(path_value)
    except OSError as exc:
        raise Pass4ArtifactLoadError(label, "file is missing or not statable") from exc
    if stat.S_ISLNK(before.st_mode):
        raise Pass4ArtifactLoadError(label, "symlinks are not accepted")
    if not stat.S_ISREG(before.st_mode):
        raise Pass4ArtifactLoadError(label, "path is not a regular file")
    if before.st_size > limit:
        raise Pass4ArtifactLoadError(label, "file exceeds the byte bound")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        try:
            descriptor = os.open(path_value, flags)
        except OSError as exc:
            raise Pass4ArtifactLoadError(label, "file could not be opened") from exc
        try:
            opened = os.fstat(descriptor)
        except OSError as exc:
            raise Pass4ArtifactLoadError(
                label, "opened file could not be statted"
            ) from exc
        if not stat.S_ISREG(opened.st_mode):
            raise Pass4ArtifactLoadError(label, "opened path is not a regular file")
        if opened.st_size > limit:
            raise Pass4ArtifactLoadError(label, "file exceeds the byte bound")
        chunks = []
        remaining = limit + 1
        while remaining:
            try:
                chunk = os.read(descriptor, min(remaining, 64 * 1024))
            except OSError as exc:
                raise Pass4ArtifactLoadError(label, "file could not be read") from exc
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if len(payload) > limit:
        raise Pass4ArtifactLoadError(label, "file exceeds the byte bound")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Pass4ArtifactLoadError(label, "file is not strict UTF-8") from exc
    if text.startswith("\ufeff"):
        raise Pass4ArtifactLoadError(label, "UTF-8 BOM is not accepted")
    return text


def _validate_json_bounds(
    raw: dict[str, Any],
    *,
    label: str,
    max_depth: int,
) -> None:
    pending: list[tuple[Any, int]] = [(raw, 1)]
    while pending:
        value, depth = pending.pop()
        if depth > max_depth:
            raise Pass4ArtifactLoadError(label, "JSON nesting exceeds the depth bound")
        if isinstance(value, dict):
            if len(value) > MAX_OBJECT_KEYS:
                raise Pass4ArtifactLoadError(
                    label, "JSON object exceeds the key-count bound"
                )
            pending.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list):
            if len(value) > MAX_ARRAY_ITEMS:
                raise Pass4ArtifactLoadError(
                    label, "JSON array exceeds the item-count bound"
                )
            pending.extend((child, depth + 1) for child in value)


def _strict_bounded_object(
    text: str,
    *,
    label: str,
    max_depth: int = MAX_JSON_DEPTH,
) -> dict[str, Any]:
    if not isinstance(text, str):
        raise TypeError("artifact text must be a string")
    if isinstance(max_depth, bool) or not isinstance(max_depth, int) or max_depth <= 0:
        raise TypeError("max_depth must be a positive integer")
    if text.startswith("\ufeff"):
        raise Pass4ArtifactLoadError(label, "UTF-8 BOM is not accepted")
    try:
        raw = strict_json_loads(text)
    except (MalformedRunReport, RecursionError) as exc:
        raise Pass4ArtifactLoadError(
            label, "JSON is malformed or excessively nested"
        ) from exc
    _validate_json_bounds(raw, label=label, max_depth=max_depth)
    return raw


def load_bounded_object(
    path,
    *,
    limit: int,
    label: str,
    max_depth: int = MAX_JSON_DEPTH,
) -> dict[str, Any]:
    text = load_bounded_text(path, limit=limit, label=label)
    return _strict_bounded_object(text, label=label, max_depth=max_depth)


@dataclass(frozen=True)
class CanonicalPass3Artifact:
    """Canonical Pass 3 representation using the existing serializer/helpers."""

    artifact_sha256: str
    canonical_text: str

    def __post_init__(self):
        if not isinstance(self.artifact_sha256, str):
            raise TypeError("artifact_sha256 must be a string")
        if not isinstance(self.canonical_text, str):
            raise TypeError("canonical_text must be a string")

    @classmethod
    def from_result(cls, result: Pass3Result) -> "CanonicalPass3Artifact":
        if not isinstance(result, Pass3Result):
            raise TypeError("result must be a Pass3Result")
        return cls.from_object(serialize_pass3(result))

    @classmethod
    def from_json(cls, text: str) -> "CanonicalPass3Artifact":
        raw = _strict_bounded_object(text, label="Pass 3 artifact")
        try:
            rendered = canonical_json(raw)
        except (MalformedRunReport, RecursionError) as exc:
            raise Pass4ArtifactLoadError(
                "Pass 3 artifact", "canonical JSON could not be produced"
            ) from exc
        return cls(sha256_text(rendered), rendered)

    @classmethod
    def from_object(cls, raw: dict[str, Any]) -> "CanonicalPass3Artifact":
        if not isinstance(raw, dict):
            raise TypeError("raw must be a JSON object")
        try:
            rendered = canonical_json(raw)
        except (MalformedRunReport, RecursionError) as exc:
            raise Pass4ArtifactLoadError(
                "Pass 3 artifact", "canonical JSON could not be produced"
            ) from exc
        return cls.from_json(rendered)

    @classmethod
    def load(cls, path) -> "CanonicalPass3Artifact":
        raw = load_bounded_object(
            path,
            limit=MAX_PASS3_ARTIFACT_BYTES,
            label="Pass 3 artifact",
        )
        return cls.from_object(raw)

    def materialize_verified(self) -> dict[str, Any]:
        try:
            raw = _strict_bounded_object(
                self.canonical_text,
                label="Pass 3 artifact",
            )
            rendered = canonical_json(raw)
        except (MalformedRunReport, RecursionError) as exc:
            raise Pass4ArtifactLoadError(
                "Pass 3 artifact", "canonical identity is malformed"
            ) from exc
        if (
            not _SHA256.fullmatch(self.artifact_sha256)
            or rendered != self.canonical_text
            or sha256_text(rendered) != self.artifact_sha256
        ):
            _fail(
                Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
                Pass4ReasonCode.PARENT_ARTIFACT_MALFORMED,
                "parent canonical identity inconsistent",
            )
        return raw


def _reclassify_load(
    exc: Pass4ArtifactLoadError,
    *,
    status: Pass4Status,
    reason: Pass4ReasonCode,
    label: str,
) -> Pass4ArtifactLoadError:
    return Pass4ArtifactLoadError(
        label,
        exc.detail.split(": ", 1)[-1],
        status=status,
        reason=reason,
    )


def load_canonical_run_report(path) -> CanonicalRunReport:
    try:
        raw = load_bounded_object(
            path, limit=MAX_RUN_REPORT_BYTES, label="run report"
        )
        return CanonicalRunReport.from_json(canonical_json(raw))
    except Pass4ArtifactLoadError as exc:
        raise _reclassify_load(
            exc,
            status=Pass4Status.SOURCE_BINDING_INCONSISTENT,
            reason=Pass4ReasonCode.RUN_REPORT_BINDING_MISMATCH,
            label="run report",
        ) from exc
    except (MalformedRunReport, RecursionError) as exc:
        raise Pass4ArtifactLoadError(
            "run report",
            "artifact is malformed",
            status=Pass4Status.SOURCE_BINDING_INCONSISTENT,
            reason=Pass4ReasonCode.RUN_REPORT_BINDING_MISMATCH,
        ) from exc


def load_canonical_layer_trace(path) -> CanonicalLayerTrace:
    try:
        raw = load_bounded_object(
            path, limit=MAX_LAYER_TRACE_BYTES, label="layer trace"
        )
        return CanonicalLayerTrace.from_json(canonical_json(raw))
    except Pass4ArtifactLoadError as exc:
        raise _reclassify_load(
            exc,
            status=Pass4Status.SOURCE_BINDING_INCONSISTENT,
            reason=Pass4ReasonCode.TRACE_SHA_MISMATCH,
            label="layer trace",
        ) from exc
    except (Pass3InputError, MalformedRunReport, RecursionError) as exc:
        raise Pass4ArtifactLoadError(
            "layer trace",
            "artifact is malformed",
            status=Pass4Status.SOURCE_BINDING_INCONSISTENT,
            reason=Pass4ReasonCode.TRACE_SHA_MISMATCH,
        ) from exc


def load_canonical_pass2_artifact(path) -> CanonicalPass2Artifact:
    try:
        raw = load_bounded_object(
            path, limit=MAX_PASS2_ARTIFACT_BYTES, label="Pass 2 artifact"
        )
        return CanonicalPass2Artifact.from_json(canonical_json(raw))
    except Pass4ArtifactLoadError as exc:
        raise _reclassify_load(
            exc,
            status=Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            reason=Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
            label="Pass 2 artifact",
        ) from exc
    except (Pass3InputError, MalformedRunReport, RecursionError) as exc:
        raise Pass4ArtifactLoadError(
            "Pass 2 artifact",
            "artifact is malformed",
            reason=Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
        ) from exc


def _coordinate(raw: Any, label: str) -> CheckpointCoordinate:
    if not isinstance(raw, dict) or set(raw) != set(_COORDINATE_FIELDS):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
            label,
        )
    values = {}
    for field in _COORDINATE_FIELDS:
        value = raw[field]
        if field == "tensor_role":
            if not isinstance(value, str) or not value:
                _fail(
                    Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
                    Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
                    label,
                )
        else:
            _require_int(value, label)
        values[field] = value
    try:
        return CheckpointCoordinate(**values)
    except ValueError as exc:
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
            label,
        )
        raise AssertionError from exc


def _coordinate_list(raw: Any, label: str) -> tuple[CheckpointCoordinate, ...]:
    if not isinstance(raw, list):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
            label,
        )
    result = tuple(_coordinate(item, label) for item in raw)
    seen = set()
    previous = None
    for item in result:
        if item.logical_key in seen or (
            previous is not None and item.order_key <= previous
        ):
            _fail(
                Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
                Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
                label,
            )
        seen.add(item.logical_key)
        previous = item.order_key
    return result


def _coverage_partition(
    coverage: dict[str, Any],
    side: str,
) -> tuple[
    tuple[CheckpointCoordinate, ...],
    tuple[CheckpointCoordinate, ...],
    tuple[CheckpointCoordinate, ...],
]:
    requested = _coordinate_list(
        coverage.get(f"{side}_requested"),
        f"parent.coverage.{side}_requested",
    )
    captured = _coordinate_list(
        coverage.get(f"{side}_captured"),
        f"parent.coverage.{side}_captured",
    )
    missing_raw = coverage.get(f"{side}_missing")
    if not isinstance(missing_raw, list):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
            f"parent.coverage.{side}_missing",
        )
    missing = []
    for entry in missing_raw:
        if not isinstance(entry, dict) or set(entry) != {
            "coordinate",
            "state",
            "detail",
        }:
            _fail(
                Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
                Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
                f"parent.coverage.{side}_missing",
            )
        coordinate = _coordinate(
            entry["coordinate"], f"parent.coverage.{side}_missing"
        )
        try:
            state = CoverageState(entry["state"])
        except (TypeError, ValueError) as exc:
            _fail(
                Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
                Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
                f"parent.coverage.{side}_missing",
            )
            raise AssertionError from exc
        if state == CoverageState.CAPTURED:
            _fail(
                Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
                Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
                f"parent.coverage.{side}_missing",
            )
        detail = entry["detail"]
        if detail is not None and not isinstance(detail, str):
            _fail(
                Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
                Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
                f"parent.coverage.{side}_missing",
            )
        missing.append(coordinate)
    missing_tuple = tuple(missing)
    seen = set()
    previous = None
    for item in missing_tuple:
        if item.logical_key in seen or (
            previous is not None and item.order_key <= previous
        ):
            _fail(
                Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
                Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
                f"parent.coverage.{side}_missing",
            )
        seen.add(item.logical_key)
        previous = item.order_key
    requested_keys = tuple(item.logical_key for item in requested)
    captured_keys = tuple(item.logical_key for item in captured)
    missing_keys = tuple(item.logical_key for item in missing_tuple)
    captured_set = set(captured_keys)
    if (
        not captured_set.issubset(set(requested_keys))
        or captured_set & set(missing_keys)
        or captured_keys
        != tuple(key for key in requested_keys if key in captured_set)
        or missing_keys
        != tuple(key for key in requested_keys if key not in captured_set)
    ):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
            f"parent.coverage.{side}",
        )
    return requested, captured, missing_tuple


def _validate_coverage_and_localization(
    raw: dict[str, Any],
    generation: str,
) -> None:
    coverage = raw.get("coverage")
    if not isinstance(coverage, dict):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
            f"{generation}.coverage",
        )
    ref_requested, ref_captured, _ = _coverage_partition(coverage, "reference")
    cand_requested, cand_captured, _ = _coverage_partition(coverage, "candidate")
    common = _coordinate_list(
        coverage.get("common_captured"), f"{generation}.coverage.common_captured"
    )
    ref_only = _coordinate_list(
        coverage.get("reference_only"), f"{generation}.coverage.reference_only"
    )
    cand_only = _coordinate_list(
        coverage.get("candidate_only"), f"{generation}.coverage.candidate_only"
    )
    comparable = _coordinate_list(
        coverage.get("common_comparable"),
        f"{generation}.coverage.common_comparable",
    )
    ref_keys = {item.logical_key for item in ref_captured}
    cand_keys = {item.logical_key for item in cand_captured}
    expected_common = tuple(
        item for item in ref_captured if item.logical_key in cand_keys
    )
    expected_ref_only = tuple(
        item for item in ref_captured if item.logical_key not in cand_keys
    )
    expected_cand_only = tuple(
        item for item in cand_captured if item.logical_key not in ref_keys
    )
    common_keys = {item.logical_key for item in common}
    comparable_keys = {item.logical_key for item in comparable}
    if (
        common != expected_common
        or ref_only != expected_ref_only
        or cand_only != expected_cand_only
        or any(key not in common_keys for key in comparable_keys)
        or comparable
        != tuple(item for item in common if item.logical_key in comparable_keys)
    ):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
            f"{generation}.coverage",
        )
    if (
        raw.get("pass3_status")
        in (
            Pass3Status.OBSERVABLE_MISMATCH_FOUND.value,
            Pass3Status.NO_MISMATCH_IN_CAPTURED_COVERAGE.value,
        )
        and (not ref_requested or not cand_requested)
    ):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
            f"{generation}.coverage.requested",
        )

    comparisons = raw.get("comparisons")
    if not isinstance(comparisons, dict):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_TYPED_ARTIFACT_INCOHERENT,
            f"{generation}.comparisons",
        )
    total = comparisons.get("total_count")
    serialized = comparisons.get("serialized_count")
    truncated = comparisons.get("truncated")
    items = comparisons.get("items")
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or total < 0
        or isinstance(serialized, bool)
        or not isinstance(serialized, int)
        or serialized < 0
        or not isinstance(truncated, bool)
        or not isinstance(items, list)
    ):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_TYPED_ARTIFACT_INCOHERENT,
            f"{generation}.comparisons",
        )
    if truncated or total != serialized or serialized != len(items):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_COMPARISONS_TRUNCATED,
            f"{generation}.comparisons",
        )
    item_coordinates = []
    equivalents = []
    for item in items:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("equivalent"), bool)
            or not isinstance(item.get("mismatching_fields"), list)
            or len(item["mismatching_fields"])
            != len(set(item["mismatching_fields"]))
            or any(
                not isinstance(value, str) or not value
                for value in item["mismatching_fields"]
            )
            or not isinstance(item.get("field_results"), list)
            or not isinstance(item.get("warnings"), list)
            or any(
                not isinstance(value, str) for value in item["warnings"]
            )
            or not isinstance(item.get("evidence_level"), str)
        ):
            _fail(
                Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
                Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
                f"{generation}.comparisons.items",
            )
        equivalent = item["equivalent"]
        expected_mismatches = [] if equivalent else [DIGEST_DECISION_FIELD]
        field_results = item["field_results"]
        if (
            item["mismatching_fields"] != expected_mismatches
            or len(field_results) != 1
            or not isinstance(field_results[0], dict)
            or field_results[0].get("field_name") != DIGEST_DECISION_FIELD
            or field_results[0].get("disposition") != "exact"
            or field_results[0].get("equivalent") is not equivalent
            or any(
                field_results[0].get(field) is not None
                for field in (
                    "abs_diff",
                    "resolved_abs_floor",
                    "resolved_rel_band",
                )
            )
        ):
            _fail(
                Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
                Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
                f"{generation}.comparisons.items",
            )
        item_coordinates.append(
            _coordinate(
                item.get("coordinate"), f"{generation}.comparisons.items"
            )
        )
        equivalents.append(equivalent)
    item_coordinates_tuple = tuple(item_coordinates)
    if len(item_coordinates_tuple) != len(
        {item.logical_key for item in item_coordinates_tuple}
    ):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
            f"{generation}.comparisons.items",
        )
    comparable_map = {item.logical_key: item for item in comparable}
    expected_items = tuple(
        comparable_map[item.logical_key]
        for item in item_coordinates_tuple
        if item.logical_key in comparable_map
    )
    if (
        len(expected_items) != len(item_coordinates_tuple)
        or expected_items != item_coordinates_tuple
    ):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
            f"{generation}.comparisons.items",
        )

    status = raw.get("pass3_status")
    localization = raw.get("localization")
    if not isinstance(localization, dict):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
            f"{generation}.localization",
        )
    mismatch_raw = localization.get("first_observed_mismatch_coordinate")
    mismatch = (
        _coordinate(mismatch_raw, f"{generation}.localization.first_mismatch")
        if mismatch_raw is not None
        else None
    )
    last_raw = localization.get("last_observed_equivalent_coordinate")
    last = (
        _coordinate(last_raw, f"{generation}.localization.last_equivalent")
        if last_raw is not None
        else None
    )
    if localization.get("last_observed_equivalent_layer") != (
        last.layer_index if last is not None else None
    ):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
            f"{generation}.localization.last_equivalent",
        )
    interval = localization.get("suspect_interval")
    if status == Pass3Status.OBSERVABLE_MISMATCH_FOUND.value:
        if mismatch is None or not isinstance(interval, dict):
            _fail(
                Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
                Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
                f"{generation}.localization",
            )
        try:
            first_false = equivalents.index(False)
        except ValueError as exc:
            _fail(
                Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
                Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
                f"{generation}.comparisons.prefix",
            )
            raise AssertionError from exc
        if (
            any(value is not True for value in equivalents[:first_false])
            or item_coordinates_tuple[first_false] != mismatch
            or mismatch.tensor_role != "layer_output"
            or mismatch.logical_key not in comparable_map
            or localization.get("first_observed_mismatching_layer")
            != mismatch.layer_index
            or localization.get("earliest_observable_suspect_layer")
            != mismatch.layer_index
        ):
            _fail(
                Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
                Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
                f"{generation}.localization",
            )
        expected_last = (
            item_coordinates_tuple[first_false - 1] if first_false else None
        )
        if last != expected_last:
            _fail(
                Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
                Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
                f"{generation}.localization.last_equivalent",
            )
        _validate_parent_interval(interval, last, mismatch, generation)
    elif mismatch is not None or interval is not None:
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
            f"{generation}.localization",
        )


def _validate_parent_interval(
    raw: dict[str, Any],
    last: Optional[CheckpointCoordinate],
    mismatch: CheckpointCoordinate,
    generation: str,
) -> None:
    if not isinstance(raw, dict):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
            f"{generation}.localization.suspect_interval",
        )
    interval_last = (
        _coordinate(
            raw.get("last_observed_equivalent_coordinate"),
            f"{generation}.localization.suspect_interval",
        )
        if raw.get("last_observed_equivalent_coordinate") is not None
        else None
    )
    interval_mismatch = _coordinate(
        raw.get("first_observed_mismatch_coordinate"),
        f"{generation}.localization.suspect_interval",
    )
    if interval_last != last or interval_mismatch != mismatch or raw.get(
        "end_inclusive"
    ) is not True:
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
            f"{generation}.localization.suspect_interval",
        )
    if last is None:
        expected = (
            raw.get("start_boundary") == "runtime_entry"
            and raw.get("start_exclusive") is False
            and raw.get("notation") == f"[entry, {mismatch.layer_index}]"
        )
        start_layer = -1
    else:
        expected = (
            raw.get("start_boundary") == "observed_checkpoint"
            and raw.get("start_exclusive") is True
            and raw.get("notation")
            == f"({last.layer_index}, {mismatch.layer_index}]"
        )
        start_layer = last.layer_index
    missing = raw.get("unobserved_layer_indices")
    expected_missing = list(range(start_layer + 1, mismatch.layer_index))
    if (
        not expected
        or not isinstance(missing, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in missing
        )
        or missing != expected_missing
    ):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
            f"{generation}.localization.suspect_interval",
        )


def _validate_parent_source_structure(raw: dict[str, Any], generation: str) -> None:
    binding = raw.get("source_binding")
    if not isinstance(binding, dict):
        return
    for side in ("reference", "candidate"):
        value = binding.get(side)
        if value is None:
            continue
        if not isinstance(value, dict):
            _fail(
                Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
                Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
                f"{generation}.source_binding.{side}",
            )
        _require_string(
            value.get("role"), f"{generation}.source_binding.{side}.role"
        )
        for field in (
            "run_report_sha256",
            "layer_trace_sha256",
            "semantic_manifest_sha256",
        ):
            _require_sha(
                value.get(field),
                f"{generation}.source_binding.{side}.{field}",
                reason=Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
            )
        set_id = value.get("artifact_set_id")
        if not isinstance(set_id, str) or not _ARTIFACT_SET_ID.fullmatch(set_id):
            _fail(
                Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
                Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
                f"{generation}.source_binding.{side}.artifact_set_id",
            )


def _validate_pass3_representation(
    raw: dict[str, Any],
    generation: str,
) -> None:
    if raw.get("schema") != SCHEMA or raw.get("kind") != "layer_localization":
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_ARTIFACT_MALFORMED,
            f"{generation}.identity",
        )
    _require_string(raw.get("contract_version"), f"{generation}.contract_version")
    status = raw.get("pass3_status")
    pass2_sha = raw.get("pass2_artifact_sha256")
    if pass2_sha is not None:
        _require_sha(
            pass2_sha,
            f"{generation}.pass2_artifact_sha256",
            reason=Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
        )
    elif status in (
        Pass3Status.OBSERVABLE_MISMATCH_FOUND.value,
        Pass3Status.NO_MISMATCH_IN_CAPTURED_COVERAGE.value,
    ):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
            f"{generation}.pass2_artifact_sha256",
        )
    if not isinstance(raw.get("pass2_object_artifact_coherence_verified"), bool):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
            f"{generation}.pass2_object_artifact_coherence_verified",
        )
    target = raw.get("target")
    if not isinstance(target, dict):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,
            f"{generation}.target",
        )
    _require_int(
        target.get("runtime_checkpoint_step"),
        f"{generation}.target.runtime_checkpoint_step",
    )
    if not isinstance(raw.get("checkpoint_artifact_binding_verified"), bool):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
            f"{generation}.checkpoint_artifact_binding_verified",
        )
    if status in (
        Pass3Status.OBSERVABLE_MISMATCH_FOUND.value,
        Pass3Status.NO_MISMATCH_IN_CAPTURED_COVERAGE.value,
    ):
        pass2_evidence = raw.get("pass2_evidence")
        if (
            raw.get("pass2_object_artifact_coherence_verified") is not True
            or raw.get("checkpoint_artifact_binding_verified") is not True
            or not isinstance(pass2_evidence, dict)
        ):
            _fail(
                Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
                Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
                f"{generation}.upstream_evidence",
            )
        tier = pass2_evidence.get("reproduction_evidence_tier")
        generated_step = pass2_evidence.get("generated_token_step")
        evidence_step = pass2_evidence.get("runtime_checkpoint_step")
        target_step = target.get("runtime_checkpoint_step")
        if (
            not isinstance(tier, str)
            or not tier
            or tier == REPRODUCTION_REQUEST_ONLY
            or isinstance(generated_step, bool)
            or not isinstance(generated_step, int)
            or generated_step < 0
            or isinstance(evidence_step, bool)
            or not isinstance(evidence_step, int)
            or evidence_step < 1
            or evidence_step != generated_step + 1
            or evidence_step != target_step
        ):
            _fail(
                Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
                Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
                f"{generation}.pass2_evidence",
            )
        source_binding = raw.get("source_binding")
        if not isinstance(source_binding, dict) or any(
            not isinstance(source_binding.get(side), dict)
            for side in ("reference", "candidate")
        ):
            _fail(
                Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
                Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
                f"{generation}.source_binding",
            )
    evidence = raw.get("evidence")
    if not isinstance(evidence, dict):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_TYPED_ARTIFACT_INCOHERENT,
            f"{generation}.evidence",
        )
    for field in (
        "decision_field",
        "decision_semantics",
        "evidence_level",
        "digest_contract_identity",
    ):
        value = evidence.get(field)
        if value is not None:
            _require_string(value, f"{generation}.evidence.{field}")
    if status in (
        Pass3Status.OBSERVABLE_MISMATCH_FOUND.value,
        Pass3Status.NO_MISMATCH_IN_CAPTURED_COVERAGE.value,
    ) and any(
        not isinstance(evidence.get(field), str) or not evidence[field]
        for field in (
            "decision_field",
            "decision_semantics",
            "evidence_level",
            "digest_contract_identity",
        )
    ):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_TYPED_ARTIFACT_INCOHERENT,
            f"{generation}.evidence",
        )
    if raw.get("semantic_limits") != _SEMANTIC_LIMITS:
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_TYPED_ARTIFACT_INCOHERENT,
            f"{generation}.semantic_limits",
        )
    reasons = raw.get("reason_codes")
    if (
        not isinstance(reasons, list)
        or not reasons
        or len(reasons) != len(set(reasons))
        or any(not isinstance(value, str) or not value for value in reasons)
    ):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_TYPED_ARTIFACT_INCOHERENT,
            f"{generation}.reason_codes",
        )
    disposition = raw.get("downstream_disposition")
    if status == Pass3Status.OBSERVABLE_MISMATCH_FOUND.value:
        expected = Pass3DownstreamDisposition.SUSPECT_INTERVAL_AVAILABLE.value
    elif status == Pass3Status.NO_MISMATCH_IN_CAPTURED_COVERAGE.value:
        expected = (
            Pass3DownstreamDisposition.EXPLORATORY_LOCALIZATION_ONLY.value
        )
    else:
        expected = Pass3DownstreamDisposition.BLOCKED.value
    if disposition != expected:
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_TYPED_ARTIFACT_INCOHERENT,
            f"{generation}.status",
        )
    _validate_parent_source_structure(raw, generation)
    _validate_coverage_and_localization(raw, generation)


def validate_pass3_artifact_coherence(
    result: Pass3Result,
    artifact: CanonicalPass3Artifact,
) -> tuple[dict[str, Any], str]:
    """Require complete equality with the existing Pass 3 representation."""
    if not isinstance(result, Pass3Result):
        raise TypeError("result must be a Pass3Result")
    if not isinstance(artifact, CanonicalPass3Artifact):
        raise TypeError("artifact must be a CanonicalPass3Artifact")
    raw = artifact.materialize_verified()
    if raw.get("schema") != SCHEMA or raw.get("kind") != "layer_localization":
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_ARTIFACT_MALFORMED,
            "parent unsupported identity",
        )
    try:
        expected = serialize_pass3(result)
    except (AttributeError, TypeError, ValueError, RecursionError) as exc:
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_TYPED_ARTIFACT_INCOHERENT,
            "parent typed representation malformed",
        )
        raise AssertionError from exc
    if raw != expected:
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_TYPED_ARTIFACT_INCOHERENT,
            "parent typed artifact disagreement",
        )
    _validate_pass3_representation(raw, "parent")
    return raw, artifact.artifact_sha256


def _source_identity(raw: dict[str, Any], side: str) -> ParentSourceIdentity:
    try:
        value = raw["source_binding"][side]
        return ParentSourceIdentity(
            value["run_report_sha256"],
            value["layer_trace_sha256"],
            value["semantic_manifest_sha256"],
            value["artifact_set_id"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
            f"parent.source_binding.{side}",
        )
        raise AssertionError from exc


def _parent_binding(
    raw: dict[str, Any],
    artifact_sha256: str,
    role: Pass3ParentRole,
) -> Pass3ParentBinding:
    try:
        return Pass3ParentBinding(
            role,
            artifact_sha256,
            raw["pass2_artifact_sha256"],
            _source_identity(raw, "reference"),
            _source_identity(raw, "candidate"),
            role == Pass3ParentRole.AUTHORITATIVE_PASS3B,
        )
    except (KeyError, TypeError, ValueError) as exc:
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
            "parent binding",
        )
        raise AssertionError from exc


def _materialize_pass2(artifact: CanonicalPass2Artifact) -> dict[str, Any]:
    try:
        raw = artifact.materialize_verified()
    except (Pass3InputError, MalformedRunReport, RecursionError, ValueError) as exc:
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
            "pass2.artifact",
        )
        raise AssertionError from exc
    if (
        raw.get("schema") != SCHEMA
        or raw.get("kind") != "prefix_policy_reproduction"
        or raw.get("contract_version") != PASS3_CONTRACT_VERSION
    ):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
            "pass2.identity",
        )
    return raw


def _pass2_binding(
    parent_raw: dict[str, Any],
    artifact: CanonicalPass2Artifact,
    *,
    label: str,
) -> dict[str, Any]:
    raw = _materialize_pass2(artifact)
    if artifact.artifact_sha256 != parent_raw.get("pass2_artifact_sha256"):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
            f"{label}.sha",
        )
    if parent_raw.get("pass2_object_artifact_coherence_verified") is not True:
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
            f"{label}.coherence",
        )
    tier = raw.get("reproduction_evidence_tier")
    if not isinstance(tier, str) or tier == REPRODUCTION_REQUEST_ONLY:
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
            f"{label}.tier",
        )
    if raw.get("pass2_status") != "reproduction_verified":
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
            f"{label}.status",
        )
    target = raw.get("target")
    parent_target = parent_raw.get("target")
    parent_evidence = parent_raw.get("pass2_evidence")
    if (
        not isinstance(target, dict)
        or not isinstance(parent_target, dict)
        or not isinstance(parent_evidence, dict)
    ):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
            f"{label}.target",
        )
    generated = target.get("generated_token_step")
    runtime_step = target.get("expected_runtime_checkpoint_step")
    if (
        isinstance(generated, bool)
        or not isinstance(generated, int)
        or generated < 0
        or isinstance(runtime_step, bool)
        or not isinstance(runtime_step, int)
        or runtime_step < 0
        or runtime_step != generated + 1
        or parent_target.get("runtime_checkpoint_step") != runtime_step
        or parent_evidence.get("runtime_checkpoint_step") != runtime_step
        or parent_evidence.get("generated_token_step") != generated
        or parent_evidence.get("reproduction_evidence_tier") != tier
    ):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
            f"{label}.step",
        )
    binding = raw.get("source_binding")
    if not isinstance(binding, dict):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
            f"{label}.source_binding",
        )
    if tier == "independent_rerun_verified":
        fields = (
            "reference_reproduction_sha256",
            "candidate_reproduction_sha256",
        )
    else:
        fields = (
            "reference_original_run_report_sha256",
            "candidate_original_run_report_sha256",
        )
    for side, field in zip(("reference", "candidate"), fields):
        parent_source = parent_raw.get("source_binding", {}).get(side)
        value = binding.get(field)
        if (
            not isinstance(parent_source, dict)
            or value != parent_source.get("run_report_sha256")
            or parent_evidence.get(field) != binding.get(field)
        ):
            _fail(
                Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
                Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
                f"{label}.source_binding.{side}",
            )
    localization = raw.get("localization_ref")
    if (
        not isinstance(localization, dict)
        or parent_evidence.get("localization_ref_sha256")
        != localization.get("sha256")
    ):
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
            f"{label}.localization_ref",
        )
    return raw


def _selected_layer(raw: dict[str, Any]) -> Optional[int]:
    localization = raw.get("localization")
    if not isinstance(localization, dict):
        return None
    value = localization.get("first_observed_mismatching_layer")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _target_step(raw: dict[str, Any]) -> Optional[int]:
    target = raw.get("target")
    value = target.get("runtime_checkpoint_step") if isinstance(target, dict) else None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _build_parent(
    classification: Pass3ParentClassification,
    discovery_binding: Pass3ParentBinding,
    authoritative_binding: Pass3ParentBinding,
    discovery_raw: dict[str, Any],
    authoritative_raw: dict[str, Any],
    authoritative_result: Pass3Result,
    *,
    cross_verified: bool,
    eligible_localization: bool,
) -> Pass3ParentEvidence:
    discovery_layer = _selected_layer(discovery_raw)
    authoritative_layer = _selected_layer(authoritative_raw)
    if (
        classification
        not in (
            Pass3ParentClassification.ELIGIBLE,
            Pass3ParentClassification.PARENT_REVALIDATION_INCONSISTENT,
        )
        and discovery_layer != authoritative_layer
    ):
        # Gate 3 outranks the gate-4 drift classification.  The frozen P4-2
        # model intentionally forbids retaining two contradictory layers on
        # any classification other than revalidation-inconsistent, so retain
        # only the already-established discovery side at this earlier exit.
        authoritative_layer = None
    values: dict[str, Any] = {
        "classification": classification,
        "discovery": discovery_binding,
        "authoritative": authoritative_binding,
        "typed_artifact_coherence_verified": True,
        "source_binding_verified": True,
        "cross_generation_semantic_coherence_verified": cross_verified,
        "discovery_selected_layer": discovery_layer,
        "authoritative_selected_layer": authoritative_layer,
        "target_runtime_checkpoint_step": _target_step(authoritative_raw),
    }
    if eligible_localization:
        evidence = authoritative_raw["evidence"]
        pass2_evidence = authoritative_raw["pass2_evidence"]
        values.update(
            parent_first_mismatch_coordinate=(
                authoritative_result.first_observed_mismatch_coordinate
            ),
            parent_last_observed_equivalent_coordinate=(
                authoritative_result.last_observed_equivalent_coordinate
            ),
            parent_suspect_interval=authoritative_result.suspect_interval,
            parent_evidence_level=authoritative_result.evidence_level,
            parent_decision_field=evidence["decision_field"],
            parent_decision_semantics=evidence["decision_semantics"],
            parent_digest_contract_identity=evidence[
                "digest_contract_identity"
            ],
            pass2_reproduction_evidence_tier=pass2_evidence[
                "reproduction_evidence_tier"
            ],
        )
    try:
        return Pass3ParentEvidence(**values)
    except (TypeError, ValueError) as exc:
        _fail(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_TYPED_ARTIFACT_INCOHERENT,
            "parent evidence construction",
        )
        raise AssertionError from exc


@dataclass(frozen=True)
class Pass4LoadedArtifactIdentities:
    discovery_pass3_sha256: str
    authoritative_pass3_sha256: str
    pass2_artifact_sha256: str
    discovery_pass2_artifact_sha256: Optional[str]
    discovery_reference: ParentSourceIdentity
    discovery_candidate: ParentSourceIdentity
    authoritative_reference: ParentSourceIdentity
    authoritative_candidate: ParentSourceIdentity

    def __post_init__(self):
        for label in (
            "discovery_pass3_sha256",
            "authoritative_pass3_sha256",
            "pass2_artifact_sha256",
        ):
            value = getattr(self, label)
            if not isinstance(value, str) or not _SHA256.fullmatch(value):
                raise ValueError(f"{label} must be a canonical SHA-256 identity")
        if self.discovery_pass2_artifact_sha256 is not None and not _SHA256.fullmatch(
            self.discovery_pass2_artifact_sha256
        ):
            raise ValueError(
                "discovery_pass2_artifact_sha256 must be canonical"
            )
        for label in (
            "discovery_reference",
            "discovery_candidate",
            "authoritative_reference",
            "authoritative_candidate",
        ):
            if not isinstance(getattr(self, label), ParentSourceIdentity):
                raise TypeError(f"{label} must be a ParentSourceIdentity")


def _bounded_tuple(values: Any, label: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{label} must be an immutable tuple")
    if len(values) > MAX_WARNINGS or len(values) != len(set(values)):
        raise ValueError(f"{label} must be unique and bounded")
    for value in values:
        if (
            not isinstance(value, str)
            or not value
            or len(value.encode("utf-8")) > MAX_DETAIL_BYTES
        ):
            raise ValueError(f"{label} contains an invalid entry")
    return values


@dataclass(frozen=True)
class Pass4ParentBindingOutcome:
    """Validated P4-3 evidence, or one frozen classified terminal outcome."""

    status: Optional[Pass4Status]
    disposition: Optional[Pass4Disposition]
    reason_codes: tuple[Pass4ReasonCode, ...]
    parent: Optional[Pass3ParentEvidence]
    reference_binding: Optional[Pass4SourceBinding]
    candidate_binding: Optional[Pass4SourceBinding]
    target_runtime_checkpoint_step: Optional[int]
    target_layer: Optional[int]
    model_family: Optional[str]
    precision_path: Optional[str]
    artifact_identities: Optional[Pass4LoadedArtifactIdentities]
    warnings: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    inherited_pass3_reason_codes: tuple[str, ...] = ()
    inherited_pass2_reason_codes: tuple[str, ...] = ()
    inherited_pass1_reason_codes: tuple[str, ...] = ()
    inherited_pass0_reason_codes: tuple[str, ...] = ()
    inherited_parent_warnings: tuple[str, ...] = ()

    def __post_init__(self):
        _bounded_tuple(self.warnings, "warnings")
        _bounded_tuple(self.diagnostics, "diagnostics")
        for label in (
            "inherited_pass3_reason_codes",
            "inherited_pass2_reason_codes",
            "inherited_pass1_reason_codes",
            "inherited_pass0_reason_codes",
            "inherited_parent_warnings",
        ):
            _bounded_tuple(getattr(self, label), label)
        if self.artifact_identities is not None and not isinstance(
            self.artifact_identities, Pass4LoadedArtifactIdentities
        ):
            raise TypeError(
                "artifact_identities must be Pass4LoadedArtifactIdentities"
            )
        if self.status is None:
            self._validate_proceed()
        else:
            self._validate_terminal()

    def _validate_proceed(self) -> None:
        if self.disposition is not None or self.reason_codes:
            raise ValueError("a proceed outcome has no terminal classification")
        if (
            not isinstance(self.parent, Pass3ParentEvidence)
            or self.parent.classification != Pass3ParentClassification.ELIGIBLE
            or not self.parent.authorizes_pass4_evidence
        ):
            raise ValueError("a proceed outcome requires an eligible parent")
        for binding in (self.reference_binding, self.candidate_binding):
            if (
                not isinstance(binding, Pass4SourceBinding)
                or not binding.parent_recorded_trace_binding_verified
            ):
                raise ValueError("a proceed outcome requires both verified bindings")
        if (
            isinstance(self.target_runtime_checkpoint_step, bool)
            or not isinstance(self.target_runtime_checkpoint_step, int)
            or self.target_runtime_checkpoint_step < 1
            or isinstance(self.target_layer, bool)
            or not isinstance(self.target_layer, int)
            or self.target_layer < 0
            or self.model_family != MODEL_FAMILY
            or not isinstance(self.precision_path, str)
            or not self.precision_path
            or len(self.precision_path.encode("utf-8")) > MAX_IDENTIFIER_BYTES
            or self.artifact_identities is None
        ):
            raise ValueError("a proceed outcome requires the complete target")
        if (
            self.parent.target_runtime_checkpoint_step
            != self.target_runtime_checkpoint_step
            or self.parent.authoritative_selected_layer != self.target_layer
        ):
            raise ValueError("the proceed target contradicts the parent")

    def _validate_terminal(self) -> None:
        if self.status not in P4_3_TERMINAL_STATUSES:
            raise ValueError("status is not reachable from P4-3")
        validate_status_algebra(
            self.status, self.disposition, self.reason_codes
        )
        if any(
            value is not None
            for value in (
                self.target_runtime_checkpoint_step,
                self.target_layer,
                self.model_family,
                self.precision_path,
            )
        ):
            raise ValueError("terminal P4-3 outcomes cannot retain a target")
        policy = FIELD_POLICY[self.status]
        parent_present = self.parent is not None
        binding_present = (
            self.reference_binding is not None
            or self.candidate_binding is not None
        )
        if (
            policy["parent_pass3"] == "required"
            and not parent_present
        ) or (
            policy["parent_pass3"] == "forbidden"
            and parent_present
        ):
            raise ValueError("terminal parent presence violates FIELD_POLICY")
        if (
            policy["bindings"] == "required"
            and not binding_present
        ) or (
            policy["bindings"] == "forbidden"
            and binding_present
        ):
            raise ValueError("terminal binding presence violates FIELD_POLICY")
        if parent_present:
            expected = PARENT_CLASSIFICATION_FOR_STATUS.get(
                self.status, Pass3ParentClassification.ELIGIBLE
            )
            if self.parent.classification != expected:
                raise ValueError("terminal parent classification is incoherent")

    @property
    def proceed(self) -> bool:
        return self.status is None


def _terminal(
    status: Pass4Status,
    reason: Pass4ReasonCode,
    *,
    parent: Optional[Pass3ParentEvidence] = None,
    reference_binding: Optional[Pass4SourceBinding] = None,
    candidate_binding: Optional[Pass4SourceBinding] = None,
    identities: Optional[Pass4LoadedArtifactIdentities] = None,
    warnings: tuple[str, ...] = (),
    diagnostic: Optional[str] = None,
    authoritative_result: Optional[Pass3Result] = None,
) -> Pass4ParentBindingOutcome:
    diagnostics = (
        (_bounded_detail(diagnostic, "validation failed"),)
        if diagnostic
        else ()
    )
    inherited = _inherited_parent_evidence(authoritative_result)
    return Pass4ParentBindingOutcome(
        status,
        STATUS_TO_DISPOSITION[status],
        (reason,),
        parent,
        reference_binding,
        candidate_binding,
        None,
        None,
        None,
        None,
        identities,
        warnings,
        diagnostics,
        *inherited,
    )


def _inherited_parent_evidence(
    result: Optional[Pass3Result],
) -> tuple[tuple[str, ...], ...]:
    if result is None:
        return ((), (), (), (), ())
    return (
        tuple(item.value for item in result.reason_codes),
        result.inherited_pass2_reason_codes,
        result.inherited_pass1_reason_codes,
        result.inherited_pass0_reason_codes,
        result.warnings,
    )


def _identity_bundle(
    discovery_raw: dict[str, Any],
    authoritative_raw: dict[str, Any],
    discovery_sha: str,
    authoritative_sha: str,
    pass2_sha: str,
    discovery_pass2_sha: Optional[str],
) -> Pass4LoadedArtifactIdentities:
    return Pass4LoadedArtifactIdentities(
        discovery_sha,
        authoritative_sha,
        pass2_sha,
        discovery_pass2_sha,
        _source_identity(discovery_raw, "reference"),
        _source_identity(discovery_raw, "candidate"),
        _source_identity(authoritative_raw, "reference"),
        _source_identity(authoritative_raw, "candidate"),
    )


def _parent_terminal(
    classification: Pass3ParentClassification,
    status: Pass4Status,
    reason: Pass4ReasonCode,
    *,
    discovery_binding: Pass3ParentBinding,
    authoritative_binding: Pass3ParentBinding,
    discovery_raw: dict[str, Any],
    authoritative_raw: dict[str, Any],
    authoritative_result: Pass3Result,
    identities: Pass4LoadedArtifactIdentities,
    warnings: tuple[str, ...],
    diagnostic: Optional[str] = None,
    cross_verified: bool = False,
) -> Pass4ParentBindingOutcome:
    parent = _build_parent(
        classification,
        discovery_binding,
        authoritative_binding,
        discovery_raw,
        authoritative_raw,
        authoritative_result,
        cross_verified=cross_verified,
        eligible_localization=False,
    )
    return _terminal(
        status,
        reason,
        parent=parent,
        identities=identities,
        warnings=warnings,
        diagnostic=diagnostic,
        authoritative_result=authoritative_result,
    )


def _gate3(
    discovery_result: Pass3Result,
    authoritative_result: Pass3Result,
    discovery_raw: dict[str, Any],
    authoritative_raw: dict[str, Any],
) -> Optional[tuple[Pass3ParentClassification, Pass4Status, Pass4ReasonCode, str]]:
    contract_version = authoritative_raw.get("contract_version")
    if contract_version != PASS3_CONTRACT_VERSION:
        return (
            Pass3ParentClassification.UNSUPPORTED_PARENT,
            Pass4Status.UNSUPPORTED_PARENT,
            Pass4ReasonCode.PARENT_CONTRACT_OR_FAMILY_UNSUPPORTED,
            "pass3b.contract_version",
        )
    status = authoritative_result.status
    if status == Pass3Status.UNSUPPORTED_CHECKPOINT_LAYOUT:
        return (
            Pass3ParentClassification.UNSUPPORTED_PARENT,
            Pass4Status.UNSUPPORTED_PARENT,
            Pass4ReasonCode.PARENT_CONTRACT_OR_FAMILY_UNSUPPORTED,
            "pass3b.status",
        )
    if status == Pass3Status.COMPARISON_POLICY_UNAVAILABLE:
        return (
            Pass3ParentClassification.UNSUPPORTED_PARENT,
            Pass4Status.UNSUPPORTED_PARENT,
            Pass4ReasonCode.PARENT_DIGEST_POLICY_UNSUPPORTED,
            "pass3b.status",
        )
    if status == Pass3Status.NO_MISMATCH_IN_CAPTURED_COVERAGE:
        return (
            Pass3ParentClassification.NOT_APPLICABLE,
            Pass4Status.NOT_APPLICABLE,
            Pass4ReasonCode.PARENT_HAS_NO_OBSERVED_MISMATCH,
            "",
        )
    if status in _PASS3_BLOCKED_STATUSES:
        return (
            Pass3ParentClassification.COMPARISON_BLOCKED_BY_PASS3,
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_STATUS_BLOCKED,
            f"pass3b.status={status.value}",
        )
    if status != Pass3Status.OBSERVABLE_MISMATCH_FOUND:
        return (
            Pass3ParentClassification.COMPARISON_BLOCKED_BY_PASS3,
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_STATUS_BLOCKED,
            "pass3b.status",
        )
    evidence = authoritative_raw["evidence"]
    if (
        evidence.get("evidence_level") != "tier1_bounded_digest"
        or evidence.get("decision_field") != DIGEST_DECISION_FIELD
        or evidence.get("decision_semantics") != DIGEST_DECISION_SEMANTICS
        or evidence.get("digest_contract_identity")
        != PASS3_MODEL_DIGEST_VERSION
    ):
        return (
            Pass3ParentClassification.UNSUPPORTED_PARENT,
            Pass4Status.UNSUPPORTED_PARENT,
            Pass4ReasonCode.PARENT_DIGEST_POLICY_UNSUPPORTED,
            "pass3b.evidence",
        )
    if _target_step(authoritative_raw) == 0:
        return (
            Pass3ParentClassification.UNSUPPORTED_PARENT,
            Pass4Status.UNSUPPORTED_PARENT,
            Pass4ReasonCode.PARENT_PHASE_UNSUPPORTED,
            "pass3b.target.runtime_checkpoint_step",
        )
    if discovery_result.status != Pass3Status.OBSERVABLE_MISMATCH_FOUND:
        return (
            Pass3ParentClassification.COMPARISON_BLOCKED_BY_PASS3,
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
            "pass3a.status",
        )
    discovery_evidence = discovery_raw["evidence"]
    if (
        discovery_raw.get("contract_version") != PASS3_CONTRACT_VERSION
        or discovery_evidence.get("evidence_level") != "tier1_bounded_digest"
        or discovery_evidence.get("decision_field") != DIGEST_DECISION_FIELD
        or discovery_evidence.get("decision_semantics")
        != DIGEST_DECISION_SEMANTICS
        or discovery_evidence.get("digest_contract_identity")
        != PASS3_DIGEST_VERSION
    ):
        return (
            Pass3ParentClassification.COMPARISON_BLOCKED_BY_PASS3,
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
            "pass3a.evidence",
        )
    return None


def _drift_path(
    discovery_raw: dict[str, Any],
    authoritative_raw: dict[str, Any],
) -> tuple[Optional[Pass4ReasonCode], Optional[str]]:
    discovery_layer = _selected_layer(discovery_raw)
    authoritative_layer = _selected_layer(authoritative_raw)
    if discovery_layer != authoritative_layer:
        return (
            Pass4ReasonCode.DISCOVERY_REBOUND_LAYER_CHANGED,
            f"discovery={discovery_layer} authoritative={authoritative_layer}",
        )
    comparisons = (
        (
            "target.runtime_checkpoint_step",
            _target_step(discovery_raw),
            _target_step(authoritative_raw),
        ),
        (
            "coverage.checkpoint_layout_basis",
            tuple(
                (item["tensor_role"], item["stage_order"])
                for item in discovery_raw["coverage"]["reference_requested"]
            ),
            tuple(
                (item["tensor_role"], item["stage_order"])
                for item in authoritative_raw["coverage"]["reference_requested"]
            ),
        ),
        (
            "coverage.reference_requested",
            discovery_raw["coverage"]["reference_requested"],
            authoritative_raw["coverage"]["reference_requested"],
        ),
        (
            "coverage.candidate_requested",
            discovery_raw["coverage"]["candidate_requested"],
            authoritative_raw["coverage"]["candidate_requested"],
        ),
        (
            "evidence.policy",
            tuple(
                discovery_raw["evidence"].get(field)
                for field in (
                    "evidence_level",
                    "decision_field",
                    "decision_semantics",
                    "digest_contract_identity",
                )
            ),
            tuple(
                authoritative_raw["evidence"].get(field)
                for field in (
                    "evidence_level",
                    "decision_field",
                    "decision_semantics",
                    "digest_contract_identity",
                )
            ),
        ),
        (
            "pass2_evidence.reproduction_evidence_tier",
            discovery_raw["pass2_evidence"].get(
                "reproduction_evidence_tier"
            ),
            authoritative_raw["pass2_evidence"].get(
                "reproduction_evidence_tier"
            ),
        ),
        (
            "pass2_evidence.generated_token_step",
            discovery_raw["pass2_evidence"].get("generated_token_step"),
            authoritative_raw["pass2_evidence"].get("generated_token_step"),
        ),
        (
            "pass2_evidence.runtime_checkpoint_step",
            discovery_raw["pass2_evidence"].get("runtime_checkpoint_step"),
            authoritative_raw["pass2_evidence"].get("runtime_checkpoint_step"),
        ),
    )
    for path, left, right in comparisons:
        if left != right:
            return Pass4ReasonCode.DISCOVERY_REBOUND_SEMANTICS_CHANGED, path
    return None, None


@dataclass(frozen=True)
class _BoundSource:
    parent_identity: ParentSourceIdentity
    role: str
    report_sha256: str
    trace_sha256: str
    manifest: dict[str, Any]
    report_raw: dict[str, Any]
    precision_path: str


def _report_materialize(
    report: CanonicalRunReport,
    slot: str,
) -> dict[str, Any]:
    try:
        raw = report.materialize()
    except (MalformedRunReport, RecursionError, ValueError) as exc:
        _fail(
            Pass4Status.SOURCE_BINDING_INCONSISTENT,
            Pass4ReasonCode.RUN_REPORT_BINDING_MISMATCH,
            slot,
        )
        raise AssertionError from exc
    return raw


def _bind_source(
    parent_raw: dict[str, Any],
    *,
    side: str,
    generation: str,
    report: CanonicalRunReport,
    trace: CanonicalLayerTrace,
    selected_layer: int,
    target_step: int,
) -> _BoundSource:
    slot = f"{generation}_{side}"
    parent_value = parent_raw["source_binding"][side]
    role = parent_value.get("role")
    expected_prefix = f"{side}_"
    if (
        not isinstance(role, str)
        or not role.startswith(expected_prefix)
        or len(role.encode("utf-8")) > MAX_IDENTIFIER_BYTES
    ):
        _fail(
            Pass4Status.SOURCE_BINDING_INCONSISTENT,
            Pass4ReasonCode.SOURCE_ROLE_MISMATCH,
            slot,
        )
    identity = trace.identity
    if identity.schema != SCHEMA or identity.kind != OUTER_TRACE_KIND:
        _fail(
            Pass4Status.SOURCE_BINDING_INCONSISTENT,
            Pass4ReasonCode.TRACE_SHA_MISMATCH,
            slot,
        )
    if (
        report.identity.schema != SCHEMA
        or report.identity.kind != "run_report"
    ):
        _fail(
            Pass4Status.SOURCE_BINDING_INCONSISTENT,
            Pass4ReasonCode.RUN_REPORT_BINDING_MISMATCH,
            slot,
        )
    if (
        report.identity.run_report_sha256
        != parent_value.get("run_report_sha256")
    ):
        _fail(
            Pass4Status.SOURCE_BINDING_INCONSISTENT,
            Pass4ReasonCode.RUN_REPORT_BINDING_MISMATCH,
            slot,
        )
    if identity.trace_sha256 != parent_value.get("layer_trace_sha256"):
        _fail(
            Pass4Status.SOURCE_BINDING_INCONSISTENT,
            Pass4ReasonCode.TRACE_SHA_MISMATCH,
            slot,
        )
    report_raw = _report_materialize(report, slot)
    manifest = report_raw.get("manifest")
    if not isinstance(manifest, dict):
        _fail(
            Pass4Status.SOURCE_BINDING_INCONSISTENT,
            Pass4ReasonCode.SEMANTIC_MANIFEST_BINDING_MISMATCH,
            slot,
        )
    try:
        manifest_sha = canonical_json_sha256(manifest)
    except (MalformedRunReport, RecursionError) as exc:
        _fail(
            Pass4Status.SOURCE_BINDING_INCONSISTENT,
            Pass4ReasonCode.SEMANTIC_MANIFEST_BINDING_MISMATCH,
            slot,
        )
        raise AssertionError from exc
    if (
        manifest_sha != identity.semantic_manifest_sha256
        or manifest_sha != parent_value.get("semantic_manifest_sha256")
    ):
        _fail(
            Pass4Status.SOURCE_BINDING_INCONSISTENT,
            Pass4ReasonCode.SEMANTIC_MANIFEST_BINDING_MISMATCH,
            slot,
        )
    report_set_id = report_raw.get("artifact_set_id")
    trace_set_id = identity.artifact_set_id
    parent_set_id = parent_value.get("artifact_set_id")
    if (
        not isinstance(report_set_id, str)
        or not _ARTIFACT_SET_ID.fullmatch(report_set_id)
        or not isinstance(trace_set_id, str)
        or not _ARTIFACT_SET_ID.fullmatch(trace_set_id)
        or report_set_id != trace_set_id
        or report_set_id != parent_set_id
    ):
        _fail(
            Pass4Status.SOURCE_BINDING_INCONSISTENT,
            Pass4ReasonCode.ARTIFACT_SET_BINDING_MISMATCH,
            slot,
        )
    model = manifest.get("model")
    runtime = manifest.get("runtime")
    if not isinstance(model, dict) or not isinstance(runtime, dict):
        _fail(
            Pass4Status.SOURCE_BINDING_INCONSISTENT,
            Pass4ReasonCode.RUNTIME_CAPTURE_IDENTITY_MISMATCH,
            f"{slot}.manifest",
        )
    for field in (
        "configured_context",
        "batch_size",
        "generation_limit",
        "thread_count",
    ):
        if field in runtime:
            value = runtime[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                _fail(
                    Pass4Status.SOURCE_BINDING_INCONSISTENT,
                    Pass4ReasonCode.RUNTIME_CAPTURE_IDENTITY_MISMATCH,
                    f"{slot}.manifest.runtime.{field}",
                )
    for field in (
        "layer_checkpoints_enabled",
        "diagnostics_enabled",
        "perf_enabled",
        "perf_per_token_enabled",
    ):
        if field in runtime and not isinstance(runtime[field], bool):
            _fail(
                Pass4Status.SOURCE_BINDING_INCONSISTENT,
                Pass4ReasonCode.RUNTIME_CAPTURE_IDENTITY_MISMATCH,
                f"{slot}.manifest.runtime.{field}",
            )
    prompt_count, prompt_count_present = _prompt_token_count(report_raw)
    if prompt_count_present and (
        isinstance(prompt_count, bool)
        or not isinstance(prompt_count, int)
        or prompt_count < 0
    ):
        _fail(
            Pass4Status.SOURCE_BINDING_INCONSISTENT,
            Pass4ReasonCode.RUNTIME_CAPTURE_IDENTITY_MISMATCH,
            f"{slot}.report.prompt_sequences[0].token_count",
        )
    if model.get("family") != MODEL_FAMILY:
        _fail(
            Pass4Status.UNSUPPORTED_PARENT,
            Pass4ReasonCode.PARENT_CONTRACT_OR_FAMILY_UNSUPPORTED,
            f"{slot}.manifest.model.family",
        )
    precision = runtime.get("precision_path")
    if (
        not isinstance(precision, str)
        or not precision
        or len(precision.encode("utf-8")) > MAX_IDENTIFIER_BYTES
    ):
        _fail(
            Pass4Status.SOURCE_BINDING_INCONSISTENT,
            Pass4ReasonCode.RUNTIME_CAPTURE_IDENTITY_MISMATCH,
            f"{slot}.manifest.runtime.precision_path",
        )
    layer_step = runtime.get("layer_checkpoint_step")
    if (
        isinstance(layer_step, bool)
        or not isinstance(layer_step, int)
        or layer_step != target_step
        or identity.runtime_checkpoint_step != target_step
    ):
        _fail(
            Pass4Status.SOURCE_BINDING_INCONSISTENT,
            Pass4ReasonCode.RUNTIME_CAPTURE_IDENTITY_MISMATCH,
            f"{slot}.manifest.runtime.layer_checkpoint_step",
        )
    enabled = runtime.get("intra_layer_checkpoints_enabled")
    capture_layer = runtime.get("intra_layer_target_layer")
    profile = runtime.get("diagnostic_capture_profile")
    if generation == "pass3b":
        valid_capture = (
            enabled is True
            and not isinstance(capture_layer, bool)
            and isinstance(capture_layer, int)
            and capture_layer == selected_layer
            and profile == DIAGNOSTIC_CAPTURE_PROFILE
        )
    else:
        valid_capture = (
            (enabled is None or enabled is False)
            and "intra_layer_target_layer" not in runtime
            and profile != DIAGNOSTIC_CAPTURE_PROFILE
        )
    if not valid_capture:
        _fail(
            Pass4Status.SOURCE_BINDING_INCONSISTENT,
            Pass4ReasonCode.RUNTIME_CAPTURE_IDENTITY_MISMATCH,
            f"{slot}.manifest.runtime.capture",
        )
    parent_identity = ParentSourceIdentity(
        parent_value["run_report_sha256"],
        parent_value["layer_trace_sha256"],
        parent_value["semantic_manifest_sha256"],
        parent_value["artifact_set_id"],
    )
    return _BoundSource(
        parent_identity,
        role,
        report.identity.run_report_sha256,
        identity.trace_sha256,
        manifest,
        report_raw,
        precision,
    )


def _prompt_token_count(raw: dict[str, Any]) -> tuple[Any, bool]:
    report = raw.get("report")
    prompts = report.get("prompt_sequences") if isinstance(report, dict) else None
    if not isinstance(prompts, list) or not prompts:
        return None, False
    first = prompts[0]
    if not isinstance(first, dict) or "token_count" not in first:
        return None, False
    return first["token_count"], True


def _at_path(value: dict[str, Any], path: tuple[str, ...]) -> tuple[Any, bool]:
    current: Any = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None, False
        current = current[key]
    return current, True


def _cross_generation(
    discovery: dict[str, _BoundSource],
    authoritative: dict[str, _BoundSource],
    warnings: list[str],
) -> Optional[str]:
    for side in ("reference", "candidate"):
        left = discovery[side]
        right = authoritative[side]
        if (
            left.trace_sha256 == right.trace_sha256
            or left.report_sha256 == right.report_sha256
        ):
            _fail(
                Pass4Status.SOURCE_BINDING_INCONSISTENT,
                Pass4ReasonCode.RUNTIME_CAPTURE_IDENTITY_MISMATCH,
                f"{side}.generation_identity",
            )
        for display, path in _CROSS_GENERATION_PATHS:
            left_value, left_present = _at_path(left.manifest, path)
            right_value, right_present = _at_path(right.manifest, path)
            if not left_present and not right_present:
                warning = (
                    "pass4.warn.cross_generation_field_unavailable:"
                    f"{display}"
                )
                if warning not in warnings and len(warnings) < MAX_WARNINGS:
                    warnings.append(warning)
                continue
            if (
                left_present != right_present
                or left_value != right_value
            ):
                return f"{side}.{display}"
        left_count, left_present = _prompt_token_count(left.report_raw)
        right_count, right_present = _prompt_token_count(right.report_raw)
        display = "report.prompt_sequences[0].token_count"
        if not left_present and not right_present:
            warning = (
                "pass4.warn.cross_generation_field_unavailable:"
                f"{display}"
            )
            if warning not in warnings and len(warnings) < MAX_WARNINGS:
                warnings.append(warning)
        elif left_present != right_present or left_count != right_count:
            return f"{side}.{display}"
    return None


def _typecheck_inputs(values: tuple[tuple[str, Any, type, bool], ...]) -> None:
    for name, value, expected, optional in values:
        if optional and value is None:
            continue
        if not isinstance(value, expected):
            raise TypeError(f"{name} must be a {expected.__name__}")


def bind_pass4_parent_inputs(
    discovery_pass3: Pass3Result,
    discovery_pass3_artifact: CanonicalPass3Artifact,
    authoritative_pass3: Pass3Result,
    authoritative_pass3_artifact: CanonicalPass3Artifact,
    pass2_artifact: CanonicalPass2Artifact,
    *,
    discovery_reference_report: CanonicalRunReport,
    discovery_candidate_report: CanonicalRunReport,
    discovery_reference_trace: CanonicalLayerTrace,
    discovery_candidate_trace: CanonicalLayerTrace,
    authoritative_reference_report: CanonicalRunReport,
    authoritative_candidate_report: CanonicalRunReport,
    authoritative_reference_trace: CanonicalLayerTrace,
    authoritative_candidate_trace: CanonicalLayerTrace,
    discovery_pass2_artifact: Optional[CanonicalPass2Artifact] = None,
) -> Pass4ParentBindingOutcome:
    """Execute frozen P4-3 gates 1 through 5 and construct P4-2 models."""
    _typecheck_inputs(
        (
            ("discovery_pass3", discovery_pass3, Pass3Result, False),
            (
                "discovery_pass3_artifact",
                discovery_pass3_artifact,
                CanonicalPass3Artifact,
                False,
            ),
            ("authoritative_pass3", authoritative_pass3, Pass3Result, False),
            (
                "authoritative_pass3_artifact",
                authoritative_pass3_artifact,
                CanonicalPass3Artifact,
                False,
            ),
            ("pass2_artifact", pass2_artifact, CanonicalPass2Artifact, False),
            (
                "discovery_reference_report",
                discovery_reference_report,
                CanonicalRunReport,
                False,
            ),
            (
                "discovery_candidate_report",
                discovery_candidate_report,
                CanonicalRunReport,
                False,
            ),
            (
                "discovery_reference_trace",
                discovery_reference_trace,
                CanonicalLayerTrace,
                False,
            ),
            (
                "discovery_candidate_trace",
                discovery_candidate_trace,
                CanonicalLayerTrace,
                False,
            ),
            (
                "authoritative_reference_report",
                authoritative_reference_report,
                CanonicalRunReport,
                False,
            ),
            (
                "authoritative_candidate_report",
                authoritative_candidate_report,
                CanonicalRunReport,
                False,
            ),
            (
                "authoritative_reference_trace",
                authoritative_reference_trace,
                CanonicalLayerTrace,
                False,
            ),
            (
                "authoritative_candidate_trace",
                authoritative_candidate_trace,
                CanonicalLayerTrace,
                False,
            ),
            (
                "discovery_pass2_artifact",
                discovery_pass2_artifact,
                CanonicalPass2Artifact,
                True,
            ),
        )
    )

    warnings: list[str] = []
    try:
        discovery_raw, discovery_sha = validate_pass3_artifact_coherence(
            discovery_pass3, discovery_pass3_artifact
        )
        authoritative_raw, authoritative_sha = (
            validate_pass3_artifact_coherence(
                authoritative_pass3, authoritative_pass3_artifact
            )
        )
        if authoritative_pass3.status in _PASS3_BLOCKED_STATUSES:
            return _terminal(
                Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
                Pass4ReasonCode.PARENT_STATUS_BLOCKED,
                warnings=tuple(warnings),
                diagnostic=(
                    f"pass3b.status={authoritative_pass3.status.value}"
                ),
                authoritative_result=authoritative_pass3,
            )
        _pass2_binding(
            authoritative_raw, pass2_artifact, label="pass3b.pass2"
        )
        discovery_pass2_sha = None
        if discovery_pass2_artifact is not None:
            _pass2_binding(
                discovery_raw,
                discovery_pass2_artifact,
                label="pass3a.pass2",
            )
            discovery_pass2_sha = discovery_pass2_artifact.artifact_sha256
        if (
            discovery_raw["pass2_artifact_sha256"]
            == authoritative_raw["pass2_artifact_sha256"]
        ):
            warnings.append("pass4.warn.pass2_lineage_not_rebuilt")
        discovery_binding = _parent_binding(
            discovery_raw, discovery_sha, Pass3ParentRole.DISCOVERY_PASS3A
        )
        authoritative_binding = _parent_binding(
            authoritative_raw,
            authoritative_sha,
            Pass3ParentRole.AUTHORITATIVE_PASS3B,
        )
        validate_pass3_parent_pair(discovery_binding, authoritative_binding)
        identities = _identity_bundle(
            discovery_raw,
            authoritative_raw,
            discovery_sha,
            authoritative_sha,
            pass2_artifact.artifact_sha256,
            discovery_pass2_sha,
        )
    except Pass4ParentInputError as exc:
        return _terminal(
            exc.status,
            exc.reason,
            warnings=tuple(warnings),
            diagnostic=exc.detail,
        )

    gate3 = _gate3(
        discovery_pass3,
        authoritative_pass3,
        discovery_raw,
        authoritative_raw,
    )
    if gate3 is not None:
        classification, status, reason, diagnostic = gate3
        if classification == Pass3ParentClassification.COMPARISON_BLOCKED_BY_PASS3:
            try:
                return _parent_terminal(
                    classification,
                    status,
                    reason,
                    discovery_binding=discovery_binding,
                    authoritative_binding=authoritative_binding,
                    discovery_raw=discovery_raw,
                    authoritative_raw=authoritative_raw,
                    authoritative_result=authoritative_pass3,
                    identities=identities,
                    warnings=tuple(warnings),
                    diagnostic=diagnostic or None,
                )
            except Pass4ParentInputError:
                return _terminal(
                    status,
                    reason,
                    identities=identities,
                    warnings=tuple(warnings),
                    diagnostic=diagnostic or None,
                    authoritative_result=authoritative_pass3,
                )
        return _parent_terminal(
            classification,
            status,
            reason,
            discovery_binding=discovery_binding,
            authoritative_binding=authoritative_binding,
            discovery_raw=discovery_raw,
            authoritative_raw=authoritative_raw,
            authoritative_result=authoritative_pass3,
            identities=identities,
            warnings=tuple(warnings),
            diagnostic=diagnostic or None,
        )

    drift_reason, drift_detail = _drift_path(
        discovery_raw, authoritative_raw
    )
    if drift_reason is not None:
        return _parent_terminal(
            Pass3ParentClassification.PARENT_REVALIDATION_INCONSISTENT,
            Pass4Status.PARENT_REVALIDATION_INCONSISTENT,
            drift_reason,
            discovery_binding=discovery_binding,
            authoritative_binding=authoritative_binding,
            discovery_raw=discovery_raw,
            authoritative_raw=authoritative_raw,
            authoritative_result=authoritative_pass3,
            identities=identities,
            warnings=tuple(warnings),
            diagnostic=drift_detail,
        )

    eligible_parent = _build_parent(
        Pass3ParentClassification.ELIGIBLE,
        discovery_binding,
        authoritative_binding,
        discovery_raw,
        authoritative_raw,
        authoritative_pass3,
        cross_verified=True,
        eligible_localization=True,
    )
    authoritative_layer = _selected_layer(authoritative_raw)
    target_step = _target_step(authoritative_raw)
    try:
        authoritative_sources = {
            "reference": _bind_source(
                authoritative_raw,
                side="reference",
                generation="pass3b",
                report=authoritative_reference_report,
                trace=authoritative_reference_trace,
                selected_layer=authoritative_layer,
                target_step=target_step,
            ),
            "candidate": _bind_source(
                authoritative_raw,
                side="candidate",
                generation="pass3b",
                report=authoritative_candidate_report,
                trace=authoritative_candidate_trace,
                selected_layer=authoritative_layer,
                target_step=target_step,
            ),
        }
        if (
            authoritative_sources["reference"].precision_path
            != authoritative_sources["candidate"].precision_path
        ):
            _fail(
                Pass4Status.SOURCE_BINDING_INCONSISTENT,
                Pass4ReasonCode.RUNTIME_CAPTURE_IDENTITY_MISMATCH,
                "pass3b.precision_path",
            )
        discovery_layer = _selected_layer(discovery_raw)
        discovery_step = _target_step(discovery_raw)
        discovery_sources = {
            "reference": _bind_source(
                discovery_raw,
                side="reference",
                generation="pass3a",
                report=discovery_reference_report,
                trace=discovery_reference_trace,
                selected_layer=discovery_layer,
                target_step=discovery_step,
            ),
            "candidate": _bind_source(
                discovery_raw,
                side="candidate",
                generation="pass3a",
                report=discovery_candidate_report,
                trace=discovery_candidate_trace,
                selected_layer=discovery_layer,
                target_step=discovery_step,
            ),
        }
        if (
            discovery_sources["reference"].precision_path
            != discovery_sources["candidate"].precision_path
        ):
            _fail(
                Pass4Status.SOURCE_BINDING_INCONSISTENT,
                Pass4ReasonCode.RUNTIME_CAPTURE_IDENTITY_MISMATCH,
                "pass3a.precision_path",
            )
        cross_detail = _cross_generation(
            discovery_sources, authoritative_sources, warnings
        )
    except Pass4ParentInputError as exc:
        if exc.status == Pass4Status.UNSUPPORTED_PARENT:
            return _parent_terminal(
                Pass3ParentClassification.UNSUPPORTED_PARENT,
                Pass4Status.UNSUPPORTED_PARENT,
                exc.reason,
                discovery_binding=discovery_binding,
                authoritative_binding=authoritative_binding,
                discovery_raw=discovery_raw,
                authoritative_raw=authoritative_raw,
                authoritative_result=authoritative_pass3,
                identities=identities,
                warnings=tuple(warnings),
                diagnostic=exc.detail,
            )
        return _terminal(
            Pass4Status.SOURCE_BINDING_INCONSISTENT,
            exc.reason,
            parent=eligible_parent,
            identities=identities,
            warnings=tuple(warnings),
            diagnostic=exc.detail,
            authoritative_result=authoritative_pass3,
        )
    if cross_detail is not None:
        return _parent_terminal(
            Pass3ParentClassification.PARENT_REVALIDATION_INCONSISTENT,
            Pass4Status.PARENT_REVALIDATION_INCONSISTENT,
            Pass4ReasonCode.DISCOVERY_REBOUND_SEMANTICS_CHANGED,
            discovery_binding=discovery_binding,
            authoritative_binding=authoritative_binding,
            discovery_raw=discovery_raw,
            authoritative_raw=authoritative_raw,
            authoritative_result=authoritative_pass3,
            identities=identities,
            warnings=tuple(warnings),
            diagnostic=cross_detail,
        )

    reference_binding = Pass4SourceBinding(
        authoritative_sources["reference"].role,
        authoritative_sources["reference"].parent_identity,
        True,
    )
    candidate_binding = Pass4SourceBinding(
        authoritative_sources["candidate"].role,
        authoritative_sources["candidate"].parent_identity,
        True,
    )
    precision_path = authoritative_sources["reference"].precision_path
    inherited = _inherited_parent_evidence(authoritative_pass3)
    return Pass4ParentBindingOutcome(
        None,
        None,
        (),
        eligible_parent,
        reference_binding,
        candidate_binding,
        target_step,
        authoritative_layer,
        MODEL_FAMILY,
        precision_path,
        identities,
        tuple(warnings),
        (),
        *inherited,
    )
