"""Strict, model-free run-report inputs for Pass 1.

The core Pass 1 API receives :class:`CanonicalRunReport` objects.  These
objects bind canonical parsed JSON to an immutable SHA-256 identity, allowing
the gate and source binding to be checked before selected-token extraction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .pass1_model import (
    NormalizedDigest,
    Pass0SourceBinding,
    Pass1ReasonCode,
    RunReportIdentity,
    RunReportMetadata,
    SelectedTokenEvidenceLevel,
    SelectedTokenSequence,
)


_SELECTED_ARRAY_PATHS = (
    ("report", "selected_token_ids"),
    ("selected_token_ids",),
)
_SELECTED_COUNT_PATHS = (
    ("report", "selected_token_count"),
    ("selected_token_count",),
)
_SELECTED_DIGEST_PATHS = (
    ("report", "selected_token_digest"),
    ("selected_token_digest",),
)


class RunReportInputError(ValueError):
    """Base class for fail-closed Pass 1 run-report errors."""

    def __init__(self, message: str, reason: Pass1ReasonCode):
        super().__init__(message)
        self.reason = reason


class MalformedRunReport(RunReportInputError):
    def __init__(self, message: str):
        super().__init__(
            message, Pass1ReasonCode.SELECTED_TOKEN_METADATA_INCONSISTENT
        )


class UnsupportedRunReport(RunReportInputError):
    def __init__(
        self,
        message: str,
        reason: Pass1ReasonCode = Pass1ReasonCode.UNSUPPORTED_RUN_ARTIFACT,
    ):
        super().__init__(message, reason)


def _pairs_no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise MalformedRunReport(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonstandard_constant(value: str):
    raise MalformedRunReport(f"non-standard JSON numeric constant: {value}")


def strict_json_loads(text: str) -> dict[str, Any]:
    """Parse one JSON object, rejecting duplicate keys and NaN/Infinity."""
    try:
        value = json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_nonstandard_constant,
        )
    except RunReportInputError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise MalformedRunReport(f"invalid run-report JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise MalformedRunReport("run-report root must be a JSON object")
    return value


def canonical_json(value: Any) -> str:
    """Deterministic compact JSON for already parsed JSON-compatible data."""
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise MalformedRunReport(
            f"run report is not canonical-JSON compatible: {exc}"
        ) from exc


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return sha256_text(canonical_json(value))


def token_ids_sha256(token_ids) -> str:
    return canonical_json_sha256(list(token_ids))


def _at(value: Any, path: tuple[str, ...]):
    current = value
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None, False
        current = current[key]
    return current, True


def _first(value: Any, paths):
    for path in paths:
        found, present = _at(value, path)
        if present:
            return found
    return None


def _normal_fingerprint(value: Any) -> Optional[str]:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        algorithm = value.get("algorithm")
        hex_value = value.get("hex")
        if isinstance(algorithm, str) and isinstance(hex_value, str):
            return f"{algorithm}:{hex_value}"
    return None


def _identity_from_object(raw: dict[str, Any], digest: str) -> RunReportIdentity:
    manifest = raw.get("manifest") if isinstance(raw.get("manifest"), dict) else {}
    return RunReportIdentity(
        run_report_sha256=digest,
        schema=raw.get("schema") if isinstance(raw.get("schema"), str) else None,
        kind=raw.get("kind") if isinstance(raw.get("kind"), str) else None,
        model_fingerprint=_normal_fingerprint(
            _first(
                manifest,
                (("model", "fingerprint"), ("model_fingerprint",)),
            )
        ),
        config_fingerprint=_normal_fingerprint(
            _first(
                manifest,
                (("config", "fingerprint"), ("config_fingerprint",)),
            )
        ),
        input_fingerprint=_normal_fingerprint(
            _first(
                manifest,
                (("input", "fingerprint"), ("input_fingerprint",)),
            )
        ),
    )


@dataclass(frozen=True)
class CanonicalRunReport:
    """Canonical parsed report retained as deterministic JSON text."""

    identity: RunReportIdentity
    canonical_text: str

    @classmethod
    def from_json(cls, text: str) -> "CanonicalRunReport":
        raw = strict_json_loads(text)
        rendered = canonical_json(raw)
        digest = sha256_text(rendered)
        return cls(
            identity=_identity_from_object(raw, digest),
            canonical_text=rendered,
        )

    @classmethod
    def from_object(cls, raw: dict[str, Any]) -> "CanonicalRunReport":
        # Reparse the deterministic rendering so this path has the same JSON
        # value normalization as from_json. Duplicate keys cannot exist in an
        # already-created dict; callers needing duplicate detection must use
        # from_json/load.
        return cls.from_json(canonical_json(raw))

    @classmethod
    def load(cls, path) -> "CanonicalRunReport":
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    def materialize(self) -> dict[str, Any]:
        # canonical_text was produced by strict parsing and cannot contain
        # duplicate object keys.
        return json.loads(self.canonical_text)


def build_source_binding(
    reference: CanonicalRunReport, candidate: CanonicalRunReport
) -> Pass0SourceBinding:
    return Pass0SourceBinding(reference.identity, candidate.identity)


def _require_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MalformedRunReport(f"{label} must be a non-negative integer")
    return value


def extract_run_metadata(
    raw: dict[str, Any], identity: RunReportIdentity, role: str
) -> RunReportMetadata:
    manifest = raw.get("manifest")
    report = raw.get("report")
    if raw.get("schema") != "lis.execution_artifact/v1":
        raise UnsupportedRunReport("unsupported or missing run-report schema")
    if raw.get("kind") != "run_report":
        raise UnsupportedRunReport("unsupported or missing run-report kind")
    if not isinstance(manifest, dict) or not isinstance(report, dict):
        raise UnsupportedRunReport("run report requires manifest and report objects")

    execution_status = report.get("execution_status")
    model_family = _first(
        manifest, (("model", "family"), ("model_family",))
    )
    batch_size = _first(
        manifest,
        (
            ("runtime", "batch_size"),
            ("runtime_settings", "batch_size"),
            ("runtime_settings", "batch"),
        ),
    )
    if batch_size is not None:
        batch_size = _require_nonnegative_int(batch_size, "batch_size")

    return RunReportMetadata(
        role=role,
        schema=raw.get("schema"),
        kind=raw.get("kind"),
        execution_status=(
            execution_status if isinstance(execution_status, str) else None
        ),
        model_family=model_family if isinstance(model_family, str) else None,
        model_fingerprint=identity.model_fingerprint,
        config_fingerprint=identity.config_fingerprint,
        input_fingerprint=identity.input_fingerprint,
        batch_size=batch_size,
    )


def _collect_locations(raw: dict[str, Any], paths):
    found = []
    for path in paths:
        value, present = _at(raw, path)
        if present:
            found.append((".".join(path), value))
    return found


def _single_metadata_value(raw: dict[str, Any], paths, label: str):
    found = _collect_locations(raw, paths)
    if not found:
        return None
    first = found[0][1]
    if any(value != first for _, value in found[1:]):
        raise MalformedRunReport(f"conflicting {label} locations")
    return first


def normalize_digest(value: Any) -> NormalizedDigest:
    if value is None:
        return NormalizedDigest(None, None, None, False)
    if isinstance(value, str):
        algorithm, separator, payload = value.partition(":")
        valid = bool(separator and algorithm and payload)
        return NormalizedDigest(
            algorithm if valid else None,
            value if valid else None,
            None,
            valid,
        )
    if isinstance(value, dict):
        algorithm = value.get("algorithm")
        payload = value.get("hex")
        size_bytes = value.get("size_bytes")
        if size_bytes is not None:
            size_bytes = _require_nonnegative_int(
                size_bytes, "selected_token_digest.size_bytes"
            )
        valid = (
            isinstance(algorithm, str)
            and bool(algorithm)
            and isinstance(payload, str)
            and bool(payload)
        )
        return NormalizedDigest(
            algorithm if valid else None,
            f"{algorithm}:{payload}" if valid else None,
            size_bytes,
            valid,
        )
    return NormalizedDigest(None, None, None, False)


def _fnv1a64_token_ids(token_ids: tuple[int, ...]) -> str:
    state = 14695981039346656037
    prime = 1099511628211
    mask = (1 << 64) - 1
    values = (len(token_ids),) + token_ids
    for value in values:
        if value > mask:
            raise MalformedRunReport("selected token ID exceeds uint64 range")
        for shift in range(0, 64, 8):
            state ^= (value >> shift) & 0xFF
            state = (state * prime) & mask
    return f"fnv1a64:{state:016x}"


def _validate_digest_for_array(
    digest: NormalizedDigest, token_ids: tuple[int, ...]
) -> NormalizedDigest:
    if not digest.valid:
        return digest
    expected: Optional[str] = None
    if digest.algorithm == "fnv1a64":
        expected = _fnv1a64_token_ids(token_ids)
    elif digest.algorithm == "sha256":
        expected = token_ids_sha256(token_ids)
    matches = expected == digest.value if expected is not None else None
    if matches is False:
        raise MalformedRunReport(
            "selected-token array contradicts selected-token digest"
        )
    if digest.size_bytes is not None and digest.size_bytes != len(token_ids) * 8:
        raise MalformedRunReport(
            "selected-token digest size contradicts selected-token array"
        )
    return NormalizedDigest(
        digest.algorithm,
        digest.value,
        digest.size_bytes,
        digest.valid,
        matches,
    )


def extract_selected_token_sequence(
    raw: dict[str, Any], role: str
) -> SelectedTokenSequence:
    array_locations = _collect_locations(raw, _SELECTED_ARRAY_PATHS)
    token_ids: Optional[tuple[int, ...]] = None
    source_locations: tuple[str, ...] = ()
    if array_locations:
        first = array_locations[0][1]
        if any(value != first for _, value in array_locations[1:]):
            raise MalformedRunReport(
                "conflicting explicit selected-token arrays"
            )
        if not isinstance(first, list):
            raise MalformedRunReport("selected_token_ids must be an array")
        normalized = []
        for index, token_id in enumerate(first):
            normalized.append(
                _require_nonnegative_int(
                    token_id, f"selected_token_ids[{index}]"
                )
            )
        token_ids = tuple(normalized)
        source_locations = tuple(location for location, _ in array_locations)

    count = _single_metadata_value(
        raw, _SELECTED_COUNT_PATHS, "selected-token count"
    )
    if count is not None:
        count = _require_nonnegative_int(count, "selected_token_count")
    digest_value = _single_metadata_value(
        raw, _SELECTED_DIGEST_PATHS, "selected-token digest"
    )
    digest = normalize_digest(digest_value)
    if digest_value is not None and not digest.valid:
        raise MalformedRunReport("selected_token_digest is malformed")

    report = raw.get("report") if isinstance(raw.get("report"), dict) else {}
    manifest = (
        raw.get("manifest") if isinstance(raw.get("manifest"), dict) else {}
    )
    stop_reason = report.get("stop_reason")
    generation_limit = _first(
        manifest,
        (
            ("runtime", "generation_limit"),
            ("runtime_settings", "generation_limit"),
            ("runtime_settings", "generate_limit"),
        ),
    )
    if generation_limit is not None:
        generation_limit = _require_nonnegative_int(
            generation_limit, "generation_limit"
        )

    if token_ids is not None:
        if count is not None and count != len(token_ids):
            raise MalformedRunReport(
                "selected_token_count does not match selected_token_ids"
            )
        digest = _validate_digest_for_array(digest, token_ids)
        level = SelectedTokenEvidenceLevel.ARRAY_EXACT
    elif count is not None and digest.valid:
        level = SelectedTokenEvidenceLevel.DIGEST_ONLY
    elif count is not None or digest.value is not None:
        level = SelectedTokenEvidenceLevel.METADATA_ONLY
    else:
        level = SelectedTokenEvidenceLevel.MISSING

    return SelectedTokenSequence(
        role=role,
        token_ids=token_ids,
        source_locations=source_locations,
        declared_count=count,
        digest=digest,
        stop_reason=stop_reason if isinstance(stop_reason, str) else None,
        generation_limit=generation_limit,
        evidence_level=level,
    )
