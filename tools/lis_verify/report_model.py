"""Immutable views over the frozen ``verification_report/v1`` value.

The M0 validator remains authoritative.  This module adds a typed, immutable
boundary without copying the schema or its status algebra into a second source
of truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any

from .product_contract import (
    CleanupStatus,
    CustomerVerdict,
    ExecutionPolicy,
    MAX_REPORT_BYTES,
    ResidueStatus,
    StageState,
    WorkflowClassification,
    canonical_json_bytes,
    validate_report,
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


@dataclass(frozen=True)
class StageResult:
    """Typed view of one canonical final stage result."""

    name: str
    state: StageState
    result_ref: str | None
    evidence_tier: str | None
    failure_class: str | None
    reason: str | None
    blocker: str | None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "StageResult":
        expected = {
            "name",
            "state",
            "result_ref",
            "evidence_tier",
            "failure_class",
            "reason",
            "blocker",
        }
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError("stage result must contain exactly the frozen fields")
        return cls(
            name=raw["name"],
            state=StageState(raw["state"]),
            result_ref=raw["result_ref"],
            evidence_tier=raw["evidence_tier"],
            failure_class=raw["failure_class"],
            reason=raw["reason"],
            blocker=raw["blocker"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "result_ref": self.result_ref,
            "evidence_tier": self.evidence_tier,
            "failure_class": self.failure_class,
            "reason": self.reason,
            "blocker": self.blocker,
        }


@dataclass(frozen=True)
class CleanupResult:
    status: CleanupStatus
    residue_status: ResidueStatus
    observed: bool
    retained_debug: bool


@dataclass(frozen=True)
class VerificationReport:
    """Canonical immutable report value.

    The bytes are the storage.  Callers receive newly decoded dictionaries, so
    mutation cannot alter the validated value held by this object.
    """

    _canonical: bytes = field(repr=False)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "VerificationReport":
        validate_report(raw)
        encoded = canonical_json_bytes(raw)
        if len(encoded) > MAX_REPORT_BYTES:
            raise ValueError("canonical report exceeds its total byte bound")
        # Decode once more so bool/int and duplicate-free JSON semantics are
        # exactly those that a persisted consumer observes.
        decoded = json.loads(encoded, object_pairs_hook=_reject_duplicate_keys)
        validate_report(decoded)
        return cls(encoded)

    @classmethod
    def from_json_bytes(cls, encoded: bytes) -> "VerificationReport":
        if not isinstance(encoded, bytes):
            raise TypeError("report input must be bytes")
        if len(encoded) > MAX_REPORT_BYTES:
            raise ValueError("report input exceeds its total byte bound")
        try:
            text = encoded.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("report input is not UTF-8") from exc
        try:
            raw = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as exc:
            raise ValueError("report input is not valid JSON") from exc
        if not isinstance(raw, dict):
            raise ValueError("report input must be a JSON object")
        report = cls.from_dict(raw)
        if encoded != report.to_json_bytes():
            raise ValueError("report input is not canonical JSON")
        return report

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self._canonical)

    def to_json_bytes(self) -> bytes:
        return self._canonical

    @property
    def attempt_id(self) -> str:
        return self.to_dict()["attempt"]["id"]

    @property
    def workflow_classification(self) -> WorkflowClassification:
        return WorkflowClassification(
            self.to_dict()["attempt"]["workflow_classification"]
        )

    @property
    def verdict(self) -> CustomerVerdict:
        return CustomerVerdict(self.to_dict()["verdict"])

    @property
    def policy(self) -> ExecutionPolicy:
        return ExecutionPolicy(self.to_dict()["policy_result"]["policy"])

    @property
    def exit_code(self) -> int:
        return self.to_dict()["policy_result"]["exit_code"]

    @property
    def stages(self) -> tuple[StageResult, ...]:
        return tuple(StageResult.from_dict(item) for item in self.to_dict()["stages"])

    @property
    def cleanup(self) -> CleanupResult:
        raw = self.to_dict()["cleanup"]
        return CleanupResult(
            status=CleanupStatus(raw["status"]),
            residue_status=ResidueStatus(raw["residue_status"]),
            observed=raw["observed"],
            retained_debug=raw["retained_debug"],
        )
