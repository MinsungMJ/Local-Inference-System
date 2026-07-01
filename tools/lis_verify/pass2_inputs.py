"""Source-bound, model-free input handling for Pass 2.

Only immutable canonical report hashes are inspected before original-report
binding succeeds.  Report materialization and all execution metadata access
happen after that boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .pass1_inputs import (
    CanonicalRunReport,
    RunReportInputError,
    extract_selected_token_sequence,
)
from .pass1_model import Pass1Result, RunReportIdentity


class Pass2InputError(ValueError):
    """A required source-bound run-report field is absent or malformed."""


def _fingerprint(value: Any) -> Optional[str]:
    if isinstance(value, str) and value:
        return value
    if isinstance(value, dict):
        algorithm = value.get("algorithm")
        payload = value.get("hex")
        if (
            isinstance(algorithm, str)
            and algorithm
            and isinstance(payload, str)
            and payload
        ):
            return f"{algorithm}:{payload}"
    return None


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Pass2InputError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Pass2InputError(f"{label} must be a non-negative integer")
    return value


def report_sha256(report) -> Optional[str]:
    """Read only the precomputed immutable report hash."""
    try:
        value = report.identity.run_report_sha256
    except AttributeError:
        return None
    return value if isinstance(value, str) and value else None


def original_binding_matches(
    pass1: Pass1Result,
    reference_original,
    candidate_original,
) -> tuple[bool, bool]:
    """Check both original hashes without materializing either report."""
    if not pass1.source_binding_verified:
        return False, False
    reference_matches = (
        report_sha256(reference_original)
        == pass1.source_binding.reference.run_report_sha256
    )
    candidate_matches = (
        report_sha256(candidate_original)
        == pass1.source_binding.candidate.run_report_sha256
    )
    return reference_matches, candidate_matches


@dataclass(frozen=True)
class RunEvidence:
    role: str
    identity: RunReportIdentity
    binary_fingerprint: str
    runtime_fingerprint: Optional[str]
    configured_context: Optional[int]
    prompt_token_count: int
    batch_size: int
    thread_count: int
    raw: dict[str, Any] = field(repr=False, compare=False)


def extract_run_evidence(
    report: CanonicalRunReport, role: str
) -> RunEvidence:
    """Materialize and validate one report after its binding gate."""
    raw = report.materialize()
    if not isinstance(raw, dict):
        raise Pass2InputError(f"{role}: run report root must be an object")
    if raw.get("schema") != "lis.execution_artifact/v1":
        raise Pass2InputError(f"{role}: unsupported run-report schema")
    if raw.get("kind") != "run_report":
        raise Pass2InputError(f"{role}: unsupported run-report kind")

    manifest = raw.get("manifest")
    body = raw.get("report")
    if not isinstance(manifest, dict) or not isinstance(body, dict):
        raise Pass2InputError(
            f"{role}: run report requires manifest and report objects"
        )
    if str(body.get("execution_status", "")).lower() != "ok":
        raise Pass2InputError(f"{role}: execution_status must be ok")

    binary = manifest.get("binary")
    runtime = manifest.get("runtime")
    if not isinstance(binary, dict) or not isinstance(runtime, dict):
        raise Pass2InputError(
            f"{role}: binary and runtime manifests are required"
        )
    binary_fingerprint = _fingerprint(binary.get("fingerprint"))
    if binary_fingerprint is None:
        raise Pass2InputError(
            f"{role}: manifest.binary.fingerprint is required"
        )

    prompt_sequences = body.get("prompt_sequences")
    if (
        not isinstance(prompt_sequences, list)
        or not prompt_sequences
        or not isinstance(prompt_sequences[0], dict)
    ):
        raise Pass2InputError(
            f"{role}: report.prompt_sequences[0] is required"
        )
    prompt_token_count = _nonnegative_int(
        prompt_sequences[0].get("token_count"),
        f"{role}: prompt token count",
    )
    batch_size = _positive_int(
        runtime.get("batch_size"), f"{role}: batch size"
    )
    thread_count = _positive_int(
        runtime.get("thread_count"), f"{role}: thread count"
    )

    configured_context = runtime.get("configured_context")
    if configured_context is not None:
        configured_context = _positive_int(
            configured_context, f"{role}: configured context"
        )
    runtime_fingerprint = _fingerprint(runtime.get("fingerprint"))

    identity = report.identity
    required_identities = (
        identity.model_fingerprint,
        identity.config_fingerprint,
        identity.input_fingerprint,
    )
    if any(value is None for value in required_identities):
        raise Pass2InputError(
            f"{role}: model/config/input fingerprints are required"
        )

    return RunEvidence(
        role=role,
        identity=identity,
        binary_fingerprint=binary_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        configured_context=configured_context,
        prompt_token_count=prompt_token_count,
        batch_size=batch_size,
        thread_count=thread_count,
        raw=raw,
    )


def reproduction_identity_matches(
    reproduction: CanonicalRunReport, original: RunEvidence
) -> bool:
    """Compare identity fields without materializing the reproduction."""
    try:
        identity = reproduction.identity
    except AttributeError:
        return False
    return bool(
        identity.model_fingerprint == original.identity.model_fingerprint
        and identity.config_fingerprint
        == original.identity.config_fingerprint
        and identity.input_fingerprint == original.identity.input_fingerprint
    )


def selected_tokens(evidence: RunEvidence) -> Optional[tuple[int, ...]]:
    """Extract exact selected tokens from already-bound report material."""
    try:
        sequence = extract_selected_token_sequence(
            evidence.raw, evidence.role
        )
    except RunReportInputError as exc:
        raise Pass2InputError(f"{evidence.role}: {exc}") from exc
    return sequence.token_ids
