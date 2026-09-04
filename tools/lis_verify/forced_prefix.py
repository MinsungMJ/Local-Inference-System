"""Source binding for the M0 forced-prefix run-report channel."""

from __future__ import annotations

from typing import Any

from .pass1_artifact import serialize as serialize_pass1
from .pass1_inputs import CanonicalRunReport, canonical_json, sha256_text
from .pass1_model import Pass1Result, Pass1Status, PrefixAvailability
from .pass2_inputs import (
    Pass2InputError,
    extract_run_evidence,
    reproduction_identity_matches,
    selected_tokens,
)
from .product_contract import (
    canonical_json_bytes,
    selection_policy_sha256,
    validate_forced_prefix_metadata,
)


class ForcedPrefixBindingError(ValueError):
    """Forced-prefix evidence does not match its source chain."""


def _pass1_sha256(pass1: Pass1Result) -> str:
    return sha256_text(canonical_json(serialize_pass1(pass1)))


def _backend(raw: dict[str, Any]) -> tuple[str | None, Any]:
    manifest = raw.get("manifest") if isinstance(raw, dict) else None
    backend = manifest.get("backend") if isinstance(manifest, dict) else None
    if not isinstance(backend, dict):
        return None, None
    return backend.get("name"), backend.get("fingerprint")


def _artifact_set_id(raw: dict[str, Any]) -> str | None:
    value = raw.get("artifact_set_id") if isinstance(raw, dict) else None
    return value if isinstance(value, str) and value else None


def _contains_prohibited_key(value: Any, prohibited: set[str]) -> bool:
    if isinstance(value, dict):
        return any(
            key in prohibited
            or _contains_prohibited_key(child, prohibited)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(
            _contains_prohibited_key(child, prohibited) for child in value
        )
    return False


def build_forced_prefix_metadata(
    pass1: Pass1Result,
    original: CanonicalRunReport,
    *,
    role: str,
    selection_policy: str = "lis_policy_modified_greedy_v1",
) -> dict[str, Any]:
    if role not in {"reference", "candidate"}:
        raise ForcedPrefixBindingError("forced-prefix role is unsupported")
    if (
        not isinstance(pass1, Pass1Result)
        or pass1.status != Pass1Status.FIRST_MISMATCH_FOUND
    ):
        raise ForcedPrefixBindingError(
            "forced-prefix binding requires a Pass 1 mismatch"
        )
    prefix = pass1.prefix_for_reproduction
    localization = pass1.localization
    if (
        prefix is None
        or localization is None
        or prefix.availability != PrefixAvailability.EMBEDDED
        or not prefix.exact_token_ids
        or prefix.token_count != len(prefix.exact_token_ids)
        or prefix.token_count > 64
    ):
        raise ForcedPrefixBindingError(
            "a bounded exact non-empty prefix is required"
        )
    expected_identity = (
        pass1.source_binding.reference
        if role == "reference"
        else pass1.source_binding.candidate
    )
    if original.identity != expected_identity:
        raise ForcedPrefixBindingError("original report is bound to the wrong role")
    try:
        run = extract_run_evidence(original, f"{role}_original")
    except Pass2InputError as exc:
        raise ForcedPrefixBindingError(str(exc)) from exc
    pass1_sha256 = _pass1_sha256(pass1)
    value = {
        "mode": "injected_selected_token_prefix_v1",
        "applied": True,
        "token_count": prefix.token_count,
        "token_ids_sha256": prefix.sha256,
        "prefix_start_generated_step": 0,
        "prefix_end_generated_step_exclusive": prefix.token_count,
        "target_generated_token_step": localization.generated_token_step,
        "runtime_checkpoint_step": localization.runtime_checkpoint_step,
        "prompt_token_count": run.prompt_token_count,
        "context_position": run.prompt_token_count + prefix.token_count,
        "selection_policy": selection_policy,
        "selection_policy_sha256": selection_policy_sha256(selection_policy),
        "source_pass0_artifact_sha256": pass1.calibration_ref.sha256,
        "source_original_run_report_sha256": original.identity.run_report_sha256,
        "source_pass1_artifact_sha256": pass1_sha256,
        "source_localization_ref_sha256": pass1_sha256,
    }
    validate_forced_prefix_metadata(value)
    return value


def forced_prefix_binding_bytes(
    pass1: Pass1Result,
    original: CanonicalRunReport,
    *,
    role: str,
    selection_policy: str = "lis_policy_modified_greedy_v1",
) -> bytes:
    return canonical_json_bytes(
        build_forced_prefix_metadata(
            pass1,
            original,
            role=role,
            selection_policy=selection_policy,
        )
    )


def validate_forced_prefix_reproduction(
    pass1: Pass1Result,
    original: CanonicalRunReport,
    reproduction: CanonicalRunReport,
    *,
    role: str,
    selection_policy: str = "lis_policy_modified_greedy_v1",
) -> None:
    expected = build_forced_prefix_metadata(
        pass1,
        original,
        role=role,
        selection_policy=selection_policy,
    )
    raw = reproduction.materialize()
    forced = raw.get("forced_prefix")
    try:
        validate_forced_prefix_metadata(forced)
    except (TypeError, ValueError) as exc:
        raise ForcedPrefixBindingError(
            "forced-prefix metadata is malformed"
        ) from exc
    if forced != expected:
        raise ForcedPrefixBindingError(
            "forced-prefix metadata is stale or source-inconsistent"
        )

    try:
        original_run = extract_run_evidence(original, f"{role}_original")
        reproduction_run = extract_run_evidence(
            reproduction, f"{role}_reproduction"
        )
    except Pass2InputError as exc:
        raise ForcedPrefixBindingError(str(exc)) from exc
    if not reproduction_identity_matches(reproduction, original_run):
        raise ForcedPrefixBindingError("model, config, or input identity changed")
    if reproduction_run.binary_fingerprint != original_run.binary_fingerprint:
        raise ForcedPrefixBindingError(
            "binary identity changed across reproduction"
        )
    if _backend(original_run.raw) != _backend(reproduction_run.raw):
        raise ForcedPrefixBindingError(
            "backend identity changed across reproduction"
        )
    if (
        reproduction_run.prompt_token_count != original_run.prompt_token_count
        or reproduction_run.batch_size != original_run.batch_size
        or reproduction_run.thread_count != original_run.thread_count
        or reproduction_run.configured_context != original_run.configured_context
    ):
        raise ForcedPrefixBindingError(
            "runtime context changed across reproduction"
        )
    original_set = _artifact_set_id(original_run.raw)
    reproduction_set = _artifact_set_id(reproduction_run.raw)
    if (
        original_set is not None
        and reproduction_set is not None
        and original_set == reproduction_set
    ):
        raise ForcedPrefixBindingError(
            "reproduction reused the original artifact set"
        )

    try:
        original_selected = selected_tokens(original_run)
        observed = selected_tokens(reproduction_run)
    except Pass2InputError as exc:
        raise ForcedPrefixBindingError(str(exc)) from exc
    if observed is None or len(observed) != 1:
        raise ForcedPrefixBindingError(
            "forced-prefix report must retain only the target selected token"
        )
    target_step = expected["target_generated_token_step"]
    if (
        original_selected is None
        or target_step >= len(original_selected)
        or observed[0] != original_selected[target_step]
    ):
        raise ForcedPrefixBindingError(
            "forced-prefix target selection did not reproduce the original role"
        )
    prohibited = {
        "full_forced_prefix_token_ids",
        "raw_prompt_text",
        "raw_generated_text",
    }
    if _contains_prohibited_key(raw, prohibited):
        raise ForcedPrefixBindingError(
            "forced-prefix report retained prohibited data"
        )


def validate_paired_forced_prefix_reproductions(
    pass1: Pass1Result,
    reference_original: CanonicalRunReport,
    candidate_original: CanonicalRunReport,
    reference_reproduction: CanonicalRunReport,
    candidate_reproduction: CanonicalRunReport,
) -> None:
    validate_forced_prefix_reproduction(
        pass1,
        reference_original,
        reference_reproduction,
        role="reference",
    )
    validate_forced_prefix_reproduction(
        pass1,
        candidate_original,
        candidate_reproduction,
        role="candidate",
    )
