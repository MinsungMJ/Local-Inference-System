"""Strict loading for the packaged, model-free seeded demonstration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import resources
import json
import re
from typing import Any, Callable

from .product_contract import canonical_json_bytes


MAX_DEMO_RESOURCE_BYTES = 512 * 1024
MAX_DEMO_BUNDLE_BYTES = 1024 * 1024
RESOURCE_FILES = {
    "profile": "profile_v1.json",
    "reference_original": "reference_original.json",
    "candidate_original": "candidate_original.json",
    "reference_reproduction": "reference_reproduction.json",
    "candidate_reproduction": "candidate_reproduction.json",
    "intra_layer_trace": "intra_layer_trace.json",
}
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")


class DemoFixtureError(ValueError):
    """The installed demo evidence is missing, malformed, or unbound."""


ResourceReader = Callable[[str], bytes]


def _reject_constant(value: str) -> None:
    raise DemoFixtureError(f"non-standard JSON constant: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DemoFixtureError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_object(data: bytes, label: str) -> dict[str, Any]:
    if not isinstance(data, bytes):
        raise DemoFixtureError(f"{label} resource reader did not return bytes")
    if not data or len(data) > MAX_DEMO_RESOURCE_BYTES:
        raise DemoFixtureError(f"{label} resource is empty or oversized")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise DemoFixtureError(f"{label} resource is not UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except DemoFixtureError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise DemoFixtureError(f"{label} resource is not valid JSON") from exc
    if not isinstance(value, dict):
        raise DemoFixtureError(f"{label} resource must be a JSON object")
    try:
        canonical_json_bytes(value)
    except (TypeError, ValueError) as exc:
        raise DemoFixtureError(f"{label} resource is not canonicalizable") from exc
    return value


def _default_reader(name: str) -> bytes:
    target = resources.files("lis_verify.demo_data").joinpath(name)
    try:
        with target.open("rb") as stream:
            data = stream.read(MAX_DEMO_RESOURCE_BYTES + 1)
    except (FileNotFoundError, OSError) as exc:
        raise DemoFixtureError("packaged demo resource is unavailable") from exc
    if len(data) > MAX_DEMO_RESOURCE_BYTES:
        raise DemoFixtureError("packaged demo resource exceeds its byte bound")
    return data


def _exact_dict(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise DemoFixtureError(f"{label} has missing or unknown fields")
    return value


def _validate_profile(profile: dict[str, Any]) -> None:
    _exact_dict(
        profile,
        {
            "schema",
            "seed",
            "comparison_mode",
            "build_calibration",
            "prompt_token_ids",
            "runtime_checkpoint_step",
            "target_layer",
            "layer_indices",
            "total_layer_count",
            "candidate_layer_digest",
            "candidate_intra_layer_mismatch_stage_order",
            "generations",
            "expected",
        },
        "demo profile",
    )
    if profile["schema"] != "lis.demo_profile/v1":
        raise DemoFixtureError("demo profile schema is unsupported")
    if profile["seed"] != 240517 or profile["comparison_mode"] != "backend_differential":
        raise DemoFixtureError("demo seed or comparison mode is unsupported")
    integer_fields = (
        "runtime_checkpoint_step",
        "target_layer",
        "total_layer_count",
        "candidate_intra_layer_mismatch_stage_order",
    )
    if any(
        isinstance(profile[field], bool)
        or not isinstance(profile[field], int)
        or profile[field] < 0
        for field in integer_fields
    ):
        raise DemoFixtureError("demo profile integer field is invalid")
    if profile["prompt_token_ids"] != [1, 2, 3]:
        raise DemoFixtureError("demo prompt identity is not frozen")
    calibration = _exact_dict(
        profile["build_calibration"],
        {
            "build_id",
            "repetition_penalty",
            "repetition_penalty_enabled",
            "structural_token_suppression",
            "rms_norm_eps_runtime_bound",
            "kv_write_round_to_nearest_even",
            "fma_contraction_backend_defined",
            "reduction_order_backend_defined",
        },
        "build calibration",
    )
    if calibration != {
        "build_id": "lis_demo_seeded_replay_v1",
        "repetition_penalty": 1.2,
        "repetition_penalty_enabled": True,
        "structural_token_suppression": True,
        "rms_norm_eps_runtime_bound": {
            "llama3_decoder": False,
            "qwen3_dense_decoder": True,
        },
        "kv_write_round_to_nearest_even": False,
        "fma_contraction_backend_defined": True,
        "reduction_order_backend_defined": True,
    }:
        raise DemoFixtureError("demo build calibration is unsupported")
    if profile["layer_indices"] != [0, 4, 8, 12]:
        raise DemoFixtureError("demo layer coverage is not frozen")
    if (
        profile["runtime_checkpoint_step"] != 18
        or profile["target_layer"] != 8
        or profile["total_layer_count"] != 13
        or profile["candidate_intra_layer_mismatch_stage_order"] != 7
    ):
        raise DemoFixtureError("demo checkpoint profile is not frozen")
    if (
        not isinstance(profile["candidate_layer_digest"], str)
        or _SHA256.fullmatch(profile["candidate_layer_digest"]) is None
    ):
        raise DemoFixtureError("demo candidate layer digest is invalid")
    generations = _exact_dict(
        profile["generations"], {"discovery", "authoritative"}, "generations"
    )
    identifiers = {
        "reference_original_artifact_set_id",
        "candidate_original_artifact_set_id",
        "reference_reproduction_artifact_set_id",
        "candidate_reproduction_artifact_set_id",
    }
    seen: set[str] = set()
    for name in ("discovery", "authoritative"):
        generation = _exact_dict(generations[name], identifiers, name)
        for value in generation.values():
            if (
                not isinstance(value, str)
                or re.fullmatch(r"aset1:[0-9a-f]{32}", value) is None
                or value in seen
            ):
                raise DemoFixtureError("artifact-set identities are invalid or reused")
            seen.add(value)
    expected = _exact_dict(
        profile["expected"],
        {
            "pass0_status",
            "pass1_status",
            "pass2_status",
            "pass3_status",
            "pass4_status",
            "generated_token_step",
            "layer_interval",
            "intra_layer_interval",
        },
        "expected result",
    )
    if expected != {
        "pass0_status": "limited_comparison_allowed",
        "pass1_status": "first_mismatch_found",
        "pass2_status": "reproduction_verified",
        "pass3_status": "observable_mismatch_found",
        "pass4_status": "observable_intra_layer_mismatch_found",
        "generated_token_step": 17,
        "layer_interval": "(4, 8]",
        "intra_layer_interval": "(rope_key_output, attention_scores]",
    }:
        raise DemoFixtureError("expected demo result is not frozen")


@dataclass(frozen=True)
class DemoFixture:
    fixture_id: str
    fixture_version: int
    manifest_sha256: str
    _canonical_values: tuple[tuple[str, bytes], ...]

    def value(self, name: str) -> dict[str, Any]:
        for candidate, encoded in self._canonical_values:
            if candidate == name:
                return json.loads(encoded)
        raise DemoFixtureError("unknown demo fixture component")


def load_demo_fixture(reader: ResourceReader | None = None) -> DemoFixture:
    """Load and byte-bind every resource before exposing any parsed value."""

    read_resource = _default_reader if reader is None else reader

    def read(name: str) -> bytes:
        try:
            return read_resource(name)
        except DemoFixtureError:
            raise
        except (KeyError, OSError) as exc:
            raise DemoFixtureError("packaged demo resource is unavailable") from exc

    manifest_bytes = read("manifest_v1.json")
    manifest = _parse_object(manifest_bytes, "manifest")
    _exact_dict(
        manifest,
        {"schema", "fixture_id", "fixture_version", "resources"},
        "demo manifest",
    )
    if manifest["schema"] != "lis.demo_fixture_manifest/v1":
        raise DemoFixtureError("demo manifest schema is unsupported")
    if (
        manifest["fixture_id"] != "lis-seeded-mismatch-240517"
        or manifest["fixture_version"] != 1
    ):
        raise DemoFixtureError("demo fixture identity is unsupported")
    entries = _exact_dict(
        manifest["resources"], set(RESOURCE_FILES), "demo resources"
    )
    total = len(manifest_bytes)
    canonical_values: list[tuple[str, bytes]] = []
    for name, expected_file in RESOURCE_FILES.items():
        entry = _exact_dict(entries[name], {"file", "sha256"}, name)
        if entry["file"] != expected_file or _SHA256.fullmatch(entry["sha256"]) is None:
            raise DemoFixtureError("demo resource name or digest is invalid")
        data = read(expected_file)
        if not isinstance(data, bytes):
            raise DemoFixtureError("demo resource reader did not return bytes")
        total += len(data)
        if total > MAX_DEMO_BUNDLE_BYTES:
            raise DemoFixtureError("demo fixture bundle exceeds its byte bound")
        actual = "sha256:" + hashlib.sha256(data).hexdigest()
        if actual != entry["sha256"]:
            raise DemoFixtureError(f"demo resource digest mismatch: {name}")
        parsed = _parse_object(data, name)
        canonical_values.append((name, canonical_json_bytes(parsed)))
    profile = json.loads(dict(canonical_values)["profile"])
    _validate_profile(profile)
    return DemoFixture(
        fixture_id=manifest["fixture_id"],
        fixture_version=manifest["fixture_version"],
        manifest_sha256="sha256:" + hashlib.sha256(manifest_bytes).hexdigest(),
        _canonical_values=tuple(canonical_values),
    )
