"""Strict packaged execution profile and local model preflight for M3."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import resources
import json
import os
from pathlib import Path
import stat
from typing import Any, Callable

from .product_contract import canonical_json_bytes, selection_policy_sha256
from .provenance import ProvenanceError, hash_regular_file


SCHEMA = "lis.supported_model_profile/v1"
KIND = "supported_model_profile"
PROFILE_RESOURCE = "plain_rope_llama_v1.json"
MAX_PROFILE_BYTES = 64 * 1024
MAX_CONFIG_BYTES = 1024 * 1024
MAX_DIRECT_TOKENS = 64


class ModelProfileError(ValueError):
    """A packaged profile or customer model failed strict preflight."""


ProfileReader = Callable[[str], bytes]


def _reject_constant(value: str) -> None:
    raise ModelProfileError(f"non-standard JSON constant: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ModelProfileError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _parse_object(data: bytes, label: str, maximum: int) -> dict[str, Any]:
    if not isinstance(data, bytes) or not data or len(data) > maximum:
        raise ModelProfileError(f"{label} is empty or oversized")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ModelProfileError(f"{label} is not UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except ModelProfileError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ModelProfileError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ModelProfileError(f"{label} must be a JSON object")
    return value


def _exact_dict(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ModelProfileError(f"{label} has missing or unknown fields")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelProfileError(f"{label} must be a positive integer")
    return value


def _default_reader(name: str) -> bytes:
    target = resources.files("lis_verify.model_profiles").joinpath(name)
    try:
        with target.open("rb") as stream:
            return stream.read(MAX_PROFILE_BYTES + 1)
    except (FileNotFoundError, OSError) as exc:
        raise ModelProfileError("packaged model profile is unavailable") from exc


def validate_model_profile(raw: Any) -> None:
    value = _exact_dict(
        raw,
        {
            "schema",
            "kind",
            "profile_id",
            "model_family",
            "model_format",
            "input_mode",
            "direct_token_ids",
            "context_length",
            "batch_size",
            "generation_limit",
            "thread_count",
            "selection_policy",
            "selection_policy_sha256",
            "config_constraints",
        },
        "model profile",
    )
    if value["schema"] != SCHEMA or value["kind"] != KIND:
        raise ModelProfileError("model profile identity is unsupported")
    if value["profile_id"] != "plain_rope_llama_direct_token_v1":
        raise ModelProfileError("model profile ID is unsupported")
    if value["model_family"] != "llama3_decoder":
        raise ModelProfileError("model family is unsupported")
    if value["model_format"] != "huggingface_local":
        raise ModelProfileError("model format is unsupported")
    if value["input_mode"] != "direct_token_ids":
        raise ModelProfileError("model profile input mode is unsupported")
    tokens = value["direct_token_ids"]
    if (
        not isinstance(tokens, list)
        or not 0 < len(tokens) <= MAX_DIRECT_TOKENS
        or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens)
    ):
        raise ModelProfileError("direct token IDs are invalid or out of bounds")
    if tokens != [1]:
        raise ModelProfileError("direct token identity is not frozen")
    for field, expected in (
        ("context_length", 128),
        ("batch_size", 1),
        ("generation_limit", 8),
        ("thread_count", 1),
    ):
        if _positive_int(value[field], field) != expected:
            raise ModelProfileError(f"{field} disagrees with the frozen profile")
    if value["selection_policy"] != "lis_policy_modified_greedy_v1":
        raise ModelProfileError("selection policy is unsupported")
    if value["selection_policy_sha256"] != selection_policy_sha256(
        value["selection_policy"]
    ):
        raise ModelProfileError("selection policy identity is inconsistent")
    constraints = _exact_dict(
        value["config_constraints"],
        {
            "allowed_architectures",
            "allowed_dtypes",
            "allowed_model_types",
            "minimum_max_position_embeddings",
            "plain_rope_only",
            "requires_merged_safetensors",
        },
        "config constraints",
    )
    if constraints != {
        "allowed_architectures": ["LlamaForCausalLM"],
        "allowed_dtypes": ["float32", "float16", "bfloat16"],
        "allowed_model_types": ["llama"],
        "minimum_max_position_embeddings": 128,
        "plain_rope_only": True,
        "requires_merged_safetensors": True,
    }:
        raise ModelProfileError("config constraints are not frozen")


@dataclass(frozen=True)
class ModelExecutionProfile:
    profile_id: str
    identity_sha256: str
    direct_token_ids: tuple[int, ...]
    context_length: int
    batch_size: int
    generation_limit: int
    thread_count: int
    selection_policy: str
    selection_policy_sha256: str
    raw: dict[str, Any]


def load_model_profile(reader: ProfileReader | None = None) -> ModelExecutionProfile:
    read = _default_reader if reader is None else reader
    try:
        data = read(PROFILE_RESOURCE)
    except ModelProfileError:
        raise
    except (KeyError, OSError) as exc:
        raise ModelProfileError("packaged model profile is unavailable") from exc
    raw = _parse_object(data, "model profile", MAX_PROFILE_BYTES)
    validate_model_profile(raw)
    canonical = canonical_json_bytes(raw)
    if data != canonical:
        raise ModelProfileError("packaged model profile is not canonical JSON")
    return ModelExecutionProfile(
        profile_id=raw["profile_id"],
        identity_sha256="sha256:" + hashlib.sha256(canonical).hexdigest(),
        direct_token_ids=tuple(raw["direct_token_ids"]),
        context_length=raw["context_length"],
        batch_size=raw["batch_size"],
        generation_limit=raw["generation_limit"],
        thread_count=raw["thread_count"],
        selection_policy=raw["selection_policy"],
        selection_policy_sha256=raw["selection_policy_sha256"],
        raw=raw,
    )


def _validate_no_symlink_components(path: Path) -> Path:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise ModelProfileError("model path is missing or inaccessible") from exc
        if stat.S_ISLNK(info.st_mode):
            raise ModelProfileError("symlink model path components are prohibited")
    return absolute


def _read_regular(path: Path, maximum: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ModelProfileError(f"cannot open model input: {path.name}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ModelProfileError(f"model input is not regular: {path.name}")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise ModelProfileError(f"model input owner is invalid: {path.name}")
        data = bytearray()
        while len(data) <= maximum:
            chunk = os.read(fd, min(65_536, maximum + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(fd)
        if (
            info.st_dev != after.st_dev
            or info.st_ino != after.st_ino
            or info.st_size != after.st_size
            or info.st_mtime_ns != after.st_mtime_ns
        ):
            raise ModelProfileError(f"model input changed while reading: {path.name}")
        if len(data) > maximum:
            raise ModelProfileError(f"model input exceeds its byte bound: {path.name}")
        return bytes(data)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class ResolvedModel:
    directory: Path
    model_path: Path
    config_path: Path
    model_sha256: str
    config_sha256: str
    layer_count: int
    vocab_size: int
    config: dict[str, Any]


def resolve_model(model: Path, profile: ModelExecutionProfile) -> ResolvedModel:
    directory = _validate_no_symlink_components(Path(model))
    try:
        info = directory.lstat()
    except OSError as exc:
        raise ModelProfileError("model directory is unavailable") from exc
    if not stat.S_ISDIR(info.st_mode):
        raise ModelProfileError("model input must be a directory")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise ModelProfileError("model directory owner is invalid")
    config_path = directory / "config.json"
    model_path = directory / "model.safetensors"
    config_data = _read_regular(config_path, MAX_CONFIG_BYTES)
    config = _parse_object(config_data, "model config", MAX_CONFIG_BYTES)
    constraints = profile.raw["config_constraints"]
    if config.get("model_type") not in constraints["allowed_model_types"]:
        raise ModelProfileError("model type is unsupported")
    architectures = config.get("architectures")
    if architectures is not None and (
        not isinstance(architectures, list)
        or len(architectures) != 1
        or architectures[0] not in constraints["allowed_architectures"]
    ):
        raise ModelProfileError("model architecture is unsupported")
    if "rope_scaling" in config and config["rope_scaling"] is not None:
        raise ModelProfileError("scaled RoPE is unsupported")
    if config.get("rope_type", "default") != "default":
        raise ModelProfileError("non-default RoPE is unsupported")
    if config.get("torch_dtype") not in constraints["allowed_dtypes"]:
        raise ModelProfileError("model dtype is unsupported")
    maximum = _positive_int(config.get("max_position_embeddings"), "max_position_embeddings")
    if maximum < profile.context_length:
        raise ModelProfileError("model context is smaller than the execution profile")
    layer_count = _positive_int(config.get("num_hidden_layers"), "num_hidden_layers")
    vocab_size = _positive_int(config.get("vocab_size"), "vocab_size")
    if any(token >= vocab_size for token in profile.direct_token_ids):
        raise ModelProfileError("direct token ID exceeds the model vocabulary")
    try:
        model_sha256, _ = hash_regular_file(model_path)
        config_sha256, _ = hash_regular_file(config_path)
    except ProvenanceError as exc:
        raise ModelProfileError(str(exc)) from exc
    return ResolvedModel(
        directory=directory,
        model_path=model_path,
        config_path=config_path,
        model_sha256=model_sha256,
        config_sha256=config_sha256,
        layer_count=layer_count,
        vocab_size=vocab_size,
        config=config,
    )
