"""Strict public-model golden manifest and local-material verification."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from importlib import resources
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
from typing import Any, Callable, Sequence
from urllib.parse import quote

from .model_profile import load_model_profile, resolve_model
from .product_contract import (
    MAX_STAGE_TIMEOUT_SECONDS,
    MAX_TEMP_DISK_BYTES,
    SHA256_RE,
    canonical_json_bytes,
)
from .provenance import hash_regular_file


SCHEMA = "lis.public_model_golden/v1"
KIND = "public_model_golden_manifest"
DEFAULT_RESOURCE = "smollm2_135m_v1.json"
MAX_MANIFEST_BYTES = 64 * 1024
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
VERSION_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z.+-]{0,63}$")


class GoldenManifestError(ValueError):
    """The golden manifest or local model material is not trustworthy."""


ManifestReader = Callable[[str], bytes]


def _reject_constant(value: str) -> None:
    raise GoldenManifestError(f"non-standard JSON constant: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise GoldenManifestError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _parse_object(data: bytes, label: str) -> dict[str, Any]:
    if not data or len(data) > MAX_MANIFEST_BYTES:
        raise GoldenManifestError(f"{label} is empty or oversized")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GoldenManifestError(f"{label} is not UTF-8") from exc
    try:
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except GoldenManifestError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise GoldenManifestError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise GoldenManifestError(f"{label} must be a JSON object")
    return value


def _exact_dict(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise GoldenManifestError(f"{label} has missing or unknown fields")
    return value


def _bounded_string(value: Any, label: str, maximum: int = 512) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > maximum
    ):
        raise GoldenManifestError(f"{label} is empty or exceeds its byte bound")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GoldenManifestError(f"{label} must be a positive integer")
    return value


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise GoldenManifestError(f"{label} is not a canonical SHA-256 identity")
    return value


def _default_reader(name: str) -> bytes:
    target = resources.files("lis_verify.golden_models").joinpath(name)
    try:
        with target.open("rb") as stream:
            return stream.read(MAX_MANIFEST_BYTES + 1)
    except (FileNotFoundError, OSError) as exc:
        raise GoldenManifestError("packaged golden manifest is unavailable") from exc


def _input_identity(profile_id: str, token_ids: list[int]) -> str:
    value = {
        "domain": "lis.verify.direct_token_input/v1",
        "profile_id": profile_id,
        "token_ids": token_ids,
    }
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_manifest(raw: Any) -> None:
    value = _exact_dict(
        raw,
        {
            "baseline_update",
            "config_constraints",
            "expected",
            "files",
            "input",
            "kind",
            "manifest_id",
            "runtime",
            "schema",
            "upstream",
            "validated_lis",
        },
        "golden manifest",
    )
    if value["schema"] != SCHEMA or value["kind"] != KIND:
        raise GoldenManifestError("golden manifest identity is unsupported")
    if value["manifest_id"] != "smollm2_135m_backend_v1":
        raise GoldenManifestError("golden manifest ID is unsupported")

    upstream = _exact_dict(
        value["upstream"],
        {"license", "license_url", "repository", "repository_url", "revision"},
        "upstream",
    )
    if upstream["repository"] != "HuggingFaceTB/SmolLM2-135M":
        raise GoldenManifestError("golden repository is unsupported")
    if upstream["repository_url"] != (
        "https://huggingface.co/" + upstream["repository"]
    ):
        raise GoldenManifestError("golden repository URL is inconsistent")
    if upstream["license"] != "Apache-2.0" or upstream["license_url"] != (
        "https://www.apache.org/licenses/LICENSE-2.0"
    ):
        raise GoldenManifestError("golden license identity is unsupported")
    revision = upstream["revision"]
    if not isinstance(revision, str) or REVISION_RE.fullmatch(revision) is None:
        raise GoldenManifestError("upstream revision is not immutable")

    files = value["files"]
    if not isinstance(files, list) or len(files) != 2:
        raise GoldenManifestError("golden manifest must name exactly two files")
    expected_paths = ["config.json", "model.safetensors"]
    total_size = 0
    for index, item in enumerate(files):
        entry = _exact_dict(item, {"path", "sha256", "size_bytes", "url"}, "file")
        path = entry["path"]
        if path != expected_paths[index] or PurePosixPath(path).parts != (path,):
            raise GoldenManifestError("golden file path is unsafe or unexpected")
        size = _positive_int(entry["size_bytes"], f"files[{index}].size_bytes")
        total_size += size
        _sha256(entry["sha256"], f"files[{index}].sha256")
        expected_url = (
            f"https://huggingface.co/{upstream['repository']}/resolve/"
            f"{revision}/{quote(path)}"
        )
        if entry["url"] != expected_url or "/main/" in entry["url"]:
            raise GoldenManifestError("golden file URL is mutable or inconsistent")
    if total_size > MAX_TEMP_DISK_BYTES:
        raise GoldenManifestError("golden files exceed the temporary disk bound")

    constraints = _exact_dict(
        value["config_constraints"],
        {
            "allowed_architectures",
            "allowed_dtypes",
            "allowed_model_types",
            "minimum_max_position_embeddings",
            "plain_rope_only",
            "requires_merged_safetensors",
            "tie_word_embeddings",
        },
        "config_constraints",
    )
    if constraints != {
        "allowed_architectures": ["LlamaForCausalLM"],
        "allowed_dtypes": ["bfloat16"],
        "allowed_model_types": ["llama"],
        "minimum_max_position_embeddings": 128,
        "plain_rope_only": True,
        "requires_merged_safetensors": True,
        "tie_word_embeddings": True,
    }:
        raise GoldenManifestError("golden configuration constraints are unsupported")

    input_spec = _exact_dict(
        value["input"], {"mode", "profile_id", "sha256", "token_ids"}, "input"
    )
    if input_spec != {
        "mode": "direct_token_ids",
        "profile_id": "plain_rope_llama_direct_token_v1",
        "sha256": _input_identity("plain_rope_llama_direct_token_v1", [1]),
        "token_ids": [1],
    }:
        raise GoldenManifestError("golden input identity is unsupported")

    runtime = _exact_dict(
        value["runtime"],
        {
            "batch_size",
            "context_length",
            "generation_limit",
            "mode",
            "required_candidate_backend_identities",
            "required_candidate_backends",
            "required_reference_backend",
            "required_reference_backend_sha256",
            "require_supported",
            "stage_timeout_seconds",
            "thread_count",
        },
        "runtime",
    )
    timeout = _positive_int(runtime["stage_timeout_seconds"], "stage timeout")
    expected_runtime = {
        "batch_size": 1,
        "context_length": 128,
        "generation_limit": 8,
        "mode": "backend",
        "required_candidate_backend_identities": {
            "avx2": "sha256:4fa1687c09367703d6f40aebcf2f334e0706f81daeef5edef3e023e9f28f8e81"
        },
        "required_candidate_backends": ["avx2"],
        "required_reference_backend": "reference",
        "required_reference_backend_sha256": (
            "sha256:df44bae84ee6d73a4c78abc72ad28ba8111ba9f559f87d3ccc11b72d9e3e6e8e"
        ),
        "require_supported": True,
        "stage_timeout_seconds": timeout,
        "thread_count": 1,
    }
    if runtime != expected_runtime or timeout > MAX_STAGE_TIMEOUT_SECONDS:
        raise GoldenManifestError("golden runtime constraints are unsupported")

    expected = _exact_dict(
        value["expected"],
        {"policy_exit_code", "report_schema", "report_version", "semantic_verdict"},
        "expected",
    )
    if expected != {
        "policy_exit_code": 0,
        "report_schema": "lis.verification_report/v1",
        "report_version": "1.0",
        "semantic_verdict": "PASS",
    }:
        raise GoldenManifestError("golden expected result is unsupported")

    validated = _exact_dict(
        value["validated_lis"], {"revision", "source_sha256", "version"}, "validated_lis"
    )
    if (
        not isinstance(validated["revision"], str)
        or REVISION_RE.fullmatch(validated["revision"]) is None
        or VERSION_RE.fullmatch(_bounded_string(validated["version"], "version", 64))
        is None
    ):
        raise GoldenManifestError("validated LIS identity is malformed")
    _sha256(validated["source_sha256"], "validated_lis.source_sha256")

    baseline = _exact_dict(
        value["baseline_update"],
        {"manifest_change_required", "review_required"},
        "baseline_update",
    )
    if baseline != {"manifest_change_required": True, "review_required": True}:
        raise GoldenManifestError("golden baseline updates must require review")


@dataclass(frozen=True)
class GoldenManifest:
    canonical_bytes: bytes
    identity_sha256: str

    def materialize(self) -> dict[str, Any]:
        return json.loads(self.canonical_bytes)

    @property
    def expected_verdict(self) -> str:
        return self.materialize()["expected"]["semantic_verdict"]

    @property
    def expected_exit_code(self) -> int:
        return self.materialize()["expected"]["policy_exit_code"]


def load_manifest(reader: ManifestReader | None = None) -> GoldenManifest:
    read = _default_reader if reader is None else reader
    try:
        data = read(DEFAULT_RESOURCE)
    except GoldenManifestError:
        raise
    except (KeyError, OSError) as exc:
        raise GoldenManifestError("golden manifest is unavailable") from exc
    raw = _parse_object(data, "golden manifest")
    validate_manifest(raw)
    canonical = canonical_json_bytes(raw)
    if data != canonical:
        raise GoldenManifestError("golden manifest is not canonical JSON")
    return GoldenManifest(
        canonical_bytes=canonical,
        identity_sha256="sha256:" + hashlib.sha256(canonical).hexdigest(),
    )


def _no_symlink_path(path: Path) -> Path:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            info = current.lstat()
        except OSError as exc:
            raise GoldenManifestError("golden model path is unavailable") from exc
        if stat.S_ISLNK(info.st_mode):
            raise GoldenManifestError("golden model symlinks are prohibited")
    return absolute


@dataclass(frozen=True)
class VerifiedGoldenMaterial:
    manifest_sha256: str
    model_sha256: str
    config_sha256: str
    total_size_bytes: int


def verify_local_model(
    manifest: GoldenManifest, model_directory: Path
) -> VerifiedGoldenMaterial:
    directory = _no_symlink_path(Path(model_directory))
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise GoldenManifestError("golden model input must be a directory")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise GoldenManifestError("golden model directory owner is invalid")

    raw = manifest.materialize()
    observed: dict[str, str] = {}
    total_size = 0
    for entry in raw["files"]:
        path = _no_symlink_path(directory / entry["path"])
        try:
            digest, size = hash_regular_file(path)
        except (OSError, ValueError) as exc:
            raise GoldenManifestError(
                f"golden model file is unavailable: {entry['path']}"
            ) from exc
        if digest != entry["sha256"] or size != entry["size_bytes"]:
            raise GoldenManifestError(
                f"golden model file identity mismatch: {entry['path']}"
            )
        observed[entry["path"]] = digest
        total_size += size

    try:
        resolved = resolve_model(directory, load_model_profile())
    except (OSError, ValueError) as exc:
        raise GoldenManifestError("golden model violates the supported profile") from exc
    config = resolved.config
    constraints = raw["config_constraints"]
    if (
        config.get("architectures") != constraints["allowed_architectures"]
        or config.get("torch_dtype") not in constraints["allowed_dtypes"]
        or config.get("model_type") not in constraints["allowed_model_types"]
        or config.get("max_position_embeddings", 0)
        < constraints["minimum_max_position_embeddings"]
        or config.get("rope_scaling") is not None
        or config.get("tie_word_embeddings") is not True
    ):
        raise GoldenManifestError("golden model configuration drifted")
    if (
        resolved.model_sha256 != observed["model.safetensors"]
        or resolved.config_sha256 != observed["config.json"]
    ):
        raise GoldenManifestError("golden model identity changed during preflight")
    return VerifiedGoldenMaterial(
        manifest_sha256=manifest.identity_sha256,
        model_sha256=resolved.model_sha256,
        config_sha256=resolved.config_sha256,
        total_size_bytes=total_size,
    )


def _reader_for_path(path: Path) -> ManifestReader:
    def read(_: str) -> bytes:
        source = _no_symlink_path(path)
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(source, flags)
        except OSError as exc:
            raise GoldenManifestError("cannot open golden manifest") from exc
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):
                raise GoldenManifestError("golden manifest is not a regular file")
            data = bytearray()
            while len(data) <= MAX_MANIFEST_BYTES:
                chunk = os.read(
                    fd, min(65_536, MAX_MANIFEST_BYTES + 1 - len(data))
                )
                if not chunk:
                    break
                data.extend(chunk)
            after = os.fstat(fd)
            if (
                before.st_dev != after.st_dev
                or before.st_ino != after.st_ino
                or before.st_size != after.st_size
                or before.st_mtime_ns != after.st_mtime_ns
                or len(data) > MAX_MANIFEST_BYTES
            ):
                raise GoldenManifestError(
                    "golden manifest changed or exceeds its byte bound"
                )
            return bytes(data)
        finally:
            os.close(fd)

    return read


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lis_verify.golden",
        description="Verify explicit local material against the public golden manifest.",
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = load_manifest(
            None if args.manifest is None else _reader_for_path(args.manifest)
        )
        material = verify_local_model(manifest, args.model)
    except (GoldenManifestError, OSError, ValueError):
        print("golden-model: verification failed closed", file=sys.stderr)
        return 2
    print(
        "golden-model: verified "
        f"manifest={material.manifest_sha256} "
        f"model={material.model_sha256} "
        f"config={material.config_sha256} "
        f"bytes={material.total_size_bytes}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
