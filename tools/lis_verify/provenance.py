"""Strict binary-adjacent build provenance for real LIS verification.

The public ``lis-verify`` command never accepts a caller-supplied source hash.
Instead, each eligible LIS executable is accompanied by a canonical
``<binary>.lis-build.json`` file that binds the binary bytes to the source tree
and build settings claimed by the builder.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile
from typing import Any, Sequence

from .product_contract import SHA256_RE, canonical_json_bytes


SCHEMA = "lis.build_provenance/v1"
KIND = "lis_build_provenance"
SOURCE_MANIFEST_DOMAIN = "lis.production_source_tree/v1"
SIDECAR_SUFFIX = ".lis-build.json"
MAX_PROVENANCE_BYTES = 64 * 1024
MAX_BUILD_FIELD_BYTES = 4096
MAX_SOURCE_FILES = 4096
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class ProvenanceError(ValueError):
    """The build provenance is missing, malformed, stale, or unbound."""


class ProvenanceUnavailableError(ProvenanceError):
    """The required provenance input does not exist."""


def _reject_constant(value: str) -> None:
    raise ProvenanceError(f"non-standard JSON constant: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise ProvenanceError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _exact_dict(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ProvenanceError(f"{label} has missing or unknown fields")
    return value


def _bounded_string(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value.encode("utf-8")) > MAX_BUILD_FIELD_BYTES
    ):
        raise ProvenanceError(f"{label} is empty or exceeds its byte bound")
    return value


def _normalized_build_field(value: str, label: str) -> str:
    if not isinstance(value, str):
        raise ProvenanceError(f"{label} must be a string")
    return _bounded_string(value.strip() or "(none)", label)


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ProvenanceError(f"{label} is not a canonical SHA-256 identity")
    return value


def sidecar_path(binary: Path) -> Path:
    return Path(os.fspath(binary) + SIDECAR_SUFFIX)


def _open_regular(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError as exc:
        raise ProvenanceUnavailableError(
            f"required provenance input is unavailable: {path.name}"
        ) from exc
    except OSError as exc:
        raise ProvenanceError(f"cannot open regular input: {path.name}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ProvenanceError(f"input is not a regular file: {path.name}")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise ProvenanceError(f"input owner is invalid: {path.name}")
        return fd, info
    except Exception:
        os.close(fd)
        raise


def _read_bounded(path: Path, maximum: int) -> bytes:
    fd, before = _open_regular(path)
    try:
        data = bytearray()
        while len(data) <= maximum:
            chunk = os.read(fd, min(65_536, maximum + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(fd)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
        ):
            raise ProvenanceError(f"input changed while reading: {path.name}")
        if len(data) > maximum:
            raise ProvenanceError(f"input exceeds its byte bound: {path.name}")
        return bytes(data)
    finally:
        os.close(fd)


def hash_regular_file(path: Path) -> tuple[str, int]:
    fd, before = _open_regular(path)
    try:
        digest = hashlib.sha256()
        total = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
        after = os.fstat(fd)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or total != after.st_size
        ):
            raise ProvenanceError(f"input changed while hashing: {path.name}")
        return "sha256:" + digest.hexdigest(), total
    finally:
        os.close(fd)


def _source_paths(root: Path) -> tuple[Path, ...]:
    paths = [root / "Makefile"]
    paths.extend((root / "srcs").rglob("*.c"))
    paths.extend((root / "srcs").rglob("*.h"))
    normalized = tuple(sorted({path for path in paths}, key=lambda item: item.as_posix()))
    if not normalized or len(normalized) > MAX_SOURCE_FILES:
        raise ProvenanceError("production source file count is outside its bound")
    return normalized


def source_tree_identity(root: Path) -> tuple[str, int]:
    source_root = Path(root).absolute()
    entries: list[dict[str, Any]] = []
    for path in _source_paths(source_root):
        try:
            relative = path.relative_to(source_root).as_posix()
        except ValueError as exc:
            raise ProvenanceError("production source escaped its root") from exc
        digest, size = hash_regular_file(path)
        entries.append({"path": relative, "sha256": digest, "size_bytes": size})
    manifest = {"domain": SOURCE_MANIFEST_DOMAIN, "files": entries}
    digest = "sha256:" + hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()
    return digest, len(entries)


def _git_identity(root: Path) -> tuple[str | None, bool | None]:
    def run(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", os.fspath(root), *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=10,
        )

    try:
        revision = run(("rev-parse", "HEAD"))
        if revision.returncode != 0:
            return None, None
        value = revision.stdout.strip()
        if _REVISION_RE.fullmatch(value) is None:
            return None, None
        status_result = run(
            ("status", "--porcelain", "--untracked-files=no", "--", "Makefile", "srcs")
        )
        if status_result.returncode != 0:
            return value, None
        return value, bool(status_result.stdout)
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None, None


def build_provenance(
    *,
    binary: Path,
    source_root: Path,
    compiler: str,
    cppflags: str,
    cflags: str,
    ldflags: str,
    ldlibs: str,
    simd: str,
) -> dict[str, Any]:
    if simd not in {"on", "off"}:
        raise ProvenanceError("SIMD build mode is unsupported")
    binary_sha256, binary_size = hash_regular_file(Path(binary))
    tree_sha256, file_count = source_tree_identity(Path(source_root))
    revision, dirty = _git_identity(Path(source_root))
    build = {
        "compiler": _normalized_build_field(compiler, "compiler"),
        "cppflags": _normalized_build_field(cppflags, "cppflags"),
        "cflags": _normalized_build_field(cflags, "cflags"),
        "ldflags": _normalized_build_field(ldflags, "ldflags"),
        "ldlibs": _normalized_build_field(ldlibs, "ldlibs"),
        "simd": simd,
    }
    return {
        "schema": SCHEMA,
        "kind": KIND,
        "source": {
            "tree_sha256": tree_sha256,
            "revision": revision,
            "dirty": dirty,
            "file_count": file_count,
        },
        "build": build,
        "binary": {"sha256": binary_sha256, "size_bytes": binary_size},
    }


def validate_provenance(raw: Any) -> None:
    value = _exact_dict(raw, {"schema", "kind", "source", "build", "binary"}, "provenance")
    if value["schema"] != SCHEMA or value["kind"] != KIND:
        raise ProvenanceError("provenance identity is unsupported")

    source = _exact_dict(
        value["source"], {"tree_sha256", "revision", "dirty", "file_count"}, "source"
    )
    _sha256(source["tree_sha256"], "source.tree_sha256")
    revision = source["revision"]
    if revision is not None and (
        not isinstance(revision, str) or _REVISION_RE.fullmatch(revision) is None
    ):
        raise ProvenanceError("source.revision is invalid")
    if source["dirty"] is not None and not isinstance(source["dirty"], bool):
        raise ProvenanceError("source.dirty is invalid")
    if (
        isinstance(source["file_count"], bool)
        or not isinstance(source["file_count"], int)
        or not 0 < source["file_count"] <= MAX_SOURCE_FILES
    ):
        raise ProvenanceError("source.file_count is outside its bound")

    build = _exact_dict(
        value["build"],
        {"compiler", "cppflags", "cflags", "ldflags", "ldlibs", "simd"},
        "build",
    )
    for field in ("compiler", "cppflags", "cflags", "ldflags", "ldlibs"):
        _bounded_string(build[field], f"build.{field}")
    if build["simd"] not in {"on", "off"}:
        raise ProvenanceError("build.simd is unsupported")

    binary = _exact_dict(value["binary"], {"sha256", "size_bytes"}, "binary")
    _sha256(binary["sha256"], "binary.sha256")
    if (
        isinstance(binary["size_bytes"], bool)
        or not isinstance(binary["size_bytes"], int)
        or binary["size_bytes"] <= 0
    ):
        raise ProvenanceError("binary.size_bytes is invalid")


@dataclass(frozen=True)
class BuildProvenance:
    source_sha256: str
    binary_sha256: str
    binary_size_bytes: int
    revision: str | None
    dirty: bool | None
    identity_sha256: str
    raw: dict[str, Any]


def load_build_provenance(
    binary: Path, *, provenance_path: Path | None = None
) -> BuildProvenance:
    binary_path = Path(binary)
    sidecar = sidecar_path(binary_path) if provenance_path is None else Path(provenance_path)
    data = _read_bounded(sidecar, MAX_PROVENANCE_BYTES)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProvenanceError("provenance is not UTF-8") from exc
    try:
        raw = json.loads(
            text,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except ProvenanceError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProvenanceError("provenance is not valid JSON") from exc
    validate_provenance(raw)
    canonical = canonical_json_bytes(raw)
    if data != canonical:
        raise ProvenanceError("provenance is not canonical JSON")
    binary_sha256, binary_size = hash_regular_file(binary_path)
    if (
        raw["binary"]["sha256"] != binary_sha256
        or raw["binary"]["size_bytes"] != binary_size
    ):
        raise ProvenanceError("provenance does not bind the resolved binary")
    return BuildProvenance(
        source_sha256=raw["source"]["tree_sha256"],
        binary_sha256=binary_sha256,
        binary_size_bytes=binary_size,
        revision=raw["source"]["revision"],
        dirty=raw["source"]["dirty"],
        identity_sha256="sha256:" + hashlib.sha256(canonical).hexdigest(),
        raw=raw,
    )


def write_build_provenance(path: Path, raw: dict[str, Any]) -> None:
    validate_provenance(raw)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(raw)
    if len(data) > MAX_PROVENANCE_BYTES:
        raise ProvenanceError("provenance exceeds its byte bound")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o644)
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise ProvenanceError("provenance write made no progress")
            view = view[written:]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lis_verify.provenance")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--binary", type=Path, required=True)
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--source-root", type=Path, required=True)
    generate.add_argument("--compiler", required=True)
    generate.add_argument("--cppflags", required=True)
    generate.add_argument("--cflags", required=True)
    generate.add_argument("--ldflags", required=True)
    generate.add_argument("--ldlibs", required=True)
    generate.add_argument("--simd", choices=("on", "off"), required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "generate":
        raise ProvenanceError("unsupported provenance command")
    expected_output = sidecar_path(args.binary)
    if args.output.absolute() != expected_output.absolute():
        raise ProvenanceError("provenance output must be adjacent to its binary")
    raw = build_provenance(
        binary=args.binary,
        source_root=args.source_root,
        compiler=args.compiler,
        cppflags=args.cppflags,
        cflags=args.cflags,
        ldflags=args.ldflags,
        ldlibs=args.ldlibs,
        simd=args.simd,
    )
    write_build_provenance(args.output, raw)
    load_build_provenance(args.binary, provenance_path=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
