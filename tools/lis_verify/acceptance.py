"""Freeze and load the internal clean-run acceptance authority."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
from typing import Any, Sequence

from .orchestrator import AcceptanceManifest, OrchestrationError
from .product_contract import SHA256_RE, canonical_json_bytes
from .provenance import hash_regular_file, source_tree_identity


SCHEMA = "lis.verify.acceptance_manifest/v1"
KIND = "acceptance_manifest"
MAX_BYTES = 64 * 1024
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")


class AcceptanceManifestError(ValueError):
    """Acceptance authority was missing, mutable, dirty, or malformed."""


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise AcceptanceManifestError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _read_private(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise AcceptanceManifestError("cannot open acceptance manifest") from exc
    try:
        before = os.fstat(fd)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or (hasattr(os, "getuid") and before.st_uid != os.getuid())
        ):
            raise AcceptanceManifestError(
                "acceptance manifest is not a private owned regular file"
            )
        data = bytearray()
        while len(data) <= MAX_BYTES:
            chunk = os.read(fd, min(65_536, MAX_BYTES + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(fd)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or len(data) > MAX_BYTES
        ):
            raise AcceptanceManifestError("acceptance manifest changed or is oversized")
        return bytes(data)
    finally:
        os.close(fd)


def load_acceptance_manifest(path: Path) -> AcceptanceManifest:
    data = _read_private(Path(path))
    try:
        raw = json.loads(data.decode("utf-8"), object_pairs_hook=_reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcceptanceManifestError("acceptance manifest is not valid JSON") from exc
    fields = {
        "clean_state_observed",
        "commands_sha256",
        "dependency_sha256",
        "kind",
        "schema",
        "source_revision",
        "source_tree_sha256",
    }
    if not isinstance(raw, dict) or set(raw) != fields:
        raise AcceptanceManifestError(
            "acceptance manifest has missing or unknown fields"
        )
    if raw["schema"] != SCHEMA or raw["kind"] != KIND:
        raise AcceptanceManifestError("acceptance manifest identity is unsupported")
    if data != canonical_json_bytes(raw):
        raise AcceptanceManifestError("acceptance manifest is not canonical JSON")
    manifest = AcceptanceManifest(
        source_revision=raw["source_revision"],
        source_tree_sha256=raw["source_tree_sha256"],
        dependency_sha256=raw["dependency_sha256"],
        commands_sha256=raw["commands_sha256"],
        clean_state_observed=raw["clean_state_observed"],
    )
    try:
        manifest.validate()
    except OrchestrationError as exc:
        raise AcceptanceManifestError("acceptance manifest fields are invalid") from exc
    return manifest


def _git(root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", os.fspath(root), *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise AcceptanceManifestError("cannot inspect the source repository") from exc
    if result.returncode != 0:
        raise AcceptanceManifestError("source repository inspection failed")
    return result.stdout.strip()


def _collection_identity(domain: str, root: Path, paths: Sequence[Path]) -> str:
    entries: list[dict[str, Any]] = []
    for path in paths:
        absolute = path.absolute()
        try:
            relative = absolute.relative_to(root.absolute()).as_posix()
        except ValueError as exc:
            raise AcceptanceManifestError("acceptance input escaped its source root") from exc
        digest, size = hash_regular_file(absolute)
        entries.append({"path": relative, "sha256": digest, "size_bytes": size})
    value = {"domain": domain, "files": entries}
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def verify_acceptance_source(
    manifest: AcceptanceManifest, source_root: Path
) -> None:
    root = Path(source_root).absolute()
    if _git(root, "rev-parse", "HEAD") != manifest.source_revision:
        raise AcceptanceManifestError("acceptance source revision changed")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise AcceptanceManifestError("acceptance source has tracked modifications")
    current_tree, _ = source_tree_identity(root)
    if current_tree != manifest.source_tree_sha256:
        raise AcceptanceManifestError("acceptance production source identity changed")


def freeze_acceptance_manifest(
    *, source_root: Path, output: Path, command_files: Sequence[Path]
) -> str:
    root = Path(source_root).absolute()
    revision = _git(root, "rev-parse", "HEAD")
    if REVISION_RE.fullmatch(revision) is None:
        raise AcceptanceManifestError("source revision is not a full commit identity")
    if _git(root, "status", "--porcelain", "--untracked-files=no"):
        raise AcceptanceManifestError("acceptance source has tracked modifications")
    source_tree_sha256, _ = source_tree_identity(root)
    dependencies = [root / "pyproject.toml", root / "tools/requirements.txt"]
    dependency_sha256 = _collection_identity(
        "lis.acceptance.dependencies/v1", root, dependencies
    )
    if not command_files:
        raise AcceptanceManifestError("acceptance command authority is empty")
    commands_sha256 = _collection_identity(
        "lis.acceptance.commands/v1",
        root,
        [root / path for path in command_files],
    )
    raw = {
        "clean_state_observed": True,
        "commands_sha256": commands_sha256,
        "dependency_sha256": dependency_sha256,
        "kind": KIND,
        "schema": SCHEMA,
        "source_revision": revision,
        "source_tree_sha256": source_tree_sha256,
    }
    data = canonical_json_bytes(raw)
    destination = Path(output)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(destination, flags, 0o600)
    except OSError as exc:
        raise AcceptanceManifestError("cannot create acceptance manifest") from exc
    try:
        os.fchmod(fd, 0o600)
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise AcceptanceManifestError("acceptance manifest write made no progress")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    load_acceptance_manifest(destination)
    return "sha256:" + hashlib.sha256(data).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lis_verify.acceptance",
        description="Freeze clean source and command authority for one CI run.",
    )
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--command-file", required=True, action="append", type=Path
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        digest = freeze_acceptance_manifest(
            source_root=args.source_root,
            output=args.output,
            command_files=args.command_file,
        )
    except (AcceptanceManifestError, OSError, ValueError):
        print("acceptance-freeze: failed closed", file=sys.stderr)
        return 2
    print(f"acceptance-freeze: verified manifest={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
