"""Private attempt workspace creation and scoped cleanup."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import secrets
import shutil
import stat

from .product_contract import (
    ATTEMPT_ID_RE,
    MAX_TEMP_DISK_BYTES,
    CleanupStatus,
    ResidueStatus,
)


class WorkspaceError(RuntimeError):
    pass


@dataclass(frozen=True)
class CleanupObservation:
    status: CleanupStatus
    residue_status: ResidueStatus
    observed: bool
    retained_debug: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "residue_status": self.residue_status.value,
            "observed": self.observed,
            "retained_debug": self.retained_debug,
        }


def new_attempt_id() -> str:
    return f"lisa1:{secrets.token_hex(16)}"


def _lstat(path: Path) -> os.stat_result:
    try:
        return path.lstat()
    except OSError as exc:
        raise WorkspaceError(f"cannot inspect output path component: {path.name}") from exc


def _validate_existing_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if not current.exists() and not current.is_symlink():
            break
        info = _lstat(current)
        if stat.S_ISLNK(info.st_mode):
            raise WorkspaceError("symlink output path components are prohibited")
        if not stat.S_ISDIR(info.st_mode):
            raise WorkspaceError("output path component is not a directory")


def _validate_private_directory(path: Path) -> None:
    info = _lstat(path)
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise WorkspaceError("output root is not a real directory")
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise WorkspaceError("output root is not owned by the current user")
    if stat.S_IMODE(info.st_mode) & 0o077:
        raise WorkspaceError("output root must not grant group or other access")


def _ensure_private_root(path: Path) -> Path:
    absolute = path.absolute()
    _validate_existing_components(absolute)
    existed = absolute.exists()
    try:
        absolute.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not existed:
            absolute.chmod(0o700)
    except OSError as exc:
        raise WorkspaceError("cannot create output root") from exc
    _validate_existing_components(absolute)
    # mkdir's mode is umask-sensitive only toward stricter permissions.  Do not
    # relax a pre-existing directory; reject it if it is not already private.
    _validate_private_directory(absolute)
    return absolute


@dataclass
class AttemptWorkspace:
    output_root: Path
    attempt_id: str
    attempt_dir: Path
    runtime_dir: Path
    _cleanup: CleanupObservation | None = None

    @classmethod
    def create(
        cls,
        output_root: Path,
        *,
        attempt_id: str | None = None,
    ) -> "AttemptWorkspace":
        identity = attempt_id or new_attempt_id()
        if ATTEMPT_ID_RE.fullmatch(identity) is None:
            raise WorkspaceError("attempt identity is not canonical")
        root = _ensure_private_root(Path(output_root))
        leaf = root / f"attempt-{identity.split(':', 1)[1]}"
        try:
            leaf.mkdir(mode=0o700)
            leaf.chmod(0o700)
            runtime = leaf / "runtime"
            runtime.mkdir(mode=0o700)
            runtime.chmod(0o700)
        except FileExistsError as exc:
            raise WorkspaceError("attempt workspace already exists") from exc
        except OSError as exc:
            try:
                leaf.rmdir()
            except OSError:
                pass
            raise WorkspaceError("cannot create private attempt workspace") from exc
        _validate_private_directory(leaf)
        _validate_private_directory(runtime)
        return cls(root, identity, leaf, runtime)

    @property
    def ledger_path(self) -> Path:
        return self.attempt_dir / "attempt.jsonl"

    @property
    def report_path(self) -> Path:
        return self.attempt_dir / "verification_report.json"

    @property
    def summary_path(self) -> Path:
        return self.attempt_dir / "summary.md"

    def temp_usage_bytes(self) -> int:
        if not self.runtime_dir.exists():
            return 0
        runtime_info = _lstat(self.runtime_dir)
        if stat.S_ISLNK(runtime_info.st_mode) or not stat.S_ISDIR(runtime_info.st_mode):
            raise WorkspaceError("runtime workspace is not an owned directory")
        if hasattr(os, "getuid") and runtime_info.st_uid != os.getuid():
            raise WorkspaceError("runtime workspace owner changed")
        total = 0
        for root, directories, files in os.walk(self.runtime_dir, followlinks=False):
            root_path = Path(root)
            for name in directories:
                info = _lstat(root_path / name)
                if stat.S_ISLNK(info.st_mode):
                    raise WorkspaceError("symlink inside runtime workspace")
            for name in files:
                info = _lstat(root_path / name)
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                    raise WorkspaceError("non-regular runtime artifact")
                total += info.st_size
                if total > MAX_TEMP_DISK_BYTES:
                    raise WorkspaceError("temporary disk limit exceeded")
        return total

    def cleanup_runtime(self, *, debug_retain: bool = False) -> CleanupObservation:
        if self._cleanup is not None:
            return self._cleanup
        if debug_retain:
            self._cleanup = CleanupObservation(
                CleanupStatus.RETAINED_DEBUG,
                ResidueStatus.RETAINED_DEBUG,
                observed=True,
                retained_debug=True,
            )
            return self._cleanup
        try:
            if self.runtime_dir.exists() or self.runtime_dir.is_symlink():
                info = _lstat(self.runtime_dir)
                if (
                    stat.S_ISLNK(info.st_mode)
                    or not stat.S_ISDIR(info.st_mode)
                    or (hasattr(os, "getuid") and info.st_uid != os.getuid())
                ):
                    raise WorkspaceError("runtime workspace authority changed")
                shutil.rmtree(self.runtime_dir)
            observed = not self.runtime_dir.exists() and not self.runtime_dir.is_symlink()
        except (OSError, WorkspaceError):
            present = self.runtime_dir.exists() or self.runtime_dir.is_symlink()
            self._cleanup = CleanupObservation(
                CleanupStatus.FAILED,
                ResidueStatus.PRESENT if present else ResidueStatus.UNKNOWN,
                observed=present,
                retained_debug=False,
            )
            return self._cleanup
        self._cleanup = CleanupObservation(
            CleanupStatus.SUCCESS,
            ResidueStatus.NONE_OBSERVED,
            observed=observed,
            retained_debug=False,
        )
        return self._cleanup


def detect_stale_attempts(output_root: Path) -> tuple[str, ...]:
    root = Path(output_root)
    if not root.exists():
        return ()
    _validate_private_directory(root)
    stale: list[str] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name):
        if not child.name.startswith("attempt-"):
            continue
        info = _lstat(child)
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise WorkspaceError("invalid attempt entry in output root")
        report_path = child / "verification_report.json"
        if report_path.is_symlink():
            raise WorkspaceError("invalid report commit marker in attempt entry")
        if not report_path.exists():
            stale.append(child.name)
        else:
            report_info = _lstat(report_path)
            if stat.S_ISLNK(report_info.st_mode) or not stat.S_ISREG(report_info.st_mode):
                raise WorkspaceError("invalid report commit marker in attempt entry")
    return tuple(stale)
