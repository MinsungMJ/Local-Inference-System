"""Strict report loading and private no-overwrite bundle publication."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import secrets
import stat

from .product_contract import MAX_REPORT_BYTES, MAX_SUMMARY_BYTES
from .report_model import VerificationReport


class ArtifactPublicationError(RuntimeError):
    pass


@dataclass
class PreparedArtifact:
    temporary_path: Path
    final_path: Path
    published: bool = False

    def discard(self) -> None:
        if not self.published:
            try:
                self.temporary_path.unlink()
            except FileNotFoundError:
                pass


@dataclass
class PreparedReportBundle:
    summary: PreparedArtifact
    report: PreparedArtifact

    def discard(self) -> None:
        self.summary.discard()
        self.report.discard()


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise ArtifactPublicationError("artifact write made no progress")
        view = view[written:]


def prepare_private_artifact(
    final_path: Path,
    data: bytes,
    *,
    maximum_bytes: int,
) -> PreparedArtifact:
    final = Path(final_path)
    if len(data) > maximum_bytes:
        raise ArtifactPublicationError("artifact exceeds its byte bound")
    if final.exists() or final.is_symlink():
        raise ArtifactPublicationError("artifact overwrite is prohibited")
    temporary = final.parent / f".{final.name}.{secrets.token_hex(8)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(temporary, flags, 0o600)
        try:
            os.fchmod(fd, 0o600)
            _write_all(fd, data)
            os.fsync(fd)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
                raise ArtifactPublicationError("prepared artifact is not private")
        finally:
            os.close(fd)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return PreparedArtifact(temporary, final)


def prepare_report_bundle(
    report: VerificationReport,
    markdown: str,
    *,
    report_path: Path,
    summary_path: Path,
) -> PreparedReportBundle:
    markdown_bytes = markdown.encode("utf-8")
    if len(markdown_bytes) > MAX_SUMMARY_BYTES:
        raise ArtifactPublicationError("Markdown summary exceeds its byte bound")
    summary = prepare_private_artifact(
        summary_path,
        markdown_bytes,
        maximum_bytes=MAX_SUMMARY_BYTES,
    )
    try:
        prepared_report = prepare_private_artifact(
            report_path,
            report.to_json_bytes(),
            maximum_bytes=MAX_REPORT_BYTES,
        )
    except Exception:
        summary.discard()
        raise
    return PreparedReportBundle(summary, prepared_report)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_report_bundle(bundle: PreparedReportBundle) -> None:
    """Publish summary first and report last as the bundle commit marker."""

    published_summary = False
    try:
        for prepared in (bundle.summary, bundle.report):
            if prepared.final_path.exists() or prepared.final_path.is_symlink():
                raise ArtifactPublicationError("artifact overwrite is prohibited")
            # A hard link creates the final name atomically and fails with
            # EEXIST instead of replacing a concurrently created destination.
            os.link(prepared.temporary_path, prepared.final_path, follow_symlinks=False)
            try:
                prepared.temporary_path.unlink()
            except Exception:
                try:
                    prepared.final_path.unlink()
                finally:
                    raise
            prepared.published = True
            if prepared is bundle.summary:
                published_summary = True
        _fsync_directory(bundle.report.final_path.parent)
    except Exception as exc:
        bundle.discard()
        # If report did not become visible, summary is not a committed bundle
        # and belongs to this private attempt, so scoped rollback is safe.
        if published_summary and not bundle.report.published:
            try:
                bundle.summary.final_path.unlink()
            except FileNotFoundError:
                pass
        if isinstance(exc, ArtifactPublicationError):
            raise
        raise ArtifactPublicationError("cannot publish report bundle") from exc


def load_report(path: Path) -> VerificationReport:
    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(source, flags)
    except OSError as exc:
        raise ArtifactPublicationError("cannot open canonical report") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ArtifactPublicationError("canonical report is not a regular file")
        if stat.S_IMODE(info.st_mode) != 0o600:
            raise ArtifactPublicationError("canonical report is not private")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            raise ArtifactPublicationError("canonical report owner is invalid")
        data = bytearray()
        while len(data) <= MAX_REPORT_BYTES:
            chunk = os.read(fd, min(65_536, MAX_REPORT_BYTES + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        if len(data) > MAX_REPORT_BYTES:
            raise ArtifactPublicationError("canonical report exceeds its byte bound")
    finally:
        os.close(fd)
    return VerificationReport.from_json_bytes(bytes(data))
