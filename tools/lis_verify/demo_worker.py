"""Internal bounded worker for the offline demonstration."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import stat
import sys

from .demo_pipeline import build_demo_report
from .product_contract import MAX_REPORT_BYTES


def _write_private(path: Path, data: bytes) -> None:
    target = Path(path)
    if not target.is_absolute():
        raise ValueError("worker output must be absolute")
    parent = target.parent
    parent_info = parent.lstat()
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or stat.S_ISLNK(parent_info.st_mode)
        or stat.S_IMODE(parent_info.st_mode) & 0o077
        or (hasattr(os, "getuid") and parent_info.st_uid != os.getuid())
    ):
        raise ValueError("worker output parent is not private and owned")
    if not data or len(data) > MAX_REPORT_BYTES:
        raise ValueError("worker result exceeds its byte bound")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(target, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("worker result write made no progress")
            view = view[written:]
        os.fsync(fd)
    except Exception:
        os.close(fd)
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise
    os.close(fd)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lis-verify-demo-worker")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = build_demo_report()
        _write_private(args.output, report.to_json_bytes())
    except Exception:
        print("lis-verify demo worker failed closed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
