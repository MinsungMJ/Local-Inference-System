"""Mode-0600 append-only attempt ledger."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import stat
from typing import Callable

from .product_contract import (
    CANONICAL_STAGES,
    MAX_LEDGER_EVENT_BYTES,
    CustomerVerdict,
    LedgerEvent,
    ResidueStatus,
    StageState,
    WorkflowClassification,
    CLI_MODES,
    canonical_json_bytes,
    validate_ledger_events,
)


class LedgerError(RuntimeError):
    pass


def utc_now_seconds() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class AppendOnlyLedger:
    def __init__(
        self,
        path: Path,
        *,
        attempt_id: str,
        workflow: WorkflowClassification,
        clock: Callable[[], str] = utc_now_seconds,
    ) -> None:
        self.path = Path(path)
        self.attempt_id = attempt_id
        self.workflow = workflow
        self._clock = clock
        self._events: list[dict[str, object]] = []
        self._started_stages: set[str] = set()
        self._finished_stages: set[str] = set()
        self._cleanup_written = False
        self._finished = False
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            self._fd = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise LedgerError("cannot create append-only attempt ledger") from exc
        os.fchmod(self._fd, 0o600)
        info = os.fstat(self._fd)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) != 0o600:
            os.close(self._fd)
            raise LedgerError("attempt ledger is not a private regular file")
        if hasattr(os, "getuid") and info.st_uid != os.getuid():
            os.close(self._fd)
            raise LedgerError("attempt ledger owner is invalid")
        self._device = info.st_dev
        self._inode = info.st_ino
        self._expected_size = 0

    @property
    def events(self) -> tuple[dict[str, object], ...]:
        return tuple(json.loads(canonical_json_bytes(item)) for item in self._events)

    @property
    def cleanup_recorded(self) -> bool:
        return self._cleanup_written

    def _check_append_authority(self) -> None:
        descriptor = os.fstat(self._fd)
        try:
            path_info = self.path.lstat()
        except OSError as exc:
            raise LedgerError("attempt ledger path authority was lost") from exc
        if (
            descriptor.st_dev != self._device
            or descriptor.st_ino != self._inode
            or path_info.st_dev != self._device
            or path_info.st_ino != self._inode
            or descriptor.st_nlink != 1
            or descriptor.st_size != self._expected_size
            or stat.S_IMODE(descriptor.st_mode) != 0o600
            or not stat.S_ISREG(descriptor.st_mode)
        ):
            raise LedgerError("attempt ledger changed outside append-only authority")
        if hasattr(os, "getuid") and descriptor.st_uid != os.getuid():
            raise LedgerError("attempt ledger owner changed")

    def _append(self, event: LedgerEvent, payload: dict[str, object]) -> None:
        if self._finished:
            raise LedgerError("attempt ledger is already finished")
        self._check_append_authority()
        timestamp = self._clock()
        if self._events and timestamp < self._events[-1]["timestamp_utc"]:
            raise LedgerError("attempt ledger clock moved backwards")
        entry: dict[str, object] = {
            "sequence": len(self._events),
            "attempt_id": self.attempt_id,
            "workflow_classification": self.workflow.value,
            "event": event.value,
            "timestamp_utc": timestamp,
            "payload": payload,
        }
        encoded = canonical_json_bytes(entry)
        if len(encoded) > MAX_LEDGER_EVENT_BYTES:
            raise LedgerError("attempt ledger event exceeds its byte bound")
        view = memoryview(encoded)
        while view:
            written = os.write(self._fd, view)
            if written <= 0:
                raise LedgerError("attempt ledger append made no progress")
            view = view[written:]
        os.fsync(self._fd)
        self._expected_size += len(encoded)
        self._events.append(entry)

    def start_attempt(self, mode: str) -> None:
        if self._events:
            raise LedgerError("attempt already started")
        if mode not in CLI_MODES:
            raise LedgerError("attempt mode is unknown")
        self._append(LedgerEvent.ATTEMPT_STARTED, {"mode": mode})

    def start_stage(self, stage: str) -> None:
        if not self._events:
            raise LedgerError("attempt has not started")
        if stage not in CANONICAL_STAGES or stage in self._started_stages:
            raise LedgerError("invalid or repeated stage start")
        self._append(LedgerEvent.STAGE_STARTED, {"stage": stage})
        self._started_stages.add(stage)

    def finish_stage(self, stage: str, state: StageState) -> None:
        if stage not in self._started_stages or stage in self._finished_stages:
            raise LedgerError("stage finish does not match one active start")
        self._append(
            LedgerEvent.STAGE_FINISHED,
            {"stage": stage, "state": state.value},
        )
        self._finished_stages.add(stage)

    def observe_cleanup(self, residue_status: ResidueStatus) -> None:
        if self._cleanup_written:
            raise LedgerError("cleanup observation is already recorded")
        self._append(
            LedgerEvent.CLEANUP_OBSERVED,
            {"residue_status": residue_status.value},
        )
        self._cleanup_written = True

    def finish_attempt(self, verdict: CustomerVerdict) -> None:
        if not self._cleanup_written:
            raise LedgerError("cleanup must be observed before attempt finish")
        if self._started_stages != self._finished_stages:
            raise LedgerError("all started stages must finish")
        candidate = [*self._events]
        timestamp = self._clock()
        candidate.append(
            {
                "sequence": len(candidate),
                "attempt_id": self.attempt_id,
                "workflow_classification": self.workflow.value,
                "event": LedgerEvent.ATTEMPT_FINISHED.value,
                "timestamp_utc": timestamp,
                "payload": {"verdict": verdict.value},
            }
        )
        validate_ledger_events(candidate)
        self._append(LedgerEvent.ATTEMPT_FINISHED, {"verdict": verdict.value})
        self._finished = True
        os.close(self._fd)

    def abort(self) -> None:
        if not self._finished:
            try:
                os.close(self._fd)
            finally:
                self._finished = True

    def __enter__(self) -> "AppendOnlyLedger":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.abort()


def load_ledger(path: Path) -> list[dict[str, object]]:
    source = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(source, flags)
    except OSError as exc:
        raise LedgerError("cannot open attempt ledger") from exc
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or (hasattr(os, "getuid") and info.st_uid != os.getuid())
        ):
            raise LedgerError("attempt ledger is not a private owned regular file")
        maximum_total = MAX_LEDGER_EVENT_BYTES * (2 * len(CANONICAL_STAGES) + 3)
        if info.st_size > maximum_total:
            raise LedgerError("attempt ledger exceeds its total structural bound")
        data = bytearray()
        while len(data) <= maximum_total:
            chunk = os.read(fd, min(65_536, maximum_total + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        if len(data) > maximum_total:
            raise LedgerError("attempt ledger exceeds its total structural bound")
    finally:
        os.close(fd)
    if not data or not data.endswith(b"\n"):
        raise LedgerError("attempt ledger must end with a complete JSON line")
    events: list[dict[str, object]] = []
    def reject_duplicates(pairs):
        value = {}
        for key, child in pairs:
            if key in value:
                raise LedgerError("attempt ledger contains a duplicate key")
            value[key] = child
        return value

    for number, line in enumerate(bytes(data).splitlines(), start=1):
        if not line:
            raise LedgerError(f"blank ledger line at {number}")
        try:
            event = json.loads(line, object_pairs_hook=reject_duplicates)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LedgerError(f"invalid ledger line at {number}") from exc
        events.append(event)
    validate_ledger_events(events)
    return events
