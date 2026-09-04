"""Bounded shell-free subprocess execution primitives."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time
from typing import Mapping, Sequence

from .product_contract import (
    DEFAULT_STAGE_TIMEOUT_SECONDS,
    MAX_STAGE_TIMEOUT_SECONDS,
    MAX_SUBPROCESS_OUTPUT_BYTES,
    TERMINATION_GRACE_SECONDS,
)


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    returncode: int | None
    stdout: bytes
    stderr: bytes
    timed_out: bool = False
    output_limited: bool = False
    interrupted_signal: str | None = None
    detail: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.returncode == 0

    @property
    def signal_exit_code(self) -> int | None:
        if self.interrupted_signal == "SIGINT":
            return 130
        if self.interrupted_signal == "SIGTERM":
            return 143
        return None


class _TerminationRequested(Exception):
    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


class BoundedExecutor:
    def __init__(
        self,
        *,
        output_limit: int = MAX_SUBPROCESS_OUTPUT_BYTES,
        termination_grace_seconds: float = TERMINATION_GRACE_SECONDS,
    ) -> None:
        if output_limit <= 0 or output_limit > MAX_SUBPROCESS_OUTPUT_BYTES:
            raise ValueError("subprocess output limit is outside the frozen bound")
        if termination_grace_seconds < 0 or (
            termination_grace_seconds > TERMINATION_GRACE_SECONDS
        ):
            raise ValueError("termination grace is outside the frozen bound")
        self.output_limit = output_limit
        self.termination_grace_seconds = termination_grace_seconds

    @staticmethod
    def _validate_argv(argv: Sequence[str]) -> tuple[str, ...]:
        if isinstance(argv, (str, bytes)) or not argv:
            raise ValueError("subprocess argv must be a non-empty sequence")
        result = tuple(argv)
        if any(not isinstance(item, str) or not item or "\0" in item for item in result):
            raise ValueError("subprocess argv contains an invalid item")
        return result

    @staticmethod
    def _environment(environment: Mapping[str, str] | None) -> dict[str, str]:
        if environment is None:
            return {
                "PATH": os.defpath,
                "LANG": "C.UTF-8",
                "LC_ALL": "C.UTF-8",
            }
        result = dict(environment)
        for key, value in result.items():
            if (
                not isinstance(key, str)
                or not key
                or "=" in key
                or "\0" in key
                or not isinstance(value, str)
                or "\0" in value
            ):
                raise ValueError("subprocess environment contains an invalid item")
        return result

    def _terminate(self, process: subprocess.Popen[bytes]) -> None:
        def group_alive() -> bool:
            if os.name != "posix":
                return process.poll() is None
            try:
                os.killpg(process.pid, 0)
                return True
            except ProcessLookupError:
                return False

        if process.poll() is not None and not group_alive():
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except ProcessLookupError:
            return
        deadline = time.monotonic() + self.termination_grace_seconds
        while group_alive() and time.monotonic() < deadline:
            if process.poll() is None:
                try:
                    process.wait(
                        timeout=min(0.05, max(0.0, deadline - time.monotonic()))
                    )
                except subprocess.TimeoutExpired:
                    pass
            else:
                time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        if group_alive():
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.wait()

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        environment: Mapping[str, str] | None = None,
        timeout_seconds: float = DEFAULT_STAGE_TIMEOUT_SECONDS,
    ) -> ExecutionResult:
        command = self._validate_argv(argv)
        if (
            isinstance(timeout_seconds, bool)
            or timeout_seconds <= 0
            or timeout_seconds > MAX_STAGE_TIMEOUT_SECONDS
        ):
            raise ValueError("stage timeout is outside the frozen bound")
        env = self._environment(environment)
        try:
            process = subprocess.Popen(
                command,
                cwd=None if cwd is None else os.fspath(cwd),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                start_new_session=(os.name == "posix"),
            )
        except OSError as exc:
            return ExecutionResult(
                status="spawn_error",
                returncode=None,
                stdout=b"",
                stderr=b"",
                detail=exc.__class__.__name__,
            )

        assert process.stdout is not None
        assert process.stderr is not None
        streams = {process.stdout.fileno(): "stdout", process.stderr.fileno(): "stderr"}
        selector = selectors.DefaultSelector()
        for stream in (process.stdout, process.stderr):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ)
        captured = {"stdout": bytearray(), "stderr": bytearray()}
        total = 0
        deadline = time.monotonic() + timeout_seconds
        status: str | None = None
        interrupted_signal: str | None = None
        installed_handlers: dict[int, object] = {}
        try:
            import threading

            if threading.current_thread() is threading.main_thread():
                def request_termination(signum, frame):
                    del frame
                    raise _TerminationRequested(signum)

                for signum in (signal.SIGINT, signal.SIGTERM):
                    installed_handlers[signum] = signal.getsignal(signum)
                    signal.signal(signum, request_termination)
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    status = "timeout"
                    self._terminate(process)
                    break
                ready = selector.select(timeout=min(0.1, remaining))
                if not ready and process.poll() is not None:
                    ready = [(key, selectors.EVENT_READ) for key in selector.get_map().values()]
                for key, _ in ready:
                    try:
                        chunk = os.read(key.fd, min(65_536, self.output_limit + 1))
                    except BlockingIOError:
                        continue
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    allowed = self.output_limit - total
                    if len(chunk) > allowed:
                        if allowed:
                            captured[streams[key.fd]].extend(chunk[:allowed])
                            total += allowed
                        status = "output_limit"
                        self._terminate(process)
                        break
                    captured[streams[key.fd]].extend(chunk)
                    total += len(chunk)
                if status is not None:
                    break
        except (_TerminationRequested, KeyboardInterrupt) as exc:
            signum = exc.signum if isinstance(exc, _TerminationRequested) else signal.SIGINT
            interrupted_signal = signal.Signals(signum).name
            status = "interrupted"
            self._terminate(process)
        finally:
            for signum, handler in installed_handlers.items():
                signal.signal(signum, handler)
            selector.close()
            process.stdout.close()
            process.stderr.close()

        if process.poll() is None:
            process.wait()
        returncode = process.returncode
        if status is None:
            status = "ok" if returncode == 0 else "nonzero_exit"
        return ExecutionResult(
            status=status,
            returncode=returncode,
            stdout=bytes(captured["stdout"]),
            stderr=bytes(captured["stderr"]),
            timed_out=status == "timeout",
            output_limited=status == "output_limit",
            interrupted_signal=interrupted_signal,
        )
