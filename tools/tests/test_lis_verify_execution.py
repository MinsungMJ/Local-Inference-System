import os
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time
import unittest

from lis_verify.execution import BoundedExecutor


class TestBoundedExecution(unittest.TestCase):
    def test_success_captures_both_channels(self):
        result = BoundedExecutor().run(
            [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"],
            timeout_seconds=5,
        )
        self.assertTrue(result.ok)
        self.assertEqual(result.stdout, b"out\n")
        self.assertEqual(result.stderr, b"err\n")

    def test_nonzero_and_spawn_error_are_classified_without_retry(self):
        result = BoundedExecutor().run(
            [sys.executable, "-c", "raise SystemExit(7)"],
            timeout_seconds=5,
        )
        self.assertEqual(result.status, "nonzero_exit")
        self.assertEqual(result.returncode, 7)
        missing = BoundedExecutor().run(["/definitely/missing/lis"], timeout_seconds=5)
        self.assertEqual(missing.status, "spawn_error")

    def test_combined_output_hard_cap(self):
        result = BoundedExecutor(output_limit=64, termination_grace_seconds=0.1).run(
            [sys.executable, "-c", "import sys; sys.stdout.write('a'*50); sys.stderr.write('b'*50)"],
            timeout_seconds=5,
        )
        self.assertEqual(result.status, "output_limit")
        self.assertTrue(result.output_limited)
        self.assertEqual(len(result.stdout) + len(result.stderr), 64)

    def test_timeout_terminates_and_reaps(self):
        result = BoundedExecutor(termination_grace_seconds=0.1).run(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            timeout_seconds=0.1,
        )
        self.assertEqual(result.status, "timeout")
        self.assertTrue(result.timed_out)
        self.assertIsNotNone(result.returncode)

    @unittest.skipUnless(os.name == "posix", "POSIX process groups required")
    def test_timeout_kills_descendant_holding_capture_pipe(self):
        script = (
            "import subprocess,sys; "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)']); "
            "print(p.pid, flush=True)"
        )
        result = BoundedExecutor(termination_grace_seconds=0.1).run(
            [sys.executable, "-c", script],
            timeout_seconds=0.2,
        )
        self.assertEqual(result.status, "timeout")
        child_pid = int(result.stdout.strip())
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            try:
                state = Path(f"/proc/{child_pid}/stat").read_text().split()[2]
            except FileNotFoundError:
                break
            if state == "Z":
                break
            time.sleep(0.01)
        else:
            self.fail("descendant process survived scoped timeout cleanup")

    @unittest.skipUnless(os.name == "posix", "POSIX signal handling required")
    def test_sigterm_is_handled_and_maps_to_143(self):
        timer = threading.Timer(0.15, os.kill, args=(os.getpid(), signal.SIGTERM))
        timer.start()
        try:
            result = BoundedExecutor(termination_grace_seconds=0.1).run(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                timeout_seconds=5,
            )
        finally:
            timer.cancel()
            timer.join()
        self.assertEqual(result.status, "interrupted")
        self.assertEqual(result.interrupted_signal, "SIGTERM")
        self.assertEqual(result.signal_exit_code, 143)

    def test_shell_metacharacters_are_plain_arguments(self):
        marker = "$(printf injected)"
        result = BoundedExecutor().run(
            [sys.executable, "-c", "import sys; print(sys.argv[1])", marker],
            timeout_seconds=5,
        )
        self.assertEqual(result.stdout, (marker + "\n").encode())

    def test_invalid_limits_fail_before_spawn(self):
        with self.assertRaises(ValueError):
            BoundedExecutor(output_limit=0)
        with self.assertRaises(ValueError):
            BoundedExecutor().run([sys.executable], timeout_seconds=0)


if __name__ == "__main__":
    unittest.main()
