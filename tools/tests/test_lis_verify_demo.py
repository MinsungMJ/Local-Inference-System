from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

from lis_verify.demo import run_demo
from lis_verify.demo_pipeline import semantic_digest
from lis_verify.execution import ExecutionResult
from lis_verify.ledger import load_ledger
from lis_verify.orchestrator import CommandRequest
from lis_verify.product_contract import CustomerVerdict, StageState


class FakeExecutor:
    def __init__(self, result: ExecutionResult, payload: bytes | None = None):
        self.result = result
        self.payload = payload
        self.calls = 0

    def run(self, argv, **kwargs):
        self.calls += 1
        if self.payload is not None:
            output = Path(argv[argv.index("--output") + 1])
            output.write_bytes(self.payload)
            output.chmod(0o600)
        return self.result


class TestDemoAdapter(unittest.TestCase):
    def request(self, root: Path, *, debug_retain: bool = False) -> CommandRequest:
        return CommandRequest(
            mode="demo",
            output_root=root,
            debug_retain=debug_retain,
        )

    def test_public_demo_writes_report_summary_and_clean_ledger(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_demo(self.request(Path(temporary) / "verify"))
            self.assertEqual(result.report.verdict, CustomerVerdict.REGRESSION)
            self.assertEqual(result.exit_code, 4)
            self.assertTrue(result.report_path.is_file())
            self.assertTrue(result.summary_path.is_file())
            self.assertFalse((result.report_path.parent / "runtime").exists())
            events = load_ledger(result.report_path.parent / "attempt.jsonl")
            self.assertEqual(events[-1]["payload"]["verdict"], "REGRESSION")

    def test_two_clean_attempts_have_different_ids_and_same_semantic_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "verify"
            left = run_demo(self.request(root))
            right = run_demo(self.request(root))
            self.assertNotEqual(left.report.attempt_id, right.report.attempt_id)
            self.assertEqual(
                semantic_digest(left.report), semantic_digest(right.report)
            )

    def test_timeout_is_inconclusive_and_never_retried(self):
        executor = FakeExecutor(
            ExecutionResult("timeout", -15, b"", b"", timed_out=True)
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = run_demo(
                self.request(Path(temporary) / "verify"), executor=executor
            )
            self.assertFalse((result.report_path.parent / "runtime").exists())
            events = load_ledger(result.report_path.parent / "attempt.jsonl")
            self.assertEqual(events[-1]["payload"]["verdict"], "INCONCLUSIVE")
        self.assertEqual(executor.calls, 1)
        self.assertEqual(result.report.verdict, CustomerVerdict.INCONCLUSIVE)
        self.assertEqual(result.exit_code, 3)
        self.assertEqual(result.report.stages[0].state, StageState.FAILED)
        self.assertTrue(
            all(stage.state == StageState.BLOCKED for stage in result.report.stages[1:10])
        )

    @unittest.skipUnless(os.name == "posix", "POSIX signal exits required")
    def test_handled_signal_preserves_process_exit_and_inconclusive_report(self):
        executor = FakeExecutor(
            ExecutionResult(
                "interrupted",
                -15,
                b"",
                b"",
                interrupted_signal="SIGTERM",
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = run_demo(
                self.request(Path(temporary) / "verify"), executor=executor
            )
            self.assertFalse((result.report_path.parent / "runtime").exists())
        self.assertEqual(executor.calls, 1)
        self.assertEqual(result.report.verdict, CustomerVerdict.INCONCLUSIVE)
        self.assertEqual(result.report.exit_code, 3)
        self.assertEqual(result.exit_code, 143)

    def test_malformed_worker_output_is_harness_error(self):
        executor = FakeExecutor(
            ExecutionResult("ok", 0, b"", b""), payload=b'{"truncated":'
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = run_demo(
                self.request(Path(temporary) / "verify"), executor=executor
            )
        self.assertEqual(executor.calls, 1)
        self.assertEqual(result.report.verdict, CustomerVerdict.HARNESS_ERROR)
        self.assertEqual(result.exit_code, 2)

    def test_worker_output_overflow_is_harness_error_and_not_retried(self):
        executor = FakeExecutor(
            ExecutionResult(
                "output_limit", -15, b"x", b"", output_limited=True
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            result = run_demo(
                self.request(Path(temporary) / "verify"), executor=executor
            )
        self.assertEqual(executor.calls, 1)
        self.assertEqual(result.report.verdict, CustomerVerdict.HARNESS_ERROR)
        self.assertEqual(result.exit_code, 2)

    def test_debug_retention_is_explicit_and_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_demo(
                self.request(Path(temporary) / "verify", debug_retain=True)
            )
            runtime = result.report_path.parent / "runtime"
            self.assertTrue(runtime.is_dir())
            worker_result = runtime / "demo_worker_result.json"
            self.assertTrue(worker_result.is_file())
            self.assertEqual(worker_result.stat().st_mode & 0o777, 0o600)
            self.assertTrue(result.report.cleanup.retained_debug)


if __name__ == "__main__":
    unittest.main()
