from pathlib import Path
import stat
import tempfile
import unittest

from lis_verify.ledger import AppendOnlyLedger, LedgerError, load_ledger
from lis_verify.product_contract import (
    CustomerVerdict,
    ResidueStatus,
    StageState,
    WorkflowClassification,
)
from lis_verify.workspace import AttemptWorkspace


class TestLedger(unittest.TestCase):
    def _ledger(self, temporary):
        workspace = AttemptWorkspace.create(Path(temporary) / "verify")
        ledger = AppendOnlyLedger(
            workspace.ledger_path,
            attempt_id=workspace.attempt_id,
            workflow=WorkflowClassification.DEVELOPMENT_DEBUGGING,
            clock=lambda: "2026-09-04T00:00:00Z",
        )
        return workspace, ledger

    def test_complete_ledger_is_private_and_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace, ledger = self._ledger(temporary)
            ledger.start_attempt("demo")
            ledger.start_stage("preflight")
            ledger.finish_stage("preflight", StageState.EXECUTED)
            ledger.observe_cleanup(ResidueStatus.NONE_OBSERVED)
            ledger.finish_attempt(CustomerVerdict.PASS)
            self.assertEqual(stat.S_IMODE(workspace.ledger_path.stat().st_mode), 0o600)
            events = load_ledger(workspace.ledger_path)
            self.assertEqual([event["sequence"] for event in events], list(range(5)))
            self.assertEqual(events[-1]["payload"]["verdict"], "PASS")

    def test_stage_retry_and_finish_before_start_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, ledger = self._ledger(temporary)
            ledger.start_attempt("demo")
            with self.assertRaises(LedgerError):
                ledger.finish_stage("preflight", StageState.FAILED)
            ledger.start_stage("preflight")
            with self.assertRaises(LedgerError):
                ledger.start_stage("preflight")
            ledger.abort()

    def test_finish_requires_cleanup_and_closed_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, ledger = self._ledger(temporary)
            ledger.start_attempt("demo")
            ledger.start_stage("preflight")
            with self.assertRaises(LedgerError):
                ledger.finish_attempt(CustomerVerdict.HARNESS_ERROR)
            ledger.abort()

    def test_unknown_mode_is_rejected_before_append(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace, ledger = self._ledger(temporary)
            with self.assertRaises(LedgerError):
                ledger.start_attempt("future")
            self.assertEqual(workspace.ledger_path.read_bytes(), b"")
            ledger.abort()

    def test_existing_ledger_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace, ledger = self._ledger(temporary)
            ledger.abort()
            with self.assertRaises(LedgerError):
                AppendOnlyLedger(
                    workspace.ledger_path,
                    attempt_id=workspace.attempt_id,
                    workflow=WorkflowClassification.DEVELOPMENT_DEBUGGING,
                )

    def test_external_append_is_detected_before_next_event(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace, ledger = self._ledger(temporary)
            ledger.start_attempt("demo")
            with workspace.ledger_path.open("ab") as stream:
                stream.write(b"tamper\n")
            with self.assertRaisesRegex(LedgerError, "append-only authority"):
                ledger.observe_cleanup(ResidueStatus.UNKNOWN)
            ledger.abort()

    def test_loader_rejects_truncated_final_line(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = AttemptWorkspace.create(Path(temporary) / "verify")
            workspace.ledger_path.write_bytes(b'{}')
            workspace.ledger_path.chmod(0o600)
            with self.assertRaisesRegex(LedgerError, "complete JSON line"):
                load_ledger(workspace.ledger_path)


if __name__ == "__main__":
    unittest.main()
