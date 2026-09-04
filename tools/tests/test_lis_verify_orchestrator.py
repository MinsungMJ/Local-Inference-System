from pathlib import Path
import tempfile
import unittest

from lis_verify.ledger import load_ledger
from lis_verify.orchestrator import (
    AcceptanceManifest,
    CommandRequest,
    OrchestrationError,
    run_orchestration,
)
from lis_verify.product_contract import CustomerVerdict, WorkflowClassification
from lis_verify.report_artifact import load_report
from lis_verify.workspace import detect_stale_attempts
from tools.tests.lis_verify_product_spine_test_support import (
    drive_pre_cleanup_stages,
    load_example,
)


class TestOrchestrator(unittest.TestCase):
    def _runner(self, name):
        def run(context):
            raw = load_example(name)
            return drive_pre_cleanup_stages(context, raw)

        return run

    def _manifest(self):
        return AcceptanceManifest(
            source_revision="a" * 40,
            source_tree_sha256="sha256:" + "b" * 64,
            dependency_sha256="sha256:" + "c" * 64,
            commands_sha256="sha256:" + "d" * 64,
            clean_state_observed=True,
        )

    def test_report_bundle_ledger_and_terminal_share_one_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_orchestration(
                CommandRequest(mode="demo", output_root=Path(temporary) / "verify"),
                self._runner("regression"),
            )
            self.assertEqual(result.exit_code, 4)
            self.assertEqual(result.report.verdict, CustomerVerdict.REGRESSION)
            self.assertEqual(load_report(result.report_path), result.report)
            self.assertIn("LIS Verify: REGRESSION", result.terminal_summary)
            events = load_ledger(result.report_path.parent / "attempt.jsonl")
            self.assertEqual(events[-1]["payload"]["verdict"], "REGRESSION")
            self.assertEqual(len(events), 27)
            self.assertFalse((result.report_path.parent / "runtime").exists())

    def test_strict_unsupported_changes_exit_not_verdict(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_orchestration(
                CommandRequest(
                    mode="backend",
                    output_root=Path(temporary) / "verify",
                    require_supported=True,
                    model=Path("model"),
                ),
                self._runner("unsupported"),
            )
            self.assertEqual(result.report.verdict, CustomerVerdict.UNSUPPORTED)
            self.assertEqual(result.exit_code, 6)

    def test_debug_retention_is_explicit_in_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_orchestration(
                CommandRequest(
                    mode="demo",
                    output_root=Path(temporary) / "verify",
                    debug_retain=True,
                ),
                self._runner("pass"),
            )
            self.assertTrue(result.report.cleanup.retained_debug)
            self.assertEqual(result.report.cleanup.residue_status.value, "retained_debug")
            self.assertTrue((result.report_path.parent / "runtime").exists())

    def test_runner_cannot_bypass_state_machine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "verify"
            with self.assertRaises(OrchestrationError):
                run_orchestration(
                    CommandRequest(mode="demo", output_root=root),
                    lambda context: load_example("pass"),
                )
            stale = detect_stale_attempts(root)
            self.assertEqual(len(stale), 1)
            attempt = root / stale[0]
            self.assertFalse((attempt / "verification_report.json").exists())
            events = load_ledger(attempt / "attempt.jsonl")
            self.assertEqual(events[-1]["payload"]["verdict"], "HARNESS_ERROR")

    def test_acceptance_requires_manifest_before_workspace_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "verify"
            with self.assertRaisesRegex(OrchestrationError, "frozen manifest"):
                run_orchestration(
                    CommandRequest(
                        mode="demo",
                        output_root=root,
                        workflow=WorkflowClassification.VERIFICATION_ACCEPTANCE,
                    ),
                    self._runner("pass"),
                )
            self.assertFalse(root.exists())

    def test_manifest_bound_acceptance_is_distinct_from_debugging(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = run_orchestration(
                CommandRequest(
                    mode="demo",
                    output_root=Path(temporary) / "verify",
                    workflow=WorkflowClassification.VERIFICATION_ACCEPTANCE,
                    acceptance_manifest=self._manifest(),
                ),
                self._runner("pass"),
            )
            self.assertEqual(
                result.report.workflow_classification,
                WorkflowClassification.VERIFICATION_ACCEPTANCE,
            )

    def test_debugging_cannot_consume_acceptance_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "verify"
            with self.assertRaisesRegex(OrchestrationError, "cannot consume"):
                run_orchestration(
                    CommandRequest(
                        mode="demo",
                        output_root=root,
                        acceptance_manifest=self._manifest(),
                    ),
                    self._runner("pass"),
                )
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
