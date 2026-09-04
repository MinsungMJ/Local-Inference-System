from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from lis_verify.execution import ExecutionResult
from lis_verify.orchestrator import CommandRequest
from lis_verify.product_contract import CustomerVerdict, StageState
from lis_verify.real import run_real
from lis_verify.provenance import sidecar_path
from tools.tests.real_execution_test_support import (
    SeededMismatchExecutor,
    write_fake_binary,
    write_tiny_llama,
)


ROOT = Path(__file__).resolve().parents[2]
BINARY = ROOT / "srcs" / "libs" / "lis"


class TimeoutExecutor:
    def __init__(self):
        self.calls = 0

    def run(self, argv, **kwargs):
        self.calls += 1
        return ExecutionResult("timeout", -15, b"", b"", timed_out=True)


class InterruptedExecutor:
    def __init__(self):
        self.calls = 0

    def run(self, argv, **kwargs):
        self.calls += 1
        return ExecutionResult(
            "interrupted",
            -15,
            b"",
            b"",
            interrupted_signal="SIGTERM",
        )


class BinaryTamperExecutor(SeededMismatchExecutor):
    def run(self, argv, **kwargs):
        result = super().run(argv, **kwargs)
        if self.calls == 1:
            binary = Path(argv[0])
            binary.write_bytes(binary.read_bytes() + b"tampered")
        return result


class RecaptureBoundaryDriftExecutor(SeededMismatchExecutor):
    def run(self, argv, **kwargs):
        result = super().run(argv, **kwargs)
        if self.calls == 6:
            report_path = Path(argv[argv.index("--report-json") + 1])
            raw = json.loads(report_path.read_text(encoding="utf-8"))
            raw["report"]["selected_token_ids"] = [0, 1, 0, 1, 0, 1, 2, 0]
            report_path.write_text(
                json.dumps(raw, separators=(",", ":"), sort_keys=True),
                encoding="utf-8",
            )
        return result


class ReusedArtifactSetExecutor(SeededMismatchExecutor):
    def __init__(self):
        super().__init__()
        self.first_artifact_set = None

    def run(self, argv, **kwargs):
        result = super().run(argv, **kwargs)
        report_path = Path(argv[argv.index("--report-json") + 1])
        raw = json.loads(report_path.read_text(encoding="utf-8"))
        if self.first_artifact_set is None:
            self.first_artifact_set = raw["artifact_set_id"]
        elif self.calls == 2:
            raw["artifact_set_id"] = self.first_artifact_set
            report_path.write_text(
                json.dumps(raw, separators=(",", ":"), sort_keys=True),
                encoding="utf-8",
            )
        return result


class StepZeroMismatchExecutor(SeededMismatchExecutor):
    def __init__(self):
        super().__init__()
        self.forced_binding_calls = 0

    def run(self, argv, **kwargs):
        if "--forced-prefix-binding-json" in argv:
            self.forced_binding_calls += 1
        result = super().run(argv, **kwargs)
        report_path = Path(argv[argv.index("--report-json") + 1])
        raw = json.loads(report_path.read_text(encoding="utf-8"))
        candidate = "candidate" in Path(argv[0]).name
        selected = [1 if candidate else 0, 1, 2, 0, 1, 2, 0, 1]
        raw["report"]["selected_token_ids"] = selected
        raw["report"]["selected_token_count"] = len(selected)
        report_path.write_text(
            json.dumps(raw, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        return result


def built_binary_available() -> bool:
    return BINARY.is_file() and sidecar_path(BINARY).is_file()


class TestRealAdapter(unittest.TestCase):
    def test_missing_model_produces_canonical_unsupported_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = run_real(
                CommandRequest(
                    mode="backend",
                    model=root / "missing-model",
                    output_root=root / "verify",
                    require_supported=True,
                )
            )
        self.assertEqual(result.report.verdict, CustomerVerdict.UNSUPPORTED)
        self.assertEqual(result.exit_code, 6)
        self.assertEqual(result.report.stages[0].state, StageState.FAILED)

    @unittest.skipUnless(built_binary_available(), "built LIS binary required")
    def test_actual_backend_equal_path_is_report_first_and_private(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = write_tiny_llama(root / "model")
            result = run_real(
                CommandRequest(
                    mode="backend",
                    model=model,
                    output_root=root / "verify",
                )
            )
            self.assertIn(
                result.report.verdict,
                {CustomerVerdict.PASS, CustomerVerdict.UNSUPPORTED},
            )
            self.assertEqual(result.report_path.stat().st_mode & 0o777, 0o600)
            self.assertFalse((result.report_path.parent / "runtime").exists())
            raw = result.report.to_json_bytes()
            self.assertNotIn(os.fsencode(model), raw)
            if result.report.verdict == CustomerVerdict.PASS:
                self.assertEqual(result.report.to_dict()["token_comparison"]["status"], "equal")
                self.assertTrue(
                    all(
                        stage.state == StageState.NOT_APPLICABLE
                        for stage in result.report.stages[5:10]
                    )
                )

    @unittest.skipUnless(built_binary_available(), "built LIS binary required")
    def test_timeout_is_inconclusive_and_is_not_retried(self):
        fake = TimeoutExecutor()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = write_tiny_llama(root / "model")
            result = run_real(
                CommandRequest(
                    mode="backend",
                    model=model,
                    output_root=root / "verify",
                ),
                executor=fake,
            )
        self.assertEqual(fake.calls, 1)
        self.assertEqual(result.report.verdict, CustomerVerdict.INCONCLUSIVE)
        self.assertEqual(result.exit_code, 3)
        self.assertEqual(
            result.report.stages[1].state,
            StageState.FAILED,
        )

    @unittest.skipUnless(built_binary_available(), "built LIS binary required")
    def test_handled_signal_preserves_exit_code_and_is_not_retried(self):
        fake = InterruptedExecutor()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = write_tiny_llama(root / "model")
            result = run_real(
                CommandRequest(
                    mode="backend",
                    model=model,
                    output_root=root / "verify",
                ),
                executor=fake,
            )
        self.assertEqual(fake.calls, 1)
        self.assertEqual(result.report.verdict, CustomerVerdict.INCONCLUSIVE)
        self.assertEqual(result.exit_code, 143)

    @unittest.skipUnless(built_binary_available(), "built LIS binary required")
    def test_backend_reference_fallback_cannot_be_reported_as_pass(self):
        fake = SeededMismatchExecutor()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = write_tiny_llama(root / "model")
            result = run_real(
                CommandRequest(
                    mode="backend",
                    model=model,
                    output_root=root / "verify",
                ),
                executor=fake,
            )
        self.assertEqual(fake.calls, 2)
        self.assertEqual(result.report.verdict, CustomerVerdict.UNSUPPORTED)
        self.assertEqual(result.report.stages[2].state, StageState.FAILED)

    @unittest.skipUnless(built_binary_available(), "built LIS binary required")
    def test_runtime_rejects_one_binary_masquerading_as_two_roles(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = write_tiny_llama(root / "model")
            result = run_real(
                CommandRequest(
                    mode="runtime",
                    model=model,
                    reference_bin=BINARY,
                    candidate_bin=BINARY,
                    output_root=root / "verify",
                    require_supported=True,
                )
            )
        self.assertEqual(result.report.verdict, CustomerVerdict.UNSUPPORTED)
        self.assertEqual(result.exit_code, 6)

    def test_missing_provenance_is_inconclusive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = write_tiny_llama(root / "model")
            reference = write_fake_binary(root / "reference-lis", "reference")
            candidate = write_fake_binary(root / "candidate-lis", "candidate")
            sidecar_path(candidate).unlink()
            result = run_real(
                CommandRequest(
                    mode="runtime",
                    model=model,
                    reference_bin=reference,
                    candidate_bin=candidate,
                    output_root=root / "verify",
                ),
                executor=SeededMismatchExecutor(),
            )
        self.assertEqual(result.report.verdict, CustomerVerdict.INCONCLUSIVE)
        self.assertEqual(result.exit_code, 3)
        self.assertEqual(result.report.stages[0].state, StageState.FAILED)

    def test_malformed_provenance_is_an_integrity_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = write_tiny_llama(root / "model")
            reference = write_fake_binary(root / "reference-lis", "reference")
            candidate = write_fake_binary(root / "candidate-lis", "candidate")
            sidecar_path(candidate).write_bytes(b"{}")
            result = run_real(
                CommandRequest(
                    mode="runtime",
                    model=model,
                    reference_bin=reference,
                    candidate_bin=candidate,
                    output_root=root / "verify",
                ),
                executor=SeededMismatchExecutor(),
            )
        self.assertEqual(result.report.verdict, CustomerVerdict.HARNESS_ERROR)
        self.assertEqual(result.exit_code, 2)

    def test_reused_artifact_set_fails_closed(self):
        fake = ReusedArtifactSetExecutor()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = write_tiny_llama(root / "model", layers=12)
            reference = write_fake_binary(root / "reference-lis", "reference")
            candidate = write_fake_binary(root / "candidate-lis", "candidate")
            result = run_real(
                CommandRequest(
                    mode="runtime",
                    model=model,
                    reference_bin=reference,
                    candidate_bin=candidate,
                    output_root=root / "verify",
                ),
                executor=fake,
            )
        self.assertEqual(fake.calls, 2)
        self.assertEqual(result.report.verdict, CustomerVerdict.HARNESS_ERROR)
        self.assertEqual(result.report.stages[2].state, StageState.FAILED)
        self.assertIn(
            "reused an artifact set",
            " ".join(result.report.to_dict()["warnings"]),
        )

    def test_runtime_mismatch_runs_forced_prefix_and_pass3a_through_pass4(self):
        fake = SeededMismatchExecutor()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = write_tiny_llama(root / "model", layers=12)
            reference = write_fake_binary(root / "reference-lis", "reference")
            candidate = write_fake_binary(root / "candidate-lis", "candidate")
            result = run_real(
                CommandRequest(
                    mode="runtime",
                    model=model,
                    reference_bin=reference,
                    candidate_bin=candidate,
                    output_root=root / "verify",
                ),
                executor=fake,
            )
            self.assertFalse((result.report_path.parent / "runtime").exists())
        self.assertEqual(fake.calls, 8)
        self.assertEqual(result.report.verdict, CustomerVerdict.REGRESSION)
        self.assertEqual(result.exit_code, 4)
        self.assertTrue(
            all(stage.state == StageState.EXECUTED for stage in result.report.stages)
        )
        raw = result.report.to_dict()
        self.assertEqual(raw["token_comparison"]["first_mismatch"]["generated_token_step"], 2)
        self.assertEqual(raw["localization"]["layer_suspect_interval"], "(0, 4]")
        self.assertEqual(
            raw["localization"]["intra_layer_suspect_interval"],
            "(rope_key_output, attention_scores]",
        )

    def test_step_zero_mismatch_uses_independent_reruns_without_forced_prefix(self):
        fake = StepZeroMismatchExecutor()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = write_tiny_llama(root / "model", layers=12)
            reference = write_fake_binary(root / "reference-lis", "reference")
            candidate = write_fake_binary(root / "candidate-lis", "candidate")
            result = run_real(
                CommandRequest(
                    mode="runtime",
                    model=model,
                    reference_bin=reference,
                    candidate_bin=candidate,
                    output_root=root / "verify",
                ),
                executor=fake,
            )
        self.assertEqual(fake.calls, 8)
        self.assertEqual(fake.forced_binding_calls, 0)
        self.assertEqual(result.report.verdict, CustomerVerdict.REGRESSION)
        self.assertEqual(
            result.report.to_dict()["token_comparison"]["first_mismatch"][
                "generated_token_step"
            ],
            0,
        )

    def test_binary_change_during_execution_fails_closed(self):
        fake = BinaryTamperExecutor()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = write_tiny_llama(root / "model", layers=12)
            reference = write_fake_binary(root / "reference-lis", "reference")
            candidate = write_fake_binary(root / "candidate-lis", "candidate")
            result = run_real(
                CommandRequest(
                    mode="runtime",
                    model=model,
                    reference_bin=reference,
                    candidate_bin=candidate,
                    output_root=root / "verify",
                ),
                executor=fake,
            )
        self.assertEqual(fake.calls, 1)
        self.assertEqual(result.report.verdict, CustomerVerdict.HARNESS_ERROR)
        self.assertEqual(result.report.stages[1].state, StageState.FAILED)

    def test_recapture_boundary_drift_retains_regression_and_blocks_pass3b(self):
        fake = RecaptureBoundaryDriftExecutor()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = write_tiny_llama(root / "model", layers=12)
            reference = write_fake_binary(root / "reference-lis", "reference")
            candidate = write_fake_binary(root / "candidate-lis", "candidate")
            result = run_real(
                CommandRequest(
                    mode="runtime",
                    model=model,
                    reference_bin=reference,
                    candidate_bin=candidate,
                    output_root=root / "verify",
                ),
                executor=fake,
            )
        self.assertEqual(result.report.verdict, CustomerVerdict.REGRESSION)
        self.assertEqual(
            result.report.stages[7].state,
            StageState.FAILED,
        )
        self.assertEqual(
            result.report.stages[8].state,
            StageState.BLOCKED,
        )


if __name__ == "__main__":
    unittest.main()
