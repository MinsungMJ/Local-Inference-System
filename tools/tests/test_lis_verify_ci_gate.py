from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

from lis_verify.ci_gate import (
    CIGateError,
    validate_attempt,
    validate_golden_report,
    validate_report_result,
)
from lis_verify.demo import run_demo
from lis_verify.golden import load_manifest
from lis_verify.orchestrator import CommandRequest
from lis_verify.product_contract import CustomerVerdict
from lis_verify.report_model import VerificationReport
from tools.tests.lis_verify_product_spine_test_support import load_example


class TestCIGate(unittest.TestCase):
    def test_all_five_verdicts_and_frozen_exits_are_distinguished(self):
        for name in (
            "pass",
            "regression",
            "inconclusive",
            "unsupported",
            "harness_error",
        ):
            report = VerificationReport.from_dict(load_example(name))
            with self.subTest(verdict=report.verdict.value):
                validate_report_result(
                    report,
                    expected_verdict=report.verdict,
                    observed_exit=report.exit_code,
                )
                with self.assertRaisesRegex(CIGateError, "exit"):
                    validate_report_result(
                        report,
                        expected_verdict=report.verdict,
                        observed_exit=99,
                    )

    def test_strict_unsupported_stays_unsupported_and_exits_six(self):
        report = VerificationReport.from_dict(load_example("unsupported"))
        self.assertEqual(report.verdict, CustomerVerdict.UNSUPPORTED)
        self.assertEqual(report.exit_code, 6)
        validate_report_result(
            report,
            expected_verdict=CustomerVerdict.UNSUPPORTED,
            observed_exit=6,
        )

    def test_golden_gate_checks_shared_and_distinct_identities(self):
        manifest = load_manifest()
        golden = manifest.materialize()
        raw = load_example("pass")
        raw["command"]["mode"] = "backend"
        raw["command"]["require_supported"] = True
        raw["policy_result"]["policy"] = "require_supported"
        files = {entry["path"]: entry["sha256"] for entry in golden["files"]}
        shared = {
            "source_sha256": "sha256:" + "1" * 64,
            "binary_sha256": "sha256:" + "2" * 64,
            "model_sha256": files["model.safetensors"],
            "config_sha256": files["config.json"],
            "input_sha256": golden["input"]["sha256"],
        }
        raw["identities"]["reference"].update(shared)
        raw["identities"]["candidate"].update(shared)
        raw["identities"]["reference"]["backend_sha256"] = golden["runtime"][
            "required_reference_backend_sha256"
        ]
        raw["identities"]["candidate"]["backend_sha256"] = next(
            iter(
                golden["runtime"]["required_candidate_backend_identities"].values()
            )
        )
        report = VerificationReport.from_dict(raw)
        validate_golden_report(report, manifest)

        fallback = copy.deepcopy(raw)
        fallback["identities"]["candidate"]["backend_sha256"] = fallback[
            "identities"
        ]["reference"]["backend_sha256"]
        with self.assertRaisesRegex(CIGateError, "fallback"):
            validate_golden_report(VerificationReport.from_dict(fallback), manifest)

        unknown_backend = copy.deepcopy(raw)
        unknown_backend["identities"]["candidate"]["backend_sha256"] = (
            "sha256:" + "8" * 64
        )
        with self.assertRaisesRegex(CIGateError, "fallback"):
            validate_golden_report(
                VerificationReport.from_dict(unknown_backend), manifest
            )

        drift = copy.deepcopy(raw)
        drift["identities"]["candidate"]["model_sha256"] = "sha256:" + "9" * 64
        with self.assertRaisesRegex(CIGateError, "input identity"):
            validate_golden_report(VerificationReport.from_dict(drift), manifest)

    def test_attempt_bundle_report_summary_and_ledger_are_consumed_together(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "verify"
            result = run_demo(CommandRequest(mode="demo", output_root=root))
            gate = validate_attempt(
                attempt_root=root,
                expected_verdict=CustomerVerdict.REGRESSION,
                observed_exit=4,
            )
            self.assertEqual(gate.verdict, CustomerVerdict.REGRESSION)
            result.summary_path.write_text("tampered\n")
            result.summary_path.chmod(0o600)
            with self.assertRaisesRegex(CIGateError, "projection"):
                validate_attempt(
                    attempt_root=root,
                    expected_verdict=CustomerVerdict.REGRESSION,
                    observed_exit=4,
                )

    def test_multiple_or_symlinked_attempts_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "verify"
            run_demo(CommandRequest(mode="demo", output_root=root))
            (root / "attempt-extra").mkdir(mode=0o700)
            with self.assertRaisesRegex(CIGateError, "exactly one"):
                validate_attempt(
                    attempt_root=root,
                    expected_verdict=CustomerVerdict.REGRESSION,
                    observed_exit=4,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "verify"
            run_demo(CommandRequest(mode="demo", output_root=root))
            link = Path(temporary) / "verify-link"
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(CIGateError, "symlink"):
                validate_attempt(
                    attempt_root=link,
                    expected_verdict=CustomerVerdict.REGRESSION,
                    observed_exit=4,
                )


if __name__ == "__main__":
    unittest.main()
