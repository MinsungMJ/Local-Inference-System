from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

from lis_verify.cli import main, parse_command
from lis_verify.orchestrator import OrchestrationResult
from lis_verify.report_model import VerificationReport
from tools.tests.lis_verify_product_spine_test_support import load_example


class TestCli(unittest.TestCase):
    def test_mode_requirements_and_defaults(self):
        demo = parse_command(["demo"])
        self.assertEqual(demo.mode, "demo")
        self.assertEqual(demo.output_root, Path(".lis/verify"))
        self.assertEqual(demo.stage_timeout_seconds, 1800)

        backend = parse_command(["backend", "--model", "model"])
        self.assertEqual(backend.model, Path("model"))

        runtime = parse_command(
            [
                "runtime",
                "--reference-bin",
                "ref",
                "--candidate-bin",
                "candidate",
                "--model",
                "model",
                "--require-supported",
            ]
        )
        self.assertEqual(runtime.reference_bin, Path("ref"))
        self.assertEqual(runtime.candidate_bin, Path("candidate"))
        self.assertTrue(runtime.require_supported)

    def test_invalid_inputs_exit_two_before_attempt(self):
        cases = (
            ["backend"],
            ["demo", "--model", "model"],
            ["runtime", "--reference-bin", "ref", "--candidate-bin", "candidate"],
            ["demo", "--stage-timeout-seconds", "0"],
            ["demo", "--stage-timeout-seconds", "7201"],
            ["demo", "--forced-prefix", "1"],
        )
        for argv in cases:
            with self.subTest(argv=argv):
                with redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as raised:
                        parse_command(argv)
                self.assertEqual(raised.exception.code, 2)

    def test_argument_error_does_not_echo_untrusted_argument(self):
        secret = "/private/example/do-not-echo"
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            with self.assertRaises(SystemExit):
                parse_command(["demo", "--unknown", secret])
        self.assertNotIn(secret, stderr.getvalue())
        self.assertLess(len(stderr.getvalue().encode()), 1024)

    def test_unavailable_runner_creates_no_attempt_or_output_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "not-created"
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = main(["demo", "--out", str(root)], runners={})
            self.assertEqual(result, 2)
            self.assertIn("not available", stderr.getvalue())
            self.assertFalse(root.exists())

    def test_injected_runner_prints_only_report_renderer_result(self):
        report = VerificationReport.from_dict(load_example("pass"))
        expected = "LIS Verify: PASS\n"

        def runner(request):
            return OrchestrationResult(
                report=report,
                report_path=Path("verification_report.json"),
                summary_path=Path("summary.md"),
                terminal_summary=expected,
                exit_code=0,
            )

        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = main(["demo"], runners={"demo": runner})
        self.assertEqual(code, 0)
        self.assertEqual(stdout.getvalue(), expected)

    def test_runner_failure_is_bounded_and_does_not_leak_traceback(self):
        stderr = io.StringIO()

        def fail(request):
            raise RuntimeError(f"private path: {request.output_root.absolute()}")

        with redirect_stderr(stderr):
            code = main(["demo"], runners={"demo": fail})
        self.assertEqual(code, 2)
        self.assertEqual(
            stderr.getvalue(),
            "lis-verify: verification orchestration failed closed\n",
        )

    def test_help_and_version_make_no_attempt(self):
        for argv in (["--help"], ["--version"]):
            with self.subTest(argv=argv), redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(argv)
                self.assertEqual(raised.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
