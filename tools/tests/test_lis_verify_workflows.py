from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / ".github/workflows/ci.yml"
GOLDEN = ROOT / ".github/workflows/golden.yml"
MAKEFILE = ROOT / "Makefile"


class TestLISVerifyWorkflows(unittest.TestCase):
    def test_all_external_actions_are_full_sha_pinned(self):
        for path in (CI, GOLDEN):
            text = path.read_text(encoding="utf-8")
            uses = re.findall(r"^\s*uses:\s*([^\s#]+)", text, flags=re.MULTILINE)
            self.assertTrue(uses, path)
            for value in uses:
                with self.subTest(path=path.name, action=value):
                    self.assertRegex(value, r"^[^@]+@[0-9a-f]{40}$")

    def test_pull_request_demo_is_acceptance_classified_and_uploaded(self):
        text = CI.read_text(encoding="utf-8")
        for required in (
            "pull_request:",
            "timeout-minutes: 15",
            "lis_verify.acceptance",
            "LIS_VERIFY_ACCEPTANCE_MANIFEST",
            "--expected-verdict REGRESSION",
            "--observed-exit",
            "--step-summary \"$GITHUB_STEP_SUMMARY\"",
            "--source-root .",
            "actions/upload-artifact@",
            "verification_report.json",
            "summary.md",
            "attempt.jsonl",
        ):
            self.assertIn(required, text)

    def test_golden_workflow_is_immutable_bounded_strict_and_off_pr(self):
        text = GOLDEN.read_text(encoding="utf-8")
        for required in (
            "schedule:",
            "workflow_dispatch:",
            "release:",
            "timeout-minutes: 30",
            "--max-filesize 704",
            "--max-filesize 269060552",
            "--max-time 600",
            "28e66ca6931668447a3bac213f23d990ad3b0e2b",
            "lis_verify.golden",
            "lis_verify.acceptance",
            "make verify-diff",
            "VERIFY_DIFF_OUT_DIR=",
            "VERIFY_STAGE_TIMEOUT_SECONDS=600",
            "--expected-verdict PASS",
            "--golden-model",
            "actions/upload-artifact@",
        ):
            self.assertIn(required, text)
        self.assertNotIn("pull_request:", text)
        self.assertNotIn("/main/", text)
        self.assertNotIn("/home/", text)
        self.assertNotIn("--retry", text)

    def test_make_wrapper_is_thin_strict_and_never_downloads(self):
        text = MAKEFILE.read_text(encoding="utf-8")
        start = text.index("verify-diff:")
        end = text.index("\nverify-test-isolation:", start)
        target = text[start:end]
        self.assertIn("VERIFY_MODEL is required", target)
        self.assertIn("python3 -m lis_verify backend", target)
        self.assertIn("--require-supported", target)
        self.assertIn("--stage-timeout-seconds", target)
        self.assertNotIn("curl", target)
        self.assertNotIn("wget", target)


if __name__ == "__main__":
    unittest.main()
