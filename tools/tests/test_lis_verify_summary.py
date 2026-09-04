import copy
import unittest

from lis_verify.product_contract import CustomerVerdict, MAX_DETAIL_BYTES, MAX_WARNINGS
from lis_verify.report_model import VerificationReport
from lis_verify.summary import render_markdown, render_terminal
from tools.tests.lis_verify_product_spine_test_support import load_example


class TestSummary(unittest.TestCase):
    def test_every_verdict_renders_deterministically(self):
        verdicts = set()
        for name in ("pass", "regression", "inconclusive", "unsupported", "harness_error"):
            report = VerificationReport.from_dict(load_example(name))
            verdicts.add(report.verdict)
            self.assertEqual(render_terminal(report), render_terminal(report))
            self.assertEqual(render_markdown(report), render_markdown(report))
            self.assertIn(report.verdict.value, render_terminal(report))
            self.assertIn(report.verdict.value, render_markdown(report))
        self.assertEqual(verdicts, set(CustomerVerdict))

    def test_renderer_escapes_control_and_markdown_content(self):
        raw = copy.deepcopy(load_example("regression"))
        raw["warnings"] = ["line1\n| line2\x1b[31m"]
        report = VerificationReport.from_dict(raw)
        terminal = render_terminal(report)
        markdown = render_markdown(report)
        self.assertNotIn("\x1b", terminal)
        self.assertNotIn("\x1b", markdown)
        self.assertNotIn("line1\n", markdown)
        self.assertIn("\\n", markdown)
        self.assertIn("\\|", markdown)

    def test_worst_case_warning_set_is_bounded(self):
        raw = copy.deepcopy(load_example("regression"))
        raw["warnings"] = [
            f"{index:02d}-" + "가" * ((MAX_DETAIL_BYTES - 3) // 3)
            for index in range(MAX_WARNINGS)
        ]
        report = VerificationReport.from_dict(raw)
        self.assertLessEqual(len(render_markdown(report).encode()), 65_536)

    def test_pass_has_no_next_action_section(self):
        report = VerificationReport.from_dict(load_example("pass"))
        self.assertNotIn("## Next action", render_markdown(report))


if __name__ == "__main__":
    unittest.main()
