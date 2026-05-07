"""Compare, interpretation, issues, and branding tests."""

from __future__ import annotations

import unittest
from pathlib import Path

from lis_inspect.app import LisInspectApp, _COMPARE_TAB_FACTORIES, _TAB_FACTORIES
from lis_inspect.cli import build_compare_session, build_session
from lis_inspect.inspection import compare_issues, compare_summary, single_issues
from lis_inspect.views.compare import render_compare
from lis_inspect.views.issues import render_issues


FIXTURES = Path(__file__).resolve().parent.parent / "test_fixtures"

RETENTION_FORBIDDEN_SUBSTRINGS = (
    "Write one short",
    "generated secret",
    "/home/",
    "/tmp/",
)


class InterpretationTest(unittest.TestCase):
    def test_full_fixture_has_compact_interpretation_labels(self) -> None:
        session = build_session(
            FIXTURES / "qwen3_1p7b_run_full.json",
            FIXTURES / "qwen3_1p7b_run_full.stderr",
        )

        self.assertEqual(session.derived.completeness_label, "full")
        self.assertEqual(session.derived.run_profile, "decode-heavy")
        self.assertIn(session.derived.warmup_effect, {"none", "mild", "present"})
        self.assertIn(session.derived.tail_spread, {"low", "moderate", "high"})

    def test_json_only_is_partial(self) -> None:
        session = build_session(FIXTURES / "qwen3_1p7b_run.json", None)
        self.assertEqual(session.derived.completeness_label, "partial")


class CompareModeTest(unittest.TestCase):
    def test_compare_summary_detects_faster_decode(self) -> None:
        compare = build_compare_session(
            FIXTURES / "qwen3_1p7b_run_full.json",
            FIXTURES / "qwen3_1p7b_run_compare.json",
            FIXTURES / "qwen3_1p7b_run_full.stderr",
            FIXTURES / "qwen3_1p7b_run_compare.stderr",
        )
        self.assertEqual(compare_summary(compare), "faster decode")

        rendered = render_compare(compare)
        self.assertIn("Compare Overview", rendered)
        self.assertIn("Performance Delta", rendered)
        self.assertIn("Per-Token Summary", rendered)
        self.assertIn("Artifact Differences", rendered)
        self.assertIn("faster decode", rendered)
        self.assertIn("decode_steady_state", rendered)

    def test_degraded_compare_when_one_side_lacks_stderr(self) -> None:
        compare = build_compare_session(
            FIXTURES / "qwen3_1p7b_run_full.json",
            FIXTURES / "qwen3_1p7b_run_compare.json",
            FIXTURES / "qwen3_1p7b_run_full.stderr",
            None,
        )
        issues = compare_issues(compare)
        rendered = render_compare(compare)
        issue_render = render_issues(compare)

        self.assertEqual(compare.right.derived.completeness_label, "partial")
        self.assertTrue(any("stderr not provided" in issue.message for issue in issues))
        self.assertIn("unavailable", rendered)
        self.assertIn("runs have different completeness levels", issue_render)

    def test_compare_and_issue_surfaces_do_not_leak_retained_content(self) -> None:
        compare = build_compare_session(
            FIXTURES / "qwen3_1p7b_run_full.json",
            FIXTURES / "qwen3_1p7b_run_compare.json",
            FIXTURES / "qwen3_1p7b_run_full.stderr",
            FIXTURES / "qwen3_1p7b_run_compare.stderr",
        )
        rendered = render_compare(compare) + "\n" + render_issues(compare)
        for forbidden in RETENTION_FORBIDDEN_SUBSTRINGS:
            self.assertNotIn(forbidden, rendered)


class IssuePanelTest(unittest.TestCase):
    def test_single_issue_panel_explains_partial_data(self) -> None:
        session = build_session(FIXTURES / "qwen3_1p7b_run.json", None)
        issues = single_issues(session)
        rendered = render_issues(session)

        self.assertTrue(any("stderr not provided" in issue.message for issue in issues))
        self.assertIn("Source Availability", rendered)
        self.assertIn("Parser completeness", rendered)
        self.assertIn("Unavailable data", rendered)
        self.assertIn("Issue count", rendered)


class CompareAndRedactionAppWiringTest(unittest.TestCase):
    def test_single_and_compare_tabs_and_branding(self) -> None:
        single_ids = [tab_id for tab_id, _title, _factory in _TAB_FACTORIES]
        compare_ids = [tab_id for tab_id, _title, _factory in _COMPARE_TAB_FACTORIES]
        self.assertEqual(single_ids, ["overview", "perf", "per_token", "artifact", "raw", "issues"])
        self.assertEqual(compare_ids, ["overview", "issues"])
        self.assertIn("LIS Inspect", LisInspectApp.TITLE)
        self.assertIn("post-execution viewer", LisInspectApp.SUB_TITLE)


if __name__ == "__main__":
    unittest.main()
