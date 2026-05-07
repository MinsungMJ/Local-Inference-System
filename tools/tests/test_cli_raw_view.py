"""Model-level tests for parser warnings and Raw-tab wiring."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lis_inspect.app import _TAB_FACTORIES, LisInspectApp
from lis_inspect.cli import build_session


FIXTURES = Path(__file__).resolve().parent.parent / "test_fixtures"


class RawViewSessionTest(unittest.TestCase):
    def test_missing_stderr_warning_is_session_level_not_path_leaking(self) -> None:
        session = build_session(FIXTURES / "qwen3_1p7b_run.json", None)
        self.assertEqual(session.derived.artifact_completeness, "json_only")
        self.assertIn("stderr log not provided", session.warnings)
        self.assertFalse(session.source.stderr.provided)

    def test_supplied_missing_stderr_marks_degraded(self) -> None:
        session = build_session(
            FIXTURES / "qwen3_1p7b_run.json",
            Path("/tmp/lis_inspect_missing.stderr"),
        )
        self.assertEqual(session.derived.artifact_completeness, "degraded")
        self.assertEqual(session.warnings, ["stderr-log not found"])
        self.assertTrue(session.source.stderr.provided)
        self.assertFalse(session.source.stderr.readable)

    def test_partial_stderr_sections_generate_compact_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stderr_path = Path(tmpdir) / "partial.stderr"
            stderr_path.write_text(
                "lis: perf-stage tag=t name=prefill ns=10 ms=1.0 tokens=2\n",
                encoding="utf-8",
            )
            session = build_session(FIXTURES / "qwen3_1p7b_run.json", stderr_path)

        self.assertEqual(session.derived.artifact_completeness, "degraded")
        self.assertIn("perf-summary section missing", session.warnings)
        self.assertTrue(session.source.stderr.readable)
        self.assertTrue(session.source.stderr.parsed)


class RawViewAppWiringTest(unittest.TestCase):
    def test_raw_tab_is_in_tab_factories_and_key_binding(self) -> None:
        tab_ids = [tab_id for tab_id, _title, _factory in _TAB_FACTORIES]
        self.assertEqual(tab_ids, ["overview", "perf", "per_token", "artifact", "raw", "issues"])

        bindings = {(binding[0], binding[1]) for binding in LisInspectApp.BINDINGS}
        self.assertIn(("5", "show_tab('raw')"), bindings)
        self.assertIn(("6", "show_tab('issues')"), bindings)
        self.assertIn(("r", "reload"), bindings)


if __name__ == "__main__":
    unittest.main()
