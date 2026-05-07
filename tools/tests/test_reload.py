"""Regression tests for the reload action (Ctrl+r / key 'r')."""

from __future__ import annotations

import unittest
from pathlib import Path

from textual.css.query import NoMatches
from textual.widgets import TabbedContent, TabPane

from lis_inspect.app import LisInspectApp
from lis_inspect.cli import build_session

_FIXTURES = Path(__file__).resolve().parents[1] / "test_fixtures"
_JSON = _FIXTURES / "qwen3_1p7b_run.json"
_STDERR = _FIXTURES / "qwen3_1p7b_run.stderr"
_COMPARE_JSON_1 = _FIXTURES / "qwen3_1p7b_run.json"
_COMPARE_JSON_2 = _FIXTURES / "qwen3_1p7b_run_compare.json"
_COMPARE_STDERR_2 = _FIXTURES / "qwen3_1p7b_run_compare.stderr"


class ReloadSingleRunTest(unittest.IsolatedAsyncioTestCase):
    async def _make_app(self, report_json, stderr_log=None):
        session = build_session(report_json, stderr_log)
        app = LisInspectApp(session, report_json=report_json, stderr_log=stderr_log)
        return app

    async def test_reload_does_not_crash(self) -> None:
        app = await self._make_app(_JSON, _STDERR)
        async with app.run_test() as pilot:
            await pilot.pause()
            # Ensure at least one tab pane exists
            tabbed = app.query_one(TabbedContent)
            self.assertIsNotNone(tabbed.query_one(TabPane))
            await pilot.press("r")
            await pilot.pause()

    async def test_reload_preserves_active_tab(self) -> None:
        app = await self._make_app(_JSON, _STDERR)
        async with app.run_test() as pilot:
            await pilot.pause()
            tabbed = app.query_one(TabbedContent)
            self.assertEqual(tabbed.active, "overview")
            # Switch to a different tab via direct assignment (key bindings
            # may not change the tab bar in run_test mode).
            tabbed.active = "perf"
            await pilot.pause()
            self.assertEqual(tabbed.active, "perf")
            await pilot.press("r")
            await pilot.pause()
            self.assertEqual(tabbed.active, "perf")

    async def test_reload_refreshes_pane_content(self) -> None:
        app = await self._make_app(_JSON, _STDERR)
        async with app.run_test() as pilot:
            await pilot.pause()
            tabbed = app.query_one(TabbedContent)
            # Before reload, capture the Overview view widget identity
            overview_pane = tabbed.get_pane("overview")
            old_children = list(overview_pane.children)
            self.assertTrue(old_children)
            await pilot.press("r")
            await pilot.pause()
            overview_pane2 = tabbed.get_pane("overview")
            new_children = list(overview_pane2.children)
            self.assertTrue(new_children)
            # Content should have been remounted (old widgets removed, new ones added).
            # The simplest safe assertion is that the child list is not identical.
            self.assertNotEqual(
                [id(c) for c in old_children],
                [id(c) for c in new_children],
            )

    async def test_reload_without_report_json_shows_notification(self) -> None:
        app = await self._make_app(_JSON, _STDERR)
        # Clear the report path so reload refuses to proceed
        app._report_json = None
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("r")
            await pilot.pause()
            # No crash, and TabbedContent should still be intact
            tabbed = app.query_one(TabbedContent)
            self.assertIsNotNone(tabbed.query_one(TabPane))


class ReloadCompareModeTest(unittest.IsolatedAsyncioTestCase):
    async def test_reload_does_not_crash_in_compare_mode(self) -> None:
        from lis_inspect.cli import build_compare_session

        session = build_compare_session(_COMPARE_JSON_1, _COMPARE_JSON_2, None, _COMPARE_STDERR_2)
        app = LisInspectApp(
            session,
            report_json=(_COMPARE_JSON_1, _COMPARE_JSON_2),
            stderr_log=(None, _COMPARE_STDERR_2),
        )
        async with app.run_test() as pilot:
            await pilot.pause()
            tabbed = app.query_one(TabbedContent)
            self.assertIn(tabbed.active, ("overview", "issues"))
            await pilot.press("r")
            await pilot.pause()
            self.assertIn(tabbed.active, ("overview", "issues"))


if __name__ == "__main__":
    unittest.main()
