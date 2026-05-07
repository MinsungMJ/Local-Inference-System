"""Focused tests for overlay render and activation paths."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

from textual.command import CommandPalette

from lis_inspect.cli import build_session
from lis_inspect.overlay import LisCommandPalette, LisThemePicker, _overlay_css

_FIXTURE_JSON = Path(__file__).resolve().parents[1] / "test_fixtures" / "qwen3_1p7b_run.json"
_FIXTURE_STDERR = Path(__file__).resolve().parents[1] / "test_fixtures" / "qwen3_1p7b_run.stderr"


class OverlayImportTest(unittest.TestCase):
    def test_overlay_imports(self) -> None:
        self.assertTrue(callable(_overlay_css))

    def test_palette_classes_are_subclasses(self) -> None:
        self.assertTrue(issubclass(LisCommandPalette, CommandPalette))
        self.assertTrue(issubclass(LisThemePicker, CommandPalette))


class OverlayCSSTest(unittest.TestCase):
    def test_css_contains_panel_rules(self) -> None:
        css = _overlay_css("LisCommandPalette")
        self.assertIn("background: $background 85%", css)
        self.assertIn("border: solid $border", css)
        self.assertIn("max-width: 80", css)
        self.assertIn("margin-top: 1", css)
        self.assertIn("CommandList", css)
        self.assertIn("overlay-footer", css)

    def test_css_is_generated_for_both_widgets(self) -> None:
        self.assertIn("LisCommandPalette", LisCommandPalette.DEFAULT_CSS)
        self.assertIn("LisThemePicker", LisThemePicker.DEFAULT_CSS)

    def test_css_contains_dimmed_help_text(self) -> None:
        css = _overlay_css("LisCommandPalette")
        self.assertIn("text-muted 80%", css)
        self.assertIn("text-style: dim", css)


class OverlayConstructionTest(unittest.TestCase):
    def test_theme_picker_uses_correct_placeholder(self) -> None:
        tp = LisThemePicker()
        self.assertIn("theme", tp._placeholder.lower())

    def test_command_palette_uses_correct_placeholder(self) -> None:
        cp = LisCommandPalette()
        self.assertIn("command", cp._placeholder.lower())

    def test_theme_picker_has_lis_id_and_class(self) -> None:
        tp = LisThemePicker()
        self.assertIn("--lis-theme-picker", tp.id)
        self.assertIn("--lis-theme-picker", tp.classes)

    def test_command_palette_has_lis_id_and_class(self) -> None:
        cp = LisCommandPalette()
        self.assertIn("--lis-command-palette", cp.id)
        self.assertIn("--lis-command-palette", cp.classes)


class OverlayActivationTest(unittest.IsolatedAsyncioTestCase):
    async def _open_palette(self, app, palette_cls):
        async with app.run_test() as pilot:
            await pilot.pause()
            app.push_screen(palette_cls())
            await pilot.pause()
            screen = app.screen
            self.assertTrue(screen.has_class("--textual-command-palette"))
            container = screen.query_one("#--container")
            self.assertIsNotNone(container)
            footer = screen.query_one(".overlay-footer")
            self.assertIsNotNone(footer)
            screen.dismiss()
            await pilot.pause()

    async def test_command_palette_opens(self) -> None:
        from lis_inspect.app import LisInspectApp

        session = build_session(_FIXTURE_JSON, _FIXTURE_STDERR)
        app = LisInspectApp(session)
        await self._open_palette(app, LisCommandPalette)

    async def test_theme_picker_opens(self) -> None:
        from lis_inspect.app import LisInspectApp

        session = build_session(_FIXTURE_JSON, _FIXTURE_STDERR)
        app = LisInspectApp(session)
        await self._open_palette(app, LisThemePicker)

    async def test_app_overrides_action_command_palette(self) -> None:
        from lis_inspect.app import LisInspectApp

        session = build_session(_FIXTURE_JSON, _FIXTURE_STDERR)
        app = LisInspectApp(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_command_palette()
            await pilot.pause()
            self.assertTrue(app.screen.has_class("--textual-command-palette"))
            self.assertIn("--lis-command-palette", app.screen.id)
            app.screen.dismiss()
            await pilot.pause()

    async def test_app_overrides_action_change_theme(self) -> None:
        from lis_inspect.app import LisInspectApp

        session = build_session(_FIXTURE_JSON, _FIXTURE_STDERR)
        app = LisInspectApp(session)
        async with app.run_test() as pilot:
            await pilot.pause()
            app.action_change_theme()
            await pilot.pause()
            self.assertTrue(app.screen.has_class("--textual-command-palette"))
            self.assertIn("--lis-theme-picker", app.screen.id)
            app.screen.dismiss()
            await pilot.pause()


if __name__ == "__main__":
    unittest.main()
