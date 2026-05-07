"""Overlay layer polish for LIS Inspect.

Provides unified modal treatment for the command palette and theme picker.
All overlays visually recede the background, center a bounded panel,
and present input + results as one coherent unit.
"""

from __future__ import annotations

from textual.command import CommandPalette
from textual.containers import Horizontal, Vertical
from textual.theme import ThemeProvider

_MAX_PANEL_WIDTH = 80
_SIDE_MARGIN = 4


def _overlay_css(widget_name: str) -> str:
    return f"""
    {widget_name} {{
        background: $background 85%;
        align-horizontal: center;
    }}

    {widget_name} > Vertical#--container {{
        display: none;
    }}

    {widget_name}.-ready > Vertical#--container {{
        display: block;
        margin-top: 1;
        margin-bottom: 2;
        margin-left: {_SIDE_MARGIN};
        margin-right: {_SIDE_MARGIN};
        min-width: 40;
        max-width: {_MAX_PANEL_WIDTH};
        border: solid $border;
        padding: 0 1;
        overflow-y: hidden;
    }}

    {widget_name} > Vertical#--container > Horizontal#--input {{
        height: auto;
        border: none;
        visibility: visible;
        padding: 1 0;
    }}

    {widget_name} > Vertical#--container > Vertical#--results {{
        border-top: solid $border;
        height: auto;
        max-height: 50vh;
        padding-bottom: 0;
    }}

    {widget_name} CommandList {{
        border: none;
        visibility: visible;
        height: auto;
        max-height: 45vh;
        padding: 0 0;
    }}

    {widget_name} CommandList.--visible {{
        visibility: visible;
    }}

    {widget_name} > .command-palette--help-text {{
        color: $text-muted 80%;
        text-style: dim;
    }}

    {widget_name} Static.overlay-footer {{
        color: $text-muted;
        text-style: dim;
        height: auto;
        padding: 0 0;
    }}
    """


class LisCommandPalette(CommandPalette):
    """Command palette with stronger modal visual separation."""

    DEFAULT_CSS = _overlay_css("LisCommandPalette")

    def __init__(self) -> None:
        from textual.command import CommandInput

        super().__init__(
            id="--lis-command-palette",
            placeholder="Search commands…",
            classes="--lis-command-palette",
        )

    def compose(self):
        from textual.command import (
            CommandInput,
            CommandList,
            LoadingIndicator,
            SearchIcon,
        )
        from textual.widgets import Button, Static

        with Vertical(id="--container"):
            with Horizontal(id="--input"):
                yield SearchIcon()
                yield CommandInput(
                    placeholder=self._placeholder,
                    select_on_focus=False,
                )
                if not self.run_on_select:
                    yield Button("\u25b6")
            with Vertical(id="--results"):
                yield CommandList()
                yield LoadingIndicator()
            yield Static(
                "   \u2191/\u2193 choose  \u2022  enter run  \u2022  esc close",
                classes="overlay-footer",
            )


class LisThemePicker(CommandPalette):
    """Theme picker built like a real modal, not a detached search.

    Reuses the same panel language as ``LisCommandPalette`` so the two
    overlays feel like a single system.
    """

    DEFAULT_CSS = _overlay_css("LisThemePicker")

    def __init__(self) -> None:
        super().__init__(
            id="--lis-theme-picker",
            placeholder="Search themes…",
            classes="--lis-theme-picker",
            providers=[ThemeProvider],
        )

    def compose(self):
        from textual.command import (
            CommandInput,
            CommandList,
            LoadingIndicator,
            SearchIcon,
        )
        from textual.widgets import Button, Static

        with Vertical(id="--container"):
            with Horizontal(id="--input"):
                yield SearchIcon()
                yield CommandInput(
                    placeholder=self._placeholder,
                    select_on_focus=False,
                )
                if not self.run_on_select:
                    yield Button("\u25b6")
            with Vertical(id="--results"):
                yield CommandList()
                yield LoadingIndicator()
            yield Static(
                "   \u2191/\u2193 choose  \u2022  enter apply  \u2022  esc close",
                classes="overlay-footer",
            )
