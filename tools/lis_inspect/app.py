"""Textual `App` shell for single-run and two-run LIS inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from textual.app import App, ComposeResult
from textual.command import CommandPalette
from textual.css.query import NoMatches
from textual.widgets import Footer, Header, TabbedContent, TabPane

from .models import CompareSession, InspectSession
from .overlay import LisCommandPalette, LisThemePicker
from .views.artifact import ArtifactView
from .views.compare import CompareView
from .views.issues import IssueView
from .views.overview import OverviewView
from .views.per_token import PerTokenView
from .views.perf import PerfView
from .views.raw import RawView


# Mapping from tab id to its view factory. Used by both `compose` and `reload`
# so the wire-up of tabs lives in one place.
_TAB_FACTORIES = (
    ("overview", "Overview", OverviewView),
    ("perf", "Perf", PerfView),
    ("per_token", "Per-Token", PerTokenView),
    ("artifact", "Artifact", ArtifactView),
    ("raw", "Raw", RawView),
    ("issues", "Issues", IssueView),
)

_COMPARE_TAB_FACTORIES = (
    ("overview", "Compare", CompareView),
    ("issues", "Issues", IssueView),
)


class LisInspectApp(App):
    """Bounded post-execution TUI inspector."""

    TITLE = "LIS Inspect"
    SUB_TITLE = "post-execution viewer"

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "reload", "Reload"),
        ("1", "show_tab('overview')", "Overview"),
        ("2", "show_tab('perf')", "Perf"),
        ("3", "show_tab('per_token')", "Per-Token"),
        ("4", "show_tab('artifact')", "Artifact"),
        ("5", "show_tab('raw')", "Raw"),
        ("6", "show_tab('issues')", "Issues"),
    ]

    def __init__(
        self,
        session: InspectSession | CompareSession,
        report_json: Optional[Path | tuple[Path, Path]] = None,
        stderr_log: Optional[Path | tuple[Optional[Path], Optional[Path]]] = None,
    ) -> None:
        super().__init__()
        self._session = session
        self._report_json = report_json
        self._stderr_log = stderr_log

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        with TabbedContent(initial="overview"):
            for tab_id, title, factory in self._tab_factories():
                with TabPane(title, id=tab_id):
                    yield factory(self._session)
        yield Footer()

    def action_show_tab(self, tab_id: str) -> None:
        tabbed = self.query_one(TabbedContent)
        try:
            tabbed.get_child_by_id(tab_id)
        except NoMatches:
            self.notify("Tab unavailable in this mode.", severity="warning")
            return
        tabbed.active = tab_id

    def action_command_palette(self) -> None:
        """Open the polished command palette instead of the stock one."""
        if self.use_command_palette and not CommandPalette.is_open(self):
            self.push_screen(LisCommandPalette())

    def action_change_theme(self) -> None:
        """Open the polished theme picker."""
        self.push_screen(LisThemePicker())

    def _tab_factories(self):
        if isinstance(self._session, CompareSession):
            return _COMPARE_TAB_FACTORIES
        return _TAB_FACTORIES

    def action_reload(self) -> None:
        if self._report_json is None:
            self.notify("Reload requires the original --report-json path.", severity="warning")
            return
        # Late import to keep app.py free of CLI-layer dependencies at import time.
        from .cli import build_compare_session, build_session
        from .parse_json import ArtifactParseError

        try:
            if isinstance(self._report_json, tuple):
                stderr = self._stderr_log if isinstance(self._stderr_log, tuple) else (None, None)
                self._session = build_compare_session(
                    self._report_json[0],
                    self._report_json[1],
                    stderr[0],
                    stderr[1],
                )
            else:
                stderr_path = self._stderr_log if isinstance(self._stderr_log, Path) else None
                self._session = build_session(self._report_json, stderr_path)
        except ArtifactParseError as exc:
            self.notify(f"Reload failed: {exc}", severity="error")
            return
        except Exception as exc:
            self.notify(f"Reload failed: {exc}", severity="error")
            return

        # Replace tab content in place, preserving the active tab.
        tabbed = self.query_one(TabbedContent)
        active = tabbed.active
        for pane in tabbed.query(TabPane):
            for child in list(pane.children):
                child.remove()
        for tab_id, _title, factory in self._tab_factories():
            try:
                pane = tabbed.get_pane(tab_id)
            except NoMatches:
                continue
            pane.mount(factory(self._session))
        tabbed.active = active
        self.notify("Reloaded.", severity="information")


def run_app(
    session: InspectSession | CompareSession,
    report_json: Optional[Path | tuple[Path, Path]] = None,
    stderr_log: Optional[Path | tuple[Optional[Path], Optional[Path]]] = None,
) -> int:
    LisInspectApp(session, report_json=report_json, stderr_log=stderr_log).run()
    return 0


__all__ = ["LisInspectApp", "run_app"]
