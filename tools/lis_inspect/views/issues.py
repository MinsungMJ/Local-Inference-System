"""Structured issue/completeness panel for single-run and compare modes."""

from __future__ import annotations

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..inspection import (
    compare_issues,
    completeness_role,
    run_interpretation,
    single_issues,
    unavailable_notes,
)
from ..models import CompareSession, InspectSession
from ..palette import style_for


def _kv(label: str, value: str, role: str = "neutral") -> str:
    return (
        f"[{style_for('muted')}]  {label:<24}[/] "
        f"[{style_for(role)}]{escape(value)}[/]"
    )


def _source_line(session: InspectSession, label: str) -> list[str]:
    lines = [f"[{style_for('focus')}]{label}[/]"]
    lines.append(
        _kv(
            "JSON source",
            "present" if session.source.json.readable else "absent",
            "success" if session.source.json.readable else "error",
        )
    )
    lines.append(
        _kv(
            "stderr source",
            "present" if session.source.stderr.readable else "absent",
            "success" if session.source.stderr.readable else "partial",
        )
    )
    lines.append(
        _kv(
            "Parser completeness",
            session.derived.completeness_label,
            completeness_role(session.derived.completeness_label),
        )
    )
    unavailable = unavailable_notes(session)
    lines.append(
        _kv(
            "Unavailable data",
            ", ".join(unavailable) if unavailable else "none",
            "partial" if unavailable else "success",
        )
    )
    return lines


def _interpretation_block(session: InspectSession, label: str) -> list[str]:
    lines = [f"[{style_for('identity')}]Interpretation: {escape(label)}[/]"]
    for name, value, role in run_interpretation(session):
        lines.append(_kv(name, value, role))
    return lines


def _issue_block(issues) -> list[str]:
    lines = [f"[{style_for('warning' if issues else 'identity')}]Issues[/]"]
    lines.append(_kv("Issue count", str(len(issues)), "warning" if issues else "success"))
    if not issues:
        lines.append(f"[{style_for('muted')}]  No inspection issues recorded.[/]")
        return lines
    for issue in issues:
        lines.append(
            f"[{style_for(issue.role)}]  • {escape(issue.source)}: "
            f"{escape(issue.message)}[/]"
        )
    return lines


def render_issues(session_or_compare: InspectSession | CompareSession) -> str:
    lines: list[str] = []
    if isinstance(session_or_compare, CompareSession):
        compare = session_or_compare
        issues = compare_issues(compare)
        lines.append(f"[{style_for('identity')}]Source Availability[/]")
        lines.extend(_source_line(compare.left, "A"))
        lines.append("")
        lines.extend(_source_line(compare.right, "B"))
        lines.append("")
        lines.extend(_interpretation_block(compare.left, "A"))
        lines.append("")
        lines.extend(_interpretation_block(compare.right, "B"))
        lines.append("")
        lines.extend(_issue_block(issues))
    else:
        session = session_or_compare
        issues = single_issues(session)
        lines.append(f"[{style_for('identity')}]Source Availability[/]")
        lines.extend(_source_line(session, "Run"))
        lines.append("")
        lines.extend(_interpretation_block(session, "Run"))
        lines.append("")
        lines.extend(_issue_block(issues))

    lines.append("")
    lines.append(f"[{style_for('muted')}]Notes[/]")
    lines.append(
        f"[{style_for('muted')}]  • Issues explain inspection completeness only; "
        "they are not telemetry events.[/]"
    )
    lines.append(
        f"[{style_for('muted')}]  • Missing stderr limits perf/per-token evidence; "
        "JSON remains canonical.[/]"
    )
    return "\n".join(lines)


class IssueView(Vertical):
    """Textual widget rendering the Issues tab."""

    DEFAULT_CSS = """
    IssueView {
        padding: 1 2;
        height: 1fr;
        overflow-y: auto;
    }
    """

    def __init__(self, session_or_compare: InspectSession | CompareSession) -> None:
        super().__init__()
        self._session_or_compare = session_or_compare

    def compose(self) -> ComposeResult:
        yield Static(render_issues(self._session_or_compare), markup=True)
