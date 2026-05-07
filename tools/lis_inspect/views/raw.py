"""Raw tab — bounded parser transparency without exposing retained text."""

from __future__ import annotations

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..models import InspectSession, SourcePreview
from ..palette import style_for

WARNING_PREVIEW_COUNT = 6


def _kv(label: str, value: str, role: str = "neutral") -> str:
    return (
        f"[{style_for('muted')}]  {label:<24}[/] "
        f"[{style_for(role)}]{escape(value)}[/]"
    )


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


def _present(source: SourcePreview) -> tuple[str, str]:
    if source.readable:
        return "present", "success"
    if source.provided:
        return "absent", "warning"
    return "absent", "muted"


def _format_source_availability(session: InspectSession) -> list[str]:
    json_present, json_role = _present(session.source.json)
    stderr_present, stderr_role = _present(session.source.stderr)
    lines = [f"[{style_for('identity')}]Source Availability[/]"]
    lines.append(_kv("JSON", json_present, json_role))
    lines.append(_kv("stderr", stderr_present, stderr_role))
    if session.source.stderr.provided and session.source.stderr.error:
        lines.append(_kv("stderr note", session.source.stderr.error, "warning"))
    elif not session.source.stderr.provided:
        lines.append(_kv("stderr note", "not provided", "muted"))
    return lines


def _format_parser_status(session: InspectSession) -> list[str]:
    perf = session.perf
    lines = [f"[{style_for('identity')}]Parser Status[/]"]
    lines.append(
        _kv(
            "JSON parsed",
            _yes_no(session.source.json.parsed),
            "success" if session.source.json.parsed else "error",
        )
    )
    lines.append(
        _kv(
            "perf-stage parsed",
            _yes_no(perf.has_stages),
            "success" if perf.has_stages else "warning",
        )
    )
    lines.append(
        _kv(
            "perf-summary parsed",
            _yes_no(perf.has_summary),
            "success" if perf.has_summary else "warning",
        )
    )
    lines.append(
        _kv(
            "per-token parsed",
            _yes_no(perf.has_per_token),
            "success" if perf.has_per_token else "muted",
        )
    )
    role = "warning" if session.derived.artifact_completeness == "degraded" else "artifact"
    lines.append(_kv("Completeness", session.derived.artifact_completeness, role))
    return lines


def _format_warnings(session: InspectSession) -> list[str]:
    count = len(session.warnings)
    role = "warning" if count else "success"
    lines = [f"[{style_for('warning' if count else 'identity')}]Warnings[/]"]
    lines.append(_kv("Count", str(count), role))
    if count == 0:
        lines.append(f"[{style_for('muted')}]  No parser warnings recorded.[/]")
        return lines
    for warning in session.warnings[:WARNING_PREVIEW_COUNT]:
        lines.append(f"[{style_for('warning')}]  • {escape(warning)}[/]")
    hidden = count - WARNING_PREVIEW_COUNT
    if hidden > 0:
        lines.append(f"[{style_for('muted')}]  … {hidden} more warning(s)[/]")
    return lines


def _format_preview(title: str, source: SourcePreview, empty_note: str) -> list[str]:
    lines = [f"[{style_for('artifact')}]{title}[/]"]
    if not source.readable:
        note = source.error or empty_note
        lines.append(f"[{style_for('muted')}]  {escape(note)}[/]")
        return lines
    if not source.preview:
        lines.append(f"[{style_for('muted')}]  (empty source)[/]")
        return lines
    lines.append(
        f"[{style_for('muted')}]  showing bounded preview from "
        f"{source.line_count} line(s)[/]"
    )
    lines.append("")
    lines.append(escape(source.preview))
    return lines


def render_raw(session: InspectSession) -> str:
    lines: list[str] = []
    lines.extend(_format_source_availability(session))
    lines.append("")
    lines.extend(_format_parser_status(session))
    lines.append("")
    lines.extend(_format_warnings(session))
    lines.append("")
    lines.extend(
        _format_preview("JSON Preview", session.source.json, "JSON source unavailable.")
    )
    lines.append("")
    lines.extend(
        _format_preview(
            "stderr Preview",
            session.source.stderr,
            "stderr source unavailable.",
        )
    )
    lines.append("")
    lines.append(f"[{style_for('muted')}]Notes[/]")
    lines.append(
        f"[{style_for('muted')}]  • Previews are bounded and redact raw prompt text, "
        "generated text, and absolute paths.[/]"
    )
    lines.append(
        f"[{style_for('muted')}]  • JSON remains canonical; stderr is "
        "supplementary parsed evidence.[/]"
    )
    return "\n".join(lines)


class RawView(Vertical):
    """Textual widget rendering the Raw tab."""

    DEFAULT_CSS = """
    RawView {
        padding: 1 2;
        height: 1fr;
        overflow-y: auto;
    }
    """

    def __init__(self, session: InspectSession) -> None:
        super().__init__()
        self._session = session

    def compose(self) -> ComposeResult:
        yield Static(render_raw(self._session), markup=True)
