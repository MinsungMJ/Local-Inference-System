"""Perf tab — answers 'where did time go?'"""

from __future__ import annotations

from typing import Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..derive import ordered_stages
from ..models import InspectSession
from ..palette import style_for

BAR_WIDTH = 32  # cells reserved for the visual bar


def _bar(ms: float, max_ms: float, width: int = BAR_WIDTH) -> str:
    if max_ms <= 0:
        return " " * width
    filled = int(round((ms / max_ms) * width))
    filled = max(0, min(width, filled))
    return "█" * filled + " " * (width - filled)


def _format_summary(session: InspectSession) -> list[str]:
    summary = session.perf.summary
    if summary is None:
        json_summary = session.artifact.perf_summary_json
        if json_summary is None:
            return [
                f"[{style_for('warning')}]No perf summary available — "
                "run with --perf and capture stderr, or include perf_summary in JSON.[/]"
            ]
        return [
            _summary_line("TTFT", _maybe_float(json_summary.get("ttft_ms")), "ms"),
            _summary_line("Mean ITL", _maybe_float(json_summary.get("itl_ms")), "ms"),
            _summary_line("TPS steady", _maybe_float(json_summary.get("tps_steady")), ""),
            _summary_line("TPS end-to-end", _maybe_float(json_summary.get("tps_end_to_end")), ""),
        ]
    return [
        _summary_line("TTFT", summary.ttft_ms, "ms"),
        _summary_line("Mean ITL", summary.itl_ms, "ms"),
        _summary_line("TPS steady", summary.tps_steady, ""),
        _summary_line("TPS end-to-end", summary.tps_end_to_end, ""),
    ]


def _summary_line(label: str, value: Optional[float], unit: str) -> str:
    if value is None:
        rendered = "—"
    elif unit:
        rendered = f"{value:.3f} {unit}"
    else:
        rendered = f"{value:.3f}"
    return (
        f"[{style_for('muted')}]{label:<18}[/] "
        f"[{style_for('perf')}]{rendered}[/]"
    )


def _maybe_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def render_perf(session: InspectSession) -> str:
    perf = session.perf
    derived = session.derived

    lines: list[str] = []
    lines.append(f"[{style_for('perf')}]Summary[/]")
    lines.extend(_format_summary(session))
    lines.append("")

    if not perf.has_stages:
        lines.append(
            f"[{style_for('warning')}]No perf-stage data parsed. "
            "Provide --stderr-log from a --perf run to see the stage breakdown.[/]"
        )
        return "\n".join(lines)

    stages = ordered_stages(perf)
    max_ms = max((ms for _, ms in stages), default=0.0)
    dominant = derived.dominant_stage

    lines.append(f"[{style_for('perf')}]Stage Breakdown[/]")
    if derived.run_profile is not None:
        lines.append(
            f"[{style_for('muted')}]Run profile:[/] "
            f"[{style_for('perf')}]{derived.run_profile}[/]"
        )
        lines.append("")

    label_width = max((len(name) for name, _ in stages), default=0)

    for name, ms in stages:
        bar = _bar(ms, max_ms)
        is_dominant = name == dominant
        bar_role = "perf" if is_dominant else "muted"
        name_role = "perf" if is_dominant else "neutral"
        marker = "▶" if is_dominant else " "
        lines.append(
            f"[{style_for(name_role)}]{marker} {name:<{label_width}}[/] "
            f"[{style_for(bar_role)}]{bar}[/] "
            f"[{style_for('perf' if is_dominant else 'neutral')}]{ms:>10.3f} ms[/]"
        )

    lines.append("")
    lines.append(f"[{style_for('muted')}]Numeric table[/]")
    for name, ms in stages:
        stage = perf.stages[name]
        lines.append(
            f"[{style_for('muted')}]  {name:<22}[/] "
            f"ns={stage.ns:<14} ms={ms:>10.3f}  tokens={stage.tokens}"
        )

    if perf.warnings:
        lines.append("")
        lines.append(f"[{style_for('warning')}]Parser warnings[/]")
        for warning in perf.warnings:
            lines.append(f"[{style_for('warning')}]• {warning}[/]")

    return "\n".join(lines)


class PerfView(Vertical):
    """Textual widget rendering the Perf tab."""

    DEFAULT_CSS = """
    PerfView {
        padding: 1 2;
        height: 1fr;
        overflow-y: auto;
    }
    """

    def __init__(self, session: InspectSession) -> None:
        super().__init__()
        self._session = session

    def compose(self) -> ComposeResult:
        yield Static(render_perf(self._session), markup=True)
