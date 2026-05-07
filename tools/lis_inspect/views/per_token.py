"""Per-Token tab — answers 'how did decode latency vary by token?'

Renders only when per-token data is present. Shows distribution stats
(count/min/max/mean/p50/p90/p95/p99), a compact per-step trend bar, and the
top slowest steps. When per-token data is absent, renders an explicit
degraded message instead of pretending data exists.
"""

from __future__ import annotations

from typing import Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..models import InspectSession, PercentileStats, PerfPerToken
from ..palette import style_for

# Eight-level block ramp for compact per-step trend.
TREND_RAMP: tuple[str, ...] = ("▁", "▂", "▃", "▄", "▅", "▆", "▇", "█")


def _stat_line(label: str, value: Optional[float], unit: str = "ms") -> str:
    if value is None:
        rendered = "—"
    elif unit:
        rendered = f"{value:.3f} {unit}"
    else:
        rendered = f"{value:.3f}"
    return (
        f"[{style_for('muted')}]  {label:<10}[/] "
        f"[{style_for('perf')}]{rendered}[/]"
    )


def _count_line(count: int) -> str:
    return (
        f"[{style_for('muted')}]  {'count':<10}[/] "
        f"[{style_for('perf')}]{count}[/]"
    )


def _trend_bar(per_token: list[PerfPerToken]) -> str:
    """Map each step's ms to a single block character, from min..max scale."""
    if not per_token:
        return ""
    values = [entry.ms for entry in per_token]
    lo = min(values)
    hi = max(values)
    span = hi - lo
    if span <= 0:
        # All identical — pick the middle ramp character.
        return TREND_RAMP[len(TREND_RAMP) // 2] * len(values)
    levels = len(TREND_RAMP)
    chars: list[str] = []
    for value in values:
        idx = int(round(((value - lo) / span) * (levels - 1)))
        idx = max(0, min(levels - 1, idx))
        chars.append(TREND_RAMP[idx])
    return "".join(chars)


def _format_stats(stats: PercentileStats) -> list[str]:
    return [
        _count_line(stats.count),
        _stat_line("min", stats.min_ms),
        _stat_line("max", stats.max_ms),
        _stat_line("mean", stats.mean_ms),
        _stat_line("p50", stats.p50_ms),
        _stat_line("p90", stats.p90_ms),
        _stat_line("p95", stats.p95_ms),
        _stat_line("p99", stats.p99_ms),
    ]


def _format_top_slow(steps: list[PerfPerToken], total: int) -> list[str]:
    if not steps:
        return []
    lines: list[str] = []
    lines.append(f"[{style_for('perf')}]Top slow steps[/]")
    rank_width = len(str(len(steps)))
    step_width = max(len(str(entry.step)) for entry in steps)
    for rank, entry in enumerate(steps, start=1):
        marker = f"#{rank:<{rank_width}}"
        lines.append(
            f"[{style_for('muted')}]  {marker}[/] "
            f"[{style_for('neutral')}]step={entry.step:<{step_width}}[/] "
            f"[{style_for('perf')}]{entry.ms:>10.3f} ms[/]"
        )
    if len(steps) < total:
        lines.append(
            f"[{style_for('muted')}]  (showing top {len(steps)} of {total})[/]"
        )
    return lines


def render_per_token(session: InspectSession) -> str:
    perf = session.perf
    derived = session.derived

    if not perf.has_per_token:
        # Honest degraded message.
        return (
            f"[{style_for('warning')}]Per-token data unavailable.[/]\n"
            f"[{style_for('muted')}]"
            "Re-run LIS with --perf-per-token and capture stderr to populate "
            "this tab. The JSON artifact alone does not carry per-step latency."
            "[/]"
        )

    stats = derived.per_token_stats
    lines: list[str] = []
    lines.append(f"[{style_for('perf')}]Distribution[/]")
    if stats is not None:
        lines.extend(_format_stats(stats))
    else:  # defensive — has_per_token is True but stats unavailable
        lines.append(
            f"[{style_for('warning')}]Distribution stats unavailable "
            "despite per-token entries being present.[/]"
        )

    lines.append("")
    lines.append(f"[{style_for('perf')}]Per-step trend[/]")
    trend = _trend_bar(perf.per_token)
    legend = f"[{style_for('muted')}](▁ smallest, █ largest)[/]"
    lines.append(f"[{style_for('muted')}]  steps 0..{len(perf.per_token) - 1}[/]  {legend}")
    lines.append(f"  [{style_for('perf')}]{trend}[/]")

    lines.append("")
    lines.extend(_format_top_slow(derived.top_slow_steps, len(perf.per_token)))
    lines.append("")
    lines.append(f"[{style_for('identity')}]Interpretation[/]")
    lines.append(
        f"[{style_for('muted')}]  {'warmup effect':<14}[/] "
        f"[{style_for('perf' if derived.warmup_effect not in (None, 'none') else 'muted')}]"
        f"{derived.warmup_effect if derived.warmup_effect is not None else 'unknown'}[/]"
    )
    lines.append(
        f"[{style_for('muted')}]  {'tail spread':<14}[/] "
        f"[{style_for('warning' if derived.tail_spread == 'high' else 'perf' if derived.tail_spread else 'muted')}]"
        f"{derived.tail_spread if derived.tail_spread is not None else 'unknown'}[/]"
    )

    if perf.warnings:
        lines.append("")
        lines.append(f"[{style_for('warning')}]Parser warnings[/]")
        for warning in perf.warnings:
            lines.append(f"[{style_for('warning')}]• {warning}[/]")

    return "\n".join(lines)


class PerTokenView(Vertical):
    """Textual widget rendering the Per-Token tab."""

    DEFAULT_CSS = """
    PerTokenView {
        padding: 1 2;
        height: 1fr;
        overflow-y: auto;
    }
    """

    def __init__(self, session: InspectSession) -> None:
        super().__init__()
        self._session = session

    def compose(self) -> ComposeResult:
        yield Static(render_per_token(self._session), markup=True)
