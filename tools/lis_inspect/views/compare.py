"""Two-run compare view."""

from __future__ import annotations

from typing import Optional

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..inspection import (
    artifact_diff_rows,
    changed_runtime_fields,
    compare_summary,
    completeness_role,
    metric_delta,
    metric_value,
    run_interpretation,
    stage_delta_rows,
)
from ..models import CompareSession, InspectSession
from ..palette import status_role, style_for


def _kv(label: str, left: str, right: str, role: str = "neutral") -> str:
    return (
        f"[{style_for('muted')}]  {label:<24}[/] "
        f"A=[{style_for(role)}]{escape(left)}[/]  "
        f"B=[{style_for(role)}]{escape(right)}[/]"
    )


def _delta_line(
    label: str,
    left: Optional[float],
    right: Optional[float],
    unit: str,
    higher_is_better: bool = False,
) -> str:
    delta, role = metric_delta(left, right, higher_is_better=higher_is_better)
    left_s = _fmt_float(left, unit)
    right_s = _fmt_float(right, unit)
    return (
        f"[{style_for('muted')}]  {label:<24}[/] "
        f"A={left_s:<14} B={right_s:<14} "
        f"[{style_for(role)}]Δ {escape(delta)}[/]"
    )


def _fmt_float(value: Optional[float], unit: str = "") -> str:
    if value is None:
        return "—"
    suffix = f" {unit}" if unit else ""
    return f"{value:.3f}{suffix}"


def _fmt(value) -> str:
    if value is None:
        return "—"
    return str(value)


def _runtime(session: InspectSession, key: str) -> str:
    return _fmt(session.artifact.runtime_settings.get(key))


def _status_line(compare: CompareSession) -> list[str]:
    left = compare.left.artifact
    right = compare.right.artifact
    lines = [f"[{style_for('identity')}]Compare Overview[/]"]
    lines.append(
        _kv(
            "Status",
            _fmt(left.execution_status),
            _fmt(right.execution_status),
            status_role(right.execution_status),
        )
    )
    lines.append(_kv("Stop reason", _fmt(left.stop_reason), _fmt(right.stop_reason)))
    lines.append(
        _kv("Model family", _fmt(left.model_family), _fmt(right.model_family), "identity")
    )
    lines.append(_kv("Backend", _fmt(left.backend), _fmt(right.backend), "identity"))
    lines.append(
        _kv("Context", _runtime(compare.left, "context"), _runtime(compare.right, "context"))
    )
    lines.append(
        _kv("Threads", _runtime(compare.left, "threads"), _runtime(compare.right, "threads"))
    )
    lines.append(
        _kv(
            "Completeness",
            compare.left.derived.completeness_label,
            compare.right.derived.completeness_label,
            completeness_role(compare.right.derived.completeness_label),
        )
    )
    lines.append(
        f"[{style_for('focus')}]  Summary                 "
        f"{escape(compare_summary(compare))}[/]"
    )
    return lines


def _interpretation(compare: CompareSession) -> list[str]:
    lines = [f"[{style_for('identity')}]Interpretation[/]"]
    left_rows = {
        name: (value, role) for name, value, role in run_interpretation(compare.left)
    }
    right_rows = {
        name: (value, role) for name, value, role in run_interpretation(compare.right)
    }
    for name in ("Run profile", "Warmup effect", "Tail spread"):
        left_value = left_rows.get(name, ("unknown", "muted"))[0]
        right_value, role = right_rows.get(name, ("unknown", "muted"))
        marker = "changed" if left_value != right_value else "same"
        lines.append(
            f"[{style_for('muted')}]  {name:<24}[/] "
            f"A={escape(left_value):<14} "
            f"B=[{style_for(role)}]{escape(right_value)}[/] "
            f"[{style_for('partial' if marker == 'changed' else 'muted')}]({marker})[/]"
        )
    return lines


def _performance(compare: CompareSession) -> list[str]:
    lines = [f"[{style_for('perf')}]Performance Delta[/]"]
    for label, key, unit, higher in (
        ("TTFT", "ttft_ms", "ms", False),
        ("Mean ITL", "itl_ms", "ms", False),
        ("TPS steady", "tps_steady", "", True),
        ("TPS end-to-end", "tps_end_to_end", "", True),
    ):
        lines.append(
            _delta_line(
                label,
                metric_value(compare.left, key),
                metric_value(compare.right, key),
                unit,
                higher_is_better=higher,
            )
        )
    lines.append("")
    lines.append(f"[{style_for('perf')}]Stage Timing Deltas[/]")
    for name, left, right, delta, role in stage_delta_rows(compare):
        if left is None and right is None:
            continue
        lines.append(
            f"[{style_for('muted')}]  {name:<24}[/] "
            f"A={_fmt_float(left, 'ms'):<14} B={_fmt_float(right, 'ms'):<14} "
            f"[{style_for(role)}]Δ {escape(delta)}[/]"
        )
    return lines


def _per_token(compare: CompareSession) -> list[str]:
    lines = [f"[{style_for('perf')}]Per-Token Summary[/]"]
    left_stats = compare.left.derived.per_token_stats
    right_stats = compare.right.derived.per_token_stats
    rows = (
        (
            "Step count",
            left_stats.count if left_stats else None,
            right_stats.count if right_stats else None,
            "",
            False,
        ),
        (
            "p50",
            left_stats.p50_ms if left_stats else None,
            right_stats.p50_ms if right_stats else None,
            "ms",
            False,
        ),
        (
            "p90",
            left_stats.p90_ms if left_stats else None,
            right_stats.p90_ms if right_stats else None,
            "ms",
            False,
        ),
        (
            "p95",
            left_stats.p95_ms if left_stats else None,
            right_stats.p95_ms if right_stats else None,
            "ms",
            False,
        ),
        (
            "p99",
            left_stats.p99_ms if left_stats else None,
            right_stats.p99_ms if right_stats else None,
            "ms",
            False,
        ),
    )
    for label, left, right, unit, higher in rows:
        lines.append(_delta_line(label, left, right, unit, higher_is_better=higher))
    lines.append(
        _kv(
            "Top slow step",
            _top_slow(compare.left),
            _top_slow(compare.right),
            "perf" if compare.right.derived.top_slow_steps else "muted",
        )
    )
    return lines


def _artifact_runtime(compare: CompareSession) -> list[str]:
    lines = [f"[{style_for('artifact')}]Runtime Differences[/]"]
    for label, left, right, role in changed_runtime_fields(compare):
        lines.append(_kv(label, left, right, role))
    lines.append("")
    lines.append(f"[{style_for('artifact')}]Artifact Differences[/]")
    for label, left, right, role in artifact_diff_rows(compare):
        lines.append(_kv(label, left, right, role))
    return lines


def _top_slow(session: InspectSession) -> str:
    if not session.derived.top_slow_steps:
        return "unavailable"
    entry = session.derived.top_slow_steps[0]
    return f"step={entry.step} {entry.ms:.3f} ms"


def render_compare(compare: CompareSession) -> str:
    lines: list[str] = []
    lines.extend(_status_line(compare))
    lines.append("")
    lines.extend(_interpretation(compare))
    lines.append("")
    lines.extend(_performance(compare))
    lines.append("")
    lines.extend(_per_token(compare))
    lines.append("")
    lines.extend(_artifact_runtime(compare))
    lines.append("")
    lines.append(f"[{style_for('muted')}]Notes[/]")
    lines.append(
        f"[{style_for('muted')}]  • Deltas are B minus A. Lower latency/stage "
        "time is better; higher TPS is better.[/]"
    )
    lines.append(
        f"[{style_for('muted')}]  • Compare mode is exactly two runs and remains "
        "post-execution only.[/]"
    )
    return "\n".join(lines)


class CompareView(Vertical):
    """Textual widget rendering the two-run Compare tab."""

    DEFAULT_CSS = """
    CompareView {
        padding: 1 2;
        height: 1fr;
        overflow-y: auto;
    }
    """

    def __init__(self, compare: CompareSession) -> None:
        super().__init__()
        self._compare = compare

    def compose(self) -> ComposeResult:
        yield Static(render_compare(self._compare), markup=True)
