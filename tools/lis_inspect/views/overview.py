"""Overview tab — answers 'what run was this and what was the outcome?'"""

from __future__ import annotations

from typing import Iterable, Optional

from rich.markup import escape
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..inspection import run_interpretation
from ..models import InspectSession, PerfStage
from ..palette import status_role, stop_reason_role, style_for


def _badge(label: str, role: str) -> str:
    style = style_for(role)
    return f"[{style}]▌[/][reverse {style}] {label} [/]"


def _kv(label: str, value: Optional[str], role: str = "neutral") -> str:
    style = style_for(role)
    rendered = value if value is not None else "—"
    return f"[{style_for('muted')}]{label:<22}[/] [{style}]{rendered}[/]"


def _stages_dict(stages: Iterable[PerfStage]) -> dict[str, PerfStage]:
    return {stage.name: stage for stage in stages}


def _format_perf_summary(session: InspectSession) -> list[str]:
    summary = session.perf.summary
    if summary is None:
        # Fall back to JSON-embedded perf summary when stderr was not provided.
        json_summary = session.artifact.perf_summary_json
        if json_summary is None:
            return [
                f"[{style_for('muted')}]Perf summary unavailable "
                f"(run with --perf and capture stderr, or include in JSON).[/]"
            ]
        return [
            _kv("TTFT (ms)", _maybe_fmt_float(json_summary.get("ttft_ms")), "perf"),
            _kv("Mean ITL (ms)", _maybe_fmt_float(json_summary.get("itl_ms")), "perf"),
            _kv("TPS steady", _maybe_fmt_float(json_summary.get("tps_steady")), "perf"),
            _kv("TPS end-to-end", _maybe_fmt_float(json_summary.get("tps_end_to_end")), "perf"),
        ]
    return [
        _kv("TTFT (ms)", f"{summary.ttft_ms:.3f}", "perf"),
        _kv("Mean ITL (ms)", f"{summary.itl_ms:.3f}", "perf"),
        _kv("TPS steady", f"{summary.tps_steady:.3f}", "perf"),
        _kv("TPS end-to-end", f"{summary.tps_end_to_end:.3f}", "perf"),
    ]


def _maybe_fmt_float(value) -> Optional[str]:
    if value is None:
        return None
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _yes_no(value: Optional[bool]) -> Optional[str]:
    if value is None:
        return None
    return "yes" if value else "no"


def render_overview(session: InspectSession) -> str:
    artifact = session.artifact
    runtime = artifact.runtime_settings
    report = artifact.report

    # Health badges.
    badges: list[str] = []
    badges.append(
        _badge(
            artifact.execution_status or "UNKNOWN",
            status_role(artifact.execution_status),
        )
    )
    if session.perf.has_stages or session.perf.has_summary:
        badges.append(_badge("perf captured", "perf"))
    else:
        badges.append(_badge("perf missing", "warning"))
    if session.perf.has_per_token:
        badges.append(_badge("per-token available", "perf"))
    retention = artifact.retention
    if retention and not any(retention.values()):
        badges.append(_badge("retention minimized", "artifact"))
    if session.warnings:
        badges.append(_badge(f"warnings {len(session.warnings)}", "warning"))

    perf_enabled = bool(runtime.get("perf_enabled"))
    per_token_enabled = bool(runtime.get("perf_per_token_enabled"))

    selected_eq = _yes_no(session.derived.selected_equals_emitted)

    lines: list[str] = []
    lines.append(" ".join(badges))
    lines.append("")
    lines.append(f"[{style_for('identity')}]Outcome[/]")
    lines.append(_kv("Status", artifact.execution_status, status_role(artifact.execution_status)))
    lines.append(_kv("Stop reason", artifact.stop_reason, stop_reason_role(artifact.stop_reason)))
    lines.append(_kv("Output mode", artifact.output_mode))
    lines.append("")
    lines.append(f"[{style_for('identity')}]Identity[/]")
    lines.append(_kv("Schema", artifact.schema, "identity"))
    lines.append(_kv("Kind", artifact.kind, "identity"))
    lines.append(_kv("Model family", artifact.model_family, "identity"))
    lines.append(_kv("Backend", artifact.backend, "identity"))
    lines.append("")
    lines.append(f"[{style_for('identity')}]Runtime[/]")
    lines.append(_kv("Context", _maybe_str(runtime.get("context"))))
    lines.append(_kv("Batch size", _maybe_str(runtime.get("batch_size"))))
    lines.append(_kv("Generation limit", _maybe_str(runtime.get("generate_limit"))))
    lines.append(_kv("Threads", _maybe_str(runtime.get("threads"))))
    lines.append(
        _kv(
            "Perf",
            "enabled" if perf_enabled else "disabled",
            "perf" if perf_enabled else "muted",
        )
    )
    lines.append(
        _kv(
            "Per-token",
            "enabled" if per_token_enabled else "disabled",
            "perf" if per_token_enabled else "muted",
        )
    )
    lines.append("")
    lines.append(f"[{style_for('identity')}]Token Accounting[/]")
    lines.append(_kv("Prompt tokens", _prompt_token_count(report)))
    lines.append(_kv("Generated tokens", _maybe_str(report.get("generated_token_count"))))
    lines.append(_kv("Selected == emitted", selected_eq, "success" if selected_eq == "yes" else "muted"))
    lines.append("")
    lines.append(f"[{style_for('perf')}]Performance Summary[/]")
    lines.extend(_format_perf_summary(session))
    if session.derived.dominant_stage is not None:
        lines.append(
            _kv(
                "Dominant stage",
                session.derived.dominant_stage,
                "perf",
            )
        )
    if session.derived.run_profile is not None:
        lines.append(_kv("Run profile", session.derived.run_profile, "perf"))
    lines.append(_kv("Artifact completeness", session.derived.artifact_completeness, "artifact"))
    lines.append("")
    lines.append(f"[{style_for('identity')}]Interpretation[/]")
    for label, value, role in run_interpretation(session):
        lines.append(_kv(label, value, role))

    if session.warnings:
        lines.append("")
        lines.append(f"[{style_for('warning')}]Notes[/]")
        lines.append(
            f"[{style_for('warning')}]Parser warnings: {len(session.warnings)}[/]"
        )
        for warning in session.warnings[:3]:
            lines.append(f"[{style_for('warning')}]• {escape(warning)}[/]")
        if len(session.warnings) > 3:
            lines.append(
                f"[{style_for('muted')}]• {len(session.warnings) - 3} more in Raw[/]"
            )
    return "\n".join(lines)


def _maybe_str(value) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _prompt_token_count(report: dict) -> Optional[str]:
    """Sum prompt-sequence token counts from the JSON report.

    Supports both a flat `prompt_token_count` and the per-sequence form documented
    in repro_execution_artifacts.md.
    """
    flat = report.get("prompt_token_count")
    if flat is not None:
        return str(flat)
    sequences = report.get("prompt_sequences")
    if isinstance(sequences, list):
        total = 0
        for seq in sequences:
            if isinstance(seq, dict):
                count = seq.get("token_count")
                if isinstance(count, int):
                    total += count
        return str(total) if total else "0"
    return None


class OverviewView(Vertical):
    """Textual widget rendering the Overview tab."""

    DEFAULT_CSS = """
    OverviewView {
        padding: 1 2;
        height: 1fr;
        overflow-y: auto;
    }
    """

    def __init__(self, session: InspectSession) -> None:
        super().__init__()
        self._session = session

    def compose(self) -> ComposeResult:
        yield Static(render_overview(self._session), markup=True)
