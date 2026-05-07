"""Artifact tab — answers 'what reproducibility/contract surface does this run expose?'

Renders LIS-specific provenance: schema/kind/output_mode, retention policy,
token accounting (counts + digests + selected==emitted + bounded ID preview),
and fingerprints (binary/model/config/input/runtime/backend). Honors retention:
no raw prompt text, no generated text, no absolute paths.
"""

from __future__ import annotations

from typing import Any, Optional

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from ..models import InspectSession, RunArtifact, TokenPreview
from ..palette import style_for


# Fingerprint manifest keys, in display order. Mirrors the sequence in
# docs/repro_execution_artifacts.md. Format: (display label, manifest key).
FINGERPRINT_KEYS: tuple[tuple[str, str], ...] = (
    ("Binary", "binary_fingerprint"),
    ("Model", "model_fingerprint"),
    ("Config", "config_fingerprint"),
    ("Input", "input_fingerprint"),
    ("Runtime", "runtime_fingerprint"),
    ("Backend", "backend_fingerprint"),
)


def _kv(label: str, value: Optional[str], role: str = "neutral") -> str:
    style = style_for(role)
    rendered = value if value is not None else "—"
    return f"[{style_for('muted')}]  {label:<22}[/] [{style}]{rendered}[/]"


def _yes_no(value: Optional[bool]) -> Optional[str]:
    if value is None:
        return None
    return "yes" if value else "no"


def _fmt_retention_value(value: Any) -> tuple[str, str]:
    """Return (rendered, role) for a retention dict value.

    Booleans encode "is this surfaced?" — `false` is the safer/calmer default
    and renders in `artifact` role; `true` (something is surfaced) renders in
    `warning` role since it deviates from minimized retention.
    """
    if isinstance(value, bool):
        return ("surfaced" if value else "omitted", "warning" if value else "artifact")
    if value is None:
        return ("—", "muted")
    return (str(value), "neutral")


def _format_retention(retention: dict) -> list[str]:
    if not retention:
        return [
            f"[{style_for('muted')}]  (no retention dict in artifact)[/]"
        ]
    lines: list[str] = []
    for key in sorted(retention.keys()):
        rendered, role = _fmt_retention_value(retention[key])
        lines.append(
            f"[{style_for('muted')}]  {key:<22}[/] "
            f"[{style_for(role)}]{rendered}[/]"
        )
    return lines


def _format_token_preview(label: str, preview: Optional[TokenPreview]) -> list[str]:
    """Render a bounded token preview block. No raw text — IDs only."""
    if preview is None:
        return [
            f"[{style_for('artifact')}]{label}[/]",
            f"[{style_for('muted')}]  (no token data in artifact)[/]",
        ]
    lines = [f"[{style_for('artifact')}]{label}[/]"]
    if preview.count is not None:
        lines.append(_kv("count", str(preview.count), "neutral"))
    else:
        lines.append(_kv("count", None))
    if preview.digest is not None:
        lines.append(_kv("digest", preview.digest, "artifact"))
    else:
        lines.append(_kv("digest", None))

    if not preview.ids_present:
        lines.append(
            f"[{style_for('muted')}]  ids                    omitted by artifact "
            "(JSON did not surface raw IDs)[/]"
        )
        return lines

    head_str = ", ".join(str(v) for v in preview.head_ids)
    if preview.tail_ids:
        tail_str = ", ".join(str(v) for v in preview.tail_ids)
        omitted = preview.omitted
        rendered = (
            f"[{head_str}, … {omitted} hidden …, {tail_str}]"
        )
    elif preview.head_ids:
        rendered = f"[{head_str}]"
    else:
        rendered = "[]"
    lines.append(_kv("ids (preview)", rendered, "neutral"))
    return lines


def _format_fingerprints(manifest: dict) -> list[str]:
    lines = [f"[{style_for('artifact')}]Fingerprints[/]"]
    for label, key in FINGERPRINT_KEYS:
        value = manifest.get(key)
        rendered = value if isinstance(value, str) else None
        lines.append(_kv(label, rendered, "artifact"))
    return lines


def _prompt_sequences_summary(report: dict) -> list[str]:
    sequences = report.get("prompt_sequences")
    if not isinstance(sequences, list) or not sequences:
        return [_kv("Prompt sequences", "0", "neutral")]
    lines = [_kv("Prompt sequences", str(len(sequences)), "neutral")]
    for idx, seq in enumerate(sequences, start=1):
        if not isinstance(seq, dict):
            continue
        token_count = seq.get("token_count")
        digest = seq.get("digest")
        rendered = (
            f"{token_count if token_count is not None else '—'} token(s)"
            f"  digest={digest if digest else '—'}"
        )
        lines.append(_kv(f"  seq {idx}", rendered, "artifact"))
    return lines


def render_artifact(session: InspectSession) -> str:
    artifact: RunArtifact = session.artifact
    derived = session.derived
    manifest = artifact.manifest
    report = artifact.report

    lines: list[str] = []

    lines.append(f"[{style_for('identity')}]Identity[/]")
    lines.append(_kv("Schema", artifact.schema, "identity"))
    lines.append(_kv("Kind", artifact.kind, "identity"))
    lines.append(_kv("Model format", _maybe_str(manifest.get("model_format")), "identity"))
    lines.append(_kv("Model family", artifact.model_family, "identity"))
    lines.append(_kv("Backend", artifact.backend, "identity"))
    lines.append(_kv("Output mode", artifact.output_mode, "identity"))
    lines.append(_kv("Input mode", _maybe_str(manifest.get("input_mode")), "identity"))
    lines.append("")

    lines.append(f"[{style_for('artifact')}]Retention Policy[/]")
    lines.extend(_format_retention(artifact.retention))
    lines.append("")

    lines.append(f"[{style_for('artifact')}]Token Accounting[/]")
    lines.extend(_prompt_sequences_summary(report))
    eq_role = "success" if derived.selected_equals_emitted else "warning" if derived.selected_equals_emitted is False else "muted"
    eq_value = _yes_no(derived.selected_equals_emitted)
    lines.append(_kv("Selected == emitted", eq_value, eq_role))
    lines.append("")

    lines.extend(_format_token_preview("Selected tokens", derived.selected_preview))
    lines.append("")
    lines.extend(_format_token_preview("Emitted tokens", derived.emitted_preview))
    lines.append("")

    lines.extend(_format_fingerprints(manifest))
    lines.append("")

    lines.append(f"[{style_for('muted')}]Notes[/]")
    lines.append(
        f"[{style_for('muted')}]  • JSON remains the canonical machine-readable "
        "source of truth.[/]"
    )
    lines.append(
        f"[{style_for('muted')}]  • Raw prompt text, generated text, and absolute "
        "paths are not surfaced.[/]"
    )
    lines.append(
        f"[{style_for('muted')}]  • Token IDs and digests are retained; raw text is not.[/]"
    )

    return "\n".join(lines)


def _maybe_str(value) -> Optional[str]:
    if value is None:
        return None
    return str(value)


class ArtifactView(Vertical):
    """Textual widget rendering the Artifact tab."""

    DEFAULT_CSS = """
    ArtifactView {
        padding: 1 2;
        height: 1fr;
        overflow-y: auto;
    }
    """

    def __init__(self, session: InspectSession) -> None:
        super().__init__()
        self._session = session

    def compose(self) -> ComposeResult:
        yield Static(render_artifact(self._session), markup=True)
