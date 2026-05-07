"""Bounded, retention-safe source previews for the Raw tab."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .models import RunArtifact, SourcePreview

JSON_PREVIEW_LINES = 32
STDERR_PREVIEW_LINES = 32
WARNING_MAX_CHARS = 160

_ABSOLUTE_PATH_RE = re.compile(r"(?<!\S)/(?:[^\s/'\"<>]+/)+[^\s'\"<>]+")
_WINDOWS_PATH_RE = re.compile(r"\b[A-Za-z]:\\[^\s'\"<>]+")
_PROMPT_FIELD_RE = re.compile(
    r"\b(prompt|prompt_text|generated_text)=('[^']*'|\"[^\"]*\"|\S+)"
)

_SENSITIVE_KEY_FRAGMENTS = (
    "raw_prompt",
    "prompt_text",
    "generated_text",
    "raw_generated",
)


def redact_sensitive_text(text: str) -> str:
    """Redact path-like and prompt/output-like substrings from raw previews."""
    redacted = _ABSOLUTE_PATH_RE.sub("<absolute-path-redacted>", text)
    redacted = _WINDOWS_PATH_RE.sub("<absolute-path-redacted>", redacted)
    return _PROMPT_FIELD_RE.sub(r"\1=<redacted>", redacted)


def normalize_warning(warning: str, max_chars: int = WARNING_MAX_CHARS) -> str:
    """Redact and bound a parser warning for compact UI display."""
    cleaned = redact_sensitive_text(str(warning)).replace("\n", " ").strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 1]}…"


def _sanitize_json_value(key: str | None, value: Any) -> Any:
    lowered = key.lower() if key is not None else ""
    if any(fragment in lowered for fragment in _SENSITIVE_KEY_FRAGMENTS):
        return "<redacted by retention policy>"
    if "path" in lowered and isinstance(value, str):
        return "<absolute-path-redacted>"
    if isinstance(value, dict):
        return {str(k): _sanitize_json_value(str(k), v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(None, item) for item in value]
    if isinstance(value, str):
        return redact_sensitive_text(value)
    return value


def bounded_preview(text: str, max_lines: int) -> tuple[str, bool, int]:
    """Return first `max_lines` from `text`, plus truncation metadata."""
    lines = text.splitlines()
    line_count = len(lines)
    if line_count <= max_lines:
        return "\n".join(lines), False, line_count
    shown = lines[:max_lines]
    hidden = line_count - max_lines
    shown.append(
        f"… truncated; showing first {max_lines} of {line_count} lines ({hidden} hidden)"
    )
    return "\n".join(shown), True, line_count


def json_source_preview(
    artifact: RunArtifact,
    max_lines: int = JSON_PREVIEW_LINES,
) -> SourcePreview:
    sanitized = _sanitize_json_value(None, artifact.raw)
    rendered = json.dumps(sanitized, indent=2, ensure_ascii=True)
    preview, truncated, line_count = bounded_preview(rendered, max_lines)
    return SourcePreview(
        provided=True,
        readable=True,
        parsed=True,
        preview=preview,
        truncated=truncated,
        line_count=line_count,
    )


def stderr_source_preview(
    *,
    provided: bool,
    text: str = "",
    parsed: bool = False,
    error: str | None = None,
    max_lines: int = STDERR_PREVIEW_LINES,
) -> SourcePreview:
    if not provided:
        return SourcePreview(provided=False, readable=False, parsed=False)
    if error is not None:
        return SourcePreview(
            provided=True,
            readable=False,
            parsed=False,
            error=normalize_warning(error),
        )
    preview, truncated, line_count = bounded_preview(redact_sensitive_text(text), max_lines)
    return SourcePreview(
        provided=True,
        readable=True,
        parsed=parsed,
        preview=preview,
        truncated=truncated,
        line_count=line_count,
    )


def read_text_for_preview(path: Path) -> tuple[str | None, str | None]:
    """Read an optional source file, returning `(text, bounded_error)`."""
    try:
        return path.read_text(encoding="utf-8", errors="replace"), None
    except FileNotFoundError:
        return None, "stderr-log not found"
    except OSError as exc:
        return None, f"stderr-log unreadable: {exc.strerror or exc.__class__.__name__}"
