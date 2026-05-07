"""Centralized semantic color palette for LIS Inspect.

All UI colors flow through `SEMANTIC` so views never hard-code style strings.
Color encodes meaning, not decoration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticStyle:
    """One semantic role and its terminal style.

    `style` is a Rich/Textual style string usable directly in markup.
    """

    role: str
    style: str
    description: str


SEMANTIC: dict[str, SemanticStyle] = {
    "success": SemanticStyle(
        role="success",
        style="bold #5fb074",
        description="Status OK / healthy / completed without error.",
    ),
    "focus": SemanticStyle(
        role="focus",
        style="bold #78a9d8",
        description="Active focus / selected tab / primary run emphasis.",
    ),
    "partial": SemanticStyle(
        role="partial",
        style="bold #c9b267",
        description="Partial but usable input coverage.",
    ),
    "warning": SemanticStyle(
        role="warning",
        style="bold #d4a25b",
        description="Degraded / missing secondary input / parse warning.",
    ),
    "error": SemanticStyle(
        role="error",
        style="bold #c45a5a",
        description="Hard failure / parse failure / incompatible artifact.",
    ),
    "identity": SemanticStyle(
        role="identity",
        style="bold #5b9fd6",
        description="Model family / backend / schema / kind.",
    ),
    "perf": SemanticStyle(
        role="perf",
        style="bold #4fb3b8",
        description="Performance emphasis / dominant stage / active metric.",
    ),
    "artifact": SemanticStyle(
        role="artifact",
        style="#9b8ec7",
        description="Reproducibility / retention / fingerprints.",
    ),
    "muted": SemanticStyle(
        role="muted",
        style="#8a8a8a",
        description="Secondary labels / footers / notes.",
    ),
    "neutral": SemanticStyle(
        role="neutral",
        style="default",
        description="Default text — no semantic emphasis.",
    ),
}


def style_for(role: str) -> str:
    """Return the Rich/Textual style string for `role`, or 'default' if unknown."""
    entry = SEMANTIC.get(role)
    return entry.style if entry is not None else "default"


def status_role(execution_status: str | None) -> str:
    """Map a JSON execution_status to the semantic role its label should use."""
    if execution_status is None:
        return "muted"
    if execution_status.lower() == "ok":
        return "success"
    return "error"


def stop_reason_role(stop_reason: str | None) -> str:
    """Map a stop_reason to the semantic role its label should use.

    Neutral by default. Reasons that imply degradation/failure escalate.
    """
    if stop_reason is None:
        return "muted"
    lowered = stop_reason.lower()
    if "error" in lowered or "fail" in lowered:
        return "error"
    return "neutral"
