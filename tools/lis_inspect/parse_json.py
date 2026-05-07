"""JSON artifact parser.

Validates the schema/kind contract from docs/repro_execution_artifacts.md and
returns a `RunArtifact`. Refuses incompatible artifacts with a clear error.
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import RunArtifact

EXPECTED_SCHEMA = "lis.execution_artifact/v1"
EXPECTED_KIND = "run_report"


class ArtifactParseError(ValueError):
    """Raised when the JSON artifact is missing or has an incompatible shape."""


def load_artifact(path: str | Path) -> RunArtifact:
    """Load and validate a LIS run-report JSON artifact.

    Validation is intentionally narrow: schema + kind. Detailed field-level
    typing is left to view code, since the canonical schema may evolve.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ArtifactParseError("report-json not found") from exc
    except OSError as exc:
        detail = exc.strerror or exc.__class__.__name__
        raise ArtifactParseError(f"report-json unreadable: {detail}") from exc

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ArtifactParseError(f"report-json is not valid JSON: {exc.msg}") from exc

    if not isinstance(raw, dict):
        raise ArtifactParseError(
            f"report-json root must be an object, got {type(raw).__name__}"
        )

    schema = raw.get("schema")
    if schema != EXPECTED_SCHEMA:
        raise ArtifactParseError(
            f"unsupported schema {schema!r} (expected {EXPECTED_SCHEMA!r})"
        )

    kind = raw.get("kind")
    if kind != EXPECTED_KIND:
        raise ArtifactParseError(
            f"unsupported kind {kind!r} (expected {EXPECTED_KIND!r})"
        )

    return RunArtifact(raw=raw, schema=schema, kind=kind)
