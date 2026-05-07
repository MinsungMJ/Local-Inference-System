"""Internal data model for LIS Inspect.

`RunArtifact` is the parsed canonical JSON report.
`PerfLog` is the parsed stderr supplementary perf data.
`InspectSession` merges both plus derived metrics and parser warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Canonical stage order from docs/performance_measurement.md.
STAGE_ORDER: tuple[str, ...] = (
    "model_load",
    "tokenizer_load",
    "tokenizer_encode",
    "runtime_init",
    "prefill",
    "first_decode",
    "decode_steady_state",
)


@dataclass(frozen=True)
class PerfStage:
    name: str
    ns: int
    ms: float
    tokens: int


@dataclass(frozen=True)
class PerfSummary:
    threads: int
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float
    itl_ms: float
    tps_steady: float
    tps_end_to_end: float


@dataclass(frozen=True)
class PerfPerToken:
    step: int
    ns: int
    ms: float


@dataclass
class PerfLog:
    """Parsed supplementary stderr lines.

    `stages`, `summary`, and optional `per_token` entries are collected by the
    parser for downstream views.
    """

    tag: Optional[str] = None
    stages: dict[str, PerfStage] = field(default_factory=dict)
    summary: Optional[PerfSummary] = None
    per_token: list[PerfPerToken] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_stages(self) -> bool:
        return bool(self.stages)

    @property
    def has_summary(self) -> bool:
        return self.summary is not None

    @property
    def has_per_token(self) -> bool:
        return bool(self.per_token)


@dataclass
class RunArtifact:
    """Parsed canonical JSON `lis.execution_artifact/v1` `run_report`.

    Stores the raw dict so views can pull fields directly without forcing every
    schema field into a Python attribute on day one.
    """

    raw: dict[str, Any]
    schema: str
    kind: str

    @property
    def manifest(self) -> dict[str, Any]:
        return self.raw.get("manifest", {}) or {}

    @property
    def report(self) -> dict[str, Any]:
        return self.raw.get("report", {}) or {}

    @property
    def runtime_settings(self) -> dict[str, Any]:
        return self.manifest.get("runtime_settings", {}) or {}

    @property
    def retention(self) -> dict[str, Any]:
        return self.manifest.get("retention", {}) or {}

    @property
    def execution_status(self) -> Optional[str]:
        value = self.report.get("execution_status")
        return str(value) if value is not None else None

    @property
    def stop_reason(self) -> Optional[str]:
        value = self.report.get("stop_reason")
        return str(value) if value is not None else None

    @property
    def output_mode(self) -> Optional[str]:
        value = self.report.get("output_mode")
        return str(value) if value is not None else None

    @property
    def model_family(self) -> Optional[str]:
        value = self.manifest.get("model_family")
        return str(value) if value is not None else None

    @property
    def backend(self) -> Optional[str]:
        value = self.manifest.get("backend")
        return str(value) if value is not None else None

    @property
    def perf_summary_json(self) -> Optional[dict[str, Any]]:
        """Bounded perf summary inside the JSON report, if present."""
        value = self.report.get("perf_summary")
        if isinstance(value, dict):
            return value
        return None


@dataclass(frozen=True)
class PercentileStats:
    """Latency distribution summary over a sequence of per-token values (ms)."""

    count: int
    min_ms: float
    max_ms: float
    mean_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float


@dataclass(frozen=True)
class TokenPreview:
    """Bounded preview of a token-ID sequence for the Artifact tab.

    `head_ids` and `tail_ids` are explicit lists. `omitted` is the number of IDs
    hidden between head and tail. `ids_present` is False when the JSON did not
    carry the raw ID list — count and digest may still be available.
    """

    count: Optional[int]
    digest: Optional[str]
    ids_present: bool
    head_ids: list[int]
    tail_ids: list[int]
    omitted: int


@dataclass
class DerivedMetrics:
    """Computed labels and summaries used by views."""

    dominant_stage: Optional[str] = None
    run_profile: Optional[str] = None  # "load-heavy" | "prefill-heavy" | "decode-heavy"
    artifact_completeness: str = "json_only"  # | "json_perf" | "json_perf_per_token"
    completeness_label: str = "partial"  # "full" | "partial" | "degraded"
    warmup_effect: Optional[str] = None  # "none" | "mild" | "present"
    tail_spread: Optional[str] = None  # "low" | "moderate" | "high"
    selected_equals_emitted: Optional[bool] = None
    per_token_stats: Optional[PercentileStats] = None
    top_slow_steps: list[PerfPerToken] = field(default_factory=list)
    selected_preview: Optional[TokenPreview] = None
    emitted_preview: Optional[TokenPreview] = None


@dataclass(frozen=True)
class SourcePreview:
    """Bounded, retention-safe preview of one inspector input."""

    provided: bool
    readable: bool
    parsed: bool
    preview: str = ""
    truncated: bool = False
    line_count: int = 0
    error: Optional[str] = None


@dataclass(frozen=True)
class SourceState:
    """Input presence/read/parse status used by transparency views."""

    json: SourcePreview
    stderr: SourcePreview


def empty_source_state() -> SourceState:
    """Default source state for tests that construct sessions directly."""
    empty = SourcePreview(provided=False, readable=False, parsed=False)
    return SourceState(json=empty, stderr=empty)


@dataclass
class InspectSession:
    """Merged view-model passed to TUI views."""

    artifact: RunArtifact
    perf: PerfLog
    derived: DerivedMetrics
    warnings: list[str] = field(default_factory=list)
    source: SourceState = field(default_factory=empty_source_state)

    @property
    def stderr_present(self) -> bool:
        return self.source.stderr.readable or (
            self.perf.has_stages or self.perf.has_summary or self.perf.has_per_token
        )


@dataclass(frozen=True)
class InspectionIssue:
    """One compact issue/completeness note for operator-facing panels."""

    source: str
    message: str
    role: str = "warning"


@dataclass
class CompareSession:
    """Exactly two LIS runs compared side by side."""

    left: InspectSession
    right: InspectSession
    warnings: list[str] = field(default_factory=list)
