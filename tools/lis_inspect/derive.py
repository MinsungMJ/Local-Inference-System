"""Derived metrics: dominant stage, run profile, artifact completeness,
per-token percentile stats, top-N slow steps, and bounded token preview.
"""

from __future__ import annotations

from statistics import median
from typing import Iterable, Optional

from .models import (
    DerivedMetrics,
    PercentileStats,
    PerfLog,
    PerfPerToken,
    RunArtifact,
    STAGE_ORDER,
    TokenPreview,
)


# Bounded token preview: cap how many IDs are surfaced from each end.
TOKEN_PREVIEW_HEAD = 8
TOKEN_PREVIEW_TAIL = 8

# Top-N slowest decode steps surfaced on the Per-Token tab.
TOP_SLOW_STEPS_DEFAULT = 5


# Stages whose magnitude defines the run profile. `tokenizer_encode` and
# `runtime_init` are excluded because they're typically noise-floor.
PROFILE_STAGES: tuple[str, ...] = (
    "model_load",
    "tokenizer_load",
    "prefill",
    "first_decode",
    "decode_steady_state",
)

LOAD_STAGES: frozenset[str] = frozenset({"model_load", "tokenizer_load"})
PREFILL_STAGES: frozenset[str] = frozenset({"prefill"})
DECODE_STAGES: frozenset[str] = frozenset({"first_decode", "decode_steady_state"})


def _dominant_stage(perf: PerfLog) -> Optional[str]:
    if not perf.has_stages:
        return None
    candidates = [
        (name, stage.ms)
        for name, stage in perf.stages.items()
        if name in PROFILE_STAGES
    ]
    if not candidates:
        # Fall back to whatever stages we do have rather than returning None.
        candidates = [(name, stage.ms) for name, stage in perf.stages.items()]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])[0]


def _run_profile(dominant: Optional[str]) -> Optional[str]:
    if dominant is None:
        return None
    if dominant in LOAD_STAGES:
        return "load-heavy"
    if dominant in PREFILL_STAGES:
        return "prefill-heavy"
    if dominant in DECODE_STAGES:
        return "decode-heavy"
    return None


def _artifact_completeness(perf: PerfLog) -> str:
    if perf.warnings:
        return "degraded"
    if perf.has_per_token:
        return "json_perf_per_token"
    if perf.has_stages or perf.has_summary:
        return "json_perf"
    return "json_only"


def _completeness_label(artifact_completeness: str) -> str:
    if artifact_completeness == "degraded":
        return "degraded"
    if artifact_completeness == "json_perf_per_token":
        return "full"
    return "partial"


def _warmup_effect(per_token: list[PerfPerToken]) -> Optional[str]:
    if len(per_token) < 3:
        return None if not per_token else "none"
    first = per_token[0].ms
    baseline = median(entry.ms for entry in per_token[1:])
    if baseline <= 0:
        return "none"
    ratio = first / baseline
    if ratio >= 1.25:
        return "present"
    if ratio >= 1.10:
        return "mild"
    return "none"


def _tail_spread(stats: Optional[PercentileStats]) -> Optional[str]:
    if stats is None or stats.p50_ms <= 0:
        return None
    spread = (stats.p99_ms - stats.p50_ms) / stats.p50_ms
    if spread >= 0.20:
        return "high"
    if spread >= 0.08:
        return "moderate"
    return "low"


def _selected_equals_emitted(artifact: RunArtifact) -> Optional[bool]:
    report = artifact.report
    selected = report.get("selected_token_digest")
    emitted = report.get("emitted_token_digest")
    if selected is None or emitted is None:
        # Fall back to id-list comparison if digests aren't carried.
        sel_ids = report.get("selected_token_ids")
        emit_ids = report.get("emitted_token_ids")
        if isinstance(sel_ids, list) and isinstance(emit_ids, list):
            return sel_ids == emit_ids
        return None
    return selected == emitted


def percentile(values: Iterable[float], pct: float) -> float:
    """Linear-interpolation percentile (NumPy-style, matches numpy.percentile default).

    `pct` is in [0, 100]. Empty input raises ValueError.
    """
    if not 0.0 <= pct <= 100.0:
        raise ValueError(f"pct must be in [0, 100], got {pct}")
    sample = sorted(float(v) for v in values)
    n = len(sample)
    if n == 0:
        raise ValueError("percentile() requires at least one value")
    if n == 1:
        return sample[0]
    rank = (pct / 100.0) * (n - 1)
    lo = int(rank)
    hi = min(lo + 1, n - 1)
    frac = rank - lo
    return sample[lo] + frac * (sample[hi] - sample[lo])


def compute_percentile_stats(per_token: list[PerfPerToken]) -> Optional[PercentileStats]:
    """Compute count/min/max/mean/p50/p90/p95/p99 over per-token ms values.

    Returns None when there is no data. Computed over all entries — does not
    exclude the warm-up step. Operators see the warm-up tail explicitly.
    """
    if not per_token:
        return None
    values = [entry.ms for entry in per_token]
    total = sum(values)
    return PercentileStats(
        count=len(values),
        min_ms=min(values),
        max_ms=max(values),
        mean_ms=total / len(values),
        p50_ms=percentile(values, 50.0),
        p90_ms=percentile(values, 90.0),
        p95_ms=percentile(values, 95.0),
        p99_ms=percentile(values, 99.0),
    )


def top_slow_steps(
    per_token: list[PerfPerToken],
    n: int = TOP_SLOW_STEPS_DEFAULT,
) -> list[PerfPerToken]:
    """Return the slowest `n` steps, descending by ms, ties broken by step asc."""
    if not per_token or n <= 0:
        return []
    return sorted(per_token, key=lambda e: (-e.ms, e.step))[:n]


def _coerce_id(value) -> Optional[int]:
    """Best-effort coercion to int. Returns None if value is not int-shaped."""
    if isinstance(value, bool):  # bools are ints in Python; reject explicitly
        return None
    if isinstance(value, int):
        return value
    return None


def bounded_token_preview(
    ids: Optional[list],
    count: Optional[int],
    digest: Optional[str],
    head: int = TOKEN_PREVIEW_HEAD,
    tail: int = TOKEN_PREVIEW_TAIL,
) -> Optional[TokenPreview]:
    """Build a `TokenPreview` from a JSON token-ID list (or absence of it).

    Returns None when nothing useful is available (no ids, no count, no digest).
    When `ids` is missing but count/digest are present, returns a preview with
    `ids_present=False` so the UI can still show count + digest.
    """
    if ids is None and count is None and digest is None:
        return None

    if ids is None:
        return TokenPreview(
            count=count,
            digest=digest,
            ids_present=False,
            head_ids=[],
            tail_ids=[],
            omitted=0,
        )

    coerced: list[int] = []
    for value in ids:
        cid = _coerce_id(value)
        if cid is not None:
            coerced.append(cid)
    actual_count = count if count is not None else len(coerced)

    if len(coerced) <= head + tail:
        return TokenPreview(
            count=actual_count,
            digest=digest,
            ids_present=True,
            head_ids=coerced,
            tail_ids=[],
            omitted=0,
        )
    return TokenPreview(
        count=actual_count,
        digest=digest,
        ids_present=True,
        head_ids=coerced[:head],
        tail_ids=coerced[-tail:],
        omitted=len(coerced) - head - tail,
    )


def _selected_preview(artifact: RunArtifact) -> Optional[TokenPreview]:
    report = artifact.report
    return bounded_token_preview(
        ids=report.get("selected_token_ids") if isinstance(report.get("selected_token_ids"), list) else None,
        count=report.get("selected_token_count"),
        digest=report.get("selected_token_digest"),
    )


def _emitted_preview(artifact: RunArtifact) -> Optional[TokenPreview]:
    report = artifact.report
    return bounded_token_preview(
        ids=report.get("emitted_token_ids") if isinstance(report.get("emitted_token_ids"), list) else None,
        count=report.get("emitted_token_count"),
        digest=report.get("emitted_token_digest"),
    )


def derive(artifact: RunArtifact, perf: PerfLog) -> DerivedMetrics:
    dominant = _dominant_stage(perf)
    artifact_completeness = _artifact_completeness(perf)
    per_token_stats = compute_percentile_stats(perf.per_token)
    return DerivedMetrics(
        dominant_stage=dominant,
        run_profile=_run_profile(dominant),
        artifact_completeness=artifact_completeness,
        completeness_label=_completeness_label(artifact_completeness),
        warmup_effect=_warmup_effect(perf.per_token),
        tail_spread=_tail_spread(per_token_stats),
        selected_equals_emitted=_selected_equals_emitted(artifact),
        per_token_stats=per_token_stats,
        top_slow_steps=top_slow_steps(perf.per_token),
        selected_preview=_selected_preview(artifact),
        emitted_preview=_emitted_preview(artifact),
    )


def ordered_stages(perf: PerfLog) -> list[tuple[str, float]]:
    """Return (name, ms) pairs in canonical stage order, including only present stages."""
    return [
        (name, perf.stages[name].ms)
        for name in STAGE_ORDER
        if name in perf.stages
    ]
