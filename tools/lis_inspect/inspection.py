"""Rule-based interpretation, issues, and two-run comparison helpers."""

from __future__ import annotations

from typing import Any, Optional

from .models import CompareSession, InspectSession, InspectionIssue, STAGE_ORDER

FINGERPRINT_KEYS: tuple[tuple[str, str], ...] = (
    ("binary", "binary_fingerprint"),
    ("model", "model_fingerprint"),
    ("config", "config_fingerprint"),
    ("input", "input_fingerprint"),
    ("runtime", "runtime_fingerprint"),
    ("backend", "backend_fingerprint"),
)


def completeness_role(label: str) -> str:
    if label == "full":
        return "success"
    if label == "degraded":
        return "warning"
    return "partial"


def single_issues(session: InspectSession, label: str = "run") -> list[InspectionIssue]:
    issues: list[InspectionIssue] = []
    source = label
    if not session.source.json.parsed:
        issues.append(InspectionIssue(source, "JSON artifact was not parsed", "error"))
    if not session.source.stderr.provided:
        issues.append(
            InspectionIssue(source, "stderr not provided; perf evidence is partial")
        )
    elif not session.source.stderr.readable:
        issues.append(InspectionIssue(source, "stderr unavailable or unreadable"))
    else:
        if not session.perf.has_stages:
            issues.append(InspectionIssue(source, "perf-stage data unavailable"))
        if not session.perf.has_summary:
            issues.append(InspectionIssue(source, "perf-summary data unavailable"))
        if (
            session.artifact.runtime_settings.get("perf_per_token_enabled")
            and not session.perf.has_per_token
        ):
            issues.append(InspectionIssue(source, "per-token data unavailable"))

    for warning in session.warnings:
        if warning == "stderr log not provided" and not session.source.stderr.provided:
            continue
        if warning.startswith("stderr-log ") and session.source.stderr.provided:
            continue
        if not any(issue.message == warning for issue in issues):
            issues.append(InspectionIssue(source, warning))

    return issues


def compare_issues(compare: CompareSession) -> list[InspectionIssue]:
    issues = single_issues(compare.left, "A")
    issues.extend(single_issues(compare.right, "B"))
    if compare.left.derived.completeness_label != compare.right.derived.completeness_label:
        issues.append(
            InspectionIssue(
                "compare",
                "runs have different completeness levels",
                "partial",
            )
        )
    for warning in compare.warnings:
        issues.append(InspectionIssue("compare", warning))
    return issues


def unavailable_notes(session: InspectSession) -> list[str]:
    notes: list[str] = []
    if not session.perf.has_summary:
        notes.append("perf summary metrics")
    if not session.perf.has_stages:
        notes.append("stage timing breakdown")
    if not session.perf.has_per_token:
        notes.append("per-token latency distribution")
    return notes


def run_interpretation(session: InspectSession) -> list[tuple[str, str, str]]:
    derived = session.derived
    rows = [
        (
            "Completeness",
            derived.completeness_label,
            completeness_role(derived.completeness_label),
        )
    ]
    if derived.run_profile is not None:
        rows.append(("Run profile", derived.run_profile, "perf"))
    else:
        rows.append(("Run profile", "unknown", "muted"))
    rows.append(
        (
            "Warmup effect",
            derived.warmup_effect if derived.warmup_effect is not None else "unknown",
            "perf" if derived.warmup_effect not in (None, "none") else "muted",
        )
    )
    rows.append(
        (
            "Tail spread",
            derived.tail_spread if derived.tail_spread is not None else "unknown",
            "warning"
            if derived.tail_spread == "high"
            else "perf"
            if derived.tail_spread
            else "muted",
        )
    )
    return rows


def compare_summary(compare: CompareSession) -> str:
    left = compare.left
    right = compare.right
    left_decode = _stage_ms(left, "decode_steady_state")
    right_decode = _stage_ms(right, "decode_steady_state")
    if left_decode is not None and right_decode is not None:
        delta = right_decode - left_decode
        threshold = max(abs(left_decode) * 0.03, 1.0)
        if delta < -threshold:
            return "faster decode"
        if delta > threshold:
            return "slower decode"

    left_load = _stage_group_ms(left, ("model_load", "tokenizer_load"))
    right_load = _stage_group_ms(right, ("model_load", "tokenizer_load"))
    if left_load is not None and right_load is not None:
        delta = right_load - left_load
        threshold = max(abs(left_load) * 0.03, 1.0)
        if delta < -threshold:
            return "faster load"
        if delta > threshold:
            return "slower load"
    return "no significant change"


def metric_value(session: InspectSession, name: str) -> Optional[float]:
    summary = session.perf.summary
    if summary is not None:
        return {
            "ttft_ms": summary.ttft_ms,
            "itl_ms": summary.itl_ms,
            "tps_steady": summary.tps_steady,
            "tps_end_to_end": summary.tps_end_to_end,
        }.get(name)
    json_summary = session.artifact.perf_summary_json
    if json_summary is None:
        return None
    value = json_summary.get(name)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_delta(
    left: Optional[float],
    right: Optional[float],
    higher_is_better: bool = False,
) -> tuple[str, str]:
    if left is None or right is None:
        return ("unavailable", "muted")
    delta = right - left
    pct = (delta / left * 100.0) if left else 0.0
    rendered = f"{delta:+.3f} ({pct:+.1f}%)"
    threshold = max(abs(left) * 0.01, 0.001)
    if abs(delta) <= threshold:
        return (rendered, "muted")
    improved = delta > 0 if higher_is_better else delta < 0
    return (rendered, "success" if improved else "warning")


def stage_delta_rows(
    compare: CompareSession,
) -> list[tuple[str, Optional[float], Optional[float], str, str]]:
    rows = []
    for name in STAGE_ORDER:
        left = _stage_ms(compare.left, name)
        right = _stage_ms(compare.right, name)
        delta, role = metric_delta(left, right)
        rows.append((name, left, right, delta, role))
    return rows


def changed_runtime_fields(compare: CompareSession) -> list[tuple[str, str, str, str]]:
    fields = ("context", "batch_size", "generate_limit", "threads")
    rows: list[tuple[str, str, str, str]] = []
    for field in fields:
        left = _fmt(compare.left.artifact.runtime_settings.get(field))
        right = _fmt(compare.right.artifact.runtime_settings.get(field))
        rows.append((field, left, right, "warning" if left != right else "muted"))
    return rows


def artifact_diff_rows(compare: CompareSession) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for key in sorted(
        set(compare.left.artifact.retention) | set(compare.right.artifact.retention)
    ):
        left = _fmt(compare.left.artifact.retention.get(key))
        right = _fmt(compare.right.artifact.retention.get(key))
        rows.append(
            (f"retention.{key}", left, right, "warning" if left != right else "muted")
        )
    for label, key in FINGERPRINT_KEYS:
        left = _fmt(compare.left.artifact.manifest.get(key))
        right = _fmt(compare.right.artifact.manifest.get(key))
        rendered_left = "same" if left == right else left
        rendered_right = "same" if left == right else right
        rows.append(
            (
                f"fingerprint.{label}",
                rendered_left,
                rendered_right,
                "warning" if left != right else "muted",
            )
        )
    for key in ("generated_token_count", "selected_token_count", "emitted_token_count"):
        left = _fmt(compare.left.artifact.report.get(key))
        right = _fmt(compare.right.artifact.report.get(key))
        rows.append((key, left, right, "warning" if left != right else "muted"))
    rows.append(
        (
            "selected == emitted",
            _fmt_bool(compare.left.derived.selected_equals_emitted),
            _fmt_bool(compare.right.derived.selected_equals_emitted),
            "warning"
            if compare.left.derived.selected_equals_emitted
            != compare.right.derived.selected_equals_emitted
            else "muted",
        )
    )
    return rows


def _stage_ms(session: InspectSession, name: str) -> Optional[float]:
    stage = session.perf.stages.get(name)
    return stage.ms if stage is not None else None


def _stage_group_ms(session: InspectSession, names: tuple[str, ...]) -> Optional[float]:
    values = [_stage_ms(session, name) for name in names]
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present)


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return str(value)


def _fmt_bool(value: Optional[bool]) -> str:
    if value is None:
        return "unknown"
    return "yes" if value else "no"
