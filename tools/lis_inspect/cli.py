"""Command-line entry: parse args, load inputs, build session, run TUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .derive import derive
from .inspection import compare_summary
from .models import CompareSession, InspectSession, PerfLog, SourceState
from .parse_json import ArtifactParseError, load_artifact
from .parse_stderr import parse_stderr_text
from .source_preview import (
    json_source_preview,
    normalize_warning,
    read_text_for_preview,
    stderr_source_preview,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lis_inspect",
        description=(
            "Post-execution TUI inspector for LIS runs. JSON is canonical; "
            "stderr is supplementary."
        ),
    )
    parser.add_argument(
        "--report-json",
        required=True,
        nargs="+",
        type=Path,
        metavar="PATH",
        help=(
            "Path to one LIS --report-json artifact, or exactly two paths for "
            "two-run compare mode."
        ),
    )
    parser.add_argument(
        "--stderr-log",
        type=Path,
        nargs="*",
        default=[],
        metavar="PATH",
        help=(
            "Optional captured stderr path(s). In compare mode, provide zero, "
            "one, or two paths matching the report-json order."
        ),
    )
    parser.add_argument(
        "--no-tui",
        action="store_true",
        help=(
            "Skip launching the TUI. Loads inputs and prints a one-line summary "
            "to stdout. Useful for CI / parser smoke checks."
        ),
    )
    return parser


def build_session(
    report_json: Path,
    stderr_log: Optional[Path],
) -> InspectSession:
    artifact = load_artifact(report_json)

    json_source = json_source_preview(artifact)
    if stderr_log is None:
        perf = PerfLog()
        stderr_source = stderr_source_preview(provided=False)
    else:
        stderr_text, stderr_error = read_text_for_preview(stderr_log)
        if stderr_error is not None:
            perf = PerfLog(warnings=[stderr_error])
            stderr_source = stderr_source_preview(provided=True, error=stderr_error)
        else:
            assert stderr_text is not None
            perf = parse_stderr_text(stderr_text)
            _augment_perf_warnings(perf, artifact.runtime_settings)
            stderr_source = stderr_source_preview(
                provided=True,
                text=stderr_text,
                parsed=perf.has_stages or perf.has_summary or perf.has_per_token,
            )

    derived = derive(artifact, perf)
    warnings: list[str] = []
    if stderr_log is None:
        warnings.append("stderr log not provided")
    warnings.extend(perf.warnings)
    warnings = [normalize_warning(warning) for warning in warnings]
    source = SourceState(json=json_source, stderr=stderr_source)
    return InspectSession(
        artifact=artifact,
        perf=perf,
        derived=derived,
        warnings=warnings,
        source=source,
    )


def build_compare_session(
    left_report_json: Path,
    right_report_json: Path,
    left_stderr_log: Optional[Path] = None,
    right_stderr_log: Optional[Path] = None,
) -> CompareSession:
    left = build_session(left_report_json, left_stderr_log)
    right = build_session(right_report_json, right_stderr_log)
    warnings: list[str] = []
    if left.derived.completeness_label != right.derived.completeness_label:
        warnings.append("compare inputs have different completeness levels")
    return CompareSession(left=left, right=right, warnings=warnings)


def _augment_perf_warnings(perf: PerfLog, runtime_settings: dict) -> None:
    """Add compact parser-completeness warnings for a readable stderr source."""
    parsed_any = perf.has_stages or perf.has_summary or perf.has_per_token
    if not parsed_any:
        perf.warnings.append("stderr log contained no LIS perf sections")
        return
    if not perf.has_stages:
        perf.warnings.append("perf-stage section missing")
    if not perf.has_summary:
        perf.warnings.append("perf-summary section missing")
    per_token_enabled = bool(runtime_settings.get("perf_per_token_enabled"))
    if per_token_enabled and not perf.has_per_token:
        perf.warnings.append("perf-per-token section missing")
    if perf.summary is not None and perf.has_per_token:
        expected = perf.summary.generated_tokens
        actual = len(perf.per_token)
        if expected != actual:
            perf.warnings.append(
                f"perf-per-token count {actual} differs from generated_tokens {expected}"
            )


def _summary_line(session: InspectSession) -> str:
    art = session.artifact
    parts = [
        f"status={art.execution_status}",
        f"stop_reason={art.stop_reason}",
        f"model_family={art.model_family}",
        f"backend={art.backend}",
        f"completeness={session.derived.artifact_completeness}",
    ]
    if session.warnings:
        parts.append(f"warnings={len(session.warnings)}")
    if session.derived.dominant_stage is not None:
        parts.append(f"dominant_stage={session.derived.dominant_stage}")
    return " ".join(parts)


def _compare_summary_line(compare: CompareSession) -> str:
    parts = [
        "mode=compare",
        f"left_status={compare.left.artifact.execution_status}",
        f"right_status={compare.right.artifact.execution_status}",
        f"left_completeness={compare.left.derived.artifact_completeness}",
        f"right_completeness={compare.right.derived.artifact_completeness}",
        f"summary={compare_summary(compare).replace(' ', '_')}",
    ]
    issue_count = len(compare.left.warnings) + len(compare.right.warnings) + len(compare.warnings)
    if issue_count:
        parts.append(f"warnings={issue_count}")
    return " ".join(parts)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    report_jsons: list[Path] = args.report_json
    stderr_logs: list[Path] = args.stderr_log
    if len(report_jsons) not in (1, 2):
        parser.error("--report-json accepts one path, or exactly two paths for compare mode")
    if len(stderr_logs) > len(report_jsons):
        parser.error("--stderr-log accepts at most one path per --report-json input")

    try:
        if len(report_jsons) == 2:
            session_or_compare: InspectSession | CompareSession = build_compare_session(
                report_jsons[0],
                report_jsons[1],
                stderr_logs[0] if len(stderr_logs) >= 1 else None,
                stderr_logs[1] if len(stderr_logs) >= 2 else None,
            )
        else:
            session_or_compare = build_session(
                report_jsons[0],
                stderr_logs[0] if stderr_logs else None,
            )
    except ArtifactParseError as exc:
        print(f"lis_inspect: {exc}", file=sys.stderr)
        return 2

    if args.no_tui:
        if isinstance(session_or_compare, CompareSession):
            print(_compare_summary_line(session_or_compare))
            for warning in session_or_compare.left.warnings:
                print(f"left warning: {warning}", file=sys.stderr)
            for warning in session_or_compare.right.warnings:
                print(f"right warning: {warning}", file=sys.stderr)
            for warning in session_or_compare.warnings:
                print(f"compare warning: {warning}", file=sys.stderr)
        else:
            print(_summary_line(session_or_compare))
            for warning in session_or_compare.warnings:
                print(f"warning: {warning}", file=sys.stderr)
        return 0

    # Defer Textual import so --no-tui works without textual installed.
    try:
        from .app import run_app
    except ImportError as exc:
        print(
            "lis_inspect: textual is required for the TUI.\n"
            "    pip install -r tools/requirements.txt\n"
            f"    (import error: {exc})",
            file=sys.stderr,
        )
        return 3

    if isinstance(session_or_compare, CompareSession):
        return run_app(
            session_or_compare,
            report_json=(report_jsons[0], report_jsons[1]),
            stderr_log=(
                stderr_logs[0] if len(stderr_logs) >= 1 else None,
                stderr_logs[1] if len(stderr_logs) >= 2 else None,
            ),
        )
    return run_app(
        session_or_compare,
        report_json=report_jsons[0],
        stderr_log=stderr_logs[0] if stderr_logs else None,
    )
