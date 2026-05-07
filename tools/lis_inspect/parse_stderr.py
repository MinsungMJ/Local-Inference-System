"""stderr parser for `lis: perf-stage`, `lis: perf-summary`, `lis: perf-per-token`.

The parser is conservative: it ignores unrelated stderr lines, collects warnings
for malformed matching lines, and never invents metrics from partial tokens.

Line formats (from docs/performance_measurement.md):

    lis: perf-stage tag=<TAG> name=<NAME> ns=<N> ms=<M> tokens=<T>
    lis: perf-summary tag=<TAG> threads=<T> prompt_tokens=<P> generated_tokens=<G> \
         ttft_ms=<M> itl_ms=<M> tps_steady=<F> tps_end_to_end=<F>
    lis: perf-per-token tag=<TAG> step=<N> ns=<N> ms=<M>
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .models import PerfLog, PerfPerToken, PerfStage, PerfSummary

PERF_STAGE_PREFIX = "lis: perf-stage "
PERF_SUMMARY_PREFIX = "lis: perf-summary "
PERF_PER_TOKEN_PREFIX = "lis: perf-per-token "


def _kv_pairs(remainder: str) -> dict[str, str]:
    """Split a `key=value key=value` tail into a dict.

    Whitespace-separated tokens. Values without `=` are skipped.
    Empty input returns an empty dict.
    """
    out: dict[str, str] = {}
    for token in remainder.split():
        if "=" not in token:
            continue
        key, _, value = token.partition("=")
        if key:
            out[key] = value
    return out


def _parse_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_perf_stage(remainder: str, log: PerfLog) -> None:
    fields = _kv_pairs(remainder)
    name = fields.get("name")
    ns = _parse_int(fields.get("ns", ""))
    ms = _parse_float(fields.get("ms", ""))
    tokens = _parse_int(fields.get("tokens", "0"))
    if name is None or ns is None or ms is None or tokens is None:
        log.warnings.append(f"malformed perf-stage line: {remainder!r}")
        return
    if log.tag is None and "tag" in fields:
        log.tag = fields["tag"]
    log.stages[name] = PerfStage(name=name, ns=ns, ms=ms, tokens=tokens)


def _parse_perf_summary(remainder: str, log: PerfLog) -> None:
    fields = _kv_pairs(remainder)
    threads = _parse_int(fields.get("threads", ""))
    prompt_tokens = _parse_int(fields.get("prompt_tokens", ""))
    generated_tokens = _parse_int(fields.get("generated_tokens", ""))
    ttft_ms = _parse_float(fields.get("ttft_ms", ""))
    itl_ms = _parse_float(fields.get("itl_ms", ""))
    tps_steady = _parse_float(fields.get("tps_steady", ""))
    tps_end_to_end = _parse_float(fields.get("tps_end_to_end", ""))

    required = (
        threads,
        prompt_tokens,
        generated_tokens,
        ttft_ms,
        itl_ms,
        tps_steady,
        tps_end_to_end,
    )
    if any(value is None for value in required):
        log.warnings.append(f"malformed perf-summary line: {remainder!r}")
        return
    if log.tag is None and "tag" in fields:
        log.tag = fields["tag"]
    log.summary = PerfSummary(
        threads=threads,  # type: ignore[arg-type]
        prompt_tokens=prompt_tokens,  # type: ignore[arg-type]
        generated_tokens=generated_tokens,  # type: ignore[arg-type]
        ttft_ms=ttft_ms,  # type: ignore[arg-type]
        itl_ms=itl_ms,  # type: ignore[arg-type]
        tps_steady=tps_steady,  # type: ignore[arg-type]
        tps_end_to_end=tps_end_to_end,  # type: ignore[arg-type]
    )


def _parse_perf_per_token(remainder: str, log: PerfLog) -> None:
    fields = _kv_pairs(remainder)
    step = _parse_int(fields.get("step", ""))
    ns = _parse_int(fields.get("ns", ""))
    ms = _parse_float(fields.get("ms", ""))
    if step is None or ns is None or ms is None:
        log.warnings.append(f"malformed perf-per-token line: {remainder!r}")
        return
    if log.tag is None and "tag" in fields:
        log.tag = fields["tag"]
    log.per_token.append(PerfPerToken(step=step, ns=ns, ms=ms))


def parse_stderr_text(text: str) -> PerfLog:
    """Parse `text` and return a `PerfLog`. Non-matching lines are ignored."""
    log = PerfLog()
    for line in text.splitlines():
        stripped = line.rstrip()
        if stripped.startswith(PERF_STAGE_PREFIX):
            _parse_perf_stage(stripped[len(PERF_STAGE_PREFIX):], log)
        elif stripped.startswith(PERF_SUMMARY_PREFIX):
            _parse_perf_summary(stripped[len(PERF_SUMMARY_PREFIX):], log)
        elif stripped.startswith(PERF_PER_TOKEN_PREFIX):
            _parse_perf_per_token(stripped[len(PERF_PER_TOKEN_PREFIX):], log)
    return log


def load_stderr_log(path: str | Path) -> PerfLog:
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        log = PerfLog()
        log.warnings.append("stderr-log not found")
        return log
    except OSError as exc:
        log = PerfLog()
        log.warnings.append(f"stderr-log unreadable: {exc.strerror or exc.__class__.__name__}")
        return log
    return parse_stderr_text(text)
