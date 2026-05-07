#!/usr/bin/env python3
"""Performance sweep harness.

Drives the LIS CLI over a matrix of:
    threads      in {1, nproc}
    prompt_size  in {short, medium, long}
    generate     in {16, 64, 256}

Per cell: 1 warm-up run (discarded) + N measured runs (default 3). The median
summary metrics across measured runs are written to tests/perf/results.csv,
a grouped markdown table is written to tests/perf/results.md, and a bounded
benchmark snapshot JSON is written to tests/perf/results.json.

Lines of interest in LIS stderr:
    lis: simd backend=<BACKEND>
    lis: perf-stage tag=<TAG> name=<STAGE> ns=<N> ms=<M> tokens=<T>
    lis: perf-summary tag=<TAG> threads=<T> prompt_tokens=<P> ...
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

STAGE_LINE_RE = re.compile(
    r"^lis:\s+perf-stage\s+(?P<fields>.*)$"
)
SUMMARY_LINE_RE = re.compile(
    r"^lis:\s+perf-summary\s+(?P<fields>.*)$"
)
BACKEND_LINE_RE = re.compile(
    r"^lis:\s+simd\s+backend=(?P<backend>[^\s]+)\s*$"
)
KEY_VALUE_RE = re.compile(r"(\w+)=([^\s]+)")


def parse_kv(text: str) -> dict[str, str]:
    return dict(KEY_VALUE_RE.findall(text))


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    replace_path(tmp_path, path)


def parse_perf_output(stderr_text: str) -> dict | None:
    """Return a dict of aggregated metrics, or None if summary not found."""
    stages: dict[str, dict[str, str]] = {}
    summary: dict[str, str] | None = None
    backend = ""

    for line in stderr_text.splitlines():
        m = BACKEND_LINE_RE.match(line)
        if m:
            backend = m.group("backend")
            continue
        m = STAGE_LINE_RE.match(line)
        if m:
            kv = parse_kv(m.group("fields"))
            name = kv.get("name")
            if name is not None:
                stages[name] = kv
            continue
        m = SUMMARY_LINE_RE.match(line)
        if m:
            summary = parse_kv(m.group("fields"))

    if summary is None:
        return None

    def stage_ms(name: str) -> float:
        entry = stages.get(name, {})
        return float(entry.get("ms", "0"))

    return {
        "backend": backend,
        "threads": int(summary.get("threads", "0")),
        "prompt_tokens": int(summary.get("prompt_tokens", "0")),
        "generated_tokens": int(summary.get("generated_tokens", "0")),
        "ttft_ms": float(summary.get("ttft_ms", "0")),
        "itl_ms": float(summary.get("itl_ms", "0")),
        "tps_steady": float(summary.get("tps_steady", "0")),
        "tps_end_to_end": float(summary.get("tps_end_to_end", "0")),
        "prefill_ms": stage_ms("prefill"),
        "first_decode_ms": stage_ms("first_decode"),
        "decode_steady_state_ms": stage_ms("decode_steady_state"),
        "model_load_ms": stage_ms("model_load"),
        "tokenizer_load_ms": stage_ms("tokenizer_load"),
        "tokenizer_encode_ms": stage_ms("tokenizer_encode"),
        "runtime_init_ms": stage_ms("runtime_init"),
    }


def median_of(records: list[dict]) -> dict:
    numeric_keys = [k for k, v in records[0].items() if isinstance(v, (int, float))]
    med: dict = {}
    for k in numeric_keys:
        vals = [r[k] for r in records]
        med[k] = statistics.median(vals)
    return med


def run_once(
    bin_path: Path,
    model: Path,
    config: Path,
    hf_tokenizer: Path,
    prompt_file: Path,
    context: int,
    generate: int,
    threads: int,
    tag: str,
) -> dict:
    prompt_text = prompt_file.read_text(encoding="utf-8").strip()
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, suffix=".json"
    ) as rf:
        report_path = Path(rf.name)
    cmd = [
        str(bin_path),
        "--model", str(model),
        "--config", str(config),
        "--hf-tokenizer", str(hf_tokenizer),
        "--prompt", prompt_text,
        "--context", str(context),
        "--batch", "1",
        "--generate", str(generate),
        "--threads", str(threads),
        "--perf",
        "--perf-tag", tag,
        "--report-json", str(report_path),
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, check=False
        )
        if proc.returncode != 0:
            sys.stderr.write(
                f"[harness] LIS exited {proc.returncode} for tag={tag}\n"
            )
            sys.stderr.write(proc.stderr)
            raise RuntimeError(f"LIS failed for tag={tag}")
        metrics = parse_perf_output(proc.stderr)
        if metrics is None:
            raise RuntimeError(
                f"Could not find perf-summary line for tag={tag}.\n"
                f"stderr:\n{proc.stderr}"
            )
        metrics["run_report"] = json.loads(
            report_path.read_text(encoding="utf-8")
        )
        return metrics
    finally:
        report_path.unlink(missing_ok=True)


def replace_path(tmp_path: Path, final_path: Path) -> None:
    os.replace(tmp_path, final_path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = [
        "tag",
        "backend",
        "threads",
        "prompt_size",
        "generate",
        "prompt_tokens",
        "generated_tokens",
        "ttft_ms",
        "itl_ms",
        "tps_steady",
        "tps_end_to_end",
        "prefill_ms",
        "first_decode_ms",
        "decode_steady_state_ms",
        "model_load_ms",
        "tokenizer_load_ms",
        "tokenizer_encode_ms",
        "runtime_init_ms",
        "ttft_ms_min",
        "ttft_ms_max",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    replace_path(tmp_path, path)


def write_markdown(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    lines = ["# LIS Performance Sweep", ""]
    for prompt_size in sorted({r["prompt_size"] for r in rows}):
        lines.append(f"## Prompt: {prompt_size}")
        lines.append("")
        lines.append(
            "| backend | threads | generate | prompt_tokens | ttft_ms | itl_ms | "
            "tps_steady | tps_end_to_end | prefill_ms | first_decode_ms |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        subset = sorted(
            [r for r in rows if r["prompt_size"] == prompt_size],
            key=lambda r: (r["threads"], r["generate"]),
        )
        for r in subset:
            lines.append(
                f"| {r.get('backend', '')} | "
                f"{r['threads']} | {r['generate']} | {r['prompt_tokens']} | "
                f"{r['ttft_ms']:.2f} | {r['itl_ms']:.2f} | "
                f"{r['tps_steady']:.2f} | {r['tps_end_to_end']:.2f} | "
                f"{r['prefill_ms']:.2f} | {r['first_decode_ms']:.2f} |"
            )
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    replace_path(tmp_path, path)


def write_json_snapshot(path: Path, cells: list[dict[str, object]]) -> None:
    write_json_atomic(
        path,
        {
            "schema": "lis.execution_artifact/v1",
            "kind": "benchmark_snapshot",
            "mode": "perf_matrix",
            "cells": cells,
        },
    )


def write_checkpoint(
    out_csv: Path,
    out_md: Path,
    out_json: Path,
    rows: list[dict],
    cells: list[dict[str, object]],
) -> None:
    write_csv(out_csv, rows)
    write_markdown(out_md, rows)
    write_json_snapshot(out_json, cells)
    sys.stderr.write(
        f"[harness] checkpoint rows={len(rows)} csv={out_csv} md={out_md} "
        f"json={out_json}\n"
    )
    sys.stderr.flush()


def log_progress(cell: int, total: int, tag: str, phase: str) -> None:
    sys.stderr.write(f"[harness] cell {cell}/{total} tag={tag} {phase}\n")
    sys.stderr.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="LIS perf sweep")
    parser.add_argument("--bin", default=str(REPO_ROOT / "srcs/libs/lis"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--hf-tokenizer", default=None)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument(
        "--threads",
        nargs="*",
        type=int,
        default=None,
        help="thread counts to sweep (default: 1, nproc)",
    )
    parser.add_argument(
        "--generates",
        nargs="*",
        type=int,
        default=[16, 64, 256],
    )
    parser.add_argument(
        "--prompt-sizes",
        nargs="*",
        default=["short", "medium", "long"],
    )
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument(
        "--out-csv",
        default=str(REPO_ROOT / "tests/perf/results.csv"),
    )
    parser.add_argument(
        "--out-md",
        default=str(REPO_ROOT / "tests/perf/results.md"),
    )
    parser.add_argument(
        "--out-json",
        default=str(REPO_ROOT / "tests/perf/results.json"),
    )
    args = parser.parse_args()

    bin_path = Path(args.bin)
    model = Path(args.model)
    config = Path(args.config) if args.config else model / "config.json"
    hf_tok = (
        Path(args.hf_tokenizer) if args.hf_tokenizer else model / "tokenizer.json"
    )
    prompts_dir = REPO_ROOT / "tests/perf/prompts"

    if not bin_path.is_file():
        sys.stderr.write(f"[harness] lis binary not found: {bin_path}\n")
        return 2

    thread_list = args.threads or [1, multiprocessing.cpu_count()]
    # De-duplicate while preserving order.
    seen: set[int] = set()
    thread_list = [t for t in thread_list if not (t in seen or seen.add(t))]

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_json = Path(args.out_json)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    cells: list[dict[str, object]] = []
    total = len(thread_list) * len(args.prompt_sizes) * len(args.generates)
    cell = 0
    for threads in thread_list:
        for prompt_size in args.prompt_sizes:
            prompt_file = prompts_dir / f"{prompt_size}.txt"
            for generate in args.generates:
                cell += 1
                tag = f"{prompt_size}/threads={threads}/gen={generate}"
                # Warm-up (discarded).
                log_progress(cell, total, tag, "warmup 1/1")
                run_once(
                    bin_path, model, config, hf_tok, prompt_file,
                    args.context, generate, threads, tag + "/warmup",
                )
                measured: list[dict] = []
                for run_idx in range(args.measured_runs):
                    log_progress(
                        cell,
                        total,
                        tag,
                        f"measured {run_idx + 1}/{args.measured_runs}",
                    )
                    metrics = run_once(
                        bin_path, model, config, hf_tok, prompt_file,
                        args.context, generate, threads,
                        f"{tag}/run{run_idx}",
                    )
                    measured.append(metrics)
                median = median_of(measured)
                backends = {m.get("backend", "") for m in measured}
                if len(backends) != 1:
                    raise RuntimeError(
                        f"backend/path changed across runs for tag={tag}: "
                        f"{sorted(backends)}"
                    )
                row = dict(median)
                row["tag"] = tag
                row["backend"] = measured[0].get("backend", "")
                row["threads"] = threads
                row["prompt_size"] = prompt_size
                row["generate"] = generate
                row["ttft_ms_min"] = min(m["ttft_ms"] for m in measured)
                row["ttft_ms_max"] = max(m["ttft_ms"] for m in measured)
                rows.append(row)
                cells.append(
                    {
                        "tag": tag,
                        "backend": row["backend"],
                        "threads": threads,
                        "prompt_size": prompt_size,
                        "generate": generate,
                        "measured_runs": len(measured),
                        "summary": dict(row),
                        "representative_run": measured[0]["run_report"],
                    }
                )
                write_checkpoint(out_csv, out_md, out_json, rows, cells)

    sys.stderr.write(f"[harness] wrote {out_csv}\n")
    sys.stderr.write(f"[harness] wrote {out_md}\n")
    sys.stderr.write(f"[harness] wrote {out_json}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
