#!/usr/bin/env python3
"""Run the Qwen3 dense first-target sanity check.

This is a model-backed, opt-in gate driven by an explicit user-supplied model
path. It is not part of the routine no-model `make verify` target. The script
uses direct token IDs to avoid chat-template/Jinja scope and validates that
the Qwen3 family path emits deterministic diagnostic evidence plus a bounded
run report for one short generation step.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def write_snapshot(
    path: Path,
    result_class: str,
    run_report: dict[str, object],
    checks: dict[str, bool],
    reason: str | None = None,
) -> None:
    payload: dict[str, object] = {
        "schema": "lis.execution_artifact/v1",
        "kind": "verification_snapshot",
        "mode": "qwen3_sanity",
        "result_class": result_class,
        "checks": checks,
        "run": run_report,
    }
    if reason is not None:
        payload["reason"] = reason
    write_json_atomic(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin", required=True, help="Path to LIS binary")
    parser.add_argument("--model", required=True, help="Qwen3 model directory")
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--token-id", type=int, default=151643)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "tests/verification/qwen3_sanity_snapshot.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bin_path = Path(args.bin)
    model_dir = Path(args.model)
    config_path = model_dir / "config.json"

    if not bin_path.is_file():
        print(
            f"qwen3-dense-sanity: result=harness_configuration_error "
            f"reason=missing_bin path={bin_path}",
            file=sys.stderr,
        )
        return 2
    if not model_dir.is_dir() or not config_path.is_file():
        print(
            f"qwen3-dense-sanity: result=harness_configuration_error "
            f"reason=missing_model path={model_dir}",
            file=sys.stderr,
        )
        return 2

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tf:
        token_path = Path(tf.name)
        tf.write(f"{args.token_id}\n")
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", delete=False, suffix=".json"
    ) as rf:
        report_path = Path(rf.name)

    cmd = [
        str(bin_path),
        "--model",
        str(model_dir),
        "--config",
        str(config_path),
        "--tokens",
        str(token_path),
        "--context",
        str(args.context),
        "--batch",
        "1",
        "--generate",
        "1",
        "--report-json",
        str(report_path),
        "--diagnostics",
        "--layer-checkpoints",
        "0",
    ]
    try:
        run = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired:
        token_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        print(
            f"qwen3-dense-sanity: result=harness_configuration_error "
            f"reason=timeout seconds={args.timeout}",
            file=sys.stderr,
        )
        return 2
    finally:
        token_path.unlink(missing_ok=True)

    if run.returncode != 0:
        report_path.unlink(missing_ok=True)
        print(
            "qwen3-dense-sanity: result=numeric_regression "
            f"reason=lis_failed exit={run.returncode}",
            file=sys.stderr,
        )
        sys.stderr.write(run.stderr)
        return 1
    run_report = json.loads(report_path.read_text(encoding="utf-8"))
    report_path.unlink(missing_ok=True)
    if "generated_token_ids:" not in run.stdout:
        write_snapshot(
            args.out_json,
            "token_parity_regression",
            run_report,
            {
                "stdout_generated_token_ids": False,
                "backend_observability": "lis: simd backend=" in run.stderr,
                "qwen3_checkpoint_diagnostic": "qwen3.logits" in run.stderr,
            },
            "missing_generated_token_ids",
        )
        print(
            "qwen3-dense-sanity: result=token_parity_regression "
            "reason=missing_generated_token_ids",
            file=sys.stderr,
        )
        sys.stderr.write(run.stderr)
        return 1
    if "lis: simd backend=" not in run.stderr:
        write_snapshot(
            args.out_json,
            "benchmark_protocol_regression",
            run_report,
            {
                "stdout_generated_token_ids": True,
                "backend_observability": False,
                "qwen3_checkpoint_diagnostic": "qwen3.logits" in run.stderr,
            },
            "missing_backend_observability",
        )
        print(
            "qwen3-dense-sanity: result=benchmark_protocol_regression "
            "reason=missing_backend_observability",
            file=sys.stderr,
        )
        sys.stderr.write(run.stderr)
        return 1
    if "qwen3.logits" not in run.stderr:
        write_snapshot(
            args.out_json,
            "numeric_regression",
            run_report,
            {
                "stdout_generated_token_ids": True,
                "backend_observability": True,
                "qwen3_checkpoint_diagnostic": False,
            },
            "missing_qwen3_checkpoint",
        )
        print(
            "qwen3-dense-sanity: result=numeric_regression "
            "reason=missing_qwen3_checkpoint",
            file=sys.stderr,
        )
        sys.stderr.write(run.stderr)
        return 1

    generated = run.stdout.strip().replace("\n", " ")
    write_snapshot(
        args.out_json,
        "pass",
        run_report,
        {
            "stdout_generated_token_ids": True,
            "backend_observability": True,
            "qwen3_checkpoint_diagnostic": True,
        },
    )
    print(
        "qwen3-dense-sanity: result=pass "
        f"artifact={model_dir} token_id={args.token_id} {generated}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
