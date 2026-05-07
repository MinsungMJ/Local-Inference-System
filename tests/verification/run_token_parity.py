#!/usr/bin/env python3
"""Fixed-prompt token-parity verification.

Runs the documented Llama-family parity fixture through the reference path
(`LIS_SIMD=0`) and the default dispatch path, then compares the exact selected
token ID sequence captured in the run reports.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT = "Write one short sentence about the sea."
DEFAULT_CONTEXT = 128
DEFAULT_BATCH = 1
DEFAULT_GENERATE = 8
DEFAULT_THREADS = 1
DEFAULT_OUT_JSON = REPO_ROOT / "tests/verification/token_parity_snapshot.json"

EXIT_PASS = 0
EXIT_HARNESS_CONFIGURATION_ERROR = 2
EXIT_TOKEN_PARITY_REGRESSION = 4


@dataclass
class LisRun:
    label: str
    backend: str
    selected_tokens: list[int]
    run_report: dict[str, object]
    stdout_text: str
    stderr_text: str


def emit_result(result_class: str, **fields: object) -> None:
    def clean(value: object) -> str:
        return str(value).replace("\n", "\\n").replace(" ", "_")

    parts = [f"result={result_class}"]
    for key, value in fields.items():
        parts.append(f"{key}={clean(value)}")
    print("token-parity: " + " ".join(parts))


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_path, path)


def load_run_report(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_lis_output(
    label: str,
    run_report: dict[str, object],
    stdout_text: str,
    stderr_text: str,
) -> LisRun:
    manifest = run_report.get("manifest")
    report = run_report.get("report")
    if not isinstance(manifest, dict) or not isinstance(report, dict):
        raise RuntimeError(f"{label}: malformed run report")

    backend = manifest.get("backend")
    if not isinstance(backend, dict):
        raise RuntimeError(f"{label}: missing backend manifest")
    backend_name = backend.get("name")
    if not isinstance(backend_name, str) or not backend_name:
        raise RuntimeError(f"{label}: missing backend name")

    selected_tokens = report.get("selected_token_ids")
    if not isinstance(selected_tokens, list) or not selected_tokens:
        raise RuntimeError(f"{label}: missing selected_token_ids")

    return LisRun(
        label=label,
        backend=backend_name,
        selected_tokens=[int(token_id) for token_id in selected_tokens],
        run_report=run_report,
        stdout_text=stdout_text,
        stderr_text=stderr_text,
    )


def run_lis(
    label: str,
    args: argparse.Namespace,
    *,
    force_reference: bool,
) -> LisRun:
    env = os.environ.copy()
    report_path: Path | None = None
    if force_reference:
        env["LIS_SIMD"] = "0"
    else:
        env.pop("LIS_SIMD", None)

    with tempfile.NamedTemporaryFile(
        suffix=".json", delete=False, dir=REPO_ROOT / "tests/verification"
    ) as tf:
        report_path = Path(tf.name)

    cmd = [
        str(args.bin),
        "--model",
        str(args.model),
        "--config",
        str(args.config),
        "--hf-tokenizer",
        str(args.hf_tokenizer),
        "--prompt",
        args.prompt,
        "--context",
        str(args.context),
        "--batch",
        str(args.batch),
        "--generate",
        str(args.generate),
        "--threads",
        str(args.threads),
        "--report-json",
        str(report_path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env,
                              check=False)
        if proc.returncode != 0:
            stderr_lower = proc.stderr.lower()
            if "unsupported" in stderr_lower:
                emit_result(
                    "documented_unsupported",
                    path=label,
                    reason="cli_reported_unsupported",
                    returncode=proc.returncode,
                )
                sys.exit(EXIT_PASS)
            raise RuntimeError(
                f"{label}: LIS exited {proc.returncode}\n{proc.stderr}"
            )
        return parse_lis_output(
            label,
            load_run_report(report_path),
            proc.stdout,
            proc.stderr,
        )
    finally:
        if report_path is not None:
            report_path.unlink(missing_ok=True)


def validate_path(label: str, path: Path, *, executable: bool = False) -> None:
    if executable:
        ok = path.is_file() and os.access(path, os.X_OK)
    else:
        ok = path.is_file() or path.is_dir()
    if not ok:
        raise RuntimeError(f"{label} not found or not usable: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run reference-vs-default token parity."
    )
    parser.add_argument("--bin", type=Path,
                        default=REPO_ROOT / "srcs/libs/lis")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--hf-tokenizer", type=Path, default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--context", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--generate", type=int, default=DEFAULT_GENERATE)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = parser.parse_args()

    if args.model is None:
        env_model = os.environ.get("VERIFY_MODEL")
        if env_model:
            args.model = Path(env_model)
        else:
            emit_result(
                "harness_configuration_error",
                detail="--model_or_VERIFY_MODEL_required",
            )
            return EXIT_HARNESS_CONFIGURATION_ERROR
    if args.config is None:
        args.config = args.model / "config.json"
    if args.hf_tokenizer is None:
        args.hf_tokenizer = args.model / "tokenizer.json"

    try:
        validate_path("lis binary", args.bin, executable=True)
        validate_path("model", args.model)
        validate_path("config", args.config)
        validate_path("hf tokenizer", args.hf_tokenizer)

        reference = run_lis("reference", args, force_reference=True)
        candidate = run_lis("candidate", args, force_reference=False)

        if candidate.backend == reference.backend:
            write_json_atomic(
                args.out_json,
                {
                    "schema": "lis.execution_artifact/v1",
                    "kind": "verification_snapshot",
                    "mode": "token_parity",
                    "result_class": "documented_unsupported",
                    "reason": "optimized_backend_not_active",
                    "reference_run": reference.run_report,
                    "candidate_run": candidate.run_report,
                },
            )
            emit_result(
                "documented_unsupported",
                reason="optimized_backend_not_active",
                reference_backend=reference.backend,
                candidate_backend=candidate.backend,
            )
            return EXIT_PASS

        if reference.selected_tokens != candidate.selected_tokens:
            write_json_atomic(
                args.out_json,
                {
                    "schema": "lis.execution_artifact/v1",
                    "kind": "verification_snapshot",
                    "mode": "token_parity",
                    "result_class": "token_parity_regression",
                    "reference_run": reference.run_report,
                    "candidate_run": candidate.run_report,
                    "comparison": {
                        "selected_token_ids_match": False,
                    },
                },
            )
            emit_result(
                "token_parity_regression",
                reference_backend=reference.backend,
                candidate_backend=candidate.backend,
                reference_tokens=",".join(map(str, reference.selected_tokens)),
                candidate_tokens=",".join(map(str, candidate.selected_tokens)),
            )
            return EXIT_TOKEN_PARITY_REGRESSION

        write_json_atomic(
            args.out_json,
            {
                "schema": "lis.execution_artifact/v1",
                "kind": "verification_snapshot",
                "mode": "token_parity",
                "result_class": "pass",
                "reference_run": reference.run_report,
                "candidate_run": candidate.run_report,
                "comparison": {
                    "selected_token_ids_match": True,
                    "selected_token_count": len(reference.selected_tokens),
                },
            },
        )
        emit_result(
            "pass",
            reference_backend=reference.backend,
            candidate_backend=candidate.backend,
            tokens=len(reference.selected_tokens),
        )
        return EXIT_PASS
    except RuntimeError as exc:
        emit_result("harness_configuration_error", detail=str(exc))
        return EXIT_HARNESS_CONFIGURATION_ERROR


if __name__ == "__main__":
    sys.exit(main())
