"""Strict customer CLI surface for LIS Verify."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Mapping, Sequence

from . import __version__
from .orchestrator import CommandRequest, OrchestrationResult
from .product_contract import (
    CLI_DEFAULTS,
    MAX_STAGE_TIMEOUT_SECONDS,
    WorkflowClassification,
)


RunnerRegistry = Mapping[str, object]
DEFAULT_RUNNERS: dict[str, object] = {}


class BoundedArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        del message
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: error: invalid arguments\n")


def _bounded_timeout(value: str) -> int:
    try:
        number = int(value, 10)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("timeout must be an integer") from exc
    if number <= 0 or number > MAX_STAGE_TIMEOUT_SECONDS:
        raise argparse.ArgumentTypeError(
            f"timeout must be between 1 and {MAX_STAGE_TIMEOUT_SECONDS} seconds"
        )
    return number


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(CLI_DEFAULTS["out"]),
        help="Private verification output root (default: .lis/verify).",
    )
    parser.add_argument(
        "--require-supported",
        action="store_true",
        help="Exit 6 instead of 0 when the semantic verdict is UNSUPPORTED.",
    )
    parser.add_argument(
        "--debug-retain",
        action="store_true",
        help="Retain bounded attempt diagnostics and report retained residue.",
    )
    parser.add_argument(
        "--stage-timeout-seconds",
        type=_bounded_timeout,
        default=CLI_DEFAULTS.get("stage_timeout_seconds", 1800),
        metavar="SECONDS",
        help="Per-stage timeout, from 1 through 7200 seconds.",
    )
    parser.add_argument("--verbose", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = BoundedArgumentParser(
        prog="lis-verify",
        description="Offline, report-driven verification for LIS.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    demo = subparsers.add_parser("demo", help="Run the seeded model-free workflow.")
    _add_common_options(demo)

    backend = subparsers.add_parser(
        "backend",
        help="Compare reference and resolved optimized backends.",
    )
    backend.add_argument("--model", required=True, type=Path)
    _add_common_options(backend)

    runtime = subparsers.add_parser(
        "runtime",
        help="Compare two explicitly identified LIS binaries.",
    )
    runtime.add_argument("--reference-bin", required=True, type=Path)
    runtime.add_argument("--candidate-bin", required=True, type=Path)
    runtime.add_argument("--model", required=True, type=Path)
    _add_common_options(runtime)
    return parser


def parse_command(argv: Sequence[str] | None = None) -> CommandRequest:
    args = build_parser().parse_args(argv)
    return CommandRequest(
        mode=args.mode,
        output_root=args.out,
        require_supported=args.require_supported,
        debug_retain=args.debug_retain,
        stage_timeout_seconds=args.stage_timeout_seconds,
        verbose=args.verbose,
        model=getattr(args, "model", None),
        reference_bin=getattr(args, "reference_bin", None),
        candidate_bin=getattr(args, "candidate_bin", None),
        workflow=WorkflowClassification.DEVELOPMENT_DEBUGGING,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    runners: RunnerRegistry | None = None,
) -> int:
    request = parse_command(argv)
    registry = DEFAULT_RUNNERS if runners is None else runners
    runner = registry.get(request.mode)
    if runner is None:
        print(
            f"lis-verify: {request.mode} workflow is not available in this milestone",
            file=sys.stderr,
        )
        return 2
    try:
        result = runner(request)
    except Exception:
        print("lis-verify: verification orchestration failed closed", file=sys.stderr)
        return 2
    if not isinstance(result, OrchestrationResult):
        raise TypeError("mode runner returned an invalid result")
    print(result.terminal_summary, end="")
    return result.exit_code
