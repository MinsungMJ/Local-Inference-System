"""Model-free M1 product-spine acceptance probe.

This is deliberately not a product mode adapter.  It injects synthetic frozen
report fixtures to verify lifecycle plumbing under a manifest-bound acceptance
classification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lis_verify.ledger import load_ledger
from lis_verify.orchestrator import (
    AcceptanceManifest,
    CommandRequest,
    run_orchestration,
)
from lis_verify.product_contract import CustomerVerdict, WorkflowClassification
from lis_verify.report_artifact import load_report
from lis_verify.summary import render_markdown, render_terminal
from tools.tests.lis_verify_product_spine_test_support import (
    drive_pre_cleanup_stages,
    load_example,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--dependency-sha256", required=True)
    parser.add_argument("--commands-sha256", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = AcceptanceManifest(
        source_revision=args.source_revision,
        source_tree_sha256=args.source_tree_sha256,
        dependency_sha256=args.dependency_sha256,
        commands_sha256=args.commands_sha256,
        clean_state_observed=True,
    )
    attempt_ids: set[str] = set()
    names = ("pass", "regression", "inconclusive", "unsupported", "harness_error")
    for name in names:
        def runner(context, fixture=name):
            return drive_pre_cleanup_stages(context, load_example(fixture))

        result = run_orchestration(
            CommandRequest(
                mode="demo",
                output_root=args.out,
                require_supported=(name == "unsupported"),
                workflow=WorkflowClassification.VERIFICATION_ACCEPTANCE,
                acceptance_manifest=manifest,
            ),
            runner,
        )
        if result.report.verdict != CustomerVerdict(name.upper()):
            raise RuntimeError("probe verdict mismatch")
        if load_report(result.report_path) != result.report:
            raise RuntimeError("published report mismatch")
        if result.summary_path.read_text() != render_markdown(result.report):
            raise RuntimeError("Markdown renderer mismatch")
        if result.terminal_summary != render_terminal(result.report):
            raise RuntimeError("terminal renderer mismatch")
        events = load_ledger(result.report_path.parent / "attempt.jsonl")
        if events[-1]["payload"]["verdict"] != result.report.verdict.value:
            raise RuntimeError("ledger verdict mismatch")
        if (result.report_path.parent / "runtime").exists():
            raise RuntimeError("unreported runtime residue")
        attempt_ids.add(result.report.attempt_id)

    if len(attempt_ids) != len(names):
        raise RuntimeError("acceptance probe reused an attempt identity")
    marker = {
        "attempts": len(attempt_ids),
        "states": ["executed", "not_applicable", "blocked", "failed"],
        "verdicts": [value.value for value in CustomerVerdict],
    }
    print("M1_ACCEPTANCE_PROBE_PASS " + json.dumps(marker, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
