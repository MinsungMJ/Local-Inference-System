"""Fail-closed CI consumption of canonical LIS Verify attempt artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import sys
from typing import Sequence

from .acceptance import (
    AcceptanceManifestError,
    load_acceptance_manifest,
    verify_acceptance_source,
)
from .golden import GoldenManifest, load_manifest, verify_local_model
from .ledger import LedgerError, load_ledger
from .product_contract import (
    MAX_SUMMARY_BYTES,
    CustomerVerdict,
    ExecutionPolicy,
    WorkflowClassification,
    expected_exit_code,
)
from .provenance import hash_regular_file
from .report_artifact import ArtifactPublicationError, load_report
from .report_model import VerificationReport
from .summary import render_markdown


class CIGateError(ValueError):
    """CI evidence is missing, inconsistent, or does not meet the gate."""


def validate_report_result(
    report: VerificationReport,
    *,
    expected_verdict: CustomerVerdict,
    observed_exit: int,
    require_acceptance: bool = False,
) -> None:
    if isinstance(observed_exit, bool) or not isinstance(observed_exit, int):
        raise CIGateError("observed exit status is invalid")
    if report.verdict != expected_verdict:
        raise CIGateError("semantic verdict did not meet the CI expectation")
    expected_policy_exit = expected_exit_code(report.verdict, report.policy)
    if report.exit_code != expected_policy_exit or observed_exit != report.exit_code:
        raise CIGateError("process exit and canonical policy result disagree")
    if require_acceptance and report.workflow_classification != (
        WorkflowClassification.VERIFICATION_ACCEPTANCE
    ):
        raise CIGateError("CI acceptance result is not acceptance-classified")
    cleanup = report.cleanup
    if (
        cleanup.status.value != "success"
        or cleanup.residue_status.value != "none_observed"
        or cleanup.observed is not True
        or cleanup.retained_debug is not False
    ):
        raise CIGateError("CI result does not prove residue-free cleanup")


def validate_golden_report(
    report: VerificationReport, manifest: GoldenManifest
) -> None:
    raw = report.to_dict()
    golden = manifest.materialize()
    expected = golden["expected"]
    runtime = golden["runtime"]
    if (
        raw["schema"] != expected["report_schema"]
        or raw["report_version"] != expected["report_version"]
        or raw["command"]["mode"] != runtime["mode"]
        or raw["command"]["require_supported"] is not True
        or report.policy != ExecutionPolicy.REQUIRE_SUPPORTED
        or raw["verdict"] != expected["semantic_verdict"]
        or raw["policy_result"]["exit_code"] != expected["policy_exit_code"]
    ):
        raise CIGateError("golden report contract did not match the manifest")
    identities = raw["identities"]
    reference = identities["reference"]
    candidate = identities["candidate"]
    file_identities = {entry["path"]: entry["sha256"] for entry in golden["files"]}
    if any(
        identity["model_sha256"] != file_identities["model.safetensors"]
        or identity["config_sha256"] != file_identities["config.json"]
        or identity["input_sha256"] != golden["input"]["sha256"]
        for identity in (reference, candidate)
    ):
        raise CIGateError("golden report input identity disagrees with the manifest")
    for field in (
        "source_sha256",
        "binary_sha256",
        "model_sha256",
        "config_sha256",
        "input_sha256",
    ):
        if reference[field] != candidate[field]:
            raise CIGateError("backend comparison changed a shared identity")
    expected_candidate_identities = runtime["required_candidate_backend_identities"]
    if (
        reference["backend_sha256"]
        != runtime["required_reference_backend_sha256"]
        or candidate["backend_sha256"] not in expected_candidate_identities.values()
        or reference["backend_sha256"] == candidate["backend_sha256"]
    ):
        raise CIGateError("optimized backend fallback cannot satisfy the golden gate")
    stage_states = {stage["name"]: stage["state"] for stage in raw["stages"]}
    required_executed = {
        "preflight",
        "reference_original_execution",
        "candidate_original_execution",
        "pass0_calibration",
        "pass1_token_localization",
        "aggregation",
        "cleanup",
    }
    if any(stage_states[name] != "executed" for name in required_executed):
        raise CIGateError("golden PASS is missing a required executed stage")
    if raw["token_comparison"]["status"] != "equal":
        raise CIGateError("golden PASS lacks equal selected-token evidence")


def _read_summary(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise CIGateError("cannot open canonical Markdown summary") from exc
    try:
        info = os.fstat(fd)
        if (
            not stat.S_ISREG(info.st_mode)
            or stat.S_IMODE(info.st_mode) != 0o600
            or (hasattr(os, "getuid") and info.st_uid != os.getuid())
            or info.st_size > MAX_SUMMARY_BYTES
        ):
            raise CIGateError("canonical Markdown summary is not private or bounded")
        data = bytearray()
        while len(data) <= MAX_SUMMARY_BYTES:
            chunk = os.read(fd, min(65_536, MAX_SUMMARY_BYTES + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(fd)
        if (
            info.st_dev != after.st_dev
            or info.st_ino != after.st_ino
            or info.st_size != after.st_size
            or info.st_mtime_ns != after.st_mtime_ns
        ):
            raise CIGateError("canonical Markdown summary changed while reading")
        if len(data) > MAX_SUMMARY_BYTES:
            raise CIGateError("canonical Markdown summary exceeds its byte bound")
    finally:
        os.close(fd)
    try:
        return bytes(data).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CIGateError("canonical Markdown summary is not UTF-8") from exc


def _resolve_attempt(root: Path) -> Path:
    source = Path(root).absolute()
    current = Path(source.anchor)
    try:
        for part in source.parts[1:]:
            current = current / part
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise CIGateError("CI attempt root contains a symlink")
    except OSError as exc:
        raise CIGateError("CI attempt root is unavailable") from exc
    info = source.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise CIGateError("CI attempt root is unsafe")
    attempts = [path for path in source.iterdir() if path.name.startswith("attempt-")]
    if len(attempts) != 1:
        raise CIGateError("CI attempt root must contain exactly one attempt")
    attempt = attempts[0]
    attempt_info = attempt.lstat()
    if (
        stat.S_ISLNK(attempt_info.st_mode)
        or not stat.S_ISDIR(attempt_info.st_mode)
        or stat.S_IMODE(attempt_info.st_mode) != 0o700
        or (hasattr(os, "getuid") and attempt_info.st_uid != os.getuid())
    ):
        raise CIGateError("CI attempt directory is not private and owned")
    return attempt


@dataclass(frozen=True)
class CIGateResult:
    verdict: CustomerVerdict
    report_sha256: str
    summary_sha256: str
    ledger_sha256: str
    manifest_sha256: str | None
    acceptance_manifest_sha256: str | None
    step_summary: str


def validate_attempt(
    *,
    attempt_root: Path,
    expected_verdict: CustomerVerdict,
    observed_exit: int,
    golden_model: Path | None = None,
    acceptance_manifest: Path | None = None,
    source_root: Path | None = None,
) -> CIGateResult:
    attempt = _resolve_attempt(attempt_root)
    report_path = attempt / "verification_report.json"
    summary_path = attempt / "summary.md"
    ledger_path = attempt / "attempt.jsonl"
    report = load_report(report_path)
    require_acceptance = acceptance_manifest is not None
    acceptance_digest = None
    if acceptance_manifest is not None:
        if source_root is None:
            raise CIGateError("acceptance validation requires the frozen source root")
        authority = load_acceptance_manifest(acceptance_manifest)
        verify_acceptance_source(authority, source_root)
        acceptance_digest, _ = hash_regular_file(acceptance_manifest)
    validate_report_result(
        report,
        expected_verdict=expected_verdict,
        observed_exit=observed_exit,
        require_acceptance=require_acceptance,
    )
    summary = _read_summary(summary_path)
    if summary != render_markdown(report):
        raise CIGateError("Markdown summary is not the canonical report projection")
    events = load_ledger(ledger_path)
    if (
        events[0]["attempt_id"] != report.attempt_id
        or events[-1]["payload"] != {"verdict": report.verdict.value}
        or events[-1]["workflow_classification"]
        != report.workflow_classification.value
    ):
        raise CIGateError("attempt ledger and report identity disagree")

    manifest_digest = None
    manifest = None
    if golden_model is not None:
        manifest = load_manifest()
        material = verify_local_model(manifest, golden_model)
        manifest_digest = material.manifest_sha256
        validate_golden_report(report, manifest)

    report_digest, _ = hash_regular_file(report_path)
    summary_digest, _ = hash_regular_file(summary_path)
    ledger_digest, _ = hash_regular_file(ledger_path)
    lines = [
        "# LIS Verify CI Gate",
        "",
        f"- Verdict: `{report.verdict.value}`",
        f"- Policy exit: `{report.exit_code}`",
        f"- Attempt: `{report.attempt_id}`",
        f"- Workflow: `{report.workflow_classification.value}`",
        f"- Report SHA-256: `{report_digest}`",
        f"- Summary SHA-256: `{summary_digest}`",
        f"- Ledger SHA-256: `{ledger_digest}`",
    ]
    if manifest_digest is not None:
        lines.append(f"- Golden manifest SHA-256: `{manifest_digest}`")
    if acceptance_digest is not None:
        lines.append(f"- Acceptance manifest SHA-256: `{acceptance_digest}`")
    lines.extend(["", summary.rstrip(), ""])
    step_summary = "\n".join(lines)
    if len(step_summary.encode("utf-8")) > MAX_SUMMARY_BYTES:
        raise CIGateError("CI step summary exceeds its byte bound")
    return CIGateResult(
        verdict=report.verdict,
        report_sha256=report_digest,
        summary_sha256=summary_digest,
        ledger_sha256=ledger_digest,
        manifest_sha256=manifest_digest,
        acceptance_manifest_sha256=acceptance_digest,
        step_summary=step_summary,
    )


def _append_step_summary(path: Path, text: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags, 0o600)
    except OSError as exc:
        raise CIGateError("cannot open the CI step summary") from exc
    try:
        data = text.encode("utf-8")
        view = memoryview(data)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise CIGateError("CI step summary write made no progress")
            view = view[written:]
    finally:
        os.close(fd)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m lis_verify.ci_gate",
        description="Validate one canonical LIS Verify attempt for CI.",
    )
    parser.add_argument("--attempt-root", required=True, type=Path)
    parser.add_argument(
        "--expected-verdict",
        required=True,
        choices=[value.value for value in CustomerVerdict],
    )
    parser.add_argument("--observed-exit", required=True, type=int)
    parser.add_argument("--golden-model", type=Path)
    parser.add_argument("--acceptance-manifest", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--step-summary", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = validate_attempt(
            attempt_root=args.attempt_root,
            expected_verdict=CustomerVerdict(args.expected_verdict),
            observed_exit=args.observed_exit,
            golden_model=args.golden_model,
            acceptance_manifest=args.acceptance_manifest,
            source_root=args.source_root,
        )
        if args.step_summary is not None:
            _append_step_summary(args.step_summary, result.step_summary)
    except (
        AcceptanceManifestError,
        ArtifactPublicationError,
        CIGateError,
        LedgerError,
        OSError,
        ValueError,
    ):
        print("lis-verify-ci: gate failed closed", file=sys.stderr)
        return 2
    print(
        "lis-verify-ci: verified "
        f"verdict={result.verdict.value} "
        f"report={result.report_sha256} "
        f"ledger={result.ledger_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
