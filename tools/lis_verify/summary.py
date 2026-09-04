"""Deterministic report-only terminal and Markdown projections."""

from __future__ import annotations

import json
from typing import Any

from .product_contract import MAX_SUMMARY_BYTES
from .report_model import VerificationReport


def _display(value: object, *, markdown: bool = False) -> str:
    """Render untrusted bounded text without terminal/Markdown control effects."""

    text = json.dumps(str(value), ensure_ascii=False)[1:-1]
    if markdown:
        text = text.replace("|", "\\|")
    return text


def _token_line(raw: dict[str, Any]) -> str:
    token = raw["token_comparison"]
    status = token["status"]
    if status == "mismatch":
        mismatch = token["first_mismatch"]
        return _display(
            f"mismatch at generated step {mismatch['generated_token_step']} "
            f"(reference={mismatch['reference_token_id']}, "
            f"candidate={mismatch['candidate_token_id']})"
        )
    if status == "equal":
        return _display(
            "equal on observed range "
            f"({token['reference_observed_count']}/"
            f"{token['candidate_observed_count']})"
        )
    return "unavailable"


def _localization_line(raw: dict[str, Any]) -> str:
    localization = raw["localization"]
    layer = localization["layer_suspect_interval"] or "none"
    intra = localization["intra_layer_suspect_interval"] or "none"
    return _display(f"layer={layer}; intra_layer={intra}")


def _coverage_line(raw: dict[str, Any]) -> str:
    coverage = raw["coverage"]
    return _display(
        f"scope={coverage['scope']}; common_layers={len(coverage['common_layers'])}; "
        f"missing_reference_layers={len(coverage['missing_reference_layers'])}; "
        f"missing_candidate_layers={len(coverage['missing_candidate_layers'])}; "
        f"common_intra_layer_stages={len(coverage['common_intra_layer_stages'])}"
    )


def render_terminal(report: VerificationReport) -> str:
    raw = report.to_dict()
    lines = [
        f"LIS Verify: {raw['verdict']}",
        f"Reason: {_display(','.join(raw['reason_codes']))}",
        f"Policy: {raw['policy_result']['policy']} "
        f"(exit={raw['policy_result']['exit_code']})",
        f"Tokens: {_token_line(raw)}",
        f"Coverage: {_coverage_line(raw)}",
        f"Localization: {_localization_line(raw)}",
        f"Evidence: {raw['evidence']['tier']} "
        f"(ceiling={raw['evidence']['ceiling']})",
        f"Cleanup: {raw['cleanup']['status']} "
        f"(residue={raw['cleanup']['residue_status']})",
    ]
    if raw["next_action"] is not None:
        lines.append(
            f"Next: {_display(raw['next_action']['code'])} — "
            f"{_display(raw['next_action']['summary'])}"
        )
    if raw["warnings"]:
        lines.append(f"Warnings: {len(raw['warnings'])}")
    text = "\n".join(lines) + "\n"
    if len(text.encode("utf-8")) > MAX_SUMMARY_BYTES:
        raise ValueError("terminal summary exceeds its total byte bound")
    return text


def render_markdown(report: VerificationReport) -> str:
    raw = report.to_dict()
    rows = [
        ("Verdict", raw["verdict"]),
        ("Reasons", ", ".join(raw["reason_codes"])),
        (
            "Policy",
            f"{raw['policy_result']['policy']} / exit "
            f"{raw['policy_result']['exit_code']}",
        ),
        ("Tokens", _token_line(raw)),
        ("Coverage", _coverage_line(raw)),
        ("Localization", _localization_line(raw)),
        (
            "Evidence",
            f"{raw['evidence']['tier']}; ceiling={raw['evidence']['ceiling']}",
        ),
        (
            "Cleanup",
            f"{raw['cleanup']['status']}; "
            f"residue={raw['cleanup']['residue_status']}",
        ),
    ]
    lines = [
        "# LIS Verify Summary",
        "",
        "| Field | Value |",
        "|---|---|",
    ]
    for name, value in rows:
        lines.append(f"| {name} | {_display(value, markdown=True)} |")
    if raw["next_action"] is not None:
        lines.extend(
            [
                "",
                "## Next action",
                "",
                f"{_display(raw['next_action']['code'], markdown=True)} — "
                f"{_display(raw['next_action']['summary'], markdown=True)}",
            ]
        )
    if raw["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(
            f"- {_display(warning, markdown=True)}" for warning in raw["warnings"]
        )
    text = "\n".join(lines) + "\n"
    if len(text.encode("utf-8")) > MAX_SUMMARY_BYTES:
        raise ValueError("Markdown summary exceeds its total byte bound")
    return text
