"""Safe Pass 1-local mappings into frozen verification-report reasons.

``first_mismatch_found`` intentionally has no mapping.  In particular, Pass 1
never uses ``token_selection_divergence`` because its frozen report mapping
implies checkpoint confirmation.
"""

from __future__ import annotations

from typing import Optional

from .pass1_model import Pass1ReasonCode, Pass1Status


LOCAL_REASON_TO_REPORT_REASON: dict[Pass1ReasonCode, str] = {
    Pass1ReasonCode.SELECTED_TOKEN_ARRAY_MISSING: (
        "partial_evidence_inconclusive"
    ),
    Pass1ReasonCode.SELECTED_TOKEN_IDENTITY_UNVERIFIED: (
        "partial_evidence_inconclusive"
    ),
    Pass1ReasonCode.SELECTED_TOKEN_METADATA_INCONSISTENT: "malformed_artifact",
    Pass1ReasonCode.GATE_RUN_IDENTITY_INCONSISTENT: "malformed_artifact",
    Pass1ReasonCode.UNSUPPORTED_RUN_ARTIFACT: "unsupported_comparison",
    Pass1ReasonCode.UNSUPPORTED_BATCH_SHAPE: "unsupported_comparison",
}


SAFE_STATUS_TO_REPORT_REASON: dict[Pass1Status, str] = {
    Pass1Status.TOKEN_EQUIVALENT_ON_OBSERVED_RANGE: (
        "equivalent_on_observed_coverage"
    ),
    Pass1Status.INPUT_TOKEN_DIVERGENCE: "input_token_divergence",
    Pass1Status.SELECTED_TOKEN_ARRAY_MISSING: (
        "partial_evidence_inconclusive"
    ),
    Pass1Status.SELECTED_TOKEN_IDENTITY_UNVERIFIED: (
        "partial_evidence_inconclusive"
    ),
    Pass1Status.UNSUPPORTED_COMPARISON: "unsupported_comparison",
    Pass1Status.INCONCLUSIVE: "partial_evidence_inconclusive",
}


def map_pass1_reason(code: Pass1ReasonCode) -> Optional[str]:
    return LOCAL_REASON_TO_REPORT_REASON.get(code)


def map_pass1_status(status: Pass1Status) -> Optional[str]:
    # FIRST_MISMATCH_FOUND and COMPARISON_BLOCKED_BY_PASS0 are deliberately
    # absent: the former stays local, and the latter uses Pass 0 mappings.
    return SAFE_STATUS_TO_REPORT_REASON.get(status)
