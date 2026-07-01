"""Safe Pass 2-local mappings into frozen verification-report reasons.

Successful boundary verification remains local evidence.  No mapping in this
module implies numeric checkpoint confirmation or confirmed first divergence.
"""

from __future__ import annotations

from typing import Optional

from .pass2_model import Pass2ReasonCode, Pass2Status


LOCAL_REASON_TO_REPORT_REASON: dict[Pass2ReasonCode, str] = {
    Pass2ReasonCode.SOURCE_BINDING_INCONSISTENT: "malformed_artifact",
    Pass2ReasonCode.PASS1_STATUS_NOT_REPRODUCIBLE: (
        "partial_evidence_inconclusive"
    ),
    Pass2ReasonCode.PREFIX_MATERIAL_UNAVAILABLE: (
        "partial_evidence_inconclusive"
    ),
    Pass2ReasonCode.PREFIX_DIGEST_MISMATCH: "prefix_reproduction_failed",
    Pass2ReasonCode.PREFIX_TOKEN_MISMATCH: "prefix_reproduction_failed",
    Pass2ReasonCode.DECODE_POLICY_REPRODUCTION_FAILED: (
        "unsupported_comparison"
    ),
    Pass2ReasonCode.CHECKPOINT_STEP_MAPPING_MISMATCH: (
        "malformed_artifact"
    ),
    Pass2ReasonCode.CONTEXT_POSITION_MISMATCH: "unsupported_comparison",
    Pass2ReasonCode.UNSUPPORTED_REPRODUCTION_MODE: (
        "unsupported_comparison"
    ),
    Pass2ReasonCode.REPRODUCTION_ARTIFACT_MALFORMED: "malformed_artifact",
    Pass2ReasonCode.VERDICT_STRENGTH_LIMIT_BLOCKS_REPRODUCTION: (
        "unsupported_comparison"
    ),
}


SAFE_STATUS_TO_REPORT_REASON: dict[Pass2Status, str] = {
    Pass2Status.TOKEN_LOCALIZATION_NOT_AVAILABLE: (
        "partial_evidence_inconclusive"
    ),
    Pass2Status.SOURCE_BINDING_INCONSISTENT: "malformed_artifact",
    Pass2Status.PREFIX_MATERIAL_UNAVAILABLE: (
        "partial_evidence_inconclusive"
    ),
    Pass2Status.PREFIX_REPRODUCTION_FAILED: "prefix_reproduction_failed",
    Pass2Status.DECODE_POLICY_REPRODUCTION_FAILED: (
        "unsupported_comparison"
    ),
    Pass2Status.CHECKPOINT_STEP_MAPPING_MISMATCH: "malformed_artifact",
    Pass2Status.CONTEXT_POSITION_MISMATCH: "unsupported_comparison",
    Pass2Status.UNSUPPORTED_REPRODUCTION_MODE: "unsupported_comparison",
    Pass2Status.INCONCLUSIVE: "partial_evidence_inconclusive",
}


def map_pass2_reason(code: Pass2ReasonCode) -> Optional[str]:
    return LOCAL_REASON_TO_REPORT_REASON.get(code)


def map_pass2_status(status: Pass2Status) -> Optional[str]:
    return SAFE_STATUS_TO_REPORT_REASON.get(status)
