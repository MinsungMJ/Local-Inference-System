"""Conservative Pass 3-local mappings into frozen report reasons.

Successful Pass 3 outcomes deliberately have no frozen mapping.
"""

from __future__ import annotations

from typing import Optional

from .pass3_model import Pass3ReasonCode, Pass3Status


LOCAL_REASON_TO_REPORT_REASON: dict[Pass3ReasonCode, str] = {
    Pass3ReasonCode.PASS2_NOT_READY: "partial_evidence_inconclusive",
    Pass3ReasonCode.REPRODUCTION_REQUEST_ONLY: "partial_evidence_inconclusive",
    Pass3ReasonCode.SOURCE_BINDING_INCONSISTENT: "malformed_artifact",
    Pass3ReasonCode.PASS2_ARTIFACT_IDENTITY_INCONSISTENT: "malformed_artifact",
    Pass3ReasonCode.PASS2_OBJECT_ARTIFACT_INCONSISTENT: "malformed_artifact",
    Pass3ReasonCode.RUN_REPORT_CANONICAL_SHA_INCONSISTENT: "malformed_artifact",
    Pass3ReasonCode.ARTIFACT_SET_ID_INCONSISTENT: "malformed_artifact",
    Pass3ReasonCode.BINDING_METADATA_MISSING: "malformed_artifact",
    Pass3ReasonCode.RUNTIME_CHECKPOINT_STEP_MISMATCH: "malformed_artifact",
    Pass3ReasonCode.INSUFFICIENT_COMMON_COVERAGE: "partial_evidence_inconclusive",
    Pass3ReasonCode.REFERENCE_CHECKPOINT_MISSING: "requested_step_absent",
    Pass3ReasonCode.CANDIDATE_CHECKPOINT_MISSING: "requested_step_absent",
    Pass3ReasonCode.CHECKPOINT_ALIGNMENT_INCONSISTENT: "unsupported_comparison",
    Pass3ReasonCode.DUPLICATE_CHECKPOINT_COORDINATE: "malformed_artifact",
    Pass3ReasonCode.CHECKPOINT_SUMMARY_MALFORMED: "malformed_artifact",
    Pass3ReasonCode.SUMMARY_FIELD_MISSING: "unsupported_comparison",
    Pass3ReasonCode.CHECKPOINT_DIGEST_INCOMPATIBLE: "unsupported_comparison",
    Pass3ReasonCode.COMPARISON_POLICY_UNAVAILABLE: "unsupported_comparison",
    Pass3ReasonCode.UNSUPPORTED_CHECKPOINT_LAYOUT: "unsupported_comparison",
    Pass3ReasonCode.ASYMMETRIC_COVERAGE: "asymmetric_checkpoint",
}


SAFE_STATUS_TO_REPORT_REASON: dict[Pass3Status, str] = {
    Pass3Status.COMPARISON_BLOCKED_BY_PASS2: "partial_evidence_inconclusive",
    Pass3Status.INSUFFICIENT_COMMON_COVERAGE: "partial_evidence_inconclusive",
    Pass3Status.SOURCE_BINDING_INCONSISTENT: "malformed_artifact",
    Pass3Status.CHECKPOINT_ALIGNMENT_INCONSISTENT: "unsupported_comparison",
    Pass3Status.CHECKPOINT_ARTIFACT_MISSING: "requested_step_absent",
    Pass3Status.CHECKPOINT_SUMMARY_MALFORMED: "malformed_artifact",
    Pass3Status.COMPARISON_POLICY_UNAVAILABLE: "unsupported_comparison",
    Pass3Status.UNSUPPORTED_CHECKPOINT_LAYOUT: "unsupported_comparison",
    Pass3Status.INCONCLUSIVE: "partial_evidence_inconclusive",
}


def map_pass3_reason(code: Pass3ReasonCode) -> Optional[str]:
    return LOCAL_REASON_TO_REPORT_REASON.get(code)


def map_pass3_status(status: Pass3Status) -> Optional[str]:
    return SAFE_STATUS_TO_REPORT_REASON.get(status)
