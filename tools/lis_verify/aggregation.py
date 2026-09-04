"""Pure product aggregation helpers backed by the frozen M0 tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .product_contract import (
    AggregationAction,
    CustomerVerdict,
    ExecutionPolicy,
    PASS0_BLOCK_REASON_VERDICTS,
    PASS_STATUS_ACTIONS,
    expected_exit_code,
)


@dataclass(frozen=True)
class AggregationDecision:
    action: AggregationAction
    target: str


def resolve_pass_status(pass_name: str, status: str) -> AggregationDecision:
    """Resolve one Pass-local status without a catch-all branch."""

    try:
        action, target = PASS_STATUS_ACTIONS[pass_name][status]
    except (KeyError, TypeError) as exc:
        raise ValueError("unknown Pass-local status; contract amendment required") from exc
    return AggregationDecision(AggregationAction(action), target)


def resolve_pass0_block_reason(reason: str) -> CustomerVerdict:
    try:
        return PASS0_BLOCK_REASON_VERDICTS[reason]
    except (KeyError, TypeError) as exc:
        raise ValueError("unknown Pass 0 blocking reason") from exc


def policy_result_for(
    verdict: CustomerVerdict,
    *,
    require_supported: bool,
    reason: str,
) -> dict[str, Any]:
    policy = (
        ExecutionPolicy.REQUIRE_SUPPORTED
        if require_supported
        else ExecutionPolicy.DEFAULT
    )
    exit_code = expected_exit_code(verdict, policy)
    return {
        "policy": policy.value,
        "satisfied": exit_code == 0,
        "exit_code": exit_code,
        "reason": reason,
    }


def apply_policy(raw: dict[str, Any], *, require_supported: bool) -> dict[str, Any]:
    """Return a copy with policy fields updated, never rewriting the verdict."""

    from .report_model import VerificationReport

    value = VerificationReport.from_dict(raw).to_dict()
    verdict = CustomerVerdict(value["verdict"])
    value["command"]["require_supported"] = require_supported
    value["policy_result"] = policy_result_for(
        verdict,
        require_supported=require_supported,
        reason=value["policy_result"]["reason"],
    )
    return VerificationReport.from_dict(value).to_dict()
