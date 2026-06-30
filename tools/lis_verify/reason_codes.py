"""Reason-code registry for Pass 0.

Correction 2: the registry stores ``code -> (domain, base_severity)`` and is
deliberately context-free. Mode-specific severity escalation (e.g.
``external_oracle_ineligible`` becoming a block in Mode C, or
``config_fingerprint_mismatch`` becoming a block outside the
``configuration_equivalence`` submode) is performed by the aggregator in
``pass0.py`` — never here.

Note on oracle-scope codes: codes that only *bound oracle scope* (e.g.
``oracle_scope_limited``, ``forced_prefix_report_json_channel_missing``,
``external_oracle_ineligible``) carry a non-blocking base severity. They limit
oracle eligibility but do not, by themselves, downgrade the internal LIS
differential — otherwise the ``comparable`` verdict would be unreachable even
for a fully calibrated raw-greedy run.
"""

from __future__ import annotations

from .model import CalibrationDomain as D
from .model import CalibrationReasonCode as C
from .model import ReasonSeverity as S


REGISTRY: dict[C, tuple[D, S]] = {
    # decode policy
    C.INCOMPATIBLE_DECODE_POLICY: (D.DECODE_POLICY, S.BLOCK),
    C.POLICY_MODIFIED_GREEDY: (D.DECODE_POLICY, S.DOWNGRADE),
    C.DECODE_POLICY_NOT_RAW: (D.DECODE_POLICY, S.DOWNGRADE),
    C.DECODE_POLICY_UNCALIBRATED: (D.DECODE_POLICY, S.DOWNGRADE),
    # tokenizer / prompt boundary
    C.TOKENIZER_BOUNDARY_UNCALIBRATED: (D.TOKENIZER_BOUNDARY, S.DOWNGRADE),
    C.PROMPT_TOKEN_ARRAY_MISSING: (D.TOKENIZER_BOUNDARY, S.DOWNGRADE),
    C.PROMPT_TOKEN_IDENTITY_UNVERIFIED: (D.TOKENIZER_BOUNDARY, S.DOWNGRADE),
    C.INPUT_TOKEN_DIVERGENCE: (D.TOKENIZER_BOUNDARY, S.BLOCK),
    C.CONFIDENCE_DOWNGRADE_TEXT_PROMPT_BOUNDARY: (D.TOKENIZER_BOUNDARY, S.DOWNGRADE),
    # config semantics
    C.CONFIG_SEMANTICS_UNCALIBRATED: (D.CONFIG_SEMANTICS, S.DOWNGRADE),
    C.RMS_NORM_EPS_RUNTIME_UNBOUND: (D.CONFIG_SEMANTICS, S.DOWNGRADE),
    C.CONFIG_FINGERPRINT_MISMATCH: (D.CONFIG_SEMANTICS, S.DOWNGRADE),  # aggregator escalates
    C.RUNTIME_CONFIG_FINGERPRINT_MISSING: (D.CONFIG_SEMANTICS, S.DOWNGRADE),
    C.REQUIRES_FIX_OR_GUARD: (D.CONFIG_SEMANTICS, S.DOWNGRADE),
    C.INCOMPATIBLE_MODEL_FAMILY: (D.CONFIG_SEMANTICS, S.BLOCK),
    # numeric policy
    C.NUMERIC_POLICY_UNCALIBRATED: (D.NUMERIC_POLICY, S.DOWNGRADE),
    C.KV_WRITE_ROUNDING_UNVERIFIED: (D.NUMERIC_POLICY, S.DOWNGRADE),
    C.FMA_POLICY_BACKEND_DEFINED: (D.NUMERIC_POLICY, S.INFORMATIONAL),
    C.REDUCTION_ORDER_BACKEND_DEFINED: (D.NUMERIC_POLICY, S.INFORMATIONAL),
    C.TOLERANCE_CAVEAT: (D.NUMERIC_POLICY, S.DOWNGRADE),
    # oracle scope
    C.EXTERNAL_ORACLE_INELIGIBLE: (D.ORACLE_SCOPE, S.INFORMATIONAL),  # aggregator escalates in Mode C
    C.HF_DEFAULT_GREEDY_INELIGIBLE: (D.ORACLE_SCOPE, S.INFORMATIONAL),
    C.HF_FORCED_TOKEN_RUNTIME_ELIGIBLE: (D.ORACLE_SCOPE, S.INFORMATIONAL),
    C.INTERNAL_LIS_DIFFERENTIAL_ONLY: (D.ORACLE_SCOPE, S.INFORMATIONAL),
    C.ORACLE_SCOPE_LIMITED: (D.ORACLE_SCOPE, S.INFORMATIONAL),
    C.FORCED_PREFIX_REPORT_JSON_CHANNEL_MISSING: (D.ORACLE_SCOPE, S.INFORMATIONAL),
}


# Codes whose *effective* severity the aggregator may raise above the base value
# depending on the declared comparison mode / submode.
AGGREGATOR_ESCALATED: frozenset[C] = frozenset({
    C.EXTERNAL_ORACLE_INELIGIBLE,
    C.CONFIG_FINGERPRINT_MISMATCH,
    C.PROMPT_TOKEN_ARRAY_MISSING,
    C.PROMPT_TOKEN_IDENTITY_UNVERIFIED,
})


_DOMAIN_ORDER: list[D] = [
    D.DECODE_POLICY,
    D.TOKENIZER_BOUNDARY,
    D.CONFIG_SEMANTICS,
    D.NUMERIC_POLICY,
    D.COMPARISON_MODE,
    D.ORACLE_SCOPE,
]

_CODE_INDEX: dict[C, int] = {code: i for i, code in enumerate(REGISTRY)}


def base_severity(code: C) -> S:
    return REGISTRY[code][1]


def domain_of(code: C) -> D:
    return REGISTRY[code][0]


def all_codes() -> list[C]:
    return list(REGISTRY.keys())


def _sort_key(code: C) -> tuple[int, int]:
    return (_DOMAIN_ORDER.index(REGISTRY[code][0]), _CODE_INDEX[code])


def order_codes(codes) -> list[C]:
    """Deduplicate and return codes in a stable (domain, registry) order."""
    return sorted(set(codes), key=_sort_key)
