"""Typed data model for P1 Pass 0 (Calibration Preflight).

Model-free: no model artifacts, no tensors, no inference — pure types only
(enums and dataclasses). Logic lives in ``domains.py`` / ``pass0.py``,
serialization in ``artifact.py``, and the reason-code registry in
``reason_codes.py``.

See ``docs/calibration_preflight.md`` and the approved implementation plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


SCHEMA = "lis.execution_artifact/v1"
KIND = "calibration_preflight"
CONTRACT_VERSION = "differential_verification_contract_v1"


# --- Enums ------------------------------------------------------------------

class ComparisonMode(str, Enum):
    """Contract-owned (Correction 1): values must match the spelling in
    ``tools/test_fixtures/differential_verification_contract.json``
    (``comparison_modes``). Do not invent or rename. Mode C is exactly
    ``external_semantic``."""

    BACKEND_DIFFERENTIAL = "backend_differential"
    RUNTIME_DIFFERENTIAL = "runtime_differential"
    EXTERNAL_SEMANTIC = "external_semantic"


class ModeBSubmode(str, Enum):
    RUNTIME_REGRESSION = "runtime_regression"
    DETERMINISM_CHECK = "determinism_check"
    CONFIGURATION_EQUIVALENCE = "configuration_equivalence"
    PRECISION_POLICY_COMPARISON = "precision_policy_comparison"


class ComparisonEligibility(str, Enum):
    COMPARABLE = "comparable"
    LIMITED_COMPARISON = "limited_comparison"
    INCOMPATIBLE = "incompatible"


class Pass0Verdict(str, Enum):
    COMPARISON_ALLOWED = "comparison_allowed"
    LIMITED_COMPARISON_ALLOWED = "limited_comparison_allowed"
    COMPARISON_BLOCKED = "comparison_blocked"


class SelectionMode(str, Enum):
    RAW_GREEDY = "raw_greedy"
    POLICY_MODIFIED_GREEDY = "policy_modified_greedy"


class PromptBoundary(str, Enum):
    DIRECT_TOKEN_IDS = "direct_token_ids"
    TEXT = "text"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PromptIdentityEvidence(str, Enum):
    """Correction 4: array equality is distinct from digest-only identity."""

    ARRAY_EQUAL = "array_equal"
    DIGEST_ONLY = "digest_only"
    UNVERIFIED = "unverified"
    DIVERGENT = "divergent"


class ConfigSemanticsStatus(str, Enum):
    CALIBRATED = "calibrated"
    REQUIRES_FIX_OR_GUARD = "requires_fix_or_guard"
    UNKNOWN = "unknown"


class KvWriteStatus(str, Enum):
    ROUND_TO_NEAREST_EVEN = "round_to_nearest_even"
    TRUNCATE_OR_UNVERIFIED = "truncate_or_unverified"
    NOT_APPLICABLE = "not_applicable"


class OracleScope(str, Enum):
    INTERNAL_LIS_ONLY = "internal_lis_only"
    INTERNAL_LIS_AND_RUNTIME = "internal_lis_and_runtime"
    EXTERNAL_SEMANTIC = "external_semantic"


class VerdictStrengthLimit(str, Enum):
    """Intentionally has no first-divergence member — structurally prevents the
    MVP from authorizing ``confirmed_first_divergence``."""

    NO_COMPARISON = "no_comparison"
    TOKEN_LOCALIZATION_ONLY = "token_localization_only"
    CHECKPOINT_CONFIRMATION_ALLOWED = "checkpoint_confirmation_allowed"


class CalibrationDomain(str, Enum):
    DECODE_POLICY = "decode_policy"
    TOKENIZER_BOUNDARY = "tokenizer_boundary"
    CONFIG_SEMANTICS = "config_semantics"
    NUMERIC_POLICY = "numeric_policy"
    COMPARISON_MODE = "comparison_mode"
    ORACLE_SCOPE = "oracle_scope"


class ReasonSeverity(str, Enum):
    BLOCK = "block"
    DOWNGRADE = "downgrade"
    INFORMATIONAL = "informational"


class CalibrationReasonCode(str, Enum):
    # decode policy
    INCOMPATIBLE_DECODE_POLICY = "incompatible_decode_policy"
    POLICY_MODIFIED_GREEDY = "policy_modified_greedy"
    DECODE_POLICY_NOT_RAW = "decode_policy_not_raw"
    DECODE_POLICY_UNCALIBRATED = "decode_policy_uncalibrated"
    # tokenizer / prompt boundary
    TOKENIZER_BOUNDARY_UNCALIBRATED = "tokenizer_boundary_uncalibrated"
    PROMPT_TOKEN_ARRAY_MISSING = "prompt_token_array_missing"
    PROMPT_TOKEN_IDENTITY_UNVERIFIED = "prompt_token_identity_unverified"
    INPUT_TOKEN_DIVERGENCE = "input_token_divergence"
    CONFIDENCE_DOWNGRADE_TEXT_PROMPT_BOUNDARY = "confidence_downgrade_text_prompt_boundary"
    # config semantics
    CONFIG_SEMANTICS_UNCALIBRATED = "config_semantics_uncalibrated"
    RMS_NORM_EPS_RUNTIME_UNBOUND = "rms_norm_eps_runtime_unbound"
    CONFIG_FINGERPRINT_MISMATCH = "config_fingerprint_mismatch"
    RUNTIME_CONFIG_FINGERPRINT_MISSING = "runtime_config_fingerprint_missing"
    REQUIRES_FIX_OR_GUARD = "requires_fix_or_guard"
    INCOMPATIBLE_MODEL_FAMILY = "incompatible_model_family"  # reuses an existing contract code
    # numeric policy
    NUMERIC_POLICY_UNCALIBRATED = "numeric_policy_uncalibrated"
    KV_WRITE_ROUNDING_UNVERIFIED = "kv_write_rounding_unverified"
    FMA_POLICY_BACKEND_DEFINED = "fma_policy_backend_defined"
    REDUCTION_ORDER_BACKEND_DEFINED = "reduction_order_backend_defined"
    TOLERANCE_CAVEAT = "tolerance_caveat"
    # oracle scope
    EXTERNAL_ORACLE_INELIGIBLE = "external_oracle_ineligible"
    HF_DEFAULT_GREEDY_INELIGIBLE = "hf_default_greedy_ineligible"
    HF_FORCED_TOKEN_RUNTIME_ELIGIBLE = "hf_forced_token_runtime_eligible"
    INTERNAL_LIS_DIFFERENTIAL_ONLY = "internal_lis_differential_only"
    ORACLE_SCOPE_LIMITED = "oracle_scope_limited"
    FORCED_PREFIX_REPORT_JSON_CHANNEL_MISSING = "forced_prefix_report_json_channel_missing"


# --- Domain output dataclasses ----------------------------------------------

@dataclass
class DecodePolicyIdentity:
    selection_mode: SelectionMode
    repetition_penalty: float
    repetition_penalty_enabled: bool
    structural_token_suppression: bool
    raw_greedy_equivalent: bool


@dataclass
class TokenizerBoundary:
    prompt_boundary: PromptBoundary
    prompt_token_array_available: bool
    prompt_token_array_equal: Optional[bool]
    prompt_token_digest_equal: Optional[bool]
    prompt_identity_evidence: PromptIdentityEvidence
    confidence: Confidence


@dataclass
class ConfigSemantics:
    rms_norm_eps_runtime_bound: bool
    status: ConfigSemanticsStatus


@dataclass
class NumericPolicy:
    kv_bf16_write: KvWriteStatus
    kv_f16_write: KvWriteStatus
    round_to_nearest_even: bool


@dataclass
class ForcedTokenRuntimeEligibility:
    """Forced-token runtime potential and artifact-backed eligibility.

    M3 sets ``artifact_supported`` only because the C producer and Python
    source binder now implement the frozen report channel.
    """

    potentially_eligible: bool
    artifact_supported: bool
    status: str


@dataclass
class OracleEligibility:
    lis_internal_backend_differential: bool
    hf_default_greedy: bool
    hf_forced_token_runtime: ForcedTokenRuntimeEligibility
    oracle_scope: OracleScope


@dataclass
class CalibrationStatus:
    decode_policy_calibrated: bool
    config_semantics_calibrated: bool
    numeric_policy_calibrated: bool
    tokenizer_boundary_calibrated: bool


@dataclass
class CalibrationPreflightArtifact:
    comparison_mode: ComparisonMode
    comparison_eligibility: ComparisonEligibility
    pass0_verdict: Pass0Verdict
    calibration_status: CalibrationStatus
    decode_policy_identity: DecodePolicyIdentity
    tokenizer_boundary: TokenizerBoundary
    config_semantics: ConfigSemantics
    numeric_policy: NumericPolicy
    oracle_eligibility: OracleEligibility
    reason_codes: list[CalibrationReasonCode] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blocking_reasons: list[CalibrationReasonCode] = field(default_factory=list)
    verdict_strength_limit: VerdictStrengthLimit = VerdictStrengthLimit.NO_COMPARISON
    schema: str = SCHEMA
    kind: str = KIND
    contract_version: str = CONTRACT_VERSION
