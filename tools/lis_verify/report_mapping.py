"""Pass 0 -> Pass 6 boundary mapping.

Maps each Pass 0 *block* reason code into an existing (frozen)
``verification_report`` ``reason_code``. This is the only place Pass 0
vocabulary touches the frozen contract enums, and it does so by mapping *into*
them — never by extending them.
"""

from __future__ import annotations

from typing import Optional

from .model import CalibrationReasonCode as C


# Pass 0 block reason -> existing verification_report reason_code.
BLOCK_REASON_TO_REPORT_REASON_CODE: dict[C, str] = {
    C.INCOMPATIBLE_DECODE_POLICY: "unsupported_comparison",
    C.INPUT_TOKEN_DIVERGENCE: "input_token_divergence",
    C.INCOMPATIBLE_MODEL_FAMILY: "incompatible_model_family",
    C.CONFIG_FINGERPRINT_MISMATCH: "unsupported_comparison",
    C.EXTERNAL_ORACLE_INELIGIBLE: "unsupported_mode",
    C.PROMPT_TOKEN_ARRAY_MISSING: "unsupported_comparison",
    C.PROMPT_TOKEN_IDENTITY_UNVERIFIED: "unsupported_comparison",
}


def map_block_reason(code: C) -> Optional[str]:
    return BLOCK_REASON_TO_REPORT_REASON_CODE.get(code)
