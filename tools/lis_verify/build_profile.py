"""LIS build calibration profile — the checked-in encoding of F1/F3 facts.

These facts are not recoverable from any run artifact: the repetition-penalty
constant (``LIS_CLI_REPETITION_PENALTY = 1.2f``) and the always-on structural
suppression live only in C source, and ``rms_norm_eps`` runtime binding is
family-specific (Qwen3 binds it, the Llama path hard-codes ``1.0e-5f``). Pass 0
therefore reads them from a checked-in profile and corroborates with the
optional ``decode_trace`` when present.

Default profile: ``tools/test_fixtures/calibration/lis_build_profile.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class BuildCalibrationProfile:
    build_id: str
    repetition_penalty: float
    repetition_penalty_enabled: bool
    structural_token_suppression: bool
    rms_norm_eps_runtime_bound: dict[str, bool]
    kv_write_round_to_nearest_even: bool
    fma_contraction_backend_defined: bool
    reduction_order_backend_defined: bool

    def eps_bound_for(self, model_family: Optional[str]) -> bool:
        if model_family is None:
            return False
        return bool(self.rms_norm_eps_runtime_bound.get(model_family, False))

    @classmethod
    def from_dict(cls, data: dict) -> "BuildCalibrationProfile":
        return cls(
            build_id=str(data.get("build_id", "unknown")),
            repetition_penalty=float(data["repetition_penalty"]),
            repetition_penalty_enabled=bool(data["repetition_penalty_enabled"]),
            structural_token_suppression=bool(data["structural_token_suppression"]),
            rms_norm_eps_runtime_bound=dict(data["rms_norm_eps_runtime_bound"]),
            kv_write_round_to_nearest_even=bool(data["kv_write_round_to_nearest_even"]),
            fma_contraction_backend_defined=bool(data["fma_contraction_backend_defined"]),
            reduction_order_backend_defined=bool(data["reduction_order_backend_defined"]),
        )

    @classmethod
    def load(cls, path) -> "BuildCalibrationProfile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


_DEFAULT_PATH = (
    Path(__file__).resolve().parents[1]
    / "test_fixtures"
    / "calibration"
    / "lis_build_profile.json"
)


def default_build_profile() -> BuildCalibrationProfile:
    """The default LIS build profile (F1/F3 facts for the current build).

    Falls back to a hard-coded conservative profile if the checked-in JSON is
    unavailable or malformed, so Pass 0 never silently fails open on a missing
    fixture.
    """
    try:
        return BuildCalibrationProfile.load(_DEFAULT_PATH)
    except (OSError, KeyError, ValueError):
        return BuildCalibrationProfile(
            build_id="lis_default_builtin",
            repetition_penalty=1.2,
            repetition_penalty_enabled=True,
            structural_token_suppression=True,
            rms_norm_eps_runtime_bound={
                "llama3_decoder": False,
                "qwen3_dense_decoder": True,
            },
            kv_write_round_to_nearest_even=False,
            fma_contraction_backend_defined=True,
            reduction_order_backend_defined=True,
        )
