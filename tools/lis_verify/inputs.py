"""Pass 0 inputs and run_report extraction.

``RunSide.from_run_report`` parses both the real (nested) ``run_report``
manifest emitted by ``srcs/core/artifact.c`` (``manifest.model.family``,
``manifest.config.fingerprint``, ``manifest.input.mode``,
``manifest.runtime.precision_path``, ``manifest.backend.name``) and the flatter
shape used by some inspector fixtures (``manifest.model_family`` etc.). The
prompt token *array* is never present in a run_report — only a count and an
``fnv1a64`` digest — so the explicit array is supplied separately by the caller
when direct token IDs are available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .build_profile import BuildCalibrationProfile, default_build_profile
from .model import ComparisonMode, ModeBSubmode, OracleScope


def _dig(obj, *paths):
    """Return the first present value among several dotted key paths."""
    for path in paths:
        cur = obj
        ok = True
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break
        if ok:
            return cur
    return None


def _kv_from_precision(precision_path) -> Optional[str]:
    if not precision_path:
        return None
    for part in str(precision_path).split(";"):
        part = part.strip()
        if part.startswith("kv="):
            return part[3:]
    return None


@dataclass
class DecodeTraceSummary:
    """Optional corroboration derived from a ``decode_trace`` artifact."""

    any_repetition_penalty_changed_selection: bool = False
    any_structural_suppression_affected: bool = False
    any_selected_token_penalized: bool = False
    decision_classes_seen: frozenset = field(default_factory=frozenset)

    @classmethod
    def from_decode_trace(cls, raw: dict) -> "DecodeTraceSummary":
        steps = raw.get("decode_trace", []) if isinstance(raw, dict) else []
        rep = supp = pen = False
        classes = set()
        for step in steps:
            if not isinstance(step, dict):
                continue
            rep = rep or bool(step.get("repetition_penalty_changed_selection"))
            supp = supp or bool(step.get("structural_suppression_affected"))
            pen = pen or bool(step.get("selected_token_penalized"))
            decision_class = step.get("decision_class")
            if decision_class:
                classes.add(decision_class)
        return cls(rep, supp, pen, frozenset(classes))


@dataclass
class RunSide:
    """One execution under comparison."""

    role: str
    model_family: Optional[str] = None
    config_fingerprint: Optional[str] = None
    binary_fingerprint: Optional[str] = None
    runtime_fingerprint: Optional[str] = None
    backend: Optional[str] = None
    input_mode: Optional[str] = None
    prompt_token_count: Optional[int] = None
    prompt_token_digest: Optional[str] = None
    prompt_token_array: Optional[list] = None
    precision_path: Optional[str] = None
    kv_storage_dtype: Optional[str] = None
    # Optional per-side decode-policy overrides (e.g. a different LIS build in
    # Mode B). When None, the build profile supplies the value.
    repetition_penalty: Optional[float] = None
    structural_token_suppression: Optional[bool] = None
    decode_trace: Optional[DecodeTraceSummary] = None

    @classmethod
    def from_run_report(
        cls,
        raw: dict,
        role: str,
        prompt_token_array: Optional[list] = None,
        decode_trace: Optional[DecodeTraceSummary] = None,
    ) -> "RunSide":
        manifest = raw.get("manifest", {}) if isinstance(raw, dict) else {}
        report = raw.get("report", {}) if isinstance(raw, dict) else {}
        seqs = report.get("prompt_sequences") or []
        first = seqs[0] if seqs and isinstance(seqs[0], dict) else {}
        precision_path = _dig(
            manifest, ["runtime", "precision_path"], ["precision_path"]
        )
        kv_dtype = _dig(report, ["kv_cache", "storage_dtype"])
        if kv_dtype is None:
            kv_dtype = _kv_from_precision(precision_path)
        return cls(
            role=role,
            model_family=_dig(manifest, ["model", "family"], ["model_family"]),
            config_fingerprint=_dig(
                manifest, ["config", "fingerprint"], ["config_fingerprint"]
            ),
            binary_fingerprint=_dig(
                manifest, ["binary", "fingerprint"], ["binary_fingerprint"]
            ),
            runtime_fingerprint=_dig(
                manifest, ["runtime", "fingerprint"], ["runtime_fingerprint"]
            ),
            backend=_dig(manifest, ["backend", "name"], ["backend"]),
            input_mode=_dig(manifest, ["input", "mode"], ["input_mode"]),
            prompt_token_count=first.get("token_count"),
            prompt_token_digest=first.get("digest"),
            prompt_token_array=(
                list(prompt_token_array) if prompt_token_array is not None else None
            ),
            precision_path=precision_path,
            kv_storage_dtype=kv_dtype,
            decode_trace=decode_trace,
        )


@dataclass
class PreflightInputs:
    reference: RunSide
    candidate: RunSide
    declared_mode: ComparisonMode
    declared_submode: Optional[ModeBSubmode] = None
    build_profile: BuildCalibrationProfile = field(default_factory=default_build_profile)
    declared_oracle_intent: Optional[OracleScope] = None
