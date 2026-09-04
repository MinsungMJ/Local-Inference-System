#!/usr/bin/env python3
"""Model-free behavior tests for P1 Pass 0 (Calibration Preflight).

No model artifacts, no tensors, no inference. Scenarios are numbered to match
the approved plan's §10 test plan.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from lis_verify import (
    BuildCalibrationProfile,
    CalibrationReasonCode as C,
    ComparisonEligibility,
    ComparisonMode,
    ModeBSubmode,
    Pass0Verdict,
    PreflightInputs,
    PromptIdentityEvidence,
    RunSide,
    VerdictStrengthLimit,
    build_gate,
    default_build_profile,
    map_block_reason,
    run_calibration_preflight,
    serialize,
)
from lis_verify.pass0 import effective_severity
from lis_verify.reason_codes import (
    AGGREGATOR_ESCALATED,
    REGISTRY,
    base_severity,
)
from lis_verify.model import CalibrationReasonCode, ReasonSeverity

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tools" / "test_fixtures" / "calibration"
CONTRACT = ROOT / "tools" / "test_fixtures" / "differential_verification_contract.json"


# Fully-calibrated raw-greedy build profile (the only path to `comparable`).
CALIBRATED = BuildCalibrationProfile(
    build_id="cal",
    repetition_penalty=1.0,
    repetition_penalty_enabled=False,
    structural_token_suppression=False,
    rms_norm_eps_runtime_bound={"llama3_decoder": True, "qwen3_dense_decoder": True},
    kv_write_round_to_nearest_even=True,
    fma_contraction_backend_defined=False,
    reduction_order_backend_defined=False,
)


def side(role, **kw):
    """Build a RunSide with sensible Mode-A defaults."""
    base = dict(
        model_family="llama3_decoder",
        config_fingerprint="fnv1a64:cfg1",
        binary_fingerprint="fnv1a64:bin1",
        backend="reference",
        input_mode="tokens",
        prompt_token_count=3,
        prompt_token_digest="fnv1a64:dig1",
        precision_path="f32_accum;weights=bf16;kv=bf16",
        kv_storage_dtype="bf16",
    )
    base.update(kw)
    return RunSide(role=role, **base)


def run(ref, cand, mode=ComparisonMode.BACKEND_DIFFERENTIAL, submode=None, profile=None):
    return run_calibration_preflight(
        PreflightInputs(
            reference=ref,
            candidate=cand,
            declared_mode=mode,
            declared_submode=submode,
            build_profile=profile if profile is not None else default_build_profile(),
        )
    )


def codes(art):
    return {c.value for c in art.reason_codes}


class TestModeStrings(unittest.TestCase):
    """Scenario 17 / Correction 1: ComparisonMode values are contract-owned."""

    def setUp(self):
        self.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    def test_every_mode_is_in_contract_fixture(self):
        allowed = set(self.contract["comparison_modes"])
        for mode in ComparisonMode:
            self.assertIn(mode.value, allowed, mode.value)

    def test_mode_c_exact_spelling(self):
        self.assertEqual(ComparisonMode.EXTERNAL_SEMANTIC.value, "external_semantic")
        self.assertIn("external_semantic", self.contract["deferred_modes"])


class TestReasonRegistry(unittest.TestCase):
    """Step S2 / Correction 2: registry is complete and context-free."""

    def test_every_code_registered_once(self):
        self.assertEqual(set(REGISTRY), set(CalibrationReasonCode))
        for code in CalibrationReasonCode:
            domain, severity = REGISTRY[code]
            self.assertIsNotNone(domain)
            self.assertIn(severity, set(ReasonSeverity))

    def test_aggregator_escalated_codes_have_nonblocking_base(self):
        for code in AGGREGATOR_ESCALATED:
            self.assertNotEqual(
                base_severity(code), ReasonSeverity.BLOCK,
                f"{code.value} must carry a non-blocking base severity",
            )

    def test_external_oracle_base_is_informational(self):
        self.assertEqual(
            base_severity(C.EXTERNAL_ORACLE_INELIGIBLE), ReasonSeverity.INFORMATIONAL
        )
        self.assertEqual(
            base_severity(C.CONFIG_FINGERPRINT_MISMATCH), ReasonSeverity.DOWNGRADE
        )


class TestDecodePolicy(unittest.TestCase):
    def test_2_penalty_1_2_disables_hf_default_greedy(self):
        art = run(side("reference"), side("candidate"))  # default profile = 1.2
        self.assertEqual(
            art.decode_policy_identity.selection_mode.value, "policy_modified_greedy"
        )
        self.assertFalse(art.decode_policy_identity.raw_greedy_equivalent)
        self.assertFalse(art.oracle_eligibility.hf_default_greedy)
        self.assertIn("policy_modified_greedy", codes(art))
        self.assertIn("hf_default_greedy_ineligible", codes(art))

    def test_3_structural_suppression_disables_raw(self):
        prof = BuildCalibrationProfile(
            "p", 1.0, False, True, {"llama3_decoder": True}, True, False, False
        )
        art = run(side("reference"), side("candidate"), profile=prof)
        self.assertFalse(art.decode_policy_identity.raw_greedy_equivalent)
        self.assertIn("decode_policy_not_raw", codes(art))
        self.assertFalse(art.oracle_eligibility.hf_default_greedy)

    def test_4_policy_mismatch_blocks(self):
        ref = side("reference", repetition_penalty=1.2)
        cand = side("candidate", repetition_penalty=1.0)
        art = run(ref, cand)
        self.assertEqual(art.pass0_verdict, Pass0Verdict.COMPARISON_BLOCKED)
        self.assertEqual(art.comparison_eligibility, ComparisonEligibility.INCOMPATIBLE)
        self.assertEqual(art.verdict_strength_limit, VerdictStrengthLimit.NO_COMPARISON)
        self.assertIn(C.INCOMPATIBLE_DECODE_POLICY, art.blocking_reasons)


class TestTokenizer(unittest.TestCase):
    def test_5_text_prompt_boundary_downgrade(self):
        ref = side("reference", input_mode="hf_tokenizer_prompt", prompt_token_array=[1, 2, 3])
        cand = side("candidate", input_mode="hf_tokenizer_prompt", prompt_token_array=[1, 2, 3])
        art = run(ref, cand, profile=CALIBRATED)
        self.assertEqual(art.tokenizer_boundary.prompt_boundary.value, "text")
        self.assertNotEqual(art.tokenizer_boundary.confidence.value, "high")
        self.assertIn("confidence_downgrade_text_prompt_boundary", codes(art))
        self.assertEqual(art.pass0_verdict, Pass0Verdict.LIMITED_COMPARISON_ALLOWED)

    def test_6_prompt_token_array_missing(self):
        art = run(side("reference"), side("candidate"), profile=CALIBRATED)
        tb = art.tokenizer_boundary
        self.assertFalse(tb.prompt_token_array_available)
        self.assertIsNone(tb.prompt_token_array_equal)
        self.assertTrue(tb.prompt_token_digest_equal)
        self.assertEqual(tb.prompt_identity_evidence, PromptIdentityEvidence.DIGEST_ONLY)
        self.assertIn("prompt_token_array_missing", codes(art))
        self.assertIn("prompt_token_identity_unverified", codes(art))

    def test_6c_array_missing_blocks_in_mode_c(self):
        art = run(side("reference"), side("candidate"), mode=ComparisonMode.EXTERNAL_SEMANTIC)
        self.assertEqual(art.pass0_verdict, Pass0Verdict.COMPARISON_BLOCKED)

    def test_2t_array_divergence_blocks(self):
        ref = side("reference", prompt_token_array=[1, 2, 3])
        cand = side("candidate", prompt_token_array=[1, 2, 9])
        art = run(ref, cand)
        self.assertEqual(
            art.tokenizer_boundary.prompt_identity_evidence,
            PromptIdentityEvidence.DIVERGENT,
        )
        self.assertIn(C.INPUT_TOKEN_DIVERGENCE, art.blocking_reasons)

    def test_15_digest_only_is_not_array_equality(self):
        # counts+digests match but no explicit arrays
        art = run(side("reference"), side("candidate"), profile=CALIBRATED)
        tb = art.tokenizer_boundary
        self.assertEqual(tb.prompt_identity_evidence, PromptIdentityEvidence.DIGEST_ONLY)
        self.assertIsNone(tb.prompt_token_array_equal)
        self.assertTrue(tb.prompt_token_digest_equal)
        self.assertEqual(art.pass0_verdict, Pass0Verdict.LIMITED_COMPARISON_ALLOWED)
        self.assertFalse(art.oracle_eligibility.hf_default_greedy)

    def test_array_equal_calibrates_tokenizer(self):
        ref = side("reference", prompt_token_array=[1, 2, 3])
        cand = side("candidate", prompt_token_array=[1, 2, 3])
        art = run(ref, cand, profile=CALIBRATED)
        self.assertEqual(
            art.tokenizer_boundary.prompt_identity_evidence,
            PromptIdentityEvidence.ARRAY_EQUAL,
        )
        self.assertTrue(art.calibration_status.tokenizer_boundary_calibrated)
        self.assertEqual(art.tokenizer_boundary.confidence.value, "high")


class TestConfigSemantics(unittest.TestCase):
    def test_7_rms_norm_eps_unbound_llama(self):
        art = run(side("reference"), side("candidate"))
        self.assertFalse(art.config_semantics.rms_norm_eps_runtime_bound)
        self.assertEqual(art.config_semantics.status.value, "requires_fix_or_guard")
        self.assertIn("config_semantics_uncalibrated", codes(art))
        self.assertIn("rms_norm_eps_runtime_unbound", codes(art))
        self.assertFalse(art.oracle_eligibility.hf_default_greedy)

    def test_7b_eps_bound_qwen3(self):
        ref = side("reference", model_family="qwen3_dense_decoder")
        cand = side("candidate", model_family="qwen3_dense_decoder")
        art = run(ref, cand)
        self.assertTrue(art.config_semantics.rms_norm_eps_runtime_bound)
        self.assertTrue(art.calibration_status.config_semantics_calibrated)

    def test_config_fp_mismatch_blocks_mode_a(self):
        ref = side("reference", prompt_token_array=[1, 2, 3])
        cand = side("candidate", config_fingerprint="fnv1a64:other", prompt_token_array=[1, 2, 3])
        art = run(ref, cand)
        self.assertEqual(art.pass0_verdict, Pass0Verdict.COMPARISON_BLOCKED)
        self.assertIn(C.CONFIG_FINGERPRINT_MISMATCH, art.blocking_reasons)

    def test_config_fp_missing_downgrades(self):
        ref = side("reference", config_fingerprint=None, prompt_token_array=[1, 2, 3])
        cand = side("candidate", config_fingerprint=None, prompt_token_array=[1, 2, 3])
        art = run(ref, cand, profile=CALIBRATED)
        self.assertIn("runtime_config_fingerprint_missing", codes(art))
        self.assertEqual(art.pass0_verdict, Pass0Verdict.LIMITED_COMPARISON_ALLOWED)


class TestNumericPolicy(unittest.TestCase):
    def test_8_bf16_kv_write_unverified(self):
        art = run(side("reference"), side("candidate"))
        self.assertEqual(art.numeric_policy.kv_bf16_write.value, "truncate_or_unverified")
        self.assertEqual(art.numeric_policy.kv_f16_write.value, "truncate_or_unverified")
        self.assertFalse(art.numeric_policy.round_to_nearest_even)
        self.assertIn("kv_write_rounding_unverified", codes(art))
        self.assertIn("numeric_policy_uncalibrated", codes(art))

    def test_8b_calibrated_numeric_same_backend(self):
        ref = side("reference", prompt_token_array=[1, 2, 3])
        cand = side("candidate", prompt_token_array=[1, 2, 3])  # same backend "reference"
        art = run(ref, cand, profile=CALIBRATED)
        self.assertTrue(art.calibration_status.numeric_policy_calibrated)
        self.assertNotIn("tolerance_caveat", codes(art))

    def test_backend_differential_adds_tolerance_caveat(self):
        ref = side("reference", backend="reference", prompt_token_array=[1, 2, 3])
        cand = side("candidate", backend="avx2", prompt_token_array=[1, 2, 3])
        art = run(ref, cand, profile=CALIBRATED)
        self.assertIn("tolerance_caveat", codes(art))
        self.assertIn("fma_policy_backend_defined", codes(art))
        self.assertIn("reduction_order_backend_defined", codes(art))


class TestModesAndOracle(unittest.TestCase):
    def test_1_raw_greedy_comparable(self):
        ref = side("reference", prompt_token_array=[1, 2, 3])
        cand = side("candidate", prompt_token_array=[1, 2, 3])
        art = run(ref, cand, profile=CALIBRATED)
        self.assertEqual(art.pass0_verdict, Pass0Verdict.COMPARISON_ALLOWED)
        self.assertEqual(art.comparison_eligibility, ComparisonEligibility.COMPARABLE)
        self.assertTrue(art.decode_policy_identity.raw_greedy_equivalent)
        self.assertEqual(art.blocking_reasons, [])

    def test_9_mode_b_configuration_equivalence_is_subject(self):
        ref = side("reference", prompt_token_array=[1, 2, 3])
        cand = side("candidate", config_fingerprint="fnv1a64:other", prompt_token_array=[1, 2, 3])
        art = run(
            ref, cand, mode=ComparisonMode.RUNTIME_DIFFERENTIAL,
            submode=ModeBSubmode.CONFIGURATION_EQUIVALENCE, profile=CALIBRATED,
        )
        self.assertNotEqual(art.pass0_verdict, Pass0Verdict.COMPARISON_BLOCKED)

    def test_9b_mode_b_regression_blocks_on_cfg_mismatch(self):
        ref = side("reference", prompt_token_array=[1, 2, 3])
        cand = side("candidate", config_fingerprint="fnv1a64:other", prompt_token_array=[1, 2, 3])
        art = run(
            ref, cand, mode=ComparisonMode.RUNTIME_DIFFERENTIAL,
            submode=ModeBSubmode.RUNTIME_REGRESSION, profile=CALIBRATED,
        )
        self.assertEqual(art.pass0_verdict, Pass0Verdict.COMPARISON_BLOCKED)

    def test_5_mode_c_blocked(self):
        art = run(side("reference", prompt_token_array=[1, 2, 3]),
                  side("candidate", prompt_token_array=[1, 2, 3]),
                  mode=ComparisonMode.EXTERNAL_SEMANTIC)
        self.assertEqual(art.pass0_verdict, Pass0Verdict.COMPARISON_BLOCKED)
        self.assertIn(C.EXTERNAL_ORACLE_INELIGIBLE, art.blocking_reasons)
        self.assertFalse(art.oracle_eligibility.lis_internal_backend_differential)

    def test_10_hf_default_greedy_only_under_strict_conditions(self):
        ref = side("reference", prompt_token_array=[1, 2, 3])
        cand = side("candidate", prompt_token_array=[1, 2, 3])
        art = run(ref, cand, profile=CALIBRATED)
        self.assertTrue(art.oracle_eligibility.hf_default_greedy)
        # Live-default build must be false.
        live = run(side("reference", prompt_token_array=[1, 2, 3]),
                   side("candidate", prompt_token_array=[1, 2, 3]))
        self.assertFalse(live.oracle_eligibility.hf_default_greedy)

    def test_16_forced_token_runtime_artifact_channel(self):
        art = run(side("reference"), side("candidate"))
        forced = art.oracle_eligibility.hf_forced_token_runtime
        self.assertTrue(forced.potentially_eligible)
        self.assertTrue(forced.artifact_supported)
        self.assertEqual(forced.status, "eligible_with_source_bound_artifact_channel")
        self.assertNotIn("forced_prefix_report_json_channel_missing", codes(art))
        self.assertEqual(art.oracle_eligibility.oracle_scope.value, "internal_lis_only")

    def test_18_aggregator_escalation_not_registry(self):
        # external_oracle_ineligible: informational base, block only in Mode C
        self.assertEqual(
            effective_severity(C.EXTERNAL_ORACLE_INELIGIBLE, ComparisonMode.BACKEND_DIFFERENTIAL, None),
            ReasonSeverity.INFORMATIONAL,
        )
        self.assertEqual(
            effective_severity(C.EXTERNAL_ORACLE_INELIGIBLE, ComparisonMode.EXTERNAL_SEMANTIC, None),
            ReasonSeverity.BLOCK,
        )


class TestVerdictMatrixInvariants(unittest.TestCase):
    def test_12_no_verdict_strength_permits_first_divergence(self):
        for member in VerdictStrengthLimit:
            self.assertNotIn("first_divergence", member.value)

    def test_block_precedence_over_downgrade(self):
        ref = side("reference", repetition_penalty=1.2, prompt_token_array=[1, 2, 3])
        cand = side("candidate", repetition_penalty=1.0, prompt_token_array=[1, 2, 3])
        art = run(ref, cand)
        self.assertEqual(art.pass0_verdict, Pass0Verdict.COMPARISON_BLOCKED)

    def test_14_never_fail_open_on_missing_metadata(self):
        ref = RunSide(role="reference")
        cand = RunSide(role="candidate")
        art = run(ref, cand)
        self.assertNotEqual(art.comparison_eligibility, ComparisonEligibility.COMPARABLE)


class TestArtifactGolden(unittest.TestCase):
    def test_11_serialized_matches_golden(self):
        ref_raw = json.loads((FIXTURES / "run_llama_reference.json").read_text())
        cand_raw = json.loads((FIXTURES / "run_llama_avx2.json").read_text())
        ref = RunSide.from_run_report(ref_raw, "reference", prompt_token_array=[1, 2, 3, 4, 5])
        cand = RunSide.from_run_report(cand_raw, "candidate", prompt_token_array=[1, 2, 3, 4, 5])
        art = run(ref, cand)
        got = serialize(art)
        golden = json.loads(
            (FIXTURES / "golden" / "calibration_preflight_llama_mode_a.json").read_text()
        )
        self.assertEqual(got, golden)

    def test_golden_is_limited_comparison(self):
        golden = json.loads(
            (FIXTURES / "golden" / "calibration_preflight_llama_mode_a.json").read_text()
        )
        self.assertEqual(golden["pass0_verdict"], "limited_comparison_allowed")
        self.assertEqual(golden["verdict_strength_limit"], "checkpoint_confirmation_allowed")
        self.assertEqual(golden["oracle_eligibility"]["oracle_scope"], "internal_lis_only")


class TestFromRunReport(unittest.TestCase):
    def test_parses_nested_real_shape(self):
        raw = json.loads((FIXTURES / "run_llama_reference.json").read_text())
        s = RunSide.from_run_report(raw, "reference")
        self.assertEqual(s.model_family, "llama3_decoder")
        self.assertEqual(s.config_fingerprint, "fnv1a64:cccc000000000001")
        self.assertEqual(s.input_mode, "tokens")
        self.assertEqual(s.backend, "reference")
        self.assertEqual(s.kv_storage_dtype, "bf16")
        self.assertEqual(s.precision_path, "f32_accum;weights=bf16;kv=bf16")
        self.assertEqual(s.prompt_token_count, 5)

    def test_parses_flat_inspector_shape(self):
        raw = {
            "schema": "lis.execution_artifact/v1",
            "kind": "run_report",
            "manifest": {
                "model_family": "qwen3_dense_decoder",
                "config_fingerprint": "fnv1a64:x",
                "input_mode": "tokens",
                "backend": "avx2",
            },
            "report": {"prompt_sequences": [{"token_count": 2, "digest": "fnv1a64:y"}]},
        }
        s = RunSide.from_run_report(raw, "candidate")
        self.assertEqual(s.model_family, "qwen3_dense_decoder")
        self.assertEqual(s.backend, "avx2")
        self.assertEqual(s.prompt_token_count, 2)


class TestGate(unittest.TestCase):
    def test_blocked_gate_does_not_proceed(self):
        ref = side("reference", repetition_penalty=1.2)
        cand = side("candidate", repetition_penalty=1.0)
        gate = build_gate(run(ref, cand))
        self.assertFalse(gate.proceed)
        self.assertEqual(gate.verdict, Pass0Verdict.COMPARISON_BLOCKED)

    def test_limited_gate_proceeds_with_ceiling(self):
        gate = build_gate(run(side("reference"), side("candidate")))
        self.assertTrue(gate.proceed)
        self.assertEqual(
            gate.verdict_strength_limit, VerdictStrengthLimit.CHECKPOINT_CONFIRMATION_ALLOWED
        )


class TestReportMapping(unittest.TestCase):
    """Step S10 / §9: block reasons map into existing report reason codes."""

    def test_block_reasons_map_to_valid_existing_codes(self):
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        valid = set(contract["reason_code_enum"])
        for code in (
            C.INCOMPATIBLE_DECODE_POLICY,
            C.INPUT_TOKEN_DIVERGENCE,
            C.INCOMPATIBLE_MODEL_FAMILY,
            C.CONFIG_FINGERPRINT_MISMATCH,
            C.EXTERNAL_ORACLE_INELIGIBLE,
        ):
            mapped = map_block_reason(code)
            self.assertIsNotNone(mapped, code.value)
            self.assertIn(mapped, valid, f"{code.value} -> {mapped}")


if __name__ == "__main__":
    unittest.main()
