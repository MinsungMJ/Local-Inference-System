#!/usr/bin/env python3
"""Model-free behavior tests for P1 Pass 1 token localization."""

from __future__ import annotations

import importlib
import json
import unittest
from dataclasses import replace
from pathlib import Path

from lis_verify import (
    CanonicalRunReport,
    ComparisonMode,
    MismatchKind,
    ModeBSubmode,
    Pass1ReasonCode,
    Pass1Status,
    Pass2Disposition,
    PreflightInputs,
    PrefixAvailability,
    RunSide,
    VerdictStrengthLimit,
    build_gate,
    build_source_binding,
    locate_first_selected_token_mismatch,
    map_pass1_reason,
    map_pass1_status,
    run_calibration_preflight,
    run_token_localization,
    runtime_checkpoint_step_for_generated,
    serialize_pass1,
)
from lis_verify.pass1_inputs import MalformedRunReport


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tools" / "test_fixtures" / "token_localization"
GOLDEN = FIXTURES / "golden"


def load(name: str) -> CanonicalRunReport:
    return CanonicalRunReport.load(FIXTURES / name)


def gate_for(
    reference: CanonicalRunReport,
    candidate: CanonicalRunReport,
    *,
    ref_prompt=(1, 2, 3),
    cand_prompt=(1, 2, 3),
    ref_penalty=None,
    cand_penalty=None,
    mode=ComparisonMode.BACKEND_DIFFERENTIAL,
    submode=None,
):
    ref = RunSide.from_run_report(
        reference.materialize(),
        "reference",
        prompt_token_array=list(ref_prompt),
    )
    cand = RunSide.from_run_report(
        candidate.materialize(),
        "candidate",
        prompt_token_array=list(cand_prompt),
    )
    ref.repetition_penalty = ref_penalty
    cand.repetition_penalty = cand_penalty
    artifact = run_calibration_preflight(
        PreflightInputs(
            reference=ref,
            candidate=cand,
            declared_mode=mode,
            declared_submode=submode,
        )
    )
    return build_gate(artifact)


def run_pair(reference, candidate, **gate_kwargs):
    gate = gate_for(reference, candidate, **gate_kwargs)
    return run_token_localization(
        gate,
        reference,
        candidate,
        build_source_binding(reference, candidate),
    )


def with_report_changes(source: CanonicalRunReport, **changes):
    raw = source.materialize()
    report = raw.setdefault("report", {})
    report.update(changes)
    return CanonicalRunReport.from_object(raw)


def with_tokens(source: CanonicalRunReport, tokens):
    raw = source.materialize()
    report = raw.setdefault("report", {})
    report["selected_token_ids"] = list(tokens)
    report["selected_token_count"] = len(tokens)
    report.pop("selected_token_digest", None)
    return CanonicalRunReport.from_object(raw)


class SentinelRunReport:
    """Identity-only source that fails if Pass 1 materializes the report."""

    def __init__(self, identity):
        self.identity = identity

    def materialize(self):
        raise AssertionError("selected-token-bearing report was materialized")


class TestCanonicalRunReportIdentity(unittest.TestCase):
    def test_whitespace_and_object_key_order_do_not_change_identity(self):
        raw = load("run_reference_base.json").materialize()
        compact = json.dumps(raw, separators=(",", ":"))
        reversed_root = json.dumps(
            {key: raw[key] for key in reversed(list(raw))},
            indent=4,
        )
        left = CanonicalRunReport.from_json(compact)
        right = CanonicalRunReport.from_json(reversed_root)
        self.assertEqual(
            left.identity.run_report_sha256,
            right.identity.run_report_sha256,
        )

    def test_parsed_value_change_changes_identity(self):
        source = load("run_reference_base.json")
        changed = with_report_changes(source, stop_reason="model_eos")
        self.assertNotEqual(
            source.identity.run_report_sha256,
            changed.identity.run_report_sha256,
        )

    def test_duplicate_json_key_is_malformed(self):
        with self.assertRaises(MalformedRunReport):
            CanonicalRunReport.from_json(
                '{"schema":"lis.execution_artifact/v1",'
                '"kind":"run_report","kind":"other"}'
            )


class TestGateAndBinding(unittest.TestCase):
    def test_binding_mismatch_stops_before_materialization(self):
        reference = load("run_reference_base.json")
        candidate = load("run_candidate_equal.json")
        other = load("run_candidate_mismatch_n.json")
        gate = gate_for(reference, candidate)
        binding = build_source_binding(reference, candidate)
        result = run_token_localization(
            gate,
            SentinelRunReport(reference.identity),
            SentinelRunReport(other.identity),
            binding,
        )
        self.assertEqual(result.status, Pass1Status.INCONCLUSIVE)
        self.assertFalse(result.source_binding_verified)
        self.assertIn(
            Pass1ReasonCode.GATE_RUN_IDENTITY_INCONSISTENT,
            result.reason_codes,
        )

    def test_pass0_block_stops_before_materialization(self):
        reference = load("run_reference_base.json")
        candidate = load("run_candidate_equal.json")
        gate = gate_for(
            reference,
            candidate,
            ref_penalty=1.2,
            cand_penalty=1.0,
        )
        binding = build_source_binding(reference, candidate)
        result = run_token_localization(
            gate,
            SentinelRunReport(reference.identity),
            SentinelRunReport(candidate.identity),
            binding,
        )
        self.assertEqual(
            result.status, Pass1Status.COMPARISON_BLOCKED_BY_PASS0
        )
        self.assertEqual(
            result.pass2_disposition, Pass2Disposition.BLOCKED_BY_PASS0
        )

    def test_prompt_divergence_is_inherited_from_pass0(self):
        reference = load("run_reference_base.json")
        candidate = load("run_candidate_equal.json")
        result = run_pair(
            reference,
            candidate,
            ref_prompt=(1, 2, 3),
            cand_prompt=(1, 2, 9),
        )
        self.assertEqual(result.status, Pass1Status.INPUT_TOKEN_DIVERGENCE)
        self.assertIsNone(result.localization)

    def test_incoherent_gate_fails_closed_before_materialization(self):
        reference = load("run_reference_base.json")
        candidate = load("run_candidate_equal.json")
        gate = gate_for(reference, candidate)
        gate = replace(gate, proceed=False)
        result = run_token_localization(
            gate,
            SentinelRunReport(reference.identity),
            SentinelRunReport(candidate.identity),
            build_source_binding(reference, candidate),
        )
        self.assertEqual(result.status, Pass1Status.INCONCLUSIVE)
        self.assertFalse(result.source_binding_verified)


class TestLocalizationAlgorithm(unittest.TestCase):
    def test_equal_arrays(self):
        result = locate_first_selected_token_mismatch((1, 2), (1, 2))
        self.assertIsNone(result.generated_token_step)
        self.assertIsNone(result.runtime_checkpoint_step)
        self.assertEqual(result.matched_generated_prefix_length, 2)

    def test_both_arrays_empty(self):
        result = locate_first_selected_token_mismatch((), ())
        self.assertIsNone(result.generated_token_step)
        self.assertEqual(result.matched_generated_prefix_length, 0)

    def test_mismatch_at_zero(self):
        result = locate_first_selected_token_mismatch((1, 2), (9, 2))
        self.assertEqual(result.generated_token_step, 0)
        self.assertEqual(result.runtime_checkpoint_step, 1)
        self.assertEqual(result.mismatch_kind, MismatchKind.TOKEN_ID_MISMATCH)

    def test_mismatch_at_n(self):
        result = locate_first_selected_token_mismatch(
            (1, 2, 3, 4), (1, 2, 3, 9)
        )
        self.assertEqual(result.generated_token_step, 3)
        self.assertEqual(result.runtime_checkpoint_step, 4)
        self.assertEqual(result.matched_generated_prefix_length, 3)

    def test_strict_prefix_boundary(self):
        result = locate_first_selected_token_mismatch(
            (1, 2, 3, 4), (1, 2, 3)
        )
        self.assertEqual(result.generated_token_step, 3)
        self.assertEqual(result.runtime_checkpoint_step, 4)
        self.assertEqual(result.reference_selected_token_id, 4)
        self.assertIsNone(result.candidate_selected_token_id)
        self.assertEqual(
            result.mismatch_kind,
            MismatchKind.LENGTH_MISMATCH_OR_EARLY_TERMINATION,
        )

    def test_checkpoint_step_examples(self):
        self.assertEqual(runtime_checkpoint_step_for_generated(0), 1)
        self.assertEqual(runtime_checkpoint_step_for_generated(17), 18)


class TestPass1Behavior(unittest.TestCase):
    def setUp(self):
        self.reference = load("run_reference_base.json")

    def test_equal_fixture(self):
        result = run_pair(
            self.reference, load("run_candidate_equal.json")
        )
        self.assertEqual(
            result.status,
            Pass1Status.TOKEN_EQUIVALENT_ON_OBSERVED_RANGE,
        )
        self.assertEqual(
            result.pass2_disposition, Pass2Disposition.NOT_REQUIRED
        )
        self.assertIsNone(result.localization.generated_token_step)

    def test_mismatch_at_zero_fixture(self):
        result = run_pair(
            self.reference, load("run_candidate_mismatch_zero.json")
        )
        self.assertEqual(result.status, Pass1Status.FIRST_MISMATCH_FOUND)
        self.assertEqual(result.localization.generated_token_step, 0)
        self.assertEqual(result.prefix_for_reproduction.exact_token_ids, ())
        self.assertEqual(
            result.prefix_for_reproduction.availability,
            PrefixAvailability.EMBEDDED,
        )

    def test_mismatch_at_n_fixture(self):
        result = run_pair(
            self.reference, load("run_candidate_mismatch_n.json")
        )
        self.assertEqual(result.localization.generated_token_step, 3)
        self.assertEqual(result.localization.runtime_checkpoint_step, 4)
        self.assertEqual(
            result.prefix_for_reproduction.exact_token_ids,
            (101, 202, 303),
        )
        self.assertEqual(result.pass2_disposition, Pass2Disposition.READY)

    def test_strict_prefix_fixture(self):
        result = run_pair(
            self.reference, load("run_candidate_strict_prefix.json")
        )
        self.assertEqual(result.localization.generated_token_step, 3)
        self.assertIsNone(result.localization.candidate_selected_token_id)
        self.assertEqual(
            result.localization.mismatch_kind,
            MismatchKind.LENGTH_MISMATCH_OR_EARLY_TERMINATION,
        )
        self.assertEqual(
            result.candidate.selected_tokens.stop_reason, "model_eos"
        )

    def test_asymmetric_missing_array(self):
        result = run_pair(
            self.reference, load("run_candidate_missing_array.json")
        )
        self.assertEqual(
            result.status, Pass1Status.SELECTED_TOKEN_ARRAY_MISSING
        )
        self.assertIsNone(result.localization)
        self.assertEqual(
            result.pass2_disposition, Pass2Disposition.BLOCKED_BY_EVIDENCE
        )

    def test_two_digest_only_arrays_are_unverified(self):
        result = run_pair(
            load("run_reference_digest_only.json"),
            load("run_candidate_digest_only.json"),
        )
        self.assertEqual(
            result.status,
            Pass1Status.SELECTED_TOKEN_IDENTITY_UNVERIFIED,
        )
        self.assertIsNone(result.localization)
        self.assertIn(
            Pass1ReasonCode.SELECTED_TOKEN_ARRAY_MISSING,
            result.reason_codes,
        )

    def test_digest_only_mismatch_does_not_localize(self):
        reference = load("run_reference_digest_only.json")
        candidate = with_report_changes(
            load("run_candidate_digest_only.json"),
            selected_token_digest="fnv1a64:0000000000000000",
        )
        result = run_pair(reference, candidate)
        self.assertEqual(
            result.status,
            Pass1Status.SELECTED_TOKEN_IDENTITY_UNVERIFIED,
        )
        self.assertIsNone(result.localization)

    def test_array_count_inconsistency_is_inconclusive(self):
        candidate = with_report_changes(
            load("run_candidate_equal.json"),
            selected_token_count=4,
        )
        result = run_pair(self.reference, candidate)
        self.assertEqual(result.status, Pass1Status.INCONCLUSIVE)
        self.assertIn(
            Pass1ReasonCode.SELECTED_TOKEN_METADATA_INCONSISTENT,
            result.reason_codes,
        )

    def test_array_digest_contradiction_is_inconclusive(self):
        candidate = with_report_changes(
            load("run_candidate_equal.json"),
            selected_token_digest="fnv1a64:0000000000000000",
        )
        result = run_pair(self.reference, candidate)
        self.assertEqual(result.status, Pass1Status.INCONCLUSIVE)

    def test_bool_token_id_is_malformed(self):
        for invalid in (True, -1, "202"):
            with self.subTest(invalid=invalid):
                candidate = with_tokens(
                    load("run_candidate_equal.json"),
                    [101, invalid, 303],
                )
                result = run_pair(self.reference, candidate)
                self.assertEqual(result.status, Pass1Status.INCONCLUSIVE)

    def test_two_exact_empty_arrays_are_equivalent(self):
        reference = with_tokens(self.reference, [])
        candidate = with_tokens(load("run_candidate_equal.json"), [])
        result = run_pair(reference, candidate)
        self.assertEqual(
            result.status,
            Pass1Status.TOKEN_EQUIVALENT_ON_OBSERVED_RANGE,
        )
        self.assertEqual(
            result.localization.matched_generated_prefix_length, 0
        )

    def test_failed_execution_is_not_localized(self):
        candidate = with_report_changes(
            load("run_candidate_equal.json"),
            execution_status="error",
        )
        result = run_pair(self.reference, candidate)
        self.assertEqual(result.status, Pass1Status.INCONCLUSIVE)
        self.assertIsNone(result.localization)

    def test_cross_family_is_blocked_by_authoritative_pass0(self):
        raw = load("run_candidate_equal.json").materialize()
        raw["manifest"]["model"]["family"] = "qwen3_dense_decoder"
        candidate = CanonicalRunReport.from_object(raw)
        gate = gate_for(self.reference, candidate)
        result = run_token_localization(
            gate,
            self.reference,
            candidate,
            build_source_binding(self.reference, candidate),
        )
        self.assertEqual(
            result.status, Pass1Status.COMPARISON_BLOCKED_BY_PASS0
        )
        self.assertIsNone(result.localization)

    def test_conflicting_supported_array_locations_fail_closed(self):
        raw = load("run_candidate_equal.json").materialize()
        raw["selected_token_ids"] = [1, 2, 3]
        candidate = CanonicalRunReport.from_object(raw)
        result = run_pair(self.reference, candidate)
        self.assertEqual(result.status, Pass1Status.INCONCLUSIVE)

    def test_limited_comparison_preserves_ceiling(self):
        result = run_pair(
            self.reference, load("run_candidate_mismatch_n.json")
        )
        self.assertEqual(
            result.verdict_strength_limit,
            VerdictStrengthLimit.CHECKPOINT_CONFIRMATION_ALLOWED,
        )
        self.assertEqual(
            result.comparison_eligibility.value, "limited_comparison"
        )

    def test_pass0_configuration_equivalence_mismatch_is_not_reinferred(self):
        raw = load("run_candidate_equal.json").materialize()
        raw["manifest"]["config"]["fingerprint"]["hex"] = "eeeeeeeeeeeeeeee"
        candidate = CanonicalRunReport.from_object(raw)
        result = run_pair(
            self.reference,
            candidate,
            mode=ComparisonMode.RUNTIME_DIFFERENTIAL,
            submode=ModeBSubmode.CONFIGURATION_EQUIVALENCE,
        )
        self.assertEqual(
            result.status,
            Pass1Status.TOKEN_EQUIVALENT_ON_OBSERVED_RANGE,
        )
        self.assertEqual(
            result.compatibility.config_state, "accepted_by_pass0"
        )

    def test_token_localization_only_blocks_pass2_not_pass1(self):
        candidate = load("run_candidate_mismatch_n.json")
        gate = gate_for(self.reference, candidate)
        artifact = replace(
            gate.artifact,
            verdict_strength_limit=VerdictStrengthLimit.TOKEN_LOCALIZATION_ONLY,
        )
        gate = replace(
            gate,
            artifact=artifact,
            verdict_strength_limit=VerdictStrengthLimit.TOKEN_LOCALIZATION_ONLY,
        )
        result = run_token_localization(
            gate,
            self.reference,
            candidate,
            build_source_binding(self.reference, candidate),
        )
        self.assertEqual(result.status, Pass1Status.FIRST_MISMATCH_FOUND)
        self.assertEqual(
            result.pass2_disposition,
            Pass2Disposition.BLOCKED_BY_STRENGTH_LIMIT,
        )


class TestPrefixSerialization(unittest.TestCase):
    def _long_prefix_result(self, fixture_name, mismatch_token):
        reference = load(fixture_name)
        raw = reference.materialize()
        tokens = list(raw["report"]["selected_token_ids"])
        tokens[-1] = mismatch_token
        candidate = with_tokens(reference, tokens)
        return run_pair(reference, candidate)

    def test_prefix_length_64_is_embedded(self):
        result = self._long_prefix_result("run_prefix_64.json", 999)
        self.assertEqual(result.localization.generated_token_step, 64)
        artifact = serialize_pass1(result)
        prefix = artifact["prefix_for_reproduction"]
        self.assertEqual(prefix["availability"], "embedded")
        self.assertEqual(len(prefix["generated_prefix_token_ids"]), 64)

    def test_prefix_length_65_requires_exact_source(self):
        result = self._long_prefix_result("run_prefix_65.json", 999)
        self.assertEqual(result.localization.generated_token_step, 65)
        self.assertEqual(
            len(result.prefix_for_reproduction.exact_token_ids), 65
        )
        artifact = serialize_pass1(result)
        prefix = artifact["prefix_for_reproduction"]
        self.assertEqual(prefix["availability"], "exact_source_required")
        self.assertIsNone(prefix["generated_prefix_token_ids"])
        self.assertEqual(prefix["generated_prefix_token_count"], 65)


class TestArtifactAndSafeBoundary(unittest.TestCase):
    def setUp(self):
        self.reference = load("run_reference_base.json")

    def test_artifact_identity_and_no_confirmation_fields(self):
        result = run_pair(
            self.reference, load("run_candidate_mismatch_n.json")
        )
        artifact = serialize_pass1(result)
        self.assertEqual(artifact["schema"], "lis.execution_artifact/v1")
        self.assertEqual(artifact["kind"], "token_localization")
        self.assertNotIn("confirmed_divergence_at_checkpoint", artifact)
        self.assertNotIn("confirmed_first_divergence", artifact)
        self.assertIsNone(map_pass1_status(Pass1Status.FIRST_MISMATCH_FOUND))

    def test_local_reason_maps_only_to_safe_frozen_reason(self):
        self.assertEqual(
            map_pass1_reason(
                Pass1ReasonCode.SELECTED_TOKEN_METADATA_INCONSISTENT
            ),
            "malformed_artifact",
        )

    def test_calibration_reference_is_bounded_by_default(self):
        result = run_pair(
            self.reference, load("run_candidate_mismatch_n.json")
        )
        artifact = serialize_pass1(result)
        calibration = artifact["calibration_ref"]
        self.assertTrue(calibration["sha256"].startswith("sha256:"))
        self.assertIsNone(calibration["embedded"])
        self.assertIn("pass0_verdict", calibration["summary"])

    def test_mismatch_golden(self):
        result = run_pair(
            self.reference, load("run_candidate_mismatch_n.json")
        )
        expected = json.loads(
            (GOLDEN / "token_localization_mismatch_n.json").read_text()
        )
        self.assertEqual(serialize_pass1(result), expected)

    def test_equal_golden(self):
        result = run_pair(
            self.reference, load("run_candidate_equal.json")
        )
        expected = json.loads(
            (GOLDEN / "token_localization_equal.json").read_text()
        )
        self.assertEqual(serialize_pass1(result), expected)

    def test_redacted_prefix_golden(self):
        reference = load("run_prefix_65.json")
        raw = reference.materialize()
        tokens = list(raw["report"]["selected_token_ids"])
        tokens[-1] = 999
        candidate = with_tokens(reference, tokens)
        result = run_pair(reference, candidate)
        expected = json.loads(
            (GOLDEN / "token_localization_redacted_prefix.json").read_text()
        )
        self.assertEqual(serialize_pass1(result), expected)

    def test_result_model_has_no_confirmation_attributes(self):
        result = run_pair(
            self.reference, load("run_candidate_mismatch_n.json")
        )
        self.assertFalse(
            hasattr(result, "confirmed_divergence_at_checkpoint")
        )
        self.assertFalse(hasattr(result, "confirmed_first_divergence"))

    def test_core_has_no_numeric_or_runtime_evidence_dependencies(self):
        module = importlib.import_module("lis_verify.pass1")
        public_names = set(vars(module))
        for forbidden in ("logits", "layer_trace", "tolerance", "tensor_reader"):
            self.assertNotIn(forbidden, public_names)


if __name__ == "__main__":
    unittest.main()
