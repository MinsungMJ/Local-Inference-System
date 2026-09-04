#!/usr/bin/env python3
"""Model-free behavior tests for P1 Pass 2 reproduction verification."""

from __future__ import annotations

import ast
import json
import unittest
from dataclasses import replace
from pathlib import Path

from lis_verify import (
    COMPUTED_STEP_EVIDENCE,
    TRACE_STEP_EVIDENCE,
    THREAD_COUNT_CAVEAT,
    CanonicalRunReport,
    ComparisonMode,
    Pass2Disposition,
    Pass2ReasonCode,
    Pass2Status,
    Pass3Disposition,
    PreflightInputs,
    ReproductionEvidenceTier,
    RunSide,
    VerdictStrengthLimit,
    build_gate,
    build_source_binding,
    map_pass2_reason,
    map_pass2_status,
    run_calibration_preflight,
    run_prefix_policy_reproduction,
    run_token_localization,
    serialize_pass2,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tools" / "test_fixtures" / "prefix_policy_reproduction"
GOLDEN = FIXTURES / "golden"
MODULES = ROOT / "tools" / "lis_verify"


def load(name: str) -> CanonicalRunReport:
    return CanonicalRunReport.load(FIXTURES / name)


def changed(source: CanonicalRunReport, mutate) -> CanonicalRunReport:
    raw = source.materialize()
    mutate(raw)
    return CanonicalRunReport.from_object(raw)


def pass1_for(
    reference: CanonicalRunReport,
    candidate: CanonicalRunReport,
    *,
    ref_penalty=None,
    cand_penalty=None,
    strength=None,
    mode=ComparisonMode.BACKEND_DIFFERENTIAL,
):
    ref = RunSide.from_run_report(
        reference.materialize(),
        "reference",
        prompt_token_array=[1, 2, 3],
    )
    cand = RunSide.from_run_report(
        candidate.materialize(),
        "candidate",
        prompt_token_array=[1, 2, 3],
    )
    ref.repetition_penalty = ref_penalty
    cand.repetition_penalty = cand_penalty
    artifact = run_calibration_preflight(
        PreflightInputs(
            reference=ref,
            candidate=cand,
            declared_mode=mode,
        )
    )
    gate = build_gate(artifact)
    if strength is not None:
        artifact = replace(artifact, verdict_strength_limit=strength)
        gate = replace(
            gate,
            artifact=artifact,
            verdict_strength_limit=strength,
        )
    return run_token_localization(
        gate,
        reference,
        candidate,
        build_source_binding(reference, candidate),
    )


def bound_pair():
    return (
        load("reference_original_bound.json"),
        load("candidate_original_bound.json"),
    )


def ready_inputs():
    reference, candidate = bound_pair()
    return pass1_for(reference, candidate), reference, candidate


def rebind_pass1(pass1, reference, candidate):
    return replace(
        pass1,
        source_binding=build_source_binding(reference, candidate),
        reference=(
            replace(pass1.reference, identity=reference.identity)
            if pass1.reference is not None
            else None
        ),
        candidate=(
            replace(pass1.candidate, identity=candidate.identity)
            if pass1.candidate is not None
            else None
        ),
        source_binding_verified=True,
    )


def long_prefix_inputs():
    reference, candidate = bound_pair()

    def set_reference(raw):
        tokens = list(range(1000, 1081))
        raw["manifest"]["runtime"]["generation_limit"] = len(tokens)
        raw["report"]["selected_token_count"] = len(tokens)
        raw["report"]["selected_token_ids"] = tokens

    def set_candidate(raw):
        tokens = list(range(1000, 1081))
        tokens[-1] = 9999
        raw["manifest"]["runtime"]["generation_limit"] = len(tokens)
        raw["report"]["selected_token_count"] = len(tokens)
        raw["report"]["selected_token_ids"] = tokens

    reference = changed(reference, set_reference)
    candidate = changed(candidate, set_candidate)
    return pass1_for(reference, candidate), reference, candidate


class BindingOnlyIdentity:
    def __init__(self, identity):
        self.run_report_sha256 = identity.run_report_sha256

    @property
    def model_fingerprint(self):
        raise AssertionError("model identity was accessed before binding")

    @property
    def config_fingerprint(self):
        raise AssertionError("config identity was accessed before binding")

    @property
    def input_fingerprint(self):
        raise AssertionError("input identity was accessed before binding")


class SentinelReport:
    def __init__(self, identity):
        self.identity = BindingOnlyIdentity(identity)

    def materialize(self):
        raise AssertionError("original metadata was accessed before binding")


class TestPass1DispositionMapping(unittest.TestCase):
    def test_ready_continues(self):
        pass1, reference, candidate = ready_inputs()
        result = run_prefix_policy_reproduction(
            pass1, reference, candidate
        )
        self.assertEqual(result.status, Pass2Status.REPRODUCTION_VERIFIED)

    def test_not_required(self):
        reference, candidate = bound_pair()

        def equalize(raw):
            raw["report"]["selected_token_ids"][-1] = 501

        candidate = changed(candidate, equalize)
        pass1 = pass1_for(reference, candidate)
        result = run_prefix_policy_reproduction(
            pass1, reference, candidate
        )
        self.assertEqual(
            result.status, Pass2Status.NO_MISMATCH_TO_REPRODUCE
        )
        self.assertEqual(result.pass3_disposition, Pass3Disposition.NOT_REQUIRED)

    def test_blocked_by_pass0(self):
        reference, candidate = bound_pair()
        pass1 = pass1_for(
            reference,
            candidate,
            ref_penalty=1.2,
            cand_penalty=1.0,
        )
        result = run_prefix_policy_reproduction(
            pass1,
            SentinelReport(reference.identity),
            SentinelReport(candidate.identity),
        )
        self.assertEqual(
            result.status, Pass2Status.COMPARISON_BLOCKED_BY_PASS0
        )
        self.assertEqual(
            result.pass3_disposition, Pass3Disposition.BLOCKED_BY_PASS0
        )

    def test_blocked_by_evidence(self):
        reference, candidate = bound_pair()

        def remove_array(raw):
            raw["report"].pop("selected_token_ids")

        candidate = changed(candidate, remove_array)
        pass1 = pass1_for(reference, candidate)
        result = run_prefix_policy_reproduction(
            pass1, reference, candidate
        )
        self.assertEqual(
            result.status, Pass2Status.TOKEN_LOCALIZATION_NOT_AVAILABLE
        )
        self.assertEqual(
            result.pass3_disposition,
            Pass3Disposition.BLOCKED_BY_PASS1_EVIDENCE,
        )

    def test_blocked_by_strength_limit(self):
        reference, candidate = bound_pair()
        pass1 = pass1_for(
            reference,
            candidate,
            strength=VerdictStrengthLimit.TOKEN_LOCALIZATION_ONLY,
        )
        result = run_prefix_policy_reproduction(
            pass1, reference, candidate
        )
        self.assertEqual(
            result.status, Pass2Status.UNSUPPORTED_REPRODUCTION_MODE
        )
        self.assertIn(
            Pass2ReasonCode.VERDICT_STRENGTH_LIMIT_BLOCKS_REPRODUCTION,
            result.reason_codes,
        )


class TestOriginalSourceBinding(unittest.TestCase):
    def test_both_original_hashes_match(self):
        pass1, reference, candidate = ready_inputs()
        result = run_prefix_policy_reproduction(
            pass1, reference, candidate
        )
        self.assertTrue(result.source_binding.reference_original_verified)
        self.assertTrue(result.source_binding.candidate_original_verified)
        self.assertTrue(result.source_binding_verified)

    def test_reference_hash_mismatch_fails_closed(self):
        pass1, _, candidate = ready_inputs()
        result = run_prefix_policy_reproduction(
            pass1,
            load("reference_original_hash_mismatch.json"),
            candidate,
        )
        self.assertEqual(
            result.status, Pass2Status.SOURCE_BINDING_INCONSISTENT
        )
        self.assertFalse(
            result.source_binding.reference_original_verified
        )

    def test_candidate_hash_mismatch_fails_closed(self):
        pass1, reference, _ = ready_inputs()
        result = run_prefix_policy_reproduction(
            pass1,
            reference,
            load("candidate_original_hash_mismatch.json"),
        )
        self.assertEqual(
            result.status, Pass2Status.SOURCE_BINDING_INCONSISTENT
        )
        self.assertFalse(
            result.source_binding.candidate_original_verified
        )

    def test_metadata_not_accessed_until_both_hashes_match(self):
        pass1, reference, _ = ready_inputs()
        sentinel = SentinelReport(reference.identity)
        result = run_prefix_policy_reproduction(
            pass1,
            sentinel,
            load("candidate_original_hash_mismatch.json"),
        )
        self.assertEqual(
            result.status, Pass2Status.SOURCE_BINDING_INCONSISTENT
        )

    def test_reproduction_identity_mismatch(self):
        pass1, reference, candidate = ready_inputs()
        result = run_prefix_policy_reproduction(
            pass1,
            reference,
            candidate,
            reference_reproduction=load(
                "reproduction_source_binding_mismatch.json"
            ),
            candidate_reproduction=load(
                "candidate_reproduction_verified.json"
            ),
        )
        self.assertEqual(
            result.status, Pass2Status.SOURCE_BINDING_INCONSISTENT
        )
        self.assertFalse(result.source_binding.reproduction_verified)


class TestOriginalMetadata(unittest.TestCase):
    def _assert_malformed(self, fixture):
        normal_pass1, _, candidate = ready_inputs()
        reference = load(fixture)
        pass1 = rebind_pass1(normal_pass1, reference, candidate)
        result = run_prefix_policy_reproduction(
            pass1, reference, candidate
        )
        self.assertEqual(result.status, Pass2Status.INCONCLUSIVE)
        self.assertIn(
            Pass2ReasonCode.REPRODUCTION_ARTIFACT_MALFORMED,
            result.reason_codes,
        )

    def test_missing_original_binary(self):
        self._assert_malformed("reference_original_missing_binary.json")

    def test_missing_original_prompt_count(self):
        self._assert_malformed(
            "reference_original_missing_prompt_count.json"
        )

    def test_missing_original_batch(self):
        self._assert_malformed("reference_original_missing_batch.json")

    def test_missing_original_thread_count(self):
        self._assert_malformed(
            "reference_original_missing_thread_count.json"
        )


class TestPrefixGate(unittest.TestCase):
    def test_n_zero_empty_prefix(self):
        reference, candidate = bound_pair()

        def first_reference(raw):
            raw["manifest"]["runtime"]["generation_limit"] = 1
            raw["report"]["selected_token_count"] = 1
            raw["report"]["selected_token_ids"] = [501]

        def first_candidate(raw):
            raw["manifest"]["runtime"]["generation_limit"] = 1
            raw["report"]["selected_token_count"] = 1
            raw["report"]["selected_token_ids"] = [999]

        reference = changed(reference, first_reference)
        candidate = changed(candidate, first_candidate)
        pass1 = pass1_for(reference, candidate)
        result = run_prefix_policy_reproduction(
            pass1, reference, candidate
        )
        self.assertEqual(result.target.generated_token_step, 0)
        self.assertEqual(result.prefix_reproduction.expected_token_count, 0)
        self.assertEqual(result.status, Pass2Status.REPRODUCTION_VERIFIED)

    def test_n_17_prefix_verified(self):
        pass1, reference, candidate = ready_inputs()
        result = run_prefix_policy_reproduction(
            pass1, reference, candidate
        )
        self.assertEqual(result.target.generated_token_step, 17)
        self.assertEqual(result.prefix_reproduction.status, "verified")
        self.assertEqual(
            result.prefix_reproduction.expected_token_count, 17
        )

    def test_redacted_prefix_with_exact_source(self):
        pass1, reference, candidate = long_prefix_inputs()
        exact = pass1.prefix_for_reproduction.exact_token_ids
        prefix = replace(pass1.prefix_for_reproduction, exact_token_ids=())
        pass1 = replace(
            pass1,
            prefix_for_reproduction=prefix,
            reference=None,
            candidate=None,
        )
        result = run_prefix_policy_reproduction(
            pass1,
            reference,
            candidate,
            exact_prefix_source=exact,
        )
        self.assertEqual(result.status, Pass2Status.REPRODUCTION_VERIFIED)

    def test_redacted_prefix_without_exact_source(self):
        pass1, reference, candidate = long_prefix_inputs()
        prefix = replace(pass1.prefix_for_reproduction, exact_token_ids=())
        pass1 = replace(
            pass1,
            prefix_for_reproduction=prefix,
            reference=None,
            candidate=None,
        )
        result = run_prefix_policy_reproduction(
            pass1, reference, candidate
        )
        self.assertEqual(
            result.status, Pass2Status.PREFIX_MATERIAL_UNAVAILABLE
        )
        self.assertEqual(
            result.reproduction_evidence_tier,
            ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY,
        )

    def test_prefix_token_mismatch(self):
        pass1, reference, candidate = ready_inputs()
        result = run_prefix_policy_reproduction(
            pass1,
            reference,
            candidate,
            reference_reproduction=load("reference_prefix_mismatch.json"),
            candidate_reproduction=load(
                "candidate_reproduction_verified.json"
            ),
        )
        self.assertEqual(
            result.status, Pass2Status.PREFIX_REPRODUCTION_FAILED
        )
        self.assertEqual(result.prefix_reproduction.first_diff_index, 5)

    def test_prefix_length_mismatch(self):
        pass1, reference, candidate = ready_inputs()
        exact = pass1.prefix_for_reproduction.exact_token_ids[:-1]
        result = run_prefix_policy_reproduction(
            pass1,
            reference,
            candidate,
            exact_prefix_source=exact,
        )
        self.assertEqual(
            result.status, Pass2Status.PREFIX_REPRODUCTION_FAILED
        )
        self.assertEqual(
            result.prefix_reproduction.mismatch_kind, "length_mismatch"
        )

    def test_prefix_digest_mismatch(self):
        pass1, reference, candidate = ready_inputs()
        exact = list(pass1.prefix_for_reproduction.exact_token_ids)
        exact[4] += 1
        result = run_prefix_policy_reproduction(
            pass1,
            reference,
            candidate,
            exact_prefix_source=exact,
        )
        self.assertEqual(
            result.status, Pass2Status.PREFIX_REPRODUCTION_FAILED
        )
        self.assertIn(
            Pass2ReasonCode.PREFIX_DIGEST_MISMATCH,
            result.reason_codes,
        )


class TestPolicyGate(unittest.TestCase):
    def test_binary_fingerprint_match(self):
        pass1, reference, candidate = ready_inputs()
        result = run_prefix_policy_reproduction(
            pass1, reference, candidate
        )
        self.assertTrue(
            result.policy_reproduction.build_continuity_verified
        )

    def test_original_binary_fingerprint_mismatch(self):
        reference = load("reference_original_bound.json")
        candidate = load("candidate_original_binary_mismatch.json")
        pass1 = pass1_for(reference, candidate)
        result = run_prefix_policy_reproduction(
            pass1, reference, candidate
        )
        self.assertEqual(
            result.status,
            Pass2Status.DECODE_POLICY_REPRODUCTION_FAILED,
        )

    def test_runtime_differential_allows_distinct_role_binaries(self):
        reference = load("reference_original_bound.json")
        candidate = load("candidate_original_binary_mismatch.json")
        pass1 = pass1_for(
            reference,
            candidate,
            mode=ComparisonMode.RUNTIME_DIFFERENTIAL,
        )
        result = run_prefix_policy_reproduction(
            pass1, reference, candidate
        )
        self.assertEqual(result.status, Pass2Status.REPRODUCTION_VERIFIED)
        self.assertTrue(
            result.policy_reproduction.build_continuity_verified
        )

    def test_runtime_differential_rejects_role_binary_drift(self):
        reference = load("reference_original_bound.json")
        candidate = load("candidate_original_binary_mismatch.json")
        pass1 = pass1_for(
            reference,
            candidate,
            mode=ComparisonMode.RUNTIME_DIFFERENTIAL,
        )
        result = run_prefix_policy_reproduction(
            pass1,
            reference,
            candidate,
            reference_reproduction=load("decode_policy_mismatch.json"),
            candidate_reproduction=changed(
                load("candidate_reproduction_verified.json"),
                lambda raw: raw["manifest"]["binary"].update(
                    {"fingerprint": "fnv1a64:bbbbbbbbbbbbbbbb"}
                ),
            ),
        )
        self.assertEqual(
            result.status,
            Pass2Status.DECODE_POLICY_REPRODUCTION_FAILED,
        )

    def test_reproduction_binary_compared_to_original(self):
        pass1, reference, candidate = ready_inputs()
        result = run_prefix_policy_reproduction(
            pass1,
            reference,
            candidate,
            reference_reproduction=load("decode_policy_mismatch.json"),
            candidate_reproduction=load(
                "candidate_reproduction_verified.json"
            ),
        )
        self.assertEqual(
            result.status,
            Pass2Status.DECODE_POLICY_REPRODUCTION_FAILED,
        )

    def test_original_multithread_warning_is_nonblocking(self):
        reference = load("reference_original_thread_count_gt_1.json")
        candidate = load("candidate_original_bound.json")
        pass1 = pass1_for(reference, candidate)
        result = run_prefix_policy_reproduction(
            pass1, reference, candidate
        )
        self.assertEqual(result.status, Pass2Status.REPRODUCTION_VERIFIED)
        self.assertIn(THREAD_COUNT_CAVEAT, result.warnings)

    def test_reproduction_multithread_warning_is_nonblocking(self):
        pass1, reference, candidate = ready_inputs()
        result = run_prefix_policy_reproduction(
            pass1,
            reference,
            candidate,
            reference_reproduction=load(
                "reference_reproduction_multithreaded.json"
            ),
            candidate_reproduction=load(
                "candidate_reproduction_verified.json"
            ),
        )
        self.assertEqual(result.status, Pass2Status.REPRODUCTION_VERIFIED)
        self.assertIn(THREAD_COUNT_CAVEAT, result.warnings)


class TestCheckpointAndContextGates(unittest.TestCase):
    def test_computed_checkpoint_step_evidence(self):
        pass1, reference, candidate = ready_inputs()
        result = run_prefix_policy_reproduction(
            pass1, reference, candidate
        )
        checkpoint = result.checkpoint_step_reproduction
        self.assertEqual(checkpoint.expected_runtime_checkpoint_step, 18)
        self.assertEqual(checkpoint.evidence, COMPUTED_STEP_EVIDENCE)
        self.assertFalse(
            checkpoint.materialized_checkpoint_artifact_verified
        )
        self.assertNotEqual(COMPUTED_STEP_EVIDENCE, TRACE_STEP_EVIDENCE)

    def test_checkpoint_step_mapping_mismatch(self):
        pass1, reference, candidate = ready_inputs()
        localization = replace(
            pass1.localization, runtime_checkpoint_step=99
        )
        pass1 = replace(pass1, localization=localization)
        result = run_prefix_policy_reproduction(
            pass1, reference, candidate
        )
        self.assertEqual(
            result.status, Pass2Status.CHECKPOINT_STEP_MAPPING_MISMATCH
        )

    def test_context_position_match(self):
        pass1, reference, candidate = ready_inputs()
        result = run_prefix_policy_reproduction(
            pass1, reference, candidate
        )
        positions = {
            run.context_position
            for run in result.context_reproduction.runs
        }
        self.assertEqual(positions, {20})

    def test_context_position_mismatch(self):
        pass1, reference, candidate = ready_inputs()
        result = run_prefix_policy_reproduction(
            pass1,
            reference,
            candidate,
            reference_reproduction=load(
                "context_position_mismatch.json"
            ),
            candidate_reproduction=load(
                "candidate_reproduction_verified.json"
            ),
        )
        self.assertEqual(
            result.status, Pass2Status.CONTEXT_POSITION_MISMATCH
        )

    def test_batch_size_mismatch(self):
        pass1, reference, candidate = ready_inputs()
        reproduction = changed(
            load("reference_reproduction_verified.json"),
            lambda raw: raw["manifest"]["runtime"].update(batch_size=2),
        )
        result = run_prefix_policy_reproduction(
            pass1,
            reference,
            candidate,
            reference_reproduction=reproduction,
            candidate_reproduction=load(
                "candidate_reproduction_verified.json"
            ),
        )
        self.assertEqual(
            result.status, Pass2Status.CONTEXT_POSITION_MISMATCH
        )


class TestEvidenceTierAndReadiness(unittest.TestCase):
    def test_original_pair_boundary_consistent(self):
        pass1, reference, candidate = ready_inputs()
        result = run_prefix_policy_reproduction(
            pass1, reference, candidate
        )
        self.assertEqual(
            result.reproduction_evidence_tier,
            ReproductionEvidenceTier.ORIGINAL_PAIR_BOUNDARY_CONSISTENT,
        )
        self.assertEqual(result.pass3_disposition, Pass3Disposition.READY)

    def test_independent_rerun_verified(self):
        pass1, reference, candidate = ready_inputs()
        result = run_prefix_policy_reproduction(
            pass1,
            reference,
            candidate,
            reference_reproduction=load(
                "reference_reproduction_verified.json"
            ),
            candidate_reproduction=load(
                "candidate_reproduction_verified.json"
            ),
        )
        self.assertEqual(
            result.reproduction_evidence_tier,
            ReproductionEvidenceTier.INDEPENDENT_RERUN_VERIFIED,
        )
        self.assertEqual(result.pass3_disposition, Pass3Disposition.READY)

    def test_request_only_cannot_be_reproduction_verified(self):
        pass1, reference, candidate = ready_inputs()
        result = run_prefix_policy_reproduction(
            pass1, reference, candidate
        )
        with self.assertRaises(ValueError):
            replace(
                result,
                reproduction_evidence_tier=(
                    ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY
                ),
            )

    def test_failure_is_not_pass3_ready(self):
        pass1, reference, candidate = ready_inputs()
        localization = replace(
            pass1.localization, runtime_checkpoint_step=99
        )
        result = run_prefix_policy_reproduction(
            replace(pass1, localization=localization),
            reference,
            candidate,
        )
        self.assertNotEqual(
            result.pass3_disposition, Pass3Disposition.READY
        )

    def test_artifact_preserves_tier_and_disposition(self):
        pass1, reference, candidate = ready_inputs()
        artifact = serialize_pass2(
            run_prefix_policy_reproduction(
                pass1, reference, candidate
            )
        )
        self.assertEqual(artifact["pass3_disposition"], "ready")
        self.assertEqual(
            artifact["reproduction_evidence_tier"],
            "original_pair_boundary_consistent",
        )


class TestSafetyAndReportBoundary(unittest.TestCase):
    def test_no_confirmation_fields(self):
        pass1, reference, candidate = ready_inputs()
        artifact = serialize_pass2(
            run_prefix_policy_reproduction(
                pass1, reference, candidate
            )
        )

        def keys(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield key
                    yield from keys(child)
            elif isinstance(value, list):
                for child in value:
                    yield from keys(child)

        all_keys = set(keys(artifact))
        self.assertNotIn("confirmed_divergence_at_checkpoint", all_keys)
        self.assertNotIn("confirmed_first_divergence", all_keys)

    def test_no_numeric_framework_dependencies(self):
        imported_roots = set()
        for name in ("pass2.py", "pass2_inputs.py", "pass2_artifact.py"):
            tree = ast.parse((MODULES / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported_roots.update(
                        alias.name.split(".")[0] for alias in node.names
                    )
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported_roots.add(node.module.split(".")[0])
        self.assertTrue(
            imported_roots.isdisjoint({"torch", "numpy", "tensorflow"})
        )

    def test_verified_status_has_no_frozen_report_mapping(self):
        self.assertIsNone(
            map_pass2_status(Pass2Status.REPRODUCTION_VERIFIED)
        )
        self.assertEqual(
            map_pass2_reason(Pass2ReasonCode.PREFIX_TOKEN_MISMATCH),
            "prefix_reproduction_failed",
        )


class TestArtifactGolden(unittest.TestCase):
    def _assert_golden(self, name, result):
        expected = json.loads((GOLDEN / name).read_text(encoding="utf-8"))
        self.assertEqual(serialize_pass2(result), expected)

    def test_verified_golden(self):
        pass1, reference, candidate = ready_inputs()
        self._assert_golden(
            "prefix_policy_reproduction_verified.json",
            run_prefix_policy_reproduction(
                pass1, reference, candidate
            ),
        )

    def test_independent_rerun_golden(self):
        pass1, reference, candidate = ready_inputs()
        self._assert_golden(
            "prefix_policy_reproduction_verified_independent_rerun.json",
            run_prefix_policy_reproduction(
                pass1,
                reference,
                candidate,
                reference_reproduction=load(
                    "reference_reproduction_verified.json"
                ),
                candidate_reproduction=load(
                    "candidate_reproduction_verified.json"
                ),
            ),
        )

    def test_prefix_failure_golden(self):
        pass1, reference, candidate = ready_inputs()
        self._assert_golden(
            "prefix_reproduction_failed.json",
            run_prefix_policy_reproduction(
                pass1,
                reference,
                candidate,
                reference_reproduction=load(
                    "reference_prefix_mismatch.json"
                ),
                candidate_reproduction=load(
                    "candidate_reproduction_verified.json"
                ),
            ),
        )

    def test_checkpoint_mismatch_golden(self):
        pass1, reference, candidate = ready_inputs()
        pass1 = replace(
            pass1,
            localization=replace(
                pass1.localization, runtime_checkpoint_step=99
            ),
        )
        self._assert_golden(
            "checkpoint_step_mapping_mismatch.json",
            run_prefix_policy_reproduction(
                pass1, reference, candidate
            ),
        )


if __name__ == "__main__":
    unittest.main()
