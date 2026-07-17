#!/usr/bin/env python3
"""Model-free Pass 3 gates, coverage, alignment, and digest policy tests."""

from __future__ import annotations

import copy
import unittest
from dataclasses import replace

from lis_verify import CanonicalLayerTrace, CanonicalRunReport
from lis_verify.pass2_model import (
    Pass2Status,
    Pass3Disposition,
    ReproductionEvidenceTier,
)
from lis_verify.pass3_inputs import CanonicalPass2Artifact
from lis_verify.pass3_model import (
    Pass3DownstreamDisposition,
    Pass3ReasonCode,
    Pass3Status,
    SummaryEvidenceLevel,
)

from .pass3_test_support import ready_case, run_case


def rerender(case, side):
    case[f"{side}_trace"] = CanonicalLayerTrace.from_object(
        case[f"{side}_raw"]
    )


class TestPass2IdentityAndGateOrder(unittest.TestCase):
    def test_typed_artifact_match(self):
        result = run_case(ready_case())
        self.assertTrue(result.pass2_object_artifact_coherence_verified)
        self.assertTrue(result.pass2_artifact_sha256.startswith("sha256:"))

    def test_each_typed_artifact_mismatch_fails_closed(self):
        mutations = (
            lambda value: replace(
                value,
                source_binding=replace(
                    value.source_binding,
                    reference_original_run_report_sha256="sha256:" + "0" * 64,
                ),
            ),
            lambda value: replace(
                value,
                localization_ref=replace(
                    value.localization_ref, sha256="sha256:" + "1" * 64
                ),
            ),
            lambda value: replace(value, warnings=value.warnings + ("changed",)),
        )
        for mutate in mutations:
            case = ready_case()
            case["pass2"] = mutate(case["pass2"])
            with self.subTest(pass2=case["pass2"]):
                result = run_case(case)
                self.assertEqual(
                    result.reason_codes,
                    (Pass3ReasonCode.PASS2_OBJECT_ARTIFACT_INCONSISTENT,),
                )

    def test_artifact_target_step_mismatch_is_a2_coherence_failure(self):
        case = ready_case()
        raw = case["pass2_artifact"].materialize_verified()
        raw["target"]["expected_runtime_checkpoint_step"] = 19
        case["pass2_artifact"] = CanonicalPass2Artifact.from_object(raw)
        self.assertEqual(
            run_case(case).reason_codes,
            (Pass3ReasonCode.PASS2_OBJECT_ARTIFACT_INCONSISTENT,),
        )

    def test_typed_status_and_disposition_mismatch_fail_closed(self):
        for field, value in (
            ("status", Pass2Status.INCONCLUSIVE),
            ("pass3_disposition", Pass3Disposition.BLOCKED_BY_REPRODUCTION),
        ):
            case = ready_case()
            object.__setattr__(case["pass2"], field, value)
            result = run_case(case)
            # Gate A1 is deliberately earlier than artifact coherence.
            self.assertEqual(
                result.reason_codes, (Pass3ReasonCode.PASS2_NOT_READY,)
            )

    def test_reproduction_request_only_stops_at_a1(self):
        case = ready_case()
        object.__setattr__(
            case["pass2"],
            "reproduction_evidence_tier",
            ReproductionEvidenceTier.REPRODUCTION_REQUEST_ONLY,
        )
        self.assertEqual(
            run_case(case).reason_codes,
            (Pass3ReasonCode.REPRODUCTION_REQUEST_ONLY,),
        )

    def test_canonical_hash_mismatch(self):
        case = ready_case()
        artifact = case["pass2_artifact"]
        case["pass2_artifact"] = CanonicalPass2Artifact(
            "sha256:" + "0" * 64, artifact.canonical_text
        )
        result = run_case(case)
        self.assertEqual(
            result.reason_codes,
            (Pass3ReasonCode.PASS2_ARTIFACT_IDENTITY_INCONSISTENT,),
        )

    def test_malformed_canonical_artifact_rejected_strictly(self):
        with self.assertRaises(Exception):
            CanonicalPass2Artifact.from_json('{"kind":"x","kind":"y"}')

    def test_no_trace_summary_access_when_pass2_not_ready(self):
        case = ready_case()
        object.__setattr__(case["pass2"], "status", Pass2Status.INCONCLUSIVE)

        class Sentinel:
            @property
            def identity(self):
                raise AssertionError("trace identity accessed before Gate A1")

            def materialize(self):
                raise AssertionError("trace summary accessed before Gate A1")

        case["reference_trace"] = Sentinel()
        case["candidate_trace"] = Sentinel()
        self.assertEqual(
            run_case(case).status, Pass3Status.COMPARISON_BLOCKED_BY_PASS2
        )

    def test_no_trace_summary_access_on_broken_binding(self):
        case = ready_case()
        wrong = copy.deepcopy(case["candidate_report"].materialize())
        wrong["report"]["selected_token_ids"][-1] = 123456
        case["candidate_report"] = CanonicalRunReport.from_object(wrong)

        class SentinelTrace(CanonicalLayerTrace):
            def materialize(self):
                raise AssertionError("summary accessed before both bindings")

        for side in ("reference", "candidate"):
            trace = case[f"{side}_trace"]
            case[f"{side}_trace"] = SentinelTrace(
                trace.identity, trace.canonical_text
            )
        result = run_case(case)
        self.assertEqual(result.status, Pass3Status.SOURCE_BINDING_INCONSISTENT)


class TestSourceBinding(unittest.TestCase):
    def test_independent_rerun_uses_reproduction_roles(self):
        result = run_case(ready_case(independent=True))
        self.assertEqual(result.reference_binding.role, "reference_reproduction")
        self.assertEqual(result.candidate_binding.role, "candidate_reproduction")
        self.assertEqual(
            result.pass2_evidence.reproduction_evidence_tier,
            "independent_rerun_verified",
        )

    def test_wrong_source_role_and_report_hash(self):
        case = ready_case()
        case["reference_report"], case["candidate_report"] = (
            case["candidate_report"], case["reference_report"]
        )
        self.assertEqual(
            run_case(case).reason_codes,
            (Pass3ReasonCode.RUN_REPORT_CANONICAL_SHA_INCONSISTENT,),
        )

    def test_artifact_set_id_mismatch(self):
        case = ready_case()
        case["reference_raw"]["artifact_set_id"] = (
            "aset1:99999999999999999999999999999999"
        )
        rerender(case, "reference")
        self.assertEqual(
            run_case(case).reason_codes,
            (Pass3ReasonCode.ARTIFACT_SET_ID_INCONSISTENT,),
        )

    def test_same_manifest_different_artifact_set_id_fails(self):
        self.test_artifact_set_id_mismatch()

    def test_forced_same_id_different_manifest_fails(self):
        case = ready_case()
        case["reference_raw"]["manifest"]["runtime"]["thread_count"] = 2
        rerender(case, "reference")
        self.assertEqual(
            run_case(case).reason_codes,
            (Pass3ReasonCode.SOURCE_BINDING_INCONSISTENT,),
        )

    def test_forced_same_id_different_report_hash_fails(self):
        case = ready_case()
        changed = case["reference_report"].materialize()
        changed["report"]["stop_reason"] = "changed"
        changed["artifact_set_id"] = case["reference_raw"]["artifact_set_id"]
        case["reference_report"] = CanonicalRunReport.from_object(changed)
        self.assertEqual(
            run_case(case).reason_codes,
            (Pass3ReasonCode.RUN_REPORT_CANONICAL_SHA_INCONSISTENT,),
        )

    def test_runtime_step_mismatch(self):
        case = ready_case()
        case["reference_raw"]["checkpoint_layout"][
            "runtime_checkpoint_step"
        ] = 17
        rerender(case, "reference")
        self.assertEqual(
            run_case(case).reason_codes,
            (Pass3ReasonCode.RUNTIME_CHECKPOINT_STEP_MISMATCH,),
        )

    def test_legacy_missing_binding_fields(self):
        case = ready_case()
        case["reference_raw"].pop("artifact_set_id")
        rerender(case, "reference")
        self.assertEqual(
            run_case(case).reason_codes,
            (Pass3ReasonCode.BINDING_METADATA_MISSING,),
        )


class TestCoverageAndLocalization(unittest.TestCase):
    def test_sparse_mismatch_interval(self):
        result = run_case(
            ready_case(candidate_digest_overrides={8: "sha256:" + "f" * 64})
        )
        self.assertEqual(result.status, Pass3Status.OBSERVABLE_MISMATCH_FOUND)
        self.assertEqual(result.first_observed_mismatch_coordinate.layer_index, 8)
        self.assertEqual(result.last_observed_equivalent_coordinate.layer_index, 4)
        self.assertEqual(result.suspect_interval.notation, "(4, 8]")
        self.assertEqual(result.suspect_interval.unobserved_layer_indices, (5, 6, 7))
        self.assertEqual(
            result.downstream_disposition,
            Pass3DownstreamDisposition.SUSPECT_INTERVAL_AVAILABLE,
        )

    def test_dense_mismatch(self):
        result = run_case(
            ready_case(
                reference_layers=(0, 1, 2),
                candidate_layers=(0, 1, 2),
                candidate_digest_overrides={2: "sha256:" + "a" * 64},
            )
        )
        self.assertEqual(result.suspect_interval.notation, "(1, 2]")

    def test_dense_no_mismatch(self):
        result = run_case(
            ready_case(reference_layers=(0, 1, 2), candidate_layers=(0, 1, 2))
        )
        self.assertEqual(
            result.status, Pass3Status.NO_MISMATCH_IN_CAPTURED_COVERAGE
        )

    def test_first_captured_mismatch(self):
        result = run_case(
            ready_case(candidate_digest_overrides={0: "sha256:" + "e" * 64})
        )
        self.assertEqual(result.suspect_interval.notation, "[entry, 0]")
        self.assertIsNone(result.last_observed_equivalent_coordinate)

    def test_no_mismatch_is_coverage_scoped(self):
        result = run_case(ready_case())
        self.assertEqual(
            result.status, Pass3Status.NO_MISMATCH_IN_CAPTURED_COVERAGE
        )
        self.assertEqual(result.evidence_level, SummaryEvidenceLevel.TIER1_BOUNDED_DIGEST)
        self.assertIn("not mathematical tensor equality", result.equality_semantics)

    def test_asymmetric_coverage_is_metadata(self):
        result = run_case(
            ready_case(
                reference_layers=(0, 4, 12),
                candidate_layers=(0, 8, 12),
            )
        )
        self.assertEqual(
            result.status, Pass3Status.NO_MISMATCH_IN_CAPTURED_COVERAGE
        )
        self.assertEqual([item.layer_index for item in result.coverage.reference_only], [4])
        self.assertEqual([item.layer_index for item in result.coverage.candidate_only], [8])
        self.assertIn(Pass3ReasonCode.ASYMMETRIC_COVERAGE, result.reason_codes)

    def test_empty_common_coverage(self):
        result = run_case(
            ready_case(reference_layers=(0,), candidate_layers=(4,))
        )
        self.assertEqual(result.status, Pass3Status.INSUFFICIENT_COMMON_COVERAGE)
        self.assertFalse(result.comparisons)

    def test_missing_state_is_preserved(self):
        result = run_case(
            ready_case(reference_captured=(0, 8, 12), candidate_captured=(0, 4, 12))
        )
        self.assertEqual(result.coverage.reference_missing[0].coordinate.layer_index, 4)
        self.assertEqual(result.coverage.reference_missing[0].state.value, "not_captured")

    def test_all_missing_states_remain_distinct(self):
        for state in ("unsupported", "malformed", "unexpectedly_absent"):
            case = ready_case(reference_captured=(0, 8, 12))
            case["reference_raw"]["checkpoint_layout"]["missing_coordinates"][0][
                "state"
            ] = state
            rerender(case, "reference")
            with self.subTest(state=state):
                result = run_case(case)
                self.assertEqual(result.coverage.reference_missing[0].state.value, state)


class TestAlignmentAndEvidence(unittest.TestCase):
    def assert_mutation_status(self, mutate, expected):
        case = ready_case()
        mutate(case["candidate_raw"])
        rerender(case, "candidate")
        self.assertEqual(run_case(case).status, expected)

    def test_shape_alignment(self):
        def mutate(raw):
            raw["layer_trace"][0]["shape"] = [1, 1, 4]
            raw["layer_trace"][0]["element_count"] = 4
            raw["layer_trace"][0]["digest"]["shape"] = [1, 1, 4]
        self.assert_mutation_status(mutate, Pass3Status.CHECKPOINT_ALIGNMENT_INCONSISTENT)

    def test_dtype_incompatible(self):
        self.assert_mutation_status(
            lambda raw: raw["layer_trace"][0].update(observed_dtype="fp16"),
            Pass3Status.CHECKPOINT_SUMMARY_MALFORMED,
        )

    def test_precision_path_alignment(self):
        case = ready_case()
        # Preserve side binding by changing both report and trace manifests,
        # then update the authoritative Pass 2 source hash/artifact.
        report = case["candidate_report"].materialize()
        report["manifest"]["runtime"]["precision_path"] = "fp32"
        case["candidate_report"] = CanonicalRunReport.from_object(report)
        case["candidate_raw"]["manifest"] = copy.deepcopy(report["manifest"])
        rerender(case, "candidate")
        binding = replace(
            case["pass2"].source_binding,
            candidate_original_run_report_sha256=(
                case["candidate_report"].identity.run_report_sha256
            ),
        )
        case["pass2"] = replace(case["pass2"], source_binding=binding)
        case["pass2_artifact"] = CanonicalPass2Artifact.from_result(case["pass2"])
        self.assertEqual(
            run_case(case).status,
            Pass3Status.CHECKPOINT_ALIGNMENT_INCONSISTENT,
        )

    def test_role_stage_batch_sequence_rejected_as_alignment(self):
        for field, value in (
            ("tensor_role", "other"),
            ("stage_order", 1),
            ("batch_index", 1),
            ("sequence_index", 1),
        ):
            def mutate(raw, field=field, value=value):
                for section in ("requested_coordinates", "captured_coordinates"):
                    raw["checkpoint_layout"][section][0][field] = value
                raw["layer_trace"][0][field] = value
                if field == "tensor_role":
                    raw["layer_trace"][0]["digest"][field] = value
            with self.subTest(field=field):
                self.assert_mutation_status(
                    mutate, Pass3Status.CHECKPOINT_ALIGNMENT_INCONSISTENT
                )

    def test_duplicate_coordinate(self):
        def mutate(raw):
            raw["checkpoint_layout"]["requested_coordinates"].append(
                copy.deepcopy(raw["checkpoint_layout"]["requested_coordinates"][0])
            )
        self.assert_mutation_status(mutate, Pass3Status.CHECKPOINT_ALIGNMENT_INCONSISTENT)

    def test_nonmonotonic_order(self):
        def mutate(raw):
            raw["checkpoint_layout"]["captured_coordinates"][0], raw["checkpoint_layout"]["captured_coordinates"][1] = (
                raw["checkpoint_layout"]["captured_coordinates"][1], raw["checkpoint_layout"]["captured_coordinates"][0]
            )
        self.assert_mutation_status(mutate, Pass3Status.CHECKPOINT_ALIGNMENT_INCONSISTENT)

    def test_unsupported_layout(self):
        self.assert_mutation_status(
            lambda raw: raw["checkpoint_layout"].update(layout_version=2),
            Pass3Status.UNSUPPORTED_CHECKPOINT_LAYOUT,
        )

    def test_qwen_model_family_is_unsupported(self):
        case = ready_case()
        report = case["candidate_report"].materialize()
        report["manifest"]["model"]["family"] = "qwen3_decoder"
        case["candidate_report"] = CanonicalRunReport.from_object(report)
        case["candidate_raw"]["manifest"] = copy.deepcopy(report["manifest"])
        rerender(case, "candidate")
        binding = replace(
            case["pass2"].source_binding,
            candidate_original_run_report_sha256=(
                case["candidate_report"].identity.run_report_sha256
            ),
        )
        case["pass2"] = replace(case["pass2"], source_binding=binding)
        case["pass2_artifact"] = CanonicalPass2Artifact.from_result(case["pass2"])
        self.assertEqual(run_case(case).status, Pass3Status.UNSUPPORTED_CHECKPOINT_LAYOUT)

    def test_legacy_layout(self):
        def mutate(raw):
            raw.pop("checkpoint_layout")
        self.assert_mutation_status(mutate, Pass3Status.UNSUPPORTED_CHECKPOINT_LAYOUT)

    def test_digest_algorithm_version_and_canonicalization(self):
        for field, value in (
            ("algorithm", "fnv1a64"),
            ("version", "other/v1"),
            ("canonicalization", "other"),
        ):
            def mutate(raw, field=field, value=value):
                raw["layer_trace"][0]["digest"][field] = value
            with self.subTest(field=field):
                self.assert_mutation_status(
                    mutate, Pass3Status.COMPARISON_POLICY_UNAVAILABLE
                )

    def test_missing_digest_has_no_uncalibrated_fallback(self):
        def mutate(raw):
            raw["layer_trace"][0].pop("digest")
            raw["layer_trace"][0]["available_summary_fields"].remove("digest")
        self.assert_mutation_status(
            mutate, Pass3Status.COMPARISON_POLICY_UNAVAILABLE
        )

    def test_no_tensor_payload_accepted(self):
        self.assert_mutation_status(
            lambda raw: raw["layer_trace"][0].update(values=[1.0, 2.0]),
            Pass3Status.CHECKPOINT_SUMMARY_MALFORMED,
        )


if __name__ == "__main__":
    unittest.main()
