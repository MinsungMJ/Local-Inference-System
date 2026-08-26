#!/usr/bin/env python3
"""Focused P4-9 coverage, alignment, comparison, and interval tests."""

from __future__ import annotations

import inspect
import unittest
from dataclasses import replace
from unittest import mock

import lis_verify
from lis_verify import pass4
from lis_verify.pass3_inputs import CanonicalLayerTrace
from lis_verify.pass4 import localize_bound_intra_layer_inputs
from lis_verify.pass4_contract import (
    INTRA_LAYER_STAGES,
    Pass4ReasonCode,
    Pass4Status,
)
from lis_verify.pass4_inputs import parse_pass4_intra_layer_inputs
from lis_verify.pass4_model import Pass4ComparisonDecision

from . import pass4_inputs_test_support as input_support
from . import pass4_localization_test_support as support
from . import pass4_parent_test_support


class Pass4LocalizationCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = pass4_parent_test_support.two_generation_case()

    def parse(self, *, mutate_reference=None, mutate_candidate=None):
        prepared = input_support.prepare_inputs(
            self.seed,
            mutate_reference=mutate_reference,
            mutate_candidate=mutate_candidate,
        )
        parsed = parse_pass4_intra_layer_inputs(
            prepared.parent,
            prepared.reference_trace,
            prepared.candidate_trace,
        )
        return prepared, parsed

    def localize(self, *, mutate_reference=None, mutate_candidate=None):
        prepared, parsed = self.parse(
            mutate_reference=mutate_reference,
            mutate_candidate=mutate_candidate,
        )
        return prepared, parsed, localize_bound_intra_layer_inputs(parsed)


class TestPass4TerminalAdaptation(Pass4LocalizationCase):
    def test_wrong_api_type_is_the_only_raised_programming_failure(self):
        with self.assertRaisesRegex(TypeError, "parsed_inputs"):
            localize_bound_intra_layer_inputs(object())

    def test_valid_parent_terminal_is_preserved_without_local_evidence(self):
        terminal_seed = pass4_parent_test_support.two_generation_case(
            authoritative_mismatch_layer=None
        )
        prepared = input_support.prepare_inputs(terminal_seed)
        parsed = parse_pass4_intra_layer_inputs(
            prepared.parent,
            prepared.reference_trace,
            prepared.candidate_trace,
        )
        result = localize_bound_intra_layer_inputs(parsed)
        self.assertEqual(result.status, Pass4Status.NOT_APPLICABLE)
        self.assertEqual(result.reason_codes, parsed.reason_codes)
        self.assertIsNone(result.coverage)
        self.assertEqual(result.comparisons, ())

    def test_p4_8_unsupported_layout_is_preserved(self):
        def unsupported(raw):
            raw["intra_layer_checkpoint_layout"]["layout_version"] = 2

        _, parsed, result = self.localize(mutate_candidate=unsupported)
        self.assertFalse(parsed.proceed)
        self.assertEqual(
            result.status, Pass4Status.UNSUPPORTED_INTRA_LAYER_LAYOUT
        )
        self.assertEqual(result.reason_codes, parsed.reason_codes)
        self.assertIsNone(result.coverage)
        self.assertIsNone(result.target_runtime_checkpoint_step)

    def test_p4_8_malformed_summary_is_representable_without_target(self):
        def malformed(raw):
            raw["intra_layer_trace"][0].pop("public_name")

        _, parsed, result = self.localize(mutate_candidate=malformed)
        self.assertEqual(
            parsed.status, Pass4Status.CHECKPOINT_SUMMARY_MALFORMED
        )
        self.assertEqual(result.status, parsed.status)
        self.assertIsNone(result.coverage)
        self.assertIsNone(result.target_token_position)

    def test_p4_8_early_alignment_is_representable_without_coverage(self):
        mutation = support.change_target_step(19)
        prepared = input_support.prepare_inputs(
            self.seed,
            mutate_reference=mutation,
            mutate_candidate=mutation,
        )
        parsed = parse_pass4_intra_layer_inputs(
            prepared.parent,
            prepared.reference_trace,
            prepared.candidate_trace,
        )
        result = localize_bound_intra_layer_inputs(parsed)
        self.assertEqual(
            result.status, Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT
        )
        self.assertEqual(
            result.reason_codes,
            (Pass4ReasonCode.STEP_ALIGNMENT_MISMATCH,),
        )
        self.assertIsNone(result.coverage)
        self.assertIsNone(result.target_runtime_checkpoint_step)

    def test_trace_identity_failure_precedes_localization(self):
        prepared, _ = self.parse()
        wrong = CanonicalLayerTrace.from_object({"wrong": True})
        parsed = parse_pass4_intra_layer_inputs(
            prepared.parent,
            prepared.reference_trace,
            wrong,
        )
        with mock.patch.object(
            pass4,
            "_coverage",
            side_effect=AssertionError("terminal evidence was analyzed"),
        ):
            result = localize_bound_intra_layer_inputs(parsed)
        self.assertEqual(
            result.status, Pass4Status.SOURCE_BINDING_INCONSISTENT
        )
        self.assertEqual(result.reason_codes, parsed.reason_codes)

    def test_warning_overflow_blocks_instead_of_truncating(self):
        _, parsed = self.parse()
        parent = replace(
            parsed.parent_outcome,
            inherited_parent_warnings=tuple(
                f"parent-warning-{index}" for index in range(32)
            ),
        )
        overfull = replace(parsed, parent_outcome=parent)
        result = localize_bound_intra_layer_inputs(overfull)
        self.assertEqual(
            result.status, Pass4Status.COMPARISON_BLOCKED_BY_PASS3
        )
        self.assertEqual(
            result.reason_codes,
            (Pass4ReasonCode.PARENT_TYPED_ARTIFACT_INCOHERENT,),
        )
        self.assertEqual(result.warnings, ())

    def test_parent_and_local_warnings_merge_once_in_stable_order(self):
        _, parsed = self.parse()
        parent = replace(
            parsed.parent_outcome,
            inherited_parent_warnings=("shared-warning", "parent-warning"),
        )
        altered = replace(
            parsed,
            parent_outcome=parent,
            warnings=("shared-warning", "local-warning"),
        )
        result = localize_bound_intra_layer_inputs(altered)
        self.assertEqual(
            result.warnings,
            ("shared-warning", "parent-warning", "local-warning"),
        )


class TestPass4CoverageOutcomes(Pass4LocalizationCase):
    def test_no_common_capture_blocks_without_digest_comparison(self):
        with mock.patch.object(
            pass4,
            "_comparison",
            side_effect=AssertionError("digest comparison was reached"),
        ):
            _, _, result = self.localize(
                mutate_reference=support.select_stages(0),
                mutate_candidate=support.select_stages(1),
            )
        self.assertEqual(
            result.status,
            Pass4Status.INSUFFICIENT_COMMON_INTRA_LAYER_COVERAGE,
        )
        self.assertEqual(result.coverage.common_captured, ())
        self.assertEqual(result.comparisons, ())

    def test_single_common_match_uses_inherited_boundary(self):
        mutation = support.select_stages(13)
        _, _, result = self.localize(
            mutate_reference=mutation,
            mutate_candidate=mutation,
        )
        self.assertEqual(
            result.status,
            Pass4Status.MISMATCH_BOUNDED_TO_INHERITED_CLOSING_BOUNDARY,
        )
        self.assertEqual(len(result.comparisons), 1)
        self.assertEqual(
            result.suspect_interval.missing_local_stage_ids,
            tuple(stage.stage_id for stage in INTRA_LAYER_STAGES[14:]),
        )

    def test_single_common_mismatch_uses_virtual_entry_interval(self):
        select = support.select_stages(13)
        _, _, result = self.localize(
            mutate_reference=select,
            mutate_candidate=support.compose(
                select, support.mismatch_digests(13)
            ),
        )
        self.assertEqual(
            result.status,
            Pass4Status.OBSERVABLE_INTRA_LAYER_MISMATCH_FOUND,
        )
        self.assertEqual(
            result.suspect_interval.notation,
            "[selected_layer_entry, mlp_gate_projection]",
        )
        self.assertEqual(
            result.suspect_interval.missing_local_stage_ids,
            tuple(stage.stage_id for stage in INTRA_LAYER_STAGES[:13]),
        )

    def test_sparse_one_sided_coverage_retains_exact_gaps(self):
        _, _, result = self.localize(
            mutate_reference=support.select_stages(0, 3, 13),
            mutate_candidate=support.compose(
                support.select_stages(0, 13, 16),
                support.mismatch_digests(13),
            ),
        )
        self.assertEqual(
            tuple(item.stage_order for item in result.coverage.common_captured),
            (0, 13),
        )
        self.assertEqual(
            tuple(item.stage_order for item in result.coverage.reference_only),
            (3,),
        )
        self.assertEqual(
            tuple(item.stage_order for item in result.coverage.candidate_only),
            (16,),
        )
        self.assertEqual(
            result.reason_codes,
            (
                Pass4ReasonCode.LOCAL_DIGEST_MISMATCH,
                Pass4ReasonCode.ASYMMETRIC_COVERAGE_RETAINED,
            ),
        )
        self.assertEqual(
            result.suspect_interval.missing_local_stage_ids,
            tuple(stage.stage_id for stage in INTRA_LAYER_STAGES[1:13]),
        )

    def test_stateful_missing_metadata_is_preserved(self):
        mutation = support.select_stages(
            0,
            state="unsupported",
            detail="attention observation unavailable",
        )
        _, _, result = self.localize(
            mutate_reference=mutation,
            mutate_candidate=mutation,
        )
        missing = result.coverage.reference_missing
        self.assertEqual(len(missing), 16)
        self.assertEqual(missing[0].state.value, "unsupported")
        self.assertEqual(
            missing[0].detail, "attention observation unavailable"
        )

    def test_asymmetric_all_match_retains_secondary_reason(self):
        _, _, result = self.localize(
            mutate_reference=support.select_stages(0, 13),
            mutate_candidate=support.select_stages(0, 13, 16),
        )
        self.assertEqual(
            result.status,
            Pass4Status.MISMATCH_BOUNDED_TO_INHERITED_CLOSING_BOUNDARY,
        )
        self.assertEqual(
            result.reason_codes,
            (
                Pass4ReasonCode.NO_LOCAL_MISMATCH_BEFORE_INHERITED_BOUNDARY,
                Pass4ReasonCode.ASYMMETRIC_COVERAGE_RETAINED,
            ),
        )


class TestPass4AlignmentAndPolicy(Pass4LocalizationCase):
    def assert_alignment(self, mutation, expected_reason):
        with mock.patch.object(
            pass4,
            "_comparison",
            side_effect=AssertionError("alignment reached digest compare"),
        ):
            _, _, result = self.localize(mutate_candidate=mutation)
        self.assertEqual(
            result.status, Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT
        )
        self.assertEqual(result.reason_codes, (expected_reason,))
        self.assertTrue(result.coverage.common_captured)
        self.assertEqual(result.coverage.common_comparable, ())

    def test_shape_and_count_alignment_precedes_digest(self):
        self.assert_alignment(
            support.shape_mismatch(7),
            Pass4ReasonCode.SHAPE_OR_COUNT_ALIGNMENT_MISMATCH,
        )

    def test_dtype_alignment_precedes_digest_policy(self):
        self.assert_alignment(
            support.dtype_policy("bf16"),
            Pass4ReasonCode.DTYPE_OR_PRECISION_ALIGNMENT_MISMATCH,
        )

    def test_precision_alignment_precedes_digest(self):
        self.assert_alignment(
            support.precision_mismatch(8),
            Pass4ReasonCode.DTYPE_OR_PRECISION_ALIGNMENT_MISMATCH,
        )

    def test_typed_phase_alignment_defense(self):
        _, parsed = self.parse()
        entries = list(parsed.candidate.entries)
        entries[4] = replace(entries[4], phase="prefill")
        candidate = replace(parsed.candidate, entries=tuple(entries))
        altered = replace(parsed, candidate=candidate)
        with mock.patch.object(
            pass4,
            "_comparison",
            side_effect=AssertionError("phase mismatch reached digest"),
        ):
            result = localize_bound_intra_layer_inputs(altered)
        self.assertEqual(
            result.reason_codes,
            (Pass4ReasonCode.PHASE_OR_POSITION_ALIGNMENT_MISMATCH,),
        )

    def test_typed_public_name_alignment_defense(self):
        _, parsed = self.parse()
        entries = list(parsed.candidate.entries)
        entries[3] = replace(entries[3], public_name="Different name")
        candidate = replace(parsed.candidate, entries=tuple(entries))
        altered = replace(parsed, candidate=candidate)
        result = localize_bound_intra_layer_inputs(altered)
        self.assertEqual(
            result.reason_codes,
            (Pass4ReasonCode.STAGE_ROLE_OR_ORDER_ALIGNMENT_MISMATCH,),
        )

    def test_unknown_digest_policy_blocks_after_complete_alignment(self):
        with mock.patch.object(
            pass4,
            "_comparison",
            side_effect=AssertionError("unknown policy reached digest"),
        ):
            _, _, result = self.localize(
                mutate_candidate=support.unknown_digest_policy()
            )
        self.assertEqual(
            result.status, Pass4Status.COMPARISON_POLICY_UNAVAILABLE
        )
        self.assertEqual(
            result.reason_codes,
            (Pass4ReasonCode.DIGEST_CONTRACT_UNKNOWN,),
        )
        self.assertEqual(
            result.coverage.common_comparable,
            result.coverage.common_captured,
        )
        self.assertEqual(result.comparisons, ())


class TestPass4DigestLocalization(Pass4LocalizationCase):
    def test_all_dense_matches_close_at_exact_parent_boundary(self):
        prepared, _, result = self.localize()
        self.assertEqual(
            result.status,
            Pass4Status.MISMATCH_BOUNDED_TO_INHERITED_CLOSING_BOUNDARY,
        )
        self.assertEqual(len(result.comparisons), 17)
        self.assertTrue(all(item.equivalent for item in result.comparisons))
        self.assertEqual(
            result.closing_boundary_decision.parent_coordinate,
            prepared.parent.parent.parent_first_mismatch_coordinate,
        )
        self.assertEqual(
            result.suspect_interval.notation,
            "(mlp_down_projection, parent:layer_output]",
        )
        self.assertEqual(
            result.inherited_pass3_reason_codes,
            ("pass3.observable_mismatch_found",),
        )
        self.assertTrue(result.inherited_pass0_reason_codes)

    def test_first_middle_and_final_mismatch_intervals(self):
        cases = (
            (0, "[selected_layer_entry, layer_input]", None),
            (
                13,
                "(mlp_norm_output, mlp_gate_projection]",
                "mlp_norm_output",
            ),
            (
                16,
                "(mlp_gated_activation, mlp_down_projection]",
                "mlp_gated_activation",
            ),
        )
        for order, notation, previous in cases:
            with self.subTest(stage_order=order):
                _, _, result = self.localize(
                    mutate_candidate=support.mismatch_digests(order)
                )
                self.assertEqual(
                    result.status,
                    Pass4Status.OBSERVABLE_INTRA_LAYER_MISMATCH_FOUND,
                )
                self.assertEqual(
                    result.first_observed_local_mismatch_coordinate.stage_order,
                    order,
                )
                self.assertEqual(result.suspect_interval.notation, notation)
                actual_previous = (
                    None
                    if result.last_observed_equivalent_coordinate is None
                    else result.last_observed_equivalent_coordinate.stage_id
                )
                self.assertEqual(actual_previous, previous)

    def test_multiple_mismatches_retain_later_comparisons(self):
        _, _, result = self.localize(
            mutate_candidate=support.mismatch_digests(5, 13)
        )
        self.assertEqual(len(result.comparisons), 17)
        self.assertEqual(
            result.first_observed_local_mismatch_coordinate.stage_order, 5
        )
        self.assertEqual(
            result.comparisons[13].decision,
            Pass4ComparisonDecision
            .MISMATCHING_OBSERVED_REPRESENTATION_DIGEST,
        )

    def test_aggregates_are_not_a_digest_fallback_or_decision_field(self):
        _, _, result = self.localize(
            mutate_candidate=support.aggregate_only_change(6)
        )
        self.assertEqual(
            result.status,
            Pass4Status.MISMATCH_BOUNDED_TO_INHERITED_CLOSING_BOUNDARY,
        )
        self.assertTrue(result.comparisons[6].equivalent)

    def test_localization_is_deterministic(self):
        _, parsed = self.parse(
            mutate_candidate=support.mismatch_digests(7, 12)
        )
        expected = localize_bound_intra_layer_inputs(parsed)
        for _ in range(50):
            self.assertEqual(
                localize_bound_intra_layer_inputs(parsed), expected
            )


class TestPass4LocalizationScope(unittest.TestCase):
    def test_core_has_no_parser_digest_serializer_io_or_export_scope(self):
        source = inspect.getsource(localize_bound_intra_layer_inputs)
        for prohibited in (
            "canonical_intra_layer_digest_stream(",
            "intra_layer_digest_sha256(",
            "def serialize",
            "def to_json",
            "json.dumps",
            "hashlib",
            "Path(",
            "open(",
            "sorted(",
        ):
            with self.subTest(prohibited=prohibited):
                self.assertNotIn(prohibited, source)
        self.assertFalse(
            hasattr(lis_verify, "localize_bound_intra_layer_inputs")
        )


if __name__ == "__main__":
    unittest.main()
