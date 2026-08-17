#!/usr/bin/env python3
"""Focused P4-8 binding-first intra-layer parser tests."""

from __future__ import annotations

import copy
import inspect
import unittest
from dataclasses import FrozenInstanceError
from unittest import mock

import lis_verify
from lis_verify import pass4_inputs
from lis_verify.pass3_inputs import CanonicalLayerTrace
from lis_verify.pass4_contract import (
    INTRA_LAYER_STAGES,
    Pass4ReasonCode,
    Pass4Status,
)
from lis_verify.pass4_inputs import parse_pass4_intra_layer_inputs

from . import pass4_inputs_test_support as support
from . import pass4_parent_test_support


EXPECTED_PUBLIC_NAMES = (
    "Layer input",
    "Pre-attention RMSNorm output",
    "Q projection output",
    "K projection output",
    "V projection output",
    "RoPE-applied Q",
    "RoPE-applied K",
    "Attention pre-softmax scores",
    "Attention softmax output",
    "Attention context",
    "Attention output projection",
    "Post-attention residual",
    "Pre-MLP RMSNorm output",
    "MLP gate projection",
    "MLP up projection",
    "MLP gated activation",
    "MLP down projection",
)


class Pass4InputsCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = pass4_parent_test_support.two_generation_case()

    def parse(self, *, mutate_reference=None, mutate_candidate=None):
        prepared = support.prepare_inputs(
            self.seed,
            mutate_reference=mutate_reference,
            mutate_candidate=mutate_candidate,
        )
        outcome = parse_pass4_intra_layer_inputs(
            prepared.parent,
            prepared.reference_trace,
            prepared.candidate_trace,
        )
        return prepared, outcome

    def assert_terminal(self, outcome, status, reason):
        self.assertFalse(outcome.proceed)
        self.assertEqual(outcome.status, status)
        self.assertEqual(outcome.reason_codes, (reason,))
        self.assertIsNone(outcome.reference)
        self.assertIsNone(outcome.candidate)


class TestP4ProducerFixture(unittest.TestCase):
    def test_fixture_is_the_exact_dense_v1_additive_shape(self):
        raw = support.fixture_blocks()
        self.assertEqual(
            set(raw),
            {"intra_layer_checkpoint_layout", "intra_layer_trace"},
        )
        layout = raw["intra_layer_checkpoint_layout"]
        requested = layout["requested_coordinates"]
        captured = layout["captured_coordinates"]
        entries = raw["intra_layer_trace"]
        self.assertEqual(len(requested), 17)
        self.assertEqual(captured, requested)
        self.assertEqual(layout["missing_coordinates"], [])
        self.assertEqual(len(entries), 17)
        self.assertEqual(
            tuple(item["stage_id"] for item in requested),
            tuple(stage.stage_id for stage in INTRA_LAYER_STAGES),
        )
        self.assertEqual(
            tuple(item["public_name"] for item in entries),
            EXPECTED_PUBLIC_NAMES,
        )
        for coordinate, entry in zip(requested, entries):
            for field in coordinate:
                self.assertEqual(entry[field], coordinate[field])


class TestP4ValidParsing(Pass4InputsCase):
    def test_valid_bound_pair_normalizes_to_immutable_sources(self):
        prepared, outcome = self.parse()
        self.assertTrue(prepared.parent.proceed)
        self.assertTrue(outcome.proceed)
        self.assertEqual(outcome.reason_codes, ())
        for parsed in (outcome.reference, outcome.candidate):
            self.assertEqual(len(parsed.entries), 17)
            self.assertEqual(
                tuple(entry.coordinate for entry in parsed.entries),
                parsed.coverage.captured_coordinates,
            )
            self.assertTrue(parsed.header.digest_contract.frozen_policy_supported)
            self.assertFalse(parsed.header.full_tensor_payload_allowed)
            self.assertFalse(hasattr(parsed, "common_comparable"))
            with self.assertRaises(FrozenInstanceError):
                parsed.header.target_layer = 9

    def test_valid_sparse_partition_preserves_missing_state_and_order(self):
        def sparse(raw):
            layout = raw["intra_layer_checkpoint_layout"]
            requested = layout["requested_coordinates"]
            layout["captured_coordinates"] = copy.deepcopy(requested[:1])
            layout["missing_coordinates"] = [
                {
                    "coordinate": copy.deepcopy(coordinate),
                    "state": "not_captured",
                    "detail": "fixture omission",
                }
                for coordinate in requested[1:]
            ]
            raw["intra_layer_trace"] = raw["intra_layer_trace"][:1]

        prepared = support.prepare_inputs(
            self.seed,
            mutate_reference=sparse,
            mutate_candidate=sparse,
        )
        outcome = parse_pass4_intra_layer_inputs(
            prepared.parent,
            prepared.reference_trace,
            prepared.candidate_trace,
        )
        self.assertTrue(outcome.proceed)
        self.assertEqual(len(outcome.reference.entries), 1)
        self.assertEqual(len(outcome.reference.coverage.missing_coordinates), 16)
        self.assertEqual(
            outcome.reference.coverage.missing_coordinates[0].detail,
            "fixture omission",
        )

    def test_unknown_but_structurally_consistent_digest_is_a_p4_9_fact(self):
        def unknown_digest(raw):
            layout = raw["intra_layer_checkpoint_layout"]
            layout["digest_contract"]["version"] = "unknown.digest/v2"
            for entry in raw["intra_layer_trace"]:
                entry["digest"]["version"] = "unknown.digest/v2"

        prepared = support.prepare_inputs(
            self.seed,
            mutate_reference=unknown_digest,
            mutate_candidate=unknown_digest,
        )
        outcome = parse_pass4_intra_layer_inputs(
            prepared.parent,
            prepared.reference_trace,
            prepared.candidate_trace,
        )
        self.assertTrue(outcome.proceed)
        self.assertFalse(
            outcome.reference.header.digest_contract.frozen_policy_supported
        )
        self.assertFalse(
            outcome.candidate.header.digest_contract.frozen_policy_supported
        )

    def test_nonfinite_marker_allows_null_aggregates(self):
        def nonfinite(raw):
            entry = raw["intra_layer_trace"][0]
            entry.update(min=None, max=None, mean=None, l2=None, nan=1)

        _, outcome = self.parse(
            mutate_reference=nonfinite,
            mutate_candidate=nonfinite,
        )
        self.assertTrue(outcome.proceed)
        self.assertIsNone(outcome.reference.entries[0].mean_value)
        self.assertTrue(outcome.reference.entries[0].nan_present)


class TestP4BindingAndPrecedence(Pass4InputsCase):
    def test_parent_terminal_propagates_without_summary_materialization(self):
        terminal_seed = pass4_parent_test_support.two_generation_case(
            authoritative_mismatch_layer=None
        )
        prepared = support.prepare_inputs(terminal_seed)
        self.assertFalse(prepared.parent.proceed)
        with mock.patch.object(
            CanonicalLayerTrace,
            "materialize",
            side_effect=AssertionError("summary access is forbidden"),
        ) as materialize:
            outcome = parse_pass4_intra_layer_inputs(
                prepared.parent,
                prepared.reference_trace,
                prepared.candidate_trace,
            )
        self.assertEqual(outcome.status, prepared.parent.status)
        self.assertEqual(outcome.reason_codes, prepared.parent.reason_codes)
        materialize.assert_not_called()

    def test_both_trace_identities_precede_either_materialization(self):
        prepared = support.prepare_inputs(self.seed)
        wrong_candidate = CanonicalLayerTrace.from_object({"wrong": True})
        with mock.patch.object(
            CanonicalLayerTrace,
            "materialize",
            side_effect=AssertionError("identity gate was bypassed"),
        ) as materialize:
            outcome = parse_pass4_intra_layer_inputs(
                prepared.parent,
                prepared.reference_trace,
                wrong_candidate,
            )
        self.assert_terminal(
            outcome,
            Pass4Status.SOURCE_BINDING_INCONSISTENT,
            Pass4ReasonCode.TRACE_SHA_MISMATCH,
        )
        materialize.assert_not_called()

    def test_both_layouts_complete_before_either_entry_gate(self):
        def malformed_reference_entry(raw):
            raw["intra_layer_trace"][0].pop("public_name")

        def unsupported_candidate_layout(raw):
            raw["intra_layer_checkpoint_layout"]["layout_version"] = 2

        _, outcome = self.parse(
            mutate_reference=malformed_reference_entry,
            mutate_candidate=unsupported_candidate_layout,
        )
        self.assert_terminal(
            outcome,
            Pass4Status.UNSUPPORTED_INTRA_LAYER_LAYOUT,
            Pass4ReasonCode.INTRA_LAYOUT_OR_TAXONOMY_UNSUPPORTED,
        )

    def test_complete_aligned_layout_is_bound_to_parent_step_and_layer(self):
        def change_target(raw):
            layout = raw["intra_layer_checkpoint_layout"]
            layout["runtime_checkpoint_step"] = 19
            for collection in (
                layout["requested_coordinates"],
                layout["captured_coordinates"],
                raw["intra_layer_trace"],
            ):
                for item in collection:
                    item["runtime_checkpoint_step"] = 19

        prepared = support.prepare_inputs(
            self.seed,
            mutate_reference=change_target,
            mutate_candidate=change_target,
        )
        outcome = parse_pass4_intra_layer_inputs(
            prepared.parent,
            prepared.reference_trace,
            prepared.candidate_trace,
        )
        self.assert_terminal(
            outcome,
            Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT,
            Pass4ReasonCode.STEP_ALIGNMENT_MISMATCH,
        )


class TestP4LayoutAndCoverageFailures(Pass4InputsCase):
    def test_legacy_or_nonfrozen_layouts_are_unsupported(self):
        cases = (
            lambda raw: raw.pop("intra_layer_checkpoint_layout"),
            lambda raw: raw["intra_layer_checkpoint_layout"].update(
                layout_name="legacy_layout"
            ),
            lambda raw: raw["intra_layer_checkpoint_layout"].update(
                layout_version=2
            ),
            lambda raw: raw["intra_layer_checkpoint_layout"].update(
                model_family="qwen_decoder"
            ),
            lambda raw: raw["intra_layer_checkpoint_layout"].update(
                phase="prefill"
            ),
            lambda raw: raw["intra_layer_checkpoint_layout"].update(
                full_tensor_payload_allowed=True
            ),
        )
        for mutate in cases:
            with self.subTest(mutate=inspect.getsource(mutate).strip()):
                _, outcome = self.parse(mutate_candidate=mutate)
                self.assert_terminal(
                    outcome,
                    Pass4Status.UNSUPPORTED_INTRA_LAYER_LAYOUT,
                    Pass4ReasonCode.INTRA_LAYOUT_OR_TAXONOMY_UNSUPPORTED,
                )

    def test_nonexact_requested_stage_set_is_unsupported(self):
        def mutate(raw):
            raw["intra_layer_checkpoint_layout"][
                "requested_coordinates"
            ].pop()

        _, outcome = self.parse(mutate_candidate=mutate)
        self.assert_terminal(
            outcome,
            Pass4Status.UNSUPPORTED_INTRA_LAYER_LAYOUT,
            Pass4ReasonCode.REQUESTED_STAGE_SET_UNSUPPORTED,
        )

    def test_duplicate_or_out_of_order_declared_coverage_is_malformed(self):
        mutations = (
            lambda layout: layout["captured_coordinates"].reverse(),
            lambda layout: layout["captured_coordinates"].__setitem__(
                1, copy.deepcopy(layout["captured_coordinates"][0])
            ),
        )
        for mutate_layout in mutations:
            def mutate(raw, apply=mutate_layout):
                apply(raw["intra_layer_checkpoint_layout"])

            with self.subTest(mutate=inspect.getsource(mutate_layout).strip()):
                _, outcome = self.parse(mutate_candidate=mutate)
                self.assert_terminal(
                    outcome,
                    Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
                    Pass4ReasonCode.DUPLICATE_OR_OUT_OF_ORDER_COORDINATE,
                )

    def test_overlap_gap_and_bad_missing_state_are_partition_malformed(self):
        def overlap(raw):
            layout = raw["intra_layer_checkpoint_layout"]
            layout["missing_coordinates"] = [
                {
                    "coordinate": copy.deepcopy(
                        layout["requested_coordinates"][0]
                    ),
                    "state": "not_captured",
                    "detail": "duplicate partition member",
                }
            ]

        def gap(raw):
            raw["intra_layer_checkpoint_layout"][
                "captured_coordinates"
            ].pop()

        def bad_state(raw):
            layout = raw["intra_layer_checkpoint_layout"]
            coordinate = layout["captured_coordinates"].pop()
            layout["missing_coordinates"] = [
                {
                    "coordinate": coordinate,
                    "state": "invented",
                    "detail": "bad state",
                }
            ]

        for mutate in (overlap, gap, bad_state):
            with self.subTest(mutate=mutate.__name__):
                _, outcome = self.parse(mutate_candidate=mutate)
                self.assert_terminal(
                    outcome,
                    Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
                    Pass4ReasonCode.COVERAGE_PARTITION_MALFORMED,
                )

    def test_entry_coverage_disagreement_is_partition_malformed(self):
        def mutate(raw):
            raw["intra_layer_trace"].pop()

        _, outcome = self.parse(mutate_candidate=mutate)
        self.assert_terminal(
            outcome,
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
            Pass4ReasonCode.COVERAGE_PARTITION_MALFORMED,
        )


class TestP4EntryFailures(Pass4InputsCase):
    def test_summary_field_shape_count_and_aggregate_fail_closed(self):
        mutations = (
            lambda entry: entry.pop("public_name"),
            lambda entry: entry.update(public_name="invented name"),
            lambda entry: entry.update(shape=[]),
            lambda entry: entry.update(element_count=2),
            lambda entry: entry.update(observed_dtype=1),
            lambda entry: entry.update(precision_path=1),
            lambda entry: entry.update(min=2, max=1),
            lambda entry: entry.update(l2=-1),
            lambda entry: entry.update(mean=10**400),
            lambda entry: entry.update(mean=None, nan=0, inf=0),
            lambda entry: entry.update(nan=True),
        )
        for mutate_entry in mutations:
            def mutate(raw, apply=mutate_entry):
                apply(raw["intra_layer_trace"][0])

            with self.subTest(mutate=inspect.getsource(mutate_entry).strip()):
                _, outcome = self.parse(mutate_candidate=mutate)
                self.assert_terminal(
                    outcome,
                    Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
                    Pass4ReasonCode.SUMMARY_FIELD_MALFORMED,
                )

    def test_entry_phase_mismatch_has_alignment_classification(self):
        def mutate(raw):
            raw["intra_layer_trace"][0]["phase"] = "prefill"

        _, outcome = self.parse(mutate_candidate=mutate)
        self.assert_terminal(
            outcome,
            Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT,
            Pass4ReasonCode.PHASE_OR_POSITION_ALIGNMENT_MISMATCH,
        )

    def test_digest_structure_and_internal_consistency_fail_closed(self):
        mutations = (
            lambda raw: raw["intra_layer_checkpoint_layout"][
                "digest_contract"
            ].pop("algorithm"),
            lambda raw: raw["intra_layer_trace"][0]["digest"].pop("value"),
            lambda raw: raw["intra_layer_trace"][0]["digest"].update(
                value="not-a-sha"
            ),
            lambda raw: raw["intra_layer_trace"][0]["digest"].update(
                tensor_role="attention_norm_output"
            ),
            lambda raw: raw["intra_layer_trace"][0]["digest"].update(
                shape=[2]
            ),
        )
        for mutate in mutations:
            with self.subTest(mutate=inspect.getsource(mutate).strip()):
                _, outcome = self.parse(mutate_candidate=mutate)
                self.assert_terminal(
                    outcome,
                    Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
                    Pass4ReasonCode.DIGEST_FIELD_MALFORMED,
                )

    def test_duplicate_entry_order_is_not_silently_normalized(self):
        def mutate(raw):
            raw["intra_layer_trace"][1] = copy.deepcopy(
                raw["intra_layer_trace"][0]
            )

        _, outcome = self.parse(mutate_candidate=mutate)
        self.assert_terminal(
            outcome,
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
            Pass4ReasonCode.DUPLICATE_OR_OUT_OF_ORDER_COORDINATE,
        )

    def test_absolute_path_payload_is_rejected_without_disclosure(self):
        def mutate(raw):
            raw["intra_layer_trace"][0][
                "precision_path"
            ] = "/private/model"

        _, outcome = self.parse(mutate_candidate=mutate)
        self.assert_terminal(
            outcome,
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
            Pass4ReasonCode.PROHIBITED_PAYLOAD_PRESENT,
        )
        joined = " ".join(outcome.diagnostics)
        self.assertNotIn("/private", joined)
        self.assertNotIn("sha256:", joined)
        self.assertNotIn("aset1:", joined)

    def test_tensor_payload_key_defense_is_classified_at_its_local_gate(self):
        with self.assertRaises(pass4_inputs.Pass4InputError) as caught:
            pass4_inputs._reject_prohibited(
                {"nested": {"values": [1.0]}}, "candidate"
            )
        self.assertEqual(
            caught.exception.status,
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
        )
        self.assertEqual(
            caught.exception.reason,
            Pass4ReasonCode.PROHIBITED_PAYLOAD_PRESENT,
        )


class TestP4ScopeSentinels(unittest.TestCase):
    def test_p4_8_does_not_own_analysis_results_serialization_or_export(self):
        source = inspect.getsource(pass4_inputs)
        self.assertNotIn("analyze_coverage(", source)
        self.assertNotIn("Pass4Result(", source)
        self.assertNotIn("serialize_pass4", source)
        self.assertFalse(
            hasattr(lis_verify, "parse_pass4_intra_layer_inputs")
        )


if __name__ == "__main__":
    unittest.main()
