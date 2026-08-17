#!/usr/bin/env python3
"""Construction, invariant, and fixture-parity tests for the Pass 4 model."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from lis_verify import pass4_contract, pass4_model
from lis_verify.pass3_model import (
    CheckpointCoordinate,
    CoverageState,
    DIGEST_DECISION_FIELD,
    DIGEST_DECISION_SEMANTICS,
    DIGEST_MATCH_SEMANTICS,
    SummaryEvidenceLevel,
    SuspectInterval,
)
from lis_verify.pass4_contract import (
    DIGEST_VERSION,
    EVIDENCE_LEVEL,
    NONCLAIMS,
    PASS3_DIGEST_VERSION,
    STAGE_IDS,
    STATUS_TO_DISPOSITION,
    IntraLayerSideCoverage,
    MissingIntraLayerCoordinate,
    ParentSourceIdentity,
    Pass3ParentClassification,
    Pass3ParentRole,
    Pass4Disposition,
    Pass4ReasonCode,
    Pass4Status,
    Pass4SuspectInterval,
    UnsupportedIntraLayerLayoutError,
    coordinate_from_mapping,
    requested_coordinates,
)
from lis_verify.pass4_model import (
    COMPARISON_STATUSES,
    FIELD_POLICY,
    FROZEN_EVIDENCE_CEILING,
    LOCAL_COVERAGE_OUTCOME_FOR_STATUS,
    MAX_COMPARISONS,
    MAX_DETAIL_BYTES,
    MAX_IDENTIFIER_BYTES,
    MAX_WARNINGS,
    Pass3ParentEvidence,
    Pass4ClosingBoundaryDecision,
    Pass4Comparison,
    Pass4ComparisonDecision,
    Pass4CoverageAnalysis,
    Pass4EvidenceCeiling,
    Pass4LocalCoverageOutcome,
    Pass4Result,
    Pass4SourceBinding,
)

from . import pass4_test_support as support


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tools" / "test_fixtures" / "intra_layer_localization"
CONTRACT = FIXTURES / "pass4_contract.json"
EXAMPLES = FIXTURES / "pass4_contract_examples.json"


def parse_parent(raw) -> CheckpointCoordinate:
    return CheckpointCoordinate(
        raw["runtime_checkpoint_step"],
        raw["layer_index"],
        raw["tensor_role"],
        raw["batch_index"],
        raw["sequence_index"],
        raw["stage_order"],
        raw["execution_ordinal"],
    )


def parse_interval(raw) -> Pass4SuspectInterval:
    return Pass4SuspectInterval(
        raw["start_kind"],
        (
            coordinate_from_mapping(raw["start_local_coordinate"])
            if raw["start_local_coordinate"] is not None
            else None
        ),
        raw["start_inclusive"],
        raw["end_kind"],
        (
            coordinate_from_mapping(raw["end_local_coordinate"])
            if raw["end_local_coordinate"] is not None
            else None
        ),
        (
            parse_parent(raw["end_parent_coordinate"])
            if raw["end_parent_coordinate"] is not None
            else None
        ),
        raw["end_evidence_origin"],
        raw["end_inclusive"],
        tuple(raw["missing_local_stage_ids"]),
        raw["notation"],
    )


class TestPass4Comparison(unittest.TestCase):
    def test_valid_comparison_and_semantics(self):
        coordinate = support.stages()[13]
        equivalent = support.comparison(coordinate)
        mismatching = support.comparison(coordinate, equivalent=False)
        self.assertTrue(equivalent.equivalent)
        self.assertFalse(mismatching.equivalent)
        self.assertEqual(
            equivalent.decision_semantics, DIGEST_MATCH_SEMANTICS
        )
        self.assertEqual(
            mismatching.decision_semantics, DIGEST_DECISION_SEMANTICS
        )

    def test_comparison_is_immutable(self):
        comparison = support.comparison(support.stages()[0])
        with self.assertRaises((AttributeError, TypeError)):
            comparison.decision = (
                Pass4ComparisonDecision
                .MISMATCHING_OBSERVED_REPRESENTATION_DIGEST
            )

    def test_decision_must_agree_with_the_observed_digests(self):
        coordinate = support.stages()[0]
        with self.assertRaisesRegex(ValueError, "contradicts"):
            Pass4Comparison(
                coordinate,
                (2,),
                support.digest("a"),
                support.digest("a"),
                Pass4ComparisonDecision
                .MISMATCHING_OBSERVED_REPRESENTATION_DIGEST,
            )
        with self.assertRaisesRegex(ValueError, "contradicts"):
            Pass4Comparison(
                coordinate,
                (2,),
                support.digest("a"),
                support.digest("b"),
                Pass4ComparisonDecision
                .EQUIVALENT_OBSERVED_REPRESENTATION_DIGEST,
            )

    def test_malformed_digest_shape_and_enum_are_rejected(self):
        coordinate = support.stages()[0]
        equivalent = (
            Pass4ComparisonDecision.EQUIVALENT_OBSERVED_REPRESENTATION_DIGEST
        )
        for digest in (
            "sha256:" + "A" * 64,
            "sha1:" + "a" * 40,
            "a" * 64,
            None,
        ):
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                Pass4Comparison(
                    coordinate, (2,), digest, digest, equivalent
                )
        for shape in ((), (0,), (True,), (-1,), [2]):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                Pass4Comparison(
                    coordinate,
                    shape,
                    support.digest("a"),
                    support.digest("a"),
                    equivalent,
                )
        with self.assertRaisesRegex(ValueError, "unknown Pass 4 comparison"):
            Pass4Comparison(
                coordinate,
                (2,),
                support.digest("a"),
                support.digest("a"),
                "equivalent_observed_representation_digest",
            )
        with self.assertRaises(TypeError):
            Pass4Comparison(
                support.parent_coordinate(),
                (2,),
                support.digest("a"),
                support.digest("a"),
                equivalent,
            )


class TestPass4Coverage(unittest.TestCase):
    def test_valid_full_and_sparse_coverage(self):
        coordinates = support.stages()
        full = support.coverage(coordinates)
        self.assertEqual(full.common_captured, coordinates)
        self.assertEqual(full.requested_coordinates, coordinates)
        self.assertFalse(full.asymmetric)
        sparse = support.coverage(
            (coordinates[0], coordinates[4]), (coordinates[0], coordinates[6])
        )
        self.assertEqual(sparse.common_captured, (coordinates[0],))
        self.assertEqual(sparse.reference_only, (coordinates[4],))
        self.assertEqual(sparse.candidate_only, (coordinates[6],))
        self.assertTrue(sparse.asymmetric)
        self.assertEqual(len(sparse.reference_missing), 15)

    def test_supplied_derived_fields_are_verified_not_recomputed(self):
        coordinates = support.stages()
        reference = support.side_coverage((coordinates[0], coordinates[4]))
        candidate = support.side_coverage((coordinates[0], coordinates[6]))
        with self.assertRaisesRegex(ValueError, "common_captured"):
            Pass4CoverageAnalysis(
                reference,
                candidate,
                (coordinates[0], coordinates[4]),
                (coordinates[0],),
                (coordinates[4],),
                (coordinates[6],),
            )
        with self.assertRaisesRegex(ValueError, "reference_only"):
            Pass4CoverageAnalysis(
                reference,
                candidate,
                (coordinates[0],),
                (coordinates[0],),
                (),
                (coordinates[6],),
            )
        with self.assertRaisesRegex(ValueError, "candidate_only"):
            Pass4CoverageAnalysis(
                reference,
                candidate,
                (coordinates[0],),
                (coordinates[0],),
                (coordinates[4],),
                (),
            )

    def test_duplicate_and_out_of_order_coverage_is_rejected_without_sorting(
        self,
    ):
        coordinates = support.stages()
        reference = support.side_coverage((coordinates[0], coordinates[4]))
        candidate = support.side_coverage((coordinates[0], coordinates[4]))
        reversed_pair = (coordinates[4], coordinates[0])
        with self.assertRaisesRegex(ValueError, "out of contracted order"):
            Pass4CoverageAnalysis(
                reference,
                candidate,
                reversed_pair,
                reversed_pair,
                (),
                (),
            )
        self.assertEqual(reversed_pair, (coordinates[4], coordinates[0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            Pass4CoverageAnalysis(
                reference,
                candidate,
                (coordinates[0], coordinates[0]),
                (coordinates[0], coordinates[0]),
                (),
                (),
            )
        with self.assertRaisesRegex(ValueError, "immutable tuple"):
            Pass4CoverageAnalysis(
                reference, candidate, [coordinates[0]], (), (), ()
            )

    def test_comparable_must_be_an_ordered_subset_of_common(self):
        coordinates = support.stages()
        with self.assertRaisesRegex(ValueError, "subset"):
            support.coverage(
                (coordinates[0],), common_comparable=(coordinates[1],)
            )

    def test_requested_list_disagreement_is_unsupported_not_sparse(self):
        coordinates = support.stages()
        other = requested_coordinates(3, 8, 12)
        reference = support.side_coverage(coordinates)
        candidate = IntraLayerSideCoverage(other, other, ())
        with self.assertRaises(UnsupportedIntraLayerLayoutError) as caught:
            Pass4CoverageAnalysis(
                reference, candidate, (), (), coordinates, other
            )
        self.assertEqual(
            caught.exception.status, "unsupported_intra_layer_layout"
        )

    def test_missing_detail_is_bounded_and_reuses_frozen_states(self):
        coordinates = support.stages()
        captured = coordinates[:16]
        oversized = MissingIntraLayerCoordinate(
            coordinates[16], CoverageState.NOT_CAPTURED, "x" * 300
        )
        side = IntraLayerSideCoverage(coordinates, captured, (oversized,))
        with self.assertRaisesRegex(ValueError, "UTF-8 bytes"):
            Pass4CoverageAnalysis(
                side, side, captured, captured, (), ()
            )
        bounded = MissingIntraLayerCoordinate(
            coordinates[16], CoverageState.UNEXPECTEDLY_ABSENT, "x" * 256
        )
        ok = IntraLayerSideCoverage(coordinates, captured, (bounded,))
        analysis = Pass4CoverageAnalysis(ok, ok, captured, captured, (), ())
        self.assertEqual(
            analysis.reference_missing[0].state,
            CoverageState.UNEXPECTEDLY_ABSENT,
        )
        self.assertEqual(MAX_DETAIL_BYTES, 256)


class TestPass4EvidenceCeiling(unittest.TestCase):
    def test_all_false_is_accepted_and_mirrors_the_frozen_nonclaims(self):
        self.assertEqual(FROZEN_EVIDENCE_CEILING.as_mapping(), NONCLAIMS)
        self.assertEqual(
            tuple(Pass4EvidenceCeiling.__dataclass_fields__),
            tuple(NONCLAIMS),
        )

    def test_any_true_or_falsey_substitute_is_rejected(self):
        for index, label in enumerate(NONCLAIMS):
            for value in (True, 0, None, ""):
                values = [False] * 8
                values[index] = value
                with self.subTest(nonclaim=label, value=value):
                    with self.assertRaisesRegex(ValueError, "exactly false"):
                        Pass4EvidenceCeiling(*values)

    def test_a_missing_nonclaim_cannot_be_defaulted(self):
        with self.assertRaises(TypeError):
            Pass4EvidenceCeiling(False, False, False, False, False, False, False)
        with self.assertRaises(TypeError):
            Pass4Result(
                Pass4Status.NOT_APPLICABLE,
                Pass4Disposition.NOT_APPLICABLE,
            )

    def test_ceiling_is_immutable(self):
        with self.assertRaises((AttributeError, TypeError)):
            FROZEN_EVIDENCE_CEILING.root_cause_identified = True


class TestPass3ParentEvidence(unittest.TestCase):
    def test_pass3a_and_pass3b_roles_are_structurally_enforced(self):
        with self.assertRaisesRegex(ValueError, "first parent"):
            support.eligible_parent(
                discovery=support.parent_binding(
                    Pass3ParentRole.AUTHORITATIVE_PASS3B
                ),
                authoritative=support.parent_binding(
                    Pass3ParentRole.DISCOVERY_PASS3A
                ),
            )
        parent = support.eligible_parent()
        self.assertEqual(
            parent.discovery.role, Pass3ParentRole.DISCOVERY_PASS3A
        )
        self.assertFalse(parent.discovery.authorizes_pass4_evidence)
        self.assertTrue(parent.authoritative.authorizes_pass4_evidence)
        self.assertTrue(parent.authorizes_pass4_evidence)

    def test_artifact_set_id_cannot_substitute_for_content_identity(self):
        with self.assertRaisesRegex(ValueError, "run_report_sha256"):
            ParentSourceIdentity(None, None, None, "aset1:" + "d" * 32)
        with self.assertRaisesRegex(ValueError, "layer_trace_sha256"):
            ParentSourceIdentity(
                support.digest("a"),
                "aset1:" + "d" * 32,
                support.digest("c"),
                "aset1:" + "d" * 32,
            )
        with self.assertRaisesRegex(ValueError, "artifact_set_id"):
            ParentSourceIdentity(
                support.digest("a"),
                support.digest("b"),
                support.digest("c"),
                support.digest("d"),
            )

    def test_eligible_parent_requires_every_verification_flag(self):
        for label in (
            "typed_artifact_coherence_verified",
            "source_binding_verified",
            "cross_generation_semantic_coherence_verified",
        ):
            with self.subTest(flag=label):
                with self.assertRaisesRegex(ValueError, "verification flag"):
                    support.eligible_parent(**{label: False})
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            support.eligible_parent(source_binding_verified=1)

    def test_eligible_parent_requires_coherent_localization(self):
        with self.assertRaisesRegex(ValueError, "first mismatch coordinate"):
            support.eligible_parent(
                parent_first_mismatch_coordinate=None,
                parent_suspect_interval=None,
            )
        with self.assertRaisesRegex(ValueError, "layer output"):
            support.eligible_parent(
                parent_first_mismatch_coordinate=CheckpointCoordinate(
                    3, 8, "mlp_down_projection", 0, 0, 0, 8
                ),
                parent_suspect_interval=None,
            )
        with self.assertRaisesRegex(ValueError, "must be the parent mismatch"):
            support.eligible_parent(
                discovery_selected_layer=9, authoritative_selected_layer=9
            )
        other_step = CheckpointCoordinate(4, 8, "layer_output", 0, 0, 0, 8)
        with self.assertRaisesRegex(ValueError, "must equal the target step"):
            support.eligible_parent(
                parent_first_mismatch_coordinate=other_step,
                parent_suspect_interval=SuspectInterval(
                    "runtime_entry",
                    None,
                    other_step,
                    False,
                    True,
                    tuple(range(0, 8)),
                    "[entry, 8]",
                ),
                target_runtime_checkpoint_step=3,
            )
        with self.assertRaisesRegex(ValueError, "does not end at"):
            support.eligible_parent(
                parent_first_mismatch_coordinate=other_step
            )
        with self.assertRaisesRegex(ValueError, "request-only"):
            support.eligible_parent(
                pass2_reproduction_evidence_tier="reproduction_request_only"
            )
        with self.assertRaisesRegex(ValueError, "tier1_bounded_digest"):
            support.eligible_parent(
                parent_evidence_level=SummaryEvidenceLevel.TIER0_STRUCTURAL
            )
        with self.assertRaisesRegex(
            ValueError, "parent_digest_contract_identity"
        ):
            support.eligible_parent(
                parent_digest_contract_identity=DIGEST_VERSION
            )

    def test_layer_drift_is_only_representable_as_revalidation_inconsistent(
        self,
    ):
        drift = support.classified_parent(
            Pass3ParentClassification.PARENT_REVALIDATION_INCONSISTENT
        )
        self.assertNotEqual(
            drift.discovery_selected_layer, drift.authoritative_selected_layer
        )
        with self.assertRaisesRegex(ValueError, "layer drift"):
            support.classified_parent(
                Pass3ParentClassification.COMPARISON_BLOCKED_BY_PASS3,
                authoritative_selected_layer=9,
            )
        with self.assertRaisesRegex(ValueError, "observed drift"):
            support.classified_parent(
                Pass3ParentClassification.PARENT_REVALIDATION_INCONSISTENT,
                authoritative_selected_layer=None,
                cross_generation_semantic_coherence_verified=True,
            )

    def test_only_an_eligible_parent_carries_localization(self):
        with self.assertRaisesRegex(ValueError, "only an eligible parent"):
            support.classified_parent(
                Pass3ParentClassification.UNSUPPORTED_PARENT,
                parent_first_mismatch_coordinate=support.parent_coordinate(),
            )
        with self.assertRaisesRegex(ValueError, "cannot select a target"):
            support.classified_parent(
                Pass3ParentClassification.NOT_APPLICABLE,
                authoritative_selected_layer=support.TARGET_LAYER,
            )

    def test_parent_evidence_is_immutable(self):
        parent = support.eligible_parent()
        self.assertIsInstance(parent, Pass3ParentEvidence)
        with self.assertRaises((AttributeError, TypeError)):
            parent.source_binding_verified = False
        with self.assertRaises(TypeError):
            Pass3ParentEvidence(
                Pass3ParentClassification.ELIGIBLE,
                support.parent_binding(Pass3ParentRole.DISCOVERY_PASS3A),
                "not-a-binding",
                True,
                True,
                True,
            )


class TestPass4ClosingBoundary(unittest.TestCase):
    def test_valid_boundary_is_inherited_pass3_evidence(self):
        closing = Pass4ClosingBoundaryDecision(support.parent_coordinate())
        self.assertEqual(closing.boundary_id, "parent:layer_output")
        self.assertEqual(closing.evidence_origin, "authoritative_pass3")
        self.assertEqual(
            closing.parent_digest_contract_identity, PASS3_DIGEST_VERSION
        )
        self.assertEqual(closing.parent_decision_field, DIGEST_DECISION_FIELD)
        self.assertNotIn("layer_output", STAGE_IDS)

    def test_boundary_cannot_become_local_evidence(self):
        with self.assertRaises(TypeError):
            Pass4ClosingBoundaryDecision(support.stages()[16])
        with self.assertRaisesRegex(ValueError, "parent layer output"):
            Pass4ClosingBoundaryDecision(
                CheckpointCoordinate(3, 8, "mlp_down_projection", 0, 0, 0, 8)
            )
        with self.assertRaisesRegex(ValueError, "evidence_origin"):
            Pass4ClosingBoundaryDecision(
                support.parent_coordinate(), evidence_origin="pass4_local"
            )
        with self.assertRaisesRegex(ValueError, "boundary_id"):
            Pass4ClosingBoundaryDecision(
                support.parent_coordinate(), boundary_id="mlp_down_projection"
            )
        with self.assertRaisesRegex(
            ValueError, "parent_digest_contract_identity"
        ):
            Pass4ClosingBoundaryDecision(
                support.parent_coordinate(),
                parent_digest_contract_identity=DIGEST_VERSION,
            )

    def test_boundary_is_never_a_coverage_member(self):
        result = support.inherited_boundary_result()
        boundary = result.closing_boundary_decision.parent_coordinate
        for group in (
            result.coverage.requested_coordinates,
            result.coverage.common_captured,
            result.coverage.common_comparable,
            tuple(item.coordinate for item in result.comparisons),
        ):
            self.assertNotIn(boundary, group)


class TestPass4ResultStatusAlgebra(unittest.TestCase):
    def test_every_status_family_constructs(self):
        for status in Pass4Status:
            with self.subTest(status=status.value):
                result = support.result_for(status)
                self.assertEqual(result.status, status)
                self.assertEqual(
                    result.disposition, STATUS_TO_DISPOSITION[status]
                )
                self.assertEqual(
                    result.evidence_ceiling.as_mapping(), NONCLAIMS
                )
                self.assertEqual(result.kind, "intra_layer_localization")
                self.assertEqual(
                    result.contract_namespace,
                    "coverage_scoped_intra_layer_localization",
                )

    def test_all_invalid_status_disposition_pairs_are_rejected(self):
        for status in Pass4Status:
            for disposition in Pass4Disposition:
                if disposition == STATUS_TO_DISPOSITION[status]:
                    continue
                with self.subTest(
                    status=status.value, disposition=disposition.value
                ):
                    with self.assertRaisesRegex(
                        ValueError, "status and disposition"
                    ):
                        support.result_for(status, disposition=disposition)

    def test_status_reason_compatibility_is_total(self):
        for status in Pass4Status:
            for reason in Pass4ReasonCode:
                allowed = (
                    status in pass4_contract.REASON_ALLOWED_STATUSES[reason]
                    and reason
                    != Pass4ReasonCode.ASYMMETRIC_COVERAGE_RETAINED
                )
                if allowed:
                    continue
                with self.subTest(status=status.value, reason=reason.value):
                    with self.assertRaises(ValueError):
                        support.result_for(status, reason_codes=(reason,))

    def test_secondary_asymmetric_reason_requires_asymmetric_coverage(self):
        with self.assertRaisesRegex(ValueError, "asymmetric"):
            support.local_mismatch_result(
                reason_codes=(
                    Pass4ReasonCode.LOCAL_DIGEST_MISMATCH,
                    Pass4ReasonCode.ASYMMETRIC_COVERAGE_RETAINED,
                )
            )
        coordinates = support.stages()
        asymmetric = support.coverage(
            (coordinates[10], coordinates[13]),
            (coordinates[10], coordinates[13], coordinates[15]),
            common_comparable=(coordinates[10], coordinates[13]),
        )
        result = support.local_mismatch_result(
            coverage=asymmetric,
            reason_codes=(
                Pass4ReasonCode.LOCAL_DIGEST_MISMATCH,
                Pass4ReasonCode.ASYMMETRIC_COVERAGE_RETAINED,
            ),
        )
        self.assertEqual(result.coverage.candidate_only, (coordinates[15],))

    def test_reason_codes_must_be_nonempty_unique_and_bounded(self):
        with self.assertRaisesRegex(ValueError, "nonempty"):
            support.result_for(Pass4Status.NOT_APPLICABLE, reason_codes=())
        with self.assertRaisesRegex(ValueError, "nonempty"):
            support.result_for(
                Pass4Status.NOT_APPLICABLE,
                reason_codes=(
                    Pass4ReasonCode.PARENT_HAS_NO_OBSERVED_MISMATCH,
                    Pass4ReasonCode.PARENT_HAS_NO_OBSERVED_MISMATCH,
                ),
            )

    def test_required_and_forbidden_fields_by_status(self):
        forbidden_probe = {
            "comparisons": (support.comparison(support.stages()[0]),),
            "local_coverage_outcome": (
                Pass4LocalCoverageOutcome.LOCAL_MISMATCH_FOUND
            ),
            "first_observed_local_mismatch_coordinate": support.stages()[0],
            "last_observed_equivalent_coordinate": support.stages()[0],
            "closing_boundary_decision": Pass4ClosingBoundaryDecision(
                support.parent_coordinate()
            ),
            "suspect_interval": support.local_interval(
                None, support.stages()[0]
            ),
            "evidence_level": EVIDENCE_LEVEL,
            "digest_contract_identity": DIGEST_VERSION,
            "coverage": support.coverage((support.stages()[0],)),
        }
        for status, policy in FIELD_POLICY.items():
            for key, rule in policy.items():
                if rule != "forbidden" or key not in forbidden_probe:
                    continue
                with self.subTest(status=status.value, field=key):
                    with self.assertRaises(ValueError):
                        support.result_for(
                            status, **{key: forbidden_probe[key]}
                        )

    def test_required_fields_cannot_be_dropped(self):
        for status, policy in FIELD_POLICY.items():
            for key in ("parent_pass3", "coverage", "suspect_interval"):
                if policy[key] != "required":
                    continue
                with self.subTest(status=status.value, field=key):
                    with self.assertRaisesRegex(ValueError, "requires"):
                        support.result_for(status, **{key: None})
        with self.assertRaisesRegex(ValueError, "both source bindings"):
            support.local_mismatch_result(candidate_binding=None)
        with self.assertRaisesRegex(ValueError, "complete target identity"):
            support.local_mismatch_result(precision_path=None)

    def test_parent_classification_must_match_the_status(self):
        with self.assertRaisesRegex(ValueError, "requires a not_applicable"):
            support.result_for(
                Pass4Status.NOT_APPLICABLE,
                parent_pass3=support.eligible_parent(),
            )
        with self.assertRaisesRegex(ValueError, "requires a eligible"):
            support.result_for(
                Pass4Status.COMPARISON_POLICY_UNAVAILABLE,
                parent_pass3=support.classified_parent(
                    Pass3ParentClassification.UNSUPPORTED_PARENT
                ),
            )

    def test_blocked_coverage_shapes_are_enforced(self):
        coordinates = support.stages()
        with self.assertRaisesRegex(ValueError, "common captured"):
            support.result_for(
                Pass4Status.INSUFFICIENT_COMMON_INTRA_LAYER_COVERAGE,
                coverage=support.coverage((coordinates[0],)),
            )
        with self.assertRaisesRegex(ValueError, "requires common captured"):
            support.result_for(
                Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT,
                coverage=support.coverage(
                    (coordinates[0],), (coordinates[1],), common_comparable=()
                ),
            )
        policy_result = support.result_for(
            Pass4Status.COMPARISON_POLICY_UNAVAILABLE
        )
        self.assertIsNotNone(policy_result.coverage)
        self.assertEqual(policy_result.comparisons, ())
        self.assertIsNone(
            policy_result.first_observed_local_mismatch_coordinate
        )
        self.assertIsNone(policy_result.last_observed_equivalent_coordinate)
        self.assertIsNone(policy_result.suspect_interval)

    def test_result_is_immutable_and_has_no_confirmation_fields(self):
        result = support.local_mismatch_result()
        with self.assertRaises((AttributeError, TypeError)):
            result.status = Pass4Status.NOT_APPLICABLE
        for prohibited in (
            "confirmed_first_divergent_stage",
            "confirmed_divergence_at_checkpoint",
            "confirmed_first_divergence",
            "root_cause",
            "numeric_divergence",
            "tensor_equality",
            "pass5_ready",
            "operation_localization",
        ):
            self.assertNotIn(prohibited, Pass4Result.__dataclass_fields__)

    def test_early_parser_failures_can_omit_unavailable_gate_evidence(self):
        absent_target = {
            field: None for field in support.target_identity()
        }
        malformed = support.result_for(
            Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
            **absent_target,
        )
        self.assertIsNone(malformed.coverage)
        self.assertIsNone(malformed.target_runtime_checkpoint_step)

        alignment = support.result_for(
            Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT,
            coverage=None,
            **absent_target,
        )
        self.assertIsNone(alignment.coverage)
        self.assertIsNone(alignment.target_runtime_checkpoint_step)

    def test_late_alignment_failure_may_retain_validated_context(self):
        result = support.result_for(
            Pass4Status.CHECKPOINT_ALIGNMENT_INCONSISTENT
        )
        self.assertIsNotNone(result.coverage)
        self.assertIsNotNone(result.target_runtime_checkpoint_step)

    def test_bounded_text_and_bool_as_int_rejection(self):
        with self.assertRaisesRegex(ValueError, "warnings"):
            support.local_mismatch_result(warnings=("x" * 300,))
        with self.assertRaisesRegex(ValueError, "entries"):
            support.local_mismatch_result(
                warnings=tuple(f"w{index}" for index in range(MAX_WARNINGS + 1))
            )
        with self.assertRaisesRegex(ValueError, "immutable tuple"):
            support.local_mismatch_result(warnings=["ok"])
        with self.assertRaisesRegex(ValueError, "inherited_pass3"):
            support.local_mismatch_result(
                inherited_pass3_reason_codes=("x" * 300,)
            )
        for label, value in (
            ("target_layer", True),
            ("target_token_position", True),
            ("layout_version", True),
            ("target_runtime_checkpoint_step", True),
        ):
            with self.subTest(field=label):
                with self.assertRaisesRegex(ValueError, "without coercion"):
                    support.local_mismatch_result(**{label: value})
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            Pass4SourceBinding("reference", support.source_identity(), 1)
        with self.assertRaisesRegex(ValueError, "UTF-8 bytes"):
            Pass4SourceBinding(
                "r" * (MAX_IDENTIFIER_BYTES + 1),
                support.source_identity(),
                True,
            )

    def test_unknown_enum_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown Pass 4 status"):
            Pass4Result(
                "not_applicable",
                Pass4Disposition.NOT_APPLICABLE,
                FROZEN_EVIDENCE_CEILING,
                (Pass4ReasonCode.PARENT_HAS_NO_OBSERVED_MISMATCH,),
                parent_pass3=support.classified_parent(
                    Pass3ParentClassification.NOT_APPLICABLE
                ),
            )
        with self.assertRaisesRegex(ValueError, "local coverage outcome"):
            support.local_mismatch_result(
                local_coverage_outcome="local_mismatch_found"
            )
        with self.assertRaises(TypeError):
            support.local_mismatch_result(coverage=object())
        with self.assertRaises(TypeError):
            support.local_mismatch_result(evidence_ceiling=None)


class TestPass4ResultLocalization(unittest.TestCase):
    def test_first_mismatch_must_be_the_earliest_and_later_ones_are_kept(self):
        coordinates = support.stages()
        captured = (coordinates[10], coordinates[13], coordinates[15])
        comparisons = (
            support.comparison(coordinates[10]),
            support.comparison(coordinates[13], equivalent=False),
            support.comparison(coordinates[15], equivalent=False),
        )
        result = support.local_mismatch_result(
            coverage=support.coverage(captured),
            comparisons=comparisons,
        )
        self.assertEqual(len(result.comparisons), 3)
        self.assertEqual(
            result.first_observed_local_mismatch_coordinate, coordinates[13]
        )
        self.assertFalse(result.comparisons[2].equivalent)
        with self.assertRaisesRegex(ValueError, "earliest"):
            support.local_mismatch_result(
                coverage=support.coverage(captured),
                comparisons=comparisons,
                first_observed_local_mismatch_coordinate=coordinates[15],
                suspect_interval=support.local_interval(
                    coordinates[13], coordinates[15], ("mlp_up_projection",)
                ),
                last_observed_equivalent_coordinate=coordinates[13],
            )

    def test_entry_interval_when_the_first_comparison_mismatches(self):
        coordinates = support.stages()
        captured = (coordinates[0], coordinates[4])
        result = support.local_mismatch_result(
            coverage=support.coverage(captured),
            comparisons=(
                support.comparison(coordinates[0], equivalent=False),
                support.comparison(coordinates[4], equivalent=False),
            ),
            first_observed_local_mismatch_coordinate=coordinates[0],
            last_observed_equivalent_coordinate=None,
            suspect_interval=support.local_interval(None, coordinates[0]),
        )
        self.assertEqual(
            result.suspect_interval.start_kind, "selected_layer_entry"
        )
        with self.assertRaisesRegex(ValueError, "entry interval"):
            support.local_mismatch_result(
                coverage=support.coverage(captured),
                comparisons=(
                    support.comparison(coordinates[0]),
                    support.comparison(coordinates[4], equivalent=False),
                ),
                first_observed_local_mismatch_coordinate=coordinates[4],
                last_observed_equivalent_coordinate=coordinates[0],
                suspect_interval=support.local_interval(
                    None, coordinates[4], ("attention_norm_output",)
                ),
            )

    def test_last_equivalent_and_interval_start_must_agree(self):
        coordinates = support.stages()
        with self.assertRaisesRegex(ValueError, "last observed equivalent"):
            support.local_mismatch_result(
                last_observed_equivalent_coordinate=coordinates[0]
            )
        with self.assertRaisesRegex(ValueError, "interval start"):
            support.local_mismatch_result(
                suspect_interval=support.local_interval(
                    coordinates[12],
                    coordinates[13],
                    (),
                )
            )

    def test_comparisons_must_equal_common_comparable_in_order(self):
        coordinates = support.stages()
        captured = (coordinates[10], coordinates[13])
        with self.assertRaisesRegex(ValueError, "common comparable"):
            support.local_mismatch_result(
                comparisons=(
                    support.comparison(coordinates[13], equivalent=False),
                    support.comparison(coordinates[10]),
                )
            )
        with self.assertRaisesRegex(ValueError, "common comparable"):
            support.local_mismatch_result(
                comparisons=(
                    support.comparison(coordinates[13], equivalent=False),
                )
            )
        outside = support.coverage(captured, common_comparable=(coordinates[10],))
        with self.assertRaisesRegex(ValueError, "common comparable"):
            support.local_mismatch_result(coverage=outside)

    def test_comparison_coordinates_must_match_the_target(self):
        other = requested_coordinates(4, 8, 11)
        with self.assertRaises(ValueError):
            support.local_mismatch_result(
                coverage=support.coverage((other[10], other[13])),
            )

    def test_comparison_bound_is_the_frozen_stage_count(self):
        self.assertEqual(MAX_COMPARISONS, 17)
        self.assertEqual(MAX_COMPARISONS, len(STAGE_IDS))
        coordinates = support.stages()
        result = support.local_mismatch_result(
            coverage=support.coverage(coordinates),
            comparisons=tuple(
                support.comparison(item, equivalent=index != 13)
                for index, item in enumerate(coordinates)
            ),
            last_observed_equivalent_coordinate=coordinates[12],
            suspect_interval=support.local_interval(
                coordinates[12], coordinates[13]
            ),
        )
        self.assertEqual(len(result.comparisons), MAX_COMPARISONS)

    def test_inherited_boundary_requires_no_local_mismatch(self):
        coordinates = support.stages()
        with self.assertRaisesRegex(ValueError, "no local mismatch"):
            support.inherited_boundary_result(
                comparisons=(
                    support.comparison(coordinates[16], equivalent=False),
                )
            )
        with self.assertRaisesRegex(ValueError, "cannot carry"):
            support.inherited_boundary_result(
                first_observed_local_mismatch_coordinate=coordinates[16]
            )
        with self.assertRaisesRegex(ValueError, "requires"):
            support.inherited_boundary_result(closing_boundary_decision=None)

    def test_inherited_interval_binds_the_exact_parent_coordinate(self):
        coordinates = support.stages()
        other_parent = CheckpointCoordinate(3, 9, "layer_output", 0, 0, 0, 9)
        with self.assertRaisesRegex(ValueError, "terminal coordinate"):
            support.inherited_boundary_result(
                closing_boundary_decision=Pass4ClosingBoundaryDecision(
                    other_parent
                )
            )
        with self.assertRaisesRegex(ValueError, "inherited interval"):
            support.inherited_boundary_result(
                suspect_interval=support.local_interval(
                    coordinates[15], coordinates[16]
                )
            )
        with self.assertRaisesRegex(ValueError, "local interval end"):
            support.local_mismatch_result(
                suspect_interval=support.inherited_interval(coordinates[10])
            )

    def test_interval_missing_stage_list_must_match_coverage(self):
        coordinates = support.stages()
        with self.assertRaisesRegex(ValueError, "missing stage list"):
            support.local_mismatch_result(
                suspect_interval=support.local_interval(
                    coordinates[10], coordinates[13], ()
                )
            )

    def test_comparison_result_requires_verified_parent_and_bindings(self):
        with self.assertRaisesRegex(ValueError, "verified source bindings"):
            support.local_mismatch_result(
                reference_binding=Pass4SourceBinding(
                    "reference_reproduction", support.source_identity(), False
                )
            )
        with self.assertRaisesRegex(ValueError, "requires a eligible"):
            support.local_mismatch_result(
                parent_pass3=support.classified_parent(
                    Pass3ParentClassification.COMPARISON_BLOCKED_BY_PASS3
                )
            )

    def test_comparison_result_requires_the_pass4_digest_identity(self):
        with self.assertRaisesRegex(ValueError, "digest_contract_identity"):
            support.local_mismatch_result(
                digest_contract_identity=PASS3_DIGEST_VERSION
            )
        with self.assertRaisesRegex(ValueError, "evidence_level"):
            support.local_mismatch_result(evidence_level="tier0_structural")
        with self.assertRaisesRegex(ValueError, "local_coverage_outcome"):
            support.local_mismatch_result(
                local_coverage_outcome=(
                    Pass4LocalCoverageOutcome
                    .NO_MISMATCH_IN_COMMON_INTRA_LAYER_COVERAGE
                )
            )

    def test_layout_and_phase_contradictions_are_rejected(self):
        for label, value in (
            ("phase", "prefill"),
            ("model_family", "qwen3_dense"),
            ("layout_name", "llama_layer_output_summary"),
            ("layout_version", 2),
            ("stage_taxonomy", "lis.llama.intra_layer_stages/v2"),
        ):
            with self.subTest(field=label):
                with self.assertRaisesRegex(ValueError, label):
                    support.local_mismatch_result(**{label: value})


class TestPass4ModelContractParity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        cls.result_model = cls.contract["result_model_contract"]
        cls.examples = json.loads(EXAMPLES.read_text(encoding="utf-8"))[
            "examples"
        ]

    def test_new_enums_are_frozen_in_the_authoritative_fixture(self):
        self.assertEqual(
            self.result_model["comparison_decision_enum"],
            [item.value for item in Pass4ComparisonDecision],
        )
        self.assertEqual(
            self.result_model["local_coverage_outcome_enum"],
            [item.value for item in Pass4LocalCoverageOutcome],
        )
        self.assertEqual(
            self.result_model["comparison_decision_semantics"],
            {
                decision.value: semantics
                for decision, semantics in (
                    pass4_model.COMPARISON_DECISION_SEMANTICS.items()
                )
            },
        )
        self.assertEqual(
            self.result_model["local_coverage_outcome_for_status"],
            {
                status.value: outcome.value
                for status, outcome in (
                    LOCAL_COVERAGE_OUTCOME_FOR_STATUS.items()
                )
            },
        )
        self.assertEqual(
            set(LOCAL_COVERAGE_OUTCOME_FOR_STATUS), COMPARISON_STATUSES
        )

    def test_fixture_bounds_and_model_roster_match_the_module(self):
        self.assertEqual(self.result_model["models"], pass4_model.__all__)
        self.assertEqual(
            self.result_model["module"], "tools/lis_verify/pass4_model.py"
        )
        self.assertEqual(
            self.result_model["max_comparisons"], MAX_COMPARISONS
        )
        self.assertEqual(
            self.result_model["max_warnings"], pass4_model.MAX_WARNINGS
        )
        self.assertEqual(
            self.result_model["max_inherited_reason_codes"],
            pass4_model.MAX_INHERITED_REASON_CODES,
        )
        self.assertEqual(
            self.result_model["max_detail_bytes"], MAX_DETAIL_BYTES
        )
        self.assertEqual(
            self.result_model["max_identifier_bytes"], MAX_IDENTIFIER_BYTES
        )
        self.assertTrue(
            self.result_model["later_comparisons_after_first_mismatch_retained"]
        )
        for prohibited in (
            "coverage_derived_from_artifacts",
            "digest_comparison_implemented",
            "localization_implemented",
            "serialization_implemented",
        ):
            self.assertFalse(self.result_model[prohibited], prohibited)

    def test_max_reason_codes_bound_is_the_frozen_p4_1_bound(self):
        self.assertEqual(self.result_model["max_reason_codes"], 32)
        reasons = tuple(Pass4ReasonCode)[:33]
        with self.assertRaisesRegex(ValueError, "bounded"):
            support.result_for(
                Pass4Status.NOT_APPLICABLE, reason_codes=reasons
            )

    def test_existing_frozen_contract_values_are_untouched(self):
        self.assertEqual(self.contract["status"], "frozen")
        self.assertEqual(self.contract["scope"], "P4-1_contract_only")
        self.assertEqual(
            self.contract["prior_contract"]["canonical_sha256"],
            "sha256:9f467c7ed31df9feea4a0757bb76faa91b11f4a17d24c89c09b60040ef8e021b",
        )
        boundary = self.contract["implementation_boundary"]
        for key in (
            "producer_implemented",
            "runtime_capture_available",
            "runtime_artifact_parser_implemented",
            "localization_algorithm_implemented",
            "localization_execution_available",
            "pass4_result_serializer_implemented",
            "production_api_added",
        ):
            self.assertFalse(boundary[key], key)
        self.assertEqual(
            self.contract["evidence_ceiling"]["nonclaims"], NONCLAIMS
        )

    def test_frozen_constants_are_shared_not_copied(self):
        for label in (
            "SCHEMA",
            "RESULT_KIND",
            "CONTRACT_VERSION",
            "CONTRACT_NAMESPACE",
            "DIGEST_VERSION",
            "PASS3_DIGEST_VERSION",
            "EVIDENCE_LEVEL",
            "INHERITED_BOUNDARY_ID",
            "INHERITED_BOUNDARY_EVIDENCE_ORIGIN",
            "LOCAL_EVIDENCE_ORIGIN",
            "INTRA_LAYER_LAYOUT_NAME",
            "STAGE_TAXONOMY",
            "MODEL_FAMILY",
            "PHASE",
            "NONCLAIMS",
            "INTRA_LAYER_STAGES",
        ):
            with self.subTest(constant=label):
                self.assertIs(
                    getattr(pass4_model, label),
                    getattr(pass4_contract, label),
                )
        self.assertIs(pass4_model.Pass4Status, pass4_contract.Pass4Status)
        self.assertIs(
            pass4_model.Pass4SuspectInterval,
            pass4_contract.Pass4SuspectInterval,
        )

    def test_model_implements_no_serialization_or_io(self):
        for prohibited in ("serialize", "to_json", "to_dict", "json", "hashlib"):
            self.assertFalse(
                hasattr(pass4_model, prohibited), prohibited
            )

    def _build_from_example(self, raw):
        status = Pass4Status(raw["status"])
        disposition = Pass4Disposition(raw["disposition"])
        reasons = tuple(Pass4ReasonCode(item) for item in raw["reason_codes"])
        common = tuple(
            coordinate_from_mapping(item)
            for item in raw["coverage"]["common_comparable"]
        )
        localization = raw["localization"]
        if localization is None:
            self.assertIsNone(raw["evidence_level"])
            return support.result_for(
                status, disposition=disposition, reason_codes=reasons
            )
        mismatch_raw = localization["first_observed_local_mismatch_coordinate"]
        mismatch = (
            coordinate_from_mapping(mismatch_raw)
            if mismatch_raw is not None
            else None
        )
        parent = parse_parent(localization["authoritative_parent_coordinate"])
        interval = parse_interval(localization["suspect_interval"])
        self.assertEqual(raw["evidence_level"], EVIDENCE_LEVEL)
        comparisons = tuple(
            support.comparison(item, equivalent=item != mismatch)
            for item in common
        )
        equivalents = [
            item.coordinate for item in comparisons if item.equivalent
        ]
        last_equivalent = equivalents[-1] if equivalents else None
        values = dict(
            parent_pass3=support.eligible_parent(
                parent_first_mismatch_coordinate=parent
            ),
            coverage=support.coverage(common),
            comparisons=comparisons,
            local_coverage_outcome=LOCAL_COVERAGE_OUTCOME_FOR_STATUS[status],
            last_observed_equivalent_coordinate=last_equivalent,
            first_observed_local_mismatch_coordinate=mismatch,
            suspect_interval=interval,
            evidence_level=raw["evidence_level"],
            digest_contract_identity=DIGEST_VERSION,
            disposition=disposition,
            reason_codes=reasons,
            **support.bindings(),
            **support.target_identity(),
        )
        if mismatch is None:
            values["closing_boundary_decision"] = Pass4ClosingBoundaryDecision(
                parent
            )
            return support.inherited_boundary_result(**values)
        return support.local_mismatch_result(**values)

    def test_every_frozen_example_is_enforced_by_the_result_model(self):
        names = {example["name"] for example in self.examples}
        self.assertEqual(
            names,
            {
                "local_mismatch_result",
                "inherited_closing_boundary_result",
                "not_applicable_result",
                "blocked_result",
                "invalid_coordinate",
                "invalid_interval",
            },
        )
        for example in self.examples:
            with self.subTest(example=example["name"]):
                raw = example["result"]
                if example["valid"]:
                    result = self._build_from_example(raw)
                    self.assertEqual(result.status.value, raw["status"])
                    self.assertEqual(
                        result.disposition.value, raw["disposition"]
                    )
                    self.assertEqual(
                        [item.value for item in result.reason_codes],
                        raw["reason_codes"],
                    )
                    self.assertEqual(
                        result.evidence_ceiling.as_mapping(), raw["nonclaims"]
                    )
                else:
                    with self.assertRaisesRegex(
                        ValueError, example["expected_error"]
                    ):
                        self._build_from_example(raw)


if __name__ == "__main__":
    unittest.main()
