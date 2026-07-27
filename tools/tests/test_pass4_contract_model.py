#!/usr/bin/env python3
"""Immutable coordinate, coverage, interval, parent, and algebra tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from lis_verify.pass3_model import CheckpointCoordinate, CoverageState
from lis_verify.pass4_contract import (
    EVIDENCE_LEVEL,
    INTRA_LAYER_STAGES,
    MissingIntraLayerCoordinate,
    ParentSourceIdentity,
    Pass3ParentBinding,
    Pass3ParentRole,
    Pass4Disposition,
    Pass4ReasonCode,
    Pass4Status,
    Pass4SuspectInterval,
    IntraLayerCoordinate,
    IntraLayerSideCoverage,
    NONCLAIMS,
    REASON_ALLOWED_STATUSES,
    STATUS_TO_DISPOSITION,
    UnsupportedIntraLayerLayoutError,
    analyze_coverage,
    coordinate_from_mapping,
    requested_coordinates,
    validate_coordinate_sequence,
    validate_interval_against_coverage,
    validate_nonclaims,
    validate_pass3_parent_pair,
    validate_requested_coordinates,
    validate_status_algebra,
)


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = (
    ROOT
    / "tools"
    / "test_fixtures"
    / "intra_layer_localization"
    / "pass4_contract_examples.json"
)


def parent_coordinate() -> CheckpointCoordinate:
    return CheckpointCoordinate(3, 8, "layer_output", 0, 0, 0, 8)


def missing_for(
    requested: tuple[IntraLayerCoordinate, ...],
    captured: tuple[IntraLayerCoordinate, ...],
) -> tuple[MissingIntraLayerCoordinate, ...]:
    keys = {coordinate.logical_key for coordinate in captured}
    return tuple(
        MissingIntraLayerCoordinate(
            coordinate, CoverageState.NOT_CAPTURED, "fixture sparse coverage"
        )
        for coordinate in requested
        if coordinate.logical_key not in keys
    )


def side_coverage(
    requested: tuple[IntraLayerCoordinate, ...],
    captured: tuple[IntraLayerCoordinate, ...],
) -> IntraLayerSideCoverage:
    return IntraLayerSideCoverage(
        requested, captured, missing_for(requested, captured)
    )


def source_identity(seed: str = "a") -> ParentSourceIdentity:
    return ParentSourceIdentity(
        "sha256:" + seed * 64,
        "sha256:" + ("b" if seed != "b" else "c") * 64,
        "sha256:" + ("c" if seed != "c" else "d") * 64,
        "aset1:" + "d" * 32,
    )


def parent_binding(
    role: Pass3ParentRole,
    authorizes: bool,
) -> Pass3ParentBinding:
    return Pass3ParentBinding(
        role,
        "sha256:" + "1" * 64,
        "sha256:" + "2" * 64,
        source_identity("a"),
        source_identity("e"),
        authorizes,
    )


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


class TestIntraLayerCoordinate(unittest.TestCase):
    def test_exact_requested_coordinate_list(self):
        requested = requested_coordinates(3, 8, 11)
        validate_requested_coordinates(requested)
        self.assertEqual(
            [coordinate.stage_id for coordinate in requested],
            [stage.stage_id for stage in INTRA_LAYER_STAGES],
        )
        self.assertEqual(
            [coordinate.execution_ordinal for coordinate in requested],
            list(range(17)),
        )

    def test_coordinate_is_immutable(self):
        coordinate = requested_coordinates(3, 8, 11)[0]
        with self.assertRaises((AttributeError, TypeError)):
            coordinate.stage_order = 2

    def test_bool_and_implicit_integer_coercion_are_rejected(self):
        raw = {
            "runtime_checkpoint_step": 3,
            "layer_index": 8,
            "stage_id": "layer_input",
            "tensor_role": "layer_input",
            "batch_index": 0,
            "sequence_index": 0,
            "token_position": 11,
            "stage_order": 0,
            "execution_ordinal": 0,
        }
        for field, value in (
            ("runtime_checkpoint_step", True),
            ("layer_index", "8"),
            ("token_position", 3.0),
            ("stage_order", False),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                coordinate_from_mapping({**raw, field: value})

    def test_invalid_range_taxonomy_role_and_order_are_rejected(self):
        base = {
            "runtime_checkpoint_step": 3,
            "layer_index": 8,
            "stage_id": "layer_input",
            "tensor_role": "layer_input",
            "batch_index": 0,
            "sequence_index": 0,
            "token_position": 11,
            "stage_order": 0,
            "execution_ordinal": 0,
        }
        mutations = (
            {"runtime_checkpoint_step": 0},
            {"layer_index": -1},
            {"token_position": -1},
            {"batch_index": 1},
            {"sequence_index": 1},
            {"stage_id": "unknown", "tensor_role": "unknown"},
            {"tensor_role": "attention_norm_output"},
            {"stage_order": 1, "execution_ordinal": 1},
            {"execution_ordinal": 1},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                coordinate_from_mapping({**base, **mutation})

    def test_duplicate_and_out_of_order_sequences_are_rejected_without_sorting(self):
        requested = requested_coordinates(3, 8, 11)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_coordinate_sequence(
                (requested[0], requested[0]), "fixture"
            )
        reversed_pair = (requested[1], requested[0])
        with self.assertRaisesRegex(ValueError, "out of contracted order"):
            validate_coordinate_sequence(reversed_pair, "fixture")
        self.assertEqual(reversed_pair, (requested[1], requested[0]))
        with self.assertRaisesRegex(ValueError, "immutable tuple"):
            validate_coordinate_sequence(list(requested), "fixture")


class TestIntraLayerCoverage(unittest.TestCase):
    def test_partition_and_common_coverage_algebra(self):
        requested = requested_coordinates(3, 8, 11)
        reference = side_coverage(
            requested, (requested[0], requested[2], requested[4])
        )
        candidate = side_coverage(
            requested, (requested[0], requested[3], requested[4])
        )
        analysis = analyze_coverage(
            reference,
            candidate,
            common_comparable=(requested[0], requested[4]),
        )
        self.assertEqual(
            analysis.common_captured, (requested[0], requested[4])
        )
        self.assertEqual(analysis.reference_only, (requested[2],))
        self.assertEqual(analysis.candidate_only, (requested[3],))
        self.assertEqual(
            set(reference.captured_coordinates)
            | {item.coordinate for item in reference.missing_coordinates},
            set(requested),
        )
        self.assertFalse(
            set(reference.captured_coordinates)
            & {item.coordinate for item in reference.missing_coordinates}
        )

    def test_bad_partition_and_noncomparable_coordinate_are_rejected(self):
        requested = requested_coordinates(3, 8, 11)
        with self.assertRaisesRegex(ValueError, "partition"):
            IntraLayerSideCoverage(requested, (requested[0],), ())
        reference = side_coverage(requested, (requested[0],))
        candidate = side_coverage(requested, (requested[0],))
        with self.assertRaisesRegex(ValueError, "subset"):
            analyze_coverage(
                reference,
                candidate,
                common_comparable=(requested[1],),
            )

    def test_requested_stage_mismatch_is_unsupported_not_sparse(self):
        requested = requested_coordinates(3, 8, 11)
        with self.assertRaises(UnsupportedIntraLayerLayoutError) as caught:
            validate_requested_coordinates(requested[:-1])
        self.assertEqual(
            caught.exception.status, "unsupported_intra_layer_layout"
        )
        reference = side_coverage(requested, requested)
        other_requested = requested_coordinates(3, 8, 12)
        candidate = side_coverage(other_requested, other_requested)
        with self.assertRaises(UnsupportedIntraLayerLayoutError) as caught:
            analyze_coverage(reference, candidate)
        self.assertEqual(
            caught.exception.status, "unsupported_intra_layer_layout"
        )


class TestPass4SuspectInterval(unittest.TestCase):
    def test_valid_local_interval_and_missing_stage_list(self):
        requested = requested_coordinates(3, 8, 11)
        interval = Pass4SuspectInterval(
            "local_coordinate",
            requested[10],
            False,
            "local_coordinate",
            requested[13],
            None,
            "pass4_local",
            True,
            (
                "post_attention_residual",
                "mlp_norm_output",
            ),
            "(attention_output_projection, mlp_gate_projection]",
        )
        validate_interval_against_coverage(
            interval,
            (requested[10], requested[13]),
            first_local_mismatch=requested[13],
            authoritative_parent_coordinate=parent_coordinate(),
        )

    def test_valid_inherited_boundary_is_exact_pass3_coordinate(self):
        requested = requested_coordinates(3, 8, 11)
        parent = parent_coordinate()
        interval = Pass4SuspectInterval(
            "local_coordinate",
            requested[16],
            False,
            "inherited_parent_boundary",
            None,
            parent,
            "authoritative_pass3",
            True,
            (),
            "(mlp_down_projection, parent:layer_output]",
        )
        validate_interval_against_coverage(
            interval,
            (requested[16],),
            first_local_mismatch=None,
            authoritative_parent_coordinate=parent,
        )
        wrong_parent = CheckpointCoordinate(
            3, 9, "layer_output", 0, 0, 0, 9
        )
        with self.assertRaisesRegex(ValueError, "bind Pass 3B exactly"):
            validate_interval_against_coverage(
                interval,
                (requested[16],),
                first_local_mismatch=None,
                authoritative_parent_coordinate=wrong_parent,
            )

    def test_incoherent_tags_and_missing_stages_are_rejected(self):
        requested = requested_coordinates(3, 8, 11)
        with self.assertRaisesRegex(ValueError, "local interval end"):
            Pass4SuspectInterval(
                "selected_layer_entry",
                None,
                True,
                "local_coordinate",
                requested[0],
                parent_coordinate(),
                "pass4_local",
                True,
                (),
                "[selected_layer_entry, layer_input]",
            )
        interval = Pass4SuspectInterval(
            "local_coordinate",
            requested[10],
            False,
            "local_coordinate",
            requested[13],
            None,
            "pass4_local",
            True,
            (),
            "(attention_output_projection, mlp_gate_projection]",
        )
        with self.assertRaisesRegex(ValueError, "missing stage list"):
            validate_interval_against_coverage(
                interval,
                (requested[10], requested[13]),
                first_local_mismatch=requested[13],
                authoritative_parent_coordinate=parent_coordinate(),
            )


class TestPass3ParentContract(unittest.TestCase):
    def test_pass3a_is_discovery_only_and_pass3b_is_authoritative(self):
        discovery = parent_binding(Pass3ParentRole.DISCOVERY_PASS3A, False)
        authoritative = parent_binding(
            Pass3ParentRole.AUTHORITATIVE_PASS3B, True
        )
        validate_pass3_parent_pair(discovery, authoritative)
        with self.assertRaisesRegex(ValueError, "Pass 3A is discovery-only"):
            parent_binding(Pass3ParentRole.DISCOVERY_PASS3A, True)
        with self.assertRaisesRegex(ValueError, "Pass 3A is discovery-only"):
            parent_binding(Pass3ParentRole.AUTHORITATIVE_PASS3B, False)

    def test_artifact_set_id_cannot_substitute_for_content_identities(self):
        with self.assertRaisesRegex(ValueError, "run_report_sha256"):
            ParentSourceIdentity(
                None,
                None,
                None,
                "aset1:" + "d" * 32,
            )
        discovery = parent_binding(Pass3ParentRole.DISCOVERY_PASS3A, False)
        authoritative = parent_binding(
            Pass3ParentRole.AUTHORITATIVE_PASS3B, True
        )
        with self.assertRaisesRegex(ValueError, "first parent"):
            validate_pass3_parent_pair(authoritative, discovery)


class TestPass4StatusAlgebra(unittest.TestCase):
    def test_every_status_has_exactly_one_frozen_disposition(self):
        self.assertEqual(set(STATUS_TO_DISPOSITION), set(Pass4Status))
        for status, disposition in STATUS_TO_DISPOSITION.items():
            primary = next(
                reason
                for reason, allowed in REASON_ALLOWED_STATUSES.items()
                if status in allowed
                and reason
                != Pass4ReasonCode.ASYMMETRIC_COVERAGE_RETAINED
            )
            validate_status_algebra(status, disposition, (primary,))

    def test_wrong_disposition_reason_mapping_and_secondary_primary_fail(self):
        with self.assertRaisesRegex(ValueError, "status and disposition"):
            validate_status_algebra(
                Pass4Status.NOT_APPLICABLE,
                Pass4Disposition.BLOCKED,
                (Pass4ReasonCode.PARENT_HAS_NO_OBSERVED_MISMATCH,),
            )
        with self.assertRaisesRegex(ValueError, "not allowed"):
            validate_status_algebra(
                Pass4Status.NOT_APPLICABLE,
                Pass4Disposition.NOT_APPLICABLE,
                (Pass4ReasonCode.LOCAL_DIGEST_MISMATCH,),
            )
        with self.assertRaisesRegex(ValueError, "primary"):
            validate_status_algebra(
                Pass4Status.OBSERVABLE_INTRA_LAYER_MISMATCH_FOUND,
                Pass4Disposition.SUSPECT_INTERVAL_AVAILABLE,
                (
                    Pass4ReasonCode.ASYMMETRIC_COVERAGE_RETAINED,
                    Pass4ReasonCode.LOCAL_DIGEST_MISMATCH,
                ),
            )

    def test_every_nonclaim_is_mandatory_and_false(self):
        validate_nonclaims(dict(NONCLAIMS))
        broken = dict(NONCLAIMS)
        broken["root_cause_identified"] = True
        with self.assertRaisesRegex(ValueError, "present and false"):
            validate_nonclaims(broken)
        self.assertEqual(EVIDENCE_LEVEL, "tier1_bounded_digest")


class TestPass4ContractExamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = json.loads(EXAMPLES.read_text(encoding="utf-8"))[
            "examples"
        ]

    def _validate(self, example):
        result = example["result"]
        self.assertEqual(result["schema"], "lis.execution_artifact/v1")
        self.assertEqual(result["kind"], "intra_layer_localization")
        status = Pass4Status(result["status"])
        disposition = Pass4Disposition(result["disposition"])
        reasons = tuple(Pass4ReasonCode(item) for item in result["reason_codes"])
        validate_status_algebra(status, disposition, reasons)
        validate_nonclaims(result["nonclaims"])
        localization = result["localization"]
        if localization is None:
            self.assertIsNone(result["evidence_level"])
            return
        self.assertEqual(result["evidence_level"], EVIDENCE_LEVEL)
        common = tuple(
            coordinate_from_mapping(item)
            for item in result["coverage"]["common_comparable"]
        )
        mismatch_raw = localization[
            "first_observed_local_mismatch_coordinate"
        ]
        mismatch = (
            coordinate_from_mapping(mismatch_raw)
            if mismatch_raw is not None
            else None
        )
        parent = parse_parent(
            localization["authoritative_parent_coordinate"]
        )
        interval = parse_interval(localization["suspect_interval"])
        validate_interval_against_coverage(
            interval,
            common,
            first_local_mismatch=mismatch,
            authoritative_parent_coordinate=parent,
        )

    def test_required_valid_and_invalid_examples_are_enforced(self):
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
                if example["valid"]:
                    self._validate(example)
                else:
                    with self.assertRaisesRegex(
                        ValueError, example["expected_error"]
                    ):
                        self._validate(example)


if __name__ == "__main__":
    unittest.main()
