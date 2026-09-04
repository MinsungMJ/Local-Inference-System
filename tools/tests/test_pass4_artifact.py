#!/usr/bin/env python3
"""P4-10 public orchestration, serializer, mapping, and export tests."""

from __future__ import annotations

import hashlib
import inspect
import json
import unittest
from pathlib import Path
from unittest import mock

import lis_verify
from lis_verify import pass4
from lis_verify import pass4_artifact
from lis_verify.pass4 import (
    localize_bound_intra_layer_inputs,
    run_coverage_scoped_intra_layer_localization,
)
from lis_verify.pass4_artifact import (
    MAX_CANONICAL_ARTIFACT_BYTES,
    serialize,
    to_json,
)
from lis_verify.pass4_contract import (
    EVIDENCE_LEVEL,
    NONCLAIMS,
    STATUS_TO_DISPOSITION,
    Pass4ReasonCode,
    Pass4Status,
)
from lis_verify.pass4_inputs import parse_pass4_intra_layer_inputs
from lis_verify.pass4_model import FROZEN_EVIDENCE_CEILING, Pass4Result
from lis_verify.pass4_report_mapping import (
    map_pass4_reason,
    map_pass4_status,
)

from . import pass4_inputs_test_support as input_support
from . import pass4_localization_test_support as localization_support
from . import pass4_parent_test_support as parent_support
from . import pass4_test_support as model_support


ROOT = Path(__file__).resolve().parents[2]
GOLDEN = (
    ROOT
    / "tools/test_fixtures/intra_layer_localization/golden"
    / "pass4_parent_blocked.json"
)

EXPECTED_TOP_LEVEL_FIELDS = {
    "schema",
    "kind",
    "contract_version",
    "contract_namespace",
    "parent_pass3",
    "source_binding",
    "target",
    "layout",
    "coverage",
    "comparisons",
    "localization",
    "status",
    "disposition",
    "reason_codes",
    "inherited_reason_codes",
    "warnings",
    "evidence_level",
    "digest_contract_identity",
    "evidence_ceiling",
    "nonclaims",
}


def recursive_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_keys(child)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Pass4ArtifactCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.seed = parent_support.two_generation_case()

    def prepared(self, *, mutate_reference=None, mutate_candidate=None):
        return input_support.prepare_inputs(
            self.seed,
            mutate_reference=mutate_reference,
            mutate_candidate=mutate_candidate,
        )

    def run_public(self, *, mutate_reference=None, mutate_candidate=None):
        prepared = self.prepared(
            mutate_reference=mutate_reference,
            mutate_candidate=mutate_candidate,
        )
        result = run_coverage_scoped_intra_layer_localization(
            **parent_support.bind_kwargs(prepared.case)
        )
        return prepared, result


class TestPass4PublicOrchestration(Pass4ArtifactCase):
    def test_public_signature_carries_every_frozen_binding_input(self):
        parameters = inspect.signature(
            run_coverage_scoped_intra_layer_localization
        ).parameters
        self.assertEqual(
            tuple(parameters),
            (
                "discovery_pass3",
                "discovery_pass3_artifact",
                "authoritative_pass3",
                "authoritative_pass3_artifact",
                "pass2_artifact",
                "discovery_reference_report",
                "discovery_candidate_report",
                "discovery_reference_trace",
                "discovery_candidate_trace",
                "authoritative_reference_report",
                "authoritative_candidate_report",
                "authoritative_reference_trace",
                "authoritative_candidate_trace",
                "discovery_pass2_artifact",
            ),
        )

    def test_public_wrapper_matches_the_composed_internal_pipeline(self):
        prepared = self.prepared(
            mutate_candidate=localization_support.mismatch_digests(13)
        )
        parsed = parse_pass4_intra_layer_inputs(
            prepared.parent,
            prepared.reference_trace,
            prepared.candidate_trace,
        )
        expected = localize_bound_intra_layer_inputs(parsed)
        actual = run_coverage_scoped_intra_layer_localization(
            **parent_support.bind_kwargs(prepared.case)
        )
        self.assertEqual(actual, expected)

    def test_public_wrapper_is_only_the_three_stage_orchestrator(self):
        parent = object()
        parsed = object()
        result = object()
        values = parent_support.bind_kwargs(self.seed)
        with (
            mock.patch.object(
                pass4, "bind_pass4_parent_inputs", return_value=parent
            ) as bind,
            mock.patch.object(
                pass4, "parse_pass4_intra_layer_inputs", return_value=parsed
            ) as parse,
            mock.patch.object(
                pass4, "localize_bound_intra_layer_inputs", return_value=result
            ) as localize,
        ):
            actual = run_coverage_scoped_intra_layer_localization(**values)
        self.assertIs(actual, result)
        bind.assert_called_once()
        parse.assert_called_once_with(
            parent,
            values["authoritative_reference_trace"],
            values["authoritative_candidate_trace"],
        )
        localize.assert_called_once_with(parsed)

    def test_wrong_public_api_type_raises_before_trace_parsing(self):
        values = parent_support.bind_kwargs(self.seed)
        values["discovery_pass3"] = object()
        with mock.patch.object(
            pass4,
            "parse_pass4_intra_layer_inputs",
            side_effect=AssertionError("parser reached after API type failure"),
        ):
            with self.assertRaisesRegex(TypeError, "discovery_pass3"):
                run_coverage_scoped_intra_layer_localization(**values)

    def test_valid_parent_terminal_never_materializes_intra_layer_entries(self):
        terminal = parent_support.two_generation_case(
            authoritative_mismatch_layer=None
        )
        result = run_coverage_scoped_intra_layer_localization(
            **parent_support.bind_kwargs(terminal)
        )
        self.assertEqual(result.status, Pass4Status.NOT_APPLICABLE)
        self.assertIsNone(result.coverage)
        self.assertEqual(result.comparisons, ())


class TestPass4ArtifactSerialization(Pass4ArtifactCase):
    def test_wrong_serializer_type_raises(self):
        with self.assertRaisesRegex(TypeError, "Pass4Result"):
            serialize(object())

    def test_required_top_level_fields_are_exact(self):
        _, result = self.run_public()
        self.assertEqual(set(serialize(result)), EXPECTED_TOP_LEVEL_FIELDS)

    def test_every_frozen_status_serializes_without_inventing_evidence(self):
        for status in Pass4Status:
            with self.subTest(status=status.value):
                result = model_support.result_for(status)
                artifact = serialize(result)
                self.assertEqual(artifact["status"], status.value)
                self.assertEqual(
                    artifact["disposition"], result.disposition.value
                )
                self.assertEqual(
                    artifact["comparisons"]["total_count"],
                    len(result.comparisons),
                )
                self.assertFalse(artifact["comparisons"]["truncated"])

    def test_frozen_valid_examples_are_exact_serializer_projections(self):
        examples = json.loads(
            (
                ROOT
                / "tools/test_fixtures/intra_layer_localization/"
                "pass4_contract_examples.json"
            ).read_text(encoding="utf-8")
        )["examples"]
        by_name = {item["name"]: item for item in examples if item["valid"]}
        results = {
            "local_mismatch_result": model_support.local_mismatch_result(),
            "inherited_closing_boundary_result": (
                model_support.inherited_boundary_result()
            ),
            "not_applicable_result": model_support.result_for(
                Pass4Status.NOT_APPLICABLE
            ),
            "blocked_result": model_support.result_for(
                Pass4Status.CHECKPOINT_SUMMARY_MALFORMED,
                reason_codes=(Pass4ReasonCode.DIGEST_FIELD_MALFORMED,),
            ),
        }
        for name, result in results.items():
            with self.subTest(name=name):
                artifact = serialize(result)
                localization = artifact["localization"]
                projection = {
                    "schema": artifact["schema"],
                    "kind": artifact["kind"],
                    "contract_version": artifact["contract_version"],
                    "contract_namespace": artifact["contract_namespace"],
                    "status": artifact["status"],
                    "disposition": artifact["disposition"],
                    "reason_codes": artifact["reason_codes"],
                    "evidence_level": artifact["evidence_level"],
                    "coverage": {
                        "common_comparable": artifact["coverage"][
                            "common_comparable"
                        ]
                    },
                    "localization": (
                        {
                            "first_observed_local_mismatch_coordinate": (
                                localization[
                                    "first_observed_local_mismatch_coordinate"
                                ]
                            ),
                            "authoritative_parent_coordinate": localization[
                                "authoritative_parent_coordinate"
                            ],
                            "suspect_interval": localization[
                                "suspect_interval"
                            ],
                        }
                        if localization is not None
                        else None
                    ),
                    "nonclaims": artifact["nonclaims"],
                }
                self.assertEqual(projection, by_name[name]["result"])

    def test_dense_comparisons_are_complete_ordered_and_not_truncated(self):
        _, result = self.run_public(
            mutate_candidate=localization_support.mismatch_digests(5, 13)
        )
        artifact = serialize(result)
        comparisons = artifact["comparisons"]
        self.assertEqual(comparisons["total_count"], 17)
        self.assertEqual(comparisons["serialized_count"], 17)
        self.assertFalse(comparisons["truncated"])
        self.assertEqual(
            tuple(
                item["coordinate"]["stage_order"]
                for item in comparisons["items"]
            ),
            tuple(range(17)),
        )
        self.assertEqual(
            artifact["localization"][
                "first_observed_local_mismatch_coordinate"
            ]["stage_order"],
            5,
        )

    def test_parent_and_source_identities_are_serialized_exactly(self):
        prepared, result = self.run_public()
        artifact = serialize(result)
        parent = artifact["parent_pass3"]
        authoritative = prepared.case["authoritative"]
        self.assertEqual(
            parent["authoritative"][
                "canonical_pass3_artifact_sha256"
            ],
            authoritative["pass3_artifact"].artifact_sha256,
        )
        self.assertEqual(
            artifact["source_binding"]["reference"][
                "layer_trace_sha256"
            ],
            authoritative["reference_trace"].identity.trace_sha256,
        )
        self.assertTrue(
            artifact["source_binding"]["reference"][
                "parent_recorded_trace_binding_verified"
            ]
        )

    def test_terminal_none_coverage_has_an_empty_wire_projection(self):
        terminal = Pass4Result(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            STATUS_TO_DISPOSITION[
                Pass4Status.COMPARISON_BLOCKED_BY_PASS3
            ],
            FROZEN_EVIDENCE_CEILING,
            (Pass4ReasonCode.PARENT_STATUS_BLOCKED,),
        )
        artifact = serialize(terminal)
        self.assertEqual(artifact["coverage"]["common_comparable"], [])
        self.assertIsNone(artifact["localization"])

    def test_minimal_parent_blocked_golden_is_exact(self):
        terminal = Pass4Result(
            Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
            STATUS_TO_DISPOSITION[
                Pass4Status.COMPARISON_BLOCKED_BY_PASS3
            ],
            FROZEN_EVIDENCE_CEILING,
            (Pass4ReasonCode.PARENT_STATUS_BLOCKED,),
        )
        self.assertEqual(
            serialize(terminal), json.loads(GOLDEN.read_text(encoding="utf-8"))
        )

    def test_json_is_deterministic_and_canonical_payload_is_bounded(self):
        _, result = self.run_public()
        artifact = serialize(result)
        self.assertEqual(serialize(result), artifact)
        self.assertEqual(to_json(result), to_json(result))
        self.assertEqual(json.loads(to_json(result)), artifact)
        canonical = json.dumps(
            artifact,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        self.assertLessEqual(len(canonical), MAX_CANONICAL_ARTIFACT_BYTES)

    def test_maximum_bounded_reason_and_warning_text_fits_the_cap(self):
        bounded = tuple(
            f"{index:02d}" + "x" * 254 for index in range(32)
        )
        result = model_support.local_mismatch_result(
            inherited_pass3_reason_codes=bounded,
            inherited_pass2_reason_codes=bounded,
            inherited_pass1_reason_codes=bounded,
            inherited_pass0_reason_codes=bounded,
            warnings=bounded,
        )
        artifact = serialize(result)
        canonical = json.dumps(
            artifact,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        self.assertLessEqual(len(canonical), MAX_CANONICAL_ARTIFACT_BYTES)

    def test_size_guard_fails_closed_instead_of_truncating(self):
        _, result = self.run_public()
        with mock.patch.object(
            pass4_artifact,
            "_canonical_size",
            return_value=MAX_CANONICAL_ARTIFACT_BYTES + 1,
        ):
            with self.assertRaisesRegex(ValueError, "256 KiB"):
                serialize(result)

    def test_no_prohibited_payload_or_confirmation_keys(self):
        _, result = self.run_public()
        keys = set(recursive_keys(serialize(result)))
        for prohibited in (
            "tensor_payload",
            "values",
            "samples",
            "prompt_text",
            "generated_text",
            "absolute_path",
            "confirmed_first_divergent_layer",
            "confirmed_divergence_at_checkpoint",
            "true_first_divergence",
            "root_cause",
            "pass5_ready",
        ):
            with self.subTest(prohibited=prohibited):
                self.assertNotIn(prohibited, keys)

    def test_nonclaims_and_inherited_reasons_are_preserved(self):
        _, result = self.run_public()
        artifact = serialize(result)
        self.assertEqual(artifact["nonclaims"], dict(NONCLAIMS))
        self.assertEqual(
            artifact["evidence_ceiling"]["evidence_level"],
            EVIDENCE_LEVEL,
        )
        self.assertEqual(
            artifact["inherited_reason_codes"]["pass3"],
            list(result.inherited_pass3_reason_codes),
        )


class TestPass4MappingBoundary(unittest.TestCase):
    def test_success_and_not_applicable_have_no_frozen_mapping(self):
        for status in (
            Pass4Status.OBSERVABLE_INTRA_LAYER_MISMATCH_FOUND,
            Pass4Status.MISMATCH_BOUNDED_TO_INHERITED_CLOSING_BOUNDARY,
            Pass4Status.NOT_APPLICABLE,
        ):
            self.assertIsNone(map_pass4_status(status))
        for reason in (
            Pass4ReasonCode.LOCAL_DIGEST_MISMATCH,
            Pass4ReasonCode.NO_LOCAL_MISMATCH_BEFORE_INHERITED_BOUNDARY,
            Pass4ReasonCode.ASYMMETRIC_COVERAGE_RETAINED,
            Pass4ReasonCode.PARENT_HAS_NO_OBSERVED_MISMATCH,
        ):
            self.assertIsNone(map_pass4_reason(reason))

    def test_defined_non_success_mappings_target_frozen_reasons(self):
        contract = json.loads(
            (
                ROOT
                / "tools/test_fixtures/differential_verification_contract.json"
            ).read_text(encoding="utf-8")
        )
        frozen = set(contract["reason_code_enum"])
        for value in Pass4Status:
            mapped = map_pass4_status(value)
            if mapped is not None:
                self.assertIn(mapped, frozen)
        for value in Pass4ReasonCode:
            mapped = map_pass4_reason(value)
            if mapped is not None:
                self.assertIn(mapped, frozen)

    def test_only_success_and_not_applicable_values_are_unmapped(self):
        self.assertEqual(
            {value for value in Pass4Status if map_pass4_status(value) is None},
            {
                Pass4Status.OBSERVABLE_INTRA_LAYER_MISMATCH_FOUND,
                Pass4Status.MISMATCH_BOUNDED_TO_INHERITED_CLOSING_BOUNDARY,
                Pass4Status.NOT_APPLICABLE,
            },
        )
        self.assertEqual(
            {
                value
                for value in Pass4ReasonCode
                if map_pass4_reason(value) is None
            },
            {
                Pass4ReasonCode.LOCAL_DIGEST_MISMATCH,
                Pass4ReasonCode.NO_LOCAL_MISMATCH_BEFORE_INHERITED_BOUNDARY,
                Pass4ReasonCode.ASYMMETRIC_COVERAGE_RETAINED,
                Pass4ReasonCode.PARENT_HAS_NO_OBSERVED_MISMATCH,
            },
        )


class TestPass4PublicExportsAndStability(unittest.TestCase):
    def test_exact_new_public_surface_is_present(self):
        expected = {
            "run_coverage_scoped_intra_layer_localization",
            "serialize_pass4",
            "pass4_to_json",
            "map_pass4_reason",
            "map_pass4_status",
            "CanonicalPass3Artifact",
            "Pass4Result",
            "Pass4Status",
            "Pass4Disposition",
            "Pass4ReasonCode",
            "Pass4ComparisonDecision",
            "Pass4LocalCoverageOutcome",
        }
        self.assertTrue(expected.issubset(set(lis_verify.__all__)))
        for name in expected:
            self.assertTrue(hasattr(lis_verify, name), name)
        self.assertIs(lis_verify.serialize_pass4, serialize)
        self.assertIs(lis_verify.pass4_to_json, to_json)
        self.assertIs(
            lis_verify.run_coverage_scoped_intra_layer_localization,
            run_coverage_scoped_intra_layer_localization,
        )
        for internal in (
            "bind_pass4_parent_inputs",
            "parse_pass4_intra_layer_inputs",
            "localize_bound_intra_layer_inputs",
        ):
            self.assertNotIn(internal, lis_verify.__all__)
            self.assertFalse(hasattr(lis_verify, internal))

    def test_frozen_contracts_and_serializer_goldens_match_current_contract(self):
        expected = {
            "tools/test_fixtures/intra_layer_localization/pass4_contract.json": (
                "fff60b99b49c5110b7438ffcc33ddf95"
                "1b67a982382ff559e7affb53a6d38b23"
            ),
            "tools/test_fixtures/intra_layer_localization/"
            "pass4_contract_examples.json": (
                "fa9f5c346cc84a9c233d04d4737a0828"
                "ed74c8c9154bd6606ef19d62b4780fd2"
            ),
            "tools/test_fixtures/differential_verification_contract.json": (
                "0d8262e76f46db4051dcf31d176e758a"
                "af388191256a6ee0cf781ab21f6678d0"
            ),
            "tools/test_fixtures/layer_localization/golden/"
            "layer_localization_entry_mismatch.json": (
                "1b3b8cc373ad3856fdb449710fdf9c48"
                "553167a7334bb4f1d5472bdd2c8067d4"
            ),
        }
        for relative, digest in expected.items():
            with self.subTest(relative=relative):
                self.assertEqual(file_sha256(ROOT / relative), digest)


if __name__ == "__main__":
    unittest.main()
