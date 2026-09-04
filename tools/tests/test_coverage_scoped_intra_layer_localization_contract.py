#!/usr/bin/env python3
"""Parity tests for the P4-1 intra-layer localization contract freeze."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from lis_verify.pass1_inputs import canonical_json_sha256
from lis_verify.pass3_model import CoverageState
from lis_verify.pass4_contract import (
    CONTRACT_NAMESPACE,
    CONTRACT_VERSION,
    DIAGNOSTIC_CAPTURE_PROFILE,
    DIGEST_VERSION,
    INHERITED_BOUNDARY_ID,
    INTRA_LAYER_LAYOUT_NAME,
    INTRA_LAYER_LAYOUT_VERSION,
    INTRA_LAYER_STAGES,
    NONCLAIMS,
    OUTER_TRACE_KIND,
    PASS3_DIGEST_VERSION,
    PHASE,
    REASON_ALLOWED_STATUSES,
    RESULT_KIND,
    SCHEMA,
    STAGE_IDS,
    STAGE_TAXONOMY,
    STATUS_TO_DISPOSITION,
    Pass4Disposition,
    Pass4ReasonCode,
    Pass4Status,
    Pass3ParentClassification,
    Pass3ParentRole,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    ROOT
    / "tools"
    / "test_fixtures"
    / "intra_layer_localization"
    / "pass4_contract.json"
)
PRIOR_FIXTURE = (
    ROOT / "tools" / "test_fixtures" / "differential_verification_contract.json"
)
MARKDOWN = ROOT / "docs" / "differential_verification.md"


class Pass4ContractFixtureTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
        text = MARKDOWN.read_text(encoding="utf-8")
        match = re.search(
            r"<!-- COVERAGE-SCOPED-INTRA-LAYER-LOCALIZATION-INDEX-BEGIN -->\s*"
            r"```json\s*(.*?)\s*```\s*"
            r"<!-- COVERAGE-SCOPED-INTRA-LAYER-LOCALIZATION-INDEX-END -->",
            text,
            flags=re.DOTALL,
        )
        if not match:
            raise AssertionError("Pass 4 Markdown contract index is missing")
        cls.markdown_index = json.loads(match.group(1))


class TestPass4ContractIdentity(Pass4ContractFixtureTestCase):
    def test_contract_and_markdown_index_are_frozen(self):
        self.assertEqual(self.contract["status"], "frozen")
        self.assertEqual(self.contract["scope"], "P4-1_contract_only")
        self.assertEqual(
            self.markdown_index, self.contract["documentation_index"]
        )

    def test_artifact_and_layout_identities_match_model(self):
        self.assertEqual(self.contract["schema"], SCHEMA)
        self.assertEqual(self.contract["contract_version"], CONTRACT_VERSION)
        self.assertEqual(
            self.contract["contract_namespace"], CONTRACT_NAMESPACE
        )
        artifacts = self.contract["artifact_identities"]
        self.assertEqual(artifacts["outer_trace"]["kind"], OUTER_TRACE_KIND)
        self.assertEqual(artifacts["result"]["kind"], RESULT_KIND)
        self.assertEqual(
            artifacts["conditional_manifest_fields"][
                "diagnostic_capture_profile"
            ],
            DIAGNOSTIC_CAPTURE_PROFILE,
        )
        layout = self.contract["layout"]
        self.assertEqual(layout["layout_name"], INTRA_LAYER_LAYOUT_NAME)
        self.assertEqual(layout["layout_version"], INTRA_LAYER_LAYOUT_VERSION)
        self.assertEqual(layout["stage_taxonomy"], STAGE_TAXONOMY)
        self.assertEqual(layout["phase"], PHASE)

    def test_exact_17_stage_taxonomy_and_order(self):
        expected = [
            {
                "order": stage.stage_order,
                "stage_id": stage.stage_id,
                "tensor_role": stage.tensor_role,
            }
            for stage in INTRA_LAYER_STAGES
        ]
        self.assertEqual(self.contract["stage_taxonomy"], expected)
        self.assertEqual(len(STAGE_IDS), 17)
        self.assertEqual(
            [stage.stage_order for stage in INTRA_LAYER_STAGES],
            list(range(17)),
        )
        self.assertTrue(
            all(stage.stage_id == stage.tensor_role for stage in INTRA_LAYER_STAGES)
        )

    def test_layer_output_is_only_the_inherited_parent_boundary(self):
        self.assertNotIn("layer_output", STAGE_IDS)
        boundary = self.contract["inherited_boundary"]
        self.assertEqual(boundary["boundary_id"], INHERITED_BOUNDARY_ID)
        self.assertEqual(boundary["evidence_origin"], "authoritative_pass3")
        self.assertFalse(boundary["local_stage"])
        for key in (
            "part_of_requested_coverage",
            "part_of_captured_coverage",
            "part_of_missing_coverage",
            "part_of_common_coverage",
        ):
            self.assertFalse(boundary[key], key)

    def test_implementation_boundary_does_not_advertise_later_work(self):
        boundary = self.contract["implementation_boundary"]
        self.assertTrue(boundary["contract_frozen"])
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


class TestPass4StatusContract(Pass4ContractFixtureTestCase):
    def test_status_disposition_and_reason_tables_match_model(self):
        algebra = self.contract["status_algebra"]
        self.assertEqual(
            algebra["statuses"], [status.value for status in Pass4Status]
        )
        self.assertEqual(
            algebra["dispositions"],
            [disposition.value for disposition in Pass4Disposition],
        )
        self.assertEqual(
            algebra["status_to_disposition"],
            {
                status.value: disposition.value
                for status, disposition in STATUS_TO_DISPOSITION.items()
            },
        )
        self.assertEqual(
            algebra["reason_allowed_statuses"],
            {
                reason.value: [status.value for status in statuses]
                for reason, statuses in REASON_ALLOWED_STATUSES.items()
            },
        )
        self.assertEqual(
            set(algebra["reason_allowed_statuses"]),
            {reason.value for reason in Pass4ReasonCode},
        )
        self.assertEqual(
            algebra["secondary_only_reasons"],
            [Pass4ReasonCode.ASYMMETRIC_COVERAGE_RETAINED.value],
        )
        self.assertEqual(
            algebra["failure_precedence"],
            [
                "wrong_api_type_raises_type_error",
                "parent_canonical_and_typed_artifact_coherence",
                "fully_validated_parent_terminal_classification",
                "discovery_rebound_semantic_or_layer_drift",
                "source_binding",
                "support_and_layout",
                "summary_and_coverage_structure",
                "common_coverage_sufficiency",
                "shared_entry_alignment",
                "digest_policy_availability",
                "digest_decisions",
            ],
        )

    def test_success_has_no_frozen_global_success_mapping(self):
        self.assertFalse(
            self.contract["status_algebra"][
                "successful_results_map_to_frozen_global_success_enums"
            ]
        )
        prior = json.loads(PRIOR_FIXTURE.read_text(encoding="utf-8"))
        for value in (
            "observable_intra_layer_mismatch_found",
            "mismatch_bounded_to_inherited_closing_boundary",
        ):
            self.assertNotIn(value, prior["comparison_outcome_enum"])
            self.assertNotIn(value, prior["reason_code_enum"])

    def test_evidence_ceiling_and_nonclaims_are_exact(self):
        ceiling = self.contract["evidence_ceiling"]
        self.assertEqual(ceiling["evidence_level"], "tier1_bounded_digest")
        self.assertEqual(ceiling["nonclaims"], NONCLAIMS)
        self.assertTrue(all(value is False for value in NONCLAIMS.values()))


class TestPass4ParentContract(Pass4ContractFixtureTestCase):
    def test_pass3a_and_pass3b_roles_are_exact(self):
        roles = self.contract["pass3_parent_contract"]["roles"]
        self.assertEqual(
            roles["pass3a"]["role"],
            Pass3ParentRole.DISCOVERY_PASS3A.value,
        )
        self.assertFalse(roles["pass3a"]["authorizes_pass4_evidence"])
        self.assertEqual(
            roles["pass3b"]["role"],
            Pass3ParentRole.AUTHORITATIVE_PASS3B.value,
        )
        self.assertTrue(roles["pass3b"]["authorizes_pass4_evidence"])

    def test_parent_classifications_are_total_for_frozen_cases(self):
        classifications = self.contract["pass3_parent_contract"][
            "classifications"
        ]
        self.assertEqual(
            set(classifications.values()),
            {classification.value for classification in Pass3ParentClassification},
        )
        self.assertEqual(
            classifications,
            {
                "valid_observed_mismatch": "eligible",
                "valid_no_mismatch": "not_applicable",
                "valid_blocked_or_malformed": "comparison_blocked_by_pass3",
                "unsupported_family_layout_or_evidence_policy":
                    "unsupported_parent",
                "pass3a_pass3b_target_or_semantic_drift":
                    "parent_revalidation_inconsistent",
            },
        )

    def test_canonical_parent_identity_uses_only_existing_helpers(self):
        wrapper = self.contract["pass3_parent_contract"][
            "canonical_wrapper"
        ]
        self.assertEqual(wrapper["serializer"], "existing_pass3_serializer")
        self.assertEqual(
            wrapper["canonical_json"], "existing_canonical_json_helper"
        )
        self.assertEqual(wrapper["hash"], "existing_sha256_helper")
        self.assertEqual(
            wrapper["json_parser"], "strict_duplicate_key_rejecting"
        )
        for prohibited in (
            "rerun_pass3_while_parsing",
            "repr_hashing",
            "pickle_hashing",
            "selected_field_identity_hashing",
            "separate_pass4_serializer_for_pass3",
        ):
            self.assertFalse(wrapper[prohibited], prohibited)

    def test_artifact_set_id_is_association_only(self):
        parent = self.contract["pass3_parent_contract"]
        self.assertEqual(
            parent["artifact_set_id_role"],
            "probabilistic_same_execution_association_evidence_only",
        )
        self.assertFalse(
            parent["artifact_set_id_substitutes_for_other_identity"]
        )
        self.assertEqual(
            set(parent["required_identity_evidence"]),
            {
                "canonical_pass3_artifact_sha256",
                "pass2_artifact_sha256",
                "run_report_sha256",
                "layer_trace_sha256",
                "semantic_manifest_sha256",
                "artifact_set_id",
            },
        )


class TestPass4BackwardCompatibility(Pass4ContractFixtureTestCase):
    def test_prior_contract_records_the_approved_m3_transition(self):
        expected = self.contract["prior_contract"]["canonical_sha256"]
        actual = canonical_json_sha256(
            json.loads(PRIOR_FIXTURE.read_text(encoding="utf-8"))
        )
        self.assertEqual(
            actual,
            "sha256:d416381eec06c011426ad1c840ec044334c0ffec73bc62e0f0ece3737df15cf2",
        )
        self.assertEqual(actual, expected)
        self.assertFalse(
            self.contract["prior_contract"][
                "all_pass0_through_pass3_serialized_values_unchanged"
            ]
        )

    def test_existing_coverage_state_values_are_reused_byte_for_value(self):
        self.assertEqual(
            self.contract["coverage_contract"]["coverage_state_enum"],
            [state.value for state in CoverageState],
        )

    def test_pass3_digest_contract_is_unchanged(self):
        self.assertEqual(PASS3_DIGEST_VERSION, "lis.checkpoint.fp32le/v1")
        self.assertEqual(
            self.contract["digest_contract"]["pass3_digest_version_unchanged"],
            PASS3_DIGEST_VERSION,
        )
        self.assertEqual(
            self.contract["digest_contract"]["version"], DIGEST_VERSION
        )
        self.assertNotEqual(DIGEST_VERSION, PASS3_DIGEST_VERSION)


if __name__ == "__main__":
    unittest.main()
