#!/usr/bin/env python3
"""Parity tests for the additive Pass 3 output contract."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from lis_verify.pass3_model import (
    AlignmentStatus,
    CoverageState,
    Pass3DownstreamDisposition,
    Pass3ReasonCode,
    Pass3Status,
    SummaryEvidenceLevel,
    SummaryFieldDisposition,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tools" / "test_fixtures" / "differential_verification_contract.json"
MARKDOWN = ROOT / "docs" / "differential_verification.md"


class CoverageScopedLocalizationContractTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.local = cls.contract["coverage_scoped_layer_localization"]
        text = MARKDOWN.read_text(encoding="utf-8")
        match = re.search(
            r"<!-- COVERAGE-SCOPED-LAYER-LOCALIZATION-INDEX-BEGIN -->\s*"
            r"```json\s*(.*?)\s*```\s*"
            r"<!-- COVERAGE-SCOPED-LAYER-LOCALIZATION-INDEX-END -->",
            text,
            flags=re.DOTALL,
        )
        if not match:
            raise AssertionError("Pass 3 Markdown contract index is missing")
        cls.markdown_index = json.loads(match.group(1))


class TestPass3ContractFreeze(CoverageScopedLocalizationContractTestCase):
    def test_markdown_namespace_is_exact(self):
        self.assertEqual(self.markdown_index, self.local)

    def test_identity_and_kind_are_additive(self):
        self.assertEqual(self.local["status"], "frozen")
        self.assertEqual(self.local["schema"], "lis.execution_artifact/v1")
        self.assertEqual(self.local["kind"], "layer_localization")
        self.assertEqual(
            self.local["contract_version"],
            "differential_verification_contract_v1",
        )

    def test_local_enums_are_exact(self):
        self.assertEqual(
            self.local["downstream_disposition_enum"],
            [
                "blocked",
                "exploratory_localization_only",
                "suspect_interval_available",
            ],
        )
        self.assertEqual(
            self.local["coverage_state_enum"],
            [
                "captured",
                "not_captured",
                "unsupported",
                "malformed",
                "unexpectedly_absent",
            ],
        )
        self.assertIn(
            "observable_mismatch_found", self.local["pass3_status_enum"]
        )
        self.assertIn(
            "pass3.pass2_object_artifact_inconsistent",
            self.local["pass3_reason_code_enum"],
        )

    def test_local_enums_match_package_model(self):
        pairs = {
            "pass3_status_enum": Pass3Status,
            "pass3_reason_code_enum": Pass3ReasonCode,
            "coverage_state_enum": CoverageState,
            "alignment_status_enum": AlignmentStatus,
            "summary_field_disposition_enum": SummaryFieldDisposition,
            "summary_evidence_level_enum": SummaryEvidenceLevel,
            "downstream_disposition_enum": Pass3DownstreamDisposition,
        }
        for key, enum in pairs.items():
            self.assertEqual(
                self.local[key], [member.value for member in enum], key
            )

    def test_pass2_identity_has_no_local_serializer(self):
        evidence = self.local["pass2_evidence"]
        self.assertEqual(
            evidence["canonical_identity_source"],
            "existing_pass2_artifact_serializer_and_canonical_json_contract",
        )
        self.assertFalse(evidence["typed_object_has_independent_identity"])
        self.assertIn(
            "pass3_local_pass2_result_serializer",
            evidence["prohibited_identity_mechanisms"],
        )

    def test_gate_order_places_source_binding_before_summaries(self):
        self.assertEqual(
            self.local["gate_order"],
            [
                "A1_typed_pass2_readiness",
                "A2_canonical_pass2_artifact_validation_hashing_and_coherence",
                "B_both_complete_source_binding_chains",
                "C_trace_headers_layout_coordinates_coverage_and_order",
                "D_bounded_checkpoint_summaries_and_digest_evidence",
            ],
        )
        self.assertFalse(
            self.local["source_binding"][
                "summary_access_before_complete_binding"
            ]
        )

    def test_full_binding_chain_is_mandatory(self):
        binding = self.local["source_binding"]
        self.assertTrue(binding["all_links_mandatory"])
        self.assertEqual(
            binding["chain"],
            self.contract["producer_checkpoint_artifact"]["source_binding"][
                "chain"
            ],
        )
        self.assertFalse(binding["matching_artifact_set_id_alone_sufficient"])
        self.assertFalse(binding["fnv1a64_satisfies_chain"])

    def test_coverage_and_interval_semantics_are_scoped(self):
        coverage = self.local["coverage_contract"]
        self.assertFalse(coverage["asymmetric_coverage_is_mismatch"])
        self.assertEqual(
            coverage["empty_common_comparable_status"],
            "insufficient_common_coverage",
        )
        localization = self.local["localization_contract"]
        self.assertEqual(localization["sparse_example"], "(4, 8]")
        self.assertEqual(localization["entry_example"], "[entry, L]")
        self.assertFalse(
            localization["earliest_observable_is_confirmed_first_divergence"]
        )

    def test_digest_is_bounded_and_no_default_tolerance_exists(self):
        policy = self.local["decision_policy"]
        self.assertEqual(policy["mvp_decision_field"], "checkpoint_digest")
        self.assertEqual(policy["evidence_level"], "tier1_bounded_digest")
        self.assertFalse(policy["digest_match_proves_tensor_equality"])
        self.assertFalse(policy["uncalibrated_default_allowed"])

    def test_report_boundary_has_no_success_mapping_or_readiness(self):
        boundary = self.local["downstream_boundary"]
        self.assertFalse(boundary["success_has_automatic_frozen_mapping"])
        self.assertFalse(boundary["pass4_or_pass5_readiness_certified"])
        self.assertFalse(boundary["frozen_verification_report_enums_modified"])
        self.assertNotIn(
            "layer_localization", self.contract["result_class_enum"]
        )
        for value in (
            "observable_mismatch_found",
            "no_mismatch_in_captured_coverage",
            "insufficient_common_coverage",
            "comparison_policy_unavailable",
        ):
            self.assertNotIn(value, self.contract["comparison_outcome_enum"])

    def test_prohibited_confirmation_claims_are_explicit(self):
        prohibited = set(self.local["prohibited_claims"])
        self.assertTrue(
            {
                "confirmed_first_divergent_layer",
                "confirmed_divergence_at_checkpoint",
                "confirmed_first_divergence",
                "pass4_ready",
                "pass5_ready",
            }.issubset(prohibited)
        )
        self.assertFalse(
            self.local["semantic_limits"]["numeric_confirmation_performed"]
        )
        self.assertFalse(
            self.local["input_contract"]["full_tensor_payload_allowed"]
        )


if __name__ == "__main__":
    unittest.main()
