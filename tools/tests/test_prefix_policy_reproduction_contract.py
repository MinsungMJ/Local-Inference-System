#!/usr/bin/env python3
"""Contract parity tests for the additive Pass 2 namespace."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from lis_verify.pass2_model import (
    COMPUTED_STEP_EVIDENCE,
    KIND,
    TRACE_STEP_EVIDENCE,
    THREAD_COUNT_CAVEAT,
    Pass2ReasonCode,
    Pass2Status,
    Pass3Disposition,
    ReproductionEvidenceTier,
)
from lis_verify.pass2_report_mapping import (
    map_pass2_reason,
    map_pass2_status,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    ROOT
    / "tools"
    / "test_fixtures"
    / "differential_verification_contract.json"
)
MARKDOWN = ROOT / "docs" / "differential_verification.md"
GOLDEN = (
    ROOT
    / "tools"
    / "test_fixtures"
    / "prefix_policy_reproduction"
    / "golden"
    / "prefix_policy_reproduction_verified.json"
)


def _values(enum):
    return [member.value for member in enum]


class PrefixPolicyContractTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.local = cls.contract["prefix_policy_reproduction"]
        text = MARKDOWN.read_text(encoding="utf-8")
        match = re.search(
            r"<!-- PREFIX-POLICY-REPRODUCTION-INDEX-BEGIN -->\s*"
            r"```json\s*(.*?)\s*```\s*"
            r"<!-- PREFIX-POLICY-REPRODUCTION-INDEX-END -->",
            text,
            flags=re.DOTALL,
        )
        if not match:
            raise AssertionError(
                "PREFIX-POLICY-REPRODUCTION-INDEX block is missing"
            )
        cls.markdown_index = json.loads(match.group(1))


class TestPrefixPolicyMarkdownParity(PrefixPolicyContractTestCase):
    def test_markdown_index_matches_fixture_namespace(self):
        self.assertEqual(self.markdown_index, self.local)

    def test_identity_is_stable(self):
        self.assertEqual(self.local["status"], "implemented")
        self.assertEqual(self.local["schema"], "lis.execution_artifact/v1")
        self.assertEqual(self.local["kind"], KIND)
        self.assertEqual(
            self.local["contract_version"],
            "differential_verification_contract_v1",
        )
        self.assertTrue(self.local["model_free"])


class TestPrefixPolicyEnumParity(PrefixPolicyContractTestCase):
    def test_enums_match_implementation(self):
        expected = {
            "pass2_status_enum": Pass2Status,
            "pass2_reason_code_enum": Pass2ReasonCode,
            "reproduction_evidence_tier_enum": ReproductionEvidenceTier,
            "pass3_disposition_enum": Pass3Disposition,
        }
        for key, enum in expected.items():
            self.assertEqual(self.local[key], _values(enum), key)

    def test_checkpoint_evidence_values_match(self):
        self.assertEqual(
            self.local["checkpoint_step_evidence_enum"],
            [COMPUTED_STEP_EVIDENCE, TRACE_STEP_EVIDENCE],
        )

    def test_only_amended_tier_names_exist(self):
        tiers = set(self.local["reproduction_evidence_tier_enum"])
        self.assertNotIn("original_pair_self_consistent", tiers)
        self.assertNotIn("plan_only_not_verified", tiers)


class TestPrefixPolicyInvariants(PrefixPolicyContractTestCase):
    def test_required_originals_precede_metadata_access(self):
        binding = self.local["source_binding"]
        self.assertEqual(
            binding["required_original_inputs"],
            ["reference_original", "candidate_original"],
        )
        self.assertTrue(
            binding["verify_both_originals_before_materialization"]
        )
        self.assertIn(
            "selected_token_ids",
            binding["verify_before_metadata_access"],
        )
        self.assertTrue(
            binding["missing_or_malformed_metadata_fails_closed"]
        )

    def test_computed_step_does_not_imply_checkpoint_artifact(self):
        evidence = self.local["checkpoint_step_evidence"]
        self.assertEqual(evidence["default"], COMPUTED_STEP_EVIDENCE)
        self.assertFalse(evidence["computed_implies_materialized_checkpoint"])
        self.assertFalse(evidence["numeric_trace_values_accessed"])

    def test_thread_caveat_is_visible_and_nonblocking(self):
        caveats = self.local["determinism_caveats"]
        self.assertEqual(
            caveats["thread_count_gt_1_warning"], THREAD_COUNT_CAVEAT
        )
        self.assertTrue(caveats["applies_to_original_or_reproduction"])
        self.assertFalse(caveats["pass2_blocking"])

    def test_downstream_readiness_preserves_tier_awareness(self):
        readiness = self.local["downstream_readiness"]
        self.assertEqual(readiness["primary_gate"], "pass3_disposition")
        self.assertEqual(
            readiness["stronger_claims_also_check"],
            "reproduction_evidence_tier",
        )
        self.assertFalse(readiness["ready_implies_independent_rerun"])

    def test_request_only_cannot_be_verified(self):
        boundary = self.local["report_boundary"]
        self.assertFalse(
            boundary["reproduction_request_only_may_be_verified"]
        )

    def test_confirmation_fields_are_prohibited(self):
        prohibited = self.local["report_boundary"][
            "prohibited_status_fields"
        ]
        self.assertIn("confirmed_divergence_at_checkpoint", prohibited)
        self.assertIn("confirmed_first_divergence", prohibited)

    def test_artifact_required_fields_match_golden(self):
        golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
        self.assertEqual(
            set(self.local["artifact_required_fields"]), set(golden)
        )

    def test_frozen_report_contract_is_not_extended(self):
        frozen_reasons = set(self.contract["reason_code_enum"])
        frozen_results = set(self.contract["result_class_enum"])
        for reason in Pass2ReasonCode:
            self.assertNotIn(reason.value, frozen_reasons)
        for status in Pass2Status:
            self.assertNotIn(status.value, frozen_results)
        self.assertFalse(
            self.local["report_boundary"][
                "frozen_verification_report_enums_modified"
            ]
        )

    def test_local_reason_mappings_target_frozen_reasons(self):
        frozen_reasons = set(self.contract["reason_code_enum"])
        for reason in Pass2ReasonCode:
            mapped = map_pass2_reason(reason)
            self.assertIsNotNone(mapped, reason.value)
            self.assertIn(mapped, frozen_reasons, reason.value)

    def test_reproduction_verified_remains_local(self):
        self.assertTrue(
            self.local["report_boundary"][
                "reproduction_verified_is_local_evidence_only"
            ]
        )
        self.assertIsNone(
            map_pass2_status(Pass2Status.REPRODUCTION_VERIFIED)
        )


if __name__ == "__main__":
    unittest.main()
