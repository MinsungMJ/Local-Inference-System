#!/usr/bin/env python3
"""Contract parity tests for the additive Pass 1 namespace."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from lis_verify.pass1_model import (
    DEFAULT_EMBEDDED_PREFIX_CAP,
    KIND,
    MismatchKind,
    Pass1ReasonCode,
    Pass1Status,
    Pass2Disposition,
    PrefixAvailability,
    SelectedTokenEvidenceLevel,
)
from lis_verify.pass1_report_mapping import (
    map_pass1_reason,
    map_pass1_status,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tools" / "test_fixtures" / "differential_verification_contract.json"
MARKDOWN = ROOT / "docs" / "differential_verification.md"


def _values(enum):
    return [member.value for member in enum]


class TokenLocalizationContractTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.local = cls.contract["token_localization"]
        text = MARKDOWN.read_text(encoding="utf-8")
        match = re.search(
            r"<!-- TOKEN-LOCALIZATION-INDEX-BEGIN -->\s*```json\s*"
            r"(.*?)\s*```\s*<!-- TOKEN-LOCALIZATION-INDEX-END -->",
            text,
            flags=re.DOTALL,
        )
        if not match:
            raise AssertionError("TOKEN-LOCALIZATION-INDEX block is missing")
        cls.markdown_index = json.loads(match.group(1))


class TestTokenLocalizationMarkdownParity(TokenLocalizationContractTestCase):
    def test_markdown_index_matches_fixture_namespace(self):
        for key, value in self.markdown_index.items():
            self.assertIn(key, self.local, key)
            self.assertEqual(self.local[key], value, key)

    def test_identity_is_stable(self):
        self.assertEqual(self.local["status"], "implemented")
        self.assertEqual(self.local["schema"], "lis.execution_artifact/v1")
        self.assertEqual(self.local["kind"], KIND)
        self.assertEqual(
            self.local["contract_version"],
            "differential_verification_contract_v1",
        )
        self.assertTrue(self.local["model_free"])


class TestTokenLocalizationEnumParity(TokenLocalizationContractTestCase):
    def test_enums_match_implementation(self):
        expected = {
            "pass1_status_enum": Pass1Status,
            "mismatch_kind_enum": MismatchKind,
            "selected_token_evidence_level_enum": SelectedTokenEvidenceLevel,
            "pass2_disposition_enum": Pass2Disposition,
            "prefix_availability_enum": PrefixAvailability,
            "pass1_reason_code_enum": Pass1ReasonCode,
        }
        for key, enum in expected.items():
            self.assertEqual(self.local[key], _values(enum), key)


class TestTokenLocalizationInvariants(TokenLocalizationContractTestCase):
    def test_source_binding_is_mandatory_and_precedes_token_access(self):
        binding = self.local["source_binding"]
        self.assertTrue(binding["mandatory"])
        self.assertEqual(binding["mvp_transport"], "Pass0SourceBinding")
        self.assertEqual(binding["digest_algorithm"], "sha256")
        self.assertFalse(binding["raw_file_bytes_hashed"])
        self.assertEqual(binding["duplicate_json_keys"], "malformed")
        self.assertTrue(binding["verify_before_selected_token_access"])

    def test_step_mapping_is_exact(self):
        mapping = self.local["step_mapping"]
        self.assertEqual(mapping["generated_token_step_base"], 0)
        self.assertEqual(
            mapping["runtime_checkpoint_step_formula"],
            "generated_token_step + 1",
        )
        self.assertIsNone(mapping["equal_arrays_generated_token_step"])
        self.assertIsNone(mapping["equal_arrays_runtime_checkpoint_step"])

    def test_prefix_cap_is_64(self):
        policy = self.local["prefix_policy"]
        self.assertEqual(
            policy["default_embedded_token_cap"],
            DEFAULT_EMBEDDED_PREFIX_CAP,
        )
        self.assertEqual(
            policy["long_prefix_availability"],
            "exact_source_required",
        )
        self.assertFalse(policy["redacted_prefix_is_reproduction_material"])

    def test_first_mismatch_stays_local(self):
        boundary = self.local["report_boundary"]
        self.assertTrue(boundary["first_mismatch_is_local_evidence_only"])
        self.assertIsNone(boundary["first_mismatch_report_reason_code"])
        self.assertIn(
            "token_selection_divergence",
            boundary["prohibited_first_mismatch_reason_codes"],
        )
        self.assertIsNone(boundary["confirmed_divergence_at_checkpoint"])
        self.assertIsNone(boundary["confirmed_first_divergence"])
        self.assertIsNone(
            map_pass1_status(Pass1Status.FIRST_MISMATCH_FOUND)
        )

    def test_frozen_report_contract_is_not_extended_by_local_values(self):
        frozen_reasons = set(self.contract["reason_code_enum"])
        frozen_results = set(self.contract["result_class_enum"])
        for reason in Pass1ReasonCode:
            self.assertNotIn(reason.value, frozen_reasons)
        for status in Pass1Status:
            self.assertNotIn(status.value, frozen_results)
        self.assertFalse(
            self.local["report_boundary"][
                "frozen_verification_report_enums_modified"
            ]
        )

    def test_local_reason_mappings_target_existing_frozen_reasons(self):
        frozen_reasons = set(self.contract["reason_code_enum"])
        for reason in Pass1ReasonCode:
            mapped = map_pass1_reason(reason)
            self.assertIsNotNone(mapped, reason.value)
            self.assertIn(mapped, frozen_reasons, reason.value)


if __name__ == "__main__":
    unittest.main()
