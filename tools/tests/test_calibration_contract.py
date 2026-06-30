#!/usr/bin/env python3
"""Model-free contract tests for the Pass 0 calibration namespace (scenario 13).

Enforces that the additive ``calibration_preflight`` block in the contract
fixture stays consistent with (a) the Markdown CALIBRATION-INDEX block, (b) the
implemented ``tools/lis_verify`` package, and (c) the frozen
``verification_report`` enums it maps into. Mirrors the discipline of
``test_differential_verification_contract.py``.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from lis_verify.model import (
    CalibrationDomain,
    CalibrationReasonCode,
    ComparisonEligibility,
    ComparisonMode,
    Confidence,
    ConfigSemanticsStatus,
    KvWriteStatus,
    OracleScope,
    Pass0Verdict,
    PromptBoundary,
    PromptIdentityEvidence,
    ReasonSeverity,
    SelectionMode,
    VerdictStrengthLimit,
)
from lis_verify.reason_codes import AGGREGATOR_ESCALATED, REGISTRY
from lis_verify.report_mapping import BLOCK_REASON_TO_REPORT_REASON_CODE

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tools" / "test_fixtures" / "differential_verification_contract.json"
MARKDOWN = ROOT / "docs" / "differential_verification.md"


def _vals(enum):
    return [m.value for m in enum]


class CalibrationContractTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.cp = cls.contract["calibration_preflight"]
        text = MARKDOWN.read_text(encoding="utf-8")
        match = re.search(
            r"<!-- CALIBRATION-INDEX-BEGIN -->\s*```json\s*(.*?)\s*```\s*<!-- CALIBRATION-INDEX-END -->",
            text,
            flags=re.DOTALL,
        )
        if not match:
            raise AssertionError("Markdown CALIBRATION-INDEX block is missing")
        cls.md_index = json.loads(match.group(1))


class TestMarkdownFixtureParity(CalibrationContractTestCase):
    def test_markdown_index_matches_fixture(self):
        for key, value in self.md_index.items():
            self.assertIn(key, self.cp, key)
            self.assertEqual(self.cp[key], value, key)

    def test_identity_fields(self):
        self.assertEqual(self.cp["schema"], "lis.execution_artifact/v1")
        self.assertEqual(self.cp["kind"], "calibration_preflight")
        self.assertEqual(self.cp["contract_version"], "differential_verification_contract_v1")
        self.assertTrue(self.cp["model_free"])
        self.assertEqual(self.cp["package"], "tools/lis_verify")


class TestEnumParityWithImplementation(CalibrationContractTestCase):
    """Ground the fixture enums in the implemented package (no drift)."""

    def test_enums_match_package(self):
        pairs = {
            "comparison_eligibility_enum": ComparisonEligibility,
            "pass0_verdict_enum": Pass0Verdict,
            "verdict_strength_limit_enum": VerdictStrengthLimit,
            "selection_mode_enum": SelectionMode,
            "prompt_boundary_enum": PromptBoundary,
            "prompt_identity_evidence_enum": PromptIdentityEvidence,
            "confidence_enum": Confidence,
            "config_semantics_status_enum": ConfigSemanticsStatus,
            "kv_write_status_enum": KvWriteStatus,
            "oracle_scope_enum": OracleScope,
            "calibration_domain_enum": CalibrationDomain,
            "reason_severity_enum": ReasonSeverity,
            "calibration_reason_code_enum": CalibrationReasonCode,
        }
        for key, enum in pairs.items():
            self.assertEqual(self.cp[key], _vals(enum), key)

    def test_reason_code_registry_matches_package(self):
        fixture_reg = self.cp["reason_code_registry"]
        self.assertEqual(set(fixture_reg), {c.value for c in CalibrationReasonCode})
        for code, (domain, severity) in REGISTRY.items():
            entry = fixture_reg[code.value]
            self.assertEqual(entry["domain"], domain.value, code.value)
            self.assertEqual(entry["base_severity"], severity.value, code.value)

    def test_aggregator_escalated_matches_package(self):
        self.assertEqual(
            set(self.cp["aggregator_escalated_codes"]),
            {c.value for c in AGGREGATOR_ESCALATED},
        )

    def test_block_reason_mapping_matches_package(self):
        fixture_map = self.cp["block_reason_to_report_reason_code"]
        package_map = {k.value: v for k, v in BLOCK_REASON_TO_REPORT_REASON_CODE.items()}
        self.assertEqual(fixture_map, package_map)


class TestNamespaceConsistency(CalibrationContractTestCase):
    def test_mode_strings_are_contract_owned(self):
        # Correction 1: ComparisonMode values must be in the frozen comparison_modes.
        allowed = set(self.contract["comparison_modes"])
        for mode in ComparisonMode:
            self.assertIn(mode.value, allowed, mode.value)

    def test_aggregator_escalated_codes_have_nonblocking_base(self):
        # Correction 2: escalation lives in the aggregator, not the registry.
        reg = self.cp["reason_code_registry"]
        for code in self.cp["aggregator_escalated_codes"]:
            self.assertNotEqual(reg[code]["base_severity"], "block", code)

    def test_verdict_strength_limit_has_no_first_divergence(self):
        for value in self.cp["verdict_strength_limit_enum"]:
            self.assertNotIn("first_divergence", value)

    def test_block_reasons_target_existing_frozen_reason_codes(self):
        valid = set(self.contract["reason_code_enum"])
        for target in self.cp["block_reason_to_report_reason_code"].values():
            self.assertIn(target, valid, target)

    def test_mvp_facts(self):
        mvp = self.cp["mvp"]
        self.assertEqual(mvp["strongest_pass0_verdict"], "comparison_allowed")
        self.assertEqual(
            mvp["strongest_downstream_strength_limit"], "checkpoint_confirmation_allowed"
        )
        self.assertFalse(mvp["enables_confirmed_first_divergence"])
        self.assertTrue(mvp["external_semantic_mode_blocked_in_mvp"])
        self.assertFalse(mvp["hf_forced_token_runtime_artifact_supported"])
        self.assertTrue(mvp["hf_default_greedy_requires_array_equal"])

    def test_calibration_does_not_mutate_frozen_report_enums(self):
        # The frozen verification_report result-class set is unchanged.
        self.assertEqual(
            self.contract["result_class_enum"],
            [
                "pass",
                "documented_unsupported",
                "numeric_regression",
                "token_parity_regression",
                "benchmark_protocol_regression",
                "harness_configuration_error",
                "verification_inconclusive",
            ],
        )


if __name__ == "__main__":
    unittest.main()
