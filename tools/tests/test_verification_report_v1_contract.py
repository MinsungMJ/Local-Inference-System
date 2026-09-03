import copy
import json
from pathlib import Path
import unittest

from lis_verify.product_contract import (
    CANONICAL_STAGES,
    CustomerVerdict,
    EVIDENCE_NONCLAIMS,
    IDENTITY_FIELDS,
    MAX_INTRA_LAYER_STAGES,
    MAX_LAYER_COLLECTION,
    MAX_NEXT_ACTION_BYTES,
    MAX_REASON_CODES,
    MAX_REPORT_BYTES,
    MAX_WARNINGS,
    REPORT_TOP_LEVEL_FIELDS,
    SCHEMA,
    canonical_json_bytes,
    validate_report,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / "tools" / "test_fixtures" / "lis_verify_contract"
EXAMPLE_ROOT = FIXTURE_ROOT / "report_examples"


def load(path):
    return json.loads(path.read_text())


class ReportContractTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = {
            path.stem: load(path) for path in sorted(EXAMPLE_ROOT.glob("*.json"))
        }
        cls.schema = load(FIXTURE_ROOT / "verification_report_v1.schema.json")


class TestGoldenReports(ReportContractTestCase):
    def test_exactly_one_example_exists_for_every_customer_verdict(self):
        self.assertEqual(
            set(self.examples),
            {"pass", "regression", "inconclusive", "unsupported", "harness_error"},
        )
        self.assertEqual(
            {report["verdict"] for report in self.examples.values()},
            {value.value for value in CustomerVerdict},
        )

    def test_all_examples_validate(self):
        for name, report in self.examples.items():
            with self.subTest(name=name):
                validate_report(report)
                self.assertLessEqual(len(canonical_json_bytes(report)), MAX_REPORT_BYTES)

    def test_strict_unsupported_keeps_semantic_verdict(self):
        report = self.examples["unsupported"]
        self.assertEqual(report["verdict"], "UNSUPPORTED")
        self.assertEqual(report["policy_result"]["policy"], "require_supported")
        self.assertFalse(report["policy_result"]["satisfied"])
        self.assertEqual(report["policy_result"]["exit_code"], 6)

    def test_regression_is_bounded_and_not_numeric_confirmation(self):
        report = self.examples["regression"]
        self.assertEqual(report["token_comparison"]["status"], "mismatch")
        self.assertEqual(report["numeric_confirmation"]["status"], "not_performed")
        self.assertIsNone(
            report["numeric_confirmation"]["confirmed_first_divergence"]
        )
        self.assertTrue(
            all(value is False for value in report["evidence"]["nonclaims"].values())
        )

    def test_equal_token_early_stop_marks_deeper_stages_not_applicable(self):
        states = {
            entry["name"]: entry["state"] for entry in self.examples["pass"]["stages"]
        }
        for stage in (
            "pass2_prefix_policy_reproduction",
            "pass3a_discovery",
            "bounded_recapture",
            "pass3b_authoritative_localization",
            "pass4_intra_layer_localization",
        ):
            self.assertEqual(states[stage], "not_applicable", stage)


class TestJsonSchemaFixture(ReportContractTestCase):
    def test_schema_identity_required_fields_and_closed_world(self):
        self.assertEqual(self.schema["$id"], "urn:lis:verification-report:v1")
        self.assertEqual(self.schema["x-lis-schema"], SCHEMA)
        self.assertFalse(self.schema["additionalProperties"])
        self.assertEqual(self.schema["required"], list(REPORT_TOP_LEVEL_FIELDS))
        self.assertEqual(set(self.schema["properties"]), set(REPORT_TOP_LEVEL_FIELDS))

    def test_identity_fields_are_exact(self):
        identity = self.schema["$defs"]["sourceIdentity"]
        self.assertFalse(identity["additionalProperties"])
        self.assertEqual(identity["required"], list(IDENTITY_FIELDS))
        self.assertEqual(set(identity["properties"]), set(IDENTITY_FIELDS))

    def test_collection_bounds_are_exact(self):
        definitions = self.schema["$defs"]
        self.assertEqual(definitions["layerCollection"]["maxItems"], MAX_LAYER_COLLECTION)
        self.assertEqual(
            definitions["stageCollection"]["maxItems"], MAX_INTRA_LAYER_STAGES
        )
        self.assertEqual(self.schema["properties"]["warnings"]["maxItems"], MAX_WARNINGS)
        self.assertEqual(
            self.schema["properties"]["reason_codes"]["maxItems"],
            MAX_REASON_CODES,
        )
        self.assertEqual(
            definitions["nextAction"]["properties"]["summary"]["maxLength"],
            MAX_NEXT_ACTION_BYTES,
        )
        self.assertEqual(
            definitions["nextAction"]["properties"]["summary"][
                "x-lis-maxUtf8Bytes"
            ],
            MAX_NEXT_ACTION_BYTES,
        )

    def test_stage_schema_has_only_terminal_states(self):
        stage = self.schema["$defs"]["stageResult"]
        self.assertEqual(
            stage["properties"]["name"]["enum"], list(CANONICAL_STAGES)
        )
        self.assertEqual(
            stage["properties"]["state"]["enum"],
            ["executed", "not_applicable", "blocked", "failed"],
        )
        self.assertFalse(stage["additionalProperties"])

    def test_evidence_nonclaims_are_required_false(self):
        nonclaims = self.schema["properties"]["evidence"]["properties"][
            "nonclaims"
        ]
        self.assertEqual(nonclaims["required"], list(EVIDENCE_NONCLAIMS))
        for field in EVIDENCE_NONCLAIMS:
            self.assertIs(nonclaims["properties"][field]["const"], False)


class TestNegativeReports(ReportContractTestCase):
    def mutate(self, name="regression"):
        return copy.deepcopy(self.examples[name])

    def assert_rejected(self, report):
        with self.assertRaises((TypeError, ValueError)):
            validate_report(report)

    def test_unknown_top_level_field_is_rejected(self):
        report = self.mutate()
        report["unknown"] = True
        self.assert_rejected(report)

    def test_missing_source_authority_is_rejected(self):
        report = self.mutate()
        del report["identities"]["reference"]["source_sha256"]
        self.assert_rejected(report)

    def test_invalid_attempt_identity_is_rejected(self):
        report = self.mutate()
        report["attempt"]["id"] = "reused-or-weak"
        self.assert_rejected(report)

    def test_policy_cannot_rewrite_or_miscode_verdict(self):
        report = self.mutate("unsupported")
        report["policy_result"]["exit_code"] = 4
        self.assert_rejected(report)

    def test_pass_cannot_carry_mismatch_or_next_action(self):
        report = self.mutate("pass")
        report["token_comparison"]["status"] = "mismatch"
        report["token_comparison"]["first_mismatch"] = {
            "generated_token_step": 0,
            "reference_token_id": 1,
            "candidate_token_id": 2,
        }
        self.assert_rejected(report)

    def test_regression_requires_selected_token_mismatch(self):
        report = self.mutate()
        report["token_comparison"]["status"] = "unavailable"
        report["token_comparison"]["first_mismatch"] = None
        self.assert_rejected(report)

    def test_bounded_evidence_cannot_promote_a_nonclaim(self):
        report = self.mutate()
        report["evidence"]["nonclaims"]["numeric_divergence_confirmed"] = True
        self.assert_rejected(report)

    def test_confirmed_first_divergence_is_always_null(self):
        report = self.mutate()
        report["numeric_confirmation"]["confirmed_first_divergence"] = {}
        self.assert_rejected(report)

    def test_confirmed_checkpoint_divergence_is_not_a_core_v1_state(self):
        report = self.mutate()
        report["numeric_confirmation"]["status"] = "confirmed_at_checkpoint"
        report["numeric_confirmation"]["confirmed_divergence_at_checkpoint"] = {}
        self.assert_rejected(report)

    def test_warning_collection_bound_is_enforced(self):
        report = self.mutate()
        report["warnings"] = [f"warning-{index}" for index in range(MAX_WARNINGS + 1)]
        self.assert_rejected(report)

    def test_next_action_utf8_byte_bound_is_enforced(self):
        report = self.mutate("inconclusive")
        report["next_action"]["summary"] = "가" * (MAX_NEXT_ACTION_BYTES // 3 + 1)
        self.assert_rejected(report)

    def test_transient_stage_state_is_rejected(self):
        report = self.mutate()
        report["stages"][0]["state"] = "running"
        self.assert_rejected(report)

    def test_stage_order_is_exact(self):
        report = self.mutate()
        report["stages"][0], report["stages"][1] = (
            report["stages"][1],
            report["stages"][0],
        )
        self.assert_rejected(report)

    def test_executed_stage_cannot_depend_on_failed_stage(self):
        report = self.mutate()
        report["stages"][6] = {
            "name": "pass3a_discovery",
            "state": "failed",
            "result_ref": None,
            "evidence_tier": None,
            "failure_class": "bounded_failure",
            "reason": "bounded failure",
            "blocker": None,
        }
        self.assert_rejected(report)

    def test_failed_cleanup_cannot_claim_zero_residue(self):
        report = self.mutate()
        report["cleanup"]["status"] = "failed"
        self.assert_rejected(report)

    def test_prohibited_privacy_field_is_rejected(self):
        report = self.mutate()
        report["coverage"]["raw_tensor_values"] = [1.0]
        self.assert_rejected(report)

    def test_absolute_private_path_in_bounded_text_is_rejected(self):
        report = self.mutate()
        report["warnings"] = ["/tmp/private-model"]
        self.assert_rejected(report)


if __name__ == "__main__":
    unittest.main()
