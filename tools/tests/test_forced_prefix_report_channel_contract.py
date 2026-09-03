import copy
import json
from pathlib import Path
import unittest

from lis_verify.product_contract import (
    FORCED_PREFIX_FIELDS,
    MAX_TOKEN_ID_PREVIEW,
    SELECTION_POLICY_PROFILES,
    selection_policy_sha256,
    validate_forced_prefix_metadata,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    ROOT
    / "tools"
    / "test_fixtures"
    / "lis_verify_contract"
    / "forced_prefix_report_channel_v1.json"
)


class TestForcedPrefixContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = json.loads(FIXTURE.read_text())
        cls.example = cls.contract["valid_example"]

    def changed(self, **updates):
        value = copy.deepcopy(self.example)
        value.update(updates)
        return value

    def assert_rejected(self, value):
        with self.assertRaises((TypeError, ValueError)):
            validate_forced_prefix_metadata(value)

    def test_design_is_frozen_but_implementation_remains_m3(self):
        self.assertEqual(self.contract["contract_status"], "design_frozen")
        self.assertEqual(self.contract["implementation_status"], "pending_m3")
        self.assertEqual(self.contract["run_report_schema"], "lis.execution_artifact/v1")

    def test_required_fields_match_contract_module(self):
        self.assertEqual(self.contract["required_fields"], list(FORCED_PREFIX_FIELDS))
        self.assertEqual(set(self.example), set(FORCED_PREFIX_FIELDS))
        self.assertEqual(self.contract["max_token_count"], MAX_TOKEN_ID_PREVIEW)

    def test_valid_nonempty_prefix_is_source_bound(self):
        validate_forced_prefix_metadata(self.example)
        self.assertEqual(self.example["target_generated_token_step"], 2)
        self.assertEqual(self.example["runtime_checkpoint_step"], 3)
        self.assertEqual(self.example["context_position"], 6)
        self.assertEqual(
            self.example["selection_policy"], "lis_policy_modified_greedy_v1"
        )
        self.assertEqual(
            self.contract["selection_policy_enum"],
            ["raw_greedy", "lis_policy_modified_greedy_v1"],
        )
        identity = self.contract["selection_policy_identity"]
        self.assertEqual(identity["profiles"], SELECTION_POLICY_PROFILES)
        self.assertEqual(
            self.example["selection_policy_sha256"],
            selection_policy_sha256(self.example["selection_policy"]),
        )

    def test_raw_material_is_not_retained(self):
        self.assertFalse(self.contract["raw_token_ids_retained_by_default"])
        self.assertFalse(self.contract["raw_prompt_text_retained"])
        self.assertFalse(self.contract["raw_generated_text_retained"])
        self.assertNotIn("token_ids", self.example)
        self.assertFalse(
            self.contract["artifact_set_id_substitutes_for_content_identity"]
        )

    def test_empty_prefix_is_rejected(self):
        self.assert_rejected(
            self.changed(
                token_count=0,
                prefix_end_generated_step_exclusive=0,
                target_generated_token_step=0,
                runtime_checkpoint_step=1,
                context_position=4,
            )
        )

    def test_prefix_count_bound_is_enforced(self):
        count = MAX_TOKEN_ID_PREVIEW + 1
        self.assert_rejected(
            self.changed(
                token_count=count,
                prefix_end_generated_step_exclusive=count,
                target_generated_token_step=count,
                runtime_checkpoint_step=count + 1,
                context_position=4 + count,
            )
        )

    def test_missing_source_sha_is_rejected(self):
        value = self.changed()
        del value["source_original_run_report_sha256"]
        self.assert_rejected(value)

    def test_malformed_or_weak_digest_is_rejected(self):
        self.assert_rejected(self.changed(token_ids_sha256="fnv1a64:1234"))

    def test_generated_step_range_mismatch_is_rejected(self):
        self.assert_rejected(self.changed(prefix_end_generated_step_exclusive=1))
        self.assert_rejected(self.changed(target_generated_token_step=1))

    def test_runtime_step_must_equal_generated_step_plus_one(self):
        self.assert_rejected(self.changed(runtime_checkpoint_step=2))

    def test_context_position_must_include_prompt_and_prefix(self):
        self.assert_rejected(self.changed(context_position=5))

    def test_policy_mismatch_is_rejected(self):
        self.assert_rejected(self.changed(selection_policy="sampling"))

    def test_request_only_and_prefill_substitution_are_prohibited(self):
        rejection = set(self.contract["rejection_classes"])
        self.assertIn("request_only_evidence", rejection)
        self.assertIn("prefill_substituted_for_decode_boundary", rejection)

    def test_current_c_cli_still_rejects_the_unimplemented_combination(self):
        source = (ROOT / "srcs" / "cli" / "driver.c").read_text()
        self.assertIn("--report-json does not support ", source)
        self.assertIn("--forced-prefix", source)

    def test_pass0_still_reports_artifact_channel_as_unimplemented(self):
        base = json.loads(
            (
                ROOT
                / "tools"
                / "test_fixtures"
                / "differential_verification_contract.json"
            ).read_text()
        )
        self.assertFalse(
            base["calibration_preflight"]["mvp"][
                "hf_forced_token_runtime_artifact_supported"
            ]
        )


if __name__ == "__main__":
    unittest.main()
