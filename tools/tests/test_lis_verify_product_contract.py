import hashlib
import json
from pathlib import Path
import unittest

from lis_verify.model import CalibrationReasonCode
from lis_verify.product_contract import (
    AggregationAction,
    CANONICAL_STAGES,
    CLEANUP_RESIDUE_PAIRS,
    CleanupStatus,
    CLI_DEFAULTS,
    CLI_COMMON_OPTIONS,
    CLI_MODES,
    CONTRACT_VERSION,
    CustomerVerdict,
    DEFAULT_STAGE_TIMEOUT_SECONDS,
    DEFAULT_EXIT_CODES,
    EVIDENCE_NONCLAIMS,
    ExecutionPolicy,
    IDENTITY_FIELDS,
    KIND,
    LedgerEvent,
    LEDGER_PAYLOAD_FIELDS,
    MAX_DETAIL_BYTES,
    MAX_IN_MEMORY_ARTIFACT_BYTES,
    MAX_IDENTIFIER_BYTES,
    MAX_INTRA_LAYER_STAGES,
    MAX_LAYER_COLLECTION,
    MAX_LEDGER_EVENT_BYTES,
    MAX_NEXT_ACTION_BYTES,
    MAX_REASON_CODES,
    MAX_REPORT_BYTES,
    MAX_STAGE_TIMEOUT_SECONDS,
    MAX_SUBPROCESS_OUTPUT_BYTES,
    MAX_SUMMARY_BYTES,
    MAX_TEMP_DISK_BYTES,
    MAX_TOKEN_ID_PREVIEW,
    MAX_WARNINGS,
    PASS0_BLOCK_REASON_VERDICTS,
    PASS3_ROLE_TRANSITIONS,
    REPORT_TOP_LEVEL_FIELDS,
    REPORT_VERSION,
    REQUIRE_SUPPORTED_EXIT_CODES,
    ResidueStatus,
    SCHEMA,
    SIGNAL_EXIT_CODES,
    STAGE_DEPENDENCIES,
    StageState,
    TERMINATION_GRACE_SECONDS,
    WorkflowClassification,
    canonical_json_bytes,
    expected_exit_code,
    runtime_status_values,
    validate_status_mapping,
    validate_ledger_events,
)
from lis_verify.report_mapping import BLOCK_REASON_TO_REPORT_REASON_CODE


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = ROOT / "tools" / "test_fixtures" / "lis_verify_contract"
LEGACY_FIXTURE = (
    ROOT / "tools" / "test_fixtures" / "differential_verification_contract.json"
)
LEGACY_SHA256 = "0d8262e76f46db4051dcf31d176e758aaf388191256a6ee0cf781ab21f6678d0"


def load(name):
    return json.loads((FIXTURE_ROOT / name).read_text())


def reject_duplicate_keys(pairs):
    value = {}
    for key, child in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


class TestProductContractIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = load("product_contract_v1.json")

    def test_identity_and_authority(self):
        self.assertEqual(self.contract["contract_status"], "approved")
        self.assertEqual(self.contract["implementation_status"], "contract_only")
        self.assertEqual(self.contract["contract_version"], CONTRACT_VERSION)
        self.assertEqual(self.contract["schema"], SCHEMA)
        self.assertEqual(self.contract["kind"], KIND)
        self.assertEqual(self.contract["report_version"], REPORT_VERSION)
        authority = self.contract["authority"]
        self.assertFalse(authority["silent_override_allowed"])
        self.assertEqual(
            authority["disagreement_policy"], "contract_validation_failure"
        )

    def test_enums_match_contract_module(self):
        pairs = (
            ("customer_verdicts", CustomerVerdict),
            ("execution_policies", ExecutionPolicy),
            ("workflow_classifications", WorkflowClassification),
            ("stage_states", StageState),
            ("cleanup_statuses", CleanupStatus),
            ("residue_statuses", ResidueStatus),
            ("aggregation_actions", AggregationAction),
        )
        for key, enum_type in pairs:
            self.assertEqual(
                self.contract[key], [value.value for value in enum_type], key
            )

    def test_report_fields_identities_and_nonclaims_match(self):
        self.assertEqual(
            self.contract["report_top_level_fields"], list(REPORT_TOP_LEVEL_FIELDS)
        )
        self.assertEqual(self.contract["identity_fields"], list(IDENTITY_FIELDS))
        self.assertEqual(
            self.contract["evidence_nonclaims"], list(EVIDENCE_NONCLAIMS)
        )

    def test_bounds_match_contract_module(self):
        self.assertEqual(
            self.contract["bounds"],
            {
                "max_report_bytes": MAX_REPORT_BYTES,
                "max_summary_bytes": MAX_SUMMARY_BYTES,
                "max_identifier_bytes": MAX_IDENTIFIER_BYTES,
                "max_detail_bytes": MAX_DETAIL_BYTES,
                "max_next_action_bytes": MAX_NEXT_ACTION_BYTES,
                "max_warnings": MAX_WARNINGS,
                "max_reason_codes": MAX_REASON_CODES,
                "max_token_id_preview": MAX_TOKEN_ID_PREVIEW,
                "max_layer_collection": MAX_LAYER_COLLECTION,
                "max_intra_layer_stages": MAX_INTRA_LAYER_STAGES,
                "max_ledger_event_bytes": MAX_LEDGER_EVENT_BYTES,
            },
        )

    def test_resource_limits_are_exact_and_bounded(self):
        self.assertEqual(
            self.contract["resource_limits"],
            {
                "default_stage_timeout_seconds": DEFAULT_STAGE_TIMEOUT_SECONDS,
                "max_stage_timeout_seconds": MAX_STAGE_TIMEOUT_SECONDS,
                "max_subprocess_output_bytes": MAX_SUBPROCESS_OUTPUT_BYTES,
                "max_temp_disk_bytes": MAX_TEMP_DISK_BYTES,
                "max_in_memory_artifact_bytes": MAX_IN_MEMORY_ARTIFACT_BYTES,
                "termination_grace_seconds": TERMINATION_GRACE_SECONDS,
                "unbounded_override_allowed": False,
                "limit_exhaustion_fails_closed": True,
            },
        )
        self.assertLessEqual(
            DEFAULT_STAGE_TIMEOUT_SECONDS, MAX_STAGE_TIMEOUT_SECONDS
        )

    def test_cli_modes_and_defaults_match(self):
        normalized_modes = {
            name: {
                "required_options": list(value["required_options"]),
                "forbidden_options": list(value["forbidden_options"]),
                "offline": value["offline"],
                "input_source": value["input_source"],
            }
            for name, value in CLI_MODES.items()
        }
        self.assertEqual(self.contract["cli_modes"], normalized_modes)
        self.assertEqual(
            self.contract["cli_common_options"], list(CLI_COMMON_OPTIONS)
        )
        self.assertEqual(self.contract["cli_defaults"], CLI_DEFAULTS)
        for mode in self.contract["cli_modes"].values():
            self.assertTrue(mode["offline"])

    def test_core_evidence_limits_are_all_fail_closed(self):
        limits = self.contract["core_v1_limits"]
        self.assertTrue(limits["confirmed_first_divergence_always_null"])
        self.assertTrue(limits["confirmed_checkpoint_divergence_always_null"])
        for key in (
            "bounded_digest_is_tensor_equality",
            "bounded_digest_is_numeric_confirmation",
            "next_action_is_evidence",
            "missing_coverage_inferred_equal",
        ):
            self.assertFalse(limits[key], key)

    def test_legacy_base_fixture_is_byte_stable(self):
        digest = hashlib.sha256(LEGACY_FIXTURE.read_bytes()).hexdigest()
        self.assertEqual(digest, LEGACY_SHA256)

    def test_all_product_fixtures_are_utf8_json_without_duplicate_keys(self):
        fixtures = sorted(FIXTURE_ROOT.rglob("*.json"))
        self.assertEqual(len(fixtures), 13)
        for path in fixtures:
            with self.subTest(path=path.relative_to(ROOT)):
                json.loads(path.read_text(), object_pairs_hook=reject_duplicate_keys)


class TestVerdictAndExitPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = load("verdict_exit_policy_v1.json")

    def test_default_and_strict_exit_maps_are_exact(self):
        default = {key.value: value for key, value in DEFAULT_EXIT_CODES.items()}
        strict = {
            key.value: value for key, value in REQUIRE_SUPPORTED_EXIT_CODES.items()
        }
        self.assertEqual(self.policy["default_exit_codes"], default)
        self.assertEqual(self.policy["require_supported_exit_codes"], strict)
        self.assertEqual(self.policy["signal_exit_codes"], SIGNAL_EXIT_CODES)

    def test_strict_policy_changes_only_unsupported_process_exit(self):
        for verdict in CustomerVerdict:
            default = expected_exit_code(verdict, ExecutionPolicy.DEFAULT)
            strict = expected_exit_code(verdict, ExecutionPolicy.REQUIRE_SUPPORTED)
            if verdict == CustomerVerdict.UNSUPPORTED:
                self.assertEqual((default, strict), (0, 6))
            else:
                self.assertEqual(default, strict, verdict)
        self.assertFalse(self.policy["semantic_verdict_rewritten_by_policy"])

    def test_cleanup_warning_does_not_change_semantic_verdict(self):
        self.assertFalse(self.policy["cleanup_warning_alters_semantic_verdict"])
        self.assertEqual(
            set(self.policy["hard_cleanup_failures"]),
            {
                "report_emission_failed",
                "confidentiality_policy_requires_hard_failure",
                "required_evidence_cannot_be_trusted",
            },
        )


class TestProhibitedEvidencePromotion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = load("prohibited_evidence_promotions_v1.json")

    def test_nonclaims_match_product_contract(self):
        self.assertEqual(set(self.fixture["nonclaims"]), set(EVIDENCE_NONCLAIMS))
        self.assertTrue(
            all(value is False for value in self.fixture["nonclaims"].values())
        )

    def test_all_eight_prohibited_promotions_are_explicit(self):
        cases = self.fixture["cases"]
        self.assertEqual(len(cases), 8)
        names = [case["name"] for case in cases]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            {(case["input_evidence"], case["prohibited_claim"]) for case in cases},
            {
                ("bounded_digest_equal", "tensor_equality"),
                ("bounded_digest_mismatch", "numeric_divergence_confirmed"),
                ("suspect_interval", "first_divergence_confirmed"),
                ("no_mismatch_in_partial_coverage", "whole_runtime_equivalence"),
                ("reproduction_request_only", "independent_rerun_verified"),
                ("prefill_construction", "decode_boundary_reproduced"),
                ("next_action", "any_verification_claim"),
                ("missing_coverage", "equivalent_coverage"),
            },
        )


class TestExhaustivePassMapping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mapping = load("pass_status_mapping_v1.json")

    def test_mapping_validates_against_live_enums(self):
        validate_status_mapping(self.mapping)

    def test_exactly_46_terminal_statuses_are_covered_once(self):
        live = runtime_status_values()
        self.assertEqual(sum(len(values) for values in live.values()), 46)
        self.assertEqual(
            sum(len(entries) for entries in self.mapping.values()), 46
        )
        for pass_name, values in live.items():
            mapped = [entry["status"] for entry in self.mapping[pass_name]]
            self.assertEqual(mapped, list(values))
            self.assertEqual(len(mapped), len(set(mapped)))

    def test_pass0_reason_partition_is_exact_and_closed(self):
        live_block_reasons = {
            value.value for value in BLOCK_REASON_TO_REPORT_REASON_CODE
        }
        self.assertEqual(set(PASS0_BLOCK_REASON_VERDICTS), live_block_reasons)
        self.assertEqual(
            set(PASS0_BLOCK_REASON_VERDICTS.values()),
            {
                CustomerVerdict.UNSUPPORTED,
                CustomerVerdict.HARNESS_ERROR,
                CustomerVerdict.INCONCLUSIVE,
            },
        )
        self.assertNotIn(
            CalibrationReasonCode.FORCED_PREFIX_REPORT_JSON_CHANNEL_MISSING.value,
            PASS0_BLOCK_REASON_VERDICTS,
        )
        contract = load("product_contract_v1.json")
        self.assertEqual(
            contract["pass0_block_reason_verdicts"],
            {
                key: verdict.value
                for key, verdict in PASS0_BLOCK_REASON_VERDICTS.items()
            },
        )

    def test_pass3_role_partition_is_finite(self):
        self.assertEqual(
            PASS3_ROLE_TRANSITIONS,
            {
                "pass3a_discovery": "bounded_recapture",
                "pass3b_authoritative_localization": (
                    "pass4_intra_layer_localization"
                ),
            },
        )
        self.assertEqual(
            load("product_contract_v1.json")["pass3_role_transitions"],
            PASS3_ROLE_TRANSITIONS,
        )


class TestStageAndLedgerContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.transitions = load("stage_state_transitions_v1.json")

    def test_stages_and_dependencies_match_module(self):
        self.assertEqual(
            self.transitions["canonical_stages"], list(CANONICAL_STAGES)
        )
        self.assertEqual(
            self.transitions["dependencies"],
            {key: list(value) for key, value in STAGE_DEPENDENCIES.items()},
        )

    def test_final_states_exclude_transient_states(self):
        self.assertEqual(
            self.transitions["stage_states"], [value.value for value in StageState]
        )
        self.assertFalse(
            self.transitions["transient_states_allowed_in_final_report"]
        )
        self.assertNotIn("pending", self.transitions["stage_states"])
        self.assertNotIn("running", self.transitions["stage_states"])

    def test_attempt_and_ledger_are_append_only(self):
        attempt = self.transitions["attempt_identity"]
        self.assertEqual(attempt["entropy_bits"], 128)
        self.assertFalse(attempt["reuse_allowed"])
        self.assertTrue(attempt["retry_requires_new_identity"])
        ledger = self.transitions["ledger"]
        self.assertTrue(ledger["append_only"])
        self.assertFalse(ledger["silent_retry_allowed"])
        self.assertEqual(ledger["file_mode"], "0600")
        self.assertEqual(ledger["run_directory_mode"], "0700")
        self.assertEqual(ledger["max_event_bytes"], MAX_LEDGER_EVENT_BYTES)
        self.assertEqual(ledger["events"], [value.value for value in LedgerEvent])
        self.assertEqual(
            ledger["event_payload_fields"],
            {
                event.value: list(fields)
                for event, fields in LEDGER_PAYLOAD_FIELDS.items()
            },
        )
        validate_ledger_events(self.transitions["valid_ledger_example"])

    def test_ledger_rejects_sequence_and_identity_changes(self):
        events = json.loads(json.dumps(self.transitions["valid_ledger_example"]))
        events[1]["sequence"] = 4
        with self.assertRaisesRegex(ValueError, "sequence"):
            validate_ledger_events(events)
        events = json.loads(json.dumps(self.transitions["valid_ledger_example"]))
        events[1]["attempt_id"] = "lisa1:88888888888888888888888888888888"
        with self.assertRaisesRegex(ValueError, "identity changed"):
            validate_ledger_events(events)

    def test_ledger_rejects_silent_stage_retry(self):
        events = json.loads(json.dumps(self.transitions["valid_ledger_example"]))
        duplicate = json.loads(json.dumps(events[1]))
        duplicate["sequence"] = 2
        duplicate["timestamp_utc"] = "2026-09-03T00:00:02Z"
        events.insert(2, duplicate)
        for sequence, event in enumerate(events):
            event["sequence"] = sequence
        with self.assertRaisesRegex(ValueError, "retry"):
            validate_ledger_events(events)

    def test_unknown_residue_is_never_inferred_zero(self):
        cleanup = self.transitions["cleanup"]
        self.assertFalse(cleanup["unknown_residue_inferred_zero"])
        self.assertFalse(cleanup["startup_stale_run_auto_delete"])

    def test_timeout_interrupt_and_cleanup_truth_table_is_explicit(self):
        conditions = {
            entry["condition"]: entry
            for entry in self.transitions["terminal_conditions"]
        }
        self.assertEqual(conditions["timeout"]["process_exit"], 3)
        self.assertEqual(conditions["handled_SIGINT"]["process_exit"], 130)
        self.assertEqual(conditions["handled_SIGTERM"]["process_exit"], 143)
        self.assertEqual(
            conditions["ordinary_cleanup_warning"]["semantic_result"],
            "unchanged",
        )
        self.assertEqual(
            conditions["cleanup_invalidates_confidentiality_or_evidence"][
                "semantic_result"
            ],
            "HARNESS_ERROR",
        )
        pairs = self.transitions["cleanup_residue_pairs"]
        self.assertEqual(
            pairs,
            {
                status.value: [residue.value for residue in residues]
                for status, residues in CLEANUP_RESIDUE_PAIRS.items()
            },
        )
        self.assertEqual(pairs["success"], ["none_observed"])
        self.assertNotIn("none_observed", pairs["failed"])
        self.assertNotIn("none_observed", pairs["partial"])


class TestCanonicalJson(unittest.TestCase):
    def test_encoding_is_deterministic_and_newline_terminated(self):
        left = canonical_json_bytes({"z": 1, "a": 2})
        right = canonical_json_bytes({"a": 2, "z": 1})
        self.assertEqual(left, right)
        self.assertEqual(left, b'{"a":2,"z":1}\n')

    def test_nonfinite_number_is_rejected(self):
        with self.assertRaises(ValueError):
            canonical_json_bytes({"value": float("nan")})


if __name__ == "__main__":
    unittest.main()
