#!/usr/bin/env python3
"""Model-free checks for the approved differential-verification design contract.

The contract is approved as a design specification; the feature itself is
planned and not yet implemented. These checks enforce that the Markdown
specification and the JSON conformance oracle stay internally consistent and
that the contract never claims the unimplemented feature already exists.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "tools" / "test_fixtures" / "differential_verification_contract.json"
MARKDOWN_PATH = ROOT / "docs" / "differential_verification.md"
RUNTIME_DIR = ROOT / "srcs" / "runtime"


def load_fixture() -> dict:
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_runtime_source(filename: str) -> str:
    """Read a current runtime translation unit (model-free; in-repo path)."""
    return (RUNTIME_DIR / filename).read_text(encoding="utf-8")


def load_markdown() -> str:
    return MARKDOWN_PATH.read_text(encoding="utf-8")


def load_markdown_index() -> dict:
    text = load_markdown()
    match = re.search(
        r"<!-- CONTRACT-INDEX-BEGIN -->\s*```json\s*(.*?)\s*```\s*<!-- CONTRACT-INDEX-END -->",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise AssertionError("Markdown contract index block is missing")
    return json.loads(match.group(1))


class ContractTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = load_fixture()
        cls.markdown = load_markdown()
        cls.markdown_index = load_markdown_index()

    def assertUnique(self, name: str) -> None:
        values = self.data[name]
        self.assertEqual(len(values), len(set(values)), f"{name} contains duplicates")


class TestStatusAndAuthority(ContractTestCase):
    def test_contract_status_is_approved(self):
        self.assertEqual(self.data["contract_status"], "approved")
        self.assertEqual(self.markdown_index["contract_status"], "approved")
        self.assertIn("Contract status: Approved", self.markdown)

    def test_implementation_status_is_planned(self):
        self.assertEqual(self.data["implementation_status"], "planned")
        self.assertEqual(self.markdown_index["implementation_status"], "planned")
        self.assertIn("Implementation status: Planned / Not yet implemented", self.markdown)

    def test_contract_version_is_stable(self):
        self.assertEqual(self.data["contract_version"], "1.0")
        self.assertIn("Contract version: 1.0", self.markdown)

    def test_schema_and_artifact_kind(self):
        self.assertEqual(self.data["schema"], "lis.execution_artifact/v1")
        self.assertEqual(self.data["kind"], "verification_report")
        self.assertEqual(self.data["artifact_kind"], "verification_report")
        self.assertEqual(self.data["report_version"], "1.0")

    def test_markdown_fixture_authority_clause(self):
        meta = self.data["fixture_metadata"]
        self.assertEqual(meta["role"], "machine_readable_conformance_oracle")
        self.assertEqual(meta["markdown_role"], "normative_human_readable_contract")
        self.assertEqual(meta["disagreement_policy"], "contract_validation_failure")
        self.assertFalse(meta["silent_override_allowed"])
        self.assertFalse(meta["stable_public_support_claim"])

    def test_contract_is_not_a_stable_support_claim(self):
        boundary = self.data["implementation_boundary"]
        self.assertFalse(boundary["verification_inconclusive_public_stable_class"])
        self.assertTrue(boundary["no_stable_public_support_claim"])
        self.assertFalse(self.data["fixture_metadata"]["stable_public_support_claim"])


class TestMarkdownFixtureConsistency(ContractTestCase):
    def test_markdown_index_matches_fixture(self):
        for key, value in self.markdown_index.items():
            self.assertIn(key, self.data)
            self.assertEqual(self.data[key], value, key)

    def test_index_covers_required_contract_sets(self):
        required = {
            "contract_status",
            "implementation_status",
            "comparison_modes",
            "mvp_modes",
            "deferred_modes",
            "mode_b_submodes",
            "phase_enum",
            "digest_algorithm_enum",
            "token_identity_levels",
            "evidence_tier_enum",
            "divergence_state_enum",
            "comparison_outcome_enum",
            "overall_verdict_enum",
            "result_class_enum",
            "reason_code_enum",
            "tolerance_profile_names",
            "coverage_fields",
            "exit_code_mapping",
        }
        self.assertTrue(required.issubset(self.markdown_index.keys()))


class TestEnumCompleteness(ContractTestCase):
    def test_all_core_enums_are_unique(self):
        for name in (
            "comparison_modes",
            "mvp_modes",
            "deferred_modes",
            "mode_b_submodes",
            "phase_enum",
            "dtype_enum",
            "model_family_enum",
            "canonical_stage_enum",
            "token_identity_levels",
            "digest_algorithm_enum",
            "evidence_tier_enum",
            "evidence_completeness_enum",
            "divergence_state_enum",
            "comparison_outcome_enum",
            "overall_verdict_enum",
            "first_divergence_scope_enum",
            "cleanup_status_enum",
            "existing_public_result_classes",
            "additive_result_classes",
            "result_class_enum",
            "reason_code_enum",
            "threshold_status_enum",
            "tolerance_profile_names",
            "coverage_fields",
            "prompt_token_identity_fields",
            "prefix_reproduction_fields",
        ):
            self.assertUnique(name)

    def test_mvp_and_deferred_modes_partition_comparison_modes(self):
        modes = set(self.data["comparison_modes"])
        mvp = set(self.data["mvp_modes"])
        deferred = set(self.data["deferred_modes"])
        self.assertEqual(modes, mvp | deferred)
        self.assertFalse(mvp & deferred)
        self.assertEqual(deferred, {"external_semantic"})

    def test_mode_b_submode_completeness(self):
        self.assertEqual(
            self.data["mode_b_submodes"],
            [
                "runtime_regression",
                "determinism_check",
                "configuration_equivalence",
                "precision_policy_comparison",
            ],
        )


class TestResultClassesAndMappings(ContractTestCase):
    def test_result_class_extension_is_exactly_one_additive_class(self):
        stable = [
            "pass",
            "documented_unsupported",
            "numeric_regression",
            "token_parity_regression",
            "benchmark_protocol_regression",
            "harness_configuration_error",
        ]
        self.assertEqual(self.data["existing_public_result_classes"], stable)
        self.assertEqual(self.data["additive_result_classes"], ["verification_inconclusive"])
        self.assertEqual(
            self.data["result_class_enum"],
            stable + ["verification_inconclusive"],
        )

    def test_every_reason_code_maps_to_valid_result_class(self):
        valid = set(self.data["result_class_enum"])
        mapping = self.data["result_class_mapping"]
        self.assertEqual(set(mapping), set(self.data["reason_code_enum"]))
        for code, result_class in mapping.items():
            self.assertIn(result_class, valid, code)

    def test_every_reason_code_maps_to_valid_outcome_and_verdict(self):
        outcomes = set(self.data["comparison_outcome_enum"])
        verdicts = set(self.data["overall_verdict_enum"])
        self.assertEqual(set(self.data["reason_code_outcome_mapping"]), set(self.data["reason_code_enum"]))
        self.assertEqual(set(self.data["reason_code_verdict_mapping"]), set(self.data["reason_code_enum"]))
        for code in self.data["reason_code_enum"]:
            self.assertIn(self.data["reason_code_outcome_mapping"][code], outcomes, code)
            self.assertIn(self.data["reason_code_verdict_mapping"][code], verdicts, code)

    def test_inconclusive_result_class_maps_only_to_inconclusive_verdicts(self):
        for code, result_class in self.data["result_class_mapping"].items():
            if result_class == "verification_inconclusive":
                self.assertEqual(self.data["reason_code_verdict_mapping"][code], "inconclusive", code)
                self.assertIn(
                    self.data["reason_code_outcome_mapping"][code],
                    {"inconclusive", "byte_difference_only", "suspected_divergence_on_coverage"},
                    code,
                )

    def test_outcome_verdict_mapping_is_closed(self):
        outcomes = set(self.data["comparison_outcome_enum"])
        verdicts = set(self.data["overall_verdict_enum"])
        mapping = self.data["outcome_allowed_overall_verdicts"]
        self.assertEqual(set(mapping), outcomes)
        for outcome, allowed in mapping.items():
            self.assertGreater(len(allowed), 0, outcome)
            self.assertTrue(set(allowed).issubset(verdicts), outcome)


class TestExitCodes(ContractTestCase):
    def test_exit_code_mapping_is_exact(self):
        self.assertEqual(
            self.data["exit_code_mapping"],
            {
                "pass": 0,
                "documented_unsupported": 0,
                "harness_configuration_error": 2,
                "verification_inconclusive": 3,
                "token_parity_regression": 4,
                "numeric_regression": 5,
            },
        )

    def test_exit_codes_have_no_collision_except_documented_skip(self):
        mapping = self.data["exit_code_mapping"]
        self.assertEqual(mapping["pass"], mapping["documented_unsupported"])
        nonzero = {
            key: value
            for key, value in mapping.items()
            if key not in {"pass", "documented_unsupported"}
        }
        self.assertEqual(len(nonzero.values()), len(set(nonzero.values())))
        self.assertEqual(set(mapping.values()), {0, 2, 3, 4, 5})

    def test_cleanup_warnings_do_not_override_valid_result(self):
        rules = self.data["exit_code_cleanup_rules"]
        self.assertFalse(rules["cleanup_warnings_alter_valid_result"])
        self.assertEqual(rules["cleanup_warning_reason_code"], "temp_cleanup_failed")
        self.assertEqual(
            set(rules["cleanup_failure_can_change_exit_status_only_when"]),
            {
                "report_emission_failed",
                "confidentiality_policy_requires_hard_failure",
                "required_evidence_cannot_be_trusted",
            },
        )


class TestStepMapping(ContractTestCase):
    def test_prefill_checkpoint_step_is_zero(self):
        mapping = self.data["checkpoint_step_mapping"]
        self.assertEqual(mapping["prefill_runtime_checkpoint_step"], 0)
        prefill = mapping["examples"][0]
        self.assertEqual(prefill["phase"], "prefill")
        self.assertIsNone(prefill["generated_token_step"])
        self.assertEqual(prefill["runtime_checkpoint_step"], 0)

    def test_first_decode_generated_token_maps_to_runtime_step_one(self):
        example = self.data["checkpoint_step_mapping"]["examples"][1]
        self.assertEqual(example["phase"], "decode")
        self.assertEqual(example["generated_token_step"], 0)
        self.assertEqual(example["runtime_checkpoint_step"], 1)

    def test_decode_token_n_maps_to_runtime_step_n_plus_one(self):
        mapping = self.data["checkpoint_step_mapping"]
        self.assertEqual(mapping["decode_runtime_checkpoint_step_formula"], "generated_token_step + 1")
        self.assertEqual(mapping["pass1_mismatch_index_to_runtime_checkpoint_step"], "N + 1")
        for example in mapping["examples"]:
            if example["phase"] == "decode":
                self.assertEqual(
                    example["runtime_checkpoint_step"],
                    example["generated_token_step"] + 1,
                )

    def test_checkpoint_identity_distinguishes_generated_and_runtime_steps(self):
        fields = set(self.data["checkpoint_identity_fields"])
        self.assertIn("generated_token_step", fields)
        self.assertIn("runtime_checkpoint_step", fields)
        self.assertIn(
            "generated_token_step",
            self.data["checkpoint_step_mapping"]["decode_report_fields_required"],
        )
        self.assertIn(
            "runtime_checkpoint_step",
            self.data["checkpoint_step_mapping"]["decode_report_fields_required"],
        )


class TestDigestSemantics(ContractTestCase):
    def test_sha256_is_the_strong_digest(self):
        self.assertEqual(self.data["strong_digest_algorithm"], "sha256")
        self.assertIn("sha256", self.data["digest_algorithm_enum"])
        self.assertIn("fnv1a64", self.data["digest_algorithm_enum"])

    def test_sha256_representation_is_precise(self):
        rep = self.data["strong_digest_representation"]
        self.assertEqual(rep["prefix"], "sha256:")
        self.assertEqual(rep["hex_characters"], 64)
        self.assertEqual(rep["hex_case"], "lowercase")
        self.assertRegex(
            "sha256:" + ("a" * 64),
            re.compile(rep["regex"]),
        )

    def test_digest_equality_is_not_array_or_numeric_equality(self):
        rep = self.data["strong_digest_representation"]
        limits = self.data["hash_only_limitations"]
        self.assertTrue(rep["not_elementwise_array_equality"])
        self.assertTrue(rep["not_tolerance_aware_numeric_evidence"])
        self.assertTrue(limits["sha256_equality_is_not_array_equality"])
        self.assertTrue(limits["sha256_mismatch_is_not_numeric_evidence"])
        self.assertTrue(limits["hash_only_byte_mismatch_cannot_confirm_checkpoint_divergence"])


class TestToleranceProfiles(ContractTestCase):
    def test_profile_names_and_objects_match(self):
        names = set(self.data["tolerance_profile_names"])
        self.assertEqual(names, set(self.data["tolerance_profiles"]))

    def test_all_profiles_have_required_metadata(self):
        required = set(self.data["tolerance_profile_required_fields"])
        valid_statuses = set(self.data["threshold_status_enum"])
        for name, profile in self.data["tolerance_profiles"].items():
            self.assertEqual(set(profile), required, name)
            self.assertEqual(profile["name"], name)
            self.assertIn(profile["threshold_status"], valid_statuses)

    def test_current_thresholds_are_uncalibrated_defaults(self):
        self.assertEqual(
            self.data["tolerance_profile_rules"]["current_numeric_values_status"],
            "uncalibrated_default",
        )
        for profile in self.data["tolerance_profiles"].values():
            self.assertEqual(profile["threshold_status"], "uncalibrated_default")

    def test_calibration_requires_provenance_holdout_and_versioning(self):
        rules = self.data["tolerance_profile_rules"]
        self.assertTrue(rules["calibrated_requires_provenance"])
        self.assertTrue(rules["calibrated_requires_holdout_validation"])
        self.assertTrue(rules["calibrated_changes_require_new_profile_version"])
        for profile in self.data["tolerance_profiles"].values():
            if profile["threshold_status"] == "calibrated":
                self.assertIsNotNone(profile["calibration_provenance"])
                self.assertIsNotNone(profile["holdout_validation"])


class TestBinaryHeader(ContractTestCase):
    def test_binary_header_freezes_required_fields(self):
        header = self.data["binary_tensor_header"]
        expected = {
            "magic",
            "major_version",
            "header_length",
            "flags",
            "dtype",
            "phase",
            "rank",
            "shape",
            "element_count",
            "payload_byte_count",
            "generated_token_step",
            "runtime_checkpoint_step",
            "layer_index",
            "execution_ordinal",
            "mode",
            "mode_b_submode",
            "model_family",
            "canonical_stage",
            "source_checkpoint_name",
            "digest_algorithm",
            "payload_digest",
            "reserved",
        }
        self.assertEqual(set(header["required_fields"]), expected)

    def test_binary_header_version_endianness_shape_and_reserved_policy(self):
        header = self.data["binary_tensor_header"]
        self.assertEqual(header["magic"], "LISB")
        self.assertEqual(header["major_version"], 1)
        self.assertEqual(header["encoding"], "fixed_width_little_endian")
        self.assertTrue(header["header_length_required"])
        self.assertTrue(header["flags_required"])
        self.assertEqual(header["rank_limit"], 4)
        self.assertEqual(header["shape_length"], 4)
        self.assertEqual(header["global_layer_sentinel"], "UINT64_MAX")
        self.assertEqual(header["payload_digest_algorithm"], "sha256")
        self.assertEqual(header["reserved_forward_compatibility_policy"], "must_be_zero_for_v1")

    def test_binary_header_validation_rules_are_complete(self):
        rules = set(self.data["binary_tensor_header"]["validation_rules"])
        expected = {
            "reject_wrong_magic",
            "reject_unsupported_major_version",
            "reject_invalid_header_length",
            "reject_unknown_required_flags",
            "reject_invalid_rank",
            "reject_nonzero_unused_shape_dimensions_where_prohibited",
            "reject_element_count_overflow",
            "reject_payload_byte_count_mismatch",
            "reject_file_length_not_exactly_header_plus_payload",
            "reject_truncated_files",
            "reject_oversized_files",
            "reject_checkpoint_identity_mismatch",
            "reject_dtype_mismatch_across_inputs",
            "reject_shape_mismatch_across_inputs",
            "reject_malformed_string_fields",
            "reject_nonzero_reserved_bytes_for_v1",
        }
        self.assertEqual(rules, expected)


class TestSecureTransportAndCleanup(ContractTestCase):
    def test_secure_directory_and_basename_policy(self):
        policy = self.data["secure_temp_file_policy"]
        self.assertEqual(policy["threat_model"], "local_single_user_mvp")
        self.assertEqual(policy["path_contract"], "directory_plus_basename")
        self.assertTrue(policy["per_run_directory_unpredictable"])
        self.assertEqual(policy["per_run_directory_mode"], "0700")
        self.assertTrue(policy["directory_owner_uid_must_match_current_user"])
        self.assertTrue(policy["reference_candidate_basenames_unpredictable"])
        self.assertTrue(policy["reference_candidate_basenames_distinct"])
        self.assertEqual(set(policy["producer_receives"]), {"validated_directory", "basename"})
        self.assertEqual(
            set(policy["basename_rejections"]),
            {"..", "absolute_basename", "directory_separator", "outside_run_directory"},
        )

    def test_secure_file_creation_and_validation_policy(self):
        policy = self.data["secure_temp_file_policy"]
        self.assertEqual(policy["file_creation_mode"], "0600")
        self.assertTrue(policy["exclusive_creation_required"])
        self.assertFalse(policy["silent_overwrite_allowed"])
        self.assertIn("o_nofollow", policy["nofollow_rule"])
        self.assertTrue(policy["validation_never_silently_skipped"])
        self.assertTrue(policy["producer_fstat_required"])
        self.assertEqual(
            set(policy["comparator_validations"]),
            {
                "regular_file",
                "owner_uid",
                "mode_0600",
                "hard_link_count_one",
                "expected_header",
                "expected_identity",
                "exact_file_length",
            },
        )

    def test_cleanup_handled_and_abnormal_paths(self):
        policy = self.data["secure_temp_file_policy"]
        self.assertEqual(policy["cleanup_on_normal_path"], "mandatory")
        self.assertEqual(policy["cleanup_on_handled_error_path"], "mandatory")
        self.assertEqual(policy["cleanup_after_abnormal_termination"], "best_effort_only")
        self.assertEqual(
            set(policy["abnormal_cleanup_not_guaranteed_after"]),
            {"SIGKILL", "host_crash", "power_loss", "kernel_failure", "filesystem_failure"},
        )
        self.assertFalse(policy["report_contains_paths"])
        self.assertFalse(policy["cleanup_failure_silently_ignored"])
        self.assertTrue(policy["numeric_verdict_remains_valid_if_cleanup_fails_after_comparison"])

    def test_stale_run_policy(self):
        stale = self.data["secure_temp_file_policy"]["stale_run_policy"]
        self.assertEqual(stale["default_stale_threshold_hours"], 24)
        self.assertEqual(stale["destructive_cleanup_default"], "explicit_cleanup_command_only")
        self.assertTrue(stale["startup_detects_and_warns"])
        self.assertFalse(stale["startup_deletes_by_default"])
        self.assertEqual(
            set(stale["metadata_marker_fields"]),
            {"format_version", "uid", "pid", "creation_time", "last_heartbeat_time", "debug_retain"},
        )
        self.assertEqual(set(stale["active_run_protection"]), {"lock", "heartbeat"})
        self.assertTrue(stale["validate_owner_and_mode_before_deletion"])
        self.assertTrue(stale["retained_debug_directories_skipped"])
        self.assertTrue(stale["audit_summary_for_deleted_or_skipped"])
        self.assertTrue(stale["warnings_use_safe_run_identifiers_only"])


class TestCheckpointCoverage(ContractTestCase):
    def test_qwen3_current_coverage_is_sparse_layer_zero(self):
        qwen = self.data["current_checkpoint_coverage"]["qwen3_dense_decoder"]
        self.assertEqual(qwen["coverage_status"], "sparse_current_layer0")
        self.assertEqual(qwen["current_layers"], [0])
        self.assertEqual(
            qwen["current_source_names"],
            [
                "qwen3.embedding",
                "qwen3.layer.0.q_after_q_norm",
                "qwen3.layer.0.k_after_k_norm",
                "qwen3.final_norm",
                "qwen3.logits",
            ],
        )

    def test_qwen3_arbitrary_layer_projection_attention_mlp_not_current(self):
        qwen = self.data["current_checkpoint_coverage"]["qwen3_dense_decoder"]
        self.assertFalse(qwen["arbitrary_layer_projection_rope_attention_mlp_current"])
        not_current = set(qwen["not_current_source_patterns"])
        self.assertIn("qwen3.layer.<n>.q_proj", not_current)
        self.assertIn("qwen3.layer.<n>.attention_output", not_current)
        self.assertIn("qwen3.layer.<n>.mlp_output", not_current)

    def test_llama_current_coverage_is_sparse_diagnostic(self):
        llama = self.data["current_checkpoint_coverage"]["llama3_decoder"]
        self.assertEqual(llama["coverage_status"], "sparse_current_diagnostic")
        self.assertTrue(llama["not_complete_taxonomy"])
        self.assertEqual(llama["current_layers"], [0, 1, 7, 8])
        # Accurate anchors that the current runtime actually emits as literals.
        for name in (
            "embedding",
            "final_norm",
            "logits",
            "layer.0.q_proj_out",
            "layer.0.q_after_rope",
            "layer.7.k_proj_weight",
        ):
            self.assertIn(name, llama["current_source_names"], name)

    def test_llama_current_source_names_are_grounded_in_runtime(self):
        """Every explicit Llama current source name must be a literal in llama.c."""
        llama = self.data["current_checkpoint_coverage"]["llama3_decoder"]
        source = load_runtime_source("llama.c")
        for name in llama["current_source_names"]:
            self.assertIn(
                f'"{name}"',
                source,
                f"{name} is listed as current but is not emitted as a literal in llama.c",
            )

    def test_llama_dynamic_output_is_a_pattern_not_a_literal(self):
        llama = self.data["current_checkpoint_coverage"]["llama3_decoder"]
        self.assertIn(
            "layer.<selected_layer>.output",
            llama["current_source_name_patterns"],
        )
        # Bracketed/templated names must never appear as concrete current literals.
        self.assertNotIn("layer.<selected_layer>.output", llama["current_source_names"])
        self.assertNotIn("layer.<n>.output", llama["current_source_names"])
        # The pattern's literal stem is genuinely emitted by snprintf in llama.c.
        self.assertIn(".output", load_runtime_source("llama.c"))

    def test_non_emitted_llama_names_are_not_current(self):
        """Regression guard: names the runtime never emits stay out of current coverage."""
        llama = self.data["current_checkpoint_coverage"]["llama3_decoder"]
        expected_not_current = {
            "layer.0.attention_norm",
            "layer.0.q_proj",
            "layer.0.k_proj",
            "layer.0.v_proj",
            "layer.0.q_rope",
            "layer.0.k_rope",
            "layer.0.attention_out",
            "layer.7.input",
            "layer.7.weights",
        }
        self.assertEqual(set(llama["not_current_source_names"]), expected_not_current)
        current = set(llama["current_source_names"])
        for name in expected_not_current:
            self.assertNotIn(name, current, name)
        # Three plausible-looking names the runtime nonetheless never emits verbatim.
        for name in ("layer.0.q_proj", "layer.0.q_rope", "layer.7.weights"):
            self.assertIn(name, llama["not_current_source_names"], name)
            self.assertNotIn(name, current, name)

    def test_qwen3_current_source_names_are_grounded_in_runtime(self):
        """The exact, sparse Qwen3 current names must be literals in qwen3.c."""
        qwen = self.data["current_checkpoint_coverage"]["qwen3_dense_decoder"]
        source = load_runtime_source("qwen3.c")
        for name in qwen["current_source_names"]:
            self.assertIn(
                f'"{name}"',
                source,
                f"{name} is listed as current but is not emitted as a literal in qwen3.c",
            )

    def test_canonical_taxonomy_is_reporting_only(self):
        taxonomy = self.data["canonical_stage_taxonomy"]
        compat = self.data["model_family_compatibility_rules"]
        self.assertTrue(taxonomy["reporting_only"])
        self.assertTrue(taxonomy["does_not_enable_cross_family_numeric_comparison"])
        self.assertFalse(compat["canonical_stage_equality_enables_numeric_compare"])
        self.assertTrue(compat["cross_family_numeric_prohibited"])


class TestReportSchema(ContractTestCase):
    def test_required_top_level_report_fields_are_explicit(self):
        fields = set(self.data["report_schema"]["required_top_level_fields"])
        expected = {
            "schema",
            "kind",
            "report_version",
            "mode",
            "submode",
            "result_class",
            "reason_code",
            "comparison_outcome",
            "overall_verdict",
            "evidence_completeness",
            "tolerance_profile",
            "sources",
            "prompt_token_identity",
            "prefix_reproduction",
            "coverage",
            "comparison",
            "suspected_first_divergence",
            "confirmed_divergence_at_checkpoint",
            "confirmed_first_divergence",
            "warnings",
            "cleanup_status",
        }
        self.assertEqual(fields, expected)

    def test_nullable_fields_are_explicit(self):
        nullable = set(self.data["report_schema"]["nullable_fields"])
        self.assertIn("confirmed_first_divergence", nullable)
        self.assertIn("confirmed_divergence_at_checkpoint", nullable)
        self.assertIn("generated_token_step", nullable)
        self.assertIn("layer", nullable)

    def test_source_provenance_coverage_prefix_and_checkpoint_fields_are_explicit(self):
        schema = self.data["report_schema"]
        for key in (
            "required_source_fields",
            "required_provenance_fields",
            "required_coverage_fields",
            "required_prefix_reproduction_fields",
            "required_confirmed_checkpoint_fields",
        ):
            self.assertGreater(len(schema[key]), 0, key)
        self.assertEqual(schema["required_coverage_fields"], self.data["coverage_fields"])
        self.assertIn("runtime_checkpoint_step", schema["required_confirmed_checkpoint_fields"])
        self.assertIn("generated_token_step", schema["required_confirmed_checkpoint_fields"])

    def test_confidential_fields_are_redacted_by_default(self):
        redacted = set(self.data["report_schema"]["redacted_by_default"])
        self.assertEqual(
            redacted,
            {
                "raw_tensor_values",
                "full_prompt_token_arrays",
                "raw_text",
                "absolute_temporary_paths",
                "model_paths",
                "temporary_artifact_paths",
            },
        )


class TestDivergenceAndMVPSemantics(ContractTestCase):
    def test_confirmed_checkpoint_requires_exhaustive_numeric_comparison(self):
        required = set(
            self.data["confirmation_eligibility_rules"][
                "confirmed_divergence_at_checkpoint_requires"
            ]
        )
        self.assertEqual(
            required,
            {
                "exhaustive_numeric_comparison",
                "all_elements_compared",
                "active_tolerance_applied",
                "at_least_one_beyond_tolerance",
                "valid_compatible_inputs",
                "verified_prefix_reproduction",
                "no_malformed_incomplete_evidence",
            },
        )

    def test_partial_coverage_prohibits_first_divergence_confirmation(self):
        rules = self.data["confirmation_eligibility_rules"]
        self.assertTrue(rules["partial_coverage_forces_confirmed_first_divergence_null"])
        self.assertIsNone(rules["mvp_confirmed_first_divergence_value"])
        self.assertIn("confirmed_first_divergence_always_null", self.data["mvp_limitations"])

    def test_mvp_includes_checkpoint_level_exhaustive_confirmation(self):
        self.assertTrue(
            self.data["confirmation_eligibility_rules"][
                "checkpoint_confirmation_over_existing_checkpoint_supported_in_mvp"
            ]
        )
        self.assertIn("exhaustive_tier2_comparison_of_one_existing_checkpoint", self.data["mvp_capabilities"])
        self.assertIn("confirmed_divergence_at_checkpoint", self.data["mvp_capabilities"])
        self.assertEqual(self.data["strongest_mvp_verdict"], "confirmed_divergence_at_checkpoint")

    def test_mvp_excludes_mode_c_and_cross_family_numeric_comparison(self):
        self.assertNotIn("external_semantic", self.data["mvp_modes"])
        self.assertIn("external_semantic", self.data["deferred_modes"])
        self.assertIn("no_mode_c_external_semantic", self.data["mvp_limitations"])
        self.assertIn("no_cross_family_comparison", self.data["mvp_limitations"])


class TestDeepVerificationAndModeC(ContractTestCase):
    def test_deep_verification_resource_controls_are_required_for_first_divergence(self):
        deep = self.data["deep_verification_resource_controls"]
        self.assertEqual(deep["status"], "outside_mvp")
        self.assertTrue(deep["required_for_confirmed_first_divergence"])
        self.assertTrue(deep["explicit_user_acknowledgment_required"])
        self.assertEqual(
            set(deep["fields"]),
            {
                "max_tensor_bytes",
                "max_checkpoints",
                "max_temp_disk_bytes",
                "sequential_predecessor_recapture",
                "retain_predecessor_captures",
                "early_stop_on_first_confirmed_divergence",
                "fail_closed_on_resource_limit",
            },
        )
        self.assertTrue(deep["sequential_predecessor_recapture"])
        self.assertFalse(deep["retain_predecessor_captures"])
        self.assertTrue(deep["fail_closed_on_resource_limit"])
        self.assertIsNone(deep["mvp_confirmed_first_divergence"])

    def test_mode_c_remains_deferred_with_required_future_provenance_fields(self):
        mode_c = self.data["mode_c_deferred_contract"]
        self.assertEqual(mode_c["mode"], "external_semantic")
        self.assertEqual(mode_c["status"], "deferred")
        self.assertTrue(mode_c["not_mvp"])
        self.assertNotIn("external_semantic", self.data["mvp_modes"])
        required = set(mode_c["required_future_fields"])
        self.assertEqual(
            required,
            {
                "explicit_injected_token_prefix_identity",
                "target_generated_token_step",
                "runtime_checkpoint_step",
                "token_position_alignment",
                "external_framework_name",
                "external_framework_version",
                "adapter_name",
                "adapter_version",
                "model_provenance",
                "config_provenance",
                "tokenizer_provenance",
                "dependency_environment_identity",
                "dtype_accumulation_policy",
                "deterministic_execution_settings",
                "hook_checkpoint_mapping_version",
                "confidentiality_policy",
                "no_raw_prompt_or_model_leakage_by_default",
            },
        )


class TestImplementationBoundary(ContractTestCase):
    def test_contract_does_not_claim_feature_surfaces_exist(self):
        boundary = self.data["implementation_boundary"]
        for key in (
            "binary_checkpoint_capture_implemented",
            "exhaustive_comparator_implemented",
            "verification_report_implemented",
            "new_cli_flags_implemented",
            "verify_diff_make_target_implemented",
            "lis_inspect_verification_view_implemented",
            "mode_c_adapter_implemented",
        ):
            self.assertFalse(boundary[key], key)
        # The contract is approved, but the feature itself is planned only.
        self.assertEqual(self.data["contract_status"], "approved")
        self.assertEqual(self.data["implementation_status"], "planned")


if __name__ == "__main__":
    unittest.main()
