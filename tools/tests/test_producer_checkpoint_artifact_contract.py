#!/usr/bin/env python3
"""Parity and byte-vector tests for the frozen P3-P1 producer contract."""

from __future__ import annotations

import hashlib
import json
import re
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tools" / "test_fixtures" / "differential_verification_contract.json"
MARKDOWN = ROOT / "docs" / "differential_verification.md"
PRODUCER_FIXTURES = (
    ROOT / "tools" / "test_fixtures" / "layer_localization" / "producer_contract"
)
EXAMPLE = PRODUCER_FIXTURES / "llama_layer_trace_vnext_schema_examples.json"
LEGACY = PRODUCER_FIXTURES / "legacy_layer_trace_without_binding.json"
VECTORS = PRODUCER_FIXTURES / "checkpoint_digest_test_vectors.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _length_prefixed(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _canonical_digest_stream(vector: dict, contract: dict) -> bytes:
    stream = bytearray(contract["domain_tag_utf8"].encode("utf-8"))
    stream.extend(bytes.fromhex(contract["domain_terminator_hex"]))
    stream.extend(_length_prefixed(contract["version"]))
    stream.extend(_length_prefixed(vector["tensor_role"]))
    stream.extend(struct.pack("<Q", len(vector["shape"])))
    for dimension in vector["shape"]:
        stream.extend(struct.pack("<Q", dimension))
    stream.extend(_length_prefixed(contract["observed_dtype"]))
    stream.extend(_length_prefixed(contract["byte_order"]))
    stream.extend(struct.pack("<Q", len(vector["input_fp32_bits_hex"])))
    for encoded_bits in vector["input_fp32_bits_hex"]:
        bits = int(encoded_bits, 16)
        if bits & 0x7F800000 == 0x7F800000 and bits & 0x007FFFFF:
            bits = int(contract["canonical_qnan_bits_hex"], 16)
        stream.extend(struct.pack("<I", bits))
    return bytes(stream)


class ProducerContractTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = _load(FIXTURE)
        cls.local = cls.contract["producer_checkpoint_artifact"]
        cls.example = _load(EXAMPLE)
        cls.legacy = _load(LEGACY)
        cls.vectors = _load(VECTORS)
        text = MARKDOWN.read_text(encoding="utf-8")
        match = re.search(
            r"<!-- PRODUCER-CHECKPOINT-ARTIFACT-INDEX-BEGIN -->\s*"
            r"```json\s*(.*?)\s*```\s*"
            r"<!-- PRODUCER-CHECKPOINT-ARTIFACT-INDEX-END -->",
            text,
            flags=re.DOTALL,
        )
        if not match:
            raise AssertionError("producer checkpoint artifact index is missing")
        cls.markdown_index = json.loads(match.group(1))


class TestProducerContractParity(ProducerContractTestCase):
    def test_markdown_namespace_is_exact(self):
        self.assertEqual(self.markdown_index, self.local)

    def test_contract_is_frozen_before_implementation(self):
        self.assertEqual(self.local["status"], "frozen")
        self.assertEqual(self.local["schema"], "lis.execution_artifact/v1")
        self.assertTrue(
            self.local["dependency_gate"][
                "post_freeze_change_requires_coordinated_contract_revision"
            ]
        )
        self.assertEqual(
            set(self.local["dependency_gate"]["p3_p1_green_unlocks"]),
            {"P3-P2-through-P3-P7", "P3-C1-through-P3-C8"},
        )

    def test_frozen_global_verification_enums_are_untouched(self):
        self.assertNotIn("observable_mismatch_found", self.contract["reason_code_enum"])
        self.assertNotIn("layer_localization", self.contract["result_class_enum"])


class TestArtifactSetIdContract(ProducerContractTestCase):
    def test_format_lifecycle_and_entropy_are_exact(self):
        artifact_set = self.local["artifact_set_id"]
        self.assertEqual(artifact_set["random_bytes"], 16)
        self.assertFalse(artifact_set["bytes_transformed_before_hex"])
        self.assertEqual(artifact_set["random_source"], "operating_system_csprng")
        self.assertRegex(
            self.example["artifact_set_id"], re.compile(artifact_set["regex"])
        )
        self.assertIn("once_at_cli_inference_execution_start", artifact_set["generation_point"])
        self.assertIn("unchanged_to_every_related_artifact", artifact_set["propagation"])
        self.assertEqual(artifact_set["separate_execution_policy"], "independently_sampled")

    def test_failure_is_closed_and_no_fallback_exists(self):
        artifact_set = self.local["artifact_set_id"]
        self.assertFalse(artifact_set["fallback_allowed"])
        self.assertEqual(
            set(artifact_set["prohibited_fallbacks"]),
            {"time", "pid", "counter", "fnv1a64", "settings", "predictable_data"},
        )
        self.assertIn("fail_closed_before_inference", artifact_set["failure_policy"])

    def test_evidence_semantics_do_not_overclaim(self):
        artifact_set = self.local["artifact_set_id"]
        self.assertEqual(
            artifact_set["evidence_semantics"],
            "probabilistic_same_cli_execution_association",
        )
        for key in (
            "absolute_uniqueness_claim",
            "content_identity",
            "configuration_compatibility",
            "matching_id_sufficient_for_source_binding",
        ):
            self.assertFalse(artifact_set[key], key)
        self.assertFalse(
            self.local["source_binding"]["forced_id_collision_bypasses_other_links"]
        )
        self.assertFalse(self.local["source_binding"]["fnv1a64_satisfies_chain"])

    def test_complete_binding_chain_is_frozen(self):
        self.assertEqual(
            self.local["source_binding"]["chain"],
            [
                "supplied_canonical_pass2_artifact",
                "canonical_pass2_artifact_sha256",
                "typed_pass2_result_artifact_coherence",
                "pass2_bound_role_run_report_sha256",
                "supplied_run_report_canonical_sha256",
                "matching_artifact_set_id",
                "matching_semantic_manifest_identity",
                "matching_target_runtime_checkpoint_step",
                "canonical_layer_trace_sha256",
            ],
        )


class TestCheckpointLayoutContract(ProducerContractTestCase):
    def test_schema_example_has_every_required_field(self):
        layout_contract = self.local["checkpoint_layout"]
        layout = self.example["checkpoint_layout"]
        self.assertEqual(self.example["schema"], self.local["schema"])
        self.assertEqual(self.example["kind"], "layer_trace")
        self.assertTrue(set(layout_contract["required_layout_fields"]).issubset(layout))
        self.assertEqual(layout["layout_name"], "llama_layer_output_summary")
        self.assertEqual(layout["layout_version"], 1)
        for entry in self.example["layer_trace"]:
            self.assertTrue(
                set(layout_contract["required_entry_fields"]).issubset(entry)
            )
            self.assertEqual(entry["name"], f"layer.{entry['layer_index']}.output")
            self.assertNotIn("tensor_payload", entry)
            self.assertNotIn("values", entry)

    def test_coverage_partition_and_entries_are_explicit(self):
        layout = self.example["checkpoint_layout"]
        requested = {self._coordinate_key(item) for item in layout["requested_coordinates"]}
        captured = {self._coordinate_key(item) for item in layout["captured_coordinates"]}
        missing = {
            self._coordinate_key(item["coordinate"])
            for item in layout["missing_coordinates"]
        }
        entries = {self._coordinate_key(item) for item in self.example["layer_trace"]}
        self.assertEqual(requested, captured | missing)
        self.assertFalse(captured & missing)
        self.assertEqual(captured, entries)
        self.assertEqual(len(captured), len(layout["captured_coordinates"]))

    @staticmethod
    def _coordinate_key(item: dict) -> tuple:
        return tuple(
            item[field]
            for field in (
                "runtime_checkpoint_step",
                "layer_index",
                "tensor_role",
                "batch_index",
                "sequence_index",
                "stage_order",
                "execution_ordinal",
            )
        )

    def test_declared_order_is_not_silently_sortable(self):
        coordinates = self.example["checkpoint_layout"]["captured_coordinates"]
        keys = [
            (
                item["runtime_checkpoint_step"],
                item["layer_index"],
                item["stage_order"],
                item["execution_ordinal"],
            )
            for item in coordinates
        ]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(
            [item["execution_ordinal"] for item in coordinates],
            list(range(len(coordinates))),
        )
        self.assertEqual(
            self.local["checkpoint_layout"]["duplicate_coordinate_policy"],
            "reject_artifact_before_write",
        )

    def test_legacy_and_qwen3_behavior_are_fail_closed(self):
        self.assertNotIn("artifact_set_id", self.legacy)
        self.assertNotIn("checkpoint_layout", self.legacy)
        compatibility = self.local["compatibility"]
        self.assertTrue(compatibility["legacy_artifacts_remain_valid_execution_artifacts"])
        self.assertEqual(compatibility["legacy_pass3_behavior"], "unsupported_checkpoint_layout")
        self.assertEqual(compatibility["qwen3_mvp_behavior"], "unsupported_checkpoint_layout")
        self.assertFalse(compatibility["qwen3_qk_norm_adaptation_allowed"])


class TestCheckpointDigestVectors(ProducerContractTestCase):
    def test_vector_contract_matches_frozen_namespace(self):
        digest = self.local["digest_contract"]
        vector_contract = self.vectors["contract"]
        for key in (
            "algorithm",
            "version",
            "domain_tag_utf8",
            "domain_terminator_hex",
            "length_prefix_encoding",
            "integer_encoding",
            "observed_dtype",
            "byte_order",
            "canonicalization",
            "canonical_qnan_bits_hex",
        ):
            self.assertEqual(vector_contract[key], digest[key], key)

    def test_every_canonical_byte_stream_and_sha256_is_exact(self):
        contract = self.vectors["contract"]
        for vector in self.vectors["vectors"]:
            stream = _canonical_digest_stream(vector, contract)
            self.assertEqual(stream.hex(), vector["canonical_stream_hex"], vector["name"])
            self.assertEqual(
                "sha256:" + hashlib.sha256(stream).hexdigest(),
                vector["digest"],
                vector["name"],
            )

    def test_vectors_cover_canonicalization_and_domain_separation(self):
        names = {vector["name"] for vector in self.vectors["vectors"]}
        self.assertEqual(
            names,
            {
                "representative_finite_values",
                "preserve_signed_zero",
                "preserve_infinities",
                "canonicalize_all_nans",
                "shape_domain_separation",
                "role_domain_separation",
            },
        )
        digests = [vector["digest"] for vector in self.vectors["vectors"]]
        self.assertEqual(len(digests), len(set(digests)))

    def test_digest_semantics_and_disabled_mode_are_bounded(self):
        digest = self.local["digest_contract"]
        self.assertTrue(digest["diagnostic_mode_only"])
        self.assertFalse(digest["normal_inference_digest_work"])
        self.assertFalse(digest["mathematical_tensor_equality_claim"])
        self.assertFalse(digest["collision_free_claim"])
        self.assertEqual(
            digest["precision_path_policy"],
            "underlying_precision_paths_must_match_exactly_for_mvp",
        )


if __name__ == "__main__":
    unittest.main()
