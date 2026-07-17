#!/usr/bin/env python3
"""Pass 3 serializer, golden, mapping, and regression-boundary tests."""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from lis_verify import map_pass3_reason, map_pass3_status, serialize_pass3
from lis_verify.pass1_inputs import canonical_json_sha256
from lis_verify.pass3_model import Pass3ReasonCode, Pass3Status

from .pass3_test_support import ready_case, run_case


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "tools" / "test_fixtures" / "differential_verification_contract.json"
GOLDEN = (
    ROOT / "tools" / "test_fixtures" / "layer_localization" / "golden"
    / "layer_localization_entry_mismatch.json"
)
MODULES = ROOT / "tools" / "lis_verify"


def recursive_keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield key
            yield from recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_keys(child)


class TestPass3Artifact(unittest.TestCase):
    def test_golden_entry_mismatch_is_exact(self):
        result = run_case(
            ready_case(
                reference_layers=(0,),
                candidate_layers=(0,),
                candidate_digest_overrides={0: "sha256:" + "f" * 64},
            )
        )
        self.assertEqual(
            serialize_pass3(result),
            json.loads(GOLDEN.read_text(encoding="utf-8")),
        )

    def test_required_contract_fields_are_exactly_present(self):
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        required = set(
            contract["coverage_scoped_layer_localization"][
                "artifact_required_fields"
            ]
        )
        self.assertEqual(required, set(serialize_pass3(run_case(ready_case()))))

    def test_serialization_is_deterministic_and_bounded(self):
        result = run_case(ready_case())
        self.assertEqual(serialize_pass3(result), serialize_pass3(result))
        comparisons = serialize_pass3(result)["comparisons"]
        self.assertEqual(comparisons["serialized_count"], 4)
        self.assertFalse(comparisons["truncated"])

    def test_no_prohibited_recursive_keys(self):
        artifact = serialize_pass3(
            run_case(
                ready_case(
                    candidate_digest_overrides={8: "sha256:" + "f" * 64}
                )
            )
        )
        keys = set(recursive_keys(artifact))
        for prohibited in (
            "tensor_payload",
            "values",
            "confirmed_first_divergent_layer",
            "confirmed_divergence_at_checkpoint",
            "confirmed_first_divergence",
            "pass4_ready",
            "pass5_ready",
        ):
            self.assertNotIn(prohibited, keys)

    def test_canonical_pass2_identity_is_serialized(self):
        case = ready_case()
        artifact = serialize_pass3(run_case(case))
        self.assertEqual(
            artifact["pass2_artifact_sha256"],
            case["pass2_artifact"].artifact_sha256,
        )
        self.assertTrue(artifact["pass2_object_artifact_coherence_verified"])


class TestPass3MappingBoundary(unittest.TestCase):
    def test_success_statuses_have_no_frozen_mapping(self):
        self.assertIsNone(
            map_pass3_status(Pass3Status.OBSERVABLE_MISMATCH_FOUND)
        )
        self.assertIsNone(
            map_pass3_status(Pass3Status.NO_MISMATCH_IN_CAPTURED_COVERAGE)
        )
        self.assertIsNone(
            map_pass3_reason(Pass3ReasonCode.OBSERVABLE_MISMATCH_FOUND)
        )
        self.assertIsNone(
            map_pass3_reason(
                Pass3ReasonCode.NO_MISMATCH_IN_CAPTURED_COVERAGE
            )
        )

    def test_all_defined_blocked_mappings_target_frozen_reasons(self):
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        frozen = set(contract["reason_code_enum"])
        for reason in Pass3ReasonCode:
            mapped = map_pass3_reason(reason)
            if mapped is not None:
                self.assertIn(mapped, frozen)
        for status in Pass3Status:
            mapped = map_pass3_status(status)
            if mapped is not None:
                self.assertIn(mapped, frozen)


class TestExistingIdentityStability(unittest.TestCase):
    def test_pass1_and_pass2_golden_hashes_are_unchanged(self):
        paths = (
            (
                ROOT / "tools/test_fixtures/token_localization/golden/token_localization_mismatch_n.json",
                "sha256:412df64c358a6d08bcd7eb8e59e1f2614cf2f89d89c47f4635adb3c66e9feaab",
            ),
            (
                ROOT / "tools/test_fixtures/prefix_policy_reproduction/golden/prefix_policy_reproduction_verified.json",
                "sha256:b99eafad812d4b202b6b2e4b6818b0af93759db7eb71efe6e17994bd06dbb26d",
            ),
        )
        for path, expected in paths:
            with self.subTest(path=path):
                self.assertEqual(
                    canonical_json_sha256(
                        json.loads(path.read_text(encoding="utf-8"))
                    ),
                    expected,
                )

    def test_pass3_defines_no_pass2_serializer(self):
        for path in MODULES.glob("pass3*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    self.assertNotIn(
                        node.name,
                        {"serialize_pass2", "serialize_pass2_result"},
                        path.name,
                    )
        source = (MODULES / "pass3_inputs.py").read_text(encoding="utf-8")
        self.assertIn(
            "from .pass2_artifact import serialize as serialize_pass2",
            source,
        )
        self.assertNotIn("pickle", source)
        self.assertNotIn("hash(repr", source)


if __name__ == "__main__":
    unittest.main()
