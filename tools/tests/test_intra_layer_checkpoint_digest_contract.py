#!/usr/bin/env python3
"""Literal-vector tests for the P4-1 contextual FP32 digest grammar."""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from lis_verify.pass4_contract import (
    DIGEST_ALGORITHM,
    DIGEST_BYTE_ORDER,
    DIGEST_CANONICALIZATION,
    DIGEST_DOMAIN_TAG,
    DIGEST_OBSERVED_DTYPE,
    DIGEST_VERSION,
    PASS3_DIGEST_VERSION,
    canonical_intra_layer_digest_stream,
    coordinate_from_mapping,
    intra_layer_digest_sha256,
    logical_fp32_bits_from_view,
)


ROOT = Path(__file__).resolve().parents[2]
VECTORS = (
    ROOT
    / "tools"
    / "test_fixtures"
    / "intra_layer_localization"
    / "intra_layer_checkpoint_digest_vectors.json"
)
FP32_BITS = re.compile(r"^[0-9a-f]{8}$")


def bits(values) -> tuple[int, ...]:
    result = []
    for value in values:
        if not isinstance(value, str) or not FP32_BITS.fullmatch(value):
            raise ValueError(
                "FP32 bit literal must be eight lowercase hexadecimal characters"
            )
        result.append(int(value, 16))
    return tuple(result)


def stream_for(semantic_input) -> bytes:
    coordinate = coordinate_from_mapping(semantic_input["coordinate"])
    shape = tuple(semantic_input["shape"])
    if "physical_fp32_bits_hex" in semantic_input:
        logical = logical_fp32_bits_from_view(
            shape,
            bits(semantic_input["physical_fp32_bits_hex"]),
            tuple(semantic_input["element_strides"]),
        )
    else:
        logical = bits(semantic_input["logical_fp32_bits_hex"])
    return canonical_intra_layer_digest_stream(
        coordinate=coordinate,
        precision_path=semantic_input["precision_path"],
        phase=semantic_input["phase"],
        shape=shape,
        logical_fp32_bits=logical,
        element_count=semantic_input.get("element_count"),
    )


class TestIntraLayerCheckpointDigestContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(VECTORS.read_text(encoding="utf-8"))
        cls.vectors = {
            vector["name"]: vector for vector in cls.fixture["vectors"]
        }

    def test_fixture_identity_matches_frozen_contract(self):
        self.assertEqual(
            self.fixture["contract"],
            {
                "algorithm": DIGEST_ALGORITHM,
                "version": DIGEST_VERSION,
                "domain_tag": DIGEST_DOMAIN_TAG,
                "observed_dtype": DIGEST_OBSERVED_DTYPE,
                "byte_order": DIGEST_BYTE_ORDER,
                "canonicalization": DIGEST_CANONICALIZATION,
            },
        )
        self.assertEqual(PASS3_DIGEST_VERSION, "lis.checkpoint.fp32le/v1")
        self.assertNotEqual(DIGEST_VERSION, PASS3_DIGEST_VERSION)

    def test_every_committed_canonical_stream_and_sha256_is_literal(self):
        self.assertEqual(len(self.vectors), 16)
        for name, vector in self.vectors.items():
            with self.subTest(vector=name):
                expected_stream = bytes.fromhex(
                    vector["canonical_stream_hex"]
                )
                actual_stream = stream_for(vector["semantic_input"])
                self.assertEqual(actual_stream, expected_stream)
                self.assertEqual(
                    intra_layer_digest_sha256(actual_stream),
                    vector["expected_sha256"],
                )
                self.assertRegex(
                    vector["canonical_stream_hex"], r"^[0-9a-f]+$"
                )
                self.assertRegex(
                    vector["expected_sha256"], r"^sha256:[0-9a-f]{64}$"
                )

    def test_signed_zero_is_preserved_and_separated(self):
        positive = self.vectors["positive_zero"]
        negative = self.vectors["negative_zero"]
        self.assertNotEqual(
            positive["canonical_stream_hex"], negative["canonical_stream_hex"]
        )
        self.assertNotEqual(
            positive["expected_sha256"], negative["expected_sha256"]
        )
        self.assertTrue(positive["canonical_stream_hex"].endswith("00000000"))
        self.assertTrue(negative["canonical_stream_hex"].endswith("00000080"))

    def test_all_nan_encodings_canonicalize_to_one_qnan(self):
        vector = self.vectors["canonical_nans"]
        stream = bytes.fromhex(vector["canonical_stream_hex"])
        self.assertTrue(stream.endswith(bytes.fromhex("0000c07f") * 4))

    def test_all_required_digest_domains_are_separated(self):
        base = self.vectors["finite_row_major_base"]["expected_sha256"]
        for name in (
            "shape_flat",
            "stage_and_role_changed",
            "layer_changed",
            "step_changed",
            "token_position_changed",
            "phase_changed",
            "precision_path_changed",
            "logical_order_changed",
        ):
            with self.subTest(vector=name):
                self.assertNotEqual(base, self.vectors[name]["expected_sha256"])
        self.assertNotEqual(
            self.vectors["string_framing_a_bc"]["expected_sha256"],
            self.vectors["string_framing_ab_c"]["expected_sha256"],
        )

    def test_strided_and_contiguous_logical_views_are_equivalent(self):
        self.assertEqual(
            self.vectors["finite_row_major_base"]["canonical_stream_hex"],
            self.vectors["strided_logical_equivalent"][
                "canonical_stream_hex"
            ],
        )
        self.assertEqual(
            self.vectors["finite_row_major_base"]["expected_sha256"],
            self.vectors["strided_logical_equivalent"]["expected_sha256"],
        )

    def test_domain_tag_is_distinct_from_the_existing_pass3_domain(self):
        stream = bytes.fromhex(
            self.vectors["finite_row_major_base"]["canonical_stream_hex"]
        )
        self.assertTrue(stream.startswith(DIGEST_DOMAIN_TAG.encode() + b"\0"))
        self.assertFalse(stream.startswith(b"LIS_CHECKPOINT_DIGEST\0"))

    def test_all_overflow_and_malformed_inputs_are_rejected(self):
        expected_names = {
            "rank_zero",
            "zero_dimension",
            "shape_overflow",
            "element_count_mismatch",
            "bool_shape_dimension",
            "unknown_stage",
            "stage_role_mismatch",
            "bool_coordinate_step",
            "coordinate_step_u64_overflow",
            "malformed_fp32_bits",
            "stride_rank_mismatch",
            "strided_view_exceeds_span",
            "bool_element_count",
        }
        rejection_vectors = self.fixture["rejection_vectors"]
        self.assertEqual(
            {vector["name"] for vector in rejection_vectors}, expected_names
        )
        for vector in rejection_vectors:
            with self.subTest(vector=vector["name"]):
                with self.assertRaisesRegex(
                    ValueError, vector["expected_error"]
                ):
                    stream_for(vector["semantic_input"])


if __name__ == "__main__":
    unittest.main()
