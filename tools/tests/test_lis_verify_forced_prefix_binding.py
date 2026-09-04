import copy
from pathlib import Path
import unittest

from lis_verify import (
    CanonicalRunReport,
    ForcedPrefixBindingError,
    build_forced_prefix_metadata,
    forced_prefix_binding_bytes,
    run_prefix_policy_reproduction,
    validate_forced_prefix_reproduction,
)
from lis_verify.pass2_model import Pass2Status
from lis_verify.product_contract import canonical_json_bytes
from tools.tests.test_pass2_prefix_policy_reproduction import ready_inputs


def changed(source, mutate):
    raw = source.materialize()
    mutate(raw)
    return CanonicalRunReport.from_object(raw)


def forced_reproduction(pass1, original, template, role):
    metadata = build_forced_prefix_metadata(pass1, original, role=role)

    def mutate(raw):
        raw["forced_prefix"] = metadata
        raw["artifact_set_id"] = "aset1:" + (
            "1" * 32 if role == "reference" else "2" * 32
        )
        original_raw = original.materialize()
        raw["manifest"]["backend"] = copy.deepcopy(
            original_raw["manifest"]["backend"]
        )
        raw["report"]["selected_token_count"] = 1
        raw["report"]["selected_token_ids"] = [
            original_raw["report"]["selected_token_ids"][-1]
        ]

    return changed(template, mutate)


class ForcedPrefixBindingTestCase(unittest.TestCase):
    def setUp(self):
        self.pass1, self.reference, self.candidate = ready_inputs()
        root = Path(__file__).resolve().parents[2]
        fixtures = root / "tools" / "test_fixtures" / "prefix_policy_reproduction"
        self.reference_reproduction = forced_reproduction(
            self.pass1,
            self.reference,
            CanonicalRunReport.load(
                fixtures / "reference_reproduction_verified.json"
            ),
            "reference",
        )
        self.candidate_reproduction = forced_reproduction(
            self.pass1,
            self.candidate,
            CanonicalRunReport.load(
                fixtures / "candidate_reproduction_verified.json"
            ),
            "candidate",
        )

    def test_binding_bytes_are_canonical_and_source_bound(self):
        raw = build_forced_prefix_metadata(
            self.pass1, self.reference, role="reference"
        )
        self.assertEqual(
            forced_prefix_binding_bytes(
                self.pass1, self.reference, role="reference"
            ),
            canonical_json_bytes(raw),
        )
        self.assertEqual(
            raw["source_original_run_report_sha256"],
            self.reference.identity.run_report_sha256,
        )
        self.assertEqual(raw["token_count"], 17)
        self.assertEqual(raw["runtime_checkpoint_step"], 18)

    def test_valid_reproduction_retains_only_target_token(self):
        validate_forced_prefix_reproduction(
            self.pass1,
            self.reference,
            self.reference_reproduction,
            role="reference",
        )
        raw = self.reference_reproduction.materialize()
        self.assertEqual(len(raw["report"]["selected_token_ids"]), 1)
        self.assertNotIn("full_forced_prefix_token_ids", raw)

    def test_pass2_accepts_digest_bound_prefix_without_raw_array(self):
        result = run_prefix_policy_reproduction(
            self.pass1,
            self.reference,
            self.candidate,
            reference_reproduction=self.reference_reproduction,
            candidate_reproduction=self.candidate_reproduction,
        )
        self.assertEqual(result.status, Pass2Status.REPRODUCTION_VERIFIED)
        self.assertEqual(
            set(result.prefix_reproduction.verified_sides),
            {
                "reference_original",
                "candidate_original",
                "reference_reproduction",
                "candidate_reproduction",
            },
        )

    def test_source_digest_count_and_wrong_side_tampering_are_rejected(self):
        mutations = (
            ("source_pass0_artifact_sha256", "sha256:" + "0" * 64),
            ("token_ids_sha256", "sha256:" + "0" * 64),
            ("token_count", 16),
            (
                "source_original_run_report_sha256",
                self.candidate.identity.run_report_sha256,
            ),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                broken = changed(
                    self.reference_reproduction,
                    lambda raw, field=field, value=value: raw[
                        "forced_prefix"
                    ].__setitem__(field, value),
                )
                with self.assertRaises(ForcedPrefixBindingError):
                    validate_forced_prefix_reproduction(
                        self.pass1,
                        self.reference,
                        broken,
                        role="reference",
                    )

    def test_binary_backend_and_context_drift_are_rejected(self):
        def binary(raw):
            raw["manifest"]["binary"]["fingerprint"]["hex"] = "0" * 16

        def backend(raw):
            raw["manifest"]["backend"]["name"] = "wrong"

        def context(raw):
            raw["manifest"]["runtime"]["configured_context"] = 64

        for label, mutation in (
            ("binary", binary),
            ("backend", backend),
            ("context", context),
        ):
            with self.subTest(label=label):
                broken = changed(self.reference_reproduction, mutation)
                with self.assertRaises(ForcedPrefixBindingError):
                    validate_forced_prefix_reproduction(
                        self.pass1,
                        self.reference,
                        broken,
                        role="reference",
                    )

    def test_full_prefix_retention_is_rejected(self):
        prefix = list(self.pass1.prefix_for_reproduction.exact_token_ids)

        def retain(raw):
            raw["report"]["selected_token_count"] = len(prefix) + 1
            raw["report"]["selected_token_ids"] = prefix + [501]

        broken = changed(self.reference_reproduction, retain)
        with self.assertRaisesRegex(ForcedPrefixBindingError, "only the target"):
            validate_forced_prefix_reproduction(
                self.pass1, self.reference, broken, role="reference"
            )

    def test_target_selection_must_reproduce_the_original_role(self):
        broken = changed(
            self.reference_reproduction,
            lambda raw: raw["report"]["selected_token_ids"].__setitem__(
                0, 999
            ),
        )
        with self.assertRaisesRegex(
            ForcedPrefixBindingError, "target selection"
        ):
            validate_forced_prefix_reproduction(
                self.pass1, self.reference, broken, role="reference"
            )


if __name__ == "__main__":
    unittest.main()
