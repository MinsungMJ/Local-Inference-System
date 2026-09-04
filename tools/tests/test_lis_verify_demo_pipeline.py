from __future__ import annotations

from unittest import mock
import unittest

from lis_verify import demo_pipeline
from lis_verify.demo_pipeline import DemoPipelineError, build_demo_report, semantic_digest
from lis_verify.pass4 import run_coverage_scoped_intra_layer_localization


class TestDemoPipeline(unittest.TestCase):
    def test_runs_real_pass_zero_through_four_and_reports_expected_interval(self):
        patches = {
            name: mock.patch.object(
                demo_pipeline,
                name,
                wraps=getattr(demo_pipeline, name),
            )
            for name in (
                "run_calibration_preflight",
                "run_token_localization",
                "run_prefix_policy_reproduction",
                "run_coverage_scoped_layer_localization",
                "run_coverage_scoped_intra_layer_localization",
            )
        }
        spies = {name: patch.start() for name, patch in patches.items()}
        self.addCleanup(lambda: [patch.stop() for patch in patches.values()])

        report = build_demo_report().to_dict()

        self.assertEqual(spies["run_calibration_preflight"].call_count, 2)
        self.assertEqual(spies["run_token_localization"].call_count, 2)
        self.assertEqual(spies["run_prefix_policy_reproduction"].call_count, 2)
        self.assertEqual(
            spies["run_coverage_scoped_layer_localization"].call_count, 2
        )
        self.assertEqual(
            spies["run_coverage_scoped_intra_layer_localization"].call_count,
            1,
        )
        self.assertEqual(report["verdict"], "REGRESSION")
        self.assertEqual(
            report["token_comparison"]["first_mismatch"]["generated_token_step"],
            17,
        )
        self.assertEqual(report["localization"]["layer_suspect_interval"], "(4, 8]")
        self.assertEqual(
            report["localization"]["intra_layer_suspect_interval"],
            "(rope_key_output, attention_scores]",
        )
        self.assertTrue(
            all(value is False for value in report["evidence"]["nonclaims"].values())
        )
        self.assertEqual(
            report["identities"]["reference"]["source_sha256"],
            report["identities"]["candidate"]["source_sha256"],
        )
        self.assertEqual(
            report["identities"]["reference"]["model_sha256"],
            report["identities"]["candidate"]["model_sha256"],
        )
        self.assertNotEqual(
            report["identities"]["reference"]["backend_sha256"],
            report["identities"]["candidate"]["backend_sha256"],
        )

    def test_two_fresh_pipelines_have_identical_semantics(self):
        left = build_demo_report()
        right = build_demo_report()
        self.assertEqual(left.to_json_bytes(), right.to_json_bytes())
        self.assertEqual(semantic_digest(left), semantic_digest(right))

    def test_missing_pass3_parent_cannot_become_regression(self):
        def missing_parent(*args, **kwargs):
            values = list(args)
            values[1] = None
            return run_coverage_scoped_intra_layer_localization(*values, **kwargs)

        with mock.patch.object(
            demo_pipeline,
            "run_coverage_scoped_intra_layer_localization",
            side_effect=missing_parent,
        ):
            with self.assertRaises((DemoPipelineError, TypeError, ValueError)):
                build_demo_report()

    def test_stale_pass3_parent_cannot_become_regression(self):
        def stale_parent(*args, **kwargs):
            values = list(args)
            values[3] = values[1]
            return run_coverage_scoped_intra_layer_localization(*values, **kwargs)

        with mock.patch.object(
            demo_pipeline,
            "run_coverage_scoped_intra_layer_localization",
            side_effect=stale_parent,
        ):
            with self.assertRaises(DemoPipelineError):
                build_demo_report()


if __name__ == "__main__":
    unittest.main()
