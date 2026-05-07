"""Unit tests for derive.derive (and ordered_stages)."""

from __future__ import annotations

import unittest
from pathlib import Path

from lis_inspect.derive import derive, ordered_stages
from lis_inspect.models import PerfLog, PerfStage, RunArtifact
from lis_inspect.parse_json import load_artifact
from lis_inspect.parse_stderr import load_stderr_log


FIXTURES = Path(__file__).resolve().parent.parent / "test_fixtures"


class DerivedMetricsTest(unittest.TestCase):
    def test_qwen3_fixture_is_decode_heavy(self) -> None:
        artifact = load_artifact(FIXTURES / "qwen3_1p7b_run.json")
        perf = load_stderr_log(FIXTURES / "qwen3_1p7b_run.stderr")
        derived = derive(artifact, perf)
        self.assertEqual(derived.dominant_stage, "decode_steady_state")
        self.assertEqual(derived.run_profile, "decode-heavy")
        self.assertEqual(derived.artifact_completeness, "json_perf")
        self.assertTrue(derived.selected_equals_emitted)

    def test_completeness_json_only_when_perf_absent(self) -> None:
        artifact = load_artifact(FIXTURES / "qwen3_1p7b_run.json")
        derived = derive(artifact, PerfLog())
        self.assertEqual(derived.artifact_completeness, "json_only")
        self.assertIsNone(derived.dominant_stage)
        self.assertIsNone(derived.run_profile)

    def test_completeness_json_perf_per_token_when_per_token_present(self) -> None:
        artifact = load_artifact(FIXTURES / "qwen3_1p7b_run.json")
        perf = load_stderr_log(FIXTURES / "qwen3_1p7b_run.stderr")
        # Synthetically inject one per-token entry.
        from lis_inspect.models import PerfPerToken

        perf.per_token.append(PerfPerToken(step=0, ns=1, ms=1.0))
        derived = derive(artifact, perf)
        self.assertEqual(derived.artifact_completeness, "json_perf_per_token")

    def test_run_profile_load_heavy(self) -> None:
        perf = PerfLog()
        perf.stages["model_load"] = PerfStage(name="model_load", ns=10, ms=10000.0, tokens=0)
        perf.stages["prefill"] = PerfStage(name="prefill", ns=10, ms=10.0, tokens=2)
        artifact = load_artifact(FIXTURES / "qwen3_1p7b_run.json")
        derived = derive(artifact, perf)
        self.assertEqual(derived.dominant_stage, "model_load")
        self.assertEqual(derived.run_profile, "load-heavy")

    def test_run_profile_prefill_heavy(self) -> None:
        perf = PerfLog()
        perf.stages["prefill"] = PerfStage(name="prefill", ns=10, ms=900.0, tokens=128)
        perf.stages["decode_steady_state"] = PerfStage(
            name="decode_steady_state", ns=10, ms=300.0, tokens=4
        )
        artifact = load_artifact(FIXTURES / "qwen3_1p7b_run.json")
        derived = derive(artifact, perf)
        self.assertEqual(derived.dominant_stage, "prefill")
        self.assertEqual(derived.run_profile, "prefill-heavy")

    def test_selected_not_equal_emitted_when_digests_differ(self) -> None:
        raw = {
            "schema": "lis.execution_artifact/v1",
            "kind": "run_report",
            "manifest": {},
            "report": {
                "selected_token_digest": "fnv1a64:aaaa",
                "emitted_token_digest": "fnv1a64:bbbb",
            },
        }
        artifact = RunArtifact(raw=raw, schema=raw["schema"], kind=raw["kind"])
        derived = derive(artifact, PerfLog())
        self.assertFalse(derived.selected_equals_emitted)

    def test_selected_equals_emitted_unknown_when_digests_missing(self) -> None:
        raw = {
            "schema": "lis.execution_artifact/v1",
            "kind": "run_report",
            "manifest": {},
            "report": {},
        }
        artifact = RunArtifact(raw=raw, schema=raw["schema"], kind=raw["kind"])
        derived = derive(artifact, PerfLog())
        self.assertIsNone(derived.selected_equals_emitted)


class OrderedStagesTest(unittest.TestCase):
    def test_ordered_stages_uses_canonical_order(self) -> None:
        perf = load_stderr_log(FIXTURES / "qwen3_1p7b_run.stderr")
        ordered = [name for name, _ in ordered_stages(perf)]
        self.assertEqual(
            ordered,
            [
                "model_load",
                "tokenizer_load",
                "tokenizer_encode",
                "runtime_init",
                "prefill",
                "first_decode",
                "decode_steady_state",
            ],
        )

    def test_ordered_stages_skips_missing(self) -> None:
        perf = PerfLog()
        perf.stages["prefill"] = PerfStage(name="prefill", ns=1, ms=1.0, tokens=1)
        perf.stages["first_decode"] = PerfStage(name="first_decode", ns=1, ms=1.0, tokens=1)
        ordered = [name for name, _ in ordered_stages(perf)]
        self.assertEqual(ordered, ["prefill", "first_decode"])


if __name__ == "__main__":
    unittest.main()
