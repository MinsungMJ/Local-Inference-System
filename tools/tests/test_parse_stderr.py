"""Unit tests for stderr parser (perf-stage, perf-summary, perf-per-token)."""

from __future__ import annotations

import unittest
from pathlib import Path

from lis_inspect.parse_stderr import load_stderr_log, parse_stderr_text


FIXTURE = Path(__file__).resolve().parent.parent / "test_fixtures" / "qwen3_1p7b_run.stderr"


class ParseStderrTextTest(unittest.TestCase):
    def test_parses_qwen3_fixture_stages_and_summary(self) -> None:
        log = load_stderr_log(FIXTURE)
        self.assertEqual(log.tag, "qwen3_1p7b")
        self.assertEqual(
            sorted(log.stages.keys()),
            sorted(
                [
                    "model_load",
                    "tokenizer_load",
                    "tokenizer_encode",
                    "runtime_init",
                    "prefill",
                    "first_decode",
                    "decode_steady_state",
                ]
            ),
        )
        decode = log.stages["decode_steady_state"]
        self.assertEqual(decode.ns, 1853223000)
        self.assertAlmostEqual(decode.ms, 1853.223, places=3)
        self.assertEqual(decode.tokens, 9)

        self.assertIsNotNone(log.summary)
        summary = log.summary
        assert summary is not None  # for type checker
        self.assertEqual(summary.threads, 1)
        self.assertEqual(summary.prompt_tokens, 2)
        self.assertEqual(summary.generated_tokens, 10)
        self.assertAlmostEqual(summary.ttft_ms, 626.268, places=3)
        self.assertAlmostEqual(summary.itl_ms, 205.914, places=3)
        self.assertAlmostEqual(summary.tps_steady, 4.856, places=3)
        self.assertAlmostEqual(summary.tps_end_to_end, 4.041, places=3)
        self.assertEqual(log.warnings, [])
        self.assertFalse(log.has_per_token)

    def test_ignores_unrelated_lines(self) -> None:
        text = (
            "warning: something else\n"
            "lis: simd backend=avx2 selected=avx2 reason=runtime\n"
            "lis: perf-stage tag=t name=prefill ns=10 ms=0.010 tokens=4\n"
        )
        log = parse_stderr_text(text)
        self.assertEqual(list(log.stages.keys()), ["prefill"])
        self.assertEqual(log.warnings, [])

    def test_collects_warning_for_malformed_perf_stage(self) -> None:
        text = "lis: perf-stage tag=t name=prefill ns=oops ms=0.5 tokens=1\n"
        log = parse_stderr_text(text)
        self.assertEqual(log.stages, {})
        self.assertEqual(len(log.warnings), 1)
        self.assertIn("malformed perf-stage", log.warnings[0])

    def test_collects_warning_for_malformed_perf_summary(self) -> None:
        # Missing tps_end_to_end.
        text = (
            "lis: perf-summary tag=t threads=1 prompt_tokens=1 generated_tokens=1 "
            "ttft_ms=1.0 itl_ms=1.0 tps_steady=1.0\n"
        )
        log = parse_stderr_text(text)
        self.assertIsNone(log.summary)
        self.assertEqual(len(log.warnings), 1)
        self.assertIn("malformed perf-summary", log.warnings[0])

    def test_collects_warning_for_malformed_perf_per_token(self) -> None:
        text = "lis: perf-per-token tag=t step=zero ns=1 ms=1.0\n"
        log = parse_stderr_text(text)
        self.assertEqual(log.per_token, [])
        self.assertEqual(len(log.warnings), 1)
        self.assertIn("malformed perf-per-token", log.warnings[0])

    def test_per_token_lines_are_collected(self) -> None:
        text = (
            "lis: perf-per-token tag=t step=0 ns=200000000 ms=200.0\n"
            "lis: perf-per-token tag=t step=1 ns=210000000 ms=210.0\n"
        )
        log = parse_stderr_text(text)
        self.assertEqual(len(log.per_token), 2)
        self.assertEqual(log.per_token[0].step, 0)
        self.assertAlmostEqual(log.per_token[1].ms, 210.0)

    def test_ignores_kv_cache_diagnostic_lines(self) -> None:
        baseline = (
            "lis: perf-stage tag=t name=prefill ns=10 ms=0.010 tokens=4\n"
            "lis: perf-summary tag=t threads=1 prompt_tokens=4 generated_tokens=1 "
            "ttft_ms=1.0 itl_ms=0.0 tps_steady=0.0 tps_end_to_end=1.0\n"
            "lis: perf-per-token tag=t step=0 ns=20 ms=0.020\n"
        )
        with_kv = (
            "lis: perf-stage tag=t name=prefill ns=10 ms=0.010 tokens=4\n"
            "lis: kv-cache: scope=run_local policy=eviction_free,monotonic "
            "dtype=f32 max_tokens=4 used_tokens=3 bytes_per_token=64 "
            "allocated_bytes=256 used_bytes=192\n"
            "lis: perf-summary tag=t threads=1 prompt_tokens=4 generated_tokens=1 "
            "ttft_ms=1.0 itl_ms=0.0 tps_steady=0.0 tps_end_to_end=1.0\n"
            "lis: perf-per-token tag=t step=0 ns=20 ms=0.020\n"
        )

        expected = parse_stderr_text(baseline)
        actual = parse_stderr_text(with_kv)
        self.assertEqual(actual.tag, expected.tag)
        self.assertEqual(actual.stages, expected.stages)
        self.assertEqual(actual.summary, expected.summary)
        self.assertEqual(actual.per_token, expected.per_token)
        self.assertEqual(actual.warnings, [])

    def test_missing_file_returns_warning_not_exception(self) -> None:
        log = load_stderr_log(Path("/no/such/file.stderr"))
        self.assertFalse(log.has_stages)
        self.assertFalse(log.has_summary)
        self.assertEqual(len(log.warnings), 1)
        self.assertIn("not found", log.warnings[0])


if __name__ == "__main__":
    unittest.main()
