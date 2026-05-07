"""Derive tests: percentile, top-N slow steps, bounded token preview."""

from __future__ import annotations

import unittest
from pathlib import Path

from lis_inspect.derive import (
    bounded_token_preview,
    compute_percentile_stats,
    percentile,
    top_slow_steps,
    derive,
)
from lis_inspect.models import PerfLog, PerfPerToken
from lis_inspect.parse_json import load_artifact
from lis_inspect.parse_stderr import load_stderr_log


FIXTURES = Path(__file__).resolve().parent.parent / "test_fixtures"


class PercentileFunctionTest(unittest.TestCase):
    def test_single_value(self) -> None:
        self.assertEqual(percentile([42.0], 50.0), 42.0)
        self.assertEqual(percentile([42.0], 99.0), 42.0)

    def test_known_percentiles_on_decade(self) -> None:
        values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
        self.assertAlmostEqual(percentile(values, 50.0), 55.0)
        self.assertAlmostEqual(percentile(values, 90.0), 91.0)
        self.assertAlmostEqual(percentile(values, 95.0), 95.5)
        self.assertAlmostEqual(percentile(values, 99.0), 99.1)

    def test_rejects_empty(self) -> None:
        with self.assertRaises(ValueError):
            percentile([], 50.0)

    def test_rejects_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            percentile([1.0], -1.0)
        with self.assertRaises(ValueError):
            percentile([1.0], 101.0)

    def test_unsorted_input_handled(self) -> None:
        # Should not require pre-sorted input.
        self.assertAlmostEqual(percentile([30.0, 10.0, 20.0], 50.0), 20.0)


class ComputePercentileStatsTest(unittest.TestCase):
    def test_returns_none_for_empty(self) -> None:
        self.assertIsNone(compute_percentile_stats([]))

    def test_decade_values_match_known_percentiles(self) -> None:
        steps = [
            PerfPerToken(step=i, ns=int(value * 1e6), ms=value)
            for i, value in enumerate(
                [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
            )
        ]
        stats = compute_percentile_stats(steps)
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.count, 10)
        self.assertAlmostEqual(stats.min_ms, 10.0)
        self.assertAlmostEqual(stats.max_ms, 100.0)
        self.assertAlmostEqual(stats.mean_ms, 55.0)
        self.assertAlmostEqual(stats.p50_ms, 55.0)
        self.assertAlmostEqual(stats.p90_ms, 91.0)
        self.assertAlmostEqual(stats.p95_ms, 95.5)
        self.assertAlmostEqual(stats.p99_ms, 99.1)

    def test_qwen3_full_fixture_mean_matches_reference(self) -> None:
        log = load_stderr_log(FIXTURES / "qwen3_1p7b_run_full.stderr")
        stats = compute_percentile_stats(log.per_token)
        self.assertIsNotNone(stats)
        assert stats is not None
        self.assertEqual(stats.count, 10)
        self.assertAlmostEqual(stats.min_ms, 198.0, places=3)
        self.assertAlmostEqual(stats.max_ms, 213.0, places=3)
        self.assertAlmostEqual(stats.mean_ms, 206.0, places=3)
        self.assertAlmostEqual(stats.p50_ms, 206.8885, places=3)
        self.assertAlmostEqual(stats.p90_ms, 212.100, places=3)
        self.assertAlmostEqual(stats.p95_ms, 212.550, places=3)
        self.assertAlmostEqual(stats.p99_ms, 212.910, places=3)


class TopSlowStepsTest(unittest.TestCase):
    def test_returns_empty_when_no_data(self) -> None:
        self.assertEqual(top_slow_steps([]), [])

    def test_returns_top_n_descending(self) -> None:
        steps = [
            PerfPerToken(step=0, ns=1, ms=100.0),
            PerfPerToken(step=1, ns=1, ms=300.0),
            PerfPerToken(step=2, ns=1, ms=200.0),
            PerfPerToken(step=3, ns=1, ms=400.0),
        ]
        top = top_slow_steps(steps, n=2)
        self.assertEqual([s.step for s in top], [3, 1])

    def test_caps_at_data_length(self) -> None:
        steps = [PerfPerToken(step=0, ns=1, ms=10.0)]
        self.assertEqual(top_slow_steps(steps, n=10), steps)

    def test_ties_broken_by_step_ascending(self) -> None:
        steps = [
            PerfPerToken(step=2, ns=1, ms=50.0),
            PerfPerToken(step=0, ns=1, ms=50.0),
            PerfPerToken(step=1, ns=1, ms=50.0),
        ]
        top = top_slow_steps(steps, n=3)
        self.assertEqual([s.step for s in top], [0, 1, 2])

    def test_n_zero_returns_empty(self) -> None:
        steps = [PerfPerToken(step=0, ns=1, ms=10.0)]
        self.assertEqual(top_slow_steps(steps, n=0), [])

    def test_qwen3_full_fixture_top5(self) -> None:
        log = load_stderr_log(FIXTURES / "qwen3_1p7b_run_full.stderr")
        top = top_slow_steps(log.per_token, n=5)
        # Sorted desc by ms: 213.0(step5), 212.0(step7), 210.0(step3),
        # 208.0(step8), 207.0(step4).
        self.assertEqual([s.step for s in top], [5, 7, 3, 8, 4])


class BoundedTokenPreviewTest(unittest.TestCase):
    def test_returns_none_when_nothing_available(self) -> None:
        self.assertIsNone(bounded_token_preview(None, None, None))

    def test_count_and_digest_only_when_ids_absent(self) -> None:
        preview = bounded_token_preview(
            ids=None, count=42, digest="fnv1a64:beef"
        )
        self.assertIsNotNone(preview)
        assert preview is not None
        self.assertEqual(preview.count, 42)
        self.assertEqual(preview.digest, "fnv1a64:beef")
        self.assertFalse(preview.ids_present)
        self.assertEqual(preview.head_ids, [])
        self.assertEqual(preview.tail_ids, [])
        self.assertEqual(preview.omitted, 0)

    def test_short_list_fits_entirely_in_head(self) -> None:
        preview = bounded_token_preview(
            ids=[101, 202, 303],
            count=3,
            digest="fnv1a64:abc",
            head=8,
            tail=8,
        )
        self.assertIsNotNone(preview)
        assert preview is not None
        self.assertTrue(preview.ids_present)
        self.assertEqual(preview.head_ids, [101, 202, 303])
        self.assertEqual(preview.tail_ids, [])
        self.assertEqual(preview.omitted, 0)
        self.assertEqual(preview.count, 3)

    def test_long_list_splits_into_head_and_tail_with_omitted(self) -> None:
        ids = list(range(100))
        preview = bounded_token_preview(
            ids=ids, count=100, digest="fnv1a64:long", head=8, tail=8
        )
        self.assertIsNotNone(preview)
        assert preview is not None
        self.assertEqual(preview.head_ids, list(range(8)))
        self.assertEqual(preview.tail_ids, list(range(92, 100)))
        self.assertEqual(preview.omitted, 100 - 16)
        self.assertEqual(preview.count, 100)

    def test_count_falls_back_to_ids_length(self) -> None:
        preview = bounded_token_preview(ids=[1, 2, 3], count=None, digest=None)
        self.assertIsNotNone(preview)
        assert preview is not None
        self.assertEqual(preview.count, 3)

    def test_non_int_ids_filtered_out(self) -> None:
        preview = bounded_token_preview(
            ids=[1, "abc", 2, None, 3, True], count=None, digest=None
        )
        self.assertIsNotNone(preview)
        assert preview is not None
        # bools are explicitly rejected too.
        self.assertEqual(preview.head_ids, [1, 2, 3])

    def test_qwen3_full_fixture_preview_round_trip(self) -> None:
        artifact = load_artifact(FIXTURES / "qwen3_1p7b_run_full.json")
        derived = derive(artifact, PerfLog())
        self.assertIsNotNone(derived.selected_preview)
        self.assertIsNotNone(derived.emitted_preview)
        assert derived.selected_preview is not None
        assert derived.emitted_preview is not None
        self.assertTrue(derived.selected_preview.ids_present)
        self.assertEqual(derived.selected_preview.count, 10)
        self.assertEqual(derived.selected_preview.head_ids, [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010])
        self.assertEqual(derived.selected_preview.tail_ids, [])
        self.assertEqual(derived.selected_preview.omitted, 0)
        self.assertEqual(
            derived.emitted_preview.head_ids,
            derived.selected_preview.head_ids,
        )

    def test_preview_when_only_digest_and_count_in_artifact(self) -> None:
        artifact = load_artifact(FIXTURES / "qwen3_1p7b_run.json")
        derived = derive(artifact, PerfLog())
        self.assertIsNotNone(derived.selected_preview)
        assert derived.selected_preview is not None
        self.assertFalse(derived.selected_preview.ids_present)
        self.assertEqual(derived.selected_preview.count, 10)
        self.assertEqual(derived.selected_preview.digest, "fnv1a64:abcdef0123456789")


class DeriveIntegrationTest(unittest.TestCase):
    def test_full_fixture_populates_artifact_metric_fields(self) -> None:
        artifact = load_artifact(FIXTURES / "qwen3_1p7b_run_full.json")
        log = load_stderr_log(FIXTURES / "qwen3_1p7b_run_full.stderr")
        derived = derive(artifact, log)
        self.assertEqual(derived.artifact_completeness, "json_perf_per_token")
        self.assertIsNotNone(derived.per_token_stats)
        assert derived.per_token_stats is not None
        self.assertEqual(derived.per_token_stats.count, 10)
        self.assertEqual(len(derived.top_slow_steps), 5)
        self.assertEqual(derived.top_slow_steps[0].step, 5)


if __name__ == "__main__":
    unittest.main()
