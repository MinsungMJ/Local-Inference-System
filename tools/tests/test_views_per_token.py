"""Render tests for the Per-Token tab."""

from __future__ import annotations

import unittest
from pathlib import Path

from lis_inspect.cli import build_session
from lis_inspect.views.per_token import render_per_token, _trend_bar
from lis_inspect.models import PerfPerToken


FIXTURES = Path(__file__).resolve().parent.parent / "test_fixtures"


class PerTokenRenderTest(unittest.TestCase):
    def test_degraded_when_no_per_token_data(self) -> None:
        session = build_session(
            FIXTURES / "qwen3_1p7b_run.json",
            FIXTURES / "qwen3_1p7b_run.stderr",
        )
        rendered = render_per_token(session)
        self.assertIn("Per-token data unavailable.", rendered)
        self.assertNotIn("Distribution", rendered)
        self.assertNotIn("Top slow steps", rendered)

    def test_degraded_when_stderr_missing_entirely(self) -> None:
        session = build_session(FIXTURES / "qwen3_1p7b_run.json", None)
        rendered = render_per_token(session)
        self.assertIn("Per-token data unavailable.", rendered)

    def test_full_fixture_renders_distribution_and_top_slow(self) -> None:
        session = build_session(
            FIXTURES / "qwen3_1p7b_run_full.json",
            FIXTURES / "qwen3_1p7b_run_full.stderr",
        )
        rendered = render_per_token(session)
        # Distribution section
        self.assertIn("Distribution", rendered)
        self.assertIn("count", rendered)
        self.assertIn("p50", rendered)
        self.assertIn("p90", rendered)
        self.assertIn("p95", rendered)
        self.assertIn("p99", rendered)
        # Mean and min/max numbers.
        self.assertIn("206.000", rendered)  # mean
        self.assertIn("198.000", rendered)  # min
        self.assertIn("213.000", rendered)  # max
        # Trend section
        self.assertIn("Per-step trend", rendered)
        self.assertIn("steps 0..9", rendered)
        # Top slow section — step 5 should lead.
        self.assertIn("Top slow steps", rendered)
        self.assertIn("step=5", rendered)


class TrendBarTest(unittest.TestCase):
    def test_empty_returns_empty(self) -> None:
        self.assertEqual(_trend_bar([]), "")

    def test_uniform_uses_middle_ramp_char(self) -> None:
        steps = [PerfPerToken(step=i, ns=1, ms=10.0) for i in range(5)]
        bar = _trend_bar(steps)
        self.assertEqual(len(bar), 5)
        self.assertEqual(len(set(bar)), 1)  # all identical chars

    def test_gradient_maps_min_to_first_max_to_last(self) -> None:
        steps = [
            PerfPerToken(step=0, ns=1, ms=1.0),
            PerfPerToken(step=1, ns=1, ms=10.0),
        ]
        bar = _trend_bar(steps)
        self.assertEqual(bar[0], "▁")
        self.assertEqual(bar[1], "█")


if __name__ == "__main__":
    unittest.main()
