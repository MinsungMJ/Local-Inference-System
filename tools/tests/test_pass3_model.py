#!/usr/bin/env python3
"""Construction and coherence tests for the Pass 3 model."""

from __future__ import annotations

import unittest

from lis_verify.pass3_model import (
    CheckpointCoordinate,
    Pass3DownstreamDisposition,
    Pass3Result,
    Pass3Status,
    SuspectInterval,
)


def coordinate(layer: int) -> CheckpointCoordinate:
    return CheckpointCoordinate(3, layer, "layer_output", 0, 0, 0, layer)


class TestPass3Model(unittest.TestCase):
    def test_mismatch_requires_interval(self):
        with self.assertRaises(ValueError):
            Pass3Result(
                Pass3Status.OBSERVABLE_MISMATCH_FOUND,
                Pass3DownstreamDisposition.SUSPECT_INTERVAL_AVAILABLE,
            )

    def test_dense_interval_is_coherent(self):
        prior = coordinate(7)
        mismatch = coordinate(8)
        interval = SuspectInterval(
            "observed_checkpoint",
            prior,
            mismatch,
            True,
            True,
            (),
            "(7, 8]",
        )
        result = Pass3Result(
            Pass3Status.OBSERVABLE_MISMATCH_FOUND,
            Pass3DownstreamDisposition.SUSPECT_INTERVAL_AVAILABLE,
            last_observed_equivalent_coordinate=prior,
            first_observed_mismatch_coordinate=mismatch,
            earliest_observable_suspect_layer=8,
            suspect_interval=interval,
        )
        self.assertEqual(result.suspect_interval.notation, "(7, 8]")

    def test_first_capture_uses_entry_interval(self):
        mismatch = coordinate(4)
        interval = SuspectInterval(
            "runtime_entry", None, mismatch, False, True, (0, 1, 2, 3),
            "[entry, 4]",
        )
        self.assertFalse(interval.start_exclusive)

    def test_no_mismatch_prohibits_interval(self):
        with self.assertRaises(ValueError):
            Pass3Result(
                Pass3Status.NO_MISMATCH_IN_CAPTURED_COVERAGE,
                Pass3DownstreamDisposition.EXPLORATORY_LOCALIZATION_ONLY,
                first_observed_mismatch_coordinate=coordinate(2),
            )

    def test_blocked_result_cannot_use_success_disposition(self):
        with self.assertRaises(ValueError):
            Pass3Result(
                Pass3Status.COMPARISON_BLOCKED_BY_PASS2,
                Pass3DownstreamDisposition.EXPLORATORY_LOCALIZATION_ONLY,
            )

    def test_model_has_no_confirmation_attributes(self):
        fields = Pass3Result.__dataclass_fields__
        for prohibited in (
            "confirmed_first_divergent_layer",
            "confirmed_divergence_at_checkpoint",
            "confirmed_first_divergence",
            "pass4_ready",
            "pass5_ready",
        ):
            self.assertNotIn(prohibited, fields)


if __name__ == "__main__":
    unittest.main()
