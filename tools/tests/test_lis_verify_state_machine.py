import unittest

from lis_verify.product_contract import CANONICAL_STAGES, StageState
from lis_verify.state_machine import StageMachine, StageTransitionError
from tools.tests.lis_verify_product_spine_test_support import load_example


class TestStageMachine(unittest.TestCase):
    def _drive(self, raw):
        machine = StageMachine()
        for stage in raw["stages"]:
            machine.start(stage["name"])
            machine.finish(
                StageState(stage["state"]),
                result_ref=stage["result_ref"],
                evidence_tier=stage["evidence_tier"],
                failure_class=stage["failure_class"],
                reason=stage["reason"],
                blocker=stage["blocker"],
            )
        return machine

    def test_all_golden_state_paths_reach_terminal(self):
        states = set()
        for name in ("pass", "regression", "inconclusive", "unsupported", "harness_error"):
            result = self._drive(load_example(name)).finalize()
            self.assertEqual(tuple(item.name for item in result), CANONICAL_STAGES)
            states.update(item.state for item in result)
        self.assertEqual(states, set(StageState))

    def test_reorder_double_start_and_incomplete_finalize_fail(self):
        machine = StageMachine()
        with self.assertRaises(StageTransitionError):
            machine.start("pass1_token_localization")
        machine.start("preflight")
        with self.assertRaises(StageTransitionError):
            machine.start("preflight")
        with self.assertRaises(StageTransitionError):
            machine.finalize()

    def test_executed_stage_needs_evidence(self):
        machine = StageMachine()
        machine.start("preflight")
        with self.assertRaises(StageTransitionError):
            machine.finish(StageState.EXECUTED)

    def test_executed_dependency_cannot_follow_failure(self):
        machine = StageMachine()
        machine.start("preflight")
        machine.finish(
            StageState.FAILED,
            failure_class="fixture_failure",
            reason="fixture failed",
        )
        machine.start("reference_original_execution")
        with self.assertRaises(StageTransitionError):
            machine.finish(
                StageState.EXECUTED,
                result_ref="sha256:" + "a" * 64,
                evidence_tier="fixture",
            )

    def test_blocker_must_be_a_prior_stage(self):
        machine = StageMachine()
        machine.start("preflight")
        with self.assertRaisesRegex(StageTransitionError, "prior blocker"):
            machine.finish(
                StageState.BLOCKED,
                reason="future dependency",
                blocker="cleanup",
            )


if __name__ == "__main__":
    unittest.main()
