import unittest

from lis_verify.aggregation import (
    apply_policy,
    policy_result_for,
    resolve_pass0_block_reason,
    resolve_pass_status,
)
from lis_verify.product_contract import (
    CustomerVerdict,
    PASS0_BLOCK_REASON_VERDICTS,
    PASS_STATUS_ACTIONS,
)
from tools.tests.lis_verify_product_spine_test_support import load_example


class TestAggregation(unittest.TestCase):
    def test_every_frozen_pass_status_resolves_without_default(self):
        count = 0
        for pass_name, statuses in PASS_STATUS_ACTIONS.items():
            for status, (action, target) in statuses.items():
                result = resolve_pass_status(pass_name, status)
                self.assertEqual(result.action.value, action)
                self.assertEqual(result.target, target)
                count += 1
        self.assertEqual(count, 46)

    def test_unknown_status_and_block_reason_fail_closed(self):
        with self.assertRaises(ValueError):
            resolve_pass_status("pass1", "future_status")
        with self.assertRaises(ValueError):
            resolve_pass0_block_reason("future_reason")

    def test_all_pass0_block_reasons_resolve(self):
        for reason, verdict in PASS0_BLOCK_REASON_VERDICTS.items():
            self.assertEqual(resolve_pass0_block_reason(reason), verdict)

    def test_strict_policy_never_rewrites_semantic_verdict(self):
        for name in ("pass", "regression", "inconclusive", "unsupported", "harness_error"):
            raw = load_example(name)
            strict = apply_policy(raw, require_supported=True)
            self.assertEqual(strict["verdict"], raw["verdict"])
            expected = 6 if raw["verdict"] == "UNSUPPORTED" else raw["policy_result"]["exit_code"]
            self.assertEqual(strict["policy_result"]["exit_code"], expected)

    def test_policy_result_is_separate(self):
        result = policy_result_for(
            CustomerVerdict.UNSUPPORTED,
            require_supported=True,
            reason="unsupported_scope",
        )
        self.assertEqual(result["exit_code"], 6)
        self.assertFalse(result["satisfied"])


if __name__ == "__main__":
    unittest.main()
