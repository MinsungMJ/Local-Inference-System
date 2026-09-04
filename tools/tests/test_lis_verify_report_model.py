import copy
import json
import unittest

from lis_verify.product_contract import CustomerVerdict, StageState
from lis_verify.report_model import VerificationReport
from tools.tests.lis_verify_product_spine_test_support import load_example


class TestVerificationReport(unittest.TestCase):
    def test_all_golden_reports_have_typed_views(self):
        states = set()
        verdicts = set()
        for name in ("pass", "regression", "inconclusive", "unsupported", "harness_error"):
            report = VerificationReport.from_dict(load_example(name))
            verdicts.add(report.verdict)
            states.update(stage.state for stage in report.stages)
            self.assertEqual(report.to_json_bytes()[-1:], b"\n")
            self.assertEqual(report.attempt_id, report.to_dict()["attempt"]["id"])
        self.assertEqual(verdicts, set(CustomerVerdict))
        self.assertEqual(states, set(StageState))

    def test_value_is_immutable_through_copy_boundary(self):
        report = VerificationReport.from_dict(load_example("pass"))
        changed = report.to_dict()
        changed["verdict"] = "HARNESS_ERROR"
        self.assertEqual(report.verdict, CustomerVerdict.PASS)

    def test_canonical_bytes_round_trip(self):
        report = VerificationReport.from_dict(load_example("regression"))
        loaded = VerificationReport.from_json_bytes(report.to_json_bytes())
        self.assertEqual(loaded, report)

    def test_noncanonical_json_is_rejected(self):
        pretty = (json.dumps(load_example("pass"), indent=2) + "\n").encode()
        with self.assertRaisesRegex(ValueError, "canonical"):
            VerificationReport.from_json_bytes(pretty)

    def test_duplicate_key_and_non_utf8_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            VerificationReport.from_json_bytes(b'{"schema":1,"schema":2}\n')
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            VerificationReport.from_json_bytes(b"\xff")

    def test_unknown_field_still_fails_closed(self):
        raw = copy.deepcopy(load_example("pass"))
        raw["extra"] = True
        with self.assertRaises(ValueError):
            VerificationReport.from_dict(raw)


if __name__ == "__main__":
    unittest.main()
