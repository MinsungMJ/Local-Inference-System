from __future__ import annotations

import copy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from lis_verify.product_contract import canonical_json_bytes
from lis_verify.usability import (
    UsabilityValidationError,
    build_aggregate_report,
    load_contract,
    load_private_dataset,
    main,
    parse_aggregate_bytes,
    parse_session_bytes,
    validate_session_record,
)


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = (
    ROOT
    / "tools"
    / "test_fixtures"
    / "lis_verify_usability"
    / "session_examples_v1.json"
)


def _examples() -> dict[str, dict]:
    data = EXAMPLES.read_bytes()
    raw = json.loads(data)
    assert data == canonical_json_bytes(raw)
    return {item["name"]: item["record"] for item in raw["examples"]}


def _complete_record(index: int) -> dict:
    record = copy.deepcopy(_examples()["complete_supported"])
    hex_id = f"{index:032x}"
    record["record_id"] = f"usr1:{hex_id}"
    record["participant_id"] = f"p1:{hex_id}"
    record["enrollment_order"] = index
    record["eligibility"]["external_to_lis_workflow"] = index <= 3
    record["eligibility"]["monthly_problem"] = index <= 4
    workflow = ((index - 1) % 4) + 1
    record["ci_followup"]["workflow_id"] = f"wf1:{workflow:032x}"
    record["ci_followup"]["followup_status"] = (
        "retained" if index <= 5 else "stopped"
    )
    record["identities"]["demo_report_sha256"] = "sha256:" + f"{index:064x}"[-64:]
    record["identities"]["backend_report_sha256"] = (
        "sha256:" + f"{index + 100:064x}"[-64:]
    )
    validate_session_record(record)
    return record


def _write_dataset(root: Path, records: list[dict]) -> None:
    root.mkdir(mode=0o700)
    root.chmod(0o700)
    for record in records:
        suffix = record["record_id"].split(":", 1)[1]
        path = root / f"record-{suffix}.json"
        path.write_bytes(canonical_json_bytes(record))
        path.chmod(0o600)


class TestUsabilityContractAndExamples(unittest.TestCase):
    def test_packaged_schema_is_canonical_closed_and_frozen(self):
        contract = load_contract()
        self.assertEqual(
            contract.identity_sha256,
            "sha256:28022ba867a7ef2b1db1d281dcfd03894ad61965a2d8812f23cbd49387f2da96",
        )
        self.assertFalse(contract.raw["additionalProperties"])
        self.assertEqual(
            set(contract.raw["required"]), set(contract.raw["properties"])
        )

    def test_all_public_examples_are_canonical_and_valid(self):
        examples = _examples()
        self.assertEqual(
            set(examples),
            {
                "complete_invalid_environment",
                "complete_supported",
                "complete_unsupported",
                "handled_error",
                "incomplete",
                "withdrawn",
            },
        )
        for name, record in examples.items():
            with self.subTest(name=name):
                self.assertEqual(
                    parse_session_bytes(canonical_json_bytes(record)), record
                )

    def test_unknown_free_text_and_identifier_leakage_fail_closed(self):
        record = _complete_record(1)
        record["raw_prompt"] = "private prompt"
        with self.assertRaisesRegex(UsabilityValidationError, "unknown"):
            validate_session_record(record)

        record = _complete_record(1)
        record["participant_id"] = "Jane Doe"
        with self.assertRaisesRegex(UsabilityValidationError, "pseudonymous"):
            validate_session_record(record)

        record = _complete_record(1)
        record["ci_followup"]["workflow_id"] = "https://example.invalid/private"
        with self.assertRaisesRegex(UsabilityValidationError, "pseudonymous"):
            validate_session_record(record)

    def test_duplicate_keys_noncanonical_and_impossible_timing_fail(self):
        record = _complete_record(1)
        pretty = (json.dumps(record, indent=2) + "\n").encode()
        with self.assertRaisesRegex(UsabilityValidationError, "canonical"):
            parse_session_bytes(pretty)

        duplicate = b'{"schema":"lis.usability_session/v1","schema":"x"}\n'
        with self.assertRaisesRegex(UsabilityValidationError, "duplicate"):
            parse_session_bytes(duplicate)

        record = _complete_record(1)
        record["setup_timing"]["hands_on_seconds"] += 1
        with self.assertRaisesRegex(UsabilityValidationError, "arithmetic"):
            validate_session_record(record)

    def test_cross_field_consent_report_and_followup_rules_fail(self):
        cases = []
        consent = _complete_record(1)
        consent["consent_state"] = "withdrawn"
        cases.append(consent)
        report = _complete_record(1)
        report["identities"]["backend_report_sha256"] = None
        cases.append(report)
        followup = _complete_record(1)
        followup["ci_followup"]["workflow_id"] = None
        cases.append(followup)
        paired = _complete_record(1)
        paired["investigation"]["prior_seconds"] = None
        cases.append(paired)
        for index, record in enumerate(cases):
            with self.subTest(case=index):
                with self.assertRaises(UsabilityValidationError):
                    validate_session_record(record)

        withdrawn = copy.deepcopy(_examples()["withdrawn"])
        withdrawn["demo"]["attempted"] = True
        withdrawn["demo"]["verdict"] = "REGRESSION"
        with self.assertRaisesRegex(UsabilityValidationError, "withdrawn session"):
            validate_session_record(withdrawn)

    def test_deeply_nested_input_is_contained(self):
        nested = "[" * 2000 + "]" * 2000
        with self.assertRaises(UsabilityValidationError):
            parse_session_bytes((nested + "\n").encode())


class TestPrivateDataset(unittest.TestCase):
    def test_private_dataset_loads_and_has_stable_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "records"
            records = [_complete_record(1), _complete_record(2)]
            _write_dataset(root, records)
            first = load_private_dataset(root)
            second = load_private_dataset(root)
            self.assertEqual(first.identity_sha256, second.identity_sha256)
            self.assertEqual(len(first.records), 2)
            self.assertGreater(first.total_size_bytes, 0)

    def test_permissions_symlink_and_unexpected_entry_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "records"
            _write_dataset(root, [_complete_record(1)])
            root.chmod(0o755)
            with self.assertRaisesRegex(UsabilityValidationError, "0700"):
                load_private_dataset(root)
            root.chmod(0o700)
            path = next(root.iterdir())
            path.chmod(0o644)
            with self.assertRaisesRegex(UsabilityValidationError, "0600"):
                load_private_dataset(root)

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "records"
            _write_dataset(root, [_complete_record(1)])
            link = base / "link"
            link.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(UsabilityValidationError, "symlink"):
                load_private_dataset(link)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "records"
            _write_dataset(root, [_complete_record(1)])
            extra = root / "notes.txt"
            extra.write_text("free form")
            extra.chmod(0o600)
            with self.assertRaisesRegex(UsabilityValidationError, "unexpected"):
                load_private_dataset(root)

        if hasattr(os, "mkfifo"):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "records"
                _write_dataset(root, [])
                fifo = root / ("record-" + "f" * 32 + ".json")
                os.mkfifo(fifo, 0o600)
                with self.assertRaisesRegex(UsabilityValidationError, "regular"):
                    load_private_dataset(root)

    def test_duplicate_participant_order_and_filename_binding_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "records"
            first = _complete_record(1)
            second = _complete_record(2)
            second["participant_id"] = first["participant_id"]
            _write_dataset(root, [first, second])
            with self.assertRaisesRegex(UsabilityValidationError, "duplicate participant"):
                load_private_dataset(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "records"
            first = _complete_record(1)
            second = _complete_record(2)
            second["enrollment_order"] = first["enrollment_order"]
            _write_dataset(root, [first, second])
            with self.assertRaisesRegex(UsabilityValidationError, "duplicate enrollment"):
                load_private_dataset(root)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "records"
            _write_dataset(root, [_complete_record(1)])
            path = next(root.iterdir())
            wrong = root / ("record-" + "f" * 32 + ".json")
            path.rename(wrong)
            with self.assertRaisesRegex(UsabilityValidationError, "disagree"):
                load_private_dataset(root)


class TestAggregateMetrics(unittest.TestCase):
    def _dataset(self, records: list[dict]):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "records"
            _write_dataset(root, records)
            return load_private_dataset(root)

    def test_empty_dataset_never_claims_beta_or_optional_authority(self):
        dataset = self._dataset([])
        report = build_aggregate_report(dataset, load_contract())
        self.assertEqual(report["beta_result"], "NOT_EVALUATED")
        self.assertEqual(report["population"]["consented_eligible_count"], 0)
        self.assertEqual(
            report["nonclaims"],
            {
                "m5_accepted": False,
                "m6_authorized": False,
                "m7_authorized": False,
                "population_generalization": False,
            },
        )

    def test_threshold_satisfying_dataset_only_becomes_ready_for_human_review(self):
        dataset = self._dataset([_complete_record(index) for index in range(1, 9)])
        report = build_aggregate_report(dataset, load_contract())
        self.assertEqual(report["beta_result"], "READY_FOR_HUMAN_REVIEW")
        self.assertTrue(all(item["status"] == "pass" for item in report["metrics"]))
        self.assertEqual(report["scope_control"]["status"], "pass")
        self.assertFalse(report["nonclaims"]["m5_accepted"])
        parsed = parse_aggregate_bytes(canonical_json_bytes(report))
        self.assertEqual(parsed, report)

    def test_failed_thresholds_are_not_accepted(self):
        records = [_complete_record(index) for index in range(1, 9)]
        records[0]["demo"]["facilitator_interventions"] = 1
        records[0]["manual_artifact_inputs"]["pass_number"] = 1
        for item in records:
            item["eligibility"]["monthly_problem"] = False
        dataset = self._dataset(records)
        report = build_aggregate_report(dataset, load_contract())
        self.assertEqual(report["beta_result"], "M5_NOT_ACCEPTED")
        statuses = {item["metric_id"]: item["status"] for item in report["metrics"]}
        self.assertEqual(statuses["clean_clone_demo_success"], "fail")
        self.assertEqual(statuses["manual_intermediate_artifact_inputs"], "fail")
        self.assertEqual(report["scope_control"]["status"], "fail")

    def test_missing_four_week_followup_is_not_evaluated(self):
        records = [_complete_record(index) for index in range(1, 9)]
        records[7]["ci_followup"]["followup_status"] = "pending"
        records[7]["ci_followup"]["observable_evidence"] = False
        dataset = self._dataset(records)
        report = build_aggregate_report(dataset, load_contract())
        self.assertEqual(report["beta_result"], "NOT_EVALUATED")
        retained = next(
            item
            for item in report["metrics"]
            if item["metric_id"] == "four_week_retained_ci_partners"
        )
        self.assertEqual(retained["status"], "incomplete")
        self.assertEqual(retained["missing_count"], 1)

    def test_aggregate_nested_drift_and_private_warning_fail_closed(self):
        dataset = self._dataset([_complete_record(index) for index in range(1, 9)])
        report = build_aggregate_report(dataset, load_contract())

        extra = copy.deepcopy(report)
        extra["metrics"][0]["unknown"] = True
        with self.assertRaisesRegex(UsabilityValidationError, "unknown"):
            parse_aggregate_bytes(canonical_json_bytes(extra))

        warning = copy.deepcopy(report)
        warning["warnings"] = ["/home/person/private"]
        with self.assertRaisesRegex(UsabilityValidationError, "warnings"):
            parse_aggregate_bytes(canonical_json_bytes(warning))

        authority = copy.deepcopy(report)
        authority["workflow_classification"] = "verification_acceptance"
        with self.assertRaisesRegex(UsabilityValidationError, "authority"):
            parse_aggregate_bytes(canonical_json_bytes(authority))

        contradiction = copy.deepcopy(report)
        contradiction["metrics"][0]["status"] = "fail"
        with self.assertRaisesRegex(UsabilityValidationError, "unmet"):
            parse_aggregate_bytes(canonical_json_bytes(contradiction))


class TestUsabilityCLI(unittest.TestCase):
    def test_cli_writes_private_no_overwrite_report_and_strict_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            records = base / "records"
            output = base / "output"
            _write_dataset(records, [])
            output.mkdir(mode=0o700)
            output.chmod(0o700)
            report = output / "aggregate.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(["--records", os.fspath(records), "--out", os.fspath(report)]),
                    0,
                )
            self.assertEqual(report.stat().st_mode & 0o777, 0o600)
            self.assertEqual(
                parse_aggregate_bytes(report.read_bytes())["beta_result"],
                "NOT_EVALUATED",
            )
            with redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main(["--records", os.fspath(records), "--out", os.fspath(report)]),
                    2,
                )

            strict_report = output / "strict.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "--records",
                            os.fspath(records),
                            "--out",
                            os.fspath(strict_report),
                            "--require-beta-ready",
                        ]
                    ),
                    3,
                )

    def test_acceptance_environment_requires_source_root_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            records = base / "records"
            output = base / "output"
            _write_dataset(records, [])
            output.mkdir(mode=0o700)
            output.chmod(0o700)
            env = {"LIS_VERIFY_ACCEPTANCE_MANIFEST": os.fspath(base / "a.json")}
            with mock.patch.dict(os.environ, env, clear=False), redirect_stderr(
                io.StringIO()
            ):
                self.assertEqual(
                    main(
                        [
                            "--records",
                            os.fspath(records),
                            "--out",
                            os.fspath(output / "missing.json"),
                        ]
                    ),
                    2,
                )

            authority = {
                "acceptance_manifest_sha256": "sha256:" + "1" * 64,
                "source_revision": "2" * 40,
                "source_tree_sha256": "sha256:" + "3" * 64,
            }
            dataset_sha = load_private_dataset(records).identity_sha256
            with mock.patch.dict(os.environ, env, clear=False), mock.patch(
                "lis_verify.usability._source_authority", return_value=authority
            ), redirect_stdout(io.StringIO()):
                report = output / "accepted.json"
                self.assertEqual(
                    main(
                        [
                            "--records",
                            os.fspath(records),
                            "--out",
                            os.fspath(report),
                            "--source-root",
                            os.fspath(ROOT),
                            "--expected-dataset-sha256",
                            dataset_sha,
                        ]
                    ),
                    0,
                )
            raw = parse_aggregate_bytes(report.read_bytes())
            self.assertEqual(raw["workflow_classification"], "verification_acceptance")
            self.assertEqual(raw["source_authority"], authority)

    def test_expected_dataset_identity_fails_closed_on_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            records = base / "records"
            output = base / "output"
            _write_dataset(records, [_complete_record(1)])
            frozen = load_private_dataset(records).identity_sha256
            output.mkdir(mode=0o700)
            output.chmod(0o700)
            record_path = next(records.iterdir())
            changed = _complete_record(1)
            changed["demo"]["facilitator_interventions"] = 1
            record_path.write_bytes(canonical_json_bytes(changed))
            record_path.chmod(0o600)
            with redirect_stderr(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "--records",
                            os.fspath(records),
                            "--out",
                            os.fspath(output / "aggregate.json"),
                            "--expected-dataset-sha256",
                            frozen,
                        ]
                    ),
                    2,
                )


if __name__ == "__main__":
    unittest.main()
