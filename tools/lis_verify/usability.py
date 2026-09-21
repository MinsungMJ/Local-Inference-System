"""Offline, privacy-bounded M5 customer-validation record aggregation.

This operator tool validates pseudonymous session records and emits descriptive
metrics.  It never contacts participants, performs telemetry, or declares M5
complete: even a threshold-satisfying dataset remains ready for human review.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
from importlib import resources
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from typing import Any, Iterable, Mapping, Sequence

from .acceptance import (
    AcceptanceManifestError,
    load_acceptance_manifest,
    verify_acceptance_source,
)
from .product_contract import (
    SHA256_RE,
    CustomerVerdict,
    WorkflowClassification,
    canonical_json_bytes,
)
from .provenance import hash_regular_file


SESSION_SCHEMA = "lis.usability_session/v1"
SESSION_KIND = "usability_session_record"
PROTOCOL_VERSION = "lis.usability.protocol/v1"
AGGREGATE_SCHEMA = "lis.usability_aggregate/v1"
AGGREGATE_KIND = "usability_aggregate_report"
REPORT_VERSION = "1.0"
CONTRACT_RESOURCE = "session_record_v1.schema.json"

MAX_CONTRACT_BYTES = 64 * 1024
MAX_SESSION_BYTES = 64 * 1024
MAX_AGGREGATE_BYTES = 128 * 1024
MAX_RECORDS = 64
MAX_DATASET_BYTES = 2 * 1024 * 1024
MAX_SECONDS = 86_400
MAX_COUNT = 100
MAX_FRACTION_COMPONENT = MAX_SECONDS * MAX_SECONDS * 2

RECORD_ID_RE = re.compile(r"^usr1:([0-9a-f]{32})$")
PARTICIPANT_ID_RE = re.compile(r"^p1:[0-9a-f]{32}$")
WORKFLOW_ID_RE = re.compile(r"^wf1:[0-9a-f]{32}$")
RECORD_FILENAME_RE = re.compile(r"^record-([0-9a-f]{32})\.json$")
REVISION_RE = re.compile(r"^[0-9a-f]{40}$")

SESSION_FIELDS = {
    "ci_followup",
    "comprehension",
    "consent_state",
    "demo",
    "eligibility",
    "enrollment_order",
    "environment",
    "identities",
    "investigation",
    "kind",
    "manual_artifact_inputs",
    "participant_id",
    "protocol_version",
    "record_id",
    "residue",
    "schema",
    "setup_timing",
    "status",
    "verification",
    "workflow_classification",
}
VERDICTS = {item.value for item in CustomerVerdict}
MANUAL_FIELDS = {
    "artifact_set_id",
    "checkpoint_step",
    "forced_prefix",
    "intermediate_artifact_path",
    "pass_number",
    "recapture_sequence",
    "target_layer",
}
METRIC_IDS = (
    "clean_clone_demo_success",
    "median_hands_on_setup_time",
    "manual_intermediate_artifact_inputs",
    "actionable_verification_rate",
    "seeded_regression_false_passes",
    "verdict_and_next_action_comprehension",
    "median_mismatch_investigation_time_reduction",
    "four_week_retained_ci_partners",
    "sensitive_tensor_residue_events",
)
AGGREGATE_WARNINGS = {
    "No participant records were supplied; no Beta conclusion is possible.",
    "The human cohort or real-workflow integration gate is incomplete.",
    "Four-week retained-use evidence is incomplete.",
    "Aggregate thresholds do not replace required human privacy and evidence review.",
}


class UsabilityValidationError(ValueError):
    """A session, dataset, aggregate, or publication boundary is invalid."""


def _reject_constant(value: str) -> None:
    raise UsabilityValidationError(f"non-standard JSON constant: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            raise UsabilityValidationError(f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _parse_canonical(data: bytes, label: str, maximum: int) -> dict[str, Any]:
    if not data or len(data) > maximum:
        raise UsabilityValidationError(f"{label} is empty or oversized")
    try:
        raw = json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_constant,
        )
    except UsabilityValidationError:
        raise
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
    ) as exc:
        raise UsabilityValidationError(f"{label} is not valid JSON") from exc
    if not isinstance(raw, dict):
        raise UsabilityValidationError(f"{label} must be a JSON object")
    try:
        canonical = canonical_json_bytes(raw)
    except (RecursionError, TypeError, ValueError) as exc:
        raise UsabilityValidationError(f"{label} cannot be canonicalized") from exc
    if data != canonical:
        raise UsabilityValidationError(f"{label} is not canonical JSON")
    return raw


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise UsabilityValidationError(f"{label} has missing or unknown fields")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise UsabilityValidationError(f"{label} must be boolean")
    return value


def _integer(
    value: Any, label: str, *, minimum: int = 0, maximum: int = MAX_COUNT
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > maximum
    ):
        raise UsabilityValidationError(f"{label} is outside its integer bound")
    return value


def _enum(value: Any, allowed: set[str], label: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise UsabilityValidationError(f"{label} is unsupported")
    return value


def _nullable_enum(value: Any, allowed: set[str], label: str) -> str | None:
    if value is None:
        return None
    return _enum(value, allowed, label)


def _nullable_sha(value: Any, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise UsabilityValidationError(f"{label} is not a canonical SHA-256")
    return value


def _read_regular(path: Path, maximum: int, *, private: bool) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise UsabilityValidationError("cannot open bounded record input") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise UsabilityValidationError("record input is not a regular file")
        if hasattr(os, "getuid") and before.st_uid != os.getuid():
            raise UsabilityValidationError("record input owner is invalid")
        if private and stat.S_IMODE(before.st_mode) != 0o600:
            raise UsabilityValidationError("session record is not private mode 0600")
        data = bytearray()
        while len(data) <= maximum:
            chunk = os.read(fd, min(65_536, maximum + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(fd)
        if (
            before.st_dev != after.st_dev
            or before.st_ino != after.st_ino
            or before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or len(data) > maximum
        ):
            raise UsabilityValidationError("record input changed or is oversized")
        return bytes(data)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class UsabilityContract:
    raw: Mapping[str, Any]
    canonical_bytes: bytes
    identity_sha256: str


def load_contract() -> UsabilityContract:
    target = resources.files("lis_verify.usability_contract").joinpath(
        CONTRACT_RESOURCE
    )
    try:
        with target.open("rb") as stream:
            data = stream.read(MAX_CONTRACT_BYTES + 1)
    except (FileNotFoundError, OSError) as exc:
        raise UsabilityValidationError("packaged usability contract is unavailable") from exc
    raw = _parse_canonical(data, "usability contract", MAX_CONTRACT_BYTES)
    _exact(raw, {"$id", "$schema", "additionalProperties", "properties", "required", "title", "type"}, "usability contract")
    if (
        raw["$schema"] != "https://json-schema.org/draft/2020-12/schema"
        or raw["type"] != "object"
        or raw["additionalProperties"] is not False
        or set(raw["required"]) != SESSION_FIELDS
        or set(raw["properties"]) != SESSION_FIELDS
    ):
        raise UsabilityValidationError("packaged usability contract identity drifted")
    return UsabilityContract(
        raw=raw,
        canonical_bytes=data,
        identity_sha256="sha256:" + hashlib.sha256(data).hexdigest(),
    )


def validate_session_record(raw: Any) -> None:
    value = _exact(raw, SESSION_FIELDS, "session record")
    if value["schema"] != SESSION_SCHEMA or value["kind"] != SESSION_KIND:
        raise UsabilityValidationError("session record identity is unsupported")
    if value["protocol_version"] != PROTOCOL_VERSION:
        raise UsabilityValidationError("session protocol version is unsupported")
    if value["workflow_classification"] != "development_debugging":
        raise UsabilityValidationError("session observations must be debugging-classified")
    record_match = (
        RECORD_ID_RE.fullmatch(value["record_id"])
        if isinstance(value["record_id"], str)
        else None
    )
    if record_match is None:
        raise UsabilityValidationError("session record identity is not pseudonymous")
    if (
        not isinstance(value["participant_id"], str)
        or PARTICIPANT_ID_RE.fullmatch(value["participant_id"]) is None
    ):
        raise UsabilityValidationError("participant identity is not pseudonymous")
    _integer(value["enrollment_order"], "enrollment_order", minimum=1, maximum=64)
    status = _enum(value["status"], {"complete", "incomplete", "withdrawn"}, "status")
    consent = _enum(value["consent_state"], {"confirmed", "withdrawn"}, "consent_state")

    eligibility = _exact(
        value["eligibility"],
        {"designer_of_pass0_4", "eligible", "exclusion_reason", "external_to_lis_workflow", "monthly_problem"},
        "eligibility",
    )
    for field in ("designer_of_pass0_4", "eligible", "external_to_lis_workflow", "monthly_problem"):
        _boolean(eligibility[field], f"eligibility.{field}")
    exclusion = _nullable_enum(
        eligibility["exclusion_reason"],
        {"not_qualifying", "designer_of_pass0_4", "consent_withdrawn", "protocol_deviation"},
        "eligibility.exclusion_reason",
    )
    if eligibility["eligible"]:
        if exclusion is not None or eligibility["designer_of_pass0_4"] or consent != "confirmed":
            raise UsabilityValidationError("eligible participant has an exclusion")
    elif exclusion is None:
        raise UsabilityValidationError("ineligible participant lacks an exclusion")
    if eligibility["designer_of_pass0_4"] and exclusion != "designer_of_pass0_4":
        raise UsabilityValidationError("Pass 0-4 designer exclusion is inconsistent")
    if consent == "withdrawn":
        if status != "withdrawn" or eligibility["eligible"] or exclusion != "consent_withdrawn":
            raise UsabilityValidationError("withdrawn consent is inconsistently classified")
    elif status == "withdrawn":
        raise UsabilityValidationError("withdrawn status requires withdrawn consent")

    environment = _exact(
        value["environment"],
        {"cpu_family", "public_golden_used", "support_status"},
        "environment",
    )
    _enum(environment["cpu_family"], {"x86_64", "aarch64", "other"}, "environment.cpu_family")
    _boolean(environment["public_golden_used"], "environment.public_golden_used")
    support_status = _enum(
        environment["support_status"], {"supported", "unsupported", "invalid"}, "environment.support_status"
    )

    demo = _exact(
        value["demo"],
        {"attempted", "canonical_report", "completed", "expected_regression_identified", "facilitator_interventions", "verdict"},
        "demo",
    )
    for field in ("attempted", "canonical_report", "completed", "expected_regression_identified"):
        _boolean(demo[field], f"demo.{field}")
    _integer(demo["facilitator_interventions"], "demo.facilitator_interventions")
    demo_verdict = _nullable_enum(demo["verdict"], VERDICTS, "demo.verdict")
    if not demo["attempted"] and (
        demo["canonical_report"]
        or demo["completed"]
        or demo["expected_regression_identified"]
        or demo["facilitator_interventions"] != 0
        or demo_verdict is not None
    ):
        raise UsabilityValidationError("unattempted demo carries observations")
    if demo["canonical_report"] and demo_verdict is None:
        raise UsabilityValidationError("canonical demo report lacks a verdict")
    if demo["completed"] and not demo["attempted"]:
        raise UsabilityValidationError("completed demo was not attempted")
    if demo["expected_regression_identified"] and demo_verdict != "REGRESSION":
        raise UsabilityValidationError("demo regression interpretation is inconsistent")

    timing = _exact(
        value["setup_timing"],
        {"hands_on_seconds", "inference_wait_seconds", "model_acquisition_wait_seconds", "observed", "wall_seconds"},
        "setup_timing",
    )
    _boolean(timing["observed"], "setup_timing.observed")
    for field in ("hands_on_seconds", "inference_wait_seconds", "model_acquisition_wait_seconds", "wall_seconds"):
        _integer(timing[field], f"setup_timing.{field}", maximum=MAX_SECONDS)
    excluded = timing["inference_wait_seconds"] + timing["model_acquisition_wait_seconds"]
    if timing["observed"]:
        if excluded > timing["wall_seconds"] or timing["hands_on_seconds"] != timing["wall_seconds"] - excluded:
            raise UsabilityValidationError("setup timing arithmetic is inconsistent")
    elif any(timing[field] != 0 for field in ("hands_on_seconds", "inference_wait_seconds", "model_acquisition_wait_seconds", "wall_seconds")):
        raise UsabilityValidationError("unobserved setup timing carries durations")

    manual = _exact(value["manual_artifact_inputs"], MANUAL_FIELDS, "manual_artifact_inputs")
    for field in MANUAL_FIELDS:
        _integer(manual[field], f"manual_artifact_inputs.{field}")

    verification = _exact(
        value["verification"],
        {"abandoned", "attempted", "canonical_verdict", "next_action_beginable", "undocumented_interventions"},
        "verification",
    )
    _boolean(verification["attempted"], "verification.attempted")
    _boolean(verification["abandoned"], "verification.abandoned")
    canonical_verdict = _nullable_enum(
        verification["canonical_verdict"], VERDICTS, "verification.canonical_verdict"
    )
    next_action = verification["next_action_beginable"]
    if next_action is not None:
        _boolean(next_action, "verification.next_action_beginable")
    _integer(verification["undocumented_interventions"], "verification.undocumented_interventions")
    if not verification["attempted"] and (
        verification["abandoned"]
        or canonical_verdict is not None
        or next_action is not None
        or verification["undocumented_interventions"] != 0
    ):
        raise UsabilityValidationError("unattempted verification carries observations")
    if canonical_verdict is not None and not verification["attempted"]:
        raise UsabilityValidationError("verification verdict lacks an attempt")
    if support_status == "unsupported" and canonical_verdict not in {None, "UNSUPPORTED"}:
        raise UsabilityValidationError("unsupported environment has a contradictory verdict")

    comprehension = _exact(
        value["comprehension"],
        {"attempted", "evidence_ceiling_correct", "next_action_correct", "used_further_documentation", "verdict_correct"},
        "comprehension",
    )
    for field in ("attempted", "evidence_ceiling_correct", "next_action_correct", "used_further_documentation", "verdict_correct"):
        _boolean(comprehension[field], f"comprehension.{field}")
    if not comprehension["attempted"] and any(
        comprehension[field]
        for field in ("evidence_ceiling_correct", "next_action_correct", "used_further_documentation", "verdict_correct")
    ):
        raise UsabilityValidationError("unattempted comprehension carries answers")

    investigation = _exact(
        value["investigation"], {"lis_verify_seconds", "paired", "prior_seconds"}, "investigation"
    )
    _boolean(investigation["paired"], "investigation.paired")
    for field in ("lis_verify_seconds", "prior_seconds"):
        child = investigation[field]
        if child is not None:
            _integer(child, f"investigation.{field}", minimum=1, maximum=MAX_SECONDS)
    if investigation["paired"] != (
        investigation["lis_verify_seconds"] is not None and investigation["prior_seconds"] is not None
    ):
        raise UsabilityValidationError("paired investigation durations are incomplete")
    if not investigation["paired"] and (
        investigation["lis_verify_seconds"] is not None or investigation["prior_seconds"] is not None
    ):
        raise UsabilityValidationError("unpaired investigation carries durations")

    followup = _exact(
        value["ci_followup"],
        {"design_partner", "followup_status", "integration_observed", "observable_evidence", "workflow_id"},
        "ci_followup",
    )
    for field in ("design_partner", "integration_observed", "observable_evidence"):
        _boolean(followup[field], f"ci_followup.{field}")
    followup_status = _enum(
        followup["followup_status"],
        {"not_due", "pending", "retained", "stopped", "unavailable", "withdrawn"},
        "ci_followup.followup_status",
    )
    workflow_id = followup["workflow_id"]
    if workflow_id is not None and (
        not isinstance(workflow_id, str) or WORKFLOW_ID_RE.fullmatch(workflow_id) is None
    ):
        raise UsabilityValidationError("workflow identity is not pseudonymous")
    if followup["integration_observed"] != (workflow_id is not None):
        raise UsabilityValidationError("workflow integration identity is inconsistent")
    if not followup["design_partner"] and (
        followup["integration_observed"]
        or followup["observable_evidence"]
        or followup_status != "not_due"
    ):
        raise UsabilityValidationError("non-partner record carries CI follow-up")
    if followup_status == "retained" and (
        not followup["design_partner"]
        or not followup["integration_observed"]
        or not followup["observable_evidence"]
    ):
        raise UsabilityValidationError("retained CI use lacks observable evidence")
    if followup["observable_evidence"] and not followup["integration_observed"]:
        raise UsabilityValidationError("CI evidence lacks an observed integration")

    residue = _exact(
        value["residue"],
        {"handled_error_checked", "handled_error_sensitive_tensor_residue", "normal_execution_checked", "normal_sensitive_tensor_residue"},
        "residue",
    )
    for field in residue:
        _boolean(residue[field], f"residue.{field}")
    if not residue["normal_execution_checked"] and residue["normal_sensitive_tensor_residue"]:
        raise UsabilityValidationError("normal residue claim lacks an observation")
    if not residue["handled_error_checked"] and residue["handled_error_sensitive_tensor_residue"]:
        raise UsabilityValidationError("handled-error residue claim lacks an observation")

    identities = _exact(
        value["identities"], {"backend_report_sha256", "demo_report_sha256"}, "identities"
    )
    demo_identity = _nullable_sha(identities["demo_report_sha256"], "identities.demo_report_sha256")
    backend_identity = _nullable_sha(identities["backend_report_sha256"], "identities.backend_report_sha256")
    if demo["canonical_report"] != (demo_identity is not None):
        raise UsabilityValidationError("demo report identity is inconsistent")
    if (canonical_verdict is not None) != (backend_identity is not None):
        raise UsabilityValidationError("backend report identity is inconsistent")

    if status == "withdrawn" and (
        demo["attempted"]
        or timing["observed"]
        or sum(manual.values()) != 0
        or verification["attempted"]
        or comprehension["attempted"]
        or investigation["paired"]
        or followup["design_partner"]
        or residue["normal_execution_checked"]
        or residue["handled_error_checked"]
        or demo_identity is not None
        or backend_identity is not None
    ):
        raise UsabilityValidationError(
            "withdrawn session must not retain behavioral observations"
        )

    if status == "complete" and eligibility["eligible"]:
        if not demo["attempted"] or not timing["observed"] or not comprehension["attempted"]:
            raise UsabilityValidationError("complete eligible session lacks required observations")
        if not residue["normal_execution_checked"] or not residue["handled_error_checked"]:
            raise UsabilityValidationError("complete eligible session lacks residue checks")
        if support_status in {"supported", "unsupported"} and not verification["attempted"]:
            raise UsabilityValidationError("complete eligible session lacks model-backed verification")


def parse_session_bytes(data: bytes) -> dict[str, Any]:
    raw = _parse_canonical(data, "session record", MAX_SESSION_BYTES)
    validate_session_record(raw)
    return raw


def _validate_private_directory(path: Path) -> Path:
    absolute = Path(path).absolute()
    current = Path(absolute.anchor)
    try:
        for part in absolute.parts[1:]:
            current = current / part
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise UsabilityValidationError("dataset path contains a symlink")
    except OSError as exc:
        raise UsabilityValidationError("dataset directory is unavailable") from exc
    info = absolute.lstat()
    if (
        not stat.S_ISDIR(info.st_mode)
        or stat.S_IMODE(info.st_mode) != 0o700
        or (hasattr(os, "getuid") and info.st_uid != os.getuid())
    ):
        raise UsabilityValidationError("dataset directory is not private mode 0700")
    return absolute


@dataclass(frozen=True)
class LoadedDataset:
    records: tuple[dict[str, Any], ...]
    identity_sha256: str
    total_size_bytes: int


def load_private_dataset(path: Path) -> LoadedDataset:
    root = _validate_private_directory(path)
    try:
        before = root.stat()
        entries = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError as exc:
        raise UsabilityValidationError("cannot enumerate the private dataset") from exc
    if len(entries) > MAX_RECORDS:
        raise UsabilityValidationError("dataset record count exceeds its bound")
    records: list[dict[str, Any]] = []
    identities: list[dict[str, Any]] = []
    seen_records: set[str] = set()
    seen_participants: set[str] = set()
    seen_orders: set[int] = set()
    total_size = 0
    for entry in entries:
        match = RECORD_FILENAME_RE.fullmatch(entry.name)
        if match is None:
            raise UsabilityValidationError("dataset contains an unexpected entry")
        data = _read_regular(entry, MAX_SESSION_BYTES, private=True)
        total_size += len(data)
        if total_size > MAX_DATASET_BYTES:
            raise UsabilityValidationError("dataset byte size exceeds its bound")
        raw = parse_session_bytes(data)
        if raw["record_id"].split(":", 1)[1] != match.group(1):
            raise UsabilityValidationError("record filename and identity disagree")
        if raw["record_id"] in seen_records:
            raise UsabilityValidationError("dataset contains a duplicate record identity")
        if raw["participant_id"] in seen_participants:
            raise UsabilityValidationError("dataset contains a duplicate participant")
        if raw["enrollment_order"] in seen_orders:
            raise UsabilityValidationError("dataset contains a duplicate enrollment order")
        seen_records.add(raw["record_id"])
        seen_participants.add(raw["participant_id"])
        seen_orders.add(raw["enrollment_order"])
        record_digest = "sha256:" + hashlib.sha256(data).hexdigest()
        identities.append(
            {"record_id": raw["record_id"], "sha256": record_digest, "size_bytes": len(data)}
        )
        records.append(raw)
    try:
        after = root.stat()
        final_names = tuple(sorted(item.name for item in root.iterdir()))
    except OSError as exc:
        raise UsabilityValidationError("cannot recheck the private dataset") from exc
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or before.st_mtime_ns != after.st_mtime_ns
        or tuple(item.name for item in entries) != final_names
    ):
        raise UsabilityValidationError("dataset directory changed while reading")
    identity_value = {"domain": "lis.usability_dataset/v1", "records": identities}
    dataset_identity = "sha256:" + hashlib.sha256(
        canonical_json_bytes(identity_value)
    ).hexdigest()
    return LoadedDataset(tuple(records), dataset_identity, total_size)


def _fraction(value: Fraction | None, unit: str) -> dict[str, Any] | None:
    if value is None:
        return None
    return {"denominator": value.denominator, "numerator": value.numerator, "unit": unit}


def _median(values: Sequence[Fraction]) -> Fraction | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _metric(
    metric_id: str,
    *,
    value: Fraction | None,
    unit: str,
    target_operator: str,
    target: Fraction,
    numerator: int | None,
    denominator: int | None,
    missing_count: int,
    complete: bool,
) -> dict[str, Any]:
    if value is None or not complete:
        status = "incomplete"
    elif target_operator == ">=":
        status = "pass" if value >= target else "fail"
    elif target_operator == "<=":
        status = "pass" if value <= target else "fail"
    elif target_operator == "==":
        status = "pass" if value == target else "fail"
    else:
        raise AssertionError("unknown target operator")
    return {
        "denominator": denominator,
        "metric_id": metric_id,
        "missing_count": missing_count,
        "numerator": numerator,
        "status": status,
        "target": {
            "operator": target_operator,
            "value": _fraction(target, unit),
        },
        "value": _fraction(value, unit),
    }


def build_aggregate_report(
    dataset: LoadedDataset,
    contract: UsabilityContract,
    *,
    workflow_classification: WorkflowClassification = WorkflowClassification.DEVELOPMENT_DEBUGGING,
    source_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    records = list(dataset.records)
    counted = [
        item
        for item in records
        if item["consent_state"] == "confirmed"
        and item["eligibility"]["eligible"]
        and not item["eligibility"]["designer_of_pass0_4"]
    ]
    complete = [item for item in counted if item["status"] == "complete"]
    external_count = sum(item["eligibility"]["external_to_lis_workflow"] for item in counted)
    workflow_ids = {
        item["ci_followup"]["workflow_id"]
        for item in counted
        if item["ci_followup"]["integration_observed"]
    }
    design_partners = sorted(
        (item for item in counted if item["ci_followup"]["design_partner"]),
        key=lambda item: item["enrollment_order"],
    )
    first_eight = design_partners[:8]
    cohort_count_ok = 8 <= len(counted) <= 12
    cohort_complete = (
        cohort_count_ok
        and len(complete) == len(counted)
        and external_count >= 3
        and 3 <= len(workflow_ids) <= 5
    )

    demo_successes = sum(
        item["demo"]["completed"]
        and item["demo"]["canonical_report"]
        and item["demo"]["verdict"] == "REGRESSION"
        and item["demo"]["expected_regression_identified"]
        and item["demo"]["facilitator_interventions"] == 0
        for item in counted
    )
    demo_missing = sum(not item["demo"]["attempted"] for item in counted)
    demo_rate = Fraction(demo_successes, len(counted)) if counted else None

    timings = [Fraction(item["setup_timing"]["hands_on_seconds"], 1) for item in counted if item["setup_timing"]["observed"]]
    timing_missing = len(counted) - len(timings)
    median_setup = _median(timings)

    manual_total = sum(
        sum(item["manual_artifact_inputs"].values()) for item in counted
    )

    supported = [
        item for item in counted if item["environment"]["support_status"] == "supported"
    ]
    actionable = sum(
        item["verification"]["attempted"]
        and item["verification"]["canonical_verdict"] in VERDICTS
        and item["verification"]["next_action_beginable"] is True
        and item["verification"]["undocumented_interventions"] == 0
        for item in supported
    )
    actionable_missing = sum(
        not item["verification"]["attempted"]
        or item["verification"]["canonical_verdict"] is None
        or item["verification"]["next_action_beginable"] is None
        for item in supported
    )
    actionable_rate = Fraction(actionable, len(supported)) if supported else None

    false_passes = sum(item["demo"]["verdict"] == "PASS" for item in counted)
    demo_attempt_count = sum(item["demo"]["attempted"] for item in counted)

    comprehension_successes = sum(
        item["comprehension"]["attempted"]
        and item["comprehension"]["verdict_correct"]
        and item["comprehension"]["evidence_ceiling_correct"]
        and item["comprehension"]["next_action_correct"]
        and not item["comprehension"]["used_further_documentation"]
        for item in counted
    )
    comprehension_missing = sum(not item["comprehension"]["attempted"] for item in counted)
    comprehension_rate = Fraction(comprehension_successes, len(counted)) if counted else None

    reductions = [
        Fraction(
            item["investigation"]["prior_seconds"] - item["investigation"]["lis_verify_seconds"],
            item["investigation"]["prior_seconds"],
        )
        for item in counted
        if item["investigation"]["paired"]
    ]
    median_reduction = _median(reductions)

    retained = sum(
        item["ci_followup"]["followup_status"] == "retained"
        and item["ci_followup"]["observable_evidence"]
        for item in first_eight
    )
    followup_missing = sum(
        item["ci_followup"]["followup_status"] in {"not_due", "pending"}
        for item in first_eight
    )
    followup_complete = len(first_eight) == 8 and followup_missing == 0

    residue_checks = 0
    residue_events = 0
    residue_missing = 0
    for item in counted:
        for checked_field, residue_field in (
            ("normal_execution_checked", "normal_sensitive_tensor_residue"),
            ("handled_error_checked", "handled_error_sensitive_tensor_residue"),
        ):
            if item["residue"][checked_field]:
                residue_checks += 1
                residue_events += int(item["residue"][residue_field])
            else:
                residue_missing += 1

    monthly_count = sum(item["eligibility"]["monthly_problem"] for item in counted)
    monthly_rate = Fraction(monthly_count, len(counted)) if counted else None

    metrics = [
        _metric(
            "clean_clone_demo_success",
            value=demo_rate,
            unit="ratio",
            target_operator=">=",
            target=Fraction(9, 10),
            numerator=demo_successes,
            denominator=len(counted),
            missing_count=demo_missing,
            complete=cohort_count_ok and demo_missing == 0,
        ),
        _metric(
            "median_hands_on_setup_time",
            value=median_setup,
            unit="seconds",
            target_operator="<=",
            target=Fraction(600, 1),
            numerator=None,
            denominator=len(timings),
            missing_count=timing_missing,
            complete=cohort_count_ok and timing_missing == 0,
        ),
        _metric(
            "manual_intermediate_artifact_inputs",
            value=Fraction(manual_total, 1) if counted else None,
            unit="count",
            target_operator="==",
            target=Fraction(0, 1),
            numerator=manual_total,
            denominator=len(counted),
            missing_count=0,
            complete=cohort_count_ok,
        ),
        _metric(
            "actionable_verification_rate",
            value=actionable_rate,
            unit="ratio",
            target_operator=">=",
            target=Fraction(9, 10),
            numerator=actionable,
            denominator=len(supported),
            missing_count=actionable_missing,
            complete=cohort_count_ok and bool(supported) and actionable_missing == 0,
        ),
        _metric(
            "seeded_regression_false_passes",
            value=Fraction(false_passes, 1) if counted else None,
            unit="count",
            target_operator="==",
            target=Fraction(0, 1),
            numerator=false_passes,
            denominator=demo_attempt_count,
            missing_count=demo_missing,
            complete=cohort_count_ok and demo_missing == 0,
        ),
        _metric(
            "verdict_and_next_action_comprehension",
            value=comprehension_rate,
            unit="ratio",
            target_operator=">=",
            target=Fraction(4, 5),
            numerator=comprehension_successes,
            denominator=len(counted),
            missing_count=comprehension_missing,
            complete=cohort_count_ok and comprehension_missing == 0,
        ),
        _metric(
            "median_mismatch_investigation_time_reduction",
            value=median_reduction,
            unit="ratio",
            target_operator=">=",
            target=Fraction(1, 2),
            numerator=None,
            denominator=len(reductions),
            missing_count=0,
            complete=cohort_count_ok and bool(reductions),
        ),
        _metric(
            "four_week_retained_ci_partners",
            value=Fraction(retained, 1) if first_eight else None,
            unit="count",
            target_operator=">=",
            target=Fraction(5, 1),
            numerator=retained,
            denominator=len(first_eight),
            missing_count=followup_missing,
            complete=cohort_count_ok and followup_complete,
        ),
        _metric(
            "sensitive_tensor_residue_events",
            value=Fraction(residue_events, 1) if residue_checks else None,
            unit="count",
            target_operator="==",
            target=Fraction(0, 1),
            numerator=residue_events,
            denominator=residue_checks,
            missing_count=residue_missing,
            complete=cohort_count_ok and residue_missing == 0,
        ),
    ]
    scope_control = _metric(
        "monthly_verification_problem_frequency",
        value=monthly_rate,
        unit="ratio",
        target_operator=">=",
        target=Fraction(1, 2),
        numerator=monthly_count,
        denominator=len(counted),
        missing_count=0,
        complete=cohort_count_ok,
    )

    metric_statuses = {item["status"] for item in metrics}
    if not cohort_complete or "incomplete" in metric_statuses:
        beta_result = "NOT_EVALUATED"
    elif "fail" in metric_statuses:
        beta_result = "M5_NOT_ACCEPTED"
    else:
        beta_result = "READY_FOR_HUMAN_REVIEW"
    warnings: list[str] = []
    if not records:
        warnings.append("No participant records were supplied; no Beta conclusion is possible.")
    if not cohort_complete:
        warnings.append("The human cohort or real-workflow integration gate is incomplete.")
    if not followup_complete:
        warnings.append("Four-week retained-use evidence is incomplete.")
    warnings.append("Aggregate thresholds do not replace required human privacy and evidence review.")

    report = {
        "beta_result": beta_result,
        "dataset": {
            "record_count": len(records),
            "sha256": dataset.identity_sha256,
            "total_size_bytes": dataset.total_size_bytes,
        },
        "human_review_required": True,
        "kind": AGGREGATE_KIND,
        "metrics": metrics,
        "nonclaims": {
            "m5_accepted": False,
            "m6_authorized": False,
            "m7_authorized": False,
            "population_generalization": False,
        },
        "population": {
            "complete_eligible_count": len(complete),
            "consented_eligible_count": len(counted),
            "design_partner_count": len(design_partners),
            "external_count": external_count,
            "incomplete_count": sum(item["status"] == "incomplete" for item in records),
            "real_workflow_count": len(workflow_ids),
            "withdrawn_count": sum(item["status"] == "withdrawn" for item in records),
        },
        "protocol": {
            "schema_sha256": contract.identity_sha256,
            "version": PROTOCOL_VERSION,
        },
        "report_version": REPORT_VERSION,
        "schema": AGGREGATE_SCHEMA,
        "scope_control": scope_control,
        "source_authority": None if source_authority is None else dict(source_authority),
        "warnings": warnings,
        "workflow_classification": workflow_classification.value,
    }
    validate_aggregate_report(report)
    return report


def validate_aggregate_report(raw: Any) -> None:
    value = _exact(
        raw,
        {"beta_result", "dataset", "human_review_required", "kind", "metrics", "nonclaims", "population", "protocol", "report_version", "schema", "scope_control", "source_authority", "warnings", "workflow_classification"},
        "aggregate report",
    )
    if (
        value["schema"] != AGGREGATE_SCHEMA
        or value["kind"] != AGGREGATE_KIND
        or value["report_version"] != REPORT_VERSION
        or value["beta_result"] not in {"NOT_EVALUATED", "M5_NOT_ACCEPTED", "READY_FOR_HUMAN_REVIEW"}
        or value["human_review_required"] is not True
    ):
        raise UsabilityValidationError("aggregate report identity is invalid")
    if value["workflow_classification"] not in {item.value for item in WorkflowClassification}:
        raise UsabilityValidationError("aggregate workflow classification is invalid")
    if not isinstance(value["metrics"], list) or len(value["metrics"]) != 9:
        raise UsabilityValidationError("aggregate metric set is incomplete")
    if any(not isinstance(item, dict) for item in value["metrics"]):
        raise UsabilityValidationError("aggregate metric entry is not an object")
    if tuple(item.get("metric_id") for item in value["metrics"]) != METRIC_IDS:
        raise UsabilityValidationError("aggregate metric identities drifted")
    for index, metric in enumerate(value["metrics"]):
        _validate_metric(metric, METRIC_IDS[index])
    _validate_metric(
        value["scope_control"], "monthly_verification_problem_frequency"
    )
    if value["scope_control"]["metric_id"] != "monthly_verification_problem_frequency":
        raise UsabilityValidationError("aggregate scope-control identity drifted")
    if value["nonclaims"] != {
        "m5_accepted": False,
        "m6_authorized": False,
        "m7_authorized": False,
        "population_generalization": False,
    }:
        raise UsabilityValidationError("aggregate report promotes a prohibited claim")
    dataset = _exact(
        value["dataset"],
        {"record_count", "sha256", "total_size_bytes"},
        "aggregate.dataset",
    )
    _integer(dataset["record_count"], "aggregate.dataset.record_count", maximum=MAX_RECORDS)
    _integer(
        dataset["total_size_bytes"],
        "aggregate.dataset.total_size_bytes",
        maximum=MAX_DATASET_BYTES,
    )
    if not isinstance(dataset["sha256"], str) or SHA256_RE.fullmatch(dataset["sha256"]) is None:
        raise UsabilityValidationError("aggregate dataset identity is invalid")
    protocol = _exact(
        value["protocol"], {"schema_sha256", "version"}, "aggregate.protocol"
    )
    if (
        protocol["version"] != PROTOCOL_VERSION
        or not isinstance(protocol["schema_sha256"], str)
        or SHA256_RE.fullmatch(protocol["schema_sha256"]) is None
    ):
        raise UsabilityValidationError("aggregate protocol identity is invalid")
    population = _exact(
        value["population"],
        {"complete_eligible_count", "consented_eligible_count", "design_partner_count", "external_count", "incomplete_count", "real_workflow_count", "withdrawn_count"},
        "aggregate.population",
    )
    for field, child in population.items():
        _integer(child, f"aggregate.population.{field}", maximum=MAX_RECORDS)
    authority = value["source_authority"]
    if authority is None:
        if value["workflow_classification"] != "development_debugging":
            raise UsabilityValidationError("acceptance aggregate lacks source authority")
    else:
        authority = _exact(
            authority,
            {"acceptance_manifest_sha256", "source_revision", "source_tree_sha256"},
            "aggregate.source_authority",
        )
        for field in ("acceptance_manifest_sha256", "source_tree_sha256"):
            if not isinstance(authority[field], str) or SHA256_RE.fullmatch(authority[field]) is None:
                raise UsabilityValidationError("aggregate source authority is malformed")
        if not isinstance(authority["source_revision"], str) or REVISION_RE.fullmatch(authority["source_revision"]) is None:
            raise UsabilityValidationError("aggregate source revision is malformed")
        if value["workflow_classification"] != "verification_acceptance":
            raise UsabilityValidationError("debugging aggregate carries acceptance authority")
    warnings = value["warnings"]
    if (
        not isinstance(warnings, list)
        or any(not isinstance(item, str) for item in warnings)
        or len(warnings) != len(set(warnings))
        or any(item not in AGGREGATE_WARNINGS for item in warnings)
    ):
        raise UsabilityValidationError("aggregate warnings are not from the bounded set")
    metric_statuses = {item["status"] for item in value["metrics"]}
    if value["beta_result"] == "READY_FOR_HUMAN_REVIEW" and metric_statuses != {"pass"}:
        raise UsabilityValidationError("ready result has an unmet metric")
    if value["beta_result"] == "M5_NOT_ACCEPTED" and (
        "fail" not in metric_statuses or "incomplete" in metric_statuses
    ):
        raise UsabilityValidationError("not-accepted result is inconsistent")
    if len(canonical_json_bytes(value)) > MAX_AGGREGATE_BYTES:
        raise UsabilityValidationError("aggregate report exceeds its byte bound")


def _validate_fraction_value(value: Any, label: str) -> None:
    child = _exact(value, {"denominator", "numerator", "unit"}, label)
    _integer(
        child["denominator"],
        f"{label}.denominator",
        minimum=1,
        maximum=MAX_FRACTION_COMPONENT,
    )
    numerator = child["numerator"]
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or abs(numerator) > MAX_FRACTION_COMPONENT
    ):
        raise UsabilityValidationError(f"{label}.numerator is outside its bound")
    _enum(child["unit"], {"count", "ratio", "seconds"}, f"{label}.unit")


def _validate_metric(raw: Any, expected_id: str) -> None:
    value = _exact(
        raw,
        {"denominator", "metric_id", "missing_count", "numerator", "status", "target", "value"},
        "aggregate metric",
    )
    if value["metric_id"] != expected_id:
        raise UsabilityValidationError("aggregate metric identity is inconsistent")
    _enum(value["status"], {"pass", "fail", "incomplete"}, "aggregate metric status")
    _integer(value["missing_count"], "aggregate metric missing_count", maximum=MAX_RECORDS * 2)
    for field in ("numerator", "denominator"):
        child = value[field]
        if child is not None:
            _integer(child, f"aggregate metric {field}", maximum=MAX_SECONDS * MAX_RECORDS)
    target = _exact(value["target"], {"operator", "value"}, "aggregate metric target")
    _enum(target["operator"], {"<=", "==", ">="}, "aggregate metric target operator")
    _validate_fraction_value(target["value"], "aggregate metric target value")
    if value["value"] is not None:
        _validate_fraction_value(value["value"], "aggregate metric value")


def parse_aggregate_bytes(data: bytes) -> dict[str, Any]:
    raw = _parse_canonical(data, "aggregate report", MAX_AGGREGATE_BYTES)
    validate_aggregate_report(raw)
    return raw


def _publish_private(path: Path, data: bytes) -> None:
    destination = Path(path)
    if destination.exists() or destination.is_symlink():
        raise UsabilityValidationError("aggregate report overwrite is prohibited")
    parent = _validate_private_directory(destination.parent)
    temporary = parent / f".{destination.name}.{secrets.token_hex(8)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(temporary, flags, 0o600)
    except OSError as exc:
        raise UsabilityValidationError("cannot create private aggregate report") from exc
    published = False
    try:
        try:
            os.fchmod(fd, 0o600)
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise UsabilityValidationError("aggregate report write made no progress")
                view = view[written:]
            os.fsync(fd)
        finally:
            os.close(fd)
        parse_aggregate_bytes(
            _read_regular(temporary, MAX_AGGREGATE_BYTES, private=True)
        )
        os.link(temporary, destination, follow_symlinks=False)
        published = True
        temporary.unlink()
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if published:
            try:
                destination.unlink()
            except FileNotFoundError:
                pass
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _source_authority(path: Path, source_root: Path) -> dict[str, Any]:
    manifest = load_acceptance_manifest(path)
    verify_acceptance_source(manifest, source_root)
    digest, _ = hash_regular_file(path)
    return {
        "acceptance_manifest_sha256": digest,
        "source_revision": manifest.source_revision,
        "source_tree_sha256": manifest.source_tree_sha256,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lis-verify-usability",
        description="Validate private M5 session records and emit bounded aggregate metrics.",
    )
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--expected-dataset-sha256")
    parser.add_argument("--require-beta-ready", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        contract = load_contract()
        dataset = load_private_dataset(args.records)
        expected_dataset = args.expected_dataset_sha256
        if expected_dataset is not None and (
            not isinstance(expected_dataset, str)
            or SHA256_RE.fullmatch(expected_dataset) is None
        ):
            raise UsabilityValidationError("expected dataset identity is malformed")
        if expected_dataset is not None and expected_dataset != dataset.identity_sha256:
            raise UsabilityValidationError("frozen dataset identity changed")
        classification = WorkflowClassification.DEVELOPMENT_DEBUGGING
        source_authority = None
        acceptance_path = os.environ.get("LIS_VERIFY_ACCEPTANCE_MANIFEST")
        if acceptance_path is not None:
            if args.source_root is None or expected_dataset is None:
                raise UsabilityValidationError(
                    "acceptance aggregation requires source and frozen dataset authority"
                )
            source_authority = _source_authority(
                Path(acceptance_path), args.source_root
            )
            classification = WorkflowClassification.VERIFICATION_ACCEPTANCE
        elif args.source_root is not None:
            raise UsabilityValidationError(
                "--source-root is only valid with acceptance authority"
            )
        report = build_aggregate_report(
            dataset,
            contract,
            workflow_classification=classification,
            source_authority=source_authority,
        )
        _publish_private(args.out, canonical_json_bytes(report))
    except (
        AcceptanceManifestError,
        OSError,
        UsabilityValidationError,
        ValueError,
    ):
        print("lis-verify-usability: validation failed closed", file=sys.stderr)
        return 2
    print(
        "lis-verify-usability: verified "
        f"result={report['beta_result']} "
        f"records={report['dataset']['record_count']} "
        f"dataset={report['dataset']['sha256']}"
    )
    if not args.require_beta_ready:
        return 0
    if report["beta_result"] == "READY_FOR_HUMAN_REVIEW":
        return 0
    if report["beta_result"] == "M5_NOT_ACCEPTED":
        return 4
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
