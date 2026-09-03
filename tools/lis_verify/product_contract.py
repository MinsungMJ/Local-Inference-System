"""Frozen Pass 5 customer-product contract for ``LIS Verify``.

This module contains enums, finite decision tables, and model-free validators.
It intentionally performs no orchestration, subprocess execution, report I/O,
or inference.  Pass 0 through Pass 4 remain the evidence-producing layers; this
module only constrains how their bounded results may be presented to customers.
"""

from __future__ import annotations

from enum import Enum
import hashlib
import json
import math
import re
from typing import Any

from .model import Pass0Verdict
from .pass1_model import Pass1Status
from .pass2_model import Pass2Status
from .pass3_model import Pass3Status
from .pass4_contract import Pass4Status


SCHEMA = "lis.verification_report/v1"
KIND = "verification_report"
REPORT_VERSION = "1.0"
CONTRACT_VERSION = "lis.verify.product_contract/v1"

MAX_REPORT_BYTES = 1_048_576
MAX_SUMMARY_BYTES = 65_536
MAX_IDENTIFIER_BYTES = 128
MAX_DETAIL_BYTES = 256
MAX_NEXT_ACTION_BYTES = 512
MAX_WARNINGS = 32
MAX_REASON_CODES = 32
MAX_TOKEN_ID_PREVIEW = 64
MAX_LAYER_COLLECTION = 4096
MAX_INTRA_LAYER_STAGES = 17
DEFAULT_STAGE_TIMEOUT_SECONDS = 1800
MAX_STAGE_TIMEOUT_SECONDS = 7200
MAX_SUBPROCESS_OUTPUT_BYTES = 1_048_576
MAX_TEMP_DISK_BYTES = 1_073_741_824
MAX_IN_MEMORY_ARTIFACT_BYTES = 67_108_864
TERMINATION_GRACE_SECONDS = 10
MAX_LEDGER_EVENT_BYTES = 65_536

SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ATTEMPT_ID_RE = re.compile(r"^lisa1:[0-9a-f]{32}$")
UTC_TIMESTAMP_RE = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$"
)


class CustomerVerdict(str, Enum):
    PASS = "PASS"
    REGRESSION = "REGRESSION"
    INCONCLUSIVE = "INCONCLUSIVE"
    UNSUPPORTED = "UNSUPPORTED"
    HARNESS_ERROR = "HARNESS_ERROR"


class ExecutionPolicy(str, Enum):
    DEFAULT = "default"
    REQUIRE_SUPPORTED = "require_supported"


class WorkflowClassification(str, Enum):
    DEVELOPMENT_DEBUGGING = "development_debugging"
    VERIFICATION_ACCEPTANCE = "verification_acceptance"


class StageState(str, Enum):
    EXECUTED = "executed"
    NOT_APPLICABLE = "not_applicable"
    BLOCKED = "blocked"
    FAILED = "failed"


class AggregationAction(str, Enum):
    CONTINUE = "continue"
    STOP = "stop"
    INHERIT = "inherit"
    RETAIN_PROVEN_REGRESSION = "retain_proven_regression_and_block_localization"
    NOT_APPLICABLE = "not_applicable"


class CleanupStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"
    RETAINED_DEBUG = "retained_debug"
    NOT_APPLICABLE = "not_applicable"


class ResidueStatus(str, Enum):
    NONE_OBSERVED = "none_observed"
    PRESENT = "present"
    UNKNOWN = "unknown"
    RETAINED_DEBUG = "retained_debug"


class LedgerEvent(str, Enum):
    ATTEMPT_STARTED = "attempt_started"
    STAGE_STARTED = "stage_started"
    STAGE_FINISHED = "stage_finished"
    CLEANUP_OBSERVED = "cleanup_observed"
    ATTEMPT_FINISHED = "attempt_finished"


DEFAULT_EXIT_CODES = {
    CustomerVerdict.PASS: 0,
    CustomerVerdict.UNSUPPORTED: 0,
    CustomerVerdict.HARNESS_ERROR: 2,
    CustomerVerdict.INCONCLUSIVE: 3,
    CustomerVerdict.REGRESSION: 4,
}

REQUIRE_SUPPORTED_EXIT_CODES = {
    **DEFAULT_EXIT_CODES,
    CustomerVerdict.UNSUPPORTED: 6,
}

SIGNAL_EXIT_CODES = {"SIGINT": 130, "SIGTERM": 143}

CLEANUP_RESIDUE_PAIRS = {
    CleanupStatus.SUCCESS: (ResidueStatus.NONE_OBSERVED,),
    CleanupStatus.FAILED: (ResidueStatus.PRESENT, ResidueStatus.UNKNOWN),
    CleanupStatus.PARTIAL: (ResidueStatus.PRESENT, ResidueStatus.UNKNOWN),
    CleanupStatus.RETAINED_DEBUG: (ResidueStatus.RETAINED_DEBUG,),
    CleanupStatus.NOT_APPLICABLE: (ResidueStatus.UNKNOWN,),
}

CANONICAL_STAGES = (
    "preflight",
    "reference_original_execution",
    "candidate_original_execution",
    "pass0_calibration",
    "pass1_token_localization",
    "pass2_prefix_policy_reproduction",
    "pass3a_discovery",
    "bounded_recapture",
    "pass3b_authoritative_localization",
    "pass4_intra_layer_localization",
    "aggregation",
    "cleanup",
)

STAGE_DEPENDENCIES = {
    "preflight": (),
    "reference_original_execution": ("preflight",),
    "candidate_original_execution": ("preflight",),
    "pass0_calibration": (
        "reference_original_execution",
        "candidate_original_execution",
    ),
    "pass1_token_localization": ("pass0_calibration",),
    "pass2_prefix_policy_reproduction": ("pass1_token_localization",),
    "pass3a_discovery": ("pass2_prefix_policy_reproduction",),
    "bounded_recapture": ("pass3a_discovery",),
    "pass3b_authoritative_localization": ("bounded_recapture",),
    "pass4_intra_layer_localization": (
        "pass3b_authoritative_localization",
    ),
    "aggregation": (
        "preflight",
        "reference_original_execution",
        "candidate_original_execution",
        "pass0_calibration",
        "pass1_token_localization",
    ),
    "cleanup": ("aggregation",),
}

CLI_MODES = {
    "demo": {
        "required_options": (),
        "forbidden_options": (
            "model",
            "reference_bin",
            "candidate_bin",
            "pass_number",
            "forced_prefix",
            "runtime_checkpoint_step",
            "target_layer",
            "intermediate_artifact",
        ),
        "offline": True,
        "input_source": "seeded_model_free_fixture",
    },
    "backend": {
        "required_options": ("model",),
        "forbidden_options": (
            "reference_bin",
            "candidate_bin",
            "pass_number",
            "forced_prefix",
            "runtime_checkpoint_step",
            "target_layer",
            "intermediate_artifact",
        ),
        "offline": True,
        "input_source": "supported_model_profile_direct_token_ids",
    },
    "runtime": {
        "required_options": ("reference_bin", "candidate_bin", "model"),
        "forbidden_options": (
            "pass_number",
            "forced_prefix",
            "runtime_checkpoint_step",
            "target_layer",
            "intermediate_artifact",
        ),
        "offline": True,
        "input_source": "supported_model_profile_direct_token_ids",
    },
}

CLI_COMMON_OPTIONS = (
    "out",
    "require_supported",
    "debug_retain",
    "stage_timeout_seconds",
    "verbose",
)

CLI_DEFAULTS = {
    "out": ".lis/verify",
    "network": "disabled",
    "telemetry": "disabled",
    "raw_text_retention": "disabled",
    "raw_tensor_retention": "disabled",
    "debug_retention": "disabled",
    "workflow_classification": "development_debugging",
    "batch_size": 1,
    "thread_count": 1,
    "generation_limit": 8,
    "input_source": "mode_contract",
    "selection_policy": "lis_policy_modified_greedy_v1",
}

SELECTION_POLICY_PROFILES = {
    "raw_greedy": {
        "domain": "lis.selection_policy/v1",
        "selection_mode": "raw_greedy",
        "repetition_penalty_decimal": "1.0",
        "structural_token_suppression": False,
    },
    "lis_policy_modified_greedy_v1": {
        "domain": "lis.selection_policy/v1",
        "selection_mode": "policy_modified_greedy",
        "repetition_penalty_decimal": "1.2",
        "structural_token_suppression": True,
    },
}

# Each Pass-local status has exactly one product aggregation action.  Values
# are (action, target).  A target may be a next stage, predecessor, verdict, or
# a finite partition name; there is deliberately no default/catch-all action.
PASS_STATUS_ACTIONS: dict[str, dict[str, tuple[str, str]]] = {
    "pass0": {
        "comparison_allowed": ("continue", "pass1_token_localization"),
        "limited_comparison_allowed": (
            "continue",
            "pass1_token_localization",
        ),
        "comparison_blocked": ("stop", "pass0_block_reason_partition"),
    },
    "pass1": {
        "comparison_blocked_by_pass0": ("inherit", "pass0_calibration"),
        "token_equivalent_on_observed_range": ("stop", "PASS"),
        "first_mismatch_found": (
            "continue",
            "pass2_prefix_policy_reproduction",
        ),
        "input_token_divergence": ("stop", "HARNESS_ERROR"),
        "selected_token_array_missing": ("stop", "HARNESS_ERROR"),
        "selected_token_identity_unverified": ("stop", "INCONCLUSIVE"),
        "unsupported_comparison": ("stop", "UNSUPPORTED"),
        "inconclusive": ("stop", "INCONCLUSIVE"),
    },
    "pass2": {
        "reproduction_verified": ("continue", "pass3a_discovery"),
        "comparison_blocked_by_pass0": ("inherit", "pass0_calibration"),
        "token_localization_not_available": (
            "inherit",
            "pass1_token_localization",
        ),
        "no_mismatch_to_reproduce": (
            "not_applicable",
            "pass1_token_localization",
        ),
        "source_binding_inconsistent": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
        "prefix_material_unavailable": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
        "prefix_reproduction_failed": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
        "decode_policy_reproduction_failed": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
        "checkpoint_step_mapping_mismatch": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
        "context_position_mismatch": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
        "unsupported_reproduction_mode": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
        "inconclusive": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
    },
    "pass3": {
        "observable_mismatch_found": ("continue", "pass3_role_partition"),
        "no_mismatch_in_captured_coverage": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
        "comparison_blocked_by_pass2": (
            "inherit",
            "pass2_prefix_policy_reproduction",
        ),
        "insufficient_common_coverage": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
        "source_binding_inconsistent": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
        "checkpoint_alignment_inconsistent": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
        "checkpoint_artifact_missing": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
        "checkpoint_summary_malformed": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
        "comparison_policy_unavailable": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
        "unsupported_checkpoint_layout": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
        "inconclusive": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
    },
    "pass4": {
        "observable_intra_layer_mismatch_found": ("stop", "REGRESSION"),
        "mismatch_bounded_to_inherited_closing_boundary": (
            "stop",
            "REGRESSION",
        ),
        "not_applicable": ("inherit", "pass3b_authoritative_localization"),
        "comparison_blocked_by_pass3": (
            "inherit",
            "pass3b_authoritative_localization",
        ),
        "insufficient_common_intra_layer_coverage": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
        "source_binding_inconsistent": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
        "checkpoint_alignment_inconsistent": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
        "checkpoint_summary_malformed": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
        "comparison_policy_unavailable": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
        "unsupported_parent": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
        "unsupported_intra_layer_layout": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
        "parent_revalidation_inconsistent": (
            "retain_proven_regression_and_block_localization",
            "pass1_token_localization",
        ),
    },
}

PASS0_BLOCK_REASON_VERDICTS = {
    "incompatible_decode_policy": CustomerVerdict.UNSUPPORTED,
    "input_token_divergence": CustomerVerdict.HARNESS_ERROR,
    "incompatible_model_family": CustomerVerdict.UNSUPPORTED,
    "config_fingerprint_mismatch": CustomerVerdict.HARNESS_ERROR,
    "external_oracle_ineligible": CustomerVerdict.UNSUPPORTED,
    "prompt_token_array_missing": CustomerVerdict.INCONCLUSIVE,
    "prompt_token_identity_unverified": CustomerVerdict.INCONCLUSIVE,
}

PASS3_ROLE_TRANSITIONS = {
    "pass3a_discovery": "bounded_recapture",
    "pass3b_authoritative_localization": "pass4_intra_layer_localization",
}

EVIDENCE_NONCLAIMS = (
    "tensor_equality",
    "numeric_divergence_confirmed",
    "first_divergence_confirmed",
    "whole_runtime_equivalence",
)

REPORT_TOP_LEVEL_FIELDS = (
    "schema",
    "kind",
    "report_version",
    "attempt",
    "command",
    "verdict",
    "reason_codes",
    "policy_result",
    "identities",
    "token_comparison",
    "localization",
    "coverage",
    "numeric_confirmation",
    "evidence",
    "stages",
    "next_action",
    "warnings",
    "cleanup",
)

IDENTITY_FIELDS = (
    "source_sha256",
    "binary_sha256",
    "model_sha256",
    "config_sha256",
    "input_sha256",
    "runtime_sha256",
    "backend_sha256",
)

FORCED_PREFIX_FIELDS = (
    "mode",
    "applied",
    "token_count",
    "token_ids_sha256",
    "prefix_start_generated_step",
    "prefix_end_generated_step_exclusive",
    "target_generated_token_step",
    "runtime_checkpoint_step",
    "prompt_token_count",
    "context_position",
    "selection_policy",
    "selection_policy_sha256",
    "source_pass0_artifact_sha256",
    "source_original_run_report_sha256",
    "source_pass1_artifact_sha256",
    "source_localization_ref_sha256",
)

LEDGER_EVENT_FIELDS = (
    "sequence",
    "attempt_id",
    "workflow_classification",
    "event",
    "timestamp_utc",
    "payload",
)

LEDGER_PAYLOAD_FIELDS = {
    LedgerEvent.ATTEMPT_STARTED: ("mode",),
    LedgerEvent.STAGE_STARTED: ("stage",),
    LedgerEvent.STAGE_FINISHED: ("stage", "state"),
    LedgerEvent.CLEANUP_OBSERVED: ("residue_status",),
    LedgerEvent.ATTEMPT_FINISHED: ("verdict",),
}

PROHIBITED_REPORT_KEYS = frozenset(
    {
        "raw_tensor_values",
        "raw_prompt_text",
        "raw_generated_text",
        "model_path",
        "temporary_artifact_path",
        "absolute_temporary_path",
        "full_forced_prefix_token_ids",
    }
)


def _require_exact_keys(value: Any, expected: tuple[str, ...], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label} must contain exactly the frozen fields")
    return value


def _require_string(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    if len(value.encode("utf-8")) > maximum:
        raise ValueError(f"{label} exceeds its UTF-8 byte bound")
    return value


def _require_int(value: Any, label: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{label} exceeds its bound")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a canonical sha256 identity")
    return value


def _require_bounded_string_list(
    value: Any,
    label: str,
    *,
    maximum_items: int,
    maximum_bytes: int,
) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum_items:
        raise ValueError(f"{label} exceeds its collection bound")
    if len(value) != len(set(value)):
        raise ValueError(f"{label} must not contain duplicates")
    for index, item in enumerate(value):
        _require_string(item, f"{label}[{index}]", maximum_bytes)
    return value


def expected_exit_code(
    verdict: CustomerVerdict,
    policy: ExecutionPolicy,
) -> int:
    if policy == ExecutionPolicy.DEFAULT:
        return DEFAULT_EXIT_CODES[verdict]
    return REQUIRE_SUPPORTED_EXIT_CODES[verdict]


def runtime_status_values() -> dict[str, tuple[str, ...]]:
    return {
        "pass0": tuple(value.value for value in Pass0Verdict),
        "pass1": tuple(value.value for value in Pass1Status),
        "pass2": tuple(value.value for value in Pass2Status),
        "pass3": tuple(value.value for value in Pass3Status),
        "pass4": tuple(value.value for value in Pass4Status),
    }


def validate_status_mapping(raw: Any) -> None:
    if not isinstance(raw, dict) or set(raw) != set(PASS_STATUS_ACTIONS):
        raise ValueError("status mapping must contain exactly Pass 0 through Pass 4")
    runtime = runtime_status_values()
    valid_actions = {value.value for value in AggregationAction}
    for pass_name, expected_entries in PASS_STATUS_ACTIONS.items():
        entries = raw[pass_name]
        if not isinstance(entries, list):
            raise ValueError(f"{pass_name} mapping must be a list")
        actual_statuses = [entry.get("status") for entry in entries]
        if tuple(actual_statuses) != runtime[pass_name]:
            raise ValueError(f"{pass_name} status mapping is missing, reordered, or stale")
        if len(actual_statuses) != len(set(actual_statuses)):
            raise ValueError(f"{pass_name} status mapping contains duplicates")
        for entry in entries:
            if set(entry) != {"status", "action", "target"}:
                raise ValueError("status mapping entries have an unknown field")
            expected_action, expected_target = expected_entries[entry["status"]]
            if entry["action"] not in valid_actions:
                raise ValueError("unknown aggregation action")
            if (entry["action"], entry["target"]) != (
                expected_action,
                expected_target,
            ):
                raise ValueError("status mapping disagrees with the frozen contract")


def validate_forced_prefix_metadata(raw: Any) -> None:
    value = _require_exact_keys(raw, FORCED_PREFIX_FIELDS, "forced_prefix")
    if value["mode"] != "injected_selected_token_prefix_v1":
        raise ValueError("forced_prefix.mode is unsupported")
    if value["applied"] is not True:
        raise ValueError("forced prefix report requires an applied prefix")
    count = _require_int(
        value["token_count"], "forced_prefix.token_count", MAX_TOKEN_ID_PREVIEW
    )
    if count == 0:
        raise ValueError("forced prefix report requires a non-empty prefix")
    start = _require_int(
        value["prefix_start_generated_step"], "prefix_start_generated_step"
    )
    end = _require_int(
        value["prefix_end_generated_step_exclusive"],
        "prefix_end_generated_step_exclusive",
    )
    target = _require_int(
        value["target_generated_token_step"], "target_generated_token_step"
    )
    runtime_step = _require_int(
        value["runtime_checkpoint_step"], "runtime_checkpoint_step"
    )
    prompt_count = _require_int(value["prompt_token_count"], "prompt_token_count")
    context_position = _require_int(value["context_position"], "context_position")
    if start != 0 or end != count or target != count:
        raise ValueError("forced prefix generated-step range is incoherent")
    if runtime_step != target + 1:
        raise ValueError("decode runtime checkpoint step must be N + 1")
    if context_position != prompt_count + count:
        raise ValueError("forced prefix context position is incoherent")
    if value["selection_policy"] not in {
        "raw_greedy",
        "lis_policy_modified_greedy_v1",
    }:
        raise ValueError("forced prefix selection policy is unsupported")
    if value["selection_policy_sha256"] != selection_policy_sha256(
        value["selection_policy"]
    ):
        raise ValueError("forced prefix selection policy digest is inconsistent")
    for field in (
        "token_ids_sha256",
        "selection_policy_sha256",
        "source_pass0_artifact_sha256",
        "source_original_run_report_sha256",
        "source_pass1_artifact_sha256",
        "source_localization_ref_sha256",
    ):
        _require_sha256(value[field], f"forced_prefix.{field}")


def _scan_privacy(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in PROHIBITED_REPORT_KEYS:
                raise ValueError(f"prohibited report field: {key}")
            _scan_privacy(child)
    elif isinstance(value, list):
        for child in value:
            _scan_privacy(child)
    elif isinstance(value, str) and (
        value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        raise ValueError("absolute paths are prohibited in the canonical report")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite JSON number is prohibited")


def validate_ledger_events(raw: Any) -> None:
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError("ledger requires at least start and finish events")
    attempt_id = None
    workflow = None
    previous_timestamp = None
    event_counts = {value.value: 0 for value in LedgerEvent}
    started_stages: set[str] = set()
    finished_stages: set[str] = set()
    valid_events = {value.value for value in LedgerEvent}
    for expected_sequence, entry in enumerate(raw):
        value = _require_exact_keys(entry, LEDGER_EVENT_FIELDS, "ledger event")
        if value["sequence"] != expected_sequence or isinstance(
            value["sequence"], bool
        ):
            raise ValueError("ledger sequence must be contiguous from zero")
        if not isinstance(value["attempt_id"], str) or ATTEMPT_ID_RE.fullmatch(
            value["attempt_id"]
        ) is None:
            raise ValueError("ledger attempt identity is not canonical")
        try:
            WorkflowClassification(value["workflow_classification"])
        except (TypeError, ValueError) as exc:
            raise ValueError("ledger workflow classification is unknown") from exc
        if value["event"] not in valid_events:
            raise ValueError("ledger event type is unknown")
        event = LedgerEvent(value["event"])
        event_counts[event.value] += 1
        if not isinstance(value["timestamp_utc"], str) or UTC_TIMESTAMP_RE.fullmatch(
            value["timestamp_utc"]
        ) is None:
            raise ValueError("ledger timestamp must be RFC3339 UTC seconds")
        if previous_timestamp is not None and value["timestamp_utc"] < previous_timestamp:
            raise ValueError("ledger timestamps must be monotonic")
        previous_timestamp = value["timestamp_utc"]
        if not isinstance(value["payload"], dict):
            raise ValueError("ledger payload must be an object")
        payload = _require_exact_keys(
            value["payload"], LEDGER_PAYLOAD_FIELDS[event], "ledger payload"
        )
        if event == LedgerEvent.ATTEMPT_STARTED:
            if payload["mode"] not in CLI_MODES:
                raise ValueError("ledger attempt mode is unknown")
        elif event == LedgerEvent.STAGE_STARTED:
            if payload["stage"] not in CANONICAL_STAGES:
                raise ValueError("ledger stage is unknown")
            if payload["stage"] in started_stages:
                raise ValueError("ledger stage retry requires a new attempt identity")
            started_stages.add(payload["stage"])
        elif event == LedgerEvent.STAGE_FINISHED:
            if payload["stage"] not in started_stages:
                raise ValueError("ledger stage finished before it started")
            if payload["stage"] in finished_stages:
                raise ValueError("ledger stage finished more than once")
            try:
                StageState(payload["state"])
            except (TypeError, ValueError) as exc:
                raise ValueError("ledger stage terminal state is unknown") from exc
            finished_stages.add(payload["stage"])
        elif event == LedgerEvent.CLEANUP_OBSERVED:
            try:
                ResidueStatus(payload["residue_status"])
            except (TypeError, ValueError) as exc:
                raise ValueError("ledger residue status is unknown") from exc
        elif payload["verdict"] not in {value.value for value in CustomerVerdict}:
            raise ValueError("ledger final verdict is unknown")
        _scan_privacy(value["payload"])
        if len(canonical_json_bytes(value)) > MAX_LEDGER_EVENT_BYTES:
            raise ValueError("ledger event exceeds its byte bound")
        if attempt_id is None:
            attempt_id = value["attempt_id"]
            workflow = value["workflow_classification"]
        elif (
            value["attempt_id"] != attempt_id
            or value["workflow_classification"] != workflow
        ):
            raise ValueError("ledger identity changed within one attempt")
    if raw[0]["event"] != LedgerEvent.ATTEMPT_STARTED.value:
        raise ValueError("ledger must start with attempt_started")
    if raw[-1]["event"] != LedgerEvent.ATTEMPT_FINISHED.value:
        raise ValueError("ledger must end with attempt_finished")
    if event_counts[LedgerEvent.ATTEMPT_STARTED.value] != 1:
        raise ValueError("ledger must contain exactly one attempt_started")
    if event_counts[LedgerEvent.ATTEMPT_FINISHED.value] != 1:
        raise ValueError("ledger must contain exactly one attempt_finished")
    if event_counts[LedgerEvent.CLEANUP_OBSERVED.value] != 1:
        raise ValueError("ledger must contain exactly one cleanup observation")
    if started_stages != finished_stages:
        raise ValueError("every started ledger stage must finish in the same attempt")


def _validate_identity_pair(raw: Any) -> None:
    value = _require_exact_keys(raw, ("reference", "candidate"), "identities")
    for role in ("reference", "candidate"):
        identity = _require_exact_keys(value[role], IDENTITY_FIELDS, role)
        for field in IDENTITY_FIELDS:
            _require_sha256(identity[field], f"{role}.{field}")


def _validate_stages(raw: Any) -> None:
    if not isinstance(raw, list) or len(raw) != len(CANONICAL_STAGES):
        raise ValueError("stages must contain the exact canonical stage count")
    names = [entry.get("name") if isinstance(entry, dict) else None for entry in raw]
    if tuple(names) != CANONICAL_STAGES:
        raise ValueError("stages must use the exact canonical order")
    states: dict[str, str] = {}
    for entry in raw:
        value = _require_exact_keys(
            entry,
            (
                "name",
                "state",
                "result_ref",
                "evidence_tier",
                "failure_class",
                "reason",
                "blocker",
            ),
            "stage",
        )
        try:
            state = StageState(value["state"])
        except (TypeError, ValueError) as exc:
            raise ValueError("stage has an unknown terminal state") from exc
        states[value["name"]] = state.value
        if state == StageState.EXECUTED:
            _require_sha256(value["result_ref"], "stage.result_ref")
            _require_string(
                value["evidence_tier"],
                "stage.evidence_tier",
                MAX_IDENTIFIER_BYTES,
            )
            if value["reason"] is not None or value["blocker"] is not None:
                raise ValueError("executed stage cannot carry reason or blocker")
            if value["failure_class"] is not None:
                raise ValueError("executed stage cannot carry a failure class")
        elif state == StageState.NOT_APPLICABLE:
            _require_string(value["reason"], "stage.reason", MAX_DETAIL_BYTES)
            if (
                value["result_ref"] is not None
                or value["evidence_tier"] is not None
                or value["failure_class"] is not None
                or value["blocker"] is not None
            ):
                raise ValueError("not_applicable stage fields are incoherent")
        elif state == StageState.BLOCKED:
            _require_string(value["reason"], "stage.reason", MAX_DETAIL_BYTES)
            if value["blocker"] not in CANONICAL_STAGES:
                raise ValueError("blocked stage requires a canonical blocker")
            if (
                value["result_ref"] is not None
                or value["evidence_tier"] is not None
                or value["failure_class"] is not None
            ):
                raise ValueError("blocked stage cannot carry a result")
        else:
            _require_string(
                value["failure_class"],
                "stage.failure_class",
                MAX_IDENTIFIER_BYTES,
            )
            _require_string(value["reason"], "stage.reason", MAX_DETAIL_BYTES)
            if (
                value["result_ref"] is not None
                or value["evidence_tier"] is not None
                or value["blocker"] is not None
            ):
                raise ValueError("failed stage fields are incoherent")

    for stage, dependencies in STAGE_DEPENDENCIES.items():
        if states[stage] != StageState.EXECUTED.value:
            continue
        if stage == "aggregation":
            # Aggregation must be able to serialize any terminal failure state.
            continue
        for dependency in dependencies:
            if states[dependency] != StageState.EXECUTED.value:
                raise ValueError("executed stage has a failed or blocked dependency")


def _validate_token_comparison(raw: Any) -> None:
    value = _require_exact_keys(
        raw,
        (
            "status",
            "reference_observed_count",
            "candidate_observed_count",
            "first_mismatch",
        ),
        "token_comparison",
    )
    if value["status"] not in {"equal", "mismatch", "unavailable"}:
        raise ValueError("token_comparison.status is unknown")
    _require_int(value["reference_observed_count"], "reference_observed_count")
    _require_int(value["candidate_observed_count"], "candidate_observed_count")
    mismatch = value["first_mismatch"]
    if value["status"] == "mismatch":
        mismatch_value = _require_exact_keys(
            mismatch,
            ("generated_token_step", "reference_token_id", "candidate_token_id"),
            "first_mismatch",
        )
        for field in mismatch_value:
            _require_int(mismatch_value[field], f"first_mismatch.{field}")
    elif mismatch is not None:
        raise ValueError("non-mismatch token result cannot carry first_mismatch")


def _validate_coverage(raw: Any) -> None:
    value = _require_exact_keys(
        raw,
        (
            "scope",
            "common_layers",
            "missing_reference_layers",
            "missing_candidate_layers",
            "common_intra_layer_stages",
            "missing_reference_intra_layer_stages",
            "missing_candidate_intra_layer_stages",
        ),
        "coverage",
    )
    _require_string(value["scope"], "coverage.scope", MAX_IDENTIFIER_BYTES)
    for field in (
        "common_layers",
        "missing_reference_layers",
        "missing_candidate_layers",
    ):
        items = value[field]
        if not isinstance(items, list) or len(items) > MAX_LAYER_COLLECTION:
            raise ValueError(f"coverage.{field} exceeds its bound")
        if len(items) != len(set(items)):
            raise ValueError(f"coverage.{field} contains duplicates")
        for item in items:
            _require_int(item, f"coverage.{field}", MAX_LAYER_COLLECTION - 1)
    for field in (
        "common_intra_layer_stages",
        "missing_reference_intra_layer_stages",
        "missing_candidate_intra_layer_stages",
    ):
        _require_bounded_string_list(
            value[field],
            f"coverage.{field}",
            maximum_items=MAX_INTRA_LAYER_STAGES,
            maximum_bytes=MAX_IDENTIFIER_BYTES,
        )


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def selection_policy_sha256(name: str) -> str:
    try:
        profile = SELECTION_POLICY_PROFILES[name]
    except (KeyError, TypeError) as exc:
        raise ValueError("selection policy is unsupported") from exc
    return "sha256:" + hashlib.sha256(canonical_json_bytes(profile)).hexdigest()


def validate_report(raw: Any) -> None:
    value = _require_exact_keys(raw, REPORT_TOP_LEVEL_FIELDS, "report")
    if value["schema"] != SCHEMA or value["kind"] != KIND:
        raise ValueError("report identity is unsupported")
    if value["report_version"] != REPORT_VERSION:
        raise ValueError("report version is unsupported")
    try:
        verdict = CustomerVerdict(value["verdict"])
    except (TypeError, ValueError) as exc:
        raise ValueError("unknown customer verdict") from exc
    _require_bounded_string_list(
        value["reason_codes"],
        "reason_codes",
        maximum_items=MAX_REASON_CODES,
        maximum_bytes=MAX_IDENTIFIER_BYTES,
    )
    if not value["reason_codes"]:
        raise ValueError("reason_codes must contain at least one final reason")

    attempt = _require_exact_keys(
        value["attempt"],
        ("id", "workflow_classification"),
        "attempt",
    )
    if not isinstance(attempt["id"], str) or ATTEMPT_ID_RE.fullmatch(
        attempt["id"]
    ) is None:
        raise ValueError("attempt.id is not canonical")
    try:
        WorkflowClassification(attempt["workflow_classification"])
    except (TypeError, ValueError) as exc:
        raise ValueError("unknown workflow classification") from exc

    command = _require_exact_keys(
        value["command"],
        ("mode", "require_supported", "output_path_redacted"),
        "command",
    )
    if command["mode"] not in CLI_MODES:
        raise ValueError("unknown command mode")
    if not isinstance(command["require_supported"], bool):
        raise ValueError("require_supported must be boolean")
    if command["output_path_redacted"] is not True:
        raise ValueError("output path must be redacted")

    policy_result = _require_exact_keys(
        value["policy_result"],
        ("policy", "satisfied", "exit_code", "reason"),
        "policy_result",
    )
    try:
        policy = ExecutionPolicy(policy_result["policy"])
    except (TypeError, ValueError) as exc:
        raise ValueError("unknown execution policy") from exc
    if command["require_supported"] != (
        policy == ExecutionPolicy.REQUIRE_SUPPORTED
    ):
        raise ValueError("command and policy_result disagree")
    exit_code = expected_exit_code(verdict, policy)
    if policy_result["exit_code"] != exit_code:
        raise ValueError("policy_result exit code contradicts semantic verdict")
    if policy_result["satisfied"] is not (exit_code == 0):
        raise ValueError("policy_result satisfied flag contradicts exit code")
    _require_string(
        policy_result["reason"], "policy_result.reason", MAX_DETAIL_BYTES
    )

    _validate_identity_pair(value["identities"])
    _validate_token_comparison(value["token_comparison"])

    localization = _require_exact_keys(
        value["localization"],
        ("layer_suspect_interval", "intra_layer_suspect_interval"),
        "localization",
    )
    for field in localization:
        child = localization[field]
        if child is not None:
            _require_string(child, f"localization.{field}", MAX_DETAIL_BYTES)

    _validate_coverage(value["coverage"])

    numeric = _require_exact_keys(
        value["numeric_confirmation"],
        ("status", "confirmed_divergence_at_checkpoint", "confirmed_first_divergence"),
        "numeric_confirmation",
    )
    if numeric["status"] not in {
        "not_performed",
        "incomplete",
    }:
        raise ValueError("unknown numeric confirmation status")
    if numeric["confirmed_first_divergence"] is not None:
        raise ValueError("confirmed_first_divergence must remain null in core v1")
    if numeric["confirmed_divergence_at_checkpoint"] is not None:
        raise ValueError(
            "core v1 cannot carry confirmed checkpoint divergence"
        )

    evidence = _require_exact_keys(
        value["evidence"],
        ("tier", "ceiling", "nonclaims"),
        "evidence",
    )
    _require_string(evidence["tier"], "evidence.tier", MAX_IDENTIFIER_BYTES)
    _require_string(
        evidence["ceiling"], "evidence.ceiling", MAX_IDENTIFIER_BYTES
    )
    nonclaims = _require_exact_keys(
        evidence["nonclaims"], EVIDENCE_NONCLAIMS, "evidence.nonclaims"
    )
    if any(nonclaims[field] is not False for field in EVIDENCE_NONCLAIMS):
        raise ValueError("bounded evidence nonclaims must all remain false")

    _validate_stages(value["stages"])

    if verdict == CustomerVerdict.PASS:
        if value["next_action"] is not None:
            raise ValueError("PASS must not require a next action")
        if value["token_comparison"]["status"] != "equal":
            raise ValueError("PASS requires equal selected tokens")
    else:
        next_action = _require_exact_keys(
            value["next_action"], ("code", "summary"), "next_action"
        )
        _require_string(
            next_action["code"], "next_action.code", MAX_IDENTIFIER_BYTES
        )
        _require_string(
            next_action["summary"],
            "next_action.summary",
            MAX_NEXT_ACTION_BYTES,
        )
    if verdict == CustomerVerdict.REGRESSION and value["token_comparison"][
        "status"
    ] != "mismatch":
        raise ValueError("REGRESSION requires a selected-token mismatch in core v1")

    _require_bounded_string_list(
        value["warnings"],
        "warnings",
        maximum_items=MAX_WARNINGS,
        maximum_bytes=MAX_DETAIL_BYTES,
    )

    cleanup = _require_exact_keys(
        value["cleanup"],
        ("status", "residue_status", "observed", "retained_debug"),
        "cleanup",
    )
    try:
        cleanup_status = CleanupStatus(cleanup["status"])
        residue_status = ResidueStatus(cleanup["residue_status"])
    except (TypeError, ValueError) as exc:
        raise ValueError("unknown cleanup or residue status") from exc
    if not isinstance(cleanup["observed"], bool) or not isinstance(
        cleanup["retained_debug"], bool
    ):
        raise ValueError("cleanup booleans are malformed")
    if residue_status not in CLEANUP_RESIDUE_PAIRS[cleanup_status]:
        raise ValueError("cleanup and residue status combination is prohibited")
    if cleanup_status == CleanupStatus.SUCCESS and (
        not cleanup["observed"]
        or residue_status != ResidueStatus.NONE_OBSERVED
        or cleanup["retained_debug"]
    ):
        raise ValueError("successful cleanup requires observed zero residue")
    if cleanup_status == CleanupStatus.RETAINED_DEBUG and (
        residue_status != ResidueStatus.RETAINED_DEBUG
        or not cleanup["retained_debug"]
    ):
        raise ValueError("retained debug cleanup state is incoherent")
    if cleanup_status != CleanupStatus.RETAINED_DEBUG and cleanup[
        "retained_debug"
    ]:
        raise ValueError("retained_debug flag contradicts cleanup status")
    if cleanup_status in {CleanupStatus.FAILED, CleanupStatus.PARTIAL} and (
        residue_status == ResidueStatus.NONE_OBSERVED
    ):
        raise ValueError("failed or partial cleanup cannot claim zero residue")

    _scan_privacy(value)
    if len(canonical_json_bytes(value)) > MAX_REPORT_BYTES:
        raise ValueError("canonical report exceeds its total byte bound")
