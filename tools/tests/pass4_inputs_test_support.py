#!/usr/bin/env python3
"""Real Pass 0-to-P4-3 builders for focused P4-8 parser tests."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from lis_verify.pass3 import run_coverage_scoped_layer_localization
from lis_verify.pass3_inputs import CanonicalLayerTrace
from lis_verify.pass4_parent import (
    CanonicalPass3Artifact,
    Pass4ParentBindingOutcome,
)

from . import pass4_parent_test_support


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "test_fixtures"
    / "intra_layer_localization"
    / "producer_intra_layer_trace_v1.json"
)


@dataclass(frozen=True)
class PreparedPass4Inputs:
    case: dict
    parent: Pass4ParentBindingOutcome
    reference_trace: CanonicalLayerTrace
    candidate_trace: CanonicalLayerTrace


def fixture_blocks() -> dict:
    """Return independent mutable copies of the committed producer blocks."""

    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _trace_with_fixture(base: dict) -> dict:
    raw = copy.deepcopy(base)
    blocks = fixture_blocks()
    raw["intra_layer_checkpoint_layout"] = blocks[
        "intra_layer_checkpoint_layout"
    ]
    raw["intra_layer_trace"] = blocks["intra_layer_trace"]
    return raw


def prepare_inputs(
    seed_case: Optional[dict] = None,
    *,
    mutate_reference: Optional[Callable[[dict], None]] = None,
    mutate_candidate: Optional[Callable[[dict], None]] = None,
) -> PreparedPass4Inputs:
    """Bind exact mutated traces through the real Pass 3B and P4-3 chain."""

    seed = seed_case or pass4_parent_test_support.two_generation_case()
    case = pass4_parent_test_support.cloned(seed)
    authoritative = case["authoritative"]

    reference_raw = _trace_with_fixture(authoritative["reference_raw"])
    candidate_raw = _trace_with_fixture(authoritative["candidate_raw"])
    if mutate_reference is not None:
        mutate_reference(reference_raw)
    if mutate_candidate is not None:
        mutate_candidate(candidate_raw)

    reference_trace = CanonicalLayerTrace.from_object(reference_raw)
    candidate_trace = CanonicalLayerTrace.from_object(candidate_raw)
    pass3 = run_coverage_scoped_layer_localization(
        authoritative["pass2"],
        authoritative["pass2_artifact"],
        reference_trace,
        candidate_trace,
        reference_source_report=authoritative["reference_report"],
        candidate_source_report=authoritative["candidate_report"],
    )
    authoritative.update(
        reference_raw=reference_raw,
        candidate_raw=candidate_raw,
        reference_trace=reference_trace,
        candidate_trace=candidate_trace,
        pass3=pass3,
        pass3_artifact=CanonicalPass3Artifact.from_result(pass3),
    )
    parent = pass4_parent_test_support.bind_case(case)
    return PreparedPass4Inputs(
        case,
        parent,
        reference_trace,
        candidate_trace,
    )
