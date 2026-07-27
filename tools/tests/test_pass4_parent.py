#!/usr/bin/env python3
"""Focused loading, identity, binding, and precedence tests for P4-3."""

from __future__ import annotations

import copy
import inspect
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from lis_verify import pass4_contract
from lis_verify.pass1_inputs import (
    CanonicalRunReport,
    canonical_json,
    sha256_text,
)
from lis_verify.pass3_inputs import (
    CanonicalLayerTrace,
    CanonicalPass2Artifact,
)
from lis_verify.pass3 import run_coverage_scoped_layer_localization
from lis_verify.pass3_model import Pass3Status
from lis_verify.pass3_model import SummaryEvidenceLevel
from lis_verify.pass4_contract import (
    ParentSourceIdentity,
    Pass3ParentClassification,
    Pass3ParentRole,
    Pass4ReasonCode,
    Pass4Status,
)
from lis_verify import pass4_parent
from lis_verify.pass4_parent import (
    MAX_ARRAY_ITEMS,
    MAX_JSON_DEPTH,
    MAX_OBJECT_KEYS,
    CanonicalPass3Artifact,
    Pass4ArtifactLoadError,
    Pass4LoadedArtifactIdentities,
    Pass4ParentBindingOutcome,
    bind_pass4_parent_inputs,
    load_bounded_object,
    load_bounded_text,
    load_canonical_layer_trace,
    load_canonical_pass2_artifact,
    load_canonical_run_report,
)
from lis_verify.pass4_model import Pass3ParentEvidence, Pass4SourceBinding

from . import pass4_parent_test_support as support


class Pass4ParentCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = support.two_generation_case()

    def bind(self, **overrides):
        return support.bind_case(self.case, **overrides)


class TestPass4StrictLoading(Pass4ParentCase):
    def _write(self, directory: str, name: str, payload: bytes) -> Path:
        path = Path(directory) / name
        path.write_bytes(payload)
        return path

    def test_all_canonical_artifact_wrappers_load_under_their_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            discovery = self.case["discovery"]
            authoritative = self.case["authoritative"]
            paths = {
                "pass3": self._write(
                    directory,
                    "pass3.json",
                    discovery["pass3_artifact"].canonical_text.encode(),
                ),
                "pass2": self._write(
                    directory,
                    "pass2.json",
                    authoritative["pass2_artifact"].canonical_text.encode(),
                ),
                "report": self._write(
                    directory,
                    "report.json",
                    authoritative["reference_report"].canonical_text.encode(),
                ),
                "trace": self._write(
                    directory,
                    "trace.json",
                    authoritative["reference_trace"].canonical_text.encode(),
                ),
            }
            self.assertEqual(
                CanonicalPass3Artifact.load(paths["pass3"]).artifact_sha256,
                discovery["pass3_artifact"].artifact_sha256,
            )
            self.assertEqual(
                load_canonical_pass2_artifact(paths["pass2"]).artifact_sha256,
                authoritative["pass2_artifact"].artifact_sha256,
            )
            self.assertEqual(
                load_canonical_run_report(paths["report"]).identity.run_report_sha256,
                authoritative["reference_report"].identity.run_report_sha256,
            )
            self.assertEqual(
                load_canonical_layer_trace(paths["trace"]).identity.trace_sha256,
                authoritative["reference_trace"].identity.trace_sha256,
            )

    def test_missing_paths_are_operational_classified_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.json"
            cases = (
                (
                    CanonicalPass3Artifact.load,
                    Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
                    Pass4ReasonCode.PARENT_ARTIFACT_MALFORMED,
                ),
                (
                    load_canonical_pass2_artifact,
                    Pass4Status.COMPARISON_BLOCKED_BY_PASS3,
                    Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,
                ),
                (
                    load_canonical_run_report,
                    Pass4Status.SOURCE_BINDING_INCONSISTENT,
                    Pass4ReasonCode.RUN_REPORT_BINDING_MISMATCH,
                ),
                (
                    load_canonical_layer_trace,
                    Pass4Status.SOURCE_BINDING_INCONSISTENT,
                    Pass4ReasonCode.TRACE_SHA_MISMATCH,
                ),
            )
            for loader, status, reason in cases:
                with self.subTest(loader=loader.__name__):
                    with self.assertRaises(Pass4ArtifactLoadError) as caught:
                        loader(missing)
                    self.assertEqual(caught.exception.status, status)
                    self.assertEqual(caught.exception.reason, reason)

    def test_directory_fifo_and_symlink_are_rejected_without_reading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.json"
            valid.write_text("{}", encoding="utf-8")
            symlink = root / "link.json"
            symlink.symlink_to(valid)
            fifo = root / "fifo"
            os.mkfifo(fifo)
            for path in (root, symlink, fifo):
                with self.subTest(path=path.name):
                    with self.assertRaises(Pass4ArtifactLoadError):
                        load_bounded_text(path, limit=64, label="fixture")

    def test_size_utf8_and_bom_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            oversized = self._write(directory, "large.json", b"x" * 65)
            invalid_utf8 = self._write(
                directory, "invalid.json", b"\xff\xfe{"
            )
            bom = self._write(
                directory, "bom.json", b"\xef\xbb\xbf{}"
            )
            with self.assertRaisesRegex(Pass4ArtifactLoadError, "byte bound"):
                load_bounded_text(oversized, limit=64, label="fixture")
            with self.assertRaisesRegex(Pass4ArtifactLoadError, "UTF-8"):
                load_bounded_text(invalid_utf8, limit=64, label="fixture")
            with self.assertRaisesRegex(Pass4ArtifactLoadError, "BOM"):
                load_bounded_text(bom, limit=64, label="fixture")

    def test_duplicate_keys_nonstandard_numbers_and_nonobject_roots_reject(self):
        payloads = (
            '{"a":1,"a":2}',
            '{"a":{"b":1,"b":2}}',
            '{"a":NaN}',
            '{"a":Infinity}',
            "[]",
            '"text"',
            "1",
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, payload in enumerate(payloads):
                path = Path(directory) / f"{index}.json"
                path.write_text(payload, encoding="utf-8")
                with self.subTest(payload=payload):
                    with self.assertRaises(Pass4ArtifactLoadError):
                        load_bounded_object(
                            path, limit=1024, label="fixture"
                        )

    def test_recursion_error_is_contained_at_the_bounded_boundary(self):
        deep = '{"a":' * 10_000 + "0" + "}" * 10_000
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deep.json"
            path.write_text(deep, encoding="utf-8")
            with self.assertRaises(Pass4ArtifactLoadError) as caught:
                load_bounded_object(
                    path, limit=128 * 1024, label="fixture"
                )
            self.assertEqual(
                caught.exception.reason,
                Pass4ReasonCode.PARENT_ARTIFACT_MALFORMED,
            )
        with mock.patch.object(
            pass4_parent, "strict_json_loads", side_effect=RecursionError
        ):
            with self.assertRaises(Pass4ArtifactLoadError):
                CanonicalPass3Artifact.from_json("{}")

    def test_depth_and_width_bounds_are_iterative_and_exact(self):
        accepted = '{"a":' * (MAX_JSON_DEPTH - 1) + "0" + "}" * (
            MAX_JSON_DEPTH - 1
        )
        rejected = '{"a":' * MAX_JSON_DEPTH + "0" + "}" * MAX_JSON_DEPTH
        with tempfile.TemporaryDirectory() as directory:
            accepted_path = Path(directory) / "accepted.json"
            rejected_path = Path(directory) / "rejected.json"
            accepted_path.write_text(accepted, encoding="utf-8")
            rejected_path.write_text(rejected, encoding="utf-8")
            self.assertIsInstance(
                load_bounded_object(
                    accepted_path,
                    limit=1024,
                    label="fixture",
                ),
                dict,
            )
            with self.assertRaisesRegex(Pass4ArtifactLoadError, "depth"):
                load_bounded_object(
                    rejected_path,
                    limit=1024,
                    label="fixture",
                )

            wide_object = Path(directory) / "wide-object.json"
            wide_object.write_text(
                json.dumps(
                    {f"k{index}": index for index in range(MAX_OBJECT_KEYS + 1)}
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Pass4ArtifactLoadError, "key-count"):
                load_bounded_object(
                    wide_object,
                    limit=1024 * 1024,
                    label="fixture",
                )

            wide_array = Path(directory) / "wide-array.json"
            wide_array.write_text(
                json.dumps({"a": [0] * (MAX_ARRAY_ITEMS + 1)}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(Pass4ArtifactLoadError, "item-count"):
                load_bounded_object(
                    wide_array,
                    limit=1024 * 1024,
                    label="fixture",
                )

    def test_invalid_path_shape_is_not_silently_normalized(self):
        with self.assertRaises(Pass4ArtifactLoadError):
            load_bounded_text("", limit=10, label="fixture")
        with self.assertRaises(Pass4ArtifactLoadError):
            load_bounded_text("bad\0path", limit=10, label="fixture")
        with self.assertRaises(TypeError):
            load_bounded_text(object(), limit=10, label="fixture")
        with self.assertRaises(TypeError):
            load_bounded_text("unused", limit=True, label="fixture")


class TestPass4CanonicalIdentity(Pass4ParentCase):
    def test_identity_is_existing_canonical_json_sha256(self):
        artifact = self.case["discovery"]["pass3_artifact"]
        raw = artifact.materialize_verified()
        self.assertEqual(
            artifact.artifact_sha256,
            sha256_text(canonical_json(raw)),
        )

    def test_reindentation_and_key_order_do_not_change_identity(self):
        raw = self.case["discovery"]["pass3_artifact"].materialize_verified()
        compact = CanonicalPass3Artifact.from_json(
            json.dumps(raw, separators=(",", ":"))
        )
        reindented = CanonicalPass3Artifact.from_json(
            json.dumps(raw, indent=4)
        )
        reordered = CanonicalPass3Artifact.from_object(
            dict(reversed(tuple(raw.items())))
        )
        self.assertEqual(compact.artifact_sha256, reindented.artifact_sha256)
        self.assertEqual(compact.artifact_sha256, reordered.artifact_sha256)

    def test_value_mutation_changes_identity_and_breaks_typed_coherence(self):
        wrapper = support.wrapper_from_mutated_parent(
            self.case,
            "discovery",
            lambda raw: raw["target"].__setitem__(
                "runtime_checkpoint_step",
                raw["target"]["runtime_checkpoint_step"] + 1,
            ),
        )
        self.assertNotEqual(
            wrapper.artifact_sha256,
            self.case["discovery"]["pass3_artifact"].artifact_sha256,
        )
        outcome = self.bind(discovery_pass3_artifact=wrapper)
        self.assertEqual(
            outcome.reason_codes,
            (Pass4ReasonCode.PARENT_TYPED_ARTIFACT_INCOHERENT,),
        )

    def test_unknown_or_missing_parent_fields_are_not_normalized(self):
        mutations = (
            lambda raw: raw.__setitem__("unknown", True),
            lambda raw: raw.pop("target"),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                wrapper = support.wrapper_from_mutated_parent(
                    self.case, "discovery", mutation
                )
                outcome = self.bind(discovery_pass3_artifact=wrapper)
                self.assertEqual(
                    outcome.reason_codes,
                    (Pass4ReasonCode.PARENT_TYPED_ARTIFACT_INCOHERENT,),
                )

    def test_bool_float_and_string_cannot_substitute_for_parent_integer(self):
        for value in (True, 18.0, "18"):
            wrapper = support.wrapper_from_mutated_parent(
                self.case,
                "discovery",
                lambda raw, value=value: raw["target"].__setitem__(
                    "runtime_checkpoint_step", value
                ),
            )
            outcome = self.bind(discovery_pass3_artifact=wrapper)
            self.assertFalse(outcome.proceed)
            self.assertEqual(
                outcome.status, Pass4Status.COMPARISON_BLOCKED_BY_PASS3
            )

    def test_self_inconsistent_sha_is_classified_not_raised(self):
        valid = self.case["discovery"]["pass3_artifact"]
        inconsistent = CanonicalPass3Artifact(
            "sha256:" + "0" * 64, valid.canonical_text
        )
        outcome = self.bind(discovery_pass3_artifact=inconsistent)
        self.assertEqual(
            outcome.reason_codes,
            (Pass4ReasonCode.PARENT_ARTIFACT_MALFORMED,),
        )

    def test_malformed_canonical_sha_syntax_is_classified_not_raised(self):
        valid = self.case["discovery"]["pass3_artifact"]
        malformed = (
            "sha256:" + "0" * 63,
            "sha256:" + "0" * 65,
            "sha256:" + "A" * 64,
            "SHA256:" + "0" * 64,
            "0" * 64,
            "",
        )
        for value in malformed:
            with self.subTest(value=value):
                wrapper = CanonicalPass3Artifact(value, valid.canonical_text)
                outcome = self.bind(discovery_pass3_artifact=wrapper)
                self.assertEqual(
                    outcome.reason_codes,
                    (Pass4ReasonCode.PARENT_ARTIFACT_MALFORMED,),
                )


class TestPass4ParentRolesAndModels(Pass4ParentCase):
    def test_complete_valid_binding_constructs_only_p4_2_models(self):
        outcome = self.bind()
        self.assertTrue(outcome.proceed)
        self.assertIsInstance(outcome, Pass4ParentBindingOutcome)
        self.assertIsInstance(outcome.parent, Pass3ParentEvidence)
        self.assertIsInstance(outcome.reference_binding, Pass4SourceBinding)
        self.assertIsInstance(outcome.candidate_binding, Pass4SourceBinding)
        self.assertIsInstance(
            outcome.artifact_identities, Pass4LoadedArtifactIdentities
        )
        self.assertEqual(
            outcome.parent.classification, Pass3ParentClassification.ELIGIBLE
        )
        self.assertEqual(
            outcome.parent.discovery.role, Pass3ParentRole.DISCOVERY_PASS3A
        )
        self.assertFalse(
            outcome.parent.discovery.authorizes_pass4_evidence
        )
        self.assertEqual(
            outcome.parent.authoritative.role,
            Pass3ParentRole.AUTHORITATIVE_PASS3B,
        )
        self.assertTrue(outcome.parent.authorizes_pass4_evidence)
        self.assertEqual(outcome.target_layer, support.TARGET_LAYER)
        self.assertEqual(outcome.target_runtime_checkpoint_step, support.TARGET_STEP)

    def test_authoritative_binding_uses_parent_recorded_roles_and_identities(self):
        outcome = self.bind()
        raw = self.case["authoritative"][
            "pass3_artifact"
        ].materialize_verified()
        for side, binding in (
            ("reference", outcome.reference_binding),
            ("candidate", outcome.candidate_binding),
        ):
            parent = raw["source_binding"][side]
            self.assertEqual(binding.role, parent["role"])
            self.assertEqual(
                binding.identity,
                ParentSourceIdentity(
                    parent["run_report_sha256"],
                    parent["layer_trace_sha256"],
                    parent["semantic_manifest_sha256"],
                    parent["artifact_set_id"],
                ),
            )
            self.assertTrue(
                binding.parent_recorded_trace_binding_verified
            )

    def test_parent_recorded_wrong_side_role_is_rejected(self):
        authoritative = self.case["authoritative"]["pass3"]
        changed = replace(
            authoritative,
            reference_binding=replace(
                authoritative.reference_binding,
                role="candidate_reproduction",
            ),
        )
        outcome = self.bind(
            authoritative_pass3=changed,
            authoritative_pass3_artifact=CanonicalPass3Artifact.from_result(
                changed
            ),
        )
        self.assertEqual(
            outcome.reason_codes, (Pass4ReasonCode.SOURCE_ROLE_MISMATCH,)
        )
        self.assertEqual(
            outcome.parent.classification,
            Pass3ParentClassification.ELIGIBLE,
        )
        self.assertIsNone(outcome.reference_binding)

    def test_discovery_artifacts_in_authoritative_slots_never_proceed(self):
        outcome = self.bind(
            authoritative_pass3=self.case["discovery"]["pass3"],
            authoritative_pass3_artifact=self.case["discovery"][
                "pass3_artifact"
            ],
        )
        self.assertFalse(outcome.proceed)

    def test_valid_authoritative_no_mismatch_is_not_applicable(self):
        case = support.two_generation_case(
            authoritative_mismatch_layer=None
        )
        outcome = support.bind_case(case)
        self.assertEqual(outcome.status, Pass4Status.NOT_APPLICABLE)
        self.assertEqual(
            outcome.reason_codes,
            (Pass4ReasonCode.PARENT_HAS_NO_OBSERVED_MISMATCH,),
        )
        self.assertEqual(
            outcome.parent.classification,
            Pass3ParentClassification.NOT_APPLICABLE,
        )
        self.assertIsNone(outcome.reference_binding)

    def test_valid_blocked_pass3b_status_is_classified_after_coherence(self):
        authoritative = self.case["authoritative"]
        blocked = run_coverage_scoped_layer_localization(
            authoritative["pass2"],
            authoritative["pass2_artifact"],
            authoritative["reference_trace"],
            authoritative["candidate_trace"],
            reference_source_report=self.case["discovery"][
                "reference_report"
            ],
            candidate_source_report=authoritative["candidate_report"],
        )
        self.assertEqual(blocked.status, Pass3Status.SOURCE_BINDING_INCONSISTENT)
        outcome = self.bind(
            authoritative_pass3=blocked,
            authoritative_pass3_artifact=CanonicalPass3Artifact.from_result(
                blocked
            ),
        )
        self.assertEqual(
            outcome.status, Pass4Status.COMPARISON_BLOCKED_BY_PASS3
        )
        self.assertEqual(
            outcome.reason_codes,
            (Pass4ReasonCode.PARENT_STATUS_BLOCKED,),
        )
        self.assertIsNone(outcome.parent)

    def test_complete_but_truncated_parent_comparisons_are_rejected(self):
        discovery = self.case["discovery"]["pass3"]
        truncated = replace(
            discovery, comparisons=discovery.comparisons * 90
        )
        outcome = self.bind(
            discovery_pass3=truncated,
            discovery_pass3_artifact=CanonicalPass3Artifact.from_result(
                truncated
            ),
        )
        self.assertEqual(
            outcome.reason_codes,
            (Pass4ReasonCode.PARENT_COMPARISONS_TRUNCATED,),
        )

    def test_duplicate_parent_comparison_coordinate_is_rejected(self):
        discovery = self.case["discovery"]["pass3"]
        duplicated = replace(
            discovery,
            comparisons=(
                discovery.comparisons[0],
                discovery.comparisons[0],
                *discovery.comparisons[1:],
            ),
        )
        outcome = self.bind(
            discovery_pass3=duplicated,
            discovery_pass3_artifact=CanonicalPass3Artifact.from_result(
                duplicated
            ),
        )
        self.assertEqual(
            outcome.reason_codes,
            (Pass4ReasonCode.PARENT_LOCALIZATION_INCOHERENT,),
        )

    def test_ready_parent_requires_verified_upstream_binding_flags(self):
        discovery = self.case["discovery"]["pass3"]
        changed = replace(
            discovery,
            checkpoint_artifact_binding_verified=False,
        )
        outcome = self.bind(
            discovery_pass3=changed,
            discovery_pass3_artifact=CanonicalPass3Artifact.from_result(
                changed
            ),
        )
        self.assertEqual(
            outcome.reason_codes,
            (Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,),
        )

    def test_unsupported_parent_policy_precedes_drift(self):
        case = support.two_generation_case(
            authoritative_mismatch_layer=12
        )
        authoritative = case["authoritative"]["pass3"]
        changed = replace(
            authoritative,
            evidence_level=SummaryEvidenceLevel.TIER1_BOUNDED_EXACT,
        )
        outcome = support.bind_case(
            case,
            authoritative_pass3=changed,
            authoritative_pass3_artifact=CanonicalPass3Artifact.from_result(
                changed
            ),
        )
        self.assertEqual(outcome.status, Pass4Status.UNSUPPORTED_PARENT)
        self.assertEqual(
            outcome.reason_codes,
            (Pass4ReasonCode.PARENT_DIGEST_POLICY_UNSUPPORTED,),
        )

    def test_type_error_is_reserved_for_api_misuse(self):
        values = support.bind_kwargs(self.case)
        positional = (
            values.pop("discovery_pass3"),
            values.pop("discovery_pass3_artifact"),
            values.pop("authoritative_pass3"),
            values.pop("authoritative_pass3_artifact"),
            values.pop("pass2_artifact"),
        )
        for index in range(len(positional)):
            broken = list(positional)
            broken[index] = object()
            with self.subTest(index=index), self.assertRaises(TypeError):
                bind_pass4_parent_inputs(*broken, **values)
        with self.assertRaises(TypeError):
            bind_pass4_parent_inputs(*positional)


class TestPass4Pass2Binding(Pass4ParentCase):
    def test_authoritative_pass2_sha_is_exact(self):
        outcome = self.bind(
            pass2_artifact=self.case["discovery"]["pass2_artifact"]
        )
        self.assertEqual(
            outcome.status, Pass4Status.COMPARISON_BLOCKED_BY_PASS3
        )
        self.assertEqual(
            outcome.reason_codes,
            (Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,),
        )

    def test_optional_discovery_pass2_exact_binding_is_accepted(self):
        outcome = self.bind(
            discovery_pass2_artifact=self.case["discovery"][
                "pass2_artifact"
            ]
        )
        self.assertTrue(outcome.proceed)
        self.assertEqual(
            outcome.artifact_identities.discovery_pass2_artifact_sha256,
            self.case["discovery"]["pass2_artifact"].artifact_sha256,
        )

    def test_wrong_optional_discovery_pass2_is_blocked(self):
        outcome = self.bind(
            discovery_pass2_artifact=self.case["authoritative"][
                "pass2_artifact"
            ]
        )
        self.assertEqual(
            outcome.reason_codes,
            (Pass4ReasonCode.PARENT_UPSTREAM_EVIDENCE_INCOHERENT,),
        )

    def test_equal_pass2_sha_is_warning_only_not_failure(self):
        discovery = self.case["discovery"]["pass3"]
        authoritative_sha = self.case["authoritative"][
            "pass2_artifact"
        ].artifact_sha256
        changed = replace(
            discovery, pass2_artifact_sha256=authoritative_sha
        )
        outcome = self.bind(
            discovery_pass3=changed,
            discovery_pass3_artifact=CanonicalPass3Artifact.from_result(
                changed
            ),
        )
        self.assertTrue(outcome.proceed)
        self.assertIn(
            "pass4.warn.pass2_lineage_not_rebuilt", outcome.warnings
        )


class TestPass4SourceBinding(Pass4ParentCase):
    def _bind_with_authoritative_reference_integer(
        self,
        field: str,
        value,
        *,
        prompt_count: bool = False,
    ):
        authoritative = self.case["authoritative"]
        report_raw = copy.deepcopy(
            authoritative["reference_report"].materialize()
        )
        trace_raw = copy.deepcopy(authoritative["reference_raw"])
        if prompt_count:
            report_raw["report"]["prompt_sequences"][0]["token_count"] = value
        else:
            report_raw["manifest"]["runtime"][field] = value
            trace_raw["manifest"]["runtime"][field] = value
        report = CanonicalRunReport.from_object(report_raw)
        trace = CanonicalLayerTrace.from_object(trace_raw)

        pass2_raw = copy.deepcopy(
            authoritative["pass2_artifact"].materialize_verified()
        )
        pass2_raw["source_binding"][
            "reference_reproduction_sha256"
        ] = report.identity.run_report_sha256
        pass2_artifact = CanonicalPass2Artifact.from_object(pass2_raw)

        parent = authoritative["pass3"]
        parent = replace(
            parent,
            pass2_artifact_sha256=pass2_artifact.artifact_sha256,
            pass2_evidence=replace(
                parent.pass2_evidence,
                reference_reproduction_sha256=(
                    report.identity.run_report_sha256
                ),
            ),
            reference_binding=replace(
                parent.reference_binding,
                run_report_sha256=report.identity.run_report_sha256,
                trace_sha256=trace.identity.trace_sha256,
                semantic_manifest_sha256=(
                    trace.identity.semantic_manifest_sha256
                ),
            ),
        )
        return self.bind(
            authoritative_pass3=parent,
            authoritative_pass3_artifact=CanonicalPass3Artifact.from_result(
                parent
            ),
            pass2_artifact=pass2_artifact,
            authoritative_reference_report=report,
            authoritative_reference_trace=trace,
        )

    def test_run_report_content_mismatch(self):
        report = self.case["authoritative"]["reference_report"]
        raw = report.materialize()
        raw["unknown_additive_field"] = True
        changed = CanonicalRunReport.from_object(raw)
        outcome = self.bind(authoritative_reference_report=changed)
        self.assertEqual(
            outcome.reason_codes,
            (Pass4ReasonCode.RUN_REPORT_BINDING_MISMATCH,),
        )

    def test_trace_content_mismatch_wins_before_artifact_set(self):
        trace = self.case["authoritative"]["reference_trace"]
        raw = json.loads(trace.canonical_text)
        raw["unknown_additive_field"] = True
        changed = CanonicalLayerTrace.from_object(raw)
        self.assertEqual(
            changed.identity.artifact_set_id,
            trace.identity.artifact_set_id,
        )
        outcome = self.bind(authoritative_reference_trace=changed)
        self.assertEqual(
            outcome.reason_codes, (Pass4ReasonCode.TRACE_SHA_MISMATCH,)
        )

    def test_semantic_manifest_identity_is_derived_and_cross_checked(self):
        trace = self.case["authoritative"]["reference_trace"]
        identity = replace(
            trace.identity,
            semantic_manifest_sha256="sha256:" + "0" * 64,
        )
        changed = CanonicalLayerTrace(identity, trace.canonical_text)
        outcome = self.bind(authoritative_reference_trace=changed)
        self.assertEqual(
            outcome.reason_codes,
            (Pass4ReasonCode.SEMANTIC_MANIFEST_BINDING_MISMATCH,),
        )

    def test_artifact_set_id_is_association_evidence_only(self):
        trace = self.case["authoritative"]["reference_trace"]
        identity = replace(
            trace.identity,
            artifact_set_id="aset1:" + "0" * 32,
        )
        changed = CanonicalLayerTrace(identity, trace.canonical_text)
        outcome = self.bind(authoritative_reference_trace=changed)
        self.assertEqual(
            outcome.reason_codes,
            (Pass4ReasonCode.ARTIFACT_SET_BINDING_MISMATCH,),
        )

    def test_cross_side_swaps_fail_on_content_identity(self):
        authoritative = self.case["authoritative"]
        trace_outcome = self.bind(
            authoritative_reference_trace=authoritative["candidate_trace"],
            authoritative_candidate_trace=authoritative["reference_trace"],
        )
        self.assertEqual(
            trace_outcome.reason_codes,
            (Pass4ReasonCode.TRACE_SHA_MISMATCH,),
        )
        report_outcome = self.bind(
            authoritative_reference_report=authoritative["candidate_report"],
            authoritative_candidate_report=authoritative["reference_report"],
        )
        self.assertEqual(
            report_outcome.reason_codes,
            (Pass4ReasonCode.RUN_REPORT_BINDING_MISMATCH,),
        )

    def test_authoritative_capture_discriminator_is_mandatory(self):
        case = support.two_generation_case(authoritative_capture=False)
        outcome = support.bind_case(case)
        self.assertEqual(
            outcome.status, Pass4Status.SOURCE_BINDING_INCONSISTENT
        )
        self.assertEqual(
            outcome.reason_codes,
            (Pass4ReasonCode.RUNTIME_CAPTURE_IDENTITY_MISMATCH,),
        )

    def test_discovery_capture_fields_are_rejected(self):
        discovery = self.case["discovery"]
        authoritative = self.case["authoritative"]
        outcome = self.bind(
            discovery_reference_report=authoritative["reference_report"],
            discovery_reference_trace=authoritative["reference_trace"],
        )
        self.assertIn(
            outcome.reason_codes[0],
            (
                Pass4ReasonCode.RUN_REPORT_BINDING_MISMATCH,
                Pass4ReasonCode.TRACE_SHA_MISMATCH,
            ),
        )

    def test_source_failure_retains_eligible_internal_parent(self):
        changed = CanonicalLayerTrace.from_object(
            {
                **self.case["authoritative"]["reference_raw"],
                "unknown_additive_field": True,
            }
        )
        outcome = self.bind(authoritative_reference_trace=changed)
        self.assertEqual(
            outcome.status, Pass4Status.SOURCE_BINDING_INCONSISTENT
        )
        self.assertEqual(
            outcome.parent.classification,
            Pass3ParentClassification.ELIGIBLE,
        )
        self.assertTrue(outcome.parent.source_binding_verified)
        self.assertIsNone(outcome.reference_binding)

    def test_bound_integer_fields_reject_bool_float_and_string(self):
        for field in (
            "configured_context",
            "batch_size",
            "generation_limit",
            "thread_count",
        ):
            for value in (True, 1.0, "1"):
                with self.subTest(field=field, value=value):
                    outcome = self._bind_with_authoritative_reference_integer(
                        field, value
                    )
                    self.assertEqual(
                        outcome.status,
                        Pass4Status.SOURCE_BINDING_INCONSISTENT,
                    )
                    self.assertEqual(
                        outcome.reason_codes,
                        (
                            Pass4ReasonCode.RUNTIME_CAPTURE_IDENTITY_MISMATCH,
                        ),
                    )

    def test_bound_prompt_count_rejects_bool_float_and_string(self):
        for value in (True, 3.0, "3"):
            with self.subTest(value=value):
                outcome = self._bind_with_authoritative_reference_integer(
                    "token_count",
                    value,
                    prompt_count=True,
                )
                self.assertEqual(
                    outcome.status,
                    Pass4Status.SOURCE_BINDING_INCONSISTENT,
                )
                self.assertEqual(
                    outcome.reason_codes,
                    (Pass4ReasonCode.RUNTIME_CAPTURE_IDENTITY_MISMATCH,),
                )


class TestPass4CrossGenerationCoherence(Pass4ParentCase):
    def test_expected_generation_differences_do_not_claim_identity(self):
        outcome = self.bind()
        self.assertTrue(outcome.proceed)
        identities = outcome.artifact_identities
        self.assertNotEqual(
            identities.discovery_pass3_sha256,
            identities.authoritative_pass3_sha256,
        )
        self.assertNotEqual(
            identities.discovery_reference.run_report_sha256,
            identities.authoritative_reference.run_report_sha256,
        )
        self.assertNotEqual(
            identities.discovery_reference.layer_trace_sha256,
            identities.authoritative_reference.layer_trace_sha256,
        )

    def test_selected_layer_drift_is_distinct_and_retains_no_localization(self):
        case = support.two_generation_case(
            authoritative_mismatch_layer=12
        )
        outcome = support.bind_case(case)
        self.assertEqual(
            outcome.status, Pass4Status.PARENT_REVALIDATION_INCONSISTENT
        )
        self.assertEqual(
            outcome.reason_codes,
            (Pass4ReasonCode.DISCOVERY_REBOUND_LAYER_CHANGED,),
        )
        self.assertNotEqual(
            outcome.parent.discovery_selected_layer,
            outcome.parent.authoritative_selected_layer,
        )
        self.assertIsNone(outcome.parent.parent_first_mismatch_coordinate)
        self.assertIsNone(outcome.parent.parent_suspect_interval)

    def test_requested_coordinate_drift_is_semantic_not_layer_drift(self):
        case = support.two_generation_case(
            authoritative_layers=(0, 4, 8)
        )
        outcome = support.bind_case(case)
        self.assertEqual(
            outcome.reason_codes,
            (Pass4ReasonCode.DISCOVERY_REBOUND_SEMANTICS_CHANGED,),
        )
        self.assertEqual(
            outcome.diagnostics, ("coverage.checkpoint_layout_basis",)
        )

    def test_bound_manifest_tuple_drift_is_classified_after_binding(self):
        case = support.two_generation_case(
            authoritative_runtime_overrides={"thread_count": 2}
        )
        outcome = support.bind_case(case)
        self.assertEqual(
            outcome.status, Pass4Status.PARENT_REVALIDATION_INCONSISTENT
        )
        self.assertEqual(
            outcome.reason_codes,
            (Pass4ReasonCode.DISCOVERY_REBOUND_SEMANTICS_CHANGED,),
        )
        self.assertIn("thread_count", outcome.diagnostics[0])

    def test_absent_in_both_optional_fields_yield_bounded_warnings(self):
        outcome = self.bind()
        self.assertTrue(outcome.proceed)
        self.assertTrue(
            any(
                value.startswith(
                    "pass4.warn.cross_generation_field_unavailable:"
                )
                for value in outcome.warnings
            )
        )
        self.assertEqual(len(outcome.warnings), len(set(outcome.warnings)))


class TestPass4FailurePrecedence(Pass4ParentCase):
    def test_parent_coherence_precedes_source_binding(self):
        malformed = support.wrapper_from_mutated_parent(
            self.case,
            "discovery",
            lambda raw: raw.__setitem__("kind", "wrong"),
        )
        outcome = self.bind(
            discovery_pass3_artifact=malformed,
            authoritative_reference_trace=self.case["authoritative"][
                "candidate_trace"
            ],
        )
        self.assertEqual(
            outcome.reason_codes,
            (Pass4ReasonCode.PARENT_ARTIFACT_MALFORMED,),
        )

    def test_layer_drift_precedes_trace_binding(self):
        case = support.two_generation_case(
            authoritative_mismatch_layer=12
        )
        outcome = support.bind_case(
            case,
            authoritative_reference_trace=case["authoritative"][
                "candidate_trace"
            ],
        )
        self.assertEqual(
            outcome.reason_codes,
            (Pass4ReasonCode.DISCOVERY_REBOUND_LAYER_CHANGED,),
        )

    def test_trace_binding_precedes_manifest_tuple_drift(self):
        case = support.two_generation_case(
            authoritative_runtime_overrides={"thread_count": 2}
        )
        trace = case["authoritative"]["reference_trace"]
        raw = json.loads(trace.canonical_text)
        raw["unknown_additive_field"] = True
        outcome = support.bind_case(
            case,
            authoritative_reference_trace=CanonicalLayerTrace.from_object(raw),
        )
        self.assertEqual(
            outcome.reason_codes, (Pass4ReasonCode.TRACE_SHA_MISMATCH,)
        )

    def test_trace_sha_precedes_artifact_set_mismatch_within_side(self):
        trace = self.case["authoritative"]["reference_trace"]
        raw = json.loads(trace.canonical_text)
        raw["artifact_set_id"] = "aset1:" + "0" * 32
        raw["unknown_additive_field"] = True
        outcome = self.bind(
            authoritative_reference_trace=CanonicalLayerTrace.from_object(raw)
        )
        self.assertEqual(
            outcome.reason_codes, (Pass4ReasonCode.TRACE_SHA_MISMATCH,)
        )

    def test_reference_side_precedes_candidate_side(self):
        authoritative = self.case["authoritative"]
        ref_raw = json.loads(authoritative["reference_trace"].canonical_text)
        cand_raw = json.loads(authoritative["candidate_trace"].canonical_text)
        ref_raw["bad"] = 1
        cand_raw["bad"] = 1
        outcome = self.bind(
            authoritative_reference_trace=CanonicalLayerTrace.from_object(
                ref_raw
            ),
            authoritative_candidate_trace=CanonicalLayerTrace.from_object(
                cand_raw
            ),
        )
        self.assertEqual(
            outcome.reason_codes, (Pass4ReasonCode.TRACE_SHA_MISMATCH,)
        )
        self.assertEqual(outcome.diagnostics, ("pass3b_reference",))

    def test_negative_outcome_is_deterministic(self):
        trace = self.case["authoritative"]["reference_trace"]
        raw = json.loads(trace.canonical_text)
        raw["bad"] = 1
        changed = CanonicalLayerTrace.from_object(raw)
        observed = {
            (
                self.bind(
                    authoritative_reference_trace=changed
                ).status,
                self.bind(
                    authoritative_reference_trace=changed
                ).disposition,
                self.bind(
                    authoritative_reference_trace=changed
                ).reason_codes,
                self.bind(
                    authoritative_reference_trace=changed
                ).warnings,
                self.bind(
                    authoritative_reference_trace=changed
                ).diagnostics,
            )
            for _ in range(10)
        }
        self.assertEqual(len(observed), 1)

    def test_every_emitted_reason_is_frozen_and_primary(self):
        outcomes = (
            self.bind(
                pass2_artifact=self.case["discovery"]["pass2_artifact"]
            ),
            support.bind_case(
                support.two_generation_case(
                    authoritative_mismatch_layer=12
                )
            ),
            self.bind(
                authoritative_reference_trace=self.case["authoritative"][
                    "candidate_trace"
                ]
            ),
        )
        for outcome in outcomes:
            reason = outcome.reason_codes[0]
            self.assertIn(
                outcome.status,
                pass4_contract.REASON_ALLOWED_STATUSES[reason],
            )
            self.assertIn(
                reason, pass4_contract.PRIMARY_REASONS[outcome.status]
            )


class TestPass4StructuralBoundary(unittest.TestCase):
    def test_module_stays_inside_the_p4_3_boundary(self):
        source = inspect.getsource(pass4_parent)
        for prohibited in (
            "intra_layer_trace",
            "intra_layer_checkpoint_layout",
            "intra_layer_digest_sha256",
            "canonical_intra_layer_digest_stream",
            "logical_fp32_bits_from_view",
            "analyze_coverage(",
            "requested_coordinates(",
            "Pass4Result(",
            "hashlib",
            "def serialize(",
            "def to_json(",
        ):
            with self.subTest(prohibited=prohibited):
                self.assertNotIn(prohibited, source)

    def test_frozen_contract_objects_are_reused_by_identity(self):
        self.assertIs(pass4_parent.NONCLAIMS, pass4_contract.NONCLAIMS)
        self.assertIs(
            pass4_parent.INTRA_LAYER_STAGES,
            pass4_contract.INTRA_LAYER_STAGES,
        )
        self.assertIs(
            pass4_parent.STATUS_TO_DISPOSITION,
            pass4_contract.STATUS_TO_DISPOSITION,
        )
        self.assertIs(
            pass4_parent.validate_pass3_parent_pair,
            pass4_contract.validate_pass3_parent_pair,
        )

    def test_no_public_package_export_was_added(self):
        import lis_verify

        exported = set(getattr(lis_verify, "__all__", ()))
        self.assertNotIn("bind_pass4_parent_inputs", exported)
        self.assertFalse(hasattr(lis_verify, "bind_pass4_parent_inputs"))

    def test_diagnostics_do_not_leak_paths_or_content_identities(self):
        case = support.two_generation_case(
            authoritative_runtime_overrides={"thread_count": 2}
        )
        outcomes = (
            support.bind_case(case),
            support.bind_case(
                support.two_generation_case(
                    authoritative_mismatch_layer=12
                )
            ),
        )
        for outcome in outcomes:
            for value in outcome.warnings + outcome.diagnostics:
                self.assertNotIn("/", value)
                self.assertNotIn("sha256:", value)
                self.assertNotIn("aset1:", value)


if __name__ == "__main__":
    unittest.main()
