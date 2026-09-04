from __future__ import annotations

from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from lis_verify.acceptance import (
    AcceptanceManifestError,
    freeze_acceptance_manifest,
    load_acceptance_manifest,
)
from lis_verify.cli import main, parse_command
from lis_verify.orchestrator import AcceptanceManifest, CommandRequest
from lis_verify.product_contract import WorkflowClassification, canonical_json_bytes
from lis_verify.provenance import BuildProvenance
from lis_verify.real_execution import RealExecutionError, ResolvedBinary
from lis_verify.real_pipeline import _validate_acceptance_candidate


def _raw_manifest() -> dict:
    return {
        "clean_state_observed": True,
        "commands_sha256": "sha256:" + "3" * 64,
        "dependency_sha256": "sha256:" + "2" * 64,
        "kind": "acceptance_manifest",
        "schema": "lis.verify.acceptance_manifest/v1",
        "source_revision": "1" * 40,
        "source_tree_sha256": "sha256:" + "1" * 64,
    }


def _write_manifest(path: Path, raw: dict | None = None) -> None:
    path.write_bytes(canonical_json_bytes(_raw_manifest() if raw is None else raw))
    path.chmod(0o600)


class TestAcceptanceManifest(unittest.TestCase):
    def test_private_canonical_manifest_loads(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "acceptance.json"
            _write_manifest(path)
            loaded = load_acceptance_manifest(path)
            self.assertTrue(loaded.clean_state_observed)
            self.assertEqual(loaded.source_revision, "1" * 40)

    def test_permissions_noncanonical_and_unknown_fields_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "acceptance.json"
            _write_manifest(path)
            path.chmod(0o644)
            with self.assertRaisesRegex(AcceptanceManifestError, "private"):
                load_acceptance_manifest(path)
            path.write_text(json.dumps(_raw_manifest(), indent=2) + "\n")
            path.chmod(0o600)
            with self.assertRaisesRegex(AcceptanceManifestError, "canonical"):
                load_acceptance_manifest(path)
            raw = _raw_manifest()
            raw["unknown"] = True
            _write_manifest(path, raw)
            with self.assertRaisesRegex(AcceptanceManifestError, "unknown"):
                load_acceptance_manifest(path)

    def test_freeze_requires_clean_source_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "tools").mkdir()
            (root / "srcs").mkdir()
            (root / "Makefile").write_text("all:\n")
            (root / "pyproject.toml").write_text("project\n")
            (root / "tools/requirements.txt").write_text("requirements\n")
            (root / "commands.yml").write_text("commands\n")
            output = root / "acceptance.json"
            with mock.patch(
                "lis_verify.acceptance._git",
                side_effect=["1" * 40, ""],
            ):
                digest = freeze_acceptance_manifest(
                    source_root=root,
                    output=output,
                    command_files=[Path("commands.yml")],
                )
            self.assertTrue(digest.startswith("sha256:"))
            self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            load_acceptance_manifest(output)
            with mock.patch(
                "lis_verify.acceptance._git",
                side_effect=["1" * 40, ""],
            ):
                with self.assertRaisesRegex(AcceptanceManifestError, "create"):
                    freeze_acceptance_manifest(
                        source_root=root,
                        output=output,
                        command_files=[Path("commands.yml")],
                    )

            dirty_output = root / "dirty.json"
            with mock.patch(
                "lis_verify.acceptance._git", side_effect=["1" * 40, " M file"]
            ):
                with self.assertRaisesRegex(AcceptanceManifestError, "tracked"):
                    freeze_acceptance_manifest(
                        source_root=root,
                        output=dirty_output,
                        command_files=[Path("commands.yml")],
                    )

    def test_real_acceptance_binds_candidate_revision_tree_and_clean_state(self):
        authority = AcceptanceManifest(
            source_revision="1" * 40,
            source_tree_sha256="sha256:" + "2" * 64,
            dependency_sha256="sha256:" + "3" * 64,
            commands_sha256="sha256:" + "4" * 64,
            clean_state_observed=True,
        )
        provenance = BuildProvenance(
            source_sha256=authority.source_tree_sha256,
            binary_sha256="sha256:" + "5" * 64,
            binary_size_bytes=1,
            revision=authority.source_revision,
            dirty=False,
            identity_sha256="sha256:" + "6" * 64,
            raw={},
        )
        context = SimpleNamespace(
            request=CommandRequest(
                mode="backend",
                model=Path("model"),
                output_root=Path("out"),
                workflow=WorkflowClassification.VERIFICATION_ACCEPTANCE,
                acceptance_manifest=authority,
            )
        )
        _validate_acceptance_candidate(
            context, ResolvedBinary(Path("lis"), provenance)
        )
        dirty = BuildProvenance(
            source_sha256=provenance.source_sha256,
            binary_sha256=provenance.binary_sha256,
            binary_size_bytes=1,
            revision=provenance.revision,
            dirty=True,
            identity_sha256=provenance.identity_sha256,
            raw={},
        )
        with self.assertRaisesRegex(RealExecutionError, "does not bind"):
            _validate_acceptance_candidate(
                context, ResolvedBinary(Path("lis"), dirty)
            )

    def test_cli_environment_switches_to_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "acceptance.json"
            _write_manifest(path)
            with mock.patch.dict(
                os.environ, {"LIS_VERIFY_ACCEPTANCE_MANIFEST": os.fspath(path)}
            ):
                request = parse_command(["demo"])
            self.assertEqual(
                request.workflow, WorkflowClassification.VERIFICATION_ACCEPTANCE
            )
            self.assertIsNotNone(request.acceptance_manifest)

            path.chmod(0o644)
            with mock.patch.dict(
                os.environ, {"LIS_VERIFY_ACCEPTANCE_MANIFEST": os.fspath(path)}
            ), redirect_stderr(io.StringIO()):
                self.assertEqual(main(["demo"]), 2)


if __name__ == "__main__":
    unittest.main()
