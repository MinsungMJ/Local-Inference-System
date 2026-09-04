import os
from pathlib import Path
import stat
import tempfile
import unittest

from lis_verify.product_contract import CleanupStatus, ResidueStatus
from lis_verify.workspace import (
    AttemptWorkspace,
    WorkspaceError,
    detect_stale_attempts,
    new_attempt_id,
)


class TestWorkspace(unittest.TestCase):
    def test_attempt_identity_is_unique_and_canonical(self):
        identities = {new_attempt_id() for _ in range(256)}
        self.assertEqual(len(identities), 256)
        for identity in identities:
            self.assertRegex(identity, r"^lisa1:[0-9a-f]{32}$")

    def test_private_modes_and_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = AttemptWorkspace.create(Path(temporary) / "verify")
            self.assertEqual(stat.S_IMODE(workspace.output_root.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(workspace.attempt_dir.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(workspace.runtime_dir.stat().st_mode), 0o700)
            (workspace.runtime_dir / "bounded.bin").write_bytes(b"abc")
            self.assertEqual(workspace.temp_usage_bytes(), 3)
            cleanup = workspace.cleanup_runtime()
            self.assertEqual(cleanup.status, CleanupStatus.SUCCESS)
            self.assertEqual(cleanup.residue_status, ResidueStatus.NONE_OBSERVED)
            self.assertFalse(workspace.runtime_dir.exists())

    def test_debug_retention_is_not_zero_residue(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = AttemptWorkspace.create(Path(temporary) / "verify")
            cleanup = workspace.cleanup_runtime(debug_retain=True)
            self.assertEqual(cleanup.status, CleanupStatus.RETAINED_DEBUG)
            self.assertEqual(cleanup.residue_status, ResidueStatus.RETAINED_DEBUG)
            self.assertTrue(workspace.runtime_dir.exists())

    def test_attempt_collision_fails_without_reuse(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "verify"
            identity = "lisa1:" + "a" * 32
            AttemptWorkspace.create(root, attempt_id=identity)
            with self.assertRaisesRegex(WorkspaceError, "already exists"):
                AttemptWorkspace.create(root, attempt_id=identity)

    def test_symlink_output_component_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            target = base / "target"
            target.mkdir(mode=0o700)
            link = base / "link"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(WorkspaceError, "symlink"):
                AttemptWorkspace.create(link)

    def test_nonprivate_existing_output_root_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "verify"
            root.mkdir(mode=0o755)
            root.chmod(0o755)
            with self.assertRaisesRegex(WorkspaceError, "group or other"):
                AttemptWorkspace.create(root)

    def test_stale_detection_does_not_delete(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = AttemptWorkspace.create(Path(temporary) / "verify")
            stale = detect_stale_attempts(workspace.output_root)
            self.assertEqual(stale, (workspace.attempt_dir.name,))
            self.assertTrue(workspace.attempt_dir.exists())

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink support required")
    def test_runtime_symlink_is_not_followed_during_cleanup(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = AttemptWorkspace.create(Path(temporary) / "verify")
            workspace.runtime_dir.rmdir()
            external = Path(temporary) / "external"
            external.mkdir()
            workspace.runtime_dir.symlink_to(external, target_is_directory=True)
            cleanup = workspace.cleanup_runtime()
            self.assertEqual(cleanup.status, CleanupStatus.FAILED)
            self.assertTrue(external.exists())


if __name__ == "__main__":
    unittest.main()
