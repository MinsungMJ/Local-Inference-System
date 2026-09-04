import copy
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

from lis_verify.report_artifact import (
    ArtifactPublicationError,
    load_report,
    prepare_report_bundle,
    publish_report_bundle,
)
from lis_verify.report_model import VerificationReport
from lis_verify.summary import render_markdown
from lis_verify.workspace import AttemptWorkspace
from tools.tests.lis_verify_product_spine_test_support import load_example


class TestReportArtifact(unittest.TestCase):
    def _prepared(self, temporary, name="pass"):
        workspace = AttemptWorkspace.create(Path(temporary) / "verify")
        report = VerificationReport.from_dict(load_example(name))
        bundle = prepare_report_bundle(
            report,
            render_markdown(report),
            report_path=workspace.report_path,
            summary_path=workspace.summary_path,
        )
        return workspace, report, bundle

    def test_report_is_last_commit_marker_and_loads_strictly(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace, report, bundle = self._prepared(temporary)
            self.assertFalse(workspace.report_path.exists())
            self.assertFalse(workspace.summary_path.exists())
            publish_report_bundle(bundle)
            self.assertTrue(workspace.report_path.exists())
            self.assertTrue(workspace.summary_path.exists())
            self.assertEqual(load_report(workspace.report_path), report)
            self.assertEqual(stat.S_IMODE(workspace.report_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(workspace.summary_path.stat().st_mode), 0o600)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = AttemptWorkspace.create(Path(temporary) / "verify")
            workspace.report_path.write_text("existing")
            report = VerificationReport.from_dict(load_example("pass"))
            with self.assertRaises(ArtifactPublicationError):
                prepare_report_bundle(
                    report,
                    render_markdown(report),
                    report_path=workspace.report_path,
                    summary_path=workspace.summary_path,
                )
            self.assertEqual(workspace.report_path.read_text(), "existing")

    def test_summary_is_rolled_back_if_report_publish_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace, _, bundle = self._prepared(temporary)
            real_link = __import__("os").link

            def fail_report(source, target, *, follow_symlinks=True):
                if Path(target).name == "verification_report.json":
                    raise OSError("injected publish failure")
                return real_link(
                    source,
                    target,
                    follow_symlinks=follow_symlinks,
                )

            with mock.patch("lis_verify.report_artifact.os.link", side_effect=fail_report):
                with self.assertRaises(ArtifactPublicationError):
                    publish_report_bundle(bundle)
            self.assertFalse(workspace.report_path.exists())
            self.assertFalse(workspace.summary_path.exists())

    def test_noncanonical_report_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace = AttemptWorkspace.create(Path(temporary) / "verify")
            workspace.report_path.write_text("{}")
            workspace.report_path.chmod(0o600)
            with self.assertRaises(ValueError):
                load_report(workspace.report_path)

    def test_nonprivate_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            workspace, _, bundle = self._prepared(temporary)
            publish_report_bundle(bundle)
            workspace.report_path.chmod(0o644)
            with self.assertRaisesRegex(ArtifactPublicationError, "private"):
                load_report(workspace.report_path)


if __name__ == "__main__":
    unittest.main()
