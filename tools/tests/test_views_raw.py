"""Render and source-preview tests for the Raw tab."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lis_inspect.cli import build_session
from lis_inspect.source_preview import STDERR_PREVIEW_LINES
from lis_inspect.views.raw import render_raw


FIXTURES = Path(__file__).resolve().parent.parent / "test_fixtures"

RETENTION_FORBIDDEN_SUBSTRINGS = (
    "Write one short",
    "generated secret",
    "/home/",
    "/tmp/",
)


class RawRenderTest(unittest.TestCase):
    def test_json_only_renders_source_and_parser_status(self) -> None:
        session = build_session(FIXTURES / "qwen3_1p7b_run.json", None)
        rendered = render_raw(session)

        self.assertIn("Source Availability", rendered)
        self.assertIn("JSON parsed", rendered)
        self.assertIn("perf-stage parsed", rendered)
        self.assertIn("Completeness", rendered)
        self.assertIn("json_only", rendered)
        self.assertIn("stderr log not provided", rendered)
        self.assertIn("JSON Preview", rendered)

    def test_json_plus_stderr_renders_all_parser_sections(self) -> None:
        session = build_session(
            FIXTURES / "qwen3_1p7b_run_full.json",
            FIXTURES / "qwen3_1p7b_run_full.stderr",
        )
        rendered = render_raw(session)

        self.assertIn("json_perf_per_token", rendered)
        self.assertIn("perf-stage parsed", rendered)
        self.assertIn("perf-summary parsed", rendered)
        self.assertIn("per-token parsed", rendered)
        self.assertIn("lis: perf-stage", rendered)
        self.assertIn("lis: perf-per-token", rendered)
        self.assertIn("No parser warnings recorded", rendered)

    def test_missing_stderr_path_is_bounded_warning_without_path_leak(self) -> None:
        missing = Path("/tmp/lis_inspect_missing.stderr")
        session = build_session(FIXTURES / "qwen3_1p7b_run.json", missing)
        rendered = render_raw(session)

        self.assertIn("stderr-log not found", rendered)
        self.assertIn("degraded", rendered)
        self.assertNotIn(str(missing), rendered)
        self.assertNotIn("/tmp/", rendered)

    def test_malformed_stderr_fragment_surfaces_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stderr_path = Path(tmpdir) / "bad.stderr"
            stderr_path.write_text(
                "lis: perf-stage tag=t name=prefill ns=10 ms=1.0 tokens=2\n"
                "lis: perf-summary tag=t threads=bad prompt_tokens=2\n"
                "lis: perf-per-token tag=t step=zero ns=1 ms=1.0\n",
                encoding="utf-8",
            )
            session = build_session(FIXTURES / "qwen3_1p7b_run.json", stderr_path)
            rendered = render_raw(session)

        self.assertIn("degraded", rendered)
        self.assertIn("malformed perf-summary", rendered)
        self.assertIn("malformed perf-per-token", rendered)
        self.assertIn("perf-summary parsed", rendered)

    def test_stderr_preview_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            stderr_path = Path(tmpdir) / "long.stderr"
            lines = [
                "lis: perf-stage tag=t name=prefill ns=10 ms=1.0 tokens=2",
                "lis: perf-summary tag=t threads=1 prompt_tokens=2 generated_tokens=1 "
                "ttft_ms=1.0 itl_ms=1.0 tps_steady=1.0 tps_end_to_end=1.0",
            ]
            lines.extend(f"unrelated line {idx}" for idx in range(STDERR_PREVIEW_LINES + 10))
            stderr_path.write_text("\n".join(lines), encoding="utf-8")
            session = build_session(FIXTURES / "qwen3_1p7b_run.json", stderr_path)
            rendered = render_raw(session)

        self.assertIn("truncated; showing first", rendered)
        self.assertNotIn(f"unrelated line {STDERR_PREVIEW_LINES + 9}", rendered)

    def test_raw_preview_redacts_retention_sensitive_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "sensitive.json"
            raw = json.loads((FIXTURES / "qwen3_1p7b_run.json").read_text(encoding="utf-8"))
            raw["manifest"]["source_path"] = "/private/path/secret/model"
            raw["report"]["prompt_text"] = "Write one short hidden prompt"
            raw["report"]["generated_text"] = "generated secret"
            artifact_path.write_text(json.dumps(raw), encoding="utf-8")

            session = build_session(artifact_path, None)
            rendered = render_raw(session)

        for forbidden in RETENTION_FORBIDDEN_SUBSTRINGS:
            self.assertNotIn(forbidden, rendered)
        self.assertIn("<redacted by retention policy>", rendered)
        self.assertIn("<absolute-path-redacted>", rendered)


if __name__ == "__main__":
    unittest.main()
