"""Render tests for the Artifact tab. Includes retention enforcement checks."""

from __future__ import annotations

import unittest
from pathlib import Path

from lis_inspect.cli import build_session
from lis_inspect.views.artifact import render_artifact


FIXTURES = Path(__file__).resolve().parent.parent / "test_fixtures"


# Strings the Artifact tab must NEVER surface (retention rule).
RETENTION_FORBIDDEN_SUBSTRINGS = (
    "Write one short",   # raw prompt text from any documentation example
    "Hello.",            # raw prompt text from another doc example
    "/home/",            # absolute paths
    "/tmp/",             # absolute paths
)


class ArtifactRenderTest(unittest.TestCase):
    def test_renders_identity_block_with_canonical_schema(self) -> None:
        session = build_session(FIXTURES / "qwen3_1p7b_run.json", None)
        rendered = render_artifact(session)
        self.assertIn("Identity", rendered)
        self.assertIn("lis.execution_artifact/v1", rendered)
        self.assertIn("run_report", rendered)
        self.assertIn("qwen3_dense_decoder", rendered)
        self.assertIn("avx2", rendered)
        self.assertIn("token_ids", rendered)

    def test_renders_retention_policy_with_omitted_for_minimized_run(self) -> None:
        session = build_session(FIXTURES / "qwen3_1p7b_run.json", None)
        rendered = render_artifact(session)
        self.assertIn("Retention Policy", rendered)
        self.assertIn("absolute_paths", rendered)
        self.assertIn("raw_prompt_text", rendered)
        self.assertIn("generated_text", rendered)
        # All three retention values are False ⇒ each rendered as "omitted".
        # The string also appears in the no-IDs token-preview message, so do
        # not pin to an exact count — just verify each retention key has its
        # value shown alongside it.
        for key in ("absolute_paths", "raw_prompt_text", "generated_text"):
            with self.subTest(key=key):
                idx = rendered.find(key)
                self.assertNotEqual(idx, -1)
                # Look for "omitted" somewhere shortly after the key on the same line.
                tail = rendered[idx:idx + 80]
                self.assertIn("omitted", tail)

    def test_renders_fingerprints_block(self) -> None:
        session = build_session(FIXTURES / "qwen3_1p7b_run.json", None)
        rendered = render_artifact(session)
        self.assertIn("Fingerprints", rendered)
        self.assertIn("fnv1a64:c12d6ff3efd1ce93", rendered)  # binary
        self.assertIn("fnv1a64:ccd87757f0c4ea58", rendered)  # model
        self.assertIn("fnv1a64:0ccf42c9ad05cc49", rendered)  # config
        self.assertIn("fnv1a64:df48170cd33ef65c", rendered)  # input
        self.assertIn("fnv1a64:ea989e5e6f0b8cdf", rendered)  # runtime
        self.assertIn("fnv1a64:43d7bf68cb75ed71", rendered)  # backend

    def test_renders_token_accounting_with_selected_equals_emitted(self) -> None:
        session = build_session(FIXTURES / "qwen3_1p7b_run.json", None)
        rendered = render_artifact(session)
        self.assertIn("Token Accounting", rendered)
        self.assertIn("Selected == emitted", rendered)
        self.assertIn("yes", rendered)

    def test_token_preview_omits_ids_when_artifact_did_not_carry_them(self) -> None:
        # Minimal fixture has count + digest but no ID list.
        session = build_session(FIXTURES / "qwen3_1p7b_run.json", None)
        rendered = render_artifact(session)
        self.assertIn("Selected tokens", rendered)
        self.assertIn("Emitted tokens", rendered)
        self.assertIn("omitted by artifact", rendered)
        self.assertIn("fnv1a64:abcdef0123456789", rendered)

    def test_token_preview_renders_ids_when_artifact_carries_them(self) -> None:
        session = build_session(FIXTURES / "qwen3_1p7b_run_full.json", None)
        rendered = render_artifact(session)
        # 10 IDs ≤ 16 budget ⇒ all in head, no tail/omitted.
        self.assertIn("ids (preview)", rendered)
        self.assertIn("101, 202, 303, 404, 505, 606, 707, 808, 909, 1010", rendered)
        self.assertNotIn("hidden", rendered)

    def test_artifact_renders_canonical_json_note(self) -> None:
        session = build_session(FIXTURES / "qwen3_1p7b_run.json", None)
        rendered = render_artifact(session)
        self.assertIn("JSON remains the canonical", rendered)
        self.assertIn("Raw prompt text, generated text, and absolute", rendered)

    def test_no_retention_violations_in_rendered_output(self) -> None:
        for fixture in ("qwen3_1p7b_run.json", "qwen3_1p7b_run_full.json"):
            with self.subTest(fixture=fixture):
                session = build_session(FIXTURES / fixture, None)
                rendered = render_artifact(session)
                for forbidden in RETENTION_FORBIDDEN_SUBSTRINGS:
                    self.assertNotIn(
                        forbidden,
                        rendered,
                        msg=f"forbidden substring {forbidden!r} appeared in {fixture}",
                    )


class ArtifactRenderWithFullSessionTest(unittest.TestCase):
    def test_full_session_includes_per_token_aware_retention(self) -> None:
        session = build_session(
            FIXTURES / "qwen3_1p7b_run_full.json",
            FIXTURES / "qwen3_1p7b_run_full.stderr",
        )
        rendered = render_artifact(session)
        # Identity / retention / token / fingerprints sections all present.
        self.assertIn("Identity", rendered)
        self.assertIn("Retention Policy", rendered)
        self.assertIn("Token Accounting", rendered)
        self.assertIn("Fingerprints", rendered)
        # JSON remains canonical even when stderr is supplied.
        self.assertIn("JSON remains the canonical", rendered)


if __name__ == "__main__":
    unittest.main()
