"""Unit tests for parse_json.load_artifact."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from lis_inspect.parse_json import (
    EXPECTED_KIND,
    EXPECTED_SCHEMA,
    ArtifactParseError,
    load_artifact,
)


FIXTURE = Path(__file__).resolve().parent.parent / "test_fixtures" / "qwen3_1p7b_run.json"


class LoadArtifactValidTest(unittest.TestCase):
    def test_loads_qwen3_fixture(self) -> None:
        artifact = load_artifact(FIXTURE)
        self.assertEqual(artifact.schema, EXPECTED_SCHEMA)
        self.assertEqual(artifact.kind, EXPECTED_KIND)
        self.assertEqual(artifact.execution_status, "OK")
        self.assertEqual(artifact.stop_reason, "decode_limit")
        self.assertEqual(artifact.model_family, "qwen3_dense_decoder")
        self.assertEqual(artifact.backend, "avx2")
        self.assertEqual(artifact.runtime_settings.get("threads"), 1)


class LoadArtifactRejectsTest(unittest.TestCase):
    def _write(self, payload) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(payload, tmp)
        tmp.close()
        return Path(tmp.name)

    def test_rejects_missing_file(self) -> None:
        with self.assertRaises(ArtifactParseError) as cm:
            load_artifact(Path("/no/such/path.json"))
        self.assertIn("not found", str(cm.exception))

    def test_rejects_non_json(self) -> None:
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        tmp.write("not json {")
        tmp.close()
        with self.assertRaises(ArtifactParseError) as cm:
            load_artifact(Path(tmp.name))
        self.assertIn("not valid JSON", str(cm.exception))

    def test_rejects_non_object_root(self) -> None:
        path = self._write([1, 2, 3])
        with self.assertRaises(ArtifactParseError) as cm:
            load_artifact(path)
        self.assertIn("must be an object", str(cm.exception))

    def test_rejects_wrong_schema(self) -> None:
        path = self._write({"schema": "other/v9", "kind": "run_report"})
        with self.assertRaises(ArtifactParseError) as cm:
            load_artifact(path)
        self.assertIn("unsupported schema", str(cm.exception))

    def test_rejects_wrong_kind(self) -> None:
        path = self._write({"schema": EXPECTED_SCHEMA, "kind": "benchmark_snapshot"})
        with self.assertRaises(ArtifactParseError) as cm:
            load_artifact(path)
        self.assertIn("unsupported kind", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
