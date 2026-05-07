"""LIS Inspect KV cache compatibility tests.

These tests protect parser tolerance for additive KV cache report fields.
They intentionally do not add KV rendering or semantic interpretation.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from lis_inspect.parse_json import EXPECTED_KIND, EXPECTED_SCHEMA, load_artifact


def _kv_cache_payload() -> dict:
    return {
        "schema": EXPECTED_SCHEMA,
        "kind": EXPECTED_KIND,
        "manifest": {
            "runtime": {
                "configured_context": 128,
                "batch_size": 1,
                "generation_limit": 4,
                "thread_count": 1,
                "precision_path": "f32_accum;weights=bf16;kv=bf16",
            },
            "backend": {
                "name": "avx2",
                "fingerprint": {
                    "algorithm": "fnv1a64",
                    "hex": "43d7bf68cb75ed71",
                    "size_bytes": 12,
                },
            },
        },
        "report": {
            "execution_status": "ok",
            "status_code": "OK",
            "stop_reason": "decode_limit",
            "perf": {
                "tag": "none",
                "threads": 1,
                "prompt_tokens": 8,
                "generated_tokens": 4,
                "ttft_ms": 1.0,
                "itl_ms": 2.0,
                "tps_steady": 10.0,
                "tps_end_to_end": 8.0,
            },
            "kv_cache": {
                "scope": "run_local",
                "policy": {
                    "eviction_free": True,
                    "monotonic_growth": True,
                    "paging": False,
                    "offload": False,
                    "sliding_window": False,
                    "prefix_reuse": False,
                },
                "storage_dtype": "bf16",
                "max_tokens": 128,
                "used_tokens": 12,
                "bytes_per_token": 114688,
                "allocated_bytes": 14680064,
                "used_bytes": 1376256,
            },
        },
    }


class InspectCompatTest(unittest.TestCase):
    def _write_payload(self, payload: dict) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(payload, tmp)
        tmp.close()
        self.addCleanup(lambda: Path(tmp.name).unlink(missing_ok=True))
        return Path(tmp.name)

    def test_parse_json_accepts_run_report_with_kv_cache(self) -> None:
        payload = _kv_cache_payload()

        artifact = load_artifact(self._write_payload(payload))

        self.assertEqual(artifact.schema, EXPECTED_SCHEMA)
        self.assertEqual(artifact.kind, EXPECTED_KIND)

    def test_parse_json_accepts_run_report_without_kv_cache(self) -> None:
        payload = _kv_cache_payload()
        payload["report"].pop("kv_cache")

        artifact = load_artifact(self._write_payload(payload))

        self.assertEqual(artifact.schema, EXPECTED_SCHEMA)
        self.assertEqual(artifact.kind, EXPECTED_KIND)
        self.assertNotIn("kv_cache", artifact.raw["report"])

    def test_parse_json_preserves_raw_payload_with_kv_cache(self) -> None:
        payload = _kv_cache_payload()
        expected = copy.deepcopy(payload)

        artifact = load_artifact(self._write_payload(payload))

        self.assertEqual(artifact.raw, expected)
        self.assertEqual(
            artifact.raw["manifest"]["runtime"]["precision_path"],
            "f32_accum;weights=bf16;kv=bf16",
        )
        self.assertEqual(artifact.raw["manifest"]["backend"]["name"], "avx2")
        self.assertIn("perf", artifact.raw["report"])
        self.assertEqual(artifact.raw["report"]["perf"], expected["report"]["perf"])
        self.assertIn("kv_cache", artifact.raw["report"])
        self.assertEqual(
            artifact.raw["report"]["kv_cache"],
            expected["report"]["kv_cache"],
        )


if __name__ == "__main__":
    unittest.main()
