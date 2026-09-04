from __future__ import annotations

import hashlib
from importlib import resources
import json
import unittest

from lis_verify.demo_fixture import (
    DemoFixtureError,
    RESOURCE_FILES,
    load_demo_fixture,
)


def resource_map() -> dict[str, bytes]:
    root = resources.files("lis_verify.demo_data")
    names = ["manifest_v1.json", *RESOURCE_FILES.values()]
    return {name: root.joinpath(name).read_bytes() for name in names}


def replace_resource(bundle: dict[str, bytes], name: str, data: bytes) -> None:
    bundle[name] = data
    manifest = json.loads(bundle["manifest_v1.json"])
    logical = next(
        key for key, filename in RESOURCE_FILES.items() if filename == name
    )
    manifest["resources"][logical]["sha256"] = (
        "sha256:" + hashlib.sha256(data).hexdigest()
    )
    bundle["manifest_v1.json"] = json.dumps(manifest).encode("utf-8")


class TestDemoFixture(unittest.TestCase):
    def test_packaged_bundle_is_complete_and_bound(self):
        fixture = load_demo_fixture()
        self.assertEqual(fixture.fixture_id, "lis-seeded-mismatch-240517")
        self.assertEqual(fixture.fixture_version, 1)
        self.assertEqual(fixture.value("profile")["seed"], 240517)
        self.assertEqual(
            set(name for name, _ in fixture._canonical_values),
            set(RESOURCE_FILES),
        )

    def test_byte_corruption_fails_before_parsing(self):
        bundle = resource_map()
        bundle["candidate_original.json"] += b" "
        with self.assertRaisesRegex(DemoFixtureError, "digest mismatch"):
            load_demo_fixture(bundle.__getitem__)

    def test_truncated_bound_resource_fails_closed(self):
        bundle = resource_map()
        replace_resource(bundle, "reference_original.json", b'{"schema":')
        with self.assertRaisesRegex(DemoFixtureError, "not valid JSON"):
            load_demo_fixture(bundle.__getitem__)

    def test_duplicate_json_key_fails_closed(self):
        bundle = resource_map()
        replace_resource(bundle, "profile_v1.json", b'{"schema":1,"schema":2}')
        with self.assertRaisesRegex(DemoFixtureError, "duplicate JSON key"):
            load_demo_fixture(bundle.__getitem__)

    def test_missing_resource_fails_closed(self):
        bundle = resource_map()
        del bundle["intra_layer_trace.json"]
        with self.assertRaisesRegex(DemoFixtureError, "unavailable"):
            load_demo_fixture(bundle.__getitem__)

    def test_stale_or_reused_generation_binding_is_rejected(self):
        bundle = resource_map()
        profile = json.loads(bundle["profile_v1.json"])
        profile["generations"]["authoritative"][
            "reference_original_artifact_set_id"
        ] = profile["generations"]["discovery"][
            "reference_original_artifact_set_id"
        ]
        replace_resource(
            bundle, "profile_v1.json", json.dumps(profile).encode("utf-8")
        )
        with self.assertRaisesRegex(DemoFixtureError, "reused"):
            load_demo_fixture(bundle.__getitem__)


if __name__ == "__main__":
    unittest.main()
