import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from lis_verify.product_contract import canonical_json_bytes
from lis_verify.provenance import (
    KIND,
    MAX_PROVENANCE_BYTES,
    SCHEMA,
    ProvenanceError,
    build_provenance,
    load_build_provenance,
    sidecar_path,
    source_tree_identity,
    validate_provenance,
    write_build_provenance,
)


class ProvenanceTestCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "srcs" / "core").mkdir(parents=True)
        (self.root / "srcs" / "includes" / "lis").mkdir(parents=True)
        (self.root / "Makefile").write_text("all:\n\t@true\n")
        (self.root / "srcs" / "core" / "sample.c").write_text("int sample(void){return 1;}\n")
        (self.root / "srcs" / "includes" / "lis" / "sample.h").write_text(
            "int sample(void);\n"
        )
        self.binary = self.root / "lis"
        self.binary.write_bytes(b"LIS-BINARY\0v1")
        self.raw = build_provenance(
            binary=self.binary,
            source_root=self.root,
            compiler="cc",
            cppflags="-Isrcs/includes",
            cflags="-std=c11 -O2",
            ldflags="(none)",
            ldlibs="-lm -lpthread",
            simd="on",
        )
        write_build_provenance(sidecar_path(self.binary), self.raw)

    def tearDown(self):
        self.temp.cleanup()

    def test_round_trip_binds_source_and_binary(self):
        value = load_build_provenance(self.binary)
        self.assertEqual(value.raw["schema"], SCHEMA)
        self.assertEqual(value.raw["kind"], KIND)
        self.assertEqual(value.source_sha256, self.raw["source"]["tree_sha256"])
        self.assertEqual(
            value.binary_sha256,
            "sha256:" + hashlib.sha256(self.binary.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            value.identity_sha256,
            "sha256:"
            + hashlib.sha256(canonical_json_bytes(self.raw)).hexdigest(),
        )

    def test_source_identity_is_path_content_and_order_bound(self):
        first, count = source_tree_identity(self.root)
        self.assertEqual(count, 3)
        (self.root / "srcs" / "core" / "sample.c").write_text(
            "int sample(void){return 2;}\n"
        )
        second, second_count = source_tree_identity(self.root)
        self.assertEqual(second_count, count)
        self.assertNotEqual(first, second)

    def test_tampered_binary_is_rejected(self):
        self.binary.write_bytes(b"LIS-BINARY\0v2")
        with self.assertRaisesRegex(ProvenanceError, "does not bind"):
            load_build_provenance(self.binary)

    def test_noncanonical_or_duplicate_sidecar_is_rejected(self):
        sidecar_path(self.binary).write_text(json.dumps(self.raw, indent=2))
        with self.assertRaisesRegex(ProvenanceError, "not canonical"):
            load_build_provenance(self.binary)
        sidecar_path(self.binary).write_text(
            '{"schema":"x","schema":"y"}\n'
        )
        with self.assertRaisesRegex(ProvenanceError, "duplicate JSON key"):
            load_build_provenance(self.binary)

    def test_missing_extra_and_malformed_fields_are_rejected(self):
        missing = dict(self.raw)
        missing.pop("build")
        with self.assertRaises(ProvenanceError):
            validate_provenance(missing)
        extra = dict(self.raw)
        extra["path"] = "/private/lis"
        with self.assertRaises(ProvenanceError):
            validate_provenance(extra)
        malformed = json.loads(json.dumps(self.raw))
        malformed["binary"]["sha256"] = "fnv1a64:1234"
        with self.assertRaises(ProvenanceError):
            validate_provenance(malformed)

    def test_symlink_binary_and_sidecar_are_rejected(self):
        alias = self.root / "lis-alias"
        alias.symlink_to(self.binary)
        write_build_provenance(sidecar_path(alias), self.raw)
        with self.assertRaisesRegex(ProvenanceError, "cannot open regular input"):
            load_build_provenance(alias)

        sidecar = sidecar_path(self.binary)
        target = self.root / "sidecar-target"
        sidecar.replace(target)
        sidecar.symlink_to(target)
        with self.assertRaisesRegex(ProvenanceError, "cannot open regular input"):
            load_build_provenance(self.binary)

    def test_oversized_sidecar_is_rejected(self):
        sidecar_path(self.binary).write_bytes(b" " * (MAX_PROVENANCE_BYTES + 1))
        with self.assertRaisesRegex(ProvenanceError, "exceeds"):
            load_build_provenance(self.binary)

    def test_write_is_canonical_and_mode_is_public_read_only(self):
        data = sidecar_path(self.binary).read_bytes()
        self.assertEqual(data, canonical_json_bytes(self.raw))
        self.assertEqual(sidecar_path(self.binary).stat().st_mode & 0o777, 0o644)


if __name__ == "__main__":
    unittest.main()
