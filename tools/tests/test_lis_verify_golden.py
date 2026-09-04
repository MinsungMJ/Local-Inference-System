from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from lis_verify.golden import (
    GoldenManifestError,
    load_manifest,
    validate_manifest,
    verify_local_model,
)
from lis_verify.product_contract import canonical_json_bytes


def _manifest_from(raw: dict):
    data = canonical_json_bytes(raw)
    return load_manifest(lambda _: data)


def _write_fake_golden(root: Path):
    model = root / "model"
    model.mkdir(mode=0o700)
    config = {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 1,
        "intermediate_size": 1,
        "max_position_embeddings": 128,
        "model_type": "llama",
        "num_attention_heads": 1,
        "num_hidden_layers": 1,
        "num_key_value_heads": 1,
        "rope_scaling": None,
        "tie_word_embeddings": True,
        "torch_dtype": "bfloat16",
        "vocab_size": 3,
    }
    config_data = canonical_json_bytes(config)
    model_data = b"bounded fake safetensors bytes"
    (model / "config.json").write_bytes(config_data)
    (model / "model.safetensors").write_bytes(model_data)
    raw = load_manifest().materialize()
    for entry, data in zip(raw["files"], (config_data, model_data)):
        entry["size_bytes"] = len(data)
        entry["sha256"] = "sha256:" + hashlib.sha256(data).hexdigest()
    return model, _manifest_from(raw)


class TestGoldenManifest(unittest.TestCase):
    def test_packaged_manifest_is_canonical_and_frozen(self):
        manifest = load_manifest()
        self.assertEqual(
            manifest.identity_sha256,
            "sha256:c9c8230064d6e3b5d80409a0e1b28d04525a183d218a0a70b4a111063e69d53a",
        )
        raw = manifest.materialize()
        self.assertEqual(raw["upstream"]["license"], "Apache-2.0")
        self.assertNotIn("/main/", "".join(item["url"] for item in raw["files"]))
        self.assertEqual(
            {item["path"] for item in raw["files"]},
            {"config.json", "model.safetensors"},
        )
        self.assertTrue(raw["baseline_update"]["review_required"])

    def test_unknown_field_mutable_url_and_bad_revision_fail(self):
        cases = []
        unknown = load_manifest().materialize()
        unknown["unknown"] = True
        cases.append(unknown)
        mutable = load_manifest().materialize()
        mutable["files"][0]["url"] = mutable["files"][0]["url"].replace(
            mutable["upstream"]["revision"], "main"
        )
        cases.append(mutable)
        revision = load_manifest().materialize()
        revision["upstream"]["revision"] = "28e66ca"
        cases.append(revision)
        for raw in cases:
            with self.subTest(case=len(cases)):
                with self.assertRaises(GoldenManifestError):
                    validate_manifest(raw)

    def test_noncanonical_manifest_fails(self):
        raw = load_manifest().materialize()
        pretty = (json.dumps(raw, indent=2) + "\n").encode()
        with self.assertRaisesRegex(GoldenManifestError, "canonical"):
            load_manifest(lambda _: pretty)

    def test_local_material_checks_size_hash_profile_and_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, manifest = _write_fake_golden(root)
            result = verify_local_model(manifest, model)
            self.assertEqual(
                result.total_size_bytes,
                sum(path.stat().st_size for path in model.iterdir()),
            )

            (model / "model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(GoldenManifestError, "identity mismatch"):
                verify_local_model(manifest, model)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, manifest = _write_fake_golden(root)
            link = root / "link"
            link.symlink_to(model, target_is_directory=True)
            with self.assertRaisesRegex(GoldenManifestError, "symlink"):
                verify_local_model(manifest, link)

    def test_config_drift_fails_even_when_manifest_hash_is_updated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, _ = _write_fake_golden(root)
            config_path = model / "config.json"
            config = json.loads(config_path.read_bytes())
            config["rope_scaling"] = {"type": "linear", "factor": 2.0}
            config_data = canonical_json_bytes(config)
            config_path.write_bytes(config_data)
            raw = copy.deepcopy(load_manifest().materialize())
            raw["files"][0]["size_bytes"] = len(config_data)
            raw["files"][0]["sha256"] = (
                "sha256:" + hashlib.sha256(config_data).hexdigest()
            )
            model_data = (model / "model.safetensors").read_bytes()
            raw["files"][1]["size_bytes"] = len(model_data)
            raw["files"][1]["sha256"] = (
                "sha256:" + hashlib.sha256(model_data).hexdigest()
            )
            with self.assertRaisesRegex(GoldenManifestError, "supported profile"):
                verify_local_model(_manifest_from(raw), model)


if __name__ == "__main__":
    unittest.main()
