import copy
import json
from pathlib import Path
import tempfile
import unittest

from lis_verify.model_profile import (
    ModelProfileError,
    PROFILE_RESOURCE,
    load_model_profile,
    resolve_model,
    validate_model_profile,
)
from lis_verify.product_contract import canonical_json_bytes


class ModelProfileTestCase(unittest.TestCase):
    def setUp(self):
        self.profile = load_model_profile()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model_dir = self.root / "model"
        self.model_dir.mkdir()
        self.config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "num_hidden_layers": 2,
            "hidden_size": 2,
            "intermediate_size": 4,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "vocab_size": 3,
            "rope_theta": 10000.0,
            "rope_scaling": None,
            "torch_dtype": "float32",
            "max_position_embeddings": 128,
        }
        (self.model_dir / "config.json").write_bytes(canonical_json_bytes(self.config))
        (self.model_dir / "model.safetensors").write_bytes(b"tiny-safetensors")

    def tearDown(self):
        self.temp.cleanup()

    def test_packaged_profile_is_frozen_and_canonical(self):
        self.assertEqual(self.profile.direct_token_ids, (1,))
        self.assertEqual(self.profile.context_length, 128)
        self.assertEqual(self.profile.generation_limit, 8)
        self.assertTrue(self.profile.identity_sha256.startswith("sha256:"))

    def test_resolve_model_binds_merged_model_and_config(self):
        resolved = resolve_model(self.model_dir, self.profile)
        self.assertEqual(resolved.layer_count, 2)
        self.assertEqual(resolved.vocab_size, 3)
        self.assertTrue(resolved.model_sha256.startswith("sha256:"))
        self.assertTrue(resolved.config_sha256.startswith("sha256:"))

    def test_profile_missing_extra_and_policy_mismatch_are_rejected(self):
        missing = copy.deepcopy(self.profile.raw)
        missing.pop("direct_token_ids")
        with self.assertRaises(ModelProfileError):
            validate_model_profile(missing)
        extra = copy.deepcopy(self.profile.raw)
        extra["prompt"] = "private"
        with self.assertRaises(ModelProfileError):
            validate_model_profile(extra)
        mismatch = copy.deepcopy(self.profile.raw)
        mismatch["selection_policy_sha256"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ModelProfileError, "policy identity"):
            validate_model_profile(mismatch)

    def test_duplicate_and_noncanonical_resource_are_rejected(self):
        duplicate = b'{"schema":"a","schema":"b"}\n'
        with self.assertRaisesRegex(ModelProfileError, "duplicate JSON key"):
            load_model_profile(lambda name: duplicate)
        pretty = json.dumps(self.profile.raw, indent=2).encode()
        with self.assertRaisesRegex(ModelProfileError, "not canonical"):
            load_model_profile(lambda name: pretty)

    def test_unsupported_model_boundaries_are_rejected(self):
        cases = (
            ("model_type", "qwen3", "model type"),
            ("rope_scaling", {"type": "linear", "factor": 2.0}, "Scaled RoPE|scaled RoPE"),
            ("torch_dtype", "int8", "dtype"),
            ("max_position_embeddings", 64, "context"),
            ("vocab_size", 1, "token ID"),
        )
        for field, value, message in cases:
            with self.subTest(field=field):
                config = dict(self.config)
                config[field] = value
                (self.model_dir / "config.json").write_bytes(canonical_json_bytes(config))
                with self.assertRaisesRegex(ModelProfileError, message):
                    resolve_model(self.model_dir, self.profile)

    def test_sharded_or_missing_merged_model_is_rejected(self):
        (self.model_dir / "model.safetensors").unlink()
        (self.model_dir / "model.safetensors.index.json").write_text("{}")
        with self.assertRaises(ModelProfileError):
            resolve_model(self.model_dir, self.profile)

    def test_symlink_model_component_is_rejected(self):
        alias = self.root / "alias"
        alias.symlink_to(self.model_dir, target_is_directory=True)
        with self.assertRaisesRegex(ModelProfileError, "symlink"):
            resolve_model(alias, self.profile)


if __name__ == "__main__":
    unittest.main()
