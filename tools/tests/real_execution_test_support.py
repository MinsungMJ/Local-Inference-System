"""Small local Llama fixture used by real-adapter tests."""

from __future__ import annotations

import json
from pathlib import Path
import struct

from lis_verify.execution import ExecutionResult
from lis_verify.provenance import (
    build_provenance,
    sidecar_path,
    write_build_provenance,
)


ROOT = Path(__file__).resolve().parents[2]


INTRA_STAGES = (
    ("layer_input", "layer_input", "Layer input"),
    ("attention_norm_output", "attention_norm_output", "Pre-attention RMSNorm output"),
    ("query_projection_output", "query_projection_output", "Q projection output"),
    ("key_projection_output", "key_projection_output", "K projection output"),
    ("value_projection_output", "value_projection_output", "V projection output"),
    ("rope_query_output", "rope_query_output", "RoPE-applied Q"),
    ("rope_key_output", "rope_key_output", "RoPE-applied K"),
    ("attention_scores", "attention_scores", "Attention pre-softmax scores"),
    ("attention_probabilities", "attention_probabilities", "Attention softmax output"),
    ("attention_context", "attention_context", "Attention context"),
    ("attention_output_projection", "attention_output_projection", "Attention output projection"),
    ("post_attention_residual", "post_attention_residual", "Post-attention residual"),
    ("mlp_norm_output", "mlp_norm_output", "Pre-MLP RMSNorm output"),
    ("mlp_gate_projection", "mlp_gate_projection", "MLP gate projection"),
    ("mlp_up_projection", "mlp_up_projection", "MLP up projection"),
    ("mlp_gated_activation", "mlp_gated_activation", "MLP gated activation"),
    ("mlp_down_projection", "mlp_down_projection", "MLP down projection"),
)


def write_tiny_llama(directory: Path, *, layers: int = 1) -> Path:
    directory.mkdir(mode=0o700)
    tensors: dict[str, dict[str, object]] = {}
    values: list[float] = []
    offset = 0

    def add(name: str, shape: list[int], data: list[float]) -> None:
        nonlocal offset
        size = len(data) * 4
        tensors[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        offset += size
        values.extend(data)

    add("model.embed_tokens.weight", [3, 1], [1.0, 2.0, 3.0])
    for layer in range(layers):
        prefix = f"model.layers.{layer}"
        for suffix in (
            "self_attn.q_proj.weight",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
        ):
            add(f"{prefix}.{suffix}", [1, 1], [0.0])
        add(f"{prefix}.input_layernorm.weight", [1], [1.0])
        add(f"{prefix}.post_attention_layernorm.weight", [1], [1.0])
    add("model.norm.weight", [1], [1.0])
    add("lm_head.weight", [3, 1], [0.9, 0.2, 0.1])
    header = json.dumps(
        tensors,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    payload = struct.pack("<Q", len(header)) + header
    payload += struct.pack(f"<{len(values)}f", *values)
    (directory / "model.safetensors").write_bytes(payload)
    config = {
        "architectures": ["LlamaForCausalLM"],
        "head_dim": 1,
        "hidden_size": 1,
        "intermediate_size": 1,
        "max_position_embeddings": 128,
        "model_type": "llama",
        "num_attention_heads": 1,
        "num_hidden_layers": layers,
        "num_key_value_heads": 1,
        "rope_theta": 10000.0,
        "torch_dtype": "float32",
        "vocab_size": 3,
    }
    (directory / "config.json").write_text(
        json.dumps(config, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    return directory


def write_fake_binary(path: Path, role: str) -> Path:
    path.write_bytes(f"#!/bin/sh\n# {role}\nexit 0\n".encode("ascii"))
    path.chmod(0o700)
    raw = build_provenance(
        binary=path,
        source_root=ROOT,
        compiler="fixture",
        cppflags="(none)",
        cflags="(none)",
        ldflags="(none)",
        ldlibs="(none)",
        simd="off",
    )
    write_build_provenance(sidecar_path(path), raw)
    return path


def _sha(number: int) -> str:
    return f"sha256:{number:064x}"


def _manifest(candidate: bool, *, checkpoint: int | None, intra: int | None) -> dict:
    binary_hex = "b" * 16 if candidate else "a" * 16
    runtime = {
        "configured_context": 128,
        "batch_size": 1,
        "thread_count": 1,
        "generation_limit": 8,
        "precision_path": "f32_accum;weights=f32;kv=f32",
        "fingerprint": {
            "algorithm": "fnv1a64",
            "hex": "2" * 16 if candidate else "1" * 16,
        },
        "layer_checkpoints_enabled": checkpoint is not None,
        "layer_checkpoint_step": checkpoint,
        "diagnostics_enabled": False,
        "perf_enabled": False,
        "perf_per_token_enabled": False,
    }
    if intra is not None:
        runtime.update(
            intra_layer_checkpoints_enabled=True,
            intra_layer_target_layer=intra,
            diagnostic_capture_profile="semantic_layer_and_intra_v1",
        )
    return {
        "binary": {
            "fingerprint": {"algorithm": "fnv1a64", "hex": binary_hex}
        },
        "model": {
            "format": "huggingface_local",
            "family": "llama3_decoder",
            "fingerprint": {"algorithm": "fnv1a64", "hex": "c" * 16},
        },
        "config": {
            "fingerprint": {"algorithm": "fnv1a64", "hex": "d" * 16}
        },
        "input": {
            "mode": "tokens",
            "fingerprint": {"algorithm": "fnv1a64", "hex": "e" * 16},
        },
        "runtime": runtime,
        "backend": {
            "name": "reference",
            "fingerprint": {
                "algorithm": "fnv1a64",
                "hex": "f" * 16,
            },
        },
    }


def _coordinate(step: int, layer: int, ordinal: int) -> dict:
    return {
        "runtime_checkpoint_step": step,
        "layer_index": layer,
        "tensor_role": "layer_output",
        "batch_index": 0,
        "sequence_index": 0,
        "stage_order": 0,
        "execution_ordinal": ordinal,
    }


def _layer_trace(
    manifest: dict,
    artifact_set_id: str,
    *,
    candidate: bool,
    checkpoint: int,
    intra: int | None,
) -> dict:
    layers = (0, 4, 8, 11)
    requested = [_coordinate(checkpoint, layer, index) for index, layer in enumerate(layers)]
    entries = []
    for index, layer in enumerate(layers):
        coordinate = requested[index]
        digest_number = layer + 1
        if candidate and layer >= 4:
            digest_number += 100
        entries.append(
            {
                "step": checkpoint,
                **coordinate,
                "phase": "decode",
                "name": f"layer.{layer}.output",
                "observed_dtype": "fp32",
                "shape": [1],
                "element_count": 1,
                "available_summary_fields": ["min", "max", "mean", "l2", "nan", "inf", "digest"],
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "l2": 0.0,
                "nan": 0,
                "inf": 0,
                "digest": {
                    "algorithm": "sha256",
                    "version": "lis.checkpoint.fp32le/v1",
                    "tensor_role": "layer_output",
                    "shape": [1],
                    "observed_dtype": "fp32",
                    "byte_order": "little",
                    "canonicalization": "ieee754-binary32-le;canonical-qnan;preserve-signed-zero",
                    "value": _sha(digest_number),
                },
            }
        )
    raw = {
        "schema": "lis.execution_artifact/v1",
        "kind": "layer_trace",
        "artifact_set_id": artifact_set_id,
        "manifest": manifest,
        "checkpoint_layout": {
            "layout_name": "llama_layer_output_summary",
            "layout_version": 1,
            "runtime_checkpoint_step": checkpoint,
            "tensor_role": "layer_output",
            "stage_order": 0,
            "ordering_semantics": "runtime_step_layer_stage_ordinal",
            "total_layer_count": 12,
            "requested_coordinates": requested,
            "captured_coordinates": requested,
            "missing_coordinates": [],
            "available_summary_fields": ["min", "max", "mean", "l2", "nan", "inf", "digest"],
            "digest_contract": {
                "algorithm": "sha256",
                "version": "lis.checkpoint.fp32le/v1",
                "observed_dtype": "fp32",
                "byte_order": "little",
                "canonicalization": "ieee754-binary32-le;canonical-qnan;preserve-signed-zero",
            },
            "duplicate_coordinate_policy": "reject_artifact_before_write",
        },
        "layer_trace": entries,
    }
    if intra is not None:
        token_position = checkpoint
        coordinates = [
            {
                "runtime_checkpoint_step": checkpoint,
                "layer_index": intra,
                "stage_id": stage_id,
                "tensor_role": tensor_role,
                "batch_index": 0,
                "sequence_index": 0,
                "token_position": token_position,
                "stage_order": order,
                "execution_ordinal": order,
            }
            for order, (stage_id, tensor_role, _) in enumerate(INTRA_STAGES)
        ]
        raw["intra_layer_checkpoint_layout"] = {
            "layout_name": "llama_intra_layer_summary",
            "layout_version": 1,
            "model_family": "llama3_decoder",
            "stage_taxonomy": "lis.llama.intra_layer_stages/v1",
            "runtime_checkpoint_step": checkpoint,
            "phase": "decode",
            "target_layer": intra,
            "batch_index": 0,
            "sequence_index": 0,
            "token_position": token_position,
            "ordering_semantics": "runtime_step_layer_stage_ordinal",
            "duplicate_coordinate_policy": "reject_artifact_before_write",
            "requested_coordinates": coordinates,
            "captured_coordinates": coordinates,
            "missing_coordinates": [],
            "available_summary_fields": ["min", "max", "mean", "l2", "nan", "inf", "digest"],
            "digest_contract": {
                "algorithm": "sha256",
                "version": "lis.checkpoint.intra_layer.fp32le/v1",
                "observed_dtype": "fp32",
                "byte_order": "little",
                "canonicalization": "ieee754-binary32-le;canonical-qnan;preserve-signed-zero",
            },
            "full_tensor_payload_allowed": False,
        }
        raw["intra_layer_trace"] = []
        for order, (stage_id, tensor_role, public_name) in enumerate(INTRA_STAGES):
            digest_number = 1000 + order
            if candidate and order >= 7:
                digest_number += 100
            raw["intra_layer_trace"].append(
                {
                    **coordinates[order],
                    "phase": "decode",
                    "public_name": public_name,
                    "shape": [1],
                    "observed_dtype": "fp32",
                    "precision_path": "f32_accum;weights=f32;kv=f32",
                    "element_count": 1,
                    "available_summary_fields": ["min", "max", "mean", "l2", "nan", "inf", "digest"],
                    "min": float(order),
                    "max": float(order),
                    "mean": float(order),
                    "l2": float(order),
                    "nan": 0,
                    "inf": 0,
                    "digest": {
                        "algorithm": "sha256",
                        "version": "lis.checkpoint.intra_layer.fp32le/v1",
                        "tensor_role": tensor_role,
                        "shape": [1],
                        "observed_dtype": "fp32",
                        "byte_order": "little",
                        "canonicalization": "ieee754-binary32-le;canonical-qnan;preserve-signed-zero",
                        "value": _sha(digest_number),
                    },
                }
            )
    return raw


class SeededMismatchExecutor:
    """Writes source-consistent artifacts for a step-2 runtime mismatch."""

    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def _option(argv, name: str) -> str | None:
        try:
            return argv[argv.index(name) + 1]
        except ValueError:
            return None

    def run(self, argv, **kwargs):
        del kwargs
        self.calls += 1
        candidate = "candidate" in Path(argv[0]).name
        checkpoint_text = self._option(argv, "--layer-checkpoints")
        checkpoint = int(checkpoint_text) if checkpoint_text is not None else None
        intra_text = self._option(argv, "--intra-layer-checkpoints")
        intra = int(intra_text) if intra_text is not None else None
        report_path = Path(self._option(argv, "--report-json"))
        artifact_set_id = f"aset1:{self.calls:032x}"
        manifest = _manifest(candidate, checkpoint=checkpoint, intra=intra)
        binding_path = self._option(argv, "--forced-prefix-binding-json")
        if binding_path is None:
            selected = [0, 1, 1 if candidate else 0, 2, 0, 1, 2, 0]
        else:
            selected = [1 if candidate else 0]
        report = {
            "schema": "lis.execution_artifact/v1",
            "kind": "run_report",
            "artifact_set_id": artifact_set_id,
            "manifest": manifest,
            "report": {
                "execution_status": "ok",
                "stop_reason": "decode_limit",
                "prompt_sequences": [{"token_count": 1}],
                "selected_token_count": len(selected),
                "selected_token_ids": selected,
            },
        }
        if binding_path is not None:
            report["forced_prefix"] = json.loads(Path(binding_path).read_text(encoding="utf-8"))
        report_path.write_text(
            json.dumps(report, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        if checkpoint is not None:
            layer_path = Path(self._option(argv, "--layer-trace-json"))
            layer_path.write_text(
                json.dumps(
                    _layer_trace(
                        manifest,
                        artifact_set_id,
                        candidate=candidate,
                        checkpoint=checkpoint,
                        intra=intra,
                    ),
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
        return ExecutionResult("ok", 0, b"", b"")
