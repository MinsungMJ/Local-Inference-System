# HuggingFace Llama Compatibility Scope

This document narrows the LIS compatibility surface for Hugging Face
Llama-family model artifacts. It defines the required boundaries, supported
layout, and explicit non-goals for direct Hugging Face testing in the engine.

## Supported Layout

LIS supports reading exactly **one** local unzipped HuggingFace model directory containing:
1. `config.json`
2. `model.safetensors`

**Requirements:**
- A single `.safetensors` weight file is supported.
- Models requiring generic `safetensors.index.json` and sharded weight files (`model-00001-of-00002.safetensors`) will be rejected explicitly as `LIS_STATUS_UNSUPPORTED_FORMAT`.
- Models saved in traditional PyTorch `.bin` / `.pt` formats are not supported and will be rejected.

## Supported Model Configurations

Only models identified by `"model_type": "llama"` or `"model_type": "llama3"` are allowed.

Weight distributions must match standard Llama-decoder architecture mappings:
- Plain RoPE from `rope_theta` only. Configs containing `rope_scaling` or a non-default `rope_type` are rejected because LIS does not implement those RoPE variants.
- `f32`, `f16`, `bf16` datatypes for `torch_dtype` are acceptable.
- All mapped weight tensors must match the parsed `torch_dtype` / `config.weight_dtype`. Mixed per-tensor dtype artifacts are documented unsupported scope; LIS does not normalize them during load.
- `F16` and `BF16` weights are preserved as native safetensors bytes and promoted to FP32 at compute boundaries. KV-cache storage follows `config.weight_dtype`; K/V are produced in FP32 scratch, converted on cache store, and promoted back to FP32 for attention math.
- Derived shapes (such as `head_dim` calculated via `hidden_size / num_attention_heads`) are resolved automatically by the generic YAML/JSON parser.

## Explicit Name Mapping

HuggingFace safetensors weights (keys) are translated aggressively into generic LIS internal tensor representations. LIS mandates mapping the standard Llama keys correctly into internal layout definitions:

| HuggingFace Name                                | LIS Internal Name                           | Rank/Shape                                                         |
| ----------------------------------------------- | ------------------------------------------- | ------------------------------------------------------------------ |
| `model.embed_tokens.weight`                     | `lis.token_embeddings.weight`               | `[vocab_size, hidden_size]`                                        |
| `model.norm.weight`                             | `lis.output_norm.weight`                    | `[hidden_size]`                                                    |
| `lm_head.weight`                                | `lis.lm_head.weight`                        | `[vocab_size, hidden_size]`; optional when `tie_word_embeddings` is true |
| `model.layers.N.input_layernorm.weight`         | `lis.layer.N.attention_norm.weight`         | `[hidden_size]`                                                    |
| `model.layers.N.post_attention_layernorm.weight`| `lis.layer.N.mlp_norm.weight`               | `[hidden_size]`                                                    |
| `model.layers.N.self_attn.q_proj.weight`        | `lis.layer.N.q_proj.weight`                 | `[attention_heads * head_dim, hidden_size]`                        |
| `model.layers.N.self_attn.k_proj.weight`        | `lis.layer.N.k_proj.weight`                 | `[kv_heads * head_dim, hidden_size]`                               |
| `model.layers.N.self_attn.v_proj.weight`        | `lis.layer.N.v_proj.weight`                 | `[kv_heads * head_dim, hidden_size]`                               |
| `model.layers.N.self_attn.o_proj.weight`        | `lis.layer.N.o_proj.weight`                 | `[hidden_size, attention_heads * head_dim]`                        |
| `model.layers.N.mlp.gate_proj.weight`           | `lis.layer.N.gate_proj.weight`              | `[intermediate_size, hidden_size]`                                 |
| `model.layers.N.mlp.up_proj.weight`             | `lis.layer.N.up_proj.weight`                | `[intermediate_size, hidden_size]`                                 |
| `model.layers.N.mlp.down_proj.weight`           | `lis.layer.N.down_proj.weight`              | `[hidden_size, intermediate_size]`                                 |

Models that supply additional unmapped arrays trigger format errors as part of
strict correctness requirements. Internal LIS validation tensors such as
`lis.validation_logits` are not valid HuggingFace import tensors.

This is a loader/import compatibility boundary. The CPU reference Llama forward
path uses this mapped tensor layout.

If `config.json` sets `tie_word_embeddings: true` and `lm_head.weight` is absent, LIS uses `lis.token_embeddings.weight` for output logits.

## Non-Goals

- Connecting to `huggingface_hub` for network downloads.
- Sharded SAFETENSOR files.
- QLoRA / LoRA adapter integration.
- GGUF/GGML integration.
- Non-Llama transformer blocks.
- RoPE scaling variants such as Llama 3.1/3.2 `rope_type: "llama3"`.

## Positional Semantics

Plain RoPE from `rope_theta`, causal mask, position indices `[0, configured_max_tokens)`, GQA layout per existing support matrix, no sliding window, no position reset, no negative positions, single sequence per batch slot.

Any `rope_scaling` value other than absent or `null` is rejected. Any `rope_type` value other than absent or `"default"` is rejected. A stable CLI diagnostic substring `unsupported configuration` is emitted on rejection, listing all rejected keys including `rope_scaling` and `rope_type`.
