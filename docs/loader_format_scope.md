# Loader Format Scope

This document records the loader format decisions for the supported local file
boundaries.

## Safetensors Supported Subset

LIS supports a narrow safetensors subset:

- local single-file path sources
- 8-byte little-endian header length
- JSON object header with top-level tensor entries
- tensor fields: `dtype`, `shape`, and `data_offsets`
- ignored `__metadata__` object
- supported dtypes: `F32`, `F16`, `BF16`, `I32`, and `U32`
- supported shapes: rank 1 through `LIS_TENSOR_MAX_RANK`, nonzero dimensions,
  contiguous byte-size validation through tensor helpers
- data offsets relative to the safetensors data section after the JSON header

Unsupported safetensors cases return explicit status values such as `LIS_STATUS_FORMAT`, `LIS_STATUS_UNSUPPORTED_DTYPE`, `LIS_STATUS_UNSUPPORTED_SHAPE`, `LIS_STATUS_SHAPE_MISMATCH`, or `LIS_STATUS_IO`.

Tensor names are accepted as simple unescaped JSON strings. Llama 3.x
weight-name semantic validation is deferred until model mapping work defines
the exact required tensor table.

The precision policy does not change safetensors parsing: `F16` and `BF16`
bytes remain native in loaded tensor views. For HuggingFace-local Llama imports,
the later mapping layer requires every mapped weight tensor to match
`config.weight_dtype`; mixed per-tensor dtype artifacts are documented
unsupported scope and are not normalized by the loader.

## Llama Config Scope

LIS includes a narrow in-memory Llama-style JSON config parser. It maps the
minimal fields needed by `lis_model_metadata`:

- `model_type`
- `num_hidden_layers`
- `hidden_size`
- `intermediate_size`
- `num_attention_heads`
- `num_key_value_heads`
- `head_dim`
- `vocab_size`
- `rope_theta`
- `torch_dtype`
- `max_position_embeddings`

The parser accepts `model_type` values `llama` and `llama3` as the initial Llama-style decoder-only boundary, but only for plain RoPE configs that use `rope_theta`. Configs containing `rope_scaling` or a non-default `rope_type` are rejected with `LIS_STATUS_UNSUPPORTED` because LIS does not implement those RoPE variants. GPT-2, Mistral, and GPT-OSS config parsing remain unsupported extension paths.

## PyTorch Compatibility Scope

LIS does not implement PyTorch checkpoint loading or broad `.pt`, `.pth`, or
`.bin` compatibility. Those paths are detected as
`LIS_MODEL_FORMAT_PYTORCH_UNSUPPORTED` and return
`LIS_STATUS_UNSUPPORTED_FORMAT`.

Future PyTorch-exported compatibility must define a specific export path, tensor naming convention, dtype scope, and comparison tests before implementation. The current implementation contains no PyTorch comparison cases or parity claims.

## Save Support Decision

Save support is deferred. LIS implements no canonical save, safetensors save,
or PyTorch-compatible export path.

Future save work must first choose and document one explicit semantic:

- LIS canonical internal format save
- limited safetensors save
- limited PyTorch-compatible export
