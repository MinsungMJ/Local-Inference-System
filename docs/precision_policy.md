# Precision Policy

This document defines the precision semantics for the existing CPU Llama-family
execution path. It documents behavior in the loader, dtype, KV cache, and
CPU-kernel code, with focused verification around that policy. It does not add
a new model family, quantization, GPU mixed precision, new decode behavior, or
new optimized precision kernels.

## Supported Scope

Supported model weight dtypes are:

- `F32`
- `F16`
- `BF16`

For HuggingFace-local Llama imports, `config.json` `torch_dtype` maps to
`config.weight_dtype` as follows:

| `torch_dtype` values | LIS dtype |
|---|---|
| `float32`, `f32` | `F32` |
| `float16`, `f16` | `F16` |
| `bfloat16`, `bf16` | `BF16` |

All mapped HuggingFace weight tensors must match `config.weight_dtype`. Mixed
per-tensor dtype artifacts are `documented_unsupported` under the documented
result classes. LIS does not normalize mixed-dtype artifacts and does not
bulk-convert model weights during loading.

The narrow Qwen3 Dense path accepts BF16 weights only.

## Conversion Points

Safetensors loading preserves native tensor bytes. Conversion happens at compute
boundaries:

- embedding reads promote stored weight values to FP32 scratch values
- matvec reads promote `F32` / `F16` / `BF16` weights to FP32 before multiply
  and accumulation
- RMSNorm weight reads promote to FP32 before scaling
- attention KV-cache reads promote stored K/V values back to FP32 before dot
  product and P/V accumulation
- diagnostics read stored weights/KV values through FP32 promotion helpers

There is no load-time bulk conversion of `F16` or `BF16` weights to `F32`.

## FP32 Compute Rule

The precision-aware execution path uses FP32 compute and accumulation:

- runtime scratch activations are `float`
- Q/K/V projection outputs are `float` before KV-cache storage
- matvec accumulators and outputs are FP32
- RMSNorm mean-square reduction and scaled outputs are FP32
- attention scores, softmax probabilities, and attention context accumulation
  are FP32
- SwiGLU, residual add, final norm, logits, and greedy selection are FP32
- layer-checkpoint and generation diagnostics report FP32-observed values

Small numeric drift can still occur when stored `F16` or `BF16` inputs represent
rounded values, and AVX reductions can differ from scalar reductions under the
documented numeric policy. Token-parity checks remain the higher-level stability
gate where applicable.

## KV-Cache Semantics

KV-cache precision is part of the precision policy:

- KV-cache storage dtype follows `config.weight_dtype`
- K/V are produced in FP32 scratch space before storage
- storing K/V converts from FP32 scratch to the KV-cache storage dtype
- reading K/V promotes stored values back to FP32 for attention math

This policy does not redesign the KV-cache layout and does not introduce a separate KV
precision policy. The cache remains tied to the model weight dtype for the
supported CPU Llama-family path.

## KV Cache Precision and Storage Policy

The KV precision/storage policy is documentation only.
`report.kv_cache` emits the resolved KV storage dtype without
changing runtime conversion behavior, introducing new C symbols, or ratifying a
KV operation API.

`kv_storage_dtype` is the resolved dtype used by the KV cache layout for stored
K/V entries. In the current implementation this is
`lis_kv_cache_layout.dtype`, computed during `lis_kv_cache_init` from
`config->weight_dtype` in `srcs/runtime/kv_cache.c`.

Current write conversion behavior:

- Qwen3 and Llama both produce K/V values in FP32 scratch buffers before KV
  storage.
- Their local `lis_store_kv` helpers obtain element pointers with
  `lis_kv_cache_key_ptr` and `lis_kv_cache_value_ptr`.
- Each stored key/value element is written with the local `lis_scalar_write`
  helper using `runtime->kv_cache.layout.dtype`.
- Current LIS does not change this conversion behavior or introduce a separate
  storage-conversion path.

Current read/compute behavior:

- Attention reads the resolved KV dtype from `runtime->kv_cache.layout.dtype`.
- Qwen3 and Llama pass that dtype to `lis_attn_qk` and `lis_attn_pv`.
- The CPU attention helpers accept a `kv_dtype` argument and promote stored KV
  values for FP32 attention math; KV storage dtype is not the accumulation
  dtype.
- Current LIS does not change attention math, scalar conversion math, or FP32
  accumulation semantics.

Precision-path reconciliation:

- LIS builds `precision_path` once after `lis_runtime_init`, using
  `model.metadata.config.weight_dtype` for `weights=<dtype>` and
  `runtime.kv_cache.layout.dtype` for `kv=<dtype>`.
- The canonical artifact string remains
  `f32_accum;weights=<dtype>;kv=<dtype>` under
  `manifest.runtime.precision_path`.
- `report.kv_cache.storage_dtype` must agree with
  the `kv=<dtype>` component of `precision_path`.
- `precision_path` remains the human-readable run precision summary.
  `report.kv_cache.storage_dtype` is the machine-readable KV storage dtype
  field in the run report.

Unsupported in the current public release:

- Mixed K and V storage dtypes.
- Per-layer KV dtype variation.
- Runtime KV dtype switching during a run.
- New dtype support beyond the existing supported dtype policy.
- Weight dtype policy changes.
- FP32 accumulation semantic changes.
- Quantized KV cache formats.

## Verification

Precision verification reuses the documented verification framework:

- `make verify` is the routine no-model gate
- `make verify-kernels` is the routine precision-aware numeric/kernel gate
- `make verify-token-parity` is the stronger, slower, model-backed gate for
  release validation points

Precision-related failures use the documented result classes:

- excessive numeric drift: `numeric_regression`
- token sequence mismatch: `token_parity_regression`
- unsupported dtype/artifact scope: `documented_unsupported`
- missing local artifacts or bad harness setup: `harness_configuration_error`

## Non-Goals

- No Qwen3 Dense precision expansion beyond the documented BF16 path.
- No int8, int4, or other quantized execution paths.
- No GPU mixed-precision path.
- No new model-family support.
- No runtime architecture redesign.
- No KV-layout redesign.
- No decode-policy changes.
- No new optimized precision kernels.
