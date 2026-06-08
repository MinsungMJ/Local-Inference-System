# Qwen3 Dense Scope

LIS supports a narrow Qwen3 Dense path for user-supplied local Dense artifacts.
It is not broad Qwen support, not general Qwen3 artifact support, not a
model-zoo importer, and not a long-context or serving project.

## First Target Artifact

LIS targets user-supplied local HuggingFace-style Qwen3 Dense directories.
The documented baseline is a merged BF16 `model.safetensors` with the observed
config below. Any compatible Qwen3 Dense artifact with the same architecture,
tensor naming, and plain-RoPE semantics may validate; this does not imply
broad Qwen3 or Qwen-family support.

The observed baseline config uses:

- `model_type: "qwen3"`
- `architectures: ["Qwen3ForCausalLM"]`
- `torch_dtype: "bfloat16"`
- `num_hidden_layers: 36`
- `hidden_size: 4096`
- `intermediate_size: 12288`
- `num_attention_heads: 32`
- `num_key_value_heads: 8`
- `head_dim: 128`
- `vocab_size: 151936`
- `rope_theta: 1000000`
- `rope_scaling: null`
- `use_sliding_window: false`
- `attention_bias: false`
- `hidden_act: "silu"`

A smaller Dense variant in the same family (e.g. Qwen3-0.6B) can serve as a
fast development-loop model if it uses the same supported config, tensor
naming, and plain-RoPE semantics. Other Qwen3 artifact layouts remain out of
scope.

## Artifact Layout

Initial public support targets the merged safetensors file in a user-supplied
local HuggingFace-style directory. If the merged `model.safetensors` file
exists, LIS loads that file and ignores any auxiliary index; if the merged
file is missing and only the index is present, LIS rejects the artifact as
unsupported. The presence of an index must not be interpreted as support for
sharded HuggingFace checkpoint loading.

## Supported Subset

LIS supports only:

- Dense decoder-only Qwen3 text execution for the narrow documented path.
- Batch-1 prefill/decode baseline.
- Short-context validation.
- BF16 real target weights, using the documented precision policy.
- Plain RoPE from `rope_theta`.
- Grouped-query causal attention.
- Gated `silu` MLP.
- Direct token-ID sanity checks.
- Bounded plain-text HuggingFace BPE tokenizer encode sanity for the
  tokenizer artifacts if they validate under the existing LIS tokenizer boundary.

Greedy decode is a deterministic sanity path. It is not a quality claim and does
not imply support for Qwen3 recommended sampling settings.

## Explicitly Unsupported

LIS does not support:

- Qwen-family umbrella compatibility.
- Qwen2 or Qwen2.5.
- Qwen3 MoE.
- Multimodal or VL variants.
- Other Qwen3 artifact layouts beyond documented Dense merged-safetensors
  targets.
- Shard-index loading.
- PyTorch `.bin`, `.pt`, or `.pth` checkpoint loading.
- GGUF or GGML.
- Quantized, adapted, LoRA, or training checkpoints.
- Non-null `rope_scaling`, YaRN, or other long-context RoPE extensions.
- `use_sliding_window == true` or sliding-window attention.
- Attention bias variants.
- Chat-template/Jinja parsing.
- Broad tokenizer-template support.
- Multi-turn session state, tool/reasoning template support, HTTP serving,
  GPU execution, or new sampling/decode policies.

Unsupported artifacts are `documented_unsupported` under the documented result-class
discipline unless they expose a malformed input, parser failure, or true
regression.

## Required Tensor Set

The supported Qwen3 Dense tensor surface is explicit. Required tensors are:

- `model.embed_tokens.weight`
- `model.norm.weight`
- `lm_head.weight`, or `tie_word_embeddings: true` with
  `model.embed_tokens.weight` reused as the output head

For each layer `N`:

- `model.layers.N.input_layernorm.weight`
- `model.layers.N.post_attention_layernorm.weight`
- `model.layers.N.self_attn.q_proj.weight`
- `model.layers.N.self_attn.k_proj.weight`
- `model.layers.N.self_attn.v_proj.weight`
- `model.layers.N.self_attn.o_proj.weight`
- `model.layers.N.self_attn.q_norm.weight`
- `model.layers.N.self_attn.k_norm.weight`
- `model.layers.N.mlp.gate_proj.weight`
- `model.layers.N.mlp.up_proj.weight`
- `model.layers.N.mlp.down_proj.weight`

Shapes must be validated from config values:

- embeddings and lm head: `[vocab_size, hidden_size]`
- Q projection: `[attention_heads * head_dim, hidden_size]`
- K/V projection: `[kv_heads * head_dim, hidden_size]`
- output projection: `[hidden_size, attention_heads * head_dim]`
- Q/K norm: `[head_dim]`
- gate/up projection: `[intermediate_size, hidden_size]`
- down projection: `[hidden_size, intermediate_size]`
- input/post/final norm: `[hidden_size]`

All mapped tensors must match `config.weight_dtype`. Mixed per-tensor dtype
artifacts are unsupported and must not be normalized at load time.

For Qwen3 Dense, `attention_heads * head_dim` is the Q projection width and may
differ from `hidden_size`. The residual stream remains `hidden_size`; K/V cache
width is `kv_heads * head_dim`. `attention_heads` must be divisible by
`kv_heads` for the grouped-query attention mapping.

## Runtime Semantics

The Qwen3 Dense forward path must be family-specific at first. It should not
opportunistically refactor the Llama runtime path before Qwen3 behavior is
validated.

The required first-pass execution order is:

1. Token embedding lookup.
2. Input RMSNorm.
3. Q/K/V projection.
4. Q/K RMSNorm using mapped `q_norm.weight` and `k_norm.weight`.
5. Plain RoPE.
6. Grouped-query causal attention.
7. Output projection and residual add.
8. Post-attention RMSNorm.
9. Gated `silu` MLP.
10. Residual add, final RMSNorm, and lm head logits.

Batch size remains 1 for the real Qwen3 forward path. Current public support does not add
sliding-window masks, YaRN, long-context validation, sampling, serving, or GPU
execution.

### Positional Semantics

Plain RoPE from `rope_theta`, causal mask, position indices `[0, configured_max_tokens)`, GQA layout per existing support matrix, no sliding window, no position reset, no negative positions, single sequence per batch slot.

Any `rope_scaling` value other than absent or `null` is rejected. Any `rope_type` value other than absent or `"default"` is rejected. These rejections apply identically in the Qwen3 config parser.

## Precision Assumptions

The Qwen3 Dense path follows the documented precision policy:

- BF16 weights remain in native safetensors storage.
- There is no bulk load-time conversion to FP32.
- Stored weights are promoted to FP32 at compute boundaries.
- Reductions, attention scores/probabilities, MLP math, logits, greedy
  selection, and diagnostics remain FP32.
- KV-cache storage follows `config.weight_dtype` unless implementation evidence
  proves a Qwen3-specific mismatch that is separately documented and approved.

## Tokenizer Boundary

The local tokenizer is HuggingFace BPE. Qwen3 Dense tokenizer support is limited to
direct token IDs plus bounded plain-text encode validation through the existing
LIS HuggingFace tokenizer import path. It must not add a Jinja/chat-template
engine or imply general Qwen chat-template compatibility.

Direct token-ID input remains the most controlled validation path. Any prompt
builder added later must be explicitly scoped and verified; it is not part of
the current public support path unless separately approved.

### Prompt and Generation Behavior Note

LIS currently passes prompts as raw tokenizer text. It does not apply
model-specific chat templates, role formatting, or thinking-mode controls.
As a result, reasoning-oriented Qwen3 models may produce extended
explanatory output even for short prompts. The observed output reflects
the model's behavior under the supplied raw prompt and greedy decoding
path; it does not indicate a runtime failure.

## Verification Expectations

Qwen3 Dense support must reuse the documented verification discipline:
- unsupported variants are explicit rejection tests
- a fixed short model-backed sanity case uses `VERIFY_QWEN3_MODEL`
- initial model-backed generation length is 1
- evidence records the artifact path, family path, selected token or top logits,
  and at least one bounded layer/logit/checkpoint sanity point when feasible
- Llama behavior remains covered by existing tests

Code, verification evidence, and support wording must agree before public
support wording is expanded.
