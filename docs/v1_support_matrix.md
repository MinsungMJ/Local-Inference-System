# LIS v1 Support Matrix

This document defines the strict, mathematically, and architecturally verified boundaries of what LIS v1 supports. The engine is deliberately narrow, optimizing for offline trustworthiness over sprawling feature sets.

## 1. Model Format & Architectures

**Supported:**
- **Local HuggingFace Llama-Family Artifacts:**
  - Standard plain-RoPE `config.json` defining Llama architectural constants (`model_type: "llama"`, `llama3`, etc.) using `rope_theta`.
  - A single accompanying uncompressed `model.safetensors` file.
  - Explicit `lm_head.weight`, or tied output embeddings via `tie_word_embeddings: true`.
- **Qwen3 Dense Targets:**
  - User-supplied local HuggingFace-style directories containing a merged BF16
    `model.safetensors` and a compatible `config.json` with Qwen3 Dense
    decoder-only architecture (`model_type: "qwen3"`,
    `architectures: ["Qwen3ForCausalLM"]`).
  - This is a narrow Qwen3 Dense path, not broad Qwen-family support.
  - The auxiliary `model.safetensors.index.json`, if present in the same
    directory, is ignored when the merged file exists. Index-only shard loading
    remains unsupported.
- **Architectures:** Decoder-only causal language models corresponding to the supported plain-RoPE Llama blueprint, plus the documented Qwen3 Dense targets.
- **Dtypes (safetensors):** `F32`, `F16`, `BF16` for Llama-family imports; BF16 only for Qwen3 Dense targets. HuggingFace-local model weights must be uniform: every mapped weight tensor must match `config.weight_dtype`; mixed per-tensor dtype artifacts are documented unsupported scope and are not normalized at load time.

**Explicitly Unsupported:**
- Non-Llama transformer families except the documented Qwen3 Dense targets (GPT-2, Mistral, GPT-OSS, Qwen2/Qwen2.5, generic Qwen-family claims, and other Qwen3 artifact layouts).
- Multimodal architectures, VL variants, MoE (Mixture of Experts).
- Network paths (HuggingFace Hub remote downloads, authentication).
- PyTorch `.bin`, `.pt`, `.pth` formats.
- GGUF/GGML local formats.
- Adapter weights (LoRA, QLoRA) and training checkpoints.
- Sharded HuggingFace distributions (`model.safetensors.index.json` mapped over multiple chunks); index-only shard loading is unsupported.
- Specific quantization schemas outside basic `F16`/`BF16`/`F32` floating limits.
- Internal LIS validation tensors inside HuggingFace model artifacts.
- RoPE scaling configs, including `rope_scaling` and non-default `rope_type` values such as Llama 3.1/3.2 `rope_type: "llama3"`.

## Context and Positional Semantics

**Contract:** `0 < configured_max_tokens ≤ trained_max_tokens` is the only accepted relation. Violations are fail-fast:

| Condition | Status Code |
|-----------|-------------|
| `configured > trained` | `LIS_STATUS_LIMIT_EXCEEDED` |
| `configured == 0` or `trained == 0` | `LIS_STATUS_INVALID_ARGUMENT` |
| `config_mode != LIS_CONTEXT_CONFIG_RUNTIME` | `LIS_STATUS_UNSUPPORTED` |
| `over_trained_policy != LIS_CONTEXT_OVER_TRAINED_REJECT` | `LIS_STATUS_UNSUPPORTED` |
| `rope_scaling` present and non-null (including `{}`) | `LIS_STATUS_UNSUPPORTED` |
| `rope_type` present and not `"default"` | `LIS_STATUS_UNSUPPORTED` |

There is no automatic clamping. No new RoPE variants, no YaRN, no llama3 rope, no sliding-window attention, no position interpolation, no context extension.

**Supported positional semantics per family:** Plain RoPE from `rope_theta`, causal mask, position indices `[0, configured_max_tokens)`, GQA layout, no sliding window, no position reset, no negative positions, single sequence per batch slot. Applies to Llama 3.x plain-RoPE subset and Qwen3 Dense only.

**Context exhaustion:** Prefill at exact `configured_max_tokens` is accepted. The following decode step returns `LIS_STATUS_LIMIT_EXCEEDED` with positions unchanged and zero new tokens emitted. CLI stop reason is `context_limit`.

## 2. Tokenizers

**Supported:**
- **LIS_VOCAB_V1:** The LIS internal canonical vocabulary binary format.
- **HuggingFace `tokenizer.json` (BPE Type):** Uncompressed, local `tokenizer.json` files representing standard Byte-Pair Encoding logic.
- Root-level special `added_tokens` entries for Llama control/chat tokens.
- Qwen3 tokenizer use is limited to direct token IDs and bounded plain-text encode validation through the existing HuggingFace BPE boundary.

**Explicitly Unsupported:**
- `tokenizer.model` SentencePiece protobufs.
- `tiktoken` definitions.
- Non-BPE tokenizers (Unigram, WordPiece).
- Pre-tokenizer regex splits inside JSON.
- Post-processor normalizations from JSON.
- Chat-template/Jinja execution, broad Qwen tokenizer templates, multi-turn role/session templating, and tool/reasoning templates.

## 3. Runtime execution

**Supported:**
- Platform: Local offline environments utilizing CPU execution.
- Policy: Greedy decode algorithm selection.
- Stop condition: scalar or array `eos_token_id` from `config.json`.
- Static batching for validation fixtures; batch 1 for the HuggingFace-local decoder forward paths.
- Validation-target execution through `lis.validation_logits` fixtures.
- CPU reference Llama forward execution for local HuggingFace Llama-family imports, using embeddings, RMSNorm, plain RoPE from `rope_theta`, grouped-query causal attention, SwiGLU MLP, KV cache, and lm_head logits.
- CPU reference Qwen3 Dense forward execution for documented Dense targets, using embeddings, RMSNorm, Q/K RMSNorm, plain RoPE, grouped-query causal attention, gated `silu` MLP, KV cache, and lm_head logits.
- Precision policy: `F16` and `BF16` weights are promoted to FP32 at compute boundaries. Runtime scratch, reductions/accumulations, attention math, logits, greedy selection, and diagnostics remain FP32. KV-cache storage dtype follows `config.weight_dtype`, with FP32 scratch K/V converted on store and promoted back to FP32 on read.

**Explicitly Unsupported:**
- Batch > 1 on the real-forward HuggingFace-local decoder paths.
- Qwen3 MoE, multimodal/VL, long-context/YaRN, sliding-window attention, attention-bias variants, and broad Qwen-family support.
- GPU accelerated backends (CUDA, ROCm, Metal) - these are interface-preparations only.
- Continuous batching and admission scheduling.
- Advanced decode policies (Speculative Decode, Top-P, Temperature).
- API Serving architecture or HTTP interfaces.
