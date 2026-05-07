# LIS Functional Design

This document describes major subsystem responsibilities at design level. It is not code-level API documentation and does not authorize implementation. Keep it synchronized with [`architecture.md`](architecture.md).

## Config and Model Metadata

**Purpose:** Represent model-family settings and execution constraints needed to load and run supported decoder-only models.

**Likely Public Interface Responsibility:**

- Parse or receive model configuration.
- Validate required fields for plain-RoPE Llama style decoder-only inference.
- Expose trained maximum context, layer count, hidden size, attention configuration, plain RoPE parameters, vocabulary size, and dtype expectations.
- Distinguish functional support envelope from validation/test target envelope.

**Key Data Structures:**

- Model config object.
- Model family enum or descriptor.
- Context window policy object.
- Dtype and shape descriptors shared with tensor/core types.

**Context Policy Invariants:**

- `lis_context_window_policy_validate` returns exactly the status codes of the
  context contract table (C1-C5): zero inputs →
  `LIS_STATUS_INVALID_ARGUMENT`; enum mismatch → `LIS_STATUS_UNSUPPORTED`;
  `configured > trained` → `LIS_STATUS_LIMIT_EXCEEDED`; otherwise →
  `LIS_STATUS_OK`.
- `config_mode` has exactly one permitted value (`LIS_CONTEXT_CONFIG_RUNTIME`) and `over_trained_policy` has exactly one permitted value (`LIS_CONTEXT_OVER_TRAINED_REJECT`). Any other value is rejected; they are not placeholders for future variants.
- The RoPE / rope-scaling validator in both Llama and Qwen3 config parsers rejects any non-null `rope_scaling` value (including `{}`) and any non-default `rope_type` with `LIS_STATUS_UNSUPPORTED`.

**Extension Points:**

- Later GPT-2, Mistral, and GPT-OSS metadata mapping.
- Extended RoPE or attention layout variants after explicit implementation and validation.
- Model-specific validation hooks.

## Tensor and Core Types

**Purpose:** Provide shared tensor metadata and storage ownership rules for loader, backend, and runtime code.

**Likely Public Interface Responsibility:**

- Represent dtype, rank, shape, stride, storage size, and optional storage ownership.
- Support tensor views without confusing view lifetime with owned storage lifetime.
- Provide safe size and byte-count calculations.

**Key Data Structures:**

- Tensor descriptor.
- Tensor storage owner.
- Tensor view.
- Dtype enum.
- Shape/stride helper structures.

**Extension Points:**

- Backend-owned storage for future GPU or specialized CPU memory.
- Additional dtypes after loader and backend support exist.
- Layout-specific views for optimized kernels.

## Tokenizer

**Purpose:** Convert between text and token IDs for prompt encoding and output decoding, using BPE over a loaded vocabulary.

**Public Interface Responsibility:**

- Load a vocabulary and merge table from a LIS_VOCAB_V1 file.
- Encode text to token IDs via byte-pair encoding merge rules.
- Decode token IDs to text via vocabulary byte-string lookup.
- Manage tokenizer lifecycle (load, destroy) with explicit ownership.

**Key Data Structures:**

- Tokenizer object owning vocabulary byte strings, lengths, merge table, and byte-to-token lookup.
- Merge table as an open-addressing hash mapping `(first, second)` token pairs to `(result, rank)`.
- BPE node linked list used internally during encoding.

**Scope Limitation:**

LIS_VOCAB_V1 is an internal canonical format with hex-encoded token byte strings and explicit merge rules. It is not a parser for SentencePiece `.model` or tiktoken files.

HuggingFace `tokenizer.json` (BPE type) import is supported as an additional
loader path behind the same `lis_tokenizer` struct, eliminating the need for
external conversion tooling for the most common real-world tokenizer artifact
format.

**Extension Points:**

- Additional vocabulary format parsers behind the same `lis_tokenizer` struct.
- Priority-queue optimization for the BPE merge loop.
- Special token management framework.

## Loader

**Purpose:** Convert supported model files into validated model metadata and tensors without leaking file format details into runtime execution.

**Likely Public Interface Responsibility:**

- Open and validate model files.
- Parse safetensors metadata for the supported subset.
- Map file tensors into LIS tensor/core structures.
- Report unsupported dtype, shape, tensor naming, file format, and I/O failures.

**Key Data Structures:**

- Loader context.
- Loaded model object.
- Tensor table or named weight registry.
- Loader error/result type.

**Extension Points:**

- Additional safetensors naming layouts.
- Limited PyTorch-exported compatibility.
- Future save support after explicit save semantics are documented.

## Runtime Context

**Purpose:** Own execution lifecycle for a loaded model under a selected backend, batch configuration, and context window policy.

**Likely Public Interface Responsibility:**

- Initialize and destroy runtime state.
- Validate model metadata, configured context length, batch size, and backend support.
- Own execution buffers and route calls to prefill and decode paths.
- Own static batch positions and a KV cache without requiring model execution.
- Run the CPU reference Llama forward path for HuggingFace-local imports with
  batch 1.
- Own a thread pool for parallel Llama forward compute. Thread count is
  configurable via `--threads N` in runtime options.
- Provide a minimal repetition-penalty generation-quality change only; it does
  not expand runtime batch size, KV cache layout, or model math.
- Keep canonical prompt construction on the CLI/tokenizer side; diagnostics
  observe existing generation decisions.
- Keep extended top-k candidate diagnostics on the CLI/diagnostic side.

**Context Policy Invariants:**

- Runtime init ordering: context/positional validation → backend/batch validation → static batch allocation → KV cache allocation → thread pool initialization. A context or positional policy failure must surface before any mutable runtime state is allocated.
- Prefill accepts `len ≤ configured_max_tokens`. Equality is allowed, with the documented consequence that decode immediately returns `LIS_STATUS_LIMIT_EXCEEDED` and zero tokens are emitted.
- Decode returns `LIS_STATUS_LIMIT_EXCEEDED` when any positions `≥ configured_max_tokens` before advancing. Positions are not mutated by the exhausted step.

**Key Data Structures:**

- Runtime context object.
- Runtime options object (including thread count).
- Backend handle or dispatch table reference.
- Execution buffer set.
- Runtime phase enum for ready, prefilled, and decoding states.
- Thread pool: owned by runtime context, created at init, destroyed at cleanup.

**Extension Points:**

- Additional decode modes.
- Backend-specific runtime state.
- Scheduler upgrades beyond static batching.
- Thread pool scaling beyond the initial fork-join model.

## Scheduler and Batching

**Purpose:** Define how token batches are presented to prefill and decode.

**Likely Public Interface Responsibility:**

- Represent static batch dimensions and per-sequence lengths.
- Enforce synchronized iterative decode constraints.
- Provide a future boundary for continuous batching or admission control.
- Reject prefill lengths that exceed configured context and advance all batch
  entries synchronously during decode.

**Key Data Structures:**

- Static batch descriptor.
- Sequence state table.
- Decode step descriptor.

**Extension Points:**

- Continuous batching.
- Request admission policy.
- Per-sequence stopping behavior in future explicitly scoped batching work.

## KV Cache

**Purpose:** Store attention key/value tensors across prefill and decode.

**Likely Public Interface Responsibility:**

- Allocate cache storage for configured model, batch, context, and dtype.
- Provide safe indexing for layers, batch entries, positions, heads, and head dimensions.
- Separate cache lifecycle from operator implementation.
- Expose key/value offset and pointer helpers without implementing attention
  computation.

**Key Data Structures:**

- KV cache object.
- KV cache layout descriptor.
- Per-layer cache view.
- Position cursor or sequence length table.

**Current Implementation Entry Points:**

- `lis_kv_cache_init` computes the current flat cache layout from
  `lis_model_config` and batch size, then allocates separate zeroed key and
  value buffers. The context length comes from
  `metadata.config.context.configured_max_tokens` through the runtime-owned
  model config.
- `lis_kv_cache_destroy` frees both buffers and clears the cache object.
- `lis_kv_cache_element_offset` validates `(layer, batch, position, head, dim)`
  against the layout and maps that logical KV address to a byte offset.
- `lis_kv_cache_key_ptr` and `lis_kv_cache_value_ptr` are the current validated
  pointer helpers for reading or writing the key/value buffers. Llama and Qwen3
  runtime paths use these helpers for KV storage and attention reads.
- `lis_runtime_init` owns KV cache creation as part of runtime lifecycle setup,
  after metadata, context-policy, backend, and batch validation.
- `lis_runtime_destroy` owns KV cache teardown as part of runtime cleanup.

These are current implementation entry points only. They do not require new C
symbols. The internal KV boundary notes record minimal boundary properties and
candidate directions only, without ratifying future function names, structs,
vtables, or a C API shape.

**Internal KV Boundary Direction:**

The internal KV boundary direction confirms boundary properties and candidate
shapes only. It does not
document any new C function as implemented, add a fake signature, or ratify a
`lis_kv_cache_ops`-style API. The safest first implementation direction, if a
helper seam is needed for KV memory transparency, is query/policy/accounting
over the existing layout and runtime state. A per-position commit helper
remains deferred future work, and it must never advance logical `used_tokens`
from per-layer writes if introduced later. Write-path migration is deferred and
must be attempted only after tests exist; it must preserve current
`lis_kv_cache_element_offset` / key-pointer / value-pointer semantics and
generated-token behavior.

**KV Precision/Storage Policy:**

The KV precision/storage policy is documentation only. It does not add C symbols or change KV
read/write logic. The current KV storage dtype is `lis_kv_cache_layout.dtype`,
computed by `lis_kv_cache_init` from `config->weight_dtype`. Qwen3 and Llama
produce K/V values in FP32 scratch buffers and store them through their current
local `lis_store_kv` helpers with `lis_scalar_write` using
`runtime->kv_cache.layout.dtype`. Attention reads pass that dtype to
`lis_attn_qk` and `lis_attn_pv`; FP32 attention accumulation semantics remain
unchanged. `report.kv_cache.storage_dtype` must agree
with the `kv=<dtype>` component of `manifest.runtime.precision_path`.

**Extension Points:**

- Alternative cache layouts for SIMD or GPU.
- Paged or segmented cache designs.
- Quantized cache storage if later authorized.

## Decode Loop

**Purpose:** Coordinate iterative generation after prefill.

**Likely Public Interface Responsibility:**

- Run synchronized decode steps.
- Consume and update KV cache state.
- Produce logits and apply greedy next-token selection for the initial mode.
- Enforce context and stop conditions.
- Provide prefill/decode state validation, synchronized position advancement,
  configured context checks, and a minimal greedy logits argmax helper. The
  static batch scheduler still does not own per-sequence EOS/stop policy.
- Provide Llama prefill/decode APIs that produce logits by executing
  embeddings, RMSNorm, attention, MLP, KV cache updates, and lm_head projection
  for batch 1. The CLI greedy loop stops on configured EOS tokens. For
  tokenizer-backed text output, the CLI also keeps known structural Llama
  chat/control tokens out of visible assistant text by stopping on end-of-turn
  controls and suppressing header/BOS controls from selection.
- Parallelize the per-token forward pass across CPU cores via thread pool
  dispatch in matvec, attention, plain RoPE, RMSNorm, and SwiGLU operations.
- Add only a minimal repetition penalty for supported user-facing generation
  without introducing top-k, top-p, temperature, beam search, speculative
  decoding, or a broad decode-policy subsystem. The CLI generation loop tracks
  tokens emitted during the current generation and applies a fixed `1.2`
  penalty to repeated candidates before greedy selection.
- Expose minimal opt-in diagnostics for the existing greedy path: selected
  token ID, selected token text when available, stop reason, and whether
  structural-token suppression or the repetition penalty affected selection.
  This does not add a new decode mode or policy framework.
- Extend the diagnostic surface to expose top-k candidate token IDs, token
  texts, raw logit scores, and adjusted scores at each generation step. The
  selected candidate is marked among the top-k entries. This does not add
  sampling, decode-policy changes, or new CLI arguments.

**Context Policy Invariants:**

- Prefill boundary: `len ≤ configured_max_tokens`. Equality is allowed; the documented consequence is that decode immediately returns `LIS_STATUS_LIMIT_EXCEEDED`.
- Decode precheck: any `positions[i] ≥ configured_max_tokens` → `LIS_STATUS_LIMIT_EXCEEDED` without advancing positions or emitting a token.
- The CLI stop reason at context exhaustion is exactly `context_limit`. The preceding emitted tokens remain in `emitted_token_ids`; the exhausted step does not append.

**Key Data Structures:**

- Decode state.
- Decode options.
- Logits view.
- Token output buffer.
- Thread pool dispatch context.

**Extension Points:**

- Sampling modes.
- Speculative decoding.
- Per-sequence stopping policy enhancements in future explicitly scoped
  batching work.

## Backend and Operator Layer

**Purpose:** Isolate runtime/model logic from concrete execution implementations.

**Likely Public Interface Responsibility:**

- Define operator contracts for supported tensor operations.
- Dispatch to CPU reference backend first.
- Leave room for SIMD, optimized matmul, custom kernels, and GPU backend interfaces.
- Validate and dispatch f32 elementwise add and f32 2D matmul against
  host-memory tensors.

**Key Data Structures:**

- Backend descriptor or vtable.
- Operator request descriptors.
- Backend capability flags.
- Backend-owned buffer handle design, when needed later.
- Backend memory-domain marker for host memory now and backend-owned buffers later.

**Extension Points:**

- CPU SIMD backend.
- Optimized matmul backend.
- GPU backend interface and later implementation.
- Custom operator insertion following a compile-time-fixed policy with centralized validation.

## CLI and Driver

**Purpose:** Provide the first local offline entrypoint for loading a model and running the narrow inference target.

**Likely Public Interface Responsibility:**

- Parse user arguments.
- Configure model path, prompt/token input, context length, batch size, and generation length.
- Connect loader, tokenizer boundary, runtime, and output.
- Report user-facing diagnostics.

**Validation Driver Scope:**

- Accept only explicit local paths and numeric limits: safetensors model path, Llama-style config path, direct token-ID path, configured context length, static batch size, and greedy generation limit.
- Use the loader/config parser, token-ID boundary, runtime lifecycle, and CPU
  backend operator dispatch.
- Treat `lis.validation_logits` as the validation tensor for the first runnable target rather than implementing full transformer model execution.
- Keep serving, sampling modes, GPU execution, and production lifecycle concerns out of the CLI.

**HuggingFace-Local Llama Support:**

- Dispatch HuggingFace-local model directories to the CPU reference Llama forward path.
- Preserve the `lis.validation_logits` path for direct safetensors validation fixtures.
- Keep the real-forward path batch-1 only until static-batch tensor execution is explicitly expanded.

**Threading Support:**

- `--threads N` CLI argument controlling thread pool size (default: 1).
- Thread count is passed through runtime options to the thread pool.

**Repetition Penalty Support:**

- Minimal repetition penalty for existing user-facing generation.
- No batch-size expansion, new sampling framework, or broad decode-policy redesign.
- Focused tests and documentation for repetition-control behavior.

**Prompt Construction and Minimal Diagnostics:**

- Canonical Llama Instruct prompt construction for the currently supported HuggingFace tokenizer-backed CLI path.
- Minimal opt-in generation diagnostics for selected token ID/text, stop reason, and suppression/penalty state.
- `--diagnostics` writes the minimal generation records to stderr and does not change default stdout generation output.
- No multi-turn session framework, broad prompt-template system, or major CLI redesign.

**Extended Candidate Diagnostics:**

- Extended top-k candidate entries in the existing `--diagnostics` output: candidate token IDs, token texts, raw logit scores, adjusted scores, and selected-candidate marking.
- No new CLI arguments, no sampling framework, no decode-policy changes.

**Decode-Trace Artifact Support:**

- `--trace-json PATH` CLI flag for bounded per-step decode-trace artifact emission.
- Trace record collected in both decode loops when `--trace-json` is active.
- Top-k extraction runs when trace is enabled regardless of `--diagnostics`.
- No change to default generation output, decode policy, or any existing CLI surface.

**BPE Vocabulary Prompt Mode:**

- `--vocab PATH --prompt TEXT` as an alternative input mode to `--tokens PATH`.
- When a tokenizer is loaded, generated tokens are decoded to text output.
- The two input modes are mutually exclusive; existing `--tokens` path is unchanged.

**Key Data Structures:**

- CLI options object.
- Driver result/status.
- Input token batch descriptor.

**Extension Points:**

- Additional input formats.
- More decode options after they are implemented.
- Future service wrapper outside the initial scope.

## Logging and Error Utilities

**Purpose:** Provide consistent diagnostics without hiding failure modes.

**Likely Public Interface Responsibility:**

- Represent subsystem error codes.
- Format concise user-facing error messages.
- Preserve enough detail for tests and debugging.

**Key Data Structures:**

- Error code enum or subsystem result types.
- Optional error detail object.
- Logging configuration.

**Extension Points:**

- Structured logs.
- Verbosity levels.
- Diagnostic tracing for backend or loader debugging.

## Performance Measurement

**Purpose:** Provide an opt-in, zero-allocation wall-clock measurement surface so the CLI driver can emit stage-by-stage timings and derived summary metrics (TTFT, mean ITL, steady-state tokens/sec, end-to-end tokens/sec) for an inference run. Strictly observational; does not participate in inference semantics.

**Likely Public Interface Responsibility:**

- Expose a monotonic nanosecond clock wrapper over `clock_gettime(CLOCK_MONOTONIC)`.
- Own a flat report struct that tracks a fixed set of seven stages and an optional per-step sample buffer, with caller-managed lifetime.
- Offer `begin`/`end` scoping and an `accumulate` convenience for already-captured interval endpoints.
- Emit stage, optional per-step, and summary lines to stderr in the stable `lis: perf-*` grammar when enabled; no-op when disabled.

**Key Symbols:**

- `lis_perf_now_ns(void) -> uint64_t`
- `lis_perf_stage_id` enum: `LIS_PERF_STAGE_MODEL_LOAD`, `LIS_PERF_STAGE_TOKENIZER_LOAD`, `LIS_PERF_STAGE_TOKENIZER_ENCODE`, `LIS_PERF_STAGE_RUNTIME_INIT`, `LIS_PERF_STAGE_PREFILL`, `LIS_PERF_STAGE_FIRST_DECODE`, `LIS_PERF_STAGE_DECODE_STEADY_STATE`
- `lis_perf_report` struct (fixed slots, no heap allocation)
- `lis_perf_report_init(lis_perf_report *)`
- `lis_perf_stage_begin(lis_perf_report *, lis_perf_stage_id)`
- `lis_perf_stage_end(lis_perf_report *, lis_perf_stage_id, uint64_t tokens)`
- `lis_perf_stage_accumulate(lis_perf_report *, lis_perf_stage_id, uint64_t ns, uint64_t tokens)`
- `lis_perf_emit_per_token(const lis_perf_report *, FILE *, size_t step, uint64_t ns)`
- `lis_perf_report_emit(const lis_perf_report *, FILE *, int threads, size_t prompt_tokens, size_t generated_tokens)`

**Key Data Structures:**

- `lis_perf_report` — contains the caller-supplied `tag` pointer, `enabled` and `per_token_enabled` flags, a start-timestamp field per stage for open scopes, and fixed-size arrays for accumulated ns and token counts indexed by `lis_perf_stage_id`.

**Extension Points:**

- Additional stage IDs may be added by extending the enum and the per-stage arrays.
- Per-operator breakdown can be layered on as separate future work without changing the existing API surface.
- GPU-era instrumentation can reuse the same report struct with a backend-specific clock source wrapper.

## SIMD Backend

**Purpose:** Expose a runtime-dispatched SIMD CPU backend alongside the
reference kernels, intended to improve CPU hot-path throughput when the AVX2
backend is enabled, without replacing or modifying the reference path, without
changing any public operator signature, and without exposing AVX intrinsics in
shared headers. Validation is driven by kernel-level diff tests and by the
performance measurement harness.

**Likely Public Interface Responsibility:**

- Expose a one-shot CPUID feature probe (`lis_cpu_features_get`) that reports the booleans used to select kernels.
- Expose a dispatch initialization call (`lis_cpu_dispatch_init`) invoked once at runtime init with the feature struct; populates the function-pointer table and honours the `LIS_SIMD=0` override.
- Keep `lis_matvec`, `lis_rms_norm`, `lis_softmax`, `lis_swiglu`, `lis_residual_add`, `lis_rope`, and attention inner-dot entry points as thin trampolines that indirect through the table; signatures are stable across the reference and SIMD dispatch paths.
- Provide per-kernel AVX2 implementations as internal symbols (`lis_matvec_avx2_fma`, etc.) with the same contract as the reference kernels and no direct caller exposure.

**Key Symbols:**

- `lis_cpu_features` struct (flat booleans: `sse2`, `avx`, `avx2`, `fma`, `f16c`, `avx512f`, `avx512vl`, `bmi2`)
- `lis_cpu_features_get(void) -> const lis_cpu_features *`
- `lis_cpu_ops` struct (function pointers for each dispatched operator)
- `lis_cpu_dispatch_init(const lis_cpu_features *) -> void`
- `lis_cpu_dispatch_backend_name(void) -> const char *` (reports `"reference"`, `"avx2"`, or `"avx512"` for the `lis: simd backend=...` diagnostic line)
- Reference kernels: `lis_matvec_reference`, `lis_rms_norm_reference`, `lis_softmax_reference`, `lis_swiglu_reference`, `lis_residual_add_reference`, `lis_rope_reference`, `lis_attn_qk_reference`, `lis_attn_pv_reference`
- AVX2 kernels: `lis_matvec_avx2_fma`, `lis_rms_norm_avx2`, `lis_softmax_avx2`, `lis_swiglu_avx2`, `lis_residual_add_avx2`, `lis_rope_avx2`, `lis_attn_qk_avx2`, `lis_attn_pv_avx2`
- Optional AVX-512 kernels: `lis_matvec_avx512f`, `lis_softmax_avx512f`

**Key Data Structures:**

- `lis_cpu_features` — cached once on first call to `lis_cpu_features_get`; plain-data, no pointers.
- `lis_cpu_ops` — one function pointer per dispatched operator, shape identical to the reference kernel signature. Populated by `lis_cpu_dispatch_init` and read via the trampolines. No per-call branch on CPU capability.

**Extension Points:**

- Adding a new optimized kernel (e.g. AVX-512, ARM NEON, or a specialized batched GEMM) is a matter of adding an implementation and a dispatch branch; no caller changes.
- GPU backends can reuse the same dispatch shape with a different `lis_cpu_ops`-equivalent table scoped to the GPU runtime; the public entry points remain the stable boundary.
- `LIS_SIMD=0` extends naturally to `LIS_SIMD=avx2`, `LIS_SIMD=avx512`, or `LIS_SIMD=reference` as additional variants land.

## Verification Framework

**Purpose:** Provide a backend-aware, repeatable verification surface for kernel checks, selected token parity, bounded CLI regression, backend/path observability, and regression-result classification. Verification is an engineering asset for runtime trustworthiness, not an ad hoc set of manual commands.

**Likely Public Interface Responsibility:**

- Expose make-based verification entry points that reuse existing binaries and scripts.
- Keep kernel verification backend-neutral; the initial `verify-kernels` implementation runs the existing AVX diff binary, but future non-SIMD kernel checks can join the same target.
- Compare fixed-prompt token parity by exact generated token IDs using existing generation diagnostics.
- Classify results as pass, documented unsupported, numeric regression, token-parity regression, benchmark/protocol regression, or harness/configuration error.
- Reuse existing `lis: ...` stderr diagnostics for backend/path observability.

**Key Entry Points:**

- `make verify`
- `make verify-kernels`
- `make verify-cli`
- `make verify-token-parity`
- `make verify-perf-smoke`
- `tests/verification/run_token_parity.py`

**Key Data Structures:**

- No new runtime data structure. Verification is driven by Makefile targets,
  existing C test binaries, a small Python token-parity harness, and documented
  result classes.

**Extension Points:**

- Precision-aware comparisons and Qwen3 Dense checks add cases under this framework rather than redefining tolerance, result-class, or backend/path-observability policy.
- `tests/perf/run_perf_matrix.py` may carry an already-emitted backend/path value into a narrow artifact field. The verification framework must not grow new benchmark modes, metrics, or reporting surfaces; bounded execution artifacts provide the shared artifact contract.

## Precision Policy

**Purpose:** Define the supported CPU Llama-family precision semantics for `F32`, `F16`, and `BF16` model weights with FP32 compute and accumulation.

**Likely Public Interface Responsibility:**

- Accept `float32`/`f32`, `float16`/`f16`, and `bfloat16`/`bf16` as config weight dtypes.
- Preserve native safetensors storage and avoid load-time bulk conversion.
- Require all mapped HuggingFace tensors to match `config.weight_dtype`; mixed per-tensor dtype artifacts are documented unsupported scope.
- Promote stored weights to FP32 at compute boundaries.
- Store KV cache in `config.weight_dtype`, after producing K/V in FP32 scratch, and promote KV values back to FP32 for attention math.

**Key Symbols:**

- `lis_dtype_scalar_read_f32`
- `lis_model_config.weight_dtype`
- `lis_kv_cache_layout.dtype`
- `lis_matvec`, `lis_rms_norm`, `lis_attn_qk`, `lis_attn_pv`

**Extension Points:**

- The precision policy does not add a generalized mixed-precision framework. Future work must explicitly authorize mixed per-tensor policy, separate KV precision, quantization, GPU mixed precision, or optimized precision kernels.
- KV storage remains single-dtype for the run. Mixed K/V storage
  dtypes, per-layer KV dtype variation, runtime KV dtype switching, new dtype
  support, weight dtype policy changes, FP32 accumulation changes, and
  quantized KV cache formats are outside current public support.

## Qwen3 Dense Support

**Purpose:** Add a narrow, family-specific Qwen3 Dense execution path for
documented local Dense artifacts loaded from user-supplied HuggingFace-style
directories, without turning LIS into a broad Qwen-family runner or general Qwen3 artifact
runner.

**Likely Public Interface Responsibility:**

- Parse and validate the documented Qwen3 Dense config subset:
  `model_type == "qwen3"`, `architectures == ["Qwen3ForCausalLM"]`, dense
  decoder-only settings, BF16 dtype, plain RoPE, no sliding-window attention,
  no attention bias, and `hidden_act == "silu"`.
- Route accepted configs to a Qwen3 Dense model family while preserving the
  existing Llama-family config and runtime behavior.
- Load only documented merged `model.safetensors` Dense target artifacts from
  user-supplied local directories. When a merged `model.safetensors`
  exists alongside an auxiliary `model.safetensors.index.json`, load the merged
  file and ignore the auxiliary index; reject index-only shard loading as
  unsupported.
- Map the explicit Qwen3 Dense tensor set, including per-layer
  `self_attn.q_norm.weight` and `self_attn.k_norm.weight`, with shape and dtype
  validation against config values. Qwen3 Dense treats `hidden_size`,
  `attention_heads * head_dim`, and `kv_heads * head_dim` as distinct widths.
- Provide a batch-1 Qwen3 Dense prefill/decode path that executes token
  embeddings, input RMSNorm, Q/K/V projection, Q/K RMSNorm, plain RoPE, GQA
  causal attention, output projection, gated `silu` MLP, final norm, and lm head
  logits.
- Reuse the documented BF16/FP32 precision semantics and verification result
  classes.
- Keep tokenizer support limited to direct token IDs and bounded plain-text
  encode validation through the existing HuggingFace BPE import path; do not add
  a Jinja/chat-template engine.

**Key Data Structures:**

- Qwen3 Dense model-family enum value.
- Existing `lis_model_config` fields where sufficient for hidden size, head
  counts, head dim, vocab size, RoPE theta, dtype, and context.
- Explicit Qwen3 tensor mapping table for the supported first target.

**Extension Points:**

- Other Qwen3 artifact layouts, sharded checkpoints, MoE, multimodal/VL,
  long-context/YaRN, sliding-window attention, chat-template support, sampling
  policies, serving, GPU execution, and quantization require later explicit
  phases.
- Shared runtime helper extraction is allowed only after the family-specific
  Qwen3 path proves the behavior and only when the extraction is
  behavior-preserving for the existing Llama path.

**Runtime Duplication Technical Debt:** The runtime helpers duplicated between
`llama.c` and `qwen3.c` are acceptable for the supported Qwen3 Dense path but
increase bug-fix drift risk and verification burden. Future work should extract
shared behavior-preserving utilities into common runtime-internal TUs while
preserving distinct family-specific entry points. No premature generic-runtime
unification.

## Reproducibility and Execution Artifact Framework

**Purpose:** Leave supported LIS runs as bounded, machine-readable artifacts
that explain what executed and how it stopped without turning LIS into a trace
system or telemetry platform. Add a human-readable Markdown companion report
for reviewer convenience without changing the canonical JSON contract.

**Likely Public Interface Responsibility:**

- Parse `--report-json PATH` and `--report-md PATH` in the CLI and keep artifact emission opt-in.
- Emit one versioned core run-report object under schema
  `lis.execution_artifact/v1`.
- Capture manifest identity for the binary, model, config, tokenizer-or-token
  input, runtime settings, and resolved backend through bounded fingerprints.
- Record prompt-sequence token counts and token-ID digests, selected token IDs,
  emitted token IDs, structured status, and stop/failure reason.
- Emit deterministic KV cache memory accounting under `report.kv_cache` and in
  the Markdown companion. The CLI computes this from `runtime.kv_cache.layout`
  and current runtime batch positions; `used_tokens` is logical populated
  positions and is not derived from per-layer writes.
- Embed optional perf summary data in the same run report when `--perf` is
  enabled.
- Reuse the same core run-report object inside verification and benchmark
  snapshot JSON rather than defining disconnected schema families.
- Provide a separate Markdown emission of the same bounded data organized by
  reviewer priority, with compact bullets, `fnv1a64:`-prefixed fingerprint
  digests, `OK`/`error` status capitalization, and bounded token-ID previews
  (up to 16 IDs in backticks, remainder as an out-of-backticks count).

**Key Symbols:**

- `lis_artifact_fingerprint`
- `lis_artifact_prompt_sequence`
- `lis_artifact_run_report`
- `lis_artifact_fingerprint_file`
- `lis_artifact_fingerprint_current_binary`
- `lis_artifact_fingerprint_token_ids`
- `lis_artifact_fingerprint_runtime`
- `lis_artifact_fingerprint_backend`
- `lis_artifact_write_run_report`
- `lis_artifact_write_run_report_md`
- `lis_artifact_kv_cache_report`

**Extension Points:**

- Future work may add bounded mode-specific snapshot wrappers around the same
  core run report, but must not create separate incompatible schema families.
- Raw prompt retention, generated-text retention, trace/logits/activation
  capture, remote upload, dashboards, and cross-backend equivalence claims stay
  out of scope unless separately authorized.

## Decode-Step Trace

**Purpose:** Capture bounded per-step decode decisions as a machine-readable artifact for debugging and regression analysis. The trace shares the run-report manifest identity and `lis.execution_artifact/v1` schema but carries kind `decode_trace` instead of `run_report`.

**Likely Public Interface Responsibility:**

- Parse `--trace-json PATH` in the CLI.
- Record per-step trace entries during decode: step index, phase, selected token ID, raw/adjusted scores, runner-up, decision margin, suppression/penalty state, decision class, top-k candidates (up to `LIS_TRACE_TOPK_SIZE` entries).
- Write a fail-closed JSON artifact with the same manifest as `run_report`.
- Omit raw prompt text, generated text, absolute paths, full-vocab rankings, logits dumps, timestamps.

**Key Symbols:**

- `lis_trace_phase` enum: `LIS_TRACE_PHASE_PREFILL_SEED` (defined but not emitted in current decode loops), `LIS_TRACE_PHASE_FIRST_DECODE` (emitted as `first_decode` for step 0), `LIS_TRACE_PHASE_DECODE` (emitted as `decode` for step > 0)
- `lis_trace_topk_entry` struct: `token_id`, `raw_score`, `adjusted_score`, `is_selected`
- `lis_trace_step` struct: `step`, `phase`, `selected_token_id`, `raw_score_selected`, `adjusted_score_selected`, `runner_up_token_id`, `runner_up_adjusted_score`, `decision_margin`, `structural_suppression_affected`, `repetition_penalty_changed_selection`, `selected_token_penalized`, `suppressed_token_count`, `penalized_token_count`, `decision_class` (values: `greedy`, `structural_suppression`, `repetition_penalty_shifted`; `structural_suppression` takes priority when both conditions apply), `topk[LIS_TRACE_TOPK_SIZE]`, `topk_count`, optional `stop_reason` (emitted in JSON only on final step when a stop condition is reached)
- `lis_trace_record` struct: `steps`, `count`, `capacity`
- `lis_trace_record_init`, `lis_trace_record_destroy`, `lis_trace_record_append`
- `lis_trace_artifact_write`

**Key Data Structures:**

- `lis_trace_record` — caller-owned, fixed-capacity array of `lis_trace_step` entries, capacity set at init to `generation_limit`.
- `lis_trace_topk_entry` — flat struct per candidate.

**Extension Points:**

- Future phases may add bounded mode-specific trace wrappers, but must not create separate incompatible schema families.
- Raw prompt retention, generated-text retention, full-vocab ranking, logits/activation dumping, remote upload, and dashboards remain out of scope.

## Precision Path Observability

**Purpose:** Surface the actually-resolved precision identity (compute accumulation, weights dtype, KV-cache dtype) alongside the backend identity, so verification consumers can observe precision path without inferring it from backend name.

**Likely Public Interface Responsibility:**

- Build `precision_path` once in `srcs/cli/driver.c` after `lis_runtime_init` succeeds, reading `model->metadata.config.weight_dtype` and `runtime->kv_cache.layout.dtype` via `lis_dtype_name`.
- Format: `"f32_accum;weights=<dtype>;kv=<dtype>"` for artifacts; human-readable stderr uses space-separated `lis: precision path=f32_accum weights=<dtype> kv=<dtype>`.
- Emit the stderr line exactly once, immediately after `lis: simd backend=...`, gated identically (`diagnostics_enabled || perf.enabled`).
- Thread `precision_path` into `lis_artifact_run_report.precision_path`, `lis_trace_artifact.precision_path`, and `lis_layer_trace_artifact.precision_path` before calling their writers.
- Writers emit `"precision_path":"..."` inside `manifest.runtime` as an additive field.
- Do not pass `precision_path` into `lis_artifact_fingerprint_runtime` or `lis_artifact_fingerprint_backend`; existing fingerprint digests must remain stable for unchanged hashed inputs. `runtime_fingerprint.size_bytes` is a stable semantic sum, so this stability statement is about digest identity and hash inputs rather than every emitted fingerprint metadata field.

**Key Symbols:**

- `precision_path` string (fixed-size stack buffer in `srcs/cli/driver.c`).
- `report->precision_path` on `lis_artifact_run_report`.
- `artifact->precision_path` on `lis_trace_artifact`.
- `artifact->precision_path` on `lis_layer_trace_artifact`.

**Key Data Structures:**

- Same `lis_artifact_run_report`, `lis_trace_artifact`, and `lis_layer_trace_artifact` structs with additive `const char *precision_path`.

**Extension Points:**

- Future work may add `precision_fingerprint` (a hashed identity) if required; it is rejected / out of scope for current public support.
- Per-step precision reporting is rejected / out of scope.

## Layer-Trace Artifact

**Purpose:** Capture stat-bearing tensor-summary `lis: layer-checkpoint` stderr lines into a compact, structured JSON artifact (`--layer-trace-json`), reusing the same `manifest` identity as `--trace-json`, with byte-exact `%.6g` parity between stderr and JSON fields.

**Likely Public Interface Responsibility:**

- `lis_cli_parse_options` recognizes `--layer-trace-json PATH` into `lis_cli_options.layer_trace_json_path`.
- `srcs/cli/driver.c` pre-validates the flag dependency (`--layer-trace-json` requires `--layer-checkpoints`), allocates `lis_layer_trace_record`, threads it through `lis_runtime_options` into `lis_runtime_context`, destroys it on all exit paths.
- `srcs/runtime/llama.c` and `srcs/runtime/qwen3.c` checkpoint helpers compute once into a stack-local `lis_layer_trace_step`, emit stderr with their existing exact `printf` format strings, then optionally append the step into `runtime->layer_trace_record`.
- `lis_layer_trace_artifact_write` emits JSON artifact with `kind:"layer_trace"`, same `manifest` shape as `decode_trace` including `precision_path`, and a top-level `layer_trace[]` array.
- Floats serialize with `%.6g`; non-finite values serialize as `null` with companion `nan`/`inf` ints (0 or 1).
- The scalar value-only `attn_scale` emit (llama layer 1) is excluded from the JSON array by construction; it is emitted directly by an inline `fprintf` and never routed through `lis_layer_trace_step`.
- Overflow is bounded: geometric growth (×2) from `LIS_LAYER_TRACE_INITIAL_CAPACITY = 64` up to `LIS_LAYER_TRACE_HARD_MAX = 8192`. A sticky `append_failed` flag suppresses the artifact (no file created or overwritten) and propagates `LIS_STATUS_OVERFLOW`.

**Key Symbols:**

- `lis_layer_trace_step` — one captured checkpoint line with fixed-size `phase[16]` and `name[96]`; truncation treated as failure.
- `lis_layer_trace_record` — growable array of steps with `append_failed` sticky flag.
- `lis_layer_trace_artifact` — manifest identity and path for the writer.
- `lis_layer_trace_record_init`, `lis_layer_trace_record_destroy`, `lis_layer_trace_record_append`, `lis_layer_trace_artifact_write`.
- `precision_path[64]` is threaded into `artifact.precision_path` before the writer call.

**Key Data Structures:**

- `lis_layer_trace_step.step`, `phase`, `name`, `shape[LIS_LAYER_TRACE_MAX_RANK]`, `rank`, `min`, `max`, `mean`, `l2`, `nan`, `inf`.
- `lis_layer_trace_record.steps`, `step_count`, `step_capacity`, `append_failed`.
- `lis_layer_trace_artifact` mirrors fields from `lis_trace_artifact` for manifest identity.

**Extension Points:**

- The scalar value-only `attn_scale` checkpoint remains stderr-only; capturing it would require an explicit contract change (no pre-staging).
- Future backend extensibility (GPU, optimized kernels) remains transparent: the checkpoint helpers themselves determine what is captured.

## LIS Inspect Compatibility Protection

**Purpose:** Lock the compatibility surface without expanding LIS Inspect.
Current LIS Inspect support covers:
`run_report` JSON and stderr perf logs. `decode_trace` and `layer_trace` are
valid LIS artifacts using schema `lis.execution_artifact/v1`, but current LIS
Inspect is not required to parse or display those artifact kinds.
Trace/layer Inspect support is deferred to future Inspect-owned work.

**Likely Public Interface Responsibility:**

- Preserve `--report-json` key names and top-level order (`schema`, `kind`,
  `manifest`, `report`) for the existing `run_report` surface.
- Keep JSON perf data under `report.perf`; do not introduce
  `report.perf_summary` as a replacement.
- Keep `manifest.runtime.precision_path` additive.
- Preserve stderr prefixes and field sets for `lis: perf-stage `,
  `lis: perf-summary `, `lis: perf-per-token `,
  `lis: generation-diagnostic `,
  `lis: generation-diagnostic-candidate `, `lis: layer-checkpoint `, and
  `lis: simd backend=`.
- Treat `lis: generation-diagnostic-reasoning ` and `lis: precision path=` as
  additive stderr lines only.
- Keep `decode_trace` and `layer_trace` schema/kind validity independent from
  current Inspect support.

**Key Data Structures:**

- Existing `lis_artifact_run_report` and `lis_perf_report` structures. No new
  runtime data structure is required.

**Extension Points:**

- Trace/layer support in LIS Inspect belongs to future Inspect-owned work.
- KV cache rendering, interpretation, graphing, comparison, or
  visualization also belongs to future Inspect-owned work. Current
  Inspect support is compatibility-only: preserve raw `report.kv_cache` and
  ignore `lis: kv-cache:` in perf stderr parsing.
- `runtime_fingerprint.size_bytes` changed from
  `sizeof(lis_cli_options)` to a stable semantic sum of hashed runtime inputs;
  this accepted nuance does not change runtime fingerprint digest identity for
  unchanged hashed inputs.

## Selection Reasoning Diagnostics

**Purpose:** Make per-step selection reasoning available from the diagnostics path and visible through both stderr and trace outputs, with the diagnostics struct as the single source of truth.

**Likely Public Interface Responsibility:**

- Extend `lis_cli_selection_diagnostics` with `raw_score_selected`, `adjusted_score_selected`, `runner_up_token_id`, `runner_up_adjusted_score`, `suppressed_token_count`, `penalized_token_count`, `decision_class`.
- Populate all fields inside `lis_cli_select_generation_token`.
- Emit `lis: generation-diagnostic-reasoning` after the existing diagnostic
  header line and before candidate lines, only when `--diagnostics` is enabled.

**Key Symbols:**

- `lis_cli_selection_diagnostics` (extended struct in `srcs/cli/driver.c`)
- Reasoning line format: `lis: generation-diagnostic-reasoning step=<S> phase=<P> decision_class=<C> margin=<M> runner_up_token_id=<T> suppressed_token_count=<N> penalized_token_count=<N>`

**Key Data Structures:**

- Same `lis_cli_selection_diagnostics` struct, extended with ten new fields. Trace step construction (`lis_cli_build_trace_step`) consumes the diagnostics struct instead of independently recomputing reasoning fields.

**Extension Points:**

- Future decision classes may be added to the fixed vocabulary.
- Raw prompt text, generated text, token text in the reasoning line, logits arrays, and full rankings remain out of scope.

## Documentation and Test Support

**Purpose:** Keep implementation progress reviewable and resumable.

**Likely Public Interface Responsibility:**

- Keep architecture and functional design aligned with code changes.
- Maintain per-source Markdown documentation for every `.c` file.
- Run tests through make-based targets.

**Key Data Structures:**

- No runtime data structure required at this design level.
- Backlog item IDs and status flags are the tracking mechanism.

**Extension Points:**

- Documentation helper make target.
- Compatibility test fixtures.
- Backend comparison tests.
