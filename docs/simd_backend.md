# SIMD Backend

This document is the reference for the AVX/SIMD acceleration layer. It covers
scope, the dispatch model, the kernel list, CPUID feature requirements, the
user-visible switches (`LIS_SIMD` env var, `SIMD=off` Makefile override), and
the verification methodology driven by the performance framework.

---

## Scope

LIS provides an optional AVX2+FMA (and optionally AVX-512F/VL) CPU backend. The
SIMD backend runs alongside the existing reference kernels and is selected once
per process at runtime init based on CPUID. The reference path is never removed.

**In scope:**

- Runtime CPUID feature detection (`sse2`, `avx`, `avx2`, `fma`, `f16c`, `avx512f`, `avx512vl`, `bmi2`).
- AVX2+FMA kernels for the hot operators: `matvec`, `rms_norm`, `softmax`, `swiglu`, `residual_add`, `rope` pair loop, attention inner dots (`Q·Kᵀ`, `P·V`).
- Reference-vs-AVX diff tests, one per kernel, at a documented shape sweep.
- A user-visible `LIS_SIMD=0` environment override and a `make SIMD=off` build override for regression diffs and pre-AVX2 hardware.
- A baseline vs AVX benchmark report driven by the performance harness.

**Out of scope:**

- Removing or bypassing the reference path. It remains the correctness baseline.
- Any change to tensor dtype contract, tensor layout, KV cache layout, or public operator signatures.
- ARM NEON, SVE, or GPU backends.
- Quantization (int8, int4) or weight layout changes.
- Operator fusion (fused attention, fused SwiGLU) — separate future work.
- Custom F16 or BF16 SIMD arithmetic paths; the existing promote-on-load to F32 is preserved.

---

## Dispatch Model

```
public entry points           dispatch table              implementations
(thin trampolines)            (function pointers)         (per-backend TUs)

  lis_matvec()       ────▶   lis_cpu_ops.matvec      ────▶   lis_matvec_reference  (cpu_kernels_reference.c)
  lis_rms_norm()     ────▶   lis_cpu_ops.rms_norm    ────▶   lis_matvec_avx2_fma   (cpu_avx.c)
  lis_softmax()      ────▶   lis_cpu_ops.softmax     ────▶   (optional) _avx512f   (cpu_avx512.c)
  lis_swiglu()       ────▶   lis_cpu_ops.swiglu
  lis_residual_add() ────▶   lis_cpu_ops.residual_add
  lis_rope()         ────▶   lis_cpu_ops.rope
  lis_attn_qk()      ────▶   lis_cpu_ops.attn_qk
  lis_attn_pv()      ────▶   lis_cpu_ops.attn_pv

  populated once at lis_runtime_init from lis_cpu_features_get() and the LIS_SIMD env override
```

**Selection logic (first match wins):**

1. `LIS_SIMD=0` or `SIMD=off` build → reference table.
2. `LIS_SIMD=reference` → reference table.
3. `LIS_SIMD=avx2` → AVX2 kernels where available; fallback to reference per slot.
4. `LIS_SIMD=avx512` + CPUID AVX-512F/VL → AVX-512 kernels where available; AVX2 fallback per slot.
5. No override:
   - AVX-512F/VL present and `cpu_avx512.c` was built → AVX-512 where provided, AVX2 otherwise, reference last.
   - AVX2+FMA present → AVX2 where provided, reference otherwise.
   - Otherwise → reference table.

The backend choice is reported once at runtime init via a single `lis: simd backend=reference|avx2|avx512` line emitted behind the existing diagnostic surface (for example under `--diagnostics` or `--perf`, so quiet runs stay quiet).

---

## Files

**New:**

| File | Responsibility |
|---|---|
| `srcs/includes/lis/cpu_features.h` | Declares `lis_cpu_features` and `lis_cpu_features_get`. |
| `srcs/core/cpu_features.c` | CPUID probe using `__builtin_cpu_supports` with inline-asm fallback. Caches on first call. |
| `srcs/includes/lis/cpu_ops.h` | Declares the public trampolines, the `lis_cpu_ops` struct, `lis_cpu_dispatch_init`, and `lis_cpu_dispatch_backend_name`. AVX-clean. |
| `srcs/backend/cpu_kernels_reference.h` | Backend-private header declaring the `_reference`-suffixed scalar kernels for `cpu_dispatch.c` to consume. Not under `srcs/includes/lis/`. |
| `srcs/backend/cpu_kernels_reference.c` | Scalar kernels extracted from `srcs/runtime/llama.c`, with `_reference` suffixed public symbols and unchanged signatures (including `const lis_loaded_tensor *weight` and `lis_thread_pool *pool` where applicable). |
| `srcs/backend/cpu_dispatch.c` | Owns `lis_cpu_ops`, public trampolines, `LIS_SIMD` env parsing, and the backend-name diagnostic. It populates the `matvec`, `rms_norm`, `rope`, `swiglu`, `residual_add`, `softmax`, `attn_qk`, and `attn_pv` slots. |
| `srcs/backend/cpu_avx.c` | AVX2+FMA kernels. Only TU permitted to include `<immintrin.h>`. Compiled with per-TU `-mavx2 -mfma -mf16c`. |
| `srcs/backend/cpu_avx512.c` (optional) | AVX-512F/VL kernels for `matvec` and `softmax`. Compiled with per-TU `-mavx512f -mavx512vl`. |
| `tests/backend/test_cpu_avx.c` | Reference-vs-AVX diff tests for every AVX kernel. |
| `docs/simd_backend.md` | This document. |

**Modified:**

| File | Change |
|---|---|
| `srcs/runtime/llama.c` | Drop the `static` copies of `lis_matvec`, `lis_rms_norm`, `lis_apply_rope`, `lis_swiglu` work, and the inline softmax/residual-add/attention-dot loops; call the public trampolines from `lis/cpu_ops.h` instead. |
| `srcs/backend/cpu_reference.c` | Holds the existing `add`/`matmul` operator-dispatch stubs; the hot scalar kernels live in `cpu_kernels_reference.c`. |
| `Makefile` | Per-TU flags for `cpu_avx.o` / `cpu_avx512.o`; `SIMD=off` opt-out; register `cpu_dispatch.c` and `cpu_kernels_reference.c` in `BACKEND_SRCS`. |
| `docs/architecture.md`, `docs/function_design.md` | Architecture subsection and function-design subsection covering the SIMD backend. |

---

## Kernel Accuracy Contract

Per-kernel accuracy is asserted by `tests/backend/test_cpu_avx.c`:

- **Relative tolerance:** ≤ 1e-4 per element.
- **Absolute tolerance:** ≤ 1e-6 per element.
- **Shape sweep:** power-of-two sizes `{64, 128, 256, 512, 4096}` for `matvec`, `rms_norm`, `softmax`, `swiglu`, `residual_add`; head dimensions `{64, 128, 256}` × sequence lengths `{1, 16, 128, 512}` for `rope` and attention inner dots.
- **Softmax edge cases:** input ranges `[0, 30]` and `[-30, 30]` to exercise the vectorized `expf` approximation near overflow.

### Rationale for the 1e-4 relative tolerance

The AVX kernels accumulate into **eight independent FMA lanes** and reduce horizontally at the end, while the reference kernel accumulates serially. Neither path is "wrong" — both are valid IEEE-754 roundings of the same mathematical sum — but they do not agree bit-exactly. The per-lane worst-case rounding error for an N-wide FMA reduction grows as roughly `sqrt(N) * eps_f32`, so for `N = 4096` columns the expected per-output drift vs the serial reference is `~sqrt(4096) * 1.19e-7 ≈ 7.6e-6` absolute, and the two reduction orders can diverge by a small multiple of that. An empirical sweep on random inputs in `[-1, 1]` shows the divergence reaching a few × 1e-5 absolute and up to ~1e-4 relative at 4096 columns; at smaller shapes (64, 128, 256) the diff sits comfortably under the reference's own rounding noise.

The previous `≤ 1e-5 rel / ≤ 1e-6 abs` gate was tighter than any lane-parallel SIMD matvec reduction can satisfy at 4096-wide accumulation without serializing the hot loop — which would sacrifice the entire purpose of the kernel. The gate is therefore relaxed to `≤ 1e-4 rel / ≤ 1e-6 abs`, tight enough to catch real kernel bugs (misindexed dtype path, truncated tail loop, swapped operand order, wrong horizontal reduction) while acknowledging the mathematical reality of SIMD reduction.

### Binding gate: end-to-end token parity

The diff test is a fast pre-flight smoke check, not the correctness gate. The **binding correctness gate** is end-to-end token parity: a fixed prompt (`"Write one short sentence about the sea."`) plus `--generate 8` on a supported Llama-family model must produce the **same token sequence** with default dispatch and with `LIS_SIMD=0`. Per-element rounding differences that stay under the diff-test tolerance are essentially never sufficient to change greedy token selection; if a specific kernel's reduction order ever drifts enough to change tokens, that kernel's reduction is reworked to more closely match the reference, even at some speed cost.

---

## CPU Feature Requirements

| Backend | Required CPUID | Target hardware |
|---|---|---|
| `reference` | baseline (no SIMD) | Any x86_64 or cross-platform |
| `avx2` | `avx2` + `fma` (+ `f16c` for future F16 path) | Haswell (2013) and later Intel; Excavator (2015) and later AMD |
| `avx512` (optional) | `avx512f` + `avx512vl` | Skylake-X, Ice Lake, Rocket Lake, Sapphire Rapids, Zen 4 |

On hardware that lacks AVX2, dispatch silently falls back to the reference table. No AVX instruction is issued anywhere in the process.

---

## User-Visible Controls

### Runtime override: `LIS_SIMD`

| Value | Effect |
|---|---|
| *(unset)* | Auto-select the best available backend (AVX-512 if built and supported, else AVX2 if supported, else reference). |
| `0` or `reference` | Force the reference table regardless of CPUID. Primary regression-diff knob. |
| `avx2` | Force the AVX2 table; per-slot fallback to reference for any kernel not yet implemented under AVX2. |
| `avx512` | Force the AVX-512 table where built and supported; per-slot fallback to AVX2, then reference. |

### Build override: `make SIMD=off`

- Drops `srcs/backend/cpu_avx.c` and `srcs/backend/cpu_avx512.c` from `BACKEND_SRCS`.
- Defines `-DLIS_DISABLE_AVX`.
- Produces a binary whose disassembly contains zero AVX instructions. Useful
  for pre-AVX2 hardware and reference-path regression diffs.

---

## Methodology

### Per-kernel

1. Build with default settings.
2. Run `./srcs/libs/test_cpu_avx`. Every reported diff must be within tolerance.

### End-to-end token parity

```bash
MODEL=/path/to/plain-rope-llama

# Baseline (reference kernels)
LIS_SIMD=0 ./srcs/libs/lis \
  --model $MODEL --config $MODEL/config.json --hf-tokenizer $MODEL/tokenizer.json \
  --prompt "Write one short sentence about the sea." \
  --context 128 --batch 1 --generate 8 --threads 1 > /tmp/tokens_ref.txt

# AVX (default dispatch)
./srcs/libs/lis \
  --model $MODEL --config $MODEL/config.json --hf-tokenizer $MODEL/tokenizer.json \
  --prompt "Write one short sentence about the sea." \
  --context 128 --batch 1 --generate 8 --threads 1 > /tmp/tokens_avx.txt

diff /tmp/tokens_ref.txt /tmp/tokens_avx.txt   # must be empty
```

### Speedup measurement

```bash
# Reference baseline
LIS_SIMD=0 python3 tests/perf/run_perf_matrix.py \
  --bin ./srcs/libs/lis --model $MODEL \
  --config $MODEL/config.json --hf-tokenizer $MODEL/tokenizer.json \
  --prompt-sizes short --generates 16 --threads 1 --measured-runs 1 \
  --out-csv /tmp/lis_perf_reference.csv \
  --out-md /tmp/lis_perf_reference.md

# AVX (default dispatch)
python3 tests/perf/run_perf_matrix.py \
  --bin ./srcs/libs/lis --model $MODEL \
  --config $MODEL/config.json --hf-tokenizer $MODEL/tokenizer.json \
  --prompt-sizes short --generates 16 --threads 1 --measured-runs 1 \
  --out-csv /tmp/lis_perf_avx.csv \
  --out-md /tmp/lis_perf_avx.md
```

`/tmp/lis_perf_*` paths are example output locations; substitute any writable
path. The harness emits CSV and Markdown summaries; users running their own
measurements should record the exact command, model, hardware, date, context
length, generated-token count, and limitations alongside any published numbers.

---

## Design Invariants (for future reviewers)

- **Reference-first.** The reference path is the correctness baseline and is never removed. Every AVX kernel must pass a diff test against it.
- **Dispatch, not replacement.** AVX runs alongside reference; both remain linkable. A runtime switch toggles between them.
- **No AVX in headers.** Only `srcs/backend/cpu_avx.c` (and `srcs/backend/cpu_avx512.c`) may include `<immintrin.h>`. Callers and shared headers stay AVX-clean.
- **Per-TU flags only.** `-mavx2 -mfma -mf16c` apply exclusively to the AVX TUs. Global `CFLAGS` stay generic so the binary loads on pre-AVX2 hardware.
- **Unaligned loads.** AVX kernels use `_mm256_loadu_ps` / `_mm512_loadu_ps`. Tensor buffers from the safetensors mapping are 4-byte aligned and unaligned loads on Haswell and later cost nothing extra.
- **No new runtime state.** The dispatch table is the only new global; it is populated once at `lis_runtime_init`.
- **Composes with threading.** AVX runs inside each thread's chunk. The AVX
  kernels are pure functions of their inputs; there is no shared mutable state
  introduced by SIMD.
- **Composes with performance measurement.** Timings remain at the stage level;
  per-operator timing is not introduced by SIMD.

---

## Related Documents

- `docs/architecture.md` — `SIMD Backend` subsection (module layout and design invariants).
- `docs/function_design.md` — `SIMD Backend` subsection (API surface).
- `docs/performance_measurement.md` — performance framework for benchmark reporting.
