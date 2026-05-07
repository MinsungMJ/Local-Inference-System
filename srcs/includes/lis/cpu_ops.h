#ifndef LIS_CPU_OPS_H
#define LIS_CPU_OPS_H

#include "lis/cpu_features.h"
#include "lis/dtype.h"
#include "lis/loader.h"
#include "lis/thread_pool.h"

#include <stddef.h>

/*
 * Public CPU-operator entry points (dispatch scaffold).
 *
 * These are thin trampolines that indirect through the per-process lis_cpu_ops
 * function-pointer table. Signatures are identical to the prior static
 * wrappers that previously lived inside srcs/runtime/llama.c; call sites did
 * not change.
 *
 * AVX intrinsics never appear in this header. Per-kernel SIMD lives in
 * srcs/backend/cpu_avx.c (and cpu_avx512.c) behind the dispatch table.
 */

typedef void (*lis_matvec_fn)(const lis_loaded_tensor *weight,
                              const float *input, size_t rows, size_t cols,
                              float *out, lis_thread_pool *pool);

typedef void (*lis_rms_norm_fn)(const float *input,
                                const lis_loaded_tensor *weight,
                                size_t hidden_size, float *out,
                                lis_thread_pool *pool);

typedef void (*lis_rope_fn)(float *values, size_t head_count, size_t head_dim,
                            size_t position, float rope_theta,
                            lis_thread_pool *pool);

typedef void (*lis_swiglu_fn)(const float *gate, const float *up,
                              size_t intermediate_size, float *out,
                              lis_thread_pool *pool);

typedef void (*lis_residual_add_fn)(float *hidden, const float *other,
                                    size_t hidden_size);

typedef void (*lis_softmax_fn)(float *values, size_t n);

/*
 * Per-position attention inner-dot kernels. The attention worker loops over
 * past positions; each slot consumes a single head-row from the KV cache, in
 * the cache's native dtype (F32 / F16 / BF16), and produces either a dot
 * product (Q·K) or an accumulated output lane (P·V).
 */
typedef void (*lis_attn_qk_fn)(const float *q_head, const void *k_row,
                               lis_dtype kv_dtype, size_t head_dim,
                               float *out_dot);

typedef void (*lis_attn_pv_fn)(float prob, const void *v_row,
                               lis_dtype kv_dtype, size_t head_dim,
                               float *out_head);

typedef struct {
    lis_matvec_fn matvec;
    lis_rms_norm_fn rms_norm;
    lis_rope_fn rope;
    lis_swiglu_fn swiglu;
    lis_residual_add_fn residual_add;
    lis_softmax_fn softmax;
    lis_attn_qk_fn attn_qk;
    lis_attn_pv_fn attn_pv;
} lis_cpu_ops;

/*
 * Populate the dispatch table exactly once at runtime init, using the CPU
 * feature struct to select reference vs AVX2 vs AVX-512 per slot. Honours the
 * LIS_SIMD environment override (0 or "reference" forces the reference table).
 * Safe to call repeatedly; only the first call has effect.
 */
void lis_cpu_dispatch_init(const lis_cpu_features *features);

/*
 * Return a stable string identifying the currently-selected backend:
 * "reference", "avx2", or "avx512". Used by the optional
 * `lis: simd backend=...` diagnostic line. Returns "reference" before
 * lis_cpu_dispatch_init has been called.
 */
const char *lis_cpu_dispatch_backend_name(void);

/* Public trampolines. Signatures match the prior static wrappers. */
void lis_matvec(const lis_loaded_tensor *weight, const float *input,
                size_t rows, size_t cols, float *out, lis_thread_pool *pool);
void lis_rms_norm(const float *input, const lis_loaded_tensor *weight,
                  size_t hidden_size, float *out, lis_thread_pool *pool);
void lis_apply_rope(float *values, size_t head_count, size_t head_dim,
                    size_t position, float rope_theta,
                    lis_thread_pool *pool);
void lis_swiglu(const float *gate, const float *up, size_t intermediate_size,
                float *out, lis_thread_pool *pool);
void lis_residual_add(float *hidden, const float *other, size_t hidden_size);
/*
 * Numerically stable in-place softmax over `n` contiguous float values. The
 * trampoline is called from inside per-head attention workers (no pool arg).
 */
void lis_softmax(float *values, size_t n);

/*
 * Per-position Q·K dot product for one attention head. `k_row` points at the
 * head-row base element in the KV cache (dim = 0), stored as `kv_dtype`.
 * Writes the scaled-before-softmax raw dot to `*out_dot`; callers apply the
 * 1/sqrt(head_dim) scale and the causal-softmax step themselves.
 */
void lis_attn_qk(const float *q_head, const void *k_row, lis_dtype kv_dtype,
                 size_t head_dim, float *out_dot);

/*
 * Per-position P·V fused multiply-add for one attention head. `v_row` points
 * at the head-row base element in the KV cache (dim = 0), stored as
 * `kv_dtype`. `out_head[0..head_dim)` must already hold the partial
 * accumulator for this head; the kernel adds `prob * V` in place.
 */
void lis_attn_pv(float prob, const void *v_row, lis_dtype kv_dtype,
                 size_t head_dim, float *out_head);

#endif
