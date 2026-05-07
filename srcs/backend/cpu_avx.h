#ifndef LIS_BACKEND_CPU_AVX_H
#define LIS_BACKEND_CPU_AVX_H

#include "lis/dtype.h"
#include "lis/loader.h"
#include "lis/thread_pool.h"

#include <stddef.h>

/*
 * AVX2+FMA CPU kernels. Backend-private: callers should use the trampolines
 * in lis/cpu_ops.h, which indirect through the lis_cpu_ops dispatch table.
 *
 * Only srcs/backend/cpu_avx.c may include <immintrin.h>; this header stays
 * AVX-clean so it is safe to include from cpu_dispatch.c regardless of the
 * per-TU compile flags.
 *
 * When the build sets -DLIS_DISABLE_AVX (e.g. make SIMD=off), cpu_avx.c is
 * dropped from the link and these symbols are not defined. Callers that
 * reference them must be guarded by the same macro.
 */

void lis_matvec_avx2_fma(const lis_loaded_tensor *weight, const float *input,
                         size_t rows, size_t cols, float *out,
                         lis_thread_pool *pool);

void lis_residual_add_avx2(float *hidden, const float *other,
                           size_t hidden_size);

void lis_rms_norm_avx2(const float *input, const lis_loaded_tensor *weight,
                       size_t hidden_size, float *out, lis_thread_pool *pool);

void lis_swiglu_avx2(const float *gate, const float *up,
                     size_t intermediate_size, float *out,
                     lis_thread_pool *pool);

void lis_softmax_avx2(float *values, size_t n);

void lis_rope_avx2(float *values, size_t head_count, size_t head_dim,
                   size_t position, float rope_theta, lis_thread_pool *pool);

void lis_attn_qk_avx2(const float *q_head, const void *k_row,
                      lis_dtype kv_dtype, size_t head_dim, float *out_dot);

void lis_attn_pv_avx2(float prob, const void *v_row, lis_dtype kv_dtype,
                      size_t head_dim, float *out_head);

#endif
