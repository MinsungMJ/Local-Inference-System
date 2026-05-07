#ifndef LIS_BACKEND_CPU_KERNELS_REFERENCE_H
#define LIS_BACKEND_CPU_KERNELS_REFERENCE_H

#include "lis/dtype.h"
#include "lis/loader.h"
#include "lis/thread_pool.h"

#include <stddef.h>

/*
 * Backend-private reference kernels. Consumed by srcs/backend/cpu_dispatch.c
 * to populate the `lis_cpu_ops` table. Not part of the public API under
 * srcs/includes/lis/; callers should use the trampolines in lis/cpu_ops.h.
 *
 * Signatures match the prior static wrappers from srcs/runtime/llama.c.
 * The reference path is never removed: every AVX kernel diff-tests against
 * these, and LIS_SIMD=0 resolves every dispatch slot back to this TU.
 */

void lis_matvec_reference(const lis_loaded_tensor *weight, const float *input,
                          size_t rows, size_t cols, float *out,
                          lis_thread_pool *pool);

void lis_rms_norm_reference(const float *input,
                            const lis_loaded_tensor *weight,
                            size_t hidden_size, float *out,
                            lis_thread_pool *pool);

void lis_apply_rope_reference(float *values, size_t head_count,
                              size_t head_dim, size_t position,
                              float rope_theta, lis_thread_pool *pool);

void lis_swiglu_reference(const float *gate, const float *up,
                          size_t intermediate_size, float *out,
                          lis_thread_pool *pool);

void lis_residual_add_reference(float *hidden, const float *other,
                                size_t hidden_size);

void lis_softmax_reference(float *values, size_t n);

void lis_attn_qk_reference(const float *q_head, const void *k_row,
                           lis_dtype kv_dtype, size_t head_dim,
                           float *out_dot);

void lis_attn_pv_reference(float prob, const void *v_row, lis_dtype kv_dtype,
                           size_t head_dim, float *out_head);

#endif
