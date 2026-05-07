#include "lis/cpu_ops.h"

#include "cpu_kernels_reference.h"

#ifndef LIS_DISABLE_AVX
#include "cpu_avx.h"
#endif

#include <stdlib.h>
#include <string.h>

static lis_cpu_ops g_cpu_ops = {
    .matvec = lis_matvec_reference,
    .rms_norm = lis_rms_norm_reference,
    .rope = lis_apply_rope_reference,
    .swiglu = lis_swiglu_reference,
    .residual_add = lis_residual_add_reference,
    .softmax = lis_softmax_reference,
    .attn_qk = lis_attn_qk_reference,
    .attn_pv = lis_attn_pv_reference,
};

static const char *g_backend_name = "reference";

typedef enum {
    LIS_SIMD_SELECT_AUTO = 0,
    LIS_SIMD_SELECT_REFERENCE,
    LIS_SIMD_SELECT_AVX2,
    LIS_SIMD_SELECT_AVX512
} lis_simd_selection;

static lis_simd_selection lis_parse_simd_env(void)
{
    const char *value = getenv("LIS_SIMD");

    if (value == NULL || value[0] == '\0') {
        return LIS_SIMD_SELECT_AUTO;
    }
    if (strcmp(value, "0") == 0 || strcmp(value, "reference") == 0) {
        return LIS_SIMD_SELECT_REFERENCE;
    }
    if (strcmp(value, "avx2") == 0) {
        return LIS_SIMD_SELECT_AVX2;
    }
    if (strcmp(value, "avx512") == 0) {
        return LIS_SIMD_SELECT_AVX512;
    }
    return LIS_SIMD_SELECT_AUTO;
}

void lis_cpu_dispatch_init(const lis_cpu_features *features)
{
    const lis_simd_selection selection = lis_parse_simd_env();
    int can_avx2 = 0;
    int can_avx512 = 0;

    if (features != NULL) {
        can_avx2 = features->avx2 && features->fma;
        can_avx512 = features->avx512f && features->avx512vl;
    }

    /*
     * Start from the reference table, then upgrade per-slot when the requested
     * backend is available. Slots without an AVX implementation stay on the
     * reference kernel, so partial SIMD coverage never leaves an empty slot.
     */
    g_cpu_ops.matvec = lis_matvec_reference;
    g_cpu_ops.rms_norm = lis_rms_norm_reference;
    g_cpu_ops.rope = lis_apply_rope_reference;
    g_cpu_ops.swiglu = lis_swiglu_reference;
    g_cpu_ops.residual_add = lis_residual_add_reference;
    g_cpu_ops.softmax = lis_softmax_reference;
    g_cpu_ops.attn_qk = lis_attn_qk_reference;
    g_cpu_ops.attn_pv = lis_attn_pv_reference;
    g_backend_name = "reference";

    (void)can_avx512;

#ifndef LIS_DISABLE_AVX
    if (selection == LIS_SIMD_SELECT_AVX2 ||
        selection == LIS_SIMD_SELECT_AVX512 ||
        selection == LIS_SIMD_SELECT_AUTO) {
        if (can_avx2) {
            g_cpu_ops.matvec = lis_matvec_avx2_fma;
            g_cpu_ops.rms_norm = lis_rms_norm_avx2;
            g_cpu_ops.rope = lis_rope_avx2;
            g_cpu_ops.swiglu = lis_swiglu_avx2;
            g_cpu_ops.residual_add = lis_residual_add_avx2;
            g_cpu_ops.softmax = lis_softmax_avx2;
            g_cpu_ops.attn_qk = lis_attn_qk_avx2;
            g_cpu_ops.attn_pv = lis_attn_pv_avx2;
            g_backend_name = "avx2";
        }
    }
#else
    (void)can_avx2;
#endif

    if (selection == LIS_SIMD_SELECT_REFERENCE) {
        g_cpu_ops.matvec = lis_matvec_reference;
        g_cpu_ops.rms_norm = lis_rms_norm_reference;
        g_cpu_ops.rope = lis_apply_rope_reference;
        g_cpu_ops.swiglu = lis_swiglu_reference;
        g_cpu_ops.residual_add = lis_residual_add_reference;
        g_cpu_ops.softmax = lis_softmax_reference;
        g_cpu_ops.attn_qk = lis_attn_qk_reference;
        g_cpu_ops.attn_pv = lis_attn_pv_reference;
        g_backend_name = "reference";
    }
}

const char *lis_cpu_dispatch_backend_name(void)
{
    return g_backend_name;
}

void lis_matvec(const lis_loaded_tensor *weight, const float *input,
                size_t rows, size_t cols, float *out, lis_thread_pool *pool)
{
    g_cpu_ops.matvec(weight, input, rows, cols, out, pool);
}

void lis_rms_norm(const float *input, const lis_loaded_tensor *weight,
                  size_t hidden_size, float *out, lis_thread_pool *pool)
{
    g_cpu_ops.rms_norm(input, weight, hidden_size, out, pool);
}

void lis_apply_rope(float *values, size_t head_count, size_t head_dim,
                    size_t position, float rope_theta, lis_thread_pool *pool)
{
    g_cpu_ops.rope(values, head_count, head_dim, position, rope_theta, pool);
}

void lis_swiglu(const float *gate, const float *up, size_t intermediate_size,
                float *out, lis_thread_pool *pool)
{
    g_cpu_ops.swiglu(gate, up, intermediate_size, out, pool);
}

void lis_residual_add(float *hidden, const float *other, size_t hidden_size)
{
    g_cpu_ops.residual_add(hidden, other, hidden_size);
}

void lis_softmax(float *values, size_t n)
{
    g_cpu_ops.softmax(values, n);
}

void lis_attn_qk(const float *q_head, const void *k_row, lis_dtype kv_dtype,
                 size_t head_dim, float *out_dot)
{
    g_cpu_ops.attn_qk(q_head, k_row, kv_dtype, head_dim, out_dot);
}

void lis_attn_pv(float prob, const void *v_row, lis_dtype kv_dtype,
                 size_t head_dim, float *out_head)
{
    g_cpu_ops.attn_pv(prob, v_row, kv_dtype, head_dim, out_head);
}
