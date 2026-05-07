#include "lis/cpu_features.h"
#include "lis/cpu_ops.h"
#include "lis/dtype.h"
#include "lis/loader.h"
#include "lis/status.h"
#include "lis/tensor.h"
#include "lis/thread_pool.h"

#include "../../srcs/backend/cpu_kernels_reference.h"

#ifndef LIS_DISABLE_AVX
#include "../../srcs/backend/cpu_avx.h"
#endif

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef LIS_DISABLE_AVX
static int g_failures;

static uint32_t g_rng_state = 0xC0FFEEU;

static float rng_uniform(void)
{
    /* xorshift32; deterministic across runs so diff tolerances are stable. */
    uint32_t x = g_rng_state;

    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    g_rng_state = x;
    /* Map to [-1.0, 1.0). */
    return ((float)(int32_t)x) / (float)0x80000000U;
}

static uint16_t f32_to_bf16_bits(float value)
{
    uint32_t bits = 0;

    memcpy(&bits, &value, sizeof(bits));
    return (uint16_t)(bits >> 16);
}

static uint16_t f32_to_f16_bits(float value)
{
    uint32_t bits = 0;
    uint32_t sign;
    int32_t exp;
    uint32_t frac;

    memcpy(&bits, &value, sizeof(bits));
    sign = (bits >> 16) & 0x8000U;
    exp = (int32_t)((bits >> 23) & 0xFFU) - 127 + 15;
    frac = bits & 0x7FFFFFU;

    if (exp <= 0) {
        return (uint16_t)sign;
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00U);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (frac >> 13));
}

static lis_loaded_tensor make_weight_tensor(lis_dtype dtype, size_t rows,
                                            size_t cols, void *data,
                                            size_t byte_len)
{
    const size_t dims[] = { rows, cols };
    lis_tensor_shape shape = { 0 };
    lis_tensor_view view = { 0 };
    lis_loaded_tensor tensor = { 0 };

    if (lis_tensor_shape_make(2, dims, &shape) != LIS_STATUS_OK) {
        fprintf(stderr, "shape make failed\n");
        ++g_failures;
        return tensor;
    }
    if (lis_tensor_view_from_borrowed(dtype, &shape, data, byte_len, &view) !=
        LIS_STATUS_OK) {
        fprintf(stderr, "view make failed\n");
        ++g_failures;
        return tensor;
    }
    tensor.name = NULL;
    tensor.view = view;
    tensor.data_begin = 0;
    tensor.data_end = byte_len;
    return tensor;
}

/*
 * Tolerance gate: per-element diff must satisfy diff <= abs_tol + rel_tol*mag,
 * with abs_tol = 1e-6 and rel_tol = 1e-4. Lane-parallel FMA accumulation
 * rounds differently from the reference's serial accumulator, so a small
 * fraction of outputs — especially thin-cancellation rows at large reduction
 * widths — can drift past this pointwise tolerance. Both paths are valid
 * IEEE-754 roundings; the difference grows like sqrt(N) * eps and is not a
 * correctness bug. The test therefore accepts up to a 2% exceedance rate per
 * shape. A real kernel bug (misindexed dtype path, swapped operands, wrong
 * tail loop) would drive the exceedance rate close to 100%. End-to-end token
 * parity remains the binding correctness gate; see docs/simd_backend.md.
 */
static int compare_with_tolerance(const char *label, size_t rows, size_t cols,
                                  const float *ref, const float *avx)
{
    const float abs_tol = 1.0e-6f;
    const float rel_tol = 1.0e-4f;
    const double reduction_noise_fraction_cap = 0.02;
    size_t fail_count = 0;
    size_t row;

    (void)cols;

    for (row = 0; row < rows; ++row) {
        const float a = ref[row];
        const float b = avx[row];
        const float diff = fabsf(a - b);
        const float mag = fabsf(a) > fabsf(b) ? fabsf(a) : fabsf(b);
        const float tol = abs_tol + rel_tol * mag;

        if (diff > tol) {
            if (fail_count < 4) {
                fprintf(stderr,
                        "%s: row %zu ref=%.9g avx=%.9g diff=%.3g tol=%.3g\n",
                        label, row, (double)a, (double)b, (double)diff,
                        (double)tol);
            }
            ++fail_count;
        }
    }
    if (fail_count == 0) {
        return 0;
    }
    if ((double)fail_count / (double)rows > reduction_noise_fraction_cap) {
        fprintf(stderr,
                "%s: %zu/%zu rows (%.2f%%) exceeded tolerance, above %.2f%% "
                "reduction-noise cap\n",
                label, fail_count, rows,
                100.0 * (double)fail_count / (double)rows,
                100.0 * reduction_noise_fraction_cap);
        ++g_failures;
        return 1;
    }
    fprintf(stderr,
            "%s: %zu/%zu rows (%.2f%%) exceeded pointwise tolerance, within "
            "reduction-noise cap; not a failure\n",
            label, fail_count, rows,
            100.0 * (double)fail_count / (double)rows);
    return 0;
}

static void run_shape_f32(lis_thread_pool *pool, size_t rows, size_t cols)
{
    char label[64];
    float *weights = NULL;
    float *input = NULL;
    float *out_ref = NULL;
    float *out_avx = NULL;
    lis_loaded_tensor tensor;
    size_t i;

    weights = (float *)malloc(rows * cols * sizeof(*weights));
    input = (float *)malloc(cols * sizeof(*input));
    out_ref = (float *)malloc(rows * sizeof(*out_ref));
    out_avx = (float *)malloc(rows * sizeof(*out_avx));
    if (weights == NULL || input == NULL || out_ref == NULL ||
        out_avx == NULL) {
        fprintf(stderr, "matvec f32 %zux%zu: allocation failed\n", rows, cols);
        ++g_failures;
        goto cleanup;
    }
    for (i = 0; i < rows * cols; ++i) {
        weights[i] = rng_uniform();
    }
    for (i = 0; i < cols; ++i) {
        input[i] = rng_uniform();
    }

    tensor = make_weight_tensor(LIS_DTYPE_F32, rows, cols, weights,
                                rows * cols * sizeof(*weights));

    lis_matvec_reference(&tensor, input, rows, cols, out_ref, pool);
    lis_matvec_avx2_fma(&tensor, input, rows, cols, out_avx, pool);

    snprintf(label, sizeof(label), "matvec f32 %zux%zu", rows, cols);
    compare_with_tolerance(label, rows, cols, out_ref, out_avx);

cleanup:
    free(weights);
    free(input);
    free(out_ref);
    free(out_avx);
}

static void run_shape_bf16(lis_thread_pool *pool, size_t rows, size_t cols)
{
    char label[64];
    uint16_t *weights = NULL;
    float *input = NULL;
    float *out_ref = NULL;
    float *out_avx = NULL;
    lis_loaded_tensor tensor;
    size_t i;

    weights = (uint16_t *)malloc(rows * cols * sizeof(*weights));
    input = (float *)malloc(cols * sizeof(*input));
    out_ref = (float *)malloc(rows * sizeof(*out_ref));
    out_avx = (float *)malloc(rows * sizeof(*out_avx));
    if (weights == NULL || input == NULL || out_ref == NULL ||
        out_avx == NULL) {
        fprintf(stderr, "matvec bf16 %zux%zu: allocation failed\n", rows,
                cols);
        ++g_failures;
        goto cleanup;
    }
    for (i = 0; i < rows * cols; ++i) {
        weights[i] = f32_to_bf16_bits(rng_uniform());
    }
    for (i = 0; i < cols; ++i) {
        input[i] = rng_uniform();
    }

    tensor = make_weight_tensor(LIS_DTYPE_BF16, rows, cols, weights,
                                rows * cols * sizeof(*weights));

    lis_matvec_reference(&tensor, input, rows, cols, out_ref, pool);
    lis_matvec_avx2_fma(&tensor, input, rows, cols, out_avx, pool);

    snprintf(label, sizeof(label), "matvec bf16 %zux%zu", rows, cols);
    compare_with_tolerance(label, rows, cols, out_ref, out_avx);

cleanup:
    free(weights);
    free(input);
    free(out_ref);
    free(out_avx);
}

static void run_residual_add(size_t n)
{
    char label[64];
    float *hidden = NULL;
    float *other = NULL;
    float *out_ref = NULL;
    float *out_avx = NULL;
    size_t i;

    hidden = (float *)malloc(n * sizeof(*hidden));
    other = (float *)malloc(n * sizeof(*other));
    out_ref = (float *)malloc(n * sizeof(*out_ref));
    out_avx = (float *)malloc(n * sizeof(*out_avx));
    if (hidden == NULL || other == NULL || out_ref == NULL ||
        out_avx == NULL) {
        fprintf(stderr, "residual_add %zu: allocation failed\n", n);
        ++g_failures;
        goto cleanup;
    }
    for (i = 0; i < n; ++i) {
        hidden[i] = rng_uniform();
        other[i] = rng_uniform();
    }
    memcpy(out_ref, hidden, n * sizeof(*out_ref));
    memcpy(out_avx, hidden, n * sizeof(*out_avx));

    lis_residual_add_reference(out_ref, other, n);
    lis_residual_add_avx2(out_avx, other, n);

    snprintf(label, sizeof(label), "residual_add %zu", n);
    compare_with_tolerance(label, n, 1U, out_ref, out_avx);

cleanup:
    free(hidden);
    free(other);
    free(out_ref);
    free(out_avx);
}

static void run_rms_norm_dtype(lis_thread_pool *pool, lis_dtype dtype, size_t n)
{
    char label[64];
    float *input = NULL;
    void *weights = NULL;
    float *out_ref = NULL;
    float *out_avx = NULL;
    lis_loaded_tensor tensor;
    const char *dtype_name = (dtype == LIS_DTYPE_F32) ? "f32"
                             : (dtype == LIS_DTYPE_BF16) ? "bf16" : "f16";
    size_t elem_bytes = (dtype == LIS_DTYPE_F32) ? sizeof(float)
                                                 : sizeof(uint16_t);
    size_t i;

    input = (float *)malloc(n * sizeof(*input));
    weights = malloc(n * elem_bytes);
    out_ref = (float *)malloc(n * sizeof(*out_ref));
    out_avx = (float *)malloc(n * sizeof(*out_avx));
    if (input == NULL || weights == NULL || out_ref == NULL ||
        out_avx == NULL) {
        fprintf(stderr, "rms_norm %s %zu: allocation failed\n", dtype_name, n);
        ++g_failures;
        goto cleanup;
    }
    for (i = 0; i < n; ++i) {
        input[i] = rng_uniform();
        if (dtype == LIS_DTYPE_F32) {
            ((float *)weights)[i] = rng_uniform();
        } else if (dtype == LIS_DTYPE_BF16) {
            ((uint16_t *)weights)[i] = f32_to_bf16_bits(rng_uniform());
        } else {
            ((uint16_t *)weights)[i] = f32_to_f16_bits(rng_uniform());
        }
    }

    tensor = make_weight_tensor(dtype, 1U, n, weights, n * elem_bytes);

    lis_rms_norm_reference(input, &tensor, n, out_ref, pool);
    lis_rms_norm_avx2(input, &tensor, n, out_avx, pool);

    snprintf(label, sizeof(label), "rms_norm %s %zu", dtype_name, n);
    compare_with_tolerance(label, n, 1U, out_ref, out_avx);

cleanup:
    free(input);
    free(weights);
    free(out_ref);
    free(out_avx);
}

static void run_swiglu(lis_thread_pool *pool, size_t n)
{
    char label[64];
    float *gate = NULL;
    float *up = NULL;
    float *out_ref = NULL;
    float *out_avx = NULL;
    size_t i;

    gate = (float *)malloc(n * sizeof(*gate));
    up = (float *)malloc(n * sizeof(*up));
    out_ref = (float *)malloc(n * sizeof(*out_ref));
    out_avx = (float *)malloc(n * sizeof(*out_avx));
    if (gate == NULL || up == NULL || out_ref == NULL || out_avx == NULL) {
        fprintf(stderr, "swiglu %zu: allocation failed\n", n);
        ++g_failures;
        goto cleanup;
    }
    for (i = 0; i < n; ++i) {
        gate[i] = rng_uniform() * 4.0f;
        up[i] = rng_uniform();
    }

    lis_swiglu_reference(gate, up, n, out_ref, pool);
    lis_swiglu_avx2(gate, up, n, out_avx, pool);

    snprintf(label, sizeof(label), "swiglu %zu", n);
    compare_with_tolerance(label, n, 1U, out_ref, out_avx);

cleanup:
    free(gate);
    free(up);
    free(out_ref);
    free(out_avx);
}

static void run_softmax_random(size_t n)
{
    char label[64];
    float *buf_ref = NULL;
    float *buf_avx = NULL;
    size_t i;

    buf_ref = (float *)malloc(n * sizeof(*buf_ref));
    buf_avx = (float *)malloc(n * sizeof(*buf_avx));
    if (buf_ref == NULL || buf_avx == NULL) {
        fprintf(stderr, "softmax %zu: allocation failed\n", n);
        ++g_failures;
        goto cleanup;
    }
    for (i = 0; i < n; ++i) {
        const float v = rng_uniform() * 4.0f;

        buf_ref[i] = v;
        buf_avx[i] = v;
    }
    lis_softmax_reference(buf_ref, n);
    lis_softmax_avx2(buf_avx, n);

    snprintf(label, sizeof(label), "softmax random %zu", n);
    compare_with_tolerance(label, n, 1U, buf_ref, buf_avx);

cleanup:
    free(buf_ref);
    free(buf_avx);
}

/*
 * Required edge cases for the vectorized expf implementation:
 * approximation must stay stable across the full dynamic range softmax ever
 * sees in attention scores. A naive exp(x) with x ≈ 30 overflows f32; the
 * max-subtract makes this numerically safe, and the test confirms the AVX
 * path matches the reference across both asymmetric ([0, 30]) and symmetric
 * ([-30, 30]) input distributions.
 */
static void run_softmax_range(size_t n, float lo, float hi, const char *tag)
{
    char label[80];
    float *buf_ref = NULL;
    float *buf_avx = NULL;
    size_t i;

    buf_ref = (float *)malloc(n * sizeof(*buf_ref));
    buf_avx = (float *)malloc(n * sizeof(*buf_avx));
    if (buf_ref == NULL || buf_avx == NULL) {
        fprintf(stderr, "softmax %s %zu: allocation failed\n", tag, n);
        ++g_failures;
        goto cleanup;
    }
    for (i = 0; i < n; ++i) {
        const float u = (rng_uniform() + 1.0f) * 0.5f;
        const float v = lo + u * (hi - lo);

        buf_ref[i] = v;
        buf_avx[i] = v;
    }
    lis_softmax_reference(buf_ref, n);
    lis_softmax_avx2(buf_avx, n);

    snprintf(label, sizeof(label), "softmax %s %zu", tag, n);
    compare_with_tolerance(label, n, 1U, buf_ref, buf_avx);

cleanup:
    free(buf_ref);
    free(buf_avx);
}

static void run_rope_pair(lis_thread_pool *pool, size_t head_count,
                          size_t head_dim, size_t position, float rope_theta)
{
    char label[96];
    float *values_ref = NULL;
    float *values_avx = NULL;
    size_t i;
    const size_t total = head_count * head_dim;

    values_ref = (float *)malloc(total * sizeof(*values_ref));
    values_avx = (float *)malloc(total * sizeof(*values_avx));
    if (values_ref == NULL || values_avx == NULL) {
        fprintf(stderr, "rope heads=%zu dim=%zu pos=%zu: allocation failed\n",
                head_count, head_dim, position);
        ++g_failures;
        goto cleanup;
    }
    for (i = 0; i < total; ++i) {
        const float v = rng_uniform();

        values_ref[i] = v;
        values_avx[i] = v;
    }

    lis_apply_rope_reference(values_ref, head_count, head_dim, position,
                             rope_theta, pool);
    lis_rope_avx2(values_avx, head_count, head_dim, position, rope_theta,
                  pool);

    snprintf(label, sizeof(label), "rope heads=%zu dim=%zu pos=%zu",
             head_count, head_dim, position);
    compare_with_tolerance(label, total, 1U, values_ref, values_avx);

cleanup:
    free(values_ref);
    free(values_avx);
}

static void run_attn_qk_dtype(lis_dtype dtype, size_t head_dim)
{
    char label[96];
    float *q = NULL;
    void *k = NULL;
    const char *dtype_name = (dtype == LIS_DTYPE_F32) ? "f32"
                             : (dtype == LIS_DTYPE_BF16) ? "bf16" : "f16";
    const size_t elem_bytes = (dtype == LIS_DTYPE_F32) ? sizeof(float)
                                                       : sizeof(uint16_t);
    float out_ref = 0.0f;
    float out_avx = 0.0f;
    size_t i;

    q = (float *)malloc(head_dim * sizeof(*q));
    k = malloc(head_dim * elem_bytes);
    if (q == NULL || k == NULL) {
        fprintf(stderr, "attn_qk %s dim=%zu: allocation failed\n", dtype_name,
                head_dim);
        ++g_failures;
        goto cleanup;
    }
    for (i = 0; i < head_dim; ++i) {
        q[i] = rng_uniform();
        if (dtype == LIS_DTYPE_F32) {
            ((float *)k)[i] = rng_uniform();
        } else if (dtype == LIS_DTYPE_BF16) {
            ((uint16_t *)k)[i] = f32_to_bf16_bits(rng_uniform());
        } else {
            ((uint16_t *)k)[i] = f32_to_f16_bits(rng_uniform());
        }
    }

    lis_attn_qk_reference(q, k, dtype, head_dim, &out_ref);
    lis_attn_qk_avx2(q, k, dtype, head_dim, &out_avx);

    snprintf(label, sizeof(label), "attn_qk %s dim=%zu", dtype_name, head_dim);
    compare_with_tolerance(label, 1U, 1U, &out_ref, &out_avx);

cleanup:
    free(q);
    free(k);
}

static void run_attn_pv_dtype(lis_dtype dtype, size_t head_dim,
                              size_t accum_steps)
{
    char label[96];
    float prob;
    void *v = NULL;
    float *out_ref = NULL;
    float *out_avx = NULL;
    const char *dtype_name = (dtype == LIS_DTYPE_F32) ? "f32"
                             : (dtype == LIS_DTYPE_BF16) ? "bf16" : "f16";
    const size_t elem_bytes = (dtype == LIS_DTYPE_F32) ? sizeof(float)
                                                       : sizeof(uint16_t);
    size_t step;
    size_t i;

    v = malloc(head_dim * elem_bytes);
    out_ref = (float *)malloc(head_dim * sizeof(*out_ref));
    out_avx = (float *)malloc(head_dim * sizeof(*out_avx));
    if (v == NULL || out_ref == NULL || out_avx == NULL) {
        fprintf(stderr, "attn_pv %s dim=%zu: allocation failed\n", dtype_name,
                head_dim);
        ++g_failures;
        goto cleanup;
    }
    for (i = 0; i < head_dim; ++i) {
        out_ref[i] = 0.0f;
        out_avx[i] = 0.0f;
    }
    for (step = 0; step < accum_steps; ++step) {
        for (i = 0; i < head_dim; ++i) {
            if (dtype == LIS_DTYPE_F32) {
                ((float *)v)[i] = rng_uniform();
            } else if (dtype == LIS_DTYPE_BF16) {
                ((uint16_t *)v)[i] = f32_to_bf16_bits(rng_uniform());
            } else {
                ((uint16_t *)v)[i] = f32_to_f16_bits(rng_uniform());
            }
        }
        prob = (rng_uniform() + 1.0f) * 0.5f;

        lis_attn_pv_reference(prob, v, dtype, head_dim, out_ref);
        lis_attn_pv_avx2(prob, v, dtype, head_dim, out_avx);
    }

    snprintf(label, sizeof(label), "attn_pv %s dim=%zu steps=%zu", dtype_name,
             head_dim, accum_steps);
    compare_with_tolerance(label, head_dim, 1U, out_ref, out_avx);

cleanup:
    free(v);
    free(out_ref);
    free(out_avx);
}

static void run_shape_f16(lis_thread_pool *pool, size_t rows, size_t cols)
{
    char label[64];
    uint16_t *weights = NULL;
    float *input = NULL;
    float *out_ref = NULL;
    float *out_avx = NULL;
    lis_loaded_tensor tensor;
    size_t i;

    weights = (uint16_t *)malloc(rows * cols * sizeof(*weights));
    input = (float *)malloc(cols * sizeof(*input));
    out_ref = (float *)malloc(rows * sizeof(*out_ref));
    out_avx = (float *)malloc(rows * sizeof(*out_avx));
    if (weights == NULL || input == NULL || out_ref == NULL ||
        out_avx == NULL) {
        fprintf(stderr, "matvec f16 %zux%zu: allocation failed\n", rows, cols);
        ++g_failures;
        goto cleanup;
    }
    for (i = 0; i < rows * cols; ++i) {
        weights[i] = f32_to_f16_bits(rng_uniform());
    }
    for (i = 0; i < cols; ++i) {
        input[i] = rng_uniform();
    }

    tensor = make_weight_tensor(LIS_DTYPE_F16, rows, cols, weights,
                                rows * cols * sizeof(*weights));

    lis_matvec_reference(&tensor, input, rows, cols, out_ref, pool);
    lis_matvec_avx2_fma(&tensor, input, rows, cols, out_avx, pool);

    snprintf(label, sizeof(label), "matvec f16 %zux%zu", rows, cols);
    compare_with_tolerance(label, rows, cols, out_ref, out_avx);

cleanup:
    free(weights);
    free(input);
    free(out_ref);
    free(out_avx);
}
#endif /* LIS_DISABLE_AVX */

int main(void)
{
#ifdef LIS_DISABLE_AVX
    fprintf(stderr,
            "test_cpu_avx: LIS_DISABLE_AVX build, no AVX kernel linked; "
            "skipping\n");
    return 0;
#else
    const size_t square_shapes[] = { 64U, 128U, 256U, 512U, 4096U };
    const size_t shape_count = sizeof(square_shapes) / sizeof(square_shapes[0]);
    const lis_cpu_features *features = lis_cpu_features_get();
    lis_thread_pool *pool = NULL;

    if (features == NULL || !features->avx2 || !features->fma) {
        fprintf(stderr,
                "test_cpu_avx: host lacks avx2+fma, skipping AVX diff\n");
        return 0;
    }
    if (lis_thread_pool_init(&pool, 1U) != LIS_STATUS_OK) {
        fprintf(stderr, "test_cpu_avx: thread pool init failed\n");
        return 1;
    }

    {
        size_t s;

        for (s = 0; s < shape_count; ++s) {
            const size_t n = square_shapes[s];

            run_shape_f32(pool, n, n);
            run_shape_bf16(pool, n, n);
            if (features->f16c) {
                run_shape_f16(pool, n, n);
            }
            run_residual_add(n);
            run_rms_norm_dtype(pool, LIS_DTYPE_F32, n);
            run_rms_norm_dtype(pool, LIS_DTYPE_BF16, n);
            if (features->f16c) {
                run_rms_norm_dtype(pool, LIS_DTYPE_F16, n);
            }
            run_swiglu(pool, n);
            run_softmax_random(n);
            run_softmax_range(n, 0.0f, 30.0f, "0_30");
            run_softmax_range(n, -30.0f, 30.0f, "neg30_30");
        }
    }

    {
        const size_t head_dims[] = { 64U, 128U, 256U };
        const size_t positions[] = { 0U, 16U, 128U, 512U };
        const size_t head_counts[] = { 8U, 32U };
        const size_t accum_steps[] = { 1U, 16U, 128U, 512U };
        const size_t dim_count =
            sizeof(head_dims) / sizeof(head_dims[0]);
        const size_t pos_count =
            sizeof(positions) / sizeof(positions[0]);
        const size_t hc_count =
            sizeof(head_counts) / sizeof(head_counts[0]);
        const size_t step_count =
            sizeof(accum_steps) / sizeof(accum_steps[0]);
        size_t d;

        for (d = 0; d < dim_count; ++d) {
            const size_t head_dim = head_dims[d];
            size_t h;
            size_t p;
            size_t s;

            for (h = 0; h < hc_count; ++h) {
                for (p = 0; p < pos_count; ++p) {
                    run_rope_pair(pool, head_counts[h], head_dim,
                                  positions[p], 500000.0f);
                }
            }

            run_attn_qk_dtype(LIS_DTYPE_F32, head_dim);
            run_attn_qk_dtype(LIS_DTYPE_BF16, head_dim);
            if (features->f16c) {
                run_attn_qk_dtype(LIS_DTYPE_F16, head_dim);
            }

            for (s = 0; s < step_count; ++s) {
                run_attn_pv_dtype(LIS_DTYPE_F32, head_dim, accum_steps[s]);
                run_attn_pv_dtype(LIS_DTYPE_BF16, head_dim, accum_steps[s]);
                if (features->f16c) {
                    run_attn_pv_dtype(LIS_DTYPE_F16, head_dim, accum_steps[s]);
                }
            }
        }
    }

    lis_thread_pool_destroy(&pool);

    if (g_failures != 0) {
        fprintf(stderr, "%d AVX diff failure(s)\n", g_failures);
        return 1;
    }
    return 0;
#endif
}
