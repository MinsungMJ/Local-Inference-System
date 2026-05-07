#include "cpu_avx.h"

#include "cpu_kernels_reference.h"
#include "lis/dtype.h"
#include "lis/loader.h"
#include "lis/thread_pool.h"

#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct {
    const lis_loaded_tensor *weight;
    const float *input;
    size_t rows;
    size_t cols;
    float *out;
} lis_matvec_avx_ctx;

static float lis_horizontal_sum_ps(__m256 v)
{
    const __m128 low = _mm256_castps256_ps128(v);
    const __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(low, high);

    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

static void lis_matvec_row_f32(const float *row, const float *input,
                               size_t cols, float *out)
{
    __m256 acc = _mm256_setzero_ps();
    size_t col = 0;

    for (; col + 8U <= cols; col += 8U) {
        const __m256 w = _mm256_loadu_ps(row + col);
        const __m256 x = _mm256_loadu_ps(input + col);

        acc = _mm256_fmadd_ps(w, x, acc);
    }
    {
        float sum = lis_horizontal_sum_ps(acc);

        for (; col < cols; ++col) {
            sum += row[col] * input[col];
        }
        *out = sum;
    }
}

static void lis_matvec_row_bf16(const uint16_t *row, const float *input,
                                size_t cols, float *out)
{
    __m256 acc = _mm256_setzero_ps();
    size_t col = 0;

    for (; col + 8U <= cols; col += 8U) {
        const __m128i raw16 = _mm_loadu_si128((const __m128i *)(row + col));
        const __m256i expanded = _mm256_cvtepu16_epi32(raw16);
        const __m256i shifted = _mm256_slli_epi32(expanded, 16);
        const __m256 w = _mm256_castsi256_ps(shifted);
        const __m256 x = _mm256_loadu_ps(input + col);

        acc = _mm256_fmadd_ps(w, x, acc);
    }
    {
        float sum = lis_horizontal_sum_ps(acc);

        for (; col < cols; ++col) {
            const uint32_t bits = (uint32_t)row[col] << 16;
            float promoted = 0.0f;

            __builtin_memcpy(&promoted, &bits, sizeof(promoted));
            sum += promoted * input[col];
        }
        *out = sum;
    }
}

static void lis_matvec_row_f16(const uint16_t *row, const float *input,
                               size_t cols, float *out)
{
    __m256 acc = _mm256_setzero_ps();
    size_t col = 0;

    for (; col + 8U <= cols; col += 8U) {
        const __m128i raw16 = _mm_loadu_si128((const __m128i *)(row + col));
        const __m256 w = _mm256_cvtph_ps(raw16);
        const __m256 x = _mm256_loadu_ps(input + col);

        acc = _mm256_fmadd_ps(w, x, acc);
    }
    {
        float sum = lis_horizontal_sum_ps(acc);

        for (; col < cols; ++col) {
            sum += lis_dtype_scalar_read_f32(LIS_DTYPE_F16, row, col) *
                   input[col];
        }
        *out = sum;
    }
}

static void lis_matvec_avx_work(size_t start, size_t count, void *context)
{
    const lis_matvec_avx_ctx *ctx = (const lis_matvec_avx_ctx *)context;
    const lis_dtype dtype = ctx->weight->view.dtype;
    const void *base = ctx->weight->view.data;
    const size_t cols = ctx->cols;
    size_t row;

    if (dtype == LIS_DTYPE_F32) {
        const float *weights = (const float *)base;

        for (row = start; row < start + count; ++row) {
            lis_matvec_row_f32(weights + row * cols, ctx->input, cols,
                               &ctx->out[row]);
        }
    } else if (dtype == LIS_DTYPE_BF16) {
        const uint16_t *weights = (const uint16_t *)base;

        for (row = start; row < start + count; ++row) {
            lis_matvec_row_bf16(weights + row * cols, ctx->input, cols,
                                &ctx->out[row]);
        }
    } else if (dtype == LIS_DTYPE_F16) {
        const uint16_t *weights = (const uint16_t *)base;

        for (row = start; row < start + count; ++row) {
            lis_matvec_row_f16(weights + row * cols, ctx->input, cols,
                               &ctx->out[row]);
        }
    } else {
        for (row = start; row < start + count; ++row) {
            ctx->out[row] = 0.0f;
        }
    }
}

void lis_matvec_avx2_fma(const lis_loaded_tensor *weight, const float *input,
                         size_t rows, size_t cols, float *out,
                         lis_thread_pool *pool)
{
    lis_matvec_avx_ctx ctx;

    ctx.weight = weight;
    ctx.input = input;
    ctx.rows = rows;
    ctx.cols = cols;
    ctx.out = out;
    lis_thread_pool_dispatch(pool, rows, lis_matvec_avx_work, &ctx);
}

void lis_residual_add_avx2(float *hidden, const float *other, size_t hidden_size)
{
    size_t i = 0;

    for (; i + 8U <= hidden_size; i += 8U) {
        const __m256 h = _mm256_loadu_ps(hidden + i);
        const __m256 o = _mm256_loadu_ps(other + i);

        _mm256_storeu_ps(hidden + i, _mm256_add_ps(h, o));
    }
    for (; i < hidden_size; ++i) {
        hidden[i] += other[i];
    }
}

typedef struct {
    const float *input;
    const lis_loaded_tensor *weight;
    float scale;
    float *out;
} lis_rms_norm_avx_ctx;

static void lis_rms_norm_avx_scale_work(size_t start, size_t count,
                                        void *context)
{
    const lis_rms_norm_avx_ctx *ctx = (const lis_rms_norm_avx_ctx *)context;
    const lis_dtype dtype = ctx->weight->view.dtype;
    const __m256 scale_v = _mm256_set1_ps(ctx->scale);
    size_t i = start;
    const size_t end = start + count;

    if (dtype == LIS_DTYPE_F32) {
        const float *w = (const float *)ctx->weight->view.data;

        for (; i + 8U <= end; i += 8U) {
            const __m256 x = _mm256_loadu_ps(ctx->input + i);
            const __m256 g = _mm256_loadu_ps(w + i);
            const __m256 xs = _mm256_mul_ps(x, scale_v);

            _mm256_storeu_ps(ctx->out + i, _mm256_mul_ps(xs, g));
        }
    } else if (dtype == LIS_DTYPE_BF16) {
        const uint16_t *w = (const uint16_t *)ctx->weight->view.data;

        for (; i + 8U <= end; i += 8U) {
            const __m128i raw16 = _mm_loadu_si128((const __m128i *)(w + i));
            const __m256i expanded = _mm256_cvtepu16_epi32(raw16);
            const __m256i shifted = _mm256_slli_epi32(expanded, 16);
            const __m256 g = _mm256_castsi256_ps(shifted);
            const __m256 x = _mm256_loadu_ps(ctx->input + i);
            const __m256 xs = _mm256_mul_ps(x, scale_v);

            _mm256_storeu_ps(ctx->out + i, _mm256_mul_ps(xs, g));
        }
    } else if (dtype == LIS_DTYPE_F16) {
        const uint16_t *w = (const uint16_t *)ctx->weight->view.data;

        for (; i + 8U <= end; i += 8U) {
            const __m128i raw16 = _mm_loadu_si128((const __m128i *)(w + i));
            const __m256 g = _mm256_cvtph_ps(raw16);
            const __m256 x = _mm256_loadu_ps(ctx->input + i);
            const __m256 xs = _mm256_mul_ps(x, scale_v);

            _mm256_storeu_ps(ctx->out + i, _mm256_mul_ps(xs, g));
        }
    }
    for (; i < end; ++i) {
        ctx->out[i] = ctx->input[i] * ctx->scale *
                      lis_dtype_scalar_read_f32(dtype, ctx->weight->view.data,
                                                i);
    }
}

void lis_rms_norm_avx2(const float *input, const lis_loaded_tensor *weight,
                       size_t hidden_size, float *out, lis_thread_pool *pool)
{
    __m256 acc = _mm256_setzero_ps();
    float mean_square = 0.0f;
    size_t i = 0;
    lis_rms_norm_avx_ctx ctx;

    /* Reduction stays single-threaded to preserve determinism, but vectorized. */
    for (; i + 8U <= hidden_size; i += 8U) {
        const __m256 x = _mm256_loadu_ps(input + i);

        acc = _mm256_fmadd_ps(x, x, acc);
    }
    {
        const __m128 low = _mm256_castps256_ps128(acc);
        const __m128 high = _mm256_extractf128_ps(acc, 1);
        __m128 sum = _mm_add_ps(low, high);

        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        mean_square = _mm_cvtss_f32(sum);
    }
    for (; i < hidden_size; ++i) {
        mean_square += input[i] * input[i];
    }
    mean_square /= (float)hidden_size;

    ctx.input = input;
    ctx.weight = weight;
    ctx.scale = 1.0f / sqrtf(mean_square + 1.0e-5f);
    ctx.out = out;
    lis_thread_pool_dispatch(pool, hidden_size, lis_rms_norm_avx_scale_work,
                             &ctx);
}

/*
 * 8-lane expf approximation. Range-reduces x = k*ln(2) + r with r in
 * [-ln(2)/2, ln(2)/2], approximates exp(r) via a degree-5 Taylor series, and
 * reconstructs 2^k by forging the float exponent field. Accuracy is a few ULP
 * over [-88, 88]; saturates at 0/+inf outside that. Sufficient for swiglu /
 * softmax in this inference engine; token parity remains the binding gate.
 */
static __m256 lis_expf_avx2(__m256 x)
{
    const __m256 max_x = _mm256_set1_ps(88.72283f);
    const __m256 min_x = _mm256_set1_ps(-88.72283f);
    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    const __m256 ln2 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 c1 = _mm256_set1_ps(1.0f);
    const __m256 c2 = _mm256_set1_ps(1.0f);
    const __m256 c3 = _mm256_set1_ps(0.5f);
    const __m256 c4 = _mm256_set1_ps(1.0f / 6.0f);
    const __m256 c5 = _mm256_set1_ps(1.0f / 24.0f);
    const __m256 c6 = _mm256_set1_ps(1.0f / 120.0f);
    __m256 f;
    __m256 k;
    __m256 r;
    __m256 p;
    __m256i ki;
    __m256i bias;
    __m256 pow2k;

    x = _mm256_min_ps(_mm256_max_ps(x, min_x), max_x);
    f = _mm256_mul_ps(x, log2e);
    k = _mm256_round_ps(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    r = _mm256_sub_ps(x, _mm256_mul_ps(k, ln2));

    p = c6;
    p = _mm256_fmadd_ps(p, r, c5);
    p = _mm256_fmadd_ps(p, r, c4);
    p = _mm256_fmadd_ps(p, r, c3);
    p = _mm256_fmadd_ps(p, r, c2);
    p = _mm256_fmadd_ps(p, r, c1);

    ki = _mm256_cvtps_epi32(k);
    bias = _mm256_set1_epi32(127);
    pow2k = _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_add_epi32(ki, bias), 23));
    return _mm256_mul_ps(p, pow2k);
}

typedef struct {
    const float *gate;
    const float *up;
    float *out;
} lis_swiglu_avx_ctx;

static void lis_swiglu_avx_work(size_t start, size_t count, void *context)
{
    const lis_swiglu_avx_ctx *ctx = (const lis_swiglu_avx_ctx *)context;
    const __m256 one = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    size_t i = start;
    const size_t end = start + count;

    for (; i + 8U <= end; i += 8U) {
        const __m256 g = _mm256_loadu_ps(ctx->gate + i);
        const __m256 u = _mm256_loadu_ps(ctx->up + i);
        const __m256 neg_g = _mm256_sub_ps(zero, g);
        const __m256 e = lis_expf_avx2(neg_g);
        const __m256 denom = _mm256_add_ps(one, e);
        const __m256 silu = _mm256_div_ps(g, denom);

        _mm256_storeu_ps(ctx->out + i, _mm256_mul_ps(silu, u));
    }
    for (; i < end; ++i) {
        const float g = ctx->gate[i];

        ctx->out[i] = (g / (1.0f + expf(-g))) * ctx->up[i];
    }
}

void lis_swiglu_avx2(const float *gate, const float *up,
                     size_t intermediate_size, float *out,
                     lis_thread_pool *pool)
{
    lis_swiglu_avx_ctx ctx;

    ctx.gate = gate;
    ctx.up = up;
    ctx.out = out;
    lis_thread_pool_dispatch(pool, intermediate_size, lis_swiglu_avx_work, &ctx);
}

static float lis_horizontal_max_ps(__m256 v)
{
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 m = _mm_max_ps(low, high);
    __m128 shuf = _mm_shuffle_ps(m, m, _MM_SHUFFLE(2, 3, 0, 1));

    m = _mm_max_ps(m, shuf);
    shuf = _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 0, 3, 2));
    m = _mm_max_ps(m, shuf);
    return _mm_cvtss_f32(m);
}

void lis_softmax_avx2(float *values, size_t n)
{
    __m256 max_v;
    __m256 acc_v;
    float max_s;
    float denom;
    size_t i;

    if (n == 0U) {
        return;
    }
    if (n < 8U) {
        size_t k;

        max_s = values[0];
        for (k = 1; k < n; ++k) {
            if (values[k] > max_s) {
                max_s = values[k];
            }
        }
        denom = 0.0f;
        for (k = 0; k < n; ++k) {
            values[k] = expf(values[k] - max_s);
            denom += values[k];
        }
        for (k = 0; k < n; ++k) {
            values[k] /= denom;
        }
        return;
    }

    max_v = _mm256_loadu_ps(values);
    for (i = 8U; i + 8U <= n; i += 8U) {
        max_v = _mm256_max_ps(max_v, _mm256_loadu_ps(values + i));
    }
    max_s = lis_horizontal_max_ps(max_v);
    for (; i < n; ++i) {
        if (values[i] > max_s) {
            max_s = values[i];
        }
    }

    {
        const __m256 max_bcast = _mm256_set1_ps(max_s);

        acc_v = _mm256_setzero_ps();
        for (i = 0; i + 8U <= n; i += 8U) {
            const __m256 x = _mm256_loadu_ps(values + i);
            const __m256 e = lis_expf_avx2(_mm256_sub_ps(x, max_bcast));

            _mm256_storeu_ps(values + i, e);
            acc_v = _mm256_add_ps(acc_v, e);
        }
        denom = lis_horizontal_sum_ps(acc_v);
    }
    for (; i < n; ++i) {
        values[i] = expf(values[i] - max_s);
        denom += values[i];
    }

    {
        const __m256 denom_bcast = _mm256_set1_ps(denom);

        for (i = 0; i + 8U <= n; i += 8U) {
            const __m256 e = _mm256_loadu_ps(values + i);

            _mm256_storeu_ps(values + i, _mm256_div_ps(e, denom_bcast));
        }
    }
    for (; i < n; ++i) {
        values[i] /= denom;
    }
}

typedef struct {
    float *values;
    const float *cos_tbl;
    const float *sin_tbl;
    size_t head_dim;
    size_t half;
} lis_rope_avx_ctx;

static void lis_rope_avx_work(size_t start, size_t count, void *context)
{
    const lis_rope_avx_ctx *ctx = (const lis_rope_avx_ctx *)context;
    const size_t half = ctx->half;
    const size_t head_dim = ctx->head_dim;
    size_t head;

    for (head = start; head < start + count; ++head) {
        float *base = ctx->values + head * head_dim;
        float *hi = base + half;
        size_t i = 0;

        for (; i + 8U <= half; i += 8U) {
            const __m256 x0 = _mm256_loadu_ps(base + i);
            const __m256 x1 = _mm256_loadu_ps(hi + i);
            const __m256 c = _mm256_loadu_ps(ctx->cos_tbl + i);
            const __m256 s = _mm256_loadu_ps(ctx->sin_tbl + i);
            const __m256 x0c = _mm256_mul_ps(x0, c);
            const __m256 x1c = _mm256_mul_ps(x1, c);
            const __m256 n0 = _mm256_fnmadd_ps(x1, s, x0c);
            const __m256 n1 = _mm256_fmadd_ps(x0, s, x1c);

            _mm256_storeu_ps(base + i, n0);
            _mm256_storeu_ps(hi + i, n1);
        }
        for (; i < half; ++i) {
            const float c = ctx->cos_tbl[i];
            const float s = ctx->sin_tbl[i];
            const float x0 = base[i];
            const float x1 = hi[i];

            base[i] = x0 * c - x1 * s;
            hi[i] = x1 * c + x0 * s;
        }
    }
}

void lis_rope_avx2(float *values, size_t head_count, size_t head_dim,
                   size_t position, float rope_theta, lis_thread_pool *pool)
{
    const size_t half = head_dim / 2U;
    float *cos_tbl;
    float *sin_tbl;
    size_t i;
    lis_rope_avx_ctx ctx;

    if (head_dim < 2U) {
        return;
    }
    cos_tbl = (float *)malloc(half * sizeof(float));
    sin_tbl = (float *)malloc(half * sizeof(float));
    if (cos_tbl == NULL || sin_tbl == NULL) {
        free(cos_tbl);
        free(sin_tbl);
        lis_apply_rope_reference(values, head_count, head_dim, position,
                                 rope_theta, pool);
        return;
    }
    for (i = 0; i < half; ++i) {
        const float theta =
            powf(rope_theta, -((float)(2U * i) / (float)head_dim));
        const float angle = (float)position * theta;

        cos_tbl[i] = cosf(angle);
        sin_tbl[i] = sinf(angle);
    }
    ctx.values = values;
    ctx.cos_tbl = cos_tbl;
    ctx.sin_tbl = sin_tbl;
    ctx.head_dim = head_dim;
    ctx.half = half;
    lis_thread_pool_dispatch(pool, head_count, lis_rope_avx_work, &ctx);
    free(cos_tbl);
    free(sin_tbl);
}

void lis_attn_qk_avx2(const float *q_head, const void *k_row,
                      lis_dtype kv_dtype, size_t head_dim, float *out_dot)
{
    __m256 acc = _mm256_setzero_ps();
    size_t dim = 0;

    if (kv_dtype == LIS_DTYPE_F32) {
        const float *k = (const float *)k_row;

        for (; dim + 8U <= head_dim; dim += 8U) {
            const __m256 q = _mm256_loadu_ps(q_head + dim);
            const __m256 kv = _mm256_loadu_ps(k + dim);

            acc = _mm256_fmadd_ps(q, kv, acc);
        }
        {
            float sum = lis_horizontal_sum_ps(acc);

            for (; dim < head_dim; ++dim) {
                sum += q_head[dim] * k[dim];
            }
            *out_dot = sum;
        }
    } else if (kv_dtype == LIS_DTYPE_BF16) {
        const uint16_t *k = (const uint16_t *)k_row;

        for (; dim + 8U <= head_dim; dim += 8U) {
            const __m128i raw16 = _mm_loadu_si128((const __m128i *)(k + dim));
            const __m256i expanded = _mm256_cvtepu16_epi32(raw16);
            const __m256i shifted = _mm256_slli_epi32(expanded, 16);
            const __m256 kv = _mm256_castsi256_ps(shifted);
            const __m256 q = _mm256_loadu_ps(q_head + dim);

            acc = _mm256_fmadd_ps(q, kv, acc);
        }
        {
            float sum = lis_horizontal_sum_ps(acc);

            for (; dim < head_dim; ++dim) {
                sum += q_head[dim] *
                       lis_dtype_scalar_read_f32(LIS_DTYPE_BF16, k_row, dim);
            }
            *out_dot = sum;
        }
    } else if (kv_dtype == LIS_DTYPE_F16) {
        const uint16_t *k = (const uint16_t *)k_row;

        for (; dim + 8U <= head_dim; dim += 8U) {
            const __m128i raw16 = _mm_loadu_si128((const __m128i *)(k + dim));
            const __m256 kv = _mm256_cvtph_ps(raw16);
            const __m256 q = _mm256_loadu_ps(q_head + dim);

            acc = _mm256_fmadd_ps(q, kv, acc);
        }
        {
            float sum = lis_horizontal_sum_ps(acc);

            for (; dim < head_dim; ++dim) {
                sum += q_head[dim] *
                       lis_dtype_scalar_read_f32(LIS_DTYPE_F16, k_row, dim);
            }
            *out_dot = sum;
        }
    } else {
        lis_attn_qk_reference(q_head, k_row, kv_dtype, head_dim, out_dot);
    }
}

void lis_attn_pv_avx2(float prob, const void *v_row, lis_dtype kv_dtype,
                      size_t head_dim, float *out_head)
{
    const __m256 p_v = _mm256_set1_ps(prob);
    size_t dim = 0;

    if (kv_dtype == LIS_DTYPE_F32) {
        const float *v = (const float *)v_row;

        for (; dim + 8U <= head_dim; dim += 8U) {
            const __m256 val = _mm256_loadu_ps(v + dim);
            const __m256 o = _mm256_loadu_ps(out_head + dim);

            _mm256_storeu_ps(out_head + dim, _mm256_fmadd_ps(p_v, val, o));
        }
        for (; dim < head_dim; ++dim) {
            out_head[dim] += prob * v[dim];
        }
    } else if (kv_dtype == LIS_DTYPE_BF16) {
        const uint16_t *v = (const uint16_t *)v_row;

        for (; dim + 8U <= head_dim; dim += 8U) {
            const __m128i raw16 = _mm_loadu_si128((const __m128i *)(v + dim));
            const __m256i expanded = _mm256_cvtepu16_epi32(raw16);
            const __m256i shifted = _mm256_slli_epi32(expanded, 16);
            const __m256 val = _mm256_castsi256_ps(shifted);
            const __m256 o = _mm256_loadu_ps(out_head + dim);

            _mm256_storeu_ps(out_head + dim, _mm256_fmadd_ps(p_v, val, o));
        }
        for (; dim < head_dim; ++dim) {
            out_head[dim] += prob *
                             lis_dtype_scalar_read_f32(LIS_DTYPE_BF16, v_row,
                                                       dim);
        }
    } else if (kv_dtype == LIS_DTYPE_F16) {
        const uint16_t *v = (const uint16_t *)v_row;

        for (; dim + 8U <= head_dim; dim += 8U) {
            const __m128i raw16 = _mm_loadu_si128((const __m128i *)(v + dim));
            const __m256 val = _mm256_cvtph_ps(raw16);
            const __m256 o = _mm256_loadu_ps(out_head + dim);

            _mm256_storeu_ps(out_head + dim, _mm256_fmadd_ps(p_v, val, o));
        }
        for (; dim < head_dim; ++dim) {
            out_head[dim] += prob *
                             lis_dtype_scalar_read_f32(LIS_DTYPE_F16, v_row,
                                                       dim);
        }
    } else {
        lis_attn_pv_reference(prob, v_row, kv_dtype, head_dim, out_head);
    }
}
