#include "cpu_kernels_reference.h"

#include "lis/dtype.h"

#include <math.h>
#include <stddef.h>

typedef struct {
    const lis_loaded_tensor *weight;
    const float *input;
    size_t rows;
    size_t cols;
    float *out;
} lis_matvec_ctx;

static void lis_matvec_work(size_t start, size_t count, void *context)
{
    const lis_matvec_ctx *ctx = (const lis_matvec_ctx *)context;
    size_t row;

    for (row = start; row < start + count; ++row) {
        float sum = 0.0f;
        size_t col;

        for (col = 0; col < ctx->cols; ++col) {
            sum += lis_dtype_scalar_read_f32(ctx->weight->view.dtype,
                                             ctx->weight->view.data,
                                             row * ctx->cols + col) *
                   ctx->input[col];
        }
        ctx->out[row] = sum;
    }
}

void lis_matvec_reference(const lis_loaded_tensor *weight, const float *input,
                          size_t rows, size_t cols, float *out,
                          lis_thread_pool *pool)
{
    lis_matvec_ctx ctx;

    ctx.weight = weight;
    ctx.input = input;
    ctx.rows = rows;
    ctx.cols = cols;
    ctx.out = out;
    lis_thread_pool_dispatch(pool, rows, lis_matvec_work, &ctx);
}

typedef struct {
    const float *input;
    const lis_loaded_tensor *weight;
    float scale;
    float *out;
} lis_rms_norm_ctx;

static void lis_rms_norm_scale_work(size_t start, size_t count, void *context)
{
    const lis_rms_norm_ctx *ctx = (const lis_rms_norm_ctx *)context;
    size_t index;

    for (index = start; index < start + count; ++index) {
        ctx->out[index] = ctx->input[index] * ctx->scale *
                          lis_dtype_scalar_read_f32(ctx->weight->view.dtype,
                                                    ctx->weight->view.data,
                                                    index);
    }
}

void lis_rms_norm_reference(const float *input,
                            const lis_loaded_tensor *weight,
                            size_t hidden_size, float *out,
                            lis_thread_pool *pool)
{
    float mean_square = 0.0f;
    size_t index;
    lis_rms_norm_ctx ctx;

    /* Reduction stays single-threaded to preserve determinism. */
    for (index = 0; index < hidden_size; ++index) {
        mean_square += input[index] * input[index];
    }
    mean_square /= (float)hidden_size;

    ctx.input = input;
    ctx.weight = weight;
    ctx.scale = 1.0f / sqrtf(mean_square + 1.0e-5f);
    ctx.out = out;
    lis_thread_pool_dispatch(pool, hidden_size, lis_rms_norm_scale_work, &ctx);
}

typedef struct {
    float *values;
    size_t head_dim;
    size_t position;
    float rope_theta;
} lis_rope_ctx;

static void lis_rope_work(size_t start, size_t count, void *context)
{
    const lis_rope_ctx *ctx = (const lis_rope_ctx *)context;
    const size_t half = ctx->head_dim / 2U;
    size_t head;

    for (head = start; head < start + count; ++head) {
        const size_t head_base = head * ctx->head_dim;
        size_t i;

        for (i = 0; i < half; ++i) {
            const float theta =
                powf(ctx->rope_theta,
                     -((float)(2U * i) / (float)ctx->head_dim));
            const float angle = (float)ctx->position * theta;
            const float c = cosf(angle);
            const float s = sinf(angle);
            const float x0 = ctx->values[head_base + i];
            const float x1 = ctx->values[head_base + i + half];

            ctx->values[head_base + i] = x0 * c - x1 * s;
            ctx->values[head_base + i + half] = x1 * c + x0 * s;
        }
    }
}

void lis_apply_rope_reference(float *values, size_t head_count,
                              size_t head_dim, size_t position,
                              float rope_theta, lis_thread_pool *pool)
{
    lis_rope_ctx ctx;

    if (head_dim < 2) {
        return;
    }
    ctx.values = values;
    ctx.head_dim = head_dim;
    ctx.position = position;
    ctx.rope_theta = rope_theta;
    lis_thread_pool_dispatch(pool, head_count, lis_rope_work, &ctx);
}

typedef struct {
    const float *gate;
    const float *up;
    float *out;
} lis_swiglu_ctx;

static void lis_swiglu_work(size_t start, size_t count, void *context)
{
    const lis_swiglu_ctx *ctx = (const lis_swiglu_ctx *)context;
    size_t index;

    for (index = start; index < start + count; ++index) {
        const float g = ctx->gate[index];

        ctx->out[index] = (g / (1.0f + expf(-g))) * ctx->up[index];
    }
}

void lis_swiglu_reference(const float *gate, const float *up,
                          size_t intermediate_size, float *out,
                          lis_thread_pool *pool)
{
    lis_swiglu_ctx ctx;

    ctx.gate = gate;
    ctx.up = up;
    ctx.out = out;
    lis_thread_pool_dispatch(pool, intermediate_size, lis_swiglu_work, &ctx);
}

void lis_residual_add_reference(float *hidden, const float *other,
                                size_t hidden_size)
{
    size_t index;

    for (index = 0; index < hidden_size; ++index) {
        hidden[index] += other[index];
    }
}

void lis_softmax_reference(float *values, size_t n)
{
    float max_v;
    float denom = 0.0f;
    size_t i;

    if (n == 0) {
        return;
    }
    max_v = values[0];
    for (i = 1; i < n; ++i) {
        if (values[i] > max_v) {
            max_v = values[i];
        }
    }
    for (i = 0; i < n; ++i) {
        values[i] = expf(values[i] - max_v);
        denom += values[i];
    }
    for (i = 0; i < n; ++i) {
        values[i] /= denom;
    }
}

void lis_attn_qk_reference(const float *q_head, const void *k_row,
                           lis_dtype kv_dtype, size_t head_dim,
                           float *out_dot)
{
    float dot = 0.0f;
    size_t dim;

    for (dim = 0; dim < head_dim; ++dim) {
        dot += q_head[dim] *
               lis_dtype_scalar_read_f32(kv_dtype, k_row, dim);
    }
    *out_dot = dot;
}

void lis_attn_pv_reference(float prob, const void *v_row, lis_dtype kv_dtype,
                           size_t head_dim, float *out_head)
{
    size_t dim;

    for (dim = 0; dim < head_dim; ++dim) {
        out_head[dim] += prob *
                         lis_dtype_scalar_read_f32(kv_dtype, v_row, dim);
    }
}
