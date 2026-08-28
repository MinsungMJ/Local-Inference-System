#include "lis/runtime.h"
#include "lis/cpu_ops.h"
#include "lis/dtype.h"
#include "lis/layer_trace.h"
#include "lis/thread_pool.h"

#ifdef LIS_TESTING
#include "lis_test_controls.h"
#endif

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* Debug-only: fp32 K snapshot for layer-1 kv-head-0, used to
   reconstruct attention scores without fp16 KV-cache loss. */
static float *s_dbg_l1_k_fp32 = NULL;
static size_t s_dbg_l1_k_fp32_cap = 0;

typedef struct {
    float *hidden;
    float *norm;
    float *q;
    float *k;
    float *v;
    float *attn;
    float *attn_out;
    float *attn_probs;
    float *gate;
    float *up;
    float *mlp;
    float *mlp_out;
    float *scores;
} lis_llama_scratch;

static void lis_llama_scratch_destroy(lis_llama_scratch *scratch);
static float lis_scalar_read(lis_dtype dtype, const void *data, size_t index);

static void lis_checkpoint_diagnostic_impl(size_t step, const char *phase,
                                            const char *name,
                                            const size_t *dims,
                                            size_t rank, const float *data,
                                            size_t count,
                                            lis_runtime_context *runtime,
                                            int is_layer_output,
                                            size_t layer_index)
{
    size_t i;
    lis_layer_trace_step lts = {0};
    int truncated = 0;
    lis_layer_trace_record *record =
        runtime != NULL ? runtime->layer_trace_record : NULL;
    const float *observed_data = data;
#ifdef LIS_TESTING
    float *perturbed_data = NULL;
#endif

    if (count == 0 || data == NULL || dims == NULL || phase == NULL) {
        return;
    }

#ifdef LIS_TESTING
    if (record != NULL && is_layer_output &&
        step == record->layout_runtime_checkpoint_step) {
        size_t perturbation_element = 0U;
        float perturbation_delta = 0.0f;

        if (lis_test_control_layer_observation(
                layer_index, count, &perturbation_element,
                &perturbation_delta)) {
            if (count > SIZE_MAX / sizeof(*perturbed_data)) {
                record->append_failed = 1;
                return;
            }
            perturbed_data = malloc(count * sizeof(*perturbed_data));
            if (perturbed_data == NULL) {
                record->append_failed = 1;
                return;
            }
            memcpy(perturbed_data, data, count * sizeof(*perturbed_data));
            perturbed_data[perturbation_element] += perturbation_delta;
            observed_data = perturbed_data;
            lis_test_control_mark_layer_observation_applied();
        }
    }
#endif

    lts.step = step;
    if (snprintf(lts.phase, sizeof(lts.phase), "%s", phase) >= (int)sizeof(lts.phase))
        truncated = 1;
    if (snprintf(lts.name, sizeof(lts.name), "%s", name) >= (int)sizeof(lts.name))
        truncated = 1;
    lts.rank = rank < LIS_LAYER_TRACE_MAX_RANK ? rank : LIS_LAYER_TRACE_MAX_RANK;
    for (i = 0; i < lts.rank; ++i) {
        lts.shape[i] = dims[i];
    }
    lts.min = observed_data[0];
    lts.max = observed_data[0];
    for (i = 0; i < count; ++i) {
        float v = observed_data[i];
        if (isnan(v)) {
            lts.nan = 1;
        } else if (isinf(v)) {
            lts.inf = 1;
        }
        if (v < lts.min && !isnan(v)) lts.min = v;
        if (v > lts.max && !isnan(v)) lts.max = v;
        if (!isnan(v) && !isinf(v)) {
            lts.mean += v;
            lts.l2 += v * v;
        }
    }
    lts.mean = lts.mean / (float)count;
    lts.l2 = sqrtf(lts.l2);

    fprintf(stderr, "lis: layer-checkpoint step=%zu phase=%s name=%s shape=[",
            step, phase, name);
    for (i = 0; i < rank; ++i) {
        fprintf(stderr, "%s%zu", i == 0 ? "" : ",", dims[i]);
    }
    fprintf(stderr, "] min=%.6g max=%.6g mean=%.6g l2=%.6g nan=%d inf=%d\n",
            lts.min, lts.max, lts.mean, lts.l2, lts.nan, lts.inf);

    if (record != NULL && !record->append_failed) {
        if (truncated) {
            record->append_failed = 1;
        } else {
            lis_status status = LIS_STATUS_OK;

            if (is_layer_output) {
                status = lis_layer_trace_step_set_layer_output(
                    &lts, layer_index, observed_data, count);
            }
            if (status != LIS_STATUS_OK ||
                lis_layer_trace_record_append(record, &lts) != LIS_STATUS_OK) {
                record->append_failed = 1;
            }
        }
    }
#ifdef LIS_TESTING
    free(perturbed_data);
#endif
}

static void lis_checkpoint_diagnostic(size_t step, const char *phase,
                                      const char *name, const size_t *dims,
                                      size_t rank, const float *data,
                                      size_t count,
                                      lis_runtime_context *runtime)
{
    lis_checkpoint_diagnostic_impl(step, phase, name, dims, rank, data, count,
                                   runtime, 0, 0U);
}

static void lis_checkpoint_layer_output_diagnostic(
    size_t step,
    const char *phase,
    const char *name,
    const size_t *dims,
    size_t rank,
    const float *data,
    size_t count,
    lis_runtime_context *runtime,
    size_t layer_index)
{
    lis_checkpoint_diagnostic_impl(step, phase, name, dims, rank, data, count,
                                   runtime, 1, layer_index);
}

static void lis_checkpoint_strided_diagnostic(size_t step, const char *phase,
                                               const char *name,
                                               const float *data,
                                               size_t rows, size_t stride,
                                               size_t cols,
                                               lis_runtime_context *runtime)
{
    size_t row;
    size_t col;
    size_t count;
    int have_value = 0;
    lis_layer_trace_step lts = {0};
    int truncated = 0;
    int first = 1;
    lis_layer_trace_record *record =
        runtime != NULL ? runtime->layer_trace_record : NULL;

    if (data == NULL || phase == NULL || rows == 0 || cols == 0 ||
        stride < cols || rows > SIZE_MAX / cols) {
        return;
    }

    count = rows * cols;
    lts.step = step;
    if (snprintf(lts.phase, sizeof(lts.phase), "%s", phase) >= (int)sizeof(lts.phase))
        truncated = 1;
    if (snprintf(lts.name, sizeof(lts.name), "%s", name) >= (int)sizeof(lts.name))
        truncated = 1;
    lts.rank = 1;
    lts.shape[0] = count;

    for (row = 0; row < rows; ++row) {
        const float *base = data + row * stride;

        for (col = 0; col < cols; ++col) {
            float v = base[col];

            if (first) {
                lts.min = v;
                lts.max = v;
                first = 0;
                have_value = 1;
            }
            if (isnan(v)) {
                lts.nan = 1;
            } else if (isinf(v)) {
                lts.inf = 1;
            }
            if (v < lts.min && !isnan(v)) lts.min = v;
            if (v > lts.max && !isnan(v)) lts.max = v;
            if (!isnan(v) && !isinf(v)) {
                lts.mean += v;
                lts.l2 += v * v;
            }
        }
    }

    if (!have_value) {
        return;
    }
    lts.mean = lts.mean / (float)count;
    lts.l2 = sqrtf(lts.l2);

    fprintf(stderr,
            "lis: layer-checkpoint step=%zu phase=%s name=%s "
            "shape=[%zu] min=%.6g max=%.6g mean=%.6g l2=%.6g "
            "nan=%d inf=%d\n",
            step, phase, name, count, lts.min, lts.max, lts.mean, lts.l2,
            lts.nan, lts.inf);

    if (record != NULL && !record->append_failed) {
        if (truncated) {
            record->append_failed = 1;
        } else {
            lis_layer_trace_record_append(record, &lts);
        }
    }
}

static void lis_checkpoint_weight_diagnostic(size_t step, const char *phase,
                                              const char *name,
                                              const lis_loaded_tensor *tensor,
                                              size_t element_count,
                                              lis_runtime_context *runtime)
{
    size_t i;
    lis_layer_trace_step lts = {0};
    int truncated = 0;
    lis_layer_trace_record *record =
        runtime != NULL ? runtime->layer_trace_record : NULL;

    if (element_count == 0 || tensor == NULL || tensor->view.data == NULL) {
        return;
    }

    lts.step = step;
    if (snprintf(lts.phase, sizeof(lts.phase), "%s", phase) >= (int)sizeof(lts.phase))
        truncated = 1;
    if (snprintf(lts.name, sizeof(lts.name), "%s", name) >= (int)sizeof(lts.name))
        truncated = 1;
    lts.rank = 1;
    lts.shape[0] = element_count;

    lts.min = lis_scalar_read(tensor->view.dtype, tensor->view.data, 0);
    lts.max = lts.min;

    for (i = 0; i < element_count; ++i) {
        float v = lis_scalar_read(tensor->view.dtype, tensor->view.data, i);
        if (isnan(v)) {
            lts.nan = 1;
        } else if (isinf(v)) {
            lts.inf = 1;
        }
        if (v < lts.min && !isnan(v)) lts.min = v;
        if (v > lts.max && !isnan(v)) lts.max = v;
        if (!isnan(v) && !isinf(v)) {
            lts.mean += v;
            lts.l2 += v * v;
        }
    }

    lts.mean = lts.mean / (float)element_count;
    lts.l2 = sqrtf(lts.l2);

    fprintf(stderr,
            "lis: layer-checkpoint step=%zu phase=%s name=%s "
            "shape=[%zu] min=%.6g max=%.6g mean=%.6g l2=%.6g "
            "nan=%d inf=%d\n",
            step, phase, name, element_count, lts.min, lts.max, lts.mean, lts.l2,
            lts.nan, lts.inf);

    if (record != NULL && !record->append_failed) {
        if (truncated) {
            record->append_failed = 1;
        } else {
            lis_layer_trace_record_append(record, &lts);
        }
    }
}

static void lis_checkpoint_per_head_diagnostic(size_t step, const char *phase,
                                                const char *base_name,
                                                const float *data,
                                                size_t head_count,
                                                size_t head_dim,
                                                lis_runtime_context *runtime)
{
    size_t head;
    size_t rank = 1;
    size_t dims[1] = { head_dim };

    for (head = 0; head < head_count; ++head) {
        const float *hd = data + head * head_dim;
        float min = hd[0];
        float max = hd[0];
        float sum = 0.0f;
        float sq_sum = 0.0f;
        float mean;
        float l2;
        size_t d;
        char name[96];
        int name_truncated = 0;

        for (d = 0; d < head_dim; ++d) {
            float v = hd[d];
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
            sq_sum += v * v;
        }
        mean = sum / (float)head_dim;
        l2 = sqrtf(sq_sum);

        if (snprintf(name, sizeof(name), "%s.head.%zu", base_name, head) >=
            (int)sizeof(name)) {
            name_truncated = 1;
        }

        /* Stderr is always emitted; never gated by JSON capture state. */
        fprintf(stderr,
                "lis: layer-checkpoint step=%zu phase=%s name=%s.head.%zu "
                "shape=[%zu] min=%.6g max=%.6g mean=%.6g l2=%.6g "
                "nan=0 inf=0\n",
                step, phase, base_name, head, head_dim,
                min, max, mean, l2);

        if (runtime != NULL && runtime->layer_trace_record != NULL &&
            !runtime->layer_trace_record->append_failed) {
            lis_layer_trace_step lts = {0};
            int truncated = name_truncated;

            lts.step = step;
            if (snprintf(lts.phase, sizeof(lts.phase), "%s", phase) >=
                (int)sizeof(lts.phase)) {
                truncated = 1;
            }
            if (!name_truncated) {
                if (snprintf(lts.name, sizeof(lts.name), "%s", name) >=
                    (int)sizeof(lts.name)) {
                    truncated = 1;
                }
            } else {
                truncated = 1;
            }
            lts.rank = rank;
            lts.shape[0] = dims[0];
            lts.min = min;
            lts.max = max;
            lts.mean = mean;
            lts.l2 = l2;
            lts.nan = 0;
            lts.inf = 0;
            if (truncated) {
                runtime->layer_trace_record->append_failed = 1;
            } else {
                lis_layer_trace_record_append(runtime->layer_trace_record, &lts);
            }
        }
    }
}

static void lis_llama_observe_intra_layer(
    lis_runtime_context *runtime,
    lis_intra_layer_stage stage,
    size_t checkpoint_step,
    size_t layer,
    size_t position,
    const lis_intra_layer_fp32_view *view)
{
    lis_status status;

    if (runtime == NULL || runtime->intra_layer_record == NULL) {
        return;
    }
    status = lis_intra_layer_observe_fp32(
        runtime->intra_layer_record, stage, checkpoint_step, layer, position,
        view);
    if (status != LIS_STATUS_OK) {
        lis_intra_layer_record_invalidate(runtime->intra_layer_record);
    }
}

static uint16_t lis_f32_to_f16(float value)
{
    uint32_t bits = 0;
    uint32_t sign = 0;
    int32_t exp = 0;
    uint32_t frac = 0;

    memcpy(&bits, &value, sizeof(bits));
    sign = (bits >> 16) & 0x8000U;
    exp = (int32_t)((bits >> 23) & 0xffU) - 127 + 15;
    frac = bits & 0x7fffffU;
    if (exp <= 0) {
        return (uint16_t)sign;
    }
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00U);
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (frac >> 13));
}

static float lis_scalar_read(lis_dtype dtype, const void *data, size_t index)
{
    return lis_dtype_scalar_read_f32(dtype, data, index);
}

static void lis_scalar_write(lis_dtype dtype, void *data, size_t index,
                             float value)
{
    if (dtype == LIS_DTYPE_F32) {
        memcpy((unsigned char *)data + index * sizeof(value), &value,
               sizeof(value));
    } else if (dtype == LIS_DTYPE_BF16) {
        uint32_t bits = 0;
        uint16_t raw = 0;

        memcpy(&bits, &value, sizeof(bits));
        raw = (uint16_t)(bits >> 16);
        memcpy((unsigned char *)data + index * sizeof(raw), &raw,
               sizeof(raw));
    } else if (dtype == LIS_DTYPE_F16) {
        uint16_t raw = lis_f32_to_f16(value);

        memcpy((unsigned char *)data + index * sizeof(raw), &raw,
               sizeof(raw));
    }
}

static const lis_loaded_tensor *lis_find_tensor(const lis_loaded_model *model,
                                                const char *name)
{
    size_t index;

    if (model == NULL || name == NULL) {
        return NULL;
    }
    for (index = 0; index < model->tensor_count; ++index) {
        if (strcmp(model->tensors[index].name, name) == 0) {
            return &model->tensors[index];
        }
    }
    return NULL;
}

static lis_status lis_expect_tensor(const lis_loaded_model *model,
                                    const char *name,
                                    const size_t *dims,
                                    size_t rank,
                                    lis_dtype expected_dtype,
                                    const lis_loaded_tensor **out_tensor)
{
    const lis_loaded_tensor *tensor = lis_find_tensor(model, name);
    size_t index;

    if (out_tensor == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    *out_tensor = NULL;
    if (tensor == NULL) {
        return LIS_STATUS_FORMAT;
    }
    if (tensor->view.shape.rank != rank) {
        return LIS_STATUS_SHAPE_MISMATCH;
    }
    for (index = 0; index < rank; ++index) {
        if (tensor->view.shape.dims[index] != dims[index]) {
            return LIS_STATUS_SHAPE_MISMATCH;
        }
    }
    if (tensor->view.dtype != expected_dtype) {
        return LIS_STATUS_UNSUPPORTED_DTYPE;
    }
    *out_tensor = tensor;
    return LIS_STATUS_OK;
}

static lis_status lis_layer_tensor(const lis_loaded_model *model, size_t layer,
                                   const char *suffix, const size_t *dims,
                                   size_t rank,
                                   const lis_loaded_tensor **out_tensor)
{
    char name[128];

    if (snprintf(name, sizeof(name), "lis.layer.%zu.%s", layer, suffix) >=
        (int)sizeof(name)) {
        return LIS_STATUS_FORMAT;
    }
    return lis_expect_tensor(model, name, dims, rank,
                             model->metadata.config.weight_dtype, out_tensor);
}

static lis_status lis_store_kv(lis_runtime_context *runtime, size_t layer,
                               size_t position, const float *k,
                               const float *v)
{
    const lis_model_config *cfg = &runtime->metadata.config;
    size_t head;

    for (head = 0; head < cfg->kv_head_count; ++head) {
        size_t dim;

        for (dim = 0; dim < cfg->head_dim; ++dim) {
            void *key_ptr = NULL;
            void *value_ptr = NULL;
            const size_t index = head * cfg->head_dim + dim;
            lis_status status =
                lis_kv_cache_key_ptr(&runtime->kv_cache, layer, 0, position,
                                     head, dim, &key_ptr);

            if (status != LIS_STATUS_OK) {
                return status;
            }
            status = lis_kv_cache_value_ptr(&runtime->kv_cache, layer, 0,
                                            position, head, dim, &value_ptr);
            if (status != LIS_STATUS_OK) {
                return status;
            }
            lis_scalar_write(runtime->kv_cache.layout.dtype, key_ptr, 0,
                             k[index]);
            lis_scalar_write(runtime->kv_cache.layout.dtype, value_ptr, 0,
                             v[index]);
        }
    }
    return LIS_STATUS_OK;
}

typedef struct {
    lis_runtime_context *runtime;
    size_t layer;
    size_t position;
    const float *q;
    float *scores_base;
    float *probs_base;
    size_t scores_stride;
    float *out;
    float scale;
    lis_status *status_out;
} lis_attention_ctx;

static void lis_attention_work(size_t start, size_t count, void *context)
{
    const lis_attention_ctx *ctx = (const lis_attention_ctx *)context;
    const lis_model_config *cfg = &ctx->runtime->metadata.config;
    const lis_dtype kv_dtype = ctx->runtime->kv_cache.layout.dtype;
    const size_t head_dim = cfg->head_dim;
    size_t head;

    for (head = start; head < start + count; ++head) {
        const size_t kv_head = head * cfg->kv_head_count /
                               cfg->attention_head_count;
        const float *q_head = ctx->q + head * head_dim;
        float *out_head = ctx->out + head * head_dim;
        float *head_scores = ctx->scores_base + head * ctx->scores_stride;
        float *head_probs = ctx->probs_base + head * ctx->scores_stride;
        size_t pos;
        size_t dim;

        for (dim = 0; dim < head_dim; ++dim) {
            out_head[dim] = 0.0f;
        }
        for (pos = 0; pos <= ctx->position; ++pos) {
            void *key_ptr = NULL;
            lis_status status =
                lis_kv_cache_key_ptr(&ctx->runtime->kv_cache, ctx->layer,
                                     0, pos, kv_head, 0, &key_ptr);
            float dot = 0.0f;

            if (status != LIS_STATUS_OK) {
                *ctx->status_out = status;
                return;
            }
            lis_attn_qk(q_head, key_ptr, kv_dtype, head_dim, &dot);
            head_scores[pos] = dot * ctx->scale;
            head_probs[pos] = head_scores[pos];
        }
        lis_softmax(head_probs, ctx->position + 1U);
        for (pos = 0; pos <= ctx->position; ++pos) {
            void *value_ptr = NULL;
            lis_status status =
                lis_kv_cache_value_ptr(&ctx->runtime->kv_cache, ctx->layer,
                                       0, pos, kv_head, 0, &value_ptr);

            if (status != LIS_STATUS_OK) {
                *ctx->status_out = status;
                return;
            }
            lis_attn_pv(head_probs[pos], value_ptr, kv_dtype, head_dim,
                        out_head);
        }
    }
}

static lis_status lis_attention(lis_runtime_context *runtime, size_t layer,
                                size_t position, const float *q,
                                float *scores_base, float *probs_base,
                                size_t scores_stride, float *out,
                                lis_thread_pool *pool)
{
    const lis_model_config *cfg = &runtime->metadata.config;
    lis_attention_ctx ctx;
    lis_status status = LIS_STATUS_OK;

    ctx.runtime = runtime;
    ctx.layer = layer;
    ctx.position = position;
    ctx.q = q;
    ctx.scores_base = scores_base;
    ctx.probs_base = probs_base;
    ctx.scores_stride = scores_stride;
    ctx.out = out;
    ctx.scale = 1.0f / sqrtf((float)cfg->head_dim);
    ctx.status_out = &status;
    lis_thread_pool_dispatch(pool, cfg->attention_head_count,
                             lis_attention_work, &ctx);
    return status;
}

static lis_status lis_llama_scratch_init(lis_llama_scratch *scratch,
                                         const lis_model_config *cfg)
{
    const size_t hidden = cfg->hidden_size;
    const size_t q_size = cfg->attention_head_count * cfg->head_dim;
    const size_t kv_size = cfg->kv_head_count * cfg->head_dim;
    const size_t intermediate = cfg->intermediate_size;
    const size_t max_tokens = cfg->context.configured_max_tokens;

    memset(scratch, 0, sizeof(*scratch));
    scratch->hidden = calloc(hidden, sizeof(*scratch->hidden));
    scratch->norm = calloc(hidden, sizeof(*scratch->norm));
    scratch->q = calloc(q_size, sizeof(*scratch->q));
    scratch->k = calloc(kv_size, sizeof(*scratch->k));
    scratch->v = calloc(kv_size, sizeof(*scratch->v));
    scratch->attn = calloc(q_size, sizeof(*scratch->attn));
    scratch->attn_out = calloc(hidden, sizeof(*scratch->attn_out));
    scratch->attn_probs = calloc(cfg->attention_head_count * max_tokens,
                                 sizeof(*scratch->attn_probs));
    scratch->gate = calloc(intermediate, sizeof(*scratch->gate));
    scratch->up = calloc(intermediate, sizeof(*scratch->up));
    scratch->mlp = calloc(intermediate, sizeof(*scratch->mlp));
    scratch->mlp_out = calloc(hidden, sizeof(*scratch->mlp_out));
    /*
     * Per-head scores buffer for parallel attention dispatch.
     * Each head gets max_tokens floats so heads can run concurrently.
     */
    scratch->scores = calloc(cfg->attention_head_count * max_tokens,
                             sizeof(*scratch->scores));
    if (scratch->hidden == NULL || scratch->norm == NULL ||
        scratch->q == NULL || scratch->k == NULL || scratch->v == NULL ||
        scratch->attn == NULL || scratch->attn_out == NULL ||
        scratch->attn_probs == NULL ||
        scratch->gate == NULL || scratch->up == NULL ||
        scratch->mlp == NULL || scratch->mlp_out == NULL ||
        scratch->scores == NULL) {
        lis_llama_scratch_destroy(scratch);
        return LIS_STATUS_NO_MEMORY;
    }
    return LIS_STATUS_OK;
}

static void lis_llama_scratch_destroy(lis_llama_scratch *scratch)
{
    if (scratch == NULL) {
        return;
    }
    free(scratch->hidden);
    free(scratch->norm);
    free(scratch->q);
    free(scratch->k);
    free(scratch->v);
    free(scratch->attn);
    free(scratch->attn_out);
    free(scratch->attn_probs);
    free(scratch->gate);
    free(scratch->up);
    free(scratch->mlp);
    free(scratch->mlp_out);
    free(scratch->scores);
    memset(scratch, 0, sizeof(*scratch));
}

static lis_status lis_llama_forward_token(lis_runtime_context *runtime,
                                          const lis_loaded_model *model,
                                          size_t token_id,
                                          size_t position,
                                          size_t checkpoint_step,
                                          const char *checkpoint_phase,
                                          lis_llama_scratch *scratch,
                                          float *out_logits,
                                          size_t logits_len)
{
    const lis_model_config *cfg = &runtime->metadata.config;
    const lis_loaded_tensor *tensor = NULL;
    size_t dims[2] = { 0 };
    size_t index;
    size_t layer;
    int emit_checkpoints;
    int emit_legacy_checkpoints;
    int capture_intra_layer;
    lis_intra_layer_trace_record *intra_record =
        runtime->intra_layer_record;
    lis_status status;

    if (token_id >= cfg->vocab_size || logits_len < cfg->vocab_size) {
        return LIS_STATUS_LIMIT_EXCEEDED;
    }
    dims[0] = cfg->vocab_size;
    dims[1] = cfg->hidden_size;
    status = lis_expect_tensor(model, "lis.token_embeddings.weight", dims, 2,
                               cfg->weight_dtype, &tensor);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    for (index = 0; index < cfg->hidden_size; ++index) {
        scratch->hidden[index] =
            lis_scalar_read(tensor->view.dtype, tensor->view.data,
                            token_id * cfg->hidden_size + index);
    }

    emit_checkpoints = runtime->layer_checkpoints_enabled &&
                       checkpoint_phase != NULL &&
                       checkpoint_step == runtime->layer_checkpoints_target_step;
    emit_legacy_checkpoints = emit_checkpoints && intra_record == NULL;
    capture_intra_layer =
        intra_record != NULL && checkpoint_phase != NULL &&
        strcmp(checkpoint_phase, LIS_INTRA_LAYER_PHASE_DECODE_NAME) == 0 &&
        checkpoint_step == intra_record->runtime_checkpoint_step &&
        position == intra_record->token_position;

    if (emit_legacy_checkpoints) {
        const size_t h_dims[1] = { cfg->hidden_size };

        lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                  "embedding", h_dims, 1, scratch->hidden,
                                  cfg->hidden_size,
            runtime);
    }

    for (layer = 0; layer < cfg->layer_count; ++layer) {
        const size_t q_dims[2] = {
            cfg->attention_head_count * cfg->head_dim,
            cfg->hidden_size
        };
        const size_t kv_dims[2] = {
            cfg->kv_head_count * cfg->head_dim,
            cfg->hidden_size
        };
        const size_t o_dims[2] = {
            cfg->hidden_size,
            cfg->attention_head_count * cfg->head_dim
        };
        const size_t mlp_in_dims[2] = {
            cfg->intermediate_size,
            cfg->hidden_size
        };
        const size_t mlp_down_dims[2] = {
            cfg->hidden_size,
            cfg->intermediate_size
        };
        const size_t norm_dims[1] = { cfg->hidden_size };

        if (capture_intra_layer && layer == intra_record->target_layer) {
            const lis_intra_layer_fp32_view view = {
                .data = scratch->hidden,
                .rank = 1U,
                .shape = { cfg->hidden_size },
                .element_strides = { 1U },
                .logical_element_count = cfg->hidden_size,
                .physical_element_count = cfg->hidden_size
            };

            lis_llama_observe_intra_layer(
                runtime, LIS_INTRA_LAYER_STAGE_LAYER_INPUT,
                checkpoint_step, layer, position, &view);
        }
        if (emit_legacy_checkpoints && layer == 0U) {
            const size_t h_dims[1] = { cfg->hidden_size };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.0.input", h_dims, 1,
                                      scratch->hidden, cfg->hidden_size,
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 1U) {
            const size_t h_dims[1] = { cfg->hidden_size };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.1.input", h_dims, 1,
                                      scratch->hidden, cfg->hidden_size,
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 8U) {
            const size_t h_dims[1] = { cfg->hidden_size };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.8.input", h_dims, 1,
                                      scratch->hidden, cfg->hidden_size,
            runtime);
        }
        status = lis_layer_tensor(model, layer, "attention_norm.weight",
                                  norm_dims, 1, &tensor);
        if (status != LIS_STATUS_OK) return status;
        lis_rms_norm(scratch->hidden, tensor, cfg->hidden_size, scratch->norm,
                     runtime->pool);
        if (capture_intra_layer && layer == intra_record->target_layer) {
            const lis_intra_layer_fp32_view view = {
                .data = scratch->norm,
                .rank = 1U,
                .shape = { cfg->hidden_size },
                .element_strides = { 1U },
                .logical_element_count = cfg->hidden_size,
                .physical_element_count = cfg->hidden_size
            };

            lis_llama_observe_intra_layer(
                runtime, LIS_INTRA_LAYER_STAGE_ATTENTION_NORM_OUTPUT,
                checkpoint_step, layer, position, &view);
        }
        if (emit_legacy_checkpoints && layer == 0U) {
            const size_t h_dims[1] = { cfg->hidden_size };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.0.input_layernorm_out", h_dims,
                                      1, scratch->norm, cfg->hidden_size,
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 1U) {
            const size_t h_dims[1] = { cfg->hidden_size };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.1.input_layernorm_out", h_dims,
                                      1, scratch->norm, cfg->hidden_size,
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 8U) {
            const size_t h_dims[1] = { cfg->hidden_size };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.8.input_layernorm_out", h_dims,
                                      1, scratch->norm, cfg->hidden_size,
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 7U) {
            const size_t h_dims[1] = { cfg->hidden_size };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.7.input_layernorm_out", h_dims,
                                      1, scratch->norm, cfg->hidden_size,
            runtime);
        }

        status = lis_layer_tensor(model, layer, "q_proj.weight", q_dims, 2,
                                  &tensor);
        if (status != LIS_STATUS_OK) return status;
        lis_matvec(tensor, scratch->norm, q_dims[0], q_dims[1], scratch->q,
                   runtime->pool);
        status = lis_layer_tensor(model, layer, "k_proj.weight", kv_dims, 2,
                                  &tensor);
        if (status != LIS_STATUS_OK) return status;
        if (emit_legacy_checkpoints && layer == 7U) {
            lis_checkpoint_weight_diagnostic(checkpoint_step, checkpoint_phase,
                                             "layer.7.k_proj_weight", tensor,
                                             kv_dims[0] * kv_dims[1],
            runtime);
        }
        lis_matvec(tensor, scratch->norm, kv_dims[0], kv_dims[1], scratch->k,
                   runtime->pool);
        status = lis_layer_tensor(model, layer, "v_proj.weight", kv_dims, 2,
                                  &tensor);
        if (status != LIS_STATUS_OK) return status;
        if (emit_legacy_checkpoints && layer == 7U) {
            lis_checkpoint_weight_diagnostic(checkpoint_step, checkpoint_phase,
                                             "layer.7.v_proj_weight", tensor,
                                             kv_dims[0] * kv_dims[1],
            runtime);
        }
        lis_matvec(tensor, scratch->norm, kv_dims[0], kv_dims[1], scratch->v,
                   runtime->pool);
        if (capture_intra_layer && layer == intra_record->target_layer) {
            const lis_intra_layer_fp32_view q_view = {
                .data = scratch->q,
                .rank = 2U,
                .shape = { cfg->attention_head_count, cfg->head_dim },
                .element_strides = { cfg->head_dim, 1U },
                .logical_element_count = q_dims[0],
                .physical_element_count = q_dims[0]
            };
            const lis_intra_layer_fp32_view k_view = {
                .data = scratch->k,
                .rank = 2U,
                .shape = { cfg->kv_head_count, cfg->head_dim },
                .element_strides = { cfg->head_dim, 1U },
                .logical_element_count = kv_dims[0],
                .physical_element_count = kv_dims[0]
            };
            const lis_intra_layer_fp32_view v_view = {
                .data = scratch->v,
                .rank = 2U,
                .shape = { cfg->kv_head_count, cfg->head_dim },
                .element_strides = { cfg->head_dim, 1U },
                .logical_element_count = kv_dims[0],
                .physical_element_count = kv_dims[0]
            };

            lis_llama_observe_intra_layer(
                runtime, LIS_INTRA_LAYER_STAGE_QUERY_PROJECTION_OUTPUT,
                checkpoint_step, layer, position, &q_view);
            lis_llama_observe_intra_layer(
                runtime, LIS_INTRA_LAYER_STAGE_KEY_PROJECTION_OUTPUT,
                checkpoint_step, layer, position, &k_view);
            lis_llama_observe_intra_layer(
                runtime, LIS_INTRA_LAYER_STAGE_VALUE_PROJECTION_OUTPUT,
                checkpoint_step, layer, position, &v_view);
        }
        if (emit_legacy_checkpoints && layer == 0U) {
            const size_t q_proj_dims[1] = { q_dims[0] };
            const size_t kv_proj_dims[1] = { kv_dims[0] };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.0.q_proj_out", q_proj_dims, 1,
                                      scratch->q, q_dims[0],
            runtime);
            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.0.k_proj_out", kv_proj_dims, 1,
                                      scratch->k, kv_dims[0],
            runtime);
            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.0.v_proj_out", kv_proj_dims, 1,
                                      scratch->v, kv_dims[0],
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 1U) {
            const size_t q_proj_dims[1] = { q_dims[0] };
            const size_t kv_proj_dims[1] = { kv_dims[0] };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.1.q_proj_out", q_proj_dims, 1,
                                      scratch->q, q_dims[0],
            runtime);
            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.1.k_proj_out", kv_proj_dims, 1,
                                      scratch->k, kv_dims[0],
            runtime);
            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.1.v_proj_out", kv_proj_dims, 1,
                                      scratch->v, kv_dims[0],
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 7U) {
            const size_t q_proj_dims[1] = { q_dims[0] };
            const size_t kv_proj_dims[1] = { kv_dims[0] };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.7.q_proj_out", q_proj_dims, 1,
                                      scratch->q, q_dims[0],
            runtime);
            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.7.k_proj_out", kv_proj_dims, 1,
                                      scratch->k, kv_dims[0],
            runtime);
            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.7.v_proj_out", kv_proj_dims, 1,
                                      scratch->v, kv_dims[0],
            runtime);
            lis_checkpoint_per_head_diagnostic(checkpoint_step,
                                               checkpoint_phase,
                                               "layer.7.k_proj_out",
                                               scratch->k,
                                               cfg->kv_head_count,
                                               cfg->head_dim,
            runtime);
            lis_checkpoint_per_head_diagnostic(checkpoint_step,
                                               checkpoint_phase,
                                               "layer.7.v_proj_out",
                                               scratch->v,
                                               cfg->kv_head_count,
                                               cfg->head_dim,
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 8U) {
            const size_t q_proj_dims[1] = { q_dims[0] };
            const size_t kv_proj_dims[1] = { kv_dims[0] };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.8.q_proj_out", q_proj_dims, 1,
                                      scratch->q, q_dims[0],
            runtime);
            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.8.k_proj_out", kv_proj_dims, 1,
                                      scratch->k, kv_dims[0],
            runtime);
            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.8.v_proj_out", kv_proj_dims, 1,
                                      scratch->v, kv_dims[0],
            runtime);
        }

        lis_apply_rope(scratch->q, cfg->attention_head_count, cfg->head_dim,
                       position, cfg->rope_theta, runtime->pool);
        lis_apply_rope(scratch->k, cfg->kv_head_count, cfg->head_dim,
                       position, cfg->rope_theta, runtime->pool);
        if (capture_intra_layer && layer == intra_record->target_layer) {
            const lis_intra_layer_fp32_view q_view = {
                .data = scratch->q,
                .rank = 2U,
                .shape = { cfg->attention_head_count, cfg->head_dim },
                .element_strides = { cfg->head_dim, 1U },
                .logical_element_count = q_dims[0],
                .physical_element_count = q_dims[0]
            };
            const lis_intra_layer_fp32_view k_view = {
                .data = scratch->k,
                .rank = 2U,
                .shape = { cfg->kv_head_count, cfg->head_dim },
                .element_strides = { cfg->head_dim, 1U },
                .logical_element_count = kv_dims[0],
                .physical_element_count = kv_dims[0]
            };

            lis_llama_observe_intra_layer(
                runtime, LIS_INTRA_LAYER_STAGE_ROPE_QUERY_OUTPUT,
                checkpoint_step, layer, position, &q_view);
            lis_llama_observe_intra_layer(
                runtime, LIS_INTRA_LAYER_STAGE_ROPE_KEY_OUTPUT,
                checkpoint_step, layer, position, &k_view);
        }
        if (emit_legacy_checkpoints && layer == 0U) {
            const size_t q_proj_dims[1] = { q_dims[0] };
            const size_t kv_proj_dims[1] = { kv_dims[0] };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.0.q_after_rope", q_proj_dims, 1,
                                      scratch->q, q_dims[0],
            runtime);
            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.0.k_after_rope", kv_proj_dims, 1,
                                      scratch->k, kv_dims[0],
            runtime);
            lis_checkpoint_per_head_diagnostic(checkpoint_step,
                                               checkpoint_phase,
                                               "layer.0.q_heads",
                                               scratch->q,
                                               cfg->attention_head_count,
                                               cfg->head_dim,
            runtime);
            lis_checkpoint_per_head_diagnostic(checkpoint_step,
                                               checkpoint_phase,
                                               "layer.0.k_heads",
                                               scratch->k,
                                               cfg->kv_head_count,
                                               cfg->head_dim,
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 1U) {
            const size_t q_proj_dims[1] = { q_dims[0] };
            const size_t kv_proj_dims[1] = { kv_dims[0] };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.1.q_after_rope", q_proj_dims, 1,
                                      scratch->q, q_dims[0],
            runtime);
            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.1.k_after_rope", kv_proj_dims, 1,
                                      scratch->k, kv_dims[0],
            runtime);
            lis_checkpoint_per_head_diagnostic(checkpoint_step,
                                               checkpoint_phase,
                                               "layer.1.q_heads",
                                               scratch->q,
                                               cfg->attention_head_count,
                                               cfg->head_dim,
            runtime);
            lis_checkpoint_per_head_diagnostic(checkpoint_step,
                                               checkpoint_phase,
                                               "layer.1.k_heads",
                                               scratch->k,
                                               cfg->kv_head_count,
                                               cfg->head_dim,
            runtime);
            {
                const float scale = 1.0f / sqrtf((float)cfg->head_dim);

                fprintf(stderr,
                        "lis: layer-checkpoint step=%zu phase=%s "
                        "name=layer.1.attn_scale value=%.6g\n",
                        checkpoint_step, checkpoint_phase, (double)scale);
            }
        }
        if (emit_legacy_checkpoints && layer == 7U) {
            const size_t q_proj_dims[1] = { q_dims[0] };
            const size_t kv_proj_dims[1] = { kv_dims[0] };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.7.q_after_rope", q_proj_dims, 1,
                                      scratch->q, q_dims[0],
            runtime);
            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.7.k_after_rope", kv_proj_dims, 1,
                                      scratch->k, kv_dims[0],
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 8U) {
            const size_t q_proj_dims[1] = { q_dims[0] };
            const size_t kv_proj_dims[1] = { kv_dims[0] };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.8.q_after_rope", q_proj_dims, 1,
                                      scratch->q, q_dims[0],
            runtime);
            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.8.k_after_rope", kv_proj_dims, 1,
                                      scratch->k, kv_dims[0],
            runtime);
        }
        /* Debug-only: snapshot layer-1 kv-head-0 K in fp32 before
           the fp16 KV-cache round-trip, so the diagnostic block can
           reconstruct scores without cache precision loss. */
        if (emit_legacy_checkpoints && layer == 1U) {
            const size_t needed =
                (position + 1U) * cfg->head_dim;

            if (s_dbg_l1_k_fp32 == NULL ||
                s_dbg_l1_k_fp32_cap < needed) {
                float *tmp = (float *)realloc(
                    s_dbg_l1_k_fp32, needed * sizeof(float));

                if (tmp != NULL) {
                    s_dbg_l1_k_fp32 = tmp;
                    s_dbg_l1_k_fp32_cap = needed;
                }
            }
            if (s_dbg_l1_k_fp32 != NULL &&
                s_dbg_l1_k_fp32_cap >= needed) {
                memcpy(s_dbg_l1_k_fp32 + position * cfg->head_dim,
                       scratch->k,
                       cfg->head_dim * sizeof(float));
            }
        }
        status = lis_store_kv(runtime, layer, position, scratch->k,
                              scratch->v);
        if (status != LIS_STATUS_OK) return status;
        status = lis_attention(runtime, layer, position, scratch->q,
                               scratch->scores, scratch->attn_probs,
                               cfg->context.configured_max_tokens,
                               scratch->attn, runtime->pool);
        if (status != LIS_STATUS_OK) return status;
        if (capture_intra_layer && layer == intra_record->target_layer) {
            const size_t used_positions = position + 1U;

            if (cfg->attention_head_count > SIZE_MAX / used_positions ||
                cfg->attention_head_count >
                    SIZE_MAX / cfg->context.configured_max_tokens) {
                lis_intra_layer_record_invalidate(intra_record);
            } else {
                const size_t logical_attention_count =
                    cfg->attention_head_count * used_positions;
                const size_t physical_attention_count =
                    cfg->attention_head_count *
                    cfg->context.configured_max_tokens;
                const lis_intra_layer_fp32_view scores_view = {
                    .data = scratch->scores,
                    .rank = 2U,
                    .shape = { cfg->attention_head_count, used_positions },
                    .element_strides = {
                        cfg->context.configured_max_tokens, 1U
                    },
                    .logical_element_count = logical_attention_count,
                    .physical_element_count = physical_attention_count
                };
                const lis_intra_layer_fp32_view probabilities_view = {
                    .data = scratch->attn_probs,
                    .rank = 2U,
                    .shape = { cfg->attention_head_count, used_positions },
                    .element_strides = {
                        cfg->context.configured_max_tokens, 1U
                    },
                    .logical_element_count = logical_attention_count,
                    .physical_element_count = physical_attention_count
                };
                const lis_intra_layer_fp32_view context_view = {
                    .data = scratch->attn,
                    .rank = 2U,
                    .shape = {
                        cfg->attention_head_count, cfg->head_dim
                    },
                    .element_strides = { cfg->head_dim, 1U },
                    .logical_element_count = q_dims[0],
                    .physical_element_count = q_dims[0]
                };

                lis_llama_observe_intra_layer(
                    runtime, LIS_INTRA_LAYER_STAGE_ATTENTION_SCORES,
                    checkpoint_step, layer, position, &scores_view);
                lis_llama_observe_intra_layer(
                    runtime, LIS_INTRA_LAYER_STAGE_ATTENTION_PROBABILITIES,
                    checkpoint_step, layer, position, &probabilities_view);
                lis_llama_observe_intra_layer(
                    runtime, LIS_INTRA_LAYER_STAGE_ATTENTION_CONTEXT,
                    checkpoint_step, layer, position, &context_view);
            }
        }
        if (emit_legacy_checkpoints && layer == 1U) {
            const size_t used_positions = position + 1U;
            const size_t h_dims[1] = {
                cfg->attention_head_count * cfg->head_dim
            };
            const size_t expanded_k_size =
                cfg->attention_head_count * cfg->head_dim *
                used_positions;
            float *expanded_k = NULL;

            expanded_k = (float *)malloc(expanded_k_size * sizeof(float));
            if (expanded_k != NULL) {
                size_t qh;
                size_t p;
                size_t d;
                int read_ok = 1;

                for (qh = 0; qh < cfg->attention_head_count && read_ok;
                     ++qh) {
                    const size_t kvh = qh * cfg->kv_head_count /
                                       cfg->attention_head_count;

                    for (p = 0; p <= position && read_ok; ++p) {
                        for (d = 0; d < cfg->head_dim; ++d) {
                            void *key_ptr = NULL;
                            lis_status ks =
                                lis_kv_cache_key_ptr(
                                    &runtime->kv_cache, layer, 0, p,
                                    kvh, d, &key_ptr);

                            if (ks != LIS_STATUS_OK) {
                                read_ok = 0;
                                break;
                            }
                            expanded_k[(qh * used_positions + p) *
                                       cfg->head_dim + d] =
                                lis_scalar_read(
                                    runtime->kv_cache.layout.dtype,
                                    key_ptr, 0);
                        }
                    }
                }
                if (read_ok) {
                    const size_t kr_dims[1] = { expanded_k_size };

                    lis_checkpoint_diagnostic(checkpoint_step,
                                              checkpoint_phase,
                                              "layer.1.k_repeated",
                                              kr_dims, 1, expanded_k,
                                              expanded_k_size,
            runtime);
                }
                free(expanded_k);
            }
            lis_checkpoint_strided_diagnostic(
                checkpoint_step, checkpoint_phase, "layer.1.attn_scores",
                scratch->scores, cfg->attention_head_count,
                cfg->context.configured_max_tokens, used_positions,
            runtime);
            {
                const size_t head0_dims[1] = { used_positions };

                lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                          "layer.1.attn_scores_head0",
                                          head0_dims, 1, scratch->scores,
                                          used_positions,
            runtime);
            }
            {
                /* Reconstruct head-0 scores offline from Q and KV cache */
                const float scale = 1.0f / sqrtf((float)cfg->head_dim);
                float *recon = (float *)malloc(
                    used_positions * sizeof(float));

                if (recon != NULL) {
                    size_t p;
                    int read_ok = 1;

                    for (p = 0; p <= position && read_ok; ++p) {
                        float dot = 0.0f;
                        size_t d;

                        for (d = 0; d < cfg->head_dim; ++d) {
                            void *key_ptr = NULL;
                            lis_status ks =
                                lis_kv_cache_key_ptr(
                                    &runtime->kv_cache, layer, 0, p,
                                    0, d, &key_ptr);

                            if (ks != LIS_STATUS_OK) {
                                read_ok = 0;
                                break;
                            }
                            dot += scratch->q[d] *
                                   lis_scalar_read(
                                       runtime->kv_cache.layout.dtype,
                                       key_ptr, 0);
                        }
                        recon[p] = dot * scale;
                    }
                    if (read_ok) {
                        const size_t r_dims[1] = { used_positions };

                        lis_checkpoint_diagnostic(
                            checkpoint_step, checkpoint_phase,
                            "layer.1.attn_scores_head0_recon",
                            r_dims, 1, recon, used_positions,
            runtime);
                    }
                    free(recon);
                }
            }
            {
                /* Reconstruct head-0 scores using the fp32 K
                   snapshot buffer (bypasses fp16 KV-cache loss). */
                const float scale = 1.0f / sqrtf((float)cfg->head_dim);
                const size_t needed_fp32 =
                    used_positions * cfg->head_dim;

                if (s_dbg_l1_k_fp32 != NULL &&
                    s_dbg_l1_k_fp32_cap >= needed_fp32) {
                    float *recon_fp32 = (float *)malloc(
                        used_positions * sizeof(float));

                    if (recon_fp32 != NULL) {
                        size_t p;

                        for (p = 0; p <= position; ++p) {
                            float dot = 0.0f;
                            size_t d;

                            for (d = 0; d < cfg->head_dim; ++d) {
                                dot += scratch->q[d] *
                                       s_dbg_l1_k_fp32[
                                           p * cfg->head_dim + d];
                            }
                            recon_fp32[p] = dot * scale;
                        }
                        {
                            const size_t r_dims[1] = {
                                used_positions
                            };

                            lis_checkpoint_diagnostic(
                                checkpoint_step, checkpoint_phase,
                                "layer.1.attn_scores_head0_fp32k",
                                r_dims, 1, recon_fp32,
                                used_positions,
            runtime);
                        }
                        free(recon_fp32);
                    }
                }
            }
            {
                /* Single-position comparison: last position only,
                   fp32 K (from scratch) vs reported (from cache). */
                const float scale = 1.0f / sqrtf((float)cfg->head_dim);
                float dot_fp32 = 0.0f;
                size_t d;

                for (d = 0; d < cfg->head_dim; ++d) {
                    dot_fp32 += scratch->q[d] * scratch->k[d];
                }
                {
                    const float val_fp32 = dot_fp32 * scale;
                    const float val_reported =
                        scratch->scores[position];
                    const size_t one_dims[1] = { 1 };

                    lis_checkpoint_diagnostic(
                        checkpoint_step, checkpoint_phase,
                        "layer.1.attn_scores_head0_pos_last_fp32",
                        one_dims, 1, &val_fp32, 1,
            runtime);
                    lis_checkpoint_diagnostic(
                        checkpoint_step, checkpoint_phase,
                        "layer.1.attn_scores_head0_pos_last_reported",
                        one_dims, 1, &val_reported, 1,
            runtime);
                }
            }
            lis_checkpoint_strided_diagnostic(
                checkpoint_step, checkpoint_phase, "layer.1.attn_probs",
                scratch->attn_probs, cfg->attention_head_count,
                cfg->context.configured_max_tokens, used_positions,
            runtime);
            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.1.attn_context", h_dims, 1,
                                      scratch->attn, h_dims[0],
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 7U) {
            const size_t used_positions = position + 1U;
            const size_t h_dims[1] = {
                cfg->attention_head_count * cfg->head_dim
            };

            lis_checkpoint_strided_diagnostic(
                checkpoint_step, checkpoint_phase, "layer.7.attn_scores",
                scratch->scores, cfg->attention_head_count,
                cfg->context.configured_max_tokens, used_positions,
            runtime);
            lis_checkpoint_strided_diagnostic(
                checkpoint_step, checkpoint_phase, "layer.7.attn_probs",
                scratch->attn_probs, cfg->attention_head_count,
                cfg->context.configured_max_tokens, used_positions,
            runtime);
            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.7.attn_context", h_dims, 1,
                                      scratch->attn, h_dims[0],
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 8U) {
            const size_t used_positions = position + 1U;
            const size_t h_dims[1] = {
                cfg->attention_head_count * cfg->head_dim
            };

            lis_checkpoint_strided_diagnostic(
                checkpoint_step, checkpoint_phase, "layer.8.attn_scores",
                scratch->scores, cfg->attention_head_count,
                cfg->context.configured_max_tokens, used_positions,
            runtime);
            lis_checkpoint_strided_diagnostic(
                checkpoint_step, checkpoint_phase, "layer.8.attn_probs",
                scratch->attn_probs, cfg->attention_head_count,
                cfg->context.configured_max_tokens, used_positions,
            runtime);
            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.8.attn_context", h_dims, 1,
                                      scratch->attn, h_dims[0],
            runtime);
        }
        status = lis_layer_tensor(model, layer, "o_proj.weight", o_dims, 2,
                                  &tensor);
        if (status != LIS_STATUS_OK) return status;
        lis_matvec(tensor, scratch->attn, o_dims[0], o_dims[1],
                   scratch->attn_out, runtime->pool);
        if (capture_intra_layer && layer == intra_record->target_layer) {
            const lis_intra_layer_fp32_view view = {
                .data = scratch->attn_out,
                .rank = 1U,
                .shape = { cfg->hidden_size },
                .element_strides = { 1U },
                .logical_element_count = cfg->hidden_size,
                .physical_element_count = cfg->hidden_size
            };

            lis_llama_observe_intra_layer(
                runtime,
                LIS_INTRA_LAYER_STAGE_ATTENTION_OUTPUT_PROJECTION,
                checkpoint_step, layer, position, &view);
        }
        if (emit_legacy_checkpoints && layer == 0U) {
            const size_t h_dims[1] = { cfg->hidden_size };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.0.attn_out", h_dims, 1,
                                      scratch->attn_out, cfg->hidden_size,
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 1U) {
            const size_t h_dims[1] = { cfg->hidden_size };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.1.attn_out", h_dims, 1,
                                      scratch->attn_out, cfg->hidden_size,
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 7U) {
            const size_t h_dims[1] = { cfg->hidden_size };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.7.attn_out", h_dims, 1,
                                      scratch->attn_out, cfg->hidden_size,
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 8U) {
            const size_t h_dims[1] = { cfg->hidden_size };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.8.attn_out", h_dims, 1,
                                      scratch->attn_out, cfg->hidden_size,
            runtime);
        }
        lis_residual_add(scratch->hidden, scratch->attn_out, cfg->hidden_size);
        if (capture_intra_layer && layer == intra_record->target_layer) {
            const lis_intra_layer_fp32_view view = {
                .data = scratch->hidden,
                .rank = 1U,
                .shape = { cfg->hidden_size },
                .element_strides = { 1U },
                .logical_element_count = cfg->hidden_size,
                .physical_element_count = cfg->hidden_size
            };

            lis_llama_observe_intra_layer(
                runtime, LIS_INTRA_LAYER_STAGE_POST_ATTENTION_RESIDUAL,
                checkpoint_step, layer, position, &view);
        }
        if (emit_legacy_checkpoints && layer == 0U) {
            const size_t h_dims[1] = { cfg->hidden_size };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.0.post_attn_residual", h_dims,
                                      1, scratch->hidden, cfg->hidden_size,
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 1U) {
            const size_t h_dims[1] = { cfg->hidden_size };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.1.post_attn_residual", h_dims,
                                      1, scratch->hidden, cfg->hidden_size,
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 7U) {
            const size_t h_dims[1] = { cfg->hidden_size };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.7.post_attn_residual", h_dims,
                                      1, scratch->hidden, cfg->hidden_size,
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 8U) {
            const size_t h_dims[1] = { cfg->hidden_size };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.8.post_attn_residual", h_dims,
                                      1, scratch->hidden, cfg->hidden_size,
            runtime);
        }

        status = lis_layer_tensor(model, layer, "mlp_norm.weight", norm_dims,
                                  1, &tensor);
        if (status != LIS_STATUS_OK) return status;
        lis_rms_norm(scratch->hidden, tensor, cfg->hidden_size, scratch->norm,
                     runtime->pool);
        if (capture_intra_layer && layer == intra_record->target_layer) {
            const lis_intra_layer_fp32_view view = {
                .data = scratch->norm,
                .rank = 1U,
                .shape = { cfg->hidden_size },
                .element_strides = { 1U },
                .logical_element_count = cfg->hidden_size,
                .physical_element_count = cfg->hidden_size
            };

            lis_llama_observe_intra_layer(
                runtime, LIS_INTRA_LAYER_STAGE_MLP_NORM_OUTPUT,
                checkpoint_step, layer, position, &view);
        }

        status = lis_layer_tensor(model, layer, "gate_proj.weight",
                                  mlp_in_dims, 2, &tensor);
        if (status != LIS_STATUS_OK) return status;
        lis_matvec(tensor, scratch->norm, mlp_in_dims[0], mlp_in_dims[1],
                   scratch->gate, runtime->pool);
        status = lis_layer_tensor(model, layer, "up_proj.weight", mlp_in_dims,
                                  2, &tensor);
        if (status != LIS_STATUS_OK) return status;
        lis_matvec(tensor, scratch->norm, mlp_in_dims[0], mlp_in_dims[1],
                   scratch->up, runtime->pool);
        if (capture_intra_layer && layer == intra_record->target_layer) {
            const lis_intra_layer_fp32_view gate_view = {
                .data = scratch->gate,
                .rank = 1U,
                .shape = { cfg->intermediate_size },
                .element_strides = { 1U },
                .logical_element_count = cfg->intermediate_size,
                .physical_element_count = cfg->intermediate_size
            };
            const lis_intra_layer_fp32_view up_view = {
                .data = scratch->up,
                .rank = 1U,
                .shape = { cfg->intermediate_size },
                .element_strides = { 1U },
                .logical_element_count = cfg->intermediate_size,
                .physical_element_count = cfg->intermediate_size
            };

            lis_llama_observe_intra_layer(
                runtime, LIS_INTRA_LAYER_STAGE_MLP_GATE_PROJECTION,
                checkpoint_step, layer, position, &gate_view);
            lis_llama_observe_intra_layer(
                runtime, LIS_INTRA_LAYER_STAGE_MLP_UP_PROJECTION,
                checkpoint_step, layer, position, &up_view);
        }
        lis_swiglu(scratch->gate, scratch->up, cfg->intermediate_size,
                   scratch->mlp, runtime->pool);
        if (capture_intra_layer && layer == intra_record->target_layer) {
            const lis_intra_layer_fp32_view view = {
                .data = scratch->mlp,
                .rank = 1U,
                .shape = { cfg->intermediate_size },
                .element_strides = { 1U },
                .logical_element_count = cfg->intermediate_size,
                .physical_element_count = cfg->intermediate_size
            };

            lis_llama_observe_intra_layer(
                runtime, LIS_INTRA_LAYER_STAGE_MLP_GATED_ACTIVATION,
                checkpoint_step, layer, position, &view);
        }
        status = lis_layer_tensor(model, layer, "down_proj.weight",
                                  mlp_down_dims, 2, &tensor);
        if (status != LIS_STATUS_OK) return status;
        lis_matvec(tensor, scratch->mlp, mlp_down_dims[0], mlp_down_dims[1],
                   scratch->mlp_out, runtime->pool);
        if (capture_intra_layer && layer == intra_record->target_layer) {
            const lis_intra_layer_fp32_view view = {
                .data = scratch->mlp_out,
                .rank = 1U,
                .shape = { cfg->hidden_size },
                .element_strides = { 1U },
                .logical_element_count = cfg->hidden_size,
                .physical_element_count = cfg->hidden_size
            };

            lis_llama_observe_intra_layer(
                runtime, LIS_INTRA_LAYER_STAGE_MLP_DOWN_PROJECTION,
                checkpoint_step, layer, position, &view);
        }
        if (emit_legacy_checkpoints && layer == 0U) {
            const size_t h_dims[1] = { cfg->hidden_size };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.0.mlp_out", h_dims, 1,
                                      scratch->mlp_out, cfg->hidden_size,
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 1U) {
            const size_t h_dims[1] = { cfg->hidden_size };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.1.mlp_out", h_dims, 1,
                                      scratch->mlp_out, cfg->hidden_size,
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 7U) {
            const size_t h_dims[1] = { cfg->hidden_size };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.7.mlp_out", h_dims, 1,
                                      scratch->mlp_out, cfg->hidden_size,
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 8U) {
            const size_t h_dims[1] = { cfg->hidden_size };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.8.mlp_out", h_dims, 1,
                                      scratch->mlp_out, cfg->hidden_size,
            runtime);
        }
        lis_residual_add(scratch->hidden, scratch->mlp_out, cfg->hidden_size);
        if (emit_legacy_checkpoints && layer == 0U) {
            const size_t h_dims[1] = { cfg->hidden_size };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.0.post_mlp_residual", h_dims,
                                      1, scratch->hidden, cfg->hidden_size,
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 1U) {
            const size_t h_dims[1] = { cfg->hidden_size };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.1.post_mlp_residual", h_dims,
                                      1, scratch->hidden, cfg->hidden_size,
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 7U) {
            const size_t h_dims[1] = { cfg->hidden_size };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.7.post_mlp_residual", h_dims,
                                      1, scratch->hidden, cfg->hidden_size,
            runtime);
        }
        if (emit_legacy_checkpoints && layer == 8U) {
            const size_t h_dims[1] = { cfg->hidden_size };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                      "layer.8.post_mlp_residual", h_dims,
                                      1, scratch->hidden, cfg->hidden_size,
            runtime);
        }

        if (emit_checkpoints &&
            lis_layer_trace_layout_selects_layer(layer, cfg->layer_count)) {
            char name[64];
            const size_t h_dims[1] = { cfg->hidden_size };

            if (snprintf(name, sizeof(name), "layer.%zu.output", layer) < (int)sizeof(name)) {
                if (layer == 7U && strcmp(name, "layer.7.output") != 0) {
                    return LIS_STATUS_INVALID_ARGUMENT;
                }
                lis_checkpoint_layer_output_diagnostic(
                    checkpoint_step, checkpoint_phase, name, h_dims, 1,
                    scratch->hidden, cfg->hidden_size, runtime, layer);
            }
        }
    }

    {
        const size_t norm_dims[1] = { cfg->hidden_size };

        status = lis_expect_tensor(model, "lis.output_norm.weight", norm_dims,
                                   1, cfg->weight_dtype, &tensor);
        if (status != LIS_STATUS_OK) return status;
        lis_rms_norm(scratch->hidden, tensor, cfg->hidden_size, scratch->norm,
                     runtime->pool);
    }

    if (emit_legacy_checkpoints) {
        const size_t h_dims[1] = { cfg->hidden_size };

        lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                  "final_norm", h_dims, 1, scratch->norm,
                                  cfg->hidden_size,
            runtime);
    }

    dims[0] = cfg->vocab_size;
    dims[1] = cfg->hidden_size;
    status = lis_expect_tensor(model,
                               cfg->tie_word_embeddings ?
                                   "lis.token_embeddings.weight" :
                                   "lis.lm_head.weight",
                               dims, 2, cfg->weight_dtype, &tensor);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    lis_matvec(tensor, scratch->norm, cfg->vocab_size, cfg->hidden_size,
               out_logits, runtime->pool);

    if (emit_legacy_checkpoints) {
        const size_t l_dims[1] = { cfg->vocab_size };

        lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                  "logits", l_dims, 1, out_logits,
                                  cfg->vocab_size,
            runtime);
    }

    return LIS_STATUS_OK;
}

lis_status lis_runtime_llama_prefill(lis_runtime_context *runtime,
                                     const lis_loaded_model *model,
                                     const size_t *tokens,
                                     const size_t *sequence_lengths,
                                     size_t sequence_count,
                                     float *out_logits,
                                     size_t logits_len)
{
    lis_llama_scratch scratch;
    size_t index;
    lis_status status;

    if (runtime == NULL || model == NULL || tokens == NULL ||
        out_logits == NULL || runtime->batch.batch_size != 1 ||
        sequence_count != 1 || runtime->phase != LIS_RUNTIME_PHASE_READY) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    status = lis_static_batch_validate_lengths(&runtime->batch,
                                               sequence_lengths,
                                               sequence_count);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_llama_scratch_init(&scratch, &runtime->metadata.config);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    for (index = 0; index < sequence_lengths[0]; ++index) {
        const char *checkpoint_phase =
            index + 1U == sequence_lengths[0] ? "prefill" : NULL;

        status = lis_llama_forward_token(runtime, model, tokens[index], index,
                                         0, checkpoint_phase,
                                         &scratch, out_logits, logits_len);
        if (status != LIS_STATUS_OK) {
            break;
        }
    }
    if (status == LIS_STATUS_OK) {
        runtime->batch.positions[0] = sequence_lengths[0];
        runtime->phase = LIS_RUNTIME_PHASE_PREFILLED;
    }
    lis_llama_scratch_destroy(&scratch);
    return status;
}

lis_status lis_runtime_llama_decode(lis_runtime_context *runtime,
                                    const lis_loaded_model *model,
                                    size_t token_id,
                                    float *out_logits,
                                    size_t logits_len)
{
    lis_llama_scratch scratch;
    lis_status status;
    size_t position;
    size_t checkpoint_step;

    if (runtime == NULL || model == NULL || out_logits == NULL ||
        runtime->batch.batch_size != 1 ||
        (runtime->phase != LIS_RUNTIME_PHASE_PREFILLED &&
         runtime->phase != LIS_RUNTIME_PHASE_DECODING)) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    position = runtime->batch.positions[0];
    if (position == 0 || position >= runtime->batch.max_tokens) {
        return LIS_STATUS_LIMIT_EXCEEDED;
    }
    if (runtime->decode_step_count == SIZE_MAX) {
        return LIS_STATUS_OVERFLOW;
    }
    checkpoint_step = runtime->decode_step_count + 1U;
    status = lis_llama_scratch_init(&scratch, &runtime->metadata.config);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_llama_forward_token(runtime, model, token_id, position,
                                     checkpoint_step, "decode",
                                     &scratch, out_logits, logits_len);
    if (status == LIS_STATUS_OK) {
        ++runtime->batch.positions[0];
        ++runtime->decode_step_count;
        runtime->phase = LIS_RUNTIME_PHASE_DECODING;
    }
    lis_llama_scratch_destroy(&scratch);
    return status;
}
