#include "lis/runtime.h"
#include "lis/cpu_ops.h"
#include "lis/dtype.h"
#include "lis/layer_trace.h"
#include "lis/thread_pool.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
} lis_qwen3_scratch;

static void lis_qwen3_scratch_destroy(lis_qwen3_scratch *scratch);

static float lis_scalar_read(lis_dtype dtype, const void *data, size_t index)
{
    return lis_dtype_scalar_read_f32(dtype, data, index);
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

static void lis_checkpoint_diagnostic(size_t step, const char *phase,
                                       const char *name, const size_t *dims,
                                       size_t rank, const float *data,
                                       size_t count,
                                       lis_runtime_context *runtime)
{
    size_t i;
    lis_layer_trace_step lts = {0};
    int truncated = 0;
    lis_layer_trace_record *record =
        runtime != NULL ? runtime->layer_trace_record : NULL;

    if (count == 0 || data == NULL || dims == NULL || phase == NULL) {
        return;
    }

    lts.step = step;
    if (snprintf(lts.phase, sizeof(lts.phase), "%s", phase) >= (int)sizeof(lts.phase))
        truncated = 1;
    if (snprintf(lts.name, sizeof(lts.name), "%s", name) >= (int)sizeof(lts.name))
        truncated = 1;
    lts.rank = rank < LIS_LAYER_TRACE_MAX_RANK ? rank : LIS_LAYER_TRACE_MAX_RANK;
    for (i = 0; i < lts.rank; ++i) {
        lts.shape[i] = dims[i];
    }
    lts.min = data[0];
    lts.max = data[0];
    for (i = 0; i < count; ++i) {
        float v = data[i];
        if (v < lts.min) lts.min = v;
        if (v > lts.max) lts.max = v;
        lts.mean += v;
        lts.l2 += v * v;
    }
    lts.mean = lts.mean / (float)count;
    lts.l2 = sqrtf(lts.l2);

    fprintf(stderr, "lis: layer-checkpoint step=%zu phase=%s name=%s shape=[",
            step, phase, name);
    for (i = 0; i < rank; ++i) {
        fprintf(stderr, "%s%zu", i == 0 ? "" : ",", dims[i]);
    }
    fprintf(stderr, "] min=%.6g max=%.6g mean=%.6g l2=%.6g nan=0 inf=0\n",
            lts.min, lts.max, lts.mean, lts.l2);

    if (record != NULL && !record->append_failed) {
        if (truncated) {
            record->append_failed = 1;
        } else {
            lis_layer_trace_record_append(record, &lts);
        }
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
                                    const char *name, const size_t *dims,
                                    size_t rank, lis_dtype expected_dtype,
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

static void lis_qwen3_rms_norm_eps(const float *input,
                                   const lis_loaded_tensor *weight,
                                   size_t n, float eps, float *out)
{
    float mean_square = 0.0f;
    float scale;
    size_t index;

    for (index = 0; index < n; ++index) {
        mean_square += input[index] * input[index];
    }
    mean_square /= (float)n;
    scale = 1.0f / sqrtf(mean_square + eps);
    for (index = 0; index < n; ++index) {
        out[index] = input[index] * scale *
                     lis_scalar_read(weight->view.dtype, weight->view.data,
                                     index);
    }
}

static void lis_qwen3_head_rms_norm(float *values, size_t head_count,
                                    size_t head_dim,
                                    const lis_loaded_tensor *weight,
                                    float eps)
{
    size_t head;

    for (head = 0; head < head_count; ++head) {
        float *base = values + head * head_dim;

        lis_qwen3_rms_norm_eps(base, weight, head_dim, eps, base);
    }
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

static lis_status lis_qwen3_scratch_init(lis_qwen3_scratch *scratch,
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
    scratch->scores = calloc(cfg->attention_head_count * max_tokens,
                             sizeof(*scratch->scores));
    if (scratch->hidden == NULL || scratch->norm == NULL ||
        scratch->q == NULL || scratch->k == NULL || scratch->v == NULL ||
        scratch->attn == NULL || scratch->attn_out == NULL ||
        scratch->attn_probs == NULL || scratch->gate == NULL ||
        scratch->up == NULL || scratch->mlp == NULL ||
        scratch->mlp_out == NULL || scratch->scores == NULL) {
        lis_qwen3_scratch_destroy(scratch);
        return LIS_STATUS_NO_MEMORY;
    }
    return LIS_STATUS_OK;
}

static void lis_qwen3_scratch_destroy(lis_qwen3_scratch *scratch)
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

static lis_status lis_qwen3_forward_token(lis_runtime_context *runtime,
                                          const lis_loaded_model *model,
                                          size_t token_id,
                                          size_t position,
                                          size_t checkpoint_step,
                                          const char *checkpoint_phase,
                                          lis_qwen3_scratch *scratch,
                                          float *out_logits,
                                          size_t logits_len)
{
    const lis_model_config *cfg = &runtime->metadata.config;
    const lis_loaded_tensor *tensor = NULL;
    size_t dims[2] = { 0 };
    size_t index;
    size_t layer;
    int emit_checkpoints;
    lis_status status;

    if (cfg->family != LIS_MODEL_FAMILY_QWEN3_DENSE_DECODER) {
        return LIS_STATUS_UNSUPPORTED_FORMAT;
    }
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
    if (emit_checkpoints) {
        const size_t h_dims[1] = { cfg->hidden_size };

        lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                  "qwen3.embedding", h_dims, 1,
                                  scratch->hidden, cfg->hidden_size,
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
        const size_t qk_norm_dims[1] = { cfg->head_dim };
        const size_t mlp_in_dims[2] = {
            cfg->intermediate_size,
            cfg->hidden_size
        };
        const size_t mlp_down_dims[2] = {
            cfg->hidden_size,
            cfg->intermediate_size
        };
        const size_t norm_dims[1] = { cfg->hidden_size };

        status = lis_layer_tensor(model, layer, "attention_norm.weight",
                                  norm_dims, 1, &tensor);
        if (status != LIS_STATUS_OK) return status;
        lis_qwen3_rms_norm_eps(scratch->hidden, tensor, cfg->hidden_size,
                               cfg->rms_norm_eps, scratch->norm);

        status = lis_layer_tensor(model, layer, "q_proj.weight", q_dims, 2,
                                  &tensor);
        if (status != LIS_STATUS_OK) return status;
        lis_matvec(tensor, scratch->norm, q_dims[0], q_dims[1], scratch->q,
                   runtime->pool);
        status = lis_layer_tensor(model, layer, "k_proj.weight", kv_dims, 2,
                                  &tensor);
        if (status != LIS_STATUS_OK) return status;
        lis_matvec(tensor, scratch->norm, kv_dims[0], kv_dims[1], scratch->k,
                   runtime->pool);
        status = lis_layer_tensor(model, layer, "v_proj.weight", kv_dims, 2,
                                  &tensor);
        if (status != LIS_STATUS_OK) return status;
        lis_matvec(tensor, scratch->norm, kv_dims[0], kv_dims[1], scratch->v,
                   runtime->pool);

        status = lis_layer_tensor(model, layer, "q_norm.weight",
                                  qk_norm_dims, 1, &tensor);
        if (status != LIS_STATUS_OK) return status;
        lis_qwen3_head_rms_norm(scratch->q, cfg->attention_head_count,
                                cfg->head_dim, tensor, cfg->rms_norm_eps);
        status = lis_layer_tensor(model, layer, "k_norm.weight",
                                  qk_norm_dims, 1, &tensor);
        if (status != LIS_STATUS_OK) return status;
        lis_qwen3_head_rms_norm(scratch->k, cfg->kv_head_count,
                                cfg->head_dim, tensor, cfg->rms_norm_eps);
        if (emit_checkpoints && layer == 0U) {
            const size_t q_proj_dims[1] = { q_dims[0] };
            const size_t kv_proj_dims[1] = { kv_dims[0] };

            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                       "qwen3.layer.0.q_after_q_norm",
                                       q_proj_dims, 1, scratch->q, q_dims[0],
                                       runtime);
            lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                       "qwen3.layer.0.k_after_k_norm",
                                       kv_proj_dims, 1, scratch->k,
                                       kv_dims[0],
                                       runtime);
        }

        lis_apply_rope(scratch->q, cfg->attention_head_count, cfg->head_dim,
                       position, cfg->rope_theta, runtime->pool);
        lis_apply_rope(scratch->k, cfg->kv_head_count, cfg->head_dim,
                       position, cfg->rope_theta, runtime->pool);
        status = lis_store_kv(runtime, layer, position, scratch->k,
                              scratch->v);
        if (status != LIS_STATUS_OK) return status;
        status = lis_attention(runtime, layer, position, scratch->q,
                               scratch->scores, scratch->attn_probs,
                               cfg->context.configured_max_tokens,
                               scratch->attn, runtime->pool);
        if (status != LIS_STATUS_OK) return status;

        status = lis_layer_tensor(model, layer, "o_proj.weight", o_dims, 2,
                                  &tensor);
        if (status != LIS_STATUS_OK) return status;
        lis_matvec(tensor, scratch->attn, o_dims[0], o_dims[1],
                   scratch->attn_out, runtime->pool);
        lis_residual_add(scratch->hidden, scratch->attn_out,
                         cfg->hidden_size);

        status = lis_layer_tensor(model, layer, "mlp_norm.weight", norm_dims,
                                  1, &tensor);
        if (status != LIS_STATUS_OK) return status;
        lis_qwen3_rms_norm_eps(scratch->hidden, tensor, cfg->hidden_size,
                               cfg->rms_norm_eps, scratch->norm);

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
        lis_swiglu(scratch->gate, scratch->up, cfg->intermediate_size,
                   scratch->mlp, runtime->pool);
        status = lis_layer_tensor(model, layer, "down_proj.weight",
                                  mlp_down_dims, 2, &tensor);
        if (status != LIS_STATUS_OK) return status;
        lis_matvec(tensor, scratch->mlp, mlp_down_dims[0], mlp_down_dims[1],
                   scratch->mlp_out, runtime->pool);
        lis_residual_add(scratch->hidden, scratch->mlp_out,
                         cfg->hidden_size);
    }

    {
        const size_t norm_dims[1] = { cfg->hidden_size };

        status = lis_expect_tensor(model, "lis.output_norm.weight", norm_dims,
                                   1, cfg->weight_dtype, &tensor);
        if (status != LIS_STATUS_OK) return status;
        lis_qwen3_rms_norm_eps(scratch->hidden, tensor, cfg->hidden_size,
                               cfg->rms_norm_eps, scratch->norm);
    }
    if (emit_checkpoints) {
        const size_t h_dims[1] = { cfg->hidden_size };

        lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                  "qwen3.final_norm", h_dims, 1,
                                  scratch->norm, cfg->hidden_size,
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
    if (emit_checkpoints) {
        const size_t l_dims[1] = { cfg->vocab_size };

        lis_checkpoint_diagnostic(checkpoint_step, checkpoint_phase,
                                  "qwen3.logits", l_dims, 1, out_logits,
                                  cfg->vocab_size,
                                  runtime);
    }

    return LIS_STATUS_OK;
}

lis_status lis_runtime_qwen3_prefill(lis_runtime_context *runtime,
                                     const lis_loaded_model *model,
                                     const size_t *tokens,
                                     const size_t *sequence_lengths,
                                     size_t sequence_count,
                                     float *out_logits,
                                     size_t logits_len)
{
    lis_qwen3_scratch scratch;
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
    status = lis_qwen3_scratch_init(&scratch, &runtime->metadata.config);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    for (index = 0; index < sequence_lengths[0]; ++index) {
        const char *checkpoint_phase =
            index + 1U == sequence_lengths[0] ? "prefill" : NULL;

        status = lis_qwen3_forward_token(runtime, model, tokens[index],
                                         index, 0, checkpoint_phase,
                                         &scratch, out_logits, logits_len);
        if (status != LIS_STATUS_OK) {
            break;
        }
    }
    if (status == LIS_STATUS_OK) {
        runtime->batch.positions[0] = sequence_lengths[0];
        runtime->phase = LIS_RUNTIME_PHASE_PREFILLED;
    }
    lis_qwen3_scratch_destroy(&scratch);
    return status;
}

lis_status lis_runtime_qwen3_decode(lis_runtime_context *runtime,
                                    const lis_loaded_model *model,
                                    size_t token_id,
                                    float *out_logits,
                                    size_t logits_len)
{
    lis_qwen3_scratch scratch;
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
    status = lis_qwen3_scratch_init(&scratch, &runtime->metadata.config);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_qwen3_forward_token(runtime, model, token_id, position,
                                     checkpoint_step, "decode",
                                     &scratch, out_logits, logits_len);
    if (status == LIS_STATUS_OK) {
        ++runtime->batch.positions[0];
        ++runtime->decode_step_count;
        runtime->phase = LIS_RUNTIME_PHASE_DECODING;
    }
    lis_qwen3_scratch_destroy(&scratch);
    return status;
}
