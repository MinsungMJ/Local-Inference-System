#include "lis/runtime.h"
#include "lis/cpu_features.h"
#include "lis/cpu_ops.h"

#include <stdlib.h>
#include <string.h>

lis_status lis_runtime_options_init(lis_runtime_options *options,
                                    const lis_model_metadata *metadata,
                                    const lis_backend *backend,
                                    size_t batch_size)
{
    lis_status status;

    if (options == NULL || metadata == NULL || backend == NULL ||
        batch_size == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    status = lis_model_metadata_validate(metadata);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_context_window_policy_validate(&metadata->config.context);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (backend->kind != LIS_BACKEND_KIND_CPU_REFERENCE ||
        backend->memory_domain != LIS_BACKEND_MEMORY_HOST ||
        backend->execute == NULL) {
        return LIS_STATUS_UNSUPPORTED;
    }

    options->metadata = *metadata;
    options->backend = backend;
    options->batch_size = batch_size;
    options->thread_count = 1;
    options->layer_checkpoints_enabled = 0;
    options->layer_checkpoints_target_step = 0;
    options->layer_trace_record = NULL;
    return LIS_STATUS_OK;
}

lis_status lis_runtime_init(lis_runtime_context *runtime,
                            const lis_runtime_options *options)
{
    lis_runtime_context local = { 0 };
    lis_status status;
    size_t thread_count;

    if (runtime == NULL || options == NULL || options->backend == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    status = lis_model_metadata_validate(&options->metadata);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_context_window_policy_validate(&options->metadata.config.context);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (options->backend->kind != LIS_BACKEND_KIND_CPU_REFERENCE ||
        options->backend->memory_domain != LIS_BACKEND_MEMORY_HOST ||
        options->backend->execute == NULL || options->batch_size == 0) {
        return LIS_STATUS_UNSUPPORTED;
    }
    thread_count = options->thread_count;
    if (thread_count == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    local.metadata = options->metadata;
    local.backend = options->backend;
    status = lis_static_batch_init(&local.batch, options->batch_size,
                                   options->metadata.config.context.configured_max_tokens);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_kv_cache_init(&local.kv_cache, &options->metadata.config,
                               options->batch_size);
    if (status != LIS_STATUS_OK) {
        lis_static_batch_destroy(&local.batch);
        return status;
    }
    status = lis_thread_pool_init(&local.pool, thread_count);
    if (status != LIS_STATUS_OK) {
        lis_kv_cache_destroy(&local.kv_cache);
        lis_static_batch_destroy(&local.batch);
        return status;
    }
    local.phase = LIS_RUNTIME_PHASE_READY;
    local.layer_checkpoints_enabled = options->layer_checkpoints_enabled;
    local.layer_checkpoints_target_step = options->layer_checkpoints_target_step;
    local.decode_step_count = 0;
    local.layer_trace_record = options->layer_trace_record;

    lis_cpu_dispatch_init(lis_cpu_features_get());

    *runtime = local;
    return LIS_STATUS_OK;
}

void lis_runtime_destroy(lis_runtime_context *runtime)
{
    if (runtime == NULL) {
        return;
    }

    lis_thread_pool_destroy(&runtime->pool);
    lis_kv_cache_destroy(&runtime->kv_cache);
    lis_static_batch_destroy(&runtime->batch);
    memset(runtime, 0, sizeof(*runtime));
}

lis_status lis_runtime_prefill(lis_runtime_context *runtime,
                               const size_t *sequence_lengths,
                               size_t sequence_count)
{
    lis_status status;
    size_t index;

    if (runtime == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (runtime->phase != LIS_RUNTIME_PHASE_READY) {
        return LIS_STATUS_BAD_STATE;
    }

    status = lis_static_batch_validate_lengths(&runtime->batch,
                                               sequence_lengths,
                                               sequence_count);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    for (index = 0; index < sequence_count; ++index) {
        runtime->batch.positions[index] = sequence_lengths[index];
    }
    runtime->phase = LIS_RUNTIME_PHASE_PREFILLED;
    return LIS_STATUS_OK;
}

lis_status lis_runtime_decode_step(lis_runtime_context *runtime)
{
    size_t index;

    if (runtime == NULL || runtime->batch.positions == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (runtime->phase != LIS_RUNTIME_PHASE_PREFILLED &&
        runtime->phase != LIS_RUNTIME_PHASE_DECODING) {
        return LIS_STATUS_BAD_STATE;
    }

    for (index = 0; index < runtime->batch.batch_size; ++index) {
        size_t offset = 0;

        if (runtime->batch.positions[index] == 0) {
            return LIS_STATUS_BAD_STATE;
        }
        if (runtime->batch.positions[index] >= runtime->batch.max_tokens) {
            return LIS_STATUS_LIMIT_EXCEEDED;
        }
        if (lis_kv_cache_element_offset(&runtime->kv_cache, 0, index,
                                        runtime->batch.positions[index] - 1U,
                                        0, 0, &offset) != LIS_STATUS_OK) {
            return LIS_STATUS_BAD_STATE;
        }
    }

    for (index = 0; index < runtime->batch.batch_size; ++index) {
        ++runtime->batch.positions[index];
    }
    runtime->phase = LIS_RUNTIME_PHASE_DECODING;
    return LIS_STATUS_OK;
}

lis_status lis_runtime_greedy_select(const float *logits, size_t vocab_size,
                                     size_t *out_token_id)
{
    size_t best = 0;
    size_t index;

    if (logits == NULL || out_token_id == NULL || vocab_size == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    for (index = 1; index < vocab_size; ++index) {
        if (logits[index] > logits[best]) {
            best = index;
        }
    }

    *out_token_id = best;
    return LIS_STATUS_OK;
}
