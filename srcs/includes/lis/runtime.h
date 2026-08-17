#ifndef LIS_RUNTIME_H
#define LIS_RUNTIME_H

#include "lis/backend.h"
#include "lis/loader.h"
#include "lis/model.h"
#include "lis/status.h"
#include "lis/thread_pool.h"
#include "lis/layer_trace.h"
#include "lis/intra_layer_trace.h"

#include <stddef.h>

typedef enum {
    LIS_RUNTIME_PHASE_READY = 1,
    LIS_RUNTIME_PHASE_PREFILLED,
    LIS_RUNTIME_PHASE_DECODING
} lis_runtime_phase;

typedef struct {
    size_t batch_size;
    size_t max_tokens;
    size_t *positions;
} lis_static_batch;

typedef struct {
    size_t layer_count;
    size_t batch_size;
    size_t context_length;
    size_t kv_head_count;
    size_t head_dim;
    lis_dtype dtype;
    size_t element_size;
    size_t elements_per_cache;
    size_t bytes_per_cache;
} lis_kv_cache_layout;

typedef struct {
    lis_kv_cache_layout layout;
    unsigned char *keys;
    unsigned char *values;
} lis_kv_cache;

typedef struct {
    lis_model_metadata metadata;
    const lis_backend *backend;
    size_t batch_size;
    size_t thread_count;
    int layer_checkpoints_enabled;
    size_t layer_checkpoints_target_step;
    lis_layer_trace_record *layer_trace_record;
    lis_intra_layer_trace_record *intra_layer_record;
} lis_runtime_options;

typedef struct {
    lis_model_metadata metadata;
    const lis_backend *backend;
    lis_static_batch batch;
    lis_kv_cache kv_cache;
    lis_runtime_phase phase;
    lis_thread_pool *pool;
    int layer_checkpoints_enabled;
    size_t layer_checkpoints_target_step;
    size_t decode_step_count;
    lis_layer_trace_record *layer_trace_record;
    lis_intra_layer_trace_record *intra_layer_record;
} lis_runtime_context;

/*
 * Observes one target coordinate without allocation or tensor mutation. A
 * NULL record or a non-target coordinate is a successful zero-work no-op. A
 * target failure invalidates the record so no partial artifact can be emitted.
 */
lis_status lis_intra_layer_observe_fp32(
    lis_intra_layer_trace_record *record,
    lis_intra_layer_stage stage,
    size_t runtime_checkpoint_step,
    size_t layer_index,
    size_t token_position,
    const lis_intra_layer_fp32_view *view);

lis_status lis_static_batch_init(lis_static_batch *batch, size_t batch_size,
                                 size_t max_tokens);
void lis_static_batch_destroy(lis_static_batch *batch);
lis_status lis_static_batch_validate_lengths(const lis_static_batch *batch,
                                             const size_t *lengths,
                                             size_t length_count);

lis_status lis_kv_cache_init(lis_kv_cache *cache,
                             const lis_model_config *config,
                             size_t batch_size);
void lis_kv_cache_destroy(lis_kv_cache *cache);
lis_status lis_kv_cache_element_offset(const lis_kv_cache *cache,
                                       size_t layer, size_t batch,
                                       size_t position, size_t head,
                                       size_t dim, size_t *out_offset);
lis_status lis_kv_cache_key_ptr(const lis_kv_cache *cache,
                                size_t layer, size_t batch,
                                size_t position, size_t head,
                                size_t dim, void **out_ptr);
lis_status lis_kv_cache_value_ptr(const lis_kv_cache *cache,
                                  size_t layer, size_t batch,
                                  size_t position, size_t head,
                                  size_t dim, void **out_ptr);

lis_status lis_runtime_options_init(lis_runtime_options *options,
                                    const lis_model_metadata *metadata,
                                    const lis_backend *backend,
                                    size_t batch_size);
lis_status lis_runtime_init(lis_runtime_context *runtime,
                            const lis_runtime_options *options);
void lis_runtime_destroy(lis_runtime_context *runtime);
lis_status lis_runtime_prefill(lis_runtime_context *runtime,
                               const size_t *sequence_lengths,
                               size_t sequence_count);
lis_status lis_runtime_decode_step(lis_runtime_context *runtime);
lis_status lis_runtime_greedy_select(const float *logits, size_t vocab_size,
                                     size_t *out_token_id);
lis_status lis_runtime_llama_prefill(lis_runtime_context *runtime,
                                     const lis_loaded_model *model,
                                     const size_t *tokens,
                                     const size_t *sequence_lengths,
                                     size_t sequence_count,
                                     float *out_logits,
                                     size_t logits_len);
lis_status lis_runtime_llama_decode(lis_runtime_context *runtime,
                                    const lis_loaded_model *model,
                                    size_t token_id,
                                    float *out_logits,
                                    size_t logits_len);
lis_status lis_runtime_qwen3_prefill(lis_runtime_context *runtime,
                                     const lis_loaded_model *model,
                                     const size_t *tokens,
                                     const size_t *sequence_lengths,
                                     size_t sequence_count,
                                     float *out_logits,
                                     size_t logits_len);
lis_status lis_runtime_qwen3_decode(lis_runtime_context *runtime,
                                    const lis_loaded_model *model,
                                    size_t token_id,
                                    float *out_logits,
                                    size_t logits_len);

#endif
