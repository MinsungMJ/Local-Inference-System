#ifndef LIS_MODEL_H
#define LIS_MODEL_H

#include "lis/context.h"
#include "lis/dtype.h"
#include "lis/status.h"

#include <stdint.h>
#include <stddef.h>

#define LIS_MODEL_FUNCTIONAL_MAX_PARAMETERS UINT64_C(10000000000)
#define LIS_MODEL_MAX_EOS_TOKENS 16

typedef enum {
    LIS_MODEL_FAMILY_UNKNOWN = 0,
    LIS_MODEL_FAMILY_LLAMA3_DECODER,
    LIS_MODEL_FAMILY_QWEN3_DENSE_DECODER,
    LIS_MODEL_FAMILY_GPT2_DECODER,
    LIS_MODEL_FAMILY_MISTRAL_DECODER,
    LIS_MODEL_FAMILY_GPT_OSS_DECODER
} lis_model_family;

typedef struct {
    uint64_t functional_max_parameters;
    uint64_t validation_target_parameters;
} lis_model_support_envelope;

typedef struct {
    lis_model_family family;
    size_t layer_count;
    size_t hidden_size;
    size_t intermediate_size;
    size_t attention_head_count;
    size_t kv_head_count;
    size_t head_dim;
    size_t vocab_size;
    float rope_theta;
    float rms_norm_eps;
    lis_dtype weight_dtype;
    int tie_word_embeddings;
    size_t eos_token_ids[LIS_MODEL_MAX_EOS_TOKENS];
    size_t eos_token_count;
    lis_context_window_policy context;
} lis_model_config;

typedef struct {
    lis_model_config config;
    lis_model_support_envelope support;
} lis_model_metadata;

const char *lis_model_family_name(lis_model_family family);
lis_model_support_envelope lis_model_support_envelope_default(void);
int lis_model_config_token_is_eos(const lis_model_config *config,
                                  size_t token_id);
lis_status lis_model_config_validate(const lis_model_config *config);
lis_status lis_model_metadata_validate(const lis_model_metadata *metadata);

#endif
