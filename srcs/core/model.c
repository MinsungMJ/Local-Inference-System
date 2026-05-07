#include "lis/model.h"

#include <stdint.h>

const char *lis_model_family_name(lis_model_family family)
{
    switch (family) {
    case LIS_MODEL_FAMILY_LLAMA3_DECODER:
        return "llama3_decoder";
    case LIS_MODEL_FAMILY_QWEN3_DENSE_DECODER:
        return "qwen3_dense_decoder";
    case LIS_MODEL_FAMILY_GPT2_DECODER:
        return "gpt2_decoder";
    case LIS_MODEL_FAMILY_MISTRAL_DECODER:
        return "mistral_decoder";
    case LIS_MODEL_FAMILY_GPT_OSS_DECODER:
        return "gpt_oss_decoder";
    case LIS_MODEL_FAMILY_UNKNOWN:
        return "unknown";
    }

    return "unknown";
}

lis_model_support_envelope lis_model_support_envelope_default(void)
{
    lis_model_support_envelope support = {
        .functional_max_parameters = LIS_MODEL_FUNCTIONAL_MAX_PARAMETERS,
        .validation_target_parameters = 0,
    };

    return support;
}

int lis_model_config_token_is_eos(const lis_model_config *config,
                                  size_t token_id)
{
    size_t index;

    if (config == NULL) {
        return 0;
    }
    for (index = 0; index < config->eos_token_count; ++index) {
        if (config->eos_token_ids[index] == token_id) {
            return 1;
        }
    }
    return 0;
}

static lis_status lis_model_validate_family(lis_model_family family)
{
    if (family == LIS_MODEL_FAMILY_LLAMA3_DECODER ||
        family == LIS_MODEL_FAMILY_QWEN3_DENSE_DECODER) {
        return LIS_STATUS_OK;
    }
    if (family == LIS_MODEL_FAMILY_UNKNOWN) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    return LIS_STATUS_UNSUPPORTED;
}

static lis_status lis_model_validate_attention_shape(const lis_model_config *config)
{
    size_t q_width = 0;
    size_t kv_width = 0;
    size_t expected_hidden = 0;

    if (config->attention_head_count == 0 || config->kv_head_count == 0 ||
        config->head_dim == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (config->kv_head_count > config->attention_head_count) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (config->attention_head_count % config->kv_head_count != 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (config->head_dim != 0 &&
        config->attention_head_count > SIZE_MAX / config->head_dim) {
        return LIS_STATUS_OVERFLOW;
    }
    if (config->head_dim != 0 &&
        config->kv_head_count > SIZE_MAX / config->head_dim) {
        return LIS_STATUS_OVERFLOW;
    }
    q_width = config->attention_head_count * config->head_dim;
    kv_width = config->kv_head_count * config->head_dim;
    if (q_width == 0 || kv_width == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    if (config->family == LIS_MODEL_FAMILY_QWEN3_DENSE_DECODER) {
        return LIS_STATUS_OK;
    }

    if (config->hidden_size / config->head_dim != config->attention_head_count ||
        config->hidden_size % config->head_dim != 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    expected_hidden = q_width;
    if (expected_hidden != config->hidden_size) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    return LIS_STATUS_OK;
}

lis_status lis_model_config_validate(const lis_model_config *config)
{
    lis_status status;

    if (config == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    status = lis_model_validate_family(config->family);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (config->layer_count == 0 || config->hidden_size == 0 ||
        config->intermediate_size == 0 || config->vocab_size == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (config->rope_theta <= 0.0f) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (config->rms_norm_eps <= 0.0f) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (!lis_dtype_is_supported(config->weight_dtype)) {
        return LIS_STATUS_UNSUPPORTED;
    }
    if (config->eos_token_count > LIS_MODEL_MAX_EOS_TOKENS) {
        return LIS_STATUS_LIMIT_EXCEEDED;
    }
    for (size_t index = 0; index < config->eos_token_count; ++index) {
        if (config->eos_token_ids[index] >= config->vocab_size) {
            return LIS_STATUS_LIMIT_EXCEEDED;
        }
    }

    status = lis_model_validate_attention_shape(config);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    return lis_context_window_policy_validate(&config->context);
}

lis_status lis_model_metadata_validate(const lis_model_metadata *metadata)
{
    lis_status status;

    if (metadata == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (metadata->support.functional_max_parameters == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (metadata->support.functional_max_parameters >
        LIS_MODEL_FUNCTIONAL_MAX_PARAMETERS) {
        return LIS_STATUS_LIMIT_EXCEEDED;
    }
    if (metadata->support.validation_target_parameters >
        metadata->support.functional_max_parameters) {
        return LIS_STATUS_LIMIT_EXCEEDED;
    }

    status = lis_model_config_validate(&metadata->config);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    return LIS_STATUS_OK;
}
