#include "lis/loader.h"

#include <stdlib.h>
#include <string.h>

static const char *lis_json_find_key(const char *json, const char *end,
                                     const char *key)
{
    const size_t key_len = strlen(key);
    const char *cursor = json;

    while (cursor + key_len <= end) {
        if (memcmp(cursor, key, key_len) == 0) {
            return cursor + key_len;
        }
        ++cursor;
    }

    return NULL;
}

static const char *lis_skip_ws(const char *cursor, const char *end)
{
    while (cursor < end && (*cursor == ' ' || *cursor == '\n' ||
                            *cursor == '\r' || *cursor == '\t')) {
        ++cursor;
    }

    return cursor;
}

static lis_status lis_find_value(const char *json, const char *end,
                                 const char *key, const char **out_value)
{
    const char *cursor = lis_json_find_key(json, end, key);

    if (cursor == NULL) {
        return LIS_STATUS_FORMAT;
    }

    cursor = lis_skip_ws(cursor, end);
    if (cursor >= end || *cursor != ':') {
        return LIS_STATUS_FORMAT;
    }
    ++cursor;
    *out_value = lis_skip_ws(cursor, end);
    return LIS_STATUS_OK;
}

static lis_status lis_parse_json_size_field(const char *json, const char *end,
                                            const char *key, size_t *out)
{
    const char *cursor = NULL;
    size_t value = 0;
    lis_status status = lis_find_value(json, end, key, &cursor);

    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (cursor >= end || *cursor < '0' || *cursor > '9') {
        return LIS_STATUS_FORMAT;
    }

    while (cursor < end && *cursor >= '0' && *cursor <= '9') {
        const size_t digit = (size_t)(*cursor - '0');

        if (value > (SIZE_MAX - digit) / 10U) {
            return LIS_STATUS_OVERFLOW;
        }
        value = value * 10U + digit;
        ++cursor;
    }

    *out = value;
    return LIS_STATUS_OK;
}

static lis_status lis_parse_json_size_at(const char **cursor,
                                         const char *end, size_t *out)
{
    size_t value = 0;

    if (cursor == NULL || *cursor == NULL || out == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    *cursor = lis_skip_ws(*cursor, end);
    if (*cursor >= end || **cursor < '0' || **cursor > '9') {
        return LIS_STATUS_FORMAT;
    }
    while (*cursor < end && **cursor >= '0' && **cursor <= '9') {
        const size_t digit = (size_t)(**cursor - '0');

        if (value > (SIZE_MAX - digit) / 10U) {
            return LIS_STATUS_OVERFLOW;
        }
        value = value * 10U + digit;
        ++(*cursor);
    }
    *out = value;
    return LIS_STATUS_OK;
}

static lis_status lis_parse_json_float_field(const char *json, const char *end,
                                             const char *key, float *out)
{
    const char *cursor = NULL;
    char number[64] = { 0 };
    char *parse_end = NULL;
    size_t len = 0;
    double value = 0.0;
    lis_status status = lis_find_value(json, end, key, &cursor);

    if (status != LIS_STATUS_OK) {
        return status;
    }

    while (cursor + len < end &&
           ((cursor[len] >= '0' && cursor[len] <= '9') ||
            cursor[len] == '-' || cursor[len] == '+' ||
            cursor[len] == '.' || cursor[len] == 'e' ||
            cursor[len] == 'E')) {
        ++len;
    }
    if (len == 0 || len >= sizeof(number)) {
        return LIS_STATUS_FORMAT;
    }

    memcpy(number, cursor, len);
    number[len] = '\0';
    value = strtod(number, &parse_end);
    if (parse_end == number || *parse_end != '\0') {
        return LIS_STATUS_FORMAT;
    }
    *out = (float)value;
    return LIS_STATUS_OK;
}

static lis_status lis_parse_json_string_field(const char *json,
                                              const char *end,
                                              const char *key,
                                              char *out,
                                              size_t out_cap)
{
    const char *cursor = NULL;
    size_t len = 0;
    lis_status status;

    if (out == NULL || out_cap == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    status = lis_find_value(json, end, key, &cursor);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (cursor >= end || *cursor != '"') {
        return LIS_STATUS_FORMAT;
    }
    ++cursor;
    while (cursor + len < end && cursor[len] != '"') {
        if (cursor[len] == '\\') {
            return LIS_STATUS_FORMAT;
        }
        ++len;
    }
    if (cursor + len >= end || len + 1U > out_cap) {
        return LIS_STATUS_FORMAT;
    }

    memcpy(out, cursor, len);
    out[len] = '\0';
    return LIS_STATUS_OK;
}

static lis_status lis_parse_optional_json_size_field(const char *json,
                                                     const char *end,
                                                     const char *key,
                                                     size_t *out,
                                                     int *out_present)
{
    const char *value = lis_json_find_key(json, end, key);
    lis_status status;

    if (value == NULL) {
        *out_present = 0;
        return LIS_STATUS_OK;
    }

    status = lis_parse_json_size_field(json, end, key, out);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    *out_present = 1;
    return LIS_STATUS_OK;
}

static lis_status lis_parse_optional_json_float_field(const char *json,
                                                      const char *end,
                                                      const char *key,
                                                      float *out,
                                                      int *out_present)
{
    const char *value = lis_json_find_key(json, end, key);
    lis_status status;

    if (out == NULL || out_present == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (value == NULL) {
        *out_present = 0;
        return LIS_STATUS_OK;
    }

    status = lis_parse_json_float_field(json, end, key, out);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    *out_present = 1;
    return LIS_STATUS_OK;
}

static lis_status lis_parse_optional_json_bool_field(const char *json,
                                                     const char *end,
                                                     const char *key,
                                                     int *out,
                                                     int *out_present)
{
    const char *cursor = NULL;
    const char *value = lis_json_find_key(json, end, key);
    lis_status status;

    if (out == NULL || out_present == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (value == NULL) {
        *out_present = 0;
        return LIS_STATUS_OK;
    }
    status = lis_find_value(json, end, key, &cursor);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (cursor + 4 <= end && memcmp(cursor, "true", 4) == 0) {
        *out = 1;
        *out_present = 1;
        return LIS_STATUS_OK;
    }
    if (cursor + 5 <= end && memcmp(cursor, "false", 5) == 0) {
        *out = 0;
        *out_present = 1;
        return LIS_STATUS_OK;
    }
    return LIS_STATUS_FORMAT;
}

static lis_status lis_parse_optional_json_string_field(const char *json,
                                                       const char *end,
                                                       const char *key,
                                                       char *out,
                                                       size_t out_cap,
                                                       int *out_present)
{
    const char *value = lis_json_find_key(json, end, key);
    lis_status status;

    if (out == NULL || out_cap == 0 || out_present == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (value == NULL) {
        *out_present = 0;
        return LIS_STATUS_OK;
    }

    status = lis_parse_json_string_field(json, end, key, out, out_cap);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    *out_present = 1;
    return LIS_STATUS_OK;
}

static lis_status lis_parse_json_single_string_array_field(const char *json,
                                                           const char *end,
                                                           const char *key,
                                                           const char *expected)
{
    const char *cursor = NULL;
    char value[64] = { 0 };
    size_t len = 0;
    lis_status status;

    if (expected == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    status = lis_find_value(json, end, key, &cursor);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    cursor = lis_skip_ws(cursor, end);
    if (cursor >= end || *cursor != '[') {
        return LIS_STATUS_FORMAT;
    }
    ++cursor;
    cursor = lis_skip_ws(cursor, end);
    if (cursor >= end || *cursor != '"') {
        return LIS_STATUS_FORMAT;
    }
    ++cursor;
    while (cursor + len < end && cursor[len] != '"') {
        if (cursor[len] == '\\') {
            return LIS_STATUS_FORMAT;
        }
        ++len;
    }
    if (cursor + len >= end || len + 1U > sizeof(value)) {
        return LIS_STATUS_FORMAT;
    }
    memcpy(value, cursor, len);
    value[len] = '\0';
    cursor += len + 1U;
    cursor = lis_skip_ws(cursor, end);
    if (cursor >= end || *cursor != ']') {
        return LIS_STATUS_FORMAT;
    }
    if (strcmp(value, expected) != 0) {
        return LIS_STATUS_UNSUPPORTED_FORMAT;
    }
    return LIS_STATUS_OK;
}

/*
 * RoPE / rope-scaling boundary validator.
 *   - rope_scaling absent                → accepted
 *   - rope_scaling present and null      → accepted
 *   - rope_scaling present and non-null   → LIS_STATUS_UNSUPPORTED
 *     (includes {}, {"type":"linear"}, {"rope_type":"llama3"}, etc.)
 *   - rope_type absent                   → accepted
 *   - rope_type present and "default"     → accepted
 *   - rope_type present and not "default" → LIS_STATUS_UNSUPPORTED
 */
static lis_status lis_validate_plain_rope_config(const char *json,
                                                  const char *end)
{
    char rope_type[32] = { 0 };
    const char *rope_scaling_value = NULL;
    const char *after_null = NULL;
    int has_rope_type = 0;
    lis_status status;

    if (lis_json_find_key(json, end, "\"rope_scaling\"") != NULL) {
        status = lis_find_value(json, end, "\"rope_scaling\"",
                                &rope_scaling_value);
        if (status != LIS_STATUS_OK) {
            return status;
        }
        if (rope_scaling_value + 4 <= end &&
            memcmp(rope_scaling_value, "null", 4) == 0) {
            after_null = lis_skip_ws(rope_scaling_value + 4, end);
            if (after_null >= end || *after_null == ',' ||
                *after_null == '}') {
                goto validate_rope_type;
            }
            return LIS_STATUS_FORMAT;
        }
        return LIS_STATUS_UNSUPPORTED;
    }

validate_rope_type:
    status = lis_parse_optional_json_string_field(json, end, "\"rope_type\"",
                                                  rope_type,
                                                  sizeof(rope_type),
                                                  &has_rope_type);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (has_rope_type && strcmp(rope_type, "default") != 0) {
        return LIS_STATUS_UNSUPPORTED;
    }

    return LIS_STATUS_OK;
}

static lis_status lis_parse_optional_json_eos_field(const char *json,
                                                    const char *end,
                                                    lis_model_config *config)
{
    const char *cursor = NULL;
    const char *value = lis_json_find_key(json, end, "\"eos_token_id\"");
    lis_status status;

    if (config == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (value == NULL) {
        return LIS_STATUS_OK;
    }
    status = lis_find_value(json, end, "\"eos_token_id\"", &cursor);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    cursor = lis_skip_ws(cursor, end);
    if (cursor < end && *cursor == '[') {
        ++cursor;
        cursor = lis_skip_ws(cursor, end);
        if (cursor < end && *cursor == ']') {
            return LIS_STATUS_FORMAT;
        }
        for (;;) {
            if (config->eos_token_count >= LIS_MODEL_MAX_EOS_TOKENS) {
                return LIS_STATUS_LIMIT_EXCEEDED;
            }
            status = lis_parse_json_size_at(&cursor, end,
                                            &config->eos_token_ids[
                                                config->eos_token_count]);
            if (status != LIS_STATUS_OK) {
                return status;
            }
            ++config->eos_token_count;
            cursor = lis_skip_ws(cursor, end);
            if (cursor >= end) {
                return LIS_STATUS_FORMAT;
            }
            if (*cursor == ']') {
                return LIS_STATUS_OK;
            }
            if (*cursor != ',') {
                return LIS_STATUS_FORMAT;
            }
            ++cursor;
        }
    }
    status = lis_parse_json_size_at(&cursor, end,
                                    &config->eos_token_ids[0]);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    config->eos_token_count = 1;
    return LIS_STATUS_OK;
}

static lis_status lis_parse_dtype_name(const char *dtype_name,
                                       lis_dtype *out_dtype)
{
    if (strcmp(dtype_name, "float32") == 0 ||
        strcmp(dtype_name, "f32") == 0) {
        *out_dtype = LIS_DTYPE_F32;
        return LIS_STATUS_OK;
    }
    if (strcmp(dtype_name, "float16") == 0 ||
        strcmp(dtype_name, "f16") == 0) {
        *out_dtype = LIS_DTYPE_F16;
        return LIS_STATUS_OK;
    }
    if (strcmp(dtype_name, "bfloat16") == 0 ||
        strcmp(dtype_name, "bf16") == 0) {
        *out_dtype = LIS_DTYPE_BF16;
        return LIS_STATUS_OK;
    }

    return LIS_STATUS_UNSUPPORTED_DTYPE;
}

lis_status lis_loader_parse_llama3_config_json(const char *json,
                                               size_t json_len,
                                               lis_model_metadata *out_metadata)
{
    const char *end = NULL;
    char model_type[32] = { 0 };
    char dtype_name[32] = { 0 };
    size_t trained_context = 0;
    int has_kv_heads = 0;
    int has_head_dim = 0;
    int has_tie_word_embeddings = 0;
    int has_rms_norm_eps = 0;
    lis_model_config config = { 0 };
    lis_model_metadata metadata = { 0 };
    lis_status status;

    if (json == NULL || json_len == 0 || out_metadata == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    end = json + json_len;
    status = lis_parse_json_string_field(json, end, "\"model_type\"",
                                         model_type, sizeof(model_type));
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (strcmp(model_type, "llama") != 0 &&
        strcmp(model_type, "llama3") != 0) {
        return LIS_STATUS_UNSUPPORTED_FORMAT;
    }

    status = lis_validate_plain_rope_config(json, end);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    config.family = LIS_MODEL_FAMILY_LLAMA3_DECODER;
    status = lis_parse_json_size_field(json, end, "\"num_hidden_layers\"",
                                       &config.layer_count);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_parse_json_size_field(json, end, "\"hidden_size\"",
                                       &config.hidden_size);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_parse_json_size_field(json, end, "\"intermediate_size\"",
                                       &config.intermediate_size);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_parse_json_size_field(json, end, "\"num_attention_heads\"",
                                       &config.attention_head_count);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_parse_optional_json_size_field(json, end,
                                               "\"num_key_value_heads\"",
                                               &config.kv_head_count,
                                               &has_kv_heads);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (!has_kv_heads) {
        config.kv_head_count = config.attention_head_count;
    }
    status = lis_parse_optional_json_size_field(json, end, "\"head_dim\"",
                                               &config.head_dim,
                                               &has_head_dim);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (!has_head_dim) {
        if (config.attention_head_count == 0 ||
            config.hidden_size % config.attention_head_count != 0) {
            return LIS_STATUS_FORMAT;
        }
        config.head_dim = config.hidden_size / config.attention_head_count;
    }
    status = lis_parse_json_size_field(json, end, "\"vocab_size\"",
                                       &config.vocab_size);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_parse_json_float_field(json, end, "\"rope_theta\"",
                                        &config.rope_theta);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_parse_optional_json_float_field(json, end,
                                                "\"rms_norm_eps\"",
                                                &config.rms_norm_eps,
                                                &has_rms_norm_eps);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (!has_rms_norm_eps) {
        config.rms_norm_eps = 1.0e-5f;
    }
    status = lis_parse_json_string_field(json, end, "\"torch_dtype\"",
                                         dtype_name, sizeof(dtype_name));
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_parse_dtype_name(dtype_name, &config.weight_dtype);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_parse_json_size_field(json, end, "\"max_position_embeddings\"",
                                       &trained_context);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_parse_optional_json_bool_field(json, end,
                                               "\"tie_word_embeddings\"",
                                               &config.tie_word_embeddings,
                                               &has_tie_word_embeddings);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    (void)has_tie_word_embeddings;
    status = lis_parse_optional_json_eos_field(json, end, &config);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    config.context = lis_context_window_policy_default(trained_context,
                                                       trained_context);

    metadata.config = config;
    metadata.support = lis_model_support_envelope_default();

    status = lis_model_metadata_validate(&metadata);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    *out_metadata = metadata;
    return LIS_STATUS_OK;
}

lis_status lis_loader_parse_qwen3_config_json(const char *json,
                                              size_t json_len,
                                              lis_model_metadata *out_metadata)
{
    const char *end = NULL;
    char model_type[32] = { 0 };
    char dtype_name[32] = { 0 };
    char hidden_act[32] = { 0 };
    size_t trained_context = 0;
    int has_tie_word_embeddings = 0;
    int attention_bias = 0;
    int has_attention_bias = 0;
    int use_sliding_window = 0;
    int has_use_sliding_window = 0;
    lis_model_config config = { 0 };
    lis_model_metadata metadata = { 0 };
    lis_status status;

    if (json == NULL || json_len == 0 || out_metadata == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    end = json + json_len;
    status = lis_parse_json_string_field(json, end, "\"model_type\"",
                                         model_type, sizeof(model_type));
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (strcmp(model_type, "qwen3") != 0) {
        return LIS_STATUS_UNSUPPORTED_FORMAT;
    }
    status = lis_parse_json_single_string_array_field(
        json, end, "\"architectures\"", "Qwen3ForCausalLM");
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_validate_plain_rope_config(json, end);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_parse_optional_json_bool_field(json, end,
                                               "\"attention_bias\"",
                                               &attention_bias,
                                               &has_attention_bias);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (!has_attention_bias || attention_bias) {
        return LIS_STATUS_UNSUPPORTED;
    }
    status = lis_parse_optional_json_bool_field(json, end,
                                               "\"use_sliding_window\"",
                                               &use_sliding_window,
                                               &has_use_sliding_window);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (!has_use_sliding_window || use_sliding_window) {
        return LIS_STATUS_UNSUPPORTED;
    }
    status = lis_parse_json_string_field(json, end, "\"hidden_act\"",
                                         hidden_act, sizeof(hidden_act));
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (strcmp(hidden_act, "silu") != 0) {
        return LIS_STATUS_UNSUPPORTED;
    }

    config.family = LIS_MODEL_FAMILY_QWEN3_DENSE_DECODER;
    status = lis_parse_json_size_field(json, end, "\"num_hidden_layers\"",
                                       &config.layer_count);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_parse_json_size_field(json, end, "\"hidden_size\"",
                                       &config.hidden_size);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_parse_json_size_field(json, end, "\"intermediate_size\"",
                                       &config.intermediate_size);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_parse_json_size_field(json, end, "\"num_attention_heads\"",
                                       &config.attention_head_count);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_parse_json_size_field(json, end, "\"num_key_value_heads\"",
                                       &config.kv_head_count);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_parse_json_size_field(json, end, "\"head_dim\"",
                                       &config.head_dim);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_parse_json_size_field(json, end, "\"vocab_size\"",
                                       &config.vocab_size);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_parse_json_float_field(json, end, "\"rope_theta\"",
                                        &config.rope_theta);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_parse_json_float_field(json, end, "\"rms_norm_eps\"",
                                        &config.rms_norm_eps);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_parse_json_string_field(json, end, "\"torch_dtype\"",
                                         dtype_name, sizeof(dtype_name));
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_parse_dtype_name(dtype_name, &config.weight_dtype);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (config.weight_dtype != LIS_DTYPE_BF16) {
        return LIS_STATUS_UNSUPPORTED_DTYPE;
    }
    status = lis_parse_json_size_field(json, end, "\"max_position_embeddings\"",
                                       &trained_context);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_parse_optional_json_bool_field(json, end,
                                               "\"tie_word_embeddings\"",
                                               &config.tie_word_embeddings,
                                               &has_tie_word_embeddings);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    (void)has_tie_word_embeddings;
    status = lis_parse_optional_json_eos_field(json, end, &config);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    config.context = lis_context_window_policy_default(trained_context,
                                                       trained_context);

    metadata.config = config;
    metadata.support = lis_model_support_envelope_default();

    status = lis_model_metadata_validate(&metadata);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    *out_metadata = metadata;
    return LIS_STATUS_OK;
}

lis_status lis_loader_parse_hf_config_json(const char *json,
                                           size_t json_len,
                                           lis_model_metadata *out_metadata)
{
    const char *end = NULL;
    char model_type[32] = { 0 };
    lis_status status;

    if (json == NULL || json_len == 0 || out_metadata == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    end = json + json_len;
    status = lis_parse_json_string_field(json, end, "\"model_type\"",
                                         model_type, sizeof(model_type));
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (strcmp(model_type, "llama") == 0 ||
        strcmp(model_type, "llama3") == 0) {
        return lis_loader_parse_llama3_config_json(json, json_len,
                                                   out_metadata);
    }
    if (strcmp(model_type, "qwen3") == 0) {
        return lis_loader_parse_qwen3_config_json(json, json_len,
                                                  out_metadata);
    }
    return LIS_STATUS_UNSUPPORTED_FORMAT;
}
