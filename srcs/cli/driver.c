#include "lis/cli.h"

#include "lis/artifact.h"
#include "lis/layer_trace.h"
#include "lis/trace.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "lis/backend.h"
#include "lis/cpu_ops.h"
#include "lis/dtype.h"
#include "lis/loader.h"
#include "lis/perf.h"
#include "lis/runtime.h"
#include "lis/tokenizer.h"

static const float LIS_CLI_REPETITION_PENALTY = 1.2f;

/*
 * Process-local test harness injection. These controls have no CLI surface,
 * default to disabled, and are used only by the real-artifact integration
 * test to establish a truthful Pass 2 boundary and a controlled observation
 * mismatch without changing production executions.
 */
typedef struct {
    int selected_token_override_enabled;
    size_t selected_token_override_step;
    size_t selected_token_override_id;
    int observation_perturbation_enabled;
    size_t observation_perturbation_layer;
    size_t observation_perturbation_element;
    float observation_perturbation_delta;
} lis_cli_test_injection;

static lis_cli_test_injection s_lis_cli_test_injection = {0};

void lis_cli_test_injection_reset(void)
{
    memset(&s_lis_cli_test_injection, 0, sizeof(s_lis_cli_test_injection));
}

void lis_cli_test_override_selected_token(size_t step, size_t token_id)
{
    s_lis_cli_test_injection.selected_token_override_enabled = 1;
    s_lis_cli_test_injection.selected_token_override_step = step;
    s_lis_cli_test_injection.selected_token_override_id = token_id;
}

void lis_cli_test_perturb_layer_observation(size_t layer_index,
                                            size_t element_index,
                                            float delta)
{
    s_lis_cli_test_injection.observation_perturbation_enabled = 1;
    s_lis_cli_test_injection.observation_perturbation_layer = layer_index;
    s_lis_cli_test_injection.observation_perturbation_element = element_index;
    s_lis_cli_test_injection.observation_perturbation_delta = delta;
}

static void lis_cli_apply_test_selected_token_override(size_t step,
                                                       size_t vocab_size,
                                                       size_t *token_id,
                                                       int *should_stop)
{
    if (token_id != NULL && should_stop != NULL &&
        s_lis_cli_test_injection.selected_token_override_enabled &&
        s_lis_cli_test_injection.selected_token_override_step == step &&
        s_lis_cli_test_injection.selected_token_override_id < vocab_size) {
        *token_id = s_lis_cli_test_injection.selected_token_override_id;
        *should_stop = 0;
    }
}

/*
 * Fixed top-k candidate count for extended token-selection diagnostics.
 * This is a compile-time constant; k is not user-configurable.
 */
#define LIS_CLI_TOPK_CANDIDATE_COUNT 5

typedef struct {
    size_t token_id;
    float raw_score;
    float adjusted_score;
    int is_selected;
} lis_cli_topk_candidate;

typedef struct {
    lis_cli_topk_candidate entries[LIS_CLI_TOPK_CANDIDATE_COUNT];
    size_t count;
} lis_cli_topk_result;

static const char LIS_CLI_LLAMA_INSTRUCT_PREFIX[] =
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n";
static const char LIS_CLI_LLAMA_INSTRUCT_SUFFIX[] =
    "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";

typedef enum {
    LIS_CLI_STOP_NONE,
    LIS_CLI_STOP_MODEL_EOS,
    LIS_CLI_STOP_STRUCTURAL_CONTROL,
    LIS_CLI_STOP_DECODE_LIMIT,
    LIS_CLI_STOP_CONTEXT_LIMIT,
    LIS_CLI_STOP_RUNTIME_ERROR
} lis_cli_generation_stop_reason;

typedef struct {
    int structural_suppression_affected;
    int repetition_penalty_changed_selection;
    int selected_token_penalized;
    float raw_score_selected;
    float adjusted_score_selected;
    size_t runner_up_token_id;
    float runner_up_adjusted_score;
    int runner_up_available;
    int decision_margin_valid;
    float decision_margin;
    size_t suppressed_token_count;
    size_t penalized_token_count;
    const char *decision_class;
} lis_cli_selection_diagnostics;

typedef struct {
    size_t *selected_token_ids;
    size_t selected_token_count;
    size_t selected_token_capacity;
    size_t *emitted_token_ids;
    size_t emitted_token_count;
    size_t emitted_token_capacity;
    lis_cli_generation_stop_reason stop_reason;
} lis_cli_execution_record;

typedef lis_status (*lis_cli_prefill_fn)(lis_runtime_context *runtime,
                                         const lis_loaded_model *model,
                                         const size_t *tokens,
                                         const size_t *sequence_lengths,
                                         size_t sequence_count,
                                         float *out_logits,
                                         size_t logits_len);
typedef lis_status (*lis_cli_decode_fn)(lis_runtime_context *runtime,
                                        const lis_loaded_model *model,
                                        size_t token_id,
                                        float *out_logits,
                                        size_t logits_len);

lis_status lis_cli_build_llama_instruct_prompt(const char *user_text,
                                               char **out_prompt,
                                               size_t *out_len)
{
    const size_t prefix_len = sizeof(LIS_CLI_LLAMA_INSTRUCT_PREFIX) - 1U;
    const size_t suffix_len = sizeof(LIS_CLI_LLAMA_INSTRUCT_SUFFIX) - 1U;
    size_t user_len;
    size_t total_len;
    char *prompt = NULL;

    if (user_text == NULL || out_prompt == NULL || out_len == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    *out_prompt = NULL;
    *out_len = 0;

    user_len = strlen(user_text);
    if (user_len > SIZE_MAX - prefix_len ||
        suffix_len > SIZE_MAX - prefix_len - user_len ||
        prefix_len + user_len + suffix_len == SIZE_MAX) {
        return LIS_STATUS_OVERFLOW;
    }
    total_len = prefix_len + user_len + suffix_len;

    prompt = malloc(total_len + 1U);
    if (prompt == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }
    memcpy(prompt, LIS_CLI_LLAMA_INSTRUCT_PREFIX, prefix_len);
    memcpy(prompt + prefix_len, user_text, user_len);
    memcpy(prompt + prefix_len + user_len, LIS_CLI_LLAMA_INSTRUCT_SUFFIX,
           suffix_len);
    prompt[total_len] = '\0';

    *out_prompt = prompt;
    *out_len = total_len;
    return LIS_STATUS_OK;
}

static const char *lis_cli_stop_reason_name(
    lis_cli_generation_stop_reason reason)
{
    switch (reason) {
    case LIS_CLI_STOP_NONE:
        return "none";
    case LIS_CLI_STOP_MODEL_EOS:
        return "model_eos";
    case LIS_CLI_STOP_STRUCTURAL_CONTROL:
        return "structural_control";
    case LIS_CLI_STOP_DECODE_LIMIT:
        return "decode_limit";
    case LIS_CLI_STOP_CONTEXT_LIMIT:
        return "context_limit";
    case LIS_CLI_STOP_RUNTIME_ERROR:
        return "runtime_error";
    }
    return "runtime_error";
}

static int lis_cli_runtime_hit_context_limit(const lis_runtime_context *runtime)
{
    size_t index;

    if (runtime == NULL || runtime->batch.positions == NULL ||
        runtime->batch.max_tokens == 0) {
        return 0;
    }
    for (index = 0; index < runtime->batch.batch_size; ++index) {
        if (runtime->batch.positions[index] >= runtime->batch.max_tokens) {
            return 1;
        }
    }
    return 0;
}

static void lis_cli_report_context_limit(const lis_runtime_context *runtime,
                                         const lis_token_id_batch *batch)
{
    size_t index;

    if (runtime == NULL || batch == NULL || runtime->batch.positions == NULL ||
        batch->lengths == NULL || runtime->batch.batch_size != batch->batch_size) {
        fprintf(stderr,
                "lis: runtime error: context limit reached during generation\n");
        return;
    }
    for (index = 0; index < runtime->batch.batch_size; ++index) {
        if (runtime->batch.positions[index] >= runtime->batch.max_tokens) {
            size_t prompt_tokens = batch->lengths[index];
            size_t generated_tokens = runtime->batch.positions[index] >
                prompt_tokens ? runtime->batch.positions[index] -
                prompt_tokens : 0;

            fprintf(stderr,
                    "lis: runtime error: context limit reached during generation: "
                    "sequence=%zu prompt_tokens=%zu generated_tokens=%zu context=%zu\n",
                    index, prompt_tokens, generated_tokens,
                    runtime->batch.max_tokens);
            return;
        }
    }
    fprintf(stderr,
            "lis: runtime error: context limit reached during generation\n");
}

static int lis_cli_family_uses_hf_forward(lis_model_family family)
{
    return family == LIS_MODEL_FAMILY_LLAMA3_DECODER ||
           family == LIS_MODEL_FAMILY_QWEN3_DENSE_DECODER;
}

static lis_status lis_cli_family_runtime_fns(lis_model_family family,
                                             lis_cli_prefill_fn *out_prefill,
                                             lis_cli_decode_fn *out_decode)
{
    if (out_prefill == NULL || out_decode == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    *out_prefill = NULL;
    *out_decode = NULL;
    if (family == LIS_MODEL_FAMILY_LLAMA3_DECODER) {
        *out_prefill = lis_runtime_llama_prefill;
        *out_decode = lis_runtime_llama_decode;
        return LIS_STATUS_OK;
    }
    if (family == LIS_MODEL_FAMILY_QWEN3_DENSE_DECODER) {
        *out_prefill = lis_runtime_qwen3_prefill;
        *out_decode = lis_runtime_qwen3_decode;
        return LIS_STATUS_OK;
    }
    return LIS_STATUS_UNSUPPORTED_FORMAT;
}

static lis_status lis_cli_execution_record_init(
    lis_cli_execution_record *record, size_t generation_limit)
{
    if (record == NULL || generation_limit == 0U) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    memset(record, 0, sizeof(*record));
    if (generation_limit > SIZE_MAX / sizeof(*record->selected_token_ids)) {
        return LIS_STATUS_OVERFLOW;
    }
    record->selected_token_ids =
        malloc(generation_limit * sizeof(*record->selected_token_ids));
    record->emitted_token_ids =
        malloc(generation_limit * sizeof(*record->emitted_token_ids));
    if (record->selected_token_ids == NULL || record->emitted_token_ids == NULL) {
        free(record->selected_token_ids);
        free(record->emitted_token_ids);
        memset(record, 0, sizeof(*record));
        return LIS_STATUS_NO_MEMORY;
    }
    record->selected_token_capacity = generation_limit;
    record->emitted_token_capacity = generation_limit;
    record->stop_reason = LIS_CLI_STOP_NONE;
    return LIS_STATUS_OK;
}

static void lis_cli_execution_record_destroy(lis_cli_execution_record *record)
{
    if (record == NULL) {
        return;
    }
    free(record->selected_token_ids);
    free(record->emitted_token_ids);
    memset(record, 0, sizeof(*record));
}

static lis_status lis_cli_execution_record_append_selected(
    lis_cli_execution_record *record, size_t token_id)
{
    if (record == NULL || record->selected_token_ids == NULL ||
        record->selected_token_count >= record->selected_token_capacity) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    record->selected_token_ids[record->selected_token_count++] = token_id;
    return LIS_STATUS_OK;
}

static lis_status lis_cli_execution_record_append_emitted(
    lis_cli_execution_record *record, size_t token_id)
{
    if (record == NULL || record->emitted_token_ids == NULL ||
        record->emitted_token_count >= record->emitted_token_capacity) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    record->emitted_token_ids[record->emitted_token_count++] = token_id;
    return LIS_STATUS_OK;
}

static lis_artifact_input_mode lis_cli_artifact_input_mode(
    const lis_cli_options *options)
{
    if (options == NULL) {
        return LIS_ARTIFACT_INPUT_MODE_TOKENS;
    }
    if (options->token_path != NULL) {
        return LIS_ARTIFACT_INPUT_MODE_TOKENS;
    }
    if (options->vocab_path != NULL) {
        return LIS_ARTIFACT_INPUT_MODE_VOCAB_PROMPT;
    }
    return LIS_ARTIFACT_INPUT_MODE_HF_TOKENIZER_PROMPT;
}

static lis_artifact_output_mode lis_cli_artifact_output_mode(int has_tokenizer)
{
    return has_tokenizer ? LIS_ARTIFACT_OUTPUT_MODE_TEXT
                         : LIS_ARTIFACT_OUTPUT_MODE_TOKEN_IDS;
}

static const char *lis_cli_artifact_input_path(const lis_cli_options *options)
{
    if (options == NULL) {
        return NULL;
    }
    if (options->token_path != NULL) {
        return options->token_path;
    }
    if (options->vocab_path != NULL) {
        return options->vocab_path;
    }
    return options->hf_tokenizer_path;
}

static lis_status lis_cli_artifact_model_path(const lis_cli_options *options,
                                              lis_model_format model_format,
                                              char *buffer,
                                              size_t buffer_size,
                                              const char **out_path)
{
    if (options == NULL || options->model_path == NULL || out_path == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (model_format == LIS_MODEL_FORMAT_HUGGINGFACE_LOCAL) {
        if (buffer == NULL || buffer_size == 0U) {
            return LIS_STATUS_INVALID_ARGUMENT;
        }
        if (snprintf(buffer, buffer_size, "%s/model.safetensors",
                     options->model_path) >= (int)buffer_size) {
            return LIS_STATUS_OVERFLOW;
        }
        *out_path = buffer;
        return LIS_STATUS_OK;
    }
    *out_path = options->model_path;
    return LIS_STATUS_OK;
}

static lis_status lis_cli_artifact_build_prompt_sequences(
    const lis_token_id_batch *batch,
    lis_artifact_prompt_sequence **out_sequences,
    size_t *out_count)
{
    lis_artifact_prompt_sequence *sequences = NULL;
    size_t sequence_idx;
    size_t token_offset = 0;
    lis_status status = LIS_STATUS_OK;

    if (batch == NULL || batch->tokens == NULL || batch->lengths == NULL ||
        batch->batch_size == 0U || out_sequences == NULL || out_count == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    *out_sequences = NULL;
    *out_count = 0;
    sequences = calloc(batch->batch_size, sizeof(*sequences));
    if (sequences == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }

    for (sequence_idx = 0; sequence_idx < batch->batch_size; ++sequence_idx) {
        size_t token_count = batch->lengths[sequence_idx];

        if (token_count > batch->token_count ||
            token_offset > batch->token_count - token_count) {
            status = LIS_STATUS_FORMAT;
            break;
        }
        sequences[sequence_idx].token_count = token_count;
        status = lis_artifact_fingerprint_token_ids(batch->tokens + token_offset,
                                                    token_count,
                                                    &sequences[sequence_idx]
                                                         .token_id_digest);
        if (status != LIS_STATUS_OK) {
            break;
        }
        token_offset += token_count;
    }
    if (status == LIS_STATUS_OK && token_offset != batch->token_count) {
        status = LIS_STATUS_FORMAT;
    }
    if (status != LIS_STATUS_OK) {
        free(sequences);
        return status;
    }

    *out_sequences = sequences;
    *out_count = batch->batch_size;
    return LIS_STATUS_OK;
}

static int lis_cli_size_mul_overflows(size_t lhs, size_t rhs, size_t *out)
{
    if (out == NULL) {
        return 1;
    }
    if (lhs != 0U && rhs > SIZE_MAX / lhs) {
        return 1;
    }
    *out = lhs * rhs;
    return 0;
}

static lis_status lis_cli_build_kv_cache_report(
    const lis_runtime_context *runtime,
    lis_artifact_kv_cache_report *out)
{
    const lis_kv_cache_layout *layout = NULL;
    lis_artifact_kv_cache_report report = { 0 };
    size_t bytes_per_token = 0;
    size_t index;

    if (runtime == NULL || out == NULL || runtime->batch.positions == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    layout = &runtime->kv_cache.layout;
    if (layout->layer_count == 0U || layout->batch_size == 0U ||
        layout->context_length == 0U || layout->kv_head_count == 0U ||
        layout->head_dim == 0U || layout->element_size == 0U ||
        layout->batch_size != runtime->batch.batch_size) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    report.storage_dtype = layout->dtype;
    report.max_tokens = layout->context_length;
    report.layer_count = layout->layer_count;
    report.batch_size = layout->batch_size;
    report.kv_head_count = layout->kv_head_count;
    report.head_dim = layout->head_dim;
    report.element_size = layout->element_size;
    for (index = 0; index < runtime->batch.batch_size; ++index) {
        if (runtime->batch.positions[index] > report.used_tokens) {
            report.used_tokens = runtime->batch.positions[index];
        }
    }
    if (report.used_tokens > report.max_tokens) {
        return LIS_STATUS_BAD_STATE;
    }

    bytes_per_token = layout->layer_count;
    if (lis_cli_size_mul_overflows(bytes_per_token, layout->batch_size,
                                   &bytes_per_token) ||
        lis_cli_size_mul_overflows(bytes_per_token, 2U, &bytes_per_token) ||
        lis_cli_size_mul_overflows(bytes_per_token, layout->kv_head_count,
                                   &bytes_per_token) ||
        lis_cli_size_mul_overflows(bytes_per_token, layout->head_dim,
                                   &bytes_per_token) ||
        lis_cli_size_mul_overflows(bytes_per_token, layout->element_size,
                                   &bytes_per_token)) {
        return LIS_STATUS_OVERFLOW;
    }
    report.bytes_per_token = bytes_per_token;
    if (lis_cli_size_mul_overflows(bytes_per_token, report.max_tokens,
                                   &report.allocated_bytes) ||
        lis_cli_size_mul_overflows(bytes_per_token, report.used_tokens,
                                   &report.used_bytes)) {
        return LIS_STATUS_OVERFLOW;
    }
    report.valid = 1;
    *out = report;
    return LIS_STATUS_OK;
}

static void lis_cli_emit_kv_cache_diagnostic(
    const lis_artifact_kv_cache_report *report)
{
    if (report == NULL || !report->valid) {
        return;
    }

    fprintf(stderr,
            "lis: kv-cache: scope=run_local policy=eviction_free,monotonic "
            "dtype=%s max_tokens=%zu used_tokens=%zu "
            "bytes_per_token=%zu allocated_bytes=%zu used_bytes=%zu\n",
            lis_dtype_name(report->storage_dtype),
            report->max_tokens,
            report->used_tokens,
            report->bytes_per_token,
            report->allocated_bytes,
            report->used_bytes);
}

static lis_status lis_cli_write_execution_artifact(
    const lis_cli_options *options,
    const lis_loaded_model *model,
    const lis_token_id_batch *batch,
    const lis_runtime_context *runtime,
    int has_tokenizer,
    const char *backend_name,
    const char *precision_path,
    const lis_artifact_set_id *artifact_set_id,
    const lis_cli_execution_record *record,
    lis_status run_status,
    const lis_perf_report *perf)
{
    lis_artifact_run_report report = { 0 };
    lis_artifact_prompt_sequence *prompt_sequences = NULL;
    lis_artifact_fingerprint selected_digest = { 0 };
    lis_artifact_fingerprint emitted_digest = { 0 };
    char model_path_buffer[1024];
    const char *model_path = NULL;
    const char *input_path = NULL;
    const char *stop_reason_name = NULL;
    static const size_t empty_token_ids[1] = { 0 };
    lis_status status;

    if (options == NULL ||
        (options->report_json_path == NULL && options->report_md_path == NULL) ||
        model == NULL || batch == NULL || backend_name == NULL ||
        runtime == NULL || artifact_set_id == NULL ||
        !artifact_set_id->valid || record == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    status = lis_cli_artifact_model_path(options, model->format,
                                           model_path_buffer,
                                           sizeof(model_path_buffer),
                                           &model_path);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    input_path = lis_cli_artifact_input_path(options);
    if (input_path == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    status = lis_cli_artifact_build_prompt_sequences(batch, &prompt_sequences,
                                                      &report.prompt_sequence_count);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    status = lis_artifact_fingerprint_token_ids(
        record->selected_token_ids, record->selected_token_count,
        &selected_digest);
    if (status != LIS_STATUS_OK) {
        free(prompt_sequences);
        return status;
    }
    status = lis_artifact_fingerprint_token_ids(
        record->emitted_token_ids, record->emitted_token_count,
        &emitted_digest);
    if (status != LIS_STATUS_OK) {
        free(prompt_sequences);
        return status;
    }

    status = lis_artifact_fingerprint_current_binary(&report.binary_fingerprint);
    if (status != LIS_STATUS_OK) {
        free(prompt_sequences);
        return status;
    }
    status = lis_artifact_fingerprint_file(model_path, &report.model_fingerprint);
    if (status != LIS_STATUS_OK) {
        free(prompt_sequences);
        return status;
    }
    status = lis_artifact_fingerprint_file(options->config_path,
                                            &report.config_fingerprint);
    if (status != LIS_STATUS_OK) {
        free(prompt_sequences);
        return status;
    }
    status = lis_artifact_fingerprint_file(input_path, &report.input_fingerprint);
    if (status != LIS_STATUS_OK) {
        free(prompt_sequences);
        return status;
    }
    status = lis_artifact_fingerprint_runtime(
        options, model->format, model->metadata.config.family,
        lis_cli_artifact_input_mode(options), backend_name,
        &report.runtime_fingerprint);
    if (status != LIS_STATUS_OK) {
        free(prompt_sequences);
        return status;
    }
    status = lis_artifact_fingerprint_backend(backend_name,
                                               options->thread_count,
                                               &report.backend_fingerprint);
    if (status != LIS_STATUS_OK) {
        free(prompt_sequences);
        return status;
    }

    stop_reason_name = lis_cli_stop_reason_name(record->stop_reason);
    report.model_format_name = lis_model_format_name(model->format);
    report.artifact_set_id = artifact_set_id;
    report.model_family_name = lis_model_family_name(model->metadata.config.family);
    report.backend_name = backend_name;
    report.precision_path = precision_path;
    report.stop_reason_name = stop_reason_name;
    report.options = options;
    report.model = model;
    report.input_mode = lis_cli_artifact_input_mode(options);
    report.output_mode = lis_cli_artifact_output_mode(has_tokenizer);
    report.prompt_sequences = prompt_sequences;
    report.selected_token_ids = record->selected_token_count > 0U
        ? record->selected_token_ids : empty_token_ids;
    report.selected_token_count = record->selected_token_count;
    report.selected_token_digest = selected_digest;
    report.emitted_token_ids = record->emitted_token_count > 0U
        ? record->emitted_token_ids : empty_token_ids;
    report.emitted_token_count = record->emitted_token_count;
    report.emitted_token_digest = emitted_digest;
    report.status = run_status;
    report.perf = perf;
    status = lis_cli_build_kv_cache_report(runtime, &report.kv_cache);
    if (status != LIS_STATUS_OK) {
        free(prompt_sequences);
        return status;
    }

    if (options->report_json_path != NULL) {
        report.path = options->report_json_path;
        status = lis_artifact_write_run_report(&report);
        if (status != LIS_STATUS_OK) {
            free(prompt_sequences);
            return status;
        }
    }
    if (options->report_md_path != NULL) {
        report.path = options->report_md_path;
        status = lis_artifact_write_run_report_md(&report);
        if (status != LIS_STATUS_OK) {
            free(prompt_sequences);
            return status;
        }
    }
    free(prompt_sequences);
    return LIS_STATUS_OK;
}

static void lis_cli_fprint_escaped_token_text(FILE *stream,
                                              const char *text,
                                              size_t text_len)
{
    static const char hex[] = "0123456789abcdef";
    size_t index;

    fputc('"', stream);
    for (index = 0; index < text_len; ++index) {
        const unsigned char ch = (unsigned char)text[index];

        if (ch == '\\' || ch == '"') {
            fputc('\\', stream);
            fputc((int)ch, stream);
        } else if (ch == '\n') {
            fputs("\\n", stream);
        } else if (ch == '\r') {
            fputs("\\r", stream);
        } else if (ch == '\t') {
            fputs("\\t", stream);
        } else if (ch >= 0x20U && ch <= 0x7eU) {
            fputc((int)ch, stream);
        } else {
            fputs("\\x", stream);
            fputc(hex[ch >> 4], stream);
            fputc(hex[ch & 0x0fU], stream);
        }
    }
    fputc('"', stream);
}

static void lis_cli_emit_generation_diagnostic(
    size_t step,
    size_t token_id,
    const lis_tokenizer *tok,
    lis_cli_generation_stop_reason reason,
    const lis_cli_selection_diagnostics *selection,
    const lis_cli_topk_result *topk,
    const char *phase)
{
    fprintf(stderr,
            "lis: generation-diagnostic step=%zu phase=%s "
            "selected_token_id=%zu selected_token_text=",
            step, phase != NULL ? phase : "decode", token_id);
    if (tok != NULL && token_id < tok->vocab_size &&
        tok->token_bytes[token_id] != NULL) {
        lis_cli_fprint_escaped_token_text(stderr, tok->token_bytes[token_id],
                                          tok->token_lens[token_id]);
    } else {
        fputs("<unavailable>", stderr);
    }
    fprintf(stderr,
            " stop_reason=%s structural_suppression_affected=%s "
            "repetition_penalty_changed_selection=%s "
            "selected_token_penalized=%s\n",
            lis_cli_stop_reason_name(reason),
            selection != NULL && selection->structural_suppression_affected ?
                "true" : "false",
            selection != NULL &&
                    selection->repetition_penalty_changed_selection ?
                "true" : "false",
            selection != NULL && selection->selected_token_penalized ?
                "true" : "false");

    if (selection != NULL) {
        char margin_buf[64];
        char runner_up_buf[64];
        if (selection->decision_margin_valid) {
            snprintf(margin_buf, sizeof(margin_buf), "%.6g",
                     selection->decision_margin);
        } else {
            snprintf(margin_buf, sizeof(margin_buf), "n/a");
        }
        if (selection->runner_up_available) {
            snprintf(runner_up_buf, sizeof(runner_up_buf), "%zu",
                     selection->runner_up_token_id);
        } else {
            snprintf(runner_up_buf, sizeof(runner_up_buf), "none");
        }
        fprintf(stderr,
                "lis: generation-diagnostic-reasoning step=%zu phase=%s "
                "decision_class=%s margin=%s runner_up_token_id=%s "
                "suppressed_token_count=%zu penalized_token_count=%zu\n",
                step, phase != NULL ? phase : "decode",
                selection->decision_class != NULL
                    ? selection->decision_class : "greedy",
                margin_buf,
                runner_up_buf,
                selection->suppressed_token_count,
                selection->penalized_token_count);
    }

    /* Emit top-k candidate entries after the existing diagnostic line. */
    if (topk != NULL) {
        size_t rank;

        for (rank = 0; rank < topk->count; ++rank) {
            const lis_cli_topk_candidate *c = &topk->entries[rank];

            fprintf(stderr,
                    "lis: generation-diagnostic-candidate step=%zu "
                    "phase=%s rank=%zu token_id=%zu token_text=",
                    step, phase != NULL ? phase : "decode",
                    rank + 1U, c->token_id);
            if (tok != NULL && c->token_id < tok->vocab_size &&
                tok->token_bytes[c->token_id] != NULL) {
                lis_cli_fprint_escaped_token_text(
                    stderr, tok->token_bytes[c->token_id],
                    tok->token_lens[c->token_id]);
            } else {
                fputs("<unavailable>", stderr);
            }
            fprintf(stderr,
                    " raw_score=%.6g adjusted_score=%.6g selected=%s\n",
                    c->raw_score, c->adjusted_score,
                    c->is_selected ? "true" : "false");
        }
    }
}

static void lis_cli_report_unsupported_config(const char *path,
                                               lis_status status)
{
    fprintf(stderr,
            "lis: config error: unsupported configuration in %s: "
            "plain RoPE from rope_theta is the only supported positional "
            "encoding; attention_bias, sliding_window, non-default "
            "hidden_act, and any non-null rope_scaling or non-default "
            "rope_type are rejected: %s\n",
            path, lis_status_name(status));
}

static lis_status lis_cli_read_text_file(const char *path, char **out_data,
                                         size_t *out_len)
{
    FILE *fp = NULL;
    long file_size = 0;
    char *data = NULL;
    lis_status status = LIS_STATUS_IO;

    if (path == NULL || out_data == NULL || out_len == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    fp = fopen(path, "rb");
    if (fp == NULL) {
        return LIS_STATUS_IO;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        goto out;
    }
    file_size = ftell(fp);
    if (file_size <= 0) {
        status = LIS_STATUS_FORMAT;
        goto out;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        goto out;
    }

    data = malloc((size_t)file_size + 1U);
    if (data == NULL) {
        status = LIS_STATUS_NO_MEMORY;
        goto out;
    }
    if (fread(data, 1, (size_t)file_size, fp) != (size_t)file_size) {
        goto out;
    }
    data[(size_t)file_size] = '\0';

    *out_data = data;
    *out_len = (size_t)file_size;
    data = NULL;
    status = LIS_STATUS_OK;

out:
    free(data);
    if (fp != NULL && fclose(fp) != 0 && status == LIS_STATUS_OK) {
        status = LIS_STATUS_IO;
    }
    return status;
}

static lis_status lis_cli_load_metadata(const lis_cli_options *options,
                                        lis_model_metadata *out_metadata)
{
    char *config_json = NULL;
    size_t config_len = 0;
    lis_status status;

    status = lis_cli_read_text_file(options->config_path, &config_json,
                                    &config_len);
    if (status != LIS_STATUS_OK) {
        fprintf(stderr, "lis: config error: could not read %s: %s\n",
                options->config_path, lis_status_name(status));
        return status;
    }

    status = lis_loader_parse_hf_config_json(config_json, config_len,
                                             out_metadata);
    free(config_json);
    if (status != LIS_STATUS_OK) {
        if (status == LIS_STATUS_UNSUPPORTED) {
            lis_cli_report_unsupported_config(options->config_path,
                                              status);
        } else {
            fprintf(stderr,
                    "lis: config error: unsupported or invalid config: %s\n",
                    lis_status_name(status));
        }
        return status;
    }

    out_metadata->config.context.configured_max_tokens =
        options->context_length;
    status = lis_model_metadata_validate(out_metadata);
    if (status != LIS_STATUS_OK) {
        fprintf(stderr, "lis: config error: context/model validation failed: %s\n",
                lis_status_name(status));
        return status;
    }

    return LIS_STATUS_OK;
}

static lis_status lis_cli_find_validation_logits(const lis_loaded_model *model,
                                                 size_t vocab_size,
                                                 lis_tensor_view *out_view)
{
    size_t index;

    if (model == NULL || out_view == NULL || vocab_size == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    for (index = 0; index < model->tensor_count; ++index) {
        const lis_loaded_tensor *tensor = &model->tensors[index];

        if (strcmp(tensor->name, "lis.validation_logits") != 0) {
            continue;
        }
        if (tensor->view.dtype != LIS_DTYPE_F32 ||
            tensor->view.shape.rank != 2 ||
            tensor->view.shape.dims[0] != 1U ||
            tensor->view.shape.dims[1] != vocab_size) {
            return LIS_STATUS_SHAPE_MISMATCH;
        }

        *out_view = tensor->view;
        return LIS_STATUS_OK;
    }

    return LIS_STATUS_UNSUPPORTED_SHAPE;
}

static lis_status lis_cli_compute_validation_logits(const lis_runtime_context *runtime,
                                                    lis_tensor_view projection,
                                                    float *out_logits,
                                                    size_t vocab_size)
{
    const float activation_data[1] = { 1.0f };
    lis_tensor_shape activation_shape = { 0 };
    lis_tensor_shape output_shape = { 0 };
    lis_tensor_view activation = { 0 };
    lis_tensor output = { 0 };
    lis_operator_request request = { 0 };
    const size_t activation_dims[2] = { 1U, 1U };
    const size_t output_dims[2] = { 1U, vocab_size };
    lis_status status;

    if (runtime == NULL || out_logits == NULL || vocab_size == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    status = lis_tensor_shape_make(2, activation_dims, &activation_shape);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_tensor_shape_make(2, output_dims, &output_shape);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_tensor_view_from_borrowed(LIS_DTYPE_F32, &activation_shape,
                                           activation_data,
                                           sizeof(activation_data),
                                           &activation);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_tensor_init_borrowed(LIS_DTYPE_F32, &output_shape,
                                      out_logits,
                                      vocab_size * sizeof(*out_logits),
                                      &output);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    request = lis_operator_make_matmul(activation, projection, &output);
    status = lis_operator_execute(runtime->backend, &request);
    lis_tensor_destroy(&output);
    return status;
}

static lis_status lis_cli_emit_token(size_t token_id, const lis_tokenizer *tok,
                                     int *emitted_token)
{
    if (tok != NULL) {
        char *piece = NULL;
        size_t piece_len = 0;
        lis_status status = lis_tokenizer_decode(tok, &token_id, 1, &piece,
                                                 &piece_len);

        if (status != LIS_STATUS_OK) {
            return status;
        }
        if (piece_len > 0) {
            fwrite(piece, 1, piece_len, stdout);
        }
        free(piece);
    } else {
        if (emitted_token != NULL && !*emitted_token) {
            printf("generated_token_ids:");
            *emitted_token = 1;
        }
        printf(" %zu", token_id);
    }
    return LIS_STATUS_OK;
}

static int lis_cli_token_text_equals(const lis_tokenizer *tok,
                                     size_t token_id, const char *text)
{
    const size_t text_len = strlen(text);

    if (tok == NULL || token_id >= tok->vocab_size ||
        tok->token_bytes[token_id] == NULL) {
        return 0;
    }
    return tok->token_lens[token_id] == text_len &&
           memcmp(tok->token_bytes[token_id], text, text_len) == 0;
}

static int lis_cli_token_text_has_reserved_prefix(const lis_tokenizer *tok,
                                                  size_t token_id)
{
    const char prefix[] = "<|reserved_special_token_";
    const char suffix[] = "|>";
    const size_t prefix_len = sizeof(prefix) - 1U;
    const size_t suffix_len = sizeof(suffix) - 1U;
    size_t token_len = 0;
    const char *text = NULL;

    if (tok == NULL || token_id >= tok->vocab_size) {
        return 0;
    }
    token_len = tok->token_lens[token_id];
    text = tok->token_bytes[token_id];
    if (text == NULL || token_len < prefix_len + suffix_len) {
        return 0;
    }
    return memcmp(text, prefix, prefix_len) == 0 &&
           memcmp(text + token_len - suffix_len, suffix, suffix_len) == 0;
}

static int lis_cli_token_is_structural_stop(const lis_model_config *config,
                                            const lis_tokenizer *tok,
                                            size_t token_id)
{
    if (tok == NULL) {
        return 0;
    }
    if (lis_model_config_token_is_eos(config, token_id)) {
        return 1;
    }
    return lis_cli_token_text_equals(tok, token_id, "<|eot_id|>") ||
           lis_cli_token_text_equals(tok, token_id, "<|end_of_text|>") ||
           lis_cli_token_text_equals(tok, token_id, "<|eom_id|>");
}

static int lis_cli_token_is_suppressed_control(const lis_tokenizer *tok,
                                               size_t token_id)
{
    if (tok == NULL) {
        return 0;
    }
    return lis_cli_token_text_equals(tok, token_id, "<|begin_of_text|>") ||
           lis_cli_token_text_equals(tok, token_id, "<|start_header_id|>") ||
           lis_cli_token_text_equals(tok, token_id, "<|end_header_id|>") ||
           lis_cli_token_text_equals(tok, token_id,
                                     "<|finetune_right_pad_id|>") ||
           lis_cli_token_text_equals(tok, token_id, "<|step_id|>") ||
           lis_cli_token_text_equals(tok, token_id, "<|python_tag|>") ||
           lis_cli_token_text_has_reserved_prefix(tok, token_id);
}

static int lis_cli_token_was_generated(const size_t *generated_tokens,
                                       size_t generated_count,
                                       size_t token_id)
{
    size_t index;

    for (index = 0; index < generated_count; ++index) {
        if (generated_tokens[index] == token_id) {
            return 1;
        }
    }
    return 0;
}

static float lis_cli_apply_repetition_penalty(float logit,
                                              const size_t *generated_tokens,
                                              size_t generated_count,
                                              size_t token_id)
{
    if (!lis_cli_token_was_generated(generated_tokens, generated_count,
                                     token_id)) {
        return logit;
    }
    if (logit > 0.0f) {
        return logit / LIS_CLI_REPETITION_PENALTY;
    }
    if (logit < 0.0f) {
        return logit * LIS_CLI_REPETITION_PENALTY;
    }
    return logit;
}

/*
 * Extract the top-k candidates by adjusted score from the logits array.
 * Suppressed tokens receive -INFINITY as their adjusted score.
 * The selected token is marked in the result.
 */
static void lis_cli_extract_topk_candidates(
    const float *logits,
    size_t vocab_size,
    const lis_tokenizer *tok,
    const size_t *generated_tokens,
    size_t generated_count,
    size_t selected_token_id,
    lis_cli_topk_result *out_topk)
{
    size_t index;
    size_t k = LIS_CLI_TOPK_CANDIDATE_COUNT;

    if (out_topk == NULL) {
        return;
    }
    out_topk->count = 0;
    if (logits == NULL || vocab_size == 0) {
        return;
    }
    if (k > vocab_size) {
        k = vocab_size;
    }

    for (index = 0; index < vocab_size; ++index) {
        const float raw = logits[index];
        const int suppressed =
            tok != NULL && lis_cli_token_is_suppressed_control(tok, index);
        float adjusted;
        lis_cli_topk_candidate candidate;
        size_t insert_pos;
        size_t shift;

        if (suppressed) {
            adjusted = -INFINITY;
        } else {
            adjusted = lis_cli_apply_repetition_penalty(
                raw, generated_tokens, generated_count, index);
        }

        /* Insert into sorted top-k if this candidate qualifies. */
        if (out_topk->count < k ||
            adjusted > out_topk->entries[out_topk->count - 1U].adjusted_score) {
            candidate.token_id = index;
            candidate.raw_score = raw;
            candidate.adjusted_score = adjusted;
            candidate.is_selected = (index == selected_token_id) ? 1 : 0;

            /* Find insertion point (descending order by adjusted score). */
            insert_pos = out_topk->count < k ? out_topk->count : k - 1U;
            for (shift = 0; shift < out_topk->count && shift < k; ++shift) {
                if (adjusted > out_topk->entries[shift].adjusted_score) {
                    insert_pos = shift;
                    break;
                }
            }
            /* Shift entries down to make room. */
            if (out_topk->count < k) {
                for (shift = out_topk->count; shift > insert_pos; --shift) {
                    out_topk->entries[shift] = out_topk->entries[shift - 1U];
                }
                ++out_topk->count;
            } else {
                for (shift = k - 1U; shift > insert_pos; --shift) {
                    out_topk->entries[shift] = out_topk->entries[shift - 1U];
                }
            }
            out_topk->entries[insert_pos] = candidate;
        }
    }
}

static lis_status lis_cli_select_generation_token(
    const float *logits,
    size_t vocab_size,
    const lis_model_config *config,
    const lis_tokenizer *tok,
    const size_t *generated_tokens,
    size_t generated_count,
    size_t *out_token_id,
    int *out_should_stop,
    lis_cli_selection_diagnostics *out_selection)
{
    size_t best = 0;
    size_t raw_best = 0;
    size_t index;
    float best_score = 0.0f;
    float raw_best_score = 0.0f;
    float suppressed_best_score = 0.0f;
    size_t runner_up = 0;
    float runner_up_score = -1.0f / 0.0f;
    size_t suppressed_count = 0;
    size_t penalized_count = 0;
    int have_best = 0;
    int have_raw_best = 0;
    int have_suppressed_best = 0;
    int have_runner_up = 0;

    if (logits == NULL || config == NULL || out_token_id == NULL ||
        out_should_stop == NULL || vocab_size == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (generated_tokens == NULL && generated_count > 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (out_selection != NULL) {
        memset(out_selection, 0, sizeof(*out_selection));
    }

    for (index = 0; index < vocab_size; ++index) {
        const int suppressed =
            tok != NULL && lis_cli_token_is_suppressed_control(tok, index);
        const int penalized =
            lis_cli_token_was_generated(generated_tokens, generated_count,
                                        index);
        const float score = lis_cli_apply_repetition_penalty(
            logits[index], generated_tokens, generated_count, index);

        if (suppressed) {
            ++suppressed_count;
            if (!have_suppressed_best || score > suppressed_best_score) {
                suppressed_best_score = score;
                have_suppressed_best = 1;
            }
            continue;
        }
        if (penalized) {
            ++penalized_count;
        }
        if (!have_raw_best || logits[index] > raw_best_score) {
            raw_best = index;
            raw_best_score = logits[index];
            have_raw_best = 1;
        }
        if (!have_best || score > best_score) {
            if (have_best) {
                runner_up = best;
                runner_up_score = best_score;
                have_runner_up = 1;
            }
            best = index;
            best_score = score;
            have_best = 1;
        } else if (!have_runner_up || score > runner_up_score) {
            runner_up = index;
            runner_up_score = score;
            have_runner_up = 1;
        }
    }
    if (!have_best) {
        *out_token_id = 0;
        *out_should_stop = 1;
        return LIS_STATUS_OK;
    }

    *out_token_id = best;
    *out_should_stop = tok != NULL &&
        lis_cli_token_is_structural_stop(config, tok, best);
    if (out_selection != NULL) {
        out_selection->structural_suppression_affected =
            have_suppressed_best && suppressed_best_score > best_score;
        out_selection->repetition_penalty_changed_selection =
            have_raw_best && raw_best != best;
        out_selection->selected_token_penalized =
            lis_cli_token_was_generated(generated_tokens, generated_count,
                                        best);
        out_selection->raw_score_selected =
            have_raw_best && best < vocab_size ? logits[best] : 0.0f;
        out_selection->adjusted_score_selected = best_score;
        out_selection->runner_up_token_id = runner_up;
        out_selection->runner_up_adjusted_score =
            have_runner_up ? runner_up_score : -1.0f / 0.0f;
        out_selection->runner_up_available = have_runner_up;
        out_selection->decision_margin_valid = have_runner_up;
        out_selection->decision_margin =
            have_runner_up ? best_score - runner_up_score : 0.0f;
        out_selection->suppressed_token_count = suppressed_count;
        out_selection->penalized_token_count = penalized_count;
        if (out_selection->structural_suppression_affected) {
            out_selection->decision_class = "structural_suppression";
        } else if (out_selection->repetition_penalty_changed_selection) {
            out_selection->decision_class = "repetition_penalty_shifted";
        } else {
            out_selection->decision_class = "greedy";
        }
    }
    return LIS_STATUS_OK;
}

static void lis_cli_build_trace_step(
    size_t step,
    lis_trace_phase phase,
    size_t selected_token_id,
    const lis_cli_selection_diagnostics *selection,
    const lis_cli_topk_result *topk,
    lis_cli_generation_stop_reason stop_reason,
    int should_stop,
    lis_trace_step *out)
{
    size_t index;

    if (selection == NULL || topk == NULL || out == NULL) {
        return;
    }

    out->step = step;
    out->phase = phase;
    out->selected_token_id = selected_token_id;
    out->raw_score_selected = selection->raw_score_selected;
    out->adjusted_score_selected = selection->adjusted_score_selected;
    out->runner_up_token_id = selection->runner_up_token_id;
    out->runner_up_adjusted_score = selection->runner_up_adjusted_score;
    out->runner_up_available = selection->runner_up_available;
    out->decision_margin = selection->decision_margin;
    out->decision_margin_valid = selection->decision_margin_valid;
    out->structural_suppression_affected =
        selection->structural_suppression_affected;
    out->repetition_penalty_changed_selection =
        selection->repetition_penalty_changed_selection;
    out->selected_token_penalized = selection->selected_token_penalized;
    out->suppressed_token_count = selection->suppressed_token_count;
    out->penalized_token_count = selection->penalized_token_count;
    out->decision_class = selection->decision_class != NULL
        ? selection->decision_class : "greedy";
    out->topk_count = topk->count < LIS_TRACE_TOPK_SIZE
        ? topk->count : LIS_TRACE_TOPK_SIZE;
    for (index = 0; index < out->topk_count; ++index) {
        out->topk[index].token_id = topk->entries[index].token_id;
        out->topk[index].raw_score = topk->entries[index].raw_score;
        out->topk[index].adjusted_score = topk->entries[index].adjusted_score;
        out->topk[index].is_selected = topk->entries[index].is_selected;
    }
    out->has_stop_reason = should_stop ? 1 : 0;
    out->stop_reason = lis_cli_stop_reason_name(stop_reason);
}

static lis_status lis_cli_write_trace_artifact(
    const lis_cli_options *options,
    const lis_loaded_model *model,
    const lis_token_id_batch *batch,
    int has_tokenizer,
    const char *backend_name,
    const char *precision_path,
    const lis_artifact_set_id *artifact_set_id,
    const lis_trace_record *trace_record)
{
    lis_trace_artifact artifact = { 0 };
    char model_path_buffer[1024];
    const char *model_path = NULL;
    const char *input_path = NULL;
    lis_status status;

    if (options == NULL || options->trace_json_path == NULL ||
        model == NULL || batch == NULL || backend_name == NULL ||
        artifact_set_id == NULL || !artifact_set_id->valid ||
        trace_record == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    status = lis_cli_artifact_model_path(options, model->format,
                                          model_path_buffer,
                                          sizeof(model_path_buffer),
                                          &model_path);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    input_path = lis_cli_artifact_input_path(options);
    if (input_path == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    artifact.path = options->trace_json_path;
    artifact.artifact_set_id = artifact_set_id;
    artifact.model_format_name = lis_model_format_name(model->format);
    artifact.model_family_name = lis_model_family_name(model->metadata.config.family);
    artifact.backend_name = backend_name;
    artifact.precision_path = precision_path;
    artifact.options = options;
    artifact.model = model;
    artifact.input_mode = lis_cli_artifact_input_mode(options);
    artifact.output_mode = lis_cli_artifact_output_mode(has_tokenizer);

    status = lis_artifact_fingerprint_current_binary(&artifact.binary_fingerprint);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_artifact_fingerprint_file(model_path, &artifact.model_fingerprint);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_artifact_fingerprint_file(options->config_path,
                                           &artifact.config_fingerprint);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_artifact_fingerprint_file(input_path, &artifact.input_fingerprint);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_artifact_fingerprint_runtime(
        options, model->format, model->metadata.config.family,
        artifact.input_mode, backend_name,
        &artifact.runtime_fingerprint);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_artifact_fingerprint_backend(backend_name,
                                               options->thread_count,
                                               &artifact.backend_fingerprint);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    status = lis_trace_artifact_write(&artifact, trace_record);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    return LIS_STATUS_OK;
}

static lis_status lis_cli_write_layer_trace_artifact(
    const lis_cli_options *options,
    const lis_loaded_model *model,
    const lis_token_id_batch *batch,
    int has_tokenizer,
    const char *backend_name,
    const char *precision_path,
    const lis_artifact_set_id *artifact_set_id,
    const lis_layer_trace_record *layer_trace_record)
{
    lis_layer_trace_artifact artifact = { 0 };
    char model_path_buffer[1024];
    const char *model_path = NULL;
    const char *input_path = NULL;
    lis_status status;

    if (options == NULL || options->layer_trace_json_path == NULL ||
        model == NULL || batch == NULL || backend_name == NULL ||
        artifact_set_id == NULL || !artifact_set_id->valid ||
        layer_trace_record == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    status = lis_cli_artifact_model_path(options, model->format,
                                          model_path_buffer,
                                          sizeof(model_path_buffer),
                                          &model_path);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    input_path = lis_cli_artifact_input_path(options);
    if (input_path == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    artifact.path = options->layer_trace_json_path;
    artifact.artifact_set_id = artifact_set_id;
    artifact.model_format_name = lis_model_format_name(model->format);
    artifact.model_family_name = lis_model_family_name(model->metadata.config.family);
    artifact.backend_name = backend_name;
    artifact.precision_path = precision_path;
    artifact.options = options;
    artifact.model = model;
    artifact.input_mode = lis_cli_artifact_input_mode(options);
    artifact.output_mode = lis_cli_artifact_output_mode(has_tokenizer);

    status = lis_artifact_fingerprint_current_binary(&artifact.binary_fingerprint);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_artifact_fingerprint_file(model_path, &artifact.model_fingerprint);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_artifact_fingerprint_file(options->config_path,
                                           &artifact.config_fingerprint);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_artifact_fingerprint_file(input_path, &artifact.input_fingerprint);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_artifact_fingerprint_runtime(
        options, model->format, model->metadata.config.family,
        artifact.input_mode, backend_name,
        &artifact.runtime_fingerprint);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_artifact_fingerprint_backend(backend_name,
                                               options->thread_count,
                                               &artifact.backend_fingerprint);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    status = lis_layer_trace_artifact_write(&artifact, layer_trace_record);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    return LIS_STATUS_OK;
}

static lis_status lis_cli_emit_generated_tokens(lis_runtime_context *runtime,
                                                 lis_tensor_view projection,
                                                 size_t generation_limit,
                                                 const lis_tokenizer *tok,
                                                 int diagnostics_enabled,
                                                 lis_cli_execution_record *record,
                                                 lis_trace_record *trace_record)
{
    float *logits = NULL;
    size_t *generated_tokens = NULL;
    size_t generated_count = 0;
    const size_t vocab_size = runtime->metadata.config.vocab_size;
    size_t step;
    int emitted_token = 0;
    lis_status status = LIS_STATUS_OK;

    if (runtime == NULL || generation_limit == 0 || vocab_size == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    if (vocab_size > SIZE_MAX / sizeof(*logits)) {
        return LIS_STATUS_OVERFLOW;
    }
    logits = malloc(vocab_size * sizeof(*logits));
    if (logits == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }
    if (generation_limit > SIZE_MAX / sizeof(*generated_tokens)) {
        free(logits);
        return LIS_STATUS_OVERFLOW;
    }
    generated_tokens = malloc(generation_limit * sizeof(*generated_tokens));
    if (generated_tokens == NULL) {
        free(logits);
        return LIS_STATUS_NO_MEMORY;
    }

    for (step = 0; step < generation_limit; ++step) {
        size_t token_id = 0;
        int should_stop = 0;
        lis_cli_generation_stop_reason stop_reason = LIS_CLI_STOP_NONE;
        lis_cli_selection_diagnostics selection = { 0 };
        lis_cli_topk_result topk = { 0 };

        status = lis_cli_compute_validation_logits(runtime, projection,
                                                    logits, vocab_size);
        if (status != LIS_STATUS_OK) {
            break;
        }
        status = lis_cli_select_generation_token(
            logits, vocab_size, &runtime->metadata.config, tok,
            generated_tokens, generated_count, &token_id, &should_stop,
            &selection);
        if (status != LIS_STATUS_OK) {
            break;
        }
        lis_cli_apply_test_selected_token_override(
            step, vocab_size, &token_id, &should_stop);
        if (record != NULL) {
            status = lis_cli_execution_record_append_selected(record, token_id);
            if (status != LIS_STATUS_OK) {
                break;
            }
        }
        if (diagnostics_enabled || trace_record != NULL) {
            lis_cli_extract_topk_candidates(logits, vocab_size, tok,
                                            generated_tokens, generated_count,
                                            token_id, &topk);
        }
        if (lis_model_config_token_is_eos(&runtime->metadata.config,
                                          token_id)) {
            stop_reason = LIS_CLI_STOP_MODEL_EOS;
        } else if (should_stop) {
            stop_reason = LIS_CLI_STOP_STRUCTURAL_CONTROL;
        } else if (step + 1U == generation_limit) {
            stop_reason = LIS_CLI_STOP_DECODE_LIMIT;
        }
        if (trace_record != NULL) {
            lis_trace_phase phase = (step == 0) ? LIS_TRACE_PHASE_FIRST_DECODE
                                                 : LIS_TRACE_PHASE_DECODE;
            lis_trace_step ts = { 0 };

            lis_cli_build_trace_step(step, phase, token_id, &selection, &topk,
                                      stop_reason, should_stop, &ts);
            lis_trace_record_append(trace_record, &ts);
        }
        if (should_stop) {
            if (record != NULL) {
                record->stop_reason = stop_reason;
            }
            if (diagnostics_enabled) {
                lis_cli_emit_generation_diagnostic(step, token_id, tok,
                                                   stop_reason, &selection,
                                                   &topk, "decode");
            }
            break;
        }
        status = lis_runtime_decode_step(runtime);
        if (status != LIS_STATUS_OK) {
            if (diagnostics_enabled) {
                stop_reason = status == LIS_STATUS_LIMIT_EXCEEDED ?
                    LIS_CLI_STOP_CONTEXT_LIMIT : LIS_CLI_STOP_RUNTIME_ERROR;
                lis_cli_emit_generation_diagnostic(step, token_id, tok,
                                                   stop_reason, &selection,
                                                   &topk, "decode");
            }
            if (record != NULL) {
                record->stop_reason = status == LIS_STATUS_LIMIT_EXCEEDED ?
                    LIS_CLI_STOP_CONTEXT_LIMIT : LIS_CLI_STOP_RUNTIME_ERROR;
            }
            break;
        }
        if (diagnostics_enabled) {
            lis_cli_emit_generation_diagnostic(step, token_id, tok,
                                               stop_reason, &selection,
                                               &topk, "decode");
        }
        status = lis_cli_emit_token(token_id, tok, &emitted_token);
        if (status != LIS_STATUS_OK) {
            if (record != NULL) {
                record->stop_reason = LIS_CLI_STOP_RUNTIME_ERROR;
            }
            break;
        }
        if (record != NULL) {
            status = lis_cli_execution_record_append_emitted(record, token_id);
            if (status != LIS_STATUS_OK) {
                break;
            }
        }
        generated_tokens[generated_count++] = token_id;
        if (record != NULL) {
            record->stop_reason = stop_reason;
        }
        if (lis_model_config_token_is_eos(&runtime->metadata.config,
                                          token_id)) {
            break;
        }
    }
    if (tok != NULL) {
        putchar('\n');
        fflush(stdout);
    } else if (emitted_token) {
        putchar('\n');
    }

    free(generated_tokens);
    free(logits);
    return status;
}

static lis_status lis_cli_emit_decoder_tokens(lis_runtime_context *runtime,
                                               const lis_loaded_model *model,
                                               const lis_token_id_batch *batch,
                                               size_t generation_limit,
                                               const lis_tokenizer *tok,
                                               int diagnostics_enabled,
                                               lis_perf_report *perf,
                                               lis_cli_execution_record *record,
                                               lis_trace_record *trace_record)
{
    float *logits = NULL;
    size_t *generated_tokens = NULL;
    size_t generated_count = 0;
    const size_t vocab_size = runtime->metadata.config.vocab_size;
    lis_cli_prefill_fn prefill = NULL;
    lis_cli_decode_fn decode = NULL;
    size_t step;
    int emitted_token = 0;
    uint64_t prompt_tokens_total = 0;
    size_t prompt_idx;
    lis_status status = LIS_STATUS_OK;

    if (runtime == NULL || model == NULL || batch == NULL ||
        generation_limit == 0 || vocab_size == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    status = lis_cli_family_runtime_fns(runtime->metadata.config.family,
                                        &prefill, &decode);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (vocab_size > SIZE_MAX / sizeof(*logits)) {
        return LIS_STATUS_OVERFLOW;
    }
    logits = malloc(vocab_size * sizeof(*logits));
    if (logits == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }
    if (generation_limit > SIZE_MAX / sizeof(*generated_tokens)) {
        free(logits);
        return LIS_STATUS_OVERFLOW;
    }
    generated_tokens = malloc(generation_limit * sizeof(*generated_tokens));
    if (generated_tokens == NULL) {
        free(logits);
        return LIS_STATUS_NO_MEMORY;
    }

    for (prompt_idx = 0; prompt_idx < batch->batch_size; ++prompt_idx) {
        prompt_tokens_total += (uint64_t)batch->lengths[prompt_idx];
    }
    lis_perf_stage_begin(perf, LIS_PERF_STAGE_PREFILL);
    status = prefill(runtime, model, batch->tokens, batch->lengths,
                     batch->batch_size, logits, vocab_size);
    lis_perf_stage_end(perf, LIS_PERF_STAGE_PREFILL, prompt_tokens_total);
    if (status != LIS_STATUS_OK) {
        free(generated_tokens);
        free(logits);
        return status;
    }

    for (step = 0; step < generation_limit; ++step) {
        size_t token_id = 0;
        int should_stop = 0;
        lis_cli_generation_stop_reason stop_reason = LIS_CLI_STOP_NONE;
        lis_cli_selection_diagnostics selection = { 0 };
        lis_cli_topk_result topk = { 0 };
        uint64_t decode_start_ns = 0;
        uint64_t decode_elapsed_ns = 0;

        status = lis_cli_select_generation_token(
            logits, vocab_size, &runtime->metadata.config, tok,
            generated_tokens, generated_count, &token_id, &should_stop,
            &selection);
        if (status != LIS_STATUS_OK) {
            break;
        }
        lis_cli_apply_test_selected_token_override(
            step, vocab_size, &token_id, &should_stop);
        if (record != NULL) {
            status = lis_cli_execution_record_append_selected(record, token_id);
            if (status != LIS_STATUS_OK) {
                break;
            }
        }
        if (diagnostics_enabled || trace_record != NULL) {
            lis_cli_extract_topk_candidates(logits, vocab_size, tok,
                                            generated_tokens, generated_count,
                                            token_id, &topk);
        }
        if (lis_model_config_token_is_eos(&runtime->metadata.config,
                                          token_id)) {
            stop_reason = LIS_CLI_STOP_MODEL_EOS;
        } else if (should_stop) {
            stop_reason = LIS_CLI_STOP_STRUCTURAL_CONTROL;
        } else if (step + 1U == generation_limit) {
            stop_reason = LIS_CLI_STOP_DECODE_LIMIT;
        }
        if (trace_record != NULL) {
            lis_trace_phase phase = (step == 0) ? LIS_TRACE_PHASE_FIRST_DECODE
                                                 : LIS_TRACE_PHASE_DECODE;
            lis_trace_step ts = { 0 };

            lis_cli_build_trace_step(step, phase, token_id, &selection, &topk,
                                      stop_reason, should_stop, &ts);
            lis_trace_record_append(trace_record, &ts);
        }
        if (should_stop) {
            if (record != NULL) {
                record->stop_reason = stop_reason;
            }
            if (diagnostics_enabled) {
                const char *phase = step == 0 ? "first_decode" : "decode";

                lis_cli_emit_generation_diagnostic(step, token_id, tok,
                                                   stop_reason, &selection,
                                                   &topk, phase);
            }
            break;
        }
        if (perf != NULL && perf->enabled) {
            decode_start_ns = lis_perf_now_ns();
        }
        status = decode(runtime, model, token_id, logits, vocab_size);
        if (perf != NULL && perf->enabled) {
            uint64_t now_ns = lis_perf_now_ns();

            decode_elapsed_ns = (now_ns >= decode_start_ns)
                ? now_ns - decode_start_ns : 0;
            lis_perf_emit_per_token(perf, stderr, step, decode_elapsed_ns);
            if (step == 0) {
                lis_perf_stage_accumulate(perf, LIS_PERF_STAGE_FIRST_DECODE,
                                          decode_elapsed_ns, 1);
            } else {
                lis_perf_stage_accumulate(perf,
                                          LIS_PERF_STAGE_DECODE_STEADY_STATE,
                                          decode_elapsed_ns, 1);
            }
        }
        if (status != LIS_STATUS_OK) {
            if (diagnostics_enabled) {
                const char *phase = step == 0 ? "first_decode" : "decode";

                stop_reason = status == LIS_STATUS_LIMIT_EXCEEDED ?
                    LIS_CLI_STOP_CONTEXT_LIMIT : LIS_CLI_STOP_RUNTIME_ERROR;
                lis_cli_emit_generation_diagnostic(step, token_id, tok,
                                                   stop_reason, &selection,
                                                   &topk, phase);
            }
            if (record != NULL) {
                record->stop_reason = status == LIS_STATUS_LIMIT_EXCEEDED ?
                    LIS_CLI_STOP_CONTEXT_LIMIT : LIS_CLI_STOP_RUNTIME_ERROR;
            }
            break;
        }
        if (diagnostics_enabled) {
            const char *phase = step == 0 ? "first_decode" : "decode";

            lis_cli_emit_generation_diagnostic(step, token_id, tok,
                                               stop_reason, &selection,
                                               &topk, phase);
        }
        status = lis_cli_emit_token(token_id, tok, &emitted_token);
        if (status != LIS_STATUS_OK) {
            if (record != NULL) {
                record->stop_reason = LIS_CLI_STOP_RUNTIME_ERROR;
            }
            break;
        }
        if (record != NULL) {
            status = lis_cli_execution_record_append_emitted(record, token_id);
            if (status != LIS_STATUS_OK) {
                break;
            }
        }
        generated_tokens[generated_count++] = token_id;
        if (record != NULL) {
            record->stop_reason = stop_reason;
        }
        if (lis_model_config_token_is_eos(&runtime->metadata.config,
                                          token_id)) {
            break;
        }
    }
    if (tok != NULL) {
        putchar('\n');
        fflush(stdout);
    } else if (emitted_token) {
        putchar('\n');
    }

    free(generated_tokens);
    free(logits);
    return status;
}

/*
 * Parse space-separated token IDs from a string.
 * Caller owns *out_ids and must free() it.
 */
static lis_status lis_cli_parse_forced_prefix(const char *text,
                                              size_t **out_ids,
                                              size_t *out_count)
{
    size_t capacity = 8;
    size_t count = 0;
    size_t *ids = NULL;
    const char *cursor;

    if (text == NULL || out_ids == NULL || out_count == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    *out_ids = NULL;
    *out_count = 0;

    ids = malloc(capacity * sizeof(*ids));
    if (ids == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }

    cursor = text;
    while (*cursor != '\0') {
        char *end = NULL;
        uintmax_t value;

        while (*cursor == ' ') {
            ++cursor;
        }
        if (*cursor == '\0') {
            break;
        }

        errno = 0;
        value = strtoumax(cursor, &end, 10);
        if (errno == ERANGE || end == cursor ||
            (*end != '\0' && *end != ' ') ||
            value > (uintmax_t)SIZE_MAX) {
            free(ids);
            return LIS_STATUS_FORMAT;
        }
        if (count == capacity) {
            size_t new_cap = capacity * 2U;
            size_t *tmp;

            if (new_cap < capacity ||
                new_cap > SIZE_MAX / sizeof(*ids)) {
                free(ids);
                return LIS_STATUS_OVERFLOW;
            }
            tmp = realloc(ids, new_cap * sizeof(*ids));
            if (tmp == NULL) {
                free(ids);
                return LIS_STATUS_NO_MEMORY;
            }
            ids = tmp;
            capacity = new_cap;
        }
        ids[count++] = (size_t)value;
        cursor = end;
    }

    if (count == 0) {
        free(ids);
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    *out_ids = ids;
    *out_count = count;
    return LIS_STATUS_OK;
}

/*
 * Forced-prefix diagnostics: prefill the prompt, feed forced prefix tokens
 * through decode, then report next-step top-k diagnostics.
 * Does not emit normal generation output.
 */
static lis_status lis_cli_run_forced_prefix_diagnostics(
    lis_runtime_context *runtime,
    const lis_loaded_model *model,
    const lis_token_id_batch *batch,
    const size_t *prefix_ids,
    size_t prefix_count,
    const lis_tokenizer *tok,
    lis_perf_report *perf)
{
    float *logits = NULL;
    const size_t vocab_size = runtime->metadata.config.vocab_size;
    size_t step;
    size_t token_id = 0;
    int should_stop = 0;
    lis_cli_selection_diagnostics selection = { 0 };
    lis_cli_topk_result topk = { 0 };
    uint64_t prompt_tokens_total = 0;
    size_t prompt_idx;
    lis_cli_prefill_fn prefill = NULL;
    lis_cli_decode_fn decode = NULL;
    lis_status status;

    if (runtime == NULL || model == NULL || batch == NULL ||
        prefix_ids == NULL || prefix_count == 0 || vocab_size == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    status = lis_cli_family_runtime_fns(runtime->metadata.config.family,
                                        &prefill, &decode);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    logits = malloc(vocab_size * sizeof(*logits));
    if (logits == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }

    /* Prefill the canonical prompt. */
    for (prompt_idx = 0; prompt_idx < batch->batch_size; ++prompt_idx) {
        prompt_tokens_total += (uint64_t)batch->lengths[prompt_idx];
    }
    lis_perf_stage_begin(perf, LIS_PERF_STAGE_PREFILL);
    status = prefill(runtime, model, batch->tokens, batch->lengths,
                     batch->batch_size, logits, vocab_size);
    lis_perf_stage_end(perf, LIS_PERF_STAGE_PREFILL, prompt_tokens_total);
    if (status != LIS_STATUS_OK) {
        free(logits);
        return status;
    }

    /* Feed each forced prefix token through decode. */
    for (step = 0; step < prefix_count; ++step) {
        uint64_t decode_start_ns = 0;
        uint64_t decode_elapsed_ns = 0;

        if (perf != NULL && perf->enabled) {
            decode_start_ns = lis_perf_now_ns();
        }
        status = decode(runtime, model, prefix_ids[step], logits,
                        vocab_size);
        if (perf != NULL && perf->enabled) {
            uint64_t now_ns = lis_perf_now_ns();

            decode_elapsed_ns = (now_ns >= decode_start_ns)
                ? now_ns - decode_start_ns : 0;
            lis_perf_emit_per_token(perf, stderr, step, decode_elapsed_ns);
            if (step == 0) {
                lis_perf_stage_accumulate(perf, LIS_PERF_STAGE_FIRST_DECODE,
                                          decode_elapsed_ns, 1);
            } else {
                lis_perf_stage_accumulate(perf,
                                          LIS_PERF_STAGE_DECODE_STEADY_STATE,
                                          decode_elapsed_ns, 1);
            }
        }
        if (status != LIS_STATUS_OK) {
            fprintf(stderr,
                    "lis: forced-prefix error: decode failed at prefix "
                    "step %zu (token %zu): %s\n",
                    step, prefix_ids[step], lis_status_name(status));
            free(logits);
            return status;
        }
    }

    /* Score the next token after the forced prefix. */
    status = lis_cli_select_generation_token(
        logits, vocab_size, &runtime->metadata.config, tok,
        prefix_ids, prefix_count, &token_id, &should_stop, &selection);
    if (status != LIS_STATUS_OK) {
        free(logits);
        return status;
    }

    lis_cli_extract_topk_candidates(logits, vocab_size, tok,
                                    prefix_ids, prefix_count,
                                    token_id, &topk);

    /* Emit forced-prefix metadata header. */
    fprintf(stderr,
            "lis: forced-prefix-info prompt_tokens=%zu "
            "forced_prefix_tokens=%zu forced_prefix_ids=",
            batch->token_count, prefix_count);
    for (step = 0; step < prefix_count; ++step) {
        fprintf(stderr, "%s%zu", step > 0 ? "," : "", prefix_ids[step]);
    }
    fputc('\n', stderr);

    /* Emit diagnostics for the next-step prediction. */
    {
        lis_cli_generation_stop_reason reason = LIS_CLI_STOP_NONE;

        if (lis_model_config_token_is_eos(&runtime->metadata.config,
                                          token_id)) {
            reason = LIS_CLI_STOP_MODEL_EOS;
        } else if (should_stop) {
            reason = LIS_CLI_STOP_STRUCTURAL_CONTROL;
        }
        lis_cli_emit_generation_diagnostic(prefix_count, token_id, tok,
                                           reason, &selection, &topk,
                                           "forced_prefix_next");
    }

    free(logits);
    return LIS_STATUS_OK;
}

lis_status lis_cli_run_inference(const lis_cli_options *options)
{
    lis_model_source source = { 0 };
    lis_loaded_model model = { 0 };
    lis_model_metadata metadata = { 0 };
    lis_token_id_batch batch = { 0 };
    lis_tokenizer tokenizer = { 0 };
    int has_tokenizer = 0;
    lis_runtime_options runtime_options = { 0 };
    lis_runtime_context runtime = { 0 };
    lis_cli_execution_record execution_record = { 0 };
    lis_trace_record trace_record_data = { 0 };
    lis_trace_record *trace_record = NULL;
    lis_tensor_view projection = { 0 };
    int artifact_requested = 0;
    int artifact_ready = 0;
    int artifact_emitted = 0;
    int use_hf_forward = 0;
    lis_perf_report perf = { 0 };
    lis_layer_trace_record layer_trace_record_data = { 0 };
    lis_layer_trace_record *layer_trace_record = NULL;
    lis_artifact_set_id artifact_set_id = {{0}, 0};
    const char *backend_name = NULL;
    char precision_path[64];
    lis_status status;

    (void)precision_path;
    (void)layer_trace_record;

    if (options == NULL || options->model_path == NULL ||
        options->config_path == NULL ||
        options->context_length == 0 || options->batch_size == 0 ||
        options->generation_limit == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (options->token_path == NULL && options->vocab_path == NULL &&
        options->hf_tokenizer_path == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (options->layer_trace_json_path != NULL &&
        !options->layer_checkpoints_enabled) {
        fprintf(stderr,
                "lis: artifact error: --layer-trace-json requires "
                "--layer-checkpoints\n");
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    status = lis_artifact_set_id_generate(&artifact_set_id);
    if (status != LIS_STATUS_OK) {
        fprintf(stderr,
                "lis: artifact association error: operating-system random "
                "source failed before inference: %s\n",
                lis_status_name(status));
        return status;
    }

    artifact_requested = options->report_json_path != NULL || options->report_md_path != NULL;
    if (options->trace_json_path != NULL) {
        trace_record = &trace_record_data;
        status = lis_trace_record_init(trace_record,
                                       options->generation_limit);
        if (status != LIS_STATUS_OK) {
            fprintf(stderr,
                    "lis: trace error: could not initialize trace record: %s\n",
                    lis_status_name(status));
            return status;
        }
    }
    if (options->layer_trace_json_path != NULL) {
        layer_trace_record = &layer_trace_record_data;
        status = lis_layer_trace_record_init(layer_trace_record,
                                             LIS_LAYER_TRACE_INITIAL_CAPACITY);
        if (status != LIS_STATUS_OK) {
            fprintf(stderr,
                    "lis: artifact error: layer-trace init failed: %s\n",
                    lis_status_name(status));
            return status;
        }
    }
    if (artifact_requested && options->forced_prefix_text != NULL) {
        fprintf(stderr,
                "lis: user-input error: --report-json does not support "
                "--forced-prefix\n");
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    lis_perf_report_init(&perf, options->perf_enabled,
                         options->perf_per_token_enabled, options->perf_tag);

    if (artifact_requested) {
        status = lis_cli_execution_record_init(&execution_record,
                                               options->generation_limit);
        if (status != LIS_STATUS_OK) {
            fprintf(stderr,
                    "lis: artifact error: could not initialize execution "
                    "record: %s\n",
                    lis_status_name(status));
            return status;
        }
    }

    lis_perf_stage_begin(&perf, LIS_PERF_STAGE_MODEL_LOAD);
    source = lis_model_source_from_path(options->model_path);
    status = lis_loader_load(&source, &model);
    if (status != LIS_STATUS_OK) {
        if (status == LIS_STATUS_UNSUPPORTED) {
            lis_cli_report_unsupported_config(options->model_path,
                                              status);
        } else {
            fprintf(stderr, "lis: loading error: could not load model %s: %s\n",
                    options->model_path, lis_status_name(status));
        }
        goto out;
    }
    use_hf_forward = model.format == LIS_MODEL_FORMAT_HUGGINGFACE_LOCAL &&
                     lis_cli_family_uses_hf_forward(
                         model.metadata.config.family);
    if (options->forced_prefix_text != NULL && !use_hf_forward) {
        fprintf(stderr,
                "lis: user-input error: --forced-prefix requires "
                "supported HuggingFace decoder model path\n");
        status = LIS_STATUS_INVALID_ARGUMENT;
        goto out;
    }

    status = lis_cli_load_metadata(options, &metadata);
    if (status != LIS_STATUS_OK) {
        goto out;
    }
    status = lis_loaded_model_attach_metadata(&model, &metadata);
    if (status != LIS_STATUS_OK) {
        fprintf(stderr, "lis: config error: could not attach metadata: %s\n",
                lis_status_name(status));
        goto out;
    }
    if (layer_trace_record != NULL &&
        model.metadata.config.family == LIS_MODEL_FAMILY_LLAMA3_DECODER) {
        status = lis_layer_trace_record_configure_llama_layout(
            layer_trace_record, options->layer_checkpoints_step,
            model.metadata.config.layer_count);
        if (status != LIS_STATUS_OK) {
            fprintf(stderr,
                    "lis: artifact error: layer-trace layout init failed: %s\n",
                    lis_status_name(status));
            goto out;
        }
        if (s_lis_cli_test_injection.observation_perturbation_enabled) {
            layer_trace_record->test_observation_perturbation_enabled = 1;
            layer_trace_record->test_observation_perturbation_layer =
                s_lis_cli_test_injection.observation_perturbation_layer;
            layer_trace_record->test_observation_perturbation_element =
                s_lis_cli_test_injection.observation_perturbation_element;
            layer_trace_record->test_observation_perturbation_delta =
                s_lis_cli_test_injection.observation_perturbation_delta;
        }
    }
    lis_perf_stage_end(&perf, LIS_PERF_STAGE_MODEL_LOAD, 0);

    /* Load input tokens: from token-ID file, LIS vocab, or HF tokenizer. */
    if (options->hf_tokenizer_path != NULL) {
        size_t *encoded_ids = NULL;
        size_t encoded_count = 0;
        char *canonical_prompt = NULL;
        size_t canonical_prompt_len = 0;

        lis_perf_stage_begin(&perf, LIS_PERF_STAGE_TOKENIZER_LOAD);
        status = lis_hf_tokenizer_load(options->hf_tokenizer_path, &tokenizer);
        lis_perf_stage_end(&perf, LIS_PERF_STAGE_TOKENIZER_LOAD, 0);
        if (status != LIS_STATUS_OK) {
            fprintf(stderr,
                    "lis: tokenizer error: could not load HF tokenizer %s: %s\n",
                    options->hf_tokenizer_path, lis_status_name(status));
            goto out;
        }
        has_tokenizer = 1;

        if (options->prompt_text == NULL) {
            fprintf(stderr,
                    "lis: user-input error: --hf-tokenizer requires --prompt\n");
            status = LIS_STATUS_INVALID_ARGUMENT;
            goto out;
        }

        if (model.metadata.config.family == LIS_MODEL_FAMILY_LLAMA3_DECODER) {
            status = lis_cli_build_llama_instruct_prompt(options->prompt_text,
                                                         &canonical_prompt,
                                                         &canonical_prompt_len);
            if (status != LIS_STATUS_OK) {
                fprintf(stderr,
                        "lis: user-input error: prompt construction failed: %s\n",
                        lis_status_name(status));
                goto out;
            }
        } else {
            canonical_prompt = (char *)options->prompt_text;
            canonical_prompt_len = strlen(options->prompt_text);
        }

        lis_perf_stage_begin(&perf, LIS_PERF_STAGE_TOKENIZER_ENCODE);
        status = lis_tokenizer_encode(&tokenizer, canonical_prompt,
                                       canonical_prompt_len,
                                       &encoded_ids, &encoded_count);
        lis_perf_stage_end(&perf, LIS_PERF_STAGE_TOKENIZER_ENCODE,
                           (status == LIS_STATUS_OK) ? encoded_count : 0);
        if (model.metadata.config.family == LIS_MODEL_FAMILY_LLAMA3_DECODER) {
            free(canonical_prompt);
        }
        canonical_prompt = NULL;
        if (status != LIS_STATUS_OK) {
            fprintf(stderr,
                    "lis: tokenizer error: encode failed: %s\n",
                    lis_status_name(status));
            goto out;
        }
        if (encoded_count == 0) {
            fprintf(stderr,
                    "lis: user-input error: prompt encoded to zero tokens\n");
            free(encoded_ids);
            status = LIS_STATUS_INVALID_ARGUMENT;
            goto out;
        }

        batch.tokens = encoded_ids;
        batch.token_count = encoded_count;
        batch.batch_size = 1;
        batch.lengths = malloc(sizeof(*batch.lengths));
        if (batch.lengths == NULL) {
            free(encoded_ids);
            batch.tokens = NULL;
            batch.token_count = 0;
            status = LIS_STATUS_NO_MEMORY;
            goto out;
        }
        batch.lengths[0] = encoded_count;
    } else if (options->vocab_path != NULL) {
        size_t *encoded_ids = NULL;
        size_t encoded_count = 0;

        lis_perf_stage_begin(&perf, LIS_PERF_STAGE_TOKENIZER_LOAD);
        status = lis_tokenizer_load(options->vocab_path, &tokenizer);
        lis_perf_stage_end(&perf, LIS_PERF_STAGE_TOKENIZER_LOAD, 0);
        if (status != LIS_STATUS_OK) {
            fprintf(stderr,
                    "lis: tokenizer error: could not load vocab %s: %s\n",
                    options->vocab_path, lis_status_name(status));
            goto out;
        }
        has_tokenizer = 1;

        if (options->prompt_text == NULL) {
            fprintf(stderr,
                    "lis: user-input error: --vocab requires --prompt\n");
            status = LIS_STATUS_INVALID_ARGUMENT;
            goto out;
        }

        lis_perf_stage_begin(&perf, LIS_PERF_STAGE_TOKENIZER_ENCODE);
        status = lis_tokenizer_encode(&tokenizer, options->prompt_text,
                                       strlen(options->prompt_text),
                                       &encoded_ids, &encoded_count);
        lis_perf_stage_end(&perf, LIS_PERF_STAGE_TOKENIZER_ENCODE,
                           (status == LIS_STATUS_OK) ? encoded_count : 0);
        if (status != LIS_STATUS_OK) {
            fprintf(stderr,
                    "lis: tokenizer error: encode failed: %s\n",
                    lis_status_name(status));
            goto out;
        }
        if (encoded_count == 0) {
            fprintf(stderr,
                    "lis: user-input error: prompt encoded to zero tokens\n");
            free(encoded_ids);
            status = LIS_STATUS_INVALID_ARGUMENT;
            goto out;
        }

        batch.tokens = encoded_ids;
        batch.token_count = encoded_count;
        batch.batch_size = 1;
        batch.lengths = malloc(sizeof(*batch.lengths));
        if (batch.lengths == NULL) {
            free(encoded_ids);
            batch.tokens = NULL;
            batch.token_count = 0;
            status = LIS_STATUS_NO_MEMORY;
            goto out;
        }
        batch.lengths[0] = encoded_count;
    } else {
        status = lis_token_id_batch_load_path(options->token_path,
                                              options->batch_size, &batch);
        if (status != LIS_STATUS_OK) {
            fprintf(stderr,
                    "lis: user-input error: invalid token ID batch: %s\n",
                    lis_status_name(status));
            goto out;
        }
    }

    status = lis_token_id_batch_validate_vocab(&batch,
                                               metadata.config.vocab_size);
    if (status != LIS_STATUS_OK) {
        fprintf(stderr, "lis: user-input error: token ID outside vocab: %s\n",
                lis_status_name(status));
        goto out;
    }
    if (!use_hf_forward) {
        status = lis_cli_find_validation_logits(&model,
                                                metadata.config.vocab_size,
                                                &projection);
        if (status != LIS_STATUS_OK) {
            fprintf(stderr,
                    "lis: loading error: missing or invalid lis.validation_logits tensor: %s\n",
                    lis_status_name(status));
            goto out;
        }
    } else if (options->batch_size != 1 || batch.batch_size != 1) {
        fprintf(stderr,
                "lis: runtime error: HuggingFace decoder execution currently supports batch 1\n");
        status = LIS_STATUS_UNSUPPORTED_SHAPE;
        goto out;
    }

    status = lis_runtime_options_init(&runtime_options, &model.metadata,
                                      lis_backend_cpu_reference(),
                                      options->batch_size);
    if (status != LIS_STATUS_OK) {
        fprintf(stderr, "lis: runtime error: invalid runtime options: %s\n",
                lis_status_name(status));
        goto out;
    }
    runtime_options.thread_count = options->thread_count;
    runtime_options.layer_checkpoints_enabled = options->layer_checkpoints_enabled;
    runtime_options.layer_checkpoints_target_step = options->layer_checkpoints_step;
    runtime_options.layer_trace_record = layer_trace_record;
    lis_perf_stage_begin(&perf, LIS_PERF_STAGE_RUNTIME_INIT);
    status = lis_runtime_init(&runtime, &runtime_options);
    lis_perf_stage_end(&perf, LIS_PERF_STAGE_RUNTIME_INIT, 0);
    if (status != LIS_STATUS_OK) {
        fprintf(stderr, "lis: runtime error: initialization failed: %s\n",
                lis_status_name(status));
        goto out;
    }
    backend_name = lis_cpu_dispatch_backend_name();
    artifact_ready = artifact_requested;
    {
        const char *weight_dtype_name = lis_dtype_name(
            model.metadata.config.weight_dtype);
        const char *kv_dtype_name = lis_dtype_name(
            runtime.kv_cache.layout.dtype);

        if (snprintf(precision_path, sizeof(precision_path),
                     "f32_accum;weights=%s;kv=%s",
                     weight_dtype_name, kv_dtype_name)
            >= (int)sizeof(precision_path)) {
            precision_path[0] = '\0';
        }
    }
    if (options->diagnostics_enabled || perf.enabled) {
        fprintf(stderr, "lis: simd backend=%s\n",
                backend_name);
        fprintf(stderr, "lis: precision path=f32_accum weights=%s kv=%s\n",
                lis_dtype_name(model.metadata.config.weight_dtype),
                lis_dtype_name(runtime.kv_cache.layout.dtype));
    }
    if (use_hf_forward && options->forced_prefix_text != NULL) {
        size_t *prefix_ids = NULL;
        size_t prefix_count = 0;

        status = lis_cli_parse_forced_prefix(options->forced_prefix_text,
                                              &prefix_ids, &prefix_count);
        if (status != LIS_STATUS_OK) {
            fprintf(stderr,
                    "lis: user-input error: invalid forced prefix: %s\n",
                    lis_status_name(status));
            goto out;
        }
        status = lis_cli_run_forced_prefix_diagnostics(
            &runtime, &model, &batch, prefix_ids, prefix_count,
            has_tokenizer ? &tokenizer : NULL, &perf);
        free(prefix_ids);
    } else if (use_hf_forward) {
        status = lis_cli_emit_decoder_tokens(&runtime, &model, &batch,
                                              options->generation_limit,
                                              has_tokenizer ? &tokenizer : NULL,
                                              options->diagnostics_enabled,
                                              &perf,
                                              artifact_requested ?
                                                  &execution_record : NULL,
                                              trace_record);
    } else {
        status = lis_runtime_prefill(&runtime, batch.lengths,
                                      batch.batch_size);
        if (status == LIS_STATUS_OK) {
            status = lis_cli_emit_generated_tokens(
                &runtime, projection, options->generation_limit,
                has_tokenizer ? &tokenizer : NULL,
                options->diagnostics_enabled,
                artifact_requested ? &execution_record : NULL,
                trace_record);
        }
    }
    if (status != LIS_STATUS_OK) {
        if (status == LIS_STATUS_LIMIT_EXCEEDED &&
            lis_cli_runtime_hit_context_limit(&runtime)) {
            if (artifact_requested &&
                execution_record.stop_reason == LIS_CLI_STOP_NONE) {
                execution_record.stop_reason = LIS_CLI_STOP_CONTEXT_LIMIT;
            }
            lis_cli_report_context_limit(&runtime, &batch);
        } else if (artifact_requested &&
                   execution_record.stop_reason == LIS_CLI_STOP_NONE) {
            execution_record.stop_reason = LIS_CLI_STOP_RUNTIME_ERROR;
        }
        fprintf(stderr, "lis: runtime error: decode/output failed: %s\n",
                lis_status_name(status));
        goto out;
    }

    if (perf.enabled) {
        size_t prompt_tokens_total = 0;
        size_t generated_tokens_total = 0;
        size_t i;

        for (i = 0; i < batch.batch_size; ++i) {
            prompt_tokens_total += batch.lengths[i];
        }
        generated_tokens_total =
            (size_t)(perf.stage_tokens[LIS_PERF_STAGE_FIRST_DECODE] +
                     perf.stage_tokens[LIS_PERF_STAGE_DECODE_STEADY_STATE]);
        lis_perf_report_emit(&perf, stderr, (int)options->thread_count,
                             prompt_tokens_total, generated_tokens_total);
    }

    if (options->diagnostics_enabled) {
        lis_artifact_kv_cache_report kv_cache_report = { 0 };

        status = lis_cli_build_kv_cache_report(&runtime, &kv_cache_report);
        if (status != LIS_STATUS_OK) {
            fprintf(stderr,
                    "lis: runtime error: kv-cache diagnostic failed: %s\n",
                    lis_status_name(status));
            goto out;
        }
        lis_cli_emit_kv_cache_diagnostic(&kv_cache_report);
    }

    if (artifact_requested) {
        if (!artifact_ready) {
            fprintf(stderr,
                    "lis: artifact error: report-json requested but required "
                    "identity was not captured\n");
            status = LIS_STATUS_INVALID_ARGUMENT;
            goto out;
        }
        status = lis_cli_write_execution_artifact(options, &model, &batch,
                                                  &runtime,
                                                  has_tokenizer, backend_name,
                                                  precision_path,
                                                  &artifact_set_id,
                                                  &execution_record,
                                                  status, &perf);
        if (status != LIS_STATUS_OK) {
            fprintf(stderr,
                    "lis: artifact error: report emission failed: %s\n",
                    lis_status_name(status));
            goto out;
        }
        artifact_emitted = 1;
    }

    if (trace_record != NULL) {
        lis_status trace_status;

        trace_record_data.steps = trace_record_data.steps;
        trace_status = lis_cli_write_trace_artifact(
            options, &model, &batch, has_tokenizer, backend_name,
            precision_path,
            &artifact_set_id,
            trace_record);
        if (trace_status != LIS_STATUS_OK) {
            fprintf(stderr,
                    "lis: trace error: trace artifact write failed: %s\n",
                    lis_status_name(trace_status));
            if (status == LIS_STATUS_OK) {
                status = trace_status;
            }
        }
    }

    if (layer_trace_record != NULL) {
        lis_status layer_trace_status;

        if (layer_trace_record->append_failed != 0) {
            fprintf(stderr,
                    "lis: artifact error: layer-trace capture overflow; "
                    "partial artifact suppressed\n");
            if (status == LIS_STATUS_OK) {
                status = LIS_STATUS_OVERFLOW;
            }
        } else {
            layer_trace_status = lis_cli_write_layer_trace_artifact(
                options, &model, &batch, has_tokenizer, backend_name,
                precision_path,
                &artifact_set_id,
                layer_trace_record);
            if (layer_trace_status != LIS_STATUS_OK) {
                fprintf(stderr,
                        "lis: artifact error: layer-trace emission failed: %s\n",
                        lis_status_name(layer_trace_status));
                if (status == LIS_STATUS_OK) {
                    status = layer_trace_status;
                }
            }
        }
    }

out:
    if (artifact_requested && artifact_ready && !artifact_emitted &&
        options->forced_prefix_text == NULL) {
        lis_status artifact_status;

        if (execution_record.stop_reason == LIS_CLI_STOP_NONE &&
            status != LIS_STATUS_OK) {
            execution_record.stop_reason = LIS_CLI_STOP_RUNTIME_ERROR;
        }
        artifact_status = lis_cli_write_execution_artifact(
            options, &model, &batch, &runtime, has_tokenizer, backend_name,
            precision_path,
            &artifact_set_id,
            &execution_record, status, &perf);
        if (artifact_status != LIS_STATUS_OK) {
            fprintf(stderr,
                    "lis: artifact error: report emission failed: %s\n",
                    lis_status_name(artifact_status));
            if (status == LIS_STATUS_OK) {
                status = artifact_status;
            }
        }
    } else if (artifact_requested && !artifact_ready &&
               options->forced_prefix_text == NULL) {
        fprintf(stderr,
                "lis: artifact error: report-json could not capture required "
                "identity before failure\n");
        if (status == LIS_STATUS_OK) {
            status = LIS_STATUS_INVALID_ARGUMENT;
        }
    }
    lis_runtime_destroy(&runtime);
    lis_token_id_batch_destroy(&batch);
    if (has_tokenizer) {
        lis_tokenizer_destroy(&tokenizer);
    }
    lis_loaded_model_destroy(&model);
    lis_cli_execution_record_destroy(&execution_record);
    lis_trace_record_destroy(&trace_record_data);
    if (layer_trace_record != NULL) {
        lis_layer_trace_record_destroy(layer_trace_record);
    }
    return status;
}
