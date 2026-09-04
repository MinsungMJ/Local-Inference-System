#include "lis/artifact.h"

#include <inttypes.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/random.h>

#define LIS_ARTIFACT_FNV64_OFFSET UINT64_C(14695981039346656037)
#define LIS_ARTIFACT_FNV64_PRIME UINT64_C(1099511628211)

static uint64_t lis_artifact_hash_init(void)
{
    return LIS_ARTIFACT_FNV64_OFFSET;
}

static lis_status lis_artifact_os_random_source(void *context,
                                                unsigned char *buffer,
                                                size_t size)
{
    size_t offset = 0;

    (void)context;
    if (buffer == NULL || size == 0U) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    while (offset < size) {
        ssize_t got = getrandom(buffer + offset, size - offset, 0);

        if (got < 0) {
            if (errno == EINTR) {
                continue;
            }
            return LIS_STATUS_IO;
        }
        if (got == 0) {
            return LIS_STATUS_IO;
        }
        offset += (size_t)got;
    }
    return LIS_STATUS_OK;
}

lis_status lis_artifact_set_id_generate_with_source(
    lis_artifact_set_id *out,
    lis_artifact_random_source_fn source,
    void *source_context)
{
    static const char hex[] = "0123456789abcdef";
    unsigned char random_bytes[LIS_ARTIFACT_SET_ID_RANDOM_BYTES];
    size_t index;
    lis_status status;

    if (out == NULL || source == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));
    status = source(source_context, random_bytes, sizeof(random_bytes));
    if (status != LIS_STATUS_OK) {
        memset(random_bytes, 0, sizeof(random_bytes));
        return status;
    }
    memcpy(out->value, LIS_ARTIFACT_SET_ID_PREFIX,
           sizeof(LIS_ARTIFACT_SET_ID_PREFIX) - 1U);
    for (index = 0; index < sizeof(random_bytes); ++index) {
        const size_t output_index =
            sizeof(LIS_ARTIFACT_SET_ID_PREFIX) - 1U + index * 2U;

        out->value[output_index] = hex[random_bytes[index] >> 4U];
        out->value[output_index + 1U] = hex[random_bytes[index] & 0x0fU];
    }
    out->value[LIS_ARTIFACT_SET_ID_LEN] = '\0';
    out->valid = 1;
    memset(random_bytes, 0, sizeof(random_bytes));
    return LIS_STATUS_OK;
}

lis_status lis_artifact_set_id_generate(lis_artifact_set_id *out)
{
    return lis_artifact_set_id_generate_with_source(
        out, lis_artifact_os_random_source, NULL);
}

static const char *lis_artifact_model_format_name(lis_model_format format)
{
    switch (format) {
    case LIS_MODEL_FORMAT_SAFETENSORS:
        return "safetensors";
    case LIS_MODEL_FORMAT_HUGGINGFACE_LOCAL:
        return "huggingface_local";
    case LIS_MODEL_FORMAT_PYTORCH_UNSUPPORTED:
        return "pytorch_unsupported";
    case LIS_MODEL_FORMAT_UNKNOWN:
        return "unknown";
    }
    return "unknown";
}

static void lis_artifact_hash_bytes(uint64_t *state,
                                    const unsigned char *data,
                                    size_t len)
{
    size_t index;

    if (state == NULL || data == NULL) {
        return;
    }
    for (index = 0; index < len; ++index) {
        *state ^= (uint64_t)data[index];
        *state *= LIS_ARTIFACT_FNV64_PRIME;
    }
}

static void lis_artifact_hash_cstr(uint64_t *state, const char *text)
{
    if (state == NULL || text == NULL) {
        return;
    }
    lis_artifact_hash_bytes(state, (const unsigned char *)text, strlen(text));
}

static void lis_artifact_hash_u64_le(uint64_t *state, uint64_t value)
{
    unsigned char bytes[8];
    size_t index;

    if (state == NULL) {
        return;
    }
    for (index = 0; index < 8; ++index) {
        bytes[index] = (unsigned char)((value >> (index * 8U)) & 0xffU);
    }
    lis_artifact_hash_bytes(state, bytes, sizeof(bytes));
}

static void lis_artifact_write_fp(FILE *fp,
                                  const lis_artifact_fingerprint *fingerprint)
{
    char hex[LIS_ARTIFACT_DIGEST_HEX_LEN + 1U];

    lis_artifact_digest_hex(fingerprint, hex);
    fprintf(fp,
            "{\"algorithm\":\"fnv1a64\",\"hex\":\"%s\",\"size_bytes\":%zu}",
            hex, fingerprint != NULL ? fingerprint->size_bytes : 0U);
}

static void lis_artifact_write_json_string(FILE *fp, const char *text)
{
    const unsigned char *cursor = (const unsigned char *)text;

    if (fp == NULL || text == NULL) {
        return;
    }
    fputc('"', fp);
    while (*cursor != '\0') {
        unsigned char ch = *cursor++;

        switch (ch) {
        case '\\':
            fputs("\\\\", fp);
            break;
        case '"':
            fputs("\\\"", fp);
            break;
        case '\b':
            fputs("\\b", fp);
            break;
        case '\f':
            fputs("\\f", fp);
            break;
        case '\n':
            fputs("\\n", fp);
            break;
        case '\r':
            fputs("\\r", fp);
            break;
        case '\t':
            fputs("\\t", fp);
            break;
        default:
            if (ch < 0x20U) {
                fprintf(fp, "\\u%04x", (unsigned int)ch);
            } else {
                fputc((int)ch, fp);
            }
            break;
        }
    }
    fputc('"', fp);
}

static int lis_artifact_sha256_id_is_valid(const char *value)
{
    size_t index;

    if (value == NULL || strncmp(value, "sha256:", 7U) != 0) {
        return 0;
    }
    for (index = 7U; index < LIS_SHA256_ID_TEXT_LEN; ++index) {
        const char ch = value[index];

        if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f'))) {
            return 0;
        }
    }
    return value[LIS_SHA256_ID_TEXT_LEN] == '\0';
}

static int lis_artifact_forced_prefix_is_valid(
    const lis_artifact_forced_prefix_report *forced)
{
    const char *expected_policy_sha = NULL;

    if (forced == NULL) {
        return 1;
    }
    if (!forced->valid || !forced->applied ||
        forced->mode == NULL ||
        strcmp(forced->mode, LIS_FORCED_PREFIX_MODE) != 0 ||
        forced->token_count == 0U ||
        forced->token_count > LIS_FORCED_PREFIX_MAX_TOKENS ||
        forced->prefix_start_generated_step != 0U ||
        forced->prefix_end_generated_step_exclusive != forced->token_count ||
        forced->target_generated_token_step != forced->token_count ||
        forced->runtime_checkpoint_step != forced->token_count + 1U ||
        forced->prompt_token_count > SIZE_MAX - forced->token_count ||
        forced->context_position !=
            forced->prompt_token_count + forced->token_count) {
        return 0;
    }
    if (forced->selection_policy != NULL &&
        strcmp(forced->selection_policy,
               LIS_SELECTION_POLICY_RAW_GREEDY) == 0) {
        expected_policy_sha = LIS_SELECTION_POLICY_RAW_GREEDY_SHA256;
    } else if (forced->selection_policy != NULL &&
               strcmp(forced->selection_policy,
                      LIS_SELECTION_POLICY_MODIFIED_GREEDY) == 0) {
        expected_policy_sha = LIS_SELECTION_POLICY_MODIFIED_GREEDY_SHA256;
    } else {
        return 0;
    }
    return lis_artifact_sha256_id_is_valid(forced->token_ids_sha256) &&
           lis_artifact_sha256_id_is_valid(
               forced->source_pass0_artifact_sha256) &&
           lis_artifact_sha256_id_is_valid(
               forced->source_original_run_report_sha256) &&
           lis_artifact_sha256_id_is_valid(
               forced->source_pass1_artifact_sha256) &&
           lis_artifact_sha256_id_is_valid(
               forced->source_localization_ref_sha256) &&
           forced->selection_policy_sha256 != NULL &&
           strcmp(forced->selection_policy_sha256, expected_policy_sha) == 0;
}

static void lis_artifact_write_forced_prefix_json(
    FILE *fp,
    const lis_artifact_forced_prefix_report *forced)
{
    fputs(",\"forced_prefix\":{\"mode\":", fp);
    lis_artifact_write_json_string(fp, forced->mode);
    fputs(",\"applied\":true", fp);
    fprintf(fp,
            ",\"token_count\":%zu,\"token_ids_sha256\":",
            forced->token_count);
    lis_artifact_write_json_string(fp, forced->token_ids_sha256);
    fprintf(fp,
            ",\"prefix_start_generated_step\":%zu"
            ",\"prefix_end_generated_step_exclusive\":%zu"
            ",\"target_generated_token_step\":%zu"
            ",\"runtime_checkpoint_step\":%zu"
            ",\"prompt_token_count\":%zu"
            ",\"context_position\":%zu"
            ",\"selection_policy\":",
            forced->prefix_start_generated_step,
            forced->prefix_end_generated_step_exclusive,
            forced->target_generated_token_step,
            forced->runtime_checkpoint_step,
            forced->prompt_token_count,
            forced->context_position);
    lis_artifact_write_json_string(fp, forced->selection_policy);
    fputs(",\"selection_policy_sha256\":", fp);
    lis_artifact_write_json_string(fp, forced->selection_policy_sha256);
    fputs(",\"source_pass0_artifact_sha256\":", fp);
    lis_artifact_write_json_string(fp, forced->source_pass0_artifact_sha256);
    fputs(",\"source_original_run_report_sha256\":", fp);
    lis_artifact_write_json_string(
        fp, forced->source_original_run_report_sha256);
    fputs(",\"source_pass1_artifact_sha256\":", fp);
    lis_artifact_write_json_string(fp, forced->source_pass1_artifact_sha256);
    fputs(",\"source_localization_ref_sha256\":", fp);
    lis_artifact_write_json_string(
        fp, forced->source_localization_ref_sha256);
    fputc('}', fp);
}

static void lis_artifact_write_bool(FILE *fp, int value)
{
    fputs(value ? "true" : "false", fp);
}

static void lis_artifact_write_perf_object(FILE *fp,
                                           const lis_artifact_run_report *report)
{
    static const char *stage_names[LIS_PERF_STAGE_COUNT] = {
        "model_load",
        "tokenizer_load",
        "tokenizer_encode",
        "runtime_init",
        "prefill",
        "first_decode",
        "decode_steady_state"
    };
    const lis_perf_report *perf = report->perf;
    uint64_t encode_ns;
    uint64_t runtime_init_ns;
    uint64_t prefill_ns;
    uint64_t first_decode_ns;
    uint64_t steady_ns;
    uint64_t steady_tokens;
    uint64_t ttft_ns;
    uint64_t end_to_end_ns;
    double itl_ms;
    double tps_steady;
    double tps_end_to_end;
    size_t index;
    size_t prompt_tokens = 0;

    if (fp == NULL || report == NULL || perf == NULL || !perf->enabled) {
        return;
    }
    for (index = 0; index < report->prompt_sequence_count; ++index) {
        prompt_tokens += report->prompt_sequences[index].token_count;
    }
    encode_ns = perf->stage_ns[LIS_PERF_STAGE_TOKENIZER_ENCODE];
    runtime_init_ns = perf->stage_ns[LIS_PERF_STAGE_RUNTIME_INIT];
    prefill_ns = perf->stage_ns[LIS_PERF_STAGE_PREFILL];
    first_decode_ns = perf->stage_ns[LIS_PERF_STAGE_FIRST_DECODE];
    steady_ns = perf->stage_ns[LIS_PERF_STAGE_DECODE_STEADY_STATE];
    steady_tokens = perf->stage_tokens[LIS_PERF_STAGE_DECODE_STEADY_STATE];
    ttft_ns = encode_ns + runtime_init_ns + prefill_ns + first_decode_ns;
    end_to_end_ns = prefill_ns + first_decode_ns + steady_ns;
    itl_ms = (steady_tokens > 0U)
        ? ((double)steady_ns / 1.0e6 / (double)steady_tokens)
        : 0.0;
    tps_steady = (steady_ns > 0U)
        ? ((double)steady_tokens * 1.0e9 / (double)steady_ns)
        : 0.0;
    tps_end_to_end = (end_to_end_ns > 0U && report->emitted_token_count > 0U)
        ? ((double)report->emitted_token_count * 1.0e9 /
           (double)end_to_end_ns)
        : 0.0;

    fputs(",\"perf\":{", fp);
    lis_artifact_write_json_string(fp, "tag");
    fputc(':', fp);
    lis_artifact_write_json_string(fp, perf->tag != NULL ? perf->tag : "none");
    fprintf(fp,
            ",\"threads\":%zu,\"prompt_tokens\":%zu,\"generated_tokens\":%zu,"
            "\"ttft_ms\":%.3f,\"itl_ms\":%.3f,\"tps_steady\":%.3f,"
            "\"tps_end_to_end\":%.3f,\"stages\":{",
            report->options != NULL ? report->options->thread_count : 0U,
            prompt_tokens,
            report->emitted_token_count,
            (double)ttft_ns / 1.0e6,
            itl_ms,
            tps_steady,
            tps_end_to_end);
    for (index = 0; index < LIS_PERF_STAGE_COUNT; ++index) {
        if (index > 0U) {
            fputc(',', fp);
        }
        lis_artifact_write_json_string(fp, stage_names[index]);
        fprintf(fp,
                ":{\"ns\":%" PRIu64 ",\"ms\":%.3f,\"tokens\":%" PRIu64 "}",
                perf->stage_ns[index],
                (double)perf->stage_ns[index] / 1.0e6,
                perf->stage_tokens[index]);
    }
    fputs("}}", fp);
}

static void lis_artifact_write_kv_cache_object(
    FILE *fp,
    const lis_artifact_kv_cache_report *kv_cache)
{
    if (fp == NULL || kv_cache == NULL || !kv_cache->valid) {
        return;
    }

    fputs(",\"kv_cache\":{\"scope\":\"run_local\",\"policy\":{"
          "\"eviction_free\":true,\"monotonic_growth\":true,"
          "\"paging\":false,\"offload\":false,\"sliding_window\":false,"
          "\"prefix_reuse\":false},\"storage_dtype\":", fp);
    lis_artifact_write_json_string(fp, lis_dtype_name(kv_cache->storage_dtype));
    fprintf(fp,
            ",\"max_tokens\":%zu,\"used_tokens\":%zu,"
            "\"bytes_per_token\":%zu,\"allocated_bytes\":%zu,"
            "\"used_bytes\":%zu,\"shape\":{\"layer_count\":%zu,"
            "\"batch_size\":%zu,\"kv_head_count\":%zu,\"head_dim\":%zu,"
            "\"element_size\":%zu}}",
            kv_cache->max_tokens,
            kv_cache->used_tokens,
            kv_cache->bytes_per_token,
            kv_cache->allocated_bytes,
            kv_cache->used_bytes,
            kv_cache->layer_count,
            kv_cache->batch_size,
            kv_cache->kv_head_count,
            kv_cache->head_dim,
            kv_cache->element_size);
}

const char *lis_artifact_input_mode_name(lis_artifact_input_mode mode)
{
    switch (mode) {
    case LIS_ARTIFACT_INPUT_MODE_TOKENS:
        return "tokens";
    case LIS_ARTIFACT_INPUT_MODE_VOCAB_PROMPT:
        return "vocab_prompt";
    case LIS_ARTIFACT_INPUT_MODE_HF_TOKENIZER_PROMPT:
        return "hf_tokenizer_prompt";
    }
    return "unknown";
}

const char *lis_artifact_output_mode_name(lis_artifact_output_mode mode)
{
    switch (mode) {
    case LIS_ARTIFACT_OUTPUT_MODE_TOKEN_IDS:
        return "token_ids";
    case LIS_ARTIFACT_OUTPUT_MODE_TEXT:
        return "text";
    }
    return "unknown";
}

void lis_artifact_digest_hex(const lis_artifact_fingerprint *fingerprint,
                             char out_hex[LIS_ARTIFACT_DIGEST_HEX_LEN + 1U])
{
    if (out_hex == NULL) {
        return;
    }
    if (fingerprint == NULL || !fingerprint->valid) {
        strcpy(out_hex, "0000000000000000");
        return;
    }
    snprintf(out_hex, LIS_ARTIFACT_DIGEST_HEX_LEN + 1U, "%016" PRIx64,
             fingerprint->digest);
}

lis_status lis_artifact_fingerprint_file(const char *path,
                                         lis_artifact_fingerprint *out)
{
    unsigned char buf[4096];
    uint64_t digest = lis_artifact_hash_init();
    FILE *fp = NULL;
    size_t total = 0;

    if (path == NULL || out == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));

    fp = fopen(path, "rb");
    if (fp == NULL) {
        return LIS_STATUS_IO;
    }
    for (;;) {
        size_t got = fread(buf, 1, sizeof(buf), fp);

        if (got > 0U) {
            lis_artifact_hash_bytes(&digest, buf, got);
            total += got;
        }
        if (got < sizeof(buf)) {
            if (ferror(fp)) {
                fclose(fp);
                return LIS_STATUS_IO;
            }
            break;
        }
    }
    if (fclose(fp) != 0) {
        return LIS_STATUS_IO;
    }
    out->digest = digest;
    out->size_bytes = total;
    out->valid = 1;
    return LIS_STATUS_OK;
}

lis_status lis_artifact_fingerprint_current_binary(lis_artifact_fingerprint *out)
{
    if (out == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    return lis_artifact_fingerprint_file("/proc/self/exe", out);
}

lis_status lis_artifact_fingerprint_token_ids(const size_t *ids,
                                              size_t count,
                                              lis_artifact_fingerprint *out)
{
    uint64_t digest = lis_artifact_hash_init();
    size_t index;

    if (out == NULL || (ids == NULL && count > 0U)) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));
    lis_artifact_hash_u64_le(&digest, (uint64_t)count);
    for (index = 0; index < count; ++index) {
        lis_artifact_hash_u64_le(&digest, (uint64_t)ids[index]);
    }
    out->digest = digest;
    out->size_bytes = count * sizeof(uint64_t);
    out->valid = 1;
    return LIS_STATUS_OK;
}

lis_status lis_artifact_fingerprint_runtime(
    const lis_cli_options *options,
    lis_model_format model_format,
    lis_model_family family,
    lis_artifact_input_mode input_mode,
    const char *backend_name,
    lis_artifact_fingerprint *out)
{
    uint64_t digest = lis_artifact_hash_init();

    if (options == NULL || backend_name == NULL || out == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));
    lis_artifact_hash_cstr(&digest,
                           lis_artifact_model_format_name(model_format));
    lis_artifact_hash_cstr(&digest, lis_model_family_name(family));
    lis_artifact_hash_cstr(&digest, lis_artifact_input_mode_name(input_mode));
    lis_artifact_hash_cstr(&digest, backend_name);
    lis_artifact_hash_u64_le(&digest, (uint64_t)options->context_length);
    lis_artifact_hash_u64_le(&digest, (uint64_t)options->batch_size);
    lis_artifact_hash_u64_le(&digest, (uint64_t)options->generation_limit);
    lis_artifact_hash_u64_le(&digest, (uint64_t)options->thread_count);
    lis_artifact_hash_u64_le(&digest, (uint64_t)options->diagnostics_enabled);
    lis_artifact_hash_u64_le(&digest, (uint64_t)options->perf_enabled);
    lis_artifact_hash_u64_le(&digest,
                             (uint64_t)options->perf_per_token_enabled);
    lis_artifact_hash_u64_le(&digest,
                             (uint64_t)options->layer_checkpoints_enabled);
    lis_artifact_hash_u64_le(&digest,
                             (uint64_t)options->layer_checkpoints_step);
    if (options->intra_layer_checkpoints_enabled) {
        lis_artifact_hash_u64_le(
            &digest, (uint64_t)options->intra_layer_checkpoints_enabled);
        lis_artifact_hash_u64_le(
            &digest, (uint64_t)options->intra_layer_target_layer);
        lis_artifact_hash_cstr(
            &digest, LIS_INTRA_LAYER_DIAGNOSTIC_CAPTURE_PROFILE);
    }
    lis_artifact_hash_u64_le(&digest,
                             (uint64_t)(options->forced_prefix_text != NULL));
    /*
     * Generation semantics depend on the fixed repetition penalty. Hash the literal
     * string instead of a binary float encoding.
     */
    lis_artifact_hash_cstr(&digest, "repetition_penalty=1.2");
    {
        size_t sz = 0;
        /* Four strings, ten numeric/boolean fields, and the repetition-penalty
         * string form the legacy stream. The three intra-layer values are
         * appended only for the enabled semantic profile. */
        sz += strlen(lis_artifact_model_format_name(model_format)) + 1;
        sz += strlen(lis_model_family_name(family)) + 1;
        sz += strlen(lis_artifact_input_mode_name(input_mode)) + 1;
        sz += strlen(backend_name) + 1;
        sz += sizeof(uint64_t) * 10U; /* numeric/boolean fields */
        if (options->intra_layer_checkpoints_enabled) {
            sz += sizeof(uint64_t) * 2U;
            sz += strlen(LIS_INTRA_LAYER_DIAGNOSTIC_CAPTURE_PROFILE) + 1U;
        }
        sz += strlen("repetition_penalty=1.2") + 1;
        out->digest = digest;
        out->size_bytes = sz;
        out->valid = 1;
    }
    return LIS_STATUS_OK;
}

lis_status lis_artifact_fingerprint_backend(const char *backend_name,
                                            size_t thread_count,
                                            lis_artifact_fingerprint *out)
{
    uint64_t digest = lis_artifact_hash_init();

    if (backend_name == NULL || out == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));
    lis_artifact_hash_cstr(&digest, backend_name);
    lis_artifact_hash_u64_le(&digest, (uint64_t)thread_count);
    out->digest = digest;
    out->size_bytes = strlen(backend_name) + sizeof(thread_count);
    out->valid = 1;
    return LIS_STATUS_OK;
}

static void lis_artifact_write_perf_md(FILE *fp,
                                       const lis_artifact_run_report *report)
{
    static const char *stage_names[LIS_PERF_STAGE_COUNT] = {
        "model_load",
        "tokenizer_load",
        "tokenizer_encode",
        "runtime_init",
        "prefill",
        "first_decode",
        "decode_steady_state"
    };
    const lis_perf_report *perf = report->perf;
    uint64_t encode_ns;
    uint64_t runtime_init_ns;
    uint64_t prefill_ns;
    uint64_t first_decode_ns;
    uint64_t steady_ns;
    uint64_t steady_tokens;
    uint64_t ttft_ns;
    uint64_t end_to_end_ns;
    double itl_ms;
    double tps_steady;
    double tps_end_to_end;
    size_t index;
    size_t prompt_tokens = 0;

    if (fp == NULL || report == NULL || perf == NULL || !perf->enabled) {
        return;
    }
    for (index = 0; index < report->prompt_sequence_count; ++index) {
        prompt_tokens += report->prompt_sequences[index].token_count;
    }
    encode_ns = perf->stage_ns[LIS_PERF_STAGE_TOKENIZER_ENCODE];
    runtime_init_ns = perf->stage_ns[LIS_PERF_STAGE_RUNTIME_INIT];
    prefill_ns = perf->stage_ns[LIS_PERF_STAGE_PREFILL];
    first_decode_ns = perf->stage_ns[LIS_PERF_STAGE_FIRST_DECODE];
    steady_ns = perf->stage_ns[LIS_PERF_STAGE_DECODE_STEADY_STATE];
    steady_tokens = perf->stage_tokens[LIS_PERF_STAGE_DECODE_STEADY_STATE];
    ttft_ns = encode_ns + runtime_init_ns + prefill_ns + first_decode_ns;
    end_to_end_ns = prefill_ns + first_decode_ns + steady_ns;
    itl_ms = (steady_tokens > 0U)
        ? ((double)steady_ns / 1.0e6 / (double)steady_tokens)
        : 0.0;
    tps_steady = (steady_ns > 0U)
        ? ((double)steady_tokens * 1.0e9 / (double)steady_ns)
        : 0.0;
    tps_end_to_end = (end_to_end_ns > 0U && report->emitted_token_count > 0U)
        ? ((double)report->emitted_token_count * 1.0e9 /
           (double)end_to_end_ns)
        : 0.0;

    fprintf(fp, "\n## Performance Summary\n");
    fprintf(fp, "- Threads: %zu\n",
            report->options != NULL ? report->options->thread_count : 0U);
    fprintf(fp, "- Prompt tokens: %zu\n", prompt_tokens);
    fprintf(fp, "- Generated tokens: %zu\n", report->emitted_token_count);
    fprintf(fp, "- TTFT: %.3f ms\n", (double)ttft_ns / 1.0e6);
    fprintf(fp, "- Mean ITL: %.3f ms\n", itl_ms);
    fprintf(fp, "- Steady-state TPS: %.3f\n", tps_steady);
    fprintf(fp, "- End-to-end TPS: %.3f\n", tps_end_to_end);

    fprintf(fp, "\n### Stage Timings\n\n");
    fprintf(fp, "| Stage | ms | Tokens |\n");
    fprintf(fp, "|---|---|---|\n");
    for (index = 0; index < LIS_PERF_STAGE_COUNT; ++index) {
        fprintf(fp, "| %s | %.3f | %" PRIu64 " |\n",
                stage_names[index],
                (double)perf->stage_ns[index] / 1.0e6,
                perf->stage_tokens[index]);
    }
}

static void lis_artifact_write_kv_cache_md(
    FILE *fp,
    const lis_artifact_kv_cache_report *kv_cache)
{
    if (fp == NULL || kv_cache == NULL || !kv_cache->valid) {
        return;
    }

    fprintf(fp, "\n## KV Cache\n\n");
    fprintf(fp, "- Scope: run_local\n");
    fprintf(fp, "- Policy: eviction_free=true, monotonic_growth=true, "
                "paging=false, offload=false, sliding_window=false, "
                "prefix_reuse=false\n");
    fprintf(fp, "- Storage dtype: %s\n",
            lis_dtype_name(kv_cache->storage_dtype));
    fprintf(fp, "- Max tokens: %zu\n", kv_cache->max_tokens);
    fprintf(fp, "- Used tokens: %zu\n", kv_cache->used_tokens);
    fprintf(fp, "- Bytes per token: %zu\n", kv_cache->bytes_per_token);
    fprintf(fp, "- Allocated bytes: %zu\n", kv_cache->allocated_bytes);
    fprintf(fp, "- Used bytes: %zu\n", kv_cache->used_bytes);
    fprintf(fp,
            "- Shape: layers=%zu, batch=%zu, kv_heads=%zu, head_dim=%zu, "
            "element_size=%zu\n",
            kv_cache->layer_count,
            kv_cache->batch_size,
            kv_cache->kv_head_count,
            kv_cache->head_dim,
            kv_cache->element_size);
}

static void lis_artifact_write_bounded_token_list_md(FILE *fp,
                                                       const size_t *ids,
                                                       size_t count)
{
    size_t index;
    size_t limit = 16;

    if (fp == NULL || ids == NULL) {
        return;
    }
    if (count == 0U) {
        fprintf(fp, "(none)\n");
        return;
    }
    fprintf(fp, "`");
    for (index = 0; index < count && index < limit; ++index) {
        fprintf(fp, "%s%zu", index > 0U ? ", " : "", ids[index]);
    }
    fprintf(fp, "`");
    if (count > limit) {
        fprintf(fp, " (%zu more)", count - limit);
    }
    fprintf(fp, "\n");
}

lis_status lis_artifact_write_run_report_md(
    const lis_artifact_run_report *report)
{
    FILE *fp = NULL;
    size_t index;
    char hex[LIS_ARTIFACT_DIGEST_HEX_LEN + 1U];

    if (report == NULL || report->path == NULL || report->options == NULL ||
        report->artifact_set_id == NULL || !report->artifact_set_id->valid ||
        report->model == NULL || report->model_format_name == NULL ||
        report->model_family_name == NULL || report->backend_name == NULL ||
        report->stop_reason_name == NULL || report->prompt_sequences == NULL ||
        report->prompt_sequence_count == 0U ||
        report->selected_token_ids == NULL ||
        report->emitted_token_ids == NULL ||
        !report->binary_fingerprint.valid ||
        !report->model_fingerprint.valid ||
        !report->config_fingerprint.valid ||
        !report->input_fingerprint.valid ||
        !report->runtime_fingerprint.valid ||
        !report->backend_fingerprint.valid ||
        !report->selected_token_digest.valid ||
        !report->emitted_token_digest.valid ||
        !report->kv_cache.valid ||
        !lis_artifact_forced_prefix_is_valid(report->forced_prefix) ||
        ((report->options->forced_prefix_text != NULL) !=
         (report->forced_prefix != NULL))) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    fp = fopen(report->path, "wb");
    if (fp == NULL) {
        return LIS_STATUS_IO;
    }

    fprintf(fp, "# LIS Execution Report\n\n");
    fprintf(fp, "> This Markdown report is a human-readable companion. "
                "The JSON artifact remains the canonical machine-readable source of truth.\n\n");

    /* Outcome */
    fprintf(fp, "## Outcome\n\n");
    fprintf(fp, "- Status: %s\n",
            report->status == LIS_STATUS_OK ? "OK" : "error");
    fprintf(fp, "- Stop reason: %s\n", report->stop_reason_name);

    /* Identity */
    fprintf(fp, "\n## Identity\n\n");
    fprintf(fp, "- Schema: `%s`\n", LIS_ARTIFACT_SCHEMA);
    fprintf(fp, "- Kind: `run_report`\n");
    fprintf(fp, "- Artifact set: `%s`\n", report->artifact_set_id->value);
    fprintf(fp, "- Model format: %s\n", report->model_format_name);
    fprintf(fp, "- Model family: %s\n", report->model_family_name);
    fprintf(fp, "- Backend: %s\n", report->backend_name);

    /* Retention Policy */
    fprintf(fp, "\n## Retention Policy\n\n");
    fprintf(fp, "- Absolute paths: omitted\n");
    fprintf(fp, "- Raw prompt text: omitted\n");
    fprintf(fp, "- Generated text: omitted\n");

    /* Runtime */
    fprintf(fp, "\n## Runtime\n\n");
    fprintf(fp, "- Configured context: %zu\n",
            report->options->context_length);
    fprintf(fp, "- Batch size: %zu\n", report->options->batch_size);
    fprintf(fp, "- Generation limit: %zu\n",
            report->options->generation_limit);
    fprintf(fp, "- Thread count: %zu\n", report->options->thread_count);
    fprintf(fp, "- Diagnostics: %s\n",
            report->options->diagnostics_enabled ? "enabled" : "disabled");
    fprintf(fp, "- Perf: %s\n",
            report->options->perf_enabled ? "enabled" : "disabled");

    if (report->forced_prefix != NULL) {
        fprintf(fp, "\n## Forced Prefix\n\n");
        fprintf(fp, "- Mode: `%s`\n", report->forced_prefix->mode);
        fprintf(fp, "- Applied tokens: %zu\n",
                report->forced_prefix->token_count);
        fprintf(fp, "- Token SHA-256: `%s`\n",
                report->forced_prefix->token_ids_sha256);
        fprintf(fp, "- Target generated step: %zu\n",
                report->forced_prefix->target_generated_token_step);
        fprintf(fp, "- Runtime checkpoint step: %zu\n",
                report->forced_prefix->runtime_checkpoint_step);
        fprintf(fp, "- Selection policy: `%s`\n",
                report->forced_prefix->selection_policy);
        fprintf(fp, "- Raw forced token IDs: omitted\n");
    }

    /* KV Cache */
    lis_artifact_write_kv_cache_md(fp, &report->kv_cache);

    /* Token Accounting */
    fprintf(fp, "\n## Token Accounting\n\n");
    fprintf(fp, "- Prompt sequences: %zu\n", report->prompt_sequence_count);
    for (index = 0; index < report->prompt_sequence_count; ++index) {
        lis_artifact_digest_hex(
            &report->prompt_sequences[index].token_id_digest, hex);
        fprintf(fp, "  - seq %zu: %zu token(s), digest `fnv1a64:%s`\n",
                index + 1U,
                report->prompt_sequences[index].token_count, hex);
    }
    fprintf(fp, "- Selected tokens: %zu\n", report->selected_token_count);
    fprintf(fp, "  - digest: `fnv1a64:");
    lis_artifact_digest_hex(&report->selected_token_digest, hex);
    fprintf(fp, "%s`\n", hex);
    fprintf(fp, "  - IDs: ");
    lis_artifact_write_bounded_token_list_md(
        fp, report->selected_token_ids, report->selected_token_count);
    fprintf(fp, "- Emitted tokens: %zu\n", report->emitted_token_count);
    fprintf(fp, "  - digest: `fnv1a64:");
    lis_artifact_digest_hex(&report->emitted_token_digest, hex);
    fprintf(fp, "%s`\n", hex);
    fprintf(fp, "  - IDs: ");
    lis_artifact_write_bounded_token_list_md(
        fp, report->emitted_token_ids, report->emitted_token_count);

    /* Fingerprints */
    fprintf(fp, "\n## Fingerprints\n\n");
    lis_artifact_digest_hex(&report->binary_fingerprint, hex);
    fprintf(fp, "- Binary: `fnv1a64:%s` (%zu bytes)\n", hex,
            report->binary_fingerprint.size_bytes);
    lis_artifact_digest_hex(&report->model_fingerprint, hex);
    fprintf(fp, "- Model: `fnv1a64:%s` (%zu bytes)\n", hex,
            report->model_fingerprint.size_bytes);
    lis_artifact_digest_hex(&report->config_fingerprint, hex);
    fprintf(fp, "- Config: `fnv1a64:%s` (%zu bytes)\n", hex,
            report->config_fingerprint.size_bytes);
    lis_artifact_digest_hex(&report->input_fingerprint, hex);
    fprintf(fp, "- Input: `fnv1a64:%s` (%zu bytes)\n", hex,
            report->input_fingerprint.size_bytes);
    lis_artifact_digest_hex(&report->runtime_fingerprint, hex);
    fprintf(fp, "- Runtime: `fnv1a64:%s`\n", hex);
    lis_artifact_digest_hex(&report->backend_fingerprint, hex);
    fprintf(fp, "- Backend: `fnv1a64:%s`\n", hex);

    /* Optional perf summary */
    lis_artifact_write_perf_md(fp, report);

    /* Notes */
    fprintf(fp, "\n## Notes\n\n");
    fprintf(fp, "- This Markdown report is a bounded human-readable summary. "
                "The JSON artifact (`lis.execution_artifact/v1`) remains the canonical source of truth.\n");
    fprintf(fp, "- Token IDs and digests are retained for reproducibility; "
                "raw text and absolute paths are omitted by default.\n");

    if (fclose(fp) != 0) {
        return LIS_STATUS_IO;
    }
    return LIS_STATUS_OK;
}

lis_status lis_artifact_write_run_report(
    const lis_artifact_run_report *report)
{
    FILE *fp = NULL;
    size_t index;

    if (report == NULL || report->path == NULL || report->options == NULL ||
        report->artifact_set_id == NULL || !report->artifact_set_id->valid ||
        report->model == NULL || report->model_format_name == NULL ||
        report->model_family_name == NULL || report->backend_name == NULL ||
        report->stop_reason_name == NULL || report->prompt_sequences == NULL ||
        report->prompt_sequence_count == 0U ||
        report->selected_token_ids == NULL ||
        report->emitted_token_ids == NULL ||
        !report->binary_fingerprint.valid ||
        !report->model_fingerprint.valid ||
        !report->config_fingerprint.valid ||
        !report->input_fingerprint.valid ||
        !report->runtime_fingerprint.valid ||
        !report->backend_fingerprint.valid ||
        !report->selected_token_digest.valid ||
        !report->emitted_token_digest.valid ||
        !report->kv_cache.valid ||
        !lis_artifact_forced_prefix_is_valid(report->forced_prefix) ||
        ((report->options->forced_prefix_text != NULL) !=
         (report->forced_prefix != NULL))) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    fp = fopen(report->path, "wb");
    if (fp == NULL) {
        return LIS_STATUS_IO;
    }

    fputs("{\"schema\":\"" LIS_ARTIFACT_SCHEMA "\",\"kind\":\"run_report\","
          "\"artifact_set_id\":", fp);
    lis_artifact_write_json_string(fp, report->artifact_set_id->value);
    if (report->forced_prefix != NULL) {
        lis_artifact_write_forced_prefix_json(fp, report->forced_prefix);
    }
    fputs(",\"manifest\":{", fp);
    fputs("\"retention_policy\":{"
          "\"absolute_paths\":\"omitted\","
          "\"raw_prompt_text\":\"omitted\","
          "\"generated_text\":\"omitted\"},", fp);
    fputs("\"binary\":{\"fingerprint\":", fp);
    lis_artifact_write_fp(fp, &report->binary_fingerprint);
    fputs("},\"model\":{\"format\":", fp);
    lis_artifact_write_json_string(fp, report->model_format_name);
    fputs(",\"family\":", fp);
    lis_artifact_write_json_string(fp, report->model_family_name);
    fputs(",\"fingerprint\":", fp);
    lis_artifact_write_fp(fp, &report->model_fingerprint);
    fputs("},\"config\":{\"fingerprint\":", fp);
    lis_artifact_write_fp(fp, &report->config_fingerprint);
    fputs("},\"input\":{\"mode\":", fp);
    lis_artifact_write_json_string(fp,
                                   lis_artifact_input_mode_name(
                                       report->input_mode));
    fputs(",\"fingerprint\":", fp);
    lis_artifact_write_fp(fp, &report->input_fingerprint);
    fprintf(fp,
            "},\"runtime\":{\"configured_context\":%zu,\"batch_size\":%zu,"
            "\"generation_limit\":%zu,\"thread_count\":%zu,"
            "\"layer_checkpoints_enabled\":",
            report->options->context_length,
            report->options->batch_size,
            report->options->generation_limit,
            report->options->thread_count);
    lis_artifact_write_bool(fp, report->options->layer_checkpoints_enabled);
    fprintf(fp, ",\"layer_checkpoint_step\":%zu",
            report->options->layer_checkpoints_step);
    if (report->options->intra_layer_checkpoints_enabled) {
        fputs(",\"intra_layer_checkpoints_enabled\":", fp);
        lis_artifact_write_bool(
            fp, report->options->intra_layer_checkpoints_enabled);
        fprintf(fp, ",\"intra_layer_target_layer\":%zu",
                report->options->intra_layer_target_layer);
        fputs(",\"diagnostic_capture_profile\":", fp);
        lis_artifact_write_json_string(
            fp, LIS_INTRA_LAYER_DIAGNOSTIC_CAPTURE_PROFILE);
    }
    fputs(",\"diagnostics_enabled\":", fp);
    lis_artifact_write_bool(fp, report->options->diagnostics_enabled);
    fputs(",\"perf_enabled\":", fp);
    lis_artifact_write_bool(fp, report->options->perf_enabled);
    fputs(",\"perf_per_token_enabled\":", fp);
    lis_artifact_write_bool(fp, report->options->perf_per_token_enabled);
    fputs(",\"precision_path\":", fp);
    lis_artifact_write_json_string(
        fp, report->precision_path != NULL ? report->precision_path : "");
    fputs(",\"fingerprint\":", fp);
    lis_artifact_write_fp(fp, &report->runtime_fingerprint);
    fputs("},\"backend\":{\"name\":", fp);
    lis_artifact_write_json_string(fp, report->backend_name);
    fputs(",\"fingerprint\":", fp);
    lis_artifact_write_fp(fp, &report->backend_fingerprint);
    fputs("}},\"report\":{\"execution_status\":", fp);
    lis_artifact_write_json_string(
        fp, report->status == LIS_STATUS_OK ? "ok" : "error");
    fputs(",\"status_code\":", fp);
    lis_artifact_write_json_string(fp, lis_status_name(report->status));
    fputs(",\"stop_reason\":", fp);
    lis_artifact_write_json_string(fp, report->stop_reason_name);
    fputs(",\"output_mode\":", fp);
    lis_artifact_write_json_string(
        fp, lis_artifact_output_mode_name(report->output_mode));
    fputs(",\"prompt_sequences\":[", fp);
    for (index = 0; index < report->prompt_sequence_count; ++index) {
        if (index > 0U) {
            fputc(',', fp);
        }
        fprintf(fp, "{\"token_count\":%zu,\"token_id_digest\":",
                report->prompt_sequences[index].token_count);
        lis_artifact_write_fp(fp,
                              &report->prompt_sequences[index].token_id_digest);
        fputc('}', fp);
    }
    fprintf(fp, "],\"selected_token_count\":%zu,\"selected_token_digest\":",
            report->selected_token_count);
    lis_artifact_write_fp(fp, &report->selected_token_digest);
    fputs(",\"selected_token_ids\":[", fp);
    for (index = 0; index < report->selected_token_count; ++index) {
        if (index > 0U) {
            fputc(',', fp);
        }
        fprintf(fp, "%zu", report->selected_token_ids[index]);
    }
    fprintf(fp, "],\"emitted_token_count\":%zu,\"emitted_token_digest\":",
            report->emitted_token_count);
    lis_artifact_write_fp(fp, &report->emitted_token_digest);
    fputs(",\"emitted_token_ids\":[", fp);
    for (index = 0; index < report->emitted_token_count; ++index) {
        if (index > 0U) {
            fputc(',', fp);
        }
        fprintf(fp, "%zu", report->emitted_token_ids[index]);
    }
    fputc(']', fp);
    lis_artifact_write_perf_object(fp, report);
    lis_artifact_write_kv_cache_object(fp, &report->kv_cache);
    fputs("}}", fp);

    if (fclose(fp) != 0) {
        return LIS_STATUS_IO;
    }
    return LIS_STATUS_OK;
}
