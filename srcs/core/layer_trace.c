#include "lis/layer_trace.h"

#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void lis_layer_trace_write_fp(FILE *fp,
                                     const lis_artifact_fingerprint *fingerprint)
{
    char hex[LIS_ARTIFACT_DIGEST_HEX_LEN + 1U];

    lis_artifact_digest_hex(fingerprint, hex);
    fprintf(fp,
            "{\"algorithm\":\"fnv1a64\",\"hex\":\"%s\",\"size_bytes\":%zu}",
            hex, fingerprint != NULL ? fingerprint->size_bytes : 0U);
}

static void lis_layer_trace_write_json_string(FILE *fp, const char *text)
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

static void lis_layer_trace_write_bool(FILE *fp, int value)
{
    fputs(value ? "true" : "false", fp);
}

static void lis_layer_trace_write_g6(FILE *fp, float v)
{
    if (isnan(v) || isinf(v)) {
        fputs("null", fp);
    } else {
        fprintf(fp, "%.6g", (double)v);
    }
}

lis_status lis_layer_trace_record_init(lis_layer_trace_record *record,
                                       size_t initial_capacity)
{
    if (record == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    memset(record, 0, sizeof(*record));
    if (initial_capacity == 0 ||
        initial_capacity > SIZE_MAX / sizeof(*record->steps)) {
        return LIS_STATUS_OVERFLOW;
    }
    record->steps = calloc(initial_capacity, sizeof(*record->steps));
    if (record->steps == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }
    record->step_count = 0;
    record->step_capacity = initial_capacity;
    return LIS_STATUS_OK;
}

void lis_layer_trace_record_destroy(lis_layer_trace_record *record)
{
    if (record == NULL) {
        return;
    }
    free(record->steps);
    memset(record, 0, sizeof(*record));
}

lis_status lis_layer_trace_record_append(lis_layer_trace_record *record,
                                         const lis_layer_trace_step *step)
{
    size_t new_cap;
    lis_layer_trace_step *tmp;

    if (record == NULL || step == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (record->append_failed != 0) {
        return LIS_STATUS_OVERFLOW;
    }
    if (record->step_count == record->step_capacity) {
        if (record->step_capacity >= LIS_LAYER_TRACE_HARD_MAX) {
            record->append_failed = 1;
            return LIS_STATUS_OVERFLOW;
        }
        new_cap = record->step_capacity * 2;
        if (new_cap > LIS_LAYER_TRACE_HARD_MAX) {
            new_cap = LIS_LAYER_TRACE_HARD_MAX;
        }
        tmp = realloc(record->steps, new_cap * sizeof(*tmp));
        if (tmp == NULL) {
            record->append_failed = 1;
            return LIS_STATUS_NO_MEMORY;
        }
        record->steps = tmp;
        record->step_capacity = new_cap;
    }
    record->steps[record->step_count] = *step;
    record->step_count += 1;
    return LIS_STATUS_OK;
}

lis_status lis_layer_trace_artifact_write(const lis_layer_trace_artifact *artifact,
                                          const lis_layer_trace_record *record)
{
    FILE *fp = NULL;
    size_t index;
    size_t rank;

    if (artifact == NULL || artifact->path == NULL ||
        artifact->options == NULL || artifact->model == NULL ||
        record == NULL || record->steps == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (record->append_failed != 0) {
        return LIS_STATUS_OVERFLOW;
    }
    if (!artifact->binary_fingerprint.valid ||
        !artifact->model_fingerprint.valid ||
        !artifact->config_fingerprint.valid ||
        !artifact->input_fingerprint.valid ||
        !artifact->runtime_fingerprint.valid ||
        !artifact->backend_fingerprint.valid) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    fp = fopen(artifact->path, "wb");
    if (fp == NULL) {
        return LIS_STATUS_IO;
    }

    fprintf(fp,
            "{\"schema\":\"lis.execution_artifact/v1\","
            "\"kind\":\"layer_trace\","
            "\"manifest\":{");
    fprintf(fp, "\"retention_policy\":{"
            "\"absolute_paths\":\"omitted\","
            "\"raw_prompt_text\":\"omitted\","
            "\"generated_text\":\"omitted\"},");
    fputs("\"binary\":{\"fingerprint\":", fp);
    lis_layer_trace_write_fp(fp, &artifact->binary_fingerprint);
    fputs("},\"model\":{\"format\":", fp);
    lis_layer_trace_write_json_string(fp, artifact->model_format_name);
    fputs(",\"family\":", fp);
    lis_layer_trace_write_json_string(fp, artifact->model_family_name);
    fputs(",\"fingerprint\":", fp);
    lis_layer_trace_write_fp(fp, &artifact->model_fingerprint);
    fputs("},\"config\":{\"fingerprint\":", fp);
    lis_layer_trace_write_fp(fp, &artifact->config_fingerprint);
    fprintf(fp,
            "},\"input\":{\"mode\":");
    lis_layer_trace_write_json_string(fp,
                                      lis_artifact_input_mode_name(
                                          artifact->input_mode));
    fputs(",\"fingerprint\":", fp);
    lis_layer_trace_write_fp(fp, &artifact->input_fingerprint);
    fprintf(fp,
            "},\"runtime\":{\"configured_context\":%zu,"
            "\"batch_size\":%zu,"
            "\"generation_limit\":%zu,"
            "\"thread_count\":%zu,"
            "\"layer_checkpoints_enabled\":",
            artifact->options->context_length,
            artifact->options->batch_size,
            artifact->options->generation_limit,
            artifact->options->thread_count);
    lis_layer_trace_write_bool(fp, artifact->options->layer_checkpoints_enabled);
    fprintf(fp, ",\"layer_checkpoint_step\":%zu,\"diagnostics_enabled\":",
            artifact->options->layer_checkpoints_step);
    lis_layer_trace_write_bool(fp, artifact->options->diagnostics_enabled);
    fputs(",\"perf_enabled\":", fp);
    lis_layer_trace_write_bool(fp, artifact->options->perf_enabled);
    fputs(",\"perf_per_token_enabled\":", fp);
    lis_layer_trace_write_bool(fp, artifact->options->perf_per_token_enabled);
    fputs(",\"precision_path\":", fp);
    lis_layer_trace_write_json_string(
        fp, artifact->precision_path != NULL ? artifact->precision_path : "");
    fputs(",\"fingerprint\":", fp);
    lis_layer_trace_write_fp(fp, &artifact->runtime_fingerprint);
    fputs("},\"backend\":{\"name\":", fp);
    lis_layer_trace_write_json_string(fp, artifact->backend_name);
    fputs(",\"fingerprint\":", fp);
    lis_layer_trace_write_fp(fp, &artifact->backend_fingerprint);
    fputs("}}", fp);

    fputs(",\"layer_trace\":[", fp);
    for (index = 0; index < record->step_count; ++index) {
        const lis_layer_trace_step *step = &record->steps[index];

        if (index > 0) {
            fputc(',', fp);
        }
        fprintf(fp,
                "{\"step\":%zu,"
                "\"phase\":\"%s\","
                "\"name\":\"%s\","
                "\"shape\":[",
                step->step, step->phase, step->name);
        for (rank = 0; rank < step->rank; ++rank) {
            if (rank > 0) {
                fputc(',', fp);
            }
            fprintf(fp, "%zu", step->shape[rank]);
        }
        fputs("],\"min\":", fp);
        lis_layer_trace_write_g6(fp, step->min);
        fputs(",\"max\":", fp);
        lis_layer_trace_write_g6(fp, step->max);
        fputs(",\"mean\":", fp);
        lis_layer_trace_write_g6(fp, step->mean);
        fputs(",\"l2\":", fp);
        lis_layer_trace_write_g6(fp, step->l2);
        fprintf(fp,
                ",\"nan\":%d,\"inf\":%d}",
                step->nan, step->inf);
    }
    fputs("]}", fp);

    if (fclose(fp) != 0) {
        return LIS_STATUS_IO;
    }
    return LIS_STATUS_OK;
}
