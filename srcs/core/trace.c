#include "lis/trace.h"

#include <errno.h>
#include <float.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *lis_trace_phase_name(lis_trace_phase phase)
{
    switch (phase) {
    case LIS_TRACE_PHASE_PREFILL_SEED:
        return "prefill_seed";
    case LIS_TRACE_PHASE_FIRST_DECODE:
        return "first_decode";
    case LIS_TRACE_PHASE_DECODE:
        return "decode";
    }
    return "decode";
}

lis_status lis_trace_record_init(lis_trace_record *record, size_t capacity)
{
    if (record == NULL || capacity == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    memset(record, 0, sizeof(*record));
    if (capacity > SIZE_MAX / sizeof(*record->steps)) {
        return LIS_STATUS_OVERFLOW;
    }
    record->steps = calloc(capacity, sizeof(*record->steps));
    if (record->steps == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }
    record->step_count = 0;
    record->step_capacity = capacity;
    return LIS_STATUS_OK;
}

void lis_trace_record_destroy(lis_trace_record *record)
{
    if (record == NULL) {
        return;
    }
    free(record->steps);
    memset(record, 0, sizeof(*record));
}

lis_status lis_trace_record_append(lis_trace_record *record,
                                   const lis_trace_step *step)
{
    if (record == NULL || step == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (record->step_count >= record->step_capacity) {
        return LIS_STATUS_OVERFLOW;
    }
    record->steps[record->step_count++] = *step;
    return LIS_STATUS_OK;
}

static void lis_trace_write_fp(FILE *fp,
                                const lis_artifact_fingerprint *fingerprint)
{
    char hex[LIS_ARTIFACT_DIGEST_HEX_LEN + 1U];

    lis_artifact_digest_hex(fingerprint, hex);
    fprintf(fp,
            "{\"algorithm\":\"fnv1a64\",\"hex\":\"%s\",\"size_bytes\":%zu}",
            hex, fingerprint != NULL ? fingerprint->size_bytes : 0U);
}

static void lis_trace_write_json_string(FILE *fp, const char *text)
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

static void lis_trace_write_bool(FILE *fp, int value)
{
    fputs(value ? "true" : "false", fp);
}

lis_status lis_trace_artifact_write(const lis_trace_artifact *artifact,
                                    const lis_trace_record *record)
{
    FILE *fp = NULL;
    size_t index;

    if (artifact == NULL || artifact->path == NULL ||
        artifact->artifact_set_id == NULL ||
        !artifact->artifact_set_id->valid ||
        artifact->options == NULL || artifact->model == NULL ||
        record == NULL || record->steps == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
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
            "\"kind\":\"decode_trace\",\"artifact_set_id\":");
    lis_trace_write_json_string(fp, artifact->artifact_set_id->value);
    fputs(",\"manifest\":{", fp);
    fprintf(fp, "\"retention_policy\":{"
            "\"absolute_paths\":\"omitted\","
            "\"raw_prompt_text\":\"omitted\","
            "\"generated_text\":\"omitted\"},");
    fputs("\"binary\":{\"fingerprint\":", fp);
    lis_trace_write_fp(fp, &artifact->binary_fingerprint);
    fputs("},\"model\":{\"format\":", fp);
    lis_trace_write_json_string(fp, artifact->model_format_name);
    fputs(",\"family\":", fp);
    lis_trace_write_json_string(fp, artifact->model_family_name);
    fputs(",\"fingerprint\":", fp);
    lis_trace_write_fp(fp, &artifact->model_fingerprint);
    fputs("},\"config\":{\"fingerprint\":", fp);
    lis_trace_write_fp(fp, &artifact->config_fingerprint);
    fprintf(fp,
            "},\"input\":{\"mode\":");
    lis_trace_write_json_string(fp,
                                lis_artifact_input_mode_name(
                                    artifact->input_mode));
    fputs(",\"fingerprint\":", fp);
    lis_trace_write_fp(fp, &artifact->input_fingerprint);
    fprintf(fp,
            "},\"runtime\":{\"configured_context\":%zu,"
            "\"batch_size\":%zu,"
            "\"generation_limit\":%zu,"
            "\"thread_count\":%zu",
            artifact->options->context_length,
            artifact->options->batch_size,
            artifact->options->generation_limit,
            artifact->options->thread_count);
    if (artifact->options->intra_layer_checkpoints_enabled) {
        fputs(",\"intra_layer_checkpoints_enabled\":", fp);
        lis_trace_write_bool(
            fp, artifact->options->intra_layer_checkpoints_enabled);
        fprintf(fp, ",\"intra_layer_target_layer\":%zu",
                artifact->options->intra_layer_target_layer);
        fputs(",\"diagnostic_capture_profile\":", fp);
        lis_trace_write_json_string(
            fp, LIS_INTRA_LAYER_DIAGNOSTIC_CAPTURE_PROFILE);
    }
    fputs(",\"diagnostics_enabled\":", fp);
    lis_trace_write_bool(fp, artifact->options->diagnostics_enabled);
    fputs(",\"perf_enabled\":", fp);
    lis_trace_write_bool(fp, artifact->options->perf_enabled);
    fputs(",\"precision_path\":", fp);
    lis_trace_write_json_string(
        fp, artifact->precision_path != NULL ? artifact->precision_path : "");
    fputs(",\"fingerprint\":", fp);
    lis_trace_write_fp(fp, &artifact->runtime_fingerprint);
    fputs("},\"backend\":{\"name\":", fp);
    lis_trace_write_json_string(fp, artifact->backend_name);
    fputs(",\"fingerprint\":", fp);
    lis_trace_write_fp(fp, &artifact->backend_fingerprint);
    fputs("}}", fp);

    fputs(",\"decode_trace\":[", fp);
    for (index = 0; index < record->step_count; ++index) {
        const lis_trace_step *step = &record->steps[index];
        size_t k;

        if (index > 0) {
            fputc(',', fp);
        }
        fprintf(fp,
                "{\"step\":%zu,"
                "\"phase\":\"%s\","
                "\"selected_token_id\":%zu,"
                "\"raw_score_selected\":%.9g,"
                "\"adjusted_score_selected\":%.9g,",
                step->step,
                lis_trace_phase_name(step->phase),
                step->selected_token_id,
                step->raw_score_selected,
                step->adjusted_score_selected);
        if (step->runner_up_available) {
            fprintf(fp,
                    "\"runner_up_token_id\":%zu,"
                    "\"runner_up_adjusted_score\":%.9g,",
                    step->runner_up_token_id,
                    step->runner_up_adjusted_score);
        } else {
            fputs("\"runner_up_token_id\":null,"
                  "\"runner_up_adjusted_score\":null,", fp);
        }
        fprintf(fp,
                "\"decision_margin\":");
        if (step->decision_margin_valid) {
            fprintf(fp, "%.9g,", step->decision_margin);
        } else {
            fputs("null,", fp);
        }
        fputs("\"structural_suppression_affected\":", fp);
        lis_trace_write_bool(fp, step->structural_suppression_affected);
        fputs(",\"repetition_penalty_changed_selection\":", fp);
        lis_trace_write_bool(fp, step->repetition_penalty_changed_selection);
        fputs(",\"selected_token_penalized\":", fp);
        lis_trace_write_bool(fp, step->selected_token_penalized);
        fprintf(fp,
                ",\"suppressed_token_count\":%zu,"
                "\"penalized_token_count\":%zu,"
                "\"decision_class\":",
                step->suppressed_token_count,
                step->penalized_token_count);
        lis_trace_write_json_string(fp, step->decision_class != NULL
                                            ? step->decision_class : "");
        fputs(",\"topk\":[", fp);
        for (k = 0; k < step->topk_count; ++k) {
            const lis_trace_topk_entry *entry = &step->topk[k];

            if (k > 0) {
                fputc(',', fp);
            }
            fprintf(fp,
                    "{\"token_id\":%zu,"
                    "\"raw_score\":%.9g,"
                    "\"adjusted_score\":%.9g,"
                    "\"is_selected\":",
                    entry->token_id,
                    entry->raw_score,
                    entry->adjusted_score);
            lis_trace_write_bool(fp, entry->is_selected);
            fputc('}', fp);
        }
        fputs("]", fp);
        if (step->has_stop_reason) {
            fputs(",\"stop_reason\":", fp);
            lis_trace_write_json_string(fp, step->stop_reason);
        }
        fputc('}', fp);
    }
    fputs("]}", fp);

    if (fclose(fp) != 0) {
        return LIS_STATUS_IO;
    }
    return LIS_STATUS_OK;
}
