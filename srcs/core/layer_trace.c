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

int lis_layer_trace_layout_selects_layer(size_t layer_index,
                                         size_t total_layer_count)
{
    const size_t three_quarter =
        (total_layer_count / 4U) * 3U +
        ((total_layer_count % 4U) * 3U) / 4U;

    return total_layer_count != 0U && layer_index < total_layer_count &&
           (layer_index == 0U ||
            layer_index == 1U ||
            layer_index == 2U ||
            layer_index == 4U ||
            layer_index == 6U ||
            layer_index == total_layer_count / 4U ||
            layer_index == total_layer_count / 2U ||
            layer_index == three_quarter ||
            layer_index + 1U == total_layer_count);
}

static lis_status lis_layer_trace_selected_ordinal(size_t layer_index,
                                                   size_t total_layer_count,
                                                   size_t *out)
{
    size_t layer;
    size_t ordinal = 0;

    if (out == NULL ||
        !lis_layer_trace_layout_selects_layer(layer_index, total_layer_count)) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    for (layer = 0; layer < layer_index; ++layer) {
        if (lis_layer_trace_layout_selects_layer(layer, total_layer_count)) {
            ++ordinal;
        }
    }
    *out = ordinal;
    return LIS_STATUS_OK;
}

lis_status lis_layer_trace_record_configure_llama_layout(
    lis_layer_trace_record *record,
    size_t runtime_checkpoint_step,
    size_t total_layer_count)
{
    if (record == NULL || record->steps == NULL || total_layer_count == 0U ||
        record->step_count != 0U) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    record->checkpoint_layout_supported = 1;
    record->layout_runtime_checkpoint_step = runtime_checkpoint_step;
    record->total_layer_count = total_layer_count;
    return LIS_STATUS_OK;
}

lis_status lis_layer_trace_step_set_layer_output(
    lis_layer_trace_step *step,
    size_t layer_index,
    const float *data,
    size_t element_count)
{
    lis_status status;

    if (step == NULL || data == NULL || element_count == 0U ||
        step->rank == 0U || step->rank > LIS_LAYER_TRACE_MAX_RANK) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    step->has_checkpoint_coordinate = 1;
    step->runtime_checkpoint_step = step->step;
    step->layer_index = layer_index;
    step->batch_index = 0U;
    step->sequence_index = 0U;
    step->stage_order = 0U;
    step->element_count = element_count;
    if (snprintf(step->tensor_role, sizeof(step->tensor_role), "%s",
                 LIS_CHECKPOINT_DIGEST_ROLE_LAYER_OUTPUT) >=
        (int)sizeof(step->tensor_role)) {
        return LIS_STATUS_FORMAT;
    }
    if (snprintf(step->observed_dtype, sizeof(step->observed_dtype), "%s",
                 LIS_CHECKPOINT_DIGEST_OBSERVED_DTYPE) >=
        (int)sizeof(step->observed_dtype)) {
        return LIS_STATUS_FORMAT;
    }
    status = lis_checkpoint_digest_fp32(
        step->tensor_role, step->shape, step->rank, data, element_count,
        &step->digest);
    if (status != LIS_STATUS_OK) {
        memset(&step->digest, 0, sizeof(step->digest));
        return status;
    }
    return LIS_STATUS_OK;
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
    lis_layer_trace_step copy;

    if (record == NULL || step == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (record->append_failed != 0) {
        return LIS_STATUS_OVERFLOW;
    }
    copy = *step;
    if (copy.has_checkpoint_coordinate) {
        size_t index;
        size_t ordinal = 0;

        if (!record->checkpoint_layout_supported || !copy.digest.valid ||
            copy.runtime_checkpoint_step !=
                record->layout_runtime_checkpoint_step ||
            copy.layer_index >= record->total_layer_count ||
            strcmp(copy.tensor_role,
                   LIS_CHECKPOINT_DIGEST_ROLE_LAYER_OUTPUT) != 0 ||
            strcmp(copy.observed_dtype,
                   LIS_CHECKPOINT_DIGEST_OBSERVED_DTYPE) != 0 ||
            copy.batch_index != 0U || copy.sequence_index != 0U ||
            copy.stage_order != 0U ||
            lis_layer_trace_selected_ordinal(
                copy.layer_index, record->total_layer_count, &ordinal) !=
                LIS_STATUS_OK) {
            record->append_failed = 1;
            return LIS_STATUS_INVALID_ARGUMENT;
        }
        for (index = 0; index < record->step_count; ++index) {
            const lis_layer_trace_step *previous = &record->steps[index];

            if (previous->has_checkpoint_coordinate &&
                previous->runtime_checkpoint_step ==
                    copy.runtime_checkpoint_step &&
                previous->layer_index == copy.layer_index &&
                previous->batch_index == copy.batch_index &&
                previous->sequence_index == copy.sequence_index &&
                previous->stage_order == copy.stage_order &&
                strcmp(previous->tensor_role, copy.tensor_role) == 0) {
                record->append_failed = 1;
                return LIS_STATUS_INVALID_ARGUMENT;
            }
        }
        if (record->layer_output_count > 0U) {
            size_t previous_index = record->step_count;

            while (previous_index > 0U) {
                const lis_layer_trace_step *previous =
                    &record->steps[--previous_index];

                if (previous->has_checkpoint_coordinate) {
                    if (previous->layer_index >= copy.layer_index ||
                        previous->execution_ordinal >= ordinal) {
                        record->append_failed = 1;
                        return LIS_STATUS_INVALID_ARGUMENT;
                    }
                    break;
                }
            }
        }
        copy.execution_ordinal = ordinal;
        if (record->digest_element_visits > SIZE_MAX - copy.element_count) {
            record->append_failed = 1;
            return LIS_STATUS_OVERFLOW;
        }
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
    record->steps[record->step_count] = copy;
    record->step_count += 1;
    if (copy.has_checkpoint_coordinate) {
        ++record->layer_output_count;
        record->digest_element_visits += copy.element_count;
    }
    return LIS_STATUS_OK;
}

static void lis_layer_trace_write_coordinate(FILE *fp,
                                             size_t runtime_checkpoint_step,
                                             size_t layer_index,
                                             size_t execution_ordinal)
{
    fprintf(fp,
            "{\"runtime_checkpoint_step\":%zu,\"layer_index\":%zu,"
            "\"tensor_role\":\"layer_output\",\"batch_index\":0,"
            "\"sequence_index\":0,\"stage_order\":0,"
            "\"execution_ordinal\":%zu}",
            runtime_checkpoint_step, layer_index, execution_ordinal);
}

static const lis_layer_trace_step *lis_layer_trace_find_layer_output(
    const lis_layer_trace_record *record,
    size_t layer_index)
{
    size_t index;

    for (index = 0; index < record->step_count; ++index) {
        if (record->steps[index].has_checkpoint_coordinate &&
            record->steps[index].layer_index == layer_index) {
            return &record->steps[index];
        }
    }
    return NULL;
}

static void lis_layer_trace_write_checkpoint_layout(
    FILE *fp,
    const lis_layer_trace_record *record)
{
    size_t layer;
    size_t ordinal = 0;
    size_t emitted = 0;

    fputs(",\"checkpoint_layout\":{"
          "\"layout_name\":\"llama_layer_output_summary\","
          "\"layout_version\":1,\"runtime_checkpoint_step\":", fp);
    fprintf(fp, "%zu", record->layout_runtime_checkpoint_step);
    fputs(",\"tensor_role\":\"layer_output\",\"stage_order\":0,"
          "\"ordering_semantics\":\"runtime_step_layer_stage_ordinal\","
          "\"total_layer_count\":", fp);
    fprintf(fp, "%zu,\"requested_coordinates\":[", record->total_layer_count);
    for (layer = 0; layer < record->total_layer_count; ++layer) {
        if (!lis_layer_trace_layout_selects_layer(
                layer, record->total_layer_count)) {
            continue;
        }
        if (emitted++ > 0U) {
            fputc(',', fp);
        }
        lis_layer_trace_write_coordinate(
            fp, record->layout_runtime_checkpoint_step, layer, ordinal++);
    }
    fputs("],\"captured_coordinates\":[", fp);
    emitted = 0;
    for (layer = 0; layer < record->step_count; ++layer) {
        const lis_layer_trace_step *step = &record->steps[layer];

        if (!step->has_checkpoint_coordinate) {
            continue;
        }
        if (emitted++ > 0U) {
            fputc(',', fp);
        }
        lis_layer_trace_write_coordinate(
            fp, step->runtime_checkpoint_step, step->layer_index,
            step->execution_ordinal);
    }
    fputs("],\"missing_coordinates\":[", fp);
    emitted = 0;
    ordinal = 0;
    for (layer = 0; layer < record->total_layer_count; ++layer) {
        if (!lis_layer_trace_layout_selects_layer(
                layer, record->total_layer_count)) {
            continue;
        }
        if (lis_layer_trace_find_layer_output(record, layer) == NULL) {
            if (emitted++ > 0U) {
                fputc(',', fp);
            }
            fputs("{\"coordinate\":", fp);
            lis_layer_trace_write_coordinate(
                fp, record->layout_runtime_checkpoint_step, layer, ordinal);
            fputs(",\"state\":\"not_captured\","
                  "\"detail\":\"target_checkpoint_not_observed\"}", fp);
        }
        ++ordinal;
    }
    fputs("],\"available_summary_fields\":[\"min\",\"max\",\"mean\","
          "\"l2\",\"nan\",\"inf\",\"digest\"],"
          "\"digest_contract\":{\"algorithm\":\"sha256\","
          "\"version\":\"" LIS_CHECKPOINT_DIGEST_VERSION "\","
          "\"observed_dtype\":\"" LIS_CHECKPOINT_DIGEST_OBSERVED_DTYPE "\","
          "\"byte_order\":\"" LIS_CHECKPOINT_DIGEST_BYTE_ORDER "\","
          "\"canonicalization\":\""
          LIS_CHECKPOINT_DIGEST_CANONICALIZATION "\"},"
          "\"duplicate_coordinate_policy\":\"reject_artifact_before_write\"}",
          fp);
}

lis_status lis_layer_trace_artifact_write(const lis_layer_trace_artifact *artifact,
                                          const lis_layer_trace_record *record)
{
    FILE *fp = NULL;
    size_t index;
    size_t rank;

    if (record == NULL || record->steps == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (record->append_failed != 0) {
        return LIS_STATUS_OVERFLOW;
    }
    if (artifact == NULL || artifact->path == NULL ||
        artifact->artifact_set_id == NULL ||
        !artifact->artifact_set_id->valid ||
        artifact->options == NULL || artifact->model == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (record->checkpoint_layout_supported &&
        (record->total_layer_count == 0U ||
         artifact->model->metadata.config.layer_count !=
             record->total_layer_count ||
         artifact->model->metadata.config.family !=
             LIS_MODEL_FAMILY_LLAMA3_DECODER)) {
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
            "\"kind\":\"layer_trace\",\"artifact_set_id\":");
    lis_layer_trace_write_json_string(fp, artifact->artifact_set_id->value);
    fputs(",\"manifest\":{", fp);
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

    if (record->checkpoint_layout_supported) {
        lis_layer_trace_write_checkpoint_layout(fp, record);
    }

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
        fprintf(fp, ",\"nan\":%d,\"inf\":%d", step->nan, step->inf);
        if (step->has_checkpoint_coordinate) {
            char digest_hex[LIS_CHECKPOINT_DIGEST_HEX_SIZE + 1U];

            lis_checkpoint_digest_hex(&step->digest, digest_hex);
            fprintf(fp,
                    ",\"runtime_checkpoint_step\":%zu,"
                    "\"layer_index\":%zu,\"tensor_role\":",
                    step->runtime_checkpoint_step, step->layer_index);
            lis_layer_trace_write_json_string(fp, step->tensor_role);
            fprintf(fp,
                    ",\"batch_index\":%zu,\"sequence_index\":%zu,"
                    "\"stage_order\":%zu,\"execution_ordinal\":%zu,"
                    "\"observed_dtype\":",
                    step->batch_index, step->sequence_index,
                    step->stage_order, step->execution_ordinal);
            lis_layer_trace_write_json_string(fp, step->observed_dtype);
            fprintf(fp,
                    ",\"element_count\":%zu,"
                    "\"available_summary_fields\":[\"min\",\"max\","
                    "\"mean\",\"l2\",\"nan\",\"inf\",\"digest\"],"
                    "\"digest\":{\"algorithm\":\"sha256\","
                    "\"version\":\"" LIS_CHECKPOINT_DIGEST_VERSION "\","
                    "\"tensor_role\":",
                    step->element_count);
            lis_layer_trace_write_json_string(fp, step->tensor_role);
            fputs(",\"shape\":[", fp);
            for (rank = 0; rank < step->rank; ++rank) {
                if (rank > 0U) {
                    fputc(',', fp);
                }
                fprintf(fp, "%zu", step->shape[rank]);
            }
            fputs("],\"observed_dtype\":\""
                  LIS_CHECKPOINT_DIGEST_OBSERVED_DTYPE "\","
                  "\"byte_order\":\"" LIS_CHECKPOINT_DIGEST_BYTE_ORDER "\","
                  "\"canonicalization\":\""
                  LIS_CHECKPOINT_DIGEST_CANONICALIZATION "\","
                  "\"value\":\"sha256:", fp);
            fputs(digest_hex, fp);
            fputs("\"}", fp);
        }
        fputc('}', fp);
    }
    fputs("]}", fp);

    if (fclose(fp) != 0) {
        return LIS_STATUS_IO;
    }
    return LIS_STATUS_OK;
}
