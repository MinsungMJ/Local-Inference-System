#include "lis/loader.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int lis_path_has_suffix(const char *path, const char *suffix)
{
    const size_t path_len = strlen(path);
    const size_t suffix_len = strlen(suffix);

    if (path_len < suffix_len) {
        return 0;
    }

    return strcmp(path + path_len - suffix_len, suffix) == 0;
}

static uint64_t lis_read_u64_le(const unsigned char bytes[8])
{
    uint64_t value = 0;
    size_t index;

    for (index = 0; index < 8; ++index) {
        value |= ((uint64_t)bytes[index]) << (index * 8);
    }

    return value;
}

static lis_status lis_probe_safetensors_header(const char *path)
{
    unsigned char len_bytes[8] = { 0 };
    uint64_t header_len = 0;
    long file_size = 0;
    int first_header_byte = 0;
    FILE *fp = NULL;
    lis_status status = LIS_STATUS_FORMAT;

    fp = fopen(path, "rb");
    if (fp == NULL) {
        return LIS_STATUS_IO;
    }
    if (fread(len_bytes, 1, sizeof(len_bytes), fp) != sizeof(len_bytes)) {
        goto out;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        status = LIS_STATUS_IO;
        goto out;
    }

    file_size = ftell(fp);
    if (file_size < 0) {
        status = LIS_STATUS_IO;
        goto out;
    }

    header_len = lis_read_u64_le(len_bytes);
    if (header_len == 0 || header_len > (uint64_t)LONG_MAX ||
        header_len > 100 * 1024 * 1024 ||
        header_len > (uint64_t)file_size - 8U) {
        goto out;
    }
    if (fseek(fp, 8, SEEK_SET) != 0) {
        status = LIS_STATUS_IO;
        goto out;
    }

    first_header_byte = fgetc(fp);
    while (first_header_byte == ' ' || first_header_byte == '\n' ||
           first_header_byte == '\r' || first_header_byte == '\t') {
        first_header_byte = fgetc(fp);
    }
    if (first_header_byte == '{') {
        status = LIS_STATUS_OK;
    }

out:
    if (fclose(fp) != 0 && status == LIS_STATUS_OK) {
        status = LIS_STATUS_IO;
    }
    return status;
}

static lis_status lis_probe_hf_model(const char *path)
{
    char config_path[1024];
    FILE *fp = NULL;

    if (snprintf(config_path, sizeof(config_path), "%s/config.json", path) >=
        (int)sizeof(config_path)) {
        return LIS_STATUS_UNSUPPORTED_FORMAT;
    }

    fp = fopen(config_path, "rb");
    if (fp != NULL) {
        fclose(fp);
        return LIS_STATUS_OK;
    }
    return LIS_STATUS_UNSUPPORTED_FORMAT;
}

lis_model_source lis_model_source_from_path(const char *path)
{
    lis_model_source source = {
        .kind = LIS_MODEL_SOURCE_PATH,
        .path = path,
    };

    return source;
}

const char *lis_model_format_name(lis_model_format format)
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

lis_status lis_loader_probe_format(const lis_model_source *source,
                                   lis_model_format *out_format)
{
    lis_status status;

    if (source == NULL || out_format == NULL || source->path == NULL ||
        source->kind != LIS_MODEL_SOURCE_PATH) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    *out_format = LIS_MODEL_FORMAT_UNKNOWN;

    if (lis_path_has_suffix(source->path, ".pt") ||
        lis_path_has_suffix(source->path, ".pth") ||
        lis_path_has_suffix(source->path, ".bin")) {
        *out_format = LIS_MODEL_FORMAT_PYTORCH_UNSUPPORTED;
        return LIS_STATUS_UNSUPPORTED_FORMAT;
    }

    lis_status st_status = lis_probe_safetensors_header(source->path);
    if (st_status == LIS_STATUS_OK) {
        *out_format = LIS_MODEL_FORMAT_SAFETENSORS;
        return LIS_STATUS_OK;
    }

    status = lis_probe_hf_model(source->path);
    if (status == LIS_STATUS_OK) {
        *out_format = LIS_MODEL_FORMAT_HUGGINGFACE_LOCAL;
        return LIS_STATUS_OK;
    }

    if (lis_path_has_suffix(source->path, ".safetensors")) {
        return st_status;
    }

    return LIS_STATUS_UNSUPPORTED_FORMAT;
}

lis_status lis_loader_load(const lis_model_source *source,
                           lis_loaded_model *out_model)
{
    lis_model_format format = LIS_MODEL_FORMAT_UNKNOWN;
    lis_status status;

    if (out_model == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    status = lis_loader_probe_format(source, &format);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    switch (format) {
    case LIS_MODEL_FORMAT_SAFETENSORS:
        return lis_loader_load_safetensors(source, out_model);
    case LIS_MODEL_FORMAT_HUGGINGFACE_LOCAL:
        return lis_loader_load_hf_model(source, out_model);
    case LIS_MODEL_FORMAT_PYTORCH_UNSUPPORTED:
    case LIS_MODEL_FORMAT_UNKNOWN:
        break;
    }

    return LIS_STATUS_UNSUPPORTED_FORMAT;
}

lis_status lis_loaded_model_attach_metadata(lis_loaded_model *model,
                                            const lis_model_metadata *metadata)
{
    lis_model_metadata local_metadata = { 0 };
    lis_status status;
    size_t index;

    if (model == NULL || metadata == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    local_metadata = *metadata;
    for (index = 0; index < model->tensor_count; ++index) {
        if (strcmp(model->tensors[index].name, "lm_head.weight") == 0 ||
            strcmp(model->tensors[index].name, "lis.lm_head.weight") == 0) {
            local_metadata.config.tie_word_embeddings = 0;
            break;
        }
    }

    status = lis_model_metadata_validate(&local_metadata);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    model->metadata = local_metadata;
    model->has_metadata = true;
    return LIS_STATUS_OK;
}

void lis_loaded_model_destroy(lis_loaded_model *model)
{
    size_t index;

    if (model == NULL) {
        return;
    }

    for (index = 0; index < model->tensor_count; ++index) {
        free(model->tensors[index].name);
    }
    free(model->tensors);
    free(model->artifact_data);
    memset(model, 0, sizeof(*model));
}
