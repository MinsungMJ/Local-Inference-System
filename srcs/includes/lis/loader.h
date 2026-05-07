#ifndef LIS_LOADER_H
#define LIS_LOADER_H

#include "lis/model.h"
#include "lis/status.h"
#include "lis/tensor.h"

#include <stdbool.h>
#include <stddef.h>

typedef enum {
    LIS_MODEL_FORMAT_UNKNOWN = 0,
    LIS_MODEL_FORMAT_SAFETENSORS,
    LIS_MODEL_FORMAT_HUGGINGFACE_LOCAL,
    LIS_MODEL_FORMAT_PYTORCH_UNSUPPORTED
} lis_model_format;

typedef enum {
    LIS_MODEL_SOURCE_PATH = 1
} lis_model_source_kind;

typedef struct {
    lis_model_source_kind kind;
    const char *path;
} lis_model_source;

typedef struct {
    char *name;
    lis_tensor_view view;
    size_t data_begin;
    size_t data_end;
} lis_loaded_tensor;

typedef struct {
    lis_model_format format;
    bool has_metadata;
    lis_model_metadata metadata;
    unsigned char *artifact_data;
    size_t artifact_size;
    lis_loaded_tensor *tensors;
    size_t tensor_count;
} lis_loaded_model;

lis_model_source lis_model_source_from_path(const char *path);
const char *lis_model_format_name(lis_model_format format);

lis_status lis_loader_probe_format(const lis_model_source *source,
                                   lis_model_format *out_format);
lis_status lis_loader_load(const lis_model_source *source,
                           lis_loaded_model *out_model);
lis_status lis_loader_load_safetensors(const lis_model_source *source,
                                       lis_loaded_model *out_model);
lis_status lis_loader_load_hf_model(const lis_model_source *source,
                                    lis_loaded_model *out_model);
lis_status lis_loader_parse_llama3_config_json(const char *json,
                                               size_t json_len,
                                               lis_model_metadata *out_metadata);
lis_status lis_loader_parse_qwen3_config_json(const char *json,
                                              size_t json_len,
                                              lis_model_metadata *out_metadata);
lis_status lis_loader_parse_hf_config_json(const char *json,
                                           size_t json_len,
                                           lis_model_metadata *out_metadata);
lis_status lis_loaded_model_attach_metadata(lis_loaded_model *model,
                                            const lis_model_metadata *metadata);
void lis_loaded_model_destroy(lis_loaded_model *model);

#endif
