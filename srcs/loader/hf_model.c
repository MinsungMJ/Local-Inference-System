#include "lis/loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static lis_status lis_read_file(const char *path, char **out_data,
                                size_t *out_size)
{
    FILE *fp = NULL;
    long file_size = 0;
    char *data = NULL;
    lis_status status = LIS_STATUS_IO;

    fp = fopen(path, "rb");
    if (fp == NULL) {
        return LIS_STATUS_IO;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        goto out;
    }
    file_size = ftell(fp);
    if (file_size <= 0 || file_size > 32 * 1024 * 1024) {
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
    *out_size = (size_t)file_size;
    data = NULL;
    status = LIS_STATUS_OK;

out:
    free(data);
    if (fp != NULL && fclose(fp) != 0 && status == LIS_STATUS_OK) {
        status = LIS_STATUS_IO;
    }
    return status;
}

static lis_status lis_rename_tensor(lis_loaded_tensor *tensor,
                                    const char *new_name)
{
    char *name = malloc(strlen(new_name) + 1U);

    if (name == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }
    strcpy(name, new_name);
    free(tensor->name);
    tensor->name = name;
    return LIS_STATUS_OK;
}

static lis_status lis_validate_shape(const lis_tensor_view *view,
                                     const size_t *expected_dims,
                                     size_t rank, lis_dtype expected_dtype)
{
    size_t index;

    if (view->dtype != expected_dtype) {
        return LIS_STATUS_UNSUPPORTED_DTYPE;
    }
    if (view->shape.rank != rank) {
        return LIS_STATUS_UNSUPPORTED_SHAPE;
    }
    for (index = 0; index < rank; ++index) {
        if (view->shape.dims[index] != expected_dims[index]) {
            return LIS_STATUS_SHAPE_MISMATCH;
        }
    }

    return LIS_STATUS_OK;
}

static lis_status lis_map_llama_hf_tensors(lis_loaded_model *model)
{
    const lis_model_config *cfg = &model->metadata.config;
    const lis_dtype dtype = cfg->weight_dtype;
    size_t index;
    int has_embed = 0;
    int has_output_norm = 0;
    int has_lm_head = 0;
    char *layer_flags = NULL;
    lis_status status = LIS_STATUS_OK;

    layer_flags = calloc(cfg->layer_count, 9);
    if (layer_flags == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }

    /* Valid shapes for Llama */
    size_t embed_shape[2] = { cfg->vocab_size, cfg->hidden_size };
    size_t q_shape[2] = { cfg->attention_head_count * cfg->head_dim, cfg->hidden_size };
    size_t k_shape[2] = { cfg->kv_head_count * cfg->head_dim, cfg->hidden_size };
    size_t v_shape[2] = { cfg->kv_head_count * cfg->head_dim, cfg->hidden_size };
    size_t o_shape[2] = { cfg->hidden_size, cfg->attention_head_count * cfg->head_dim };
    size_t gate_shape[2] = { cfg->intermediate_size, cfg->hidden_size };
    size_t up_shape[2] = { cfg->intermediate_size, cfg->hidden_size };
    size_t down_shape[2] = { cfg->hidden_size, cfg->intermediate_size };
    size_t norm_shape[1] = { cfg->hidden_size };

    for (index = 0; index < model->tensor_count; ++index) {
        lis_loaded_tensor *t = &model->tensors[index];
        const char *name = t->name;
        size_t layer_idx = 0;
        char new_name[256];

        if (strcmp(name, "model.embed_tokens.weight") == 0) {
            if (has_embed) {
                status = LIS_STATUS_FORMAT;
                goto out;
            }
            status = lis_validate_shape(&t->view, embed_shape, 2, dtype);
            if (status != LIS_STATUS_OK) goto out;
            status = lis_rename_tensor(t, "lis.token_embeddings.weight");
            if (status != LIS_STATUS_OK) goto out;
            has_embed = 1;
            continue;
        }

        if (strcmp(name, "model.norm.weight") == 0) {
            if (has_output_norm) {
                status = LIS_STATUS_FORMAT;
                goto out;
            }
            status = lis_validate_shape(&t->view, norm_shape, 1, dtype);
            if (status != LIS_STATUS_OK) goto out;
            status = lis_rename_tensor(t, "lis.output_norm.weight");
            if (status != LIS_STATUS_OK) goto out;
            has_output_norm = 1;
            continue;
        }

        if (strcmp(name, "lm_head.weight") == 0) {
            if (has_lm_head) {
                status = LIS_STATUS_FORMAT;
                goto out;
            }
            status = lis_validate_shape(&t->view, embed_shape, 2, dtype);
            if (status != LIS_STATUS_OK) goto out;
            status = lis_rename_tensor(t, "lis.lm_head.weight");
            if (status != LIS_STATUS_OK) goto out;
            has_lm_head = 1;
            continue;
        }

        const char *prefix = "model.layers.";
        if (strncmp(name, prefix, strlen(prefix)) == 0) {
            char *endptr;
            layer_idx = strtoul(name + strlen(prefix), &endptr, 10);
            if (*endptr == '.') {
                const char *suffix = endptr + 1;

                if (layer_idx >= cfg->layer_count) { status = LIS_STATUS_FORMAT; goto out; }

                if (strcmp(suffix, "self_attn.q_proj.weight") == 0) {
                    status = lis_validate_shape(&t->view, q_shape, 2, dtype);
                    if (status != LIS_STATUS_OK) goto out;
                    snprintf(new_name, sizeof(new_name), "lis.layer.%zu.q_proj.weight", layer_idx);
                    status = lis_rename_tensor(t, new_name);
                    if (status != LIS_STATUS_OK) goto out;
                    if (layer_flags[layer_idx * 9 + 0]++) { status = LIS_STATUS_FORMAT; goto out; }
                    continue;
                }
                if (strcmp(suffix, "self_attn.k_proj.weight") == 0) {
                    status = lis_validate_shape(&t->view, k_shape, 2, dtype);
                    if (status != LIS_STATUS_OK) goto out;
                    snprintf(new_name, sizeof(new_name), "lis.layer.%zu.k_proj.weight", layer_idx);
                    status = lis_rename_tensor(t, new_name);
                    if (status != LIS_STATUS_OK) goto out;
                    if (layer_flags[layer_idx * 9 + 1]++) { status = LIS_STATUS_FORMAT; goto out; }
                    continue;
                }
                if (strcmp(suffix, "self_attn.v_proj.weight") == 0) {
                    status = lis_validate_shape(&t->view, v_shape, 2, dtype);
                    if (status != LIS_STATUS_OK) goto out;
                    snprintf(new_name, sizeof(new_name), "lis.layer.%zu.v_proj.weight", layer_idx);
                    status = lis_rename_tensor(t, new_name);
                    if (status != LIS_STATUS_OK) goto out;
                    if (layer_flags[layer_idx * 9 + 2]++) { status = LIS_STATUS_FORMAT; goto out; }
                    continue;
                }
                if (strcmp(suffix, "self_attn.o_proj.weight") == 0) {
                    status = lis_validate_shape(&t->view, o_shape, 2, dtype);
                    if (status != LIS_STATUS_OK) goto out;
                    snprintf(new_name, sizeof(new_name), "lis.layer.%zu.o_proj.weight", layer_idx);
                    status = lis_rename_tensor(t, new_name);
                    if (status != LIS_STATUS_OK) goto out;
                    if (layer_flags[layer_idx * 9 + 3]++) { status = LIS_STATUS_FORMAT; goto out; }
                    continue;
                }
                if (strcmp(suffix, "mlp.gate_proj.weight") == 0) {
                    status = lis_validate_shape(&t->view, gate_shape, 2, dtype);
                    if (status != LIS_STATUS_OK) goto out;
                    snprintf(new_name, sizeof(new_name), "lis.layer.%zu.gate_proj.weight", layer_idx);
                    status = lis_rename_tensor(t, new_name);
                    if (status != LIS_STATUS_OK) goto out;
                    if (layer_flags[layer_idx * 9 + 4]++) { status = LIS_STATUS_FORMAT; goto out; }
                    continue;
                }
                if (strcmp(suffix, "mlp.up_proj.weight") == 0) {
                    status = lis_validate_shape(&t->view, up_shape, 2, dtype);
                    if (status != LIS_STATUS_OK) goto out;
                    snprintf(new_name, sizeof(new_name), "lis.layer.%zu.up_proj.weight", layer_idx);
                    status = lis_rename_tensor(t, new_name);
                    if (status != LIS_STATUS_OK) goto out;
                    if (layer_flags[layer_idx * 9 + 5]++) { status = LIS_STATUS_FORMAT; goto out; }
                    continue;
                }
                if (strcmp(suffix, "mlp.down_proj.weight") == 0) {
                    status = lis_validate_shape(&t->view, down_shape, 2, dtype);
                    if (status != LIS_STATUS_OK) goto out;
                    snprintf(new_name, sizeof(new_name), "lis.layer.%zu.down_proj.weight", layer_idx);
                    status = lis_rename_tensor(t, new_name);
                    if (status != LIS_STATUS_OK) goto out;
                    if (layer_flags[layer_idx * 9 + 6]++) { status = LIS_STATUS_FORMAT; goto out; }
                    continue;
                }
                if (strcmp(suffix, "input_layernorm.weight") == 0) {
                    status = lis_validate_shape(&t->view, norm_shape, 1, dtype);
                    if (status != LIS_STATUS_OK) goto out;
                    snprintf(new_name, sizeof(new_name), "lis.layer.%zu.attention_norm.weight", layer_idx);
                    status = lis_rename_tensor(t, new_name);
                    if (status != LIS_STATUS_OK) goto out;
                    if (layer_flags[layer_idx * 9 + 7]++) { status = LIS_STATUS_FORMAT; goto out; }
                    continue;
                }
                if (strcmp(suffix, "post_attention_layernorm.weight") == 0) {
                    status = lis_validate_shape(&t->view, norm_shape, 1, dtype);
                    if (status != LIS_STATUS_OK) goto out;
                    snprintf(new_name, sizeof(new_name), "lis.layer.%zu.mlp_norm.weight", layer_idx);
                    status = lis_rename_tensor(t, new_name);
                    if (status != LIS_STATUS_OK) goto out;
                    if (layer_flags[layer_idx * 9 + 8]++) { status = LIS_STATUS_FORMAT; goto out; }
                    continue;
                }
            }
        }

        /* Unknown or unsupported tensor layout in the model */
        status = LIS_STATUS_FORMAT;
        goto out;
    }

    if (!has_embed || !has_output_norm ||
        (!has_lm_head && !cfg->tie_word_embeddings)) {
        status = LIS_STATUS_FORMAT;
        goto out;
    }
    for (index = 0; index < cfg->layer_count * 9; ++index) {
        if (layer_flags[index] == 0) {
            status = LIS_STATUS_FORMAT;
            goto out;
        }
    }

out:
    if (status == LIS_STATUS_OK && has_lm_head) {
        model->metadata.config.tie_word_embeddings = 0;
    }
    free(layer_flags);
    return status;
}

static lis_status lis_map_qwen3_hf_tensors(lis_loaded_model *model)
{
    const lis_model_config *cfg = &model->metadata.config;
    const lis_dtype dtype = cfg->weight_dtype;
    const size_t per_layer = 11U;
    size_t index;
    int has_embed = 0;
    int has_output_norm = 0;
    int has_lm_head = 0;
    char *layer_flags = NULL;
    lis_status status = LIS_STATUS_OK;

    layer_flags = calloc(cfg->layer_count, per_layer);
    if (layer_flags == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }

    size_t embed_shape[2] = { cfg->vocab_size, cfg->hidden_size };
    size_t q_shape[2] = { cfg->attention_head_count * cfg->head_dim,
                          cfg->hidden_size };
    size_t k_shape[2] = { cfg->kv_head_count * cfg->head_dim,
                          cfg->hidden_size };
    size_t v_shape[2] = { cfg->kv_head_count * cfg->head_dim,
                          cfg->hidden_size };
    size_t o_shape[2] = { cfg->hidden_size,
                          cfg->attention_head_count * cfg->head_dim };
    size_t qk_norm_shape[1] = { cfg->head_dim };
    size_t gate_shape[2] = { cfg->intermediate_size, cfg->hidden_size };
    size_t up_shape[2] = { cfg->intermediate_size, cfg->hidden_size };
    size_t down_shape[2] = { cfg->hidden_size, cfg->intermediate_size };
    size_t norm_shape[1] = { cfg->hidden_size };

    for (index = 0; index < model->tensor_count; ++index) {
        lis_loaded_tensor *t = &model->tensors[index];
        const char *name = t->name;
        size_t layer_idx = 0;
        char new_name[256];

        if (strcmp(name, "model.embed_tokens.weight") == 0) {
            if (has_embed) {
                status = LIS_STATUS_FORMAT;
                goto out;
            }
            status = lis_validate_shape(&t->view, embed_shape, 2, dtype);
            if (status != LIS_STATUS_OK) goto out;
            status = lis_rename_tensor(t, "lis.token_embeddings.weight");
            if (status != LIS_STATUS_OK) goto out;
            has_embed = 1;
            continue;
        }
        if (strcmp(name, "model.norm.weight") == 0) {
            if (has_output_norm) {
                status = LIS_STATUS_FORMAT;
                goto out;
            }
            status = lis_validate_shape(&t->view, norm_shape, 1, dtype);
            if (status != LIS_STATUS_OK) goto out;
            status = lis_rename_tensor(t, "lis.output_norm.weight");
            if (status != LIS_STATUS_OK) goto out;
            has_output_norm = 1;
            continue;
        }
        if (strcmp(name, "lm_head.weight") == 0) {
            if (has_lm_head) {
                status = LIS_STATUS_FORMAT;
                goto out;
            }
            status = lis_validate_shape(&t->view, embed_shape, 2, dtype);
            if (status != LIS_STATUS_OK) goto out;
            status = lis_rename_tensor(t, "lis.lm_head.weight");
            if (status != LIS_STATUS_OK) goto out;
            has_lm_head = 1;
            continue;
        }

        const char *prefix = "model.layers.";
        if (strncmp(name, prefix, strlen(prefix)) == 0) {
            char *endptr;
            layer_idx = strtoul(name + strlen(prefix), &endptr, 10);
            if (*endptr == '.') {
                const char *suffix = endptr + 1;
                size_t flag = 0;
                const size_t *shape = NULL;
                size_t rank = 0;
                const char *mapped_suffix = NULL;

                if (layer_idx >= cfg->layer_count) {
                    status = LIS_STATUS_FORMAT;
                    goto out;
                }
                if (strcmp(suffix, "self_attn.q_proj.weight") == 0) {
                    flag = 0; shape = q_shape; rank = 2;
                    mapped_suffix = "q_proj.weight";
                } else if (strcmp(suffix, "self_attn.k_proj.weight") == 0) {
                    flag = 1; shape = k_shape; rank = 2;
                    mapped_suffix = "k_proj.weight";
                } else if (strcmp(suffix, "self_attn.v_proj.weight") == 0) {
                    flag = 2; shape = v_shape; rank = 2;
                    mapped_suffix = "v_proj.weight";
                } else if (strcmp(suffix, "self_attn.o_proj.weight") == 0) {
                    flag = 3; shape = o_shape; rank = 2;
                    mapped_suffix = "o_proj.weight";
                } else if (strcmp(suffix, "self_attn.q_norm.weight") == 0) {
                    flag = 4; shape = qk_norm_shape; rank = 1;
                    mapped_suffix = "q_norm.weight";
                } else if (strcmp(suffix, "self_attn.k_norm.weight") == 0) {
                    flag = 5; shape = qk_norm_shape; rank = 1;
                    mapped_suffix = "k_norm.weight";
                } else if (strcmp(suffix, "mlp.gate_proj.weight") == 0) {
                    flag = 6; shape = gate_shape; rank = 2;
                    mapped_suffix = "gate_proj.weight";
                } else if (strcmp(suffix, "mlp.up_proj.weight") == 0) {
                    flag = 7; shape = up_shape; rank = 2;
                    mapped_suffix = "up_proj.weight";
                } else if (strcmp(suffix, "mlp.down_proj.weight") == 0) {
                    flag = 8; shape = down_shape; rank = 2;
                    mapped_suffix = "down_proj.weight";
                } else if (strcmp(suffix, "input_layernorm.weight") == 0) {
                    flag = 9; shape = norm_shape; rank = 1;
                    mapped_suffix = "attention_norm.weight";
                } else if (strcmp(suffix,
                                  "post_attention_layernorm.weight") == 0) {
                    flag = 10; shape = norm_shape; rank = 1;
                    mapped_suffix = "mlp_norm.weight";
                }
                if (mapped_suffix != NULL) {
                    status = lis_validate_shape(&t->view, shape, rank, dtype);
                    if (status != LIS_STATUS_OK) goto out;
                    snprintf(new_name, sizeof(new_name),
                             "lis.layer.%zu.%s", layer_idx, mapped_suffix);
                    status = lis_rename_tensor(t, new_name);
                    if (status != LIS_STATUS_OK) goto out;
                    if (layer_flags[layer_idx * per_layer + flag]++) {
                        status = LIS_STATUS_FORMAT;
                        goto out;
                    }
                    continue;
                }
            }
        }

        status = LIS_STATUS_FORMAT;
        goto out;
    }

    if (!has_embed || !has_output_norm ||
        (!has_lm_head && !cfg->tie_word_embeddings)) {
        status = LIS_STATUS_FORMAT;
        goto out;
    }
    for (index = 0; index < cfg->layer_count * per_layer; ++index) {
        if (layer_flags[index] == 0) {
            status = LIS_STATUS_FORMAT;
            goto out;
        }
    }

out:
    if (status == LIS_STATUS_OK && has_lm_head) {
        model->metadata.config.tie_word_embeddings = 0;
    }
    free(layer_flags);
    return status;
}

static lis_status lis_map_hf_tensors(lis_loaded_model *model)
{
    if (model == NULL || !model->has_metadata) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (model->metadata.config.family == LIS_MODEL_FAMILY_LLAMA3_DECODER) {
        return lis_map_llama_hf_tensors(model);
    }
    if (model->metadata.config.family ==
        LIS_MODEL_FAMILY_QWEN3_DENSE_DECODER) {
        return lis_map_qwen3_hf_tensors(model);
    }
    return LIS_STATUS_UNSUPPORTED_FORMAT;
}

lis_status lis_loader_load_hf_model(const lis_model_source *source,
                                    lis_loaded_model *out_model)
{
    char config_path[1024];
    char st_path[1024];
    char st_index_path[1024];
    char *config_json = NULL;
    size_t config_len = 0;
    lis_model_metadata metadata = { 0 };
    lis_model_source st_source = { 0 };
    lis_status status;
    FILE *fp = NULL;
    FILE *st_fp = NULL;

    if (source == NULL || out_model == NULL || source->path == NULL ||
        source->kind != LIS_MODEL_SOURCE_PATH) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    if (snprintf(config_path, sizeof(config_path), "%s/config.json",
                 source->path) >= (int)sizeof(config_path)) {
        return LIS_STATUS_FORMAT;
    }
    if (snprintf(st_index_path, sizeof(st_index_path),
                 "%s/model.safetensors.index.json",
                 source->path) >= (int)sizeof(st_index_path)) {
        return LIS_STATUS_FORMAT;
    }
    if (snprintf(st_path, sizeof(st_path), "%s/model.safetensors",
                 source->path) >= (int)sizeof(st_path)) {
        return LIS_STATUS_FORMAT;
    }

    st_fp = fopen(st_path, "rb");
    if (st_fp != NULL) {
        if (fclose(st_fp) != 0) {
            return LIS_STATUS_IO;
        }
    } else {
        fp = fopen(st_index_path, "rb");
        if (fp != NULL) {
            fclose(fp);
            return LIS_STATUS_UNSUPPORTED_FORMAT;
        }
        return LIS_STATUS_IO;
    }

    status = lis_read_file(config_path, &config_json, &config_len);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    status = lis_loader_parse_hf_config_json(config_json, config_len,
                                             &metadata);
    free(config_json);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    st_source = lis_model_source_from_path(st_path);
    status = lis_loader_load_safetensors(&st_source, out_model);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    status = lis_loaded_model_attach_metadata(out_model, &metadata);
    if (status != LIS_STATUS_OK) {
        lis_loaded_model_destroy(out_model);
        return status;
    }

    status = lis_map_hf_tensors(out_model);
    if (status != LIS_STATUS_OK) {
        lis_loaded_model_destroy(out_model);
        return status;
    }

    out_model->format = LIS_MODEL_FORMAT_HUGGINGFACE_LOCAL;
    return LIS_STATUS_OK;
}
