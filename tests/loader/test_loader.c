#include "lis/loader.h"
#include "lis/status.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures;

static void expect_status(const char *name, lis_status actual,
                          lis_status expected)
{
    if (actual != expected) {
        fprintf(stderr, "%s: expected %s, got %s\n", name,
                lis_status_name(expected), lis_status_name(actual));
        ++g_failures;
    }
}

static void expect_size(const char *name, size_t actual, size_t expected)
{
    if (actual != expected) {
        fprintf(stderr, "%s: expected %zu, got %zu\n", name, expected,
                actual);
        ++g_failures;
    }
}

static void write_u64_le(FILE *fp, uint64_t value)
{
    size_t index;

    for (index = 0; index < 8; ++index) {
        fputc((int)((value >> (index * 8)) & 0xffU), fp);
    }
}

static lis_status write_safetensors_file(const char *path, const char *header,
                                         const unsigned char *data,
                                         size_t data_len)
{
    FILE *fp = fopen(path, "wb");

    if (fp == NULL) {
        return LIS_STATUS_IO;
    }
    write_u64_le(fp, (uint64_t)strlen(header));
    if (fwrite(header, 1, strlen(header), fp) != strlen(header)) {
        fclose(fp);
        return LIS_STATUS_IO;
    }
    if (data_len != 0 && fwrite(data, 1, data_len, fp) != data_len) {
        fclose(fp);
        return LIS_STATUS_IO;
    }
    if (fclose(fp) != 0) {
        return LIS_STATUS_IO;
    }

    return LIS_STATUS_OK;
}

static void test_probe_and_load_safetensors(void)
{
    const char *path = "srcs/libs/test_valid.safetensors";
    const char *header =
        "{\"weight\":{\"dtype\":\"F32\",\"shape\":[2,2],"
        "\"data_offsets\":[0,16]},\"__metadata__\":{\"format\":\"pt\"}}";
    const unsigned char data[16] = { 0 };
    lis_model_source source = lis_model_source_from_path(path);
    lis_model_format format = LIS_MODEL_FORMAT_UNKNOWN;
    lis_loaded_model model = { 0 };

    expect_status("write valid safetensors",
                  write_safetensors_file(path, header, data, sizeof(data)),
                  LIS_STATUS_OK);
    expect_status("probe safetensors",
                  lis_loader_probe_format(&source, &format), LIS_STATUS_OK);
    expect_size("format safetensors", (size_t)format,
                (size_t)LIS_MODEL_FORMAT_SAFETENSORS);
    expect_status("load safetensors", lis_loader_load(&source, &model),
                  LIS_STATUS_OK);
    expect_size("tensor count", model.tensor_count, 1);
    expect_size("metadata absent", (size_t)model.has_metadata, 0);
    expect_size("tensor dtype", (size_t)model.tensors[0].view.dtype,
                (size_t)LIS_DTYPE_F32);
    expect_size("tensor rank", model.tensors[0].view.shape.rank, 2);
    expect_size("tensor dim 0", model.tensors[0].view.shape.dims[0], 2);
    expect_size("tensor dim 1", model.tensors[0].view.shape.dims[1], 2);
    expect_size("tensor byte length", model.tensors[0].view.byte_len, 16);
    if (strcmp(model.tensors[0].name, "weight") != 0) {
        fprintf(stderr, "tensor name mismatch\n");
        ++g_failures;
    }
    lis_loaded_model_destroy(&model);
    remove(path);
}

static void test_load_safetensors_precision_dtypes(void)
{
    const char *path = "srcs/libs/test_precision_dtypes.safetensors";
    const char *header =
        "{\"half\":{\"dtype\":\"F16\",\"shape\":[2],"
        "\"data_offsets\":[0,4]},"
        "\"brain\":{\"dtype\":\"BF16\",\"shape\":[2],"
        "\"data_offsets\":[4,8]}}";
    const unsigned char data[8] = { 0 };
    lis_model_source source = lis_model_source_from_path(path);
    lis_loaded_model model = { 0 };

    expect_status("write precision safetensors",
                  write_safetensors_file(path, header, data, sizeof(data)),
                  LIS_STATUS_OK);
    expect_status("load precision safetensors", lis_loader_load(&source, &model),
                  LIS_STATUS_OK);
    expect_size("precision tensor count", model.tensor_count, 2);
    expect_size("f16 tensor dtype", (size_t)model.tensors[0].view.dtype,
                (size_t)LIS_DTYPE_F16);
    expect_size("f16 tensor byte length", model.tensors[0].view.byte_len, 4);
    expect_size("bf16 tensor dtype", (size_t)model.tensors[1].view.dtype,
                (size_t)LIS_DTYPE_BF16);
    expect_size("bf16 tensor byte length", model.tensors[1].view.byte_len, 4);
    lis_loaded_model_destroy(&model);
    remove(path);
}

static void test_malformed_and_unsupported_safetensors(void)
{
    const char *bad_path = "srcs/libs/test_bad.safetensors";
    const char *dtype_path = "srcs/libs/test_unsupported_dtype.safetensors";
    const char *truncated_path = "srcs/libs/test_truncated.safetensors";
    const char *empty_shape_path = "srcs/libs/test_empty_shape.safetensors";
    const char *mismatch_path = "srcs/libs/test_shape_mismatch.safetensors";
    const char *unsupported_header =
        "{\"weight\":{\"dtype\":\"I64\",\"shape\":[1],"
        "\"data_offsets\":[0,8]}}";
    const char *truncated_header =
        "{\"weight\":{\"dtype\":\"F32\",\"shape\":[2,2],"
        "\"data_offsets\":[0,16]}}";
    const char *empty_shape_header =
        "{\"weight\":{\"dtype\":\"F32\",\"shape\":[],"
        "\"data_offsets\":[0,0]}}";
    const char *mismatch_header =
        "{\"weight\":{\"dtype\":\"F32\",\"shape\":[2,2],"
        "\"data_offsets\":[0,12]}}";
    const unsigned char data[8] = { 0 };
    const unsigned char mismatch_data[12] = { 0 };
    FILE *fp = fopen(bad_path, "wb");
    lis_model_source source = lis_model_source_from_path(bad_path);
    lis_loaded_model model = { 0 };

    if (fp != NULL) {
        fputs("bad", fp);
        fclose(fp);
    }
    expect_status("load malformed safetensors", lis_loader_load(&source, &model),
                  LIS_STATUS_FORMAT);
    lis_loaded_model_destroy(&model);
    remove(bad_path);

    source = lis_model_source_from_path(dtype_path);
    expect_status("write unsupported dtype",
                  write_safetensors_file(dtype_path, unsupported_header, data,
                                         sizeof(data)),
                  LIS_STATUS_OK);
    expect_status("load unsupported dtype", lis_loader_load(&source, &model),
                  LIS_STATUS_UNSUPPORTED_DTYPE);
    lis_loaded_model_destroy(&model);
    remove(dtype_path);

    source = lis_model_source_from_path(truncated_path);
    expect_status("write truncated tensor",
                  write_safetensors_file(truncated_path, truncated_header,
                                         data, sizeof(data)),
                  LIS_STATUS_OK);
    expect_status("load truncated tensor", lis_loader_load(&source, &model),
                  LIS_STATUS_FORMAT);
    lis_loaded_model_destroy(&model);
    remove(truncated_path);

    source = lis_model_source_from_path(empty_shape_path);
    expect_status("write empty shape",
                  write_safetensors_file(empty_shape_path, empty_shape_header,
                                         data, 0),
                  LIS_STATUS_OK);
    expect_status("load empty shape", lis_loader_load(&source, &model),
                  LIS_STATUS_UNSUPPORTED_SHAPE);
    lis_loaded_model_destroy(&model);
    remove(empty_shape_path);

    source = lis_model_source_from_path(mismatch_path);
    expect_status("write byte mismatch",
                  write_safetensors_file(mismatch_path, mismatch_header,
                                         mismatch_data, sizeof(mismatch_data)),
                  LIS_STATUS_OK);
    expect_status("load byte mismatch", lis_loader_load(&source, &model),
                  LIS_STATUS_SHAPE_MISMATCH);
    lis_loaded_model_destroy(&model);
    remove(mismatch_path);

    /* Defensively reject massive headers. */
    const char *oversized_path = "srcs/libs/test_oversize.safetensors";
    fp = fopen(oversized_path, "wb");
    if (fp != NULL) {
        write_u64_le(fp, 100ULL * 1024ULL * 1024ULL + 1ULL); /* > 100MB bound */
        fclose(fp);
    }
    source = lis_model_source_from_path(oversized_path);
    expect_status("load oversized safetensors header",
                  lis_loader_load(&source, &model), LIS_STATUS_FORMAT);
    remove(oversized_path);
}

static void test_unsupported_pytorch_path(void)
{
    lis_model_source source = lis_model_source_from_path("model.pt");
    lis_model_format format = LIS_MODEL_FORMAT_UNKNOWN;

    expect_status("probe pytorch unsupported",
                  lis_loader_probe_format(&source, &format),
                  LIS_STATUS_UNSUPPORTED_FORMAT);
    expect_size("pytorch unsupported format", (size_t)format,
                (size_t)LIS_MODEL_FORMAT_PYTORCH_UNSUPPORTED);
}

static void test_llama_config_parser(void)
{
    const char *json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":2,"
        "\"hidden_size\":128,\"intermediate_size\":256,"
        "\"num_attention_heads\":4,\"num_key_value_heads\":2,"
        "\"head_dim\":32,\"vocab_size\":32000,"
        "\"rope_theta\":500000.0,\"torch_dtype\":\"float16\","
        "\"max_position_embeddings\":8192}";
    const char *unsupported_json =
        "{\"model_type\":\"gpt2\",\"num_hidden_layers\":2,"
        "\"hidden_size\":128,\"intermediate_size\":256,"
        "\"num_attention_heads\":4,\"head_dim\":32,"
        "\"vocab_size\":32000,\"rope_theta\":10000.0,"
        "\"torch_dtype\":\"float16\",\"max_position_embeddings\":1024}";
    const char *derived_head_dim_json =
        "{\"model_type\":\"llama3\",\"num_hidden_layers\":2,"
        "\"hidden_size\":128,\"intermediate_size\":256,"
        "\"num_attention_heads\":4,\"num_key_value_heads\":2,"
        "\"vocab_size\":32000,\"rope_theta\":500000.0,"
        "\"torch_dtype\":\"bf16\",\"max_position_embeddings\":8192}";
    const char *eos_array_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":2,"
        "\"hidden_size\":128,\"intermediate_size\":256,"
        "\"num_attention_heads\":4,\"num_key_value_heads\":2,"
        "\"head_dim\":32,\"vocab_size\":32000,"
        "\"rope_theta\":500000.0,\"torch_dtype\":\"float16\","
        "\"eos_token_id\":[1,2,3],"
        "\"max_position_embeddings\":8192}";
    const char *eos_scalar_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":2,"
        "\"hidden_size\":128,\"intermediate_size\":256,"
        "\"num_attention_heads\":4,\"num_key_value_heads\":2,"
        "\"head_dim\":32,\"vocab_size\":32000,"
        "\"rope_theta\":500000.0,\"torch_dtype\":\"float16\","
        "\"eos_token_id\":1,"
        "\"max_position_embeddings\":8192}";
    const char *bad_dtype_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":2,"
        "\"hidden_size\":128,\"intermediate_size\":256,"
        "\"num_attention_heads\":4,\"head_dim\":32,"
        "\"vocab_size\":32000,\"rope_theta\":10000.0,"
        "\"torch_dtype\":\"int8\",\"max_position_embeddings\":1024}";
    const char *missing_field_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":2,"
        "\"hidden_size\":128,\"intermediate_size\":256,"
        "\"num_attention_heads\":4,\"head_dim\":32,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float16\","
        "\"max_position_embeddings\":1024}";
    const char *bad_eos_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":2,"
        "\"hidden_size\":128,\"intermediate_size\":256,"
        "\"num_attention_heads\":4,\"num_key_value_heads\":2,"
        "\"head_dim\":32,\"vocab_size\":32000,"
        "\"rope_theta\":500000.0,\"torch_dtype\":\"float16\","
        "\"eos_token_id\":[1,\"x\"],"
        "\"max_position_embeddings\":8192}";
    const char *out_of_vocab_eos_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":2,"
        "\"hidden_size\":128,\"intermediate_size\":256,"
        "\"num_attention_heads\":4,\"num_key_value_heads\":2,"
        "\"head_dim\":32,\"vocab_size\":3,"
        "\"rope_theta\":500000.0,\"torch_dtype\":\"float16\","
        "\"eos_token_id\":3,"
        "\"max_position_embeddings\":8192}";
    const char *rope_scaling_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":2,"
        "\"hidden_size\":128,\"intermediate_size\":256,"
        "\"num_attention_heads\":4,\"num_key_value_heads\":2,"
        "\"head_dim\":32,\"vocab_size\":32000,"
        "\"rope_theta\":500000.0,"
        "\"rope_scaling\":{\"rope_type\":\"llama3\",\"factor\":32.0},"
        "\"torch_dtype\":\"float16\","
        "\"max_position_embeddings\":8192}";
    const char *rope_scaling_null_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":2,"
        "\"hidden_size\":128,\"intermediate_size\":256,"
        "\"num_attention_heads\":4,\"num_key_value_heads\":2,"
        "\"head_dim\":32,\"vocab_size\":32000,"
        "\"rope_theta\":500000.0,\"rope_scaling\":null,"
        "\"torch_dtype\":\"float16\","
        "\"max_position_embeddings\":8192}";
    const char *unsupported_rope_type_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":2,"
        "\"hidden_size\":128,\"intermediate_size\":256,"
        "\"num_attention_heads\":4,\"num_key_value_heads\":2,"
        "\"head_dim\":32,\"vocab_size\":32000,"
        "\"rope_theta\":500000.0,\"rope_type\":\"llama3\","
        "\"torch_dtype\":\"float16\","
        "\"max_position_embeddings\":8192}";
    const char *default_rope_type_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":2,"
        "\"hidden_size\":128,\"intermediate_size\":256,"
        "\"num_attention_heads\":4,\"num_key_value_heads\":2,"
        "\"head_dim\":32,\"vocab_size\":32000,"
        "\"rope_theta\":500000.0,\"rope_type\":\"default\","
        "\"torch_dtype\":\"float16\","
        "\"max_position_embeddings\":8192}";
    const char *rope_scaling_empty_object_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":2,"
        "\"hidden_size\":128,\"intermediate_size\":256,"
        "\"num_attention_heads\":4,\"num_key_value_heads\":2,"
        "\"head_dim\":32,\"vocab_size\":32000,"
        "\"rope_theta\":500000.0,\"rope_scaling\":{},"
        "\"torch_dtype\":\"float16\","
        "\"max_position_embeddings\":8192}";
    const char *rope_scaling_linear_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":2,"
        "\"hidden_size\":128,\"intermediate_size\":256,"
        "\"num_attention_heads\":4,\"num_key_value_heads\":2,"
        "\"head_dim\":32,\"vocab_size\":32000,"
        "\"rope_theta\":500000.0,"
        "\"rope_scaling\":{\"type\":\"linear\",\"factor\":2.0},"
        "\"torch_dtype\":\"float16\","
        "\"max_position_embeddings\":8192}";
    const char *meta_llama3_8b_instruct_json =
        "{\"architectures\":[\"LlamaForCausalLM\"],"
        "\"attention_bias\":false,\"attention_dropout\":0.0,"
        "\"bos_token_id\":128000,\"eos_token_id\":128009,"
        "\"hidden_act\":\"silu\",\"hidden_size\":4096,"
        "\"initializer_range\":0.02,\"intermediate_size\":14336,"
        "\"max_position_embeddings\":8192,\"model_type\":\"llama\","
        "\"num_attention_heads\":32,\"num_hidden_layers\":32,"
        "\"num_key_value_heads\":8,\"pretraining_tp\":1,"
        "\"rms_norm_eps\":1e-05,\"rope_scaling\":null,"
        "\"rope_theta\":500000.0,\"tie_word_embeddings\":false,"
        "\"torch_dtype\":\"bfloat16\","
        "\"transformers_version\":\"4.40.0.dev0\","
        "\"use_cache\":true,\"vocab_size\":128256}";
    lis_model_metadata metadata = { 0 };
    lis_loaded_model model = { 0 };

    expect_status("parse llama config",
                  lis_loader_parse_llama3_config_json(json, strlen(json),
                                                      &metadata),
                  LIS_STATUS_OK);
    expect_size("metadata family", (size_t)metadata.config.family,
                (size_t)LIS_MODEL_FAMILY_LLAMA3_DECODER);
    expect_size("trained context",
                metadata.config.context.trained_max_tokens, 8192);
    expect_status("attach metadata",
                  lis_loaded_model_attach_metadata(&model, &metadata),
                  LIS_STATUS_OK);
    expect_size("attached metadata", (size_t)model.has_metadata, 1);

    expect_status("unsupported config model type",
                  lis_loader_parse_llama3_config_json(unsupported_json,
                                                      strlen(unsupported_json),
                                                      &metadata),
                  LIS_STATUS_UNSUPPORTED_FORMAT);
    expect_status("parse derived head dim",
                  lis_loader_parse_llama3_config_json(derived_head_dim_json,
                                                      strlen(derived_head_dim_json),
                                                      &metadata),
                  LIS_STATUS_OK);
    expect_size("derived head dim", metadata.config.head_dim, 32);
    expect_size("derived dtype", (size_t)metadata.config.weight_dtype,
                (size_t)LIS_DTYPE_BF16);
    expect_status("parse eos array",
                  lis_loader_parse_llama3_config_json(eos_array_json,
                                                      strlen(eos_array_json),
                                                      &metadata),
                  LIS_STATUS_OK);
    expect_size("eos array count", metadata.config.eos_token_count, 3);
    expect_size("eos array value 2", metadata.config.eos_token_ids[2], 3);
    expect_status("parse eos scalar",
                  lis_loader_parse_llama3_config_json(eos_scalar_json,
                                                      strlen(eos_scalar_json),
                                                      &metadata),
                  LIS_STATUS_OK);
    expect_size("eos scalar count", metadata.config.eos_token_count, 1);
    expect_size("eos scalar value", metadata.config.eos_token_ids[0], 1);
    expect_status("unsupported config dtype",
                  lis_loader_parse_llama3_config_json(bad_dtype_json,
                                                      strlen(bad_dtype_json),
                                                      &metadata),
                  LIS_STATUS_UNSUPPORTED_DTYPE);
    expect_status("missing config field",
                  lis_loader_parse_llama3_config_json(missing_field_json,
                                                      strlen(missing_field_json),
                                                      &metadata),
                  LIS_STATUS_FORMAT);
    expect_status("malformed eos field",
                  lis_loader_parse_llama3_config_json(bad_eos_json,
                                                      strlen(bad_eos_json),
                                                      &metadata),
                  LIS_STATUS_FORMAT);
    expect_status("out of vocab eos field",
                  lis_loader_parse_llama3_config_json(out_of_vocab_eos_json,
                                                      strlen(out_of_vocab_eos_json),
                                                      &metadata),
                  LIS_STATUS_LIMIT_EXCEEDED);
    expect_status("unsupported rope scaling",
                  lis_loader_parse_llama3_config_json(rope_scaling_json,
                                                      strlen(rope_scaling_json),
                                                      &metadata),
                  LIS_STATUS_UNSUPPORTED);
    expect_status("null rope scaling remains supported",
                  lis_loader_parse_llama3_config_json(rope_scaling_null_json,
                                                      strlen(rope_scaling_null_json),
                                                      &metadata),
                  LIS_STATUS_OK);
    expect_status("unsupported rope type",
                  lis_loader_parse_llama3_config_json(unsupported_rope_type_json,
                                                      strlen(unsupported_rope_type_json),
                                                      &metadata),
                  LIS_STATUS_UNSUPPORTED);
    expect_status("default rope type remains supported",
                  lis_loader_parse_llama3_config_json(default_rope_type_json,
                                                      strlen(default_rope_type_json),
                                                      &metadata),
                  LIS_STATUS_OK);
    expect_status("rope_scaling empty object rejected",
                  lis_loader_parse_llama3_config_json(rope_scaling_empty_object_json,
                                                      strlen(rope_scaling_empty_object_json),
                                                      &metadata),
                  LIS_STATUS_UNSUPPORTED);
    expect_status("rope_scaling linear type rejected",
                  lis_loader_parse_llama3_config_json(rope_scaling_linear_json,
                                                      strlen(rope_scaling_linear_json),
                                                      &metadata),
                  LIS_STATUS_UNSUPPORTED);
    expect_status("meta llama3 8b instruct null rope scaling",
                  lis_loader_parse_llama3_config_json(meta_llama3_8b_instruct_json,
                                                      strlen(meta_llama3_8b_instruct_json),
                                                      &metadata),
                  LIS_STATUS_OK);
    expect_size("meta llama3 8b instruct derived head dim",
                metadata.config.head_dim, 128);
}

static lis_status write_text_file(const char *path, const char *text)
{
    FILE *fp = fopen(path, "wb");

    if (fp == NULL) {
        return LIS_STATUS_IO;
    }
    if (fwrite(text, 1, strlen(text), fp) != strlen(text)) {
        fclose(fp);
        return LIS_STATUS_IO;
    }
    if (fclose(fp) != 0) {
        return LIS_STATUS_IO;
    }
    return LIS_STATUS_OK;
}

static const char *HF_TEST_DIR = "srcs/libs/hf_test";
static const char *HF_CONFIG_PATH = "srcs/libs/hf_test/config.json";
static const char *HF_ST_PATH = "srcs/libs/hf_test/model.safetensors";
static const char *HF_ST_INDEX_PATH =
    "srcs/libs/hf_test/model.safetensors.index.json";

static const char *HF_CONFIG_JSON =
    "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
    "\"hidden_size\":16,\"intermediate_size\":32,"
    "\"num_attention_heads\":2,\"num_key_value_heads\":2,"
    "\"head_dim\":8,\"vocab_size\":32,"
    "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
    "\"max_position_embeddings\":128}";

static const char *HF_CONFIG_TINY_F16_JSON =
    "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
    "\"hidden_size\":1,\"intermediate_size\":1,"
    "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
    "\"head_dim\":1,\"vocab_size\":3,"
    "\"rope_theta\":10000.0,\"torch_dtype\":\"float16\","
    "\"max_position_embeddings\":4}";

static const char *HF_CONFIG_TINY_BF16_JSON =
    "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
    "\"hidden_size\":1,\"intermediate_size\":1,"
    "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
    "\"head_dim\":1,\"vocab_size\":3,"
    "\"rope_theta\":10000.0,\"torch_dtype\":\"bfloat16\","
    "\"max_position_embeddings\":4}";

static const char *HF_CONFIG_QWEN3_TINY_JSON =
    "{\"architectures\":[\"Qwen3ForCausalLM\"],"
    "\"model_type\":\"qwen3\",\"num_hidden_layers\":1,"
    "\"hidden_size\":2,\"intermediate_size\":2,"
    "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
    "\"head_dim\":2,\"vocab_size\":4,"
    "\"rope_theta\":1000000.0,\"rms_norm_eps\":1e-06,"
    "\"torch_dtype\":\"bfloat16\",\"max_position_embeddings\":8,"
    "\"attention_bias\":false,\"use_sliding_window\":false,"
    "\"hidden_act\":\"silu\",\"rope_scaling\":null,"
    "\"tie_word_embeddings\":false,\"eos_token_id\":3}";

static const char *HF_CONFIG_QWEN3_EXPANDED_Q_JSON =
    "{\"architectures\":[\"Qwen3ForCausalLM\"],"
    "\"model_type\":\"qwen3\",\"num_hidden_layers\":1,"
    "\"hidden_size\":2,\"intermediate_size\":2,"
    "\"num_attention_heads\":2,\"num_key_value_heads\":1,"
    "\"head_dim\":2,\"vocab_size\":4,"
    "\"rope_theta\":1000000.0,\"rms_norm_eps\":1e-06,"
    "\"torch_dtype\":\"bfloat16\",\"max_position_embeddings\":8,"
    "\"attention_bias\":false,\"use_sliding_window\":false,"
    "\"hidden_act\":\"silu\",\"rope_scaling\":null,"
    "\"tie_word_embeddings\":false,\"eos_token_id\":3}";

static const char *HF_CONFIG_QWEN3_NON_DIVISIBLE_GQA_JSON =
    "{\"architectures\":[\"Qwen3ForCausalLM\"],"
    "\"model_type\":\"qwen3\",\"num_hidden_layers\":1,"
    "\"hidden_size\":2,\"intermediate_size\":2,"
    "\"num_attention_heads\":3,\"num_key_value_heads\":2,"
    "\"head_dim\":2,\"vocab_size\":4,"
    "\"rope_theta\":1000000.0,\"rms_norm_eps\":1e-06,"
    "\"torch_dtype\":\"bfloat16\",\"max_position_embeddings\":8,"
    "\"attention_bias\":false,\"use_sliding_window\":false,"
    "\"hidden_act\":\"silu\",\"rope_scaling\":null,"
    "\"tie_word_embeddings\":false,\"eos_token_id\":3}";

static const char *HF_CONFIG_LLAMA_EXPANDED_Q_JSON =
    "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
    "\"hidden_size\":2,\"intermediate_size\":2,"
    "\"num_attention_heads\":2,\"num_key_value_heads\":1,"
    "\"head_dim\":2,\"vocab_size\":4,"
    "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
    "\"max_position_embeddings\":8}";

static const char *HF_CONFIG_QWEN3_SLIDING_JSON =
    "{\"architectures\":[\"Qwen3ForCausalLM\"],"
    "\"model_type\":\"qwen3\",\"num_hidden_layers\":1,"
    "\"hidden_size\":2,\"intermediate_size\":2,"
    "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
    "\"head_dim\":2,\"vocab_size\":4,"
    "\"rope_theta\":1000000.0,\"rms_norm_eps\":1e-06,"
    "\"torch_dtype\":\"bfloat16\",\"max_position_embeddings\":8,"
    "\"attention_bias\":false,\"use_sliding_window\":true,"
    "\"hidden_act\":\"silu\",\"rope_scaling\":null}";

static const char *HF_CONFIG_QWEN3_ROPE_SCALING_JSON =
    "{\"architectures\":[\"Qwen3ForCausalLM\"],"
    "\"model_type\":\"qwen3\",\"num_hidden_layers\":1,"
    "\"hidden_size\":2,\"intermediate_size\":2,"
    "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
    "\"head_dim\":2,\"vocab_size\":4,"
    "\"rope_theta\":1000000.0,\"rms_norm_eps\":1e-06,"
    "\"torch_dtype\":\"bfloat16\",\"max_position_embeddings\":8,"
    "\"attention_bias\":false,\"use_sliding_window\":false,"
    "\"hidden_act\":\"silu\","
    "\"rope_scaling\":{\"type\":\"yarn\",\"factor\":2.0}}";

static const char *HF_CONFIG_QWEN3_BIAS_JSON =
    "{\"architectures\":[\"Qwen3ForCausalLM\"],"
    "\"model_type\":\"qwen3\",\"num_hidden_layers\":1,"
    "\"hidden_size\":2,\"intermediate_size\":2,"
    "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
    "\"head_dim\":2,\"vocab_size\":4,"
    "\"rope_theta\":1000000.0,\"rms_norm_eps\":1e-06,"
    "\"torch_dtype\":\"bfloat16\",\"max_position_embeddings\":8,"
    "\"attention_bias\":true,\"use_sliding_window\":false,"
    "\"hidden_act\":\"silu\",\"rope_scaling\":null}";

static const char *HF_CONFIG_QWEN3_F16_JSON =
    "{\"architectures\":[\"Qwen3ForCausalLM\"],"
    "\"model_type\":\"qwen3\",\"num_hidden_layers\":1,"
    "\"hidden_size\":2,\"intermediate_size\":2,"
    "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
    "\"head_dim\":2,\"vocab_size\":4,"
    "\"rope_theta\":1000000.0,\"rms_norm_eps\":1e-06,"
    "\"torch_dtype\":\"float16\",\"max_position_embeddings\":8,"
    "\"attention_bias\":false,\"use_sliding_window\":false,"
    "\"hidden_act\":\"silu\",\"rope_scaling\":null}";

static const char *HF_CONFIG_QWEN3_ROPE_SCALING_EMPTY_JSON =
    "{\"architectures\":[\"Qwen3ForCausalLM\"],"
    "\"model_type\":\"qwen3\",\"num_hidden_layers\":1,"
    "\"hidden_size\":2,\"intermediate_size\":2,"
    "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
    "\"head_dim\":2,\"vocab_size\":4,"
    "\"rope_theta\":1000000.0,\"rms_norm_eps\":1e-06,"
    "\"torch_dtype\":\"bfloat16\",\"max_position_embeddings\":8,"
    "\"attention_bias\":false,\"use_sliding_window\":false,"
    "\"hidden_act\":\"silu\",\"rope_scaling\":{}}";

static const char *HF_CONFIG_QWEN3_ROPE_TYPE_LLAMA3_JSON =
    "{\"architectures\":[\"Qwen3ForCausalLM\"],"
    "\"model_type\":\"qwen3\",\"num_hidden_layers\":1,"
    "\"hidden_size\":2,\"intermediate_size\":2,"
    "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
    "\"head_dim\":2,\"vocab_size\":4,"
    "\"rope_theta\":1000000.0,\"rms_norm_eps\":1e-06,"
    "\"torch_dtype\":\"bfloat16\",\"max_position_embeddings\":8,"
    "\"attention_bias\":false,\"use_sliding_window\":false,"
    "\"hidden_act\":\"silu\",\"rope_type\":\"llama3\"}";

static const char *HF_CONFIG_QWEN2_JSON =
    "{\"architectures\":[\"Qwen2ForCausalLM\"],"
    "\"model_type\":\"qwen2\",\"num_hidden_layers\":1,"
    "\"hidden_size\":2,\"intermediate_size\":2,"
    "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
    "\"head_dim\":2,\"vocab_size\":4,"
    "\"rope_theta\":1000000.0,\"rms_norm_eps\":1e-06,"
    "\"torch_dtype\":\"bfloat16\",\"max_position_embeddings\":8}";

static const char *HF_CONFIG_QWEN25_JSON =
    "{\"architectures\":[\"Qwen2ForCausalLM\"],"
    "\"model_type\":\"qwen2_5\",\"num_hidden_layers\":1,"
    "\"hidden_size\":2,\"intermediate_size\":2,"
    "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
    "\"head_dim\":2,\"vocab_size\":4,"
    "\"rope_theta\":1000000.0,\"rms_norm_eps\":1e-06,"
    "\"torch_dtype\":\"bfloat16\",\"max_position_embeddings\":8}";

static const char *HF_CONFIG_QWEN3_MOE_JSON =
    "{\"architectures\":[\"Qwen3MoeForCausalLM\"],"
    "\"model_type\":\"qwen3\",\"num_hidden_layers\":1,"
    "\"hidden_size\":2,\"intermediate_size\":2,"
    "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
    "\"head_dim\":2,\"vocab_size\":4,"
    "\"rope_theta\":1000000.0,\"rms_norm_eps\":1e-06,"
    "\"torch_dtype\":\"bfloat16\",\"max_position_embeddings\":8,"
    "\"attention_bias\":false,\"use_sliding_window\":false,"
    "\"hidden_act\":\"silu\",\"rope_scaling\":null,"
    "\"num_experts\":8}";

static const char *HF_CONFIG_QWEN3_VL_JSON =
    "{\"architectures\":[\"Qwen3VLForConditionalGeneration\"],"
    "\"model_type\":\"qwen3\",\"num_hidden_layers\":1,"
    "\"hidden_size\":2,\"intermediate_size\":2,"
    "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
    "\"head_dim\":2,\"vocab_size\":4,"
    "\"rope_theta\":1000000.0,\"rms_norm_eps\":1e-06,"
    "\"torch_dtype\":\"bfloat16\",\"max_position_embeddings\":8,"
    "\"attention_bias\":false,\"use_sliding_window\":false,"
    "\"hidden_act\":\"silu\",\"rope_scaling\":null}";

static const char *HF_CONFIG_NON_LLAMA_JSON =
    "{\"model_type\":\"gpt2\",\"num_hidden_layers\":1,"
    "\"hidden_size\":16,\"intermediate_size\":32,"
    "\"num_attention_heads\":2,\"num_key_value_heads\":2,"
    "\"head_dim\":8,\"vocab_size\":32,"
    "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
    "\"max_position_embeddings\":128}";

static const char *HF_CONFIG_TIED_JSON =
    "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
    "\"hidden_size\":16,\"intermediate_size\":32,"
    "\"num_attention_heads\":2,\"num_key_value_heads\":2,"
    "\"head_dim\":8,\"vocab_size\":32,"
    "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
    "\"tie_word_embeddings\":true,"
    "\"max_position_embeddings\":128}";

static const char *HF_CONFIG_MISSING_FIELD_JSON =
    "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
    "\"hidden_size\":16,\"intermediate_size\":32,"
    "\"num_attention_heads\":2,\"num_key_value_heads\":2,"
    "\"head_dim\":8,"
    "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
    "\"max_position_embeddings\":128}";

static const char *HF_CONFIG_ROPE_SCALING_JSON =
    "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
    "\"hidden_size\":16,\"intermediate_size\":32,"
    "\"num_attention_heads\":2,\"num_key_value_heads\":2,"
    "\"head_dim\":8,\"vocab_size\":32,"
    "\"rope_theta\":10000.0,"
    "\"rope_scaling\":{\"rope_type\":\"llama3\",\"factor\":32.0},"
    "\"torch_dtype\":\"float32\","
    "\"max_position_embeddings\":128}";

static const char *HF_CONFIG_ROPE_SCALING_NULL_JSON =
    "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
    "\"hidden_size\":16,\"intermediate_size\":32,"
    "\"num_attention_heads\":2,\"num_key_value_heads\":2,"
    "\"head_dim\":8,\"vocab_size\":32,"
    "\"rope_theta\":10000.0,\"rope_scaling\":null,"
    "\"torch_dtype\":\"float32\","
    "\"max_position_embeddings\":128}";

static const char *HF_CONFIG_ROPE_TYPE_JSON =
    "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
    "\"hidden_size\":16,\"intermediate_size\":32,"
    "\"num_attention_heads\":2,\"num_key_value_heads\":2,"
    "\"head_dim\":8,\"vocab_size\":32,"
    "\"rope_theta\":10000.0,\"rope_type\":\"llama3\","
    "\"torch_dtype\":\"float32\","
    "\"max_position_embeddings\":128}";

static const char *HF_HEADER_VALID =
    "{\"model.embed_tokens.weight\":{\"dtype\":\"F32\",\"shape\":[32,16],"
    "\"data_offsets\":[0,2048]},"
    "\"model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16,16],\"data_offsets\":[2048,3072]},"
    "\"model.layers.0.self_attn.k_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16,16],\"data_offsets\":[3072,4096]},"
    "\"model.layers.0.self_attn.v_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16,16],\"data_offsets\":[4096,5120]},"
    "\"model.layers.0.self_attn.o_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16,16],\"data_offsets\":[5120,6144]},"
    "\"model.layers.0.mlp.gate_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[32,16],\"data_offsets\":[6144,8192]},"
    "\"model.layers.0.mlp.up_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[32,16],\"data_offsets\":[8192,10240]},"
    "\"model.layers.0.mlp.down_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16,32],\"data_offsets\":[10240,12288]},"
    "\"model.layers.0.input_layernorm.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16],\"data_offsets\":[12288,12352]},"
    "\"model.layers.0.post_attention_layernorm.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16],\"data_offsets\":[12352,12416]},"
    "\"model.norm.weight\":{\"dtype\":\"F32\",\"shape\":[16],"
    "\"data_offsets\":[12416,12480]},"
    "\"lm_head.weight\":{\"dtype\":\"F32\",\"shape\":[32,16],"
    "\"data_offsets\":[12480,14528]}}";

static const char *HF_HEADER_TINY_F16 =
    "{\"model.embed_tokens.weight\":{\"dtype\":\"F16\",\"shape\":[3,1],"
    "\"data_offsets\":[0,6]},"
    "\"model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"F16\","
    "\"shape\":[1,1],\"data_offsets\":[6,8]},"
    "\"model.layers.0.self_attn.k_proj.weight\":{\"dtype\":\"F16\","
    "\"shape\":[1,1],\"data_offsets\":[8,10]},"
    "\"model.layers.0.self_attn.v_proj.weight\":{\"dtype\":\"F16\","
    "\"shape\":[1,1],\"data_offsets\":[10,12]},"
    "\"model.layers.0.self_attn.o_proj.weight\":{\"dtype\":\"F16\","
    "\"shape\":[1,1],\"data_offsets\":[12,14]},"
    "\"model.layers.0.mlp.gate_proj.weight\":{\"dtype\":\"F16\","
    "\"shape\":[1,1],\"data_offsets\":[14,16]},"
    "\"model.layers.0.mlp.up_proj.weight\":{\"dtype\":\"F16\","
    "\"shape\":[1,1],\"data_offsets\":[16,18]},"
    "\"model.layers.0.mlp.down_proj.weight\":{\"dtype\":\"F16\","
    "\"shape\":[1,1],\"data_offsets\":[18,20]},"
    "\"model.layers.0.input_layernorm.weight\":{\"dtype\":\"F16\","
    "\"shape\":[1],\"data_offsets\":[20,22]},"
    "\"model.layers.0.post_attention_layernorm.weight\":{\"dtype\":\"F16\","
    "\"shape\":[1],\"data_offsets\":[22,24]},"
    "\"model.norm.weight\":{\"dtype\":\"F16\",\"shape\":[1],"
    "\"data_offsets\":[24,26]},"
    "\"lm_head.weight\":{\"dtype\":\"F16\",\"shape\":[3,1],"
    "\"data_offsets\":[26,32]}}";

static const char *HF_HEADER_TINY_BF16 =
    "{\"model.embed_tokens.weight\":{\"dtype\":\"BF16\",\"shape\":[3,1],"
    "\"data_offsets\":[0,6]},"
    "\"model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[1,1],\"data_offsets\":[6,8]},"
    "\"model.layers.0.self_attn.k_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[1,1],\"data_offsets\":[8,10]},"
    "\"model.layers.0.self_attn.v_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[1,1],\"data_offsets\":[10,12]},"
    "\"model.layers.0.self_attn.o_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[1,1],\"data_offsets\":[12,14]},"
    "\"model.layers.0.mlp.gate_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[1,1],\"data_offsets\":[14,16]},"
    "\"model.layers.0.mlp.up_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[1,1],\"data_offsets\":[16,18]},"
    "\"model.layers.0.mlp.down_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[1,1],\"data_offsets\":[18,20]},"
    "\"model.layers.0.input_layernorm.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[1],\"data_offsets\":[20,22]},"
    "\"model.layers.0.post_attention_layernorm.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[1],\"data_offsets\":[22,24]},"
    "\"model.norm.weight\":{\"dtype\":\"BF16\",\"shape\":[1],"
    "\"data_offsets\":[24,26]},"
    "\"lm_head.weight\":{\"dtype\":\"BF16\",\"shape\":[3,1],"
    "\"data_offsets\":[26,32]}}";

static const char *HF_HEADER_TINY_MIXED_BF16_F32 =
    "{\"model.embed_tokens.weight\":{\"dtype\":\"BF16\",\"shape\":[3,1],"
    "\"data_offsets\":[0,6]},"
    "\"model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[1,1],\"data_offsets\":[6,10]},"
    "\"model.layers.0.self_attn.k_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[1,1],\"data_offsets\":[10,12]},"
    "\"model.layers.0.self_attn.v_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[1,1],\"data_offsets\":[12,14]},"
    "\"model.layers.0.self_attn.o_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[1,1],\"data_offsets\":[14,16]},"
    "\"model.layers.0.mlp.gate_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[1,1],\"data_offsets\":[16,18]},"
    "\"model.layers.0.mlp.up_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[1,1],\"data_offsets\":[18,20]},"
    "\"model.layers.0.mlp.down_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[1,1],\"data_offsets\":[20,22]},"
    "\"model.layers.0.input_layernorm.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[1],\"data_offsets\":[22,24]},"
    "\"model.layers.0.post_attention_layernorm.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[1],\"data_offsets\":[24,26]},"
    "\"model.norm.weight\":{\"dtype\":\"BF16\",\"shape\":[1],"
    "\"data_offsets\":[26,28]},"
    "\"lm_head.weight\":{\"dtype\":\"BF16\",\"shape\":[3,1],"
    "\"data_offsets\":[28,34]}}";

static const char *HF_HEADER_QWEN3_TINY_BF16 =
    "{\"model.embed_tokens.weight\":{\"dtype\":\"BF16\",\"shape\":[4,2],"
    "\"data_offsets\":[0,16]},"
    "\"model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[16,24]},"
    "\"model.layers.0.self_attn.k_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[24,32]},"
    "\"model.layers.0.self_attn.v_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[32,40]},"
    "\"model.layers.0.self_attn.o_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[40,48]},"
    "\"model.layers.0.self_attn.q_norm.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2],\"data_offsets\":[48,52]},"
    "\"model.layers.0.self_attn.k_norm.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2],\"data_offsets\":[52,56]},"
    "\"model.layers.0.mlp.gate_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[56,64]},"
    "\"model.layers.0.mlp.up_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[64,72]},"
    "\"model.layers.0.mlp.down_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[72,80]},"
    "\"model.layers.0.input_layernorm.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2],\"data_offsets\":[80,84]},"
    "\"model.layers.0.post_attention_layernorm.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2],\"data_offsets\":[84,88]},"
    "\"model.norm.weight\":{\"dtype\":\"BF16\",\"shape\":[2],"
    "\"data_offsets\":[88,92]},"
    "\"lm_head.weight\":{\"dtype\":\"BF16\",\"shape\":[4,2],"
    "\"data_offsets\":[92,108]}}";

static const char *HF_HEADER_QWEN3_EXPANDED_Q_BF16 =
    "{\"model.embed_tokens.weight\":{\"dtype\":\"BF16\",\"shape\":[4,2],"
    "\"data_offsets\":[0,16]},"
    "\"model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[4,2],\"data_offsets\":[16,32]},"
    "\"model.layers.0.self_attn.k_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[32,40]},"
    "\"model.layers.0.self_attn.v_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[40,48]},"
    "\"model.layers.0.self_attn.o_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,4],\"data_offsets\":[48,64]},"
    "\"model.layers.0.self_attn.q_norm.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2],\"data_offsets\":[64,68]},"
    "\"model.layers.0.self_attn.k_norm.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2],\"data_offsets\":[68,72]},"
    "\"model.layers.0.mlp.gate_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[72,80]},"
    "\"model.layers.0.mlp.up_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[80,88]},"
    "\"model.layers.0.mlp.down_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[88,96]},"
    "\"model.layers.0.input_layernorm.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2],\"data_offsets\":[96,100]},"
    "\"model.layers.0.post_attention_layernorm.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2],\"data_offsets\":[100,104]},"
    "\"model.norm.weight\":{\"dtype\":\"BF16\",\"shape\":[2],"
    "\"data_offsets\":[104,108]},"
    "\"lm_head.weight\":{\"dtype\":\"BF16\",\"shape\":[4,2],"
    "\"data_offsets\":[108,124]}}";

static const char *HF_HEADER_QWEN3_EXPANDED_Q_BAD_O_BF16 =
    "{\"model.embed_tokens.weight\":{\"dtype\":\"BF16\",\"shape\":[4,2],"
    "\"data_offsets\":[0,16]},"
    "\"model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[4,2],\"data_offsets\":[16,32]},"
    "\"model.layers.0.self_attn.k_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[32,40]},"
    "\"model.layers.0.self_attn.v_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[40,48]},"
    "\"model.layers.0.self_attn.o_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[48,56]},"
    "\"model.layers.0.self_attn.q_norm.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2],\"data_offsets\":[56,60]},"
    "\"model.layers.0.self_attn.k_norm.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2],\"data_offsets\":[60,64]},"
    "\"model.layers.0.mlp.gate_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[64,72]},"
    "\"model.layers.0.mlp.up_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[72,80]},"
    "\"model.layers.0.mlp.down_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[80,88]},"
    "\"model.layers.0.input_layernorm.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2],\"data_offsets\":[88,92]},"
    "\"model.layers.0.post_attention_layernorm.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2],\"data_offsets\":[92,96]},"
    "\"model.norm.weight\":{\"dtype\":\"BF16\",\"shape\":[2],"
    "\"data_offsets\":[96,100]},"
    "\"lm_head.weight\":{\"dtype\":\"BF16\",\"shape\":[4,2],"
    "\"data_offsets\":[100,116]}}";

static const char *HF_HEADER_QWEN3_MISSING_Q_NORM =
    "{\"model.embed_tokens.weight\":{\"dtype\":\"BF16\",\"shape\":[4,2],"
    "\"data_offsets\":[0,16]},"
    "\"model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[16,24]},"
    "\"model.layers.0.self_attn.k_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[24,32]},"
    "\"model.layers.0.self_attn.v_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[32,40]},"
    "\"model.layers.0.self_attn.o_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[40,48]},"
    "\"model.layers.0.self_attn.k_norm.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2],\"data_offsets\":[52,56]},"
    "\"model.layers.0.mlp.gate_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[56,64]},"
    "\"model.layers.0.mlp.up_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[64,72]},"
    "\"model.layers.0.mlp.down_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[72,80]},"
    "\"model.layers.0.input_layernorm.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2],\"data_offsets\":[80,84]},"
    "\"model.layers.0.post_attention_layernorm.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2],\"data_offsets\":[84,88]},"
    "\"model.norm.weight\":{\"dtype\":\"BF16\",\"shape\":[2],"
    "\"data_offsets\":[88,92]},"
    "\"lm_head.weight\":{\"dtype\":\"BF16\",\"shape\":[4,2],"
    "\"data_offsets\":[92,108]}}";

static const char *HF_HEADER_QWEN3_MIXED_DTYPE =
    "{\"model.embed_tokens.weight\":{\"dtype\":\"BF16\",\"shape\":[4,2],"
    "\"data_offsets\":[0,16]},"
    "\"model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[2,2],\"data_offsets\":[16,32]},"
    "\"model.layers.0.self_attn.k_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[32,40]},"
    "\"model.layers.0.self_attn.v_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[40,48]},"
    "\"model.layers.0.self_attn.o_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[48,56]},"
    "\"model.layers.0.self_attn.q_norm.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2],\"data_offsets\":[56,60]},"
    "\"model.layers.0.self_attn.k_norm.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2],\"data_offsets\":[60,64]},"
    "\"model.layers.0.mlp.gate_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[64,72]},"
    "\"model.layers.0.mlp.up_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[72,80]},"
    "\"model.layers.0.mlp.down_proj.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2,2],\"data_offsets\":[80,88]},"
    "\"model.layers.0.input_layernorm.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2],\"data_offsets\":[88,92]},"
    "\"model.layers.0.post_attention_layernorm.weight\":{\"dtype\":\"BF16\","
    "\"shape\":[2],\"data_offsets\":[92,96]},"
    "\"model.norm.weight\":{\"dtype\":\"BF16\",\"shape\":[2],"
    "\"data_offsets\":[96,100]},"
    "\"lm_head.weight\":{\"dtype\":\"BF16\",\"shape\":[4,2],"
    "\"data_offsets\":[100,116]}}";

static const char *HF_HEADER_MISSING_LM_HEAD =
    "{\"model.embed_tokens.weight\":{\"dtype\":\"F32\",\"shape\":[32,16],"
    "\"data_offsets\":[0,2048]},"
    "\"model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16,16],\"data_offsets\":[2048,3072]},"
    "\"model.layers.0.self_attn.k_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16,16],\"data_offsets\":[3072,4096]},"
    "\"model.layers.0.self_attn.v_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16,16],\"data_offsets\":[4096,5120]},"
    "\"model.layers.0.self_attn.o_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16,16],\"data_offsets\":[5120,6144]},"
    "\"model.layers.0.mlp.gate_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[32,16],\"data_offsets\":[6144,8192]},"
    "\"model.layers.0.mlp.up_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[32,16],\"data_offsets\":[8192,10240]},"
    "\"model.layers.0.mlp.down_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16,32],\"data_offsets\":[10240,12288]},"
    "\"model.layers.0.input_layernorm.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16],\"data_offsets\":[12288,12352]},"
    "\"model.layers.0.post_attention_layernorm.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16],\"data_offsets\":[12352,12416]},"
    "\"model.norm.weight\":{\"dtype\":\"F32\",\"shape\":[16],"
    "\"data_offsets\":[12416,12480]}}";

static const char *HF_HEADER_WRONG_EMBED_SHAPE =
    "{\"model.embed_tokens.weight\":{\"dtype\":\"F32\",\"shape\":[16,32],"
    "\"data_offsets\":[0,2048]},"
    "\"model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16,16],\"data_offsets\":[2048,3072]},"
    "\"model.layers.0.self_attn.k_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16,16],\"data_offsets\":[3072,4096]},"
    "\"model.layers.0.self_attn.v_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16,16],\"data_offsets\":[4096,5120]},"
    "\"model.layers.0.self_attn.o_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16,16],\"data_offsets\":[5120,6144]},"
    "\"model.layers.0.mlp.gate_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[32,16],\"data_offsets\":[6144,8192]},"
    "\"model.layers.0.mlp.up_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[32,16],\"data_offsets\":[8192,10240]},"
    "\"model.layers.0.mlp.down_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16,32],\"data_offsets\":[10240,12288]},"
    "\"model.layers.0.input_layernorm.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16],\"data_offsets\":[12288,12352]},"
    "\"model.layers.0.post_attention_layernorm.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16],\"data_offsets\":[12352,12416]},"
    "\"model.norm.weight\":{\"dtype\":\"F32\",\"shape\":[16],"
    "\"data_offsets\":[12416,12480]},"
    "\"lm_head.weight\":{\"dtype\":\"F32\",\"shape\":[32,16],"
    "\"data_offsets\":[12480,14528]}}";

static const char *HF_HEADER_UNSUPPORTED_DTYPE =
    "{\"model.embed_tokens.weight\":{\"dtype\":\"I32\",\"shape\":[32,16],"
    "\"data_offsets\":[0,2048]}}";

static const char *HF_HEADER_WRONG_LAYER =
    "{\"model.embed_tokens.weight\":{\"dtype\":\"F32\",\"shape\":[32,16],"
    "\"data_offsets\":[0,2048]},"
    "\"model.layers.1.self_attn.q_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16,16],\"data_offsets\":[2048,3072]}}";

static const char *HF_HEADER_FUSED_QKV =
    "{\"model.embed_tokens.weight\":{\"dtype\":\"F32\",\"shape\":[32,16],"
    "\"data_offsets\":[0,2048]},"
    "\"model.layers.0.self_attn.qkv_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[48,16],\"data_offsets\":[2048,5120]}}";

static const char *HF_HEADER_INTERNAL_EXTRA =
    "{\"model.embed_tokens.weight\":{\"dtype\":\"F32\",\"shape\":[32,16],"
    "\"data_offsets\":[0,2048]},"
    "\"lis.validation_logits\":{\"dtype\":\"F32\",\"shape\":[1,32],"
    "\"data_offsets\":[2048,2176]}}";

static const char *HF_HEADER_DUPLICATE_LM_HEAD =
    "{\"model.embed_tokens.weight\":{\"dtype\":\"F32\",\"shape\":[32,16],"
    "\"data_offsets\":[0,2048]},"
    "\"model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16,16],\"data_offsets\":[2048,3072]},"
    "\"model.layers.0.self_attn.k_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16,16],\"data_offsets\":[3072,4096]},"
    "\"model.layers.0.self_attn.v_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16,16],\"data_offsets\":[4096,5120]},"
    "\"model.layers.0.self_attn.o_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16,16],\"data_offsets\":[5120,6144]},"
    "\"model.layers.0.mlp.gate_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[32,16],\"data_offsets\":[6144,8192]},"
    "\"model.layers.0.mlp.up_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[32,16],\"data_offsets\":[8192,10240]},"
    "\"model.layers.0.mlp.down_proj.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16,32],\"data_offsets\":[10240,12288]},"
    "\"model.layers.0.input_layernorm.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16],\"data_offsets\":[12288,12352]},"
    "\"model.layers.0.post_attention_layernorm.weight\":{\"dtype\":\"F32\","
    "\"shape\":[16],\"data_offsets\":[12352,12416]},"
    "\"model.norm.weight\":{\"dtype\":\"F32\",\"shape\":[16],"
    "\"data_offsets\":[12416,12480]},"
    "\"lm_head.weight\":{\"dtype\":\"F32\",\"shape\":[32,16],"
    "\"data_offsets\":[12480,14528]},"
    "\"lm_head.weight\":{\"dtype\":\"F32\",\"shape\":[32,16],"
    "\"data_offsets\":[14528,16576]}}";

static void test_qwen3_config_parser(void)
{
    lis_model_metadata metadata = { 0 };

    expect_status("parse qwen3 config",
                  lis_loader_parse_hf_config_json(HF_CONFIG_QWEN3_TINY_JSON,
                                                  strlen(HF_CONFIG_QWEN3_TINY_JSON),
                                                  &metadata),
                  LIS_STATUS_OK);
    expect_size("qwen3 family", (size_t)metadata.config.family,
                (size_t)LIS_MODEL_FAMILY_QWEN3_DENSE_DECODER);
    expect_size("qwen3 dtype", (size_t)metadata.config.weight_dtype,
                (size_t)LIS_DTYPE_BF16);
    expect_size("qwen3 eos count", metadata.config.eos_token_count, 1);
    expect_size("qwen3 eos value", metadata.config.eos_token_ids[0], 3);
    expect_status("parse qwen3 expanded q geometry",
                  lis_loader_parse_hf_config_json(
                      HF_CONFIG_QWEN3_EXPANDED_Q_JSON,
                      strlen(HF_CONFIG_QWEN3_EXPANDED_Q_JSON),
                      &metadata),
                  LIS_STATUS_OK);
    expect_size("qwen3 expanded hidden", metadata.config.hidden_size, 2);
    expect_size("qwen3 expanded heads",
                metadata.config.attention_head_count, 2);
    expect_size("qwen3 expanded head dim", metadata.config.head_dim, 2);
    expect_status("qwen3 non-divisible gqa rejected",
                  lis_loader_parse_hf_config_json(
                      HF_CONFIG_QWEN3_NON_DIVISIBLE_GQA_JSON,
                      strlen(HF_CONFIG_QWEN3_NON_DIVISIBLE_GQA_JSON),
                      &metadata),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_status("llama expanded q geometry rejected",
                  lis_loader_parse_hf_config_json(
                      HF_CONFIG_LLAMA_EXPANDED_Q_JSON,
                      strlen(HF_CONFIG_LLAMA_EXPANDED_Q_JSON),
                      &metadata),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_status("qwen2 rejected",
                  lis_loader_parse_hf_config_json(HF_CONFIG_QWEN2_JSON,
                                                  strlen(HF_CONFIG_QWEN2_JSON),
                                                  &metadata),
                  LIS_STATUS_UNSUPPORTED_FORMAT);
    expect_status("qwen2.5 rejected",
                  lis_loader_parse_hf_config_json(HF_CONFIG_QWEN25_JSON,
                                                  strlen(HF_CONFIG_QWEN25_JSON),
                                                  &metadata),
                  LIS_STATUS_UNSUPPORTED_FORMAT);
    expect_status("qwen3 moe rejected",
                  lis_loader_parse_hf_config_json(HF_CONFIG_QWEN3_MOE_JSON,
                                                  strlen(HF_CONFIG_QWEN3_MOE_JSON),
                                                  &metadata),
                  LIS_STATUS_UNSUPPORTED_FORMAT);
    expect_status("qwen3 vl rejected",
                  lis_loader_parse_hf_config_json(HF_CONFIG_QWEN3_VL_JSON,
                                                  strlen(HF_CONFIG_QWEN3_VL_JSON),
                                                  &metadata),
                  LIS_STATUS_UNSUPPORTED_FORMAT);
    expect_status("qwen3 rope scaling rejected",
                  lis_loader_parse_hf_config_json(
                      HF_CONFIG_QWEN3_ROPE_SCALING_JSON,
                      strlen(HF_CONFIG_QWEN3_ROPE_SCALING_JSON),
                      &metadata),
                  LIS_STATUS_UNSUPPORTED);
    expect_status("qwen3 sliding window rejected",
                  lis_loader_parse_hf_config_json(
                      HF_CONFIG_QWEN3_SLIDING_JSON,
                      strlen(HF_CONFIG_QWEN3_SLIDING_JSON),
                      &metadata),
                  LIS_STATUS_UNSUPPORTED);
    expect_status("qwen3 attention bias rejected",
                  lis_loader_parse_hf_config_json(HF_CONFIG_QWEN3_BIAS_JSON,
                                                  strlen(HF_CONFIG_QWEN3_BIAS_JSON),
                                                  &metadata),
                  LIS_STATUS_UNSUPPORTED);
    expect_status("qwen3 non-bf16 rejected",
                  lis_loader_parse_hf_config_json(HF_CONFIG_QWEN3_F16_JSON,
                                                  strlen(HF_CONFIG_QWEN3_F16_JSON),
                                                  &metadata),
                  LIS_STATUS_UNSUPPORTED_DTYPE);
    expect_status("qwen3 rope_scaling empty object rejected",
                  lis_loader_parse_hf_config_json(
                      HF_CONFIG_QWEN3_ROPE_SCALING_EMPTY_JSON,
                      strlen(HF_CONFIG_QWEN3_ROPE_SCALING_EMPTY_JSON),
                      &metadata),
                  LIS_STATUS_UNSUPPORTED);
    expect_status("qwen3 rope_type llama3 rejected",
                  lis_loader_parse_hf_config_json(
                      HF_CONFIG_QWEN3_ROPE_TYPE_LLAMA3_JSON,
                      strlen(HF_CONFIG_QWEN3_ROPE_TYPE_LLAMA3_JSON),
                      &metadata),
                  LIS_STATUS_UNSUPPORTED);
}

static void cleanup_hf_fixture(void)
{
    remove(HF_CONFIG_PATH);
    remove(HF_ST_PATH);
    remove(HF_ST_INDEX_PATH);
    if (system("rmdir srcs/libs/hf_test 2>/dev/null") != 0) {
        /* Best effort cleanup; failures usually mean the directory is absent. */
    }
}

static void write_hf_fixture(const char *name, const char *config_json,
                             const char *st_header, size_t st_data_len)
{
    unsigned char *st_data = NULL;

    cleanup_hf_fixture();
    if (system("mkdir -p srcs/libs/hf_test") != 0) {
        fprintf(stderr, "%s: mkdir failed\n", name);
        ++g_failures;
        return;
    }

    if (config_json != NULL) {
        expect_status(name, write_text_file(HF_CONFIG_PATH, config_json),
                      LIS_STATUS_OK);
    }
    if (st_header == NULL) {
        return;
    }

    st_data = calloc(1, st_data_len);
    if (st_data == NULL && st_data_len != 0) {
        fprintf(stderr, "%s: allocation failed\n", name);
        ++g_failures;
        return;
    }
    expect_status(name, write_safetensors_file(HF_ST_PATH, st_header, st_data,
                                               st_data_len),
                  LIS_STATUS_OK);
    free(st_data);
}

static void expect_hf_load_status(const char *name, const char *config_json,
                                  const char *st_header, size_t st_data_len,
                                  lis_status expected)
{
    lis_model_source source = lis_model_source_from_path(HF_TEST_DIR);
    lis_loaded_model model = { 0 };

    write_hf_fixture(name, config_json, st_header, st_data_len);
    expect_status(name, lis_loader_load(&source, &model), expected);
    lis_loaded_model_destroy(&model);
    cleanup_hf_fixture();
}

static void test_hf_local_import(void)
{
    lis_model_source source = lis_model_source_from_path(HF_TEST_DIR);
    lis_model_format format = LIS_MODEL_FORMAT_UNKNOWN;
    lis_loaded_model model = { 0 };

    write_hf_fixture("write hf local fixture", HF_CONFIG_JSON, HF_HEADER_VALID,
                     14528);

    expect_status("probe hf local format",
                  lis_loader_probe_format(&source, &format), LIS_STATUS_OK);
    expect_size("hf format matched", (size_t)format,
                (size_t)LIS_MODEL_FORMAT_HUGGINGFACE_LOCAL);
    
    expect_status("load hf local model", lis_loader_load(&source, &model), LIS_STATUS_OK);
    expect_size("hf tensor count", model.tensor_count, 12);
    expect_size("hf metadata attached", (size_t)model.has_metadata, 1);
    
    if (strcmp(model.tensors[0].name, "lis.token_embeddings.weight") != 0) {
        fprintf(stderr, "tensor name mapped incorrectly, got: %s\n", model.tensors[0].name);
        ++g_failures;
    }
    expect_size("hf metadata layers", model.metadata.config.layer_count, 1);
    expect_size("hf metadata hidden", model.metadata.config.hidden_size, 16);
    lis_loaded_model_destroy(&model);
    
    /* Merged model.safetensors is the concrete policy; an auxiliary index is
       ignored when the merged file is present, while index-only is rejected. */
    expect_status("write hf safetensors index",
                  write_text_file(HF_ST_INDEX_PATH, "{}"), LIS_STATUS_OK);
    expect_status("load hf merged with auxiliary index",
                  lis_loader_load(&source, &model), LIS_STATUS_OK);
    lis_loaded_model_destroy(&model);
    remove(HF_ST_PATH);
    expect_status("load hf index-only reject", lis_loader_load(&source, &model),
                  LIS_STATUS_UNSUPPORTED_FORMAT);
    cleanup_hf_fixture();
}

static void test_hf_local_import_rejections(void)
{
    lis_model_source source = { 0 };
    lis_loaded_model model = { 0 };

    expect_hf_load_status("hf missing config", NULL, HF_HEADER_VALID, 14528,
                          LIS_STATUS_UNSUPPORTED_FORMAT);
    expect_hf_load_status("hf missing safetensors", HF_CONFIG_JSON, NULL, 0,
                          LIS_STATUS_IO);
    expect_hf_load_status("hf non-llama config", HF_CONFIG_NON_LLAMA_JSON,
                          HF_HEADER_VALID, 14528,
                          LIS_STATUS_UNSUPPORTED_FORMAT);
    expect_hf_load_status("hf malformed config", "{", HF_HEADER_VALID, 14528,
                          LIS_STATUS_FORMAT);
    expect_hf_load_status("hf missing config field",
                          HF_CONFIG_MISSING_FIELD_JSON, HF_HEADER_VALID,
                          14528, LIS_STATUS_FORMAT);
    expect_hf_load_status("hf unsupported rope scaling",
                          HF_CONFIG_ROPE_SCALING_JSON, HF_HEADER_VALID,
                          14528, LIS_STATUS_UNSUPPORTED);
    expect_hf_load_status("hf null rope scaling",
                          HF_CONFIG_ROPE_SCALING_NULL_JSON, HF_HEADER_VALID,
                          14528, LIS_STATUS_OK);
    expect_hf_load_status("hf unsupported rope type",
                          HF_CONFIG_ROPE_TYPE_JSON, HF_HEADER_VALID,
                          14528, LIS_STATUS_UNSUPPORTED);
    expect_hf_load_status("hf missing tensor", HF_CONFIG_JSON,
                          HF_HEADER_MISSING_LM_HEAD, 12480,
                          LIS_STATUS_FORMAT);
    expect_hf_load_status("hf tied embeddings without lm head",
                          HF_CONFIG_TIED_JSON, HF_HEADER_MISSING_LM_HEAD,
                          12480, LIS_STATUS_OK);
    expect_hf_load_status("hf transposed tensor shape", HF_CONFIG_JSON,
                          HF_HEADER_WRONG_EMBED_SHAPE, 14528,
                          LIS_STATUS_SHAPE_MISMATCH);
    expect_hf_load_status("hf unsupported tensor dtype", HF_CONFIG_JSON,
                          HF_HEADER_UNSUPPORTED_DTYPE, 2048,
                          LIS_STATUS_UNSUPPORTED_DTYPE);
    expect_hf_load_status("hf wrong layer index", HF_CONFIG_JSON,
                          HF_HEADER_WRONG_LAYER, 3072, LIS_STATUS_FORMAT);
    expect_hf_load_status("hf fused qkv tensor", HF_CONFIG_JSON,
                          HF_HEADER_FUSED_QKV, 5120, LIS_STATUS_FORMAT);
    expect_hf_load_status("hf internal tensor rejected", HF_CONFIG_JSON,
                          HF_HEADER_INTERNAL_EXTRA, 2176, LIS_STATUS_FORMAT);
    expect_hf_load_status("hf duplicate tensor rejected", HF_CONFIG_JSON,
                          HF_HEADER_DUPLICATE_LM_HEAD, 16576,
                          LIS_STATUS_FORMAT);

    source = lis_model_source_from_path("https://huggingface.co/example/model");
    expect_status("hf remote url rejected", lis_loader_load(&source, &model),
                  LIS_STATUS_UNSUPPORTED_FORMAT);
    lis_loaded_model_destroy(&model);

    source = lis_model_source_from_path("model.gguf");
    expect_status("hf gguf rejected", lis_loader_load(&source, &model),
                  LIS_STATUS_UNSUPPORTED_FORMAT);
    lis_loaded_model_destroy(&model);
}

static void test_hf_tied_output_policy(void)
{
    lis_model_source source = lis_model_source_from_path(HF_TEST_DIR);
    lis_loaded_model model = { 0 };
    lis_model_metadata metadata = { 0 };
    int saw_lm_head = 0;
    size_t index;

    write_hf_fixture("write tied explicit lm_head fixture",
                     HF_CONFIG_TIED_JSON, HF_HEADER_VALID, 14528);
    expect_status("hf tied explicit lm_head accepted",
                  lis_loader_load(&source, &model), LIS_STATUS_OK);
    expect_size("hf tied explicit uses lm_head",
                (size_t)model.metadata.config.tie_word_embeddings, 0);
    for (index = 0; index < model.tensor_count; ++index) {
        if (strcmp(model.tensors[index].name, "lis.lm_head.weight") == 0) {
            saw_lm_head = 1;
        }
    }
    expect_size("hf tied explicit lm_head mapped", (size_t)saw_lm_head, 1);
    expect_status("parse tied explicit metadata",
                  lis_loader_parse_hf_config_json(HF_CONFIG_TIED_JSON,
                                                  strlen(HF_CONFIG_TIED_JSON),
                                                  &metadata),
                  LIS_STATUS_OK);
    expect_status("reattach tied explicit metadata",
                  lis_loaded_model_attach_metadata(&model, &metadata),
                  LIS_STATUS_OK);
    expect_size("reattach tied explicit keeps lm_head",
                (size_t)model.metadata.config.tie_word_embeddings, 0);
    lis_loaded_model_destroy(&model);
    cleanup_hf_fixture();

    saw_lm_head = 0;
    write_hf_fixture("write tied missing lm_head fixture",
                     HF_CONFIG_TIED_JSON, HF_HEADER_MISSING_LM_HEAD, 12480);
    expect_status("hf tied missing lm_head accepted",
                  lis_loader_load(&source, &model), LIS_STATUS_OK);
    expect_size("hf tied missing reuses embeddings",
                (size_t)model.metadata.config.tie_word_embeddings, 1);
    for (index = 0; index < model.tensor_count; ++index) {
        if (strcmp(model.tensors[index].name, "lis.lm_head.weight") == 0) {
            saw_lm_head = 1;
        }
    }
    expect_size("hf tied missing has no lm_head tensor",
                (size_t)saw_lm_head, 0);
    lis_loaded_model_destroy(&model);
    cleanup_hf_fixture();
}

static void test_hf_precision_dtype_consistency(void)
{
    expect_hf_load_status("hf f16 uniform model",
                          HF_CONFIG_TINY_F16_JSON, HF_HEADER_TINY_F16, 32,
                          LIS_STATUS_OK);
    expect_hf_load_status("hf bf16 uniform model",
                          HF_CONFIG_TINY_BF16_JSON, HF_HEADER_TINY_BF16, 32,
                          LIS_STATUS_OK);
    expect_hf_load_status("hf mixed dtype rejected",
                          HF_CONFIG_TINY_BF16_JSON,
                          HF_HEADER_TINY_MIXED_BF16_F32, 34,
                          LIS_STATUS_UNSUPPORTED_DTYPE);
}

static void test_hf_qwen3_dense_import(void)
{
    lis_model_source source = lis_model_source_from_path(HF_TEST_DIR);
    lis_loaded_model model = { 0 };
    int saw_q_norm = 0;
    int saw_k_norm = 0;
    int saw_expanded_q = 0;
    int saw_expanded_o = 0;
    size_t index;

    expect_hf_load_status("hf qwen3 dense model",
                          HF_CONFIG_QWEN3_TINY_JSON,
                          HF_HEADER_QWEN3_TINY_BF16, 108,
                          LIS_STATUS_OK);
    write_hf_fixture("write qwen3 dense fixture",
                     HF_CONFIG_QWEN3_TINY_JSON,
                     HF_HEADER_QWEN3_TINY_BF16, 108);
    expect_status("load qwen3 dense model", lis_loader_load(&source, &model),
                  LIS_STATUS_OK);
    expect_size("qwen3 tensor count", model.tensor_count, 14);
    expect_size("qwen3 metadata family", (size_t)model.metadata.config.family,
                (size_t)LIS_MODEL_FAMILY_QWEN3_DENSE_DECODER);
    for (index = 0; index < model.tensor_count; ++index) {
        if (strcmp(model.tensors[index].name,
                   "lis.layer.0.q_norm.weight") == 0) {
            saw_q_norm = 1;
        }
        if (strcmp(model.tensors[index].name,
                   "lis.layer.0.k_norm.weight") == 0) {
            saw_k_norm = 1;
        }
    }
    expect_size("qwen3 q_norm mapped", (size_t)saw_q_norm, 1);
    expect_size("qwen3 k_norm mapped", (size_t)saw_k_norm, 1);
    lis_loaded_model_destroy(&model);
    cleanup_hf_fixture();

    write_hf_fixture("write qwen3 expanded q fixture",
                     HF_CONFIG_QWEN3_EXPANDED_Q_JSON,
                     HF_HEADER_QWEN3_EXPANDED_Q_BF16, 124);
    expect_status("load qwen3 expanded q model",
                  lis_loader_load(&source, &model), LIS_STATUS_OK);
    for (index = 0; index < model.tensor_count; ++index) {
        if (strcmp(model.tensors[index].name,
                   "lis.layer.0.q_proj.weight") == 0 &&
            model.tensors[index].view.shape.dims[0] == 4 &&
            model.tensors[index].view.shape.dims[1] == 2) {
            saw_expanded_q = 1;
        }
        if (strcmp(model.tensors[index].name,
                   "lis.layer.0.o_proj.weight") == 0 &&
            model.tensors[index].view.shape.dims[0] == 2 &&
            model.tensors[index].view.shape.dims[1] == 4) {
            saw_expanded_o = 1;
        }
    }
    expect_size("qwen3 expanded q shape mapped", (size_t)saw_expanded_q, 1);
    expect_size("qwen3 expanded o shape mapped", (size_t)saw_expanded_o, 1);
    lis_loaded_model_destroy(&model);
    cleanup_hf_fixture();

    expect_hf_load_status("hf qwen3 old o shape rejected",
                          HF_CONFIG_QWEN3_EXPANDED_Q_JSON,
                          HF_HEADER_QWEN3_EXPANDED_Q_BAD_O_BF16, 116,
                          LIS_STATUS_SHAPE_MISMATCH);
    expect_hf_load_status("hf qwen3 missing q norm rejected",
                          HF_CONFIG_QWEN3_TINY_JSON,
                          HF_HEADER_QWEN3_MISSING_Q_NORM, 108,
                          LIS_STATUS_FORMAT);
    expect_hf_load_status("hf qwen3 mixed dtype rejected",
                          HF_CONFIG_QWEN3_TINY_JSON,
                          HF_HEADER_QWEN3_MIXED_DTYPE, 116,
                          LIS_STATUS_UNSUPPORTED_DTYPE);
}

int main(void)
{
    test_probe_and_load_safetensors();
    test_load_safetensors_precision_dtypes();
    test_malformed_and_unsupported_safetensors();
    test_unsupported_pytorch_path();
    test_llama_config_parser();
    test_qwen3_config_parser();
    test_hf_local_import();
    test_hf_local_import_rejections();
    test_hf_tied_output_policy();
    test_hf_precision_dtype_consistency();
    test_hf_qwen3_dense_import();

    if (g_failures != 0) {
        fprintf(stderr, "%d loader test failure(s)\n", g_failures);
        return 1;
    }

    return 0;
}
