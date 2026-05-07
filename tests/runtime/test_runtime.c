#include "lis/backend.h"
#include "lis/dtype.h"
#include "lis/runtime.h"
#include "lis/status.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
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

static void expect_float_close(const char *name, float actual, float expected,
                               float tolerance)
{
    const float diff = actual > expected ? actual - expected : expected - actual;

    if (diff > tolerance) {
        fprintf(stderr, "%s: expected %.9g +/- %.3g, got %.9g\n",
                name, (double)expected, (double)tolerance, (double)actual);
        ++g_failures;
    }
}

static lis_model_metadata valid_metadata(void)
{
    lis_model_config config = {
        .family = LIS_MODEL_FAMILY_LLAMA3_DECODER,
        .layer_count = 2,
        .hidden_size = 8,
        .intermediate_size = 16,
        .attention_head_count = 2,
        .kv_head_count = 1,
        .head_dim = 4,
        .vocab_size = 32,
        .rope_theta = 10000.0f,
        .rms_norm_eps = 1.0e-5f,
        .weight_dtype = LIS_DTYPE_F32,
        .context = lis_context_window_policy_default(16, 8),
    };
    lis_model_metadata metadata = {
        .config = config,
        .support = lis_model_support_envelope_default(),
    };

    return metadata;
}

static void test_runtime_lifecycle(void)
{
    lis_model_metadata metadata = valid_metadata();
    lis_runtime_options options = { 0 };
    lis_runtime_context runtime = { 0 };

    expect_status("runtime options",
                  lis_runtime_options_init(&options, &metadata,
                                           lis_backend_cpu_reference(), 2),
                  LIS_STATUS_OK);
    expect_status("runtime init", lis_runtime_init(&runtime, &options),
                  LIS_STATUS_OK);
    expect_size("runtime batch size", runtime.batch.batch_size, 2);
    expect_size("runtime max tokens", runtime.batch.max_tokens, 8);
    expect_size("kv layers", runtime.kv_cache.layout.layer_count, 2);
    expect_size("kv batch", runtime.kv_cache.layout.batch_size, 2);
    expect_size("kv context", runtime.kv_cache.layout.context_length, 8);
    expect_size("kv heads", runtime.kv_cache.layout.kv_head_count, 1);
    expect_size("kv head dim", runtime.kv_cache.layout.head_dim, 4);
    lis_runtime_destroy(&runtime);
    lis_runtime_destroy(&runtime);
}

static void test_prefill_decode_transitions(void)
{
    lis_model_metadata metadata = valid_metadata();
    lis_runtime_options options = { 0 };
    lis_runtime_context runtime = { 0 };
    const size_t lengths[] = { 3, 5 };

    expect_status("runtime options for transitions",
                  lis_runtime_options_init(&options, &metadata,
                                           lis_backend_cpu_reference(), 2),
                  LIS_STATUS_OK);
    expect_status("runtime init for transitions",
                  lis_runtime_init(&runtime, &options), LIS_STATUS_OK);
    expect_status("decode before prefill", lis_runtime_decode_step(&runtime),
                  LIS_STATUS_BAD_STATE);
    expect_status("runtime prefill", lis_runtime_prefill(&runtime, lengths, 2),
                  LIS_STATUS_OK);
    expect_size("prefill position 0", runtime.batch.positions[0], 3);
    expect_size("prefill position 1", runtime.batch.positions[1], 5);
    expect_status("runtime decode step", lis_runtime_decode_step(&runtime),
                  LIS_STATUS_OK);
    expect_size("decode position 0", runtime.batch.positions[0], 4);
    expect_size("decode position 1", runtime.batch.positions[1], 6);
    expect_status("prefill twice", lis_runtime_prefill(&runtime, lengths, 2),
                  LIS_STATUS_BAD_STATE);
    lis_runtime_destroy(&runtime);
}

static void test_prefill_validation(void)
{
    lis_model_metadata metadata = valid_metadata();
    lis_runtime_options options = { 0 };
    lis_runtime_context runtime = { 0 };
    const size_t wrong_count[] = { 3 };
    const size_t zero_length[] = { 3, 0 };

    expect_status("runtime options for prefill validation",
                  lis_runtime_options_init(&options, &metadata,
                                           lis_backend_cpu_reference(), 2),
                  LIS_STATUS_OK);
    expect_status("runtime init for prefill validation",
                  lis_runtime_init(&runtime, &options), LIS_STATUS_OK);
    expect_status("prefill null lengths",
                  lis_runtime_prefill(&runtime, NULL, 2),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_status("prefill wrong count",
                  lis_runtime_prefill(&runtime, wrong_count, 1),
                  LIS_STATUS_SHAPE_MISMATCH);
    expect_status("prefill zero length",
                  lis_runtime_prefill(&runtime, zero_length, 2),
                  LIS_STATUS_INVALID_ARGUMENT);
    lis_runtime_destroy(&runtime);
}

static void test_context_limit_validation(void)
{
    lis_model_metadata metadata = valid_metadata();
    lis_runtime_options options = { 0 };
    lis_runtime_context runtime = { 0 };
    const size_t too_long[] = { 9, 1 };
    const size_t fills_to_limit[] = { 8, 8 };

    expect_status("runtime options for limits",
                  lis_runtime_options_init(&options, &metadata,
                                           lis_backend_cpu_reference(), 2),
                  LIS_STATUS_OK);
    expect_status("runtime init for limits", lis_runtime_init(&runtime, &options),
                  LIS_STATUS_OK);
    expect_status("prefill too long",
                  lis_runtime_prefill(&runtime, too_long, 2),
                  LIS_STATUS_LIMIT_EXCEEDED);
    expect_status("prefill at limit",
                  lis_runtime_prefill(&runtime, fills_to_limit, 2),
                  LIS_STATUS_OK);
    expect_status("decode past limit", lis_runtime_decode_step(&runtime),
                  LIS_STATUS_LIMIT_EXCEEDED);
    lis_runtime_destroy(&runtime);
}

static void test_kv_cache_indexing(void)
{
    lis_model_metadata metadata = valid_metadata();
    lis_kv_cache cache = { 0 };
    size_t offset = 0;
    void *ptr = NULL;

    expect_status("kv init", lis_kv_cache_init(&cache, &metadata.config, 2),
                  LIS_STATUS_OK);
    expect_status("kv offset",
                  lis_kv_cache_element_offset(&cache, 1, 1, 7, 0, 3,
                                              &offset),
                  LIS_STATUS_OK);
    expect_size("kv offset value", offset,
                (((((1 * 2 + 1) * 8 + 7) * 1 + 0) * 4 + 3) * 4));
    expect_status("kv key ptr",
                  lis_kv_cache_key_ptr(&cache, 0, 0, 0, 0, 0, &ptr),
                  LIS_STATUS_OK);
    if (ptr == NULL) {
        fprintf(stderr, "kv key ptr was null\n");
        ++g_failures;
    }
    expect_status("kv out of bounds",
                  lis_kv_cache_element_offset(&cache, 0, 0, 8, 0, 0,
                                              &offset),
                  LIS_STATUS_LIMIT_EXCEEDED);
    ptr = (void *)1;
    expect_status("kv key ptr clears on error",
                  lis_kv_cache_key_ptr(&cache, 0, 0, 8, 0, 0, &ptr),
                  LIS_STATUS_LIMIT_EXCEEDED);
    if (ptr != NULL) {
        fprintf(stderr, "kv key ptr error did not clear pointer\n");
        ++g_failures;
    }
    expect_status("kv null out pointer",
                  lis_kv_cache_key_ptr(&cache, 0, 0, 0, 0, 0, NULL),
                  LIS_STATUS_INVALID_ARGUMENT);
    lis_kv_cache_destroy(&cache);
}

static void test_greedy_select(void)
{
    const float logits[] = { -1.0f, 2.0f, 1.0f, 5.0f, 3.0f };
    size_t token_id = 0;

    expect_status("greedy select", lis_runtime_greedy_select(logits, 5,
                                                             &token_id),
                  LIS_STATUS_OK);
    expect_size("greedy token", token_id, 3);
    expect_status("greedy invalid", lis_runtime_greedy_select(NULL, 5,
                                                              &token_id),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_status("greedy zero vocab", lis_runtime_greedy_select(logits, 0,
                                                                 &token_id),
                  LIS_STATUS_INVALID_ARGUMENT);
}

static char *copy_name(const char *name)
{
    char *copy = malloc(strlen(name) + 1U);

    if (copy != NULL) {
        strcpy(copy, name);
    }
    return copy;
}

static lis_status set_test_tensor_raw(lis_loaded_tensor *tensor,
                                      const char *name, lis_dtype dtype,
                                      const void *data, size_t rank,
                                      const size_t *dims)
{
    lis_status status;

    tensor->name = copy_name(name);
    if (tensor->name == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }
    status = lis_tensor_shape_make(rank, dims, &tensor->view.shape);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    tensor->view.dtype = dtype;
    tensor->view.data = data;
    status = lis_tensor_shape_byte_size(&tensor->view.shape, tensor->view.dtype,
                                        &tensor->view.byte_len);
    return status;
}

static lis_status set_test_tensor(lis_loaded_tensor *tensor, const char *name,
                                  const float *data, size_t rank,
                                  const size_t *dims)
{
    return set_test_tensor_raw(tensor, name, LIS_DTYPE_F32, data, rank, dims);
}

static void test_llama_forward_tiny_fixture(void)
{
    lis_model_config config = {
        .family = LIS_MODEL_FAMILY_LLAMA3_DECODER,
        .layer_count = 1,
        .hidden_size = 1,
        .intermediate_size = 1,
        .attention_head_count = 1,
        .kv_head_count = 1,
        .head_dim = 1,
        .vocab_size = 3,
        .rope_theta = 10000.0f,
        .rms_norm_eps = 1.0e-5f,
        .weight_dtype = LIS_DTYPE_F32,
        .context = lis_context_window_policy_default(8, 4),
    };
    lis_model_metadata metadata = {
        .config = config,
        .support = lis_model_support_envelope_default(),
    };
    const float embed[] = { 1.0f, 2.0f, 3.0f };
    const float zero[] = { 0.0f };
    const float one[] = { 1.0f };
    const float head[] = { 0.1f, 0.2f, 0.9f };
    const size_t embed_dims[] = { 3, 1 };
    const size_t hidden_dims[] = { 1, 1 };
    const size_t norm_dims[] = { 1 };
    const size_t head_dims[] = { 3, 1 };
    const size_t tokens[] = { 0, 1 };
    const size_t lengths[] = { 2 };
    float logits[3] = { 0.0f };
    lis_loaded_model model = { 0 };
    lis_runtime_options options = { 0 };
    lis_runtime_context runtime = { 0 };
    size_t token_id = 0;

    model.format = LIS_MODEL_FORMAT_HUGGINGFACE_LOCAL;
    model.has_metadata = 1;
    model.metadata = metadata;
    model.tensor_count = 12;
    model.tensors = calloc(model.tensor_count, sizeof(*model.tensors));
    if (model.tensors == NULL) {
        fprintf(stderr, "tiny llama fixture allocation failed\n");
        ++g_failures;
        return;
    }

    expect_status("tiny llama tensor embed",
                  set_test_tensor(&model.tensors[0],
                                  "lis.token_embeddings.weight", embed, 2,
                                  embed_dims),
                  LIS_STATUS_OK);
    expect_status("tiny llama tensor attn norm",
                  set_test_tensor(&model.tensors[1],
                                  "lis.layer.0.attention_norm.weight", one, 1,
                                  norm_dims),
                  LIS_STATUS_OK);
    expect_status("tiny llama tensor q",
                  set_test_tensor(&model.tensors[2],
                                  "lis.layer.0.q_proj.weight", zero, 2,
                                  hidden_dims),
                  LIS_STATUS_OK);
    expect_status("tiny llama tensor k",
                  set_test_tensor(&model.tensors[3],
                                  "lis.layer.0.k_proj.weight", zero, 2,
                                  hidden_dims),
                  LIS_STATUS_OK);
    expect_status("tiny llama tensor v",
                  set_test_tensor(&model.tensors[4],
                                  "lis.layer.0.v_proj.weight", zero, 2,
                                  hidden_dims),
                  LIS_STATUS_OK);
    expect_status("tiny llama tensor o",
                  set_test_tensor(&model.tensors[5],
                                  "lis.layer.0.o_proj.weight", zero, 2,
                                  hidden_dims),
                  LIS_STATUS_OK);
    expect_status("tiny llama tensor mlp norm",
                  set_test_tensor(&model.tensors[6],
                                  "lis.layer.0.mlp_norm.weight", one, 1,
                                  norm_dims),
                  LIS_STATUS_OK);
    expect_status("tiny llama tensor gate",
                  set_test_tensor(&model.tensors[7],
                                  "lis.layer.0.gate_proj.weight", zero, 2,
                                  hidden_dims),
                  LIS_STATUS_OK);
    expect_status("tiny llama tensor up",
                  set_test_tensor(&model.tensors[8],
                                  "lis.layer.0.up_proj.weight", zero, 2,
                                  hidden_dims),
                  LIS_STATUS_OK);
    expect_status("tiny llama tensor down",
                  set_test_tensor(&model.tensors[9],
                                  "lis.layer.0.down_proj.weight", zero, 2,
                                  hidden_dims),
                  LIS_STATUS_OK);
    expect_status("tiny llama tensor output norm",
                  set_test_tensor(&model.tensors[10],
                                  "lis.output_norm.weight", one, 1,
                                  norm_dims),
                  LIS_STATUS_OK);
    expect_status("tiny llama tensor lm head",
                  set_test_tensor(&model.tensors[11],
                                  "lis.lm_head.weight", head, 2,
                                  head_dims),
                  LIS_STATUS_OK);

    expect_status("tiny llama runtime options",
                  lis_runtime_options_init(&options, &metadata,
                                           lis_backend_cpu_reference(), 1),
                  LIS_STATUS_OK);
    expect_status("tiny llama runtime init",
                  lis_runtime_init(&runtime, &options), LIS_STATUS_OK);
    expect_status("tiny llama prefill",
                  lis_runtime_llama_prefill(&runtime, &model, tokens, lengths,
                                            1, logits, 3),
                  LIS_STATUS_OK);
    expect_size("tiny llama prefill position", runtime.batch.positions[0], 2);
    expect_status("tiny llama greedy after prefill",
                  lis_runtime_greedy_select(logits, 3, &token_id),
                  LIS_STATUS_OK);
    expect_size("tiny llama prefill token", token_id, 2);
    expect_status("tiny llama decode",
                  lis_runtime_llama_decode(&runtime, &model, token_id, logits,
                                           3),
                  LIS_STATUS_OK);
    expect_size("tiny llama decode position", runtime.batch.positions[0], 3);
    expect_status("tiny llama greedy after decode",
                  lis_runtime_greedy_select(logits, 3, &token_id),
                  LIS_STATUS_OK);
    expect_size("tiny llama decode token", token_id, 2);

    lis_runtime_destroy(&runtime);
    lis_loaded_model_destroy(&model);
}

static lis_status build_precision_tiny_llama_model(lis_loaded_model *model,
                                                   lis_model_metadata *metadata,
                                                   lis_dtype dtype,
                                                   int mixed_embedding_dtype)
{
    lis_model_config config = {
        .family = LIS_MODEL_FAMILY_LLAMA3_DECODER,
        .layer_count = 1,
        .hidden_size = 1,
        .intermediate_size = 1,
        .attention_head_count = 1,
        .kv_head_count = 1,
        .head_dim = 1,
        .vocab_size = 3,
        .rope_theta = 10000.0f,
        .rms_norm_eps = 1.0e-5f,
        .weight_dtype = dtype,
        .context = lis_context_window_policy_default(8, 4),
    };
    static const uint16_t f16_embed[] = { 0x3c00U, 0x4000U, 0x4200U };
    static const uint16_t f16_zero[] = { 0x0000U };
    static const uint16_t f16_one[] = { 0x3c00U };
    static const uint16_t f16_two[] = { 0x4000U };
    static const uint16_t f16_head[] = { 0x3c00U, 0x4000U, 0x4400U };
    static const uint16_t bf16_embed[] = { 0x3f80U, 0x4000U, 0x4040U };
    static const uint16_t bf16_zero[] = { 0x0000U };
    static const uint16_t bf16_one[] = { 0x3f80U };
    static const uint16_t bf16_two[] = { 0x4000U };
    static const uint16_t bf16_head[] = { 0x3f80U, 0x4000U, 0x4080U };
    static const float f32_embed[] = { 1.0f, 2.0f, 3.0f };
    const void *embed = dtype == LIS_DTYPE_F16 ?
        (const void *)f16_embed : (const void *)bf16_embed;
    const void *zero = dtype == LIS_DTYPE_F16 ?
        (const void *)f16_zero : (const void *)bf16_zero;
    const void *one = dtype == LIS_DTYPE_F16 ?
        (const void *)f16_one : (const void *)bf16_one;
    const void *two = dtype == LIS_DTYPE_F16 ?
        (const void *)f16_two : (const void *)bf16_two;
    const void *head = dtype == LIS_DTYPE_F16 ?
        (const void *)f16_head : (const void *)bf16_head;
    const size_t embed_dims[] = { 3, 1 };
    const size_t hidden_dims[] = { 1, 1 };
    const size_t norm_dims[] = { 1 };
    const size_t head_dims[] = { 3, 1 };

    *metadata = (lis_model_metadata){
        .config = config,
        .support = lis_model_support_envelope_default(),
    };

    memset(model, 0, sizeof(*model));
    model->format = LIS_MODEL_FORMAT_HUGGINGFACE_LOCAL;
    model->has_metadata = 1;
    model->metadata = *metadata;
    model->tensor_count = 12;
    model->tensors = calloc(model->tensor_count, sizeof(*model->tensors));
    if (model->tensors == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }

    if (set_test_tensor_raw(&model->tensors[0], "lis.token_embeddings.weight",
                            mixed_embedding_dtype ? LIS_DTYPE_F32 : dtype,
                            mixed_embedding_dtype ?
                                (const void *)f32_embed : embed,
                            2, embed_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[1],
                            "lis.layer.0.attention_norm.weight", dtype, one,
                            1, norm_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[2], "lis.layer.0.q_proj.weight",
                            dtype, zero, 2, hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[3], "lis.layer.0.k_proj.weight",
                            dtype, one, 2, hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[4], "lis.layer.0.v_proj.weight",
                            dtype, two, 2, hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[5], "lis.layer.0.o_proj.weight",
                            dtype, zero, 2, hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[6],
                            "lis.layer.0.mlp_norm.weight", dtype, one, 1,
                            norm_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[7], "lis.layer.0.gate_proj.weight",
                            dtype, zero, 2, hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[8], "lis.layer.0.up_proj.weight",
                            dtype, zero, 2, hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[9], "lis.layer.0.down_proj.weight",
                            dtype, zero, 2, hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[10], "lis.output_norm.weight",
                            dtype, one, 1, norm_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[11], "lis.lm_head.weight",
                            dtype, head, 2, head_dims) != LIS_STATUS_OK) {
        lis_loaded_model_destroy(model);
        return LIS_STATUS_NO_MEMORY;
    }
    return LIS_STATUS_OK;
}

static void run_precision_tiny_fixture(lis_dtype dtype)
{
    lis_loaded_model model = { 0 };
    lis_model_metadata metadata = { 0 };
    lis_runtime_options options = { 0 };
    lis_runtime_context runtime = { 0 };
    const size_t tokens[] = { 0, 1 };
    const size_t lengths[] = { 2 };
    float logits[3] = { 0.0f };
    size_t token_id = 0;
    void *key_ptr = NULL;
    void *value_ptr = NULL;

    if (build_precision_tiny_llama_model(&model, &metadata, dtype, 0) !=
        LIS_STATUS_OK) {
        fprintf(stderr, "precision fixture build failed\n");
        ++g_failures;
        return;
    }
    expect_status("precision runtime options",
                  lis_runtime_options_init(&options, &metadata,
                                           lis_backend_cpu_reference(), 1),
                  LIS_STATUS_OK);
    expect_status("precision runtime init", lis_runtime_init(&runtime, &options),
                  LIS_STATUS_OK);
    expect_size("precision kv dtype", (size_t)runtime.kv_cache.layout.dtype,
                (size_t)dtype);
    expect_size("precision kv element size",
                runtime.kv_cache.layout.element_size, 2);
    expect_status("precision prefill",
                  lis_runtime_llama_prefill(&runtime, &model, tokens, lengths,
                                            1, logits, 3),
                  LIS_STATUS_OK);
    expect_status("precision greedy after prefill",
                  lis_runtime_greedy_select(logits, 3, &token_id),
                  LIS_STATUS_OK);
    expect_size("precision prefill token", token_id, 2);
    expect_status("precision kv key ptr",
                  lis_kv_cache_key_ptr(&runtime.kv_cache, 0, 0, 0, 0, 0,
                                       &key_ptr),
                  LIS_STATUS_OK);
    expect_status("precision kv value ptr",
                  lis_kv_cache_value_ptr(&runtime.kv_cache, 0, 0, 0, 0, 0,
                                         &value_ptr),
                  LIS_STATUS_OK);
    expect_float_close("precision kv key promoted",
                       lis_dtype_scalar_read_f32(dtype, key_ptr, 0),
                       1.0f, 1.0e-2f);
    expect_float_close("precision kv value promoted",
                       lis_dtype_scalar_read_f32(dtype, value_ptr, 0),
                       2.0f, 1.0e-2f);
    expect_status("precision decode",
                  lis_runtime_llama_decode(&runtime, &model, token_id, logits,
                                           3),
                  LIS_STATUS_OK);
    expect_status("precision greedy after decode",
                  lis_runtime_greedy_select(logits, 3, &token_id),
                  LIS_STATUS_OK);
    expect_size("precision decode token", token_id, 2);

    lis_runtime_destroy(&runtime);
    lis_loaded_model_destroy(&model);
}

static void test_precision_f16_bf16_forward_and_kv_cache(void)
{
    run_precision_tiny_fixture(LIS_DTYPE_F16);
    run_precision_tiny_fixture(LIS_DTYPE_BF16);
}

static void test_precision_mixed_runtime_dtype_rejected(void)
{
    lis_loaded_model model = { 0 };
    lis_model_metadata metadata = { 0 };
    lis_runtime_options options = { 0 };
    lis_runtime_context runtime = { 0 };
    const size_t tokens[] = { 0 };
    const size_t lengths[] = { 1 };
    float logits[3] = { 0.0f };

    if (build_precision_tiny_llama_model(&model, &metadata, LIS_DTYPE_BF16,
                                         1) != LIS_STATUS_OK) {
        fprintf(stderr, "mixed precision fixture build failed\n");
        ++g_failures;
        return;
    }
    expect_status("mixed precision runtime options",
                  lis_runtime_options_init(&options, &metadata,
                                           lis_backend_cpu_reference(), 1),
                  LIS_STATUS_OK);
    expect_status("mixed precision runtime init",
                  lis_runtime_init(&runtime, &options), LIS_STATUS_OK);
    expect_status("mixed precision prefill rejected",
                  lis_runtime_llama_prefill(&runtime, &model, tokens, lengths,
                                            1, logits, 3),
                  LIS_STATUS_UNSUPPORTED_DTYPE);

    lis_runtime_destroy(&runtime);
    lis_loaded_model_destroy(&model);
}

static lis_status build_qwen3_tiny_model(lis_loaded_model *model,
                                         lis_model_metadata *metadata,
                                         int scaled_q_norm,
                                         int mixed_embedding_dtype)
{
    lis_model_config config = {
        .family = LIS_MODEL_FAMILY_QWEN3_DENSE_DECODER,
        .layer_count = 1,
        .hidden_size = 2,
        .intermediate_size = 2,
        .attention_head_count = 1,
        .kv_head_count = 1,
        .head_dim = 2,
        .vocab_size = 3,
        .rope_theta = 10000.0f,
        .rms_norm_eps = 1.0e-6f,
        .weight_dtype = LIS_DTYPE_BF16,
        .context = lis_context_window_policy_default(8, 4),
    };
    static const uint16_t bf16_embed[] = {
        0x3f80U, 0x0000U, 0x0000U, 0x3f80U, 0x3f80U, 0x3f80U
    };
    static const float f32_embed[] = {
        1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f
    };
    static const uint16_t bf16_identity[] = {
        0x3f80U, 0x0000U, 0x0000U, 0x3f80U
    };
    static const uint16_t bf16_zero4[] = {
        0x0000U, 0x0000U, 0x0000U, 0x0000U
    };
    static const uint16_t bf16_one2[] = { 0x3f80U, 0x3f80U };
    static const uint16_t bf16_q_norm_scaled[] = { 0x4000U, 0x3f80U };
    static const uint16_t bf16_head[] = {
        0x3f80U, 0x0000U, 0x0000U, 0x3f80U, 0x3f80U, 0x3f80U
    };
    const size_t embed_dims[] = { 3, 2 };
    const size_t hidden_dims[] = { 2, 2 };
    const size_t norm_dims[] = { 2 };
    const size_t head_dims[] = { 3, 2 };
    const uint16_t *q_norm = scaled_q_norm ? bf16_q_norm_scaled : bf16_one2;

    *metadata = (lis_model_metadata){
        .config = config,
        .support = lis_model_support_envelope_default(),
    };

    memset(model, 0, sizeof(*model));
    model->format = LIS_MODEL_FORMAT_HUGGINGFACE_LOCAL;
    model->has_metadata = 1;
    model->metadata = *metadata;
    model->tensor_count = 14;
    model->tensors = calloc(model->tensor_count, sizeof(*model->tensors));
    if (model->tensors == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }
    if (set_test_tensor_raw(&model->tensors[0], "lis.token_embeddings.weight",
                            mixed_embedding_dtype ? LIS_DTYPE_F32 :
                                LIS_DTYPE_BF16,
                            mixed_embedding_dtype ?
                                (const void *)f32_embed :
                                (const void *)bf16_embed,
                            2, embed_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[1],
                            "lis.layer.0.attention_norm.weight",
                            LIS_DTYPE_BF16, bf16_one2, 1,
                            norm_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[2], "lis.layer.0.q_proj.weight",
                            LIS_DTYPE_BF16, bf16_identity, 2,
                            hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[3], "lis.layer.0.k_proj.weight",
                            LIS_DTYPE_BF16, bf16_identity, 2,
                            hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[4], "lis.layer.0.v_proj.weight",
                            LIS_DTYPE_BF16, bf16_identity, 2,
                            hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[5], "lis.layer.0.o_proj.weight",
                            LIS_DTYPE_BF16, bf16_identity, 2,
                            hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[6], "lis.layer.0.q_norm.weight",
                            LIS_DTYPE_BF16, q_norm, 1,
                            norm_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[7], "lis.layer.0.k_norm.weight",
                            LIS_DTYPE_BF16, bf16_one2, 1,
                            norm_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[8], "lis.layer.0.mlp_norm.weight",
                            LIS_DTYPE_BF16, bf16_one2, 1,
                            norm_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[9], "lis.layer.0.gate_proj.weight",
                            LIS_DTYPE_BF16, bf16_zero4, 2,
                            hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[10], "lis.layer.0.up_proj.weight",
                            LIS_DTYPE_BF16, bf16_zero4, 2,
                            hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[11], "lis.layer.0.down_proj.weight",
                            LIS_DTYPE_BF16, bf16_zero4, 2,
                            hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[12], "lis.output_norm.weight",
                            LIS_DTYPE_BF16, bf16_one2, 1,
                            norm_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model->tensors[13], "lis.lm_head.weight",
                            LIS_DTYPE_BF16, bf16_head, 2,
                            head_dims) != LIS_STATUS_OK) {
        lis_loaded_model_destroy(model);
        return LIS_STATUS_NO_MEMORY;
    }
    return LIS_STATUS_OK;
}

static lis_status run_qwen3_tiny_logits(int scaled_q_norm, float logits[3],
                                        lis_runtime_context *out_runtime,
                                        lis_loaded_model *out_model)
{
    lis_model_metadata metadata = { 0 };
    lis_runtime_options options = { 0 };
    const size_t tokens[] = { 0, 2 };
    const size_t lengths[] = { 2 };
    lis_status status;

    status = build_qwen3_tiny_model(out_model, &metadata, scaled_q_norm, 0);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_runtime_options_init(&options, &metadata,
                                      lis_backend_cpu_reference(), 1);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_runtime_init(out_runtime, &options);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    return lis_runtime_qwen3_prefill(out_runtime, out_model, tokens, lengths,
                                     1, logits, 3);
}

static void test_qwen3_forward_tiny_fixture(void)
{
    lis_loaded_model model_base = { 0 };
    lis_loaded_model model_scaled = { 0 };
    lis_runtime_context runtime_base = { 0 };
    lis_runtime_context runtime_scaled = { 0 };
    float logits_base[3] = { 0.0f };
    float logits_scaled[3] = { 0.0f };
    float diff = 0.0f;
    size_t token_id = 0;
    void *value_ptr = NULL;

    expect_status("qwen3 base prefill",
                  run_qwen3_tiny_logits(0, logits_base, &runtime_base,
                                        &model_base),
                  LIS_STATUS_OK);
    expect_status("qwen3 scaled q norm prefill",
                  run_qwen3_tiny_logits(1, logits_scaled, &runtime_scaled,
                                        &model_scaled),
                  LIS_STATUS_OK);
    expect_size("qwen3 kv dtype",
                (size_t)runtime_base.kv_cache.layout.dtype,
                (size_t)LIS_DTYPE_BF16);
    expect_status("qwen3 kv value ptr",
                  lis_kv_cache_value_ptr(&runtime_base.kv_cache, 0, 0, 0, 0,
                                         0, &value_ptr),
                  LIS_STATUS_OK);
    if (lis_dtype_scalar_read_f32(LIS_DTYPE_BF16, value_ptr, 0) <= 1.0f) {
        fprintf(stderr, "qwen3 kv value did not preserve promoted BF16 data\n");
        ++g_failures;
    }
    diff = fabsf(logits_scaled[0] - logits_base[0]) +
           fabsf(logits_scaled[1] - logits_base[1]) +
           fabsf(logits_scaled[2] - logits_base[2]);
    if (diff <= 1.0e-4f) {
        fprintf(stderr, "qwen3 q_norm did not affect logits enough\n");
        ++g_failures;
    }
    expect_status("qwen3 greedy after prefill",
                  lis_runtime_greedy_select(logits_scaled, 3, &token_id),
                  LIS_STATUS_OK);
    expect_status("qwen3 decode",
                  lis_runtime_qwen3_decode(&runtime_scaled, &model_scaled,
                                           token_id, logits_scaled, 3),
                  LIS_STATUS_OK);
    expect_size("qwen3 decode position", runtime_scaled.batch.positions[0],
                3);

    lis_runtime_destroy(&runtime_base);
    lis_runtime_destroy(&runtime_scaled);
    lis_loaded_model_destroy(&model_base);
    lis_loaded_model_destroy(&model_scaled);
}

static void test_qwen3_expanded_q_geometry_runtime(void)
{
    lis_model_config config = {
        .family = LIS_MODEL_FAMILY_QWEN3_DENSE_DECODER,
        .layer_count = 1,
        .hidden_size = 2,
        .intermediate_size = 2,
        .attention_head_count = 2,
        .kv_head_count = 1,
        .head_dim = 2,
        .vocab_size = 3,
        .rope_theta = 10000.0f,
        .rms_norm_eps = 1.0e-6f,
        .weight_dtype = LIS_DTYPE_BF16,
        .context = lis_context_window_policy_default(8, 4),
    };
    lis_model_metadata metadata = {
        .config = config,
        .support = lis_model_support_envelope_default(),
    };
    static const uint16_t bf16_embed[] = {
        0x3f80U, 0x0000U, 0x0000U, 0x3f80U, 0x3f80U, 0x3f80U
    };
    static const uint16_t bf16_q_proj[] = {
        0x3f80U, 0x0000U, 0x0000U, 0x3f80U,
        0x3f80U, 0x0000U, 0x0000U, 0x3f80U
    };
    static const uint16_t bf16_identity[] = {
        0x3f80U, 0x0000U, 0x0000U, 0x3f80U
    };
    static const uint16_t bf16_o_proj[] = {
        0x3f80U, 0x0000U, 0x3f80U, 0x0000U,
        0x0000U, 0x3f80U, 0x0000U, 0x3f80U
    };
    static const uint16_t bf16_zero4[] = {
        0x0000U, 0x0000U, 0x0000U, 0x0000U
    };
    static const uint16_t bf16_one2[] = { 0x3f80U, 0x3f80U };
    static const uint16_t bf16_head[] = {
        0x3f80U, 0x0000U, 0x0000U, 0x3f80U, 0x3f80U, 0x3f80U
    };
    const size_t embed_dims[] = { 3, 2 };
    const size_t hidden_dims[] = { 2, 2 };
    const size_t q_dims[] = { 4, 2 };
    const size_t o_dims[] = { 2, 4 };
    const size_t norm_dims[] = { 2 };
    const size_t head_dims[] = { 3, 2 };
    lis_loaded_model model = { 0 };
    lis_runtime_options options = { 0 };
    lis_runtime_context runtime = { 0 };
    const size_t tokens[] = { 0 };
    const size_t lengths[] = { 1 };
    float logits[3] = { 0.0f };

    memset(&model, 0, sizeof(model));
    model.format = LIS_MODEL_FORMAT_HUGGINGFACE_LOCAL;
    model.has_metadata = 1;
    model.metadata = metadata;
    model.tensor_count = 14;
    model.tensors = calloc(model.tensor_count, sizeof(*model.tensors));
    if (model.tensors == NULL) {
        fprintf(stderr, "qwen3 expanded q fixture allocation failed\n");
        ++g_failures;
        return;
    }
    if (set_test_tensor_raw(&model.tensors[0], "lis.token_embeddings.weight",
                            LIS_DTYPE_BF16, bf16_embed, 2,
                            embed_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model.tensors[1],
                            "lis.layer.0.attention_norm.weight",
                            LIS_DTYPE_BF16, bf16_one2, 1,
                            norm_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model.tensors[2], "lis.layer.0.q_proj.weight",
                            LIS_DTYPE_BF16, bf16_q_proj, 2,
                            q_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model.tensors[3], "lis.layer.0.k_proj.weight",
                            LIS_DTYPE_BF16, bf16_identity, 2,
                            hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model.tensors[4], "lis.layer.0.v_proj.weight",
                            LIS_DTYPE_BF16, bf16_identity, 2,
                            hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model.tensors[5], "lis.layer.0.o_proj.weight",
                            LIS_DTYPE_BF16, bf16_o_proj, 2,
                            o_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model.tensors[6], "lis.layer.0.q_norm.weight",
                            LIS_DTYPE_BF16, bf16_one2, 1,
                            norm_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model.tensors[7], "lis.layer.0.k_norm.weight",
                            LIS_DTYPE_BF16, bf16_one2, 1,
                            norm_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model.tensors[8], "lis.layer.0.mlp_norm.weight",
                            LIS_DTYPE_BF16, bf16_one2, 1,
                            norm_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model.tensors[9], "lis.layer.0.gate_proj.weight",
                            LIS_DTYPE_BF16, bf16_zero4, 2,
                            hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model.tensors[10], "lis.layer.0.up_proj.weight",
                            LIS_DTYPE_BF16, bf16_zero4, 2,
                            hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model.tensors[11], "lis.layer.0.down_proj.weight",
                            LIS_DTYPE_BF16, bf16_zero4, 2,
                            hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model.tensors[12], "lis.output_norm.weight",
                            LIS_DTYPE_BF16, bf16_one2, 1,
                            norm_dims) != LIS_STATUS_OK ||
        set_test_tensor_raw(&model.tensors[13], "lis.lm_head.weight",
                            LIS_DTYPE_BF16, bf16_head, 2,
                            head_dims) != LIS_STATUS_OK) {
        fprintf(stderr, "qwen3 expanded q fixture build failed\n");
        ++g_failures;
        lis_loaded_model_destroy(&model);
        return;
    }

    expect_status("qwen3 expanded q runtime options",
                  lis_runtime_options_init(&options, &metadata,
                                           lis_backend_cpu_reference(), 1),
                  LIS_STATUS_OK);
    expect_status("qwen3 expanded q runtime init",
                  lis_runtime_init(&runtime, &options), LIS_STATUS_OK);
    expect_size("qwen3 expanded q kv heads",
                runtime.kv_cache.layout.kv_head_count, 1);
    expect_size("qwen3 expanded q head dim",
                runtime.kv_cache.layout.head_dim, 2);
    expect_status("qwen3 expanded q prefill",
                  lis_runtime_qwen3_prefill(&runtime, &model, tokens,
                                            lengths, 1, logits, 3),
                  LIS_STATUS_OK);

    lis_runtime_destroy(&runtime);
    lis_loaded_model_destroy(&model);
}

static void test_qwen3_mixed_runtime_dtype_rejected(void)
{
    lis_loaded_model model = { 0 };
    lis_model_metadata metadata = { 0 };
    lis_runtime_options options = { 0 };
    lis_runtime_context runtime = { 0 };
    const size_t tokens[] = { 0 };
    const size_t lengths[] = { 1 };
    float logits[3] = { 0.0f };

    if (build_qwen3_tiny_model(&model, &metadata, 0, 1) != LIS_STATUS_OK) {
        fprintf(stderr, "qwen3 mixed fixture build failed\n");
        ++g_failures;
        return;
    }
    expect_status("qwen3 mixed runtime options",
                  lis_runtime_options_init(&options, &metadata,
                                           lis_backend_cpu_reference(), 1),
                  LIS_STATUS_OK);
    expect_status("qwen3 mixed runtime init",
                  lis_runtime_init(&runtime, &options), LIS_STATUS_OK);
    expect_status("qwen3 mixed prefill rejected",
                  lis_runtime_qwen3_prefill(&runtime, &model, tokens,
                                            lengths, 1, logits, 3),
                  LIS_STATUS_UNSUPPORTED_DTYPE);

    lis_runtime_destroy(&runtime);
    lis_loaded_model_destroy(&model);
}

static void test_context_semantics_exhaustion_and_init_validate(void)
{
    lis_model_metadata metadata = valid_metadata();
    lis_runtime_options options = { 0 };
    lis_runtime_context runtime = { 0 };
    lis_runtime_context zeroed = { 0 };
    const size_t at_context_max[] = { 8 };

    /* Prefill at exactly configured_max_tokens is accepted. */
    expect_status("runtime options",
                  lis_runtime_options_init(&options, &metadata,
                                           lis_backend_cpu_reference(), 1),
                  LIS_STATUS_OK);
    expect_status("runtime init", lis_runtime_init(&runtime, &options),
                  LIS_STATUS_OK);
    expect_size("runtime max tokens", runtime.batch.max_tokens, 8);
    expect_status("prefill at configured_max_tokens",
                  lis_runtime_prefill(&runtime, at_context_max, 1),
                  LIS_STATUS_OK);
    /* Decode from that state returns LIMIT_EXCEEDED without mutating. */
    expect_status("decode past context returns LIMIT_EXCEEDED",
                  lis_runtime_decode_step(&runtime),
                  LIS_STATUS_LIMIT_EXCEEDED);
    expect_size("position unchanged after exhaustion",
                runtime.batch.positions[0], 8);
    lis_runtime_destroy(&runtime);

    /* Invalid context policy at init returns error and runtime is zeroed. */
    metadata.config.context = lis_context_window_policy_default(8, 16);
    memset(&options, 0, sizeof(options));
    memset(&runtime, 0, sizeof(runtime));
    expect_status("configured > trained at options init",
                  lis_runtime_options_init(&options, &metadata,
                                           lis_backend_cpu_reference(), 1),
                  LIS_STATUS_LIMIT_EXCEEDED);
    /* runtime remains zeroed — no allocations leaked */
    expect_size("runtime zeroed after invalid policy",
                 (size_t)memcmp(&runtime, &zeroed, sizeof(runtime)), 0);
}

static void test_context_semantics_runtime_init_invalid_context(void)
{
    lis_model_metadata metadata = valid_metadata();
    lis_runtime_options options = { 0 };
    lis_runtime_context runtime = { 0 };
    lis_runtime_context zeroed = { 0 };

    metadata.config.context =
        lis_context_window_policy_default(8, 16);
    options.metadata = metadata;
    options.backend = lis_backend_cpu_reference();
    options.batch_size = 1;
    options.thread_count = 1;
    expect_status("runtime init invalid context direct",
                  lis_runtime_init(&runtime, &options),
                  LIS_STATUS_LIMIT_EXCEEDED);
    expect_size("runtime zeroed after invalid context init",
                (size_t)memcmp(&runtime, &zeroed, sizeof(runtime)), 0);
}

int main(void)
{
    test_runtime_lifecycle();
    test_prefill_decode_transitions();
    test_prefill_validation();
    test_context_limit_validation();
    test_kv_cache_indexing();
    test_greedy_select();
    test_llama_forward_tiny_fixture();
    test_precision_f16_bf16_forward_and_kv_cache();
    test_precision_mixed_runtime_dtype_rejected();
    test_qwen3_forward_tiny_fixture();
    test_qwen3_expanded_q_geometry_runtime();
    test_qwen3_mixed_runtime_dtype_rejected();

    test_context_semantics_exhaustion_and_init_validate();
    test_context_semantics_runtime_init_invalid_context();

    if (g_failures != 0) {
        fprintf(stderr, "%d runtime test failure(s)\n", g_failures);
        return 1;
    }

    return 0;
}
