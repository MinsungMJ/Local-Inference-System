#include "lis/backend.h"
#include "lis/runtime.h"
#include "lis/thread_pool.h"
#include "lis/status.h"

#include <stddef.h>
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

static void expect_float(const char *name, float actual, float expected)
{
    if (actual != expected) {
        fprintf(stderr, "%s: expected %g, got %g\n", name,
                (double)expected, (double)actual);
        ++g_failures;
    }
}

/* --- Thread pool lifecycle and dispatch --- */

static void test_pool_init_destroy_single(void)
{
    lis_thread_pool *pool = NULL;

    expect_status("pool init single", lis_thread_pool_init(&pool, 1),
                  LIS_STATUS_OK);
    expect_size("pool thread count single", lis_thread_pool_thread_count(pool),
                1);
    lis_thread_pool_destroy(&pool);
    if (pool != NULL) {
        fprintf(stderr, "pool destroy did not set pointer to NULL\n");
        ++g_failures;
    }
}

static void test_pool_init_destroy_multi(void)
{
    lis_thread_pool *pool = NULL;

    expect_status("pool init multi", lis_thread_pool_init(&pool, 4),
                  LIS_STATUS_OK);
    expect_size("pool thread count multi", lis_thread_pool_thread_count(pool),
                4);
    lis_thread_pool_destroy(&pool);
    if (pool != NULL) {
        fprintf(stderr, "pool destroy multi did not set pointer to NULL\n");
        ++g_failures;
    }
}

static void test_pool_init_zero_rejected(void)
{
    lis_thread_pool *pool = NULL;

    expect_status("pool init zero", lis_thread_pool_init(&pool, 0),
                  LIS_STATUS_INVALID_ARGUMENT);
}

static void test_pool_null_safety(void)
{
    expect_status("pool init null out", lis_thread_pool_init(NULL, 1),
                  LIS_STATUS_INVALID_ARGUMENT);
    lis_thread_pool_destroy(NULL);
    expect_size("pool thread count null", lis_thread_pool_thread_count(NULL),
                0);
}

static void test_pool_double_destroy(void)
{
    lis_thread_pool *pool = NULL;

    expect_status("pool init for double destroy",
                  lis_thread_pool_init(&pool, 2), LIS_STATUS_OK);
    lis_thread_pool_destroy(&pool);
    lis_thread_pool_destroy(&pool);
}

typedef struct {
    float *values;
} add_one_ctx;

static void add_one_work(size_t start, size_t count, void *context)
{
    add_one_ctx *ctx = (add_one_ctx *)context;
    size_t i;

    for (i = start; i < start + count; ++i) {
        ctx->values[i] += 1.0f;
    }
}

static void test_pool_dispatch_single_thread(void)
{
    lis_thread_pool *pool = NULL;
    float values[8] = { 0 };
    add_one_ctx ctx = { .values = values };
    size_t i;

    expect_status("pool init for dispatch single",
                  lis_thread_pool_init(&pool, 1), LIS_STATUS_OK);
    expect_status("dispatch single",
                  lis_thread_pool_dispatch(pool, 8, add_one_work, &ctx),
                  LIS_STATUS_OK);
    for (i = 0; i < 8; ++i) {
        expect_float("dispatch single value", values[i], 1.0f);
    }
    lis_thread_pool_destroy(&pool);
}

static void test_pool_dispatch_multi_thread(void)
{
    lis_thread_pool *pool = NULL;
    float values[100] = { 0 };
    add_one_ctx ctx = { .values = values };
    size_t i;

    expect_status("pool init for dispatch multi",
                  lis_thread_pool_init(&pool, 4), LIS_STATUS_OK);
    expect_status("dispatch multi",
                  lis_thread_pool_dispatch(pool, 100, add_one_work, &ctx),
                  LIS_STATUS_OK);
    for (i = 0; i < 100; ++i) {
        expect_float("dispatch multi value", values[i], 1.0f);
    }
    lis_thread_pool_destroy(&pool);
}

static void test_pool_dispatch_total_less_than_threads(void)
{
    lis_thread_pool *pool = NULL;
    float values[3] = { 0 };
    add_one_ctx ctx = { .values = values };
    size_t i;

    expect_status("pool init total<threads",
                  lis_thread_pool_init(&pool, 8), LIS_STATUS_OK);
    expect_status("dispatch total<threads",
                  lis_thread_pool_dispatch(pool, 3, add_one_work, &ctx),
                  LIS_STATUS_OK);
    for (i = 0; i < 3; ++i) {
        expect_float("total<threads value", values[i], 1.0f);
    }
    lis_thread_pool_destroy(&pool);
}

static void test_pool_dispatch_zero_total(void)
{
    lis_thread_pool *pool = NULL;

    expect_status("pool init for zero total",
                  lis_thread_pool_init(&pool, 2), LIS_STATUS_OK);
    expect_status("dispatch zero total",
                  lis_thread_pool_dispatch(pool, 0, add_one_work, NULL),
                  LIS_STATUS_OK);
    lis_thread_pool_destroy(&pool);
}

static void test_pool_dispatch_multiple_rounds(void)
{
    lis_thread_pool *pool = NULL;
    float values[16] = { 0 };
    add_one_ctx ctx = { .values = values };
    int round;
    size_t i;

    expect_status("pool init for multi-round",
                  lis_thread_pool_init(&pool, 3), LIS_STATUS_OK);
    for (round = 0; round < 5; ++round) {
        expect_status("dispatch round",
                      lis_thread_pool_dispatch(pool, 16, add_one_work, &ctx),
                      LIS_STATUS_OK);
    }
    for (i = 0; i < 16; ++i) {
        expect_float("multi-round value", values[i], 5.0f);
    }
    lis_thread_pool_destroy(&pool);
}

/* --- Runtime thread pool integration --- */

static lis_model_metadata valid_metadata(void)
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

    return metadata;
}

static void test_runtime_init_thread_count_default(void)
{
    lis_model_metadata metadata = valid_metadata();
    lis_runtime_options options = { 0 };
    lis_runtime_context runtime = { 0 };

    expect_status("runtime options default threads",
                  lis_runtime_options_init(&options, &metadata,
                                           lis_backend_cpu_reference(), 1),
                  LIS_STATUS_OK);
    expect_size("default thread count", options.thread_count, 1);
    expect_status("runtime init default threads",
                  lis_runtime_init(&runtime, &options), LIS_STATUS_OK);
    expect_size("runtime pool threads",
                lis_thread_pool_thread_count(runtime.pool), 1);
    lis_runtime_destroy(&runtime);
}

static void test_runtime_init_thread_count_multi(void)
{
    lis_model_metadata metadata = valid_metadata();
    lis_runtime_options options = { 0 };
    lis_runtime_context runtime = { 0 };

    expect_status("runtime options for multi-thread",
                  lis_runtime_options_init(&options, &metadata,
                                           lis_backend_cpu_reference(), 1),
                  LIS_STATUS_OK);
    options.thread_count = 4;
    expect_status("runtime init multi-thread",
                  lis_runtime_init(&runtime, &options), LIS_STATUS_OK);
    expect_size("runtime pool multi-threads",
                lis_thread_pool_thread_count(runtime.pool), 4);
    lis_runtime_destroy(&runtime);
}

static void test_runtime_init_thread_count_zero_rejected(void)
{
    lis_model_metadata metadata = valid_metadata();
    lis_runtime_options options = { 0 };
    lis_runtime_context runtime = { 0 };

    expect_status("runtime options for zero-thread",
                  lis_runtime_options_init(&options, &metadata,
                                           lis_backend_cpu_reference(), 1),
                  LIS_STATUS_OK);
    options.thread_count = 0;
    expect_status("runtime init zero threads",
                  lis_runtime_init(&runtime, &options),
                  LIS_STATUS_INVALID_ARGUMENT);
}

/* --- Llama forward with threading --- */

static char *copy_name(const char *name)
{
    char *copy = malloc(strlen(name) + 1U);

    if (copy != NULL) {
        strcpy(copy, name);
    }
    return copy;
}

static lis_status set_test_tensor(lis_loaded_tensor *tensor, const char *name,
                                  const float *data, size_t rank,
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
    tensor->view.dtype = LIS_DTYPE_F32;
    tensor->view.data = data;
    status = lis_tensor_shape_byte_size(&tensor->view.shape, tensor->view.dtype,
                                        &tensor->view.byte_len);
    return status;
}

static lis_status build_tiny_llama_model(lis_loaded_model *model,
                                          lis_model_metadata *metadata)
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
    static const float embed[] = { 1.0f, 2.0f, 3.0f };
    static const float zero[] = { 0.0f };
    static const float one[] = { 1.0f };
    static const float head[] = { 0.1f, 0.2f, 0.9f };
    static const size_t embed_dims[] = { 3, 1 };
    static const size_t hidden_dims[] = { 1, 1 };
    static const size_t norm_dims[] = { 1 };
    static const size_t head_dims[] = { 3, 1 };

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

    if (set_test_tensor(&model->tensors[0], "lis.token_embeddings.weight",
                        embed, 2, embed_dims) != LIS_STATUS_OK ||
        set_test_tensor(&model->tensors[1],
                        "lis.layer.0.attention_norm.weight", one, 1,
                        norm_dims) != LIS_STATUS_OK ||
        set_test_tensor(&model->tensors[2], "lis.layer.0.q_proj.weight",
                        zero, 2, hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor(&model->tensors[3], "lis.layer.0.k_proj.weight",
                        zero, 2, hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor(&model->tensors[4], "lis.layer.0.v_proj.weight",
                        zero, 2, hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor(&model->tensors[5], "lis.layer.0.o_proj.weight",
                        zero, 2, hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor(&model->tensors[6],
                        "lis.layer.0.mlp_norm.weight", one, 1,
                        norm_dims) != LIS_STATUS_OK ||
        set_test_tensor(&model->tensors[7], "lis.layer.0.gate_proj.weight",
                        zero, 2, hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor(&model->tensors[8], "lis.layer.0.up_proj.weight",
                        zero, 2, hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor(&model->tensors[9], "lis.layer.0.down_proj.weight",
                        zero, 2, hidden_dims) != LIS_STATUS_OK ||
        set_test_tensor(&model->tensors[10], "lis.output_norm.weight",
                        one, 1, norm_dims) != LIS_STATUS_OK ||
        set_test_tensor(&model->tensors[11], "lis.lm_head.weight",
                        head, 2, head_dims) != LIS_STATUS_OK) {
        lis_loaded_model_destroy(model);
        return LIS_STATUS_NO_MEMORY;
    }
    return LIS_STATUS_OK;
}

static void run_llama_tiny_fixture(size_t thread_count,
                                    float *logits_prefill,
                                    float *logits_decode,
                                    size_t *prefill_token,
                                    size_t *decode_token)
{
    lis_loaded_model model = { 0 };
    lis_model_metadata metadata = { 0 };
    lis_runtime_options options = { 0 };
    lis_runtime_context runtime = { 0 };
    const size_t tokens[] = { 0, 1 };
    const size_t lengths[] = { 2 };
    float logits[3] = { 0 };

    if (build_tiny_llama_model(&model, &metadata) != LIS_STATUS_OK) {
        fprintf(stderr, "failed to build tiny llama model\n");
        ++g_failures;
        return;
    }
    if (lis_runtime_options_init(&options, &metadata,
                                  lis_backend_cpu_reference(), 1) !=
        LIS_STATUS_OK) {
        fprintf(stderr, "runtime options init failed\n");
        ++g_failures;
        lis_loaded_model_destroy(&model);
        return;
    }
    options.thread_count = thread_count;
    if (lis_runtime_init(&runtime, &options) != LIS_STATUS_OK) {
        fprintf(stderr, "runtime init failed for thread_count=%zu\n",
                thread_count);
        ++g_failures;
        lis_loaded_model_destroy(&model);
        return;
    }
    if (lis_runtime_llama_prefill(&runtime, &model, tokens, lengths, 1,
                                   logits, 3) != LIS_STATUS_OK) {
        fprintf(stderr, "prefill failed for thread_count=%zu\n", thread_count);
        ++g_failures;
        lis_runtime_destroy(&runtime);
        lis_loaded_model_destroy(&model);
        return;
    }
    memcpy(logits_prefill, logits, sizeof(logits));
    lis_runtime_greedy_select(logits, 3, prefill_token);

    if (lis_runtime_llama_decode(&runtime, &model, *prefill_token, logits,
                                  3) != LIS_STATUS_OK) {
        fprintf(stderr, "decode failed for thread_count=%zu\n", thread_count);
        ++g_failures;
        lis_runtime_destroy(&runtime);
        lis_loaded_model_destroy(&model);
        return;
    }
    memcpy(logits_decode, logits, sizeof(logits));
    lis_runtime_greedy_select(logits, 3, decode_token);

    lis_runtime_destroy(&runtime);
    lis_loaded_model_destroy(&model);
}

static void test_llama_forward_threads_1_identity(void)
{
    float logits_prefill_1[3] = { 0 };
    float logits_decode_1[3] = { 0 };
    size_t prefill_token_1 = 0;
    size_t decode_token_1 = 0;

    run_llama_tiny_fixture(1, logits_prefill_1, logits_decode_1,
                            &prefill_token_1, &decode_token_1);
    expect_size("threads-1 prefill token", prefill_token_1, 2);
    expect_size("threads-1 decode token", decode_token_1, 2);
}

static void test_llama_forward_threads_2_identical(void)
{
    float logits_prefill_1[3] = { 0 };
    float logits_decode_1[3] = { 0 };
    float logits_prefill_2[3] = { 0 };
    float logits_decode_2[3] = { 0 };
    size_t pt1 = 0, dt1 = 0, pt2 = 0, dt2 = 0;
    size_t i;

    run_llama_tiny_fixture(1, logits_prefill_1, logits_decode_1, &pt1, &dt1);
    run_llama_tiny_fixture(2, logits_prefill_2, logits_decode_2, &pt2, &dt2);

    expect_size("threads-2 prefill token matches", pt2, pt1);
    expect_size("threads-2 decode token matches", dt2, dt1);
    for (i = 0; i < 3; ++i) {
        expect_float("threads-2 prefill logit", logits_prefill_2[i],
                     logits_prefill_1[i]);
        expect_float("threads-2 decode logit", logits_decode_2[i],
                     logits_decode_1[i]);
    }
}

static void test_llama_forward_threads_4_identical(void)
{
    float logits_prefill_1[3] = { 0 };
    float logits_decode_1[3] = { 0 };
    float logits_prefill_4[3] = { 0 };
    float logits_decode_4[3] = { 0 };
    size_t pt1 = 0, dt1 = 0, pt4 = 0, dt4 = 0;
    size_t i;

    run_llama_tiny_fixture(1, logits_prefill_1, logits_decode_1, &pt1, &dt1);
    run_llama_tiny_fixture(4, logits_prefill_4, logits_decode_4, &pt4, &dt4);

    expect_size("threads-4 prefill token matches", pt4, pt1);
    expect_size("threads-4 decode token matches", dt4, dt1);
    for (i = 0; i < 3; ++i) {
        expect_float("threads-4 prefill logit", logits_prefill_4[i],
                     logits_prefill_1[i]);
        expect_float("threads-4 decode logit", logits_decode_4[i],
                     logits_decode_1[i]);
    }
}

int main(void)
{
    /* Thread pool lifecycle and dispatch */
    test_pool_init_destroy_single();
    test_pool_init_destroy_multi();
    test_pool_init_zero_rejected();
    test_pool_null_safety();
    test_pool_double_destroy();
    test_pool_dispatch_single_thread();
    test_pool_dispatch_multi_thread();
    test_pool_dispatch_total_less_than_threads();
    test_pool_dispatch_zero_total();
    test_pool_dispatch_multiple_rounds();

    /* Runtime thread pool integration */
    test_runtime_init_thread_count_default();
    test_runtime_init_thread_count_multi();
    test_runtime_init_thread_count_zero_rejected();

    /* Parallel forward path correctness */
    test_llama_forward_threads_1_identity();
    test_llama_forward_threads_2_identical();
    test_llama_forward_threads_4_identical();

    if (g_failures != 0) {
        fprintf(stderr, "%d threading test failure(s)\n", g_failures);
        return 1;
    }

    printf("threading tests passed\n");
    return 0;
}
