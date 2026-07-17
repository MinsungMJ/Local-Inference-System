#include "lis/context.h"
#include "lis/checkpoint_digest.h"
#include "lis/cpu_features.h"
#include "lis/dtype.h"
#include "lis/layer_trace.h"
#include "lis/model.h"
#include "lis/status.h"
#include "lis/tensor.h"

#include <stdint.h>
#include <stdio.h>
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

static void test_dtype(void)
{
    size_t size = 0;

    expect_status("f32 size", lis_dtype_size_bytes(LIS_DTYPE_F32, &size),
                  LIS_STATUS_OK);
    expect_size("f32 byte size", size, 4);
    expect_status("bf16 size", lis_dtype_size_bytes(LIS_DTYPE_BF16, &size),
                  LIS_STATUS_OK);
    expect_size("bf16 byte size", size, 2);
    expect_status("invalid dtype size",
                  lis_dtype_size_bytes(LIS_DTYPE_INVALID, &size),
                  LIS_STATUS_UNSUPPORTED);
    expect_status("dtype null out",
                  lis_dtype_size_bytes(LIS_DTYPE_F32, NULL),
                  LIS_STATUS_INVALID_ARGUMENT);
}

static void test_tensor_shape(void)
{
    const size_t dims[] = { 2, 3, 4 };
    const size_t overflow_dims[] = { SIZE_MAX, 2 };
    lis_tensor_shape shape = { 0 };
    size_t count = 0;
    size_t byte_size = 0;

    expect_status("shape make", lis_tensor_shape_make(3, dims, &shape),
                  LIS_STATUS_OK);
    expect_size("shape rank", shape.rank, 3);
    expect_size("shape stride 0", shape.strides[0], 12);
    expect_size("shape stride 1", shape.strides[1], 4);
    expect_size("shape stride 2", shape.strides[2], 1);
    expect_status("element count",
                  lis_tensor_shape_element_count(&shape, &count),
                  LIS_STATUS_OK);
    expect_size("element count value", count, 24);
    expect_status("byte size",
                  lis_tensor_shape_byte_size(&shape, LIS_DTYPE_F16,
                                             &byte_size),
                  LIS_STATUS_OK);
    expect_size("byte size value", byte_size, 48);
    expect_status("shape rank zero",
                  lis_tensor_shape_make(0, dims, &shape),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_status("shape null dims",
                  lis_tensor_shape_make(3, NULL, &shape),
                  LIS_STATUS_INVALID_ARGUMENT);
    {
        const size_t zero_dims[] = { 2, 0 };

        expect_status("shape zero dim",
                      lis_tensor_shape_make(2, zero_dims, &shape),
                      LIS_STATUS_INVALID_ARGUMENT);
    }
    expect_status("shape overflow",
                  lis_tensor_shape_make(2, overflow_dims, &shape),
                  LIS_STATUS_OVERFLOW);
}

static void release_counter(void *data, void *user_data)
{
    int *counter = user_data;

    (void)data;
    ++(*counter);
}

static void test_tensor_ownership(void)
{
    const size_t dims[] = { 2, 2 };
    float data[4] = { 0.0f };
    lis_tensor_shape shape = { 0 };
    lis_tensor_view view = { 0 };
    lis_tensor borrowed = { 0 };
    lis_tensor owned = { 0 };
    int release_count = 0;

    expect_status("shape for tensor", lis_tensor_shape_make(2, dims, &shape),
                  LIS_STATUS_OK);
    expect_status("borrowed view too small",
                  lis_tensor_view_from_borrowed(LIS_DTYPE_F32, &shape, data,
                                                sizeof(data) - 1, &view),
                  LIS_STATUS_SHAPE_MISMATCH);
    expect_status("borrowed view null data",
                  lis_tensor_view_from_borrowed(LIS_DTYPE_F32, &shape, NULL,
                                                sizeof(data), &view),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_status("borrowed tensor",
                  lis_tensor_init_borrowed(LIS_DTYPE_F32, &shape, data,
                                           sizeof(data), &borrowed),
                  LIS_STATUS_OK);
    lis_tensor_destroy(&borrowed);
    expect_size("borrowed destroy no release", (size_t)release_count, 0);

    expect_status("owned tensor",
                  lis_tensor_init_owned(LIS_DTYPE_F32, &shape, data,
                                        sizeof(data), release_counter,
                                        &release_count, &owned),
                  LIS_STATUS_OK);
    lis_tensor_destroy(&owned);
    expect_size("owned destroy release", (size_t)release_count, 1);
}

static lis_model_config valid_llama3_config(void)
{
    lis_model_config config = {
        .family = LIS_MODEL_FAMILY_LLAMA3_DECODER,
        .layer_count = 2,
        .hidden_size = 128,
        .intermediate_size = 256,
        .attention_head_count = 4,
        .kv_head_count = 2,
        .head_dim = 32,
        .vocab_size = 32000,
        .rope_theta = 500000.0f,
        .rms_norm_eps = 1.0e-5f,
        .weight_dtype = LIS_DTYPE_F16,
        .context = lis_context_window_policy_default(8192, 4096),
    };

    return config;
}

static void test_context_policy(void)
{
    lis_context_window_policy policy =
        lis_context_window_policy_default(8192, 4096);
    lis_context_window_policy too_large =
        lis_context_window_policy_default(4096, 8192);
    lis_context_window_policy equal =
        lis_context_window_policy_default(4096, 4096);
    lis_context_window_policy zero_trained =
        lis_context_window_policy_default(0, 8192);
    lis_context_window_policy zero_configured =
        lis_context_window_policy_default(8192, 0);
    lis_context_window_policy bad_mode;
    lis_context_window_policy bad_over_trained;

    expect_status("context policy valid",
                  lis_context_window_policy_validate(&policy),
                  LIS_STATUS_OK);
    expect_status("context request valid",
                  lis_context_window_validate_request(&policy, 4096),
                  LIS_STATUS_OK);
    expect_status("context request too large",
                  lis_context_window_validate_request(&policy, 4097),
                  LIS_STATUS_LIMIT_EXCEEDED);
    expect_status("context request zero",
                  lis_context_window_validate_request(&policy, 0),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_status("configured exceeds trained",
                  lis_context_window_policy_validate(&too_large),
                  LIS_STATUS_LIMIT_EXCEEDED);

    /* configured == trained is accepted (terminal: decode-0) */
    expect_status("configured equals trained",
                  lis_context_window_policy_validate(&equal), LIS_STATUS_OK);

    /* zero trained returns INVALID_ARGUMENT */
    expect_status("zero trained_max_tokens",
                  lis_context_window_policy_validate(&zero_trained),
                  LIS_STATUS_INVALID_ARGUMENT);

    /* zero configured returns INVALID_ARGUMENT */
    expect_status("zero configured_max_tokens",
                  lis_context_window_policy_validate(&zero_configured),
                  LIS_STATUS_INVALID_ARGUMENT);

    /* NULL pointer returns INVALID_ARGUMENT */
    expect_status("null policy",
                  lis_context_window_policy_validate(NULL),
                  LIS_STATUS_INVALID_ARGUMENT);

    /* config_mode other than RUNTIME returns UNSUPPORTED */
    bad_mode = lis_context_window_policy_default(8192, 4096);
    bad_mode.config_mode = (lis_context_config_mode)0;
    expect_status("bad config_mode",
                  lis_context_window_policy_validate(&bad_mode),
                  LIS_STATUS_UNSUPPORTED);

    /* over_trained_policy other than REJECT returns UNSUPPORTED */
    bad_over_trained = lis_context_window_policy_default(8192, 4096);
    bad_over_trained.over_trained_policy =
        (lis_context_over_trained_policy)0;
    expect_status("bad over_trained_policy",
                  lis_context_window_policy_validate(&bad_over_trained),
                  LIS_STATUS_UNSUPPORTED);
}

static void test_model_metadata(void)
{
    lis_model_config config = valid_llama3_config();
    lis_model_metadata metadata = {
        .config = config,
        .support = lis_model_support_envelope_default(),
    };

    expect_status("llama3 config", lis_model_config_validate(&config),
                  LIS_STATUS_OK);
    expect_status("metadata", lis_model_metadata_validate(&metadata),
                  LIS_STATUS_OK);
    expect_size("non eos token",
                (size_t)lis_model_config_token_is_eos(&config, 1), 0);

    config = valid_llama3_config();
    config.eos_token_ids[0] = 2;
    config.eos_token_count = 1;
    expect_status("eos config", lis_model_config_validate(&config),
                  LIS_STATUS_OK);
    expect_size("eos token",
                (size_t)lis_model_config_token_is_eos(&config, 2), 1);
    expect_size("eos null config",
                (size_t)lis_model_config_token_is_eos(NULL, 2), 0);

    config.family = LIS_MODEL_FAMILY_GPT2_DECODER;
    expect_status("future family unsupported",
                  lis_model_config_validate(&config), LIS_STATUS_UNSUPPORTED);

    config = valid_llama3_config();
    config.kv_head_count = config.attention_head_count + 1;
    expect_status("invalid kv heads", lis_model_config_validate(&config),
                  LIS_STATUS_INVALID_ARGUMENT);

    config = valid_llama3_config();
    config.weight_dtype = LIS_DTYPE_INVALID;
    expect_status("invalid weight dtype", lis_model_config_validate(&config),
                  LIS_STATUS_UNSUPPORTED);

    config = valid_llama3_config();
    config.eos_token_ids[0] = config.vocab_size;
    config.eos_token_count = 1;
    expect_status("eos outside vocab", lis_model_config_validate(&config),
                  LIS_STATUS_LIMIT_EXCEEDED);

    metadata.config = valid_llama3_config();
    metadata.support.functional_max_parameters = 1024;
    metadata.support.validation_target_parameters = 2048;
    expect_status("validation envelope over functional",
                  lis_model_metadata_validate(&metadata),
                  LIS_STATUS_LIMIT_EXCEEDED);
}

static void test_cpu_features(void)
{
    const lis_cpu_features *first = lis_cpu_features_get();
    const lis_cpu_features *second = lis_cpu_features_get();

    if (first == NULL || second == NULL) {
        fprintf(stderr, "cpu features: null pointer returned\n");
        ++g_failures;
        return;
    }
    if (first != second) {
        fprintf(stderr, "cpu features: pointer not stable across calls\n");
        ++g_failures;
    }
    if (first->sse2 != second->sse2 || first->avx != second->avx ||
        first->avx2 != second->avx2 || first->fma != second->fma ||
        first->f16c != second->f16c ||
        first->avx512f != second->avx512f ||
        first->avx512vl != second->avx512vl ||
        first->bmi2 != second->bmi2) {
        fprintf(stderr, "cpu features: struct not stable across calls\n");
        ++g_failures;
    }
#if defined(__x86_64__) || defined(__i386__)
    if (first->sse2 != 1) {
        fprintf(stderr,
                "cpu features: expected sse2=1 on x86 build, got %d\n",
                first->sse2);
        ++g_failures;
    }
#endif
    if (first->avx2 != 0 && first->avx2 != 1) {
        fprintf(stderr, "cpu features: avx2 must be 0 or 1\n");
        ++g_failures;
    }
}

static void test_layer_trace_record_growth(void)
{
    lis_layer_trace_record record = {0};
    lis_layer_trace_step step = {0};
    lis_status status;
    size_t i;

    status = lis_layer_trace_record_init(&record, 1);
    expect_status("growth init", status, LIS_STATUS_OK);
    expect_size("growth initial capacity", record.step_capacity, 1);

    for (i = 0; i < LIS_LAYER_TRACE_HARD_MAX; ++i) {
        status = lis_layer_trace_record_append(&record, &step);
        if (status != LIS_STATUS_OK) {
            fprintf(stderr, "growth append %zu: expected OK, got %s\n",
                    i, lis_status_name(status));
            ++g_failures;
            break;
        }
        if (record.step_count == 64) {
            expect_size("growth cap 64", record.step_capacity, 64);
        }
        if (record.step_count == 128) {
            expect_size("growth cap 128", record.step_capacity, 128);
        }
    }
    expect_size("growth hard_max reached", record.step_count,
                LIS_LAYER_TRACE_HARD_MAX);
    expect_size("growth hard_max capacity", record.step_capacity,
                LIS_LAYER_TRACE_HARD_MAX);

    lis_layer_trace_record_destroy(&record);
}

static void test_layer_trace_record_overflow(void)
{
    lis_layer_trace_record record = {0};
    lis_layer_trace_step step = {0};
    lis_status status;
    size_t i;

    status = lis_layer_trace_record_init(&record, 1);
    expect_status("overflow init", status, LIS_STATUS_OK);

    for (i = 0; i < LIS_LAYER_TRACE_HARD_MAX; ++i) {
        step.step = i;
        status = lis_layer_trace_record_append(&record, &step);
        if (status != LIS_STATUS_OK) {
            fprintf(stderr, "overflow append %zu: expected OK, got %s\n",
                    i, lis_status_name(status));
            ++g_failures;
            break;
        }
    }

    status = lis_layer_trace_record_append(&record, &step);
    expect_status("overflow append after max", status, LIS_STATUS_OVERFLOW);
    expect_size("overflow append_failed", (size_t)record.append_failed, 1);

    /* artifact_write with append_failed must return OVERFLOW */
    {
        char dummy;
        lis_layer_trace_artifact artifact = {
            .path = "test_overflow.json",
            .options = (lis_cli_options *)&dummy,
            .model = (lis_loaded_model *)&dummy,
            .binary_fingerprint = {.valid = 1},
            .model_fingerprint = {.valid = 1},
            .config_fingerprint = {.valid = 1},
            .input_fingerprint = {.valid = 1},
            .runtime_fingerprint = {.valid = 1},
            .backend_fingerprint = {.valid = 1},
        };
        status = lis_layer_trace_artifact_write(&artifact, &record);
        expect_status("overflow artifact_write", status, LIS_STATUS_OVERFLOW);
    }

    lis_layer_trace_record_destroy(&record);
}

static void test_checkpoint_digest_vectors(void)
{
    static const struct {
        const char *name;
        const char *role;
        size_t rank;
        size_t shape[2];
        size_t count;
        uint32_t bits[3];
        const char *expected;
    } vectors[] = {
        {
            "finite", "layer_output", 1, {3, 0}, 3,
            {UINT32_C(0x3f800000), UINT32_C(0xc0200000),
             UINT32_C(0x40500000)},
            "f63cac06920e4310fa013b38a233c88b917b7bad77720b212d444c610cb36da4"
        },
        {
            "signed zero", "layer_output", 1, {2, 0}, 2,
            {UINT32_C(0x00000000), UINT32_C(0x80000000), 0},
            "3571b1a3c12497675f5034337d264e82d24d280fd0f93f3fa363e09e338a5a26"
        },
        {
            "infinities", "layer_output", 1, {2, 0}, 2,
            {UINT32_C(0x7f800000), UINT32_C(0xff800000), 0},
            "29d7bbf249f921f0eac64c6a982e0cc9e27ac2f480cb3a503bf3241c40988de0"
        },
        {
            "nan canonical", "layer_output", 1, {3, 0}, 3,
            {UINT32_C(0x7fc00001), UINT32_C(0x7fa12345),
             UINT32_C(0xffc54321)},
            "262ca9bbf40acca6b7f0a510772fd78b1eb1d7f445f4d850135a8811f4ce3445"
        },
        {
            "shape domain", "layer_output", 2, {1, 3}, 3,
            {UINT32_C(0x3f800000), UINT32_C(0xc0200000),
             UINT32_C(0x40500000)},
            "2d77fec8148696842f00a5705b9421af62c03cb2c4365aef02bae5a09fbdc2ba"
        },
        {
            "role domain", "attention_output", 1, {3, 0}, 3,
            {UINT32_C(0x3f800000), UINT32_C(0xc0200000),
             UINT32_C(0x40500000)},
            "f6ba69456253afd7a3c610298a96fd3d597601f449486756fd73473400df6b5a"
        }
    };
    size_t index;

    for (index = 0; index < sizeof(vectors) / sizeof(vectors[0]); ++index) {
        float values[3] = {0};
        lis_checkpoint_digest digest = {{0}, 0};
        char hex[LIS_CHECKPOINT_DIGEST_HEX_SIZE + 1U];
        size_t value_index;

        for (value_index = 0; value_index < vectors[index].count;
             ++value_index) {
            memcpy(values + value_index, vectors[index].bits + value_index,
                   sizeof(values[value_index]));
        }
        expect_status(vectors[index].name,
                      lis_checkpoint_digest_fp32(
                          vectors[index].role, vectors[index].shape,
                          vectors[index].rank, values, vectors[index].count,
                          &digest),
                      LIS_STATUS_OK);
        lis_checkpoint_digest_hex(&digest, hex);
        if (strcmp(hex, vectors[index].expected) != 0) {
            fprintf(stderr, "%s digest: expected %s, got %s\n",
                    vectors[index].name, vectors[index].expected, hex);
            ++g_failures;
        }
    }
    {
        const size_t shape[] = {2};
        const float values[] = {1.0f};
        lis_checkpoint_digest digest = {{0}, 0};

        expect_status("digest shape mismatch",
                      lis_checkpoint_digest_fp32(
                          LIS_CHECKPOINT_DIGEST_ROLE_LAYER_OUTPUT,
                          shape, 1, values, 1, &digest),
                      LIS_STATUS_SHAPE_MISMATCH);
    }
}

static lis_status deterministic_random_source(void *context,
                                              unsigned char *buffer,
                                              size_t size)
{
    size_t index;

    (void)context;
    for (index = 0; index < size; ++index) {
        buffer[index] = (unsigned char)index;
    }
    return LIS_STATUS_OK;
}

static lis_status failing_random_source(void *context,
                                        unsigned char *buffer,
                                        size_t size)
{
    (void)context;
    (void)buffer;
    (void)size;
    return LIS_STATUS_IO;
}

static void test_artifact_set_id_lifecycle(void)
{
    lis_artifact_set_id deterministic = {{0}, 0};
    lis_artifact_set_id failed = {{0}, 0};
    lis_artifact_set_id first = {{0}, 0};
    lis_artifact_set_id second = {{0}, 0};

    expect_status("artifact set deterministic",
                  lis_artifact_set_id_generate_with_source(
                      &deterministic, deterministic_random_source, NULL),
                  LIS_STATUS_OK);
    if (!deterministic.valid ||
        strcmp(deterministic.value,
               "aset1:000102030405060708090a0b0c0d0e0f") != 0) {
        fprintf(stderr, "artifact set deterministic bytes changed: %s\n",
                deterministic.value);
        ++g_failures;
    }
    expect_status("artifact set failure",
                  lis_artifact_set_id_generate_with_source(
                      &failed, failing_random_source, NULL),
                  LIS_STATUS_IO);
    expect_size("artifact set failure invalid", (size_t)failed.valid, 0);
    expect_size("artifact set failure empty", strlen(failed.value), 0);

    expect_status("artifact set os first",
                  lis_artifact_set_id_generate(&first), LIS_STATUS_OK);
    expect_status("artifact set os second",
                  lis_artifact_set_id_generate(&second), LIS_STATUS_OK);
    expect_size("artifact set os first length", strlen(first.value),
                LIS_ARTIFACT_SET_ID_LEN);
    expect_size("artifact set os second length", strlen(second.value),
                LIS_ARTIFACT_SET_ID_LEN);
    if (strcmp(first.value, second.value) == 0) {
        fprintf(stderr, "artifact set bounded sample unexpectedly collided\n");
        ++g_failures;
    }
}

int main(void)
{
    test_dtype();
    test_tensor_shape();
    test_tensor_ownership();
    test_context_policy();
    test_model_metadata();
    test_cpu_features();
    test_layer_trace_record_growth();
    test_layer_trace_record_overflow();
    test_checkpoint_digest_vectors();
    test_artifact_set_id_lifecycle();

    if (g_failures != 0) {
        fprintf(stderr, "%d core test failure(s)\n", g_failures);
        return 1;
    }

    return 0;
}
