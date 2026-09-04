#include "lis/context.h"
#include "lis/checkpoint_digest.h"
#include "lis/cpu_features.h"
#include "lis/dtype.h"
#include "lis/intra_layer_trace.h"
#include "lis/layer_trace.h"
#include "lis/model.h"
#include "lis/status.h"
#include "lis/tensor.h"

#include <float.h>
#include <inttypes.h>
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

static void test_generic_sha256_vectors(void)
{
    static const struct {
        const char *input;
        const char *expected;
    } vectors[] = {
        {
            "",
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        },
        {
            "[0,1]",
            "463f2998327eb3a694145e6014444480b2235be84aa6cfd57871cc64f1cd816c"
        }
    };
    size_t index;

    for (index = 0U; index < sizeof(vectors) / sizeof(vectors[0]); ++index) {
        lis_checkpoint_digest digest = {{0}, 0};
        char hex[LIS_CHECKPOINT_DIGEST_HEX_SIZE + 1U];

        expect_status("generic sha256 vector",
                      lis_sha256_digest_bytes(vectors[index].input,
                                              strlen(vectors[index].input),
                                              &digest),
                      LIS_STATUS_OK);
        lis_checkpoint_digest_hex(&digest, hex);
        if (strcmp(hex, vectors[index].expected) != 0) {
            fprintf(stderr, "generic sha256: expected %s, got %s\n",
                    vectors[index].expected, hex);
            ++g_failures;
        }
    }
    expect_status("generic sha256 null input",
                  lis_sha256_digest_bytes(NULL, 1U, NULL),
                  LIS_STATUS_INVALID_ARGUMENT);
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

static lis_layer_trace_step make_layer_output_step(size_t step_index,
                                                   size_t layer_index,
                                                   float value)
{
    lis_layer_trace_step step = {0};

    step.step = step_index;
    step.rank = 1;
    step.shape[0] = 1;
    snprintf(step.phase, sizeof(step.phase), "%s", "decode");
    snprintf(step.name, sizeof(step.name), "layer.%zu.output", layer_index);
    step.min = value;
    step.max = value;
    step.mean = value;
    step.l2 = value < 0.0f ? -value : value;
    expect_status("make layer output digest",
                  lis_layer_trace_step_set_layer_output(
                      &step, layer_index, &value, 1),
                  LIS_STATUS_OK);
    return step;
}

static void test_layer_trace_coordinate_guards(void)
{
    lis_layer_trace_record disabled = {0};
    lis_layer_trace_record duplicate = {0};
    lis_layer_trace_record nonmonotonic = {0};
    lis_layer_trace_step layer_zero = make_layer_output_step(3, 0, 1.0f);
    lis_layer_trace_step layer_two = make_layer_output_step(3, 2, 2.0f);

    expect_status("digest disabled record init",
                  lis_layer_trace_record_init(&disabled, 1), LIS_STATUS_OK);
    expect_size("digest disabled element visits",
                disabled.digest_element_visits, 0);
    lis_layer_trace_record_destroy(&disabled);

    expect_status("duplicate init",
                  lis_layer_trace_record_init(&duplicate, 2), LIS_STATUS_OK);
    expect_status("duplicate layout",
                  lis_layer_trace_record_configure_llama_layout(
                      &duplicate, 3, 4),
                  LIS_STATUS_OK);
    expect_status("duplicate first append",
                  lis_layer_trace_record_append(&duplicate, &layer_zero),
                  LIS_STATUS_OK);
    expect_size("digest enabled element visits",
                duplicate.digest_element_visits, 1);
    expect_status("duplicate rejected",
                  lis_layer_trace_record_append(&duplicate, &layer_zero),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_size("duplicate sticky failure", (size_t)duplicate.append_failed, 1);
    expect_size("duplicate rejection adds no digest visits",
                duplicate.digest_element_visits, 1);
    lis_layer_trace_record_destroy(&duplicate);

    expect_status("nonmonotonic init",
                  lis_layer_trace_record_init(&nonmonotonic, 2),
                  LIS_STATUS_OK);
    expect_status("nonmonotonic layout",
                  lis_layer_trace_record_configure_llama_layout(
                      &nonmonotonic, 3, 4),
                  LIS_STATUS_OK);
    expect_status("nonmonotonic first append",
                  lis_layer_trace_record_append(&nonmonotonic, &layer_two),
                  LIS_STATUS_OK);
    expect_status("nonmonotonic rejected",
                  lis_layer_trace_record_append(&nonmonotonic, &layer_zero),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_size("nonmonotonic sticky failure",
                (size_t)nonmonotonic.append_failed, 1);
    lis_layer_trace_record_destroy(&nonmonotonic);
}

/* ---------------------------------------------------------------------------
 * P4-4 intra-layer producer layout tests.
 *
 * All helpers in this section are prefixed `intra_` and all test entry points
 * `test_intra_layer_*`. They exercise the module's authoritative stage table
 * rather than any second in-test taxonomy: the golden arrays below are literals
 * transcribed from the frozen external authority
 * tools/test_fixtures/intra_layer_localization/pass4_contract.json, in the same
 * spirit as the golden digest vectors above, so that an accidental edit to the
 * C table fails here.
 * ------------------------------------------------------------------------- */

static const char *const intra_frozen_stage_ids[] = {
    "layer_input",
    "attention_norm_output",
    "query_projection_output",
    "key_projection_output",
    "value_projection_output",
    "rope_query_output",
    "rope_key_output",
    "attention_scores",
    "attention_probabilities",
    "attention_context",
    "attention_output_projection",
    "post_attention_residual",
    "mlp_norm_output",
    "mlp_gate_projection",
    "mlp_up_projection",
    "mlp_gated_activation",
    "mlp_down_projection"
};

static void intra_expect_string(const char *name, const char *actual,
                                const char *expected)
{
    if (actual == NULL || expected == NULL || strcmp(actual, expected) != 0) {
        fprintf(stderr, "%s: expected \"%s\", got \"%s\"\n", name,
                expected != NULL ? expected : "(null)",
                actual != NULL ? actual : "(null)");
        ++g_failures;
    }
}

static void intra_expect_null(const char *name, const void *pointer)
{
    if (pointer != NULL) {
        fprintf(stderr, "%s: expected NULL pointer\n", name);
        ++g_failures;
    }
}

static void intra_expect_state(const char *name,
                               const lis_intra_layer_trace_record *record,
                               lis_intra_layer_record_state expected)
{
    lis_intra_layer_record_state actual =
        lis_intra_layer_record_get_state(record);

    if (actual != expected) {
        fprintf(stderr, "%s: expected state %d, got %d\n", name,
                (int)expected, (int)actual);
        ++g_failures;
    }
}

static void test_intra_layer_stage_taxonomy(void)
{
    size_t index;

    expect_size("intra frozen id count",
                sizeof(intra_frozen_stage_ids) /
                    sizeof(intra_frozen_stage_ids[0]),
                LIS_INTRA_LAYER_STAGE_COUNT);
    expect_size("intra stage count", (size_t)LIS_INTRA_LAYER_STAGE_COUNT, 17U);

    for (index = 0; index < LIS_INTRA_LAYER_STAGE_COUNT; ++index) {
        const lis_intra_layer_stage_info *info =
            lis_intra_layer_stage_lookup(index);

        if (info == NULL) {
            fprintf(stderr, "intra stage %zu: unexpected NULL\n", index);
            ++g_failures;
            continue;
        }
        expect_size("intra stage order", info->stage_order, index);
        intra_expect_string("intra stage id", info->stage_id,
                            intra_frozen_stage_ids[index]);
        /* stage_id == tensor_role is a frozen v1 identity. */
        intra_expect_string("intra tensor role", info->tensor_role,
                            intra_frozen_stage_ids[index]);
        if (info->public_name == NULL || info->public_name[0] == '\0') {
            fprintf(stderr, "intra stage %zu: empty public name\n", index);
            ++g_failures;
        }
        /* layer_output is the inherited parent boundary, not a local stage. */
        if (strcmp(info->stage_id, "layer_output") == 0 ||
            strcmp(info->tensor_role, "layer_output") == 0) {
            fprintf(stderr, "intra stage %zu: layer_output is not a local "
                            "stage\n", index);
            ++g_failures;
        }
    }

    /* Enum identifiers agree with the table order they index. */
    expect_size("intra enum layer_input",
                (size_t)LIS_INTRA_LAYER_STAGE_LAYER_INPUT, 0U);
    expect_size("intra enum attention_scores",
                (size_t)LIS_INTRA_LAYER_STAGE_ATTENTION_SCORES, 7U);
    expect_size("intra enum mlp_down_projection",
                (size_t)LIS_INTRA_LAYER_STAGE_MLP_DOWN_PROJECTION, 16U);
}

static void test_intra_layer_stage_lookup_rejects_unknown(void)
{
    intra_expect_null("intra lookup 17",
                      lis_intra_layer_stage_lookup(LIS_INTRA_LAYER_STAGE_COUNT));
    intra_expect_null("intra lookup size max",
                      lis_intra_layer_stage_lookup(SIZE_MAX));
    intra_expect_null("intra lookup wrapped negative",
                      lis_intra_layer_stage_lookup((size_t)-1));
    intra_expect_null(
        "intra lookup cast enum overflow",
        lis_intra_layer_stage_lookup(
            (size_t)(lis_intra_layer_stage)LIS_INTRA_LAYER_STAGE_COUNT));
}

static void test_intra_layer_record_configure_guards(void)
{
    lis_intra_layer_trace_record record;
    char oversized[LIS_INTRA_LAYER_IDENTIFIER_MAX + 2U];
    char control[8];

    memset(oversized, 'a', sizeof(oversized) - 1U);
    oversized[sizeof(oversized) - 1U] = '\0';
    memcpy(control, "bf16\tx", 7U);

    expect_status("intra configure init",
                  lis_intra_layer_record_init(&record), LIS_STATUS_OK);
    intra_expect_state("intra configure initial state", &record,
                       LIS_INTRA_LAYER_RECORD_UNINITIALIZED);

    expect_status("intra configure step zero",
                  lis_intra_layer_record_configure(&record, 0U, 1U, 4U, 0U,
                                                   "bf16"),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_status("intra configure layer out of range",
                  lis_intra_layer_record_configure(&record, 3U, 4U, 4U, 0U,
                                                   "bf16"),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_status("intra configure zero layer count",
                  lis_intra_layer_record_configure(&record, 3U, 0U, 0U, 0U,
                                                   "bf16"),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_status("intra configure null precision path",
                  lis_intra_layer_record_configure(&record, 3U, 1U, 4U, 0U,
                                                   NULL),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_status("intra configure empty precision path",
                  lis_intra_layer_record_configure(&record, 3U, 1U, 4U, 0U,
                                                   ""),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_status("intra configure oversized precision path",
                  lis_intra_layer_record_configure(&record, 3U, 1U, 4U, 0U,
                                                   oversized),
                  LIS_STATUS_FORMAT);
    expect_status("intra configure control byte precision path",
                  lis_intra_layer_record_configure(&record, 3U, 1U, 4U, 0U,
                                                   control),
                  LIS_STATUS_INVALID_ARGUMENT);
    intra_expect_state("intra configure stays uninitialized", &record,
                       LIS_INTRA_LAYER_RECORD_UNINITIALIZED);

    expect_status("intra configure accepted",
                  lis_intra_layer_record_configure(&record, 3U, 1U, 4U, 5U,
                                                   "bf16"),
                  LIS_STATUS_OK);
    intra_expect_state("intra configure active", &record,
                       LIS_INTRA_LAYER_RECORD_ACTIVE);
    expect_status("intra configure twice",
                  lis_intra_layer_record_configure(&record, 3U, 1U, 4U, 5U,
                                                   "bf16"),
                  LIS_STATUS_BAD_STATE);
    expect_size("intra configure step", record.runtime_checkpoint_step, 3U);
    expect_size("intra configure target layer", record.target_layer, 1U);
    expect_size("intra configure total layers", record.total_layer_count, 4U);
    expect_size("intra configure token position", record.token_position, 5U);
    intra_expect_string("intra configure precision path",
                        record.precision_path, "bf16");

    /* A boundary-length identifier is accepted, not truncated. */
    expect_status("intra configure boundary init",
                  lis_intra_layer_record_init(&record), LIS_STATUS_OK);
    oversized[LIS_INTRA_LAYER_IDENTIFIER_MAX] = '\0';
    expect_status("intra configure boundary length",
                  lis_intra_layer_record_configure(&record, 1U, 0U, 1U, 0U,
                                                   oversized),
                  LIS_STATUS_OK);
    expect_size("intra configure boundary stored",
                strlen(record.precision_path),
                (size_t)LIS_INTRA_LAYER_IDENTIFIER_MAX);
    lis_intra_layer_record_destroy(&record);
}

static void test_intra_layer_fp32_view_validation(void)
{
    static const float storage[64] = {0.0f};
    lis_intra_layer_fp32_view view;

    /* Valid contiguous [2,3] view. */
    memset(&view, 0, sizeof(view));
    view.data = storage;
    view.rank = 2U;
    view.shape[0] = 2U;
    view.shape[1] = 3U;
    view.element_strides[0] = 3U;
    view.element_strides[1] = 1U;
    view.logical_element_count = 6U;
    view.physical_element_count = 6U;
    expect_status("intra view contiguous",
                  lis_intra_layer_fp32_view_validate(&view), LIS_STATUS_OK);

    /* Valid strided [A,S] view over a larger physical span. */
    view.shape[0] = 4U;
    view.shape[1] = 3U;
    view.element_strides[0] = 16U;
    view.element_strides[1] = 1U;
    view.logical_element_count = 12U;
    view.physical_element_count = 64U;
    expect_status("intra view strided",
                  lis_intra_layer_fp32_view_validate(&view), LIS_STATUS_OK);

    /* max_offset == physical_element_count - 1 is the inclusive boundary. */
    view.physical_element_count = 51U;
    expect_status("intra view span boundary inclusive",
                  lis_intra_layer_fp32_view_validate(&view), LIS_STATUS_OK);
    /* One-past-span is rejected. */
    view.physical_element_count = 50U;
    expect_status("intra view span one past",
                  lis_intra_layer_fp32_view_validate(&view),
                  LIS_STATUS_INVALID_ARGUMENT);
    view.physical_element_count = 8U;
    expect_status("intra view span exceeded",
                  lis_intra_layer_fp32_view_validate(&view),
                  LIS_STATUS_INVALID_ARGUMENT);
    view.physical_element_count = 64U;

    expect_status("intra view null",
                  lis_intra_layer_fp32_view_validate(NULL),
                  LIS_STATUS_INVALID_ARGUMENT);
    view.data = NULL;
    expect_status("intra view null data",
                  lis_intra_layer_fp32_view_validate(&view),
                  LIS_STATUS_INVALID_ARGUMENT);
    view.data = storage;

    view.rank = 0U;
    expect_status("intra view rank zero",
                  lis_intra_layer_fp32_view_validate(&view),
                  LIS_STATUS_UNSUPPORTED_SHAPE);
    view.rank = LIS_INTRA_LAYER_MAX_RANK + 1U;
    expect_status("intra view rank too large",
                  lis_intra_layer_fp32_view_validate(&view),
                  LIS_STATUS_UNSUPPORTED_SHAPE);
    view.rank = 2U;

    view.shape[1] = 0U;
    expect_status("intra view zero dimension",
                  lis_intra_layer_fp32_view_validate(&view),
                  LIS_STATUS_UNSUPPORTED_SHAPE);
    view.shape[1] = 3U;

    view.logical_element_count = 11U;
    expect_status("intra view element count mismatch",
                  lis_intra_layer_fp32_view_validate(&view),
                  LIS_STATUS_SHAPE_MISMATCH);
    view.logical_element_count = 12U;

    view.element_strides[1] = 0U;
    expect_status("intra view zero stride",
                  lis_intra_layer_fp32_view_validate(&view),
                  LIS_STATUS_INVALID_ARGUMENT);
    view.element_strides[1] = 1U;

    /* Shape product overflow is detected before any span arithmetic. */
    view.shape[0] = SIZE_MAX;
    view.shape[1] = 2U;
    expect_status("intra view shape product overflow",
                  lis_intra_layer_fp32_view_validate(&view),
                  LIS_STATUS_OVERFLOW);
    view.shape[0] = 4U;
    view.shape[1] = 3U;

    /* Stride span overflow. */
    view.element_strides[0] = SIZE_MAX;
    expect_status("intra view span overflow",
                  lis_intra_layer_fp32_view_validate(&view),
                  LIS_STATUS_OVERFLOW);
    view.element_strides[0] = 16U;

    view.physical_element_count = 0U;
    expect_status("intra view empty physical span",
                  lis_intra_layer_fp32_view_validate(&view),
                  LIS_STATUS_INVALID_ARGUMENT);
}

static lis_intra_layer_observation intra_make_observation(
    lis_intra_layer_stage stage,
    size_t step,
    size_t layer,
    size_t token)
{
    lis_intra_layer_observation observation;
    size_t index;

    memset(&observation, 0, sizeof(observation));
    observation.stage = stage;
    observation.phase = LIS_INTRA_LAYER_PHASE_DECODE;
    observation.runtime_checkpoint_step = step;
    observation.layer_index = layer;
    observation.token_position = token;
    observation.batch_index = 0U;
    observation.sequence_index = 0U;
    observation.stage_order = (size_t)stage;
    observation.execution_ordinal = (size_t)stage;
    observation.rank = 1U;
    observation.shape[0] = 4U;
    observation.element_count = 4U;
    observation.min = -1.0f;
    observation.max = 2.0f;
    observation.mean = 0.5f;
    observation.l2 = 2.5f;
    observation.nan = 0;
    observation.inf = 0;
    /* A distinctive caller-supplied digest. P4-4 carries these bytes; it never
     * derives them from any payload. */
    observation.digest.valid = 1;
    for (index = 0; index < LIS_CHECKPOINT_DIGEST_SIZE; ++index) {
        observation.digest.bytes[index] =
            (unsigned char)(index + (size_t)stage);
    }
    return observation;
}

/* ---------------------------------------------------------------------------
 * P4-5 contextual/strided digest tests.
 *
 * The literals below are transcribed from the frozen P4-1 JSON fixture. Three
 * of its 16 positive encoder vectors use non-producer phases ("prefill", "bc",
 * and "c"); the P4-4 C producer deliberately cannot represent those as valid
 * observations. The 13 decode vectors are therefore the complete positive C
 * API parity set, while the Python contract suite retains all 16 encoder
 * vectors and this suite separately pins non-decode rejection.
 * ------------------------------------------------------------------------- */

static void intra_expect_digest_invalid_zero(const char *name,
                                             const lis_checkpoint_digest *digest)
{
    static const unsigned char zero[LIS_CHECKPOINT_DIGEST_SIZE] = {0U};

    if (digest->valid != 0 ||
        memcmp(digest->bytes, zero, sizeof(digest->bytes)) != 0) {
        fprintf(stderr, "%s: failed digest output was not zero/invalid\n",
                name);
        ++g_failures;
    }
}

static void intra_expect_digest_status(
    const char *name,
    const lis_intra_layer_trace_record *record,
    const lis_intra_layer_observation *observation,
    const lis_intra_layer_fp32_view *view,
    lis_status expected)
{
    lis_checkpoint_digest digest;
    lis_status status;

    memset(&digest, 0xa5, sizeof(digest));
    status = lis_intra_layer_checkpoint_digest_fp32(
        record, observation, view, &digest);
    expect_status(name, status, expected);
    if (expected != LIS_STATUS_OK) {
        intra_expect_digest_invalid_zero(name, &digest);
    }
}

static void intra_digest_setup_base(
    lis_intra_layer_trace_record *record,
    lis_intra_layer_observation *observation,
    lis_intra_layer_fp32_view *view,
    float storage[6])
{
    static const uint32_t bits[4] = {
        UINT32_C(0x3f800000), UINT32_C(0xc0200000),
        UINT32_C(0x00000001), UINT32_C(0x7f7fffff)
    };
    size_t index;

    expect_status("intra digest base init",
                  lis_intra_layer_record_init(record), LIS_STATUS_OK);
    expect_status("intra digest base configure",
                  lis_intra_layer_record_configure(record, 3U, 8U, 9U, 11U,
                                                   "f32"),
                  LIS_STATUS_OK);
    *observation = intra_make_observation(
        LIS_INTRA_LAYER_STAGE_MLP_GATE_PROJECTION, 3U, 8U, 11U);
    observation->rank = 2U;
    observation->shape[0] = 2U;
    observation->shape[1] = 2U;
    observation->element_count = 4U;
    memset(storage, 0, sizeof(float) * 6U);
    for (index = 0U; index < 4U; ++index) {
        memcpy(storage + index, bits + index, sizeof(storage[index]));
    }
    memset(view, 0, sizeof(*view));
    view->data = storage;
    view->rank = 2U;
    view->shape[0] = 2U;
    view->shape[1] = 2U;
    view->element_strides[0] = 2U;
    view->element_strides[1] = 1U;
    view->logical_element_count = 4U;
    view->physical_element_count = 4U;
}

static void test_intra_layer_checkpoint_digest_vectors(void)
{
    static const struct {
        const char *name;
        size_t step;
        size_t layer;
        size_t stage_order;
        size_t token;
        const char *precision_path;
        size_t rank;
        size_t shape[LIS_INTRA_LAYER_MAX_RANK];
        size_t strides[LIS_INTRA_LAYER_MAX_RANK];
        size_t physical_count;
        uint32_t physical_bits[6];
        const char *expected;
    } vectors[] = {
        {
            "finite_row_major_base", 3U, 8U, 13U, 11U, "f32", 2U,
            {2U, 2U}, {2U, 1U}, 4U,
            {UINT32_C(0x3f800000), UINT32_C(0xc0200000),
             UINT32_C(0x00000001), UINT32_C(0x7f7fffff)},
            "b87b96f63353fb16d193180220f2cca8e7c906f7cb88ff90bd82e09984f8f2fd"
        },
        {
            "positive_zero", 3U, 8U, 13U, 11U, "f32", 1U,
            {1U}, {1U}, 1U, {UINT32_C(0x00000000)},
            "8bac4d9995a4fdf2671675161ee440bea5bbfb10f0af175735e1b74a61a08b5e"
        },
        {
            "negative_zero", 3U, 8U, 13U, 11U, "f32", 1U,
            {1U}, {1U}, 1U, {UINT32_C(0x80000000)},
            "d6b7b9a12888541a3754318df1e23c4d2535e701167e8bf49788a378e29075bc"
        },
        {
            "infinities", 3U, 8U, 13U, 11U, "f32", 1U,
            {2U}, {1U}, 2U,
            {UINT32_C(0x7f800000), UINT32_C(0xff800000)},
            "8b28af0d4e191488f74223c251b89beceb94995c5d382862bad5649358aaef14"
        },
        {
            "canonical_nans", 3U, 8U, 13U, 11U, "f32", 1U,
            {4U}, {1U}, 4U,
            {UINT32_C(0x7fc00001), UINT32_C(0xffc12345),
             UINT32_C(0x7fffffff), UINT32_C(0x7f800001)},
            "b3297b811f9c0d8c9601152e2ff9f933e982274dc4942e5547232521a68a53c3"
        },
        {
            "shape_flat", 3U, 8U, 13U, 11U, "f32", 1U,
            {4U}, {1U}, 4U,
            {UINT32_C(0x3f800000), UINT32_C(0xc0200000),
             UINT32_C(0x00000001), UINT32_C(0x7f7fffff)},
            "0d0d459aea4398cbebc045e118813f32e08c2b39c166aa297e7fdbbae114e682"
        },
        {
            "stage_and_role_changed", 3U, 8U, 1U, 11U, "f32", 2U,
            {2U, 2U}, {2U, 1U}, 4U,
            {UINT32_C(0x3f800000), UINT32_C(0xc0200000),
             UINT32_C(0x00000001), UINT32_C(0x7f7fffff)},
            "c8cc44cf90c8003d1d031c4e896a7878b133ff6a87a510935229ee7b784f0e70"
        },
        {
            "layer_changed", 3U, 9U, 13U, 11U, "f32", 2U,
            {2U, 2U}, {2U, 1U}, 4U,
            {UINT32_C(0x3f800000), UINT32_C(0xc0200000),
             UINT32_C(0x00000001), UINT32_C(0x7f7fffff)},
            "8ffde18efea277b64a305797d1e28ea8760335ef41d54f4c112d11091ad1bec3"
        },
        {
            "step_changed", 4U, 8U, 13U, 11U, "f32", 2U,
            {2U, 2U}, {2U, 1U}, 4U,
            {UINT32_C(0x3f800000), UINT32_C(0xc0200000),
             UINT32_C(0x00000001), UINT32_C(0x7f7fffff)},
            "4887ed77898f20c1bf95b05301b6022c72e79984c8694588092ac404366d5d58"
        },
        {
            "token_position_changed", 3U, 8U, 13U, 12U, "f32", 2U,
            {2U, 2U}, {2U, 1U}, 4U,
            {UINT32_C(0x3f800000), UINT32_C(0xc0200000),
             UINT32_C(0x00000001), UINT32_C(0x7f7fffff)},
            "b746e5293168f5723238a7a0a9754f7fe7d874a08c5b171c3feea7e24c73764a"
        },
        {
            "precision_path_changed", 3U, 8U, 13U, 11U, "bf16", 2U,
            {2U, 2U}, {2U, 1U}, 4U,
            {UINT32_C(0x3f800000), UINT32_C(0xc0200000),
             UINT32_C(0x00000001), UINT32_C(0x7f7fffff)},
            "4c6c99a3ecadaff8fafa6f93f06ef2518811fdd48a18aba8096412471f3d4a9f"
        },
        {
            "logical_order_changed", 3U, 8U, 13U, 11U, "f32", 2U,
            {2U, 2U}, {2U, 1U}, 4U,
            {UINT32_C(0x3f800000), UINT32_C(0x00000001),
             UINT32_C(0xc0200000), UINT32_C(0x7f7fffff)},
            "a1bdac70f42b0d255608b2ddba9fcab7436a1acbc879d70eb9fe68a5fc041831"
        },
        {
            "strided_logical_equivalent", 3U, 8U, 13U, 11U, "f32", 2U,
            {2U, 2U}, {3U, 1U}, 6U,
            {UINT32_C(0x3f800000), UINT32_C(0xc0200000),
             UINT32_C(0xdeadbeef), UINT32_C(0x00000001),
             UINT32_C(0x7f7fffff), UINT32_C(0xdeadbeef)},
            "b87b96f63353fb16d193180220f2cca8e7c906f7cb88ff90bd82e09984f8f2fd"
        }
    };
    size_t vector_index;

    expect_size("intra digest C vector count",
                sizeof(vectors) / sizeof(vectors[0]), 13U);
    for (vector_index = 0U;
         vector_index < sizeof(vectors) / sizeof(vectors[0]);
         ++vector_index) {
        lis_intra_layer_trace_record record;
        lis_intra_layer_trace_record record_before;
        lis_intra_layer_observation observation;
        lis_intra_layer_observation observation_before;
        lis_intra_layer_fp32_view view;
        lis_intra_layer_fp32_view view_before;
        lis_checkpoint_digest digest;
        float storage[6] = {0.0f};
        float storage_before[6];
        char hex[LIS_CHECKPOINT_DIGEST_HEX_SIZE + 1U];
        size_t logical_count = 1U;
        size_t index;

        expect_status(vectors[vector_index].name,
                      lis_intra_layer_record_init(&record), LIS_STATUS_OK);
        expect_status(vectors[vector_index].name,
                      lis_intra_layer_record_configure(
                          &record, vectors[vector_index].step,
                          vectors[vector_index].layer,
                          vectors[vector_index].layer + 1U,
                          vectors[vector_index].token,
                          vectors[vector_index].precision_path),
                      LIS_STATUS_OK);
        observation = intra_make_observation(
            (lis_intra_layer_stage)vectors[vector_index].stage_order,
            vectors[vector_index].step, vectors[vector_index].layer,
            vectors[vector_index].token);
        observation.rank = vectors[vector_index].rank;
        memset(observation.shape, 0, sizeof(observation.shape));
        for (index = 0U; index < observation.rank; ++index) {
            observation.shape[index] = vectors[vector_index].shape[index];
            logical_count *= observation.shape[index];
        }
        observation.element_count = logical_count;
        for (index = 0U; index < vectors[vector_index].physical_count;
             ++index) {
            memcpy(storage + index,
                   vectors[vector_index].physical_bits + index,
                   sizeof(storage[index]));
        }
        memset(&view, 0, sizeof(view));
        view.data = storage;
        view.rank = observation.rank;
        memcpy(view.shape, observation.shape, sizeof(view.shape));
        memcpy(view.element_strides, vectors[vector_index].strides,
               sizeof(view.element_strides));
        view.logical_element_count = logical_count;
        view.physical_element_count = vectors[vector_index].physical_count;

        record_before = record;
        observation_before = observation;
        view_before = view;
        memcpy(storage_before, storage, sizeof(storage));
        memset(&digest, 0xa5, sizeof(digest));
        expect_status(vectors[vector_index].name,
                      lis_intra_layer_checkpoint_digest_fp32(
                          &record, &observation, &view, &digest),
                      LIS_STATUS_OK);
        expect_size(vectors[vector_index].name, (size_t)digest.valid, 1U);
        lis_checkpoint_digest_hex(&digest, hex);
        intra_expect_string(vectors[vector_index].name, hex,
                            vectors[vector_index].expected);
        if (memcmp(&record, &record_before, sizeof(record)) != 0 ||
            memcmp(&observation, &observation_before,
                   sizeof(observation)) != 0 ||
            memcmp(&view, &view_before, sizeof(view)) != 0 ||
            memcmp(storage, storage_before, sizeof(storage)) != 0) {
            fprintf(stderr, "%s: digest mutated an input\n",
                    vectors[vector_index].name);
            ++g_failures;
        }
    }
}

static void test_intra_layer_checkpoint_digest_guards(void)
{
    static const struct {
        const char *name;
        const unsigned char bytes[5];
        size_t length;
    } invalid_utf8[] = {
        {"intra digest utf8 overlong", {0xe0U, 0x80U, 0x80U}, 3U},
        {"intra digest utf8 surrogate", {0xedU, 0xa0U, 0x80U}, 3U},
        {"intra digest utf8 above max", {0xf4U, 0x90U, 0x80U, 0x80U}, 4U},
        {"intra digest utf8 truncated", {0xe2U, 0x82U}, 2U},
        {"intra digest utf8 bad continuation", {0xe2U, 0x28U, 0xa1U}, 3U},
        {"intra digest utf8 invalid lead", {0xf5U, 0x80U, 0x80U, 0x80U}, 4U}
    };
    lis_intra_layer_trace_record record;
    lis_intra_layer_trace_record record_before;
    lis_intra_layer_observation observation;
    lis_intra_layer_fp32_view view;
    float storage[6];
    size_t utf8_index;

    intra_digest_setup_base(&record, &observation, &view, storage);
    expect_status("intra digest null output",
                  lis_intra_layer_checkpoint_digest_fp32(
                      &record, &observation, &view, NULL),
                  LIS_STATUS_INVALID_ARGUMENT);
    intra_expect_digest_status("intra digest null record", NULL,
                               &observation, &view,
                               LIS_STATUS_INVALID_ARGUMENT);
    intra_expect_digest_status("intra digest null observation", &record,
                               NULL, &view, LIS_STATUS_INVALID_ARGUMENT);
    intra_expect_digest_status("intra digest null view", &record,
                               &observation, NULL,
                               LIS_STATUS_INVALID_ARGUMENT);

    intra_digest_setup_base(&record, &observation, &view, storage);
    record.state = LIS_INTRA_LAYER_RECORD_READY;
    intra_expect_digest_status("intra digest requires active", &record,
                               &observation, &view, LIS_STATUS_BAD_STATE);

    intra_digest_setup_base(&record, &observation, &view, storage);
    record_before = record;
    observation.phase = (lis_intra_layer_phase)2;
    intra_expect_digest_status("intra digest prefill rejected", &record,
                               &observation, &view, LIS_STATUS_UNSUPPORTED);
    if (memcmp(&record, &record_before, sizeof(record)) != 0) {
        fprintf(stderr, "intra digest failure mutated record\n");
        ++g_failures;
    }

    intra_digest_setup_base(&record, &observation, &view, storage);
    observation.stage = (lis_intra_layer_stage)LIS_INTRA_LAYER_STAGE_COUNT;
    intra_expect_digest_status("intra digest unknown stage", &record,
                               &observation, &view,
                               LIS_STATUS_INVALID_ARGUMENT);

#define INTRA_DIGEST_COORDINATE_FAILURE(field, value, label)                 \
    do {                                                                     \
        intra_digest_setup_base(&record, &observation, &view, storage);      \
        observation.field = (value);                                         \
        intra_expect_digest_status((label), &record, &observation, &view,    \
                                   LIS_STATUS_INVALID_ARGUMENT);             \
    } while (0)

    INTRA_DIGEST_COORDINATE_FAILURE(runtime_checkpoint_step, 4U,
                                    "intra digest step mismatch");
    INTRA_DIGEST_COORDINATE_FAILURE(layer_index, 7U,
                                    "intra digest layer mismatch");
    INTRA_DIGEST_COORDINATE_FAILURE(token_position, 12U,
                                    "intra digest token mismatch");
    INTRA_DIGEST_COORDINATE_FAILURE(batch_index, 1U,
                                    "intra digest batch nonzero");
    INTRA_DIGEST_COORDINATE_FAILURE(sequence_index, 1U,
                                    "intra digest sequence nonzero");
    INTRA_DIGEST_COORDINATE_FAILURE(stage_order, 12U,
                                    "intra digest stage order mismatch");
    INTRA_DIGEST_COORDINATE_FAILURE(execution_ordinal, 12U,
                                    "intra digest ordinal mismatch");
#undef INTRA_DIGEST_COORDINATE_FAILURE

    intra_digest_setup_base(&record, &observation, &view, storage);
    observation.rank = 0U;
    intra_expect_digest_status("intra digest rank zero", &record,
                               &observation, &view,
                               LIS_STATUS_UNSUPPORTED_SHAPE);
    intra_digest_setup_base(&record, &observation, &view, storage);
    observation.rank = LIS_INTRA_LAYER_MAX_RANK + 1U;
    intra_expect_digest_status("intra digest rank too large", &record,
                               &observation, &view,
                               LIS_STATUS_UNSUPPORTED_SHAPE);
    intra_digest_setup_base(&record, &observation, &view, storage);
    observation.shape[1] = 0U;
    intra_expect_digest_status("intra digest zero dimension", &record,
                               &observation, &view,
                               LIS_STATUS_UNSUPPORTED_SHAPE);
    intra_digest_setup_base(&record, &observation, &view, storage);
    observation.shape[0] = SIZE_MAX;
    observation.shape[1] = 2U;
    intra_expect_digest_status("intra digest shape overflow", &record,
                               &observation, &view, LIS_STATUS_OVERFLOW);
    intra_digest_setup_base(&record, &observation, &view, storage);
    observation.element_count = 3U;
    intra_expect_digest_status("intra digest element count mismatch", &record,
                               &observation, &view,
                               LIS_STATUS_SHAPE_MISMATCH);

    intra_digest_setup_base(&record, &observation, &view, storage);
    view.data = NULL;
    intra_expect_digest_status("intra digest null view data", &record,
                               &observation, &view,
                               LIS_STATUS_INVALID_ARGUMENT);
    intra_digest_setup_base(&record, &observation, &view, storage);
    view.element_strides[0] = 0U;
    intra_expect_digest_status("intra digest zero stride", &record,
                               &observation, &view,
                               LIS_STATUS_INVALID_ARGUMENT);
    intra_digest_setup_base(&record, &observation, &view, storage);
    view.element_strides[0] = 4U;
    intra_expect_digest_status("intra digest view exceeds span", &record,
                               &observation, &view,
                               LIS_STATUS_INVALID_ARGUMENT);
    intra_digest_setup_base(&record, &observation, &view, storage);
    view.element_strides[0] = SIZE_MAX;
    intra_expect_digest_status("intra digest view offset overflow", &record,
                               &observation, &view, LIS_STATUS_OVERFLOW);
    intra_digest_setup_base(&record, &observation, &view, storage);
    view.shape[0] = 1U;
    view.shape[1] = 4U;
    view.element_strides[0] = 4U;
    view.physical_element_count = 4U;
    intra_expect_digest_status("intra digest observation view shape mismatch",
                               &record, &observation, &view,
                               LIS_STATUS_SHAPE_MISMATCH);

    intra_digest_setup_base(&record, &observation, &view, storage);
    record.precision_path[0] = (char)0xc0;
    record.precision_path[1] = (char)0x80;
    record.precision_path[2] = '\0';
    intra_expect_digest_status("intra digest invalid utf8", &record,
                               &observation, &view, LIS_STATUS_FORMAT);
    for (utf8_index = 0U;
         utf8_index < sizeof(invalid_utf8) / sizeof(invalid_utf8[0]);
         ++utf8_index) {
        intra_digest_setup_base(&record, &observation, &view, storage);
        memcpy(record.precision_path, invalid_utf8[utf8_index].bytes,
               invalid_utf8[utf8_index].length);
        record.precision_path[invalid_utf8[utf8_index].length] = '\0';
        intra_expect_digest_status(invalid_utf8[utf8_index].name, &record,
                                   &observation, &view, LIS_STATUS_FORMAT);
    }
    intra_digest_setup_base(&record, &observation, &view, storage);
    memset(record.precision_path, 'x', sizeof(record.precision_path));
    intra_expect_digest_status("intra digest unterminated precision", &record,
                               &observation, &view, LIS_STATUS_FORMAT);

    intra_digest_setup_base(&record, &observation, &view, storage);
    memcpy(record.precision_path,
           "f32-\xc2\xa2-\xe2\x82\xac-\xf0\x90\x8d\x88", 15U);
    record.precision_path[15] = '\0';
    intra_expect_digest_status("intra digest valid multibyte utf8", &record,
                               &observation, &view, LIS_STATUS_OK);

    intra_digest_setup_base(&record, &observation, &view, storage);
    memset(&observation.digest, 0xa5, sizeof(observation.digest));
    expect_status("intra digest append integration",
                  lis_intra_layer_checkpoint_digest_fp32(
                      &record, &observation, &view, &observation.digest),
                  LIS_STATUS_OK);
    expect_status("intra digest accepted by append",
                  lis_intra_layer_record_append_observation(
                      &record, &observation),
                  LIS_STATUS_OK);
}

static void test_intra_layer_checkpoint_digest_striding(void)
{
    lis_intra_layer_trace_record record;
    lis_intra_layer_observation observation;
    lis_intra_layer_fp32_view contiguous_view;
    lis_intra_layer_fp32_view strided_view;
    lis_checkpoint_digest contiguous_digest;
    lis_checkpoint_digest strided_digest;
    lis_checkpoint_digest changed_digest;
    float contiguous[4];
    float strided[6];
    static const uint32_t logical_bits[4] = {
        UINT32_C(0x3f800000), UINT32_C(0xc0200000),
        UINT32_C(0x00000001), UINT32_C(0x7f7fffff)
    };
    static const uint32_t padding_a = UINT32_C(0xdeadbeef);
    static const uint32_t padding_b = UINT32_C(0x01234567);
    static const uint32_t changed = UINT32_C(0x40000000);
    size_t index;

    intra_digest_setup_base(&record, &observation, &contiguous_view,
                            strided);
    for (index = 0U; index < 4U; ++index) {
        memcpy(contiguous + index, logical_bits + index,
               sizeof(contiguous[index]));
    }
    contiguous_view.data = contiguous;

    memcpy(strided, logical_bits, sizeof(float) * 2U);
    memcpy(strided + 2U, &padding_a, sizeof(strided[2]));
    memcpy(strided + 3U, logical_bits + 2U, sizeof(float) * 2U);
    memcpy(strided + 5U, &padding_a, sizeof(strided[5]));
    strided_view = contiguous_view;
    strided_view.data = strided;
    strided_view.element_strides[0] = 3U;
    strided_view.physical_element_count = 6U;

    expect_status("intra digest contiguous logical view",
                  lis_intra_layer_checkpoint_digest_fp32(
                      &record, &observation, &contiguous_view,
                      &contiguous_digest),
                  LIS_STATUS_OK);
    expect_status("intra digest strided logical view",
                  lis_intra_layer_checkpoint_digest_fp32(
                      &record, &observation, &strided_view, &strided_digest),
                  LIS_STATUS_OK);
    if (memcmp(contiguous_digest.bytes, strided_digest.bytes,
               sizeof(contiguous_digest.bytes)) != 0) {
        fprintf(stderr, "intra digest strided logical equivalence failed\n");
        ++g_failures;
    }

    memcpy(strided + 2U, &padding_b, sizeof(strided[2]));
    memcpy(strided + 5U, &padding_b, sizeof(strided[5]));
    expect_status("intra digest ignores physical padding",
                  lis_intra_layer_checkpoint_digest_fp32(
                      &record, &observation, &strided_view, &changed_digest),
                  LIS_STATUS_OK);
    if (memcmp(strided_digest.bytes, changed_digest.bytes,
               sizeof(strided_digest.bytes)) != 0) {
        fprintf(stderr, "intra digest included physical padding\n");
        ++g_failures;
    }

    memcpy(strided + 3U, &changed, sizeof(strided[3]));
    expect_status("intra digest logical value change",
                  lis_intra_layer_checkpoint_digest_fp32(
                      &record, &observation, &strided_view, &changed_digest),
                  LIS_STATUS_OK);
    if (memcmp(strided_digest.bytes, changed_digest.bytes,
               sizeof(strided_digest.bytes)) == 0) {
        fprintf(stderr, "intra digest ignored a logical value change\n");
        ++g_failures;
    }
}

/*
 * Behaviourally identical to the layer-trace writer's own escaper and
 * %.6g-or-null float writer, expressed without <math.h> so the core suite keeps
 * linking without libm.
 */
static void intra_test_write_string(FILE *fp, const char *text)
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

static void intra_test_write_float(FILE *fp, float value)
{
    if (value != value || value > FLT_MAX || value < -FLT_MAX) {
        fputs("null", fp);
    } else {
        fprintf(fp, "%.6g", (double)value);
    }
}

static const lis_intra_layer_json_hooks intra_test_hooks = {
    intra_test_write_string,
    intra_test_write_float
};

static char intra_json_buffer[131072];

static size_t intra_capture_json(const char *name,
                                 const lis_intra_layer_trace_record *record,
                                 lis_status expected)
{
    FILE *fp = tmpfile();
    lis_status status;
    long emitted;
    size_t read_bytes;

    intra_json_buffer[0] = '\0';
    if (fp == NULL) {
        fprintf(stderr, "%s: tmpfile() unavailable\n", name);
        ++g_failures;
        return 0;
    }
    status = lis_intra_layer_record_write_json(fp, record, &intra_test_hooks);
    expect_status(name, status, expected);
    emitted = ftell(fp);
    if (emitted < 0) {
        fprintf(stderr, "%s: ftell failed\n", name);
        ++g_failures;
        (void)fclose(fp);
        return 0;
    }
    rewind(fp);
    read_bytes = fread(intra_json_buffer, 1U,
                       sizeof(intra_json_buffer) - 1U, fp);
    intra_json_buffer[read_bytes] = '\0';
    (void)fclose(fp);
    expect_size(name, read_bytes, (size_t)emitted);
    return read_bytes;
}

static void intra_expect_contains(const char *name, const char *haystack,
                                  const char *needle)
{
    if (strstr(haystack, needle) == NULL) {
        fprintf(stderr, "%s: missing expected fragment \"%s\"\n", name,
                needle);
        ++g_failures;
    }
}

static void intra_expect_absent(const char *name, const char *haystack,
                                const char *needle)
{
    if (strstr(haystack, needle) != NULL) {
        fprintf(stderr, "%s: forbidden fragment \"%s\" present\n", name,
                needle);
        ++g_failures;
    }
}

static void intra_expect_prefix(const char *name, const char *haystack,
                                const char *prefix)
{
    if (strncmp(haystack, prefix, strlen(prefix)) != 0) {
        fprintf(stderr, "%s: output does not begin with \"%s\"\n", name,
                prefix);
        ++g_failures;
    }
}

/*
 * Requires `expected` to appear immediately after `anchor`, not merely
 * somewhere in the document. This is what pins the *first* element of an
 * emitted list to its position; a containment check cannot, because every
 * fragment of a reordered list is still present.
 */
static void intra_expect_after(const char *name, const char *haystack,
                               const char *anchor, const char *expected)
{
    const char *found = strstr(haystack, anchor);

    if (found == NULL) {
        fprintf(stderr, "%s: anchor \"%s\" not found\n", name, anchor);
        ++g_failures;
        return;
    }
    found += strlen(anchor);
    if (strncmp(found, expected, strlen(expected)) != 0) {
        fprintf(stderr, "%s: expected \"%s\" immediately after the anchor\n",
                name, expected);
        ++g_failures;
    }
}

/*
 * Asserts that the JSON array starting at `list_key` — and ending before
 * `end_key`, or at the end of the buffer when `end_key` is NULL — carries
 * exactly `expected_count` "stage_order" values, each inside the frozen
 * taxonomy and each strictly greater than its predecessor.
 *
 * Strict ascent is the frozen ordering rule itself ("malformed order is never
 * silently sorted"), and for a full 17-element list it is also exact: 17
 * strictly increasing values all below 17 can only be 0..16. Cardinality
 * checks and containment checks are both order-invariant, so this is the
 * assertion that actually fails when an emission loop is reversed.
 */
static void intra_expect_stage_order_sequence(const char *name,
                                              const char *haystack,
                                              const char *list_key,
                                              const char *end_key,
                                              size_t expected_count)
{
    static const char field[] = "\"stage_order\":";
    const char *cursor = strstr(haystack, list_key);
    const char *end;
    size_t found = 0;
    size_t previous = 0;

    if (cursor == NULL) {
        fprintf(stderr, "%s: list key \"%s\" not found\n", name, list_key);
        ++g_failures;
        return;
    }
    cursor += strlen(list_key);
    if (end_key != NULL) {
        end = strstr(cursor, end_key);
        if (end == NULL) {
            fprintf(stderr, "%s: end key \"%s\" not found\n", name, end_key);
            ++g_failures;
            return;
        }
    } else {
        end = cursor + strlen(cursor);
    }
    while ((cursor = strstr(cursor, field)) != NULL && cursor < end) {
        size_t value = 0;
        size_t digits = 0;

        cursor += sizeof(field) - 1U;
        while (*cursor >= '0' && *cursor <= '9') {
            value = value * 10U + (size_t)(*cursor - '0');
            ++cursor;
            ++digits;
        }
        if (digits == 0U) {
            fprintf(stderr, "%s: malformed stage_order at element %zu\n",
                    name, found);
            ++g_failures;
            return;
        }
        if (value >= LIS_INTRA_LAYER_STAGE_COUNT) {
            fprintf(stderr, "%s: stage_order %zu outside the taxonomy\n",
                    name, value);
            ++g_failures;
            return;
        }
        if (found > 0U && value <= previous) {
            fprintf(stderr,
                    "%s: stage_order %zu does not follow %zu; emitted order is "
                    "never sorted or repaired\n", name, value, previous);
            ++g_failures;
            return;
        }
        previous = value;
        ++found;
    }
    expect_size(name, found, expected_count);
}

static size_t intra_count_occurrences(const char *haystack,
                                      const char *needle)
{
    size_t count = 0;
    size_t step = strlen(needle);
    const char *cursor = haystack;

    if (step == 0U) {
        return 0;
    }
    while ((cursor = strstr(cursor, needle)) != NULL) {
        ++count;
        cursor += step;
    }
    return count;
}

/*
 * Configures a record and resolves every stage in canonical order: stages
 * before `unavailable_stage` and after it are captured, that one is declared
 * unavailable. Pass LIS_INTRA_LAYER_STAGE_COUNT to capture all 17.
 */
static void intra_fill_record(const char *name,
                              lis_intra_layer_trace_record *record,
                              size_t unavailable_stage)
{
    size_t index;

    expect_status("intra fill init", lis_intra_layer_record_init(record),
                  LIS_STATUS_OK);
    expect_status("intra fill configure",
                  lis_intra_layer_record_configure(record, 3U, 1U, 8U, 5U,
                                                   "bf16"),
                  LIS_STATUS_OK);
    for (index = 0; index < LIS_INTRA_LAYER_STAGE_COUNT; ++index) {
        if (index == unavailable_stage) {
            expect_status(name,
                          lis_intra_layer_record_mark_unavailable(
                              record, (lis_intra_layer_stage)index,
                              LIS_INTRA_LAYER_MISSING_UNSUPPORTED,
                              "observation_unavailable"),
                          LIS_STATUS_OK);
        } else {
            lis_intra_layer_observation observation =
                intra_make_observation((lis_intra_layer_stage)index, 3U, 1U,
                                       5U);

            expect_status(name,
                          lis_intra_layer_record_append_observation(
                              record, &observation),
                          LIS_STATUS_OK);
        }
    }
    expect_status(name, lis_intra_layer_record_finalize(record),
                  LIS_STATUS_OK);
}

static void test_intra_layer_append_ordering_and_duplicates(void)
{
    lis_intra_layer_trace_record record;
    lis_intra_layer_observation observation;

    /* Valid ordered append. */
    expect_status("intra order init", lis_intra_layer_record_init(&record),
                  LIS_STATUS_OK);
    expect_status("intra order configure",
                  lis_intra_layer_record_configure(&record, 3U, 1U, 4U, 5U,
                                                   "bf16"),
                  LIS_STATUS_OK);
    observation = intra_make_observation(LIS_INTRA_LAYER_STAGE_LAYER_INPUT,
                                         3U, 1U, 5U);
    expect_status("intra order first append",
                  lis_intra_layer_record_append_observation(&record,
                                                            &observation),
                  LIS_STATUS_OK);
    expect_size("intra order captured count", record.captured_count, 1U);
    expect_size("intra order element visits", record.digest_element_visits,
                4U);
    observation = intra_make_observation(
        LIS_INTRA_LAYER_STAGE_QUERY_PROJECTION_OUTPUT, 3U, 1U, 5U);
    expect_status("intra order second append",
                  lis_intra_layer_record_append_observation(&record,
                                                            &observation),
                  LIS_STATUS_OK);
    expect_size("intra order captured count 2", record.captured_count, 2U);

    /* Out-of-order arrival is rejected, never sorted. */
    observation = intra_make_observation(
        LIS_INTRA_LAYER_STAGE_ATTENTION_NORM_OUTPUT, 3U, 1U, 5U);
    expect_status("intra order backwards rejected",
                  lis_intra_layer_record_append_observation(&record,
                                                            &observation),
                  LIS_STATUS_INVALID_ARGUMENT);
    intra_expect_state("intra order sticky invalid", &record,
                       LIS_INTRA_LAYER_RECORD_INVALID);
    expect_size("intra order counters untouched", record.captured_count, 2U);
    expect_size("intra order visits untouched", record.digest_element_visits,
                8U);
    expect_status("intra order finalize after reject",
                  lis_intra_layer_record_finalize(&record),
                  LIS_STATUS_BAD_STATE);

    /* Duplicate stage rejection. */
    expect_status("intra duplicate init", lis_intra_layer_record_init(&record),
                  LIS_STATUS_OK);
    expect_status("intra duplicate configure",
                  lis_intra_layer_record_configure(&record, 3U, 1U, 4U, 5U,
                                                   "bf16"),
                  LIS_STATUS_OK);
    observation = intra_make_observation(LIS_INTRA_LAYER_STAGE_LAYER_INPUT,
                                         3U, 1U, 5U);
    expect_status("intra duplicate first",
                  lis_intra_layer_record_append_observation(&record,
                                                            &observation),
                  LIS_STATUS_OK);
    expect_status("intra duplicate second",
                  lis_intra_layer_record_append_observation(&record,
                                                            &observation),
                  LIS_STATUS_INVALID_ARGUMENT);
    intra_expect_state("intra duplicate sticky", &record,
                       LIS_INTRA_LAYER_RECORD_INVALID);
    expect_size("intra duplicate captured unchanged", record.captured_count,
                1U);
    expect_size("intra duplicate visits unchanged",
                record.digest_element_visits, 4U);

    /* Duplicate coordinate across the captured/unavailable boundary. */
    expect_status("intra cross init", lis_intra_layer_record_init(&record),
                  LIS_STATUS_OK);
    expect_status("intra cross configure",
                  lis_intra_layer_record_configure(&record, 3U, 1U, 4U, 5U,
                                                   "bf16"),
                  LIS_STATUS_OK);
    expect_status("intra cross append",
                  lis_intra_layer_record_append_observation(&record,
                                                            &observation),
                  LIS_STATUS_OK);
    expect_status("intra cross mark same stage",
                  lis_intra_layer_record_mark_unavailable(
                      &record, LIS_INTRA_LAYER_STAGE_LAYER_INPUT,
                      LIS_INTRA_LAYER_MISSING_NOT_CAPTURED, "already_present"),
                  LIS_STATUS_INVALID_ARGUMENT);
    intra_expect_state("intra cross sticky", &record,
                       LIS_INTRA_LAYER_RECORD_INVALID);

    expect_status("intra cross2 init", lis_intra_layer_record_init(&record),
                  LIS_STATUS_OK);
    expect_status("intra cross2 configure",
                  lis_intra_layer_record_configure(&record, 3U, 1U, 4U, 5U,
                                                   "bf16"),
                  LIS_STATUS_OK);
    expect_status("intra cross2 mark",
                  lis_intra_layer_record_mark_unavailable(
                      &record, LIS_INTRA_LAYER_STAGE_LAYER_INPUT,
                      LIS_INTRA_LAYER_MISSING_NOT_CAPTURED, "unavailable"),
                  LIS_STATUS_OK);
    expect_status("intra cross2 append same stage",
                  lis_intra_layer_record_append_observation(&record,
                                                            &observation),
                  LIS_STATUS_INVALID_ARGUMENT);
    intra_expect_state("intra cross2 sticky", &record,
                       LIS_INTRA_LAYER_RECORD_INVALID);
    lis_intra_layer_record_destroy(&record);
}

static void test_intra_layer_append_coordinate_guards(void)
{
    static const struct {
        const char *name;
        size_t      step;
        size_t      layer;
        size_t      token;
        size_t      batch;
        size_t      sequence;
        size_t      stage_order_override;
        int         override_stage_order;
        size_t      execution_ordinal_override;
        int         override_execution_ordinal;
    } cases[] = {
        { "intra coord step mismatch", 4U, 1U, 5U, 0U, 0U, 0U, 0, 0U, 0 },
        { "intra coord layer mismatch", 3U, 2U, 5U, 0U, 0U, 0U, 0, 0U, 0 },
        { "intra coord token mismatch", 3U, 1U, 6U, 0U, 0U, 0U, 0, 0U, 0 },
        { "intra coord batch nonzero", 3U, 1U, 5U, 1U, 0U, 0U, 0, 0U, 0 },
        { "intra coord sequence nonzero", 3U, 1U, 5U, 0U, 1U, 0U, 0, 0U, 0 },
        { "intra coord stage order mismatch", 3U, 1U, 5U, 0U, 0U, 9U, 1, 0U,
          0 },
        { "intra coord execution ordinal mismatch", 3U, 1U, 5U, 0U, 0U, 0U, 0,
          9U, 1 }
    };
    size_t index;

    for (index = 0; index < sizeof(cases) / sizeof(cases[0]); ++index) {
        lis_intra_layer_trace_record record;
        lis_intra_layer_observation observation;

        expect_status("intra coord init",
                      lis_intra_layer_record_init(&record), LIS_STATUS_OK);
        expect_status("intra coord configure",
                      lis_intra_layer_record_configure(&record, 3U, 1U, 4U,
                                                       5U, "bf16"),
                      LIS_STATUS_OK);
        observation = intra_make_observation(
            LIS_INTRA_LAYER_STAGE_LAYER_INPUT, cases[index].step,
            cases[index].layer, cases[index].token);
        observation.batch_index = cases[index].batch;
        observation.sequence_index = cases[index].sequence;
        if (cases[index].override_stage_order) {
            observation.stage_order = cases[index].stage_order_override;
            observation.execution_ordinal = cases[index].stage_order_override;
        }
        if (cases[index].override_execution_ordinal) {
            observation.execution_ordinal =
                cases[index].execution_ordinal_override;
        }
        expect_status(cases[index].name,
                      lis_intra_layer_record_append_observation(&record,
                                                                &observation),
                      LIS_STATUS_INVALID_ARGUMENT);
        intra_expect_state(cases[index].name, &record,
                           LIS_INTRA_LAYER_RECORD_INVALID);
        lis_intra_layer_record_destroy(&record);
    }
}

static void test_intra_layer_append_payload_guards(void)
{
    lis_intra_layer_trace_record record;
    lis_intra_layer_observation observation;
    lis_intra_layer_observation zeroed;

    memset(&zeroed, 0, sizeof(zeroed));

#define INTRA_RESET_RECORD()                                                 \
    do {                                                                     \
        expect_status("intra payload init",                                  \
                      lis_intra_layer_record_init(&record), LIS_STATUS_OK);  \
        expect_status("intra payload configure",                             \
                      lis_intra_layer_record_configure(&record, 3U, 1U, 4U,  \
                                                       5U, "bf16"),          \
                      LIS_STATUS_OK);                                        \
        observation = intra_make_observation(                                \
            LIS_INTRA_LAYER_STAGE_LAYER_INPUT, 3U, 1U, 5U);                  \
    } while (0)

    /* Unknown stage identifiers. */
    INTRA_RESET_RECORD();
    observation.stage = (lis_intra_layer_stage)LIS_INTRA_LAYER_STAGE_COUNT;
    observation.stage_order = LIS_INTRA_LAYER_STAGE_COUNT;
    observation.execution_ordinal = LIS_INTRA_LAYER_STAGE_COUNT;
    expect_status("intra unknown stage 17",
                  lis_intra_layer_record_append_observation(&record,
                                                            &observation),
                  LIS_STATUS_INVALID_ARGUMENT);
    intra_expect_state("intra unknown stage 17 sticky", &record,
                       LIS_INTRA_LAYER_RECORD_INVALID);

    INTRA_RESET_RECORD();
    observation.stage =
        (lis_intra_layer_stage)(LIS_INTRA_LAYER_STAGE_COUNT + 4096U);
    expect_status("intra unknown stage far",
                  lis_intra_layer_record_append_observation(&record,
                                                            &observation),
                  LIS_STATUS_INVALID_ARGUMENT);

    /* Phase gating: a zeroed observation is invalid, not decode. */
    INTRA_RESET_RECORD();
    expect_status("intra phase zeroed",
                  lis_intra_layer_record_append_observation(&record, &zeroed),
                  LIS_STATUS_UNSUPPORTED);
    intra_expect_state("intra phase sticky", &record,
                       LIS_INTRA_LAYER_RECORD_INVALID);

    /* Shape, rank, and element-count coherence. */
    INTRA_RESET_RECORD();
    observation.rank = 0U;
    expect_status("intra rank zero",
                  lis_intra_layer_record_append_observation(&record,
                                                            &observation),
                  LIS_STATUS_UNSUPPORTED_SHAPE);

    INTRA_RESET_RECORD();
    observation.rank = LIS_INTRA_LAYER_MAX_RANK + 1U;
    expect_status("intra rank too large",
                  lis_intra_layer_record_append_observation(&record,
                                                            &observation),
                  LIS_STATUS_UNSUPPORTED_SHAPE);

    INTRA_RESET_RECORD();
    observation.rank = 2U;
    observation.shape[0] = 4U;
    observation.shape[1] = 0U;
    expect_status("intra zero dimension",
                  lis_intra_layer_record_append_observation(&record,
                                                            &observation),
                  LIS_STATUS_UNSUPPORTED_SHAPE);

    INTRA_RESET_RECORD();
    observation.element_count = 5U;
    expect_status("intra element count mismatch",
                  lis_intra_layer_record_append_observation(&record,
                                                            &observation),
                  LIS_STATUS_SHAPE_MISMATCH);

    INTRA_RESET_RECORD();
    observation.rank = 1U;
    observation.shape[0] = 0U;
    observation.element_count = 0U;
    expect_status("intra empty tensor",
                  lis_intra_layer_record_append_observation(&record,
                                                            &observation),
                  LIS_STATUS_UNSUPPORTED_SHAPE);

    INTRA_RESET_RECORD();
    observation.rank = 2U;
    observation.shape[0] = SIZE_MAX;
    observation.shape[1] = 2U;
    observation.element_count = 2U;
    expect_status("intra shape product overflow",
                  lis_intra_layer_record_append_observation(&record,
                                                            &observation),
                  LIS_STATUS_OVERFLOW);

    /* Contract integer flags. */
    INTRA_RESET_RECORD();
    observation.nan = 2;
    expect_status("intra nan flag out of range",
                  lis_intra_layer_record_append_observation(&record,
                                                            &observation),
                  LIS_STATUS_INVALID_ARGUMENT);

    INTRA_RESET_RECORD();
    observation.inf = -1;
    expect_status("intra inf flag out of range",
                  lis_intra_layer_record_append_observation(&record,
                                                            &observation),
                  LIS_STATUS_INVALID_ARGUMENT);

    /* The digest is required evidence, never derived here. */
    INTRA_RESET_RECORD();
    observation.digest.valid = 0;
    expect_status("intra digest not valid",
                  lis_intra_layer_record_append_observation(&record,
                                                            &observation),
                  LIS_STATUS_INVALID_ARGUMENT);

    /* Element-visit accumulation overflow. */
    INTRA_RESET_RECORD();
    record.digest_element_visits = SIZE_MAX - 3U;
    expect_status("intra visit accumulation overflow",
                  lis_intra_layer_record_append_observation(&record,
                                                            &observation),
                  LIS_STATUS_OVERFLOW);
    expect_size("intra visit unchanged after overflow",
                record.digest_element_visits, SIZE_MAX - 3U);

    /* The inclusive boundary still succeeds. */
    INTRA_RESET_RECORD();
    record.digest_element_visits = SIZE_MAX - 4U;
    expect_status("intra visit accumulation boundary",
                  lis_intra_layer_record_append_observation(&record,
                                                            &observation),
                  LIS_STATUS_OK);
    expect_size("intra visit saturated", record.digest_element_visits,
                SIZE_MAX);

    /* NULL observation. */
    INTRA_RESET_RECORD();
    expect_status("intra null observation",
                  lis_intra_layer_record_append_observation(&record, NULL),
                  LIS_STATUS_INVALID_ARGUMENT);
    intra_expect_state("intra null observation sticky", &record,
                       LIS_INTRA_LAYER_RECORD_INVALID);
    expect_status("intra append null record",
                  lis_intra_layer_record_append_observation(NULL,
                                                            &observation),
                  LIS_STATUS_INVALID_ARGUMENT);

#undef INTRA_RESET_RECORD
    lis_intra_layer_record_destroy(&record);
}

static void test_intra_layer_mark_unavailable_guards(void)
{
    static const lis_intra_layer_missing_state legal[] = {
        LIS_INTRA_LAYER_MISSING_NOT_CAPTURED,
        LIS_INTRA_LAYER_MISSING_UNSUPPORTED,
        LIS_INTRA_LAYER_MISSING_MALFORMED,
        LIS_INTRA_LAYER_MISSING_UNEXPECTEDLY_ABSENT
    };
    lis_intra_layer_trace_record record;
    char oversized[LIS_INTRA_LAYER_DETAIL_MAX + 2U];
    size_t index;

    memset(oversized, 'd', sizeof(oversized) - 1U);
    oversized[sizeof(oversized) - 1U] = '\0';

    for (index = 0; index < sizeof(legal) / sizeof(legal[0]); ++index) {
        expect_status("intra missing legal init",
                      lis_intra_layer_record_init(&record), LIS_STATUS_OK);
        expect_status("intra missing legal configure",
                      lis_intra_layer_record_configure(&record, 3U, 1U, 4U,
                                                       5U, "bf16"),
                      LIS_STATUS_OK);
        expect_status("intra missing legal state",
                      lis_intra_layer_record_mark_unavailable(
                          &record, LIS_INTRA_LAYER_STAGE_LAYER_INPUT,
                          legal[index], "reason"),
                      LIS_STATUS_OK);
        expect_size("intra missing count", record.missing_count, 1U);
        expect_size("intra missing adds no visits",
                    record.digest_element_visits, 0U);
    }

    expect_status("intra missing invalid init",
                  lis_intra_layer_record_init(&record), LIS_STATUS_OK);
    expect_status("intra missing invalid configure",
                  lis_intra_layer_record_configure(&record, 3U, 1U, 4U, 5U,
                                                   "bf16"),
                  LIS_STATUS_OK);
    expect_status("intra missing invalid sentinel",
                  lis_intra_layer_record_mark_unavailable(
                      &record, LIS_INTRA_LAYER_STAGE_LAYER_INPUT,
                      LIS_INTRA_LAYER_MISSING_INVALID, "reason"),
                  LIS_STATUS_INVALID_ARGUMENT);
    intra_expect_state("intra missing invalid sticky", &record,
                       LIS_INTRA_LAYER_RECORD_INVALID);

#define INTRA_MARK_CASE(label, stage, state, detail, expected)               \
    do {                                                                     \
        expect_status("intra mark init",                                     \
                      lis_intra_layer_record_init(&record), LIS_STATUS_OK);  \
        expect_status("intra mark configure",                                \
                      lis_intra_layer_record_configure(&record, 3U, 1U, 4U,  \
                                                       5U, "bf16"),          \
                      LIS_STATUS_OK);                                        \
        expect_status((label),                                               \
                      lis_intra_layer_record_mark_unavailable(               \
                          &record, (stage), (state), (detail)),              \
                      (expected));                                           \
    } while (0)

    INTRA_MARK_CASE("intra mark null detail",
                    LIS_INTRA_LAYER_STAGE_LAYER_INPUT,
                    LIS_INTRA_LAYER_MISSING_NOT_CAPTURED, NULL,
                    LIS_STATUS_INVALID_ARGUMENT);
    INTRA_MARK_CASE("intra mark empty detail",
                    LIS_INTRA_LAYER_STAGE_LAYER_INPUT,
                    LIS_INTRA_LAYER_MISSING_NOT_CAPTURED, "",
                    LIS_STATUS_INVALID_ARGUMENT);
    INTRA_MARK_CASE("intra mark oversized detail",
                    LIS_INTRA_LAYER_STAGE_LAYER_INPUT,
                    LIS_INTRA_LAYER_MISSING_NOT_CAPTURED, oversized,
                    LIS_STATUS_FORMAT);
    INTRA_MARK_CASE("intra mark control byte detail",
                    LIS_INTRA_LAYER_STAGE_LAYER_INPUT,
                    LIS_INTRA_LAYER_MISSING_NOT_CAPTURED, "bad\ndetail",
                    LIS_STATUS_INVALID_ARGUMENT);
    INTRA_MARK_CASE("intra mark unknown stage",
                    (lis_intra_layer_stage)LIS_INTRA_LAYER_STAGE_COUNT,
                    LIS_INTRA_LAYER_MISSING_NOT_CAPTURED, "reason",
                    LIS_STATUS_INVALID_ARGUMENT);

    /* Boundary-length detail is accepted, not truncated. */
    oversized[LIS_INTRA_LAYER_DETAIL_MAX] = '\0';
    INTRA_MARK_CASE("intra mark boundary detail",
                    LIS_INTRA_LAYER_STAGE_LAYER_INPUT,
                    LIS_INTRA_LAYER_MISSING_NOT_CAPTURED, oversized,
                    LIS_STATUS_OK);
    expect_size("intra mark boundary stored",
                strlen(record.slots[0].detail),
                (size_t)LIS_INTRA_LAYER_DETAIL_MAX);

#undef INTRA_MARK_CASE

    expect_status("intra mark null record",
                  lis_intra_layer_record_mark_unavailable(
                      NULL, LIS_INTRA_LAYER_STAGE_LAYER_INPUT,
                      LIS_INTRA_LAYER_MISSING_NOT_CAPTURED, "reason"),
                  LIS_STATUS_INVALID_ARGUMENT);
    lis_intra_layer_record_destroy(&record);
}

static void test_intra_layer_finalize_partition(void)
{
    lis_intra_layer_trace_record record;
    size_t index;

    /* 16 resolved of 17 must not finalize. */
    expect_status("intra partial init", lis_intra_layer_record_init(&record),
                  LIS_STATUS_OK);
    expect_status("intra partial configure",
                  lis_intra_layer_record_configure(&record, 3U, 1U, 4U, 5U,
                                                   "bf16"),
                  LIS_STATUS_OK);
    for (index = 0; index + 1U < LIS_INTRA_LAYER_STAGE_COUNT; ++index) {
        lis_intra_layer_observation observation =
            intra_make_observation((lis_intra_layer_stage)index, 3U, 1U, 5U);

        expect_status("intra partial append",
                      lis_intra_layer_record_append_observation(&record,
                                                                &observation),
                      LIS_STATUS_OK);
    }
    expect_status("intra partial finalize",
                  lis_intra_layer_record_finalize(&record),
                  LIS_STATUS_BAD_STATE);
    intra_expect_state("intra partial sticky", &record,
                       LIS_INTRA_LAYER_RECORD_INVALID);

    /* 16 captured + 1 unavailable is READY. */
    intra_fill_record("intra mixed fill", &record, 3U);
    intra_expect_state("intra mixed ready", &record,
                       LIS_INTRA_LAYER_RECORD_READY);
    expect_size("intra mixed captured", record.captured_count, 16U);
    expect_size("intra mixed missing", record.missing_count, 1U);
    expect_size("intra mixed union", record.captured_count +
                                         record.missing_count,
                (size_t)LIS_INTRA_LAYER_STAGE_COUNT);
    expect_size("intra mixed is_ready",
                (size_t)lis_intra_layer_record_is_ready(&record), 1U);
    /* Finalize is idempotent once READY, and the record is sealed. */
    expect_status("intra mixed finalize idempotent",
                  lis_intra_layer_record_finalize(&record), LIS_STATUS_OK);
    {
        lis_intra_layer_observation observation = intra_make_observation(
            LIS_INTRA_LAYER_STAGE_LAYER_INPUT, 3U, 1U, 5U);

        expect_status("intra ready append rejected",
                      lis_intra_layer_record_append_observation(&record,
                                                                &observation),
                      LIS_STATUS_BAD_STATE);
        intra_expect_state("intra ready append invalidates", &record,
                           LIS_INTRA_LAYER_RECORD_INVALID);
    }

    /* 17 captured is READY. */
    intra_fill_record("intra full fill", &record,
                      LIS_INTRA_LAYER_STAGE_COUNT);
    intra_expect_state("intra full ready", &record,
                       LIS_INTRA_LAYER_RECORD_READY);
    expect_size("intra full captured", record.captured_count,
                (size_t)LIS_INTRA_LAYER_STAGE_COUNT);
    expect_size("intra full missing", record.missing_count, 0U);
    expect_size("intra full visits", record.digest_element_visits, 68U);

    expect_status("intra finalize null",
                  lis_intra_layer_record_finalize(NULL),
                  LIS_STATUS_INVALID_ARGUMENT);
    lis_intra_layer_record_destroy(&record);
}

static void test_intra_layer_sticky_invalidity(void)
{
    lis_intra_layer_trace_record record;
    lis_intra_layer_observation observation =
        intra_make_observation(LIS_INTRA_LAYER_STAGE_LAYER_INPUT, 3U, 1U, 5U);

    expect_status("intra sticky init", lis_intra_layer_record_init(&record),
                  LIS_STATUS_OK);
    expect_status("intra sticky configure",
                  lis_intra_layer_record_configure(&record, 3U, 1U, 4U, 5U,
                                                   "bf16"),
                  LIS_STATUS_OK);
    lis_intra_layer_record_invalidate(&record);
    intra_expect_state("intra sticky invalid", &record,
                       LIS_INTRA_LAYER_RECORD_INVALID);
    lis_intra_layer_record_invalidate(&record);
    intra_expect_state("intra sticky idempotent", &record,
                       LIS_INTRA_LAYER_RECORD_INVALID);

    expect_status("intra sticky configure rejected",
                  lis_intra_layer_record_configure(&record, 3U, 1U, 4U, 5U,
                                                   "bf16"),
                  LIS_STATUS_BAD_STATE);
    expect_status("intra sticky append rejected",
                  lis_intra_layer_record_append_observation(&record,
                                                            &observation),
                  LIS_STATUS_BAD_STATE);
    expect_status("intra sticky mark rejected",
                  lis_intra_layer_record_mark_unavailable(
                      &record, LIS_INTRA_LAYER_STAGE_LAYER_INPUT,
                      LIS_INTRA_LAYER_MISSING_NOT_CAPTURED, "reason"),
                  LIS_STATUS_BAD_STATE);
    expect_status("intra sticky finalize rejected",
                  lis_intra_layer_record_finalize(&record),
                  LIS_STATUS_BAD_STATE);
    intra_expect_state("intra sticky never cleared", &record,
                       LIS_INTRA_LAYER_RECORD_INVALID);
    expect_size("intra sticky not ready",
                (size_t)lis_intra_layer_record_is_ready(&record), 0U);

    /* init is the only recovery. */
    expect_status("intra sticky reinit", lis_intra_layer_record_init(&record),
                  LIS_STATUS_OK);
    intra_expect_state("intra sticky reinit state", &record,
                       LIS_INTRA_LAYER_RECORD_UNINITIALIZED);
    lis_intra_layer_record_destroy(&record);
}

static void test_intra_layer_json_serialization(void)
{
    static const char expected_layout_head[] =
        ",\"intra_layer_checkpoint_layout\":{"
        "\"layout_name\":\"llama_intra_layer_summary\","
        "\"layout_version\":1,"
        "\"model_family\":\"llama3_decoder\","
        "\"stage_taxonomy\":\"lis.llama.intra_layer_stages/v1\","
        "\"runtime_checkpoint_step\":3,"
        "\"phase\":\"decode\","
        "\"target_layer\":1,"
        "\"batch_index\":0,"
        "\"sequence_index\":0,"
        "\"token_position\":5,"
        "\"ordering_semantics\":\"runtime_step_layer_stage_ordinal\","
        "\"duplicate_coordinate_policy\":\"reject_artifact_before_write\","
        "\"requested_coordinates\":[";
    static const char expected_first_coordinate[] =
        "{\"runtime_checkpoint_step\":3,\"layer_index\":1,"
        "\"stage_id\":\"layer_input\",\"tensor_role\":\"layer_input\","
        "\"batch_index\":0,\"sequence_index\":0,\"token_position\":5,"
        "\"stage_order\":0,\"execution_ordinal\":0}";
    static const char expected_last_coordinate[] =
        "{\"runtime_checkpoint_step\":3,\"layer_index\":1,"
        "\"stage_id\":\"mlp_down_projection\","
        "\"tensor_role\":\"mlp_down_projection\","
        "\"batch_index\":0,\"sequence_index\":0,\"token_position\":5,"
        "\"stage_order\":16,\"execution_ordinal\":16}";
    static const char expected_layout_tail[] =
        "],\"available_summary_fields\":[\"min\",\"max\",\"mean\",\"l2\","
        "\"nan\",\"inf\",\"digest\"],"
        "\"digest_contract\":{\"algorithm\":\"sha256\","
        "\"version\":\"lis.checkpoint.intra_layer.fp32le/v1\","
        "\"observed_dtype\":\"fp32\",\"byte_order\":\"little\","
        "\"canonicalization\":"
        "\"ieee754-binary32-le;canonical-qnan;preserve-signed-zero\"},"
        "\"full_tensor_payload_allowed\":false},\"intra_layer_trace\":[";
    static const char expected_first_entry[] =
        "{\"runtime_checkpoint_step\":3,\"phase\":\"decode\","
        "\"layer_index\":1,\"stage_id\":\"layer_input\","
        "\"tensor_role\":\"layer_input\",\"public_name\":\"Layer input\","
        "\"batch_index\":0,\"sequence_index\":0,\"token_position\":5,"
        "\"stage_order\":0,\"execution_ordinal\":0,\"shape\":[4],"
        "\"observed_dtype\":\"fp32\",\"precision_path\":\"bf16\","
        "\"element_count\":4,"
        "\"available_summary_fields\":[\"min\",\"max\",\"mean\",\"l2\","
        "\"nan\",\"inf\",\"digest\"],"
        "\"min\":-1,\"max\":2,\"mean\":0.5,\"l2\":2.5,\"nan\":0,\"inf\":0,"
        "\"digest\":{\"algorithm\":\"sha256\","
        "\"version\":\"lis.checkpoint.intra_layer.fp32le/v1\","
        "\"tensor_role\":\"layer_input\",\"shape\":[4],"
        "\"observed_dtype\":\"fp32\",\"byte_order\":\"little\","
        "\"canonicalization\":"
        "\"ieee754-binary32-le;canonical-qnan;preserve-signed-zero\","
        "\"value\":\"sha256:"
        "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f\"}}";
    lis_intra_layer_trace_record record;
    size_t size;

    intra_fill_record("intra json fill", &record,
                      LIS_INTRA_LAYER_STAGE_COUNT);
    size = intra_capture_json("intra json full", &record, LIS_STATUS_OK);
    if (size == 0U) {
        lis_intra_layer_record_destroy(&record);
        return;
    }

    intra_expect_prefix("intra json layout head", intra_json_buffer,
                        expected_layout_head);
    intra_expect_contains("intra json last coordinate", intra_json_buffer,
                          expected_last_coordinate);
    intra_expect_contains("intra json layout tail", intra_json_buffer,
                          expected_layout_tail);
    intra_expect_contains("intra json empty missing list", intra_json_buffer,
                          "\"missing_coordinates\":[],");

    /*
     * Each list must BEGIN with its stage-0 element, anchored to the byte that
     * opens the list, and must then ascend through the canonical order. The
     * head anchors and the sequence walks together pin every element position;
     * containment and cardinality alone are order-invariant.
     */
    intra_expect_after("intra json requested head", intra_json_buffer,
                       expected_layout_head, expected_first_coordinate);
    intra_expect_after("intra json captured head", intra_json_buffer,
                       "],\"captured_coordinates\":[",
                       expected_first_coordinate);
    intra_expect_after("intra json entry head", intra_json_buffer,
                       expected_layout_tail, expected_first_entry);
    intra_expect_stage_order_sequence("intra json requested order",
                                      intra_json_buffer,
                                      "\"requested_coordinates\":[",
                                      "\"captured_coordinates\":[",
                                      (size_t)LIS_INTRA_LAYER_STAGE_COUNT);
    intra_expect_stage_order_sequence("intra json captured order",
                                      intra_json_buffer,
                                      "\"captured_coordinates\":[",
                                      "\"missing_coordinates\":[",
                                      (size_t)LIS_INTRA_LAYER_STAGE_COUNT);
    intra_expect_stage_order_sequence("intra json entry order",
                                      intra_json_buffer,
                                      "\"intra_layer_trace\":[", NULL,
                                      (size_t)LIS_INTRA_LAYER_STAGE_COUNT);

    /* 17 requested + 17 captured + 0 missing + 17 entries. */
    expect_size("intra json coordinate objects",
                intra_count_occurrences(intra_json_buffer,
                                        "\"execution_ordinal\":"),
                51U);
    expect_size("intra json entries",
                intra_count_occurrences(intra_json_buffer, "\"public_name\":"),
                (size_t)LIS_INTRA_LAYER_STAGE_COUNT);
    expect_size("intra json digest values",
                intra_count_occurrences(intra_json_buffer, "\"value\":\"sha256:"),
                (size_t)LIS_INTRA_LAYER_STAGE_COUNT);
    /* The parent boundary role is never emitted as an intra entry. */
    intra_expect_absent("intra json no layer_output", intra_json_buffer,
                        "layer_output");
    /* No tensor payload field exists in any P4-4 structure. */
    intra_expect_absent("intra json no payload", intra_json_buffer,
                        "\"values\"");
    if (intra_json_buffer[size - 1U] != ']') {
        fprintf(stderr, "intra json terminator: expected ']', got '%c'\n",
                intra_json_buffer[size - 1U]);
        ++g_failures;
    }
    lis_intra_layer_record_destroy(&record);
}

static void test_intra_layer_json_mixed_and_discipline(void)
{
    lis_intra_layer_trace_record record;
    lis_intra_layer_observation observation;
    uint32_t nan_bits = UINT32_C(0x7fc00000);
    float nan_value;
    size_t index;

    intra_fill_record("intra mixed json fill", &record, 3U);
    if (intra_capture_json("intra mixed json", &record, LIS_STATUS_OK) == 0U) {
        lis_intra_layer_record_destroy(&record);
        return;
    }
    intra_expect_contains(
        "intra mixed missing entry", intra_json_buffer,
        "\"missing_coordinates\":[{\"coordinate\":"
        "{\"runtime_checkpoint_step\":3,\"layer_index\":1,"
        "\"stage_id\":\"key_projection_output\","
        "\"tensor_role\":\"key_projection_output\","
        "\"batch_index\":0,\"sequence_index\":0,\"token_position\":5,"
        "\"stage_order\":3,\"execution_ordinal\":3},"
        "\"state\":\"unsupported\","
        "\"detail\":\"observation_unavailable\"}],");
    /* The unavailable stage has no trace entry (public_name is entry-only). */
    intra_expect_absent("intra mixed no entry for missing", intra_json_buffer,
                        "\"public_name\":\"K projection output\"");
    expect_size("intra mixed entry count",
                intra_count_occurrences(intra_json_buffer, "\"public_name\":"),
                16U);
    /*
     * With stage 3 unavailable, captured and entry lists are the ordered
     * 16-element complement and missing is the ordered 1-element remainder.
     * Ascent must hold across the gap, not merely the counts.
     */
    intra_expect_stage_order_sequence("intra mixed requested order",
                                      intra_json_buffer,
                                      "\"requested_coordinates\":[",
                                      "\"captured_coordinates\":[",
                                      (size_t)LIS_INTRA_LAYER_STAGE_COUNT);
    intra_expect_stage_order_sequence("intra mixed captured order",
                                      intra_json_buffer,
                                      "\"captured_coordinates\":[",
                                      "\"missing_coordinates\":[", 16U);
    intra_expect_stage_order_sequence("intra mixed missing order",
                                      intra_json_buffer,
                                      "\"missing_coordinates\":[",
                                      "\"available_summary_fields\":", 1U);
    intra_expect_stage_order_sequence("intra mixed entry order",
                                      intra_json_buffer,
                                      "\"intra_layer_trace\":[", NULL, 16U);
    /* 17 requested + 16 captured + 1 missing + 16 entries. */
    expect_size("intra mixed coordinate objects",
                intra_count_occurrences(intra_json_buffer,
                                        "\"execution_ordinal\":"),
                50U);

    /* Contract booleans are JSON literals; contract integer flags are bare
     * integers. Neither is coerced into the other. */
    intra_expect_contains("intra boolean literal", intra_json_buffer,
                          "\"full_tensor_payload_allowed\":false}");
    intra_expect_absent("intra boolean not integer", intra_json_buffer,
                        "\"full_tensor_payload_allowed\":0");
    intra_expect_absent("intra boolean not one", intra_json_buffer,
                        "\"full_tensor_payload_allowed\":1");
    intra_expect_absent("intra nan flag not boolean", intra_json_buffer,
                        "\"nan\":false");
    intra_expect_absent("intra inf flag not boolean", intra_json_buffer,
                        "\"inf\":true");
    intra_expect_contains("intra layout version unquoted", intra_json_buffer,
                          "\"layout_version\":1,");

    /* Non-finite summaries render as JSON null, matching the Pass 3 writer. */
    memcpy(&nan_value, &nan_bits, sizeof(nan_value));
    expect_status("intra nonfinite init", lis_intra_layer_record_init(&record),
                  LIS_STATUS_OK);
    expect_status("intra nonfinite configure",
                  lis_intra_layer_record_configure(&record, 3U, 1U, 4U, 5U,
                                                   "bf16"),
                  LIS_STATUS_OK);
    for (index = 0; index < LIS_INTRA_LAYER_STAGE_COUNT; ++index) {
        observation = intra_make_observation((lis_intra_layer_stage)index, 3U,
                                             1U, 5U);
        if (index == 0U) {
            observation.mean = nan_value;
            observation.nan = 1;
            observation.inf = 1;
        }
        expect_status("intra nonfinite append",
                      lis_intra_layer_record_append_observation(&record,
                                                                &observation),
                      LIS_STATUS_OK);
    }
    expect_status("intra nonfinite finalize",
                  lis_intra_layer_record_finalize(&record), LIS_STATUS_OK);
    if (intra_capture_json("intra nonfinite json", &record,
                           LIS_STATUS_OK) != 0U) {
        intra_expect_contains("intra nonfinite null", intra_json_buffer,
                              "\"mean\":null,");
        intra_expect_contains("intra nonfinite flags", intra_json_buffer,
                              "\"nan\":1,\"inf\":1,");
    }
    lis_intra_layer_record_destroy(&record);
}

static void test_intra_layer_json_rejects_unready(void)
{
    lis_intra_layer_trace_record record;
    lis_intra_layer_observation observation;
    lis_intra_layer_json_hooks broken;

    expect_status("intra unready init", lis_intra_layer_record_init(&record),
                  LIS_STATUS_OK);
    expect_size("intra unready uninitialized bytes",
                intra_capture_json("intra json uninitialized", &record,
                                   LIS_STATUS_BAD_STATE),
                0U);

    expect_status("intra unready configure",
                  lis_intra_layer_record_configure(&record, 3U, 1U, 4U, 5U,
                                                   "bf16"),
                  LIS_STATUS_OK);
    observation = intra_make_observation(LIS_INTRA_LAYER_STAGE_LAYER_INPUT,
                                         3U, 1U, 5U);
    expect_status("intra unready append",
                  lis_intra_layer_record_append_observation(&record,
                                                            &observation),
                  LIS_STATUS_OK);
    expect_size("intra unready active bytes",
                intra_capture_json("intra json active", &record,
                                   LIS_STATUS_BAD_STATE),
                0U);

    lis_intra_layer_record_invalidate(&record);
    expect_size("intra unready invalid bytes",
                intra_capture_json("intra json invalid", &record,
                                   LIS_STATUS_BAD_STATE),
                0U);

    /* Null and incomplete hook arguments. */
    intra_fill_record("intra hook fill", &record,
                      LIS_INTRA_LAYER_STAGE_COUNT);
    expect_status("intra json null fp",
                  lis_intra_layer_record_write_json(NULL, &record,
                                                    &intra_test_hooks),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_status("intra json null record",
                  lis_intra_layer_record_write_json(stdout, NULL,
                                                    &intra_test_hooks),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_status("intra json null hooks",
                  lis_intra_layer_record_write_json(stdout, &record, NULL),
                  LIS_STATUS_INVALID_ARGUMENT);
    broken.write_string = NULL;
    broken.write_float = intra_test_write_float;
    expect_status("intra json null string hook",
                  lis_intra_layer_record_write_json(stdout, &record, &broken),
                  LIS_STATUS_INVALID_ARGUMENT);
    broken.write_string = intra_test_write_string;
    broken.write_float = NULL;
    expect_status("intra json null float hook",
                  lis_intra_layer_record_write_json(stdout, &record, &broken),
                  LIS_STATUS_INVALID_ARGUMENT);
    lis_intra_layer_record_destroy(&record);
}

static void test_intra_layer_digest_is_carried_not_computed(void)
{
    static const char distinctive[] =
        "\"value\":\"sha256:"
        "0f0e0d0c0b0a09080706050403020100"
        "f0e0d0c0b0a090807060504030201000\"";
    lis_intra_layer_trace_record record;
    lis_intra_layer_observation observation;
    size_t index;

    expect_status("intra carry init", lis_intra_layer_record_init(&record),
                  LIS_STATUS_OK);
    expect_status("intra carry configure",
                  lis_intra_layer_record_configure(&record, 3U, 1U, 4U, 5U,
                                                   "bf16"),
                  LIS_STATUS_OK);
    for (index = 0; index < LIS_INTRA_LAYER_STAGE_COUNT; ++index) {
        observation = intra_make_observation((lis_intra_layer_stage)index, 3U,
                                             1U, 5U);
        if (index == 0U) {
            size_t byte;

            /* A pattern no digest function would produce from this payload. */
            for (byte = 0; byte < 16U; ++byte) {
                observation.digest.bytes[byte] = (unsigned char)(0x0FU - byte);
            }
            for (byte = 16U; byte < LIS_CHECKPOINT_DIGEST_SIZE; ++byte) {
                observation.digest.bytes[byte] =
                    (unsigned char)((0xFU - (byte - 16U)) << 4);
            }
        }
        expect_status("intra carry append",
                      lis_intra_layer_record_append_observation(&record,
                                                                &observation),
                      LIS_STATUS_OK);
    }
    expect_status("intra carry finalize",
                  lis_intra_layer_record_finalize(&record), LIS_STATUS_OK);
    /* The stored observation is a verbatim copy of what the caller supplied. */
    expect_size("intra carry digest byte 0",
                (size_t)record.slots[0].observation.digest.bytes[0], 0x0FU);
    expect_size("intra carry digest byte 31",
                (size_t)record.slots[0].observation.digest.bytes[31], 0x00U);
    if (intra_capture_json("intra carry json", &record, LIS_STATUS_OK) != 0U) {
        intra_expect_contains("intra carry digest hex", intra_json_buffer,
                              distinctive);
    }
    lis_intra_layer_record_destroy(&record);
}

/*
 * Writer-integration fixture. Every value is fixed so the emitted artifact is
 * byte-deterministic and the additive-insertion relation can be compared
 * exactly.
 */
typedef struct {
    lis_cli_options       options;
    lis_loaded_model      model;
    lis_artifact_set_id   set_id;
    lis_layer_trace_record record;
    lis_layer_trace_artifact artifact;
} intra_writer_fixture;

static void intra_writer_fixture_init(intra_writer_fixture *fixture,
                                      const char *path)
{
    lis_layer_trace_step step;

    memset(&fixture->options, 0, sizeof(fixture->options));
    memset(&fixture->model, 0, sizeof(fixture->model));
    memset(&fixture->set_id, 0, sizeof(fixture->set_id));
    memset(&fixture->record, 0, sizeof(fixture->record));
    memset(&fixture->artifact, 0, sizeof(fixture->artifact));

    fixture->options.context_length = 128U;
    fixture->options.batch_size = 1U;
    fixture->options.generation_limit = 4U;
    fixture->options.thread_count = 1U;
    fixture->options.layer_checkpoints_enabled = 1;
    fixture->options.layer_checkpoints_step = 3U;

    /*
     * Eight layers so the frozen Pass 3 selected subset is a proper subset:
     * {0,1,2,4,6,7} are selected, 3 and 5 are not.
     */
    fixture->model.metadata.config = valid_llama3_config();
    fixture->model.metadata.config.layer_count = 8U;

    expect_status("intra writer set id",
                  lis_artifact_set_id_generate_with_source(
                      &fixture->set_id, deterministic_random_source, NULL),
                  LIS_STATUS_OK);

    expect_status("intra writer record init",
                  lis_layer_trace_record_init(&fixture->record, 4),
                  LIS_STATUS_OK);
    expect_status("intra writer layout",
                  lis_layer_trace_record_configure_llama_layout(
                      &fixture->record, 3U, 8U),
                  LIS_STATUS_OK);
    step = make_layer_output_step(3U, 0U, 1.0f);
    expect_status("intra writer append",
                  lis_layer_trace_record_append(&fixture->record, &step),
                  LIS_STATUS_OK);

    fixture->artifact.path = path;
    fixture->artifact.artifact_set_id = &fixture->set_id;
    fixture->artifact.model_format_name = "safetensors";
    fixture->artifact.model_family_name = "llama3_decoder";
    fixture->artifact.backend_name = "cpu_reference";
    fixture->artifact.precision_path = "bf16";
    fixture->artifact.options = &fixture->options;
    fixture->artifact.model = &fixture->model;
    fixture->artifact.input_mode = LIS_ARTIFACT_INPUT_MODE_TOKENS;
    fixture->artifact.output_mode = LIS_ARTIFACT_OUTPUT_MODE_TOKEN_IDS;
    fixture->artifact.binary_fingerprint.valid = 1;
    fixture->artifact.binary_fingerprint.digest = UINT64_C(1);
    fixture->artifact.binary_fingerprint.size_bytes = 1U;
    fixture->artifact.model_fingerprint = fixture->artifact.binary_fingerprint;
    fixture->artifact.config_fingerprint = fixture->artifact.binary_fingerprint;
    fixture->artifact.input_fingerprint = fixture->artifact.binary_fingerprint;
    fixture->artifact.runtime_fingerprint = fixture->artifact.binary_fingerprint;
    fixture->artifact.backend_fingerprint = fixture->artifact.binary_fingerprint;
    fixture->artifact.intra_layer_record = NULL;
}

static void intra_writer_fixture_destroy(intra_writer_fixture *fixture)
{
    lis_layer_trace_record_destroy(&fixture->record);
}

static size_t intra_read_file(const char *name, const char *path,
                              char *buffer, size_t buffer_size)
{
    FILE *fp = fopen(path, "rb");
    size_t read_bytes;

    buffer[0] = '\0';
    if (fp == NULL) {
        fprintf(stderr, "%s: cannot open %s\n", name, path);
        ++g_failures;
        return 0;
    }
    read_bytes = fread(buffer, 1U, buffer_size - 1U, fp);
    buffer[read_bytes] = '\0';
    (void)fclose(fp);
    return read_bytes;
}

/*
 * Compares a captured artifact against a chunked golden byte sequence. The
 * chunking is purely an ISO C string-literal length concession; the comparison
 * is over the concatenation, byte for byte.
 */
static void intra_expect_golden(const char *name, const char *actual,
                                size_t actual_size,
                                const char *const *chunks,
                                size_t chunk_count)
{
    size_t expected_size = 0;
    size_t offset = 0;
    size_t index;

    for (index = 0; index < chunk_count; ++index) {
        expected_size += strlen(chunks[index]);
    }
    expect_size(name, actual_size, expected_size);
    for (index = 0; index < chunk_count; ++index) {
        size_t length = strlen(chunks[index]);

        if (offset + length > actual_size ||
            memcmp(actual + offset, chunks[index], length) != 0) {
            fprintf(stderr,
                    "%s: diverges at offset %zu; expected \"%s\"\n",
                    name, offset, chunks[index]);
            ++g_failures;
            return;
        }
        offset += length;
    }
}

static int intra_path_exists(const char *path)
{
    FILE *fp = fopen(path, "rb");

    if (fp == NULL) {
        return 0;
    }
    (void)fclose(fp);
    return 1;
}

static char intra_absent_buffer[131072];
static char intra_present_buffer[131072];

static void test_intra_layer_writer_rejects_invalid_record(void)
{
    static const char path[] = "test_intra_writer_reject.json";
    intra_writer_fixture fixture;
    lis_intra_layer_trace_record intra;

    /* Not-ready records: nothing may be created on the target path. */
    expect_status("intra writer reject init",
                  lis_intra_layer_record_init(&intra), LIS_STATUS_OK);
    expect_status("intra writer reject configure",
                  lis_intra_layer_record_configure(&intra, 3U, 1U, 4U, 5U,
                                                   "bf16"),
                  LIS_STATUS_OK);
    intra_writer_fixture_init(&fixture, path);
    (void)remove(path);
    fixture.artifact.intra_layer_record = &intra;
    expect_status("intra writer active record",
                  lis_layer_trace_artifact_write(&fixture.artifact,
                                                 &fixture.record),
                  LIS_STATUS_BAD_STATE);
    expect_size("intra writer active no file",
                (size_t)intra_path_exists(path), 0U);

    lis_intra_layer_record_invalidate(&intra);
    expect_status("intra writer invalid record",
                  lis_layer_trace_artifact_write(&fixture.artifact,
                                                 &fixture.record),
                  LIS_STATUS_BAD_STATE);
    expect_size("intra writer invalid no file",
                (size_t)intra_path_exists(path), 0U);
    intra_writer_fixture_destroy(&fixture);

    /* Cross-object coherence failures, each leaving no file behind. */
#define INTRA_WRITER_INCOHERENT(label, mutate, expected)                     \
    do {                                                                     \
        intra_writer_fixture_init(&fixture, path);                           \
        intra_fill_record("intra writer coherence fill", &intra,             \
                          LIS_INTRA_LAYER_STAGE_COUNT);                      \
        fixture.artifact.intra_layer_record = &intra;                        \
        (void)remove(path);                                                  \
        mutate;                                                              \
        expect_status((label),                                               \
                      lis_layer_trace_artifact_write(&fixture.artifact,      \
                                                     &fixture.record),       \
                      (expected));                                           \
        expect_size((label),(size_t)intra_path_exists(path), 0U);            \
        intra_writer_fixture_destroy(&fixture);                              \
    } while (0)

    INTRA_WRITER_INCOHERENT("intra writer step mismatch",
                            fixture.record.layout_runtime_checkpoint_step = 4U,
                            LIS_STATUS_INVALID_ARGUMENT);
    INTRA_WRITER_INCOHERENT("intra writer layer count mismatch",
                            intra.total_layer_count = 4U,
                            LIS_STATUS_INVALID_ARGUMENT);
    INTRA_WRITER_INCOHERENT("intra writer layout unsupported",
                            fixture.record.checkpoint_layout_supported = 0,
                            LIS_STATUS_INVALID_ARGUMENT);
    INTRA_WRITER_INCOHERENT("intra writer unselected layer",
                            intra.target_layer = 3U,
                            LIS_STATUS_INVALID_ARGUMENT);
    INTRA_WRITER_INCOHERENT("intra writer precision mismatch",
                            fixture.artifact.precision_path = "f32",
                            LIS_STATUS_INVALID_ARGUMENT);
    INTRA_WRITER_INCOHERENT("intra writer precision null",
                            fixture.artifact.precision_path = NULL,
                            LIS_STATUS_INVALID_ARGUMENT);

#undef INTRA_WRITER_INCOHERENT
    (void)remove(path);
}

static void test_intra_layer_writer_absent_bytes_unchanged(void)
{
    /*
     * Golden bytes captured from the pre-P4-4 writer for this exact
     * fixture, chunked only because ISO C limits a single string
     * literal to 4095 bytes. Any perturbation of the existing artifact
     * body -- including the split of the "]}" terminator -- fails here.
     */
    static const char *const expected_absent[] = {
        "{\"schema\":\"lis.execution_artifact/v1\",",
        "\"kind\":\"layer_trace\",\"artifact_set_id\":\"aset1:0001020",
        "30405060708090a0b0c0d0e0f\",\"manifest\":{\"retention_policy",
        "\":{\"absolute_paths\":\"omitted\",\"raw_prompt_text\":\"omi",
        "tted\",\"generated_text\":\"omitted\"},",
        "\"binary\":{\"fingerprint\":{\"algorithm\":\"fnv1a64\",",
        "\"hex\":\"0000000000000001\",\"size_bytes\":1}},",
        "\"model\":{\"format\":\"safetensors\",",
        "\"family\":\"llama3_decoder\",\"fingerprint\":{\"algorithm\"",
        ":\"fnv1a64\",\"hex\":\"0000000000000001\",",
        "\"size_bytes\":1}},\"config\":{\"fingerprint\":{\"algorithm",
        "\":\"fnv1a64\",\"hex\":\"0000000000000001\",",
        "\"size_bytes\":1}},\"input\":{\"mode\":\"tokens\",",
        "\"fingerprint\":{\"algorithm\":\"fnv1a64\",",
        "\"hex\":\"0000000000000001\",\"size_bytes\":1}},",
        "\"runtime\":{\"configured_context\":128,",
        "\"batch_size\":1,\"generation_limit\":4,",
        "\"thread_count\":1,\"layer_checkpoints_enabled\":true,",
        "\"layer_checkpoint_step\":3,\"diagnostics_enabled\":false,",
        "\"perf_enabled\":false,\"perf_per_token_enabled\":false,",
        "\"precision_path\":\"bf16\",\"fingerprint\":{\"algorithm\":",
        "\"fnv1a64\",\"hex\":\"0000000000000001\",",
        "\"size_bytes\":1}},\"backend\":{\"name\":\"cpu_reference\",",
        "\"fingerprint\":{\"algorithm\":\"fnv1a64\",",
        "\"hex\":\"0000000000000001\",\"size_bytes\":1}}},",
        "\"checkpoint_layout\":{\"layout_name\":\"llama_layer_output_",
        "summary\",\"layout_version\":1,\"runtime_checkpoint_step\":3",
        ",\"tensor_role\":\"layer_output\",\"stage_order\":0,",
        "\"ordering_semantics\":\"runtime_step_layer_stage_ordinal\",",
        "\"total_layer_count\":8,\"requested_coordinates\":[{\"runtim",
        "e_checkpoint_step\":3,\"layer_index\":0,",
        "\"tensor_role\":\"layer_output\",\"batch_index\":0,",
        "\"sequence_index\":0,\"stage_order\":0,",
        "\"execution_ordinal\":0},{\"runtime_checkpoint_step\":3,",
        "\"layer_index\":1,\"tensor_role\":\"layer_output\",",
        "\"batch_index\":0,\"sequence_index\":0,",
        "\"stage_order\":0,\"execution_ordinal\":1},{\"runtime_checkp",
        "oint_step\":3,\"layer_index\":2,\"tensor_role\":\"layer_outp",
        "ut\",\"batch_index\":0,\"sequence_index\":0,",
        "\"stage_order\":0,\"execution_ordinal\":2},{\"runtime_checkp",
        "oint_step\":3,\"layer_index\":4,\"tensor_role\":\"layer_outp",
        "ut\",\"batch_index\":0,\"sequence_index\":0,",
        "\"stage_order\":0,\"execution_ordinal\":3},{\"runtime_checkp",
        "oint_step\":3,\"layer_index\":6,\"tensor_role\":\"layer_outp",
        "ut\",\"batch_index\":0,\"sequence_index\":0,",
        "\"stage_order\":0,\"execution_ordinal\":4},{\"runtime_checkp",
        "oint_step\":3,\"layer_index\":7,\"tensor_role\":\"layer_outp",
        "ut\",\"batch_index\":0,\"sequence_index\":0,",
        "\"stage_order\":0,\"execution_ordinal\":5}],",
        "\"captured_coordinates\":[{\"runtime_checkpoint_step\":3,",
        "\"layer_index\":0,\"tensor_role\":\"layer_output\",",
        "\"batch_index\":0,\"sequence_index\":0,",
        "\"stage_order\":0,\"execution_ordinal\":0}],",
        "\"missing_coordinates\":[{\"coordinate\":{\"runtime_checkpoi",
        "nt_step\":3,\"layer_index\":1,\"tensor_role\":\"layer_output",
        "\",\"batch_index\":0,\"sequence_index\":0,",
        "\"stage_order\":0,\"execution_ordinal\":1},",
        "\"state\":\"not_captured\",\"detail\":\"target_checkpoint_no",
        "t_observed\"},{\"coordinate\":{\"runtime_checkpoint_step\":3",
        ",\"layer_index\":2,\"tensor_role\":\"layer_output\",",
        "\"batch_index\":0,\"sequence_index\":0,",
        "\"stage_order\":0,\"execution_ordinal\":2},",
        "\"state\":\"not_captured\",\"detail\":\"target_checkpoint_no",
        "t_observed\"},{\"coordinate\":{\"runtime_checkpoint_step\":3",
        ",\"layer_index\":4,\"tensor_role\":\"layer_output\",",
        "\"batch_index\":0,\"sequence_index\":0,",
        "\"stage_order\":0,\"execution_ordinal\":3},",
        "\"state\":\"not_captured\",\"detail\":\"target_checkpoint_no",
        "t_observed\"},{\"coordinate\":{\"runtime_checkpoint_step\":3",
        ",\"layer_index\":6,\"tensor_role\":\"layer_output\",",
        "\"batch_index\":0,\"sequence_index\":0,",
        "\"stage_order\":0,\"execution_ordinal\":4},",
        "\"state\":\"not_captured\",\"detail\":\"target_checkpoint_no",
        "t_observed\"},{\"coordinate\":{\"runtime_checkpoint_step\":3",
        ",\"layer_index\":7,\"tensor_role\":\"layer_output\",",
        "\"batch_index\":0,\"sequence_index\":0,",
        "\"stage_order\":0,\"execution_ordinal\":5},",
        "\"state\":\"not_captured\",\"detail\":\"target_checkpoint_no",
        "t_observed\"}],\"available_summary_fields\":[\"min\",",
        "\"max\",\"mean\",\"l2\",\"nan\",\"inf\",",
        "\"digest\"],\"digest_contract\":{\"algorithm\":\"sha256\",",
        "\"version\":\"lis.checkpoint.fp32le/v1\",",
        "\"observed_dtype\":\"fp32\",\"byte_order\":\"little\",",
        "\"canonicalization\":\"ieee754-binary32-le;canonical-qnan;pr",
        "eserve-signed-zero\"},\"duplicate_coordinate_policy\":\"reje",
        "ct_artifact_before_write\"},\"layer_trace\":[{\"step\":3,",
        "\"phase\":\"decode\",\"name\":\"layer.0.output\",",
        "\"shape\":[1],\"min\":1,\"max\":1,\"mean\":1,",
        "\"l2\":1,\"nan\":0,\"inf\":0,\"runtime_checkpoint_step\":3,",
        "\"layer_index\":0,\"tensor_role\":\"layer_output\",",
        "\"batch_index\":0,\"sequence_index\":0,",
        "\"stage_order\":0,\"execution_ordinal\":0,",
        "\"observed_dtype\":\"fp32\",\"element_count\":1,",
        "\"available_summary_fields\":[\"min\",",
        "\"max\",\"mean\",\"l2\",\"nan\",\"inf\",",
        "\"digest\"],\"digest\":{\"algorithm\":\"sha256\",",
        "\"version\":\"lis.checkpoint.fp32le/v1\",",
        "\"tensor_role\":\"layer_output\",\"shape\":[1],",
        "\"observed_dtype\":\"fp32\",\"byte_order\":\"little\",",
        "\"canonicalization\":\"ieee754-binary32-le;canonical-qnan;pr",
        "eserve-signed-zero\",\"value\":\"sha256:cdaac9a2e6cc308d7cc2",
        "564d24086969b6e791408406c048f2e6fccd38543d2e\"}}]}"
    };

    static const char path[] = "test_intra_writer_absent.json";
    intra_writer_fixture fixture;
    size_t absent_size;

    intra_writer_fixture_init(&fixture, path);
    (void)remove(path);
    expect_status("intra absent write",
                  lis_layer_trace_artifact_write(&fixture.artifact,
                                                 &fixture.record),
                  LIS_STATUS_OK);
    absent_size = intra_read_file("intra absent read", path,
                                  intra_absent_buffer,
                                  sizeof(intra_absent_buffer));
    intra_writer_fixture_destroy(&fixture);
    (void)remove(path);

    /* No new key appears anywhere in absent mode. */
    intra_expect_absent("intra absent no new keys", intra_absent_buffer,
                        "intra_layer");
    intra_expect_golden("intra absent golden", intra_absent_buffer,
                        absent_size, expected_absent,
                        sizeof(expected_absent) /
                            sizeof(expected_absent[0]));
}

static void test_intra_layer_writer_additive_insertion(void)
{
    static const char path[] = "test_intra_writer_present.json";
    intra_writer_fixture fixture;
    lis_intra_layer_trace_record intra;
    size_t absent_size;
    size_t present_size;
    size_t blocks_size;

    absent_size = strlen(intra_absent_buffer);
    if (absent_size == 0U) {
        fprintf(stderr, "intra additive: absent reference missing\n");
        ++g_failures;
        return;
    }

    intra_fill_record("intra additive fill", &intra,
                      LIS_INTRA_LAYER_STAGE_COUNT);
    intra_writer_fixture_init(&fixture, path);
    fixture.artifact.intra_layer_record = &intra;
    (void)remove(path);
    expect_status("intra additive write",
                  lis_layer_trace_artifact_write(&fixture.artifact,
                                                 &fixture.record),
                  LIS_STATUS_OK);
    present_size = intra_read_file("intra additive read", path,
                                   intra_present_buffer,
                                   sizeof(intra_present_buffer));
    intra_writer_fixture_destroy(&fixture);
    (void)remove(path);
    if (present_size == 0U) {
        return;
    }

    /* Capture the module's own blocks for the exact insertion comparison. */
    blocks_size = intra_capture_json("intra additive blocks", &intra,
                                     LIS_STATUS_OK);
    lis_intra_layer_record_destroy(&intra);
    if (blocks_size == 0U) {
        return;
    }

    /*
     * with_record == absent_without_final_brace + blocks + final_brace
     */
    expect_size("intra additive size", present_size,
                absent_size - 1U + blocks_size + 1U);
    if (intra_absent_buffer[absent_size - 1U] != '}' ||
        intra_present_buffer[present_size - 1U] != '}') {
        fprintf(stderr, "intra additive: artifact does not end with '}'\n");
        ++g_failures;
        return;
    }
    if (strncmp(intra_present_buffer, intra_absent_buffer,
                absent_size - 1U) != 0) {
        fprintf(stderr, "intra additive: pre-existing bytes were perturbed\n");
        ++g_failures;
    }
    if (strncmp(intra_present_buffer + (absent_size - 1U), intra_json_buffer,
                blocks_size) != 0) {
        fprintf(stderr, "intra additive: inserted blocks differ from the "
                        "module output\n");
        ++g_failures;
    }
    /* The new content is strictly appended after layer_trace[]. */
    intra_expect_contains("intra additive junction", intra_present_buffer,
                          "\"}}],\"intra_layer_checkpoint_layout\":{");
}

/*
 * The validator accepts every byte >= 0x20 except DEL, so '"' and '\\' are
 * legal in caller-supplied identifiers and details and must be JSON-escaped on
 * output.
 *
 * This is asserted on BOTH paths deliberately. The module-level tests inject
 * their own hook pair, so without a production-writer leg the real escaper's
 * quote and backslash branches are never executed by any P4-4 test, and a
 * regression in either escaper — or a divergence between them — would go
 * unnoticed. Running one record through both and demanding the same escaped
 * bytes is what makes this test resistant to that.
 */
static void test_intra_layer_json_escapes_caller_strings(void)
{
    static const char quoted_path[] = "a\"b\\c";
    static const char quoted_detail[] = "unavailable:\"quoted\"\\path";
    static const char expected_path_bytes[] =
        "\"precision_path\":\"a\\\"b\\\\c\"";
    static const char expected_detail_bytes[] =
        "\"detail\":\"unavailable:\\\"quoted\\\"\\\\path\"";
    static const char unescaped_path_bytes[] = "\"precision_path\":\"a\"b";
    static const char path[] = "test_intra_writer_escape.json";
    lis_intra_layer_trace_record record;
    intra_writer_fixture fixture;
    size_t index;

    expect_status("intra escape init", lis_intra_layer_record_init(&record),
                  LIS_STATUS_OK);
    expect_status("intra escape configure",
                  lis_intra_layer_record_configure(&record, 3U, 1U, 8U, 5U,
                                                   quoted_path),
                  LIS_STATUS_OK);
    intra_expect_string("intra escape stored path", record.precision_path,
                        quoted_path);
    for (index = 0; index < LIS_INTRA_LAYER_STAGE_COUNT; ++index) {
        if (index == 3U) {
            expect_status("intra escape mark",
                          lis_intra_layer_record_mark_unavailable(
                              &record, (lis_intra_layer_stage)index,
                              LIS_INTRA_LAYER_MISSING_UNSUPPORTED,
                              quoted_detail),
                          LIS_STATUS_OK);
        } else {
            lis_intra_layer_observation observation =
                intra_make_observation((lis_intra_layer_stage)index, 3U, 1U,
                                       5U);

            expect_status("intra escape append",
                          lis_intra_layer_record_append_observation(&record,
                                                                    &observation),
                          LIS_STATUS_OK);
        }
    }
    expect_status("intra escape finalize",
                  lis_intra_layer_record_finalize(&record), LIS_STATUS_OK);

    /* Leg 1: the module writer, through the injected test hooks. */
    if (intra_capture_json("intra escape module json", &record,
                           LIS_STATUS_OK) != 0U) {
        intra_expect_contains("intra escape module path", intra_json_buffer,
                              expected_path_bytes);
        intra_expect_contains("intra escape module detail", intra_json_buffer,
                              expected_detail_bytes);
        intra_expect_absent("intra escape module raw quote", intra_json_buffer,
                            unescaped_path_bytes);
    }

    /* Leg 2: the production layer-trace writer, through its own escaper. */
    intra_writer_fixture_init(&fixture, path);
    fixture.artifact.precision_path = quoted_path;
    fixture.artifact.intra_layer_record = &record;
    (void)remove(path);
    expect_status("intra escape write",
                  lis_layer_trace_artifact_write(&fixture.artifact,
                                                 &fixture.record),
                  LIS_STATUS_OK);
    if (intra_read_file("intra escape read", path, intra_json_buffer,
                        sizeof(intra_json_buffer)) != 0U) {
        intra_expect_contains("intra escape writer path", intra_json_buffer,
                              expected_path_bytes);
        intra_expect_contains("intra escape writer detail", intra_json_buffer,
                              expected_detail_bytes);
        intra_expect_absent("intra escape writer raw quote", intra_json_buffer,
                            unescaped_path_bytes);
    }
    intra_writer_fixture_destroy(&fixture);
    (void)remove(path);
    lis_intra_layer_record_destroy(&record);
}

static void test_intra_layer_null_handling(void)
{
    expect_status("intra init null",
                  lis_intra_layer_record_init(NULL),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_status("intra configure null record",
                  lis_intra_layer_record_configure(NULL, 1U, 0U, 1U, 0U,
                                                   "bf16"),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_size("intra is_ready null",
                (size_t)lis_intra_layer_record_is_ready(NULL), 0U);
    intra_expect_state("intra get_state null", NULL,
                       LIS_INTRA_LAYER_RECORD_INVALID);
    /* Safe no-ops. */
    lis_intra_layer_record_invalidate(NULL);
    lis_intra_layer_record_destroy(NULL);
}

static void test_intra_layer_runtime_fingerprint_identity(void)
{
    lis_cli_options options = {0};
    lis_artifact_fingerprint absent = {0};
    lis_artifact_fingerprint absent_with_ignored_target = {0};
    lis_artifact_fingerprint target_zero = {0};
    lis_artifact_fingerprint target_zero_repeat = {0};
    lis_artifact_fingerprint target_one = {0};
    const size_t enabled_size =
        157U + 2U * sizeof(uint64_t) +
        sizeof(LIS_INTRA_LAYER_DIAGNOSTIC_CAPTURE_PROFILE);

    options.context_length = 8U;
    options.batch_size = 1U;
    options.generation_limit = 2U;
    options.thread_count = 1U;
    options.layer_checkpoints_enabled = 1;
    options.layer_checkpoints_step = 1U;
    expect_status(
        "intra fingerprint absent",
        lis_artifact_fingerprint_runtime(
            &options, LIS_MODEL_FORMAT_HUGGINGFACE_LOCAL,
            LIS_MODEL_FAMILY_LLAMA3_DECODER,
            LIS_ARTIFACT_INPUT_MODE_TOKENS, "cpu_reference", &absent),
        LIS_STATUS_OK);
    if (absent.digest != UINT64_C(0x79f32118f17592b7) ||
        absent.size_bytes != 157U) {
        fprintf(stderr,
                "intra absent fingerprint drifted: %016" PRIx64 " %zu\n",
                absent.digest, absent.size_bytes);
        ++g_failures;
    }

    options.intra_layer_target_layer = 9U;
    expect_status(
        "intra fingerprint absent ignores target",
        lis_artifact_fingerprint_runtime(
            &options, LIS_MODEL_FORMAT_HUGGINGFACE_LOCAL,
            LIS_MODEL_FAMILY_LLAMA3_DECODER,
            LIS_ARTIFACT_INPUT_MODE_TOKENS, "cpu_reference",
            &absent_with_ignored_target),
        LIS_STATUS_OK);
    if (memcmp(&absent, &absent_with_ignored_target, sizeof(absent)) != 0) {
        fprintf(stderr, "disabled intra target changed runtime identity\n");
        ++g_failures;
    }

    options.intra_layer_checkpoints_enabled = 1;
    options.intra_layer_target_layer = 0U;
    expect_status(
        "intra fingerprint target zero",
        lis_artifact_fingerprint_runtime(
            &options, LIS_MODEL_FORMAT_HUGGINGFACE_LOCAL,
            LIS_MODEL_FAMILY_LLAMA3_DECODER,
            LIS_ARTIFACT_INPUT_MODE_TOKENS, "cpu_reference", &target_zero),
        LIS_STATUS_OK);
    expect_status(
        "intra fingerprint target zero repeat",
        lis_artifact_fingerprint_runtime(
            &options, LIS_MODEL_FORMAT_HUGGINGFACE_LOCAL,
            LIS_MODEL_FAMILY_LLAMA3_DECODER,
            LIS_ARTIFACT_INPUT_MODE_TOKENS, "cpu_reference",
            &target_zero_repeat),
        LIS_STATUS_OK);
    expect_size("intra fingerprint enabled size", target_zero.size_bytes,
                enabled_size);
    if (memcmp(&target_zero, &target_zero_repeat,
               sizeof(target_zero)) != 0 ||
        target_zero.digest == absent.digest) {
        fprintf(stderr, "enabled intra runtime identity was not repeatable\n");
        ++g_failures;
    }

    options.intra_layer_target_layer = 1U;
    expect_status(
        "intra fingerprint target one",
        lis_artifact_fingerprint_runtime(
            &options, LIS_MODEL_FORMAT_HUGGINGFACE_LOCAL,
            LIS_MODEL_FAMILY_LLAMA3_DECODER,
            LIS_ARTIFACT_INPUT_MODE_TOKENS, "cpu_reference", &target_one),
        LIS_STATUS_OK);
    if (target_one.digest == target_zero.digest ||
        target_one.size_bytes != target_zero.size_bytes) {
        fprintf(stderr, "intra target layer did not separate runtime identity\n");
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
    test_generic_sha256_vectors();
    test_artifact_set_id_lifecycle();
    test_layer_trace_coordinate_guards();
    test_intra_layer_stage_taxonomy();
    test_intra_layer_stage_lookup_rejects_unknown();
    test_intra_layer_record_configure_guards();
    test_intra_layer_fp32_view_validation();
    test_intra_layer_checkpoint_digest_vectors();
    test_intra_layer_checkpoint_digest_guards();
    test_intra_layer_checkpoint_digest_striding();
    test_intra_layer_append_ordering_and_duplicates();
    test_intra_layer_append_coordinate_guards();
    test_intra_layer_append_payload_guards();
    test_intra_layer_mark_unavailable_guards();
    test_intra_layer_finalize_partition();
    test_intra_layer_sticky_invalidity();
    test_intra_layer_json_serialization();
    test_intra_layer_json_mixed_and_discipline();
    test_intra_layer_json_rejects_unready();
    test_intra_layer_digest_is_carried_not_computed();
    test_intra_layer_writer_rejects_invalid_record();
    test_intra_layer_writer_absent_bytes_unchanged();
    test_intra_layer_writer_additive_insertion();
    test_intra_layer_json_escapes_caller_strings();
    test_intra_layer_null_handling();
    test_intra_layer_runtime_fingerprint_identity();

    if (g_failures != 0) {
        fprintf(stderr, "%d core test failure(s)\n", g_failures);
        return 1;
    }

    return 0;
}
