#define _POSIX_C_SOURCE 200809L

#include "lis/cli.h"
#include "lis/artifact.h"

#include "lis/status.h"
#include "lis/tokenizer.h"

#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>

static int g_failures;

static void expect_int(const char *name, int actual, int expected)
{
    if (actual != expected) {
        fprintf(stderr, "%s: expected %d, got %d\n", name, expected, actual);
        ++g_failures;
    }
}

static void expect_status(const char *name, lis_status actual,
                          lis_status expected)
{
    if (actual != expected) {
        fprintf(stderr, "%s: expected %s, got %s\n", name,
                lis_status_name(expected), lis_status_name(actual));
        ++g_failures;
    }
}

static void expect_string_equals(const char *name, const char *actual,
                                 size_t actual_len, const char *expected)
{
    const size_t expected_len = strlen(expected);

    if (actual == NULL || actual_len != expected_len ||
        memcmp(actual, expected, expected_len) != 0) {
        fprintf(stderr, "%s: expected \"%s\", got \"%s\"\n",
                name, expected, actual != NULL ? actual : "(null)");
        ++g_failures;
    }
}

static void expect_file_empty(const char *name, const char *path)
{
    FILE *fp = fopen(path, "rb");
    long file_size = -1;

    if (fp == NULL) {
        fprintf(stderr, "%s: could not open %s\n", name, path);
        ++g_failures;
        return;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "%s: could not seek %s\n", name, path);
        ++g_failures;
        fclose(fp);
        return;
    }
    file_size = ftell(fp);
    if (file_size != 0) {
        fprintf(stderr, "%s: expected empty file %s, got %ld bytes\n",
                name, path, file_size);
        ++g_failures;
    }
    fclose(fp);
}

static void expect_file_missing(const char *name, const char *path)
{
    if (access(path, F_OK) == 0) {
        fprintf(stderr, "%s: expected missing file %s\n", name, path);
        ++g_failures;
    }
}

static void expect_file_contains(const char *name, const char *path,
                                 const char *needle)
{
    FILE *fp = fopen(path, "rb");
    long file_size = 0;
    char *data = NULL;

    if (fp == NULL) {
        fprintf(stderr, "%s: could not open %s\n", name, path);
        ++g_failures;
        return;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "%s: could not seek %s\n", name, path);
        ++g_failures;
        fclose(fp);
        return;
    }
    file_size = ftell(fp);
    if (file_size < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "%s: could not size %s\n", name, path);
        ++g_failures;
        fclose(fp);
        return;
    }
    data = malloc((size_t)file_size + 1U);
    if (data == NULL) {
        fprintf(stderr, "%s: allocation failed\n", name);
        ++g_failures;
        fclose(fp);
        return;
    }
    if (fread(data, 1, (size_t)file_size, fp) != (size_t)file_size) {
        fprintf(stderr, "%s: could not read %s\n", name, path);
        ++g_failures;
        free(data);
        fclose(fp);
        return;
    }
    data[(size_t)file_size] = '\0';
    if (strstr(data, needle) == NULL) {
        fprintf(stderr, "%s: expected %s to contain \"%s\"\n",
                name, path, needle);
        ++g_failures;
    }
    free(data);
    fclose(fp);
}

static void expect_file_occurrences(const char *name, const char *path,
                                    const char *needle, size_t expected)
{
    FILE *fp = fopen(path, "rb");
    long file_size = 0;
    char *data = NULL;
    char *cursor;
    size_t actual = 0;
    const size_t needle_len = strlen(needle);

    if (fp == NULL) {
        fprintf(stderr, "%s: could not open %s\n", name, path);
        ++g_failures;
        return;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "%s: could not seek %s\n", name, path);
        ++g_failures;
        fclose(fp);
        return;
    }
    file_size = ftell(fp);
    if (file_size < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "%s: could not size %s\n", name, path);
        ++g_failures;
        fclose(fp);
        return;
    }
    data = malloc((size_t)file_size + 1U);
    if (data == NULL) {
        fprintf(stderr, "%s: allocation failed\n", name);
        ++g_failures;
        fclose(fp);
        return;
    }
    if (fread(data, 1, (size_t)file_size, fp) != (size_t)file_size) {
        fprintf(stderr, "%s: could not read %s\n", name, path);
        ++g_failures;
        free(data);
        fclose(fp);
        return;
    }
    data[(size_t)file_size] = '\0';
    cursor = data;
    while (needle_len > 0 && (cursor = strstr(cursor, needle)) != NULL) {
        ++actual;
        cursor += needle_len;
    }
    if (actual != expected) {
        fprintf(stderr, "%s: expected %s to contain \"%s\" %zu times, got %zu\n",
                name, path, needle, expected, actual);
        ++g_failures;
    }
    free(data);
    fclose(fp);
}

static void expect_file_equals(const char *name, const char *path,
                               const char *expected)
{
    FILE *fp = fopen(path, "rb");
    long file_size = 0;
    char *data = NULL;
    const size_t expected_len = strlen(expected);

    if (fp == NULL) {
        fprintf(stderr, "%s: could not open %s\n", name, path);
        ++g_failures;
        return;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "%s: could not seek %s\n", name, path);
        ++g_failures;
        fclose(fp);
        return;
    }
    file_size = ftell(fp);
    if (file_size < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "%s: could not size %s\n", name, path);
        ++g_failures;
        fclose(fp);
        return;
    }
    data = malloc((size_t)file_size + 1U);
    if (data == NULL) {
        fprintf(stderr, "%s: allocation failed\n", name);
        ++g_failures;
        fclose(fp);
        return;
    }
    if (fread(data, 1, (size_t)file_size, fp) != (size_t)file_size) {
        fprintf(stderr, "%s: could not read %s\n", name, path);
        ++g_failures;
        free(data);
        fclose(fp);
        return;
    }
    data[(size_t)file_size] = '\0';
    if ((size_t)file_size != expected_len ||
        memcmp(data, expected, expected_len) != 0) {
        fprintf(stderr, "%s: expected %s to equal \"%s\", got \"%s\"\n",
                name, path, expected, data);
        ++g_failures;
    }
    free(data);
    fclose(fp);
}

static char *read_file_content(const char *path)
{
    FILE *fp = fopen(path, "rb");
    long file_size = 0;
    char *data = NULL;

    if (fp == NULL) {
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    file_size = ftell(fp);
    if (file_size < 0 || fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return NULL;
    }
    data = malloc((size_t)file_size + 1U);
    if (data == NULL) {
        fclose(fp);
        return NULL;
    }
    if (fread(data, 1, (size_t)file_size, fp) != (size_t)file_size) {
        free(data);
        fclose(fp);
        return NULL;
    }
    data[(size_t)file_size] = '\0';
    fclose(fp);
    return data;
}

static void expect_file_not_contains(const char *name, const char *path,
                                     const char *needle)
{
    char *data = read_file_content(path);

    if (data == NULL) {
        return;
    }
    if (strstr(data, needle) != NULL) {
        fprintf(stderr, "%s: expected %s to NOT contain \"%s\"\n",
                name, path, needle);
        ++g_failures;
    }
    free(data);
}

static int run_cli_capture(int argc, char **argv, const char *stdout_path,
                           const char *stderr_path)
{
    int saved_stdout = -1;
    int saved_stderr = -1;
    int stdout_fd = -1;
    int stderr_fd = -1;
    int result = -1;

    fflush(NULL);
    saved_stdout = dup(STDOUT_FILENO);
    saved_stderr = dup(STDERR_FILENO);
    stdout_fd = open(stdout_path, O_CREAT | O_TRUNC | O_WRONLY, 0600);
    stderr_fd = open(stderr_path, O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (saved_stdout < 0 || saved_stderr < 0 ||
        stdout_fd < 0 || stderr_fd < 0 ||
        dup2(stdout_fd, STDOUT_FILENO) < 0 ||
        dup2(stderr_fd, STDERR_FILENO) < 0) {
        goto out;
    }

    result = lis_cli_run(argc, argv);

out:
    fflush(NULL);
    if (saved_stdout >= 0) {
        dup2(saved_stdout, STDOUT_FILENO);
    }
    if (saved_stderr >= 0) {
        dup2(saved_stderr, STDERR_FILENO);
    }
    if (stdout_fd >= 0) {
        close(stdout_fd);
    }
    if (stderr_fd >= 0) {
        close(stderr_fd);
    }
    if (saved_stdout >= 0) {
        close(saved_stdout);
    }
    if (saved_stderr >= 0) {
        close(saved_stderr);
    }
    if (result < 0) {
        fprintf(stderr, "cli capture setup failed\n");
        ++g_failures;
    }
    return result;
}

static void write_u64_le(FILE *fp, uint64_t value)
{
    size_t index;

    for (index = 0; index < 8; ++index) {
        fputc((int)((value >> (index * 8)) & 0xffU), fp);
    }
}

static lis_status write_text_file(const char *path, const char *text)
{
    FILE *fp = fopen(path, "wb");
    const size_t len = strlen(text);

    if (fp == NULL) {
        return LIS_STATUS_IO;
    }
    if (fwrite(text, 1, len, fp) != len) {
        fclose(fp);
        return LIS_STATUS_IO;
    }
    if (fclose(fp) != 0) {
        return LIS_STATUS_IO;
    }

    return LIS_STATUS_OK;
}

static lis_status write_safetensors_file(const char *path)
{
    const char *header =
        "{\"lis.validation_logits\":{\"dtype\":\"F32\",\"shape\":[1,4],"
        "\"data_offsets\":[0,16]}}";
    const float logits[4] = { -1.0f, 1.0f, 5.0f, 2.0f };
    FILE *fp = fopen(path, "wb");

    if (fp == NULL) {
        return LIS_STATUS_IO;
    }
    write_u64_le(fp, (uint64_t)strlen(header));
    if (fwrite(header, 1, strlen(header), fp) != strlen(header)) {
        fclose(fp);
        return LIS_STATUS_IO;
    }
    if (fwrite(logits, 1, sizeof(logits), fp) != sizeof(logits)) {
        fclose(fp);
        return LIS_STATUS_IO;
    }
    if (fclose(fp) != 0) {
        return LIS_STATUS_IO;
    }

    return LIS_STATUS_OK;
}

static lis_status write_safetensors_file_with_data(const char *path,
                                                   const char *header,
                                                   const void *data,
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

static int append_safetensors_header_tensor(char *header,
                                            size_t header_cap,
                                            size_t *header_len,
                                            const char *name,
                                            const char *shape,
                                            size_t value_count,
                                            size_t *data_offset,
                                            int *first)
{
    size_t byte_count;
    size_t start;
    size_t end;
    int written;

    if (header == NULL || header_len == NULL || name == NULL ||
        shape == NULL || data_offset == NULL || first == NULL ||
        value_count > SIZE_MAX / sizeof(float) ||
        *header_len >= header_cap) {
        return 0;
    }
    byte_count = value_count * sizeof(float);
    if (*data_offset > SIZE_MAX - byte_count) {
        return 0;
    }
    start = *data_offset;
    end = start + byte_count;
    written = snprintf(header + *header_len, header_cap - *header_len,
                       "%s\"%s\":{\"dtype\":\"F32\",\"shape\":%s,"
                       "\"data_offsets\":[%zu,%zu]}",
                       *first ? "" : ",", name, shape, start, end);
    if (written < 0 || (size_t)written >= header_cap - *header_len) {
        return 0;
    }
    *header_len += (size_t)written;
    *data_offset = end;
    *first = 0;
    return 1;
}

static lis_status write_llama_checkpoint_fixture_with_embeddings(
    const char *path,
    size_t layer_count,
    float embedding0,
    float embedding1,
    float embedding2)
{
    const size_t header_cap = 65536U;
    size_t value_count;
    char *header = NULL;
    float *data = NULL;
    size_t header_len = 0;
    size_t data_offset = 0;
    size_t data_index = 0;
    size_t layer;
    int first = 1;
    lis_status status;

    if (path == NULL || layer_count == 0 ||
        layer_count > (SIZE_MAX - 7U) / 9U) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    value_count = 3U + layer_count * 9U + 1U + 3U;
    header = malloc(header_cap);
    data = calloc(value_count, sizeof(*data));
    if (header == NULL || data == NULL) {
        free(header);
        free(data);
        return LIS_STATUS_NO_MEMORY;
    }
    header[header_len++] = '{';

    if (!append_safetensors_header_tensor(header, header_cap, &header_len,
                                          "model.embed_tokens.weight", "[3,1]",
                                          3, &data_offset, &first)) {
        free(header);
        free(data);
        return LIS_STATUS_OVERFLOW;
    }
    data[data_index++] = embedding0;
    data[data_index++] = embedding1;
    data[data_index++] = embedding2;

    for (layer = 0; layer < layer_count; ++layer) {
        char name[128];

        snprintf(name, sizeof(name),
                 "model.layers.%zu.self_attn.q_proj.weight", layer);
        if (!append_safetensors_header_tensor(header, header_cap, &header_len,
                                              name, "[1,1]", 1,
                                              &data_offset, &first)) {
            free(header);
            free(data);
            return LIS_STATUS_OVERFLOW;
        }
        ++data_index;
        snprintf(name, sizeof(name),
                 "model.layers.%zu.self_attn.k_proj.weight", layer);
        if (!append_safetensors_header_tensor(header, header_cap, &header_len,
                                              name, "[1,1]", 1,
                                              &data_offset, &first)) {
            free(header);
            free(data);
            return LIS_STATUS_OVERFLOW;
        }
        ++data_index;
        snprintf(name, sizeof(name),
                 "model.layers.%zu.self_attn.v_proj.weight", layer);
        if (!append_safetensors_header_tensor(header, header_cap, &header_len,
                                              name, "[1,1]", 1,
                                              &data_offset, &first)) {
            free(header);
            free(data);
            return LIS_STATUS_OVERFLOW;
        }
        ++data_index;
        snprintf(name, sizeof(name),
                 "model.layers.%zu.self_attn.o_proj.weight", layer);
        if (!append_safetensors_header_tensor(header, header_cap, &header_len,
                                              name, "[1,1]", 1,
                                              &data_offset, &first)) {
            free(header);
            free(data);
            return LIS_STATUS_OVERFLOW;
        }
        ++data_index;
        snprintf(name, sizeof(name),
                 "model.layers.%zu.mlp.gate_proj.weight", layer);
        if (!append_safetensors_header_tensor(header, header_cap, &header_len,
                                              name, "[1,1]", 1,
                                              &data_offset, &first)) {
            free(header);
            free(data);
            return LIS_STATUS_OVERFLOW;
        }
        ++data_index;
        snprintf(name, sizeof(name),
                 "model.layers.%zu.mlp.up_proj.weight", layer);
        if (!append_safetensors_header_tensor(header, header_cap, &header_len,
                                              name, "[1,1]", 1,
                                              &data_offset, &first)) {
            free(header);
            free(data);
            return LIS_STATUS_OVERFLOW;
        }
        ++data_index;
        snprintf(name, sizeof(name),
                 "model.layers.%zu.mlp.down_proj.weight", layer);
        if (!append_safetensors_header_tensor(header, header_cap, &header_len,
                                              name, "[1,1]", 1,
                                              &data_offset, &first)) {
            free(header);
            free(data);
            return LIS_STATUS_OVERFLOW;
        }
        ++data_index;
        snprintf(name, sizeof(name),
                 "model.layers.%zu.input_layernorm.weight", layer);
        if (!append_safetensors_header_tensor(header, header_cap, &header_len,
                                              name, "[1]", 1,
                                              &data_offset, &first)) {
            free(header);
            free(data);
            return LIS_STATUS_OVERFLOW;
        }
        data[data_index++] = 1.0f;
        snprintf(name, sizeof(name),
                 "model.layers.%zu.post_attention_layernorm.weight", layer);
        if (!append_safetensors_header_tensor(header, header_cap, &header_len,
                                              name, "[1]", 1,
                                              &data_offset, &first)) {
            free(header);
            free(data);
            return LIS_STATUS_OVERFLOW;
        }
        data[data_index++] = 1.0f;
    }

    if (!append_safetensors_header_tensor(header, header_cap, &header_len,
                                          "model.norm.weight", "[1]", 1,
                                          &data_offset, &first)) {
        free(header);
        free(data);
        return LIS_STATUS_OVERFLOW;
    }
    data[data_index++] = 1.0f;
    if (!append_safetensors_header_tensor(header, header_cap, &header_len,
                                          "lm_head.weight", "[3,1]", 3,
                                          &data_offset, &first)) {
        free(header);
        free(data);
        return LIS_STATUS_OVERFLOW;
    }
    data[data_index++] = 0.9f;
    data[data_index++] = 0.2f;
    data[data_index++] = 0.1f;
    if (header_len + 1U >= header_cap) {
        free(header);
        free(data);
        return LIS_STATUS_OVERFLOW;
    }
    header[header_len++] = '}';
    header[header_len] = '\0';

    status = write_safetensors_file_with_data(path, header, data,
                                              value_count * sizeof(*data));
    free(header);
    free(data);
    return status;
}

static lis_status write_llama_checkpoint_fixture(const char *path,
                                                 size_t layer_count)
{
    return write_llama_checkpoint_fixture_with_embeddings(
        path, layer_count, 1.0f, 2.0f, 3.0f);
}

static void test_cli_llama_instruct_prompt_builder(void)
{
    char *prompt = NULL;
    size_t prompt_len = 0;
    const char *expected_empty =
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    const char *expected_newline =
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "line1\nline2"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
    const char *expected_utf8 =
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "caf\303\251"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";

    expect_status("prompt builder empty",
                  lis_cli_build_llama_instruct_prompt("", &prompt,
                                                      &prompt_len),
                  LIS_STATUS_OK);
    expect_string_equals("prompt builder empty bytes", prompt, prompt_len,
                         expected_empty);
    free(prompt);
    prompt = NULL;

    expect_status("prompt builder newline",
                  lis_cli_build_llama_instruct_prompt("line1\nline2",
                                                      &prompt, &prompt_len),
                  LIS_STATUS_OK);
    expect_string_equals("prompt builder newline bytes", prompt, prompt_len,
                         expected_newline);
    free(prompt);
    prompt = NULL;

    expect_status("prompt builder ordinary bytes",
                  lis_cli_build_llama_instruct_prompt("caf\303\251", &prompt,
                                                      &prompt_len),
                  LIS_STATUS_OK);
    expect_string_equals("prompt builder ordinary bytes", prompt, prompt_len,
                         expected_utf8);
    free(prompt);
}

static void write_valid_fixtures(const char *model_path,
                                 const char *config_path,
                                 const char *token_path)
{
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":4,\"intermediate_size\":8,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":4,\"vocab_size\":4,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";

    expect_status("write cli model fixture",
                  write_safetensors_file(model_path), LIS_STATUS_OK);
    expect_status("write cli config fixture",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("write cli tokens fixture",
                  write_text_file(token_path, "0 1\n2\n"), LIS_STATUS_OK);
}

static void test_cli_help(void)
{
    char *argv[] = { "lis", "--help" };

    expect_int("cli help", lis_cli_run(2, argv), 0);
}

static void test_cli_invalid_arguments(void)
{
    char *argv[] = { "lis", "--model", "x" };
    char *zero_argv[] = {
        "lis",
        "--model", "x",
        "--config", "x",
        "--tokens", "x",
        "--context", "0",
        "--batch", "1",
        "--generate", "1",
    };
    char *unknown_argv[] = { "lis", "--serve" };
    char *unknown_kv_argv[] = { "lis", "--kv-cache-diagnostics" };

    expect_int("cli invalid arguments", lis_cli_run(3, argv), 2);
    expect_int("cli zero numeric argument", lis_cli_run(13, zero_argv), 2);
    expect_int("cli unknown argument", lis_cli_run(2, unknown_argv), 2);
    expect_int("cli no kv diagnostics flag", lis_cli_run(2, unknown_kv_argv),
               2);
}

static void test_cli_happy_path(void)
{
    const char *model_path = "srcs/libs/test_cli_model.safetensors";
    const char *config_path = "srcs/libs/test_cli_config.json";
    const char *token_path = "srcs/libs/test_cli_tokens.txt";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_model.safetensors",
        "--config", "srcs/libs/test_cli_config.json",
        "--tokens", "srcs/libs/test_cli_tokens.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "2",
    };

    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli happy path", lis_cli_run(13, argv), 0);
    remove(model_path);
    remove(config_path);
    remove(token_path);
}

static void test_cli_report_json_success(void)
{
    const char *model_path = "srcs/libs/test_cli_report_model.safetensors";
    const char *config_path = "srcs/libs/test_cli_report_config.json";
    const char *token_path = "srcs/libs/test_cli_report_tokens.txt";
    const char *report_path = "srcs/libs/test_cli_report_success.json";
    const char *stdout_path = "srcs/libs/test_cli_report_success.out";
    const char *stderr_path = "srcs/libs/test_cli_report_success.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_report_model.safetensors",
        "--config", "srcs/libs/test_cli_report_config.json",
        "--tokens", "srcs/libs/test_cli_report_tokens.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "2",
        "--report-json", "srcs/libs/test_cli_report_success.json",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli report success",
               run_cli_capture(15, argv, stdout_path, stderr_path), 0);
    expect_file_equals("cli report success stdout", stdout_path,
                       "generated_token_ids: 2 2\n");
    expect_file_empty("cli report success stderr", stderr_path);
    expect_file_contains("cli report schema", report_path,
                         "\"schema\":\"lis.execution_artifact/v1\"");
    expect_file_contains("cli report kind", report_path,
                         "\"kind\":\"run_report\"");
    expect_file_contains("cli report model format", report_path,
                         "\"format\":\"safetensors\"");
    expect_file_contains("cli report input mode", report_path,
                         "\"mode\":\"tokens\"");
    expect_file_contains("cli report output mode", report_path,
                         "\"output_mode\":\"token_ids\"");
    expect_file_contains("cli report success status", report_path,
                         "\"execution_status\":\"ok\"");
    expect_file_contains("cli report success stop reason", report_path,
                         "\"stop_reason\":\"decode_limit\"");
    expect_file_contains("cli report selected ids", report_path,
                         "\"selected_token_ids\":[2,2]");
    expect_file_contains("cli report emitted ids", report_path,
                         "\"emitted_token_ids\":[2,2]");
    expect_file_contains("cli report kv cache object", report_path,
                         "\"kv_cache\":{\"scope\":\"run_local\"");
    expect_file_contains("cli report kv cache policy", report_path,
                         "\"policy\":{\"eviction_free\":true,"
                         "\"monotonic_growth\":true,\"paging\":false,"
                         "\"offload\":false,\"sliding_window\":false,"
                         "\"prefix_reuse\":false}");
    expect_file_contains("cli report kv cache dtype", report_path,
                         "\"storage_dtype\":\"f32\"");
    expect_file_contains("cli report kv cache max tokens", report_path,
                         "\"max_tokens\":4");
    expect_file_contains("cli report kv cache used tokens", report_path,
                         "\"used_tokens\":4");
    expect_file_contains("cli report kv cache bytes per token", report_path,
                         "\"bytes_per_token\":64");
    expect_file_contains("cli report kv cache allocated bytes", report_path,
                         "\"allocated_bytes\":256");
    expect_file_contains("cli report kv cache used bytes", report_path,
                         "\"used_bytes\":256");
    expect_file_contains("cli report kv cache shape", report_path,
                         "\"shape\":{\"layer_count\":1,\"batch_size\":2,"
                         "\"kv_head_count\":1,\"head_dim\":4,"
                         "\"element_size\":4}");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_token_batch_rejection(void)
{
    const char *model_path = "srcs/libs/test_cli_bad_tokens.safetensors";
    const char *config_path = "srcs/libs/test_cli_bad_tokens.json";
    const char *token_path = "srcs/libs/test_cli_bad_tokens.txt";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_bad_tokens.safetensors",
        "--config", "srcs/libs/test_cli_bad_tokens.json",
        "--tokens", "srcs/libs/test_cli_bad_tokens.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "1",
    };

    write_valid_fixtures(model_path, config_path, token_path);
    expect_status("overwrite bad token batch",
                  write_text_file(token_path, "0 1\n"), LIS_STATUS_OK);
    expect_int("cli token batch rejection", lis_cli_run(13, argv), 1);
    remove(model_path);
    remove(config_path);
    remove(token_path);
}

static void test_cli_token_format_rejection(void)
{
    const char *model_path = "srcs/libs/test_cli_bad_token_format.safetensors";
    const char *config_path = "srcs/libs/test_cli_bad_token_format.json";
    const char *token_path = "srcs/libs/test_cli_bad_token_format.txt";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_bad_token_format.safetensors",
        "--config", "srcs/libs/test_cli_bad_token_format.json",
        "--tokens", "srcs/libs/test_cli_bad_token_format.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "1",
    };

    write_valid_fixtures(model_path, config_path, token_path);
    expect_status("overwrite malformed token batch",
                  write_text_file(token_path, "0 x\n2\n"), LIS_STATUS_OK);
    expect_int("cli malformed token rejection", lis_cli_run(13, argv), 1);
    remove(model_path);
    remove(config_path);
    remove(token_path);
}

static void test_cli_vocab_rejection(void)
{
    const char *model_path = "srcs/libs/test_cli_bad_vocab.safetensors";
    const char *config_path = "srcs/libs/test_cli_bad_vocab.json";
    const char *token_path = "srcs/libs/test_cli_bad_vocab.txt";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_bad_vocab.safetensors",
        "--config", "srcs/libs/test_cli_bad_vocab.json",
        "--tokens", "srcs/libs/test_cli_bad_vocab.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "1",
    };

    write_valid_fixtures(model_path, config_path, token_path);
    expect_status("overwrite out of vocab token batch",
                  write_text_file(token_path, "0 4\n2\n"), LIS_STATUS_OK);
    expect_int("cli vocab rejection", lis_cli_run(13, argv), 1);
    remove(model_path);
    remove(config_path);
    remove(token_path);
}

static void test_cli_validation_logits_rejection(void)
{
    const char *model_path = "srcs/libs/test_cli_bad_logits.safetensors";
    const char *config_path = "srcs/libs/test_cli_bad_logits.json";
    const char *token_path = "srcs/libs/test_cli_bad_logits.txt";
    const char *header =
        "{\"other.tensor\":{\"dtype\":\"F32\",\"shape\":[1,4],"
        "\"data_offsets\":[0,16]}}";
    const float logits[4] = { 0.0f };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_bad_logits.safetensors",
        "--config", "srcs/libs/test_cli_bad_logits.json",
        "--tokens", "srcs/libs/test_cli_bad_logits.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "1",
    };

    expect_status("write bad logits model",
                  write_safetensors_file_with_data(model_path, header,
                                                   logits, sizeof(logits)),
                  LIS_STATUS_OK);
    expect_status("write bad logits config",
                  write_text_file(config_path,
                                  "{\"model_type\":\"llama\","
                                  "\"num_hidden_layers\":1,"
                                  "\"hidden_size\":4,"
                                  "\"intermediate_size\":8,"
                                  "\"num_attention_heads\":1,"
                                  "\"num_key_value_heads\":1,"
                                  "\"head_dim\":4,\"vocab_size\":4,"
                                  "\"rope_theta\":10000.0,"
                                  "\"torch_dtype\":\"float32\","
                                  "\"max_position_embeddings\":8}"),
                  LIS_STATUS_OK);
    expect_status("write bad logits tokens",
                  write_text_file(token_path, "0 1\n2\n"), LIS_STATUS_OK);
    expect_int("cli validation logits rejection", lis_cli_run(13, argv), 1);
    remove(model_path);
    remove(config_path);
    remove(token_path);
}

static void test_token_batch_public_validation(void)
{
    size_t tokens[] = { 0, 1 };
    size_t mismatched_lengths[] = { 1, 2 };
    lis_token_id_batch batch = {
        .tokens = tokens,
        .token_count = 2,
        .lengths = mismatched_lengths,
        .batch_size = 2,
    };

    expect_status("token batch length mismatch",
                  lis_token_id_batch_validate_vocab(&batch, 4),
                  LIS_STATUS_SHAPE_MISMATCH);
    mismatched_lengths[1] = 1;
    expect_status("token batch vocab valid",
                  lis_token_id_batch_validate_vocab(&batch, 4),
                  LIS_STATUS_OK);
    tokens[1] = 4;
    expect_status("token batch vocab limit",
                  lis_token_id_batch_validate_vocab(&batch, 4),
                  LIS_STATUS_LIMIT_EXCEEDED);
}

static void test_cli_context_rejection(void)
{
    const char *model_path = "srcs/libs/test_cli_bad_context.safetensors";
    const char *config_path = "srcs/libs/test_cli_bad_context.json";
    const char *token_path = "srcs/libs/test_cli_bad_context.txt";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_bad_context.safetensors",
        "--config", "srcs/libs/test_cli_bad_context.json",
        "--tokens", "srcs/libs/test_cli_bad_context.txt",
        "--context", "9",
        "--batch", "2",
        "--generate", "1",
    };

    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli context rejection", lis_cli_run(13, argv), 1);
    remove(model_path);
    remove(config_path);
    remove(token_path);
}

static void test_cli_unsupported_rope_config_rejection(void)
{
    const char *model_path = "srcs/libs/test_cli_rope_scaling.safetensors";
    const char *config_path = "srcs/libs/test_cli_rope_scaling.json";
    const char *token_path = "srcs/libs/test_cli_rope_scaling.txt";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":4,\"intermediate_size\":8,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":4,\"vocab_size\":4,"
        "\"rope_theta\":10000.0,"
        "\"rope_scaling\":{\"rope_type\":\"llama3\",\"factor\":32.0},"
        "\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_rope_scaling.safetensors",
        "--config", "srcs/libs/test_cli_rope_scaling.json",
        "--tokens", "srcs/libs/test_cli_rope_scaling.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "1",
    };

    expect_status("write rope scaling model", write_safetensors_file(model_path),
                  LIS_STATUS_OK);
    expect_status("write rope scaling config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("write rope scaling tokens",
                  write_text_file(token_path, "0 1\n2\n"), LIS_STATUS_OK);
    expect_int("cli unsupported rope config rejection", lis_cli_run(13, argv),
               1);
    remove(model_path);
    remove(config_path);
    remove(token_path);
}

static void test_cli_decode_limit_rejection(void)
{
    const char *model_path = "srcs/libs/test_cli_decode_limit.safetensors";
    const char *config_path = "srcs/libs/test_cli_decode_limit.json";
    const char *token_path = "srcs/libs/test_cli_decode_limit.txt";
    const char *stdout_path = "srcs/libs/test_cli_decode_limit.out";
    const char *stderr_path = "srcs/libs/test_cli_decode_limit.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_decode_limit.safetensors",
        "--config", "srcs/libs/test_cli_decode_limit.json",
        "--tokens", "srcs/libs/test_cli_decode_limit.txt",
        "--context", "2",
        "--batch", "2",
        "--generate", "1",
    };

    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli decode limit rejection",
               run_cli_capture(13, argv, stdout_path, stderr_path), 1);
    expect_file_empty("cli decode limit stdout empty", stdout_path);
    expect_file_contains("cli decode limit context detail", stderr_path,
                         "context limit reached during generation");
    expect_file_contains("cli decode limit stderr", stderr_path,
                         "decode/output failed: LIMIT_EXCEEDED");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_report_json_context_limit(void)
{
    const char *model_path = "srcs/libs/test_cli_report_limit.safetensors";
    const char *config_path = "srcs/libs/test_cli_report_limit.json";
    const char *token_path = "srcs/libs/test_cli_report_limit.txt";
    const char *report_path = "srcs/libs/test_cli_report_limit.json.out";
    const char *stdout_path = "srcs/libs/test_cli_report_limit.out";
    const char *stderr_path = "srcs/libs/test_cli_report_limit.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_report_limit.safetensors",
        "--config", "srcs/libs/test_cli_report_limit.json",
        "--tokens", "srcs/libs/test_cli_report_limit.txt",
        "--context", "2",
        "--batch", "2",
        "--generate", "1",
        "--report-json", "srcs/libs/test_cli_report_limit.json.out",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli report context limit",
               run_cli_capture(15, argv, stdout_path, stderr_path), 1);
    expect_file_empty("cli report context stdout", stdout_path);
    expect_file_contains("cli report context stderr detail", stderr_path,
                         "context limit reached during generation");
    expect_file_contains("cli report context stderr status", stderr_path,
                         "decode/output failed: LIMIT_EXCEEDED");
    expect_file_contains("cli report context execution status", report_path,
                         "\"execution_status\":\"error\"");
    expect_file_contains("cli report context status code", report_path,
                         "\"status_code\":\"LIMIT_EXCEEDED\"");
    expect_file_contains("cli report context stop reason", report_path,
                         "\"stop_reason\":\"context_limit\"");
    expect_file_contains("cli report context selected count", report_path,
                         "\"selected_token_count\":1");
    expect_file_contains("cli report context selected ids", report_path,
                         "\"selected_token_ids\":[2]");
    expect_file_contains("cli report context kv used tokens", report_path,
                         "\"used_tokens\":2");
    expect_file_contains("cli report context kv used bytes", report_path,
                         "\"used_bytes\":128");
    expect_file_contains("cli report context emitted count", report_path,
                         "\"emitted_token_count\":0");
    expect_file_contains("cli report context emitted ids", report_path,
                         "\"emitted_token_ids\":[]");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_validation_eos_stop(void)
{
    const char *model_path = "srcs/libs/test_cli_eos_model.safetensors";
    const char *config_path = "srcs/libs/test_cli_eos_config.json";
    const char *token_path = "srcs/libs/test_cli_eos_tokens.txt";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":4,\"intermediate_size\":8,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":4,\"vocab_size\":4,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"eos_token_id\":2,\"max_position_embeddings\":8}";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_eos_model.safetensors",
        "--config", "srcs/libs/test_cli_eos_config.json",
        "--tokens", "srcs/libs/test_cli_eos_tokens.txt",
        "--context", "2",
        "--batch", "2",
        "--generate", "100",
    };

    expect_status("write eos model", write_safetensors_file(model_path),
                  LIS_STATUS_OK);
    expect_status("write eos config", write_text_file(config_path, config_json),
                  LIS_STATUS_OK);
    expect_status("write eos tokens", write_text_file(token_path, "0\n1\n"),
                  LIS_STATUS_OK);
    expect_int("cli validation eos stop", lis_cli_run(13, argv), 0);
    remove(model_path);
    remove(config_path);
    remove(token_path);
}

static void write_structural_token_fixtures(const char *model_path,
                                            const char *config_path,
                                            const char *tokenizer_path,
                                            const float *logits)
{
    const char *header =
        "{\"lis.validation_logits\":{\"dtype\":\"F32\",\"shape\":[1,10],"
        "\"data_offsets\":[0,40]}}";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":4,\"intermediate_size\":8,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":4,\"vocab_size\":10,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":64}";
    const char *tokenizer_json =
        "{\"added_tokens\":["
        "{\"id\":2,\"content\":\"<|begin_of_text|>\",\"special\":true},"
        "{\"id\":3,\"content\":\"<|start_header_id|>\",\"special\":true},"
        "{\"id\":4,\"content\":\"<|end_header_id|>\",\"special\":true},"
        "{\"id\":5,\"content\":\"<|eot_id|>\",\"special\":true},"
        "{\"id\":7,\"content\":\"user\",\"special\":true},"
        "{\"id\":8,\"content\":\"assistant\",\"special\":true},"
        "{\"id\":9,\"content\":\"\\n\\n\",\"special\":true}],"
        "\"model\":{\"type\":\"BPE\","
        "\"vocab\":{\"a\":0,\"b\":1,\"c\":6},\"merges\":[]}}";

    expect_status("write structural model",
                  write_safetensors_file_with_data(model_path, header,
                                                   logits,
                                                   10U * sizeof(*logits)),
                  LIS_STATUS_OK);
    expect_status("write structural config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("write structural tokenizer",
                  write_text_file(tokenizer_path, tokenizer_json),
                  LIS_STATUS_OK);
}

static void test_cli_structural_control_token_suppression(void)
{
    const char *model_path = "srcs/libs/test_cli_structural.safetensors";
    const char *config_path = "srcs/libs/test_cli_structural.json";
    const char *tokenizer_path = "srcs/libs/test_cli_structural_tokenizer.json";
    const char *stdout_path = "srcs/libs/test_cli_structural.out";
    const char *stderr_path = "srcs/libs/test_cli_structural.err";
    const float logits[10] = { 0.0f, 8.0f, 10.0f, 12.0f, 11.0f,
                               -1.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_structural.safetensors",
        "--config", "srcs/libs/test_cli_structural.json",
        "--hf-tokenizer", "srcs/libs/test_cli_structural_tokenizer.json",
        "--prompt", "a",
        "--context", "32",
        "--batch", "1",
        "--generate", "2",
    };

    remove(model_path);
    remove(config_path);
    remove(tokenizer_path);
    remove(stdout_path);
    remove(stderr_path);
    write_structural_token_fixtures(model_path, config_path, tokenizer_path,
                                    logits);
    expect_int("cli structural token suppression",
               run_cli_capture(15, argv, stdout_path, stderr_path), 0);
    expect_file_equals("cli structural stdout", stdout_path, "bb\n");
    expect_file_empty("cli structural stderr", stderr_path);
    remove(model_path);
    remove(config_path);
    remove(tokenizer_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_structural_stop_token_hidden(void)
{
    const char *model_path = "srcs/libs/test_cli_structural_stop.safetensors";
    const char *config_path = "srcs/libs/test_cli_structural_stop.json";
    const char *tokenizer_path =
        "srcs/libs/test_cli_structural_stop_tokenizer.json";
    const char *stdout_path = "srcs/libs/test_cli_structural_stop.out";
    const char *stderr_path = "srcs/libs/test_cli_structural_stop.err";
    const float logits[10] = { 0.0f, 8.0f, 0.0f, 7.0f, 6.0f,
                               12.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_structural_stop.safetensors",
        "--config", "srcs/libs/test_cli_structural_stop.json",
        "--hf-tokenizer", "srcs/libs/test_cli_structural_stop_tokenizer.json",
        "--prompt", "a",
        "--context", "32",
        "--batch", "1",
        "--generate", "2",
    };

    remove(model_path);
    remove(config_path);
    remove(tokenizer_path);
    remove(stdout_path);
    remove(stderr_path);
    write_structural_token_fixtures(model_path, config_path, tokenizer_path,
                                    logits);
    expect_int("cli structural stop hidden",
               run_cli_capture(15, argv, stdout_path, stderr_path), 0);
    expect_file_equals("cli structural stop stdout", stdout_path, "\n");
    expect_file_empty("cli structural stop stderr", stderr_path);
    remove(model_path);
    remove(config_path);
    remove(tokenizer_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_canonical_hf_prompt_empty_user_text(void)
{
    const char *model_path = "srcs/libs/test_cli_canonical.safetensors";
    const char *config_path = "srcs/libs/test_cli_canonical.json";
    const char *tokenizer_path = "srcs/libs/test_cli_canonical_tokenizer.json";
    const char *stdout_path = "srcs/libs/test_cli_canonical.out";
    const char *stderr_path = "srcs/libs/test_cli_canonical.err";
    const char *header =
        "{\"lis.validation_logits\":{\"dtype\":\"F32\",\"shape\":[1,10],"
        "\"data_offsets\":[0,40]}}";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":4,\"intermediate_size\":8,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":4,\"vocab_size\":10,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":64}";
    const char *tokenizer_json =
        "{\"added_tokens\":["
        "{\"id\":2,\"content\":\"<|begin_of_text|>\",\"special\":true},"
        "{\"id\":3,\"content\":\"<|start_header_id|>\",\"special\":true},"
        "{\"id\":4,\"content\":\"<|end_header_id|>\",\"special\":true},"
        "{\"id\":5,\"content\":\"<|eot_id|>\",\"special\":true},"
        "{\"id\":6,\"content\":\"user\",\"special\":true},"
        "{\"id\":7,\"content\":\"assistant\",\"special\":true},"
        "{\"id\":8,\"content\":\"\\n\\n\",\"special\":true}],"
        "\"model\":{\"type\":\"BPE\","
        "\"vocab\":{\"b\":1},\"merges\":[]}}";
    const float logits[10] = { 0.0f, 9.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_canonical.safetensors",
        "--config", "srcs/libs/test_cli_canonical.json",
        "--hf-tokenizer", "srcs/libs/test_cli_canonical_tokenizer.json",
        "--prompt", "",
        "--context", "32",
        "--batch", "1",
        "--generate", "1",
    };

    remove(model_path);
    remove(config_path);
    remove(tokenizer_path);
    remove(stdout_path);
    remove(stderr_path);
    expect_status("write canonical model",
                  write_safetensors_file_with_data(model_path, header, logits,
                                                   sizeof(logits)),
                  LIS_STATUS_OK);
    expect_status("write canonical config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("write canonical tokenizer",
                  write_text_file(tokenizer_path, tokenizer_json),
                  LIS_STATUS_OK);
    expect_int("cli canonical hf prompt",
               run_cli_capture(15, argv, stdout_path, stderr_path), 0);
    expect_file_equals("cli canonical stdout", stdout_path, "b\n");
    expect_file_empty("cli canonical stderr", stderr_path);
    remove(model_path);
    remove(config_path);
    remove(tokenizer_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_generation_diagnostics_token_output(void)
{
    const char *model_path = "srcs/libs/test_cli_diag_ids.safetensors";
    const char *config_path = "srcs/libs/test_cli_diag_ids.json";
    const char *token_path = "srcs/libs/test_cli_diag_ids.txt";
    const char *stdout_path = "srcs/libs/test_cli_diag_ids.out";
    const char *stderr_path = "srcs/libs/test_cli_diag_ids.err";
    const char *header =
        "{\"lis.validation_logits\":{\"dtype\":\"F32\",\"shape\":[1,3],"
        "\"data_offsets\":[0,12]}}";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":4,\"intermediate_size\":8,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":4,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    const float logits[3] = { 10.0f, 9.0f, 0.0f };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_diag_ids.safetensors",
        "--config", "srcs/libs/test_cli_diag_ids.json",
        "--tokens", "srcs/libs/test_cli_diag_ids.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "2",
        "--diagnostics",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    expect_status("write diagnostic ids model",
                  write_safetensors_file_with_data(model_path, header, logits,
                                                   sizeof(logits)),
                  LIS_STATUS_OK);
    expect_status("write diagnostic ids config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("write diagnostic ids tokens",
                  write_text_file(token_path, "0\n"), LIS_STATUS_OK);
    expect_int("cli diagnostics token ids",
               run_cli_capture(14, argv, stdout_path, stderr_path), 0);
    expect_file_equals("cli diagnostics ids stdout", stdout_path,
                       "generated_token_ids: 0 1\n");
    expect_file_contains("cli diagnostics unavailable text", stderr_path,
                         "selected_token_text=<unavailable>");
    expect_file_contains("cli diagnostics simd backend", stderr_path,
                         "lis: simd backend=");
    expect_file_contains("cli diagnostics selected id 0", stderr_path,
                         "selected_token_id=0");
    expect_file_contains("cli diagnostics selected id 1", stderr_path,
                         "selected_token_id=1");
    expect_file_contains("cli diagnostics decode limit", stderr_path,
                         "stop_reason=decode_limit");
    expect_file_contains("cli diagnostics repetition selection", stderr_path,
                         "repetition_penalty_changed_selection=true");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_generation_diagnostics_model_eos(void)
{
    const char *model_path = "srcs/libs/test_cli_diag_eos.safetensors";
    const char *config_path = "srcs/libs/test_cli_diag_eos.json";
    const char *token_path = "srcs/libs/test_cli_diag_eos.txt";
    const char *stdout_path = "srcs/libs/test_cli_diag_eos.out";
    const char *stderr_path = "srcs/libs/test_cli_diag_eos.err";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":4,\"intermediate_size\":8,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":4,\"vocab_size\":4,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"eos_token_id\":2,\"max_position_embeddings\":8}";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_diag_eos.safetensors",
        "--config", "srcs/libs/test_cli_diag_eos.json",
        "--tokens", "srcs/libs/test_cli_diag_eos.txt",
        "--context", "2",
        "--batch", "2",
        "--generate", "100",
        "--diagnostics",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    expect_status("write diagnostic eos model", write_safetensors_file(model_path),
                  LIS_STATUS_OK);
    expect_status("write diagnostic eos config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("write diagnostic eos tokens",
                  write_text_file(token_path, "0\n1\n"), LIS_STATUS_OK);
    expect_int("cli diagnostics model eos",
               run_cli_capture(14, argv, stdout_path, stderr_path), 0);
    expect_file_equals("cli diagnostics eos stdout", stdout_path,
                       "generated_token_ids: 2\n");
    expect_file_contains("cli diagnostics model eos reason", stderr_path,
                         "stop_reason=model_eos");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_generation_diagnostics_tokenizer_output(void)
{
    const char *model_path = "srcs/libs/test_cli_diag_text.safetensors";
    const char *config_path = "srcs/libs/test_cli_diag_text.json";
    const char *tokenizer_path = "srcs/libs/test_cli_diag_text_tokenizer.json";
    const char *stdout_path = "srcs/libs/test_cli_diag_text.out";
    const char *stderr_path = "srcs/libs/test_cli_diag_text.err";
    const float logits[10] = { 0.0f, 8.0f, 10.0f, 12.0f, 11.0f,
                               -1.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_diag_text.safetensors",
        "--config", "srcs/libs/test_cli_diag_text.json",
        "--hf-tokenizer", "srcs/libs/test_cli_diag_text_tokenizer.json",
        "--prompt", "a",
        "--context", "32",
        "--batch", "1",
        "--generate", "1",
        "--diagnostics",
    };

    remove(model_path);
    remove(config_path);
    remove(tokenizer_path);
    remove(stdout_path);
    remove(stderr_path);
    write_structural_token_fixtures(model_path, config_path, tokenizer_path,
                                    logits);
    expect_int("cli diagnostics tokenizer",
               run_cli_capture(16, argv, stdout_path, stderr_path), 0);
    expect_file_equals("cli diagnostics tokenizer stdout", stdout_path,
                       "b\n");
    expect_file_contains("cli diagnostics text available", stderr_path,
                         "selected_token_text=\"b\"");
    expect_file_contains("cli diagnostics suppression", stderr_path,
                         "structural_suppression_affected=true");
    expect_file_contains("cli diagnostics text decode limit", stderr_path,
                         "stop_reason=decode_limit");
    remove(model_path);
    remove(config_path);
    remove(tokenizer_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_report_json_tokenizer_perf(void)
{
    const char *model_path = "srcs/libs/test_cli_report_text.safetensors";
    const char *config_path = "srcs/libs/test_cli_report_text.json";
    const char *tokenizer_path = "srcs/libs/test_cli_report_text_tokenizer.json";
    const char *report_path = "srcs/libs/test_cli_report_text.json.out";
    const char *stdout_path = "srcs/libs/test_cli_report_text.out";
    const char *stderr_path = "srcs/libs/test_cli_report_text.err";
    const float logits[10] = { 0.0f, 8.0f, 10.0f, 12.0f, 11.0f,
                               -1.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_report_text.safetensors",
        "--config", "srcs/libs/test_cli_report_text.json",
        "--hf-tokenizer", "srcs/libs/test_cli_report_text_tokenizer.json",
        "--prompt", "a",
        "--context", "32",
        "--batch", "1",
        "--generate", "1",
        "--perf",
        "--report-json", "srcs/libs/test_cli_report_text.json.out",
    };

    remove(model_path);
    remove(config_path);
    remove(tokenizer_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
    write_structural_token_fixtures(model_path, config_path, tokenizer_path,
                                    logits);
    expect_int("cli report tokenizer perf",
               run_cli_capture(18, argv, stdout_path, stderr_path), 0);
    expect_file_equals("cli report tokenizer stdout", stdout_path, "b\n");
    expect_file_contains("cli report tokenizer stderr backend", stderr_path,
                         "lis: simd backend=");
    expect_file_contains("cli report tokenizer stderr perf", stderr_path,
                         "lis: perf-summary");
    expect_file_contains("cli report tokenizer input mode", report_path,
                         "\"mode\":\"hf_tokenizer_prompt\"");
    expect_file_contains("cli report tokenizer output mode", report_path,
                         "\"output_mode\":\"text\"");
    expect_file_contains("cli report tokenizer status", report_path,
                         "\"execution_status\":\"ok\"");
    expect_file_contains("cli report tokenizer selected ids", report_path,
                         "\"selected_token_ids\":[1]");
    expect_file_contains("cli report tokenizer emitted ids", report_path,
                         "\"emitted_token_ids\":[1]");
    expect_file_contains("cli report tokenizer perf object", report_path,
                         "\"perf\":{\"tag\":\"none\"");
    expect_file_contains("cli report tokenizer generated tokens", report_path,
                         "\"generated_tokens\":1");
    remove(model_path);
    remove(config_path);
    remove(tokenizer_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_generation_diagnostics_structural_stop(void)
{
    const char *model_path = "srcs/libs/test_cli_diag_stop.safetensors";
    const char *config_path = "srcs/libs/test_cli_diag_stop.json";
    const char *tokenizer_path = "srcs/libs/test_cli_diag_stop_tokenizer.json";
    const char *stdout_path = "srcs/libs/test_cli_diag_stop.out";
    const char *stderr_path = "srcs/libs/test_cli_diag_stop.err";
    const float logits[10] = { 0.0f, 8.0f, 0.0f, 7.0f, 6.0f,
                               12.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_diag_stop.safetensors",
        "--config", "srcs/libs/test_cli_diag_stop.json",
        "--hf-tokenizer", "srcs/libs/test_cli_diag_stop_tokenizer.json",
        "--prompt", "a",
        "--context", "32",
        "--batch", "1",
        "--generate", "2",
        "--diagnostics",
    };

    remove(model_path);
    remove(config_path);
    remove(tokenizer_path);
    remove(stdout_path);
    remove(stderr_path);
    write_structural_token_fixtures(model_path, config_path, tokenizer_path,
                                    logits);
    expect_int("cli diagnostics structural stop",
               run_cli_capture(16, argv, stdout_path, stderr_path), 0);
    expect_file_equals("cli diagnostics stop stdout", stdout_path, "\n");
    expect_file_contains("cli diagnostics structural stop reason", stderr_path,
                         "stop_reason=structural_control");
    expect_file_contains("cli diagnostics structural stop text", stderr_path,
                         "selected_token_text=\"<|eot_id|>\"");
    remove(model_path);
    remove(config_path);
    remove(tokenizer_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_generation_diagnostics_context_limit(void)
{
    const char *model_path = "srcs/libs/test_cli_diag_context.safetensors";
    const char *config_path = "srcs/libs/test_cli_diag_context.json";
    const char *token_path = "srcs/libs/test_cli_diag_context.txt";
    const char *stdout_path = "srcs/libs/test_cli_diag_context.out";
    const char *stderr_path = "srcs/libs/test_cli_diag_context.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_diag_context.safetensors",
        "--config", "srcs/libs/test_cli_diag_context.json",
        "--tokens", "srcs/libs/test_cli_diag_context.txt",
        "--context", "2",
        "--batch", "2",
        "--generate", "1",
        "--diagnostics",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli diagnostics context limit",
               run_cli_capture(14, argv, stdout_path, stderr_path), 1);
    expect_file_empty("cli diagnostics context stdout", stdout_path);
    expect_file_contains("cli diagnostics context reason", stderr_path,
                         "stop_reason=context_limit");
    expect_file_contains("cli context limit stderr message", stderr_path,
                         "context limit reached during generation");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_repetition_penalty_text_output(void)
{
    const char *model_path = "srcs/libs/test_cli_repetition.safetensors";
    const char *config_path = "srcs/libs/test_cli_repetition.json";
    const char *tokenizer_path = "srcs/libs/test_cli_repetition_tokenizer.json";
    const char *stdout_path = "srcs/libs/test_cli_repetition.out";
    const char *stderr_path = "srcs/libs/test_cli_repetition.err";
    const char *header =
        "{\"lis.validation_logits\":{\"dtype\":\"F32\",\"shape\":[1,10],"
        "\"data_offsets\":[0,40]}}";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":4,\"intermediate_size\":8,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":4,\"vocab_size\":10,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":64}";
    const char *tokenizer_json =
        "{\"added_tokens\":["
        "{\"id\":3,\"content\":\"<|begin_of_text|>\",\"special\":true},"
        "{\"id\":4,\"content\":\"<|start_header_id|>\",\"special\":true},"
        "{\"id\":5,\"content\":\"<|end_header_id|>\",\"special\":true},"
        "{\"id\":6,\"content\":\"<|eot_id|>\",\"special\":true},"
        "{\"id\":7,\"content\":\"user\",\"special\":true},"
        "{\"id\":8,\"content\":\"assistant\",\"special\":true},"
        "{\"id\":9,\"content\":\"\\n\\n\",\"special\":true}],"
        "\"model\":{\"type\":\"BPE\","
        "\"vocab\":{\"a\":0,\"b\":1,\"c\":2},\"merges\":[]}}";
    const float logits[10] = { 10.0f, 9.0f, 0.0f, 0.0f, 0.0f,
                               0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_repetition.safetensors",
        "--config", "srcs/libs/test_cli_repetition.json",
        "--hf-tokenizer", "srcs/libs/test_cli_repetition_tokenizer.json",
        "--prompt", "a",
        "--context", "32",
        "--batch", "1",
        "--generate", "2",
    };

    remove(model_path);
    remove(config_path);
    remove(tokenizer_path);
    remove(stdout_path);
    remove(stderr_path);
    expect_status("write repetition model",
                  write_safetensors_file_with_data(model_path, header, logits,
                                                   sizeof(logits)),
                  LIS_STATUS_OK);
    expect_status("write repetition config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("write repetition tokenizer",
                  write_text_file(tokenizer_path, tokenizer_json),
                  LIS_STATUS_OK);
    expect_int("cli repetition penalty text",
               run_cli_capture(15, argv, stdout_path, stderr_path), 0);
    expect_file_equals("cli repetition stdout", stdout_path, "ab\n");
    expect_file_empty("cli repetition stderr", stderr_path);
    remove(model_path);
    remove(config_path);
    remove(tokenizer_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_repetition_penalty_token_output(void)
{
    const char *model_path = "srcs/libs/test_cli_repetition_ids.safetensors";
    const char *config_path = "srcs/libs/test_cli_repetition_ids.json";
    const char *token_path = "srcs/libs/test_cli_repetition_ids.txt";
    const char *stdout_path = "srcs/libs/test_cli_repetition_ids.out";
    const char *stderr_path = "srcs/libs/test_cli_repetition_ids.err";
    const char *header =
        "{\"lis.validation_logits\":{\"dtype\":\"F32\",\"shape\":[1,3],"
        "\"data_offsets\":[0,12]}}";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":4,\"intermediate_size\":8,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":4,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    const float logits[3] = { 10.0f, 9.0f, 0.0f };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_repetition_ids.safetensors",
        "--config", "srcs/libs/test_cli_repetition_ids.json",
        "--tokens", "srcs/libs/test_cli_repetition_ids.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "2",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    expect_status("write repetition ids model",
                  write_safetensors_file_with_data(model_path, header, logits,
                                                   sizeof(logits)),
                  LIS_STATUS_OK);
    expect_status("write repetition ids config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("write repetition ids tokens",
                  write_text_file(token_path, "0\n"), LIS_STATUS_OK);
    expect_int("cli repetition penalty token ids",
               run_cli_capture(13, argv, stdout_path, stderr_path), 0);
    expect_file_equals("cli repetition ids stdout", stdout_path,
                       "generated_token_ids: 0 1\n");
    expect_file_empty("cli repetition ids stderr", stderr_path);
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_hf_llama_forward_path(void)
{
    const char *model_path = "srcs/libs/test_cli_hf_model/model.safetensors";
    const char *config_path = "srcs/libs/test_cli_hf_model/config.json";
    const char *token_path = "srcs/libs/test_cli_hf_tokens.txt";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":1,\"intermediate_size\":1,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":1,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"eos_token_id\":2,"
        "\"max_position_embeddings\":4}";
    const char *header =
        "{\"model.embed_tokens.weight\":{\"dtype\":\"F32\",\"shape\":[3,1],"
        "\"data_offsets\":[0,12]},"
        "\"model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[12,16]},"
        "\"model.layers.0.self_attn.k_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[16,20]},"
        "\"model.layers.0.self_attn.v_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[20,24]},"
        "\"model.layers.0.self_attn.o_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[24,28]},"
        "\"model.layers.0.mlp.gate_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[28,32]},"
        "\"model.layers.0.mlp.up_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[32,36]},"
        "\"model.layers.0.mlp.down_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[36,40]},"
        "\"model.layers.0.input_layernorm.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1],\"data_offsets\":[40,44]},"
        "\"model.layers.0.post_attention_layernorm.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1],\"data_offsets\":[44,48]},"
        "\"model.norm.weight\":{\"dtype\":\"F32\",\"shape\":[1],"
        "\"data_offsets\":[48,52]},"
        "\"lm_head.weight\":{\"dtype\":\"F32\",\"shape\":[3,1],"
        "\"data_offsets\":[52,64]}}";
    const float data[16] = {
        1.0f, 2.0f, 3.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f,
        0.1f, 0.2f, 0.9f
    };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_hf_model",
        "--config", "srcs/libs/test_cli_hf_model/config.json",
        "--tokens", "srcs/libs/test_cli_hf_tokens.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "100",
    };
    const char *stdout_path = "srcs/libs/test_cli_hf_limit.out";
    const char *stderr_path = "srcs/libs/test_cli_hf_limit.err";
    char *limit_argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_hf_model",
        "--config", "srcs/libs/test_cli_hf_model/config.json",
        "--tokens", "srcs/libs/test_cli_hf_tokens.txt",
        "--context", "2",
        "--batch", "1",
        "--generate", "1",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    if (system("mkdir -p srcs/libs/test_cli_hf_model") != 0) {
        fprintf(stderr, "mkdir hf cli fixture failed\n");
        ++g_failures;
        return;
    }
    expect_status("write cli hf config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("write cli hf model",
                  write_safetensors_file_with_data(model_path, header, data,
                                                   sizeof(data)),
                  LIS_STATUS_OK);
    expect_status("write cli hf tokens",
                  write_text_file(token_path, "0 1\n"), LIS_STATUS_OK);
    expect_int("cli hf llama forward path", lis_cli_run(13, argv), 0);
    expect_int("cli hf llama decode limit rejection",
               run_cli_capture(13, limit_argv, stdout_path, stderr_path), 1);
    expect_file_empty("cli hf decode limit stdout empty", stdout_path);
    expect_file_contains("cli hf decode limit context detail", stderr_path,
                         "context limit reached during generation");
    expect_file_contains("cli hf decode limit stderr", stderr_path,
                         "decode/output failed: LIMIT_EXCEEDED");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    if (system("rmdir srcs/libs/test_cli_hf_model 2>/dev/null") != 0) {
        /* best effort cleanup */
    }
}

static void test_cli_hf_qwen3_forward_path(void)
{
    const char *model_path = "srcs/libs/test_cli_qwen3/model.safetensors";
    const char *config_path = "srcs/libs/test_cli_qwen3/config.json";
    const char *token_path = "srcs/libs/test_cli_qwen3_tokens.txt";
    const char *tokenizer_path = "srcs/libs/test_cli_qwen3_tokenizer.json";
    const char *stdout_path = "srcs/libs/test_cli_qwen3.out";
    const char *stderr_path = "srcs/libs/test_cli_qwen3.err";
    const char *layer_trace_path = "srcs/libs/test_cli_qwen3_layer.json";
    const char *config_json =
        "{\"architectures\":[\"Qwen3ForCausalLM\"],"
        "\"model_type\":\"qwen3\",\"num_hidden_layers\":1,"
        "\"hidden_size\":2,\"intermediate_size\":2,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":2,\"vocab_size\":3,"
        "\"rope_theta\":1000000.0,\"rms_norm_eps\":1e-06,"
        "\"torch_dtype\":\"bfloat16\",\"max_position_embeddings\":8,"
        "\"attention_bias\":false,\"use_sliding_window\":false,"
        "\"hidden_act\":\"silu\",\"rope_scaling\":null}";
    const char *header =
        "{\"model.embed_tokens.weight\":{\"dtype\":\"BF16\",\"shape\":[3,2],"
        "\"data_offsets\":[0,12]},"
        "\"model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"BF16\","
        "\"shape\":[2,2],\"data_offsets\":[12,20]},"
        "\"model.layers.0.self_attn.k_proj.weight\":{\"dtype\":\"BF16\","
        "\"shape\":[2,2],\"data_offsets\":[20,28]},"
        "\"model.layers.0.self_attn.v_proj.weight\":{\"dtype\":\"BF16\","
        "\"shape\":[2,2],\"data_offsets\":[28,36]},"
        "\"model.layers.0.self_attn.o_proj.weight\":{\"dtype\":\"BF16\","
        "\"shape\":[2,2],\"data_offsets\":[36,44]},"
        "\"model.layers.0.self_attn.q_norm.weight\":{\"dtype\":\"BF16\","
        "\"shape\":[2],\"data_offsets\":[44,48]},"
        "\"model.layers.0.self_attn.k_norm.weight\":{\"dtype\":\"BF16\","
        "\"shape\":[2],\"data_offsets\":[48,52]},"
        "\"model.layers.0.mlp.gate_proj.weight\":{\"dtype\":\"BF16\","
        "\"shape\":[2,2],\"data_offsets\":[52,60]},"
        "\"model.layers.0.mlp.up_proj.weight\":{\"dtype\":\"BF16\","
        "\"shape\":[2,2],\"data_offsets\":[60,68]},"
        "\"model.layers.0.mlp.down_proj.weight\":{\"dtype\":\"BF16\","
        "\"shape\":[2,2],\"data_offsets\":[68,76]},"
        "\"model.layers.0.input_layernorm.weight\":{\"dtype\":\"BF16\","
        "\"shape\":[2],\"data_offsets\":[76,80]},"
        "\"model.layers.0.post_attention_layernorm.weight\":{\"dtype\":\"BF16\","
        "\"shape\":[2],\"data_offsets\":[80,84]},"
        "\"model.norm.weight\":{\"dtype\":\"BF16\",\"shape\":[2],"
        "\"data_offsets\":[84,88]},"
        "\"lm_head.weight\":{\"dtype\":\"BF16\",\"shape\":[3,2],"
        "\"data_offsets\":[88,100]}}";
    const uint16_t data[50] = {
        0x3f80U, 0x0000U, 0x0000U, 0x3f80U, 0x3f80U, 0x3f80U,
        0x3f80U, 0x0000U, 0x0000U, 0x3f80U,
        0x3f80U, 0x0000U, 0x0000U, 0x3f80U,
        0x3f80U, 0x0000U, 0x0000U, 0x3f80U,
        0x3f80U, 0x0000U, 0x0000U, 0x3f80U,
        0x4000U, 0x3f80U,
        0x3f80U, 0x3f80U,
        0x0000U, 0x0000U, 0x0000U, 0x0000U,
        0x0000U, 0x0000U, 0x0000U, 0x0000U,
        0x0000U, 0x0000U, 0x0000U, 0x0000U,
        0x3f80U, 0x3f80U,
        0x3f80U, 0x3f80U,
        0x3f80U, 0x3f80U,
        0x3f80U, 0x0000U, 0x0000U, 0x3f80U, 0x3f80U, 0x3f80U
    };
    const char *tokenizer_json =
        "{\"model\":{\"type\":\"BPE\","
        "\"vocab\":{\"a\":0,\"b\":1,\"c\":2},\"merges\":[]}}";
    char *argv_tokens[] = {
        "lis",
        "--model", "srcs/libs/test_cli_qwen3",
        "--config", "srcs/libs/test_cli_qwen3/config.json",
        "--tokens", "srcs/libs/test_cli_qwen3_tokens.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "1",
        "--layer-checkpoints", "0",
        "--layer-trace-json", "srcs/libs/test_cli_qwen3_layer.json",
    };
    char *argv_text[] = {
        "lis",
        "--model", "srcs/libs/test_cli_qwen3",
        "--config", "srcs/libs/test_cli_qwen3/config.json",
        "--hf-tokenizer", "srcs/libs/test_cli_qwen3_tokenizer.json",
        "--prompt", "a",
        "--context", "4",
        "--batch", "1",
        "--generate", "1",
        "--diagnostics",
    };
    char *argv_intra[] = {
        "lis",
        "--model", "srcs/libs/test_cli_qwen3",
        "--config", "srcs/libs/test_cli_qwen3/config.json",
        "--tokens", "srcs/libs/test_cli_qwen3_tokens.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "1",
        "--layer-checkpoints", "1",
        "--layer-trace-json", "srcs/libs/test_cli_qwen3_layer.json",
        "--intra-layer-checkpoints", "0",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(tokenizer_path);
    remove(stdout_path);
    remove(stderr_path);
    remove(layer_trace_path);
    if (system("mkdir -p srcs/libs/test_cli_qwen3") != 0) {
        fprintf(stderr, "mkdir qwen3 cli fixture failed\n");
        ++g_failures;
        return;
    }
    expect_status("write cli qwen3 config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("write cli qwen3 model",
                  write_safetensors_file_with_data(model_path, header, data,
                                                   sizeof(data)),
                  LIS_STATUS_OK);
    expect_status("write cli qwen3 tokens",
                  write_text_file(token_path, "0 2\n"), LIS_STATUS_OK);
    expect_status("write cli qwen3 tokenizer",
                  write_text_file(tokenizer_path, tokenizer_json),
                  LIS_STATUS_OK);
    expect_int("cli qwen3 direct token path",
               run_cli_capture(17, argv_tokens, stdout_path, stderr_path), 0);
    expect_file_contains("cli qwen3 token stdout", stdout_path,
                         "generated_token_ids:");
    expect_file_contains("cli qwen3 legacy trace kind", layer_trace_path,
                         "\"kind\":\"layer_trace\"");
    expect_file_contains("cli qwen3 trace role remains q norm",
                         layer_trace_path, "qwen3.layer.0.q_after_q_norm");
    expect_file_not_contains("cli qwen3 has no llama checkpoint layout",
                             layer_trace_path, "\"checkpoint_layout\"");
    expect_file_not_contains("cli qwen3 q norm not layer output",
                             layer_trace_path,
                             "\"tensor_role\":\"layer_output\"");
    remove(stdout_path);
    remove(stderr_path);
    remove(layer_trace_path);
    expect_int("cli qwen3 intra unsupported",
               run_cli_capture(
                   (int)(sizeof(argv_intra) / sizeof(argv_intra[0])),
                   argv_intra, stdout_path, stderr_path), 1);
    expect_file_contains("cli qwen3 intra family error", stderr_path,
                         "lis: artifact error: --intra-layer-checkpoints "
                         "requires the Llama decoder family");
    expect_file_missing("cli qwen3 intra no layer artifact",
                        layer_trace_path);
    remove(stdout_path);
    remove(stderr_path);
    expect_int("cli qwen3 plain text tokenizer path",
               run_cli_capture(16, argv_text, stdout_path, stderr_path), 0);
    expect_file_contains("cli qwen3 diagnostics backend", stderr_path,
                         "lis: simd backend=");
    expect_file_contains("cli qwen3 diagnostics phase", stderr_path,
                         "phase=first_decode");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(tokenizer_path);
    remove(stdout_path);
    remove(stderr_path);
    if (system("rmdir srcs/libs/test_cli_qwen3 2>/dev/null") != 0) {
        /* best effort cleanup */
    }
}

/* === Extended token selection diagnostics tests === */

/*
 * Test top-k candidate ordering, selected marking, raw/adjusted scores,
 * and unavailable token text in token-ID mode (no tokenizer).
 */
static void test_token_selection_candidates_token_ids(void)
{
    const char *model_path = "srcs/libs/test_token_selection_ids.safetensors";
    const char *config_path = "srcs/libs/test_token_selection_ids.json";
    const char *token_path = "srcs/libs/test_token_selection_ids.txt";
    const char *stdout_path = "srcs/libs/test_token_selection_ids.out";
    const char *stderr_path = "srcs/libs/test_token_selection_ids.err";
    const char *header =
        "{\"lis.validation_logits\":{\"dtype\":\"F32\",\"shape\":[1,6],"
        "\"data_offsets\":[0,24]}}";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":4,\"intermediate_size\":8,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":4,\"vocab_size\":6,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    /* Vocab 6: logits order token2(15)>token4(12)>token5(8)>token0(5)>token1(3)>token3(1). */
    const float logits[6] = { 5.0f, 3.0f, 15.0f, 1.0f, 12.0f, 8.0f };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_token_selection_ids.safetensors",
        "--config", "srcs/libs/test_token_selection_ids.json",
        "--tokens", "srcs/libs/test_token_selection_ids.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "1",
        "--diagnostics",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    expect_status("token selection ids write model",
                  write_safetensors_file_with_data(model_path, header, logits,
                                                   sizeof(logits)),
                  LIS_STATUS_OK);
    expect_status("token selection ids write config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("token selection ids write tokens",
                  write_text_file(token_path, "0\n"), LIS_STATUS_OK);
    expect_int("token selection ids run",
               run_cli_capture(14, argv, stdout_path, stderr_path), 0);

    /* stdout: normal token-ID generation output, not changed by diagnostics. */
    expect_file_equals("token selection ids stdout", stdout_path,
                       "generated_token_ids: 2\n");

    /* Existing diagnostic fields still present. */
    expect_file_contains("token selection ids diagnostic selected", stderr_path,
                         "selected_token_id=2");
    expect_file_contains("token selection ids diagnostic text unavailable",
                         stderr_path,
                         "selected_token_text=<unavailable>");
    expect_file_contains("token selection ids diagnostic stop", stderr_path,
                         "stop_reason=decode_limit");

    /* Candidate lines: ranked by descending adjusted score. */
    expect_file_contains("token selection rank1", stderr_path,
                         "rank=1 token_id=2");
    expect_file_contains("token selection rank1 selected", stderr_path,
                         "rank=1 token_id=2 token_text=<unavailable> "
                         "raw_score=15 adjusted_score=15 selected=true");
    expect_file_contains("token selection rank2", stderr_path,
                         "rank=2 token_id=4");
    expect_file_contains("token selection rank2 not selected", stderr_path,
                         "rank=2 token_id=4 token_text=<unavailable> "
                         "raw_score=12 adjusted_score=12 selected=false");
    expect_file_contains("token selection rank3", stderr_path,
                         "rank=3 token_id=5");
    expect_file_contains("token selection rank4", stderr_path,
                         "rank=4 token_id=0");
    expect_file_contains("token selection rank5", stderr_path,
                         "rank=5 token_id=1");

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

/*
 * Test top-k candidates with a tokenizer: token text reporting, structural
 * suppression producing -inf adjusted scores, and repetition penalty
 * producing raw vs adjusted score divergence.
 */
static void test_token_selection_candidates_with_tokenizer(void)
{
    const char *model_path = "srcs/libs/test_token_selection_tok.safetensors";
    const char *config_path = "srcs/libs/test_token_selection_tok.json";
    const char *tokenizer_path =
        "srcs/libs/test_token_selection_tok_tokenizer.json";
    const char *stdout_path = "srcs/libs/test_token_selection_tok.out";
    const char *stderr_path = "srcs/libs/test_token_selection_tok.err";
    /* Vocab=10. Tokens 2-5,7-9 are structural. Token 0="a", 1="b", 6="c".
     * Logits: token 3 (structural <|start_header_id|>) has highest raw=12,
     * token 1 ("b") has raw=8, token 6 ("c") has raw=7.
     * After suppression, structural tokens get -inf. Greedy picks token 1.
     */
    const float logits[10] = { 0.0f, 8.0f, 10.0f, 12.0f, 11.0f,
                               -1.0f, 7.0f, 0.0f, 0.0f, 0.0f };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_token_selection_tok.safetensors",
        "--config", "srcs/libs/test_token_selection_tok.json",
        "--hf-tokenizer",
        "srcs/libs/test_token_selection_tok_tokenizer.json",
        "--prompt", "a",
        "--context", "32",
        "--batch", "1",
        "--generate", "1",
        "--diagnostics",
    };

    remove(model_path);
    remove(config_path);
    remove(tokenizer_path);
    remove(stdout_path);
    remove(stderr_path);
    write_structural_token_fixtures(model_path, config_path, tokenizer_path,
                                    logits);
    expect_int("token selection tokenizer run",
               run_cli_capture(16, argv, stdout_path, stderr_path), 0);

    /* stdout: text output, not changed by diagnostics. */
    expect_file_equals("token selection tokenizer stdout", stdout_path, "b\n");

    /* Existing diagnostic fields preserved. */
    expect_file_contains("token selection tokenizer diagnostic selected",
                         stderr_path,
                         "selected_token_id=1");
    expect_file_contains("token selection tokenizer diagnostic text",
                         stderr_path,
                         "selected_token_text=\"b\"");
    expect_file_contains("token selection tokenizer diagnostic suppression",
                         stderr_path,
                         "structural_suppression_affected=true");

    /* Candidate rank 1 should be token 1 ("b"), selected. */
    expect_file_contains("token selection tokenizer rank1 text", stderr_path,
                         "rank=1 token_id=1 token_text=\"b\"");
    expect_file_contains("token selection tokenizer rank1 selected",
                         stderr_path,
                         "rank=1 token_id=1 token_text=\"b\" "
                         "raw_score=8 adjusted_score=8 selected=true");
    /* rank 2 should be token 6 ("c"). */
    expect_file_contains("token selection tokenizer rank2", stderr_path,
                         "rank=2 token_id=6 token_text=\"c\"");
    expect_file_contains("token selection tokenizer rank2 scores",
                         stderr_path,
                         "rank=2 token_id=6 token_text=\"c\" "
                         "raw_score=7 adjusted_score=7 selected=false");
    /* rank 3 should be token 0 ("a"). */
    expect_file_contains("token selection tokenizer rank3", stderr_path,
                         "rank=3 token_id=0 token_text=\"a\"");

    remove(model_path);
    remove(config_path);
    remove(tokenizer_path);
    remove(stdout_path);
    remove(stderr_path);
}

/*
 * Test that repetition penalty produces raw vs adjusted score divergence
 * in top-k candidate reporting. Generate 2 steps so the second step has
 * a penalized token from step 1.
 */
static void test_token_selection_repetition_penalty_scores(void)
{
    const char *model_path = "srcs/libs/test_token_selection_rep.safetensors";
    const char *config_path = "srcs/libs/test_token_selection_rep.json";
    const char *token_path = "srcs/libs/test_token_selection_rep.txt";
    const char *stdout_path = "srcs/libs/test_token_selection_rep.out";
    const char *stderr_path = "srcs/libs/test_token_selection_rep.err";
    const char *header =
        "{\"lis.validation_logits\":{\"dtype\":\"F32\",\"shape\":[1,3],"
        "\"data_offsets\":[0,12]}}";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":4,\"intermediate_size\":8,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":4,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    /* Logits: token0=10, token1=9, token2=0.
     * Step 0: greedy selects token 0 (raw=10, adjusted=10).
     * Step 1: token 0 penalized to 10/1.2=8.333... Token 1 raw=9 > 8.333.
     * So step 1 greedy selects token 1, and candidate for token 0 shows
     * raw_score=10, adjusted_score=8.33333 (divergent).
     */
    const float logits[3] = { 10.0f, 9.0f, 0.0f };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_token_selection_rep.safetensors",
        "--config", "srcs/libs/test_token_selection_rep.json",
        "--tokens", "srcs/libs/test_token_selection_rep.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "2",
        "--diagnostics",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    expect_status("token selection rep write model",
                  write_safetensors_file_with_data(model_path, header, logits,
                                                   sizeof(logits)),
                  LIS_STATUS_OK);
    expect_status("token selection rep write config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("token selection rep write tokens",
                  write_text_file(token_path, "0\n"), LIS_STATUS_OK);
    expect_int("token selection rep run",
               run_cli_capture(14, argv, stdout_path, stderr_path), 0);

    /* stdout: normal output. */
    expect_file_equals("token selection rep stdout", stdout_path,
                       "generated_token_ids: 0 1\n");

    /* Step 0: token 0 selected, raw=adjusted=10 (no penalty yet). */
    expect_file_contains("token selection rep step zero rank1", stderr_path,
                         "step=0 phase=decode rank=1 token_id=0 "
                         "token_text=<unavailable> "
                         "raw_score=10 adjusted_score=10 selected=true");
    expect_file_contains("token selection rep step zero rank2", stderr_path,
                         "step=0 phase=decode rank=2 token_id=1 "
                         "token_text=<unavailable> "
                         "raw_score=9 adjusted_score=9 selected=false");

    /* Step 1: token 1 selected (adjusted=9). Token 0 has raw=10 but
     * adjusted=10/1.2=8.33333, showing penalty divergence. */
    expect_file_contains("token selection rep step one rank1 selected",
                         stderr_path,
                         "step=1 phase=decode rank=1 token_id=1 "
                         "token_text=<unavailable> "
                         "raw_score=9 adjusted_score=9 selected=true");
    expect_file_contains("token selection rep step one rank2 penalized",
                         stderr_path,
                         "step=1 phase=decode rank=2 token_id=0 "
                         "token_text=<unavailable> "
                         "raw_score=10 adjusted_score=8.33333 selected=false");

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

/*
 * Test that default stdout output is unaffected when diagnostics are disabled
 * (no top-k candidate lines appear).
 */
static void test_token_selection_no_diagnostics_no_output(void)
{
    const char *model_path =
        "srcs/libs/test_token_selection_nodiag.safetensors";
    const char *config_path = "srcs/libs/test_token_selection_nodiag.json";
    const char *token_path = "srcs/libs/test_token_selection_nodiag.txt";
    const char *stdout_path = "srcs/libs/test_token_selection_nodiag.out";
    const char *stderr_path = "srcs/libs/test_token_selection_nodiag.err";
    const char *header =
        "{\"lis.validation_logits\":{\"dtype\":\"F32\",\"shape\":[1,3],"
        "\"data_offsets\":[0,12]}}";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":4,\"intermediate_size\":8,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":4,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    const float logits[3] = { 10.0f, 9.0f, 0.0f };
    /* Note: NO --diagnostics flag. */
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_token_selection_nodiag.safetensors",
        "--config", "srcs/libs/test_token_selection_nodiag.json",
        "--tokens", "srcs/libs/test_token_selection_nodiag.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "1",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    expect_status("token selection nodiag write model",
                  write_safetensors_file_with_data(model_path, header, logits,
                                                   sizeof(logits)),
                  LIS_STATUS_OK);
    expect_status("token selection nodiag write config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("token selection nodiag write tokens",
                  write_text_file(token_path, "0\n"), LIS_STATUS_OK);
    expect_int("token selection nodiag run",
               run_cli_capture(13, argv, stdout_path, stderr_path), 0);

    /* stdout: normal generation output. */
    expect_file_equals("token selection nodiag stdout", stdout_path,
                       "generated_token_ids: 0\n");
    /* stderr: empty (no diagnostic or candidate lines). */
    expect_file_empty("token selection nodiag stderr", stderr_path);

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

/*
 * Test top-k with vocab smaller than k (edge case: vocab=3 < k=5).
 * Should report only 3 candidates without error.
 */
static void test_token_selection_small_vocab(void)
{
    const char *model_path =
        "srcs/libs/test_token_selection_small.safetensors";
    const char *config_path = "srcs/libs/test_token_selection_small.json";
    const char *token_path = "srcs/libs/test_token_selection_small.txt";
    const char *stdout_path = "srcs/libs/test_token_selection_small.out";
    const char *stderr_path = "srcs/libs/test_token_selection_small.err";
    const char *header =
        "{\"lis.validation_logits\":{\"dtype\":\"F32\",\"shape\":[1,3],"
        "\"data_offsets\":[0,12]}}";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":4,\"intermediate_size\":8,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":4,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    const float logits[3] = { 5.0f, 10.0f, 1.0f };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_token_selection_small.safetensors",
        "--config", "srcs/libs/test_token_selection_small.json",
        "--tokens", "srcs/libs/test_token_selection_small.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "1",
        "--diagnostics",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    expect_status("token selection small write model",
                  write_safetensors_file_with_data(model_path, header, logits,
                                                   sizeof(logits)),
                  LIS_STATUS_OK);
    expect_status("token selection small write config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("token selection small write tokens",
                  write_text_file(token_path, "0\n"), LIS_STATUS_OK);
    expect_int("token selection small vocab run",
               run_cli_capture(14, argv, stdout_path, stderr_path), 0);

    /* Should have exactly 3 candidate lines (vocab=3 < k=5). */
    expect_file_contains("token selection small rank1", stderr_path,
                         "rank=1 token_id=1");
    expect_file_contains("token selection small rank2", stderr_path,
                         "rank=2 token_id=0");
    expect_file_contains("token selection small rank3", stderr_path,
                         "rank=3 token_id=2");
    /* No rank=4 or rank=5 should exist; just verify rank=1..3 present. */

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

/*
 * Verify that the validation path emits phase=decode in diagnostic output
 * (no prefill concept, so all steps are "decode").
 */
static void test_diagnostics_phase_decode_validation_path(void)
{
    const char *model_path = "srcs/libs/test_phase_decode.safetensors";
    const char *config_path = "srcs/libs/test_phase_decode.json";
    const char *token_path = "srcs/libs/test_phase_decode.txt";
    const char *stdout_path = "srcs/libs/test_phase_decode.out";
    const char *stderr_path = "srcs/libs/test_phase_decode.err";
    const char *header =
        "{\"lis.validation_logits\":{\"dtype\":\"F32\",\"shape\":[1,3],"
        "\"data_offsets\":[0,12]}}";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":4,\"intermediate_size\":8,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":4,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    const float logits[3] = { 10.0f, 9.0f, 0.0f };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_phase_decode.safetensors",
        "--config", "srcs/libs/test_phase_decode.json",
        "--tokens", "srcs/libs/test_phase_decode.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "1",
        "--diagnostics",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    expect_status("phase decode write model",
                  write_safetensors_file_with_data(model_path, header, logits,
                                                   sizeof(logits)),
                  LIS_STATUS_OK);
    expect_status("phase decode write config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("phase decode write tokens",
                  write_text_file(token_path, "0\n"), LIS_STATUS_OK);
    expect_int("phase decode run",
               run_cli_capture(14, argv, stdout_path, stderr_path), 0);
    expect_file_contains("phase decode header", stderr_path,
                         "step=0 phase=decode selected_token_id=0");
    expect_file_contains("phase decode candidate", stderr_path,
                         "step=0 phase=decode rank=1");

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

/*
 * Verify that the Llama forward path emits phase=first_decode at step 0
 * in diagnostic output.
 */
static void test_diagnostics_phase_first_decode_llama_path(void)
{
    const char *model_path = "srcs/libs/test_phase_fd_model/model.safetensors";
    const char *config_path = "srcs/libs/test_phase_fd_model/config.json";
    const char *token_path = "srcs/libs/test_phase_fd.txt";
    const char *stdout_path = "srcs/libs/test_phase_fd.out";
    const char *stderr_path = "srcs/libs/test_phase_fd.err";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":1,\"intermediate_size\":1,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":1,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"eos_token_id\":2,"
        "\"max_position_embeddings\":4}";
    const char *header =
        "{\"model.embed_tokens.weight\":{\"dtype\":\"F32\",\"shape\":[3,1],"
        "\"data_offsets\":[0,12]},"
        "\"model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[12,16]},"
        "\"model.layers.0.self_attn.k_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[16,20]},"
        "\"model.layers.0.self_attn.v_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[20,24]},"
        "\"model.layers.0.self_attn.o_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[24,28]},"
        "\"model.layers.0.mlp.gate_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[28,32]},"
        "\"model.layers.0.mlp.up_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[32,36]},"
        "\"model.layers.0.mlp.down_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[36,40]},"
        "\"model.layers.0.input_layernorm.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1],\"data_offsets\":[40,44]},"
        "\"model.layers.0.post_attention_layernorm.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1],\"data_offsets\":[44,48]},"
        "\"model.norm.weight\":{\"dtype\":\"F32\",\"shape\":[1],"
        "\"data_offsets\":[48,52]},"
        "\"lm_head.weight\":{\"dtype\":\"F32\",\"shape\":[3,1],"
        "\"data_offsets\":[52,64]}}";
    const float data[16] = {
        1.0f, 2.0f, 3.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f,
        0.1f, 0.2f, 0.9f
    };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_phase_fd_model",
        "--config", "srcs/libs/test_phase_fd_model/config.json",
        "--tokens", "srcs/libs/test_phase_fd.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "1",
        "--diagnostics",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    if (system("mkdir -p srcs/libs/test_phase_fd_model") != 0) {
        fprintf(stderr, "mkdir phase fd fixture failed\n");
        ++g_failures;
        return;
    }
    expect_status("phase fd write config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("phase fd write model",
                  write_safetensors_file_with_data(model_path, header, data,
                                                   sizeof(data)),
                  LIS_STATUS_OK);
    expect_status("phase fd write tokens",
                  write_text_file(token_path, "0 1\n"), LIS_STATUS_OK);
    expect_int("phase fd run",
               run_cli_capture(14, argv, stdout_path, stderr_path), 0);
    expect_file_contains("phase first_decode header", stderr_path,
                         "step=0 phase=first_decode");
    expect_file_contains("phase first_decode candidate", stderr_path,
                         "step=0 phase=first_decode rank=1");

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    if (system("rmdir srcs/libs/test_phase_fd_model 2>/dev/null") != 0) {
        /* best effort cleanup */
    }
}

/* === Layer-trace artifact tests === */

static void test_cli_layer_trace_json_dependency_error(void)
{
    const char *model_path = "srcs/libs/test_layer_trace_err.safetensors";
    const char *config_path = "srcs/libs/test_layer_trace_err.json";
    const char *token_path = "srcs/libs/test_layer_trace_err.txt";
    const char *layer_trace_path = "srcs/libs/test_layer_trace_err_artifact.json";
    const char *stdout_path = "srcs/libs/test_layer_trace_err.out";
    const char *stderr_path = "srcs/libs/test_layer_trace_err.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_layer_trace_err.safetensors",
        "--config", "srcs/libs/test_layer_trace_err.json",
        "--tokens", "srcs/libs/test_layer_trace_err.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "1",
        "--layer-trace-json", "srcs/libs/test_layer_trace_err_artifact.json",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(layer_trace_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("layer trace dependency error",
               run_cli_capture(15, argv, stdout_path, stderr_path), 1);
    expect_file_contains("layer trace dependency stderr",
                         stderr_path,
                         "lis: artifact error: --layer-trace-json requires "
                         "--layer-checkpoints");
    expect_file_missing("layer trace no artifact on dep error",
                        layer_trace_path);
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(layer_trace_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_layer_trace_json_happy_path(void)
{
    const char *model_path = "srcs/libs/test_layer_trace_happy/model.safetensors";
    const char *config_path = "srcs/libs/test_layer_trace_happy/config.json";
    const char *token_path = "srcs/libs/test_layer_trace_happy.txt";
    const char *layer_trace_path = "srcs/libs/test_layer_trace_happy.json";
    const char *stdout_path = "srcs/libs/test_layer_trace_happy.out";
    const char *stderr_path = "srcs/libs/test_layer_trace_happy.err";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":1,\"intermediate_size\":1,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":1,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    const char *header =
        "{\"model.embed_tokens.weight\":{\"dtype\":\"F32\","
        "\"shape\":[3,1],\"data_offsets\":[0,12]},"
        "\"model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[12,16]},"
        "\"model.layers.0.self_attn.k_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[16,20]},"
        "\"model.layers.0.self_attn.v_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[20,24]},"
        "\"model.layers.0.self_attn.o_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[24,28]},"
        "\"model.layers.0.mlp.gate_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[28,32]},"
        "\"model.layers.0.mlp.up_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[32,36]},"
        "\"model.layers.0.mlp.down_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[36,40]},"
        "\"model.layers.0.input_layernorm.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1],\"data_offsets\":[40,44]},"
        "\"model.layers.0.post_attention_layernorm.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1],\"data_offsets\":[44,48]},"
        "\"model.norm.weight\":{\"dtype\":\"F32\",\"shape\":[1],"
        "\"data_offsets\":[48,52]},"
        "\"lm_head.weight\":{\"dtype\":\"F32\",\"shape\":[3,1],"
        "\"data_offsets\":[52,64]}}";
    const float data[16] = {
        1.0f, 2.0f, 3.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f,
        0.1f, 0.2f, 0.9f
    };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_layer_trace_happy",
        "--config", "srcs/libs/test_layer_trace_happy/config.json",
        "--tokens", "srcs/libs/test_layer_trace_happy.txt",
        "--context", "8",
        "--batch", "1",
        "--generate", "1",
        "--layer-checkpoints", "0",
        "--layer-trace-json", "srcs/libs/test_layer_trace_happy.json",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(layer_trace_path);
    remove(stdout_path);
    remove(stderr_path);
    if (system("mkdir -p srcs/libs/test_layer_trace_happy") != 0) {
        fprintf(stderr, "mkdir layer_trace happy fixture failed\n");
        ++g_failures;
        return;
    }
    expect_status("layer trace happy write config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("layer trace happy write model",
                  write_safetensors_file_with_data(model_path, header, data,
                                                   sizeof(data)),
                  LIS_STATUS_OK);
    expect_status("layer trace happy write tokens",
                  write_text_file(token_path, "0\n"), LIS_STATUS_OK);
    expect_int("layer trace happy path",
               run_cli_capture(17, argv, stdout_path, stderr_path), 0);
    expect_file_contains("layer trace happy schema", layer_trace_path,
                         "\"schema\":\"lis.execution_artifact/v1\"");
    expect_file_contains("layer trace happy kind", layer_trace_path,
                         "\"kind\":\"layer_trace\"");
    expect_file_contains("layer trace happy manifest", layer_trace_path,
                         "\"manifest\":");
    expect_file_contains("layer trace happy precision_path", layer_trace_path,
                         "\"precision_path\":\"f32_accum;weights=");
    expect_file_contains("layer trace happy layer_trace array", layer_trace_path,
                         "\"layer_trace\":");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(layer_trace_path);
    remove(stdout_path);
    remove(stderr_path);
    if (system("rmdir srcs/libs/test_layer_trace_happy 2>/dev/null") != 0) {
        /* best effort */
    }
}

static void test_cli_layer_trace_json_no_op_when_omitted(void)
{
    const char *model_path = "srcs/libs/test_layer_trace_noop/model.safetensors";
    const char *config_path = "srcs/libs/test_layer_trace_noop/config.json";
    const char *token_path = "srcs/libs/test_layer_trace_noop.txt";
    const char *layer_trace_path = "srcs/libs/test_layer_trace_noop.json";
    const char *stdout_path = "srcs/libs/test_layer_trace_noop.out";
    const char *stderr_path = "srcs/libs/test_layer_trace_noop.err";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":1,\"intermediate_size\":1,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":1,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_layer_trace_noop",
        "--config", "srcs/libs/test_layer_trace_noop/config.json",
        "--tokens", "srcs/libs/test_layer_trace_noop.txt",
        "--context", "8",
        "--batch", "1",
        "--generate", "1",
        "--layer-checkpoints", "0",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(layer_trace_path);
    remove(stdout_path);
    remove(stderr_path);
    if (system("mkdir -p srcs/libs/test_layer_trace_noop") != 0) {
        fprintf(stderr, "mkdir layer_trace noop fixture failed\n");
        ++g_failures;
        return;
    }
    expect_status("layer trace noop write config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("layer trace noop write model",
                  write_llama_checkpoint_fixture(model_path, 1),
                  LIS_STATUS_OK);
    expect_status("layer trace noop write tokens",
                  write_text_file(token_path, "0\n"), LIS_STATUS_OK);
    expect_int("layer trace no-op when omitted",
               run_cli_capture(15, argv, stdout_path, stderr_path), 0);
    expect_file_missing("layer trace no artifact when omitted",
                        layer_trace_path);
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(layer_trace_path);
    remove(stdout_path);
    remove(stderr_path);
    if (system("rmdir srcs/libs/test_layer_trace_noop 2>/dev/null") != 0) {
        /* best effort */
    }
}

static void test_cli_layer_trace_json_stderr_unchanged(void)
{
    const char *model_path = "srcs/libs/test_layer_trace_std/model.safetensors";
    const char *config_path = "srcs/libs/test_layer_trace_std/config.json";
    const char *token_path = "srcs/libs/test_layer_trace_std.txt";
    const char *layer_trace_path = "srcs/libs/test_layer_trace_std.json";
    const char *stdout_path = "srcs/libs/test_layer_trace_std.out";
    const char *stderr_path = "srcs/libs/test_layer_trace_std.err";
    const char *stderr_nolayer_path = "srcs/libs/test_layer_trace_std_nolayer.err";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":1,\"intermediate_size\":1,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":1,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    char *argv_with[] = {
        "lis",
        "--model", "srcs/libs/test_layer_trace_std",
        "--config", "srcs/libs/test_layer_trace_std/config.json",
        "--tokens", "srcs/libs/test_layer_trace_std.txt",
        "--context", "8",
        "--batch", "1",
        "--generate", "1",
        "--layer-checkpoints", "0",
        "--layer-trace-json", "srcs/libs/test_layer_trace_std.json",
    };
    char *argv_without[] = {
        "lis",
        "--model", "srcs/libs/test_layer_trace_std",
        "--config", "srcs/libs/test_layer_trace_std/config.json",
        "--tokens", "srcs/libs/test_layer_trace_std.txt",
        "--context", "8",
        "--batch", "1",
        "--generate", "1",
        "--layer-checkpoints", "0",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(layer_trace_path);
    remove(stdout_path);
    remove(stderr_path);
    remove(stderr_nolayer_path);
    if (system("mkdir -p srcs/libs/test_layer_trace_std") != 0) {
        fprintf(stderr, "mkdir layer_trace std fixture failed\n");
        ++g_failures;
        return;
    }
    expect_status("layer trace std write config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("layer trace std write model",
                  write_llama_checkpoint_fixture(model_path, 1),
                  LIS_STATUS_OK);
    expect_status("layer trace std write tokens",
                  write_text_file(token_path, "0\n"), LIS_STATUS_OK);
    expect_int("layer trace stderr unchanged with",
               run_cli_capture(17, argv_with, stdout_path, stderr_path), 0);
    expect_int("layer trace stderr unchanged without",
               run_cli_capture(15, argv_without, stdout_path, stderr_nolayer_path), 0);
    {
        char *with_data = read_file_content(stderr_path);
        char *without_data = read_file_content(stderr_nolayer_path);
        expect_string_equals("layer trace stderr byte identical",
                             with_data, with_data ? strlen(with_data) : 0,
                             without_data ? without_data : "");
        free(with_data);
        free(without_data);
    }
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(layer_trace_path);
    remove(stdout_path);
    remove(stderr_path);
    remove(stderr_nolayer_path);
    if (system("rmdir srcs/libs/test_layer_trace_std 2>/dev/null") != 0) {
        /* best effort */
    }
}

/*
 * Verify that --layer-checkpoints emits the tensor summary statistics for the
 * 32-layer comparison checkpoint set.
 */
static void test_cli_layer_checkpoints(void)
{
    const char *model_path = "srcs/libs/test_layer_chk_model/model.safetensors";
    const char *config_path = "srcs/libs/test_layer_chk_model/config.json";
    const char *token_path = "srcs/libs/test_layer_chk.txt";
    const char *stdout_path = "srcs/libs/test_layer_chk.out";
    const char *stderr_path = "srcs/libs/test_layer_chk.err";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":32,"
        "\"hidden_size\":1,\"intermediate_size\":1,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":1,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"eos_token_id\":2,"
        "\"max_position_embeddings\":8}";
    const char *step_values[4] = { "0", "1", "2", "3" };
    const char *step_phases[4] = { "prefill", "decode", "decode", "decode" };
    const size_t checkpoint_layers[5] = { 0, 8, 16, 24, 31 };
    size_t step;
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_layer_chk_model",
        "--config", "srcs/libs/test_layer_chk_model/config.json",
        "--tokens", "srcs/libs/test_layer_chk.txt",
        "--context", "8",
        "--batch", "1",
        "--generate", "4",
        "--layer-checkpoints", "0",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    if (system("mkdir -p srcs/libs/test_layer_chk_model") != 0) {
        fprintf(stderr, "mkdir layer_chk fixture failed\n");
        ++g_failures;
        return;
    }
    expect_status("layer chk write config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("layer chk write model",
                  write_llama_checkpoint_fixture(model_path, 32),
                  LIS_STATUS_OK);
    expect_status("layer chk write tokens",
                  write_text_file(token_path, "0 1\n"), LIS_STATUS_OK);
    for (step = 0; step < 4; ++step) {
        char expected[128];

        argv[14] = (char *)step_values[step];
        expect_int("layer chk run",
                   run_cli_capture(15, argv, stdout_path, stderr_path), 0);
        expect_file_equals("layer chk stdout unchanged", stdout_path,
                           "generated_token_ids: 0 0 0 0\n");
        expect_file_occurrences("layer chk one checkpoint set", stderr_path,
                                "lis: layer-checkpoint ", 79);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=embedding shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk embedding output", stderr_path,
                             expected);
        {
            size_t layer_index;

            for (layer_index = 0; layer_index < 5; ++layer_index) {
                snprintf(expected, sizeof(expected),
                         "lis: layer-checkpoint step=%zu phase=%s "
                         "name=layer.%zu.output shape=[1]",
                         step, step_phases[step],
                         checkpoint_layers[layer_index]);
                expect_file_contains("layer chk layer output", stderr_path,
                                     expected);
            }
        }
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.7.q_proj_out shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk layer 7 q proj out", stderr_path,
                             expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.7.k_proj_out shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk layer 7 k proj out", stderr_path,
                             expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.7.v_proj_out shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk layer 7 v proj out", stderr_path,
                             expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.7.q_after_rope shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk layer 7 q after rope", stderr_path,
                             expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.7.k_after_rope shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk layer 7 k after rope", stderr_path,
                             expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.7.attn_scores shape=[%zu]",
                 step, step_phases[step], step + 2U);
        expect_file_contains("layer chk layer 7 attn scores", stderr_path,
                             expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.7.attn_probs shape=[%zu]",
                 step, step_phases[step], step + 2U);
        expect_file_contains("layer chk layer 7 attn probs", stderr_path,
                             expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.7.attn_context shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk layer 7 attn context", stderr_path,
                             expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.7.attn_out shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk layer 7 attn out", stderr_path,
                             expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.7.post_attn_residual shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk layer 7 post attn residual",
                             stderr_path, expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.7.mlp_out shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk layer 7 mlp out", stderr_path,
                             expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.7.post_mlp_residual shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk layer 7 post mlp residual",
                             stderr_path, expected);
        expect_file_not_contains("layer chk duplicate layer 7 output",
                                 stderr_path, "name=layer.7.output shape=");
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.8.input shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk layer 8 input", stderr_path,
                             expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.8.input_layernorm_out shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk layer 8 input layernorm out",
                             stderr_path, expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.8.q_proj_out shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk layer 8 q proj out", stderr_path,
                             expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.8.k_proj_out shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk layer 8 k proj out", stderr_path,
                             expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.8.v_proj_out shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk layer 8 v proj out", stderr_path,
                             expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.8.q_after_rope shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk layer 8 q after rope", stderr_path,
                             expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.8.k_after_rope shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk layer 8 k after rope", stderr_path,
                             expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.8.attn_scores shape=[%zu]",
                 step, step_phases[step], step + 2U);
        expect_file_contains("layer chk layer 8 attn scores", stderr_path,
                             expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.8.attn_probs shape=[%zu]",
                 step, step_phases[step], step + 2U);
        expect_file_contains("layer chk layer 8 attn probs", stderr_path,
                             expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.8.attn_context shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk layer 8 attn context", stderr_path,
                             expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.8.attn_out shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk layer 8 attn out", stderr_path,
                             expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.8.post_attn_residual shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk layer 8 post attn residual",
                             stderr_path, expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.8.mlp_out shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk layer 8 mlp out", stderr_path,
                             expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=layer.8.post_mlp_residual shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk layer 8 post mlp residual",
                             stderr_path, expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=final_norm shape=[1]",
                 step, step_phases[step]);
        expect_file_contains("layer chk final norm", stderr_path, expected);
        snprintf(expected, sizeof(expected),
                 "lis: layer-checkpoint step=%zu phase=%s name=logits shape=[3]",
                 step, step_phases[step]);
        expect_file_contains("layer chk logits", stderr_path, expected);
    }

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    if (system("rmdir srcs/libs/test_layer_chk_model 2>/dev/null") != 0) {
        /* best effort cleanup */
    }
}

/*
 * Verify that --forced-prefix without --diagnostics is rejected.
 */
static void test_forced_prefix_requires_diagnostics(void)
{
    const char *stdout_path = "srcs/libs/test_fp_nodiag.out";
    const char *stderr_path = "srcs/libs/test_fp_nodiag.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_phase_fd_model",
        "--config", "srcs/libs/test_phase_fd_model/config.json",
        "--tokens", "srcs/libs/test_phase_fd.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "1",
        "--forced-prefix", "0 1",
    };

    remove(stdout_path);
    remove(stderr_path);
    /* Should fail with exit code 2 (argument parse rejection). */
    expect_int("forced prefix requires diagnostics",
               run_cli_capture(15, argv, stdout_path, stderr_path), 2);
    remove(stdout_path);
    remove(stderr_path);
}

/*
 * Verify that --forced-prefix with validation path model is rejected.
 */
static void test_forced_prefix_requires_llama_path(void)
{
    const char *model_path = "srcs/libs/test_fp_val.safetensors";
    const char *config_path = "srcs/libs/test_fp_val.json";
    const char *token_path = "srcs/libs/test_fp_val.txt";
    const char *stdout_path = "srcs/libs/test_fp_val.out";
    const char *stderr_path = "srcs/libs/test_fp_val.err";
    const char *header =
        "{\"lis.validation_logits\":{\"dtype\":\"F32\",\"shape\":[1,3],"
        "\"data_offsets\":[0,12]}}";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":4,\"intermediate_size\":8,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":4,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    const float logits[3] = { 10.0f, 9.0f, 0.0f };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_fp_val.safetensors",
        "--config", "srcs/libs/test_fp_val.json",
        "--tokens", "srcs/libs/test_fp_val.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "1",
        "--diagnostics",
        "--forced-prefix", "0 1",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    expect_status("fp val write model",
                  write_safetensors_file_with_data(model_path, header, logits,
                                                   sizeof(logits)),
                  LIS_STATUS_OK);
    expect_status("fp val write config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("fp val write tokens",
                  write_text_file(token_path, "0\n"), LIS_STATUS_OK);
    expect_int("forced prefix rejects validation path",
               run_cli_capture(16, argv, stdout_path, stderr_path), 1);
    expect_file_contains("fp val stderr", stderr_path,
                         "--forced-prefix requires supported HuggingFace decoder model path");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_report_json_requires_forced_prefix_binding(void)
{
    const char *report_path = "srcs/libs/test_cli_report_forced_prefix.json";
    const char *stdout_path = "srcs/libs/test_cli_report_forced_prefix.out";
    const char *stderr_path = "srcs/libs/test_cli_report_forced_prefix.err";
    char *argv[] = {
        "lis",
        "--model", "unused-model",
        "--config", "unused-config",
        "--tokens", "unused-tokens",
        "--context", "1",
        "--batch", "1",
        "--generate", "1",
        "--diagnostics",
        "--forced-prefix", "0",
        "--report-json", "srcs/libs/test_cli_report_forced_prefix.json",
    };

    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
    expect_int("cli report forced prefix rejection",
               run_cli_capture(18, argv, stdout_path, stderr_path), 1);
    expect_file_empty("cli report forced prefix stdout", stdout_path);
    expect_file_contains("cli report forced prefix stderr", stderr_path,
                         "--report-json with --forced-prefix requires "
                         "--forced-prefix-binding-json");
    expect_file_missing("cli report forced prefix artifact", report_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
}

static int write_forced_binding_variant(
    const char *path,
    int applied,
    size_t token_count,
    size_t target_step,
    size_t runtime_step,
    size_t context_position,
    const char *policy_sha256,
    int include_localization,
    const char *suffix)
{
    char json[4096];
    const char *localization = include_localization
        ? ",\"source_localization_ref_sha256\":"
          "\"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd\""
        : "";
    int written = snprintf(
        json, sizeof(json),
        "{\"mode\":\"injected_selected_token_prefix_v1\","
        "\"applied\":%s,\"token_count\":%zu,"
        "\"token_ids_sha256\":"
        "\"sha256:463f2998327eb3a694145e6014444480b2235be84aa6cfd57871cc64f1cd816c\","
        "\"prefix_start_generated_step\":0,"
        "\"prefix_end_generated_step_exclusive\":%zu,"
        "\"target_generated_token_step\":%zu,"
        "\"runtime_checkpoint_step\":%zu,"
        "\"prompt_token_count\":1,\"context_position\":%zu,"
        "\"selection_policy\":\"lis_policy_modified_greedy_v1\","
        "\"selection_policy_sha256\":\"%s\","
        "\"source_pass0_artifact_sha256\":"
        "\"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\","
        "\"source_original_run_report_sha256\":"
        "\"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\","
        "\"source_pass1_artifact_sha256\":"
        "\"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\""
        "%s%s}",
        applied ? "true" : "false", token_count, token_count, target_step,
        runtime_step, context_position, policy_sha256, localization,
        suffix != NULL ? suffix : "");

    if (written < 0 || (size_t)written >= sizeof(json)) {
        return 0;
    }
    return write_text_file(path, json) == LIS_STATUS_OK;
}

static void expect_forced_binding_rejected(const char *name,
                                           const char *binding_path)
{
    const char *report_path = "srcs/libs/test_fp_binding_reject_report.json";
    const char *stdout_path = "srcs/libs/test_fp_binding_reject.out";
    const char *stderr_path = "srcs/libs/test_fp_binding_reject.err";
    char *argv[] = {
        "lis", "--model", "unused-model", "--config", "unused-config",
        "--tokens", "unused-tokens", "--context", "8", "--batch", "1",
        "--generate", "3", "--forced-prefix", "0 1",
        "--forced-prefix-binding-json", (char *)binding_path,
        "--report-json", (char *)report_path,
    };

    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
    expect_int(name, run_cli_capture(19, argv, stdout_path, stderr_path), 1);
    expect_file_contains(name, stderr_path, "invalid forced-prefix binding");
    expect_file_missing(name, report_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_forced_prefix_binding_adversarial_inputs(void)
{
    static const char policy_sha[] =
        "sha256:63f64c98586bc3cf31bbcccda5f6f354faba9ba47675780e77608309f7c912d0";
    static const char wrong_policy_sha[] =
        "sha256:0000000000000000000000000000000000000000000000000000000000000000";
    const char *binding_path = "srcs/libs/test_fp_binding_reject.json";
    const char *target_path = "srcs/libs/test_fp_binding_target.json";
    const char *link_path = "srcs/libs/test_fp_binding_link.json";
    const char *stdout_path = "srcs/libs/test_fp_sign.out";
    const char *stderr_path = "srcs/libs/test_fp_sign.err";
    char too_many[256] = {0};
    size_t offset = 0U;
    size_t index;
    char *signed_argv[] = {
        "lis", "--model", "unused", "--config", "unused", "--tokens",
        "unused", "--context", "8", "--batch", "1", "--generate", "3",
        "--forced-prefix", "+0 1", "--forced-prefix-binding-json",
        (char *)binding_path, "--report-json", "unused-report",
    };

    remove(binding_path);
    remove(target_path);
    remove(link_path);
    expect_int("write missing-field binding",
               write_forced_binding_variant(binding_path, 1, 2U, 2U, 3U,
                                            3U, policy_sha, 0, ""), 1);
    expect_forced_binding_rejected("forced binding missing field",
                                   binding_path);

    expect_int("write extra-field binding",
               write_forced_binding_variant(
                   binding_path, 1, 2U, 2U, 3U, 3U, policy_sha, 1,
                   ",\"unexpected\":0"), 1);
    expect_forced_binding_rejected("forced binding extra field",
                                   binding_path);

    expect_int("write duplicate-field binding",
               write_forced_binding_variant(
                   binding_path, 1, 2U, 2U, 3U, 3U, policy_sha, 0,
                   ",\"mode\":\"injected_selected_token_prefix_v1\""), 1);
    expect_forced_binding_rejected("forced binding duplicate field",
                                   binding_path);

    expect_int("write wrong-policy binding",
               write_forced_binding_variant(binding_path, 1, 2U, 2U, 3U,
                                            3U, wrong_policy_sha, 1, ""), 1);
    expect_forced_binding_rejected("forced binding wrong policy SHA",
                                   binding_path);

    expect_int("write wrong-step binding",
               write_forced_binding_variant(binding_path, 1, 2U, 3U, 3U,
                                            3U, policy_sha, 1, ""), 1);
    expect_forced_binding_rejected("forced binding wrong target step",
                                   binding_path);

    expect_int("write wrong-context binding",
               write_forced_binding_variant(binding_path, 1, 2U, 2U, 3U,
                                            4U, policy_sha, 1, ""), 1);
    expect_forced_binding_rejected("forced binding wrong context",
                                   binding_path);

    expect_int("write count-65 binding",
               write_forced_binding_variant(binding_path, 1, 65U, 65U, 66U,
                                            66U, policy_sha, 1, ""), 1);
    expect_forced_binding_rejected("forced binding count 65",
                                   binding_path);

    expect_int("write false-applied binding",
               write_forced_binding_variant(binding_path, 0, 2U, 2U, 3U,
                                            3U, policy_sha, 1, ""), 1);
    expect_forced_binding_rejected("forced binding applied false",
                                   binding_path);

    expect_int("write symlink target binding",
               write_forced_binding_variant(target_path, 1, 2U, 2U, 3U,
                                            3U, policy_sha, 1, ""), 1);
    expect_int("create binding symlink", symlink(target_path, link_path), 0);
    expect_forced_binding_rejected("forced binding symlink", link_path);

    expect_int("write signed-prefix binding",
               write_forced_binding_variant(binding_path, 1, 2U, 2U, 3U,
                                            3U, policy_sha, 1, ""), 1);
    remove(stdout_path);
    remove(stderr_path);
    expect_int("forced prefix rejects explicit sign",
               run_cli_capture(19, signed_argv, stdout_path, stderr_path), 1);
    expect_file_contains("forced prefix signed stderr", stderr_path,
                         "invalid forced prefix");

    for (index = 0U; index < 65U; ++index) {
        int written = snprintf(too_many + offset, sizeof(too_many) - offset,
                               "%s0", index == 0U ? "" : " ");

        if (written < 0 || (size_t)written >= sizeof(too_many) - offset) {
            ++g_failures;
            break;
        }
        offset += (size_t)written;
    }
    signed_argv[14] = too_many;
    remove(stdout_path);
    remove(stderr_path);
    expect_int("forced prefix rejects count 65",
               run_cli_capture(19, signed_argv, stdout_path, stderr_path), 1);
    expect_file_contains("forced prefix count stderr", stderr_path,
                         "invalid forced prefix");

    remove(binding_path);
    remove(target_path);
    remove(link_path);
    remove(stdout_path);
    remove(stderr_path);
    remove("unused-report");
}

/*
 * Verify that --forced-prefix on the Llama path produces forced_prefix_next
 * diagnostics with the expected info header and candidate output.
 */
static void test_forced_prefix_diagnostics_llama_path(void)
{
    const char *model_path = "srcs/libs/test_fp_llama/model.safetensors";
    const char *config_path = "srcs/libs/test_fp_llama/config.json";
    const char *token_path = "srcs/libs/test_fp_llama.txt";
    const char *stdout_path = "srcs/libs/test_fp_llama.out";
    const char *stderr_path = "srcs/libs/test_fp_llama.err";
    const char *binding_path = "srcs/libs/test_fp_llama_binding.json";
    const char *bad_binding_path = "srcs/libs/test_fp_llama_bad_binding.json";
    const char *report_path = "srcs/libs/test_fp_llama_report.json";
    const char *bad_report_path = "srcs/libs/test_fp_llama_bad_report.json";
    const char *report_stdout_path = "srcs/libs/test_fp_llama_report.out";
    const char *report_stderr_path = "srcs/libs/test_fp_llama_report.err";
    const char *bad_stdout_path = "srcs/libs/test_fp_llama_bad.out";
    const char *bad_stderr_path = "srcs/libs/test_fp_llama_bad.err";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":1,\"intermediate_size\":1,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":1,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"eos_token_id\":2,"
        "\"max_position_embeddings\":8}";
    const char *header =
        "{\"model.embed_tokens.weight\":{\"dtype\":\"F32\",\"shape\":[3,1],"
        "\"data_offsets\":[0,12]},"
        "\"model.layers.0.self_attn.q_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[12,16]},"
        "\"model.layers.0.self_attn.k_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[16,20]},"
        "\"model.layers.0.self_attn.v_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[20,24]},"
        "\"model.layers.0.self_attn.o_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[24,28]},"
        "\"model.layers.0.mlp.gate_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[28,32]},"
        "\"model.layers.0.mlp.up_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[32,36]},"
        "\"model.layers.0.mlp.down_proj.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1,1],\"data_offsets\":[36,40]},"
        "\"model.layers.0.input_layernorm.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1],\"data_offsets\":[40,44]},"
        "\"model.layers.0.post_attention_layernorm.weight\":{\"dtype\":\"F32\","
        "\"shape\":[1],\"data_offsets\":[44,48]},"
        "\"model.norm.weight\":{\"dtype\":\"F32\",\"shape\":[1],"
        "\"data_offsets\":[48,52]},"
        "\"lm_head.weight\":{\"dtype\":\"F32\",\"shape\":[3,1],"
        "\"data_offsets\":[52,64]}}";
    const float data[16] = {
        1.0f, 2.0f, 3.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f,
        0.1f, 0.2f, 0.9f
    };
    const char *binding_json =
        "{\"mode\":\"injected_selected_token_prefix_v1\","
        "\"applied\":true,\"token_count\":2,"
        "\"token_ids_sha256\":\"sha256:463f2998327eb3a694145e6014444480b2235be84aa6cfd57871cc64f1cd816c\","
        "\"prefix_start_generated_step\":0,"
        "\"prefix_end_generated_step_exclusive\":2,"
        "\"target_generated_token_step\":2,"
        "\"runtime_checkpoint_step\":3,"
        "\"prompt_token_count\":1,\"context_position\":3,"
        "\"selection_policy\":\"lis_policy_modified_greedy_v1\","
        "\"selection_policy_sha256\":\"sha256:63f64c98586bc3cf31bbcccda5f6f354faba9ba47675780e77608309f7c912d0\","
        "\"source_pass0_artifact_sha256\":\"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\","
        "\"source_original_run_report_sha256\":\"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\","
        "\"source_pass1_artifact_sha256\":\"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\","
        "\"source_localization_ref_sha256\":\"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd\"}";
    const char *bad_binding_json =
        "{\"mode\":\"injected_selected_token_prefix_v1\","
        "\"applied\":true,\"token_count\":2,"
        "\"token_ids_sha256\":\"sha256:0000000000000000000000000000000000000000000000000000000000000000\","
        "\"prefix_start_generated_step\":0,"
        "\"prefix_end_generated_step_exclusive\":2,"
        "\"target_generated_token_step\":2,"
        "\"runtime_checkpoint_step\":3,"
        "\"prompt_token_count\":1,\"context_position\":3,"
        "\"selection_policy\":\"lis_policy_modified_greedy_v1\","
        "\"selection_policy_sha256\":\"sha256:63f64c98586bc3cf31bbcccda5f6f354faba9ba47675780e77608309f7c912d0\","
        "\"source_pass0_artifact_sha256\":\"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\","
        "\"source_original_run_report_sha256\":\"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\","
        "\"source_pass1_artifact_sha256\":\"sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc\","
        "\"source_localization_ref_sha256\":\"sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd\"}";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_fp_llama",
        "--config", "srcs/libs/test_fp_llama/config.json",
        "--tokens", "srcs/libs/test_fp_llama.txt",
        "--context", "8",
        "--batch", "1",
        "--generate", "1",
        "--diagnostics",
        "--forced-prefix", "0 1",
    };
    char *report_argv[] = {
        "lis",
        "--model", "srcs/libs/test_fp_llama",
        "--config", "srcs/libs/test_fp_llama/config.json",
        "--tokens", "srcs/libs/test_fp_llama.txt",
        "--context", "8",
        "--batch", "1",
        "--generate", "3",
        "--forced-prefix", "0 1",
        "--forced-prefix-binding-json",
        "srcs/libs/test_fp_llama_binding.json",
        "--report-json", "srcs/libs/test_fp_llama_report.json",
    };
    char *bad_report_argv[] = {
        "lis",
        "--model", "srcs/libs/test_fp_llama",
        "--config", "srcs/libs/test_fp_llama/config.json",
        "--tokens", "srcs/libs/test_fp_llama.txt",
        "--context", "8",
        "--batch", "1",
        "--generate", "3",
        "--forced-prefix", "0 1",
        "--forced-prefix-binding-json",
        "srcs/libs/test_fp_llama_bad_binding.json",
        "--report-json", "srcs/libs/test_fp_llama_bad_report.json",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    remove(binding_path);
    remove(bad_binding_path);
    remove(report_path);
    remove(bad_report_path);
    remove(report_stdout_path);
    remove(report_stderr_path);
    remove(bad_stdout_path);
    remove(bad_stderr_path);
    if (system("mkdir -p srcs/libs/test_fp_llama") != 0) {
        fprintf(stderr, "mkdir fp llama fixture failed\n");
        ++g_failures;
        return;
    }
    expect_status("fp llama write config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("fp llama write model",
                  write_safetensors_file_with_data(model_path, header, data,
                                                   sizeof(data)),
                  LIS_STATUS_OK);
    expect_status("fp llama write tokens",
                  write_text_file(token_path, "0\n"), LIS_STATUS_OK);
    expect_status("fp llama write binding",
                  write_text_file(binding_path, binding_json), LIS_STATUS_OK);
    expect_status("fp llama write bad binding",
                  write_text_file(bad_binding_path, bad_binding_json),
                  LIS_STATUS_OK);
    expect_int("fp llama run",
               run_cli_capture(16, argv, stdout_path, stderr_path), 0);

    /* Forced-prefix info header. */
    expect_file_contains("fp info header", stderr_path,
                         "forced-prefix-info prompt_tokens=1 "
                         "forced_prefix_tokens=2 forced_prefix_ids=0,1");

    /* phase=forced_prefix_next in diagnostic header and candidates. */
    expect_file_contains("fp phase header", stderr_path,
                         "phase=forced_prefix_next");
    expect_file_contains("fp step index", stderr_path,
                         "step=2 phase=forced_prefix_next");
    expect_file_contains("fp candidate rank1", stderr_path,
                         "step=2 phase=forced_prefix_next rank=1");

    /* No normal generation output on stdout. */
    expect_file_empty("fp llama stdout empty", stdout_path);

    expect_int("fp report-bound run",
               run_cli_capture(19, report_argv, report_stdout_path,
                               report_stderr_path), 0);
    expect_file_empty("fp report-bound stdout empty", report_stdout_path);
    expect_file_empty("fp report-bound stderr empty", report_stderr_path);
    expect_file_contains("fp report object", report_path,
                         "\"forced_prefix\":{\"mode\":"
                         "\"injected_selected_token_prefix_v1\"");
    expect_file_contains("fp report digest", report_path,
                         "\"token_ids_sha256\":"
                         "\"sha256:463f2998327eb3a694145e6014444480b2235be84aa6cfd57871cc64f1cd816c\"");
    expect_file_contains("fp report source binding", report_path,
                         "\"source_original_run_report_sha256\":"
                         "\"sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\"");
    expect_file_contains("fp report selected target", report_path,
                         "\"selected_token_count\":1");
    expect_file_not_contains("fp report omits raw prefix IDs", report_path,
                             "\"selected_token_ids\":[0,1");
    expect_file_not_contains("fp report omits raw prefix field", report_path,
                             "full_forced_prefix_token_ids");

    expect_int("fp report rejects digest mismatch",
               run_cli_capture(19, bad_report_argv, bad_stdout_path,
                               bad_stderr_path), 1);
    expect_file_empty("fp bad report stdout empty", bad_stdout_path);
    expect_file_contains("fp bad binding stderr", bad_stderr_path,
                         "does not match the applied prefix");
    expect_file_missing("fp bad report absent", bad_report_path);

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    remove(binding_path);
    remove(bad_binding_path);
    remove(report_path);
    remove(bad_report_path);
    remove(report_stdout_path);
    remove(report_stderr_path);
    remove(bad_stdout_path);
    remove(bad_stderr_path);
    if (system("rmdir srcs/libs/test_fp_llama 2>/dev/null") != 0) {
        /* best effort cleanup */
    }
}

/*
 * Markdown companion report tests
 */

static void test_cli_report_md_success(void)
{
    const char *model_path = "srcs/libs/test_cli_report_md_model.safetensors";
    const char *config_path = "srcs/libs/test_cli_report_md_config.json";
    const char *token_path = "srcs/libs/test_cli_report_md_tokens.txt";
    const char *md_path = "srcs/libs/test_cli_report_md.md";
    const char *stdout_path = "srcs/libs/test_cli_report_md.out";
    const char *stderr_path = "srcs/libs/test_cli_report_md.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_report_md_model.safetensors",
        "--config", "srcs/libs/test_cli_report_md_config.json",
        "--tokens", "srcs/libs/test_cli_report_md_tokens.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "2",
        "--report-md", "srcs/libs/test_cli_report_md.md",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(md_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli report md success",
               run_cli_capture(15, argv, stdout_path, stderr_path), 0);
    expect_file_equals("cli report md stdout", stdout_path,
                       "generated_token_ids: 2 2\n");
    expect_file_empty("cli report md stderr", stderr_path);
    /* Markdown heading and canonical note */
    expect_file_contains("cli report md heading", md_path,
                         "# LIS Execution Report");
    expect_file_contains("cli report md canonical note", md_path,
                         "JSON artifact remains the canonical machine-readable source of truth");
    /* Outcome section */
    expect_file_contains("cli report md status ok", md_path,
                          "Status: OK");
    expect_file_contains("cli report md stop reason", md_path,
                         "Stop reason: decode_limit");
    /* Identity section */
    expect_file_contains("cli report md schema", md_path,
                         "Schema: `lis.execution_artifact/v1`");
    expect_file_contains("cli report md kind", md_path,
                         "Kind: `run_report`");
    /* Retention Policy */
    expect_file_contains("cli report md retention", md_path,
                         "## Retention Policy");
    expect_file_contains("cli report md retention paths", md_path,
                         "Absolute paths: omitted");
    /* Runtime section */
    expect_file_contains("cli report md runtime", md_path,
                         "## Runtime");
    expect_file_contains("cli report md context", md_path,
                         "Configured context: 4");
    /* KV Cache */
    expect_file_contains("cli report md kv cache", md_path,
                         "## KV Cache");
    expect_file_contains("cli report md kv scope", md_path,
                         "Scope: run_local");
    expect_file_contains("cli report md kv dtype", md_path,
                         "Storage dtype: f32");
    expect_file_contains("cli report md kv max tokens", md_path,
                         "Max tokens: 4");
    expect_file_contains("cli report md kv used tokens", md_path,
                         "Used tokens: 4");
    expect_file_contains("cli report md kv bytes per token", md_path,
                         "Bytes per token: 64");
    expect_file_contains("cli report md kv allocated", md_path,
                         "Allocated bytes: 256");
    expect_file_contains("cli report md kv used bytes", md_path,
                         "Used bytes: 256");
    expect_file_contains("cli report md kv shape", md_path,
                         "Shape: layers=1, batch=2, kv_heads=1, head_dim=4, "
                         "element_size=4");
    /* Token Accounting */
    expect_file_contains("cli report md token accounting", md_path,
                         "## Token Accounting");
    /* Selected and Emitted tokens bounded list */
    expect_file_contains("cli report md selected tokens", md_path,
                         "Selected tokens: 2");
    expect_file_contains("cli report md emitted tokens", md_path,
                         "Emitted tokens: 2");
    /* Bounded token list with backtick code markdown */
    expect_file_contains("cli report md token list", md_path, "`");
    /* Fingerprints */
    expect_file_contains("cli report md fingerprints", md_path,
                         "## Fingerprints");
    expect_file_contains("cli report md binary fingerprint", md_path,
                         "- Binary:");
    /* Notes */
    expect_file_contains("cli report md notes", md_path,
                         "## Notes");
    expect_file_contains("cli report md notes json canonical", md_path,
                         "JSON artifact");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(md_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_report_md_and_json_coexist(void)
{
    const char *model_path = "srcs/libs/test_cli_report_both_model.safetensors";
    const char *config_path = "srcs/libs/test_cli_report_both_config.json";
    const char *token_path = "srcs/libs/test_cli_report_both_tokens.txt";
    const char *json_path = "srcs/libs/test_cli_report_both.json";
    const char *md_path = "srcs/libs/test_cli_report_both.md";
    const char *stdout_path = "srcs/libs/test_cli_report_both.out";
    const char *stderr_path = "srcs/libs/test_cli_report_both.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_report_both_model.safetensors",
        "--config", "srcs/libs/test_cli_report_both_config.json",
        "--tokens", "srcs/libs/test_cli_report_both_tokens.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "2",
        "--report-json", "srcs/libs/test_cli_report_both.json",
        "--report-md", "srcs/libs/test_cli_report_both.md",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(json_path);
    remove(md_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli report json+md coexist",
               run_cli_capture(17, argv, stdout_path, stderr_path), 0);
    expect_file_equals("cli report json+md stdout", stdout_path,
                       "generated_token_ids: 2 2\n");
    expect_file_empty("cli report json+md stderr", stderr_path);
    expect_file_contains("cli report json exists", json_path,
                         "\"schema\":\"lis.execution_artifact/v1\"");
    expect_file_contains("cli report md exists", md_path,
                         "# LIS Execution Report");
    expect_file_contains("cli report md status from both", md_path,
                          "Status: OK");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(json_path);
    remove(md_path);
    remove(stdout_path);
    remove(stderr_path);
}

/*
 * Verify that a run with neither --report-json nor --report-md does not emit
 * either file and behaves identically to before.
 */
static void test_cli_no_report_md_unchanged_behavior(void)
{
    const char *model_path = "srcs/libs/test_cli_no_report_model.safetensors";
    const char *config_path = "srcs/libs/test_cli_no_report_config.json";
    const char *token_path = "srcs/libs/test_cli_no_report_tokens.txt";
    const char *json_path = "srcs/libs/test_cli_no_report.json";
    const char *md_path = "srcs/libs/test_cli_no_report.md";
    const char *stdout_path = "srcs/libs/test_cli_no_report.out";
    const char *stderr_path = "srcs/libs/test_cli_no_report.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_no_report_model.safetensors",
        "--config", "srcs/libs/test_cli_no_report_config.json",
        "--tokens", "srcs/libs/test_cli_no_report_tokens.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "2",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(json_path);
    remove(md_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli no report unchanged",
               run_cli_capture(13, argv, stdout_path, stderr_path), 0);
    expect_file_equals("cli no report stdout", stdout_path,
                       "generated_token_ids: 2 2\n");
    expect_file_empty("cli no report stderr", stderr_path);
    expect_file_missing("cli no report json missing", json_path);
    expect_file_missing("cli no report md missing", md_path);
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(json_path);
    remove(md_path);
    remove(stdout_path);
    remove(stderr_path);
}

/*
 * Verify --report-md with perf enabled includes performance sections.
 */
static void test_cli_report_md_with_perf(void)
{
    const char *model_path = "srcs/libs/test_cli_report_md_perf_model.safetensors";
    const char *config_path = "srcs/libs/test_cli_report_md_perf_config.json";
    const char *token_path = "srcs/libs/test_cli_report_md_perf_tokens.txt";
    const char *md_path = "srcs/libs/test_cli_report_md_perf.md";
    const char *stdout_path = "srcs/libs/test_cli_report_md_perf.out";
    const char *stderr_path = "srcs/libs/test_cli_report_md_perf.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_report_md_perf_model.safetensors",
        "--config", "srcs/libs/test_cli_report_md_perf_config.json",
        "--tokens", "srcs/libs/test_cli_report_md_perf_tokens.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "2",
        "--perf",
        "--report-md", "srcs/libs/test_cli_report_md_perf.md",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(md_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli report md with perf",
               run_cli_capture(16, argv, stdout_path, stderr_path), 0);
    expect_file_contains("cli report md perf summary heading", md_path,
                         "## Performance Summary");
    expect_file_contains("cli report md stage timings heading", md_path,
                         "### Stage Timings");
    expect_file_contains("cli report md stage table", md_path,
                         "| Stage | ms | Tokens |");
    expect_file_contains("cli report md ttfs", md_path,
                         "- TTFT:");
    expect_file_contains("cli report md tps", md_path,
                         "- Steady-state TPS:");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(md_path);
    remove(stdout_path);
    remove(stderr_path);
}

/*
 * Verify --report-md does not enforce --report-json.
 */
static void test_cli_report_md_without_json(void)
{
    const char *model_path = "srcs/libs/test_cli_report_md_only_model.safetensors";
    const char *config_path = "srcs/libs/test_cli_report_md_only_config.json";
    const char *token_path = "srcs/libs/test_cli_report_md_only_tokens.txt";
    const char *json_path = "srcs/libs/test_cli_report_md_only.json";
    const char *md_path = "srcs/libs/test_cli_report_md_only.md";
    const char *stdout_path = "srcs/libs/test_cli_report_md_only.out";
    const char *stderr_path = "srcs/libs/test_cli_report_md_only.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_report_md_only_model.safetensors",
        "--config", "srcs/libs/test_cli_report_md_only_config.json",
        "--tokens", "srcs/libs/test_cli_report_md_only_tokens.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "2",
        "--report-md", "srcs/libs/test_cli_report_md_only.md",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(json_path);
    remove(md_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli report md without json",
               run_cli_capture(15, argv, stdout_path, stderr_path), 0);
    expect_file_contains("cli report md only md exists", md_path,
                         "# LIS Execution Report");
    expect_file_missing("cli report md only json missing", json_path);
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(json_path);
    remove(md_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_trace_json_success(void)
{
    const char *model_path = "srcs/libs/test_cli_trace_model.safetensors";
    const char *config_path = "srcs/libs/test_cli_trace_config.json";
    const char *token_path = "srcs/libs/test_cli_trace_tokens.txt";
    const char *trace_path = "srcs/libs/test_cli_trace_success.json";
    const char *stdout_path = "srcs/libs/test_cli_trace_success.out";
    const char *stderr_path = "srcs/libs/test_cli_trace_success.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_trace_model.safetensors",
        "--config", "srcs/libs/test_cli_trace_config.json",
        "--tokens", "srcs/libs/test_cli_trace_tokens.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "2",
        "--trace-json", "srcs/libs/test_cli_trace_success.json",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli trace success",
               run_cli_capture(15, argv, stdout_path, stderr_path), 0);
    expect_file_contains("cli trace schema", trace_path,
                          "\"schema\":\"lis.execution_artifact/v1\"");
    expect_file_contains("cli trace kind", trace_path,
                          "\"kind\":\"decode_trace\"");
    expect_file_contains("cli trace manifest", trace_path,
                          "\"manifest\":{");
    expect_file_contains("cli trace decode_trace array", trace_path,
                          "\"decode_trace\":[");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_trace_json_absent_no_change(void)
{
    const char *model_path = "srcs/libs/test_cli_trace_absent_model.safetensors";
    const char *config_path = "srcs/libs/test_cli_trace_absent_config.json";
    const char *token_path = "srcs/libs/test_cli_trace_absent_tokens.txt";
    const char *stdout_path = "srcs/libs/test_cli_trace_absent.out";
    const char *stderr_path = "srcs/libs/test_cli_trace_absent.err";
    char *argv_no_trace[] = {
        "lis",
        "--model", "srcs/libs/test_cli_trace_absent_model.safetensors",
        "--config", "srcs/libs/test_cli_trace_absent_config.json",
        "--tokens", "srcs/libs/test_cli_trace_absent_tokens.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "2",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli trace absent",
               run_cli_capture(13, argv_no_trace, stdout_path, stderr_path), 0);
    expect_file_equals("cli trace absent stdout", stdout_path,
                       "generated_token_ids: 2 2\n");
    expect_file_empty("cli trace absent stderr", stderr_path);
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_trace_json_fields(void)
{
    const char *model_path = "srcs/libs/test_cli_trace_fields_model.safetensors";
    const char *config_path = "srcs/libs/test_cli_trace_fields_config.json";
    const char *token_path = "srcs/libs/test_cli_trace_fields_tokens.txt";
    const char *trace_path = "srcs/libs/test_cli_trace_fields.json";
    const char *stdout_path = "srcs/libs/test_cli_trace_fields.out";
    const char *stderr_path = "srcs/libs/test_cli_trace_fields.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_trace_fields_model.safetensors",
        "--config", "srcs/libs/test_cli_trace_fields_config.json",
        "--tokens", "srcs/libs/test_cli_trace_fields_tokens.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "2",
        "--trace-json", "srcs/libs/test_cli_trace_fields.json",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli trace fields",
               run_cli_capture(15, argv, stdout_path, stderr_path), 0);
    expect_file_contains("cli trace step field", trace_path,
                          "\"step\":");
    expect_file_contains("cli trace phase field", trace_path,
                          "\"phase\":");
    expect_file_contains("cli trace selected_token_id", trace_path,
                          "\"selected_token_id\":");
    expect_file_contains("cli trace raw_score_selected", trace_path,
                          "\"raw_score_selected\":");
    expect_file_contains("cli trace adjusted_score_selected", trace_path,
                          "\"adjusted_score_selected\":");
    expect_file_contains("cli trace runner_up_token_id", trace_path,
                          "\"runner_up_token_id\":");
    expect_file_contains("cli trace runner_up_adjusted_score", trace_path,
                          "\"runner_up_adjusted_score\":");
    expect_file_contains("cli trace decision_margin", trace_path,
                          "\"decision_margin\":");
    expect_file_contains("cli trace structural_suppression_affected", trace_path,
                          "\"structural_suppression_affected\":");
    expect_file_contains("cli trace repetition_penalty_changed_selection",
                         trace_path,
                          "\"repetition_penalty_changed_selection\":");
    expect_file_contains("cli trace selected_token_penalized", trace_path,
                          "\"selected_token_penalized\":");
    expect_file_contains("cli trace suppressed_token_count", trace_path,
                          "\"suppressed_token_count\":");
    expect_file_contains("cli trace penalized_token_count", trace_path,
                          "\"penalized_token_count\":");
    expect_file_contains("cli trace decision_class", trace_path,
                          "\"decision_class\":");
    expect_file_contains("cli trace topk array", trace_path,
                          "\"topk\":[");
    expect_file_contains("cli trace topk token_id", trace_path,
                          "\"token_id\":");
    expect_file_contains("cli trace topk raw_score", trace_path,
                          "\"raw_score\":");
    expect_file_contains("cli trace topk adjusted_score", trace_path,
                          "\"adjusted_score\":");
    expect_file_contains("cli trace topk is_selected", trace_path,
                          "\"is_selected\":");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_trace_json_decision_margin(void)
{
    const char *model_path = "srcs/libs/test_cli_trace_margin_model.safetensors";
    const char *config_path = "srcs/libs/test_cli_trace_margin_config.json";
    const char *token_path = "srcs/libs/test_cli_trace_margin_tokens.txt";
    const char *trace_path = "srcs/libs/test_cli_trace_margin.json";
    const char *stdout_path = "srcs/libs/test_cli_trace_margin.out";
    const char *stderr_path = "srcs/libs/test_cli_trace_margin.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_trace_margin_model.safetensors",
        "--config", "srcs/libs/test_cli_trace_margin_config.json",
        "--tokens", "srcs/libs/test_cli_trace_margin_tokens.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "2",
        "--trace-json", "srcs/libs/test_cli_trace_margin.json",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli trace margin",
               run_cli_capture(15, argv, stdout_path, stderr_path), 0);
    expect_file_contains("cli trace margin positive", trace_path,
                          "\"decision_margin\":");
    expect_file_occurrences("cli trace step count matches generate",
                            trace_path, "\"step\":0", 1);
    expect_file_occurrences("cli trace step 1 count", trace_path,
                            "\"step\":1", 1);
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_trace_json_topk_selected(void)
{
    const char *model_path = "srcs/libs/test_cli_trace_topk_model.safetensors";
    const char *config_path = "srcs/libs/test_cli_trace_topk_config.json";
    const char *token_path = "srcs/libs/test_cli_trace_topk_tokens.txt";
    const char *trace_path = "srcs/libs/test_cli_trace_topk.json";
    const char *stdout_path = "srcs/libs/test_cli_trace_topk.out";
    const char *stderr_path = "srcs/libs/test_cli_trace_topk.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_trace_topk_model.safetensors",
        "--config", "srcs/libs/test_cli_trace_topk_config.json",
        "--tokens", "srcs/libs/test_cli_trace_topk_tokens.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "2",
        "--trace-json", "srcs/libs/test_cli_trace_topk.json",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli trace topk",
               run_cli_capture(15, argv, stdout_path, stderr_path), 0);
    expect_file_contains("cli trace topk has selected true", trace_path,
                          "\"is_selected\":true");
    expect_file_occurrences("cli trace topk selected true count",
                            trace_path, "\"is_selected\":true", 2);
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_trace_json_no_report_json_mutation(void)
{
    const char *model_path = "srcs/libs/test_cli_trace_no_mut_model.safetensors";
    const char *config_path = "srcs/libs/test_cli_trace_no_mut_config.json";
    const char *token_path = "srcs/libs/test_cli_trace_no_mut_tokens.txt";
    const char *report_path = "srcs/libs/test_cli_trace_no_mut_report.json";
    const char *trace_path = "srcs/libs/test_cli_trace_no_mut_trace.json";
    const char *stdout_path = "srcs/libs/test_cli_trace_no_mut.out";
    const char *stderr_path = "srcs/libs/test_cli_trace_no_mut.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_trace_no_mut_model.safetensors",
        "--config", "srcs/libs/test_cli_trace_no_mut_config.json",
        "--tokens", "srcs/libs/test_cli_trace_no_mut_tokens.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "2",
        "--report-json", "srcs/libs/test_cli_trace_no_mut_report.json",
        "--trace-json", "srcs/libs/test_cli_trace_no_mut_trace.json",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli trace no mut",
               run_cli_capture(17, argv, stdout_path, stderr_path), 0);
    expect_file_contains("cli trace report unchanged kind", report_path,
                          "\"kind\":\"run_report\"");
    expect_file_contains("cli trace report unchanged schema", report_path,
                          "\"schema\":\"lis.execution_artifact/v1\"");
    expect_file_contains("cli trace report has selected ids", report_path,
                          "\"selected_token_ids\":[2,2]");
    expect_file_contains("cli trace artifact kind", trace_path,
                          "\"kind\":\"decode_trace\"");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_trace_json_no_stderr_change(void)
{
    const char *model_path = "srcs/libs/test_cli_trace_stderr_model.safetensors";
    const char *config_path = "srcs/libs/test_cli_trace_stderr_config.json";
    const char *token_path = "srcs/libs/test_cli_trace_stderr_tokens.txt";
    const char *stdout_path_no_trace = "srcs/libs/test_cli_trace_stderr_no.out";
    const char *stderr_path_no_trace = "srcs/libs/test_cli_trace_stderr_no.err";
    const char *trace_path = "srcs/libs/test_cli_trace_stderr_trace.json";
    const char *stdout_path_with_trace = "srcs/libs/test_cli_trace_stderr_with.out";
    const char *stderr_path_with_trace = "srcs/libs/test_cli_trace_stderr_with.err";
    char *argv_no_trace[] = {
        "lis",
        "--model", "srcs/libs/test_cli_trace_stderr_model.safetensors",
        "--config", "srcs/libs/test_cli_trace_stderr_config.json",
        "--tokens", "srcs/libs/test_cli_trace_stderr_tokens.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "2",
    };
    char *argv_with_trace[] = {
        "lis",
        "--model", "srcs/libs/test_cli_trace_stderr_model.safetensors",
        "--config", "srcs/libs/test_cli_trace_stderr_config.json",
        "--tokens", "srcs/libs/test_cli_trace_stderr_tokens.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "2",
        "--trace-json", "srcs/libs/test_cli_trace_stderr_trace.json",
    };
    long no_trace_size = 0;
    long with_trace_size = 0;
    FILE *fp = NULL;

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path_no_trace);
    remove(stderr_path_no_trace);
    remove(trace_path);
    remove(stdout_path_with_trace);
    remove(stderr_path_with_trace);
    write_valid_fixtures(model_path, config_path, token_path);

    expect_int("cli trace stderr no trace",
               run_cli_capture(13, argv_no_trace, stdout_path_no_trace,
                              stderr_path_no_trace), 0);
    expect_int("cli trace stderr with trace",
               run_cli_capture(15, argv_with_trace, stdout_path_with_trace,
                              stderr_path_with_trace), 0);

    fp = fopen(stderr_path_no_trace, "rb");
    if (fp != NULL) {
        if (fseek(fp, 0, SEEK_END) == 0) {
            no_trace_size = ftell(fp);
        }
        fclose(fp);
    }
    fp = fopen(stderr_path_with_trace, "rb");
    if (fp != NULL) {
        if (fseek(fp, 0, SEEK_END) == 0) {
            with_trace_size = ftell(fp);
        }
        fclose(fp);
    }
    expect_int("cli trace stderr same size", (int)with_trace_size,
               (int)no_trace_size);

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path_no_trace);
    remove(stderr_path_no_trace);
    remove(trace_path);
    remove(stdout_path_with_trace);
    remove(stderr_path_with_trace);
}

static void test_cli_reasoning_line_with_diagnostics(void)
{
    const char *model_path = "srcs/libs/test_cli_reason_diag.safetensors";
    const char *config_path = "srcs/libs/test_cli_reason_diag.json";
    const char *token_path = "srcs/libs/test_cli_reason_diag.txt";
    const char *stdout_path = "srcs/libs/test_cli_reason_diag.out";
    const char *stderr_path = "srcs/libs/test_cli_reason_diag.err";
    const char *header =
        "{\"lis.validation_logits\":{\"dtype\":\"F32\",\"shape\":[1,3],"
        "\"data_offsets\":[0,12]}}";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":4,\"intermediate_size\":8,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":4,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    const float logits[3] = { 10.0f, 9.0f, 0.0f };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_reason_diag.safetensors",
        "--config", "srcs/libs/test_cli_reason_diag.json",
        "--tokens", "srcs/libs/test_cli_reason_diag.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "2",
        "--diagnostics",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    expect_status("write reason diag model",
                  write_safetensors_file_with_data(model_path, header, logits,
                                                   sizeof(logits)),
                  LIS_STATUS_OK);
    expect_status("write reason diag config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("write reason diag tokens",
                  write_text_file(token_path, "0\n"), LIS_STATUS_OK);
    expect_int("cli reasoning diag",
               run_cli_capture(14, argv, stdout_path, stderr_path), 0);
    expect_file_contains("cli reasoning line present", stderr_path,
                          "lis: generation-diagnostic-reasoning");
    expect_file_contains("cli reasoning decision_class", stderr_path,
                          "decision_class=");
    expect_file_contains("cli reasoning margin", stderr_path,
                          "margin=");
    expect_file_contains("cli reasoning runner_up", stderr_path,
                          "runner_up_token_id=");
    expect_file_contains("cli reasoning suppressed_count", stderr_path,
                          "suppressed_token_count=");
    expect_file_contains("cli reasoning penalized_count", stderr_path,
                          "penalized_token_count=");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_reasoning_line_absent_without_diagnostics(void)
{
    const char *model_path = "srcs/libs/test_cli_reason_no_diag.safetensors";
    const char *config_path = "srcs/libs/test_cli_reason_no_diag.json";
    const char *token_path = "srcs/libs/test_cli_reason_no_diag.txt";
    const char *stdout_path = "srcs/libs/test_cli_reason_no_diag.out";
    const char *stderr_path = "srcs/libs/test_cli_reason_no_diag.err";
    const char *header =
        "{\"lis.validation_logits\":{\"dtype\":\"F32\",\"shape\":[1,3],"
        "\"data_offsets\":[0,12]}}";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":4,\"intermediate_size\":8,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":4,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    const float logits[3] = { 10.0f, 9.0f, 0.0f };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_reason_no_diag.safetensors",
        "--config", "srcs/libs/test_cli_reason_no_diag.json",
        "--tokens", "srcs/libs/test_cli_reason_no_diag.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "2",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    expect_status("write reason no diag model",
                  write_safetensors_file_with_data(model_path, header, logits,
                                                   sizeof(logits)),
                  LIS_STATUS_OK);
    expect_status("write reason no diag config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("write reason no diag tokens",
                  write_text_file(token_path, "0\n"), LIS_STATUS_OK);
    expect_int("cli reasoning absent",
               run_cli_capture(13, argv, stdout_path, stderr_path), 0);
    expect_file_not_contains("cli no reasoning without diagnostics",
                             stderr_path,
                             "lis: generation-diagnostic-reasoning");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_reasoning_line_ordering(void)
{
    const char *model_path = "srcs/libs/test_cli_reason_order.safetensors";
    const char *config_path = "srcs/libs/test_cli_reason_order.json";
    const char *token_path = "srcs/libs/test_cli_reason_order.txt";
    const char *stdout_path = "srcs/libs/test_cli_reason_order.out";
    const char *stderr_path = "srcs/libs/test_cli_reason_order.err";
    const char *header =
        "{\"lis.validation_logits\":{\"dtype\":\"F32\",\"shape\":[1,3],"
        "\"data_offsets\":[0,12]}}";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":4,\"intermediate_size\":8,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":4,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    const float logits[3] = { 10.0f, 9.0f, 0.0f };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_reason_order.safetensors",
        "--config", "srcs/libs/test_cli_reason_order.json",
        "--tokens", "srcs/libs/test_cli_reason_order.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "2",
        "--diagnostics",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    expect_status("write reason order model",
                  write_safetensors_file_with_data(model_path, header, logits,
                                                   sizeof(logits)),
                  LIS_STATUS_OK);
    expect_status("write reason order config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("write reason order tokens",
                  write_text_file(token_path, "0\n"), LIS_STATUS_OK);
    expect_int("cli reasoning order",
               run_cli_capture(14, argv, stdout_path, stderr_path), 0);
    {
        char *content = read_file_content(stderr_path);
        if (content != NULL) {
            char *diag = strstr(content,
                                "lis: generation-diagnostic step=");
            char *reason = strstr(content,
                                  "lis: generation-diagnostic-reasoning");
            if (diag != NULL && reason != NULL) {
                expect_int("cli reasoning after header",
                           (reason > diag) ? 1 : 0, 1);
            }
            free(content);
        }
    }
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_reasoning_line_trace_only_absent(void)
{
    const char *model_path = "srcs/libs/test_cli_reason_trace_only.safetensors";
    const char *config_path = "srcs/libs/test_cli_reason_trace_only.json";
    const char *token_path = "srcs/libs/test_cli_reason_trace_only.txt";
    const char *trace_path = "srcs/libs/test_cli_reason_trace_only.json";
    const char *stdout_path = "srcs/libs/test_cli_reason_trace_only.out";
    const char *stderr_path = "srcs/libs/test_cli_reason_trace_only.err";
    const char *header =
        "{\"lis.validation_logits\":{\"dtype\":\"F32\",\"shape\":[1,3],"
        "\"data_offsets\":[0,12]}}";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":4,\"intermediate_size\":8,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":4,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    const float logits[3] = { 10.0f, 9.0f, 0.0f };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_reason_trace_only.safetensors",
        "--config", "srcs/libs/test_cli_reason_trace_only.json",
        "--tokens", "srcs/libs/test_cli_reason_trace_only.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "2",
        "--trace-json", "srcs/libs/test_cli_reason_trace_only.json",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
    expect_status("write reason trace only model",
                  write_safetensors_file_with_data(model_path, header, logits,
                                                   sizeof(logits)),
                  LIS_STATUS_OK);
    expect_status("write reason trace only config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("write reason trace only tokens",
                  write_text_file(token_path, "0\n"), LIS_STATUS_OK);
    expect_int("cli reasoning trace only",
               run_cli_capture(15, argv, stdout_path, stderr_path), 0);
    expect_file_not_contains("cli no reasoning with trace only",
                             stderr_path,
                             "lis: generation-diagnostic-reasoning");
    expect_file_not_contains("cli no diag header with trace only",
                             stderr_path,
                             "lis: generation-diagnostic step=");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
}

/* Precision path observability tests */
static void test_cli_precision_stderr_diagnostics(void)
{
    const char *model_path = "srcs/libs/test_cli_precision_diag.safetensors";
    const char *config_path = "srcs/libs/test_cli_precision_diag.json";
    const char *token_path = "srcs/libs/test_cli_precision_diag.txt";
    const char *stdout_path = "srcs/libs/test_cli_precision_diag.out";
    const char *stderr_path = "srcs/libs/test_cli_precision_diag.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_precision_diag.safetensors",
        "--config", "srcs/libs/test_cli_precision_diag.json",
        "--tokens", "srcs/libs/test_cli_precision_diag.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "1",
        "--diagnostics",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli precision stderr diagnostics",
               run_cli_capture(14, argv, stdout_path, stderr_path), 0);
    expect_file_contains("cli precision diag stderr line", stderr_path,
                         "lis: precision path=f32_accum weights=f32 kv=f32");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_precision_stderr_perf(void)
{
    const char *model_path = "srcs/libs/test_cli_precision_perf.safetensors";
    const char *config_path = "srcs/libs/test_cli_precision_perf.json";
    const char *token_path = "srcs/libs/test_cli_precision_perf.txt";
    const char *stdout_path = "srcs/libs/test_cli_precision_perf.out";
    const char *stderr_path = "srcs/libs/test_cli_precision_perf.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_precision_perf.safetensors",
        "--config", "srcs/libs/test_cli_precision_perf.json",
        "--tokens", "srcs/libs/test_cli_precision_perf.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "1",
        "--perf",
    };
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli precision stderr perf",
               run_cli_capture(14, argv, stdout_path, stderr_path), 0);
    expect_file_contains("cli precision perf stderr line", stderr_path,
                         "lis: precision path=f32_accum weights=f32 kv=f32");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_precision_stderr_quiet_absent(void)
{
    const char *model_path = "srcs/libs/test_cli_precision_quiet.safetensors";
    const char *config_path = "srcs/libs/test_cli_precision_quiet.json";
    const char *token_path = "srcs/libs/test_cli_precision_quiet.txt";
    const char *stdout_path = "srcs/libs/test_cli_precision_quiet.out";
    const char *stderr_path = "srcs/libs/test_cli_precision_quiet.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_precision_quiet.safetensors",
        "--config", "srcs/libs/test_cli_precision_quiet.json",
        "--tokens", "srcs/libs/test_cli_precision_quiet.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "1",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli precision quiet run",
               run_cli_capture(13, argv, stdout_path, stderr_path), 0);
    expect_file_not_contains("cli precision quiet absent", stderr_path,
                             "lis: precision path=");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_precision_line_ordering(void)
{
    const char *model_path = "srcs/libs/test_cli_precision_order.safetensors";
    const char *config_path = "srcs/libs/test_cli_precision_order.json";
    const char *token_path = "srcs/libs/test_cli_precision_order.txt";
    const char *stdout_path = "srcs/libs/test_cli_precision_order.out";
    const char *stderr_path = "srcs/libs/test_cli_precision_order.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_precision_order.safetensors",
        "--config", "srcs/libs/test_cli_precision_order.json",
        "--tokens", "srcs/libs/test_cli_precision_order.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "1",
        "--diagnostics",
    };
    char *content;

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli precision ordering",
               run_cli_capture(14, argv, stdout_path, stderr_path), 0);
    content = read_file_content(stderr_path);
    if (content != NULL) {
        char *simd = strstr(content, "lis: simd backend=");
        char *prec = strstr(content, "lis: precision path=");

        if (simd == NULL || prec == NULL || prec < simd) {
            fprintf(stderr,
                    "cli precision ordering: expected simd before precision\n");
            ++g_failures;
        }
        free(content);
    } else {
        fprintf(stderr, "cli precision ordering: could not read stderr\n");
        ++g_failures;
    }
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_precision_report_json(void)
{
    const char *model_path = "srcs/libs/test_cli_precision_report.safetensors";
    const char *config_path = "srcs/libs/test_cli_precision_report.json";
    const char *token_path = "srcs/libs/test_cli_precision_report.txt";
    const char *report_path = "srcs/libs/test_cli_precision_report.json.out";
    const char *stdout_path = "srcs/libs/test_cli_precision_report.out";
    const char *stderr_path = "srcs/libs/test_cli_precision_report.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_precision_report.safetensors",
        "--config", "srcs/libs/test_cli_precision_report.json",
        "--tokens", "srcs/libs/test_cli_precision_report.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "1",
        "--report-json", "srcs/libs/test_cli_precision_report.json.out",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli precision report json",
               run_cli_capture(15, argv, stdout_path, stderr_path), 0);
    expect_file_contains("cli precision report json field", report_path,
                         "\"precision_path\":\"f32_accum;weights=f32;kv=f32\"");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_precision_trace_json(void)
{
    const char *model_path = "srcs/libs/test_cli_precision_trace.safetensors";
    const char *config_path = "srcs/libs/test_cli_precision_trace.json";
    const char *token_path = "srcs/libs/test_cli_precision_trace.txt";
    const char *trace_path = "srcs/libs/test_cli_precision_trace.json.out";
    const char *stdout_path = "srcs/libs/test_cli_precision_trace.out";
    const char *stderr_path = "srcs/libs/test_cli_precision_trace.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_precision_trace.safetensors",
        "--config", "srcs/libs/test_cli_precision_trace.json",
        "--tokens", "srcs/libs/test_cli_precision_trace.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "1",
        "--trace-json", "srcs/libs/test_cli_precision_trace.json.out",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli precision trace json",
               run_cli_capture(15, argv, stdout_path, stderr_path), 0);
    expect_file_contains("cli precision trace json field", trace_path,
                         "\"precision_path\":\"f32_accum;weights=f32;kv=f32\"");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_precision_fingerprint_stable(void)
{
    const char *model_path = "srcs/libs/test_cli_precision_fp.safetensors";
    const char *config_path = "srcs/libs/test_cli_precision_fp.json";
    const char *token_path = "srcs/libs/test_cli_precision_fp.txt";
    const char *report_path = "srcs/libs/test_cli_precision_fp.json.out";
    const char *stdout_path = "srcs/libs/test_cli_precision_fp.out";
    const char *stderr_path = "srcs/libs/test_cli_precision_fp.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_precision_fp.safetensors",
        "--config", "srcs/libs/test_cli_precision_fp.json",
        "--tokens", "srcs/libs/test_cli_precision_fp.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "1",
        "--report-json", "srcs/libs/test_cli_precision_fp.json.out",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli precision fingerprint stable",
               run_cli_capture(15, argv, stdout_path, stderr_path), 0);
    /* Assert that runtime_fingerprint and backend_fingerprint are still present
     * and not zeroed out, confirming no new inputs were introduced. */
    expect_file_contains("cli precision fp runtime fingerprint", report_path,
                         "\"runtime\":{");
    expect_file_contains("cli precision fp backend fingerprint", report_path,
                         "\"backend\":{\"name\":");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
}

/* Eviction-free KV cache diagnostics */
static void test_cli_kv_cache_diagnostic_with_diagnostics(void)
{
    const char *model_path = "srcs/libs/test_cli_kv_diag.safetensors";
    const char *config_path = "srcs/libs/test_cli_kv_diag.json";
    const char *token_path = "srcs/libs/test_cli_kv_diag.txt";
    const char *stdout_path = "srcs/libs/test_cli_kv_diag.out";
    const char *stderr_path = "srcs/libs/test_cli_kv_diag.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_kv_diag.safetensors",
        "--config", "srcs/libs/test_cli_kv_diag.json",
        "--tokens", "srcs/libs/test_cli_kv_diag.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "1",
        "--diagnostics",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli kv diagnostic run",
               run_cli_capture(14, argv, stdout_path, stderr_path), 0);
    expect_file_occurrences("cli kv diagnostic one line", stderr_path,
                            "lis: kv-cache:", 1U);
    expect_file_contains(
        "cli kv diagnostic format", stderr_path,
        "lis: kv-cache: scope=run_local policy=eviction_free,monotonic "
        "dtype=f32 max_tokens=4 used_tokens=3 bytes_per_token=64 "
        "allocated_bytes=256 used_bytes=192");
    expect_file_not_contains("cli kv diagnostic no exhausted line",
                             stderr_path, "lis: kv-cache: exhausted");

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_kv_cache_diagnostic_quiet_absent(void)
{
    const char *model_path = "srcs/libs/test_cli_kv_quiet.safetensors";
    const char *config_path = "srcs/libs/test_cli_kv_quiet.json";
    const char *token_path = "srcs/libs/test_cli_kv_quiet.txt";
    const char *stdout_path = "srcs/libs/test_cli_kv_quiet.out";
    const char *stderr_path = "srcs/libs/test_cli_kv_quiet.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_kv_quiet.safetensors",
        "--config", "srcs/libs/test_cli_kv_quiet.json",
        "--tokens", "srcs/libs/test_cli_kv_quiet.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "1",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli kv quiet run",
               run_cli_capture(13, argv, stdout_path, stderr_path), 0);
    expect_file_not_contains("cli kv quiet absent", stderr_path,
                             "lis: kv-cache:");

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_kv_cache_diagnostic_perf_only_absent(void)
{
    const char *model_path = "srcs/libs/test_cli_kv_perf.safetensors";
    const char *config_path = "srcs/libs/test_cli_kv_perf.json";
    const char *token_path = "srcs/libs/test_cli_kv_perf.txt";
    const char *stdout_path = "srcs/libs/test_cli_kv_perf.out";
    const char *stderr_path = "srcs/libs/test_cli_kv_perf.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_kv_perf.safetensors",
        "--config", "srcs/libs/test_cli_kv_perf.json",
        "--tokens", "srcs/libs/test_cli_kv_perf.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "1",
        "--perf",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli kv perf-only run",
               run_cli_capture(14, argv, stdout_path, stderr_path), 0);
    expect_file_contains("cli kv perf keeps perf summary", stderr_path,
                         "lis: perf-summary tag=none threads=1 "
                         "prompt_tokens=");
    expect_file_not_contains("cli kv perf-only absent", stderr_path,
                             "lis: kv-cache:");

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_kv_cache_diagnostic_with_perf_preserves_perf(void)
{
    const char *model_path = "srcs/libs/test_cli_kv_diag_perf.safetensors";
    const char *config_path = "srcs/libs/test_cli_kv_diag_perf.json";
    const char *token_path = "srcs/libs/test_cli_kv_diag_perf.txt";
    const char *stdout_path = "srcs/libs/test_cli_kv_diag_perf.out";
    const char *stderr_path = "srcs/libs/test_cli_kv_diag_perf.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_cli_kv_diag_perf.safetensors",
        "--config", "srcs/libs/test_cli_kv_diag_perf.json",
        "--tokens", "srcs/libs/test_cli_kv_diag_perf.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "1",
        "--diagnostics",
        "--perf",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("cli kv diagnostics perf run",
               run_cli_capture(15, argv, stdout_path, stderr_path), 0);
    expect_file_contains("cli kv diagnostics perf summary", stderr_path,
                         "lis: perf-summary tag=none threads=1 "
                         "prompt_tokens=");
    expect_file_contains("cli kv diagnostics generation line", stderr_path,
                         "lis: generation-diagnostic step=0 phase=decode ");
    expect_file_occurrences("cli kv diagnostics perf one line", stderr_path,
                            "lis: kv-cache:", 1U);

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_kv_cache_flag_absent_from_help(void)
{
    const char *stdout_path = "srcs/libs/test_cli_kv_help.out";
    const char *stderr_path = "srcs/libs/test_cli_kv_help.err";
    char *argv[] = { "lis", "--help" };

    remove(stdout_path);
    remove(stderr_path);
    expect_int("cli kv help run",
               run_cli_capture(2, argv, stdout_path, stderr_path), 0);
    expect_file_not_contains("cli kv flag absent from help", stdout_path,
                             "--kv-cache");

    remove(stdout_path);
    remove(stderr_path);
}

#include "test_cli_layer_trace_tests.c"
#include "test_cli_artifact_compat.c"
#include "test_cli_artifact_regression.c"
#include "test_cli_kv_cache_semantics.c"
#include "test_cli_pass3_producer.c"
#include "test_cli_intra_layer.c"
#include "test_cli_pass4_integration.c"

int main(void)
{
    test_cli_llama_instruct_prompt_builder();
    test_cli_help();
    test_cli_invalid_arguments();
    test_cli_happy_path();
    test_cli_report_json_success();
    test_cli_token_batch_rejection();
    test_cli_token_format_rejection();
    test_cli_vocab_rejection();
    test_cli_validation_logits_rejection();
    test_token_batch_public_validation();
    test_cli_context_rejection();
    test_cli_unsupported_rope_config_rejection();
    test_cli_decode_limit_rejection();
    test_cli_report_json_context_limit();
    test_cli_validation_eos_stop();
    test_cli_structural_control_token_suppression();
    test_cli_structural_stop_token_hidden();
    test_cli_canonical_hf_prompt_empty_user_text();
    test_cli_generation_diagnostics_token_output();
    test_cli_generation_diagnostics_model_eos();
    test_cli_generation_diagnostics_tokenizer_output();
    test_cli_report_json_tokenizer_perf();
    test_cli_generation_diagnostics_structural_stop();
    test_cli_generation_diagnostics_context_limit();
    test_cli_repetition_penalty_text_output();
    test_cli_repetition_penalty_token_output();
    test_cli_hf_llama_forward_path();
    test_cli_hf_qwen3_forward_path();

    /* Extended token selection diagnostics */
    test_token_selection_candidates_token_ids();
    test_token_selection_candidates_with_tokenizer();
    test_token_selection_repetition_penalty_scores();
    test_token_selection_no_diagnostics_no_output();
    test_token_selection_small_vocab();

    /* First-decode-step phase marker diagnostics */
    test_diagnostics_phase_decode_validation_path();
    test_diagnostics_phase_first_decode_llama_path();
    
    /* Layer-trace artifact */
    test_cli_layer_trace_json_dependency_error();
    test_cli_layer_trace_json_happy_path();
    test_cli_layer_trace_json_no_op_when_omitted();
    test_cli_layer_trace_json_stderr_unchanged();

    /* Layer-checkpoint diagnostics */
    test_cli_layer_checkpoints();

    /* Forced-prefix diagnostics */
    test_forced_prefix_requires_diagnostics();
    test_forced_prefix_requires_llama_path();
    test_cli_report_json_requires_forced_prefix_binding();
    test_forced_prefix_binding_adversarial_inputs();
    test_forced_prefix_diagnostics_llama_path();

    /* Markdown companion report tests */
    test_cli_report_md_success();
    test_cli_report_md_and_json_coexist();
    test_cli_no_report_md_unchanged_behavior();
    test_cli_report_md_with_perf();
    test_cli_report_md_without_json();

    /* --trace-json tests */
    test_cli_trace_json_success();
    test_cli_trace_json_absent_no_change();
    test_cli_trace_json_fields();
    test_cli_trace_json_decision_margin();
    test_cli_trace_json_topk_selected();
    test_cli_trace_json_no_report_json_mutation();
    test_cli_trace_json_no_stderr_change();

    /* Reasoning diagnostics */
    test_cli_reasoning_line_with_diagnostics();
    test_cli_reasoning_line_absent_without_diagnostics();
    test_cli_reasoning_line_ordering();
    test_cli_reasoning_line_trace_only_absent();

    /* Precision path observability */
    test_cli_precision_stderr_diagnostics();
    test_cli_precision_stderr_perf();
    test_cli_precision_stderr_quiet_absent();
    test_cli_precision_line_ordering();
    test_cli_precision_report_json();
    test_cli_precision_trace_json();
    test_cli_precision_fingerprint_stable();

    /* Eviction-free KV cache diagnostics */
    test_cli_kv_cache_diagnostic_with_diagnostics();
    test_cli_kv_cache_diagnostic_quiet_absent();
    test_cli_kv_cache_diagnostic_perf_only_absent();
    test_cli_kv_cache_diagnostic_with_perf_preserves_perf();
    test_cli_kv_cache_flag_absent_from_help();

    /* KV correctness and boundary regression tests */
    test_kv_cache_semantics_accounting_formulas();
    test_kv_cache_semantics_precision_storage_consistency();
    test_kv_cache_semantics_kv_cache_determinism();
    test_kv_cache_semantics_context_limit_accounting();
    test_kv_cache_semantics_qwen3_smoke_skip_gated();

    test_cli_layer_trace_json_fingerprint_stable();
    test_cli_layer_trace_json_precision_path();
    test_cli_layer_trace_json_attn_scale_exclusion();

    /* LIS Inspect compatibility protection */
    test_cli_artifact_compat_run_report_legacy_surface();
    test_cli_artifact_compat_run_report_perf_surface();
    test_cli_artifact_compat_stderr_perf_surface();
    test_cli_artifact_compat_generation_diagnostic_surface();
    test_cli_artifact_compat_layer_checkpoint_surface();
    test_cli_artifact_compat_trace_artifact_validity();
    test_cli_artifact_compat_layer_trace_artifact_validity();

    /* Regression and validation coverage */
    test_cli_artifact_regression_default_additive_off();
    test_cli_artifact_regression_run_report_key_order();
    test_cli_artifact_regression_report_perf_canonical();
    test_cli_artifact_regression_trace_margin_identity();
    test_cli_artifact_regression_trace_no_runner_up_null_triple();
    test_cli_artifact_regression_trace_topk_bounds_and_selected();
    test_cli_artifact_regression_trace_phase_and_stop();
    test_cli_artifact_regression_decision_class_greedy();
    test_cli_artifact_regression_decision_class_penalty_shifted();
    test_cli_artifact_regression_decision_class_structural_and_taxonomy();
    test_cli_artifact_regression_layer_trace_parity_and_precision();
    test_cli_artifact_regression_runtime_fingerprint_repeatable();
    test_cli_artifact_regression_trace_determinism();
    test_cli_artifact_regression_trace_eos_stop();

    test_cli_intra_layer_surface();

    test_cli_pass3_producer_contract();

    test_cli_pass4_real_artifact_integration();

    if (g_failures != 0) {
        fprintf(stderr, "%d CLI test failure(s)\n", g_failures);
        return 1;
    }

    return 0;
}
