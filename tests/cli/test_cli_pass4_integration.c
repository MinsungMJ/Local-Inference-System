/* P4-12 two-generation real-artifact integration, included by test_cli.c. */

#include "lis_test_controls.h"

#include <sys/stat.h>

#define P4_12_PATH_CAPACITY 512U
#define P4_12_CASE_RUN_COUNT 9U
#define P4_12_TARGET_LAYER 4U

typedef struct {
    char report[P4_12_PATH_CAPACITY];
    char decode[P4_12_PATH_CAPACITY];
    char layer[P4_12_PATH_CAPACITY];
    char stdout_path[P4_12_PATH_CAPACITY];
    char stderr_path[P4_12_PATH_CAPACITY];
} p4_12_cli_artifacts;

typedef struct {
    char directory[P4_12_PATH_CAPACITY];
    char config[P4_12_PATH_CAPACITY];
    char model[P4_12_PATH_CAPACITY];
    char same_tokens[P4_12_PATH_CAPACITY];
    char different_tokens[P4_12_PATH_CAPACITY];
    p4_12_cli_artifacts runs[P4_12_CASE_RUN_COUNT];
} p4_12_case_fixture;

static const char *const p4_12_run_names[P4_12_CASE_RUN_COUNT] = {
    "discovery_original_reference",
    "discovery_original_candidate",
    "discovery_reproduction_reference",
    "discovery_reproduction_candidate",
    "authoritative_original_reference",
    "authoritative_original_candidate",
    "authoritative_reproduction_reference",
    "authoritative_reproduction_candidate",
    "authoritative_different_input",
};

static int p4_12_path(char *out, size_t capacity, const char *directory,
                      const char *name, const char *suffix)
{
    const int length = snprintf(out, capacity, "%s/%s%s", directory, name,
                                suffix);

    return length >= 0 && (size_t)length < capacity;
}

static int p4_12_init_artifacts(p4_12_cli_artifacts *artifacts,
                                const char *directory, const char *name)
{
    return p4_12_path(artifacts->report, sizeof(artifacts->report), directory,
                      name, "_report.json") &&
           p4_12_path(artifacts->decode, sizeof(artifacts->decode), directory,
                      name, "_decode.json") &&
           p4_12_path(artifacts->layer, sizeof(artifacts->layer), directory,
                      name, "_layer.json") &&
           p4_12_path(artifacts->stdout_path,
                      sizeof(artifacts->stdout_path), directory, name,
                      ".out") &&
           p4_12_path(artifacts->stderr_path,
                      sizeof(artifacts->stderr_path), directory, name,
                      ".err");
}

static void p4_12_remove_artifacts(const p4_12_cli_artifacts *artifacts)
{
    remove(artifacts->report);
    remove(artifacts->decode);
    remove(artifacts->layer);
    remove(artifacts->stdout_path);
    remove(artifacts->stderr_path);
}

static void p4_12_cleanup_case(p4_12_case_fixture *fixture)
{
    size_t index;

    lis_cli_test_injection_reset();
    for (index = 0U; index < P4_12_CASE_RUN_COUNT; ++index) {
        p4_12_remove_artifacts(&fixture->runs[index]);
    }
    remove(fixture->model);
    remove(fixture->config);
    remove(fixture->same_tokens);
    remove(fixture->different_tokens);
    if (fixture->directory[0] != '\0' &&
        rmdir(fixture->directory) != 0) {
        fprintf(stderr, "p4-12 private fixture cleanup failed: %s\n",
                fixture->directory);
        ++g_failures;
    }
}

static int p4_12_init_case(p4_12_case_fixture *fixture,
                           const char *case_label)
{
    static const char config_json[] =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":12,"
        "\"hidden_size\":1,\"intermediate_size\":1,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":1,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    struct stat info;
    lis_artifact_set_id directory_id;
    char directory_path[P4_12_PATH_CAPACITY];
    size_t index;

    memset(fixture, 0, sizeof(*fixture));
    if (lis_artifact_set_id_generate(&directory_id) != LIS_STATUS_OK ||
        !directory_id.valid ||
        snprintf(directory_path, sizeof(directory_path),
                 "srcs/libs/.test_p4_12_%s_%s", case_label,
                 directory_id.value + strlen(LIS_ARTIFACT_SET_ID_PREFIX)) >=
            (int)sizeof(directory_path)) {
        return 0;
    }
    if (mkdir(directory_path, 0700) != 0 ||
        strlen(directory_path) >= sizeof(fixture->directory)) {
        return 0;
    }
    memcpy(fixture->directory, directory_path, strlen(directory_path) + 1U);
    if (stat(fixture->directory, &info) != 0 || !S_ISDIR(info.st_mode) ||
        info.st_uid != getuid() ||
        (info.st_mode & 0777U) != 0700U) {
        fprintf(stderr, "p4-12 private fixture authority check failed\n");
        ++g_failures;
        p4_12_cleanup_case(fixture);
        return 0;
    }
    if (!p4_12_path(fixture->config, sizeof(fixture->config),
                    fixture->directory, "config", ".json") ||
        !p4_12_path(fixture->model, sizeof(fixture->model),
                    fixture->directory, "model", ".safetensors") ||
        !p4_12_path(fixture->same_tokens, sizeof(fixture->same_tokens),
                    fixture->directory, "same_tokens", ".txt") ||
        !p4_12_path(fixture->different_tokens,
                    sizeof(fixture->different_tokens), fixture->directory,
                    "different_tokens", ".txt")) {
        p4_12_cleanup_case(fixture);
        return 0;
    }
    for (index = 0U; index < P4_12_CASE_RUN_COUNT; ++index) {
        if (!p4_12_init_artifacts(&fixture->runs[index], fixture->directory,
                                  p4_12_run_names[index])) {
            p4_12_cleanup_case(fixture);
            return 0;
        }
    }
    if (write_text_file(fixture->config, config_json) != LIS_STATUS_OK ||
        write_llama_checkpoint_fixture_with_embeddings(
            fixture->model, 12U, 1.0f, -2.0f, 3.0f) != LIS_STATUS_OK ||
        write_text_file(fixture->same_tokens, "0\n") != LIS_STATUS_OK ||
        write_text_file(fixture->different_tokens, "1\n") != LIS_STATUS_OK) {
        fprintf(stderr, "p4-12 private fixture creation failed\n");
        ++g_failures;
        p4_12_cleanup_case(fixture);
        return 0;
    }
    return 1;
}

static int p4_12_run_cli(const p4_12_case_fixture *fixture,
                         const char *token_path,
                         const p4_12_cli_artifacts *artifacts,
                         int intra_layer_capture)
{
    char target_layer[32];
    char *argv[27];
    int argc = 0;
    pid_t child;
    int child_status = 0;

    if (snprintf(target_layer, sizeof(target_layer), "%u",
                 (unsigned int)P4_12_TARGET_LAYER) >=
        (int)sizeof(target_layer)) {
        return 1;
    }
    argv[argc++] = "lis";
    argv[argc++] = "--model";
    argv[argc++] = (char *)fixture->directory;
    argv[argc++] = "--config";
    argv[argc++] = (char *)fixture->config;
    argv[argc++] = "--tokens";
    argv[argc++] = (char *)token_path;
    argv[argc++] = "--context";
    argv[argc++] = "8";
    argv[argc++] = "--batch";
    argv[argc++] = "1";
    argv[argc++] = "--generate";
    argv[argc++] = "2";
    argv[argc++] = "--threads";
    argv[argc++] = "1";
    argv[argc++] = "--report-json";
    argv[argc++] = (char *)artifacts->report;
    argv[argc++] = "--trace-json";
    argv[argc++] = (char *)artifacts->decode;
    argv[argc++] = "--layer-checkpoints";
    argv[argc++] = "1";
    argv[argc++] = "--layer-trace-json";
    argv[argc++] = (char *)artifacts->layer;
    if (intra_layer_capture) {
        argv[argc++] = "--intra-layer-checkpoints";
        argv[argc++] = target_layer;
    }

    child = fork();
    if (child < 0) {
        return 1;
    }
    if (child == 0) {
        const int status = run_cli_capture(
            argc, argv, artifacts->stdout_path, artifacts->stderr_path);

        _exit(status == 0 ? 0 : 1);
    }
    if (waitpid(child, &child_status, 0) != child ||
        !WIFEXITED(child_status)) {
        return 1;
    }
    return WEXITSTATUS(child_status);
}

static int p4_12_run_generation(p4_12_case_fixture *fixture,
                                size_t offset, int intra_layer_capture,
                                int perturb_intra_layer)
{
    lis_cli_test_injection_reset();
    if (p4_12_run_cli(fixture, fixture->same_tokens,
                      &fixture->runs[offset], intra_layer_capture) != 0) {
        return 1;
    }
    if (lis_cli_test_override_selected_token(0U, 1U) != LIS_STATUS_OK ||
        p4_12_run_cli(fixture, fixture->same_tokens,
                      &fixture->runs[offset + 1U],
                      intra_layer_capture) != 0) {
        lis_cli_test_injection_reset();
        return 1;
    }
    lis_cli_test_injection_reset();
    if (p4_12_run_cli(fixture, fixture->same_tokens,
                      &fixture->runs[offset + 2U],
                      intra_layer_capture) != 0) {
        return 1;
    }
    if (lis_cli_test_perturb_layer_observation(
            P4_12_TARGET_LAYER, 0U, 0.25f) != LIS_STATUS_OK ||
        (perturb_intra_layer &&
         lis_cli_test_perturb_intra_layer_observation(
             LIS_INTRA_LAYER_STAGE_MLP_GATE_PROJECTION, 0U, 0.5f) !=
             LIS_STATUS_OK) ||
        p4_12_run_cli(fixture, fixture->same_tokens,
                      &fixture->runs[offset + 3U],
                      intra_layer_capture) != 0) {
        lis_cli_test_injection_reset();
        return 1;
    }
    lis_cli_test_injection_reset();
    return 0;
}

static int p4_12_run_case(p4_12_case_fixture *fixture,
                          int perturb_intra_layer)
{
    if (p4_12_run_generation(fixture, 0U, 0, 0) != 0 ||
        p4_12_run_generation(fixture, 4U, 1, perturb_intra_layer) != 0 ||
        p4_12_run_cli(fixture, fixture->different_tokens,
                      &fixture->runs[8], 1) != 0) {
        lis_cli_test_injection_reset();
        return 1;
    }
    return 0;
}

static int p4_12_run_probe(const char *case_a, const char *case_b)
{
    pid_t child = fork();
    int child_status = 0;

    if (child < 0) {
        return 1;
    }
    if (child == 0) {
        execl("/usr/bin/env", "env", "PYTHONDONTWRITEBYTECODE=1",
              "PYTHONPATH=tools", "python3",
              "tools/tests/pass4_real_artifact_probe.py", case_a, case_b,
              (char *)NULL);
        _exit(1);
    }
    if (waitpid(child, &child_status, 0) != child ||
        !WIFEXITED(child_status)) {
        return 1;
    }
    return WEXITSTATUS(child_status);
}

static void test_cli_pass4_real_artifact_integration(void)
{
    p4_12_case_fixture case_a;
    p4_12_case_fixture case_b;
    int case_a_ready = 0;
    int case_b_ready = 0;

    case_a_ready = p4_12_init_case(&case_a, "a");
    case_b_ready = p4_12_init_case(&case_b, "b");
    if (!case_a_ready || !case_b_ready) {
        fprintf(stderr, "p4-12 private case setup failed\n");
        ++g_failures;
        if (case_a_ready) {
            p4_12_cleanup_case(&case_a);
        }
        if (case_b_ready) {
            p4_12_cleanup_case(&case_b);
        }
        return;
    }

    expect_int("p4-12 actual case A generation",
               p4_12_run_case(&case_a, 0), 0);
    expect_int("p4-12 actual case B generation",
               p4_12_run_case(&case_b, 1), 0);
    expect_int("p4-12 actual two-generation probe",
               p4_12_run_probe(case_a.directory, case_b.directory), 0);

    p4_12_cleanup_case(&case_a);
    p4_12_cleanup_case(&case_b);
}
