/* Real-artifact Pass 3 integration cases, included by test_cli.c. */

#include "lis_test_controls.h"

typedef struct {
    const char *report;
    const char *decode;
    const char *layer;
    const char *stdout_path;
    const char *stderr_path;
} pass3_cli_artifacts;

static int extract_artifact_set_id(const char *json,
                                   char out[LIS_ARTIFACT_SET_ID_LEN + 1U])
{
    static const char marker[] = "\"artifact_set_id\":\"";
    const char *start;
    size_t index;

    if (json == NULL || out == NULL) {
        return 0;
    }
    start = strstr(json, marker);
    if (start == NULL) {
        return 0;
    }
    start += sizeof(marker) - 1U;
    if (strncmp(start, "aset1:", 6U) != 0 ||
        start[LIS_ARTIFACT_SET_ID_LEN] != '"') {
        return 0;
    }
    for (index = 6U; index < LIS_ARTIFACT_SET_ID_LEN; ++index) {
        if (!((start[index] >= '0' && start[index] <= '9') ||
              (start[index] >= 'a' && start[index] <= 'f'))) {
            return 0;
        }
    }
    memcpy(out, start, LIS_ARTIFACT_SET_ID_LEN);
    out[LIS_ARTIFACT_SET_ID_LEN] = '\0';
    return 1;
}

static int run_pass3_cli_case(const char *token_path,
                              const pass3_cli_artifacts *artifacts)
{
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_pass3_producer",
        "--config", "srcs/libs/test_pass3_producer/config.json",
        "--tokens", (char *)token_path,
        "--context", "8",
        "--batch", "1",
        "--generate", "2",
        "--threads", "1",
        "--report-json", (char *)artifacts->report,
        "--trace-json", (char *)artifacts->decode,
        "--layer-checkpoints", "1",
        "--layer-trace-json", (char *)artifacts->layer,
    };

    pid_t child = fork();
    int child_status = 0;

    if (child < 0) {
        return 1;
    }
    if (child == 0) {
        const int cli_status = run_cli_capture(
            (int)(sizeof(argv) / sizeof(argv[0])), argv,
            artifacts->stdout_path, artifacts->stderr_path);

        _exit(cli_status == 0 ? 0 : 1);
    }
    if (waitpid(child, &child_status, 0) != child ||
        !WIFEXITED(child_status)) {
        return 1;
    }
    return WEXITSTATUS(child_status);
}

static void remove_pass3_cli_artifacts(const pass3_cli_artifacts *artifacts)
{
    remove(artifacts->report);
    remove(artifacts->decode);
    remove(artifacts->layer);
    remove(artifacts->stdout_path);
    remove(artifacts->stderr_path);
}

static void test_cli_pass3_producer_contract(void)
{
    const char *model_path = "srcs/libs/test_pass3_producer/model.safetensors";
    const char *config_path = "srcs/libs/test_pass3_producer/config.json";
    const char *same_token_path = "srcs/libs/test_pass3_producer_same.txt";
    const char *different_token_path = "srcs/libs/test_pass3_producer_diff.txt";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":12,"
        "\"hidden_size\":1,\"intermediate_size\":1,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":1,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    const pass3_cli_artifacts original_reference = {
        "srcs/libs/test_pass3_original_reference_report.json",
        "srcs/libs/test_pass3_original_reference_decode.json",
        "srcs/libs/test_pass3_original_reference_layer.json",
        "srcs/libs/test_pass3_original_reference.out",
        "srcs/libs/test_pass3_original_reference.err",
    };
    const pass3_cli_artifacts original_candidate = {
        "srcs/libs/test_pass3_original_candidate_report.json",
        "srcs/libs/test_pass3_original_candidate_decode.json",
        "srcs/libs/test_pass3_original_candidate_layer.json",
        "srcs/libs/test_pass3_original_candidate.out",
        "srcs/libs/test_pass3_original_candidate.err",
    };
    const pass3_cli_artifacts same_reference = {
        "srcs/libs/test_pass3_same_reference_report.json",
        "srcs/libs/test_pass3_same_reference_decode.json",
        "srcs/libs/test_pass3_same_reference_layer.json",
        "srcs/libs/test_pass3_same_reference.out",
        "srcs/libs/test_pass3_same_reference.err",
    };
    const pass3_cli_artifacts same_candidate = {
        "srcs/libs/test_pass3_same_candidate_report.json",
        "srcs/libs/test_pass3_same_candidate_decode.json",
        "srcs/libs/test_pass3_same_candidate_layer.json",
        "srcs/libs/test_pass3_same_candidate.out",
        "srcs/libs/test_pass3_same_candidate.err",
    };
    const pass3_cli_artifacts controlled_candidate = {
        "srcs/libs/test_pass3_controlled_report.json",
        "srcs/libs/test_pass3_controlled_decode.json",
        "srcs/libs/test_pass3_controlled_layer.json",
        "srcs/libs/test_pass3_controlled.out",
        "srcs/libs/test_pass3_controlled.err",
    };
    const pass3_cli_artifacts different_input = {
        "srcs/libs/test_pass3_different_input_report.json",
        "srcs/libs/test_pass3_different_input_decode.json",
        "srcs/libs/test_pass3_different_input_layer.json",
        "srcs/libs/test_pass3_different_input.out",
        "srcs/libs/test_pass3_different_input.err",
    };
    const pass3_cli_artifacts *all_artifacts[] = {
        &original_reference,
        &original_candidate,
        &same_reference,
        &same_candidate,
        &controlled_candidate,
        &different_input,
    };
    char report_id[LIS_ARTIFACT_SET_ID_LEN + 1U] = {0};
    char decode_id[LIS_ARTIFACT_SET_ID_LEN + 1U] = {0};
    char layer_id[LIS_ARTIFACT_SET_ID_LEN + 1U] = {0};
    char *report = NULL;
    char *decode = NULL;
    char *layer = NULL;
    size_t index;

    remove(model_path);
    remove(config_path);
    remove(same_token_path);
    remove(different_token_path);
    for (index = 0; index < sizeof(all_artifacts) / sizeof(all_artifacts[0]);
         ++index) {
        remove_pass3_cli_artifacts(all_artifacts[index]);
    }
    if (system("mkdir -p srcs/libs/test_pass3_producer") != 0) {
        fprintf(stderr, "mkdir pass3 producer fixture failed\n");
        ++g_failures;
        return;
    }
    expect_status("pass3 producer config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("pass3 producer model",
                  write_llama_checkpoint_fixture_with_embeddings(
                      model_path, 12, 1.0f, -2.0f, 3.0f),
                  LIS_STATUS_OK);
    expect_status("pass3 same tokens",
                  write_text_file(same_token_path, "0\n"), LIS_STATUS_OK);
    expect_status("pass3 different tokens",
                  write_text_file(different_token_path, "1\n"),
                  LIS_STATUS_OK);

    lis_cli_test_injection_reset();
    expect_int("pass3 original reference run",
               run_pass3_cli_case(same_token_path, &original_reference), 0);

    expect_status("pass3 selected-token control configure",
                  lis_cli_test_override_selected_token(0U, 1U),
                  LIS_STATUS_OK);
    expect_int("pass3 original controlled boundary run",
               run_pass3_cli_case(same_token_path, &original_candidate), 0);
    lis_cli_test_injection_reset();

    expect_int("pass3 independent same reference run",
               run_pass3_cli_case(same_token_path, &same_reference), 0);
    expect_int("pass3 independent same candidate run",
               run_pass3_cli_case(same_token_path, &same_candidate), 0);

    expect_status("pass3 layer-observation control configure",
                  lis_cli_test_perturb_layer_observation(4U, 0U, 0.25f),
                  LIS_STATUS_OK);
    expect_int("pass3 same-boundary controlled observation run",
               run_pass3_cli_case(same_token_path, &controlled_candidate), 0);
    lis_cli_test_injection_reset();

    expect_int("pass3 different-input classification run",
               run_pass3_cli_case(different_token_path, &different_input), 0);

    report = read_file_content(same_reference.report);
    decode = read_file_content(same_reference.decode);
    layer = read_file_content(same_reference.layer);
    if (!extract_artifact_set_id(report, report_id) ||
        !extract_artifact_set_id(decode, decode_id) ||
        !extract_artifact_set_id(layer, layer_id)) {
        fprintf(stderr, "pass3 producer artifact-set format missing\n");
        ++g_failures;
    } else if (strcmp(report_id, decode_id) != 0 ||
               strcmp(report_id, layer_id) != 0) {
        fprintf(stderr, "pass3 producer sibling artifact-set IDs differ\n");
        ++g_failures;
    }
    if (layer == NULL) {
        fprintf(stderr, "pass3 producer layer artifact missing\n");
        ++g_failures;
    } else {
        const char *required[] = {
            "\"layout_name\":\"llama_layer_output_summary\"",
            "\"layout_version\":1",
            "\"runtime_checkpoint_step\":1",
            "\"ordering_semantics\":\"runtime_step_layer_stage_ordinal\"",
            "\"total_layer_count\":12",
            "\"requested_coordinates\":[",
            "\"captured_coordinates\":[",
            "\"missing_coordinates\":[]",
            "\"duplicate_coordinate_policy\":\"reject_artifact_before_write\"",
            "\"observed_dtype\":\"fp32\"",
            "\"element_count\":1",
            "\"execution_ordinal\":0",
            "\"execution_ordinal\":1",
            "\"version\":\"lis.checkpoint.fp32le/v1\"",
            "\"byte_order\":\"little\"",
            "\"canonicalization\":\"ieee754-binary32-le;canonical-qnan;preserve-signed-zero\"",
            "\"value\":\"sha256:",
        };

        for (index = 0; index < sizeof(required) / sizeof(required[0]);
             ++index) {
            if (strstr(layer, required[index]) == NULL) {
                fprintf(stderr, "pass3 producer missing field: %s\n",
                        required[index]);
                ++g_failures;
            }
        }
    }
    expect_file_occurrences("pass3 producer layer 0 unique",
                            same_reference.layer,
                            "\"name\":\"layer.0.output\"", 1);
    expect_file_occurrences("pass3 producer layer 1 unique",
                            same_reference.layer,
                            "\"name\":\"layer.1.output\"", 1);

    expect_int(
        "pass3 actual producer revalidation probe",
        system(
            "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=tools python3 "
            "tools/tests/pass3_real_artifact_probe.py "
            "srcs/libs/test_pass3_original_reference_report.json "
            "srcs/libs/test_pass3_original_candidate_report.json "
            "srcs/libs/test_pass3_same_reference_report.json "
            "srcs/libs/test_pass3_same_reference_layer.json "
            "srcs/libs/test_pass3_same_candidate_report.json "
            "srcs/libs/test_pass3_same_candidate_layer.json "
            "srcs/libs/test_pass3_controlled_report.json "
            "srcs/libs/test_pass3_controlled_layer.json "
            "srcs/libs/test_pass3_different_input_report.json "
            "srcs/libs/test_pass3_different_input_layer.json"),
        0);

    free(report);
    free(decode);
    free(layer);
    lis_cli_test_injection_reset();
    remove(model_path);
    remove(config_path);
    remove(same_token_path);
    remove(different_token_path);
    for (index = 0; index < sizeof(all_artifacts) / sizeof(all_artifacts[0]);
         ++index) {
        remove_pass3_cli_artifacts(all_artifacts[index]);
    }
    if (system("rmdir srcs/libs/test_pass3_producer 2>/dev/null") != 0) {
        /* best effort */
    }
}
