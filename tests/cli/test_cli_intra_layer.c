/* P4-7 CLI capture/identity tests, included by test_cli.c. */

static int p4_7_extract_runtime_fingerprint(
    const char *json,
    char out[LIS_ARTIFACT_DIGEST_HEX_LEN + 1U])
{
    static const char marker[] =
        "\"fingerprint\":{\"algorithm\":\"fnv1a64\",\"hex\":\"";
    const char *runtime;
    const char *start;

    if (json == NULL || out == NULL) {
        return 0;
    }
    runtime = strstr(json, "\"runtime\":{");
    start = runtime != NULL ? strstr(runtime, marker) : NULL;
    if (start == NULL) {
        return 0;
    }
    start += sizeof(marker) - 1U;
    if (start[LIS_ARTIFACT_DIGEST_HEX_LEN] != '"') {
        return 0;
    }
    memcpy(out, start, LIS_ARTIFACT_DIGEST_HEX_LEN);
    out[LIS_ARTIFACT_DIGEST_HEX_LEN] = '\0';
    return 1;
}

static size_t p4_7_count_after(const char *text, const char *start_marker,
                               const char *needle)
{
    const char *cursor;
    size_t count = 0U;
    const size_t needle_len = strlen(needle);

    if (text == NULL || start_marker == NULL || needle == NULL ||
        needle_len == 0U) {
        return 0U;
    }
    cursor = strstr(text, start_marker);
    if (cursor == NULL) {
        return 0U;
    }
    while ((cursor = strstr(cursor, needle)) != NULL) {
        ++count;
        cursor += needle_len;
    }
    return count;
}

static void p4_7_expect_capture_manifest(const char *name,
                                         const char *path,
                                         size_t target_layer)
{
    char fragment[256];

    if (snprintf(
            fragment, sizeof(fragment),
            "\"intra_layer_checkpoints_enabled\":true,"
            "\"intra_layer_target_layer\":%zu,"
            "\"diagnostic_capture_profile\":\"%s\"",
            target_layer, LIS_INTRA_LAYER_DIAGNOSTIC_CAPTURE_PROFILE) >=
            (int)sizeof(fragment)) {
        fprintf(stderr, "%s: conditional manifest fragment overflow\n", name);
        ++g_failures;
        return;
    }
    expect_file_contains(name, path, fragment);
    expect_file_occurrences(name, path,
                            "\"intra_layer_checkpoints_enabled\":true", 1U);
}

static void p4_7_remove_artifacts(const char *report, const char *trace,
                                  const char *layer, const char *stdout_path,
                                  const char *stderr_path)
{
    remove(report);
    remove(trace);
    remove(layer);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_intra_layer_dependency_errors(void)
{
    const char *stdout_path = "srcs/libs/test_p4_7_dependency.out";
    const char *stderr_path = "srcs/libs/test_p4_7_dependency.err";
    const char *layer_path = "srcs/libs/test_p4_7_dependency.json";
    char *missing_parent[] = {
        "lis", "--model", "unused", "--config", "unused",
        "--tokens", "unused", "--context", "8", "--batch", "1",
        "--generate", "1", "--intra-layer-checkpoints", "0"
    };
    char *missing_layer_path[] = {
        "lis", "--model", "unused", "--config", "unused",
        "--tokens", "unused", "--context", "8", "--batch", "1",
        "--generate", "1", "--layer-checkpoints", "1",
        "--intra-layer-checkpoints", "0"
    };
    char *prefill_step[] = {
        "lis", "--model", "unused", "--config", "unused",
        "--tokens", "unused", "--context", "8", "--batch", "1",
        "--generate", "1", "--layer-checkpoints", "0",
        "--layer-trace-json", "srcs/libs/test_p4_7_dependency.json",
        "--intra-layer-checkpoints", "0"
    };
    char *batch_two[] = {
        "lis", "--model", "unused", "--config", "unused",
        "--tokens", "unused", "--context", "8", "--batch", "2",
        "--generate", "1", "--layer-checkpoints", "1",
        "--layer-trace-json", "srcs/libs/test_p4_7_dependency.json",
        "--intra-layer-checkpoints", "0"
    };
    char *bad_layer_text[] = {
        "lis", "--model", "unused", "--config", "unused",
        "--tokens", "unused", "--context", "8", "--batch", "1",
        "--generate", "1", "--intra-layer-checkpoints", "invalid"
    };
    char *new_output_path[] = {
        "lis", "--model", "unused", "--config", "unused",
        "--tokens", "unused", "--context", "8", "--batch", "1",
        "--generate", "1", "--intra-layer-trace-json", "unused"
    };
    char *help[] = {"lis", "--help"};

    p4_7_remove_artifacts(layer_path, layer_path, layer_path, stdout_path,
                          stderr_path);
    expect_int("p4-7 help run",
               run_cli_capture(2, help, stdout_path, stderr_path), 0);
    expect_file_contains("p4-7 help flag", stdout_path,
                         "--intra-layer-checkpoints LAYER");
    expect_int("p4-7 intra requires parent checkpoint",
               run_cli_capture(
                   (int)(sizeof(missing_parent) / sizeof(missing_parent[0])),
                   missing_parent, stdout_path, stderr_path), 1);
    expect_file_contains("p4-7 missing parent diagnostic", stderr_path,
                         "lis: artifact error: --intra-layer-checkpoints "
                         "requires --layer-checkpoints STEP");
    expect_file_missing("p4-7 missing parent no artifact", layer_path);

    expect_int("p4-7 intra requires layer path",
               run_cli_capture(
                   (int)(sizeof(missing_layer_path) /
                         sizeof(missing_layer_path[0])),
                   missing_layer_path, stdout_path, stderr_path), 1);
    expect_file_contains("p4-7 missing path diagnostic", stderr_path,
                         "lis: artifact error: --intra-layer-checkpoints "
                         "requires --layer-checkpoints STEP");

    expect_int("p4-7 intra rejects prefill step",
               run_cli_capture(
                   (int)(sizeof(prefill_step) / sizeof(prefill_step[0])),
                   prefill_step, stdout_path, stderr_path), 1);
    expect_file_contains("p4-7 prefill diagnostic", stderr_path,
                         "STEP > 0");
    expect_file_missing("p4-7 prefill no artifact", layer_path);

    expect_int("p4-7 intra rejects batch two",
               run_cli_capture(
                   (int)(sizeof(batch_two) / sizeof(batch_two[0])),
                   batch_two, stdout_path, stderr_path), 1);
    expect_file_contains("p4-7 batch diagnostic", stderr_path,
                         "--batch 1");
    expect_file_missing("p4-7 batch no artifact", layer_path);

    expect_int("p4-7 intra rejects malformed layer",
               run_cli_capture(
                   (int)(sizeof(bad_layer_text) /
                         sizeof(bad_layer_text[0])),
                   bad_layer_text, stdout_path, stderr_path), 2);
    expect_file_contains("p4-7 malformed layer usage", stderr_path,
                         "lis: usage error: invalid arguments");

    expect_int("p4-7 no standalone output flag",
               run_cli_capture(
                   (int)(sizeof(new_output_path) /
                         sizeof(new_output_path[0])),
                   new_output_path, stdout_path, stderr_path), 2);
    expect_file_contains("p4-7 standalone output usage", stderr_path,
                         "lis: usage error: invalid arguments");

    p4_7_remove_artifacts(layer_path, layer_path, layer_path, stdout_path,
                          stderr_path);
}

static void test_cli_intra_layer_artifact_identity(void)
{
    const char *model_path = "srcs/libs/test_p4_7_cli/model.safetensors";
    const char *config_path = "srcs/libs/test_p4_7_cli/config.json";
    const char *token_path = "srcs/libs/test_p4_7_cli_tokens.txt";
    const char *legacy_report = "srcs/libs/test_p4_7_legacy_report.json";
    const char *legacy_trace = "srcs/libs/test_p4_7_legacy_trace.json";
    const char *legacy_layer = "srcs/libs/test_p4_7_legacy_layer.json";
    const char *target0_report = "srcs/libs/test_p4_7_target0_report.json";
    const char *target0_trace = "srcs/libs/test_p4_7_target0_trace.json";
    const char *target0_layer = "srcs/libs/test_p4_7_target0_layer.json";
    const char *target1_report = "srcs/libs/test_p4_7_target1_report.json";
    const char *target1_trace = "srcs/libs/test_p4_7_target1_trace.json";
    const char *target1_layer = "srcs/libs/test_p4_7_target1_layer.json";
    const char *failed_layer = "srcs/libs/test_p4_7_failed_layer.json";
    const char *stdout_path = "srcs/libs/test_p4_7_cli.out";
    const char *stderr_path = "srcs/libs/test_p4_7_cli.err";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":8,"
        "\"hidden_size\":1,\"intermediate_size\":1,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":1,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    char *legacy_argv[] = {
        "lis", "--model", "srcs/libs/test_p4_7_cli",
        "--config", "srcs/libs/test_p4_7_cli/config.json",
        "--tokens", "srcs/libs/test_p4_7_cli_tokens.txt",
        "--context", "8", "--batch", "1", "--generate", "2",
        "--threads", "1",
        "--report-json", "srcs/libs/test_p4_7_legacy_report.json",
        "--trace-json", "srcs/libs/test_p4_7_legacy_trace.json",
        "--layer-checkpoints", "1",
        "--layer-trace-json", "srcs/libs/test_p4_7_legacy_layer.json"
    };
    char *target0_argv[] = {
        "lis", "--model", "srcs/libs/test_p4_7_cli",
        "--config", "srcs/libs/test_p4_7_cli/config.json",
        "--tokens", "srcs/libs/test_p4_7_cli_tokens.txt",
        "--context", "8", "--batch", "1", "--generate", "2",
        "--threads", "1",
        "--report-json", "srcs/libs/test_p4_7_target0_report.json",
        "--trace-json", "srcs/libs/test_p4_7_target0_trace.json",
        "--layer-checkpoints", "1",
        "--layer-trace-json", "srcs/libs/test_p4_7_target0_layer.json",
        "--intra-layer-checkpoints", "0"
    };
    char *target1_argv[] = {
        "lis", "--model", "srcs/libs/test_p4_7_cli",
        "--config", "srcs/libs/test_p4_7_cli/config.json",
        "--tokens", "srcs/libs/test_p4_7_cli_tokens.txt",
        "--context", "8", "--batch", "1", "--generate", "2",
        "--threads", "1",
        "--report-json", "srcs/libs/test_p4_7_target1_report.json",
        "--trace-json", "srcs/libs/test_p4_7_target1_trace.json",
        "--layer-checkpoints", "1",
        "--layer-trace-json", "srcs/libs/test_p4_7_target1_layer.json",
        "--intra-layer-checkpoints", "1"
    };
    char *out_of_range[] = {
        "lis", "--model", "srcs/libs/test_p4_7_cli",
        "--config", "srcs/libs/test_p4_7_cli/config.json",
        "--tokens", "srcs/libs/test_p4_7_cli_tokens.txt",
        "--context", "8", "--batch", "1", "--generate", "1",
        "--layer-checkpoints", "1",
        "--layer-trace-json", "srcs/libs/test_p4_7_failed_layer.json",
        "--intra-layer-checkpoints", "8"
    };
    char *not_selected[] = {
        "lis", "--model", "srcs/libs/test_p4_7_cli",
        "--config", "srcs/libs/test_p4_7_cli/config.json",
        "--tokens", "srcs/libs/test_p4_7_cli_tokens.txt",
        "--context", "8", "--batch", "1", "--generate", "1",
        "--layer-checkpoints", "1",
        "--layer-trace-json", "srcs/libs/test_p4_7_failed_layer.json",
        "--intra-layer-checkpoints", "3"
    };
    char *out_of_context[] = {
        "lis", "--model", "srcs/libs/test_p4_7_cli",
        "--config", "srcs/libs/test_p4_7_cli/config.json",
        "--tokens", "srcs/libs/test_p4_7_cli_tokens.txt",
        "--context", "8", "--batch", "1", "--generate", "1",
        "--layer-checkpoints", "8",
        "--layer-trace-json", "srcs/libs/test_p4_7_failed_layer.json",
        "--intra-layer-checkpoints", "0"
    };
    char *incomplete[] = {
        "lis", "--model", "srcs/libs/test_p4_7_cli",
        "--config", "srcs/libs/test_p4_7_cli/config.json",
        "--tokens", "srcs/libs/test_p4_7_cli_tokens.txt",
        "--context", "8", "--batch", "1", "--generate", "1",
        "--layer-checkpoints", "2",
        "--layer-trace-json", "srcs/libs/test_p4_7_failed_layer.json",
        "--intra-layer-checkpoints", "0"
    };
    char *report_json = NULL;
    char *trace_json = NULL;
    char *layer_json = NULL;
    char *target1_json = NULL;
    char report_fp[LIS_ARTIFACT_DIGEST_HEX_LEN + 1U] = {0};
    char trace_fp[LIS_ARTIFACT_DIGEST_HEX_LEN + 1U] = {0};
    char layer_fp[LIS_ARTIFACT_DIGEST_HEX_LEN + 1U] = {0};
    char target1_fp[LIS_ARTIFACT_DIGEST_HEX_LEN + 1U] = {0};
    char report_id[LIS_ARTIFACT_SET_ID_LEN + 1U] = {0};
    char trace_id[LIS_ARTIFACT_SET_ID_LEN + 1U] = {0};
    char layer_id[LIS_ARTIFACT_SET_ID_LEN + 1U] = {0};

    remove(model_path);
    remove(config_path);
    remove(token_path);
    p4_7_remove_artifacts(legacy_report, legacy_trace, legacy_layer,
                          stdout_path, stderr_path);
    p4_7_remove_artifacts(target0_report, target0_trace, target0_layer,
                          stdout_path, stderr_path);
    p4_7_remove_artifacts(target1_report, target1_trace, target1_layer,
                          stdout_path, stderr_path);
    remove(failed_layer);
    if (system("mkdir -p srcs/libs/test_p4_7_cli") != 0) {
        fprintf(stderr, "p4-7 fixture directory creation failed\n");
        ++g_failures;
        return;
    }
    expect_status("p4-7 write config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("p4-7 write model",
                  write_llama_checkpoint_fixture(model_path, 8U),
                  LIS_STATUS_OK);
    expect_status("p4-7 write tokens",
                  write_text_file(token_path, "0\n"), LIS_STATUS_OK);

    expect_int("p4-7 legacy sibling run",
               run_cli_capture(
                   (int)(sizeof(legacy_argv) / sizeof(legacy_argv[0])),
                   legacy_argv, stdout_path, stderr_path), 0);
    expect_file_not_contains("p4-7 legacy report fields", legacy_report,
                             "intra_layer_");
    expect_file_not_contains("p4-7 legacy trace fields", legacy_trace,
                             "intra_layer_");
    expect_file_not_contains("p4-7 legacy layer fields", legacy_layer,
                             "intra_layer_");
    expect_file_not_contains("p4-7 legacy report profile", legacy_report,
                             "semantic_layer_and_intra_v1");

    expect_int("p4-7 target zero run",
               run_cli_capture(
                   (int)(sizeof(target0_argv) / sizeof(target0_argv[0])),
                   target0_argv, stdout_path, stderr_path), 0);
    p4_7_expect_capture_manifest("p4-7 report capture manifest",
                                 target0_report, 0U);
    p4_7_expect_capture_manifest("p4-7 trace capture manifest",
                                 target0_trace, 0U);
    p4_7_expect_capture_manifest("p4-7 layer capture manifest",
                                 target0_layer, 0U);
    expect_file_contains("p4-7 additive layout", target0_layer,
                         "\"intra_layer_checkpoint_layout\":{");
    expect_file_contains("p4-7 additive trace", target0_layer,
                         "\"intra_layer_trace\":[");
    expect_file_contains("p4-7 fixed no missing coordinates", target0_layer,
                         "\"missing_coordinates\":[]");
    expect_file_occurrences("p4-7 semantic stderr only parent outputs",
                            stderr_path, "lis: layer-checkpoint ", 6U);
    expect_file_not_contains("p4-7 semantic stderr no embedding", stderr_path,
                             "name=embedding");

    report_json = read_file_content(target0_report);
    trace_json = read_file_content(target0_trace);
    layer_json = read_file_content(target0_layer);
    if (p4_7_count_after(layer_json, "\"intra_layer_trace\":[",
                         "\"stage_id\":") !=
            17U) {
        fprintf(stderr, "p4-7 intra trace did not contain exactly 17 entries\n");
        ++g_failures;
    }
    if (!p4_7_extract_runtime_fingerprint(report_json, report_fp) ||
        !p4_7_extract_runtime_fingerprint(trace_json, trace_fp) ||
        !p4_7_extract_runtime_fingerprint(layer_json, layer_fp) ||
        strcmp(report_fp, trace_fp) != 0 || strcmp(report_fp, layer_fp) != 0) {
        fprintf(stderr, "p4-7 sibling runtime fingerprints differ\n");
        ++g_failures;
    }
    {
        const char *report_manifest = report_json != NULL ?
            strstr(report_json, "\"manifest\":{") : NULL;
        const char *layer_manifest = layer_json != NULL ?
            strstr(layer_json, "\"manifest\":{") : NULL;
        const char *report_manifest_end = report_manifest != NULL ?
            strstr(report_manifest, "},\"report\":{") : NULL;
        const char *layer_manifest_end = layer_manifest != NULL ?
            strstr(layer_manifest, "},\"checkpoint_layout\":{") : NULL;
        const size_t report_manifest_len =
            report_manifest != NULL && report_manifest_end != NULL ?
                (size_t)(report_manifest_end - report_manifest) : 0U;
        const size_t layer_manifest_len =
            layer_manifest != NULL && layer_manifest_end != NULL ?
                (size_t)(layer_manifest_end - layer_manifest) : 0U;

        if (report_manifest_len == 0U ||
            report_manifest_len != layer_manifest_len ||
            memcmp(report_manifest, layer_manifest,
                   report_manifest_len) != 0) {
            fprintf(stderr,
                    "p4-7 report/layer semantic manifests differ\n");
            ++g_failures;
        }
    }
    if (!extract_artifact_set_id(report_json, report_id) ||
        !extract_artifact_set_id(trace_json, trace_id) ||
        !extract_artifact_set_id(layer_json, layer_id) ||
        strcmp(report_id, trace_id) != 0 || strcmp(report_id, layer_id) != 0) {
        fprintf(stderr, "p4-7 sibling artifact-set IDs differ\n");
        ++g_failures;
    }
    free(report_json);
    free(trace_json);
    free(layer_json);

    expect_int("p4-7 target one run",
               run_cli_capture(
                   (int)(sizeof(target1_argv) / sizeof(target1_argv[0])),
                   target1_argv, stdout_path, stderr_path), 0);
    p4_7_expect_capture_manifest("p4-7 target one manifest", target1_layer,
                                 1U);
    target1_json = read_file_content(target1_layer);
    if (!p4_7_extract_runtime_fingerprint(target1_json, target1_fp) ||
        strcmp(layer_fp, target1_fp) == 0) {
        fprintf(stderr, "p4-7 target layer did not separate runtime identity\n");
        ++g_failures;
    }
    free(target1_json);

    remove(failed_layer);
    expect_int("p4-7 target layer bound",
               run_cli_capture(
                   (int)(sizeof(out_of_range) / sizeof(out_of_range[0])),
                   out_of_range, stdout_path, stderr_path), 1);
    expect_file_contains("p4-7 target layer diagnostic", stderr_path,
                         "intra-layer target is outside the model layer range");
    expect_file_missing("p4-7 target layer no artifact", failed_layer);

    expect_int("p4-7 target selected by parent layout",
               run_cli_capture(
                   (int)(sizeof(not_selected) / sizeof(not_selected[0])),
                   not_selected, stdout_path, stderr_path), 1);
    expect_file_contains("p4-7 parent layout diagnostic", stderr_path,
                         "intra-layer target is not selected by the frozen "
                         "Pass 3 checkpoint layout");
    expect_file_missing("p4-7 parent layout no artifact", failed_layer);

    expect_int("p4-7 target token context bound",
               run_cli_capture(
                   (int)(sizeof(out_of_context) / sizeof(out_of_context[0])),
                   out_of_context, stdout_path, stderr_path), 1);
    expect_file_contains("p4-7 target context diagnostic", stderr_path,
                         "intra-layer token position is outside the configured context");
    expect_file_missing("p4-7 target context no artifact", failed_layer);

    expect_int("p4-7 incomplete capture suppressed",
               run_cli_capture(
                   (int)(sizeof(incomplete) / sizeof(incomplete[0])),
                   incomplete, stdout_path, stderr_path), 1);
    expect_file_contains("p4-7 incomplete diagnostic", stderr_path,
                         "intra-layer capture incomplete; layer-trace artifact suppressed");
    expect_file_missing("p4-7 incomplete no artifact", failed_layer);

    remove(model_path);
    remove(config_path);
    remove(token_path);
    p4_7_remove_artifacts(legacy_report, legacy_trace, legacy_layer,
                          stdout_path, stderr_path);
    p4_7_remove_artifacts(target0_report, target0_trace, target0_layer,
                          stdout_path, stderr_path);
    p4_7_remove_artifacts(target1_report, target1_trace, target1_layer,
                          stdout_path, stderr_path);
    remove(failed_layer);
    if (system("rmdir srcs/libs/test_p4_7_cli 2>/dev/null") != 0) {
        /* best effort */
    }
}

static void test_cli_intra_layer_surface(void)
{
    test_cli_intra_layer_dependency_errors();
    test_cli_intra_layer_artifact_identity();
}
