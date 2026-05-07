static void expect_file_order4(const char *name, const char *path,
                               const char *a, const char *b,
                               const char *c, const char *d)
{
    char *data = read_file_content(path);

    if (data == NULL) {
        fprintf(stderr, "%s: could not read %s\n", name, path);
        ++g_failures;
        return;
    }
    if (strstr(data, a) == NULL || strstr(data, b) == NULL ||
        strstr(data, c) == NULL || strstr(data, d) == NULL ||
        strstr(data, a) > strstr(data, b) ||
        strstr(data, b) > strstr(data, c) ||
        strstr(data, c) > strstr(data, d)) {
        fprintf(stderr,
                "%s: expected ordered substrings in %s: %s, %s, %s, %s\n",
                name, path, a, b, c, d);
        ++g_failures;
    }
    free(data);
}

static void expect_file_order3(const char *name, const char *path,
                               const char *a, const char *b,
                               const char *c)
{
    char *data = read_file_content(path);

    if (data == NULL) {
        fprintf(stderr, "%s: could not read %s\n", name, path);
        ++g_failures;
        return;
    }
    if (strstr(data, a) == NULL || strstr(data, b) == NULL ||
        strstr(data, c) == NULL ||
        strstr(data, a) > strstr(data, b) ||
        strstr(data, b) > strstr(data, c)) {
        fprintf(stderr,
                "%s: expected ordered substrings in %s: %s, %s, %s\n",
                name, path, a, b, c);
        ++g_failures;
    }
    free(data);
}

static void test_cli_artifact_compat_run_report_legacy_surface(void)
{
    const char *model_path = "srcs/libs/test_artifact_compat_report.safetensors";
    const char *config_path = "srcs/libs/test_artifact_compat_report.json";
    const char *token_path = "srcs/libs/test_artifact_compat_report.txt";
    const char *report_path = "srcs/libs/test_artifact_compat_report.out.json";
    const char *stdout_path = "srcs/libs/test_artifact_compat_report.out";
    const char *stderr_path = "srcs/libs/test_artifact_compat_report.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_artifact_compat_report.safetensors",
        "--config", "srcs/libs/test_artifact_compat_report.json",
        "--tokens", "srcs/libs/test_artifact_compat_report.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "2",
        "--report-json", "srcs/libs/test_artifact_compat_report.out.json",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("artifact-compat report run",
               run_cli_capture(15, argv, stdout_path, stderr_path), 0);

    expect_file_order4("artifact-compat report top-level order", report_path,
                       "\"schema\":\"lis.execution_artifact/v1\"",
                       "\"kind\":\"run_report\"",
                       "\"manifest\":{",
                       "\"report\":{");
    expect_file_contains("artifact-compat runtime precision additive", report_path,
                         "\"perf_per_token_enabled\":false,"
                         "\"precision_path\":\"f32_accum;weights=f32;kv=f32\","
                         "\"fingerprint\":");
    expect_file_contains("artifact-compat runtime configured_context", report_path,
                         "\"configured_context\":4");
    expect_file_contains("artifact-compat runtime batch_size", report_path,
                         "\"batch_size\":2");
    expect_file_contains("artifact-compat runtime generation_limit", report_path,
                         "\"generation_limit\":2");
    expect_file_contains("artifact-compat runtime thread_count", report_path,
                         "\"thread_count\":1");
    expect_file_contains("artifact-compat runtime diagnostics", report_path,
                         "\"diagnostics_enabled\":false");
    expect_file_contains("artifact-compat runtime perf", report_path,
                         "\"perf_enabled\":false");
    expect_file_contains("artifact-compat runtime perf-per-token", report_path,
                         "\"perf_per_token_enabled\":false");
    expect_file_contains("artifact-compat report execution_status", report_path,
                         "\"execution_status\":\"ok\"");
    expect_file_contains("artifact-compat report status_code", report_path,
                         "\"status_code\":\"OK\"");
    expect_file_contains("artifact-compat report stop_reason", report_path,
                         "\"stop_reason\":\"decode_limit\"");
    expect_file_contains("artifact-compat report output_mode", report_path,
                         "\"output_mode\":\"token_ids\"");
    expect_file_contains("artifact-compat report prompt_sequences", report_path,
                         "\"prompt_sequences\":[");
    expect_file_contains("artifact-compat report selected count", report_path,
                         "\"selected_token_count\":2");
    expect_file_contains("artifact-compat report selected digest", report_path,
                         "\"selected_token_digest\":");
    expect_file_contains("artifact-compat report selected ids", report_path,
                         "\"selected_token_ids\":[2,2]");
    expect_file_contains("artifact-compat report emitted count", report_path,
                         "\"emitted_token_count\":2");
    expect_file_contains("artifact-compat report emitted digest", report_path,
                         "\"emitted_token_digest\":");
    expect_file_contains("artifact-compat report emitted ids", report_path,
                         "\"emitted_token_ids\":[2,2]");
    expect_file_not_contains("artifact-compat no perf_summary rename", report_path,
                             "\"perf_summary\"");

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_artifact_compat_run_report_perf_surface(void)
{
    const char *model_path = "srcs/libs/test_artifact_compat_perf.safetensors";
    const char *config_path = "srcs/libs/test_artifact_compat_perf.json";
    const char *token_path = "srcs/libs/test_artifact_compat_perf.txt";
    const char *report_path = "srcs/libs/test_artifact_compat_perf.out.json";
    const char *stdout_path = "srcs/libs/test_artifact_compat_perf.out";
    const char *stderr_path = "srcs/libs/test_artifact_compat_perf.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_artifact_compat_perf.safetensors",
        "--config", "srcs/libs/test_artifact_compat_perf.json",
        "--tokens", "srcs/libs/test_artifact_compat_perf.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "2",
        "--perf",
        "--report-json", "srcs/libs/test_artifact_compat_perf.out.json",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("artifact-compat perf report run",
               run_cli_capture(16, argv, stdout_path, stderr_path), 0);

    expect_file_contains("artifact-compat perf object name", report_path,
                         "\"perf\":{\"tag\":\"none\"");
    expect_file_contains("artifact-compat perf threads", report_path,
                         "\"threads\":1");
    expect_file_contains("artifact-compat perf prompt_tokens", report_path,
                         "\"prompt_tokens\":3");
    expect_file_contains("artifact-compat perf generated_tokens", report_path,
                         "\"generated_tokens\":2");
    expect_file_contains("artifact-compat perf ttft", report_path,
                         "\"ttft_ms\":");
    expect_file_contains("artifact-compat perf itl", report_path,
                         "\"itl_ms\":");
    expect_file_contains("artifact-compat perf tps steady", report_path,
                         "\"tps_steady\":");
    expect_file_contains("artifact-compat perf tps e2e", report_path,
                         "\"tps_end_to_end\":");
    expect_file_contains("artifact-compat perf stages", report_path,
                         "\"stages\":{");
    expect_file_contains("artifact-compat perf model_load", report_path,
                         "\"model_load\":");
    expect_file_contains("artifact-compat perf first_decode", report_path,
                         "\"first_decode\":");
    expect_file_not_contains("artifact-compat perf no summary rename", report_path,
                             "\"perf_summary\"");

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_artifact_compat_stderr_perf_surface(void)
{
    const char *model_path = "srcs/libs/test_artifact_compat_stderr_perf.safetensors";
    const char *config_path = "srcs/libs/test_artifact_compat_stderr_perf.json";
    const char *token_path = "srcs/libs/test_artifact_compat_stderr_perf.txt";
    const char *stdout_path = "srcs/libs/test_artifact_compat_stderr_perf.out";
    const char *stderr_path = "srcs/libs/test_artifact_compat_stderr_perf.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_artifact_compat_stderr_perf.safetensors",
        "--config", "srcs/libs/test_artifact_compat_stderr_perf.json",
        "--tokens", "srcs/libs/test_artifact_compat_stderr_perf.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "2",
        "--perf-per-token",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("artifact-compat stderr perf run",
               run_cli_capture(14, argv, stdout_path, stderr_path), 0);

    expect_file_contains("artifact-compat simd prefix", stderr_path,
                         "lis: simd backend=");
    expect_file_contains("artifact-compat precision additive", stderr_path,
                         "lis: precision path=f32_accum weights=f32 kv=f32");
    expect_file_contains("artifact-compat perf-stage field set", stderr_path,
                         "lis: perf-stage tag=none name=model_load ns=");
    expect_file_contains("artifact-compat perf-stage ms tokens", stderr_path,
                         " ms=");
    expect_file_contains("artifact-compat perf-stage tokens", stderr_path,
                         " tokens=");
    expect_file_contains("artifact-compat perf-summary field set", stderr_path,
                         "lis: perf-summary tag=none threads=1 "
                         "prompt_tokens=");
    expect_file_contains("artifact-compat perf-summary generated tokens", stderr_path,
                         " generated_tokens=");
    expect_file_contains("artifact-compat perf-summary tps", stderr_path,
                         "itl_ms=");
    expect_file_contains("artifact-compat perf-summary tps steady", stderr_path,
                         "tps_steady=");
    expect_file_contains("artifact-compat perf-summary tps e2e", stderr_path,
                         "tps_end_to_end=");
    expect_file_order3("artifact-compat backend precision perf order", stderr_path,
                       "lis: simd backend=",
                       "lis: precision path=",
                       "lis: perf-stage ");

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_artifact_compat_generation_diagnostic_surface(void)
{
    const char *model_path = "srcs/libs/test_artifact_compat_diag.safetensors";
    const char *config_path = "srcs/libs/test_artifact_compat_diag.json";
    const char *token_path = "srcs/libs/test_artifact_compat_diag.txt";
    const char *stdout_path = "srcs/libs/test_artifact_compat_diag.out";
    const char *stderr_path = "srcs/libs/test_artifact_compat_diag.err";
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
        "--model", "srcs/libs/test_artifact_compat_diag.safetensors",
        "--config", "srcs/libs/test_artifact_compat_diag.json",
        "--tokens", "srcs/libs/test_artifact_compat_diag.txt",
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
    expect_status("artifact-compat diag write model",
                  write_safetensors_file_with_data(model_path, header, logits,
                                                   sizeof(logits)),
                  LIS_STATUS_OK);
    expect_status("artifact-compat diag write config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("artifact-compat diag write tokens",
                  write_text_file(token_path, "0\n"), LIS_STATUS_OK);
    expect_int("artifact-compat diag run",
               run_cli_capture(14, argv, stdout_path, stderr_path), 0);

    expect_file_contains("artifact-compat diagnostic field set", stderr_path,
                         "lis: generation-diagnostic step=0 phase=decode "
                         "selected_token_id=0 selected_token_text=<unavailable> "
                         "stop_reason=decode_limit "
                         "structural_suppression_affected=false "
                         "repetition_penalty_changed_selection=false "
                         "selected_token_penalized=false");
    expect_file_contains("artifact-compat reasoning additive", stderr_path,
                         "lis: generation-diagnostic-reasoning step=0 "
                         "phase=decode decision_class=greedy margin=");
    expect_file_contains("artifact-compat candidate field set", stderr_path,
                         "lis: generation-diagnostic-candidate step=0 "
                         "phase=decode rank=1 token_id=0 "
                         "token_text=<unavailable> raw_score=10 "
                         "adjusted_score=10 selected=true");
    expect_file_order3("artifact-compat diag reasoning candidate order", stderr_path,
                       "lis: generation-diagnostic step=",
                       "lis: generation-diagnostic-reasoning ",
                       "lis: generation-diagnostic-candidate ");

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_artifact_compat_layer_checkpoint_surface(void)
{
    const char *model_path =
        "srcs/libs/test_artifact_compat_layer_chk/model.safetensors";
    const char *config_path =
        "srcs/libs/test_artifact_compat_layer_chk/config.json";
    const char *token_path = "srcs/libs/test_artifact_compat_layer_chk.txt";
    const char *stdout_path = "srcs/libs/test_artifact_compat_layer_chk.out";
    const char *stderr_path = "srcs/libs/test_artifact_compat_layer_chk.err";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":2,"
        "\"hidden_size\":1,\"intermediate_size\":1,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":1,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_artifact_compat_layer_chk",
        "--config", "srcs/libs/test_artifact_compat_layer_chk/config.json",
        "--tokens", "srcs/libs/test_artifact_compat_layer_chk.txt",
        "--context", "8",
        "--batch", "1",
        "--generate", "1",
        "--layer-checkpoints", "0",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    if (system("mkdir -p srcs/libs/test_artifact_compat_layer_chk") != 0) {
        fprintf(stderr, "mkdir artifact-compat layer fixture failed\n");
        ++g_failures;
        return;
    }
    expect_status("artifact-compat layer write config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("artifact-compat layer write model",
                  write_llama_checkpoint_fixture(model_path, 2),
                  LIS_STATUS_OK);
    expect_status("artifact-compat layer write tokens",
                  write_text_file(token_path, "0\n"), LIS_STATUS_OK);
    expect_int("artifact-compat layer run",
               run_cli_capture(15, argv, stdout_path, stderr_path), 0);

    expect_file_contains("artifact-compat layer checkpoint field set", stderr_path,
                         "lis: layer-checkpoint step=0 phase=prefill "
                         "name=embedding shape=[1] min=");
    expect_file_contains("artifact-compat layer checkpoint max", stderr_path,
                         " max=");
    expect_file_contains("artifact-compat layer checkpoint mean", stderr_path,
                         " mean=");
    expect_file_contains("artifact-compat layer checkpoint l2", stderr_path,
                         " l2=");
    expect_file_contains("artifact-compat layer checkpoint nan", stderr_path,
                         " nan=");
    expect_file_contains("artifact-compat layer checkpoint inf", stderr_path,
                         " inf=");
    expect_file_contains("artifact-compat layer attn_scale scalar", stderr_path,
                         "lis: layer-checkpoint step=0 phase=prefill "
                         "name=layer.1.attn_scale value=");

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_path);
    remove(stderr_path);
    if (system("rmdir srcs/libs/test_artifact_compat_layer_chk 2>/dev/null") != 0) {
        /* best effort */
    }
}

static void test_cli_artifact_compat_trace_artifact_validity(void)
{
    const char *model_path = "srcs/libs/test_artifact_compat_trace.safetensors";
    const char *config_path = "srcs/libs/test_artifact_compat_trace.json";
    const char *token_path = "srcs/libs/test_artifact_compat_trace.txt";
    const char *trace_path = "srcs/libs/test_artifact_compat_trace.out.json";
    const char *stdout_path = "srcs/libs/test_artifact_compat_trace.out";
    const char *stderr_path = "srcs/libs/test_artifact_compat_trace.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_artifact_compat_trace.safetensors",
        "--config", "srcs/libs/test_artifact_compat_trace.json",
        "--tokens", "srcs/libs/test_artifact_compat_trace.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "2",
        "--trace-json", "srcs/libs/test_artifact_compat_trace.out.json",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("artifact-compat trace run",
               run_cli_capture(15, argv, stdout_path, stderr_path), 0);
    expect_file_contains("artifact-compat decode trace schema", trace_path,
                         "\"schema\":\"lis.execution_artifact/v1\"");
    expect_file_contains("artifact-compat decode trace kind", trace_path,
                         "\"kind\":\"decode_trace\"");

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_artifact_compat_layer_trace_artifact_validity(void)
{
    const char *model_path =
        "srcs/libs/test_artifact_compat_layer_trace/model.safetensors";
    const char *config_path =
        "srcs/libs/test_artifact_compat_layer_trace/config.json";
    const char *token_path = "srcs/libs/test_artifact_compat_layer_trace.txt";
    const char *layer_trace_path =
        "srcs/libs/test_artifact_compat_layer_trace.out.json";
    const char *stdout_path = "srcs/libs/test_artifact_compat_layer_trace.out";
    const char *stderr_path = "srcs/libs/test_artifact_compat_layer_trace.err";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":1,\"intermediate_size\":1,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":1,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_artifact_compat_layer_trace",
        "--config", "srcs/libs/test_artifact_compat_layer_trace/config.json",
        "--tokens", "srcs/libs/test_artifact_compat_layer_trace.txt",
        "--context", "8",
        "--batch", "1",
        "--generate", "1",
        "--layer-checkpoints", "0",
        "--layer-trace-json", "srcs/libs/test_artifact_compat_layer_trace.out.json",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(layer_trace_path);
    remove(stdout_path);
    remove(stderr_path);
    if (system("mkdir -p srcs/libs/test_artifact_compat_layer_trace") != 0) {
        fprintf(stderr, "mkdir artifact-compat layer-trace fixture failed\n");
        ++g_failures;
        return;
    }
    expect_status("artifact-compat layer-trace write config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("artifact-compat layer-trace write model",
                  write_llama_checkpoint_fixture(model_path, 1),
                  LIS_STATUS_OK);
    expect_status("artifact-compat layer-trace write tokens",
                  write_text_file(token_path, "0\n"), LIS_STATUS_OK);
    expect_int("artifact-compat layer-trace run",
               run_cli_capture(17, argv, stdout_path, stderr_path), 0);
    expect_file_contains("artifact-compat layer trace schema", layer_trace_path,
                         "\"schema\":\"lis.execution_artifact/v1\"");
    expect_file_contains("artifact-compat layer trace kind", layer_trace_path,
                         "\"kind\":\"layer_trace\"");

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(layer_trace_path);
    remove(stdout_path);
    remove(stderr_path);
    if (system("rmdir srcs/libs/test_artifact_compat_layer_trace 2>/dev/null") != 0) {
        /* best effort */
    }
}
