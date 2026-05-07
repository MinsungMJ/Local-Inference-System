static void test_cli_layer_trace_json_fingerprint_stable(void)
{
    const char *model_path = "srcs/libs/test_layer_trace_fp/model.safetensors";
    const char *config_path = "srcs/libs/test_layer_trace_fp/config.json";
    const char *token_path = "srcs/libs/test_layer_trace_fp.txt";
    const char *report_path = "srcs/libs/test_layer_trace_fp_report.json";
    const char *trace_path = "srcs/libs/test_layer_trace_fp_trace.json";
    const char *stdout_path = "srcs/libs/test_layer_trace_fp.out";
    const char *stderr_path = "srcs/libs/test_layer_trace_fp.err";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":1,\"intermediate_size\":1,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":1,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_layer_trace_fp",
        "--config", "srcs/libs/test_layer_trace_fp/config.json",
        "--tokens", "srcs/libs/test_layer_trace_fp.txt",
        "--context", "8",
        "--batch", "1",
        "--generate", "1",
        "--report-json", "srcs/libs/test_layer_trace_fp_report.json",
        "--trace-json", "srcs/libs/test_layer_trace_fp_trace.json",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
    if (system("mkdir -p srcs/libs/test_layer_trace_fp") != 0) {
        fprintf(stderr, "mkdir layer_trace fp fixture failed\n");
        ++g_failures;
        return;
    }
    expect_status("layer trace fp write config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("layer trace fp write model",
                  write_llama_checkpoint_fixture(model_path, 1),
                  LIS_STATUS_OK);
    expect_status("layer trace fp write tokens",
                  write_text_file(token_path, "0\n"), LIS_STATUS_OK);
    expect_int("layer trace fp stable run",
               run_cli_capture(17, argv, stdout_path, stderr_path), 0);
    {
        char *r = read_file_content(report_path);
        char *t = read_file_content(trace_path);
        expect_int("layer trace fp report exists", r != NULL ? 0 : 1, 0);
        expect_int("layer trace fp trace exists", t != NULL ? 0 : 1, 0);
        if (r != NULL) free(r);
        if (t != NULL) free(t);
    }
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
    if (system("rmdir srcs/libs/test_layer_trace_fp 2>/dev/null") != 0) {
        /* best effort */
    }
}

static void test_cli_layer_trace_json_precision_path(void)
{
    const char *model_path = "srcs/libs/test_layer_trace_pp/model.safetensors";
    const char *config_path = "srcs/libs/test_layer_trace_pp/config.json";
    const char *token_path = "srcs/libs/test_layer_trace_pp.txt";
    const char *layer_trace_path = "srcs/libs/test_layer_trace_pp.json";
    const char *stdout_path = "srcs/libs/test_layer_trace_pp.out";
    const char *stderr_path = "srcs/libs/test_layer_trace_pp.err";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":1,\"intermediate_size\":1,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":1,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_layer_trace_pp",
        "--config", "srcs/libs/test_layer_trace_pp/config.json",
        "--tokens", "srcs/libs/test_layer_trace_pp.txt",
        "--context", "8",
        "--batch", "1",
        "--generate", "1",
        "--layer-checkpoints", "0",
        "--layer-trace-json", "srcs/libs/test_layer_trace_pp.json",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(layer_trace_path);
    remove(stdout_path);
    remove(stderr_path);
    if (system("mkdir -p srcs/libs/test_layer_trace_pp") != 0) {
        fprintf(stderr, "mkdir layer_trace pp fixture failed\n");
        ++g_failures;
        return;
    }
    expect_status("layer trace pp write config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("layer trace pp write model",
                  write_llama_checkpoint_fixture(model_path, 1),
                  LIS_STATUS_OK);
    expect_status("layer trace pp write tokens",
                  write_text_file(token_path, "0\n"), LIS_STATUS_OK);
    expect_int("layer trace pp path run",
               run_cli_capture(17, argv, stdout_path, stderr_path), 0);
    expect_file_contains("layer trace pp precision_path", layer_trace_path,
                         "\"precision_path\":\"f32_accum;weights=");
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(layer_trace_path);
    remove(stdout_path);
    remove(stderr_path);
    if (system("rmdir srcs/libs/test_layer_trace_pp 2>/dev/null") != 0) {
        /* best effort */
    }
}

static void test_cli_layer_trace_json_attn_scale_exclusion(void)
{
    const char *model_path = "srcs/libs/test_layer_trace_att/model.safetensors";
    const char *config_path = "srcs/libs/test_layer_trace_att/config.json";
    const char *token_path = "srcs/libs/test_layer_trace_att.txt";
    const char *layer_trace_path = "srcs/libs/test_layer_trace_att.json";
    const char *stdout_path = "srcs/libs/test_layer_trace_att.out";
    const char *stderr_path = "srcs/libs/test_layer_trace_att.err";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":2,"
        "\"hidden_size\":1,\"intermediate_size\":1,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":1,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_layer_trace_att",
        "--config", "srcs/libs/test_layer_trace_att/config.json",
        "--tokens", "srcs/libs/test_layer_trace_att.txt",
        "--context", "8",
        "--batch", "1",
        "--generate", "1",
        "--layer-checkpoints", "0",
        "--layer-trace-json", "srcs/libs/test_layer_trace_att.json",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(layer_trace_path);
    remove(stdout_path);
    remove(stderr_path);
    if (system("mkdir -p srcs/libs/test_layer_trace_att") != 0) {
        fprintf(stderr, "mkdir layer_trace att fixture failed\n");
        ++g_failures;
        return;
    }
    expect_status("layer trace att write config",
                  write_text_file(config_path, config_json), LIS_STATUS_OK);
    expect_status("layer trace att write model",
                  write_llama_checkpoint_fixture(model_path, 2),
                  LIS_STATUS_OK);
    expect_status("layer trace att write tokens",
                  write_text_file(token_path, "0\n"), LIS_STATUS_OK);
    expect_int("layer trace att run",
               run_cli_capture(17, argv, stdout_path, stderr_path), 0);
    expect_file_contains("layer trace att stderr attn_scale", stderr_path,
                         "name=layer.1.attn_scale value=");
    {
        char *json = read_file_content(layer_trace_path);
        if (json != NULL) {
            if (strstr(json, ".attn_scale") != NULL) {
                fprintf(stderr, "layer trace att json attn_scale: "
                                "unexpected .attn_scale entry\n");
                ++g_failures;
            }
            free(json);
        }
    }
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(layer_trace_path);
    remove(stdout_path);
    remove(stderr_path);
    if (system("rmdir srcs/libs/test_layer_trace_att 2>/dev/null") != 0) {
        /* best effort */
    }
}
