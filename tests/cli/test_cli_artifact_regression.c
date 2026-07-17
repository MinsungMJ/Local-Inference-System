/* Regression and validation coverage */

static void test_cli_artifact_regression_default_additive_off(void)
{
    const char *model_path = "srcs/libs/test_artifact_regression_off.safetensors";
    const char *config_path = "srcs/libs/test_artifact_regression_off.json";
    const char *token_path = "srcs/libs/test_artifact_regression_off.txt";
    const char *stdout_a = "srcs/libs/test_artifact_regression_off_a.out";
    const char *stderr_a = "srcs/libs/test_artifact_regression_off_a.err";
    const char *stdout_b = "srcs/libs/test_artifact_regression_off_b.out";
    const char *stderr_b = "srcs/libs/test_artifact_regression_off_b.err";
    
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_artifact_regression_off.safetensors",
        "--config", "srcs/libs/test_artifact_regression_off.json",
        "--tokens", "srcs/libs/test_artifact_regression_off.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "2",
    };

    remove(model_path);
    remove(config_path);
    remove(token_path);
    write_valid_fixtures(model_path, config_path, token_path);
    write_text_file(token_path, "0\n");

    expect_int("artifact_regression_off run a", run_cli_capture(sizeof(argv)/sizeof(argv[0]), argv, stdout_a, stderr_a), 0);
    expect_int("artifact_regression_off run b", run_cli_capture(sizeof(argv)/sizeof(argv[0]), argv, stdout_b, stderr_b), 0);

    {
        char *out_a = read_file_content(stdout_a);
        char *out_b = read_file_content(stdout_b);
        char *err_a = read_file_content(stderr_a);
        char *err_b = read_file_content(stderr_b);
        
        expect_string_equals("artifact_regression_off stdout eq", out_a, out_a ? strlen(out_a) : 0, out_b ? out_b : "");
        expect_string_equals("artifact_regression_off stderr eq", err_a, err_a ? strlen(err_a) : 0, err_b ? err_b : "");

        free(out_a);
        free(out_b);
        free(err_a);
        free(err_b);
    }
    
    expect_file_not_contains("artifact_regression_off stderr precision", stderr_a, "lis: precision path=");
    expect_file_not_contains("artifact_regression_off stderr diag", stderr_a, "lis: generation-diagnostic");
    expect_file_not_contains("artifact_regression_off stderr diag reason", stderr_a, "lis: generation-diagnostic-reasoning");
    expect_file_not_contains("artifact_regression_off stderr diag cand", stderr_a, "lis: generation-diagnostic-candidate");
    expect_file_not_contains("artifact_regression_off stderr layer stat", stderr_a, "lis: layer-checkpoint");

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(stdout_a);
    remove(stderr_a);
    remove(stdout_b);
    remove(stderr_b);
}

static void test_cli_artifact_regression_run_report_key_order(void)
{
    const char *model_path = "srcs/libs/test_artifact_regression_report.safetensors";
    const char *config_path = "srcs/libs/test_artifact_regression_report.json";
    const char *token_path = "srcs/libs/test_artifact_regression_report.txt";
    const char *report_path = "srcs/libs/test_artifact_regression_report.json";
    const char *stdout_path = "srcs/libs/test_artifact_regression_report.out";
    const char *stderr_path = "srcs/libs/test_artifact_regression_report.err";
    
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_artifact_regression_report.safetensors",
        "--config", "srcs/libs/test_artifact_regression_report.json",
        "--tokens", "srcs/libs/test_artifact_regression_report.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "2",
        "--report-json", "srcs/libs/test_artifact_regression_report.json",
    };

    write_valid_fixtures(model_path, config_path, token_path);
    write_text_file(token_path, "0\n");
    expect_int("artifact_regression_report run", run_cli_capture(sizeof(argv)/sizeof(argv[0]), argv, stdout_path, stderr_path), 0);
    
    {
        char *data = read_file_content(report_path);
        if (data) {
            char *p1 = strstr(data, "\"schema\":");
            char *p2 = p1 ? strstr(p1, "\"kind\":") : NULL;
            char *p3 = p2 ? strstr(p2, "\"manifest\":{") : NULL;
            char *p4 = p3 ? strstr(p3, "\"report\":{") : NULL;
            
            if (!p1 || !p2 || !p3 || !p4) {
                fprintf(stderr, "artifact_regression_report key order mismatch\n");
                g_failures++;
            }
            
            char *r1 = strstr(data, "\"perf_per_token_enabled\":");
            char *r2 = r1 ? strstr(r1, "\"precision_path\":") : NULL;
            char *r3 = r2 ? strstr(r2, "\"fingerprint\":") : NULL;
            
            if (!r1 || !r2 || !r3) {
                fprintf(stderr, "artifact_regression_report runtime order mismatch\n");
                g_failures++;
            }
            
            free(data);
        } else {
            fprintf(stderr, "artifact_regression_report file read failed\n");
            g_failures++;
        }
    }

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_artifact_regression_report_perf_canonical(void)
{
    const char *model_path = "srcs/libs/test_artifact_regression_perf.safetensors";
    const char *config_path = "srcs/libs/test_artifact_regression_perf.json";
    const char *token_path = "srcs/libs/test_artifact_regression_perf.txt";
    const char *report_path = "srcs/libs/test_artifact_regression_perf.json";
    const char *stdout_path = "srcs/libs/test_artifact_regression_perf.out";
    const char *stderr_path = "srcs/libs/test_artifact_regression_perf.err";
    
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_artifact_regression_perf.safetensors",
        "--config", "srcs/libs/test_artifact_regression_perf.json",
        "--tokens", "srcs/libs/test_artifact_regression_perf.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "2",
        "--perf",
        "--report-json", "srcs/libs/test_artifact_regression_perf.json",
    };

    write_valid_fixtures(model_path, config_path, token_path);
    write_text_file(token_path, "0\n");
    expect_int("artifact_regression_perf run", run_cli_capture(sizeof(argv)/sizeof(argv[0]), argv, stdout_path, stderr_path), 0);
    
    {
        char *data = read_file_content(report_path);
        if (data) {
            char *rep = strstr(data, "\"report\":{");
            if (rep) {
                if (!strstr(rep, "\"perf\":{")) {
                    fprintf(stderr, "artifact_regression_perf canonical perf not found\n");
                    g_failures++;
                }
                if (strstr(rep, "\"perf_summary\":")) {
                    fprintf(stderr, "artifact_regression_perf summary found\n");
                    g_failures++;
                }
            } else {
                fprintf(stderr, "artifact_regression_perf report not found\n");
                g_failures++;
            }
            free(data);
        } else {
            fprintf(stderr, "artifact_regression_perf file read failed\n");
            g_failures++;
        }
    }
    
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_artifact_regression_trace_margin_identity(void)
{
    const char *model_path = "srcs/libs/test_artifact_regression_margin.safetensors";
    const char *config_path = "srcs/libs/test_artifact_regression_margin.json";
    const char *token_path = "srcs/libs/test_artifact_regression_margin.txt";
    const char *trace_path = "srcs/libs/test_artifact_regression_margin_trace.json";
    const char *stdout_path = "srcs/libs/test_artifact_regression_margin.out";
    const char *stderr_path = "srcs/libs/test_artifact_regression_margin.err";
    
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
        "--model", "srcs/libs/test_artifact_regression_margin.safetensors",
        "--config", "srcs/libs/test_artifact_regression_margin.json",
        "--tokens", "srcs/libs/test_artifact_regression_margin.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "2",
        "--trace-json", "srcs/libs/test_artifact_regression_margin_trace.json",
    };

    write_safetensors_file_with_data(model_path, header, logits, sizeof(logits));
    write_text_file(config_path, config_json);
    write_text_file(token_path, "0\n");
    
    expect_int("artifact_regression_margin run", run_cli_capture(sizeof(argv)/sizeof(argv[0]), argv, stdout_path, stderr_path), 0);
    
    {
        char *data = read_file_content(trace_path);
        if (data) {
            char *step = strstr(data, "{\"step\":");
            if (step) {
                double adj = 0, run = 0, margin = 0;
                char *p1 = strstr(step, "\"adjusted_score_selected\":");
                char *p2 = strstr(step, "\"runner_up_adjusted_score\":");
                char *p3 = strstr(step, "\"decision_margin\":");
                if (p1 && p2 && p3) {
                    sscanf(p1, "\"adjusted_score_selected\":%lf", &adj);
                    sscanf(p2, "\"runner_up_adjusted_score\":%lf", &run);
                    sscanf(p3, "\"decision_margin\":%lf", &margin);
                    if (margin > (adj - run) + 1e-5 || margin < (adj - run) - 1e-5) {
                        fprintf(stderr, "artifact_regression_margin identity failed: %f != %f - %f\n", margin, adj, run);
                        g_failures++;
                    }
                } else {
                    fprintf(stderr, "artifact_regression_margin missing fields\n");
                    g_failures++;
                }
            } else {
                fprintf(stderr, "artifact_regression_margin step not found\n");
                g_failures++;
            }
            free(data);
        } else {
            fprintf(stderr, "artifact_regression_margin file read failed\n");
            g_failures++;
        }
    }
    
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_artifact_regression_trace_no_runner_up_null_triple(void)
{
    const char *model_path = "srcs/libs/test_artifact_regression_null.safetensors";
    const char *config_path = "srcs/libs/test_artifact_regression_null.json";
    const char *token_path = "srcs/libs/test_artifact_regression_null.txt";
    const char *trace_path = "srcs/libs/test_artifact_regression_null_trace.json";
    const char *stdout_path = "srcs/libs/test_artifact_regression_null.out";
    const char *stderr_path = "srcs/libs/test_artifact_regression_null.err";
    
    const char *header =
        "{\"lis.validation_logits\":{\"dtype\":\"F32\",\"shape\":[1,1],"
        "\"data_offsets\":[0,4]}}";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":4,\"intermediate_size\":8,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":4,\"vocab_size\":1,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    const float logits[1] = { 10.0f };
    
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_artifact_regression_null.safetensors",
        "--config", "srcs/libs/test_artifact_regression_null.json",
        "--tokens", "srcs/libs/test_artifact_regression_null.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "2",
        "--trace-json", "srcs/libs/test_artifact_regression_null_trace.json",
    };

    write_safetensors_file_with_data(model_path, header, logits, sizeof(logits));
    write_text_file(config_path, config_json);
    write_text_file(token_path, "0\n");
    
    expect_int("artifact_regression_null run", run_cli_capture(sizeof(argv)/sizeof(argv[0]), argv, stdout_path, stderr_path), 0);
    
    expect_file_contains("artifact_regression_null triple", trace_path, "\"runner_up_token_id\":null");
    expect_file_contains("artifact_regression_null triple", trace_path, "\"runner_up_adjusted_score\":null");
    expect_file_contains("artifact_regression_null triple", trace_path, "\"decision_margin\":null");
    
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_artifact_regression_trace_topk_bounds_and_selected(void)
{
    const char *model_path = "srcs/libs/test_artifact_regression_topk.safetensors";
    const char *config_path = "srcs/libs/test_artifact_regression_topk.json";
    const char *token_path = "srcs/libs/test_artifact_regression_topk.txt";
    const char *trace_path = "srcs/libs/test_artifact_regression_topk_trace.json";
    const char *stdout_path = "srcs/libs/test_artifact_regression_topk.out";
    const char *stderr_path = "srcs/libs/test_artifact_regression_topk.err";
    
    const char *header =
        "{\"lis.validation_logits\":{\"dtype\":\"F32\",\"shape\":[1,8],"
        "\"data_offsets\":[0,32]}}";
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":1,"
        "\"hidden_size\":4,\"intermediate_size\":8,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":4,\"vocab_size\":8,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
    const float logits[8] = { 8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f };
    
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_artifact_regression_topk.safetensors",
        "--config", "srcs/libs/test_artifact_regression_topk.json",
        "--tokens", "srcs/libs/test_artifact_regression_topk.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "2",
        "--trace-json", "srcs/libs/test_artifact_regression_topk_trace.json",
    };

    write_safetensors_file_with_data(model_path, header, logits, sizeof(logits));
    write_text_file(config_path, config_json);
    write_text_file(token_path, "0\n");
    
    expect_int("artifact_regression_topk run", run_cli_capture(sizeof(argv)/sizeof(argv[0]), argv, stdout_path, stderr_path), 0);
    
    {
        char *data = read_file_content(trace_path);
        if (data) {
            char *topk = data;
            while ((topk = strstr(topk, "\"topk\":[")) != NULL) {
                char *end = strstr(topk, "]");
                if (!end) break;
                int count = 0;
                char *entry = topk;
                while ((entry = strstr(entry, "{\"token_id\":")) != NULL && entry < end) {
                    count++;
                    entry++;
                }
                if (count > 5) {
                    fprintf(stderr, "artifact_regression_topk exceeded 5: got %d\n", count);
                    g_failures++;
                }
                
                int selected_count = 0;
                entry = topk;
                while ((entry = strstr(entry, "\"is_selected\":true")) != NULL && entry < end) {
                    selected_count++;
                    entry++;
                }
                if (selected_count != 1) {
                    fprintf(stderr, "artifact_regression_topk selected unique failed: got %d\n", selected_count);
                    g_failures++;
                }
                
                topk = end;
            }
            free(data);
        } else {
            fprintf(stderr, "artifact_regression_topk file read failed\n");
            g_failures++;
        }
    }
    
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_artifact_regression_trace_phase_and_stop(void)
{
    const char *model_path = "srcs/libs/test_artifact_regression_phase.safetensors";
    const char *config_path = "srcs/libs/test_artifact_regression_phase.json";
    const char *token_path = "srcs/libs/test_artifact_regression_phase.txt";
    const char *trace_path = "srcs/libs/test_artifact_regression_phase_trace.json";
    const char *stdout_path = "srcs/libs/test_artifact_regression_phase.out";
    const char *stderr_path = "srcs/libs/test_artifact_regression_phase.err";
    
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_artifact_regression_phase.safetensors",
        "--config", "srcs/libs/test_artifact_regression_phase.json",
        "--tokens", "srcs/libs/test_artifact_regression_phase.txt",
        "--context", "8",
        "--batch", "1",
        "--generate", "4",
        "--trace-json", "srcs/libs/test_artifact_regression_phase_trace.json",
    };

    write_valid_fixtures(model_path, config_path, token_path);
    write_text_file(token_path, "0\n");
    expect_int("artifact_regression_phase run", run_cli_capture(sizeof(argv)/sizeof(argv[0]), argv, stdout_path, stderr_path), 0);
    
    {
        char *data = read_file_content(trace_path);
        if (data) {
            if (!strstr(data, "{\"step\":0,\"phase\":\"first_decode\"")) {
                fprintf(stderr, "artifact_regression_phase step 0 missing\n");
                g_failures++;
            }
            if (!strstr(data, "{\"step\":1,\"phase\":\"decode\"")) {
                fprintf(stderr, "artifact_regression_phase step 1 missing\n");
                g_failures++;
            }
            if (!strstr(data, "{\"step\":2,\"phase\":\"decode\"")) {
                fprintf(stderr, "artifact_regression_phase step 2 missing\n");
                g_failures++;
            }
            if (!strstr(data, "{\"step\":3,\"phase\":\"decode\"")) {
                fprintf(stderr, "artifact_regression_phase step 3 missing\n");
                g_failures++;
            }
            if (strstr(data, "\"phase\":\"prefill_seed\"")) {
                fprintf(stderr, "artifact_regression_phase has prefill_seed\n");
                g_failures++;
            }
            
            free(data);
        } else {
            fprintf(stderr, "artifact_regression_phase file read failed\n");
            g_failures++;
        }
    }
    
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_artifact_regression_decision_class_greedy(void)
{
    const char *model_path = "srcs/libs/test_artifact_regression_decision.safetensors";
    const char *config_path = "srcs/libs/test_artifact_regression_decision.json";
    const char *token_path = "srcs/libs/test_artifact_regression_decision.txt";
    const char *trace_path = "srcs/libs/test_artifact_regression_decision_trace.json";
    const char *stdout_path = "srcs/libs/test_artifact_regression_decision.out";
    const char *stderr_path = "srcs/libs/test_artifact_regression_decision.err";
    
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
        "--model", "srcs/libs/test_artifact_regression_decision.safetensors",
        "--config", "srcs/libs/test_artifact_regression_decision.json",
        "--tokens", "srcs/libs/test_artifact_regression_decision.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "2",
        "--trace-json", "srcs/libs/test_artifact_regression_decision_trace.json",
    };

    write_safetensors_file_with_data(model_path, header, logits, sizeof(logits));
    write_text_file(config_path, config_json);
    write_text_file(token_path, "0\n");
    
    expect_int("artifact_regression_decision run", run_cli_capture(sizeof(argv)/sizeof(argv[0]), argv, stdout_path, stderr_path), 0);
    
    expect_file_contains("artifact_regression_decision greedy", trace_path, "\"decision_class\":\"greedy\"");
    expect_file_contains("artifact_regression_decision structural false", trace_path, "\"structural_suppression_affected\":false");
    expect_file_contains("artifact_regression_decision repetition false", trace_path, "\"repetition_penalty_changed_selection\":false");
    expect_file_contains("artifact_regression_decision penalized false", trace_path, "\"selected_token_penalized\":false");
    expect_file_contains("artifact_regression_decision suppressed 0", trace_path, "\"suppressed_token_count\":0");
    expect_file_contains("artifact_regression_decision penalized 0", trace_path, "\"penalized_token_count\":0");
    
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_artifact_regression_decision_class_penalty_shifted(void)
{
    const char *model_path = "srcs/libs/test_artifact_regression_pen.safetensors";
    const char *config_path = "srcs/libs/test_artifact_regression_pen.json";
    const char *token_path = "srcs/libs/test_artifact_regression_pen.txt";
    const char *trace_path = "srcs/libs/test_artifact_regression_pen_trace.json";
    const char *stdout_path = "srcs/libs/test_artifact_regression_pen.out";
    const char *stderr_path = "srcs/libs/test_artifact_regression_pen.err";
    
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
        "--model", "srcs/libs/test_artifact_regression_pen.safetensors",
        "--config", "srcs/libs/test_artifact_regression_pen.json",
        "--tokens", "srcs/libs/test_artifact_regression_pen.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "2",
        "--trace-json", "srcs/libs/test_artifact_regression_pen_trace.json",
    };

    write_safetensors_file_with_data(model_path, header, logits, sizeof(logits));
    write_text_file(config_path, config_json);
    write_text_file(token_path, "0\n");
    
    expect_int("artifact_regression_pen run", run_cli_capture(sizeof(argv)/sizeof(argv[0]), argv, stdout_path, stderr_path), 0);
    
    expect_file_contains("artifact_regression_pen rep shift", trace_path, "\"decision_class\":\"repetition_penalty_shifted\"");
    expect_file_contains("artifact_regression_pen rep changed", trace_path, "\"repetition_penalty_changed_selection\":true");
    expect_file_contains("artifact_regression_pen pen count 1", trace_path, "\"penalized_token_count\":1");
    
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_artifact_regression_decision_class_structural_and_taxonomy(void)
{
    const char *model_path = "srcs/libs/test_artifact_regression_str.safetensors";
    const char *config_path = "srcs/libs/test_artifact_regression_str.json";
    const char *tokenizer_path = "srcs/libs/test_artifact_regression_str_tok.json";
    const char *trace_path = "srcs/libs/test_artifact_regression_str_trace.json";
    const char *stdout_path = "srcs/libs/test_artifact_regression_str.out";
    const char *stderr_path = "srcs/libs/test_artifact_regression_str.err";
    
    const float logits[10] = { 0.0f, 8.0f, 10.0f, 12.0f, 11.0f,
                               -1.0f, 0.0f, 0.0f, 0.0f, 0.0f };
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_artifact_regression_str.safetensors",
        "--config", "srcs/libs/test_artifact_regression_str.json",
        "--hf-tokenizer", "srcs/libs/test_artifact_regression_str_tok.json",
        "--prompt", "a",
        "--context", "32",
        "--batch", "1",
        "--generate", "2",
        "--trace-json", "srcs/libs/test_artifact_regression_str_trace.json",
    };

    write_structural_token_fixtures(model_path, config_path, tokenizer_path, logits);
    
    expect_int("artifact_regression_str run", run_cli_capture(sizeof(argv)/sizeof(argv[0]), argv, stdout_path, stderr_path), 0);
    
    expect_file_contains("artifact_regression_str struct class", trace_path, "\"decision_class\":\"structural_suppression\"");
    expect_file_contains("artifact_regression_str struct affected", trace_path, "\"structural_suppression_affected\":true");
    expect_file_not_contains("artifact_regression_str struct count 0", trace_path, "\"suppressed_token_count\":0");
    
    {
        char *data = read_file_content(trace_path);
        if (data) {
            char *cls = data;
            while ((cls = strstr(cls, "\"decision_class\":")) != NULL) {
                if (strncmp(cls, "\"decision_class\":\"greedy\"", 25) != 0 &&
                    strncmp(cls, "\"decision_class\":\"repetition_penalty_shifted\"", 47) != 0 &&
                    strncmp(cls, "\"decision_class\":\"structural_suppression\"", 41) != 0) {
                    fprintf(stderr, "artifact_regression_str invalid decision_class\n");
                    g_failures++;
                }
                cls += 17;
            }
            free(data);
        }
    }
    
    remove(model_path);
    remove(config_path);
    remove(tokenizer_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_artifact_regression_layer_trace_parity_and_precision(void)
{
    const char *model_path = "srcs/libs/test_artifact_regression_layer/model.safetensors";
    const char *config_path = "srcs/libs/test_artifact_regression_layer/config.json";
    const char *token_path = "srcs/libs/test_artifact_regression_layer.txt";
    const char *trace_path = "srcs/libs/test_artifact_regression_layer_trace.json";
    const char *stdout_path = "srcs/libs/test_artifact_regression_layer.out";
    const char *stderr_path = "srcs/libs/test_artifact_regression_layer.err";
    
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":2,"
        "\"hidden_size\":1,\"intermediate_size\":1,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":1,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
        
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_artifact_regression_layer",
        "--config", "srcs/libs/test_artifact_regression_layer/config.json",
        "--tokens", "srcs/libs/test_artifact_regression_layer.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "1",
        "--layer-checkpoints", "0",
        "--layer-trace-json", "srcs/libs/test_artifact_regression_layer_trace.json",
    };

    if (system("mkdir -p srcs/libs/test_artifact_regression_layer") != 0) {
        fprintf(stderr, "mkdir artifact_regression_layer fixture failed\n");
        g_failures++;
    }
    write_llama_checkpoint_fixture(model_path, 2);
    write_text_file(config_path, config_json);
    write_text_file(token_path, "0\n");
    
    expect_int("artifact_regression_layer run", run_cli_capture(sizeof(argv)/sizeof(argv[0]), argv, stdout_path, stderr_path), 0);
    
    expect_file_not_contains("artifact_regression_layer absent precision stderr", stderr_path, "lis: precision path=");
    expect_file_contains("artifact_regression_layer precision trace", trace_path, "\"precision_path\":\"f32_accum;weights=");
    
    {
        char *err_data = read_file_content(stderr_path);
        char *trace_data = read_file_content(trace_path);
        if (err_data && trace_data) {
            int err_count = 0;
            char *p = err_data;
            while ((p = strstr(p, "lis: layer-checkpoint ")) != NULL) {
                char *nl = strchr(p, '\n');
                char *shape = strstr(p, "shape=[");
                if (shape && (!nl || shape < nl)) {
                    err_count++;
                }
                p += 22;
            }
            
            int trace_count = 0;
            p = strstr(trace_data, "\"layer_trace\":[");
            if (p) {
                while ((p = strstr(p, "{\"step\":")) != NULL) {
                    trace_count++;
                    p++;
                }
            }
            
            if (err_count != trace_count) {
                fprintf(stderr, "artifact_regression_layer count mismatch: %d != %d\n", err_count, trace_count);
                g_failures++;
            }
            
            /* Check one value parity */
            p = strstr(err_data, "name=embedding shape=[1] min=");
            if (!p) {
                fprintf(stderr, "artifact_regression_layer missing embedding checkpoint in stderr\n");
                g_failures++;
            } else {
                char min_v[64] = {0}, max_v[64] = {0}, mean_v[64] = {0};
                char l2_v[64] = {0}, nan_v[64] = {0}, inf_v[64] = {0};
                int n = sscanf(p,
                    "name=embedding shape=[1] min=%63s max=%63s mean=%63s l2=%63s nan=%63s inf=%63s",
                    min_v, max_v, mean_v, l2_v, nan_v, inf_v);
                if (n != 6) {
                    fprintf(stderr, "artifact_regression_layer embedding parse got %d/6 fields\n", n);
                    g_failures++;
                } else {
                    char probe[256];
                    snprintf(probe, sizeof(probe),
                        "\"name\":\"embedding\",\"shape\":[1],\"min\":%s,\"max\":%s",
                        min_v, max_v);
                    if (!strstr(trace_data, probe)) {
                        fprintf(stderr, "artifact_regression_layer min/max parity failed\n");
                        g_failures++;
                    }
                    snprintf(probe, sizeof(probe),
                        "\"mean\":%s,\"l2\":%s", mean_v, l2_v);
                    if (!strstr(trace_data, probe)) {
                        fprintf(stderr, "artifact_regression_layer mean/l2 parity failed\n");
                        g_failures++;
                    }
                    snprintf(probe, sizeof(probe),
                        "\"nan\":%s,\"inf\":%s", nan_v, inf_v);
                    if (!strstr(trace_data, probe)) {
                        fprintf(stderr, "artifact_regression_layer nan/inf parity failed\n");
                        g_failures++;
                    }
                }
            }
        }
        free(err_data);
        free(trace_data);
    }
    
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
    if (system("rm -rf srcs/libs/test_artifact_regression_layer") != 0) {
        /* best effort */
    }
}

static void test_cli_artifact_regression_runtime_fingerprint_repeatable(void)
{
    const char *model_path = "srcs/libs/test_artifact_regression_fp.safetensors";
    const char *config_path = "srcs/libs/test_artifact_regression_fp.json";
    const char *token_path = "srcs/libs/test_artifact_regression_fp.txt";
    const char *report_a = "srcs/libs/test_artifact_regression_fpa.json";
    const char *report_b = "srcs/libs/test_artifact_regression_fpb.json";
    const char *stdout_path = "srcs/libs/test_artifact_regression_fp.out";
    const char *stderr_path = "srcs/libs/test_artifact_regression_fp.err";
    
    char *argv_a[] = {
        "lis",
        "--model", "srcs/libs/test_artifact_regression_fp.safetensors",
        "--config", "srcs/libs/test_artifact_regression_fp.json",
        "--tokens", "srcs/libs/test_artifact_regression_fp.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "2",
        "--report-json", "srcs/libs/test_artifact_regression_fpa.json",
    };
    
    char *argv_b[] = {
        "lis",
        "--model", "srcs/libs/test_artifact_regression_fp.safetensors",
        "--config", "srcs/libs/test_artifact_regression_fp.json",
        "--tokens", "srcs/libs/test_artifact_regression_fp.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "2",
        "--report-json", "srcs/libs/test_artifact_regression_fpb.json",
    };

    write_valid_fixtures(model_path, config_path, token_path);
    write_text_file(token_path, "0\n");
    expect_int("artifact_regression_fp run a", run_cli_capture(sizeof(argv_a)/sizeof(argv_a[0]), argv_a, stdout_path, stderr_path), 0);
    expect_int("artifact_regression_fp run b", run_cli_capture(sizeof(argv_b)/sizeof(argv_b[0]), argv_b, stdout_path, stderr_path), 0);
    
    char *data_a = read_file_content(report_a);
    char *data_b = read_file_content(report_b);
    if (data_a && data_b) {
        char *rt_a = strstr(data_a, "\"runtime\":{");
        char *rt_b = strstr(data_b, "\"runtime\":{");
        if (rt_a && rt_b) {
            char *fp_a = strstr(rt_a, "\"fingerprint\":{\"algorithm\":\"fnv1a64\",\"hex\":\"");
            char *fp_b = strstr(rt_b, "\"fingerprint\":{\"algorithm\":\"fnv1a64\",\"hex\":\"");
            if (fp_a && fp_b) {
                char digest_a[65] = {0};
                char digest_b[65] = {0};
                sscanf(fp_a, "\"fingerprint\":{\"algorithm\":\"fnv1a64\",\"hex\":\"%64[^\"]\"", digest_a);
                sscanf(fp_b, "\"fingerprint\":{\"algorithm\":\"fnv1a64\",\"hex\":\"%64[^\"]\"", digest_b);
                if (strcmp(digest_a, digest_b) != 0) {
                    fprintf(stderr, "artifact_regression_fp repeatable fingerprint failed\n");
                    g_failures++;
                }
            } else {
                fprintf(stderr, "artifact_regression_fp fingerprint missing\n");
                g_failures++;
            }
        }
    }
    free(data_a);
    free(data_b);
    
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_a);
    remove(report_b);
    remove(stdout_path);
    remove(stderr_path);
}

static void test_cli_artifact_regression_trace_determinism(void)
{
    const char *model_path = "srcs/libs/test_artifact_regression_det/model.safetensors";
    const char *config_path = "srcs/libs/test_artifact_regression_det/config.json";
    const char *token_path = "srcs/libs/test_artifact_regression_det.txt";
    const char *trace_a = "srcs/libs/test_artifact_regression_det_trace_a.json";
    const char *trace_b = "srcs/libs/test_artifact_regression_det_trace_b.json";
    const char *layer_a = "srcs/libs/test_artifact_regression_det_layer_a.json";
    const char *layer_b = "srcs/libs/test_artifact_regression_det_layer_b.json";
    const char *stdout_path = "srcs/libs/test_artifact_regression_det.out";
    const char *stderr_path = "srcs/libs/test_artifact_regression_det.err";
    
    const char *config_json =
        "{\"model_type\":\"llama\",\"num_hidden_layers\":2,"
        "\"hidden_size\":1,\"intermediate_size\":1,"
        "\"num_attention_heads\":1,\"num_key_value_heads\":1,"
        "\"head_dim\":1,\"vocab_size\":3,"
        "\"rope_theta\":10000.0,\"torch_dtype\":\"float32\","
        "\"max_position_embeddings\":8}";
        
    char *argv_a[] = {
        "lis",
        "--model", "srcs/libs/test_artifact_regression_det",
        "--config", "srcs/libs/test_artifact_regression_det/config.json",
        "--tokens", "srcs/libs/test_artifact_regression_det.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "1",
        "--trace-json", "srcs/libs/test_artifact_regression_det_trace_a.json",
        "--layer-checkpoints", "0",
        "--layer-trace-json", "srcs/libs/test_artifact_regression_det_layer_a.json",
    };
    
    char *argv_b[] = {
        "lis",
        "--model", "srcs/libs/test_artifact_regression_det",
        "--config", "srcs/libs/test_artifact_regression_det/config.json",
        "--tokens", "srcs/libs/test_artifact_regression_det.txt",
        "--context", "4",
        "--batch", "1",
        "--generate", "1",
        "--trace-json", "srcs/libs/test_artifact_regression_det_trace_b.json",
        "--layer-checkpoints", "0",
        "--layer-trace-json", "srcs/libs/test_artifact_regression_det_layer_b.json",
    };

    if (system("mkdir -p srcs/libs/test_artifact_regression_det") != 0) {
        fprintf(stderr, "mkdir artifact_regression_det fixture failed\n");
        g_failures++;
    }
    write_llama_checkpoint_fixture(model_path, 2);
    write_text_file(config_path, config_json);
    write_text_file(token_path, "0\n");
    
    expect_int("artifact_regression_det run a", run_cli_capture(sizeof(argv_a)/sizeof(argv_a[0]), argv_a, stdout_path, stderr_path), 0);
    expect_int("artifact_regression_det run b", run_cli_capture(sizeof(argv_b)/sizeof(argv_b[0]), argv_b, stdout_path, stderr_path), 0);
    
    char *ta = read_file_content(trace_a);
    char *tb = read_file_content(trace_b);
    char *la = read_file_content(layer_a);
    char *lb = read_file_content(layer_b);
    
    if (ta && tb && la && lb) {
        static const char marker[] = "\"artifact_set_id\":\"aset1:";
        char *ta_id = strstr(ta, marker);
        char *tb_id = strstr(tb, marker);
        char *la_id = strstr(la, marker);
        char *lb_id = strstr(lb, marker);
        const size_t prefix_len = sizeof(marker) - 1U;

        if (!ta_id || !tb_id || !la_id || !lb_id) {
            fprintf(stderr, "artifact_regression_det artifact-set ID missing\n");
            g_failures++;
        } else {
            if (memcmp(ta_id + prefix_len, la_id + prefix_len, 32U) != 0 ||
                memcmp(tb_id + prefix_len, lb_id + prefix_len, 32U) != 0) {
                fprintf(stderr,
                        "artifact_regression_det same-execution IDs differ\n");
                g_failures++;
            }
            if (memcmp(ta_id + prefix_len, tb_id + prefix_len, 32U) == 0) {
                fprintf(stderr,
                        "artifact_regression_det separate executions reused ID\n");
                g_failures++;
            }
            memset(ta_id + prefix_len, '0', 32U);
            memset(tb_id + prefix_len, '0', 32U);
            memset(la_id + prefix_len, '0', 32U);
            memset(lb_id + prefix_len, '0', 32U);
        }
    }
    expect_string_equals("artifact_regression_det trace eq except ID",
                         ta, ta ? strlen(ta) : 0, tb ? tb : "");
    expect_string_equals("artifact_regression_det layer eq except ID",
                         la, la ? strlen(la) : 0, lb ? lb : "");
    
    const char *forbidden[] = {
        "timestamp", "created_at", "wall_", "clock_", "epoch", "unix_",
        "0x", "random", "rand_", "uuid", "guid", "INFINITY", "NaN", ":NaN", "nan,"
    };
    
    if (ta && la) {
        for (size_t i = 0; i < sizeof(forbidden) / sizeof(forbidden[0]); i++) {
            if (strstr(ta, forbidden[i]) || strstr(la, forbidden[i])) {
                fprintf(stderr, "artifact_regression_det forbidden token found: %s\n", forbidden[i]);
                g_failures++;
            }
        }
    }
    
    free(ta);
    free(tb);
    free(la);
    free(lb);
    
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(trace_a);
    remove(trace_b);
    remove(layer_a);
    remove(layer_b);
    remove(stdout_path);
    remove(stderr_path);
    if (system("rm -rf srcs/libs/test_artifact_regression_det") != 0) {
        /* best effort */
    }
}

static void test_cli_artifact_regression_trace_eos_stop(void)
{
    const char *model_path = "srcs/libs/test_artifact_regression_eos.safetensors";
    const char *config_path = "srcs/libs/test_artifact_regression_eos.json";
    const char *tok_path = "srcs/libs/test_artifact_regression_eos_tok.json";
    const char *trace_path = "srcs/libs/test_artifact_regression_eos_trace.json";
    const char *stdout_path = "srcs/libs/test_artifact_regression_eos.out";
    const char *stderr_path = "srcs/libs/test_artifact_regression_eos.err";

    const float logits[10] = {
        -1.0f, -1.0f, -1.0f, -1.0f, -1.0f,
         5.0f, -1.0f, -1.0f, -1.0f, -1.0f
    };

    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_artifact_regression_eos.safetensors",
        "--config", "srcs/libs/test_artifact_regression_eos.json",
        "--hf-tokenizer", "srcs/libs/test_artifact_regression_eos_tok.json",
        "--prompt", "a",
        "--context", "32",
        "--batch", "1",
        "--generate", "100",
        "--trace-json", "srcs/libs/test_artifact_regression_eos_trace.json",
    };

    write_structural_token_fixtures(model_path, config_path, tok_path, logits);

    expect_int("artifact_regression_eos run", run_cli_capture(sizeof(argv)/sizeof(argv[0]), argv, stdout_path, stderr_path), 0);

    {
        char *data = read_file_content(trace_path);
        if (data) {
            char *stop1 = strstr(data, "\"stop_reason\":");
            if (!stop1) {
                fprintf(stderr, "artifact_regression_eos missing stop_reason\n");
                g_failures++;
            } else {
                char *stop2 = strstr(stop1 + 1, "\"stop_reason\":");
                if (stop2) {
                    fprintf(stderr, "artifact_regression_eos multiple stop_reason\n");
                    g_failures++;
                }
                char *next_step = strstr(stop1, "{\"step\":");
                if (next_step) {
                    fprintf(stderr, "artifact_regression_eos stop_reason not in terminal step\n");
                    g_failures++;
                }
            }
            free(data);
        } else {
            fprintf(stderr, "artifact_regression_eos file read failed\n");
            g_failures++;
        }
    }

    remove(model_path);
    remove(config_path);
    remove(tok_path);
    remove(trace_path);
    remove(stdout_path);
    remove(stderr_path);
}
