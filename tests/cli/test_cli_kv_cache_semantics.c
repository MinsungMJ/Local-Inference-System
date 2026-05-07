/* KV correctness and boundary regression tests */

/*
 * Simple JSON value extraction helpers: find a key like "key":<value>
 * and extract the integer or string value. No full JSON parser needed.
 */

static int extract_json_int(const char *json, const char *key,
                            long *out_value)
{
    char search[256];
    int n = snprintf(search, sizeof(search), "\"%s\":", key);
    const char *pos;
    char *end;

    if (n < 0 || (size_t)n >= sizeof(search)) {
        return 0;
    }
    pos = strstr(json, search);
    if (pos == NULL) {
        return 0;
    }
    pos += (size_t)n;
    *out_value = strtol(pos, &end, 10);
    return (end != pos) ? 1 : 0;
}

static const char *extract_json_string(const char *json, const char *key,
                                       char *out_buf, size_t out_buf_sz)
{
    char search[256];
    int n = snprintf(search, sizeof(search), "\"%s\":\"", key);
    const char *pos;
    const char *end;

    if (n < 0 || (size_t)n >= sizeof(search)) {
        return NULL;
    }
    pos = strstr(json, search);
    if (pos == NULL) {
        return NULL;
    }
    pos += (size_t)n;
    end = strchr(pos, '"');
    if (end == NULL) {
        return NULL;
    }
    {
        size_t len = (size_t)(end - pos);
        if (len >= out_buf_sz) {
            len = out_buf_sz - 1;
        }
        memcpy(out_buf, pos, len);
        out_buf[len] = '\0';
    }
    return out_buf;
}

static char *extract_json_object(const char *json, const char *key)
{
    char search[256];
    int n = snprintf(search, sizeof(search), "\"%s\":{", key);
    const char *pos;
    const char *start;
    const char *p;
    int depth = 0;
    int in_string = 0;
    int escaped = 0;

    if (n < 0 || (size_t)n >= sizeof(search)) {
        return NULL;
    }
    pos = strstr(json, search);
    if (pos == NULL) {
        return NULL;
    }
    start = strchr(pos, '{');
    if (start == NULL) {
        return NULL;
    }

    for (p = start; *p != '\0'; ++p) {
        if (in_string) {
            if (escaped) {
                escaped = 0;
            } else if (*p == '\\') {
                escaped = 1;
            } else if (*p == '"') {
                in_string = 0;
            }
            continue;
        }
        if (*p == '"') {
            in_string = 1;
        } else if (*p == '{') {
            ++depth;
        } else if (*p == '}') {
            --depth;
            if (depth == 0) {
                size_t len = (size_t)(p - start + 1);
                char *out = (char *)malloc(len + 1);
                if (out == NULL) {
                    return NULL;
                }
                memcpy(out, start, len);
                out[len] = '\0';
                return out;
            }
        }
    }

    return NULL;
}

/*
 * Extract the kv=<dtype> token from a precision_path string such as
 * "f32_accum;weights=f32;kv=f16".
 */
static const char *parse_kv_dtype_from_precision(const char *path,
                                                 char *out_buf,
                                                 size_t out_buf_sz)
{
    const char *kv = strstr(path, ";kv=");

    if (kv == NULL) {
        return NULL;
    }
    kv += 4;
    {
        const char *end = strchr(kv, ';');
        size_t len;

        if (end != NULL) {
            len = (size_t)(end - kv);
        } else {
            len = strlen(kv);
        }
        if (len >= out_buf_sz) {
            len = out_buf_sz - 1;
        }
        memcpy(out_buf, kv, len);
        out_buf[len] = '\0';
    }
    return out_buf;
}

/*
 * Test 1: Accounting formula invariants.
 *
 * Generate a run_report via the validation-logits path, then verify:
 *   allocated_bytes == bytes_per_token * max_tokens
 *   used_bytes     == bytes_per_token * used_tokens
 *   max_tokens equals the configured context capacity (--context)
 *   used_tokens is a logical token-position count (prompt + generated tokens)
 */
static void test_kv_cache_semantics_accounting_formulas(void)
{
    const char *model_path = "srcs/libs/test_kv_cache_formula.safetensors";
    const char *config_path = "srcs/libs/test_kv_cache_formula.json";
    const char *token_path = "srcs/libs/test_kv_cache_formula.txt";
    const char *report_path = "srcs/libs/test_kv_cache_formula.json.out";
    const char *stdout_path = "srcs/libs/test_kv_cache_formula.out";
    const char *stderr_path = "srcs/libs/test_kv_cache_formula.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_kv_cache_formula.safetensors",
        "--config", "srcs/libs/test_kv_cache_formula.json",
        "--tokens", "srcs/libs/test_kv_cache_formula.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "2",
        "--report-json", "srcs/libs/test_kv_cache_formula.json.out",
    };
    char *report_content;
    long max_tokens = 0;
    long used_tokens = 0;
    long bytes_per_token = 0;
    long allocated_bytes = 0;
    long used_bytes = 0;
    long computed_allocated = 0;
    long computed_used = 0;
    char *kv_cache;

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("kv cache formula run", run_cli_capture(15, argv, stdout_path,
                                                       stderr_path), 0);

    report_content = read_file_content(report_path);
    if (report_content == NULL) {
        fprintf(stderr, "kv cache formula: could not read report file\n");
        ++g_failures;
        goto cleanup;
    }

    kv_cache = extract_json_object(report_content, "kv_cache");
    if (kv_cache == NULL) {
        fprintf(stderr, "kv cache formula: could not extract kv_cache object\n");
        ++g_failures;
        free(report_content);
        goto cleanup;
    }

    if (!extract_json_int(kv_cache, "max_tokens", &max_tokens) ||
        !extract_json_int(kv_cache, "used_tokens", &used_tokens) ||
        !extract_json_int(kv_cache, "bytes_per_token", &bytes_per_token) ||
        !extract_json_int(kv_cache, "allocated_bytes", &allocated_bytes) ||
        !extract_json_int(kv_cache, "used_bytes", &used_bytes)) {
        fprintf(stderr, "kv cache formula: could not extract kv_cache fields\n");
        ++g_failures;
        free(kv_cache);
        free(report_content);
        goto cleanup;
    }

    computed_allocated = bytes_per_token * max_tokens;
    computed_used = bytes_per_token * used_tokens;

    expect_int("kv cache formula allocated_bytes", (int)allocated_bytes,
               (int)computed_allocated);
    expect_int("kv cache formula used_bytes", (int)used_bytes,
               (int)computed_used);
    expect_int("kv cache formula max_tokens == context", (int)max_tokens, 4);
    expect_int("kv cache formula used_tokens logical", (int)used_tokens, 4);

    {
        const int prompt_tokens_per_seq = 2;
        const int generated_tokens = 2;
        expect_int("kv cache formula used_tokens formula",
                   (int)used_tokens,
                   prompt_tokens_per_seq + generated_tokens);
    }

    free(kv_cache);
    free(report_content);

cleanup:
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
}

/*
 * Test 2: Precision/storage consistency.
 *
 * Parse manifest.runtime.precision_path, extract the bounded kv=<dtype> token,
 * then extract report.kv_cache.storage_dtype and assert they match.
 */
static void test_kv_cache_semantics_precision_storage_consistency(void)
{
    const char *model_path = "srcs/libs/test_kv_cache_dtype.safetensors";
    const char *config_path = "srcs/libs/test_kv_cache_dtype.json";
    const char *token_path = "srcs/libs/test_kv_cache_dtype.txt";
    const char *report_path = "srcs/libs/test_kv_cache_dtype.json.out";
    const char *stdout_path = "srcs/libs/test_kv_cache_dtype.out";
    const char *stderr_path = "srcs/libs/test_kv_cache_dtype.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_kv_cache_dtype.safetensors",
        "--config", "srcs/libs/test_kv_cache_dtype.json",
        "--tokens", "srcs/libs/test_kv_cache_dtype.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "1",
        "--report-json", "srcs/libs/test_kv_cache_dtype.json.out",
    };
    char *report_content;
    char precision_path[256] = { 0 };
    char kv_from_path[32] = { 0 };
    char storage_dtype[32] = { 0 };
    char *kv_cache;

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("kv cache dtype run", run_cli_capture(15, argv, stdout_path,
                                                     stderr_path), 0);

    report_content = read_file_content(report_path);
    if (report_content == NULL) {
        fprintf(stderr, "kv cache dtype: could not read report file\n");
        ++g_failures;
        goto cleanup;
    }

    if (!extract_json_string(report_content, "precision_path", precision_path,
                             sizeof(precision_path))) {
        fprintf(stderr, "kv cache dtype: precision_path not found in report\n");
        ++g_failures;
        free(report_content);
        goto cleanup;
    }

    if (!parse_kv_dtype_from_precision(precision_path, kv_from_path,
                                       sizeof(kv_from_path))) {
        fprintf(stderr, "kv cache dtype: could not parse kv=<dtype> from %s\n",
                precision_path);
        ++g_failures;
        free(report_content);
        goto cleanup;
    }

    kv_cache = extract_json_object(report_content, "kv_cache");
    if (kv_cache == NULL) {
        fprintf(stderr, "kv cache dtype: kv_cache object not found in report\n");
        ++g_failures;
        free(report_content);
        goto cleanup;
    }

    if (!extract_json_string(kv_cache, "storage_dtype", storage_dtype,
                             sizeof(storage_dtype))) {
        fprintf(stderr, "kv cache dtype: storage_dtype not found in report\n");
        ++g_failures;
        free(kv_cache);
        free(report_content);
        goto cleanup;
    }

    expect_string_equals("kv cache dtype matches storage_dtype",
                         storage_dtype, strlen(storage_dtype), kv_from_path);

    free(kv_cache);
    free(report_content);

cleanup:
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
}

/*
 * Test 3: report.kv_cache determinism.
 *
 * Run two equivalent CLI report generations with identical options, extract
 * the semantic kv_cache fields, and verify they are identical across runs.
 * Do NOT compare full artifacts — perf/timing/path-like fields may vary.
 */
static void test_kv_cache_semantics_kv_cache_determinism(void)
{
    const char *model_path = "srcs/libs/test_kv_cache_det.safetensors";
    const char *config_path = "srcs/libs/test_kv_cache_det.json";
    const char *token_path = "srcs/libs/test_kv_cache_det.txt";
    const char *report_a = "srcs/libs/test_kv_cache_det_a.json";
    const char *report_b = "srcs/libs/test_kv_cache_det_b.json";
    const char *stdout_a = "srcs/libs/test_kv_cache_det_a.out";
    const char *stderr_a = "srcs/libs/test_kv_cache_det_a.err";
    const char *stdout_b = "srcs/libs/test_kv_cache_det_b.out";
    const char *stderr_b = "srcs/libs/test_kv_cache_det_b.err";
    char *argv_a[] = {
        "lis",
        "--model", "srcs/libs/test_kv_cache_det.safetensors",
        "--config", "srcs/libs/test_kv_cache_det.json",
        "--tokens", "srcs/libs/test_kv_cache_det.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "2",
        "--report-json", "srcs/libs/test_kv_cache_det_a.json",
    };
    char *argv_b[] = {
        "lis",
        "--model", "srcs/libs/test_kv_cache_det.safetensors",
        "--config", "srcs/libs/test_kv_cache_det.json",
        "--tokens", "srcs/libs/test_kv_cache_det.txt",
        "--context", "4",
        "--batch", "2",
        "--generate", "2",
        "--report-json", "srcs/libs/test_kv_cache_det_b.json",
    };
    char *content_a;
    char *content_b;
    char *kv_cache_a;
    char *kv_cache_b;
    const char *semantic_keys[] = {
        "\"scope\":\"run_local\"",
        "\"policy\":{\"eviction_free\":true,\"monotonic_growth\":true,"
            "\"paging\":false,\"offload\":false,\"sliding_window\":false,"
            "\"prefix_reuse\":false}",
        "\"storage_dtype\":\"f32\"",
        "\"max_tokens\":4",
        "\"used_tokens\":4",
        "\"bytes_per_token\":64",
        "\"allocated_bytes\":256",
        "\"used_bytes\":256",
        "\"shape\":{\"layer_count\":1,\"batch_size\":2,"
            "\"kv_head_count\":1,\"head_dim\":4,\"element_size\":4}",
    };
    size_t ki;

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_a);
    remove(report_b);
    remove(stdout_a);
    remove(stderr_a);
    remove(stdout_b);
    remove(stderr_b);
    write_valid_fixtures(model_path, config_path, token_path);

    expect_int("kv cache determinism run A",
               run_cli_capture(15, argv_a, stdout_a, stderr_a), 0);
    expect_int("kv cache determinism run B",
               run_cli_capture(15, argv_b, stdout_b, stderr_b), 0);

    content_a = read_file_content(report_a);
    content_b = read_file_content(report_b);
    if (content_a == NULL || content_b == NULL) {
        fprintf(stderr, "kv cache determinism: could not read report files\n");
        ++g_failures;
        free(content_a);
        free(content_b);
        goto cleanup;
    }

    kv_cache_a = extract_json_object(content_a, "kv_cache");
    kv_cache_b = extract_json_object(content_b, "kv_cache");
    if (kv_cache_a == NULL || kv_cache_b == NULL) {
        fprintf(stderr,
                "kv cache determinism: could not extract kv_cache objects\n");
        ++g_failures;
        free(kv_cache_a);
        free(kv_cache_b);
        free(content_a);
        free(content_b);
        goto cleanup;
    }

    expect_string_equals("kv cache determinism kv_cache object", kv_cache_a,
                         strlen(kv_cache_a), kv_cache_b);

    for (ki = 0; ki < sizeof(semantic_keys) / sizeof(semantic_keys[0]); ++ki) {
        char msg[128];

        snprintf(msg, sizeof(msg),
                 "kv cache determinism field %zu present A", ki);
        if (strstr(kv_cache_a, semantic_keys[ki]) == NULL) {
            fprintf(stderr, "%s: missing \"%s\" in report A\n",
                    msg, semantic_keys[ki]);
            ++g_failures;
        }
        snprintf(msg, sizeof(msg),
                 "kv cache determinism field %zu present B", ki);
        if (strstr(kv_cache_b, semantic_keys[ki]) == NULL) {
            fprintf(stderr, "%s: missing \"%s\" in report B\n",
                    msg, semantic_keys[ki]);
            ++g_failures;
        }
    }

    free(kv_cache_a);
    free(kv_cache_b);
    free(content_a);
    free(content_b);

cleanup:
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_a);
    remove(report_b);
    remove(stdout_a);
    remove(stderr_a);
    remove(stdout_b);
    remove(stderr_b);
}

/*
 * Test 4: context_limit invariant — strengthened.
 *
 * Extend the existing context-limit artifact checks with additional
 * report.kv_cache assertions that verify the accounting formulas hold
 * even in the error/context-limit path.
 *
 * Existing coverage (in test_cli_report_json_context_limit):
 *   - execution_status == "error"
 *   - stop_reason == "context_limit"
 *   - selected_token_count == 1
 *   - emitted_token_count == 0
 *   - emitted_token_ids == []
 *   - used_tokens == 2
 *   - used_bytes == 128
 *
 * Additional assertions added here:
 *   - used_tokens does not exceed context capacity
 *   - used_bytes == bytes_per_token * used_tokens
 *   - allocated_bytes == bytes_per_token * max_tokens
 *   - max_tokens == configured context capacity
 */
static void test_kv_cache_semantics_context_limit_accounting(void)
{
    const char *model_path = "srcs/libs/test_kv_cache_ctx.safetensors";
    const char *config_path = "srcs/libs/test_kv_cache_ctx.json";
    const char *token_path = "srcs/libs/test_kv_cache_ctx.txt";
    const char *report_path = "srcs/libs/test_kv_cache_ctx.json.out";
    const char *stdout_path = "srcs/libs/test_kv_cache_ctx.out";
    const char *stderr_path = "srcs/libs/test_kv_cache_ctx.err";
    char *argv[] = {
        "lis",
        "--model", "srcs/libs/test_kv_cache_ctx.safetensors",
        "--config", "srcs/libs/test_kv_cache_ctx.json",
        "--tokens", "srcs/libs/test_kv_cache_ctx.txt",
        "--context", "2",
        "--batch", "2",
        "--generate", "1",
        "--report-json", "srcs/libs/test_kv_cache_ctx.json.out",
    };
    char *report_content;
    long max_tokens = 0;
    long used_tokens = 0;
    long bytes_per_token = 0;
    long allocated_bytes = 0;
    long used_bytes = 0;
    char *kv_cache;

    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
    write_valid_fixtures(model_path, config_path, token_path);
    expect_int("kv cache context run",
               run_cli_capture(15, argv, stdout_path, stderr_path), 1);

    report_content = read_file_content(report_path);
    if (report_content == NULL) {
        fprintf(stderr, "kv cache context: could not read report file\n");
        ++g_failures;
        goto cleanup;
    }

    kv_cache = extract_json_object(report_content, "kv_cache");
    if (kv_cache == NULL) {
        fprintf(stderr, "kv cache context: could not extract kv_cache object\n");
        ++g_failures;
        free(report_content);
        goto cleanup;
    }

    if (!extract_json_int(kv_cache, "max_tokens", &max_tokens) ||
        !extract_json_int(kv_cache, "used_tokens", &used_tokens) ||
        !extract_json_int(kv_cache, "bytes_per_token", &bytes_per_token) ||
        !extract_json_int(kv_cache, "allocated_bytes", &allocated_bytes) ||
        !extract_json_int(kv_cache, "used_bytes", &used_bytes)) {
        fprintf(stderr, "kv cache context: could not extract kv_cache fields\n");
        ++g_failures;
        free(kv_cache);
        free(report_content);
        goto cleanup;
    }

    expect_int("kv cache context max_tokens == configured", (int)max_tokens,
               2);
    expect_int("kv cache context used_tokens <= max_tokens",
               (int)used_tokens <= (int)max_tokens ? 1 : 0, 1);
    expect_int("kv cache context allocated_bytes", (int)allocated_bytes,
               (int)(bytes_per_token * max_tokens));
    expect_int("kv cache context used_bytes", (int)used_bytes,
               (int)(bytes_per_token * used_tokens));

    expect_int("kv cache context used_tokens unchanged after exhaustion",
               (int)used_tokens, 2);

    free(kv_cache);
    free(report_content);

cleanup:
    remove(model_path);
    remove(config_path);
    remove(token_path);
    remove(report_path);
    remove(stdout_path);
    remove(stderr_path);
}

/*
 * Test 5: Qwen3 dense smoke — local/manual only.
 *
 * Skip-gated by default. Enabling it requires both:
 *   LIS_QWEN3_SMOKE=1
 *   LIS_QWEN3_MODEL=/path/to/qwen3-dense
 *
 * Without LIS_QWEN3_SMOKE the test is a no-op pass and is not part of any
 * CI gate. With LIS_QWEN3_SMOKE=1 set but LIS_QWEN3_MODEL unset the test
 * fails fast rather than silently falling back to any local path.
 */
static void test_kv_cache_semantics_qwen3_smoke_skip_gated(void)
{
    const char *smoke_env = getenv("LIS_QWEN3_SMOKE");
    const char *model_dir;
    char config_path[512];
    char tokenizer_path[512];
    int n;

    if (smoke_env == NULL || strcmp(smoke_env, "1") != 0) {
        return;
    }

    model_dir = getenv("LIS_QWEN3_MODEL");
    if (model_dir == NULL || model_dir[0] == '\0') {
        fprintf(stderr,
                "kv cache qwen3 smoke: LIS_QWEN3_SMOKE=1 requires "
                "LIS_QWEN3_MODEL=/path/to/qwen3-dense\n");
        ++g_failures;
        return;
    }

    n = snprintf(config_path, sizeof(config_path),
                 "%s/config.json", model_dir);
    if (n < 0 || (size_t)n >= sizeof(config_path)) {
        fprintf(stderr,
                "kv cache qwen3 smoke: LIS_QWEN3_MODEL path too long\n");
        ++g_failures;
        return;
    }
    n = snprintf(tokenizer_path, sizeof(tokenizer_path),
                 "%s/tokenizer.json", model_dir);
    if (n < 0 || (size_t)n >= sizeof(tokenizer_path)) {
        fprintf(stderr,
                "kv cache qwen3 smoke: LIS_QWEN3_MODEL path too long\n");
        ++g_failures;
        return;
    }

    {
        const char *stdout_path = "srcs/libs/test_kv_cache_qwen_smoke.out";
        const char *stderr_path = "srcs/libs/test_kv_cache_qwen_smoke.err";
        const char *report_path = "srcs/libs/test_kv_cache_qwen_smoke.json";
        char *argv[] = {
            "lis",
            "--model", (char *)model_dir,
            "--config", config_path,
            "--hf-tokenizer", tokenizer_path,
            "--prompt", "hello",
            "--context", "128",
            "--batch", "1",
            "--generate", "4",
            "--report-json", "srcs/libs/test_kv_cache_qwen_smoke.json",
        };
        char *report_content;
        long used_tokens = 0;
        long max_tokens = 0;
        long bytes_per_token = 0;
        long allocated_bytes = 0;
        long used_bytes = 0;
        char *kv_cache;

        remove(stdout_path);
        remove(stderr_path);
        remove(report_path);

        expect_int("kv cache qwen3 smoke run",
                   run_cli_capture(17, argv, stdout_path, stderr_path), 0);

        report_content = read_file_content(report_path);
        if (report_content == NULL) {
            fprintf(stderr, "kv cache qwen3 smoke: could not read report\n");
            ++g_failures;
            goto smoke_cleanup;
        }

        kv_cache = extract_json_object(report_content, "kv_cache");
        if (kv_cache == NULL) {
            fprintf(stderr,
                    "kv cache qwen3 smoke: could not extract kv_cache object\n");
            ++g_failures;
            free(report_content);
            goto smoke_cleanup;
        }

        if (!extract_json_int(kv_cache, "used_tokens", &used_tokens) ||
            !extract_json_int(kv_cache, "max_tokens", &max_tokens) ||
            !extract_json_int(kv_cache, "bytes_per_token", &bytes_per_token) ||
            !extract_json_int(kv_cache, "allocated_bytes", &allocated_bytes) ||
            !extract_json_int(kv_cache, "used_bytes", &used_bytes)) {
            fprintf(stderr,
                    "kv cache qwen3 smoke: could not extract kv fields\n");
            ++g_failures;
            free(kv_cache);
            free(report_content);
            goto smoke_cleanup;
        }

        expect_int("kv cache qwen3 smoke used <= max",
                   (int)used_tokens <= (int)max_tokens ? 1 : 0, 1);
        expect_int("kv cache qwen3 smoke allocated", (int)allocated_bytes,
                   (int)(bytes_per_token * max_tokens));
        expect_int("kv cache qwen3 smoke used", (int)used_bytes,
                   (int)(bytes_per_token * used_tokens));
        expect_int("kv cache qwen3 smoke used gt 0",
                   (int)used_tokens > 0, 1);

        free(kv_cache);
        free(report_content);

    smoke_cleanup:
        remove(stdout_path);
        remove(stderr_path);
        remove(report_path);
    }
}
