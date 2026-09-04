#include "lis/cli.h"

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void lis_cli_print_usage(FILE *stream, const char *program)
{
    fprintf(stream,
            "Usage: %s --model PATH --config PATH --context N --batch N "
            "--generate N\n"
            "       [--tokens PATH | --vocab PATH --prompt TEXT | "
            "--hf-tokenizer PATH --prompt TEXT]\n"
            "       [--threads N] [--diagnostics]\n"
            "       [--forced-prefix \"ID ...\"]\n"
            "       [--forced-prefix-binding-json PATH]\n"
            "       [--layer-checkpoints STEP]\n"
            "       [--intra-layer-checkpoints LAYER]\n"
            "       [--report-json PATH] [--report-md PATH]\n"
            "       [--trace-json PATH] [--layer-trace-json PATH]\n"
            "       [--perf] [--perf-per-token] [--perf-tag TAG]\n"
            "\n"
            "LIS v1 Local CPU Inference Engine (plain-RoPE Llama and "
            "Qwen3 Dense targets supported)\n"
            "\n"
            "Inputs:\n"
            "  --model PATH         HF supported decoder import dir or validation safetensors path\n"
            "  --context N          Configured max sequence context length\n"
            "  --batch N            Static batch size\n"
            "  --generate N         Limit token generation steps\n"
            "  --threads N          CPU thread count (default: 1)\n"
            "  --diagnostics        Print opt-in generation diagnostics to stderr\n"
            "  --forced-prefix \"ID ...\"  Forced token IDs for diagnostics comparison\n"
            "  --forced-prefix-binding-json PATH  Source binding for a forced-prefix run report\n"
            "  --layer-checkpoints STEP    Print layer checkpoint stats at STEP (0=prefill)\n"
            "  --intra-layer-checkpoints LAYER  Capture the fixed semantic intra-layer profile\n"
            "  --layer-trace-json PATH  Write a compact per-layer/per-stage trace artifact\n"
            "  --report-json PATH   Write a versioned machine-readable execution artifact\n"
            "  --report-md PATH   Write a bounded human-readable Markdown execution report\n"
            "  --trace-json PATH    Write a bounded decode-step trace artifact\n"
            "\n"
            "Performance Measurement:\n"
            "  --perf               Emit per-stage timings and summary to stderr\n"
            "  --perf-per-token     Also emit one line per decode step (implies --perf)\n"
            "  --perf-tag TAG       Label echoed into every lis: perf-* line (default: none)\n"
            "\n"
            "Tokenizer & Prompt Mode (Choose One):\n"
            "  --tokens PATH                       Direct LIS token-ID input\n"
            "  --vocab PATH --prompt TEXT          LIS BPE vocab/merge binary input\n"
            "  --hf-tokenizer PATH --prompt TEXT   HuggingFace BPE tokenizer.json import\n",
            program);
}

static lis_status lis_cli_parse_size(const char *text, size_t *out)
{
    char *end = NULL;
    uintmax_t value = 0;

    if (text == NULL || *text == '\0' || out == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    errno = 0;
    value = strtoumax(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        value == 0 || value > (uintmax_t)SIZE_MAX) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    *out = (size_t)value;
    return LIS_STATUS_OK;
}

static lis_status lis_cli_parse_step(const char *text, size_t *out)
{
    char *end = NULL;
    uintmax_t value = 0;

    if (text == NULL || *text == '\0' || out == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    errno = 0;
    value = strtoumax(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' ||
        value > (uintmax_t)SIZE_MAX) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    *out = (size_t)value;
    return LIS_STATUS_OK;
}

static lis_status lis_cli_require_value(int argc, char **argv, int *index,
                                        const char **out_value)
{
    if (index == NULL || out_value == NULL || *index + 1 >= argc) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (argv[*index + 1][0] == '-') {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    ++(*index);
    *out_value = argv[*index];
    return LIS_STATUS_OK;
}

static lis_status lis_cli_parse_options(int argc, char **argv,
                                        lis_cli_options *out_options,
                                        int *out_help_requested)
{
    lis_cli_options options = { 0 };
    int index;

    if (argv == NULL || out_options == NULL || out_help_requested == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    *out_help_requested = 0;
    if (argc == 1) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    for (index = 1; index < argc; ++index) {
        const char *arg = argv[index];

        if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            *out_help_requested = 1;
            return LIS_STATUS_OK;
        }
        if (strcmp(arg, "--model") == 0) {
            if (lis_cli_require_value(argc, argv, &index,
                                      &options.model_path) != LIS_STATUS_OK) {
                return LIS_STATUS_INVALID_ARGUMENT;
            }
        } else if (strcmp(arg, "--config") == 0) {
            if (lis_cli_require_value(argc, argv, &index,
                                      &options.config_path) != LIS_STATUS_OK) {
                return LIS_STATUS_INVALID_ARGUMENT;
            }
        } else if (strcmp(arg, "--tokens") == 0) {
            if (lis_cli_require_value(argc, argv, &index,
                                      &options.token_path) != LIS_STATUS_OK) {
                return LIS_STATUS_INVALID_ARGUMENT;
            }
        } else if (strcmp(arg, "--vocab") == 0) {
            if (lis_cli_require_value(argc, argv, &index,
                                      &options.vocab_path) != LIS_STATUS_OK) {
                return LIS_STATUS_INVALID_ARGUMENT;
            }
        } else if (strcmp(arg, "--hf-tokenizer") == 0) {
            if (lis_cli_require_value(argc, argv, &index,
                                      &options.hf_tokenizer_path) != LIS_STATUS_OK) {
                return LIS_STATUS_INVALID_ARGUMENT;
            }
        } else if (strcmp(arg, "--prompt") == 0) {
            if (lis_cli_require_value(argc, argv, &index,
                                      &options.prompt_text) != LIS_STATUS_OK) {
                return LIS_STATUS_INVALID_ARGUMENT;
            }
        } else if (strcmp(arg, "--report-json") == 0) {
            if (lis_cli_require_value(argc, argv, &index,
                                      &options.report_json_path) != LIS_STATUS_OK) {
                return LIS_STATUS_INVALID_ARGUMENT;
            }
        } else if (strcmp(arg, "--report-md") == 0) {
            if (lis_cli_require_value(argc, argv, &index,
                                      &options.report_md_path) != LIS_STATUS_OK) {
                return LIS_STATUS_INVALID_ARGUMENT;
            }
        } else if (strcmp(arg, "--context") == 0) {
            const char *value = NULL;

            if (lis_cli_require_value(argc, argv, &index, &value) != LIS_STATUS_OK ||
                lis_cli_parse_size(value, &options.context_length) != LIS_STATUS_OK) {
                return LIS_STATUS_INVALID_ARGUMENT;
            }
        } else if (strcmp(arg, "--batch") == 0) {
            const char *value = NULL;

            if (lis_cli_require_value(argc, argv, &index, &value) != LIS_STATUS_OK ||
                lis_cli_parse_size(value, &options.batch_size) != LIS_STATUS_OK) {
                return LIS_STATUS_INVALID_ARGUMENT;
            }
        } else if (strcmp(arg, "--generate") == 0) {
            const char *value = NULL;

            if (lis_cli_require_value(argc, argv, &index, &value) != LIS_STATUS_OK ||
                lis_cli_parse_size(value, &options.generation_limit) != LIS_STATUS_OK) {
                return LIS_STATUS_INVALID_ARGUMENT;
            }
        } else if (strcmp(arg, "--threads") == 0) {
            const char *value = NULL;

            if (lis_cli_require_value(argc, argv, &index, &value) != LIS_STATUS_OK ||
                lis_cli_parse_size(value, &options.thread_count) != LIS_STATUS_OK) {
                return LIS_STATUS_INVALID_ARGUMENT;
            }
        } else if (strcmp(arg, "--diagnostics") == 0) {
            options.diagnostics_enabled = 1;
        } else if (strcmp(arg, "--layer-checkpoints") == 0) {
            const char *value = NULL;

            if (lis_cli_require_value(argc, argv, &index, &value) != LIS_STATUS_OK ||
                lis_cli_parse_step(value, &options.layer_checkpoints_step) != LIS_STATUS_OK) {
                return LIS_STATUS_INVALID_ARGUMENT;
            }
            options.layer_checkpoints_enabled = 1;
        } else if (strcmp(arg, "--intra-layer-checkpoints") == 0) {
            const char *value = NULL;

            if (lis_cli_require_value(argc, argv, &index, &value) !=
                    LIS_STATUS_OK ||
                lis_cli_parse_step(value,
                                   &options.intra_layer_target_layer) !=
                    LIS_STATUS_OK) {
                return LIS_STATUS_INVALID_ARGUMENT;
            }
            options.intra_layer_checkpoints_enabled = 1;
        } else if (strcmp(arg, "--forced-prefix") == 0) {
            if (lis_cli_require_value(argc, argv, &index,
                                      &options.forced_prefix_text) != LIS_STATUS_OK) {
                return LIS_STATUS_INVALID_ARGUMENT;
            }
        } else if (strcmp(arg, "--forced-prefix-binding-json") == 0) {
            if (lis_cli_require_value(
                    argc, argv, &index,
                    &options.forced_prefix_binding_json_path) !=
                LIS_STATUS_OK) {
                return LIS_STATUS_INVALID_ARGUMENT;
            }
        } else if (strcmp(arg, "--perf") == 0) {
            options.perf_enabled = 1;
        } else if (strcmp(arg, "--perf-per-token") == 0) {
            options.perf_enabled = 1;
            options.perf_per_token_enabled = 1;
        } else if (strcmp(arg, "--perf-tag") == 0) {
            if (lis_cli_require_value(argc, argv, &index,
                                      &options.perf_tag) != LIS_STATUS_OK) {
                return LIS_STATUS_INVALID_ARGUMENT;
            }
        } else if (strcmp(arg, "--trace-json") == 0) {
            if (lis_cli_require_value(argc, argv, &index,
                                      &options.trace_json_path) != LIS_STATUS_OK) {
                return LIS_STATUS_INVALID_ARGUMENT;
            }
        } else if (strcmp(arg, "--layer-trace-json") == 0) {
            if (lis_cli_require_value(argc, argv, &index,
                                      &options.layer_trace_json_path) != LIS_STATUS_OK) {
                return LIS_STATUS_INVALID_ARGUMENT;
            }
        } else {
            return LIS_STATUS_INVALID_ARGUMENT;
        }
    }

    if (options.thread_count == 0) {
        options.thread_count = 1;
    }
    if (options.model_path == NULL || options.config_path == NULL ||
        options.context_length == 0 || options.batch_size == 0 ||
        options.generation_limit == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    /* Exactly one input mode must be selected. */
    {
        int mode_count = 0;

        if (options.token_path != NULL) {
            ++mode_count;
        }
        if (options.vocab_path != NULL) {
            ++mode_count;
        }
        if (options.hf_tokenizer_path != NULL) {
            ++mode_count;
        }
        if (mode_count != 1) {
            return LIS_STATUS_INVALID_ARGUMENT;
        }
    }
    if (options.vocab_path != NULL && options.prompt_text == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (options.hf_tokenizer_path != NULL && options.prompt_text == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (options.prompt_text != NULL && options.vocab_path == NULL &&
        options.hf_tokenizer_path == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (options.forced_prefix_binding_json_path != NULL &&
        (options.forced_prefix_text == NULL ||
         options.report_json_path == NULL)) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (options.forced_prefix_text != NULL &&
        !options.diagnostics_enabled &&
        options.forced_prefix_binding_json_path == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    *out_options = options;
    return LIS_STATUS_OK;
}

int lis_cli_run(int argc, char **argv)
{
    lis_cli_options options = { 0 };
    lis_status status;
    int help_requested = 0;
    const char *program = "lis";

    if (argc > 0 && argv != NULL && argv[0] != NULL) {
        program = argv[0];
    }

    status = lis_cli_parse_options(argc, argv, &options, &help_requested);
    if (help_requested) {
        lis_cli_print_usage(stdout, program);
        return 0;
    }
    if (status != LIS_STATUS_OK) {
        fprintf(stderr, "lis: usage error: invalid arguments\n");
        lis_cli_print_usage(stderr, program);
        return 2;
    }

    status = lis_cli_run_inference(&options);
    if (status != LIS_STATUS_OK) {
        return 1;
    }

    return 0;
}
