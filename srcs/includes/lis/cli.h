#ifndef LIS_CLI_H
#define LIS_CLI_H

#include "lis/status.h"

#include <stddef.h>

typedef struct {
    const char *model_path;
    const char *config_path;
    const char *token_path;
    const char *vocab_path;
    const char *hf_tokenizer_path;
    const char *report_json_path;
    const char *report_md_path;
    const char *prompt_text;
    size_t context_length;
    size_t batch_size;
    size_t generation_limit;
    size_t thread_count;
    int diagnostics_enabled;
    const char *forced_prefix_text;
    int layer_checkpoints_enabled;
    size_t layer_checkpoints_step;
    int perf_enabled;
    int perf_per_token_enabled;
    const char *perf_tag;
    const char *trace_json_path;
    const char *layer_trace_json_path;
} lis_cli_options;

/*
 * Owns the minimal local executable boundary. Keeps this surface
 * deliberately narrow: local safetensors validation path or supported
 * HuggingFace decoder directory, explicit config path, direct token-ID or
 * bounded tokenizer-backed prompt input, context length, static batch size,
 * and greedy generation limit.
 */
int lis_cli_run(int argc, char **argv);
lis_status lis_cli_run_inference(const lis_cli_options *options);

/*
 * Builds the narrow canonical prompt for the currently supported single
 * user-prompt Llama Instruct path:
 *
 * <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n
 * {user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
 *
 * Caller owns *out_prompt and must free() it.
 */
lis_status lis_cli_build_llama_instruct_prompt(const char *user_text,
                                               char **out_prompt,
                                               size_t *out_len);

#endif
