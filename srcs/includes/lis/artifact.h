#ifndef LIS_ARTIFACT_H
#define LIS_ARTIFACT_H

#include "lis/cli.h"
#include "lis/loader.h"
#include "lis/model.h"
#include "lis/perf.h"
#include "lis/status.h"

#include <stddef.h>
#include <stdint.h>

#define LIS_ARTIFACT_SCHEMA "lis.execution_artifact/v1"
#define LIS_ARTIFACT_DIGEST_HEX_LEN 16
#define LIS_ARTIFACT_SET_ID_RANDOM_BYTES 16
#define LIS_ARTIFACT_SET_ID_HEX_LEN 32
#define LIS_ARTIFACT_SET_ID_PREFIX "aset1:"
#define LIS_ARTIFACT_SET_ID_LEN 38
#define LIS_INTRA_LAYER_DIAGNOSTIC_CAPTURE_PROFILE \
    "semantic_layer_and_intra_v1"
#define LIS_FORCED_PREFIX_MAX_TOKENS 64U
#define LIS_SHA256_ID_HEX_LEN 64U
#define LIS_SHA256_ID_TEXT_LEN 71U
#define LIS_FORCED_PREFIX_MODE "injected_selected_token_prefix_v1"
#define LIS_SELECTION_POLICY_RAW_GREEDY "raw_greedy"
#define LIS_SELECTION_POLICY_MODIFIED_GREEDY \
    "lis_policy_modified_greedy_v1"
#define LIS_SELECTION_POLICY_RAW_GREEDY_SHA256 \
    "sha256:fde534e94802f32b2aa573c90294fdaacb2bf2dd36e44b0e0e070d8edfd0724d"
#define LIS_SELECTION_POLICY_MODIFIED_GREEDY_SHA256 \
    "sha256:63f64c98586bc3cf31bbcccda5f6f354faba9ba47675780e77608309f7c912d0"

typedef struct {
    char value[LIS_ARTIFACT_SET_ID_LEN + 1U];
    int valid;
} lis_artifact_set_id;

typedef lis_status (*lis_artifact_random_source_fn)(
    void *context,
    unsigned char *buffer,
    size_t size);

typedef struct {
    uint64_t digest;
    size_t size_bytes;
    int valid;
} lis_artifact_fingerprint;

typedef enum {
    LIS_ARTIFACT_INPUT_MODE_TOKENS = 0,
    LIS_ARTIFACT_INPUT_MODE_VOCAB_PROMPT,
    LIS_ARTIFACT_INPUT_MODE_HF_TOKENIZER_PROMPT
} lis_artifact_input_mode;

typedef enum {
    LIS_ARTIFACT_OUTPUT_MODE_TOKEN_IDS = 0,
    LIS_ARTIFACT_OUTPUT_MODE_TEXT
} lis_artifact_output_mode;

typedef struct {
    size_t token_count;
    lis_artifact_fingerprint token_id_digest;
} lis_artifact_prompt_sequence;

typedef struct {
    int valid;
    lis_dtype storage_dtype;
    size_t max_tokens;
    size_t used_tokens;
    size_t bytes_per_token;
    size_t allocated_bytes;
    size_t used_bytes;
    size_t layer_count;
    size_t batch_size;
    size_t kv_head_count;
    size_t head_dim;
    size_t element_size;
} lis_artifact_kv_cache_report;

typedef struct {
    int valid;
    const char *mode;
    int applied;
    size_t token_count;
    char token_ids_sha256[LIS_SHA256_ID_TEXT_LEN + 1U];
    size_t prefix_start_generated_step;
    size_t prefix_end_generated_step_exclusive;
    size_t target_generated_token_step;
    size_t runtime_checkpoint_step;
    size_t prompt_token_count;
    size_t context_position;
    const char *selection_policy;
    const char *selection_policy_sha256;
    char source_pass0_artifact_sha256[LIS_SHA256_ID_TEXT_LEN + 1U];
    char source_original_run_report_sha256[LIS_SHA256_ID_TEXT_LEN + 1U];
    char source_pass1_artifact_sha256[LIS_SHA256_ID_TEXT_LEN + 1U];
    char source_localization_ref_sha256[LIS_SHA256_ID_TEXT_LEN + 1U];
} lis_artifact_forced_prefix_report;

typedef struct {
    const char *path;
    const lis_artifact_set_id *artifact_set_id;
    const char *model_format_name;
    const char *model_family_name;
    const char *backend_name;
    const char *stop_reason_name;
    const char *precision_path;
    const lis_cli_options *options;
    const lis_loaded_model *model;
    lis_artifact_input_mode input_mode;
    lis_artifact_output_mode output_mode;
    lis_artifact_fingerprint binary_fingerprint;
    lis_artifact_fingerprint model_fingerprint;
    lis_artifact_fingerprint config_fingerprint;
    lis_artifact_fingerprint input_fingerprint;
    lis_artifact_fingerprint runtime_fingerprint;
    lis_artifact_fingerprint backend_fingerprint;
    const lis_artifact_prompt_sequence *prompt_sequences;
    size_t prompt_sequence_count;
    const size_t *selected_token_ids;
    size_t selected_token_count;
    lis_artifact_fingerprint selected_token_digest;
    const size_t *emitted_token_ids;
    size_t emitted_token_count;
    lis_artifact_fingerprint emitted_token_digest;
    lis_status status;
    const lis_perf_report *perf;
    lis_artifact_kv_cache_report kv_cache;
    const lis_artifact_forced_prefix_report *forced_prefix;
} lis_artifact_run_report;

const char *lis_artifact_input_mode_name(lis_artifact_input_mode mode);
const char *lis_artifact_output_mode_name(lis_artifact_output_mode mode);

void lis_artifact_digest_hex(const lis_artifact_fingerprint *fingerprint,
                             char out_hex[LIS_ARTIFACT_DIGEST_HEX_LEN + 1U]);

lis_status lis_artifact_set_id_generate(lis_artifact_set_id *out);
lis_status lis_artifact_set_id_generate_with_source(
    lis_artifact_set_id *out,
    lis_artifact_random_source_fn source,
    void *source_context);

lis_status lis_artifact_fingerprint_file(const char *path,
                                         lis_artifact_fingerprint *out);
lis_status lis_artifact_fingerprint_current_binary(lis_artifact_fingerprint *out);
lis_status lis_artifact_fingerprint_token_ids(const size_t *ids,
                                              size_t count,
                                              lis_artifact_fingerprint *out);
lis_status lis_artifact_fingerprint_runtime(
    const lis_cli_options *options,
    lis_model_format model_format,
    lis_model_family family,
    lis_artifact_input_mode input_mode,
    const char *backend_name,
    lis_artifact_fingerprint *out);
lis_status lis_artifact_fingerprint_backend(const char *backend_name,
                                            size_t thread_count,
                                            lis_artifact_fingerprint *out);
lis_status lis_artifact_write_run_report(
    const lis_artifact_run_report *report);
lis_status lis_artifact_write_run_report_md(
    const lis_artifact_run_report *report);

#endif
