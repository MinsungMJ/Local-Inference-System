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
