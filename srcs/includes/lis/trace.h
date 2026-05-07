#ifndef LIS_TRACE_H
#define LIS_TRACE_H

#include "lis/artifact.h"
#include "lis/cli.h"
#include "lis/loader.h"
#include "lis/model.h"
#include "lis/status.h"

#include <stddef.h>
#include <stdint.h>

#define LIS_TRACE_TOPK_SIZE 5

typedef enum {
    LIS_TRACE_PHASE_PREFILL_SEED = 0,
    LIS_TRACE_PHASE_FIRST_DECODE,
    LIS_TRACE_PHASE_DECODE
} lis_trace_phase;

typedef struct {
    size_t token_id;
    float raw_score;
    float adjusted_score;
    int is_selected;
} lis_trace_topk_entry;

typedef struct {
    size_t step;
    lis_trace_phase phase;
    size_t selected_token_id;
    float raw_score_selected;
    float adjusted_score_selected;
    size_t runner_up_token_id;
    float runner_up_adjusted_score;
    int runner_up_available;
    float decision_margin;
    int decision_margin_valid;
    int structural_suppression_affected;
    int repetition_penalty_changed_selection;
    int selected_token_penalized;
    size_t suppressed_token_count;
    size_t penalized_token_count;
    const char *decision_class;
    lis_trace_topk_entry topk[LIS_TRACE_TOPK_SIZE];
    size_t topk_count;
    int has_stop_reason;
    const char *stop_reason;
} lis_trace_step;

typedef struct {
    lis_trace_step *steps;
    size_t step_count;
    size_t step_capacity;
} lis_trace_record;

typedef struct {
    const char *path;
    const char *model_format_name;
    const char *model_family_name;
    const char *backend_name;
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
} lis_trace_artifact;

lis_status lis_trace_record_init(lis_trace_record *record, size_t capacity);
void lis_trace_record_destroy(lis_trace_record *record);
lis_status lis_trace_record_append(lis_trace_record *record,
                                   const lis_trace_step *step);

lis_status lis_trace_artifact_write(const lis_trace_artifact *artifact,
                                    const lis_trace_record *record);

const char *lis_trace_phase_name(lis_trace_phase phase);

#endif