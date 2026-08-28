#ifndef LIS_LAYER_TRACE_H
#define LIS_LAYER_TRACE_H

#include "lis/artifact.h"
#include "lis/checkpoint_digest.h"
#include "lis/cli.h"
#include "lis/loader.h"
#include "lis/model.h"
#include "lis/status.h"

#include <stddef.h>

#define LIS_LAYER_TRACE_MAX_RANK         4
#define LIS_LAYER_TRACE_NAME_MAX         96
#define LIS_LAYER_TRACE_PHASE_MAX        16
#define LIS_LAYER_TRACE_INITIAL_CAPACITY 64
#define LIS_LAYER_TRACE_HARD_MAX         8192
#define LIS_LAYER_TRACE_ROLE_MAX         32
#define LIS_LAYER_TRACE_DTYPE_MAX        16

typedef struct {
    size_t step;
    char   phase[LIS_LAYER_TRACE_PHASE_MAX];
    char   name[LIS_LAYER_TRACE_NAME_MAX];
    size_t shape[LIS_LAYER_TRACE_MAX_RANK];
    size_t rank;
    float  min;
    float  max;
    float  mean;
    float  l2;
    int    nan;   /* 0 or 1 */
    int    inf;   /* 0 or 1 */
    int    has_checkpoint_coordinate;
    size_t runtime_checkpoint_step;
    size_t layer_index;
    char   tensor_role[LIS_LAYER_TRACE_ROLE_MAX];
    size_t batch_index;
    size_t sequence_index;
    size_t stage_order;
    size_t execution_ordinal;
    char   observed_dtype[LIS_LAYER_TRACE_DTYPE_MAX];
    size_t element_count;
    lis_checkpoint_digest digest;
} lis_layer_trace_step;

typedef struct {
    lis_layer_trace_step *steps;
    size_t step_count;
    size_t step_capacity;
    int    append_failed;   /* sticky: alloc, hard-cap, or string-truncation */
    int    checkpoint_layout_supported;
    size_t layout_runtime_checkpoint_step;
    size_t total_layer_count;
    size_t layer_output_count;
    size_t digest_element_visits;
} lis_layer_trace_record;

/* Optional Pass 4 producer record; declared, never dereferenced, here. */
struct lis_intra_layer_trace_record;

typedef struct {
    const char *path;
    const lis_artifact_set_id *artifact_set_id;
    const char *model_format_name;
    const char *model_family_name;
    const char *backend_name;
    const char *precision_path;
    const lis_cli_options *options;
    const lis_loaded_model *model;
    lis_artifact_input_mode  input_mode;
    lis_artifact_output_mode output_mode;
    lis_artifact_fingerprint binary_fingerprint;
    lis_artifact_fingerprint model_fingerprint;
    lis_artifact_fingerprint config_fingerprint;
    lis_artifact_fingerprint input_fingerprint;
    lis_artifact_fingerprint runtime_fingerprint;
    lis_artifact_fingerprint backend_fingerprint;
    /*
     * Optional, borrowed, and NULL by default. When NULL the emitted artifact
     * bytes are identical to a build without this field. The record's storage,
     * lifetime, and lifecycle belong entirely to the caller; the writer never
     * mutates or frees it.
     */
    const struct lis_intra_layer_trace_record *intra_layer_record;
} lis_layer_trace_artifact;

lis_status lis_layer_trace_record_init(lis_layer_trace_record *record,
                                       size_t initial_capacity);
void       lis_layer_trace_record_destroy(lis_layer_trace_record *record);
lis_status lis_layer_trace_record_append(lis_layer_trace_record *record,
                                         const lis_layer_trace_step *step);
lis_status lis_layer_trace_record_configure_llama_layout(
    lis_layer_trace_record *record,
    size_t runtime_checkpoint_step,
    size_t total_layer_count);
int lis_layer_trace_layout_selects_layer(size_t layer_index,
                                         size_t total_layer_count);
lis_status lis_layer_trace_step_set_layer_output(
    lis_layer_trace_step *step,
    size_t layer_index,
    const float *data,
    size_t element_count);
lis_status lis_layer_trace_artifact_write(const lis_layer_trace_artifact *artifact,
                                          const lis_layer_trace_record *record);

#endif
