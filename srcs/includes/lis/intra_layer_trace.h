#ifndef LIS_INTRA_LAYER_TRACE_H
#define LIS_INTRA_LAYER_TRACE_H

/*
 * Producer-side structural layer for the frozen Pass 4 intra-layer contract.
 *
 * This module owns the single authoritative 17-stage Llama taxonomy, a fixed
 * 17-slot allocation-free record for exactly one
 * (runtime_checkpoint_step, layer_index, token_position) target, the strict
 * structural validation that rejects malformed producer input, and the JSON
 * emission of the two additive blocks "intra_layer_checkpoint_layout" and
 * "intra_layer_trace".
 *
 * It carries a caller-supplied checkpoint digest and never computes one: no
 * entry point reads tensor elements, and lis_intra_layer_fp32_view is validated
 * for declared span coherence without ever dereferencing its data pointer. The
 * intra-layer digest stream and its computation are a separate work package.
 *
 * The include list below is deliberately minimal: it is the mechanism that
 * keeps runtime, loader, and CLI state out of this module.
 */

#include "lis/checkpoint_digest.h"
#include "lis/status.h"

#include <stddef.h>
#include <stdio.h>

#define LIS_INTRA_LAYER_STAGE_COUNT     17U
#define LIS_INTRA_LAYER_MAX_RANK         4U
#define LIS_INTRA_LAYER_IDENTIFIER_MAX 128U
#define LIS_INTRA_LAYER_DETAIL_MAX     256U

#define LIS_INTRA_LAYER_LAYOUT_NAME "llama_intra_layer_summary"
#define LIS_INTRA_LAYER_LAYOUT_VERSION 1U
#define LIS_INTRA_LAYER_STAGE_TAXONOMY "lis.llama.intra_layer_stages/v1"
#define LIS_INTRA_LAYER_MODEL_FAMILY "llama3_decoder"
#define LIS_INTRA_LAYER_PHASE_DECODE_NAME "decode"
#define LIS_INTRA_LAYER_ORDERING_SEMANTICS "runtime_step_layer_stage_ordinal"
#define LIS_INTRA_LAYER_DUPLICATE_POLICY "reject_artifact_before_write"

/*
 * Digest identity strings only. The canonical byte stream and the digest
 * function are owned by the digest work package, which must include this
 * header and must not redefine either macro.
 */
#define LIS_INTRA_LAYER_DIGEST_VERSION "lis.checkpoint.intra_layer.fp32le/v1"
#define LIS_INTRA_LAYER_DIGEST_DOMAIN_TAG "LIS_INTRA_LAYER_CHECKPOINT_DIGEST"

typedef enum {
    LIS_INTRA_LAYER_STAGE_LAYER_INPUT = 0,
    LIS_INTRA_LAYER_STAGE_ATTENTION_NORM_OUTPUT,
    LIS_INTRA_LAYER_STAGE_QUERY_PROJECTION_OUTPUT,
    LIS_INTRA_LAYER_STAGE_KEY_PROJECTION_OUTPUT,
    LIS_INTRA_LAYER_STAGE_VALUE_PROJECTION_OUTPUT,
    LIS_INTRA_LAYER_STAGE_ROPE_QUERY_OUTPUT,
    LIS_INTRA_LAYER_STAGE_ROPE_KEY_OUTPUT,
    LIS_INTRA_LAYER_STAGE_ATTENTION_SCORES,
    LIS_INTRA_LAYER_STAGE_ATTENTION_PROBABILITIES,
    LIS_INTRA_LAYER_STAGE_ATTENTION_CONTEXT,
    LIS_INTRA_LAYER_STAGE_ATTENTION_OUTPUT_PROJECTION,
    LIS_INTRA_LAYER_STAGE_POST_ATTENTION_RESIDUAL,
    LIS_INTRA_LAYER_STAGE_MLP_NORM_OUTPUT,
    LIS_INTRA_LAYER_STAGE_MLP_GATE_PROJECTION,
    LIS_INTRA_LAYER_STAGE_MLP_UP_PROJECTION,
    LIS_INTRA_LAYER_STAGE_MLP_GATED_ACTIVATION,
    LIS_INTRA_LAYER_STAGE_MLP_DOWN_PROJECTION
} lis_intra_layer_stage;

/*
 * One row of the single authoritative stage table. The enum carries no
 * strings; every stage string in the program originates here.
 */
typedef struct {
    size_t      stage_order;
    const char *stage_id;
    const char *tensor_role;   /* equals stage_id for v1 */
    const char *public_name;   /* presentation metadata; never parsed */
} lis_intra_layer_stage_info;

/*
 * Returns NULL for any stage_order outside the frozen taxonomy. The parameter
 * is size_t so that a negative or out-of-range enum cast wraps and is rejected
 * rather than indexing out of bounds.
 */
const lis_intra_layer_stage_info *
lis_intra_layer_stage_lookup(size_t stage_order);

/* Prefill is unrepresentable in v1; a zeroed struct is invalid, not decode. */
typedef enum {
    LIS_INTRA_LAYER_PHASE_INVALID = 0,
    LIS_INTRA_LAYER_PHASE_DECODE  = 1
} lis_intra_layer_phase;

/*
 * Missing-coordinate states. "captured" is deliberately absent: it is not a
 * missing state, mirroring the frozen model's rejection of it.
 */
typedef enum {
    LIS_INTRA_LAYER_MISSING_INVALID = 0,
    LIS_INTRA_LAYER_MISSING_NOT_CAPTURED,
    LIS_INTRA_LAYER_MISSING_UNSUPPORTED,
    LIS_INTRA_LAYER_MISSING_MALFORMED,
    LIS_INTRA_LAYER_MISSING_UNEXPECTEDLY_ABSENT
} lis_intra_layer_missing_state;

typedef enum {
    LIS_INTRA_LAYER_RECORD_UNINITIALIZED = 0,
    LIS_INTRA_LAYER_RECORD_ACTIVE,
    LIS_INTRA_LAYER_RECORD_READY,
    LIS_INTRA_LAYER_RECORD_INVALID   /* sticky */
} lis_intra_layer_record_state;

/*
 * A declared strided FP32 view. Structure only: the validator checks span
 * coherence and never dereferences data.
 */
typedef struct {
    const float *data;
    size_t rank;
    size_t shape[LIS_INTRA_LAYER_MAX_RANK];
    size_t element_strides[LIS_INTRA_LAYER_MAX_RANK];  /* in float elements */
    size_t logical_element_count;
    size_t physical_element_count;
} lis_intra_layer_fp32_view;

lis_status lis_intra_layer_fp32_view_validate(
    const lis_intra_layer_fp32_view *view);

/*
 * One captured observation. Pure value semantics: the record stores a copy, so
 * the caller may pass a stack temporary. Stage, role, public name, dtype, and
 * phase strings are never stored; they are looked up from the table or emitted
 * from the frozen macros above.
 *
 * The coordinate fields are supplied explicitly and validated against the
 * record's configured target. The redundancy is deliberate: it makes caller
 * confusion detectable instead of silently absorbed.
 */
typedef struct {
    lis_intra_layer_stage stage;
    lis_intra_layer_phase phase;

    size_t runtime_checkpoint_step;   /* frozen minimum 1 */
    size_t layer_index;
    size_t token_position;
    size_t batch_index;               /* frozen: 0 */
    size_t sequence_index;            /* frozen: 0 */
    size_t stage_order;               /* frozen: equals table order */
    size_t execution_ordinal;         /* frozen: equals stage_order */

    size_t rank;
    size_t shape[LIS_INTRA_LAYER_MAX_RANK];
    size_t element_count;

    float min;
    float max;
    float mean;
    float l2;
    int   nan;                        /* exactly 0 or 1 */
    int   inf;                        /* exactly 0 or 1 */

    lis_checkpoint_digest digest;     /* carried, never computed here */
} lis_intra_layer_observation;

/*
 * Fixed 17-slot record for one target. The slot array makes the
 * captured/missing partition and its canonical ordering structural rather than
 * checked: emission walks slots 0..16, so no sort exists that could silently
 * repair malformed input. Out-of-order arrival is still rejected independently
 * through last_resolved_stage_order.
 */
typedef struct lis_intra_layer_trace_record {
    lis_intra_layer_record_state state;

    size_t runtime_checkpoint_step;
    size_t target_layer;
    size_t total_layer_count;
    size_t token_position;
    char   precision_path[LIS_INTRA_LAYER_IDENTIFIER_MAX + 1U];

    struct {
        int                           occupied;
        int                           captured;
        lis_intra_layer_observation   observation;   /* valid iff captured */
        lis_intra_layer_missing_state missing_state; /* valid iff !captured */
        char detail[LIS_INTRA_LAYER_DETAIL_MAX + 1U];
    } slots[LIS_INTRA_LAYER_STAGE_COUNT];

    int    has_last_resolved;
    size_t last_resolved_stage_order;

    size_t captured_count;
    size_t missing_count;
    size_t digest_element_visits;
} lis_intra_layer_trace_record;

_Static_assert(sizeof(lis_intra_layer_trace_record) <= 16384U,
               "intra-layer record must stay a small fixed-size object");

/*
 * JSON primitives injected by the caller. The layer-trace writer passes its own
 * existing escaper and %.6g-or-null float writer, so the intra blocks are
 * byte-compatible with every other LIS writer without duplicating either
 * helper.
 */
typedef struct {
    void (*write_string)(FILE *fp, const char *text);  /* escaped JSON string */
    void (*write_float)(FILE *fp, float value);        /* %.6g, or null */
} lis_intra_layer_json_hooks;

lis_status lis_intra_layer_record_init(lis_intra_layer_trace_record *record);

lis_status lis_intra_layer_record_configure(
    lis_intra_layer_trace_record *record,
    size_t runtime_checkpoint_step,   /* >= 1 */
    size_t target_layer,              /* < total_layer_count */
    size_t total_layer_count,         /* > 0 */
    size_t token_position,
    const char *precision_path);      /* non-empty, bounded, no control bytes */

lis_status lis_intra_layer_record_append_observation(
    lis_intra_layer_trace_record *record,
    const lis_intra_layer_observation *observation);

lis_status lis_intra_layer_record_mark_unavailable(
    lis_intra_layer_trace_record *record,
    lis_intra_layer_stage stage,
    lis_intra_layer_missing_state state,
    const char *detail);              /* non-empty, bounded */

lis_status lis_intra_layer_record_finalize(
    lis_intra_layer_trace_record *record);

void lis_intra_layer_record_invalidate(
    lis_intra_layer_trace_record *record);   /* sticky, idempotent */

void lis_intra_layer_record_destroy(
    lis_intra_layer_trace_record *record);   /* memset; owns no heap */

lis_intra_layer_record_state lis_intra_layer_record_get_state(
    const lis_intra_layer_trace_record *record);

int lis_intra_layer_record_is_ready(
    const lis_intra_layer_trace_record *record);   /* 1 or 0; NULL -> 0 */

/*
 * Emits, with a leading comma:
 *   ,"intra_layer_checkpoint_layout":{...},"intra_layer_trace":[...]
 * Requires a READY record and both hook members non-NULL. Nothing is written
 * unless every precondition passes.
 */
lis_status lis_intra_layer_record_write_json(
    FILE *fp,
    const lis_intra_layer_trace_record *record,
    const lis_intra_layer_json_hooks *hooks);

#endif
