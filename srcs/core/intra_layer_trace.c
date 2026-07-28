#include "lis/intra_layer_trace.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

/*
 * The single authoritative stage table. Orders are exactly 0..16 and
 * stage_id equals tensor_role for v1, matching the frozen contract fixture.
 * The inherited parent boundary role is not a local stage and therefore
 * appears nowhere in this table.
 */
static const lis_intra_layer_stage_info lis_intra_layer_stage_table[] = {
    {  0U, "layer_input", "layer_input",
       "Layer input" },
    {  1U, "attention_norm_output", "attention_norm_output",
       "Pre-attention RMSNorm output" },
    {  2U, "query_projection_output", "query_projection_output",
       "Q projection output" },
    {  3U, "key_projection_output", "key_projection_output",
       "K projection output" },
    {  4U, "value_projection_output", "value_projection_output",
       "V projection output" },
    {  5U, "rope_query_output", "rope_query_output",
       "RoPE-applied Q" },
    {  6U, "rope_key_output", "rope_key_output",
       "RoPE-applied K" },
    {  7U, "attention_scores", "attention_scores",
       "Attention pre-softmax scores" },
    {  8U, "attention_probabilities", "attention_probabilities",
       "Attention softmax output" },
    {  9U, "attention_context", "attention_context",
       "Attention context" },
    { 10U, "attention_output_projection", "attention_output_projection",
       "Attention output projection" },
    { 11U, "post_attention_residual", "post_attention_residual",
       "Post-attention residual" },
    { 12U, "mlp_norm_output", "mlp_norm_output",
       "Pre-MLP RMSNorm output" },
    { 13U, "mlp_gate_projection", "mlp_gate_projection",
       "MLP gate projection" },
    { 14U, "mlp_up_projection", "mlp_up_projection",
       "MLP up projection" },
    { 15U, "mlp_gated_activation", "mlp_gated_activation",
       "MLP gated activation" },
    { 16U, "mlp_down_projection", "mlp_down_projection",
       "MLP down projection" }
};

_Static_assert(sizeof(lis_intra_layer_stage_table) /
                   sizeof(lis_intra_layer_stage_table[0]) ==
                   LIS_INTRA_LAYER_STAGE_COUNT,
               "intra-layer stage table must contain exactly 17 frozen stages");

_Static_assert(sizeof(size_t) <= sizeof(uint64_t),
               "intra-layer coordinates must convert losslessly to uint64");

/*
 * Repository convention is a file-static checked multiply per translation unit;
 * there is no shared checked-arithmetic header.
 */
static int lis_intra_layer_mul_overflows(size_t left, size_t right)
{
    return left != 0U && right > SIZE_MAX / left;
}

const lis_intra_layer_stage_info *
lis_intra_layer_stage_lookup(size_t stage_order)
{
    if (stage_order >= LIS_INTRA_LAYER_STAGE_COUNT) {
        return NULL;
    }
    return &lis_intra_layer_stage_table[stage_order];
}

/*
 * Bounded identifier/detail validation. Rejects NULL, empty, oversized, and
 * control-byte input. Escaping at emission time is a second line of defence,
 * not the only one.
 */
static lis_status lis_intra_layer_validate_text(const char *text,
                                                size_t max_bytes)
{
    size_t index;

    if (text == NULL || text[0] == '\0') {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    for (index = 0; text[index] != '\0'; ++index) {
        unsigned char ch = (unsigned char)text[index];

        if (index >= max_bytes) {
            return LIS_STATUS_FORMAT;
        }
        if (ch < 0x20U || ch == 0x7FU) {
            return LIS_STATUS_INVALID_ARGUMENT;
        }
    }
    return LIS_STATUS_OK;
}

lis_status lis_intra_layer_fp32_view_validate(
    const lis_intra_layer_fp32_view *view)
{
    size_t index;
    size_t product = 1U;
    size_t max_offset = 0U;

    if (view == NULL || view->data == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (view->rank == 0U || view->rank > LIS_INTRA_LAYER_MAX_RANK) {
        return LIS_STATUS_UNSUPPORTED_SHAPE;
    }
    for (index = 0; index < view->rank; ++index) {
        if (view->shape[index] == 0U) {
            return LIS_STATUS_UNSUPPORTED_SHAPE;
        }
        if (lis_intra_layer_mul_overflows(product, view->shape[index])) {
            return LIS_STATUS_OVERFLOW;
        }
        product *= view->shape[index];
    }
    if (product != view->logical_element_count) {
        return LIS_STATUS_SHAPE_MISMATCH;
    }
    for (index = 0; index < view->rank; ++index) {
        size_t extent;
        size_t contribution;

        if (view->element_strides[index] == 0U) {
            return LIS_STATUS_INVALID_ARGUMENT;
        }
        extent = view->shape[index] - 1U;
        if (lis_intra_layer_mul_overflows(extent,
                                          view->element_strides[index])) {
            return LIS_STATUS_OVERFLOW;
        }
        contribution = extent * view->element_strides[index];
        if (max_offset > SIZE_MAX - contribution) {
            return LIS_STATUS_OVERFLOW;
        }
        max_offset += contribution;
    }
    if (view->physical_element_count == 0U ||
        max_offset >= view->physical_element_count) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    return LIS_STATUS_OK;
}

lis_status lis_intra_layer_record_init(lis_intra_layer_trace_record *record)
{
    if (record == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    memset(record, 0, sizeof(*record));
    record->state = LIS_INTRA_LAYER_RECORD_UNINITIALIZED;
    return LIS_STATUS_OK;
}

void lis_intra_layer_record_invalidate(lis_intra_layer_trace_record *record)
{
    if (record == NULL) {
        return;
    }
    record->state = LIS_INTRA_LAYER_RECORD_INVALID;
}

void lis_intra_layer_record_destroy(lis_intra_layer_trace_record *record)
{
    if (record == NULL) {
        return;
    }
    memset(record, 0, sizeof(*record));
}

lis_intra_layer_record_state lis_intra_layer_record_get_state(
    const lis_intra_layer_trace_record *record)
{
    if (record == NULL) {
        return LIS_INTRA_LAYER_RECORD_INVALID;
    }
    return record->state;
}

int lis_intra_layer_record_is_ready(
    const lis_intra_layer_trace_record *record)
{
    return record != NULL && record->state == LIS_INTRA_LAYER_RECORD_READY;
}

lis_status lis_intra_layer_record_configure(
    lis_intra_layer_trace_record *record,
    size_t runtime_checkpoint_step,
    size_t target_layer,
    size_t total_layer_count,
    size_t token_position,
    const char *precision_path)
{
    lis_status status;
    size_t length;

    if (record == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (record->state != LIS_INTRA_LAYER_RECORD_UNINITIALIZED) {
        return LIS_STATUS_BAD_STATE;
    }
    if (runtime_checkpoint_step < 1U) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (total_layer_count == 0U || target_layer >= total_layer_count) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    status = lis_intra_layer_validate_text(precision_path,
                                           LIS_INTRA_LAYER_IDENTIFIER_MAX);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    /* Nothing above mutates the record, so a rejected configure leaves it
     * exactly as it was: UNINITIALIZED and re-configurable. */
    length = strlen(precision_path);
    memcpy(record->precision_path, precision_path, length);
    record->precision_path[length] = '\0';
    record->runtime_checkpoint_step = runtime_checkpoint_step;
    record->target_layer = target_layer;
    record->total_layer_count = total_layer_count;
    record->token_position = token_position;
    record->state = LIS_INTRA_LAYER_RECORD_ACTIVE;
    return LIS_STATUS_OK;
}

/*
 * Shared entry guard for the two record-mutating resolution calls. A record
 * that is not ACTIVE fails closed: it transitions to sticky INVALID so a
 * missed return-value check downstream degrades to "artifact suppressed",
 * never to "silently partial artifact".
 */
static lis_status lis_intra_layer_require_active(
    lis_intra_layer_trace_record *record)
{
    if (record->state != LIS_INTRA_LAYER_RECORD_ACTIVE) {
        record->state = LIS_INTRA_LAYER_RECORD_INVALID;
        return LIS_STATUS_BAD_STATE;
    }
    return LIS_STATUS_OK;
}

/*
 * Occupancy and strict ordering are shared by captured and unavailable
 * resolutions: the 17-stage sequence is resolved in a single increasing pass.
 * Malformed order is rejected, never sorted.
 */
static lis_status lis_intra_layer_check_slot_order(
    const lis_intra_layer_trace_record *record,
    size_t stage_order)
{
    if (record->slots[stage_order].occupied != 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (record->has_last_resolved != 0 &&
        stage_order <= record->last_resolved_stage_order) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    return LIS_STATUS_OK;
}

lis_status lis_intra_layer_record_append_observation(
    lis_intra_layer_trace_record *record,
    const lis_intra_layer_observation *observation)
{
    const lis_intra_layer_stage_info *info;
    lis_status status;
    size_t stage_order;
    size_t product = 1U;
    size_t index;

    if (record == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (observation == NULL) {
        record->state = LIS_INTRA_LAYER_RECORD_INVALID;
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    status = lis_intra_layer_require_active(record);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    info = lis_intra_layer_stage_lookup((size_t)observation->stage);
    if (info == NULL) {
        record->state = LIS_INTRA_LAYER_RECORD_INVALID;
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    stage_order = info->stage_order;

    if (observation->phase != LIS_INTRA_LAYER_PHASE_DECODE) {
        record->state = LIS_INTRA_LAYER_RECORD_INVALID;
        return LIS_STATUS_UNSUPPORTED;
    }
    if (observation->stage_order != stage_order ||
        observation->execution_ordinal != observation->stage_order ||
        observation->batch_index != 0U ||
        observation->sequence_index != 0U ||
        observation->runtime_checkpoint_step !=
            record->runtime_checkpoint_step ||
        observation->layer_index != record->target_layer ||
        observation->token_position != record->token_position) {
        record->state = LIS_INTRA_LAYER_RECORD_INVALID;
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    status = lis_intra_layer_check_slot_order(record, stage_order);
    if (status != LIS_STATUS_OK) {
        record->state = LIS_INTRA_LAYER_RECORD_INVALID;
        return status;
    }

    if (observation->rank == 0U ||
        observation->rank > LIS_INTRA_LAYER_MAX_RANK) {
        record->state = LIS_INTRA_LAYER_RECORD_INVALID;
        return LIS_STATUS_UNSUPPORTED_SHAPE;
    }
    for (index = 0; index < observation->rank; ++index) {
        if (observation->shape[index] == 0U) {
            record->state = LIS_INTRA_LAYER_RECORD_INVALID;
            return LIS_STATUS_UNSUPPORTED_SHAPE;
        }
        if (lis_intra_layer_mul_overflows(product, observation->shape[index])) {
            record->state = LIS_INTRA_LAYER_RECORD_INVALID;
            return LIS_STATUS_OVERFLOW;
        }
        product *= observation->shape[index];
    }
    if (observation->element_count == 0U ||
        product != observation->element_count) {
        record->state = LIS_INTRA_LAYER_RECORD_INVALID;
        return LIS_STATUS_SHAPE_MISMATCH;
    }

    /* Contract integer flags carry exactly two legal values; there is no
     * implicit coercion from an arbitrary non-zero integer. */
    if ((observation->nan != 0 && observation->nan != 1) ||
        (observation->inf != 0 && observation->inf != 1)) {
        record->state = LIS_INTRA_LAYER_RECORD_INVALID;
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    /* The digest is required evidence and is carried, never computed here. */
    if (observation->digest.valid != 1) {
        record->state = LIS_INTRA_LAYER_RECORD_INVALID;
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (record->digest_element_visits >
        SIZE_MAX - observation->element_count) {
        record->state = LIS_INTRA_LAYER_RECORD_INVALID;
        return LIS_STATUS_OVERFLOW;
    }

    /* Every check has passed, so a rejected append leaves all counters and all
     * slots untouched. */
    record->slots[stage_order].occupied = 1;
    record->slots[stage_order].captured = 1;
    record->slots[stage_order].observation = *observation;
    record->slots[stage_order].missing_state = LIS_INTRA_LAYER_MISSING_INVALID;
    record->slots[stage_order].detail[0] = '\0';
    record->has_last_resolved = 1;
    record->last_resolved_stage_order = stage_order;
    record->captured_count += 1U;
    record->digest_element_visits += observation->element_count;
    return LIS_STATUS_OK;
}

lis_status lis_intra_layer_record_mark_unavailable(
    lis_intra_layer_trace_record *record,
    lis_intra_layer_stage stage,
    lis_intra_layer_missing_state state,
    const char *detail)
{
    const lis_intra_layer_stage_info *info;
    lis_status status;
    size_t stage_order;
    size_t length;

    if (record == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    status = lis_intra_layer_require_active(record);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    info = lis_intra_layer_stage_lookup((size_t)stage);
    if (info == NULL) {
        record->state = LIS_INTRA_LAYER_RECORD_INVALID;
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    stage_order = info->stage_order;

    /* "captured" is not a missing state, and neither is the zero sentinel. */
    if (state != LIS_INTRA_LAYER_MISSING_NOT_CAPTURED &&
        state != LIS_INTRA_LAYER_MISSING_UNSUPPORTED &&
        state != LIS_INTRA_LAYER_MISSING_MALFORMED &&
        state != LIS_INTRA_LAYER_MISSING_UNEXPECTEDLY_ABSENT) {
        record->state = LIS_INTRA_LAYER_RECORD_INVALID;
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    status = lis_intra_layer_validate_text(detail, LIS_INTRA_LAYER_DETAIL_MAX);
    if (status != LIS_STATUS_OK) {
        record->state = LIS_INTRA_LAYER_RECORD_INVALID;
        return status;
    }
    status = lis_intra_layer_check_slot_order(record, stage_order);
    if (status != LIS_STATUS_OK) {
        record->state = LIS_INTRA_LAYER_RECORD_INVALID;
        return status;
    }

    length = strlen(detail);
    memcpy(record->slots[stage_order].detail, detail, length);
    record->slots[stage_order].detail[length] = '\0';
    record->slots[stage_order].occupied = 1;
    record->slots[stage_order].captured = 0;
    record->slots[stage_order].missing_state = state;
    memset(&record->slots[stage_order].observation, 0,
           sizeof(record->slots[stage_order].observation));
    record->has_last_resolved = 1;
    record->last_resolved_stage_order = stage_order;
    record->missing_count += 1U;
    return LIS_STATUS_OK;
}

lis_status lis_intra_layer_record_finalize(
    lis_intra_layer_trace_record *record)
{
    size_t index;

    if (record == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (record->state == LIS_INTRA_LAYER_RECORD_READY) {
        return LIS_STATUS_OK;
    }
    if (record->state != LIS_INTRA_LAYER_RECORD_ACTIVE) {
        record->state = LIS_INTRA_LAYER_RECORD_INVALID;
        return LIS_STATUS_BAD_STATE;
    }
    for (index = 0; index < LIS_INTRA_LAYER_STAGE_COUNT; ++index) {
        if (record->slots[index].occupied == 0) {
            record->state = LIS_INTRA_LAYER_RECORD_INVALID;
            return LIS_STATUS_BAD_STATE;
        }
    }
    if (record->captured_count > LIS_INTRA_LAYER_STAGE_COUNT ||
        record->captured_count + record->missing_count !=
            LIS_INTRA_LAYER_STAGE_COUNT) {
        record->state = LIS_INTRA_LAYER_RECORD_INVALID;
        return LIS_STATUS_BAD_STATE;
    }
    /* Disjointness and canonical ordering need no runtime check: one slot per
     * stage cannot be both, and emission walks slots 0..16. */
    record->state = LIS_INTRA_LAYER_RECORD_READY;
    return LIS_STATUS_OK;
}

static void lis_intra_layer_write_coordinate(
    FILE *fp,
    const lis_intra_layer_trace_record *record,
    const lis_intra_layer_stage_info *info,
    const lis_intra_layer_json_hooks *hooks)
{
    fprintf(fp, "{\"runtime_checkpoint_step\":%zu,\"layer_index\":%zu,"
                "\"stage_id\":",
            record->runtime_checkpoint_step, record->target_layer);
    hooks->write_string(fp, info->stage_id);
    fputs(",\"tensor_role\":", fp);
    hooks->write_string(fp, info->tensor_role);
    fprintf(fp, ",\"batch_index\":0,\"sequence_index\":0,"
                "\"token_position\":%zu,\"stage_order\":%zu,"
                "\"execution_ordinal\":%zu}",
            record->token_position, info->stage_order, info->stage_order);
}

static const char *lis_intra_layer_missing_state_name(
    lis_intra_layer_missing_state state)
{
    switch (state) {
    case LIS_INTRA_LAYER_MISSING_NOT_CAPTURED:
        return "not_captured";
    case LIS_INTRA_LAYER_MISSING_UNSUPPORTED:
        return "unsupported";
    case LIS_INTRA_LAYER_MISSING_MALFORMED:
        return "malformed";
    case LIS_INTRA_LAYER_MISSING_UNEXPECTEDLY_ABSENT:
        return "unexpectedly_absent";
    case LIS_INTRA_LAYER_MISSING_INVALID:
    default:
        break;
    }
    return "";
}

static void lis_intra_layer_write_shape(
    FILE *fp,
    const lis_intra_layer_observation *observation)
{
    size_t rank;

    fputc('[', fp);
    for (rank = 0; rank < observation->rank; ++rank) {
        if (rank > 0U) {
            fputc(',', fp);
        }
        fprintf(fp, "%zu", observation->shape[rank]);
    }
    fputc(']', fp);
}

static void lis_intra_layer_write_layout(
    FILE *fp,
    const lis_intra_layer_trace_record *record,
    const lis_intra_layer_json_hooks *hooks)
{
    size_t index;
    size_t emitted;

    fputs(",\"intra_layer_checkpoint_layout\":{\"layout_name\":", fp);
    hooks->write_string(fp, LIS_INTRA_LAYER_LAYOUT_NAME);
    fprintf(fp, ",\"layout_version\":%u,\"model_family\":",
            LIS_INTRA_LAYER_LAYOUT_VERSION);
    hooks->write_string(fp, LIS_INTRA_LAYER_MODEL_FAMILY);
    fputs(",\"stage_taxonomy\":", fp);
    hooks->write_string(fp, LIS_INTRA_LAYER_STAGE_TAXONOMY);
    fprintf(fp, ",\"runtime_checkpoint_step\":%zu,\"phase\":",
            record->runtime_checkpoint_step);
    hooks->write_string(fp, LIS_INTRA_LAYER_PHASE_DECODE_NAME);
    fprintf(fp, ",\"target_layer\":%zu,\"batch_index\":0,"
                "\"sequence_index\":0,\"token_position\":%zu,"
                "\"ordering_semantics\":",
            record->target_layer, record->token_position);
    hooks->write_string(fp, LIS_INTRA_LAYER_ORDERING_SEMANTICS);
    fputs(",\"duplicate_coordinate_policy\":", fp);
    hooks->write_string(fp, LIS_INTRA_LAYER_DUPLICATE_POLICY);

    fputs(",\"requested_coordinates\":[", fp);
    for (index = 0; index < LIS_INTRA_LAYER_STAGE_COUNT; ++index) {
        if (index > 0U) {
            fputc(',', fp);
        }
        lis_intra_layer_write_coordinate(
            fp, record, lis_intra_layer_stage_lookup(index), hooks);
    }

    fputs("],\"captured_coordinates\":[", fp);
    emitted = 0;
    for (index = 0; index < LIS_INTRA_LAYER_STAGE_COUNT; ++index) {
        if (record->slots[index].captured == 0) {
            continue;
        }
        if (emitted++ > 0U) {
            fputc(',', fp);
        }
        lis_intra_layer_write_coordinate(
            fp, record, lis_intra_layer_stage_lookup(index), hooks);
    }

    fputs("],\"missing_coordinates\":[", fp);
    emitted = 0;
    for (index = 0; index < LIS_INTRA_LAYER_STAGE_COUNT; ++index) {
        if (record->slots[index].captured != 0) {
            continue;
        }
        if (emitted++ > 0U) {
            fputc(',', fp);
        }
        fputs("{\"coordinate\":", fp);
        lis_intra_layer_write_coordinate(
            fp, record, lis_intra_layer_stage_lookup(index), hooks);
        fputs(",\"state\":", fp);
        hooks->write_string(fp, lis_intra_layer_missing_state_name(
                                    record->slots[index].missing_state));
        fputs(",\"detail\":", fp);
        hooks->write_string(fp, record->slots[index].detail);
        fputc('}', fp);
    }

    fputs("],\"available_summary_fields\":[\"min\",\"max\",\"mean\",\"l2\","
          "\"nan\",\"inf\",\"digest\"],"
          "\"digest_contract\":{\"algorithm\":\"sha256\","
          "\"version\":\"" LIS_INTRA_LAYER_DIGEST_VERSION "\","
          "\"observed_dtype\":\"" LIS_CHECKPOINT_DIGEST_OBSERVED_DTYPE "\","
          "\"byte_order\":\"" LIS_CHECKPOINT_DIGEST_BYTE_ORDER "\","
          "\"canonicalization\":\""
          LIS_CHECKPOINT_DIGEST_CANONICALIZATION "\"},"
          "\"full_tensor_payload_allowed\":false}", fp);
}

static void lis_intra_layer_write_entry(
    FILE *fp,
    const lis_intra_layer_trace_record *record,
    const lis_intra_layer_stage_info *info,
    const lis_intra_layer_observation *observation,
    const lis_intra_layer_json_hooks *hooks)
{
    char digest_hex[LIS_CHECKPOINT_DIGEST_HEX_SIZE + 1U];

    fprintf(fp, "{\"runtime_checkpoint_step\":%zu,\"phase\":",
            observation->runtime_checkpoint_step);
    hooks->write_string(fp, LIS_INTRA_LAYER_PHASE_DECODE_NAME);
    fprintf(fp, ",\"layer_index\":%zu,\"stage_id\":",
            observation->layer_index);
    hooks->write_string(fp, info->stage_id);
    fputs(",\"tensor_role\":", fp);
    hooks->write_string(fp, info->tensor_role);
    fputs(",\"public_name\":", fp);
    hooks->write_string(fp, info->public_name);
    fprintf(fp, ",\"batch_index\":%zu,\"sequence_index\":%zu,"
                "\"token_position\":%zu,\"stage_order\":%zu,"
                "\"execution_ordinal\":%zu,\"shape\":",
            observation->batch_index, observation->sequence_index,
            observation->token_position, observation->stage_order,
            observation->execution_ordinal);
    lis_intra_layer_write_shape(fp, observation);
    fputs(",\"observed_dtype\":", fp);
    hooks->write_string(fp, LIS_CHECKPOINT_DIGEST_OBSERVED_DTYPE);
    fputs(",\"precision_path\":", fp);
    hooks->write_string(fp, record->precision_path);
    fprintf(fp, ",\"element_count\":%zu,"
                "\"available_summary_fields\":[\"min\",\"max\",\"mean\","
                "\"l2\",\"nan\",\"inf\",\"digest\"],\"min\":",
            observation->element_count);
    hooks->write_float(fp, observation->min);
    fputs(",\"max\":", fp);
    hooks->write_float(fp, observation->max);
    fputs(",\"mean\":", fp);
    hooks->write_float(fp, observation->mean);
    fputs(",\"l2\":", fp);
    hooks->write_float(fp, observation->l2);
    fprintf(fp, ",\"nan\":%d,\"inf\":%d,"
                "\"digest\":{\"algorithm\":\"sha256\","
                "\"version\":\"" LIS_INTRA_LAYER_DIGEST_VERSION "\","
                "\"tensor_role\":",
            observation->nan, observation->inf);
    hooks->write_string(fp, info->tensor_role);
    fputs(",\"shape\":", fp);
    lis_intra_layer_write_shape(fp, observation);
    fputs(",\"observed_dtype\":\"" LIS_CHECKPOINT_DIGEST_OBSERVED_DTYPE "\","
          "\"byte_order\":\"" LIS_CHECKPOINT_DIGEST_BYTE_ORDER "\","
          "\"canonicalization\":\""
          LIS_CHECKPOINT_DIGEST_CANONICALIZATION "\","
          "\"value\":\"sha256:", fp);
    lis_checkpoint_digest_hex(&observation->digest, digest_hex);
    fputs(digest_hex, fp);
    fputs("\"}}", fp);
}

lis_status lis_intra_layer_record_write_json(
    FILE *fp,
    const lis_intra_layer_trace_record *record,
    const lis_intra_layer_json_hooks *hooks)
{
    size_t index;
    size_t emitted = 0;

    if (fp == NULL || record == NULL || hooks == NULL ||
        hooks->write_string == NULL || hooks->write_float == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (!lis_intra_layer_record_is_ready(record)) {
        return LIS_STATUS_BAD_STATE;
    }

    /* All content validation completed at append/mark/finalize time, so the
     * emission below is total: nothing can fail on content grounds after the
     * first byte is written. */
    lis_intra_layer_write_layout(fp, record, hooks);
    fputs(",\"intra_layer_trace\":[", fp);
    for (index = 0; index < LIS_INTRA_LAYER_STAGE_COUNT; ++index) {
        if (record->slots[index].captured == 0) {
            continue;
        }
        if (emitted++ > 0U) {
            fputc(',', fp);
        }
        lis_intra_layer_write_entry(fp, record,
                                    lis_intra_layer_stage_lookup(index),
                                    &record->slots[index].observation, hooks);
    }
    fputc(']', fp);
    return LIS_STATUS_OK;
}
