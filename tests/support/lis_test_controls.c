#include "lis_test_controls.h"

#include <math.h>
#include <string.h>

typedef struct {
    int selected_token_enabled;
    size_t selected_token_step;
    size_t selected_token_id;

    int layer_observation_enabled;
    size_t layer_observation_layer;
    size_t layer_observation_element;
    float layer_observation_delta;
    int layer_observation_applied;

    int intra_layer_observation_enabled;
    lis_intra_layer_stage intra_layer_observation_stage;
    size_t intra_layer_observation_element;
    float intra_layer_observation_delta;
    int intra_layer_observation_applied;
} lis_test_control_state;

static lis_test_control_state s_lis_test_controls;

void lis_cli_test_injection_reset(void)
{
    memset(&s_lis_test_controls, 0, sizeof(s_lis_test_controls));
}

lis_status lis_cli_test_override_selected_token(size_t step,
                                                size_t token_id)
{
    s_lis_test_controls.selected_token_enabled = 1;
    s_lis_test_controls.selected_token_step = step;
    s_lis_test_controls.selected_token_id = token_id;
    return LIS_STATUS_OK;
}

lis_status lis_cli_test_perturb_layer_observation(size_t layer_index,
                                                  size_t element_index,
                                                  float delta)
{
    if (!isfinite(delta)) {
        s_lis_test_controls.layer_observation_enabled = 0;
        s_lis_test_controls.layer_observation_applied = 0;
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    s_lis_test_controls.layer_observation_enabled = 1;
    s_lis_test_controls.layer_observation_layer = layer_index;
    s_lis_test_controls.layer_observation_element = element_index;
    s_lis_test_controls.layer_observation_delta = delta;
    s_lis_test_controls.layer_observation_applied = 0;
    return LIS_STATUS_OK;
}

lis_status lis_cli_test_perturb_intra_layer_observation(
    lis_intra_layer_stage stage,
    size_t logical_element_index,
    float delta)
{
    if (lis_intra_layer_stage_lookup((size_t)stage) == NULL ||
        !isfinite(delta)) {
        s_lis_test_controls.intra_layer_observation_enabled = 0;
        s_lis_test_controls.intra_layer_observation_applied = 0;
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    s_lis_test_controls.intra_layer_observation_enabled = 1;
    s_lis_test_controls.intra_layer_observation_stage = stage;
    s_lis_test_controls.intra_layer_observation_element =
        logical_element_index;
    s_lis_test_controls.intra_layer_observation_delta = delta;
    s_lis_test_controls.intra_layer_observation_applied = 0;
    return LIS_STATUS_OK;
}

int lis_test_control_apply_selected_token(size_t step,
                                          size_t vocab_size,
                                          size_t *token_id,
                                          int *should_stop)
{
    if (token_id == NULL || should_stop == NULL ||
        !s_lis_test_controls.selected_token_enabled ||
        s_lis_test_controls.selected_token_step != step ||
        s_lis_test_controls.selected_token_id >= vocab_size) {
        return 0;
    }
    *token_id = s_lis_test_controls.selected_token_id;
    *should_stop = 0;
    return 1;
}

int lis_test_control_layer_observation(size_t layer_index,
                                       size_t element_count,
                                       size_t *element_index,
                                       float *delta)
{
    if (element_index == NULL || delta == NULL ||
        !s_lis_test_controls.layer_observation_enabled ||
        s_lis_test_controls.layer_observation_applied ||
        s_lis_test_controls.layer_observation_layer != layer_index ||
        s_lis_test_controls.layer_observation_element >= element_count) {
        return 0;
    }
    *element_index = s_lis_test_controls.layer_observation_element;
    *delta = s_lis_test_controls.layer_observation_delta;
    return 1;
}

void lis_test_control_mark_layer_observation_applied(void)
{
    if (s_lis_test_controls.layer_observation_enabled) {
        s_lis_test_controls.layer_observation_applied = 1;
    }
}

lis_status lis_test_control_prepare_intra_layer_observation(
    lis_intra_layer_stage stage,
    size_t logical_element_count,
    int *active)
{
    if (active == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    *active = 0;
    if (!s_lis_test_controls.intra_layer_observation_enabled ||
        s_lis_test_controls.intra_layer_observation_applied ||
        s_lis_test_controls.intra_layer_observation_stage != stage) {
        return LIS_STATUS_OK;
    }
    if (s_lis_test_controls.intra_layer_observation_element >=
        logical_element_count) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    *active = 1;
    return LIS_STATUS_OK;
}

float lis_test_control_intra_layer_value(lis_intra_layer_stage stage,
                                         size_t logical_element_index,
                                         float runtime_value)
{
    if (!s_lis_test_controls.intra_layer_observation_enabled ||
        s_lis_test_controls.intra_layer_observation_applied ||
        s_lis_test_controls.intra_layer_observation_stage != stage ||
        s_lis_test_controls.intra_layer_observation_element !=
            logical_element_index) {
        return runtime_value;
    }
    return runtime_value + s_lis_test_controls.intra_layer_observation_delta;
}

void lis_test_control_mark_intra_layer_observation_applied(
    lis_intra_layer_stage stage)
{
    if (s_lis_test_controls.intra_layer_observation_enabled &&
        !s_lis_test_controls.intra_layer_observation_applied &&
        s_lis_test_controls.intra_layer_observation_stage == stage) {
        s_lis_test_controls.intra_layer_observation_applied = 1;
    }
}

int lis_test_control_intra_layer_observation_applied(void)
{
    return s_lis_test_controls.intra_layer_observation_applied;
}
