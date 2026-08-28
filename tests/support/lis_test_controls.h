#ifndef LIS_TEST_CONTROLS_H
#define LIS_TEST_CONTROLS_H

#ifndef LIS_TESTING
#error "lis_test_controls.h is available only to LIS_TESTING builds"
#endif

#include "lis/intra_layer_trace.h"
#include "lis/status.h"

#include <stddef.h>

/* Process-local controls linked only into isolated test binaries. */
void lis_cli_test_injection_reset(void);

lis_status lis_cli_test_override_selected_token(size_t step,
                                                size_t token_id);

lis_status lis_cli_test_perturb_layer_observation(size_t layer_index,
                                                  size_t element_index,
                                                  float delta);

lis_status lis_cli_test_perturb_intra_layer_observation(
    lis_intra_layer_stage stage,
    size_t logical_element_index,
    float delta);

/* Internal queries used only by LIS_TESTING translation-unit variants. */
int lis_test_control_apply_selected_token(size_t step,
                                          size_t vocab_size,
                                          size_t *token_id,
                                          int *should_stop);

int lis_test_control_layer_observation(size_t layer_index,
                                       size_t element_count,
                                       size_t *element_index,
                                       float *delta);

void lis_test_control_mark_layer_observation_applied(void);

lis_status lis_test_control_prepare_intra_layer_observation(
    lis_intra_layer_stage stage,
    size_t logical_element_count,
    int *active);

float lis_test_control_intra_layer_value(lis_intra_layer_stage stage,
                                         size_t logical_element_index,
                                         float runtime_value);

void lis_test_control_mark_intra_layer_observation_applied(
    lis_intra_layer_stage stage);

int lis_test_control_intra_layer_observation_applied(void);

#endif
