#ifndef LIS_CONTEXT_H
#define LIS_CONTEXT_H

#include "lis/status.h"

#include <stddef.h>

typedef enum {
    /*
     * Contract: exactly one supported value. Any other enum value
     * is rejected with LIS_STATUS_UNSUPPORTED — it is not a placeholder
     * for future expansion.
     */
    LIS_CONTEXT_CONFIG_RUNTIME = 1
} lis_context_config_mode;

typedef enum {
    /*
     * Contract: exactly one supported value. Any other enum value
     * is rejected with LIS_STATUS_UNSUPPORTED — it is not a placeholder
     * for future expansion.
     */
    LIS_CONTEXT_OVER_TRAINED_REJECT = 1
} lis_context_over_trained_policy;

typedef struct {
    /*
     * Config/model metadata owns trained_max_tokens. A later CLI or driver
     * provides configured_max_tokens as a runtime setting. RoPE position
     * handling, KV cache capacity, and temporary buffers are sized from the
     * validated configured limit.
     *
     * Contract: 0 < configured_max_tokens <= trained_max_tokens.
     *   - configured > trained  → LIS_STATUS_LIMIT_EXCEEDED
     *   - configured == trained → supported (terminal: decode-0)
     *   - either == 0           → LIS_STATUS_INVALID_ARGUMENT
     */
    size_t trained_max_tokens;
    size_t configured_max_tokens;
    lis_context_config_mode config_mode;
    lis_context_over_trained_policy over_trained_policy;
} lis_context_window_policy;

lis_context_window_policy lis_context_window_policy_default(size_t trained_max_tokens,
                                                            size_t configured_max_tokens);

/*
 * Contract status codes:
 *   NULL policy                           → LIS_STATUS_INVALID_ARGUMENT
 *   trained == 0 or configured == 0       → LIS_STATUS_INVALID_ARGUMENT
 *   config_mode != LIS_CONTEXT_CONFIG_RUNTIME           → LIS_STATUS_UNSUPPORTED
 *   over_trained_policy != LIS_CONTEXT_OVER_TRAINED_REJECT → LIS_STATUS_UNSUPPORTED
 *   configured_max_tokens > trained_max_tokens          → LIS_STATUS_LIMIT_EXCEEDED
 *   otherwise                             → LIS_STATUS_OK
 */
lis_status lis_context_window_policy_validate(const lis_context_window_policy *policy);

/*
 * Validates that requested_tokens > 0 and requested_tokens <= configured_max_tokens
 * after first passing lis_context_window_policy_validate().
 */
lis_status lis_context_window_validate_request(const lis_context_window_policy *policy,
                                               size_t requested_tokens);

#endif
