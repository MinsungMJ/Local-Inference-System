/*
 * srcs/core/context.c — Context window policy implementation.
 *
 * Scope: comment-only hardening. The implementation already respects
 * the contract table; changes are strictly comments, ordering, and
 * diagnostics. No structural change, no new behavior, no new enum values.
 */

#include "lis/context.h"

lis_context_window_policy lis_context_window_policy_default(size_t trained_max_tokens,
                                                             size_t configured_max_tokens)
{
    lis_context_window_policy policy = {
        .trained_max_tokens = trained_max_tokens,
        .configured_max_tokens = configured_max_tokens,
        /* Exactly one supported enum value; callers must not change. */
        .config_mode = LIS_CONTEXT_CONFIG_RUNTIME,
        .over_trained_policy = LIS_CONTEXT_OVER_TRAINED_REJECT,
    };

    return policy;
}

lis_status lis_context_window_policy_validate(const lis_context_window_policy *policy)
{
    if (policy == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    /*
     * Ordering: zero checks first (INVALID_ARGUMENT), then enum
     * mismatch (UNSUPPORTED), then configured>trained (LIMIT_EXCEEDED).
     * Do not clamp — fail-fast.
     */
    if (policy->trained_max_tokens == 0 || policy->configured_max_tokens == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (policy->config_mode != LIS_CONTEXT_CONFIG_RUNTIME) {
        return LIS_STATUS_UNSUPPORTED;
    }
    if (policy->over_trained_policy != LIS_CONTEXT_OVER_TRAINED_REJECT) {
        return LIS_STATUS_UNSUPPORTED;
    }
    if (policy->configured_max_tokens > policy->trained_max_tokens) {
        return LIS_STATUS_LIMIT_EXCEEDED;
    }

    return LIS_STATUS_OK;
}

lis_status lis_context_window_validate_request(const lis_context_window_policy *policy,
                                                size_t requested_tokens)
{
    lis_status status;

    status = lis_context_window_policy_validate(policy);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (requested_tokens == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (requested_tokens > policy->configured_max_tokens) {
        return LIS_STATUS_LIMIT_EXCEEDED;
    }

    return LIS_STATUS_OK;
}
