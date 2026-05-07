#include "lis/runtime.h"

#include <stdlib.h>
#include <string.h>

lis_status lis_static_batch_init(lis_static_batch *batch, size_t batch_size,
                                 size_t max_tokens)
{
    lis_static_batch local = { 0 };

    if (batch == NULL || batch_size == 0 || max_tokens == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (batch_size > SIZE_MAX / sizeof(*local.positions)) {
        return LIS_STATUS_OVERFLOW;
    }

    local.positions = calloc(batch_size, sizeof(*local.positions));
    if (local.positions == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }
    local.batch_size = batch_size;
    local.max_tokens = max_tokens;
    *batch = local;
    return LIS_STATUS_OK;
}

void lis_static_batch_destroy(lis_static_batch *batch)
{
    if (batch == NULL) {
        return;
    }

    free(batch->positions);
    memset(batch, 0, sizeof(*batch));
}

lis_status lis_static_batch_validate_lengths(const lis_static_batch *batch,
                                             const size_t *lengths,
                                             size_t length_count)
{
    size_t index;

    if (batch == NULL || lengths == NULL || batch->positions == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (length_count != batch->batch_size) {
        return LIS_STATUS_SHAPE_MISMATCH;
    }

    for (index = 0; index < length_count; ++index) {
        if (lengths[index] == 0) {
            return LIS_STATUS_INVALID_ARGUMENT;
        }
        if (lengths[index] > batch->max_tokens) {
            return LIS_STATUS_LIMIT_EXCEEDED;
        }
    }

    return LIS_STATUS_OK;
}
