#ifndef LIS_DTYPE_H
#define LIS_DTYPE_H

#include "lis/status.h"

#include <stdbool.h>
#include <stddef.h>

typedef enum {
    LIS_DTYPE_INVALID = 0,
    LIS_DTYPE_F32,
    LIS_DTYPE_F16,
    LIS_DTYPE_BF16,
    LIS_DTYPE_I32,
    LIS_DTYPE_U32
} lis_dtype;

bool lis_dtype_is_supported(lis_dtype dtype);
const char *lis_dtype_name(lis_dtype dtype);
lis_status lis_dtype_size_bytes(lis_dtype dtype, size_t *out_size);

/*
 * Read one scalar element at `index` from `data` with the given dtype, promoted
 * to f32. Supports LIS_DTYPE_F32, LIS_DTYPE_BF16, and LIS_DTYPE_F16. Returns
 * 0.0f for unsupported dtypes. Used by the CPU reference kernels to promote
 * mixed-dtype weights and KV cache values on the fly.
 */
float lis_dtype_scalar_read_f32(lis_dtype dtype, const void *data,
                                size_t index);

#endif
