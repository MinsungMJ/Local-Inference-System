#include "lis/dtype.h"

#include <stdint.h>
#include <string.h>

static float lis_f16_to_f32(uint16_t value)
{
    const uint32_t sign = ((uint32_t)value & 0x8000U) << 16;
    const uint32_t exp_bits = ((uint32_t)value >> 10) & 0x1fU;
    uint32_t frac = (uint32_t)value & 0x03ffU;
    uint32_t out = 0;
    float result = 0.0f;

    if (exp_bits == 0) {
        if (frac == 0) {
            out = sign;
        } else {
            int32_t exp = -14;

            while ((frac & 0x0400U) == 0) {
                frac <<= 1;
                --exp;
            }
            frac &= 0x03ffU;
            out = sign | ((uint32_t)(exp + 127) << 23) | (frac << 13);
        }
    } else if (exp_bits == 31U) {
        out = sign | 0x7f800000U | (frac << 13);
    } else {
        out = sign | ((exp_bits + 112U) << 23) | (frac << 13);
    }
    memcpy(&result, &out, sizeof(result));
    return result;
}

float lis_dtype_scalar_read_f32(lis_dtype dtype, const void *data, size_t index)
{
    if (dtype == LIS_DTYPE_F32) {
        float value = 0.0f;

        memcpy(&value, (const unsigned char *)data + index * sizeof(value),
               sizeof(value));
        return value;
    }
    if (dtype == LIS_DTYPE_BF16) {
        uint16_t raw = 0;
        uint32_t bits = 0;
        float value = 0.0f;

        memcpy(&raw, (const unsigned char *)data + index * sizeof(raw),
               sizeof(raw));
        bits = (uint32_t)raw << 16;
        memcpy(&value, &bits, sizeof(value));
        return value;
    }
    if (dtype == LIS_DTYPE_F16) {
        uint16_t raw = 0;

        memcpy(&raw, (const unsigned char *)data + index * sizeof(raw),
               sizeof(raw));
        return lis_f16_to_f32(raw);
    }
    return 0.0f;
}

bool lis_dtype_is_supported(lis_dtype dtype)
{
    switch (dtype) {
    case LIS_DTYPE_F32:
    case LIS_DTYPE_F16:
    case LIS_DTYPE_BF16:
    case LIS_DTYPE_I32:
    case LIS_DTYPE_U32:
        return true;
    case LIS_DTYPE_INVALID:
        return false;
    }

    return false;
}

const char *lis_dtype_name(lis_dtype dtype)
{
    switch (dtype) {
    case LIS_DTYPE_F32:
        return "f32";
    case LIS_DTYPE_F16:
        return "f16";
    case LIS_DTYPE_BF16:
        return "bf16";
    case LIS_DTYPE_I32:
        return "i32";
    case LIS_DTYPE_U32:
        return "u32";
    case LIS_DTYPE_INVALID:
        return "invalid";
    }

    return "unknown";
}

lis_status lis_dtype_size_bytes(lis_dtype dtype, size_t *out_size)
{
    if (out_size == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    switch (dtype) {
    case LIS_DTYPE_F32:
    case LIS_DTYPE_I32:
    case LIS_DTYPE_U32:
        *out_size = 4;
        return LIS_STATUS_OK;
    case LIS_DTYPE_F16:
    case LIS_DTYPE_BF16:
        *out_size = 2;
        return LIS_STATUS_OK;
    case LIS_DTYPE_INVALID:
        *out_size = 0;
        return LIS_STATUS_UNSUPPORTED;
    }

    *out_size = 0;
    return LIS_STATUS_UNSUPPORTED;
}
