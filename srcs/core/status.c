#include "lis/status.h"

const char *lis_status_name(lis_status status)
{
    switch (status) {
    case LIS_STATUS_OK:
        return "OK";
    case LIS_STATUS_INVALID_ARGUMENT:
        return "INVALID_ARGUMENT";
    case LIS_STATUS_UNSUPPORTED:
        return "UNSUPPORTED";
    case LIS_STATUS_UNSUPPORTED_FORMAT:
        return "UNSUPPORTED_FORMAT";
    case LIS_STATUS_UNSUPPORTED_DTYPE:
        return "UNSUPPORTED_DTYPE";
    case LIS_STATUS_UNSUPPORTED_SHAPE:
        return "UNSUPPORTED_SHAPE";
    case LIS_STATUS_OVERFLOW:
        return "OVERFLOW";
    case LIS_STATUS_LIMIT_EXCEEDED:
        return "LIMIT_EXCEEDED";
    case LIS_STATUS_SHAPE_MISMATCH:
        return "SHAPE_MISMATCH";
    case LIS_STATUS_FORMAT:
        return "FORMAT";
    case LIS_STATUS_IO:
        return "IO";
    case LIS_STATUS_NO_MEMORY:
        return "NO_MEMORY";
    case LIS_STATUS_BAD_STATE:
        return "BAD_STATE";
    }

    return "UNKNOWN_STATUS";
}
