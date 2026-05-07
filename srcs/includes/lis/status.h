#ifndef LIS_STATUS_H
#define LIS_STATUS_H

typedef enum {
    LIS_STATUS_OK = 0,
    LIS_STATUS_INVALID_ARGUMENT,
    LIS_STATUS_UNSUPPORTED,
    LIS_STATUS_UNSUPPORTED_FORMAT,
    LIS_STATUS_UNSUPPORTED_DTYPE,
    LIS_STATUS_UNSUPPORTED_SHAPE,
    LIS_STATUS_OVERFLOW,
    LIS_STATUS_LIMIT_EXCEEDED,
    LIS_STATUS_SHAPE_MISMATCH,
    LIS_STATUS_FORMAT,
    LIS_STATUS_IO,
    LIS_STATUS_NO_MEMORY,
    LIS_STATUS_BAD_STATE
} lis_status;

/*
 * Returns a static string owned by the program. Callers must not free or
 * mutate the returned pointer.
 */
const char *lis_status_name(lis_status status);

#endif
