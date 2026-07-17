#ifndef LIS_CHECKPOINT_DIGEST_H
#define LIS_CHECKPOINT_DIGEST_H

#include "lis/status.h"

#include <stddef.h>

#define LIS_CHECKPOINT_DIGEST_SIZE 32
#define LIS_CHECKPOINT_DIGEST_HEX_SIZE 64
#define LIS_CHECKPOINT_DIGEST_VERSION "lis.checkpoint.fp32le/v1"
#define LIS_CHECKPOINT_DIGEST_ROLE_LAYER_OUTPUT "layer_output"
#define LIS_CHECKPOINT_DIGEST_OBSERVED_DTYPE "fp32"
#define LIS_CHECKPOINT_DIGEST_BYTE_ORDER "little"
#define LIS_CHECKPOINT_DIGEST_CANONICALIZATION \
    "ieee754-binary32-le;canonical-qnan;preserve-signed-zero"

typedef struct {
    unsigned char bytes[LIS_CHECKPOINT_DIGEST_SIZE];
    int valid;
} lis_checkpoint_digest;

lis_status lis_checkpoint_digest_fp32(
    const char *tensor_role,
    const size_t *shape,
    size_t rank,
    const float *values,
    size_t element_count,
    lis_checkpoint_digest *out);

void lis_checkpoint_digest_hex(
    const lis_checkpoint_digest *digest,
    char out_hex[LIS_CHECKPOINT_DIGEST_HEX_SIZE + 1U]);

#endif
