#include "lis/checkpoint_digest.h"

#include <stdint.h>
#include <string.h>

#define LIS_SHA256_BLOCK_SIZE 64U

typedef struct {
    uint32_t state[8];
    uint64_t total_bytes;
    unsigned char block[LIS_SHA256_BLOCK_SIZE];
    size_t block_size;
} lis_sha256_context;

static uint32_t lis_sha256_rotr(uint32_t value, unsigned int shift)
{
    return (value >> shift) | (value << (32U - shift));
}

static uint32_t lis_sha256_load_be32(const unsigned char *data)
{
    return ((uint32_t)data[0] << 24U) |
           ((uint32_t)data[1] << 16U) |
           ((uint32_t)data[2] << 8U) |
           (uint32_t)data[3];
}

static void lis_sha256_store_be32(unsigned char *data, uint32_t value)
{
    data[0] = (unsigned char)(value >> 24U);
    data[1] = (unsigned char)(value >> 16U);
    data[2] = (unsigned char)(value >> 8U);
    data[3] = (unsigned char)value;
}

static void lis_sha256_transform(lis_sha256_context *context,
                                 const unsigned char block[64])
{
    static const uint32_t constants[64] = {
        UINT32_C(0x428a2f98), UINT32_C(0x71374491), UINT32_C(0xb5c0fbcf),
        UINT32_C(0xe9b5dba5), UINT32_C(0x3956c25b), UINT32_C(0x59f111f1),
        UINT32_C(0x923f82a4), UINT32_C(0xab1c5ed5), UINT32_C(0xd807aa98),
        UINT32_C(0x12835b01), UINT32_C(0x243185be), UINT32_C(0x550c7dc3),
        UINT32_C(0x72be5d74), UINT32_C(0x80deb1fe), UINT32_C(0x9bdc06a7),
        UINT32_C(0xc19bf174), UINT32_C(0xe49b69c1), UINT32_C(0xefbe4786),
        UINT32_C(0x0fc19dc6), UINT32_C(0x240ca1cc), UINT32_C(0x2de92c6f),
        UINT32_C(0x4a7484aa), UINT32_C(0x5cb0a9dc), UINT32_C(0x76f988da),
        UINT32_C(0x983e5152), UINT32_C(0xa831c66d), UINT32_C(0xb00327c8),
        UINT32_C(0xbf597fc7), UINT32_C(0xc6e00bf3), UINT32_C(0xd5a79147),
        UINT32_C(0x06ca6351), UINT32_C(0x14292967), UINT32_C(0x27b70a85),
        UINT32_C(0x2e1b2138), UINT32_C(0x4d2c6dfc), UINT32_C(0x53380d13),
        UINT32_C(0x650a7354), UINT32_C(0x766a0abb), UINT32_C(0x81c2c92e),
        UINT32_C(0x92722c85), UINT32_C(0xa2bfe8a1), UINT32_C(0xa81a664b),
        UINT32_C(0xc24b8b70), UINT32_C(0xc76c51a3), UINT32_C(0xd192e819),
        UINT32_C(0xd6990624), UINT32_C(0xf40e3585), UINT32_C(0x106aa070),
        UINT32_C(0x19a4c116), UINT32_C(0x1e376c08), UINT32_C(0x2748774c),
        UINT32_C(0x34b0bcb5), UINT32_C(0x391c0cb3), UINT32_C(0x4ed8aa4a),
        UINT32_C(0x5b9cca4f), UINT32_C(0x682e6ff3), UINT32_C(0x748f82ee),
        UINT32_C(0x78a5636f), UINT32_C(0x84c87814), UINT32_C(0x8cc70208),
        UINT32_C(0x90befffa), UINT32_C(0xa4506ceb), UINT32_C(0xbef9a3f7),
        UINT32_C(0xc67178f2)
    };
    uint32_t words[64];
    uint32_t a;
    uint32_t b;
    uint32_t c;
    uint32_t d;
    uint32_t e;
    uint32_t f;
    uint32_t g;
    uint32_t h;
    size_t index;

    for (index = 0; index < 16U; ++index) {
        words[index] = lis_sha256_load_be32(block + index * 4U);
    }
    for (index = 16U; index < 64U; ++index) {
        const uint32_t s0 = lis_sha256_rotr(words[index - 15U], 7U) ^
                            lis_sha256_rotr(words[index - 15U], 18U) ^
                            (words[index - 15U] >> 3U);
        const uint32_t s1 = lis_sha256_rotr(words[index - 2U], 17U) ^
                            lis_sha256_rotr(words[index - 2U], 19U) ^
                            (words[index - 2U] >> 10U);

        words[index] = words[index - 16U] + s0 + words[index - 7U] + s1;
    }

    a = context->state[0];
    b = context->state[1];
    c = context->state[2];
    d = context->state[3];
    e = context->state[4];
    f = context->state[5];
    g = context->state[6];
    h = context->state[7];

    for (index = 0; index < 64U; ++index) {
        const uint32_t sum1 = lis_sha256_rotr(e, 6U) ^
                              lis_sha256_rotr(e, 11U) ^
                              lis_sha256_rotr(e, 25U);
        const uint32_t choose = (e & f) ^ ((~e) & g);
        const uint32_t temp1 = h + sum1 + choose + constants[index] + words[index];
        const uint32_t sum0 = lis_sha256_rotr(a, 2U) ^
                              lis_sha256_rotr(a, 13U) ^
                              lis_sha256_rotr(a, 22U);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = sum0 + majority;

        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }

    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}

static void lis_sha256_init(lis_sha256_context *context)
{
    static const uint32_t initial_state[8] = {
        UINT32_C(0x6a09e667), UINT32_C(0xbb67ae85),
        UINT32_C(0x3c6ef372), UINT32_C(0xa54ff53a),
        UINT32_C(0x510e527f), UINT32_C(0x9b05688c),
        UINT32_C(0x1f83d9ab), UINT32_C(0x5be0cd19)
    };

    memset(context, 0, sizeof(*context));
    memcpy(context->state, initial_state, sizeof(initial_state));
}

static void lis_sha256_update(lis_sha256_context *context,
                              const unsigned char *data,
                              size_t size)
{
    size_t consumed = 0;

    context->total_bytes += (uint64_t)size;
    while (consumed < size) {
        size_t available = LIS_SHA256_BLOCK_SIZE - context->block_size;
        size_t take = size - consumed < available ? size - consumed : available;

        memcpy(context->block + context->block_size, data + consumed, take);
        context->block_size += take;
        consumed += take;
        if (context->block_size == LIS_SHA256_BLOCK_SIZE) {
            lis_sha256_transform(context, context->block);
            context->block_size = 0;
        }
    }
}

static void lis_sha256_final(lis_sha256_context *context,
                             unsigned char out[LIS_CHECKPOINT_DIGEST_SIZE])
{
    const uint64_t total_bits = context->total_bytes * UINT64_C(8);
    unsigned char length_bytes[8];
    unsigned char marker = 0x80U;
    unsigned char zero = 0;
    size_t index;

    for (index = 0; index < 8U; ++index) {
        length_bytes[7U - index] =
            (unsigned char)((total_bits >> (index * 8U)) & UINT64_C(0xff));
    }
    lis_sha256_update(context, &marker, 1U);
    while (context->block_size != 56U) {
        lis_sha256_update(context, &zero, 1U);
    }
    lis_sha256_update(context, length_bytes, sizeof(length_bytes));
    for (index = 0; index < 8U; ++index) {
        lis_sha256_store_be32(out + index * 4U, context->state[index]);
    }
}

static void lis_checkpoint_digest_update_u64_le(lis_sha256_context *context,
                                                 uint64_t value)
{
    unsigned char bytes[8];
    size_t index;

    for (index = 0; index < sizeof(bytes); ++index) {
        bytes[index] = (unsigned char)((value >> (index * 8U)) & UINT64_C(0xff));
    }
    lis_sha256_update(context, bytes, sizeof(bytes));
}

static void lis_checkpoint_digest_update_tag(lis_sha256_context *context,
                                              const char *tag)
{
    const size_t length = strlen(tag);

    lis_checkpoint_digest_update_u64_le(context, (uint64_t)length);
    lis_sha256_update(context, (const unsigned char *)tag, length);
}

lis_status lis_checkpoint_digest_fp32(
    const char *tensor_role,
    const size_t *shape,
    size_t rank,
    const float *values,
    size_t element_count,
    lis_checkpoint_digest *out)
{
    static const unsigned char domain[] = "LIS_CHECKPOINT_DIGEST";
    lis_sha256_context context;
    size_t shape_product = 1U;
    size_t index;
    unsigned char zero = 0;

    if (tensor_role == NULL || shape == NULL || rank == 0U || values == NULL ||
        element_count == 0U || out == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));
    for (index = 0; index < rank; ++index) {
        if (shape[index] == 0U || shape_product > SIZE_MAX / shape[index]) {
            return LIS_STATUS_OVERFLOW;
        }
        shape_product *= shape[index];
    }
    if (shape_product != element_count) {
        return LIS_STATUS_SHAPE_MISMATCH;
    }

    lis_sha256_init(&context);
    lis_sha256_update(&context, domain, sizeof(domain) - 1U);
    lis_sha256_update(&context, &zero, 1U);
    lis_checkpoint_digest_update_tag(&context, LIS_CHECKPOINT_DIGEST_VERSION);
    lis_checkpoint_digest_update_tag(&context, tensor_role);
    lis_checkpoint_digest_update_u64_le(&context, (uint64_t)rank);
    for (index = 0; index < rank; ++index) {
        lis_checkpoint_digest_update_u64_le(&context, (uint64_t)shape[index]);
    }
    lis_checkpoint_digest_update_tag(&context,
                                     LIS_CHECKPOINT_DIGEST_OBSERVED_DTYPE);
    lis_checkpoint_digest_update_tag(&context, LIS_CHECKPOINT_DIGEST_BYTE_ORDER);
    lis_checkpoint_digest_update_u64_le(&context, (uint64_t)element_count);
    for (index = 0; index < element_count; ++index) {
        uint32_t bits = 0;
        unsigned char bytes[4];

        memcpy(&bits, values + index, sizeof(bits));
        if ((bits & UINT32_C(0x7f800000)) == UINT32_C(0x7f800000) &&
            (bits & UINT32_C(0x007fffff)) != 0U) {
            bits = UINT32_C(0x7fc00000);
        }
        bytes[0] = (unsigned char)(bits & UINT32_C(0xff));
        bytes[1] = (unsigned char)((bits >> 8U) & UINT32_C(0xff));
        bytes[2] = (unsigned char)((bits >> 16U) & UINT32_C(0xff));
        bytes[3] = (unsigned char)((bits >> 24U) & UINT32_C(0xff));
        lis_sha256_update(&context, bytes, sizeof(bytes));
    }
    lis_sha256_final(&context, out->bytes);
    out->valid = 1;
    return LIS_STATUS_OK;
}

void lis_checkpoint_digest_hex(
    const lis_checkpoint_digest *digest,
    char out_hex[LIS_CHECKPOINT_DIGEST_HEX_SIZE + 1U])
{
    static const char hex[] = "0123456789abcdef";
    size_t index;

    if (out_hex == NULL) {
        return;
    }
    if (digest == NULL || !digest->valid) {
        memset(out_hex, '0', LIS_CHECKPOINT_DIGEST_HEX_SIZE);
        out_hex[LIS_CHECKPOINT_DIGEST_HEX_SIZE] = '\0';
        return;
    }
    for (index = 0; index < LIS_CHECKPOINT_DIGEST_SIZE; ++index) {
        out_hex[index * 2U] = hex[digest->bytes[index] >> 4U];
        out_hex[index * 2U + 1U] = hex[digest->bytes[index] & 0x0fU];
    }
    out_hex[LIS_CHECKPOINT_DIGEST_HEX_SIZE] = '\0';
}
