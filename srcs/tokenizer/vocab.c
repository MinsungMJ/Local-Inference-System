#include "lis/tokenizer.h"

#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* --- Hex decoding helpers --- */

static int lis_hex_value(unsigned char ch, unsigned char *out)
{
    if (ch >= '0' && ch <= '9') {
        *out = (unsigned char)(ch - '0');
        return 1;
    }
    if (ch >= 'a' && ch <= 'f') {
        *out = (unsigned char)(ch - 'a' + 10);
        return 1;
    }
    if (ch >= 'A' && ch <= 'F') {
        *out = (unsigned char)(ch - 'A' + 10);
        return 1;
    }
    return 0;
}

static lis_status lis_vocab_decode_hex(const char *hex, size_t hex_len,
                                       char **out_bytes, size_t *out_len)
{
    size_t byte_count;
    char *bytes;
    size_t i;

    if (hex_len == 0) {
        *out_bytes = NULL;
        *out_len = 0;
        return LIS_STATUS_OK;
    }
    if (hex_len % 2 != 0) {
        return LIS_STATUS_FORMAT;
    }
    byte_count = hex_len / 2;

    bytes = malloc(byte_count + 1);
    if (bytes == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }

    for (i = 0; i < byte_count; ++i) {
        unsigned char hi;
        unsigned char lo;

        if (!lis_hex_value((unsigned char)hex[2 * i], &hi) ||
            !lis_hex_value((unsigned char)hex[2 * i + 1], &lo)) {
            free(bytes);
            return LIS_STATUS_FORMAT;
        }
        bytes[i] = (char)((hi << 4) | lo);
    }
    bytes[byte_count] = '\0';

    *out_bytes = bytes;
    *out_len = byte_count;
    return LIS_STATUS_OK;
}

/* --- Line-oriented parsing helpers --- */

static const char *lis_vocab_skip_line(const char *p, const char *end)
{
    while (p < end && *p != '\n') {
        ++p;
    }
    if (p < end) {
        ++p;
    }
    return p;
}

static size_t lis_vocab_line_len(const char *p, const char *end)
{
    const char *start = p;

    while (p < end && *p != '\n' && *p != '\r') {
        ++p;
    }
    return (size_t)(p - start);
}

static lis_status lis_vocab_parse_size(const char *text, size_t len,
                                       size_t *out)
{
    size_t value = 0;
    size_t i;

    if (len == 0) {
        return LIS_STATUS_FORMAT;
    }
    for (i = 0; i < len; ++i) {
        size_t digit;

        if (!isdigit((unsigned char)text[i])) {
            return LIS_STATUS_FORMAT;
        }
        digit = (size_t)(text[i] - '0');
        if (value > (SIZE_MAX - digit) / 10) {
            return LIS_STATUS_OVERFLOW;
        }
        value = value * 10 + digit;
    }
    *out = value;
    return LIS_STATUS_OK;
}

static const char *lis_vocab_find_space(const char *p, size_t len)
{
    size_t i;

    for (i = 0; i < len; ++i) {
        if (p[i] == ' ') {
            return p + i;
        }
    }
    return NULL;
}

/* --- Main parser --- */

lis_status lis_vocab_parse(const char *data, size_t data_len,
                           lis_tokenizer *out)
{
    const char *cursor;
    const char *end;
    size_t line_len;
    size_t vocab_size = 0;
    size_t merge_count = 0;
    size_t i;
    lis_status status;

    if (data == NULL || data_len == 0 || out == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));

    cursor = data;
    end = data + data_len;

    /* Line 1: magic header */
    line_len = lis_vocab_line_len(cursor, end);
    if (line_len < 12 || memcmp(cursor, "LIS_VOCAB_V1", 12) != 0) {
        return LIS_STATUS_FORMAT;
    }
    cursor = lis_vocab_skip_line(cursor, end);

    /* Line 2: "<vocab_size> <merge_count>" */
    line_len = lis_vocab_line_len(cursor, end);
    {
        const char *space = lis_vocab_find_space(cursor, line_len);

        if (space == NULL) {
            return LIS_STATUS_FORMAT;
        }
        status = lis_vocab_parse_size(cursor, (size_t)(space - cursor),
                                       &vocab_size);
        if (status != LIS_STATUS_OK) {
            return status;
        }
        status = lis_vocab_parse_size(
            space + 1, line_len - (size_t)(space - cursor) - 1, &merge_count);
        if (status != LIS_STATUS_OK) {
            return status;
        }
    }
    cursor = lis_vocab_skip_line(cursor, end);

    if (vocab_size == 0) {
        return LIS_STATUS_FORMAT;
    }

    /* Allocate vocabulary arrays. */
    out->token_bytes = calloc(vocab_size, sizeof(*out->token_bytes));
    out->token_lens = calloc(vocab_size, sizeof(*out->token_lens));
    if (out->token_bytes == NULL || out->token_lens == NULL) {
        lis_tokenizer_destroy(out);
        return LIS_STATUS_NO_MEMORY;
    }
    out->vocab_size = vocab_size;

    /* Parse hex-encoded token lines. */
    for (i = 0; i < vocab_size; ++i) {
        if (cursor >= end) {
            lis_tokenizer_destroy(out);
            return LIS_STATUS_FORMAT;
        }
        line_len = lis_vocab_line_len(cursor, end);
        while (line_len > 0 && (cursor[line_len - 1] == ' ' ||
               cursor[line_len - 1] == '\t' ||
               cursor[line_len - 1] == '\r')) {
            --line_len;
        }
        status = lis_vocab_decode_hex(cursor, line_len,
                                       &out->token_bytes[i],
                                       &out->token_lens[i]);
        if (status != LIS_STATUS_OK) {
            lis_tokenizer_destroy(out);
            return status;
        }
        cursor = lis_vocab_skip_line(cursor, end);
    }

    /* Parse merge rules. */
    if (merge_count > 0) {
        size_t table_cap = merge_count * 2;

        if (table_cap < 16) {
            table_cap = 16;
        }
        status = lis_merge_table_init(&out->merges, table_cap);
        if (status != LIS_STATUS_OK) {
            lis_tokenizer_destroy(out);
            return status;
        }

        for (i = 0; i < merge_count; ++i) {
            size_t first_id;
            size_t second_id;
            size_t result_id;
            const char *sp1;
            const char *sp2;

            if (cursor >= end) {
                lis_tokenizer_destroy(out);
                return LIS_STATUS_FORMAT;
            }
            line_len = lis_vocab_line_len(cursor, end);

            sp1 = lis_vocab_find_space(cursor, line_len);
            if (sp1 == NULL) {
                lis_tokenizer_destroy(out);
                return LIS_STATUS_FORMAT;
            }
            sp2 = lis_vocab_find_space(sp1 + 1,
                                        line_len - (size_t)(sp1 - cursor) - 1);
            if (sp2 == NULL) {
                lis_tokenizer_destroy(out);
                return LIS_STATUS_FORMAT;
            }

            status = lis_vocab_parse_size(cursor, (size_t)(sp1 - cursor),
                                           &first_id);
            if (status != LIS_STATUS_OK) {
                lis_tokenizer_destroy(out);
                return status;
            }
            status = lis_vocab_parse_size(sp1 + 1, (size_t)(sp2 - sp1 - 1),
                                           &second_id);
            if (status != LIS_STATUS_OK) {
                lis_tokenizer_destroy(out);
                return status;
            }
            status = lis_vocab_parse_size(
                sp2 + 1, line_len - (size_t)(sp2 - cursor) - 1, &result_id);
            if (status != LIS_STATUS_OK) {
                lis_tokenizer_destroy(out);
                return status;
            }

            if (first_id >= vocab_size || second_id >= vocab_size ||
                result_id >= vocab_size) {
                lis_tokenizer_destroy(out);
                return LIS_STATUS_FORMAT;
            }

            status = lis_merge_table_insert(&out->merges, first_id, second_id,
                                             result_id, i);
            if (status != LIS_STATUS_OK) {
                lis_tokenizer_destroy(out);
                return status;
            }
            cursor = lis_vocab_skip_line(cursor, end);
        }
    }

    /* Build byte-to-token lookup from single-byte vocabulary entries. */
    memset(out->byte_to_token_valid, 0, sizeof(out->byte_to_token_valid));
    for (i = 0; i < vocab_size; ++i) {
        if (out->token_lens[i] == 1) {
            unsigned char bval = (unsigned char)out->token_bytes[i][0];

            if (!out->byte_to_token_valid[bval]) {
                out->byte_to_token[bval] = i;
                out->byte_to_token_valid[bval] = 1;
            }
        }
    }

    return LIS_STATUS_OK;
}
