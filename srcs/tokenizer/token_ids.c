#include "lis/tokenizer.h"

#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static lis_status lis_token_file_read(const char *path, char **out_data,
                                      size_t *out_len)
{
    FILE *fp = NULL;
    long file_size = 0;
    char *data = NULL;
    lis_status status = LIS_STATUS_IO;

    if (path == NULL || out_data == NULL || out_len == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    fp = fopen(path, "rb");
    if (fp == NULL) {
        return LIS_STATUS_IO;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        goto out;
    }
    file_size = ftell(fp);
    if (file_size <= 0) {
        status = LIS_STATUS_FORMAT;
        goto out;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        goto out;
    }

    data = malloc((size_t)file_size + 1U);
    if (data == NULL) {
        status = LIS_STATUS_NO_MEMORY;
        goto out;
    }
    if (fread(data, 1, (size_t)file_size, fp) != (size_t)file_size) {
        goto out;
    }
    data[(size_t)file_size] = '\0';

    *out_data = data;
    *out_len = (size_t)file_size;
    data = NULL;
    status = LIS_STATUS_OK;

out:
    free(data);
    if (fp != NULL && fclose(fp) != 0 && status == LIS_STATUS_OK) {
        status = LIS_STATUS_IO;
    }
    return status;
}

static lis_status lis_token_batch_push(lis_token_id_batch *batch,
                                       size_t token)
{
    size_t new_count = 0;
    size_t *new_tokens = NULL;

    if (batch == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (batch->token_count == SIZE_MAX) {
        return LIS_STATUS_OVERFLOW;
    }
    new_count = batch->token_count + 1U;
    if (new_count > SIZE_MAX / sizeof(*new_tokens)) {
        return LIS_STATUS_OVERFLOW;
    }

    new_tokens = realloc(batch->tokens, new_count * sizeof(*new_tokens));
    if (new_tokens == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }
    batch->tokens = new_tokens;
    batch->tokens[batch->token_count] = token;
    batch->token_count = new_count;
    return LIS_STATUS_OK;
}

static lis_status lis_token_parse_number(const char *data, size_t len,
                                         size_t *index, size_t *out_token)
{
    size_t value = 0;

    if (data == NULL || index == NULL || out_token == NULL ||
        *index >= len || !isdigit((unsigned char)data[*index])) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    while (*index < len && isdigit((unsigned char)data[*index])) {
        const size_t digit = (size_t)(data[*index] - '0');

        if (value > (SIZE_MAX - digit) / 10U) {
            return LIS_STATUS_OVERFLOW;
        }
        value = value * 10U + digit;
        ++(*index);
    }

    *out_token = value;
    return LIS_STATUS_OK;
}

static lis_status lis_token_finish_sequence(lis_token_id_batch *batch,
                                            size_t expected_batch_size,
                                            size_t *sequence_index,
                                            size_t sequence_len)
{
    if (batch == NULL || sequence_index == NULL || sequence_len == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (*sequence_index >= expected_batch_size) {
        return LIS_STATUS_SHAPE_MISMATCH;
    }

    batch->lengths[*sequence_index] = sequence_len;
    ++(*sequence_index);
    return LIS_STATUS_OK;
}

static lis_status lis_token_parse_batch(const char *data, size_t len,
                                        size_t expected_batch_size,
                                        lis_token_id_batch *out_batch)
{
    lis_token_id_batch batch = { 0 };
    size_t sequence_index = 0;
    size_t sequence_len = 0;
    size_t index = 0;
    lis_status status = LIS_STATUS_OK;

    if (data == NULL || out_batch == NULL || expected_batch_size == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (expected_batch_size > SIZE_MAX / sizeof(*batch.lengths)) {
        return LIS_STATUS_OVERFLOW;
    }

    batch.lengths = calloc(expected_batch_size, sizeof(*batch.lengths));
    if (batch.lengths == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }
    batch.batch_size = expected_batch_size;

    while (index < len) {
        const unsigned char ch = (unsigned char)data[index];

        if (ch == '\n') {
            if (sequence_len != 0) {
                status = lis_token_finish_sequence(&batch,
                                                   expected_batch_size,
                                                   &sequence_index,
                                                   sequence_len);
                if (status != LIS_STATUS_OK) {
                    goto fail;
                }
                sequence_len = 0;
            }
            ++index;
            continue;
        }
        if (ch == '\r' || ch == ' ' || ch == '\t' || ch == '\v' ||
            ch == '\f') {
            ++index;
            continue;
        }
        if (isdigit(ch)) {
            size_t token = 0;

            status = lis_token_parse_number(data, len, &index, &token);
            if (status != LIS_STATUS_OK) {
                goto fail;
            }
            status = lis_token_batch_push(&batch, token);
            if (status != LIS_STATUS_OK) {
                goto fail;
            }
            ++sequence_len;
            continue;
        }

        status = LIS_STATUS_FORMAT;
        goto fail;
    }

    if (sequence_len != 0) {
        status = lis_token_finish_sequence(&batch, expected_batch_size,
                                           &sequence_index, sequence_len);
        if (status != LIS_STATUS_OK) {
            goto fail;
        }
    }
    if (sequence_index != expected_batch_size) {
        status = LIS_STATUS_SHAPE_MISMATCH;
        goto fail;
    }

    *out_batch = batch;
    return LIS_STATUS_OK;

fail:
    lis_token_id_batch_destroy(&batch);
    return status;
}

lis_status lis_token_id_batch_load_path(const char *path,
                                        size_t expected_batch_size,
                                        lis_token_id_batch *out_batch)
{
    char *data = NULL;
    size_t len = 0;
    lis_status status;

    if (out_batch == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    memset(out_batch, 0, sizeof(*out_batch));

    status = lis_token_file_read(path, &data, &len);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_token_parse_batch(data, len, expected_batch_size, out_batch);
    free(data);
    return status;
}

lis_status lis_token_id_batch_validate_vocab(const lis_token_id_batch *batch,
                                             size_t vocab_size)
{
    size_t index;
    size_t length_sum = 0;

    if (batch == NULL || batch->tokens == NULL || batch->lengths == NULL ||
        batch->batch_size == 0 || vocab_size == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    for (index = 0; index < batch->batch_size; ++index) {
        if (batch->lengths[index] == 0) {
            return LIS_STATUS_INVALID_ARGUMENT;
        }
        if (length_sum > SIZE_MAX - batch->lengths[index]) {
            return LIS_STATUS_OVERFLOW;
        }
        length_sum += batch->lengths[index];
    }
    if (length_sum != batch->token_count) {
        return LIS_STATUS_SHAPE_MISMATCH;
    }

    for (index = 0; index < batch->token_count; ++index) {
        if (batch->tokens[index] >= vocab_size) {
            return LIS_STATUS_LIMIT_EXCEEDED;
        }
    }

    return LIS_STATUS_OK;
}

void lis_token_id_batch_destroy(lis_token_id_batch *batch)
{
    if (batch == NULL) {
        return;
    }

    free(batch->tokens);
    free(batch->lengths);
    memset(batch, 0, sizeof(*batch));
}
