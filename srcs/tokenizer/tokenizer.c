#include "lis/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static lis_status lis_tokenizer_read_file(const char *path, char **out_data,
                                          size_t *out_len)
{
    FILE *fp = NULL;
    long file_size;
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

lis_status lis_tokenizer_load(const char *path, lis_tokenizer *out)
{
    char *data = NULL;
    size_t data_len = 0;
    lis_status status;

    if (path == NULL || out == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));

    status = lis_tokenizer_read_file(path, &data, &data_len);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    status = lis_vocab_parse(data, data_len, out);
    free(data);
    return status;
}

void lis_tokenizer_destroy(lis_tokenizer *tok)
{
    size_t i;

    if (tok == NULL) {
        return;
    }

    if (tok->token_bytes != NULL) {
        for (i = 0; i < tok->vocab_size; ++i) {
            free(tok->token_bytes[i]);
        }
        free(tok->token_bytes);
    }
    if (tok->special_token_bytes != NULL) {
        for (i = 0; i < tok->special_token_count; ++i) {
            free(tok->special_token_bytes[i]);
        }
        free(tok->special_token_bytes);
    }
    free(tok->token_lens);
    free(tok->special_token_lens);
    free(tok->special_token_ids);
    lis_merge_table_destroy(&tok->merges);
    memset(tok, 0, sizeof(*tok));
}

size_t lis_tokenizer_vocab_size(const lis_tokenizer *tok)
{
    if (tok == NULL) {
        return 0;
    }
    return tok->vocab_size;
}
