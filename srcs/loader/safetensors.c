#include "lis/loader.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char *cur;
    const char *end;
} lis_json_cursor;

static void lis_json_skip_ws(lis_json_cursor *cursor)
{
    while (cursor->cur < cursor->end &&
           (*cursor->cur == ' ' || *cursor->cur == '\n' ||
            *cursor->cur == '\r' || *cursor->cur == '\t')) {
        ++cursor->cur;
    }
}

static lis_status lis_json_expect(lis_json_cursor *cursor, char expected)
{
    lis_json_skip_ws(cursor);
    if (cursor->cur >= cursor->end || *cursor->cur != expected) {
        return LIS_STATUS_FORMAT;
    }
    ++cursor->cur;
    return LIS_STATUS_OK;
}

static lis_status lis_json_parse_string(lis_json_cursor *cursor, char **out)
{
    const char *start = NULL;
    size_t len = 0;
    char *copy = NULL;

    if (out == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    *out = NULL;
    lis_json_skip_ws(cursor);
    if (cursor->cur >= cursor->end || *cursor->cur != '"') {
        return LIS_STATUS_FORMAT;
    }

    ++cursor->cur;
    start = cursor->cur;
    while (cursor->cur < cursor->end && *cursor->cur != '"') {
        if (*cursor->cur == '\\') {
            return LIS_STATUS_FORMAT;
        }
        ++cursor->cur;
    }
    if (cursor->cur >= cursor->end) {
        return LIS_STATUS_FORMAT;
    }

    len = (size_t)(cursor->cur - start);
    copy = malloc(len + 1);
    if (copy == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }
    memcpy(copy, start, len);
    copy[len] = '\0';
    ++cursor->cur;
    *out = copy;
    return LIS_STATUS_OK;
}

static lis_status lis_json_parse_size(lis_json_cursor *cursor, size_t *out)
{
    size_t value = 0;

    if (out == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    lis_json_skip_ws(cursor);
    if (cursor->cur >= cursor->end || *cursor->cur < '0' ||
        *cursor->cur > '9') {
        return LIS_STATUS_FORMAT;
    }

    while (cursor->cur < cursor->end && *cursor->cur >= '0' &&
           *cursor->cur <= '9') {
        const size_t digit = (size_t)(*cursor->cur - '0');

        if (value > (SIZE_MAX - digit) / 10U) {
            return LIS_STATUS_OVERFLOW;
        }
        value = value * 10U + digit;
        ++cursor->cur;
    }

    *out = value;
    return LIS_STATUS_OK;
}

static lis_status lis_json_skip_value(lis_json_cursor *cursor)
{
    size_t depth = 0;
    int in_string = 0;

    lis_json_skip_ws(cursor);
    if (cursor->cur >= cursor->end) {
        return LIS_STATUS_FORMAT;
    }

    do {
        const char ch = *cursor->cur;

        if (in_string) {
            if (ch == '\\') {
                return LIS_STATUS_FORMAT;
            }
            if (ch == '"') {
                in_string = 0;
            }
            ++cursor->cur;
            continue;
        }

        if (ch == '"') {
            in_string = 1;
            ++cursor->cur;
            continue;
        }
        if (ch == '{' || ch == '[') {
            ++depth;
            ++cursor->cur;
            continue;
        }
        if (ch == '}' || ch == ']') {
            if (depth == 0) {
                return LIS_STATUS_FORMAT;
            }
            --depth;
            ++cursor->cur;
            if (depth == 0) {
                return LIS_STATUS_OK;
            }
            continue;
        }
        if (depth == 0 && (ch == ',' || ch == '}')) {
            return LIS_STATUS_OK;
        }
        ++cursor->cur;
    } while (cursor->cur < cursor->end);

    return depth == 0 ? LIS_STATUS_OK : LIS_STATUS_FORMAT;
}

static lis_status lis_parse_safetensors_dtype(const char *name,
                                              lis_dtype *out_dtype)
{
    if (strcmp(name, "F32") == 0) {
        *out_dtype = LIS_DTYPE_F32;
        return LIS_STATUS_OK;
    }
    if (strcmp(name, "F16") == 0) {
        *out_dtype = LIS_DTYPE_F16;
        return LIS_STATUS_OK;
    }
    if (strcmp(name, "BF16") == 0) {
        *out_dtype = LIS_DTYPE_BF16;
        return LIS_STATUS_OK;
    }
    if (strcmp(name, "I32") == 0) {
        *out_dtype = LIS_DTYPE_I32;
        return LIS_STATUS_OK;
    }
    if (strcmp(name, "U32") == 0) {
        *out_dtype = LIS_DTYPE_U32;
        return LIS_STATUS_OK;
    }

    return LIS_STATUS_UNSUPPORTED_DTYPE;
}

static lis_status lis_json_parse_shape(lis_json_cursor *cursor,
                                       lis_tensor_shape *out_shape)
{
    size_t dims[LIS_TENSOR_MAX_RANK] = { 0 };
    size_t rank = 0;
    lis_status status;

    status = lis_json_expect(cursor, '[');
    if (status != LIS_STATUS_OK) {
        return status;
    }
    lis_json_skip_ws(cursor);
    if (cursor->cur < cursor->end && *cursor->cur == ']') {
        return LIS_STATUS_UNSUPPORTED_SHAPE;
    }

    for (;;) {
        if (rank >= LIS_TENSOR_MAX_RANK) {
            return LIS_STATUS_UNSUPPORTED_SHAPE;
        }
        status = lis_json_parse_size(cursor, &dims[rank]);
        if (status != LIS_STATUS_OK) {
            return status;
        }
        ++rank;

        lis_json_skip_ws(cursor);
        if (cursor->cur >= cursor->end) {
            return LIS_STATUS_FORMAT;
        }
        if (*cursor->cur == ']') {
            ++cursor->cur;
            break;
        }
        if (*cursor->cur != ',') {
            return LIS_STATUS_FORMAT;
        }
        ++cursor->cur;
    }

    status = lis_tensor_shape_make(rank, dims, out_shape);
    if (status == LIS_STATUS_INVALID_ARGUMENT) {
        return LIS_STATUS_UNSUPPORTED_SHAPE;
    }
    return status;
}

static lis_status lis_json_parse_offsets(lis_json_cursor *cursor,
                                         size_t *out_begin, size_t *out_end)
{
    lis_status status;

    status = lis_json_expect(cursor, '[');
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_json_parse_size(cursor, out_begin);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_json_expect(cursor, ',');
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_json_parse_size(cursor, out_end);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    return lis_json_expect(cursor, ']');
}

static lis_status lis_loaded_model_push_tensor(lis_loaded_model *model,
                                              lis_loaded_tensor *tensor)
{
    lis_loaded_tensor *new_tensors = NULL;
    const size_t new_count = model->tensor_count + 1U;

    if (new_count > SIZE_MAX / sizeof(*new_tensors)) {
        return LIS_STATUS_OVERFLOW;
    }

    new_tensors = realloc(model->tensors, new_count * sizeof(*new_tensors));
    if (new_tensors == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }

    model->tensors = new_tensors;
    model->tensors[model->tensor_count] = *tensor;
    model->tensor_count = new_count;
    memset(tensor, 0, sizeof(*tensor));
    return LIS_STATUS_OK;
}

static lis_status lis_parse_tensor_entry(lis_json_cursor *cursor,
                                         const char *name,
                                         unsigned char *data_base,
                                         size_t data_size,
                                         lis_loaded_model *model)
{
    lis_loaded_tensor tensor = { 0 };
    lis_tensor_shape shape = { 0 };
    lis_dtype dtype = LIS_DTYPE_INVALID;
    size_t begin = 0;
    size_t end = 0;
    size_t expected_bytes = 0;
    int has_dtype = 0;
    int has_shape = 0;
    int has_offsets = 0;
    lis_status status;

    status = lis_json_expect(cursor, '{');
    if (status != LIS_STATUS_OK) {
        return status;
    }

    for (;;) {
        char *field = NULL;

        lis_json_skip_ws(cursor);
        if (cursor->cur < cursor->end && *cursor->cur == '}') {
            ++cursor->cur;
            break;
        }

        status = lis_json_parse_string(cursor, &field);
        if (status != LIS_STATUS_OK) {
            return status;
        }
        status = lis_json_expect(cursor, ':');
        if (status != LIS_STATUS_OK) {
            free(field);
            return status;
        }

        if (strcmp(field, "dtype") == 0) {
            char *dtype_name = NULL;

            status = lis_json_parse_string(cursor, &dtype_name);
            if (status == LIS_STATUS_OK) {
                status = lis_parse_safetensors_dtype(dtype_name, &dtype);
            }
            free(dtype_name);
            has_dtype = status == LIS_STATUS_OK;
        } else if (strcmp(field, "shape") == 0) {
            status = lis_json_parse_shape(cursor, &shape);
            has_shape = status == LIS_STATUS_OK;
        } else if (strcmp(field, "data_offsets") == 0) {
            status = lis_json_parse_offsets(cursor, &begin, &end);
            has_offsets = status == LIS_STATUS_OK;
        } else {
            status = lis_json_skip_value(cursor);
        }

        free(field);
        if (status != LIS_STATUS_OK) {
            return status;
        }

        lis_json_skip_ws(cursor);
        if (cursor->cur >= cursor->end) {
            return LIS_STATUS_FORMAT;
        }
        if (*cursor->cur == ',') {
            ++cursor->cur;
            continue;
        }
        if (*cursor->cur == '}') {
            ++cursor->cur;
            break;
        }
        return LIS_STATUS_FORMAT;
    }

    if (!has_dtype || !has_shape || !has_offsets) {
        return LIS_STATUS_FORMAT;
    }
    if (end < begin || end > data_size) {
        return LIS_STATUS_FORMAT;
    }

    status = lis_tensor_shape_byte_size(&shape, dtype, &expected_bytes);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (expected_bytes != end - begin) {
        return LIS_STATUS_SHAPE_MISMATCH;
    }

    tensor.name = malloc(strlen(name) + 1U);
    if (tensor.name == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }
    strcpy(tensor.name, name);
    tensor.data_begin = begin;
    tensor.data_end = end;
    status = lis_tensor_view_from_borrowed(dtype, &shape, data_base + begin,
                                           end - begin, &tensor.view);
    if (status != LIS_STATUS_OK) {
        free(tensor.name);
        return status;
    }

    return lis_loaded_model_push_tensor(model, &tensor);
}

static uint64_t lis_read_u64_le(const unsigned char bytes[8])
{
    uint64_t value = 0;
    size_t index;

    for (index = 0; index < 8; ++index) {
        value |= ((uint64_t)bytes[index]) << (index * 8);
    }

    return value;
}

static lis_status lis_read_file(const char *path, unsigned char **out_data,
                                size_t *out_size)
{
    FILE *fp = NULL;
    long file_size = 0;
    unsigned char *data = NULL;
    lis_status status = LIS_STATUS_IO;

    if (path == NULL || out_data == NULL || out_size == NULL) {
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

    data = malloc((size_t)file_size);
    if (data == NULL) {
        status = LIS_STATUS_NO_MEMORY;
        goto out;
    }
    if (fread(data, 1, (size_t)file_size, fp) != (size_t)file_size) {
        goto out;
    }

    *out_data = data;
    *out_size = (size_t)file_size;
    data = NULL;
    status = LIS_STATUS_OK;

out:
    free(data);
    if (fp != NULL && fclose(fp) != 0 && status == LIS_STATUS_OK) {
        status = LIS_STATUS_IO;
    }
    return status;
}

lis_status lis_loader_load_safetensors(const lis_model_source *source,
                                       lis_loaded_model *out_model)
{
    lis_loaded_model model = { 0 };
    lis_json_cursor cursor = { 0 };
    uint64_t header_len64 = 0;
    size_t header_len = 0;
    size_t data_offset = 0;
    lis_status status;

    if (source == NULL || out_model == NULL || source->path == NULL ||
        source->kind != LIS_MODEL_SOURCE_PATH) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    status = lis_read_file(source->path, &model.artifact_data,
                           &model.artifact_size);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (model.artifact_size < 8U) {
        status = LIS_STATUS_FORMAT;
        goto fail;
    }

    header_len64 = lis_read_u64_le(model.artifact_data);
    if (header_len64 == 0 || header_len64 > SIZE_MAX ||
        header_len64 > 100 * 1024 * 1024 ||
        header_len64 > model.artifact_size - 8U) {
        status = LIS_STATUS_FORMAT;
        goto fail;
    }

    header_len = (size_t)header_len64;
    data_offset = 8U + header_len;
    cursor.cur = (const char *)model.artifact_data + 8U;
    cursor.end = cursor.cur + header_len;
    model.format = LIS_MODEL_FORMAT_SAFETENSORS;

    status = lis_json_expect(&cursor, '{');
    if (status != LIS_STATUS_OK) {
        goto fail;
    }

    for (;;) {
        char *name = NULL;

        lis_json_skip_ws(&cursor);
        if (cursor.cur < cursor.end && *cursor.cur == '}') {
            ++cursor.cur;
            break;
        }

        status = lis_json_parse_string(&cursor, &name);
        if (status != LIS_STATUS_OK) {
            goto fail;
        }
        status = lis_json_expect(&cursor, ':');
        if (status != LIS_STATUS_OK) {
            free(name);
            goto fail;
        }

        if (strcmp(name, "__metadata__") == 0) {
            status = lis_json_skip_value(&cursor);
        } else {
            status = lis_parse_tensor_entry(&cursor, name,
                                            model.artifact_data + data_offset,
                                            model.artifact_size - data_offset,
                                            &model);
        }
        free(name);
        if (status != LIS_STATUS_OK) {
            goto fail;
        }

        lis_json_skip_ws(&cursor);
        if (cursor.cur >= cursor.end) {
            status = LIS_STATUS_FORMAT;
            goto fail;
        }
        if (*cursor.cur == ',') {
            ++cursor.cur;
            continue;
        }
        if (*cursor.cur == '}') {
            ++cursor.cur;
            break;
        }
        status = LIS_STATUS_FORMAT;
        goto fail;
    }

    if (model.tensor_count == 0) {
        status = LIS_STATUS_FORMAT;
        goto fail;
    }

    *out_model = model;
    return LIS_STATUS_OK;

fail:
    lis_loaded_model_destroy(&model);
    return status;
}
