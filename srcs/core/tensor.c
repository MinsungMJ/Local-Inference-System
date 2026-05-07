#include "lis/tensor.h"

#include <stdint.h>
#include <string.h>

static int lis_mul_size_overflows(size_t lhs, size_t rhs, size_t *out)
{
    if (out == NULL) {
        return 1;
    }
    if (lhs != 0 && rhs > SIZE_MAX / lhs) {
        return 1;
    }

    *out = lhs * rhs;
    return 0;
}

lis_status lis_tensor_shape_make(size_t rank, const size_t *dims,
                                 lis_tensor_shape *out_shape)
{
    lis_tensor_shape shape = { 0 };
    size_t stride = 1;
    size_t index;

    if (out_shape == NULL || dims == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (rank == 0 || rank > LIS_TENSOR_MAX_RANK) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    shape.rank = rank;
    for (index = 0; index < rank; ++index) {
        if (dims[index] == 0) {
            return LIS_STATUS_INVALID_ARGUMENT;
        }
        shape.dims[index] = dims[index];
    }

    for (index = rank; index > 0; --index) {
        const size_t dim_index = index - 1;
        size_t next_stride = 0;

        shape.strides[dim_index] = stride;
        if (lis_mul_size_overflows(stride, shape.dims[dim_index],
                                   &next_stride)) {
            return LIS_STATUS_OVERFLOW;
        }
        stride = next_stride;
    }

    *out_shape = shape;
    return LIS_STATUS_OK;
}

lis_status lis_tensor_shape_element_count(const lis_tensor_shape *shape,
                                          size_t *out_count)
{
    size_t count = 1;
    size_t index;

    if (shape == NULL || out_count == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (shape->rank == 0 || shape->rank > LIS_TENSOR_MAX_RANK) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    for (index = 0; index < shape->rank; ++index) {
        if (shape->dims[index] == 0) {
            return LIS_STATUS_INVALID_ARGUMENT;
        }
        if (lis_mul_size_overflows(count, shape->dims[index], &count)) {
            return LIS_STATUS_OVERFLOW;
        }
    }

    *out_count = count;
    return LIS_STATUS_OK;
}

lis_status lis_tensor_shape_byte_size(const lis_tensor_shape *shape,
                                      lis_dtype dtype, size_t *out_size)
{
    size_t element_count = 0;
    size_t dtype_size = 0;
    lis_status status;

    if (out_size == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    status = lis_tensor_shape_element_count(shape, &element_count);
    if (status != LIS_STATUS_OK) {
        *out_size = 0;
        return status;
    }

    status = lis_dtype_size_bytes(dtype, &dtype_size);
    if (status != LIS_STATUS_OK) {
        *out_size = 0;
        return status;
    }

    if (lis_mul_size_overflows(element_count, dtype_size, out_size)) {
        *out_size = 0;
        return LIS_STATUS_OVERFLOW;
    }

    return LIS_STATUS_OK;
}

lis_status lis_tensor_view_from_borrowed(lis_dtype dtype,
                                         const lis_tensor_shape *shape,
                                         const void *data, size_t byte_len,
                                         lis_tensor_view *out_view)
{
    lis_tensor_view view = { 0 };
    size_t required_bytes = 0;
    lis_status status;

    if (shape == NULL || data == NULL || out_view == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    status = lis_tensor_shape_byte_size(shape, dtype, &required_bytes);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (byte_len < required_bytes) {
        return LIS_STATUS_SHAPE_MISMATCH;
    }

    view.dtype = dtype;
    view.shape = *shape;
    view.data = data;
    view.byte_len = byte_len;
    *out_view = view;

    return LIS_STATUS_OK;
}

lis_status lis_tensor_init_borrowed(lis_dtype dtype,
                                    const lis_tensor_shape *shape,
                                    void *data, size_t byte_len,
                                    lis_tensor *out_tensor)
{
    lis_tensor_view view = { 0 };
    lis_tensor tensor = { 0 };
    lis_status status;

    if (out_tensor == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    status = lis_tensor_view_from_borrowed(dtype, shape, data, byte_len, &view);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    tensor.dtype = dtype;
    tensor.shape = view.shape;
    tensor.storage.kind = LIS_TENSOR_STORAGE_BORROWED;
    tensor.storage.data = data;
    tensor.storage.byte_len = byte_len;
    *out_tensor = tensor;

    return LIS_STATUS_OK;
}

lis_status lis_tensor_init_owned(lis_dtype dtype,
                                 const lis_tensor_shape *shape,
                                 void *data, size_t byte_len,
                                 lis_tensor_release_fn release,
                                 void *release_user_data,
                                 lis_tensor *out_tensor)
{
    lis_tensor_view view = { 0 };
    lis_tensor tensor = { 0 };
    lis_status status;

    if (out_tensor == NULL || release == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    status = lis_tensor_view_from_borrowed(dtype, shape, data, byte_len, &view);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    tensor.dtype = dtype;
    tensor.shape = view.shape;
    tensor.storage.kind = LIS_TENSOR_STORAGE_OWNED;
    tensor.storage.data = data;
    tensor.storage.byte_len = byte_len;
    tensor.storage.release = release;
    tensor.storage.release_user_data = release_user_data;
    *out_tensor = tensor;

    return LIS_STATUS_OK;
}

void lis_tensor_destroy(lis_tensor *tensor)
{
    if (tensor == NULL) {
        return;
    }

    if (tensor->storage.kind == LIS_TENSOR_STORAGE_OWNED &&
        tensor->storage.release != NULL) {
        tensor->storage.release(tensor->storage.data,
                                tensor->storage.release_user_data);
    }

    memset(tensor, 0, sizeof(*tensor));
}
