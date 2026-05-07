#ifndef LIS_TENSOR_H
#define LIS_TENSOR_H

#include "lis/dtype.h"
#include "lis/status.h"

#include <stddef.h>

#define LIS_TENSOR_MAX_RANK 8

typedef struct {
    size_t rank;
    size_t dims[LIS_TENSOR_MAX_RANK];
    size_t strides[LIS_TENSOR_MAX_RANK];
} lis_tensor_shape;

typedef enum {
    LIS_TENSOR_STORAGE_NONE = 0,
    LIS_TENSOR_STORAGE_BORROWED,
    LIS_TENSOR_STORAGE_OWNED
} lis_tensor_storage_kind;

typedef void (*lis_tensor_release_fn)(void *data, void *user_data);

typedef struct {
    lis_tensor_storage_kind kind;
    void *data;
    size_t byte_len;
    lis_tensor_release_fn release;
    void *release_user_data;
} lis_tensor_storage;

typedef struct {
    lis_dtype dtype;
    lis_tensor_shape shape;
    lis_tensor_storage storage;
} lis_tensor;

typedef struct {
    lis_dtype dtype;
    lis_tensor_shape shape;
    const void *data;
    size_t byte_len;
} lis_tensor_view;

lis_status lis_tensor_shape_make(size_t rank, const size_t *dims,
                                 lis_tensor_shape *out_shape);
lis_status lis_tensor_shape_element_count(const lis_tensor_shape *shape,
                                          size_t *out_count);
lis_status lis_tensor_shape_byte_size(const lis_tensor_shape *shape,
                                      lis_dtype dtype, size_t *out_size);

lis_status lis_tensor_view_from_borrowed(lis_dtype dtype,
                                         const lis_tensor_shape *shape,
                                         const void *data, size_t byte_len,
                                         lis_tensor_view *out_view);
lis_status lis_tensor_init_borrowed(lis_dtype dtype,
                                    const lis_tensor_shape *shape,
                                    void *data, size_t byte_len,
                                    lis_tensor *out_tensor);
lis_status lis_tensor_init_owned(lis_dtype dtype,
                                 const lis_tensor_shape *shape,
                                 void *data, size_t byte_len,
                                 lis_tensor_release_fn release,
                                 void *release_user_data,
                                 lis_tensor *out_tensor);
void lis_tensor_destroy(lis_tensor *tensor);

#endif
