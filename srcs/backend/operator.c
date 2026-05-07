#include "lis/backend.h"

static int lis_shapes_equal(const lis_tensor_shape *lhs,
                            const lis_tensor_shape *rhs)
{
    size_t index;

    if (lhs == NULL || rhs == NULL || lhs->rank != rhs->rank) {
        return 0;
    }

    for (index = 0; index < lhs->rank; ++index) {
        if (lhs->dims[index] != rhs->dims[index]) {
            return 0;
        }
    }

    return 1;
}

static lis_status lis_validate_view(const lis_tensor_view *view)
{
    size_t expected_bytes = 0;
    lis_status status;

    if (view == NULL || view->data == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    status = lis_tensor_shape_byte_size(&view->shape, view->dtype,
                                        &expected_bytes);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (view->byte_len < expected_bytes) {
        return LIS_STATUS_SHAPE_MISMATCH;
    }

    return LIS_STATUS_OK;
}

static lis_status lis_validate_output(const lis_tensor *out)
{
    size_t expected_bytes = 0;
    lis_status status;

    if (out == NULL || out->storage.data == NULL ||
        out->storage.kind == LIS_TENSOR_STORAGE_NONE) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    status = lis_tensor_shape_byte_size(&out->shape, out->dtype,
                                        &expected_bytes);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (out->storage.byte_len < expected_bytes) {
        return LIS_STATUS_SHAPE_MISMATCH;
    }

    return LIS_STATUS_OK;
}

static lis_status lis_validate_add_request(const lis_operator_binary *request)
{
    lis_status status;

    if (request == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    status = lis_validate_view(&request->lhs);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_validate_view(&request->rhs);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_validate_output(request->out);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (request->lhs.dtype != LIS_DTYPE_F32 ||
        request->rhs.dtype != LIS_DTYPE_F32 ||
        request->out->dtype != LIS_DTYPE_F32) {
        return LIS_STATUS_UNSUPPORTED_DTYPE;
    }
    if (!lis_shapes_equal(&request->lhs.shape, &request->rhs.shape) ||
        !lis_shapes_equal(&request->lhs.shape, &request->out->shape)) {
        return LIS_STATUS_SHAPE_MISMATCH;
    }

    return LIS_STATUS_OK;
}

static lis_status lis_validate_matmul_request(const lis_operator_matmul *request)
{
    lis_status status;

    if (request == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    status = lis_validate_view(&request->lhs);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_validate_view(&request->rhs);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    status = lis_validate_output(request->out);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (request->lhs.dtype != LIS_DTYPE_F32 ||
        request->rhs.dtype != LIS_DTYPE_F32 ||
        request->out->dtype != LIS_DTYPE_F32) {
        return LIS_STATUS_UNSUPPORTED_DTYPE;
    }
    if (request->lhs.shape.rank != 2 || request->rhs.shape.rank != 2 ||
        request->out->shape.rank != 2) {
        return LIS_STATUS_UNSUPPORTED_SHAPE;
    }
    if (request->lhs.shape.dims[1] != request->rhs.shape.dims[0] ||
        request->out->shape.dims[0] != request->lhs.shape.dims[0] ||
        request->out->shape.dims[1] != request->rhs.shape.dims[1]) {
        return LIS_STATUS_SHAPE_MISMATCH;
    }

    return LIS_STATUS_OK;
}

const char *lis_backend_kind_name(lis_backend_kind kind)
{
    switch (kind) {
    case LIS_BACKEND_KIND_CPU_REFERENCE:
        return "cpu_reference";
    case LIS_BACKEND_KIND_SIMD_RESERVED:
        return "simd_reserved";
    case LIS_BACKEND_KIND_GPU_RESERVED:
        return "gpu_reserved";
    }

    return "unknown_backend";
}

const char *lis_operator_kind_name(lis_operator_kind kind)
{
    switch (kind) {
    case LIS_OPERATOR_ADD:
        return "add";
    case LIS_OPERATOR_MATMUL:
        return "matmul";
    }

    return "unknown_operator";
}

lis_operator_request lis_operator_make_add(lis_tensor_view lhs,
                                           lis_tensor_view rhs,
                                           lis_tensor *out)
{
    lis_operator_request request = {
        .kind = LIS_OPERATOR_ADD,
        .args.binary = {
            .lhs = lhs,
            .rhs = rhs,
            .out = out,
        },
    };

    return request;
}

lis_operator_request lis_operator_make_matmul(lis_tensor_view lhs,
                                              lis_tensor_view rhs,
                                              lis_tensor *out)
{
    lis_operator_request request = {
        .kind = LIS_OPERATOR_MATMUL,
        .args.matmul = {
            .lhs = lhs,
            .rhs = rhs,
            .out = out,
        },
    };

    return request;
}

lis_status lis_operator_execute(const lis_backend *backend,
                                const lis_operator_request *request)
{
    lis_status status;

    if (backend == NULL || request == NULL || backend->execute == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (backend->memory_domain != LIS_BACKEND_MEMORY_HOST) {
        return LIS_STATUS_UNSUPPORTED;
    }

    switch (request->kind) {
    case LIS_OPERATOR_ADD:
        status = lis_validate_add_request(&request->args.binary);
        break;
    case LIS_OPERATOR_MATMUL:
        status = lis_validate_matmul_request(&request->args.matmul);
        break;
    default:
        return LIS_STATUS_UNSUPPORTED;
    }
    if (status != LIS_STATUS_OK) {
        return status;
    }

    return backend->execute(backend, request);
}
