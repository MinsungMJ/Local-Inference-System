#include "lis/backend.h"

#include <stddef.h>

static lis_status lis_cpu_reference_add(const lis_operator_binary *request)
{
    const float *lhs = request->lhs.data;
    const float *rhs = request->rhs.data;
    float *out = request->out->storage.data;
    size_t count = 0;
    size_t index;
    lis_status status;

    status = lis_tensor_shape_element_count(&request->lhs.shape, &count);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    for (index = 0; index < count; ++index) {
        out[index] = lhs[index] + rhs[index];
    }

    return LIS_STATUS_OK;
}

static lis_status lis_cpu_reference_matmul(const lis_operator_matmul *request)
{
    const float *lhs = request->lhs.data;
    const float *rhs = request->rhs.data;
    float *out = request->out->storage.data;
    const size_t rows = request->lhs.shape.dims[0];
    const size_t inner = request->lhs.shape.dims[1];
    const size_t cols = request->rhs.shape.dims[1];
    size_t row;

    for (row = 0; row < rows; ++row) {
        size_t col;

        for (col = 0; col < cols; ++col) {
            float sum = 0.0f;
            size_t inner_index;

            for (inner_index = 0; inner_index < inner; ++inner_index) {
                sum += lhs[row * inner + inner_index] *
                       rhs[inner_index * cols + col];
            }
            out[row * cols + col] = sum;
        }
    }

    return LIS_STATUS_OK;
}

static lis_status lis_cpu_reference_execute(const lis_backend *backend,
                                            const lis_operator_request *request)
{
    (void)backend;

    switch (request->kind) {
    case LIS_OPERATOR_ADD:
        return lis_cpu_reference_add(&request->args.binary);
    case LIS_OPERATOR_MATMUL:
        return lis_cpu_reference_matmul(&request->args.matmul);
    }

    return LIS_STATUS_UNSUPPORTED;
}

const lis_backend *lis_backend_cpu_reference(void)
{
    static const lis_backend backend = {
        .kind = LIS_BACKEND_KIND_CPU_REFERENCE,
        .name = "cpu_reference",
        .memory_domain = LIS_BACKEND_MEMORY_HOST,
        .execute = lis_cpu_reference_execute,
    };

    return &backend;
}
