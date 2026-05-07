#include "lis/backend.h"
#include "lis/status.h"
#include "lis/tensor.h"

#include <stddef.h>
#include <stdio.h>

static int g_failures;

static void expect_status(const char *name, lis_status actual,
                          lis_status expected)
{
    if (actual != expected) {
        fprintf(stderr, "%s: expected %s, got %s\n", name,
                lis_status_name(expected), lis_status_name(actual));
        ++g_failures;
    }
}

static void expect_float(const char *name, float actual, float expected)
{
    const float diff = actual > expected ? actual - expected : expected - actual;

    if (diff > 0.00001f) {
        fprintf(stderr, "%s: expected %.6f, got %.6f\n", name,
                (double)expected, (double)actual);
        ++g_failures;
    }
}

static lis_tensor_shape make_shape(size_t rank, const size_t *dims)
{
    lis_tensor_shape shape = { 0 };

    expect_status("make test shape", lis_tensor_shape_make(rank, dims, &shape),
                  LIS_STATUS_OK);
    return shape;
}

static lis_tensor_view make_view(lis_tensor_shape shape, const float *data,
                                 size_t count)
{
    lis_tensor_view view = { 0 };

    expect_status("make test view",
                  lis_tensor_view_from_borrowed(LIS_DTYPE_F32, &shape, data,
                                                count * sizeof(*data), &view),
                  LIS_STATUS_OK);
    return view;
}

static lis_tensor make_output(lis_tensor_shape shape, float *data, size_t count)
{
    lis_tensor tensor = { 0 };

    expect_status("make test output",
                  lis_tensor_init_borrowed(LIS_DTYPE_F32, &shape, data,
                                           count * sizeof(*data), &tensor),
                  LIS_STATUS_OK);
    return tensor;
}

static void test_add(void)
{
    const size_t dims[] = { 4 };
    const float lhs[] = { 1.0f, 2.0f, 3.0f, 4.0f };
    const float rhs[] = { 10.0f, 20.0f, 30.0f, 40.0f };
    float out_data[4] = { 0.0f };
    lis_tensor_shape shape = make_shape(1, dims);
    lis_tensor out = make_output(shape, out_data, 4);
    lis_operator_request request =
        lis_operator_make_add(make_view(shape, lhs, 4), make_view(shape, rhs, 4),
                              &out);

    expect_status("cpu add",
                  lis_operator_execute(lis_backend_cpu_reference(), &request),
                  LIS_STATUS_OK);
    expect_float("add 0", out_data[0], 11.0f);
    expect_float("add 1", out_data[1], 22.0f);
    expect_float("add 2", out_data[2], 33.0f);
    expect_float("add 3", out_data[3], 44.0f);
}

static void test_matmul(void)
{
    const size_t lhs_dims[] = { 2, 3 };
    const size_t rhs_dims[] = { 3, 2 };
    const size_t out_dims[] = { 2, 2 };
    const float lhs[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    const float rhs[] = { 7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f };
    float out_data[4] = { 0.0f };
    lis_tensor_shape lhs_shape = make_shape(2, lhs_dims);
    lis_tensor_shape rhs_shape = make_shape(2, rhs_dims);
    lis_tensor_shape out_shape = make_shape(2, out_dims);
    lis_tensor out = make_output(out_shape, out_data, 4);
    lis_operator_request request =
        lis_operator_make_matmul(make_view(lhs_shape, lhs, 6),
                                 make_view(rhs_shape, rhs, 6), &out);

    expect_status("cpu matmul",
                  lis_operator_execute(lis_backend_cpu_reference(), &request),
                  LIS_STATUS_OK);
    expect_float("matmul 0", out_data[0], 58.0f);
    expect_float("matmul 1", out_data[1], 64.0f);
    expect_float("matmul 2", out_data[2], 139.0f);
    expect_float("matmul 3", out_data[3], 154.0f);
}

static void test_validation(void)
{
    const size_t dims[] = { 2 };
    const size_t wrong_dims[] = { 3 };
    const size_t wrong_rank_dims[] = { 2, 1, 1 };
    const size_t matmul_a_dims[] = { 2, 3 };
    const size_t matmul_b_dims[] = { 4, 2 };
    const size_t matmul_out_dims[] = { 2, 2 };
    float lhs[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f };
    float rhs[] = { 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f, 11.0f };
    float out_data[4] = { 0.0f };
    lis_tensor_shape shape = make_shape(1, dims);
    lis_tensor_shape wrong_shape = make_shape(1, wrong_dims);
    lis_tensor_shape wrong_rank_shape = make_shape(3, wrong_rank_dims);
    lis_tensor_shape matmul_a = make_shape(2, matmul_a_dims);
    lis_tensor_shape matmul_b = make_shape(2, matmul_b_dims);
    lis_tensor_shape matmul_out = make_shape(2, matmul_out_dims);
    lis_tensor out = make_output(shape, out_data, 2);
    lis_tensor wrong_rank_output = make_output(wrong_rank_shape, out_data, 2);
    lis_tensor matmul_output = make_output(matmul_out, out_data, 4);
    lis_operator_request request =
        lis_operator_make_add(make_view(shape, lhs, 2),
                              make_view(wrong_shape, rhs, 3), &out);

    expect_status("add shape mismatch",
                  lis_operator_execute(lis_backend_cpu_reference(), &request),
                  LIS_STATUS_SHAPE_MISMATCH);

    request = lis_operator_make_matmul(make_view(matmul_a, lhs, 6),
                                       make_view(matmul_b, rhs, 8),
                                       &matmul_output);
    expect_status("matmul shape mismatch",
                  lis_operator_execute(lis_backend_cpu_reference(), &request),
                  LIS_STATUS_SHAPE_MISMATCH);

    request = lis_operator_make_add(make_view(shape, lhs, 2),
                                    make_view(shape, rhs, 2), NULL);
    expect_status("missing output",
                  lis_operator_execute(lis_backend_cpu_reference(), &request),
                  LIS_STATUS_INVALID_ARGUMENT);

    request = lis_operator_make_matmul(make_view(wrong_rank_shape, lhs, 2),
                                       make_view(shape, rhs, 2),
                                       &wrong_rank_output);
    expect_status("matmul unsupported rank",
                  lis_operator_execute(lis_backend_cpu_reference(), &request),
                  LIS_STATUS_UNSUPPORTED_SHAPE);

    {
        lis_tensor i32_out = out;

        i32_out.dtype = LIS_DTYPE_I32;
        request = lis_operator_make_add(make_view(shape, lhs, 2),
                                        make_view(shape, rhs, 2), &i32_out);
        expect_status("add unsupported output dtype",
                      lis_operator_execute(lis_backend_cpu_reference(), &request),
                      LIS_STATUS_UNSUPPORTED_DTYPE);
    }
}

static lis_status unsupported_execute(const lis_backend *backend,
                                      const lis_operator_request *request)
{
    (void)backend;
    (void)request;
    return LIS_STATUS_OK;
}

static void test_backend_boundary(void)
{
    lis_backend future_gpu = {
        .kind = LIS_BACKEND_KIND_GPU_RESERVED,
        .name = "future_gpu",
        .memory_domain = LIS_BACKEND_MEMORY_BACKEND_OWNED,
        .execute = unsupported_execute,
    };
    lis_operator_request request = { 0 };

    expect_status("null backend", lis_operator_execute(NULL, &request),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_status("null request",
                  lis_operator_execute(lis_backend_cpu_reference(), NULL),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_status("reserved gpu backend", lis_operator_execute(&future_gpu, &request),
                  LIS_STATUS_UNSUPPORTED);
    request.kind = (lis_operator_kind)99;
    expect_status("unsupported operator kind",
                  lis_operator_execute(lis_backend_cpu_reference(), &request),
                  LIS_STATUS_UNSUPPORTED);
}

int main(void)
{
    test_add();
    test_matmul();
    test_validation();
    test_backend_boundary();

    if (g_failures != 0) {
        fprintf(stderr, "%d backend test failure(s)\n", g_failures);
        return 1;
    }

    return 0;
}
