#ifndef LIS_BACKEND_H
#define LIS_BACKEND_H

#include "lis/status.h"
#include "lis/tensor.h"

typedef enum {
    LIS_BACKEND_KIND_CPU_REFERENCE = 1,
    LIS_BACKEND_KIND_SIMD_RESERVED,
    LIS_BACKEND_KIND_GPU_RESERVED
} lis_backend_kind;

typedef enum {
    LIS_BACKEND_MEMORY_HOST = 1,
    LIS_BACKEND_MEMORY_BACKEND_OWNED
} lis_backend_memory_domain;

typedef enum {
    LIS_OPERATOR_ADD = 1,
    LIS_OPERATOR_MATMUL
} lis_operator_kind;

typedef struct {
    lis_tensor_view lhs;
    lis_tensor_view rhs;
    lis_tensor *out;
} lis_operator_binary;

typedef struct {
    lis_tensor_view lhs;
    lis_tensor_view rhs;
    lis_tensor *out;
} lis_operator_matmul;

typedef struct {
    lis_operator_kind kind;
    union {
        lis_operator_binary binary;
        lis_operator_matmul matmul;
    } args;
} lis_operator_request;

struct lis_backend;

typedef lis_status (*lis_backend_execute_fn)(const struct lis_backend *backend,
                                             const lis_operator_request *request);

typedef struct lis_backend {
    lis_backend_kind kind;
    const char *name;
    /*
     * Supports host-memory tensors only. The memory-domain field is
     * the explicit future extension point for backend-owned buffers, such as a
     * later GPU allocation handle, without embedding those assumptions in core
     * tensor metadata.
     */
    lis_backend_memory_domain memory_domain;
    lis_backend_execute_fn execute;
} lis_backend;

const char *lis_backend_kind_name(lis_backend_kind kind);
const char *lis_operator_kind_name(lis_operator_kind kind);

lis_operator_request lis_operator_make_add(lis_tensor_view lhs,
                                           lis_tensor_view rhs,
                                           lis_tensor *out);
lis_operator_request lis_operator_make_matmul(lis_tensor_view lhs,
                                              lis_tensor_view rhs,
                                              lis_tensor *out);

/*
 * Executes one validated operator request. Inputs are borrowed tensor views.
 * Output storage is caller-owned and must be writable for the duration of the
 * call. Implements f32 add and f32 2D matmul through the CPU reference
 * backend only.
 */
lis_status lis_operator_execute(const lis_backend *backend,
                                const lis_operator_request *request);
const lis_backend *lis_backend_cpu_reference(void);

#endif
