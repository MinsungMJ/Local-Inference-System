#include "lis/cpu_features.h"

#include <stdatomic.h>

static lis_cpu_features g_features;
static atomic_int g_initialized;

static void lis_cpu_features_populate(lis_cpu_features *out)
{
    out->sse2 = 0;
    out->avx = 0;
    out->avx2 = 0;
    out->fma = 0;
    out->f16c = 0;
    out->avx512f = 0;
    out->avx512vl = 0;
    out->bmi2 = 0;
#if (defined(__x86_64__) || defined(__i386__)) && \
    (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    out->sse2 = __builtin_cpu_supports("sse2") ? 1 : 0;
    out->avx = __builtin_cpu_supports("avx") ? 1 : 0;
    out->avx2 = __builtin_cpu_supports("avx2") ? 1 : 0;
    out->fma = __builtin_cpu_supports("fma") ? 1 : 0;
    out->f16c = __builtin_cpu_supports("f16c") ? 1 : 0;
    out->avx512f = __builtin_cpu_supports("avx512f") ? 1 : 0;
    out->avx512vl = __builtin_cpu_supports("avx512vl") ? 1 : 0;
    out->bmi2 = __builtin_cpu_supports("bmi2") ? 1 : 0;
#endif
}

const lis_cpu_features *lis_cpu_features_get(void)
{
    if (atomic_load_explicit(&g_initialized, memory_order_acquire) != 0) {
        return &g_features;
    }
    /*
     * Multiple threads racing here write identical values, so the
     * struct remains self-consistent. The release/acquire on
     * g_initialized publishes those writes to later readers.
     */
    lis_cpu_features_populate(&g_features);
    atomic_store_explicit(&g_initialized, 1, memory_order_release);
    return &g_features;
}
