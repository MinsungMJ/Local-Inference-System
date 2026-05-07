#ifndef LIS_CPU_FEATURES_H
#define LIS_CPU_FEATURES_H

/*
 * Runtime CPUID-derived feature flags used by the SIMD dispatcher.
 *
 * The struct is plain-data (no pointers, no ownership). Values are 0 or 1.
 * On non-x86 builds every field is 0. Populated once on first call to
 * lis_cpu_features_get; callers must treat the returned pointer as read-only.
 * Safe to call concurrently after the first completed call.
 */
typedef struct {
    int sse2;
    int avx;
    int avx2;
    int fma;
    int f16c;
    int avx512f;
    int avx512vl;
    int bmi2;
} lis_cpu_features;

const lis_cpu_features *lis_cpu_features_get(void);

#endif
