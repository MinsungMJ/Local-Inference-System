#ifndef LIS_PERF_H
#define LIS_PERF_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/*
 * Wall-clock measurement primitive. Owned and driven by the CLI
 * driver. Strictly observational: no runtime, operator, loader, or tokenizer
 * module references this API. Fixed-slot, no dynamic allocation, no global
 * state.
 */

typedef enum {
    LIS_PERF_STAGE_MODEL_LOAD = 0,
    LIS_PERF_STAGE_TOKENIZER_LOAD,
    LIS_PERF_STAGE_TOKENIZER_ENCODE,
    LIS_PERF_STAGE_RUNTIME_INIT,
    LIS_PERF_STAGE_PREFILL,
    LIS_PERF_STAGE_FIRST_DECODE,
    LIS_PERF_STAGE_DECODE_STEADY_STATE,
    LIS_PERF_STAGE_COUNT
} lis_perf_stage_id;

typedef struct {
    int enabled;
    int per_token_enabled;
    const char *tag;
    uint64_t stage_start_ns[LIS_PERF_STAGE_COUNT];
    int       stage_open[LIS_PERF_STAGE_COUNT];
    uint64_t  stage_ns[LIS_PERF_STAGE_COUNT];
    uint64_t  stage_tokens[LIS_PERF_STAGE_COUNT];
} lis_perf_report;

/*
 * Monotonic nanosecond timestamp. Wraps clock_gettime(CLOCK_MONOTONIC).
 */
uint64_t lis_perf_now_ns(void);

/*
 * Initialize an on-stack report. All counters zeroed; enabled = 0; tag defaults
 * to "none" when tag is NULL.
 */
void lis_perf_report_init(lis_perf_report *report,
                          int enabled,
                          int per_token_enabled,
                          const char *tag);

/*
 * Begin a stage scope. No-op when report is NULL or disabled. Safe to call
 * multiple times on the same stage; each begin overwrites the stored start
 * timestamp.
 */
void lis_perf_stage_begin(lis_perf_report *report, lis_perf_stage_id stage);

/*
 * End a stage scope. Adds elapsed ns since matching begin() to the stage
 * accumulator and records the token count passed in. No-op when report is NULL
 * or disabled, or when no matching begin is open.
 */
void lis_perf_stage_end(lis_perf_report *report,
                        lis_perf_stage_id stage,
                        uint64_t tokens);

/*
 * Add a precomputed interval to a stage accumulator without using the
 * begin/end scope. Useful when the caller already has start/end timestamps
 * captured by an outer loop.
 */
void lis_perf_stage_accumulate(lis_perf_report *report,
                               lis_perf_stage_id stage,
                               uint64_t ns,
                               uint64_t tokens);

/*
 * Emit a single `lis: perf-per-token` line when per_token_enabled is set.
 * No-op when report is NULL or per-token emission is off.
 */
void lis_perf_emit_per_token(const lis_perf_report *report,
                             FILE *stream,
                             size_t step,
                             uint64_t ns);

/*
 * Emit the seven stage lines and the summary line to stream when enabled is
 * set. prompt_tokens and generated_tokens are the summary totals the driver
 * reports; threads is the runtime-configured worker count.
 */
void lis_perf_report_emit(const lis_perf_report *report,
                          FILE *stream,
                          int threads,
                          size_t prompt_tokens,
                          size_t generated_tokens);

#endif
