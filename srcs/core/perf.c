#if !defined(_POSIX_C_SOURCE) || (_POSIX_C_SOURCE - 0) < 199309L
#undef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 199309L
#endif

#include "lis/perf.h"

#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define LIS_PERF_NS_PER_SEC UINT64_C(1000000000)

static const char *lis_perf_stage_name(lis_perf_stage_id stage)
{
    switch (stage) {
    case LIS_PERF_STAGE_MODEL_LOAD:
        return "model_load";
    case LIS_PERF_STAGE_TOKENIZER_LOAD:
        return "tokenizer_load";
    case LIS_PERF_STAGE_TOKENIZER_ENCODE:
        return "tokenizer_encode";
    case LIS_PERF_STAGE_RUNTIME_INIT:
        return "runtime_init";
    case LIS_PERF_STAGE_PREFILL:
        return "prefill";
    case LIS_PERF_STAGE_FIRST_DECODE:
        return "first_decode";
    case LIS_PERF_STAGE_DECODE_STEADY_STATE:
        return "decode_steady_state";
    case LIS_PERF_STAGE_COUNT:
    default:
        return "unknown";
    }
}

uint64_t lis_perf_now_ns(void)
{
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t)ts.tv_sec * LIS_PERF_NS_PER_SEC + (uint64_t)ts.tv_nsec;
}

void lis_perf_report_init(lis_perf_report *report,
                          int enabled,
                          int per_token_enabled,
                          const char *tag)
{
    if (report == NULL) {
        return;
    }
    memset(report, 0, sizeof(*report));
    report->enabled = enabled ? 1 : 0;
    report->per_token_enabled = per_token_enabled ? 1 : 0;
    report->tag = (tag != NULL && tag[0] != '\0') ? tag : "none";
}

void lis_perf_stage_begin(lis_perf_report *report, lis_perf_stage_id stage)
{
    if (report == NULL || !report->enabled ||
        (int)stage < 0 || stage >= LIS_PERF_STAGE_COUNT) {
        return;
    }
    report->stage_start_ns[stage] = lis_perf_now_ns();
    report->stage_open[stage] = 1;
}

void lis_perf_stage_end(lis_perf_report *report,
                        lis_perf_stage_id stage,
                        uint64_t tokens)
{
    uint64_t now;

    if (report == NULL || !report->enabled ||
        (int)stage < 0 || stage >= LIS_PERF_STAGE_COUNT ||
        !report->stage_open[stage]) {
        return;
    }
    now = lis_perf_now_ns();
    if (now >= report->stage_start_ns[stage]) {
        report->stage_ns[stage] += now - report->stage_start_ns[stage];
    }
    report->stage_tokens[stage] += tokens;
    report->stage_open[stage] = 0;
}

void lis_perf_stage_accumulate(lis_perf_report *report,
                               lis_perf_stage_id stage,
                               uint64_t ns,
                               uint64_t tokens)
{
    if (report == NULL || !report->enabled ||
        (int)stage < 0 || stage >= LIS_PERF_STAGE_COUNT) {
        return;
    }
    report->stage_ns[stage] += ns;
    report->stage_tokens[stage] += tokens;
}

void lis_perf_emit_per_token(const lis_perf_report *report,
                             FILE *stream,
                             size_t step,
                             uint64_t ns)
{
    double ms;

    if (report == NULL || !report->enabled || !report->per_token_enabled ||
        stream == NULL) {
        return;
    }
    ms = (double)ns / 1.0e6;
    fprintf(stream,
            "lis: perf-per-token tag=%s step=%zu ns=%" PRIu64 " ms=%.3f\n",
            report->tag, step, ns, ms);
}

static double lis_perf_ns_to_ms(uint64_t ns)
{
    return (double)ns / 1.0e6;
}

void lis_perf_report_emit(const lis_perf_report *report,
                          FILE *stream,
                          int threads,
                          size_t prompt_tokens,
                          size_t generated_tokens)
{
    uint64_t encode_ns;
    uint64_t runtime_init_ns;
    uint64_t prefill_ns;
    uint64_t first_decode_ns;
    uint64_t steady_ns;
    uint64_t steady_tokens;
    uint64_t ttft_ns;
    uint64_t end_to_end_ns;
    double   itl_ms;
    double   tps_steady;
    double   tps_end_to_end;
    size_t   i;

    if (report == NULL || !report->enabled || stream == NULL) {
        return;
    }

    for (i = 0; i < LIS_PERF_STAGE_COUNT; ++i) {
        fprintf(stream,
                "lis: perf-stage tag=%s name=%s ns=%" PRIu64
                " ms=%.3f tokens=%" PRIu64 "\n",
                report->tag,
                lis_perf_stage_name((lis_perf_stage_id)i),
                report->stage_ns[i],
                lis_perf_ns_to_ms(report->stage_ns[i]),
                report->stage_tokens[i]);
    }

    encode_ns       = report->stage_ns[LIS_PERF_STAGE_TOKENIZER_ENCODE];
    runtime_init_ns = report->stage_ns[LIS_PERF_STAGE_RUNTIME_INIT];
    prefill_ns      = report->stage_ns[LIS_PERF_STAGE_PREFILL];
    first_decode_ns = report->stage_ns[LIS_PERF_STAGE_FIRST_DECODE];
    steady_ns       = report->stage_ns[LIS_PERF_STAGE_DECODE_STEADY_STATE];
    steady_tokens   = report->stage_tokens[LIS_PERF_STAGE_DECODE_STEADY_STATE];

    ttft_ns = encode_ns + runtime_init_ns + prefill_ns + first_decode_ns;
    end_to_end_ns = prefill_ns + first_decode_ns + steady_ns;

    itl_ms = (steady_tokens > 0)
        ? (lis_perf_ns_to_ms(steady_ns) / (double)steady_tokens)
        : 0.0;
    tps_steady = (steady_ns > 0)
        ? ((double)steady_tokens * 1.0e9 / (double)steady_ns)
        : 0.0;
    tps_end_to_end = (end_to_end_ns > 0 && generated_tokens > 0)
        ? ((double)generated_tokens * 1.0e9 / (double)end_to_end_ns)
        : 0.0;

    fprintf(stream,
            "lis: perf-summary tag=%s threads=%d prompt_tokens=%zu "
            "generated_tokens=%zu ttft_ms=%.3f itl_ms=%.3f "
            "tps_steady=%.3f tps_end_to_end=%.3f\n",
            report->tag,
            threads,
            prompt_tokens,
            generated_tokens,
            lis_perf_ns_to_ms(ttft_ns),
            itl_ms,
            tps_steady,
            tps_end_to_end);
}
