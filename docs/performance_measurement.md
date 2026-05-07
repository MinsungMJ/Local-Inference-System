# Performance Measurement

LIS v1 is correctness-first, but operators need wall-clock numbers to reason
about threading, prompt length, and generation length trade-offs. LIS provides
an opt-in performance measurement framework: stage-by-stage timings,
TTFT/ITL/throughput summary, and a sweep harness over threads × prompt length ×
generation length. Measurement is pure observation — inference semantics are
unchanged.

## Scope

- Wall-clock stage breakdown per run, emitted on stderr in the existing `lis: ...` convention.
- Summary metrics: TTFT, mean ITL, steady-state tokens/sec, end-to-end tokens/sec.
- Comparison matrix driven by an external harness that keeps the stable stderr protocol for CSV/Markdown and also writes a bounded JSON benchmark snapshot reusing the core run-report contract.
- CPU only. The only clock source is POSIX `CLOCK_MONOTONIC`.

Out of scope: per-operator timing (matvec, attention, MLP, RoPE, softmax), GPU
timers, allocation tracking, `perf record` integration, HF-side comparison.

## Stages measured

| Stage name              | What it covers                                              |
|-------------------------|-------------------------------------------------------------|
| `model_load`            | `lis_loader_load` + metadata attach                         |
| `tokenizer_load`        | `lis_hf_tokenizer_load` or `lis_tokenizer_load` (BPE)       |
| `tokenizer_encode`      | Prompt text → token IDs                                     |
| `runtime_init`          | `lis_runtime_init` (thread pool creation, KV cache sizing)  |
| `prefill`               | `lis_runtime_llama_prefill` over the full prompt batch      |
| `first_decode`          | The **first** `lis_runtime_llama_decode` call only          |
| `decode_steady_state`   | Sum of all subsequent `lis_runtime_llama_decode` calls      |

Token-ID input (`--tokens PATH`) has no `tokenizer_encode`; that stage is reported as `ns=0`.

## Summary metrics

- **TTFT** (Time To First Token) = `tokenizer_encode + runtime_init + prefill + first_decode`.
  - `model_load` and `tokenizer_load` are excluded because they amortize across runs and are dominated by the filesystem page cache after the first invocation.
- **ITL** (mean Inter-Token Latency) = `decode_steady_state_ns / decode_steady_state_tokens`.
  - The first decode step is excluded — it carries attention-mask warm-up costs that aren't representative of steady state.
- **tps_steady** (steady-state tokens/sec) = `decode_steady_state_tokens / (decode_steady_state_ns / 1e9)`.
- **tps_end_to_end** = `generated_tokens / ((prefill + first_decode + decode_steady_state) / 1e9)`.

## Output format

All lines go to stderr, one key=value pair per field, line-oriented for trivial parsing. `<TAG>` is whatever was passed to `--perf-tag` (default `"none"`).

```
lis: perf-stage tag=<TAG> name=model_load         ns=<N> ms=<M> tokens=0
lis: perf-stage tag=<TAG> name=tokenizer_load     ns=<N> ms=<M> tokens=0
lis: perf-stage tag=<TAG> name=tokenizer_encode   ns=<N> ms=<M> tokens=0
lis: perf-stage tag=<TAG> name=runtime_init       ns=<N> ms=<M> tokens=0
lis: perf-stage tag=<TAG> name=prefill            ns=<N> ms=<M> tokens=<P>
lis: perf-stage tag=<TAG> name=first_decode       ns=<N> ms=<M> tokens=1
lis: perf-stage tag=<TAG> name=decode_steady_state ns=<N> ms=<M> tokens=<D>
lis: perf-summary tag=<TAG> threads=<T> prompt_tokens=<P> generated_tokens=<G> \
     ttft_ms=<M> itl_ms=<M> tps_steady=<F> tps_end_to_end=<F>
```

Optional per-step lines under `--perf-per-token`:

```
lis: perf-per-token tag=<TAG> step=<N> ns=<N> ms=<M>
```

## CLI flags

| Flag                 | Effect                                                        |
|----------------------|---------------------------------------------------------------|
| `--perf`             | Enable stage timing and summary emission.                     |
| `--perf-per-token`   | In addition, emit one line per decode step. Implies `--perf`. |
| `--perf-tag TAG`     | Free-form label echoed into every `lis: perf-*` line.         |

Without `--perf`, nothing related to measurement is emitted. Existing
`--diagnostics` / `--layer-checkpoints` output is unchanged.

## Sweep harness

`tests/perf/run_perf_matrix.py` runs a full matrix and writes a comparison table:

- `threads ∈ {1, nproc}`
- prompt lengths `∈ {short, medium, long}` — fixed files under `tests/perf/prompts/`
- generation lengths `∈ {16, 64, 256}`
- Per cell: 1 warm-up (discarded) + 3 measured runs; median reported, min/max in a side column.

Output files:

- `tests/perf/results.csv` — one row per (threads, prompt, generate) cell.
- `tests/perf/results.md` — grouped Markdown summary for pasting into docs.
- `tests/perf/results.json` — bounded benchmark snapshot JSON; each cell embeds one representative core run report plus the aggregated summary fields.

The harness also parses the already-emitted
`lis: simd backend=...` stderr line produced by `--perf` runs and carries that
value as a narrow `backend` field in generated CSV/Markdown artifacts. The same
harness additionally requests `--report-json` and writes the bounded JSON
snapshot described above. The stage-line protocol itself is
unchanged.

Convenience target: `make bench` invokes the harness when `BENCH_MODEL` is
explicitly set:
`BENCH_MODEL=/path/to/plain-rope-llama make bench`

Precision work treats performance effects as measured consequences of the
documented precision path. It does not add new benchmark modes, metrics, or
reporting surfaces.

## Methodology caveats

- **Page cache**: `model_load` on a cold boot reads gigabytes from disk; subsequent runs hit the OS page cache and are ~2 orders of magnitude faster. The harness discards its first run per cell to normalize this.
- **Thread pool warm-up**: the first prefill often pays a pthread spin/wake-up cost. `first_decode` absorbs attention-mask and KV-cache-tail initialization. ITL explicitly excludes these.
- **Clock resolution**: Linux `CLOCK_MONOTONIC` is typically nanosecond-precision but microsecond-accurate. Stages shorter than a few microseconds will show low-significance noise.
- **`--perf` overhead**: one `clock_gettime` per stage begin/end; on the order of 50 ns per call, far below the resolution of the measurements themselves.

## Verification

Run the smoke command with a local plain-RoPE Llama-family model:

```bash
MODEL=/path/to/plain-rope-llama

./srcs/libs/lis \
  --model "$MODEL" \
  --config "$MODEL/config.json" \
  --hf-tokenizer "$MODEL/tokenizer.json" \
  --prompt "Hello." --context 512 --batch 1 --generate 8 \
  --threads 1 --perf --perf-tag "smoke"
```

Expect seven `lis: perf-stage` lines and one `lis: perf-summary` line on stderr,
all `ms` values > 0. Re-running with `--threads $(nproc)` should reduce
`prefill_ms` and `decode_steady_state_ms`. When `--report-json PATH` is added,
the emitted run report carries the same bounded perf summary in machine-readable
form.

Performance is environment-dependent. Do not treat any single measurement as a
general performance claim; record the model, hardware, date, thread count,
context length, generation length, and known limitations when comparing runs.
