# LIS Development Style Guide

This document defines the expected coding style for C contributions to LIS. It is a public contributor reference, not an exhaustive or immutable specification. When in doubt, follow the conventions established in the existing codebase and ask for maintainer review.

## Language Baseline

- Target **C11 or newer**.
- Prefer language features that are widely practical in real systems code.
- Do not introduce newer-standard-only features unless the build environment clearly supports them.

## Primary Objective

Optimize for:

1. **Correctness**
2. **Clarity of contract**
3. **Maintainability**
4. **Predictable systems behaviour**
5. **Reviewability by another systems engineer**

Do not optimise for novelty, terseness, or macro cleverness.

## Type Discipline

- Use fixed-width integer types from `<stdint.h>` when width matters (`uint32_t`, `int64_t`, etc.).
- Use `size_t` for sizes, lengths, and capacities.
- Use semantically correct signed types for differences or negative-capable quantities.
- Do not use vague types such as `int` or `long` when ABI, layout, storage, protocol, or hardware semantics depend on width.

## API Contract Clarity

Every nontrivial function should make the following clear through its signature, naming, or comments:

- **Ownership** — who owns the memory, who frees it
- **Lifetime** — when the data is valid
- **Mutability** — whether the caller may modify the data
- **Nullability** — whether pointer parameters may be `NULL`
- **Size and capacity expectations** — buffer sizes, array lengths
- **Failure model** — how errors are signalled and what happens on failure

Do not design APIs that force callers to guess these properties.

## Buffer Safety

- Pass pointer and length/capacity together.
- Bound all parse, copy, and encode operations explicitly.
- Do not assume hidden buffer sizes.
- Prefer interfaces that allow callers to reason about truncation, overflow, and written length.

## Error Handling

- Use a consistent error model within a module.
- Prefer explicit status codes or clearly documented failure conventions.
- Do not collapse distinct failure modes into an ambiguous boolean unless the task truly requires it.

## Memory and Allocation

- Allocate using `sizeof(*ptr)`.
- Check allocation results unless the environment contract guarantees otherwise.
- Keep cleanup paths consistent and leak-resistant.
- Avoid duplicated cleanup logic when a structured cleanup path is clearer.

## Compile-Time Enforcement

- Use `_Static_assert` for size, layout, alignment, constant assumptions, and invariants.
- Do not leave layout-sensitive assumptions as comments only.

## Macros

- Prefer `static inline` over function-like macros.
- Use macros only where the language is insufficient:
  - conditional compilation
  - include guards
  - carefully justified generic wrappers
  - token and string generation
- Do not use macros to hide ordinary logic that should be written as functions.

## Concurrency

- Do not use `volatile` as a thread-synchronisation primitive.
- Use `<stdatomic.h>` when lock-free shared-state synchronisation is needed.
- If mutex-based synchronisation is simpler and correct, prefer the simpler correct design.
- Concurrency assumptions must be explicit.

## Portability Boundaries

- Make compiler, platform, POSIX, GNU, kernel-only, or architecture-specific assumptions explicit.
- Do not present environment-specific code as portable ISO C.

## Tooling Discipline

Contributed code should be compatible with a workflow that expects:

- Strong compiler warnings (`-Wall -Wextra -Wpedantic -Werror`)
- Sanitiser-friendly behaviour
- Static analysis compatibility

Do not write code that depends on warning suppression or undefined behaviour by default.

## Preferred Patterns

### Explicit Width Where Required

```c
#include <stdint.h>

typedef struct {
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    uint64_t payload_len;
} msg_header_t;
```

### Size-Aware API

```c
int encode_packet(uint8_t *dst, size_t dst_cap,
                  const struct packet *pkt, size_t *out_len);
```

### Designated Initialiser

```c
typedef struct {
    int port;
    int timeout_ms;
    int retry_count;
} conn_cfg_t;

conn_cfg_t cfg = {
    .port = 8080,
    .timeout_ms = 1000,
    .retry_count = 3,
};
```

### Inline Helper Instead of Unsafe Macro

```c
static inline int max_int(int a, int b)
{
    return (a > b) ? a : b;
}
```

### Allocation Bound to the Actual Object

```c
worker_t *worker = malloc(sizeof(*worker));
if (worker == NULL) {
    return ERR_NO_MEMORY;
}
```

### Compile-Time Contract

```c
_Static_assert(sizeof(struct on_disk_hdr) == 32,
               "on_disk_hdr layout changed");
```

### Explicit Error Model

```c
typedef enum {
    LOAD_OK = 0,
    LOAD_ERR_OPEN,
    LOAD_ERR_READ,
    LOAD_ERR_FORMAT,
} load_result_t;

load_result_t load_config(const char *path, config_t *out_cfg);
```

### Atomics for Shared State

```c
#include <stdatomic.h>

atomic_bool stop_requested = false;
```

### Structured Cleanup

```c
int process_file(const char *path)
{
    FILE *fp = NULL;
    char *buf = NULL;
    int rc = -1;

    fp = fopen(path, "rb");
    if (fp == NULL)
        goto out;

    buf = malloc(4096);
    if (buf == NULL)
        goto out;

    if (fread(buf, 1, 4096, fp) != 4096)
        goto out;

    rc = 0;

out:
    free(buf);
    if (fp != NULL)
        fclose(fp);
    return rc;
}
```

## Header and Source Ownership

- Headers declare public interfaces; sources implement them.
- Prefer a clear ownership boundary between header and source files.
- Subsystem boundaries should be maintained. Do not mix runtime internals into loader headers, loader concerns into backend headers, and so on.

## Summary

Generate C code that a careful systems programmer would describe as:

- Explicit
- Bounded
- Contract-aware
- Warning-clean
- Concurrency-conscious
- Maintainable under review

When forced to choose, prefer **clarity and correctness over terseness and cleverness**.