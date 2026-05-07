#ifndef LIS_THREAD_POOL_H
#define LIS_THREAD_POOL_H

#include "lis/status.h"

#include <stddef.h>

typedef struct lis_thread_pool lis_thread_pool;

/*
 * Work function signature for thread pool dispatch.
 * Called once per chunk with the chunk's start index, element count,
 * and the caller-provided context pointer.
 */
typedef void (*lis_work_fn)(size_t start, size_t count, void *context);

/*
 * Create a thread pool with the given thread count.
 * When thread_count == 1, a lightweight stub is created that executes
 * work inline in the calling thread with zero dispatch overhead.
 * thread_count must be >= 1.
 * Caller owns the returned pool and must call lis_thread_pool_destroy.
 */
lis_status lis_thread_pool_init(lis_thread_pool **out_pool,
                                size_t thread_count);

/*
 * Destroy a thread pool, joining all worker threads and freeing resources.
 * Safe to call with NULL or a pointer to NULL.
 */
void lis_thread_pool_destroy(lis_thread_pool **pool);

/*
 * Dispatch work across pool threads using fork-join semantics.
 * Partitions [0, total) into chunks and calls work_fn once per chunk.
 * All chunks complete before dispatch returns.
 * When total == 0, returns immediately.
 * When total < thread_count, only total chunks are dispatched.
 */
lis_status lis_thread_pool_dispatch(lis_thread_pool *pool, size_t total,
                                    lis_work_fn work_fn, void *context);

/* Return the thread count of the pool. Returns 0 if pool is NULL. */
size_t lis_thread_pool_thread_count(const lis_thread_pool *pool);

#endif
