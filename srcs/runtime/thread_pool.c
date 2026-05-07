#include "lis/thread_pool.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    lis_work_fn work_fn;
    void *context;
    size_t start;
    size_t count;
} lis_work_item;

typedef struct {
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond_work;
    pthread_cond_t cond_done;
    lis_work_item item;
    int has_work;
    int shutdown;
} lis_worker;

struct lis_thread_pool {
    size_t thread_count;
    lis_worker *workers;
};

static void *lis_worker_loop(void *arg)
{
    lis_worker *worker = (lis_worker *)arg;

    pthread_mutex_lock(&worker->mutex);
    for (;;) {
        while (!worker->has_work && !worker->shutdown) {
            pthread_cond_wait(&worker->cond_work, &worker->mutex);
        }
        if (worker->shutdown && !worker->has_work) {
            break;
        }
        {
            lis_work_item item = worker->item;

            pthread_mutex_unlock(&worker->mutex);
            item.work_fn(item.start, item.count, item.context);
            pthread_mutex_lock(&worker->mutex);
        }
        worker->has_work = 0;
        pthread_cond_signal(&worker->cond_done);
    }
    pthread_mutex_unlock(&worker->mutex);
    return NULL;
}

static void lis_worker_init(lis_worker *worker)
{
    memset(worker, 0, sizeof(*worker));
    pthread_mutex_init(&worker->mutex, NULL);
    pthread_cond_init(&worker->cond_work, NULL);
    pthread_cond_init(&worker->cond_done, NULL);
}

static void lis_worker_start(lis_worker *worker)
{
    pthread_create(&worker->thread, NULL, lis_worker_loop, worker);
}

static void lis_worker_shutdown(lis_worker *worker)
{
    pthread_mutex_lock(&worker->mutex);
    worker->shutdown = 1;
    pthread_cond_signal(&worker->cond_work);
    pthread_mutex_unlock(&worker->mutex);
    pthread_join(worker->thread, NULL);
    pthread_mutex_destroy(&worker->mutex);
    pthread_cond_destroy(&worker->cond_work);
    pthread_cond_destroy(&worker->cond_done);
}

static void lis_worker_submit(lis_worker *worker, const lis_work_item *item)
{
    pthread_mutex_lock(&worker->mutex);
    worker->item = *item;
    worker->has_work = 1;
    pthread_cond_signal(&worker->cond_work);
    pthread_mutex_unlock(&worker->mutex);
}

static void lis_worker_wait(lis_worker *worker)
{
    pthread_mutex_lock(&worker->mutex);
    while (worker->has_work) {
        pthread_cond_wait(&worker->cond_done, &worker->mutex);
    }
    pthread_mutex_unlock(&worker->mutex);
}

lis_status lis_thread_pool_init(lis_thread_pool **out_pool,
                                size_t thread_count)
{
    lis_thread_pool *pool = NULL;
    size_t index;

    if (out_pool == NULL || thread_count == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    *out_pool = NULL;

    pool = malloc(sizeof(*pool));
    if (pool == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }
    memset(pool, 0, sizeof(*pool));
    pool->thread_count = thread_count;

    if (thread_count == 1) {
        pool->workers = NULL;
        *out_pool = pool;
        return LIS_STATUS_OK;
    }

    pool->workers = malloc(thread_count * sizeof(*pool->workers));
    if (pool->workers == NULL) {
        free(pool);
        return LIS_STATUS_NO_MEMORY;
    }
    for (index = 0; index < thread_count; ++index) {
        lis_worker_init(&pool->workers[index]);
    }
    for (index = 0; index < thread_count; ++index) {
        lis_worker_start(&pool->workers[index]);
    }

    *out_pool = pool;
    return LIS_STATUS_OK;
}

void lis_thread_pool_destroy(lis_thread_pool **pool)
{
    lis_thread_pool *p = NULL;
    size_t index;

    if (pool == NULL || *pool == NULL) {
        return;
    }
    p = *pool;
    if (p->workers != NULL) {
        for (index = 0; index < p->thread_count; ++index) {
            lis_worker_shutdown(&p->workers[index]);
        }
        free(p->workers);
    }
    free(p);
    *pool = NULL;
}

lis_status lis_thread_pool_dispatch(lis_thread_pool *pool, size_t total,
                                    lis_work_fn work_fn, void *context)
{
    size_t chunk_count;
    size_t base_chunk;
    size_t remainder;
    size_t offset;
    size_t index;

    if (pool == NULL || work_fn == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (total == 0) {
        return LIS_STATUS_OK;
    }

    /* Single-thread fast path: call directly, no dispatch overhead. */
    if (pool->thread_count == 1) {
        work_fn(0, total, context);
        return LIS_STATUS_OK;
    }

    chunk_count = (total < pool->thread_count) ? total : pool->thread_count;
    base_chunk = total / chunk_count;
    remainder = total % chunk_count;
    offset = 0;

    for (index = 0; index < chunk_count; ++index) {
        lis_work_item item;
        size_t count = base_chunk + (index < remainder ? 1 : 0);

        item.work_fn = work_fn;
        item.context = context;
        item.start = offset;
        item.count = count;
        lis_worker_submit(&pool->workers[index], &item);
        offset += count;
    }
    for (index = 0; index < chunk_count; ++index) {
        lis_worker_wait(&pool->workers[index]);
    }

    return LIS_STATUS_OK;
}

size_t lis_thread_pool_thread_count(const lis_thread_pool *pool)
{
    if (pool == NULL) {
        return 0;
    }
    return pool->thread_count;
}
