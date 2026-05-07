#include "lis/runtime.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static int lis_mul_size_overflows(size_t lhs, size_t rhs, size_t *out)
{
    if (out == NULL) {
        return 1;
    }
    if (lhs != 0 && rhs > SIZE_MAX / lhs) {
        return 1;
    }

    *out = lhs * rhs;
    return 0;
}

static lis_status lis_kv_cache_compute_layout(const lis_model_config *config,
                                              size_t batch_size,
                                              lis_kv_cache_layout *out_layout)
{
    lis_kv_cache_layout layout = { 0 };
    size_t elements = 0;
    lis_status status;

    if (config == NULL || out_layout == NULL || batch_size == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    status = lis_model_config_validate(config);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (config->context.configured_max_tokens == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    layout.layer_count = config->layer_count;
    layout.batch_size = batch_size;
    layout.context_length = config->context.configured_max_tokens;
    layout.kv_head_count = config->kv_head_count;
    layout.head_dim = config->head_dim;
    layout.dtype = config->weight_dtype;

    status = lis_dtype_size_bytes(layout.dtype, &layout.element_size);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    elements = layout.layer_count;
    if (lis_mul_size_overflows(elements, layout.batch_size, &elements) ||
        lis_mul_size_overflows(elements, layout.context_length, &elements) ||
        lis_mul_size_overflows(elements, layout.kv_head_count, &elements) ||
        lis_mul_size_overflows(elements, layout.head_dim, &elements)) {
        return LIS_STATUS_OVERFLOW;
    }
    layout.elements_per_cache = elements;
    if (lis_mul_size_overflows(elements, layout.element_size,
                               &layout.bytes_per_cache)) {
        return LIS_STATUS_OVERFLOW;
    }

    *out_layout = layout;
    return LIS_STATUS_OK;
}

lis_status lis_kv_cache_init(lis_kv_cache *cache,
                             const lis_model_config *config,
                             size_t batch_size)
{
    lis_kv_cache local = { 0 };
    lis_status status;

    if (cache == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    status = lis_kv_cache_compute_layout(config, batch_size, &local.layout);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    local.keys = calloc(1, local.layout.bytes_per_cache);
    local.values = calloc(1, local.layout.bytes_per_cache);
    if (local.keys == NULL || local.values == NULL) {
        lis_kv_cache_destroy(&local);
        return LIS_STATUS_NO_MEMORY;
    }

    *cache = local;
    return LIS_STATUS_OK;
}

void lis_kv_cache_destroy(lis_kv_cache *cache)
{
    if (cache == NULL) {
        return;
    }

    free(cache->keys);
    free(cache->values);
    memset(cache, 0, sizeof(*cache));
}

lis_status lis_kv_cache_element_offset(const lis_kv_cache *cache,
                                       size_t layer, size_t batch,
                                       size_t position, size_t head,
                                       size_t dim, size_t *out_offset)
{
    size_t index = 0;

    if (cache == NULL || out_offset == NULL || cache->keys == NULL ||
        cache->values == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (layer >= cache->layout.layer_count ||
        batch >= cache->layout.batch_size ||
        position >= cache->layout.context_length ||
        head >= cache->layout.kv_head_count ||
        dim >= cache->layout.head_dim) {
        return LIS_STATUS_LIMIT_EXCEEDED;
    }

    index = layer;
    index = index * cache->layout.batch_size + batch;
    index = index * cache->layout.context_length + position;
    index = index * cache->layout.kv_head_count + head;
    index = index * cache->layout.head_dim + dim;
    if (lis_mul_size_overflows(index, cache->layout.element_size,
                               out_offset)) {
        return LIS_STATUS_OVERFLOW;
    }

    return LIS_STATUS_OK;
}

lis_status lis_kv_cache_key_ptr(const lis_kv_cache *cache,
                                size_t layer, size_t batch,
                                size_t position, size_t head,
                                size_t dim, void **out_ptr)
{
    size_t offset = 0;
    lis_status status;

    if (out_ptr == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    status = lis_kv_cache_element_offset(cache, layer, batch, position, head,
                                         dim, &offset);
    if (status != LIS_STATUS_OK) {
        *out_ptr = NULL;
        return status;
    }

    *out_ptr = cache->keys + offset;
    return LIS_STATUS_OK;
}

lis_status lis_kv_cache_value_ptr(const lis_kv_cache *cache,
                                  size_t layer, size_t batch,
                                  size_t position, size_t head,
                                  size_t dim, void **out_ptr)
{
    size_t offset = 0;
    lis_status status;

    if (out_ptr == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    status = lis_kv_cache_element_offset(cache, layer, batch, position, head,
                                         dim, &offset);
    if (status != LIS_STATUS_OK) {
        *out_ptr = NULL;
        return status;
    }

    *out_ptr = cache->values + offset;
    return LIS_STATUS_OK;
}
