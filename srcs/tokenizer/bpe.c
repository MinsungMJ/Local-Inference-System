#include "lis/tokenizer.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* --- Merge table implementation --- */

static size_t lis_merge_hash(size_t first, size_t second, size_t capacity)
{
    size_t h = first * 2654435761U + second * 40503U;

    return h % capacity;
}

lis_status lis_merge_table_init(lis_merge_table *table, size_t capacity)
{
    if (table == NULL || capacity == 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (capacity > SIZE_MAX / sizeof(*table->entries)) {
        return LIS_STATUS_OVERFLOW;
    }

    table->entries = calloc(capacity, sizeof(*table->entries));
    if (table->entries == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }
    table->capacity = capacity;
    table->count = 0;
    return LIS_STATUS_OK;
}

void lis_merge_table_destroy(lis_merge_table *table)
{
    if (table == NULL) {
        return;
    }
    free(table->entries);
    memset(table, 0, sizeof(*table));
}

lis_status lis_merge_table_insert(lis_merge_table *table,
                                  size_t first, size_t second,
                                  size_t result, size_t rank)
{
    size_t idx;
    size_t i;

    if (table == NULL || table->entries == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (table->count >= table->capacity) {
        return LIS_STATUS_NO_MEMORY;
    }

    idx = lis_merge_hash(first, second, table->capacity);
    for (i = 0; i < table->capacity; ++i) {
        size_t probe = (idx + i) % table->capacity;

        if (!table->entries[probe].occupied) {
            table->entries[probe].first = first;
            table->entries[probe].second = second;
            table->entries[probe].result = result;
            table->entries[probe].rank = rank;
            table->entries[probe].occupied = 1;
            ++table->count;
            return LIS_STATUS_OK;
        }
        if (table->entries[probe].first == first &&
            table->entries[probe].second == second) {
            if (rank < table->entries[probe].rank) {
                table->entries[probe].rank = rank;
                table->entries[probe].result = result;
            }
            return LIS_STATUS_OK;
        }
    }
    return LIS_STATUS_NO_MEMORY;
}

lis_status lis_merge_table_lookup(const lis_merge_table *table,
                                  size_t first, size_t second,
                                  size_t *out_rank, size_t *out_result)
{
    size_t idx;
    size_t i;

    if (table == NULL || out_rank == NULL || out_result == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (table->entries == NULL || table->capacity == 0) {
        return LIS_STATUS_UNSUPPORTED;
    }

    idx = lis_merge_hash(first, second, table->capacity);
    for (i = 0; i < table->capacity; ++i) {
        size_t probe = (idx + i) % table->capacity;

        if (!table->entries[probe].occupied) {
            return LIS_STATUS_UNSUPPORTED;
        }
        if (table->entries[probe].first == first &&
            table->entries[probe].second == second) {
            *out_rank = table->entries[probe].rank;
            *out_result = table->entries[probe].result;
            return LIS_STATUS_OK;
        }
    }
    return LIS_STATUS_UNSUPPORTED;
}

/* --- BPE encode --- */

typedef struct lis_bpe_node {
    size_t token;
    struct lis_bpe_node *prev;
    struct lis_bpe_node *next;
} lis_bpe_node;

static lis_status lis_token_id_append(size_t **ids, size_t *count,
                                      size_t *capacity, size_t token_id)
{
    size_t *new_ids = NULL;

    if (*count == *capacity) {
        size_t new_capacity = *capacity == 0 ? 8 : *capacity * 2U;

        if (new_capacity < *capacity || new_capacity > SIZE_MAX / sizeof(**ids)) {
            return LIS_STATUS_OVERFLOW;
        }
        new_ids = realloc(*ids, new_capacity * sizeof(**ids));
        if (new_ids == NULL) {
            return LIS_STATUS_NO_MEMORY;
        }
        *ids = new_ids;
        *capacity = new_capacity;
    }
    (*ids)[*count] = token_id;
    ++(*count);
    return LIS_STATUS_OK;
}

static lis_status lis_token_ids_append_many(size_t **ids, size_t *count,
                                            size_t *capacity,
                                            const size_t *src,
                                            size_t src_count)
{
    size_t index;
    lis_status status;

    for (index = 0; index < src_count; ++index) {
        status = lis_token_id_append(ids, count, capacity, src[index]);
        if (status != LIS_STATUS_OK) {
            return status;
        }
    }
    return LIS_STATUS_OK;
}

static lis_status lis_tokenizer_encode_bpe_span(const lis_tokenizer *tok,
                                                const char *text,
                                                size_t text_len,
                                                size_t **out_ids,
                                                size_t *out_count)
{
    lis_bpe_node *nodes = NULL;
    lis_bpe_node *head = NULL;
    lis_bpe_node *node;
    size_t node_count;
    size_t *result = NULL;
    size_t i;
    int changed;

    if (tok == NULL || out_ids == NULL || out_count == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (text == NULL && text_len > 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    *out_ids = NULL;
    *out_count = 0;

    if (text_len == 0) {
        return LIS_STATUS_OK;
    }

    if (text_len > SIZE_MAX / sizeof(*nodes)) {
        return LIS_STATUS_OVERFLOW;
    }
    nodes = calloc(text_len, sizeof(*nodes));
    if (nodes == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }

    /* Map each input byte to its initial token ID. */
    for (i = 0; i < text_len; ++i) {
        unsigned char byte_val = (unsigned char)text[i];

        if (!tok->byte_to_token_valid[byte_val]) {
            free(nodes);
            return LIS_STATUS_UNSUPPORTED;
        }
        nodes[i].token = tok->byte_to_token[byte_val];
        nodes[i].prev = (i > 0) ? &nodes[i - 1] : NULL;
        nodes[i].next = (i + 1 < text_len) ? &nodes[i + 1] : NULL;
    }
    head = &nodes[0];
    node_count = text_len;

    /*
     * BPE merge loop: each iteration finds the highest-priority merge
     * (lowest rank) present in the current token sequence, then applies
     * all occurrences of that merge in a single left-to-right pass.
     */
    do {
        size_t best_rank = SIZE_MAX;
        size_t best_result = 0;

        changed = 0;

        for (node = head; node != NULL && node->next != NULL;
             node = node->next) {
            size_t rank;
            size_t res;

            if (lis_merge_table_lookup(&tok->merges, node->token,
                                        node->next->token, &rank,
                                        &res) == LIS_STATUS_OK) {
                if (rank < best_rank) {
                    best_rank = rank;
                    best_result = res;
                }
            }
        }

        if (best_rank == SIZE_MAX) {
            break;
        }

        /* Apply all occurrences of the best-ranked merge. */
        for (node = head; node != NULL && node->next != NULL; ) {
            size_t rank;
            size_t res;

            if (lis_merge_table_lookup(&tok->merges, node->token,
                                        node->next->token, &rank,
                                        &res) == LIS_STATUS_OK &&
                rank == best_rank) {
                lis_bpe_node *removed = node->next;

                node->token = best_result;
                node->next = removed->next;
                if (removed->next != NULL) {
                    removed->next->prev = node;
                }
                --node_count;
                changed = 1;
                node = node->next;
            } else {
                node = node->next;
            }
        }
    } while (changed);

    /* Collect the final token sequence. */
    if (node_count > SIZE_MAX / sizeof(*result)) {
        free(nodes);
        return LIS_STATUS_OVERFLOW;
    }
    result = malloc(node_count * sizeof(*result));
    if (result == NULL) {
        free(nodes);
        return LIS_STATUS_NO_MEMORY;
    }

    i = 0;
    for (node = head; node != NULL; node = node->next) {
        result[i++] = node->token;
    }

    free(nodes);
    *out_ids = result;
    *out_count = node_count;
    return LIS_STATUS_OK;
}

static size_t lis_match_special_token(const lis_tokenizer *tok,
                                      const char *text, size_t text_len,
                                      size_t *out_token_id)
{
    size_t best_len = 0;
    size_t index;

    if (out_token_id != NULL) {
        *out_token_id = 0;
    }
    for (index = 0; index < tok->special_token_count; ++index) {
        const size_t len = tok->special_token_lens[index];

        if (len > best_len && len <= text_len &&
            memcmp(text, tok->special_token_bytes[index], len) == 0) {
            best_len = len;
            if (out_token_id != NULL) {
                *out_token_id = tok->special_token_ids[index];
            }
        }
    }
    return best_len;
}

lis_status lis_tokenizer_encode(const lis_tokenizer *tok,
                                const char *text, size_t text_len,
                                size_t **out_ids, size_t *out_count)
{
    size_t *result = NULL;
    size_t result_count = 0;
    size_t result_capacity = 0;
    size_t cursor = 0;
    size_t span_start = 0;
    lis_status status = LIS_STATUS_OK;

    if (tok == NULL || out_ids == NULL || out_count == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (text == NULL && text_len > 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    *out_ids = NULL;
    *out_count = 0;
    if (text_len == 0) {
        return LIS_STATUS_OK;
    }

    while (cursor < text_len) {
        size_t token_id = 0;
        const size_t special_len =
            lis_match_special_token(tok, text + cursor, text_len - cursor,
                                    &token_id);

        if (special_len == 0) {
            ++cursor;
            continue;
        }
        if (cursor > span_start) {
            size_t *span_ids = NULL;
            size_t span_count = 0;

            status = lis_tokenizer_encode_bpe_span(tok, text + span_start,
                                                   cursor - span_start,
                                                   &span_ids, &span_count);
            if (status == LIS_STATUS_OK) {
                status = lis_token_ids_append_many(&result, &result_count,
                                                   &result_capacity, span_ids,
                                                   span_count);
            }
            free(span_ids);
            if (status != LIS_STATUS_OK) {
                free(result);
                return status;
            }
        }
        status = lis_token_id_append(&result, &result_count,
                                     &result_capacity, token_id);
        if (status != LIS_STATUS_OK) {
            free(result);
            return status;
        }
        cursor += special_len;
        span_start = cursor;
    }
    if (span_start < text_len) {
        size_t *span_ids = NULL;
        size_t span_count = 0;

        status = lis_tokenizer_encode_bpe_span(tok, text + span_start,
                                               text_len - span_start,
                                               &span_ids, &span_count);
        if (status == LIS_STATUS_OK) {
            status = lis_token_ids_append_many(&result, &result_count,
                                               &result_capacity, span_ids,
                                               span_count);
        }
        free(span_ids);
        if (status != LIS_STATUS_OK) {
            free(result);
            return status;
        }
    }

    *out_ids = result;
    *out_count = result_count;
    return LIS_STATUS_OK;
}

/* --- BPE decode --- */

lis_status lis_tokenizer_decode(const lis_tokenizer *tok,
                                const size_t *ids, size_t count,
                                char **out_text, size_t *out_len)
{
    size_t total_len = 0;
    char *text;
    size_t offset;
    size_t i;

    if (tok == NULL || out_text == NULL || out_len == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    if (ids == NULL && count > 0) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }

    *out_text = NULL;
    *out_len = 0;

    for (i = 0; i < count; ++i) {
        if (ids[i] >= tok->vocab_size) {
            return LIS_STATUS_LIMIT_EXCEEDED;
        }
        if (total_len > SIZE_MAX - tok->token_lens[ids[i]]) {
            return LIS_STATUS_OVERFLOW;
        }
        total_len += tok->token_lens[ids[i]];
    }

    text = malloc(total_len + 1);
    if (text == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }

    offset = 0;
    for (i = 0; i < count; ++i) {
        if (tok->token_lens[ids[i]] > 0) {
            memcpy(text + offset, tok->token_bytes[ids[i]],
                   tok->token_lens[ids[i]]);
            offset += tok->token_lens[ids[i]];
        }
    }
    text[total_len] = '\0';

    *out_text = text;
    *out_len = total_len;
    return LIS_STATUS_OK;
}
