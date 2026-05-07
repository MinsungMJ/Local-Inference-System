#ifndef LIS_TOKENIZER_H
#define LIS_TOKENIZER_H

#include "lis/status.h"

#include <stddef.h>

/* --- Token ID batch --- */

typedef struct {
    size_t *tokens;
    size_t token_count;
    size_t *lengths;
    size_t batch_size;
} lis_token_id_batch;

/*
 * Loads a direct token-ID batch from a local text file. Each nonempty
 * line is one static-batch sequence; each token is an unsigned decimal ID.
 * This is a tokenizer boundary adapter, not a text tokenizer implementation.
 */
lis_status lis_token_id_batch_load_path(const char *path,
                                        size_t expected_batch_size,
                                        lis_token_id_batch *out_batch);
lis_status lis_token_id_batch_validate_vocab(const lis_token_id_batch *batch,
                                             size_t vocab_size);
void lis_token_id_batch_destroy(lis_token_id_batch *batch);

/* --- Tokenizer integration --- */

/*
 * Open-addressing hash table entry for BPE merge pair lookup.
 * Maps (first, second) -> (result, rank). Rank 0 = highest priority.
 */
typedef struct {
    size_t first;
    size_t second;
    size_t result;
    size_t rank;
    int occupied;
} lis_merge_entry;

/*
 * Hash table for BPE merge pair lookup.
 */
typedef struct {
    lis_merge_entry *entries;
    size_t capacity;
    size_t count;
} lis_merge_table;

/*
 * Tokenizer state. Owns vocabulary byte strings, a merge table for BPE,
 * and a byte-to-token lookup for initial text segmentation.
 *
 * Loaded from a LIS_VOCAB_V1 file via lis_tokenizer_load().
 * File format:
 *   Line 1: "LIS_VOCAB_V1"
 *   Line 2: "<vocab_size> <merge_count>"
 *   Next vocab_size lines: hex-encoded byte string per token (ID order)
 *   Next merge_count lines: "<first_id> <second_id> <result_id>" (priority order)
 *
 * Merges are listed highest-priority first (rank 0 = first line).
 */
typedef struct {
    char **token_bytes;
    size_t *token_lens;
    size_t vocab_size;
    lis_merge_table merges;
    size_t byte_to_token[256];
    int byte_to_token_valid[256];
    char **special_token_bytes;
    size_t *special_token_lens;
    size_t *special_token_ids;
    size_t special_token_count;
} lis_tokenizer;

/*
 * Load a tokenizer from a LIS_VOCAB_V1 file at the given path.
 * Caller must call lis_tokenizer_destroy() when done.
 */
lis_status lis_tokenizer_load(const char *path, lis_tokenizer *out);

void lis_tokenizer_destroy(lis_tokenizer *tok);

size_t lis_tokenizer_vocab_size(const lis_tokenizer *tok);

/*
 * Encode text into token IDs using byte-pair encoding.
 * Caller owns the returned *out_ids array and must free() it.
 * Empty text (text_len == 0) produces zero tokens.
 */
lis_status lis_tokenizer_encode(const lis_tokenizer *tok,
                                const char *text, size_t text_len,
                                size_t **out_ids, size_t *out_count);

/*
 * Decode token IDs into a byte string.
 * Caller owns the returned *out_text buffer and must free() it.
 * Output is null-terminated; *out_len excludes the terminator.
 */
lis_status lis_tokenizer_decode(const lis_tokenizer *tok,
                                const size_t *ids, size_t count,
                                char **out_text, size_t *out_len);

/* Merge table operations (subsystem-internal, shared across tokenizer files) */
lis_status lis_merge_table_init(lis_merge_table *table, size_t capacity);
void lis_merge_table_destroy(lis_merge_table *table);
lis_status lis_merge_table_insert(lis_merge_table *table,
                                  size_t first, size_t second,
                                  size_t result, size_t rank);
lis_status lis_merge_table_lookup(const lis_merge_table *table,
                                  size_t first, size_t second,
                                  size_t *out_rank, size_t *out_result);

/* Vocabulary parser (subsystem-internal, called by lis_tokenizer_load) */
lis_status lis_vocab_parse(const char *data, size_t data_len,
                           lis_tokenizer *out);

/* --- Minimal JSON subset parser (subsystem-internal) --- */

typedef enum {
    LIS_JSON_NULL,
    LIS_JSON_BOOL,
    LIS_JSON_NUMBER,
    LIS_JSON_STRING,
    LIS_JSON_ARRAY,
    LIS_JSON_OBJECT
} lis_json_type;

typedef struct lis_json_value lis_json_value;

struct lis_json_value {
    lis_json_type type;
    union {
        int boolean;
        double number;
        struct { char *data; size_t len; } string;
        struct { lis_json_value *items; size_t count; } array;
        struct {
            char **keys;
            size_t *key_lens;
            lis_json_value *values;
            size_t count;
        } object;
    } as;
};

/*
 * Parse an in-memory JSON buffer into a value tree. Caller must call
 * lis_json_destroy() on the result. Not a general-purpose JSON library;
 * sufficient for HuggingFace tokenizer.json extraction only.
 */
lis_status lis_json_parse(const char *text, size_t text_len,
                          lis_json_value *out);
void lis_json_destroy(lis_json_value *value);

/* Look up a key in a JSON object. Returns NULL if not found or not object. */
const lis_json_value *lis_json_object_get(const lis_json_value *obj,
                                          const char *key);

/* --- HuggingFace BPE tokenizer import --- */

/*
 * Load a tokenizer from a HuggingFace tokenizer.json file (BPE type only).
 * Populates the same lis_tokenizer struct as lis_tokenizer_load().
 * Caller must call lis_tokenizer_destroy() when done.
 */
lis_status lis_hf_tokenizer_load(const char *path, lis_tokenizer *out);

#endif
