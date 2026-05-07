#include "lis/tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * GPT-2 byte-to-unicode reverse mapping.
 *
 * HuggingFace byte-level BPE tokenizers represent raw byte values as Unicode
 * codepoints using a fixed bijective mapping from GPT-2. This file reverses
 * that mapping during import so that LIS stores raw byte strings internally.
 *
 * Direct-mapped bytes (188 values): codepoint == byte value.
 *   33-126, 161-172, 174-255
 *
 * Indirect-mapped bytes (68 values): codepoints 256-323.
 *   Bytes 0-32   -> codepoints 256-288
 *   Bytes 127-160 -> codepoints 289-322
 *   Byte 173     -> codepoint 323
 */

/* Build the codepoint-to-byte table. Returns 1 on valid codepoint, 0 otherwise. */
static int lis_hf_codepoint_to_byte(uint32_t cp, unsigned char *out)
{
    /* Direct-mapped ranges */
    if (cp >= 33 && cp <= 126) {
        *out = (unsigned char)cp;
        return 1;
    }
    if (cp >= 161 && cp <= 172) {
        *out = (unsigned char)cp;
        return 1;
    }
    if (cp >= 174 && cp <= 255) {
        *out = (unsigned char)cp;
        return 1;
    }

    /* Indirect-mapped: codepoints 256-323 */
    if (cp >= 256 && cp <= 288) {
        /* bytes 0-32 */
        *out = (unsigned char)(cp - 256);
        return 1;
    }
    if (cp >= 289 && cp <= 322) {
        /* bytes 127-160 */
        *out = (unsigned char)(cp - 289 + 127);
        return 1;
    }
    if (cp == 323) {
        /* byte 173 */
        *out = 173;
        return 1;
    }

    return 0;
}

/*
 * Decode a UTF-8 string (HF token key) into raw bytes by reversing the
 * GPT-2 byte-to-unicode mapping. Each decoded Unicode codepoint must map
 * to exactly one byte via the mapping above.
 *
 * Returns LIS_STATUS_OK on success. Caller owns *out_bytes.
 */
static lis_status lis_hf_decode_token_string(const char *utf8, size_t utf8_len,
                                             char **out_bytes,
                                             size_t *out_byte_len)
{
    const unsigned char *p = (const unsigned char *)utf8;
    const unsigned char *end = p + utf8_len;
    size_t count = 0;
    char *result;
    size_t write_pos;
    const unsigned char *scan;

    /* Pass 1: count output bytes (one per codepoint) */
    scan = p;
    while (scan < end) {
        uint32_t cp;
        unsigned char dummy;

        if (scan[0] <= 0x7F) {
            cp = scan[0];
            scan += 1;
        } else if ((scan[0] & 0xE0) == 0xC0 && scan + 1 < end) {
            cp = ((uint32_t)(scan[0] & 0x1F) << 6) |
                 ((uint32_t)(scan[1] & 0x3F));
            scan += 2;
        } else if ((scan[0] & 0xF0) == 0xE0 && scan + 2 < end) {
            cp = ((uint32_t)(scan[0] & 0x0F) << 12) |
                 ((uint32_t)(scan[1] & 0x3F) << 6) |
                 ((uint32_t)(scan[2] & 0x3F));
            scan += 3;
        } else if ((scan[0] & 0xF8) == 0xF0 && scan + 3 < end) {
            cp = ((uint32_t)(scan[0] & 0x07) << 18) |
                 ((uint32_t)(scan[1] & 0x3F) << 12) |
                 ((uint32_t)(scan[2] & 0x3F) << 6) |
                 ((uint32_t)(scan[3] & 0x3F));
            scan += 4;
        } else {
            return LIS_STATUS_FORMAT;
        }

        if (!lis_hf_codepoint_to_byte(cp, &dummy)) {
            return LIS_STATUS_FORMAT;
        }
        ++count;
    }

    if (count == 0) {
        *out_bytes = NULL;
        *out_byte_len = 0;
        return LIS_STATUS_OK;
    }

    result = malloc(count + 1);
    if (result == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }

    /* Pass 2: write bytes */
    scan = p;
    write_pos = 0;
    while (scan < end) {
        uint32_t cp;
        unsigned char byte_val = 0;

        if (scan[0] <= 0x7F) {
            cp = scan[0];
            scan += 1;
        } else if ((scan[0] & 0xE0) == 0xC0) {
            cp = ((uint32_t)(scan[0] & 0x1F) << 6) |
                 ((uint32_t)(scan[1] & 0x3F));
            scan += 2;
        } else if ((scan[0] & 0xF0) == 0xE0) {
            cp = ((uint32_t)(scan[0] & 0x0F) << 12) |
                 ((uint32_t)(scan[1] & 0x3F) << 6) |
                 ((uint32_t)(scan[2] & 0x3F));
            scan += 3;
        } else {
            cp = ((uint32_t)(scan[0] & 0x07) << 18) |
                 ((uint32_t)(scan[1] & 0x3F) << 12) |
                 ((uint32_t)(scan[2] & 0x3F) << 6) |
                 ((uint32_t)(scan[3] & 0x3F));
            scan += 4;
        }

        (void)lis_hf_codepoint_to_byte(cp, &byte_val);
        result[write_pos++] = (char)byte_val;
    }
    result[count] = '\0';

    *out_bytes = result;
    *out_byte_len = count;
    return LIS_STATUS_OK;
}

/* --- Temporary string-to-ID hash table for merge resolution --- */

typedef struct {
    char *key;
    size_t key_len;
    size_t value;
    int occupied;
} lis_str_id_entry;

typedef struct {
    lis_str_id_entry *entries;
    size_t capacity;
} lis_str_id_table;

static size_t lis_str_hash(const char *data, size_t len)
{
    size_t h = 5381;
    size_t i;

    for (i = 0; i < len; ++i) {
        h = ((h << 5) + h) ^ (unsigned char)data[i];
    }
    return h;
}

static void lis_str_id_table_destroy(lis_str_id_table *table)
{
    if (table == NULL) {
        return;
    }
    free(table->entries);
    table->entries = NULL;
    table->capacity = 0;
}

static lis_status lis_str_id_table_init(lis_str_id_table *table,
                                        size_t capacity)
{
    table->entries = calloc(capacity, sizeof(*table->entries));
    if (table->entries == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }
    table->capacity = capacity;
    return LIS_STATUS_OK;
}

static lis_status lis_str_id_table_insert(lis_str_id_table *table,
                                          const char *key, size_t key_len,
                                          size_t value)
{
    size_t idx = lis_str_hash(key, key_len) % table->capacity;
    size_t i;

    for (i = 0; i < table->capacity; ++i) {
        size_t probe = (idx + i) % table->capacity;

        if (!table->entries[probe].occupied) {
            table->entries[probe].key = (char *)key;
            table->entries[probe].key_len = key_len;
            table->entries[probe].value = value;
            table->entries[probe].occupied = 1;
            return LIS_STATUS_OK;
        }
        if (table->entries[probe].key_len == key_len &&
            memcmp(table->entries[probe].key, key, key_len) == 0) {
            table->entries[probe].value = value;
            return LIS_STATUS_OK;
        }
    }
    return LIS_STATUS_NO_MEMORY;
}

static int lis_str_id_table_lookup(const lis_str_id_table *table,
                                   const char *key, size_t key_len,
                                   size_t *out_value)
{
    size_t idx = lis_str_hash(key, key_len) % table->capacity;
    size_t i;

    for (i = 0; i < table->capacity; ++i) {
        size_t probe = (idx + i) % table->capacity;

        if (!table->entries[probe].occupied) {
            return 0;
        }
        if (table->entries[probe].key_len == key_len &&
            memcmp(table->entries[probe].key, key, key_len) == 0) {
            *out_value = table->entries[probe].value;
            return 1;
        }
    }
    return 0;
}

/* --- File reading (same pattern as tokenizer.c) --- */

static lis_status lis_hf_read_file(const char *path, char **out_data,
                                   size_t *out_len)
{
    FILE *fp = NULL;
    long file_size;
    char *data = NULL;
    lis_status status = LIS_STATUS_IO;

    fp = fopen(path, "rb");
    if (fp == NULL) {
        return LIS_STATUS_IO;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        goto out;
    }
    file_size = ftell(fp);
    if (file_size <= 0) {
        status = LIS_STATUS_FORMAT;
        goto out;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        goto out;
    }

    data = malloc((size_t)file_size + 1U);
    if (data == NULL) {
        status = LIS_STATUS_NO_MEMORY;
        goto out;
    }
    if (fread(data, 1, (size_t)file_size, fp) != (size_t)file_size) {
        goto out;
    }
    data[(size_t)file_size] = '\0';

    *out_data = data;
    *out_len = (size_t)file_size;
    data = NULL;
    status = LIS_STATUS_OK;

out:
    free(data);
    if (fp != NULL && fclose(fp) != 0 && status == LIS_STATUS_OK) {
        status = LIS_STATUS_IO;
    }
    return status;
}

/* --- Core import logic --- */

/*
 * Extract vocab from model.vocab JSON object.
 * Populates tok->token_bytes, tok->token_lens, tok->vocab_size.
 * Also builds string-to-ID table in *str_table for merge resolution.
 */
static lis_status lis_hf_extract_vocab(const lis_json_value *vocab_obj,
                                       lis_tokenizer *tok,
                                       lis_str_id_table *str_table)
{
    size_t vocab_size;
    size_t i;
    size_t max_id = 0;
    lis_status status;

    if (vocab_obj->type != LIS_JSON_OBJECT) {
        return LIS_STATUS_FORMAT;
    }

    vocab_size = vocab_obj->as.object.count;
    if (vocab_size == 0) {
        return LIS_STATUS_FORMAT;
    }

    /* Find the maximum token ID to determine array size. */
    for (i = 0; i < vocab_size; ++i) {
        const lis_json_value *val = &vocab_obj->as.object.values[i];
        size_t id;

        if (val->type != LIS_JSON_NUMBER) {
            return LIS_STATUS_FORMAT;
        }
        if (val->as.number < 0.0 || val->as.number != (double)(size_t)val->as.number) {
            return LIS_STATUS_FORMAT;
        }
        id = (size_t)val->as.number;
        if (id > max_id) {
            max_id = id;
        }
    }

    tok->vocab_size = max_id + 1;
    tok->token_bytes = calloc(tok->vocab_size, sizeof(*tok->token_bytes));
    tok->token_lens = calloc(tok->vocab_size, sizeof(*tok->token_lens));
    if (tok->token_bytes == NULL || tok->token_lens == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }

    /* Init string-to-ID table for merge resolution (use original HF strings) */
    {
        size_t table_cap = vocab_size * 3;

        if (table_cap < 16) {
            table_cap = 16;
        }
        status = lis_str_id_table_init(str_table, table_cap);
        if (status != LIS_STATUS_OK) {
            return status;
        }
    }

    /* Decode each vocab entry */
    for (i = 0; i < vocab_size; ++i) {
        const char *key = vocab_obj->as.object.keys[i];
        size_t key_len = vocab_obj->as.object.key_lens[i];
        size_t id = (size_t)vocab_obj->as.object.values[i].as.number;
        char *decoded = NULL;
        size_t decoded_len = 0;

        status = lis_hf_decode_token_string(key, key_len,
                                            &decoded, &decoded_len);
        if (status != LIS_STATUS_OK) {
            return status;
        }

        if (tok->token_bytes[id] != NULL) {
            /* Duplicate ID — reject */
            free(decoded);
            return LIS_STATUS_FORMAT;
        }
        tok->token_bytes[id] = decoded;
        tok->token_lens[id] = decoded_len;

        /* Insert original HF string into lookup table */
        status = lis_str_id_table_insert(str_table, key, key_len, id);
        if (status != LIS_STATUS_OK) {
            return status;
        }
    }

    return LIS_STATUS_OK;
}

static lis_status lis_hf_ensure_vocab_size(lis_tokenizer *tok,
                                           size_t required_size)
{
    char **new_bytes = NULL;
    size_t *new_lens = NULL;

    if (required_size <= tok->vocab_size) {
        return LIS_STATUS_OK;
    }
    if (required_size > SIZE_MAX / sizeof(*new_bytes)) {
        return LIS_STATUS_OVERFLOW;
    }
    new_bytes = realloc(tok->token_bytes, required_size * sizeof(*new_bytes));
    if (new_bytes == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }
    tok->token_bytes = new_bytes;
    new_lens = realloc(tok->token_lens, required_size * sizeof(*new_lens));
    if (new_lens == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }
    tok->token_lens = new_lens;
    memset(tok->token_bytes + tok->vocab_size, 0,
           (required_size - tok->vocab_size) * sizeof(*tok->token_bytes));
    memset(tok->token_lens + tok->vocab_size, 0,
           (required_size - tok->vocab_size) * sizeof(*tok->token_lens));
    tok->vocab_size = required_size;
    return LIS_STATUS_OK;
}

static lis_status lis_hf_add_special_token(lis_tokenizer *tok,
                                           const char *content,
                                           size_t content_len,
                                           size_t id)
{
    char **new_bytes = NULL;
    size_t *new_lens = NULL;
    size_t *new_ids = NULL;
    char *copy = NULL;
    lis_status status;

    status = lis_hf_ensure_vocab_size(tok, id + 1U);
    if (status != LIS_STATUS_OK) {
        return status;
    }
    if (tok->token_bytes[id] == NULL) {
        tok->token_bytes[id] = malloc(content_len + 1U);
        if (tok->token_bytes[id] == NULL) {
            return LIS_STATUS_NO_MEMORY;
        }
        memcpy(tok->token_bytes[id], content, content_len);
        tok->token_bytes[id][content_len] = '\0';
        tok->token_lens[id] = content_len;
    } else if (tok->token_lens[id] != content_len ||
               memcmp(tok->token_bytes[id], content, content_len) != 0) {
        return LIS_STATUS_FORMAT;
    }

    new_bytes = realloc(tok->special_token_bytes,
                        (tok->special_token_count + 1U) * sizeof(*new_bytes));
    if (new_bytes == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }
    tok->special_token_bytes = new_bytes;
    new_lens = realloc(tok->special_token_lens,
                       (tok->special_token_count + 1U) * sizeof(*new_lens));
    if (new_lens == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }
    tok->special_token_lens = new_lens;
    new_ids = realloc(tok->special_token_ids,
                      (tok->special_token_count + 1U) * sizeof(*new_ids));
    if (new_ids == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }
    tok->special_token_ids = new_ids;

    copy = malloc(content_len + 1U);
    if (copy == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }
    memcpy(copy, content, content_len);
    copy[content_len] = '\0';

    tok->special_token_bytes[tok->special_token_count] = copy;
    tok->special_token_lens[tok->special_token_count] = content_len;
    tok->special_token_ids[tok->special_token_count] = id;
    ++tok->special_token_count;
    return LIS_STATUS_OK;
}

static lis_status lis_hf_extract_added_tokens(const lis_json_value *root,
                                              lis_tokenizer *tok)
{
    const lis_json_value *added = lis_json_object_get(root, "added_tokens");
    size_t index;

    if (added == NULL) {
        return LIS_STATUS_OK;
    }
    if (added->type != LIS_JSON_ARRAY) {
        return LIS_STATUS_FORMAT;
    }
    for (index = 0; index < added->as.array.count; ++index) {
        const lis_json_value *entry = &added->as.array.items[index];
        const lis_json_value *id_val = NULL;
        const lis_json_value *content_val = NULL;
        const lis_json_value *special_val = NULL;
        size_t id;
        lis_status status;

        if (entry->type != LIS_JSON_OBJECT) {
            return LIS_STATUS_FORMAT;
        }
        id_val = lis_json_object_get(entry, "id");
        content_val = lis_json_object_get(entry, "content");
        special_val = lis_json_object_get(entry, "special");
        if (id_val == NULL || id_val->type != LIS_JSON_NUMBER ||
            content_val == NULL || content_val->type != LIS_JSON_STRING) {
            return LIS_STATUS_FORMAT;
        }
        if (special_val != NULL && special_val->type == LIS_JSON_BOOL &&
            !special_val->as.boolean) {
            continue;
        }
        if (id_val->as.number < 0.0 ||
            id_val->as.number != (double)(size_t)id_val->as.number) {
            return LIS_STATUS_FORMAT;
        }
        id = (size_t)id_val->as.number;
        status = lis_hf_add_special_token(tok, content_val->as.string.data,
                                          content_val->as.string.len, id);
        if (status != LIS_STATUS_OK) {
            return status;
        }
    }
    return LIS_STATUS_OK;
}

/*
 * Parse merge rules from model.merges JSON array.
 *
 * Supports two entry formats:
 * 1. Legacy string: "token_a token_b" (space-delimited)
 * 2. Modern array: ["token_a", "token_b"]
 *
 * Look up IDs via str_table, concatenate to find result ID,
 * and insert into tok->merges.
 */
static lis_status lis_hf_extract_merges(const lis_json_value *merges_arr,
                                        lis_tokenizer *tok,
                                        const lis_str_id_table *str_table)
{
    size_t merge_count;
    size_t table_cap;
    size_t i;
    lis_status status;

    if (merges_arr->type != LIS_JSON_ARRAY) {
        return LIS_STATUS_FORMAT;
    }

    merge_count = merges_arr->as.array.count;
    if (merge_count == 0) {
        return LIS_STATUS_OK;
    }

    table_cap = merge_count * 2;
    if (table_cap < 16) {
        table_cap = 16;
    }
    status = lis_merge_table_init(&tok->merges, table_cap);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    for (i = 0; i < merge_count; ++i) {
        const lis_json_value *entry = &merges_arr->as.array.items[i];
        const char *first_str;
        size_t first_len;
        const char *second_str;
        size_t second_len;
        size_t first_id;
        size_t second_id;
        size_t result_id;
        char *concat = NULL;
        size_t concat_len;

        if (entry->type == LIS_JSON_STRING) {
            /*
             * Legacy HF tokenizer.json format: each merge is a single
             * space-delimited string, e.g. "Ġ Ġ".
             */
            const char *str = entry->as.string.data;
            size_t str_len = entry->as.string.len;
            const char *space = memchr(str, ' ', str_len);

            if (space == NULL) {
                return LIS_STATUS_FORMAT;
            }
            first_str = str;
            first_len = (size_t)(space - str);
            second_str = space + 1;
            second_len = str_len - first_len - 1;
        } else if (entry->type == LIS_JSON_ARRAY &&
                   entry->as.array.count == 2 &&
                   entry->as.array.items[0].type == LIS_JSON_STRING &&
                   entry->as.array.items[1].type == LIS_JSON_STRING) {
            /*
             * Modern HF tokenizer.json format (e.g. Qwen3-8B): each
             * merge is an array of two strings, e.g. ["Ġ", "Ġ"].
             */
            first_str = entry->as.array.items[0].as.string.data;
            first_len = entry->as.array.items[0].as.string.len;
            second_str = entry->as.array.items[1].as.string.data;
            second_len = entry->as.array.items[1].as.string.len;
        } else {
            return LIS_STATUS_FORMAT;
        }

        /* Look up first and second token IDs */
        if (!lis_str_id_table_lookup(str_table, first_str, first_len,
                                     &first_id)) {
            return LIS_STATUS_FORMAT;
        }
        if (!lis_str_id_table_lookup(str_table, second_str, second_len,
                                     &second_id)) {
            return LIS_STATUS_FORMAT;
        }

        /* Build concatenated string and look up result ID */
        concat_len = first_len + second_len;
        concat = malloc(concat_len + 1);
        if (concat == NULL) {
            return LIS_STATUS_NO_MEMORY;
        }
        memcpy(concat, first_str, first_len);
        memcpy(concat + first_len, second_str, second_len);
        concat[concat_len] = '\0';

        if (!lis_str_id_table_lookup(str_table, concat, concat_len,
                                     &result_id)) {
            free(concat);
            return LIS_STATUS_FORMAT;
        }
        free(concat);

        status = lis_merge_table_insert(&tok->merges, first_id, second_id,
                                        result_id, i);
        if (status != LIS_STATUS_OK) {
            return status;
        }
    }

    return LIS_STATUS_OK;
}

/* Build byte_to_token lookup from single-byte vocabulary entries. */
static void lis_hf_build_byte_lookup(lis_tokenizer *tok)
{
    size_t i;

    memset(tok->byte_to_token_valid, 0, sizeof(tok->byte_to_token_valid));
    for (i = 0; i < tok->vocab_size; ++i) {
        if (tok->token_bytes[i] != NULL && tok->token_lens[i] == 1) {
            unsigned char bval = (unsigned char)tok->token_bytes[i][0];

            if (!tok->byte_to_token_valid[bval]) {
                tok->byte_to_token[bval] = i;
                tok->byte_to_token_valid[bval] = 1;
            }
        }
    }
}

/* --- Public API --- */

lis_status lis_hf_tokenizer_load(const char *path, lis_tokenizer *out)
{
    char *file_data = NULL;
    size_t file_len = 0;
    lis_json_value root;
    int root_valid = 0;
    lis_str_id_table str_table;
    int str_table_valid = 0;
    const lis_json_value *model;
    const lis_json_value *type_val;
    const lis_json_value *vocab_val;
    const lis_json_value *merges_val;
    lis_status status;

    if (path == NULL || out == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));
    memset(&root, 0, sizeof(root));
    memset(&str_table, 0, sizeof(str_table));

    /* Read and parse JSON */
    status = lis_hf_read_file(path, &file_data, &file_len);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    status = lis_json_parse(file_data, file_len, &root);
    free(file_data);
    file_data = NULL;
    if (status != LIS_STATUS_OK) {
        return status;
    }
    root_valid = 1;

    /* Navigate to model object */
    model = lis_json_object_get(&root, "model");
    if (model == NULL || model->type != LIS_JSON_OBJECT) {
        status = LIS_STATUS_FORMAT;
        goto out;
    }

    /* Validate model.type == "BPE" */
    type_val = lis_json_object_get(model, "type");
    if (type_val == NULL || type_val->type != LIS_JSON_STRING) {
        status = LIS_STATUS_UNSUPPORTED;
        goto out;
    }
    if (type_val->as.string.len != 3 ||
        memcmp(type_val->as.string.data, "BPE", 3) != 0) {
        status = LIS_STATUS_UNSUPPORTED;
        goto out;
    }

    /* Get model.vocab and model.merges */
    vocab_val = lis_json_object_get(model, "vocab");
    if (vocab_val == NULL || vocab_val->type != LIS_JSON_OBJECT) {
        status = LIS_STATUS_FORMAT;
        goto out;
    }

    merges_val = lis_json_object_get(model, "merges");
    if (merges_val == NULL || merges_val->type != LIS_JSON_ARRAY) {
        status = LIS_STATUS_FORMAT;
        goto out;
    }

    /* Extract vocabulary */
    status = lis_hf_extract_vocab(vocab_val, out, &str_table);
    if (status != LIS_STATUS_OK) {
        goto out;
    }
    str_table_valid = 1;

    status = lis_hf_extract_added_tokens(&root, out);
    if (status != LIS_STATUS_OK) {
        goto out;
    }

    /* Extract merges */
    status = lis_hf_extract_merges(merges_val, out, &str_table);
    if (status != LIS_STATUS_OK) {
        goto out;
    }

    /* Build byte-to-token lookup */
    lis_hf_build_byte_lookup(out);

    status = LIS_STATUS_OK;

out:
    if (str_table_valid) {
        lis_str_id_table_destroy(&str_table);
    }
    if (root_valid) {
        lis_json_destroy(&root);
    }
    if (status != LIS_STATUS_OK) {
        lis_tokenizer_destroy(out);
    }
    return status;
}
