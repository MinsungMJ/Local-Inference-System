#include "lis/tokenizer.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define LIS_JSON_MAX_DEPTH 64

typedef struct {
    const char *cursor;
    const char *end;
    int depth;
} lis_json_parser;

/* --- Forward declarations --- */

static lis_status lis_json_parse_value_r(lis_json_parser *p,
                                         lis_json_value *out);

/* --- Whitespace and character helpers --- */

static void lis_json_skip_ws(lis_json_parser *p)
{
    while (p->cursor < p->end) {
        char ch = *p->cursor;

        if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
            ++p->cursor;
        } else {
            break;
        }
    }
}

static int lis_json_peek(lis_json_parser *p)
{
    lis_json_skip_ws(p);
    if (p->cursor >= p->end) {
        return -1;
    }
    return (unsigned char)*p->cursor;
}

static int lis_json_consume(lis_json_parser *p, char expected)
{
    lis_json_skip_ws(p);
    if (p->cursor < p->end && *p->cursor == expected) {
        ++p->cursor;
        return 1;
    }
    return 0;
}

static int lis_json_match(lis_json_parser *p, const char *word, size_t len)
{
    if ((size_t)(p->end - p->cursor) < len) {
        return 0;
    }
    if (memcmp(p->cursor, word, len) != 0) {
        return 0;
    }
    p->cursor += len;
    return 1;
}

/* --- String parsing --- */

static int lis_json_hex_digit(char ch)
{
    if (ch >= '0' && ch <= '9') {
        return ch - '0';
    }
    if (ch >= 'a' && ch <= 'f') {
        return ch - 'a' + 10;
    }
    if (ch >= 'A' && ch <= 'F') {
        return ch - 'A' + 10;
    }
    return -1;
}

static lis_status lis_json_parse_hex4(lis_json_parser *p, uint32_t *out)
{
    uint32_t value = 0;
    int i;

    if (p->cursor + 4 > p->end) {
        return LIS_STATUS_FORMAT;
    }
    for (i = 0; i < 4; ++i) {
        int d = lis_json_hex_digit(p->cursor[i]);

        if (d < 0) {
            return LIS_STATUS_FORMAT;
        }
        value = (value << 4) | (uint32_t)d;
    }
    p->cursor += 4;
    *out = value;
    return LIS_STATUS_OK;
}

/*
 * Encode a Unicode codepoint as UTF-8 into buf. Returns bytes written (1-4),
 * or 0 on invalid codepoint.
 */
static size_t lis_json_encode_utf8(uint32_t cp, char *buf)
{
    if (cp <= 0x7F) {
        buf[0] = (char)cp;
        return 1;
    }
    if (cp <= 0x7FF) {
        buf[0] = (char)(0xC0 | (cp >> 6));
        buf[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    if (cp <= 0xFFFF) {
        buf[0] = (char)(0xE0 | (cp >> 12));
        buf[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    if (cp <= 0x10FFFF) {
        buf[0] = (char)(0xF0 | (cp >> 18));
        buf[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
        buf[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
        buf[3] = (char)(0x80 | (cp & 0x3F));
        return 4;
    }
    return 0;
}

static lis_status lis_json_parse_string(lis_json_parser *p,
                                        char **out_str, size_t *out_len)
{
    /*
     * Two-pass: first count output bytes, then write. This avoids realloc
     * chains for potentially large strings (tokenizer vocab keys).
     */
    const char *start;
    const char *scan;
    size_t output_len = 0;
    char *result;
    size_t write_pos;

    if (p->cursor >= p->end || *p->cursor != '"') {
        return LIS_STATUS_FORMAT;
    }
    ++p->cursor;
    start = p->cursor;

    /* Pass 1: compute output length */
    scan = p->cursor;
    while (scan < p->end && *scan != '"') {
        if (*scan == '\\') {
            ++scan;
            if (scan >= p->end) {
                return LIS_STATUS_FORMAT;
            }
            switch (*scan) {
            case '"': case '\\': case '/':
                output_len += 1;
                ++scan;
                break;
            case 'b': case 'f': case 'n': case 'r': case 't':
                output_len += 1;
                ++scan;
                break;
            case 'u': {
                uint32_t cp;
                const char *saved = p->cursor;

                p->cursor = scan + 1;
                if (lis_json_parse_hex4(p, &cp) != LIS_STATUS_OK) {
                    p->cursor = saved;
                    return LIS_STATUS_FORMAT;
                }
                /* Handle surrogate pairs */
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    uint32_t lo;

                    if (p->cursor + 2 > p->end ||
                        p->cursor[0] != '\\' || p->cursor[1] != 'u') {
                        p->cursor = saved;
                        return LIS_STATUS_FORMAT;
                    }
                    p->cursor += 2;
                    if (lis_json_parse_hex4(p, &lo) != LIS_STATUS_OK ||
                        lo < 0xDC00 || lo > 0xDFFF) {
                        p->cursor = saved;
                        return LIS_STATUS_FORMAT;
                    }
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                }
                scan = p->cursor;
                p->cursor = saved;

                {
                    char tmp[4];
                    size_t enc_len = lis_json_encode_utf8(cp, tmp);

                    if (enc_len == 0) {
                        return LIS_STATUS_FORMAT;
                    }
                    output_len += enc_len;
                }
                break;
            }
            default:
                return LIS_STATUS_FORMAT;
            }
        } else {
            output_len += 1;
            ++scan;
        }
    }
    if (scan >= p->end) {
        return LIS_STATUS_FORMAT;
    }

    /* Pass 2: write output */
    result = malloc(output_len + 1);
    if (result == NULL) {
        return LIS_STATUS_NO_MEMORY;
    }

    p->cursor = start;
    write_pos = 0;
    while (p->cursor < p->end && *p->cursor != '"') {
        if (*p->cursor == '\\') {
            ++p->cursor;
            switch (*p->cursor) {
            case '"':  result[write_pos++] = '"';  ++p->cursor; break;
            case '\\': result[write_pos++] = '\\'; ++p->cursor; break;
            case '/':  result[write_pos++] = '/';  ++p->cursor; break;
            case 'b':  result[write_pos++] = '\b'; ++p->cursor; break;
            case 'f':  result[write_pos++] = '\f'; ++p->cursor; break;
            case 'n':  result[write_pos++] = '\n'; ++p->cursor; break;
            case 'r':  result[write_pos++] = '\r'; ++p->cursor; break;
            case 't':  result[write_pos++] = '\t'; ++p->cursor; break;
            case 'u': {
                uint32_t cp;

                ++p->cursor;
                (void)lis_json_parse_hex4(p, &cp);
                if (cp >= 0xD800 && cp <= 0xDBFF) {
                    uint32_t lo;

                    p->cursor += 2; /* skip \u */
                    (void)lis_json_parse_hex4(p, &lo);
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                }
                write_pos += lis_json_encode_utf8(cp,
                                                   result + write_pos);
                break;
            }
            default:
                break;
            }
        } else {
            result[write_pos++] = *p->cursor;
            ++p->cursor;
        }
    }
    result[output_len] = '\0';

    /* Skip closing quote */
    ++p->cursor;

    *out_str = result;
    *out_len = output_len;
    return LIS_STATUS_OK;
}

/* --- Number parsing --- */

static lis_status lis_json_parse_number(lis_json_parser *p, double *out)
{
    const char *start = p->cursor;
    char *end_ptr = NULL;
    double value;

    /* Validate that it starts like a JSON number */
    if (p->cursor >= p->end) {
        return LIS_STATUS_FORMAT;
    }
    if (*p->cursor == '-') {
        ++p->cursor;
    }
    if (p->cursor >= p->end ||
        (*p->cursor < '0' || *p->cursor > '9')) {
        p->cursor = start;
        return LIS_STATUS_FORMAT;
    }

    /* Use strtod for the actual parsing */
    p->cursor = start;
    value = strtod(p->cursor, &end_ptr);
    if (end_ptr == p->cursor) {
        return LIS_STATUS_FORMAT;
    }
    p->cursor = end_ptr;
    *out = value;
    return LIS_STATUS_OK;
}

/* --- Object parsing --- */

static lis_status lis_json_parse_object(lis_json_parser *p,
                                        lis_json_value *out)
{
    size_t capacity = 8;
    size_t count = 0;
    char **keys = NULL;
    size_t *key_lens = NULL;
    lis_json_value *values = NULL;
    lis_status status;

    if (p->depth >= LIS_JSON_MAX_DEPTH) {
        return LIS_STATUS_LIMIT_EXCEEDED;
    }
    ++p->depth;

    /* Skip '{' */
    ++p->cursor;

    keys = calloc(capacity, sizeof(*keys));
    key_lens = calloc(capacity, sizeof(*key_lens));
    values = calloc(capacity, sizeof(*values));
    if (keys == NULL || key_lens == NULL || values == NULL) {
        status = LIS_STATUS_NO_MEMORY;
        goto fail;
    }

    if (lis_json_peek(p) == '}') {
        ++p->cursor;
        goto done;
    }

    for (;;) {
        char *key = NULL;
        size_t key_len = 0;

        lis_json_skip_ws(p);
        status = lis_json_parse_string(p, &key, &key_len);
        if (status != LIS_STATUS_OK) {
            goto fail;
        }

        if (!lis_json_consume(p, ':')) {
            free(key);
            status = LIS_STATUS_FORMAT;
            goto fail;
        }

        if (count >= capacity) {
            size_t new_cap = capacity * 2;
            char **new_keys = realloc(keys, new_cap * sizeof(*keys));
            size_t *new_klens = realloc(key_lens,
                                         new_cap * sizeof(*key_lens));
            lis_json_value *new_vals = realloc(values,
                                                new_cap * sizeof(*values));

            if (new_keys == NULL || new_klens == NULL || new_vals == NULL) {
                /* Realloc failure: originals still valid if non-NULL */
                if (new_keys != NULL) {
                    keys = new_keys;
                }
                if (new_klens != NULL) {
                    key_lens = new_klens;
                }
                if (new_vals != NULL) {
                    values = new_vals;
                }
                free(key);
                status = LIS_STATUS_NO_MEMORY;
                goto fail;
            }
            keys = new_keys;
            key_lens = new_klens;
            values = new_vals;
            memset(values + capacity, 0,
                   (new_cap - capacity) * sizeof(*values));
            capacity = new_cap;
        }

        keys[count] = key;
        key_lens[count] = key_len;

        lis_json_skip_ws(p);
        status = lis_json_parse_value_r(p, &values[count]);
        if (status != LIS_STATUS_OK) {
            goto fail;
        }
        ++count;

        if (lis_json_consume(p, '}')) {
            break;
        }
        if (!lis_json_consume(p, ',')) {
            status = LIS_STATUS_FORMAT;
            goto fail;
        }
    }

done:
    --p->depth;
    out->type = LIS_JSON_OBJECT;
    out->as.object.keys = keys;
    out->as.object.key_lens = key_lens;
    out->as.object.values = values;
    out->as.object.count = count;
    return LIS_STATUS_OK;

fail:
    --p->depth;
    {
        size_t i;

        for (i = 0; i < count; ++i) {
            free(keys[i]);
            lis_json_destroy(&values[i]);
        }
    }
    free(keys);
    free(key_lens);
    free(values);
    return status;
}

/* --- Array parsing --- */

static lis_status lis_json_parse_array(lis_json_parser *p,
                                       lis_json_value *out)
{
    size_t capacity = 8;
    size_t count = 0;
    lis_json_value *items = NULL;
    lis_status status;

    if (p->depth >= LIS_JSON_MAX_DEPTH) {
        return LIS_STATUS_LIMIT_EXCEEDED;
    }
    ++p->depth;

    /* Skip '[' */
    ++p->cursor;

    items = calloc(capacity, sizeof(*items));
    if (items == NULL) {
        --p->depth;
        return LIS_STATUS_NO_MEMORY;
    }

    if (lis_json_peek(p) == ']') {
        ++p->cursor;
        goto done;
    }

    for (;;) {
        if (count >= capacity) {
            size_t new_cap = capacity * 2;
            lis_json_value *new_items = realloc(items,
                                                 new_cap * sizeof(*items));

            if (new_items == NULL) {
                status = LIS_STATUS_NO_MEMORY;
                goto fail;
            }
            items = new_items;
            memset(items + capacity, 0,
                   (new_cap - capacity) * sizeof(*items));
            capacity = new_cap;
        }

        lis_json_skip_ws(p);
        status = lis_json_parse_value_r(p, &items[count]);
        if (status != LIS_STATUS_OK) {
            goto fail;
        }
        ++count;

        if (lis_json_consume(p, ']')) {
            break;
        }
        if (!lis_json_consume(p, ',')) {
            status = LIS_STATUS_FORMAT;
            goto fail;
        }
    }

done:
    --p->depth;
    out->type = LIS_JSON_ARRAY;
    out->as.array.items = items;
    out->as.array.count = count;
    return LIS_STATUS_OK;

fail:
    --p->depth;
    {
        size_t i;

        for (i = 0; i < count; ++i) {
            lis_json_destroy(&items[i]);
        }
    }
    free(items);
    return status;
}

/* --- Generic value parsing --- */

static lis_status lis_json_parse_value_r(lis_json_parser *p,
                                         lis_json_value *out)
{
    int ch;

    memset(out, 0, sizeof(*out));
    ch = lis_json_peek(p);
    if (ch < 0) {
        return LIS_STATUS_FORMAT;
    }

    switch (ch) {
    case '"': {
        char *str = NULL;
        size_t len = 0;
        lis_status s = lis_json_parse_string(p, &str, &len);

        if (s != LIS_STATUS_OK) {
            return s;
        }
        out->type = LIS_JSON_STRING;
        out->as.string.data = str;
        out->as.string.len = len;
        return LIS_STATUS_OK;
    }
    case '{':
        return lis_json_parse_object(p, out);
    case '[':
        return lis_json_parse_array(p, out);
    case 't':
        if (!lis_json_match(p, "true", 4)) {
            return LIS_STATUS_FORMAT;
        }
        out->type = LIS_JSON_BOOL;
        out->as.boolean = 1;
        return LIS_STATUS_OK;
    case 'f':
        if (!lis_json_match(p, "false", 5)) {
            return LIS_STATUS_FORMAT;
        }
        out->type = LIS_JSON_BOOL;
        out->as.boolean = 0;
        return LIS_STATUS_OK;
    case 'n':
        if (!lis_json_match(p, "null", 4)) {
            return LIS_STATUS_FORMAT;
        }
        out->type = LIS_JSON_NULL;
        return LIS_STATUS_OK;
    default:
        if (ch == '-' || (ch >= '0' && ch <= '9')) {
            double num = 0.0;
            lis_status s = lis_json_parse_number(p, &num);

            if (s != LIS_STATUS_OK) {
                return s;
            }
            out->type = LIS_JSON_NUMBER;
            out->as.number = num;
            return LIS_STATUS_OK;
        }
        return LIS_STATUS_FORMAT;
    }
}

/* --- Public API --- */

lis_status lis_json_parse(const char *text, size_t text_len,
                          lis_json_value *out)
{
    lis_json_parser parser;
    lis_status status;

    if (text == NULL || out == NULL) {
        return LIS_STATUS_INVALID_ARGUMENT;
    }
    memset(out, 0, sizeof(*out));

    if (text_len == 0) {
        return LIS_STATUS_FORMAT;
    }

    parser.cursor = text;
    parser.end = text + text_len;
    parser.depth = 0;

    status = lis_json_parse_value_r(&parser, out);
    if (status != LIS_STATUS_OK) {
        return status;
    }

    /* Verify no trailing non-whitespace */
    lis_json_skip_ws(&parser);
    if (parser.cursor != parser.end) {
        lis_json_destroy(out);
        memset(out, 0, sizeof(*out));
        return LIS_STATUS_FORMAT;
    }

    return LIS_STATUS_OK;
}

void lis_json_destroy(lis_json_value *value)
{
    size_t i;

    if (value == NULL) {
        return;
    }

    switch (value->type) {
    case LIS_JSON_STRING:
        free(value->as.string.data);
        break;
    case LIS_JSON_ARRAY:
        for (i = 0; i < value->as.array.count; ++i) {
            lis_json_destroy(&value->as.array.items[i]);
        }
        free(value->as.array.items);
        break;
    case LIS_JSON_OBJECT:
        for (i = 0; i < value->as.object.count; ++i) {
            free(value->as.object.keys[i]);
            lis_json_destroy(&value->as.object.values[i]);
        }
        free(value->as.object.keys);
        free(value->as.object.key_lens);
        free(value->as.object.values);
        break;
    default:
        break;
    }
    memset(value, 0, sizeof(*value));
}

const lis_json_value *lis_json_object_get(const lis_json_value *obj,
                                          const char *key)
{
    size_t i;
    size_t key_len;

    if (obj == NULL || key == NULL || obj->type != LIS_JSON_OBJECT) {
        return NULL;
    }

    key_len = strlen(key);
    for (i = 0; i < obj->as.object.count; ++i) {
        if (obj->as.object.key_lens[i] == key_len &&
            memcmp(obj->as.object.keys[i], key, key_len) == 0) {
            return &obj->as.object.values[i];
        }
    }
    return NULL;
}
