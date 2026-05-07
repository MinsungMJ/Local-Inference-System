#include "lis/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_pass;
static int g_fail;

static void expect_status(const char *label, lis_status got,
                          lis_status expected)
{
    if (got == expected) {
        ++g_pass;
    } else {
        fprintf(stderr, "FAIL: %s: expected %s, got %s\n", label,
                lis_status_name(expected), lis_status_name(got));
        ++g_fail;
    }
}

static void expect_size(const char *label, size_t got, size_t expected)
{
    if (got == expected) {
        ++g_pass;
    } else {
        fprintf(stderr, "FAIL: %s: expected %zu, got %zu\n", label,
                expected, got);
        ++g_fail;
    }
}

static void expect_bytes(const char *label, const char *got, size_t got_len,
                         const char *expected, size_t expected_len)
{
    if (got_len == expected_len &&
        (got_len == 0 || memcmp(got, expected, expected_len) == 0)) {
        ++g_pass;
    } else {
        fprintf(stderr, "FAIL: %s: byte string mismatch (got %zu, want %zu)\n",
                label, got_len, expected_len);
        ++g_fail;
    }
}

static int write_file(const char *path, const char *content, size_t len)
{
    FILE *fp = fopen(path, "wb");

    if (fp == NULL) {
        return 0;
    }
    if (fwrite(content, 1, len, fp) != len) {
        fclose(fp);
        return 0;
    }
    fclose(fp);
    return 1;
}

/*
 * Test vocabulary:
 *   0: 'a' (61)      1: 'b' (62)       2: 'c' (63)
 *   3: 'ab' (6162)   4: 'abc' (616263)
 *
 * Merges (priority order):
 *   rank 0: (0, 1) -> 3    a + b  -> ab
 *   rank 1: (3, 2) -> 4    ab + c -> abc
 */
static const char TEST_VOCAB[] =
    "LIS_VOCAB_V1\n"
    "5 2\n"
    "61\n"
    "62\n"
    "63\n"
    "6162\n"
    "616263\n"
    "0 1 3\n"
    "3 2 4\n";

static const char *VOCAB_PATH = "srcs/libs/test_tokenizer_vocab.txt";

/* --- Load / lifecycle tests --- */

static void test_load_valid(void)
{
    lis_tokenizer tok = { 0 };
    lis_status status;

    write_file(VOCAB_PATH, TEST_VOCAB, sizeof(TEST_VOCAB) - 1);

    status = lis_tokenizer_load(VOCAB_PATH, &tok);
    expect_status("load valid", status, LIS_STATUS_OK);
    expect_size("vocab_size", lis_tokenizer_vocab_size(&tok), 5);
    expect_size("byte 'a' valid",
                (size_t)tok.byte_to_token_valid[0x61], 1);
    expect_size("byte 'a' -> 0", tok.byte_to_token[0x61], 0);
    expect_size("byte 'b' -> 1", tok.byte_to_token[0x62], 1);
    expect_size("byte 'c' -> 2", tok.byte_to_token[0x63], 2);

    lis_tokenizer_destroy(&tok);
}

static void test_load_nonexistent(void)
{
    lis_tokenizer tok = { 0 };

    expect_status("load nonexistent",
                  lis_tokenizer_load("srcs/libs/no_such_file.txt", &tok),
                  LIS_STATUS_IO);
}

static void test_load_bad_magic(void)
{
    lis_tokenizer tok = { 0 };
    const char data[] = "WRONG_MAGIC\n3 0\n61\n62\n63\n";

    write_file(VOCAB_PATH, data, sizeof(data) - 1);
    expect_status("load bad magic",
                  lis_tokenizer_load(VOCAB_PATH, &tok), LIS_STATUS_FORMAT);
}

static void test_load_truncated(void)
{
    lis_tokenizer tok = { 0 };
    const char data[] = "LIS_VOCAB_V1\n5 0\n61\n";

    write_file(VOCAB_PATH, data, sizeof(data) - 1);
    expect_status("load truncated",
                  lis_tokenizer_load(VOCAB_PATH, &tok), LIS_STATUS_FORMAT);
}

static void test_load_invalid_hex(void)
{
    lis_tokenizer tok = { 0 };
    const char data[] = "LIS_VOCAB_V1\n2 0\n6G\n62\n";

    write_file(VOCAB_PATH, data, sizeof(data) - 1);
    expect_status("invalid hex digit",
                  lis_tokenizer_load(VOCAB_PATH, &tok), LIS_STATUS_FORMAT);
}

static void test_load_odd_hex(void)
{
    lis_tokenizer tok = { 0 };
    const char data[] = "LIS_VOCAB_V1\n2 0\n6\n62\n";

    write_file(VOCAB_PATH, data, sizeof(data) - 1);
    expect_status("odd hex length",
                  lis_tokenizer_load(VOCAB_PATH, &tok), LIS_STATUS_FORMAT);
}

static void test_load_merge_out_of_range(void)
{
    lis_tokenizer tok = { 0 };
    const char data[] = "LIS_VOCAB_V1\n3 1\n61\n62\n6162\n0 99 2\n";

    write_file(VOCAB_PATH, data, sizeof(data) - 1);
    expect_status("merge id out of range",
                  lis_tokenizer_load(VOCAB_PATH, &tok), LIS_STATUS_FORMAT);
}

static void test_load_null_args(void)
{
    lis_tokenizer tok = { 0 };

    expect_status("load null path",
                  lis_tokenizer_load(NULL, &tok),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_status("load null out",
                  lis_tokenizer_load(VOCAB_PATH, NULL),
                  LIS_STATUS_INVALID_ARGUMENT);
}

/* --- Encode tests --- */

static void test_encode_basic(void)
{
    lis_tokenizer tok = { 0 };
    size_t *ids = NULL;
    size_t count = 0;

    write_file(VOCAB_PATH, TEST_VOCAB, sizeof(TEST_VOCAB) - 1);
    lis_tokenizer_load(VOCAB_PATH, &tok);

    /* "abc" merges: a+b->ab, ab+c->abc => [4] */
    expect_status("encode abc",
                  lis_tokenizer_encode(&tok, "abc", 3, &ids, &count),
                  LIS_STATUS_OK);
    expect_size("abc count", count, 1);
    if (count == 1) {
        expect_size("abc[0]", ids[0], 4);
    }
    free(ids);

    /* "ab" merges: a+b->ab => [3] */
    ids = NULL;
    count = 0;
    expect_status("encode ab",
                  lis_tokenizer_encode(&tok, "ab", 2, &ids, &count),
                  LIS_STATUS_OK);
    expect_size("ab count", count, 1);
    if (count == 1) {
        expect_size("ab[0]", ids[0], 3);
    }
    free(ids);

    /* "a" => no merge, [0] */
    ids = NULL;
    count = 0;
    expect_status("encode a",
                  lis_tokenizer_encode(&tok, "a", 1, &ids, &count),
                  LIS_STATUS_OK);
    expect_size("a count", count, 1);
    if (count == 1) {
        expect_size("a[0]", ids[0], 0);
    }
    free(ids);

    /* "abab" => [3, 3] */
    ids = NULL;
    count = 0;
    expect_status("encode abab",
                  lis_tokenizer_encode(&tok, "abab", 4, &ids, &count),
                  LIS_STATUS_OK);
    expect_size("abab count", count, 2);
    if (count == 2) {
        expect_size("abab[0]", ids[0], 3);
        expect_size("abab[1]", ids[1], 3);
    }
    free(ids);

    /* "cba" => [2, 1, 0] (no applicable merges) */
    ids = NULL;
    count = 0;
    expect_status("encode cba",
                  lis_tokenizer_encode(&tok, "cba", 3, &ids, &count),
                  LIS_STATUS_OK);
    expect_size("cba count", count, 3);
    if (count == 3) {
        expect_size("cba[0]", ids[0], 2);
        expect_size("cba[1]", ids[1], 1);
        expect_size("cba[2]", ids[2], 0);
    }
    free(ids);

    lis_tokenizer_destroy(&tok);
}

static void test_encode_empty(void)
{
    lis_tokenizer tok = { 0 };
    size_t *ids = NULL;
    size_t count = 0;

    write_file(VOCAB_PATH, TEST_VOCAB, sizeof(TEST_VOCAB) - 1);
    lis_tokenizer_load(VOCAB_PATH, &tok);

    expect_status("encode empty",
                  lis_tokenizer_encode(&tok, "", 0, &ids, &count),
                  LIS_STATUS_OK);
    expect_size("empty count", count, 0);
    free(ids);

    lis_tokenizer_destroy(&tok);
}

static void test_encode_unmapped_byte(void)
{
    lis_tokenizer tok = { 0 };
    size_t *ids = NULL;
    size_t count = 0;

    write_file(VOCAB_PATH, TEST_VOCAB, sizeof(TEST_VOCAB) - 1);
    lis_tokenizer_load(VOCAB_PATH, &tok);

    expect_status("encode unmapped",
                  lis_tokenizer_encode(&tok, "xyz", 3, &ids, &count),
                  LIS_STATUS_UNSUPPORTED);

    lis_tokenizer_destroy(&tok);
}

static void test_encode_no_merges(void)
{
    lis_tokenizer tok = { 0 };
    size_t *ids = NULL;
    size_t count = 0;
    const char data[] = "LIS_VOCAB_V1\n3 0\n61\n62\n63\n";

    write_file(VOCAB_PATH, data, sizeof(data) - 1);
    expect_status("no-merge load",
                  lis_tokenizer_load(VOCAB_PATH, &tok), LIS_STATUS_OK);

    expect_status("no-merge encode",
                  lis_tokenizer_encode(&tok, "abc", 3, &ids, &count),
                  LIS_STATUS_OK);
    expect_size("no-merge count", count, 3);
    if (count == 3) {
        expect_size("no-merge[0]", ids[0], 0);
        expect_size("no-merge[1]", ids[1], 1);
        expect_size("no-merge[2]", ids[2], 2);
    }
    free(ids);

    lis_tokenizer_destroy(&tok);
}

static void test_encode_null_args(void)
{
    lis_tokenizer tok = { 0 };
    size_t *ids = NULL;
    size_t count = 0;

    expect_status("encode null tok",
                  lis_tokenizer_encode(NULL, "a", 1, &ids, &count),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_status("encode null out_ids",
                  lis_tokenizer_encode(&tok, "a", 1, NULL, &count),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_status("encode null out_count",
                  lis_tokenizer_encode(&tok, "a", 1, &ids, NULL),
                  LIS_STATUS_INVALID_ARGUMENT);
}

/* --- Decode tests --- */

static void test_decode_basic(void)
{
    lis_tokenizer tok = { 0 };
    char *text = NULL;
    size_t text_len = 0;
    size_t ids[3];

    write_file(VOCAB_PATH, TEST_VOCAB, sizeof(TEST_VOCAB) - 1);
    lis_tokenizer_load(VOCAB_PATH, &tok);

    /* [4] -> "abc" */
    ids[0] = 4;
    expect_status("decode [4]",
                  lis_tokenizer_decode(&tok, ids, 1, &text, &text_len),
                  LIS_STATUS_OK);
    expect_bytes("decode [4] text", text, text_len, "abc", 3);
    free(text);

    /* [3, 2] -> "abc" */
    ids[0] = 3;
    ids[1] = 2;
    text = NULL;
    text_len = 0;
    expect_status("decode [3,2]",
                  lis_tokenizer_decode(&tok, ids, 2, &text, &text_len),
                  LIS_STATUS_OK);
    expect_bytes("decode [3,2] text", text, text_len, "abc", 3);
    free(text);

    /* [0, 1, 2] -> "abc" */
    ids[0] = 0;
    ids[1] = 1;
    ids[2] = 2;
    text = NULL;
    text_len = 0;
    expect_status("decode [0,1,2]",
                  lis_tokenizer_decode(&tok, ids, 3, &text, &text_len),
                  LIS_STATUS_OK);
    expect_bytes("decode [0,1,2] text", text, text_len, "abc", 3);
    free(text);

    lis_tokenizer_destroy(&tok);
}

static void test_decode_empty(void)
{
    lis_tokenizer tok = { 0 };
    char *text = NULL;
    size_t text_len = 0;

    write_file(VOCAB_PATH, TEST_VOCAB, sizeof(TEST_VOCAB) - 1);
    lis_tokenizer_load(VOCAB_PATH, &tok);

    expect_status("decode empty",
                  lis_tokenizer_decode(&tok, NULL, 0, &text, &text_len),
                  LIS_STATUS_OK);
    expect_size("decode empty len", text_len, 0);
    free(text);

    lis_tokenizer_destroy(&tok);
}

static void test_decode_out_of_range(void)
{
    lis_tokenizer tok = { 0 };
    char *text = NULL;
    size_t text_len = 0;
    size_t bad_id = 999;

    write_file(VOCAB_PATH, TEST_VOCAB, sizeof(TEST_VOCAB) - 1);
    lis_tokenizer_load(VOCAB_PATH, &tok);

    expect_status("decode out-of-range",
                  lis_tokenizer_decode(&tok, &bad_id, 1, &text, &text_len),
                  LIS_STATUS_LIMIT_EXCEEDED);

    lis_tokenizer_destroy(&tok);
}

static void test_decode_null_args(void)
{
    lis_tokenizer tok = { 0 };
    char *text = NULL;
    size_t text_len = 0;
    size_t id = 0;

    expect_status("decode null tok",
                  lis_tokenizer_decode(NULL, &id, 1, &text, &text_len),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_status("decode null out_text",
                  lis_tokenizer_decode(&tok, &id, 1, NULL, &text_len),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_status("decode null out_len",
                  lis_tokenizer_decode(&tok, &id, 1, &text, NULL),
                  LIS_STATUS_INVALID_ARGUMENT);
}

/* --- Round-trip tests --- */

static void test_roundtrip(void)
{
    lis_tokenizer tok = { 0 };
    size_t *ids = NULL;
    size_t count = 0;
    char *text = NULL;
    size_t text_len = 0;

    write_file(VOCAB_PATH, TEST_VOCAB, sizeof(TEST_VOCAB) - 1);
    lis_tokenizer_load(VOCAB_PATH, &tok);

    expect_status("rt encode",
                  lis_tokenizer_encode(&tok, "abcab", 5, &ids, &count),
                  LIS_STATUS_OK);
    expect_status("rt decode",
                  lis_tokenizer_decode(&tok, ids, count, &text, &text_len),
                  LIS_STATUS_OK);
    expect_bytes("rt result", text, text_len, "abcab", 5);

    free(ids);
    free(text);
    lis_tokenizer_destroy(&tok);
}

static void test_roundtrip_single_bytes(void)
{
    lis_tokenizer tok = { 0 };
    size_t *ids = NULL;
    size_t count = 0;
    char *text = NULL;
    size_t text_len = 0;

    write_file(VOCAB_PATH, TEST_VOCAB, sizeof(TEST_VOCAB) - 1);
    lis_tokenizer_load(VOCAB_PATH, &tok);

    expect_status("rt single encode",
                  lis_tokenizer_encode(&tok, "cba", 3, &ids, &count),
                  LIS_STATUS_OK);
    expect_status("rt single decode",
                  lis_tokenizer_decode(&tok, ids, count, &text, &text_len),
                  LIS_STATUS_OK);
    expect_bytes("rt single result", text, text_len, "cba", 3);

    free(ids);
    free(text);
    lis_tokenizer_destroy(&tok);
}

/* --- Merge table edge cases --- */

static void test_merge_table_basic(void)
{
    lis_merge_table table = { 0 };
    size_t rank;
    size_t result;

    expect_status("mt init",
                  lis_merge_table_init(&table, 16), LIS_STATUS_OK);

    expect_status("mt insert",
                  lis_merge_table_insert(&table, 10, 20, 30, 0),
                  LIS_STATUS_OK);

    expect_status("mt lookup found",
                  lis_merge_table_lookup(&table, 10, 20, &rank, &result),
                  LIS_STATUS_OK);
    expect_size("mt rank", rank, 0);
    expect_size("mt result", result, 30);

    expect_status("mt lookup miss",
                  lis_merge_table_lookup(&table, 10, 99, &rank, &result),
                  LIS_STATUS_UNSUPPORTED);

    lis_merge_table_destroy(&table);
}

static void test_destroy_null(void)
{
    lis_tokenizer_destroy(NULL);
    lis_merge_table_destroy(NULL);
    ++g_pass;
}

int main(void)
{
    g_pass = 0;
    g_fail = 0;

    test_load_valid();
    test_load_nonexistent();
    test_load_bad_magic();
    test_load_truncated();
    test_load_invalid_hex();
    test_load_odd_hex();
    test_load_merge_out_of_range();
    test_load_null_args();

    test_encode_basic();
    test_encode_empty();
    test_encode_unmapped_byte();
    test_encode_no_merges();
    test_encode_null_args();

    test_decode_basic();
    test_decode_empty();
    test_decode_out_of_range();
    test_decode_null_args();

    test_roundtrip();
    test_roundtrip_single_bytes();

    test_merge_table_basic();
    test_destroy_null();

    printf("test_tokenizer: %d passed, %d failed\n", g_pass, g_fail);

    remove(VOCAB_PATH);

    return g_fail > 0 ? 1 : 0;
}
