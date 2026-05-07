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

/* ===== JSON parser tests ===== */

static void test_json_parse_valid(void)
{
    const char *json = "{\"a\": 42, \"b\": \"hello\", \"c\": true, \"d\": null, "
                       "\"e\": [1, 2, 3]}";
    lis_json_value root;
    const lis_json_value *val;

    expect_status("json_valid_parse",
                  lis_json_parse(json, strlen(json), &root), LIS_STATUS_OK);

    val = lis_json_object_get(&root, "a");
    if (val != NULL && val->type == LIS_JSON_NUMBER) {
        expect_size("json_valid_a", (size_t)val->as.number, 42);
    } else {
        fprintf(stderr, "FAIL: json_valid_a: missing or wrong type\n");
        ++g_fail;
    }

    val = lis_json_object_get(&root, "b");
    if (val != NULL && val->type == LIS_JSON_STRING) {
        expect_bytes("json_valid_b", val->as.string.data, val->as.string.len,
                     "hello", 5);
    } else {
        fprintf(stderr, "FAIL: json_valid_b: missing or wrong type\n");
        ++g_fail;
    }

    val = lis_json_object_get(&root, "c");
    if (val != NULL && val->type == LIS_JSON_BOOL) {
        expect_size("json_valid_c", (size_t)val->as.boolean, 1);
    } else {
        fprintf(stderr, "FAIL: json_valid_c: missing or wrong type\n");
        ++g_fail;
    }

    val = lis_json_object_get(&root, "d");
    if (val != NULL) {
        expect_size("json_valid_d_type", (size_t)val->type,
                    (size_t)LIS_JSON_NULL);
    } else {
        fprintf(stderr, "FAIL: json_valid_d: missing\n");
        ++g_fail;
    }

    val = lis_json_object_get(&root, "e");
    if (val != NULL && val->type == LIS_JSON_ARRAY) {
        expect_size("json_valid_e_count", val->as.array.count, 3);
    } else {
        fprintf(stderr, "FAIL: json_valid_e: missing or wrong type\n");
        ++g_fail;
    }

    lis_json_destroy(&root);
}

static void test_json_parse_malformed(void)
{
    lis_json_value root;

    expect_status("json_malformed_empty",
                  lis_json_parse("", 0, &root), LIS_STATUS_FORMAT);
    expect_status("json_malformed_brace",
                  lis_json_parse("{", 1, &root), LIS_STATUS_FORMAT);
    expect_status("json_malformed_trailing",
                  lis_json_parse("42 extra", 8, &root), LIS_STATUS_FORMAT);
    expect_status("json_malformed_null_text",
                  lis_json_parse(NULL, 5, &root), LIS_STATUS_INVALID_ARGUMENT);
}

static void test_json_parse_string_escapes(void)
{
    /* Test \n, \t, \", \\, \uXXXX */
    const char *json = "\"hello\\nworld\\t\\\"end\\\\\"";
    lis_json_value root;

    expect_status("json_escape_parse",
                  lis_json_parse(json, strlen(json), &root), LIS_STATUS_OK);
    if (root.type == LIS_JSON_STRING) {
        expect_bytes("json_escape_val", root.as.string.data,
                     root.as.string.len,
                     "hello\nworld\t\"end\\", 17);
    } else {
        fprintf(stderr, "FAIL: json_escape_val: not string\n");
        ++g_fail;
    }
    lis_json_destroy(&root);
}

static void test_json_parse_unicode_escape(void)
{
    /* \u0041 = 'A' */
    const char *json = "\"\\u0041\"";
    lis_json_value root;

    expect_status("json_unicode_parse",
                  lis_json_parse(json, strlen(json), &root), LIS_STATUS_OK);
    if (root.type == LIS_JSON_STRING) {
        expect_bytes("json_unicode_val", root.as.string.data,
                     root.as.string.len, "A", 1);
    } else {
        fprintf(stderr, "FAIL: json_unicode_val: not string\n");
        ++g_fail;
    }
    lis_json_destroy(&root);
}

static void test_json_nested(void)
{
    const char *json = "{\"model\": {\"type\": \"BPE\", \"inner\": [1]}}";
    lis_json_value root;
    const lis_json_value *model;
    const lis_json_value *type_val;

    expect_status("json_nested_parse",
                  lis_json_parse(json, strlen(json), &root), LIS_STATUS_OK);

    model = lis_json_object_get(&root, "model");
    if (model != NULL && model->type == LIS_JSON_OBJECT) {
        type_val = lis_json_object_get(model, "type");
        if (type_val != NULL && type_val->type == LIS_JSON_STRING) {
            expect_bytes("json_nested_type", type_val->as.string.data,
                         type_val->as.string.len, "BPE", 3);
        } else {
            fprintf(stderr, "FAIL: json_nested_type: missing\n");
            ++g_fail;
        }
    } else {
        fprintf(stderr, "FAIL: json_nested_model: missing\n");
        ++g_fail;
    }
    lis_json_destroy(&root);
}

/* ===== HF import tests ===== */

/*
 * Build a minimal valid HF tokenizer.json with a small BPE vocabulary.
 * Uses the GPT-2 byte-to-unicode encoding for byte values.
 *
 * Vocab: 4 single-byte tokens + 1 merged token
 *   "!" (byte 0x21) -> ID 0
 *   "a" (byte 0x61) -> ID 1    (codepoint 97, direct-mapped)
 *   "b" (byte 0x62) -> ID 2    (codepoint 98, direct-mapped)
 *   "ab" -> ID 3
 *
 * Merge: "a b" -> "ab" (rank 0)
 */
static const char *g_minimal_hf_json =
    "{\n"
    "  \"model\": {\n"
    "    \"type\": \"BPE\",\n"
    "    \"vocab\": {\n"
    "      \"!\": 0,\n"
    "      \"a\": 1,\n"
    "      \"b\": 2,\n"
    "      \"ab\": 3\n"
    "    },\n"
    "    \"merges\": [\n"
    "      \"a b\"\n"
    "    ]\n"
    "  }\n"
    "}\n";

static void test_hf_import_valid_bpe(void)
{
    const char *path = "/tmp/lis_test_hf_valid.json";
    lis_tokenizer tok;

    if (!write_file(path, g_minimal_hf_json, strlen(g_minimal_hf_json))) {
        fprintf(stderr, "FAIL: hf_valid_bpe: could not write test file\n");
        ++g_fail;
        return;
    }

    expect_status("hf_valid_load",
                  lis_hf_tokenizer_load(path, &tok), LIS_STATUS_OK);

    expect_size("hf_valid_vocab_size", tok.vocab_size, 4);

    /* Token 0: "!" = byte 0x21 */
    expect_size("hf_valid_tok0_len", tok.token_lens[0], 1);
    if (tok.token_bytes[0] != NULL) {
        expect_size("hf_valid_tok0_byte", (size_t)(unsigned char)tok.token_bytes[0][0], 0x21);
    }

    /* Token 1: "a" = byte 0x61 */
    expect_size("hf_valid_tok1_len", tok.token_lens[1], 1);
    if (tok.token_bytes[1] != NULL) {
        expect_bytes("hf_valid_tok1", tok.token_bytes[1], tok.token_lens[1],
                     "a", 1);
    }

    /* Token 3: "ab" = bytes 0x61 0x62 */
    expect_size("hf_valid_tok3_len", tok.token_lens[3], 2);
    if (tok.token_bytes[3] != NULL) {
        expect_bytes("hf_valid_tok3", tok.token_bytes[3], tok.token_lens[3],
                     "ab", 2);
    }

    /* Verify merge: (1,2) -> 3 exists */
    {
        size_t rank = 0;
        size_t result = 0;

        expect_status("hf_valid_merge_lookup",
                      lis_merge_table_lookup(&tok.merges, 1, 2,
                                             &rank, &result),
                      LIS_STATUS_OK);
        expect_size("hf_valid_merge_result", result, 3);
        expect_size("hf_valid_merge_rank", rank, 0);
    }

    /* Verify byte_to_token for 'a' (0x61) */
    expect_size("hf_valid_byte_lookup_a_valid",
                (size_t)tok.byte_to_token_valid[0x61], 1);
    expect_size("hf_valid_byte_lookup_a_id",
                tok.byte_to_token[0x61], 1);

    lis_tokenizer_destroy(&tok);
    remove(path);
}

static void test_hf_import_unsupported_type(void)
{
    const char *path = "/tmp/lis_test_hf_unigram.json";
    const char *json =
        "{\"model\": {\"type\": \"Unigram\", \"vocab\": {}, \"merges\": []}}";
    lis_tokenizer tok;

    if (!write_file(path, json, strlen(json))) {
        fprintf(stderr, "FAIL: hf_unsupported_type: write error\n");
        ++g_fail;
        return;
    }

    expect_status("hf_unsupported_type",
                  lis_hf_tokenizer_load(path, &tok), LIS_STATUS_UNSUPPORTED);
    remove(path);
}

static void test_hf_import_missing_model(void)
{
    const char *path = "/tmp/lis_test_hf_nomodel.json";
    const char *json = "{\"version\": \"1.0\"}";
    lis_tokenizer tok;

    if (!write_file(path, json, strlen(json))) {
        fprintf(stderr, "FAIL: hf_missing_model: write error\n");
        ++g_fail;
        return;
    }

    expect_status("hf_missing_model",
                  lis_hf_tokenizer_load(path, &tok), LIS_STATUS_FORMAT);
    remove(path);
}

static void test_hf_import_missing_vocab(void)
{
    const char *path = "/tmp/lis_test_hf_novocab.json";
    const char *json =
        "{\"model\": {\"type\": \"BPE\", \"merges\": []}}";
    lis_tokenizer tok;

    if (!write_file(path, json, strlen(json))) {
        fprintf(stderr, "FAIL: hf_missing_vocab: write error\n");
        ++g_fail;
        return;
    }

    expect_status("hf_missing_vocab",
                  lis_hf_tokenizer_load(path, &tok), LIS_STATUS_FORMAT);
    remove(path);
}

static void test_hf_import_missing_merges(void)
{
    const char *path = "/tmp/lis_test_hf_nomerges.json";
    const char *json =
        "{\"model\": {\"type\": \"BPE\", \"vocab\": {\"a\": 0}}}";
    lis_tokenizer tok;

    if (!write_file(path, json, strlen(json))) {
        fprintf(stderr, "FAIL: hf_missing_merges: write error\n");
        ++g_fail;
        return;
    }

    expect_status("hf_missing_merges",
                  lis_hf_tokenizer_load(path, &tok), LIS_STATUS_FORMAT);
    remove(path);
}

static void test_hf_import_bad_merge_token(void)
{
    const char *path = "/tmp/lis_test_hf_badmerge.json";
    /* Merge references "c" which is not in vocab */
    const char *json =
        "{\"model\": {\"type\": \"BPE\", "
        "\"vocab\": {\"a\": 0, \"b\": 1, \"ab\": 2}, "
        "\"merges\": [\"a c\"]}}";
    lis_tokenizer tok;

    if (!write_file(path, json, strlen(json))) {
        fprintf(stderr, "FAIL: hf_bad_merge_token: write error\n");
        ++g_fail;
        return;
    }

    expect_status("hf_bad_merge_token",
                  lis_hf_tokenizer_load(path, &tok), LIS_STATUS_FORMAT);
    remove(path);
}

static void test_hf_import_io_error(void)
{
    lis_tokenizer tok;

    expect_status("hf_io_error",
                  lis_hf_tokenizer_load("/tmp/lis_nonexistent_file.json",
                                        &tok),
                  LIS_STATUS_IO);
}

static void test_hf_import_null_args(void)
{
    lis_tokenizer tok;

    expect_status("hf_null_path",
                  lis_hf_tokenizer_load(NULL, &tok),
                  LIS_STATUS_INVALID_ARGUMENT);
    expect_status("hf_null_out",
                  lis_hf_tokenizer_load("/tmp/dummy", NULL),
                  LIS_STATUS_INVALID_ARGUMENT);
}

/* ===== Byte-to-unicode mapping tests ===== */

/*
 * Test that indirect-mapped bytes are correctly reversed.
 * Byte 0 maps to codepoint 256, which in UTF-8 is \xC4\x80 (Ā).
 * Byte 10 (\n) maps to codepoint 266, which in UTF-8 is \xC4\x8A.
 */
static void test_hf_import_indirect_bytes(void)
{
    const char *path = "/tmp/lis_test_hf_indirect.json";
    lis_tokenizer tok;
    /*
     * Vocab has one token: the UTF-8 encoding of codepoint 256 (Ā) -> ID 0
     * Codepoint 256 = byte 0 in the GPT-2 mapping.
     */
    const char *json =
        "{\"model\": {\"type\": \"BPE\", "
        "\"vocab\": {\"\xC4\x80\": 0}, "
        "\"merges\": []}}";

    if (!write_file(path, json, strlen(json))) {
        fprintf(stderr, "FAIL: hf_indirect_bytes: write error\n");
        ++g_fail;
        return;
    }

    expect_status("hf_indirect_load",
                  lis_hf_tokenizer_load(path, &tok), LIS_STATUS_OK);

    expect_size("hf_indirect_vocab_size", tok.vocab_size, 1);
    expect_size("hf_indirect_tok0_len", tok.token_lens[0], 1);
    if (tok.token_bytes[0] != NULL) {
        /* Should decode to byte 0 */
        expect_size("hf_indirect_tok0_byte",
                    (size_t)(unsigned char)tok.token_bytes[0][0], 0);
    }

    lis_tokenizer_destroy(&tok);
    remove(path);
}

/* ===== Round-trip test: HF import -> encode -> decode ===== */

static void test_hf_import_roundtrip(void)
{
    const char *path = "/tmp/lis_test_hf_roundtrip.json";
    lis_tokenizer tok;
    size_t *ids = NULL;
    size_t count = 0;
    char *decoded = NULL;
    size_t decoded_len = 0;

    if (!write_file(path, g_minimal_hf_json, strlen(g_minimal_hf_json))) {
        fprintf(stderr, "FAIL: hf_roundtrip: write error\n");
        ++g_fail;
        return;
    }

    expect_status("hf_roundtrip_load",
                  lis_hf_tokenizer_load(path, &tok), LIS_STATUS_OK);

    /* Encode "ab" — should merge (a,b)->ab, producing [3] */
    expect_status("hf_roundtrip_encode",
                  lis_tokenizer_encode(&tok, "ab", 2, &ids, &count),
                  LIS_STATUS_OK);
    expect_size("hf_roundtrip_encode_count", count, 1);
    if (count == 1) {
        expect_size("hf_roundtrip_encode_id", ids[0], 3);
    }

    /* Decode [3] back to "ab" */
    if (ids != NULL) {
        expect_status("hf_roundtrip_decode",
                      lis_tokenizer_decode(&tok, ids, count,
                                           &decoded, &decoded_len),
                      LIS_STATUS_OK);
        expect_bytes("hf_roundtrip_decoded", decoded, decoded_len, "ab", 2);
        free(decoded);
    }

    free(ids);
    lis_tokenizer_destroy(&tok);
    remove(path);
}

static void test_hf_import_added_special_token(void)
{
    const char *path = "/tmp/lis_test_hf_added_special.json";
    const char *json =
        "{\"added_tokens\":[{\"id\":5,\"content\":\"<x>\",\"special\":true}],"
        "\"model\":{\"type\":\"BPE\",\"vocab\":{\"a\":0},\"merges\":[]}}";
    lis_tokenizer tok;
    size_t *ids = NULL;
    size_t count = 0;
    char *decoded = NULL;
    size_t decoded_len = 0;

    if (!write_file(path, json, strlen(json))) {
        fprintf(stderr, "FAIL: hf_added_special: write error\n");
        ++g_fail;
        return;
    }

    expect_status("hf_added_special_load",
                  lis_hf_tokenizer_load(path, &tok), LIS_STATUS_OK);
    expect_size("hf_added_special_vocab_size", tok.vocab_size, 6);
    expect_size("hf_added_special_count", tok.special_token_count, 1);
    expect_status("hf_added_special_encode",
                  lis_tokenizer_encode(&tok, "<x>", 3, &ids, &count),
                  LIS_STATUS_OK);
    expect_size("hf_added_special_encode_count", count, 1);
    if (count == 1) {
        expect_size("hf_added_special_encode_id", ids[0], 5);
    }
    expect_status("hf_added_special_decode",
                  lis_tokenizer_decode(&tok, ids, count, &decoded,
                                       &decoded_len),
                  LIS_STATUS_OK);
    expect_bytes("hf_added_special_decoded", decoded, decoded_len, "<x>", 3);
    free(decoded);
    free(ids);
    lis_tokenizer_destroy(&tok);
    remove(path);
}

static void test_hf_import_array_merges(void)
{
    const char *path = "/tmp/lis_test_hf_array_merges.json";
    lis_tokenizer tok;
    /* Same vocab as g_minimal_hf_json, but merges use array format */
    const char *json =
        "{\n"
        "  \"model\": {\n"
        "    \"type\": \"BPE\",\n"
        "    \"vocab\": {\n"
        "      \"!\": 0,\n"
        "      \"a\": 1,\n"
        "      \"b\": 2,\n"
        "      \"ab\": 3\n"
        "    },\n"
        "    \"merges\": [\n"
        "      [\"a\", \"b\"]\n"
        "    ]\n"
        "  }\n"
        "}\n";
    size_t *ids = NULL;
    size_t count = 0;
    char *decoded = NULL;
    size_t decoded_len = 0;

    if (!write_file(path, json, strlen(json))) {
        fprintf(stderr, "FAIL: hf_array_merges: write error\n");
        ++g_fail;
        return;
    }

    expect_status("hf_array_merges_load",
                  lis_hf_tokenizer_load(path, &tok), LIS_STATUS_OK);
    expect_size("hf_array_merges_vocab_size", tok.vocab_size, 4);

    /* Verify merge: (1,2) -> 3 exists */
    {
        size_t rank = 0;
        size_t result = 0;

        expect_status("hf_array_merges_lookup",
                      lis_merge_table_lookup(&tok.merges, 1, 2,
                                             &rank, &result),
                      LIS_STATUS_OK);
        expect_size("hf_array_merges_result", result, 3);
        expect_size("hf_array_merges_rank", rank, 0);
    }

    /* Round-trip encode and decode */
    expect_status("hf_array_merges_encode",
                  lis_tokenizer_encode(&tok, "ab", 2, &ids, &count),
                  LIS_STATUS_OK);
    expect_size("hf_array_merges_encode_count", count, 1);
    if (count == 1) {
        expect_size("hf_array_merges_encode_id", ids[0], 3);
    }

    if (ids != NULL) {
        expect_status("hf_array_merges_decode",
                      lis_tokenizer_decode(&tok, ids, count,
                                           &decoded, &decoded_len),
                      LIS_STATUS_OK);
        expect_bytes("hf_array_merges_decoded", decoded, decoded_len, "ab", 2);
        free(decoded);
    }
    free(ids);

    /* Also verify the legacy string format is still accepted */
    {
        const char *legacy_path = "/tmp/lis_test_hf_legacy.json";
        lis_tokenizer legacy_tok;

        if (write_file(legacy_path, g_minimal_hf_json,
                       strlen(g_minimal_hf_json))) {
            expect_status("hf_array_merges_legacy_load",
                          lis_hf_tokenizer_load(legacy_path, &legacy_tok),
                          LIS_STATUS_OK);
            expect_size("hf_array_merges_legacy_vocab_size",
                        legacy_tok.vocab_size, 4);
            {
                size_t rank = 0;
                size_t result = 0;
                expect_status("hf_array_merges_legacy_lookup",
                              lis_merge_table_lookup(&legacy_tok.merges, 1,
                                                     2, &rank, &result),
                              LIS_STATUS_OK);
                expect_size("hf_array_merges_legacy_result", result, 3);
            }
            lis_tokenizer_destroy(&legacy_tok);
            remove(legacy_path);
        } else {
            fprintf(stderr, "FAIL: hf_array_merges: legacy write error\n");
            ++g_fail;
        }
    }

    lis_tokenizer_destroy(&tok);
    remove(path);
}

/* ===== Main ===== */

int main(void)
{
    g_pass = 0;
    g_fail = 0;

    /* JSON parser tests */
    test_json_parse_valid();
    test_json_parse_malformed();
    test_json_parse_string_escapes();
    test_json_parse_unicode_escape();
    test_json_nested();

    /* HF import tests */
    test_hf_import_valid_bpe();
    test_hf_import_unsupported_type();
    test_hf_import_missing_model();
    test_hf_import_missing_vocab();
    test_hf_import_missing_merges();
    test_hf_import_bad_merge_token();
    test_hf_import_io_error();
    test_hf_import_null_args();
    test_hf_import_indirect_bytes();

    /* Round-trip test */
    test_hf_import_roundtrip();
    test_hf_import_added_special_token();
    test_hf_import_array_merges();

    printf("test_hf_import: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
